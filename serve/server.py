"""server.py -- OpenAI Chat Completions HTTP server for kimi-k3-lean.

Endpoints:
    GET  /health                 liveness, plus what is loaded
    GET  /v1/models              the one model this process holds
    POST /v1/chat/completions    streaming and not

Stdlib only -- ``http.server.ThreadingHTTPServer`` and ``BaseHTTPRequestHandler``.
The C engine is single-caller per ctx, so generations queue on the engine lock.
That is the honest design: on a model streaming at a few tokens per second,
waiting for the lock is small next to waiting for the answer.

Streaming writes ``data: <json>\n\n`` SSE lines, with a final ``data: [DONE]\n\n``
sentinel. The article's engine is greedy, so we never branch on a stop string
mid-stream -- the engine emits tokens one at a time and we forward each one
to the client after detokenizing.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from . import api as api_mod
from .chatfmt import build_prompt
from .engine import Engine, EngineError


SERVER_NAME = "kimi-k3-lean"

# Bodies larger than this are refused before they hit memory. A chat request is
# text; anything at this size is a degenerate payload, not a conversation.
MAX_BODY_BYTES = 16 * 1024 * 1024


class ChatServer(ThreadingHTTPServer):
    """Threaded HTTP server; one engine, one lock."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, *, engine: Engine, model_id: str,
                 api_key: Optional[str] = None,
                 default_max_tokens: int = 256,
                 log_requests: bool = True):
        super().__init__(addr, handler)
        self.engine = engine
        self.model_id = model_id
        self.api_key = api_key
        self.default_max_tokens = default_max_tokens
        self.log_requests = log_requests
        self.started = int(time.time())

    def handle_error(self, request, client_address):
        """A client hanging up is not an error worth a traceback."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError,
                            TimeoutError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVER_NAME}/1"
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------

    def log_message(self, format, *args):
        if getattr(self.server, "log_requests", True):
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_json(self, status: int, payload: dict, *, headers=None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, err: api_mod.APIError) -> None:
        # The OpenAI shape is {"error": {"message": ..., "type": ..., ...}}.
        body = json.dumps(
            {"error": err.to_dict()}, ensure_ascii=False
        ).encode()
        self.send_response(err.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        # Refusing a request without consuming its body desynchronises a
        # keep-alive connection: the bytes we did not read become the next
        # request line. Every path that rejects before reading closes the
        # connection.
        def refuse(message: str, status: int) -> api_mod.APIError:
            self.close_connection = True
            return api_mod.APIError(message, status=status)

        length = self.headers.get("Content-Length")
        if length is None:
            raise refuse("Content-Length is required", 411)
        try:
            n = int(length)
        except ValueError:
            raise refuse("Content-Length is not a number", 400)
        if n < 0 or n > MAX_BODY_BYTES:
            raise refuse(
                f"request body may not exceed {MAX_BODY_BYTES} bytes", 413
            )
        raw = self.rfile.read(n) if n else b""
        if not raw:
            raise api_mod.APIError("request body is empty")
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise api_mod.APIError(f"request body is not valid JSON: {e}")
        if not isinstance(body, dict):
            raise api_mod.APIError("request body must be a JSON object")
        return body

    def _authorized(self) -> bool:
        key = self.server.api_key
        if not key:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            # Compare in constant time: a server that returns faster on a
            # wrong first byte leaks the key one byte at a time.
            import hmac
            return hmac.compare_digest(header[7:], key)
        return False

    # ---- routing --------------------------------------------------------

    def do_GET(self):
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                return self._health()
            if not self._authorized():
                raise api_mod.APIError(
                    "invalid API key", status=401,
                    type_="authentication_error",
                )
            if path == "/v1/models":
                return self._models()
            raise api_mod.APIError(
                f"no route for GET {path}", status=404,
                type_="not_found_error",
            )
        except api_mod.APIError as e:
            self._send_error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if not self._authorized():
                self.close_connection = True
                raise api_mod.APIError(
                    "invalid API key", status=401,
                    type_="authentication_error",
                )
            if path == "/v1/chat/completions":
                return self._chat()
            if path == "/v1/state/save":
                return self._state_save()
            if path == "/v1/state/load":
                return self._state_load()
            self.close_connection = True
            raise api_mod.APIError(
                f"no route for POST {path}", status=404,
                type_="not_found_error",
            )
        except api_mod.APIError as e:
            self._send_error(e)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except EngineError as e:
            self._send_error(api_mod.APIError(
                str(e), status=500, type_="engine_error",
            ))

    # ---- endpoints ------------------------------------------------------

    def _health(self):
        self._send_json(200, {
            "status": "ok",
            "model": self.server.model_id,
            "engine_version": "1.0.0",
            "uptime_s": int(time.time()) - self.server.started,
        })

    def _models(self):
        model_id = self.server.model_id
        self._send_json(200, api_mod.build_models_list(model_id))

    def _state_save(self):
        """POST /v1/state/save { "path": "/some/safe/location/state.bin" }

        Persists the engine's recurrent state + KV cache. The harness
        (chat agent) calls this after a conversation finishes so the
        next turn can resume without re-reading the whole prompt.
        """
        body = self._read_body()
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise api_mod.APIError(
                "path must be a non-empty string",
                status=400, type_="invalid_request_error",
                param="path",
            )
        # Refuse to write outside well-known locations. The state file is
        # opaque engine-internal data; the harness should pass an
        # explicit path. We allow /tmp and /var/tmp for convenience and
        # any path under $HOME, and refuse anything else.
        import os
        path_abs = os.path.abspath(path)
        home = os.path.expanduser("~")
        if not (path_abs.startswith("/tmp/")
                or path_abs.startswith("/var/tmp/")
                or path_abs.startswith(home + "/")):
            raise api_mod.APIError(
                f"path must be under /tmp, /var/tmp, or {home}/",
                status=400, type_="invalid_request_error",
                param="path",
            )
        try:
            with self.server.engine._lock:
                self.server.engine.save_state(path_abs)
        except EngineError as e:
            raise api_mod.APIError(
                str(e), status=500, type_="engine_error",
            )
        except OSError as e:
            raise api_mod.APIError(
                f"could not write state file: {e}",
                status=500, type_="io_error",
            )
        self._send_json(200, {
            "object": "state",
            "path": path_abs,
            "bytes_written": os.path.getsize(path_abs)
                              if os.path.exists(path_abs) else 0,
        })

    def _state_load(self):
        """POST /v1/state/load { "path": "/some/safe/location/state.bin" }

        Restores the engine's recurrent state + KV cache. The harness
        calls this at the start of a continued conversation.
        """
        body = self._read_body()
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise api_mod.APIError(
                "path must be a non-empty string",
                status=400, type_="invalid_request_error",
                param="path",
            )
        import os
        if not os.path.exists(path):
            raise api_mod.APIError(
                f"state file not found: {path}",
                status=404, type_="not_found_error",
                param="path",
            )
        try:
            with self.server.engine._lock:
                self.server.engine.load_state(path)
        except EngineError as e:
            raise api_mod.APIError(
                str(e), status=500, type_="engine_error",
            )
        except (OSError, RuntimeError) as e:
            raise api_mod.APIError(
                f"could not load state file: {e}",
                status=500, type_="io_error",
            )
        self._send_json(200, {
            "object": "state",
            "path": os.path.abspath(path),
            "restored": True,
        })

    # ---- chat -----------------------------------------------------------

    def _chat(self):
        body = self._read_body()
        srv = self.server
        engine = srv.engine

        try:
            req = api_mod.parse_chat_request(body)
        except api_mod.APIError:
            raise

        stream = req.stream

        # The lock spans prompt building *and* generation. The article's
        # engine carries no per-ctx state between calls in this lean build
        # (no multi-turn resume here -- that's the CLI's job), so a per-
        # request lock is enough to keep two concurrent requests from
        # racing the C engine.
        with engine._lock:
            prompt_ids = build_prompt(engine, req.messages)
            prompt_tokens = len(prompt_ids)

            if not stream:
                self._chat_blocking(
                    req, prompt_ids, prompt_tokens,
                )
            else:
                self._chat_stream(
                    req, prompt_ids, prompt_tokens,
                )

    def _chat_blocking(self, req, prompt_ids, prompt_tokens):
        engine = self.server.engine
        # k3_step is synchronous; prefill + decode in one call.
        try:
            out_ids = engine.complete(prompt_ids, max_tokens=req.max_tokens)
        except EngineError as e:
            raise api_mod.APIError(
                str(e), status=500, type_="engine_error",
            )

        # The article's engine returns the first argmax after the prompt
        # plus N-1 generated tokens. prompt_tokens + len(out_ids) is the
        # total tokens billed.
        completion_tokens = len(out_ids)
        finish_reason = "length" if completion_tokens >= req.max_tokens else "stop"
        text = engine.detokenize(out_ids) if out_ids else ""

        payload = api_mod.build_chat_response(
            request=req,
            finish_reason=finish_reason,
            content=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._send_json(200, payload)

    def _chat_stream(self, req, prompt_ids, prompt_tokens):
        engine = self.server.engine
        # SSE headers.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # Chunked because the length is unknown until the model stops.
        # Without Content-Length and on HTTP/1.1, the alternative is to
        # close the socket to signal end -- which costs the client its
        # keep-alive.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # First chunk: announce the assistant role.
        first = api_mod.build_chat_chunk(
            request=req,
            delta_text="",
            delta_role="assistant",
        )
        self._write_sse(first)

        completion_tokens = 0
        finish_reason = "stop"
        client_gone = False

        try:
            for tid in engine.stream(
                prompt_ids, max_tokens=req.max_tokens,
            ):
                if client_gone:
                    # We can't abort the C engine once it's started; the
                    # only signal is the lock. Break here and let the
                    # engine finish in the background.
                    break
                try:
                    piece = engine.detokenize([tid])
                except EngineError:
                    piece = ""
                completion_tokens += 1
                try:
                    chunk = api_mod.build_chat_chunk(
                        request=req, delta_text=piece,
                    )
                    self._write_sse(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    client_gone = True
                if completion_tokens >= req.max_tokens:
                    finish_reason = "length"
                    break
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected; nothing more to do.
            return

        # Final chunk: empty delta with finish_reason.
        final = api_mod.build_chat_chunk(
            request=req,
            delta_text=None,
            finish_reason=finish_reason,
        )
        try:
            self._write_sse(final)
            # The OpenAI sentinel.
            self._write_sse(api_mod.build_done_chunk(), raw=True)
            self._finish_chunked()
        except (BrokenPipeError, ConnectionResetError):
            return

    # ---- SSE framing -----------------------------------------------------

    def _write_sse(self, payload, *, raw: bool = False) -> None:
        """Write one SSE event. `payload` is the JSON body, or the raw
        sentinel string if `raw=True` (e.g. ``[DONE]``)."""
        if raw:
            body_bytes = payload.encode("utf-8")
        else:
            body_bytes = f"data: {payload}\n\n".encode("utf-8")
        # Manual chunked framing: BaseHTTPRequestHandler does not do it.
        self.wfile.write(b"%X\r\n" % len(body_bytes))
        self.wfile.write(body_bytes)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _finish_chunked(self) -> None:
        """Emit the terminating 0-length chunk."""
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def serve(engine: Engine, *, host: str = "127.0.0.1", port: int = 8080,
          model_id: str = "kimi-k3-lean",
          api_key: Optional[str] = None,
          default_max_tokens: int = 256,
          log_requests: bool = True,
          ready: Optional[threading.Event] = None) -> ChatServer:
    """Build the server. The caller decides whether to serve_forever."""
    if ":" in host:
        ChatServer.address_family = socket.AF_INET6
    srv = ChatServer(
        (host, port), Handler,
        engine=engine, model_id=model_id, api_key=api_key,
        default_max_tokens=default_max_tokens,
        log_requests=log_requests,
    )
    if ready is not None:
        ready.set()
    return srv


__all__ = ["ChatServer", "Handler", "serve", "MAX_BODY_BYTES"]
