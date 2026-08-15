"""__main__.py -- `python3 serve/__main__.py <model_dir>`.

OpenAI-compatible HTTP server over liblitmoe.so. The model argument is a
path to the article's tiny_k3 fixture (or a real K3 model directory).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

if __package__ in (None, ""):                    # python3 serve/__main__.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "serve"

from .engine import Engine, EngineError            # noqa: E402
from .server import serve                          # noqa: E402


class _NullEngine:
    """Sentinel used when the real engine failed to open (no model on disk).

    The OpenAI server still comes up so /v1/models works (Hermes/bookmarks
    can register the URL); /v1/chat/completions requests get an
    engine_error envelope. Shape mirrors Engine's read-only introspection.
    """

    def __init__(self, model_id: str = "kimi-k3") -> None:
        self.model_id   = model_id
        self.n_layers   = 0
        self.vocab_size = 0
        self.ctx_size   = 0
        # The http layer does `with engine._lock:` per request.
        from threading import Lock
        self._lock = Lock()

    def close(self) -> None:
        pass

    # Operations the http layer calls. Each raises EngineError which the
    # server catches and converts to a 500 engine_error JSON envelope.
    def tokenize(self, text: str):
        raise EngineError("engine not loaded (no model on disk; run "
                          "`litMoE fetch` to download K3 weights)")

    def generate(self, prompt_ids, *, max_tokens: int = 0):
        raise EngineError("engine not loaded (no model on disk)")

    def complete(self, prompt_ids, *, max_tokens: int = 0):
        raise EngineError("engine not loaded (no model on disk)")

    def stream(self, prompt_ids, *, max_tokens: int = 0):
        raise EngineError("engine not loaded (no model on disk)")

    def detokenize(self, ids, *, out_cap: int = 4096):
        raise EngineError("engine not loaded (no model on disk)")

    def reset_stats(self) -> None: pass
    def get_stats(self): return None
    def save_state(self, path) -> None:
        raise EngineError("engine not loaded")
    def load_state(self, path) -> None:
        raise EngineError("engine not loaded")


def bounded_int(lo: int, hi: int):
    def parse(text: str) -> int:
        try:
            value = int(text, 10)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {text}") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(
                f"integer must be between {lo} and {hi}: {text}")
        return value
    return parse


def humanize_bytes(n: int) -> str:
    """For startup banner only."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m serve",
        description="OpenAI-compatible server for the article's kimi-k3 engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 serve/__main__.py tests/fixtures/tiny_k3.bin --port 8080
  python3 serve/__main__.py tests/fixtures/tiny_k3.bin --api-key "$K3_KEY" \\
      --host 0.0.0.0

  curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{"model":"litMoE","messages":[{"role":"user","content":"hi"}]}'
""")
    ap.add_argument("model", help="path to the model directory (or tiny_k3.bin fixture)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="default 127.0.0.1 — loopback only. Use 0.0.0.0 to "
                         "accept from the network, and set --api-key if you do")
    ap.add_argument("--port", type=bounded_int(0, 65535), default=8080)
    ap.add_argument("--model-id", default=None,
                    help="name reported by /v1/models (default: directory name)")
    ap.add_argument("--api-key", default=os.environ.get("LITMOE_API_KEY"),
                    help="require this bearer token (default $LITMOE_API_KEY)")

    # Engine options that map to liblitmoe's k3_open_args.
    g = ap.add_argument_group("engine")
    g.add_argument("--preset", default=None,
                   choices=["laptop", "desktop", "workstation", "server",
                            "max", "auto"],
                   help="memory preset (laptop=8GB, server=high RAM)")
    g.add_argument("--layers", type=bounded_int(-1, 1024), default=-1,
                   help="bind only first N layers; -1 = all")
    g.add_argument("--cache-gb", type=float, default=0.0,
                   help="routed-expert cache budget (0 = engine picks)")
    g.add_argument("--trunk-gb", type=float, default=0.0,
                   help="trunk ring/pinned-layer budget (0 = engine picks)")
    g.add_argument("--trunk-dir", default=None,
                   help="optional packed-trunk directory")
    g.add_argument("--config", default=None,
                   help="optional config.json override")
    g.add_argument("--tok-dir", default=None,
                   help="optional tiktoken.model directory")
    g.add_argument("--no-incremental", action="store_true",
                   help="do not carry KV cache + recurrent state between turns")

    s = ap.add_argument_group("serving")
    s.add_argument("--max-tokens", type=bounded_int(1, 32768), default=256,
                   help="default cap when a request does not set one "
                        "(default 256)")
    s.add_argument("--log-requests", action="store_true",
                   help="log each request to stderr (default: quiet)")

    args = ap.parse_args(argv)

    model = Path(args.model).expanduser()
    model_id = args.model_id or model.name

    engine = None
    if model.exists():
        try:
            engine = Engine(
                str(model),
                preset=args.preset,
                trunk_dir=args.trunk_dir,
                config_path=args.config,
                tok_dir=args.tok_dir,
                layers=args.layers,
                cache_gb=args.cache_gb,
                trunk_gb=args.trunk_gb,
                incremental=not args.no_incremental,
            )
        except EngineError as e:
            print(f"engine open failed: {e}", file=sys.stderr)
            print("the HTTP scaffold will still come up; /v1/models works,", file=sys.stderr)
            print("but /v1/chat/completions will return engine_error until", file=sys.stderr)
            print("weights are on disk. Run: litMoE fetch", file=sys.stderr)
    else:
        print(f"no model at {model} — starting scaffold (engine_error on chat)", file=sys.stderr)
        print("download weights with: litMoE fetch", file=sys.stderr)

    # Wrap engine (or its absence) in a uniform interface for the http
    # layer. When the real engine loaded, use it directly. When it didn't,
    # the wrapper exposes the same .model_id / .n_layers / .vocab_size /
    # .ctx_size as Engine does, so /v1/models still returns a useful body;
    # /v1/chat/completions returns the engine_error envelope.
    if engine is None:
        engine = _NullEngine(model_id=model_id or "kimi-k3")

    print(f"litMoE OpenAI server")
    print(f"  model    {model_id} — {engine.model_id}, "
          f"{engine.n_layers} layers, vocab {engine.vocab_size}, "
          f"ctx {engine.ctx_size}")
    print(f"  preset   {args.preset or 'auto'}")
    if not args.api_key and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\nWARNING: listening on {args.host} with no --api-key: "
              f"anyone who can reach this port can use the model.",
              file=sys.stderr)

    srv = serve(
        engine,
        host=args.host,
        port=args.port,
        model_id=model_id,
        api_key=args.api_key,
        default_max_tokens=args.max_tokens,
        log_requests=args.log_requests,
    )

    shown = args.host if ":" not in args.host else f"[{args.host}]"
    print(f"\nlistening on http://{shown}:{args.port}  "
          f"(POST /v1/chat/completions)")

    # Signal handling: shut down cleanly on SIGINT / SIGTERM.
    stop_event = {"flag": False}

    def _on_signal(signum, frame):
        stop_event["flag"] = True
        try:
            srv.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        engine.close()
        print("\nstopping", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
