"""api.py -- OpenAI Chat Completions request/response shaping.

Two main entry points:
- `parse_chat_request(payload)` -- validate and normalize the request JSON.
- `build_chat_response(...)` -- shape the engine's output into OpenAI-shape JSON.

Streaming uses `build_chat_chunk(...)` for each token and `build_done_chunk()`
for the final `[DONE]` sentinel.

Field names follow the OpenAI Chat Completions spec as of 2025:
https://platform.openai.com/docs/api-reference/chat
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# Default model name returned in /v1/models. Override via the request if the
# client sends a specific model.
DEFAULT_MODEL = "kimi-k3-lean"


# ---------------------------------------------------------------------------- request

@dataclass
class ChatRequest:
    """Normalized OpenAI Chat Completions request.

    Fields match the public OpenAI spec; defaults are sane for a local model.
    """
    messages: list[dict]               # required
    model: str = DEFAULT_MODEL         # echoed back; engine doesn't care
    max_tokens: int = 256              # cap on generation
    temperature: float = 1.0           # accepted but not used (article uses greedy)
    top_p: float = 1.0                 # accepted but not used
    stream: bool = False               # SSE if true
    stop: list[str] | None = None      # accepted; v1 implementation uses first only
    user: str | None = None            # opaque; logged but not used
    presence_penalty: float = 0.0      # accepted but not used
    frequency_penalty: float = 0.0     # accepted but not used
    seed: int | None = None            # accepted; ignored
    n: int = 1                         # accepted; v1 always returns 1
    # Non-OpenAI fields (pass-through, used by kimi-k3-lean).
    top_k: int = -1                    # engine-specific; -1 = default
    incremental: bool = True           # engine carries KV cache between turns
    state: str | None = None           # engine-specific; state file path

    @property
    def request_id(self) -> str:
        return f"chatcmpl-{uuid.uuid4().hex}"


class APIError(Exception):
    """An HTTP error in the OpenAI Chat Completions shape.

    `status` is the HTTP status code the server returns; `type` is the
    OpenAI error category string; `param`/`code` are the OpenAI-shape
    optional fields.
    """
    def __init__(self, message: str, type_: str = "invalid_request_error",
                 status: int = 400,
                 param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.type = type_
        self.status = status
        self.param = param
        self.code = code

    def to_dict(self) -> dict:
        """Body of the OpenAI-shape error envelope."""
        out: dict = {
            "message": self.message,
            "type": self.type,
        }
        if self.param is not None:
            out["param"] = self.param
        if self.code is not None:
            out["code"] = self.code
        return out


def parse_chat_request(payload: Any) -> ChatRequest:
    """Validate and normalize a request body.

    Raises APIError for any 400-class issue. Returns a ChatRequest.
    """
    if not isinstance(payload, dict):
        raise APIError("request body must be a JSON object", type_="invalid_request_error")

    msgs = payload.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise APIError(
            "messages must be a non-empty list",
            type_="invalid_request_error",
            param="messages",
        )
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            raise APIError(
                f"messages[{i}] must be an object",
                type_="invalid_request_error",
                param=f"messages[{i}]",
            )
        if not isinstance(m.get("role"), str):
            raise APIError(
                f"messages[{i}].role is required and must be a string",
                type_="invalid_request_error",
                param=f"messages[{i}].role",
            )

    mt = payload.get("max_tokens", 256)
    if not isinstance(mt, int) or mt < 1 or mt > 32768:
        raise APIError(
            "max_tokens must be an integer in [1, 32768]",
            type_="invalid_request_error",
            param="max_tokens",
        )

    # Optional fields. We accept a wide range but normalize.
    out = ChatRequest(
        messages=msgs,
        model=str(payload.get("model", DEFAULT_MODEL)),
        max_tokens=mt,
        temperature=float(payload.get("temperature", 1.0)),
        top_p=float(payload.get("top_p", 1.0)),
        stream=bool(payload.get("stream", False)),
        stop=payload.get("stop"),
        user=payload.get("user"),
        presence_penalty=float(payload.get("presence_penalty", 0.0)),
        frequency_penalty=float(payload.get("frequency_penalty", 0.0)),
        seed=payload.get("seed"),
        n=int(payload.get("n", 1)),
        top_k=int(payload.get("top_k", -1)),
        incremental=bool(payload.get("incremental", True)),
        state=payload.get("state"),
    )
    return out


# ---------------------------------------------------------------------------- response

def build_chat_response(
    *,
    request: ChatRequest,
    finish_reason: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    created_ts: int | None = None,
    system_fingerprint: str | None = None,
) -> dict:
    """Shape one complete Chat Completions response."""
    return {
        "id": request.request_id,
        "object": "chat.completion",
        "created": created_ts if created_ts is not None else int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": _normalize_finish_reason(finish_reason),
                "logprobs": None,
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": system_fingerprint or "fp_kimi_k3_lean",
    }


def build_chat_chunk(
    *,
    request: ChatRequest,
    delta_text: Optional[str],
    delta_role: Optional[str] = None,
    finish_reason: Optional[str] = None,
    created_ts: int | None = None,
) -> str:
    """Format one SSE chunk. Returns the bytes to write (incl. SSE framing).

    The caller is expected to send `data: <chunk>\n\n`. We emit the chunk
    content as JSON; the caller prefixes `data: ` and trailing `\n\n`.
    """
    delta: dict = {}
    if delta_role:
        delta["role"] = delta_role
    if delta_text is not None:
        delta["content"] = delta_text

    chunk = {
        "id": request.request_id,
        "object": "chat.completion.chunk",
        "created": created_ts if created_ts is not None else int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": _normalize_finish_reason(finish_reason) if finish_reason else None,
            },
        ],
    }
    return json.dumps(chunk, ensure_ascii=False)


def build_done_chunk() -> str:
    """The OpenAI sentinel `data: [DONE]\n\n`. We return just the body."""
    return "[DONE]"


def _normalize_finish_reason(reason: str | None) -> str | None:
    """Engine uses 'eof' / 'max' / 'stop'; OpenAI uses 'stop' / 'length'."""
    if reason is None:
        return None
    r = reason.lower()
    if r in ("eof", "stop"):
        return "stop"
    if r in ("max", "length"):
        return "length"
    if r in ("abort",):
        return "stop"
    return r


# ---------------------------------------------------------------------------- /v1/models

def build_models_list(model_id: str = DEFAULT_MODEL) -> dict:
    """Response shape for GET /v1/models."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kimi-k3-lean",
                "permission": [],
            },
        ],
    }


__all__ = [
    "ChatRequest",
    "APIError",
    "parse_chat_request",
    "build_chat_response",
    "build_chat_chunk",
    "build_done_chunk",
    "build_models_list",
    "DEFAULT_MODEL",
]