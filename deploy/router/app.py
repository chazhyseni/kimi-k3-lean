"""router — multi-model OpenAI-compatible router for kimi-k3-lean.

WHAT THIS IS
  Sits in front of one or more model backends. The model backends are
  themselves OpenAI-compatible Chat Completions servers (they just
  happen to load different weights). This router:

  1. Receives `/v1/chat/completions` requests with a `model:` field.
  2. Looks the model name up in its registry (env-driven).
  3. Forwards the request to the matching backend's HTTP URL.

When you start the stack with only one backend, the router is a no-op
(the model name is the only one in the registry). When you bring up a
second backend (Linear, a future Kimi variant, an Anthropic-format
translator, etc.), you list it in `MODELS` env and the router handles
both.

This is the kimi-k3-lean analogue of LiteLLM in the llm-server stack,
but stripped to OpenAI-only (no Anthropic translation — there's no need
for it here). The Python `openai` SDK, Open WebUI, aider, Claude Code
all speak OpenAI natively.

USAGE
  Standalone:    MODELS='{"kimi-k3":"http://k3:8080/v1"}' python3 deploy/router/app.py
  Docker:        part of the deploy/compose.yml stack

ENV
  MODELS        JSON dict mapping model name -> upstream base URL.
                Example: {"kimi-k3":"http://k3:8080/v1","kimi-linear":"http://linear:8080/v1"}
  DEFAULT_MODEL default model name if a request omits `model` field (default: "kimi-k3")
  PORT          default: 8080
"""
from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

_raw_models = os.environ.get("MODELS", '{"kimi-k3":"http://k3:8080/v1"}')
MODELS: dict[str, str] = json.loads(_raw_models)
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", next(iter(MODELS), "kimi-k3"))

app = FastAPI(title="kimi-k3-lean router", docs_url=None, redoc_url=None, openapi_url=None)

client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=30.0, read=1800.0, write=1800.0, pool=30.0)
)


def _upstream_for(model_name: str) -> str:
    if model_name in MODELS:
        return MODELS[model_name]
    # Fallback to default if the request names an unknown model.
    return MODELS[DEFAULT_MODEL]


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "service": "router",
        "models": list(MODELS.keys()),
        "default": DEFAULT_MODEL,
    }


@app.get("/v1/models")
async def list_models() -> dict:
    """Return the model list. Models are grouped by backend in production;
    here we assume one model per backend."""
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "kimi-k3-lean"}
            for name in MODELS
        ],
    }


@app.post("/v1/chat/completions")
async def chat(request: Request) -> Response:
    body = await request.json()
    model_name = body.get("model", DEFAULT_MODEL)
    upstream_base = _upstream_for(model_name)

    # Forward everything to the matching backend.
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    is_stream = body.get("stream", False)

    if is_stream:
        async def stream() -> "AsyncIterator[bytes]":
            async with client.stream(
                "POST", f"{upstream_base}/chat/completions",
                headers=headers, json=body,
            ) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(stream(), media_type="text/event-stream")

    # Non-streaming.
    r = await client.post(
        f"{upstream_base}/chat/completions",
        headers=headers, json=body,
    )
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers={k: v for k, v in r.headers.items() if k.lower() not in {"content-length", "transfer-encoding"}},
    )


@app.post("/v1/state/{action}")
async def state(action: str, request: Request) -> Response:
    """Forward state-save / state-load to the named backend."""
    body = await request.json()
    model_name = body.get("model", DEFAULT_MODEL)
    upstream_base = _upstream_for(model_name)
    r = await client.post(
        f"{upstream_base}/state/{action}",
        headers={"content-type": "application/json"},
        json=body,
    )
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.get("/v1/state")
async def state_list() -> dict:
    return {"models": list(MODELS.keys())}