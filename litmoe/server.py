"""OpenAI-compatible HTTP proxy.

This is a thin pass-through that forwards requests to engine processes.
Engines speak OpenAI-compatible HTTP; we just route by model name.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from litmoe.config import GatewayConfig, ModelEntry
from litmoe.engines import make_engine, Engine

logger = logging.getLogger(__name__)


class Gateway:
    """Routes OpenAI requests to the right engine based on model name."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.engines: dict[str, Engine] = {}
        self.app = FastAPI(title="litmoe gateway")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {"id": m.id, "object": "model", "owned_by": "litmoe",
                     "engine": m.engine}
                    for m in self.config.models
                ],
            }

        @self.app.get("/health")
        async def health():
            return {
                "status": "ok",
                "engines": {
                    model_id: {
                        "running": eng.process is not None and eng.process.poll() is None,
                        "port": eng.default_port(),
                        "base_url": eng.base_url,
                        "log": str(eng._log_path) if eng._log_path else None,
                    }
                    for model_id, eng in self.engines.items()
                },
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._proxy(request, "chat/completions")

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            return await self._proxy(request, "completions")

        @self.app.post("/v1/messages")
        async def messages(request: Request):
            """Anthropic Messages API → forward to OpenAI Chat Completions."""
            return await self._proxy(request, "messages", anthropic=True)

    async def _proxy(self, request: Request, endpoint: str, anthropic: bool = False):
        """Forward request to the right engine."""
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            raise HTTPException(400, "invalid JSON body")

        # Determine model
        model_id = payload.get("model")
        if not model_id:
            raise HTTPException(400, "missing 'model' field")

        engine = self.engines.get(model_id)
        if not engine:
            raise HTTPException(404, f"model not loaded: {model_id}")

        if not engine.base_url:
            raise HTTPException(503, f"engine for {model_id} not ready")

        # Translate Anthropic Messages → OpenAI Chat Completions
        if anthropic:
            payload = _anthropic_to_openai(payload)
            target_url = f"{engine.base_url}/v1/chat/completions"
        else:
            target_url = f"{engine.base_url}/v1/{endpoint}"

        # Use translated payload if anthropic, otherwise original body
        send_body = json.dumps(payload).encode() if anthropic else body
        stream = payload.get("stream", False)
        timeout = httpx.Timeout(connect=10.0, read=None, write=600.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if stream:
                return StreamingResponse(
                    _stream_response(client, target_url, send_body),
                    media_type="text/event-stream",
                )
            else:
                r = await client.post(target_url, content=send_body, headers={"content-type": "application/json"})
                return JSONResponse(content=r.json(), status_code=r.status_code)

    def load_engines(self, log_dir: str | None = None) -> None:
        """Start all configured engines."""
        from pathlib import Path
        ld = Path(log_dir) if log_dir else None
        for model in self.config.models:
            logger.info("Loading %s via %s...", model.id, model.engine)
            engine = make_engine(model)
            engine.start(log_dir=ld)
            self.engines[model.id] = engine

    async def wait_all_ready(self, timeout: float = 600.0) -> bool:
        """Wait for all engines to be ready."""
        tasks = [eng.wait_ready(timeout=timeout) for eng in self.engines.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results)

    def shutdown(self) -> None:
        """Stop all engines."""
        for engine in self.engines.values():
            engine.stop()


async def _stream_response(client: httpx.AsyncClient, url: str, body: bytes):
    """Stream responses from upstream engine."""
    async with client.stream("POST", url, content=body, headers={"content-type": "application/json"}) as r:
        async for chunk in r.aiter_bytes():
            yield chunk


def _anthropic_to_openai(payload: dict) -> dict:
    """Translate Anthropic Messages API to OpenAI Chat Completions.

    Best-effort translation. Supports the common subset.
    """
    messages = []
    system = payload.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            for block in system:
                if block.get("type") == "text":
                    messages.append({"role": "system", "content": block.get("text", "")})

    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    # Pass through as data URL or skip
                    text_parts.append(f"[image: {block.get('source', {}).get('type', 'unknown')}]")
            messages.append({"role": role, "content": "\n".join(text_parts)})

    out = {
        "model": payload.get("model"),
        "messages": messages,
        "max_tokens": payload.get("max_tokens", 4096),
        "stream": False,  # Anthropic streaming is different SSE format
    }
    if "temperature" in payload:
        out["temperature"] = payload["temperature"]
    if "top_p" in payload:
        out["top_p"] = payload["top_p"]
    if "stop_sequences" in payload:
        out["stop"] = payload["stop_sequences"]

    return out


def run(config: GatewayConfig, log_dir: str | None = None) -> None:
    """Entry point: start gateway."""
    gateway = Gateway(config)
    gateway.load_engines(log_dir=log_dir)

    # Wait for engines in background
    async def startup():
        ok = await gateway.wait_all_ready(timeout=600)
        if not ok:
            logger.warning("Not all engines became ready")
        else:
            logger.info("All engines ready.")

    # Use the existing event loop or create one for startup checks
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(startup())

    try:
        uvicorn.run(gateway.app, host=config.host, port=config.port, log_level="info")
    finally:
        gateway.shutdown()
