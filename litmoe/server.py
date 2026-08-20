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
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)

        # Strip Authorization header — the gateway handles auth, not the engine.
        # llama-server rejects Bearer tokens that don't match its own key.
        fwd_headers = {"content-type": "application/json"}

        if stream:
            # For streaming, create the client outside async with so it
            # stays open for the duration of the StreamingResponse iterator
            return StreamingResponse(
                _stream_response(target_url, send_body, timeout, fwd_headers),
                media_type="text/event-stream",
            )
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(target_url, content=send_body, headers=fwd_headers)
                return JSONResponse(content=r.json(), status_code=r.status_code)

    def load_engines(self, log_dir: str | None = None) -> None:
        """Start all configured engines."""
        from pathlib import Path
        ld = Path(log_dir) if log_dir else None
        for idx, model in enumerate(self.config.models):
            # Auto-fix stale context sizes — set to model's native context
            # Also check if model + KV cache fits in available memory
            if model.n_ctx and model.n_ctx < 16384:
                model_id = model.id or ""
                native_ctx = {
                    "deepseek-v4-flash": 131072,   # 128K MLA
                    "kimi-linear-48b": 1048576,     # 1M KDA+MLA
                    "kimi-k3": 262144,              # 256K MLA
                    "qwen3.8-2.4t": 262144,         # 256K (1M with YaRN)
                    "qwen3.8-9b-distill": 131072,   # 128K (1M with YaRN)
                    "minimax-m3": 1048576,          # 1M
                    "gemma-4-12b": 131072,          # 128K
                    "gemma-4-31b": 131072,          # 128K
                    "llama-4-scout": 10485760,     # 10M
                }
                new_ctx = native_ctx.get(model_id, 131072)  # default 128K

                # Check memory fit
                import os as _os
                try:
                    total_mem = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
                    total_mem_gb = total_mem / 1e9
                    # KV cache rate per token (bytes) — rough per-model estimates
                    kv_rates = {
                        "deepseek-v4-flash": 23.1 / 131072,
                        "kimi-linear-48b": 15.0 / 1048576,
                        "kimi-k3": 49.9 / 262144,
                        "qwen3.8-9b-distill": 17.2 / 131072,
                        "minimax-m3": 386.5 / 1048576,
                        "gemma-4-12b": 85.9 / 131072,
                        "gemma-4-31b": 86.0 / 131072,
                        "llama-4-scout": 206.2 / 10485760,
                    }
                    kv_rate = kv_rates.get(model_id, 17.2 / 131072)
                    kv_gb = kv_rate * new_ctx
                    # Estimate model size from config — we don't know exact
                    # but 90% of memory minus 3 GB overhead is the budget
                    avail_gb = total_mem_gb * 0.9 - 3
                    if kv_gb > avail_gb:
                        # Reduce context to fit (assume model is already loaded)
                        max_ctx = int(avail_gb / kv_rate)
                        new_ctx = max((max_ctx // 4096) * 4096, 8192)
                except (ValueError, OSError, AttributeError):
                    pass

                logger.warning(
                    "Model %s has n_ctx=%d (too small), setting to %d",
                    model.id, model.n_ctx, new_ctx
                )
                model.n_ctx = new_ctx
                # Also persist the fix to models.yaml so it doesn't need auto-fixing every run
                try:
                    import yaml as _yaml
                    config_path = Path(_yaml.__file__).parent  # just to ensure yaml is imported
                    from litmoe.config import load_config, GatewayConfig
                    # Find the config file path
                    import os
                    cfg_path = os.environ.get("LITMOE_CONFIG")
                    if not cfg_path:
                        for candidate in ["models.yaml", "deploy/models.yaml", "config/models.yaml"]:
                            if Path(candidate).exists():
                                cfg_path = candidate
                                break
                    if cfg_path and Path(cfg_path).exists():
                        with open(cfg_path) as f:
                            raw = _yaml.safe_load(f)
                        for m in raw.get("models", []):
                            if m.get("id") == model.id:
                                m["n_ctx"] = new_ctx
                                break
                        with open(cfg_path, "w") as f:
                            _yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
                        logger.info("Persisted n_ctx=%d for %s to %s", new_ctx, model.id, cfg_path)
                except Exception:
                    pass  # best effort — runtime fix is what matters
            logger.info("Loading %s via %s...", model.id, model.engine)
            engine = make_engine(model)
            # Assign unique port: first model 8081, second 8082, etc.
            engine._assigned_port = 8081 + idx
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


async def _stream_response(url: str, body: bytes, timeout: httpx.Timeout, headers: dict | None = None):
    """Stream SSE responses from upstream engine. Owns the httpx client lifecycle."""
    fwd_headers = headers or {"content-type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, content=body, headers=fwd_headers) as r:
            async for line in r.aiter_lines():
                if line:
                    yield line.encode() + b"\n"


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
        "max_tokens": payload.get("max_tokens", 8192),
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
