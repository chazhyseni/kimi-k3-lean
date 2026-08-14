"""gateway — auth + size limits + reverse proxy for the kimi-k3-lean router.

WHAT THIS IS
  The public-facing entry point. Clients (browser, CLI agents, scripts)
  hit `https://<host>/llm/v1/...` and this service:

  1. Requires a bearer token (constant-time compare).
  2. Caps request body size (10 MB by default).
  3. Proxies to the router (`http://router:8080/v1/...`) with the
     router's internal key.

This is the kimi-k3-lean analogue of /mnt/scratch/private-llm/gateway/app.py
in your llm-server deployment. It uses the same hmac.compare_digest
pattern for constant-time auth, the same 10 MB cap, and the same
httpx streaming pattern that preserves SSE chunks.

USAGE
  Standalone:    python3 deploy/gateway/app.py
  Docker:        part of the deploy/compose.yml stack

ENV
  UPSTREAM_BASE_URL   default: http://router:8080/v1
  UPSTREAM_API_KEY    bearer to send to upstream
  INTERNAL_API_KEY    bearer to require from clients
  ALLOWED_ORIGINS     comma-separated CORS origins (default: empty = no CORS)
  MAX_REQUEST_BYTES   default: 10485760  (10 MB)
  PORT                default: 8080
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://router:8080/v1").rstrip("/")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")
_ALLOWED_ORIGINS_RAW = os.environ.get("ALLOWED_ORIGINS", "")
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", "10485760"))

app = FastAPI(title="kimi-k3-lean gateway", docs_url=None, redoc_url=None, openapi_url=None)

if _ALLOWED_ORIGINS_RAW:
    ALLOWED_ORIGINS = [
        x.strip() for x in _ALLOWED_ORIGINS_RAW.split(",") if x.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

# Long timeouts because model inference can be slow (especially on CPU).
client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=30.0, read=1800.0, write=1800.0, pool=30.0)
)


def _authorized(auth_header: str) -> bool:
    """Constant-time bearer-token check. Returns True if no key required
    or the provided header matches the expected key."""
    if not INTERNAL_API_KEY:
        return True
    import hmac
    if not auth_header.startswith("Bearer "):
        return False
    provided = auth_header[7:].encode("utf-8")
    expected = INTERNAL_API_KEY.encode("utf-8")
    return hmac.compare_digest(provided, expected)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — does not require auth."""
    return {"status": "ok", "service": "gateway", "upstream": UPSTREAM_BASE_URL}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request,
                authorization: str | None = Header(default=None)) -> Response:
    """Forward any request to the upstream router, after auth + size checks."""
    if not _authorized(authorization or ""):
        raise HTTPException(status_code=401, detail="invalid API key")

    # Cap request body size.
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"request body may not exceed {MAX_REQUEST_BYTES} bytes",
        )

    upstream_url = f"{UPSTREAM_BASE_URL}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    if UPSTREAM_API_KEY:
        headers["authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    # Streaming if the client requested it.
    if request.headers.get("accept") == "text/event-stream":
        async def stream() -> "AsyncIterator[bytes]":
            async with client.stream(
                request.method, upstream_url, headers=headers, content=body,
            ) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(stream(), media_type="text/event-stream")

    # Non-streaming.
    r = await client.request(request.method, upstream_url, headers=headers, content=body)
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers={k: v for k, v in r.headers.items() if k.lower() not in {"content-length", "transfer-encoding"}},
    )


@app.exception_handler(HTTPException)
async def http_exc_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "gateway_error"}},
    )