"""workspace — model browser page + static laptop installer host.

WHAT THIS IS
  A small FastAPI service that exposes:

  1. GET /workspace/models      — HTML page with model cards (no auth).
  2. GET /api/models            — JSON model list (mirrors OpenAI-style).
  3. GET /client/*              — static laptop installer files.

This is the kimi-k3-lean analogue of the workspace service in
the llm-server deployment. The page is generated from the same model
list that the router knows about; nothing else here.

ENV
  MODELS         JSON dict mapping model name -> description. Same as
                 what the router uses.
  STATIC_DIR     default: /srv/llm-client  (where the installer lives)
  PORT           default: 8080
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

MODELS = json.loads(os.environ.get("MODELS", '{"kimi-k3": "Kimi K3 (2.78T params, MXFP4 experts)"}'))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "/srv/llm-client"))

app = FastAPI(title="kimi-k3-lean workspace", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/workspace/models")
async def models_html() -> HTMLResponse:
    cards = "\n".join(
        f"""<article class="card"><h3>{html.escape(name)}</h3>
<p>{html.escape(desc)}</p>
<code>POST /v1/chat/completions</code></article>"""
        for name, desc in MODELS.items()
    )
    page = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>kimi-k3-lean — models</title>
<style>
  body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ margin-top: 0; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }}
  code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }}
</style>
</head><body>
<h1>kimi-k3-lean — models</h1>
<p>Self-hosted OpenAI-compatible endpoints. No data leaves this host.</p>
{cards}
</body></html>"""
    return HTMLResponse(page)


@app.get("/api/models")
async def models_json() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "kimi-k3-lean"}
            for name in MODELS
        ],
    })


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "service": "workspace", "models": list(MODELS.keys())}


# Static file server for the laptop installer (off by default if dir is empty).
if STATIC_DIR.is_dir():
    app.mount("/client", StaticFiles(directory=str(STATIC_DIR)), name="client")
else:
    @app.get("/client/{path:path}")
    async def client_missing(_: str) -> JSONResponse:
        return JSONResponse({"error": {"message": "client installer not installed"}}, status_code=404)