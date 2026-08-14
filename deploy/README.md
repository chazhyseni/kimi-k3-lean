# kimi-k3-lean — Network Deployment

Run kimi-k3-lean as a multi-user LAN or VPN service so that laptops,
phones, browsers, and agent CLIs (Claude Code, Pi, OpenCode, aider,
Qwen Code) all share one local inference server.

```
LAN clients (browser / agent CLI / scripts)
        │  HTTPS :443  (only published port)
        ▼
   Caddy reverse proxy ── TLS + routing
   ├─ /v1/*         ──► gateway ──► router ──► kimi-k3-lean backend
   ├─ /workspace/*  ──► model browser page
   ├─ /client/*     ──► laptop installer (curl one-liner)
   └─ /             ──► Open WebUI (browser chat, optional)
```

This is the kimi-k3-lean analogue of /mnt/scratch/private-llm/ in the
llm-server deployment. Same architecture, smaller scope, no GPU, no
vLLM, no LiteLLM. The router replaces LiteLLM (LiteLLM is for
translating OpenAI ↔ Anthropic ↔ etc.; we don`t need that). The
gateway has the same auth/size-limit/streaming pattern.

## One-command bring-up

After `bootstrap.sh` has installed the `kimi-k3-lean` launcher:

```bash
# bare server (Caddy + gateway + model, no chat UI)
kimi-k3-lean stack up

# full stack including Open WebUI for browser chats
kimi-k3-lean stack up --webui

# tear down
kimi-k3-lean stack down
kimi-k3-lean stack status
kimi-k3-lean stack logs gateway
```

This is a thin wrapper around `docker compose -f deploy/compose.yml`.
On a fresh install it copies `deploy/.env.example` to `.env` for you;
edit that file before production to set INTERNAL_API_KEY, SERVER_NAME,
and WEBUI_ADMIN_PASSWORD.

What you get:

- `http://localhost/v1/models`     — Caddy-fronted OpenAI endpoint
- `http://localhost/llm/v1/...`    — same, with `/llm` prefix stripped
- `http://localhost/workspace/...` — file browser
- `http://localhost/client/...`    — laptop installer (`curl | bash`)
- `http://localhost/`              — Open WebUI chat (with `--webui`)

## Bring-up

```bash
cd deploy/
cp .env.example .env
# Edit .env — set INTERNAL_API_KEY (openssl rand -hex 32), SERVER_NAME, etc.

docker compose up -d --build         # gateway + router + workspace + caddy
docker compose --profile with-k3 up -d --build    # + the kimi-k3 backend
```

The first start downloads the k3 image, builds the gateway/router/
workspace images, and brings everything up. The model container stays
in `created` until you mount a populated checkpoint directory at
`deploy/checkpoints/k3`. To do that:

```bash
cd ..
./scripts/setup-and-serve.sh --download-only    # fetch weights into deploy/checkpoints/k3
./scripts/setup-and-serve.sh --convert-only     # convert in place

cd deploy
docker compose --profile with-k3 up -d k3        # start the k3 backend
```

## Bring-up modes

| Goal | Command |
|---|---|
| Stand up the proxy stack (no model yet) | `docker compose up -d --build` |
| Add the k3 backend after bringing up the proxy | `docker compose --profile with-k3 up -d k3` |
| Add a second model (linear) | replicate the `k3` service block under a different profile |
| Add Open WebUI for browser chat | add `--profile with-webui` to any `up` |
| Free the disk (stop the model only) | `docker compose stop k3` |
| Stop everything, keep state | `docker compose down` |

## Endpoints

After `docker compose up -d`:

| URL | Auth | Used by |
|---|---|---|
| `https://$SERVER_NAME/v1/models` | none | browser, scripts |
| `https://$SERVER_NAME/v1/chat/completions` | Bearer `$INTERNAL_API_KEY` | OpenAI-style clients |
| `https://$SERVER_NAME/llm/healthz` | none | ops |
| `https://$SERVER_NAME/workspace/models` | none | browser |
| `https://$SERVER_NAME/api/models` | none | browser JS |
| `https://$SERVER_NAME/client/install.sh` | none | laptop installer |

(With `--profile with-webui`: `https://$SERVER_NAME/` opens Open WebUI;
`WEBUI_ADMIN_EMAIL` / `WEBUI_ADMIN_PASSWORD` from `.env`.)

## Laptop access (any host on the LAN)

```bash
curl -fsSL -k https://10.10.4.104/client/install.sh | bash
```

The installer lands `~/llm-client/`:

- `remote-env.sh` — sets `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`,
  `NODE_EXTRA_CA_CERTS`, etc. every session.
- `secrets.env.example` — template (copy to `secrets.env`, fill in
  the keys from the server admin).
- `caddy-root.crt` — the LAN's internal CA cert (for curl/Node clients).

Then:

```bash
nano ~/llm-client/secrets.env            # paste INTERNAL_API_KEY from admin
chmod 600 ~/llm-client/secrets.env
source ~/llm-client/remote-env.sh        # every session

curl -k https://10.10.4.104/v1/models -H "Authorization: Bearer $INTERNAL_API_KEY"

claude                                   # Claude Code now talks to kimi-k3-lean
opencode run -m kimi-k3 "hello"          # OpenCode does the same
qwen -p "hi"                             # Qwen Code does the same
```

If your agent CLI uses a different env-var convention (Pi, aider, etc.),
check the README of that CLI. The `remote-env.sh` sets the
OpenAI-compatible standard variables (`OPENAI_BASE_URL`,
`OPENAI_API_KEY`) which all modern agent CLIs honor.

## TLS

The default Caddy mode is `tls internal`, which uses Caddy's own CA.
That's fine for a LAN or VPN:

- Browsers will warn once (click through).
- curl/Python/SDK clients need the CA cert. The installer pulls it
  to `~/llm-client/caddy-root.crt`. `remote-env.sh` points Node and
  Python at it via `NODE_EXTRA_CA_CERTS` / `SSL_CERT_FILE`.
- For system-wide trust on the client, see the
  `update-ca-certificates` / `add-trusted-cert` / `certutil` lines
  in `static/install.sh`.

For a public-facing deployment, swap `tls internal` for a real DNS
challenge (the `Caddyfile` is parameterized — set `TLS_MODE=dns-cloudflare`
or similar, and add the corresponding env vars to the Caddy service).

## Operations

```bash
cd deploy

# Logs
docker compose logs -f                  # all services
docker compose logs -f gateway          # one service
docker compose logs -f k3               # the model backend

# Health
curl -sk https://10.10.4.104/llm/healthz
curl -sk https://10.10.4.104/v1/models -H "Authorization: Bearer $INTERNAL_API_KEY"
docker compose ps                        # container state

# Restart one piece
docker compose restart gateway          # no state lost
docker compose up -d --force-recreate k3 # model backend (re-loads weights)

# Stop / start the model without touching the proxy
docker compose stop k3
docker compose start k3

# Bring everything down
docker compose down                     # state persists in volumes
```

## Architecture in detail

### Caddy
Single TLS-terminating reverse proxy. Only 80/443 published. Routes
URL prefixes to backends:

- `/v1/*`, `/llm/*`     → gateway (auth, body-cap, streaming)
- `/workspace/*`, `/api/models` → workspace (browser UI, model browser)
- `/client/*`           → workspace (static installer files)
- `/`                   → optional Open WebUI (gated by `--profile with-webui`)

### Gateway (FastAPI)
- Bearer-token auth (constant-time compare, no logs).
- 10 MB body cap (configurable via `MAX_REQUEST_BYTES`).
- Streams SSE chunks unchanged (preserves OpenAI token-by-token output).
- Forwards to the router with the upstream API key.

### Router (FastAPI)
- Multi-model registry (env-driven JSON dict, `MODELS`).
- Forwards `/v1/chat/completions` to the matching backend's HTTP URL.
- Forwards `/v1/state/save` and `/v1/state/load` for conversation
  resume (the state file lives on whichever backend the model is on).

### Workspace (FastAPI)
- `/workspace/models` — HTML page listing models (card view).
- `/api/models` — JSON model list.
- `/client/*` — static installer files.

### Model backend (the kimi-k3-lean image)
- Loads the checkpoint from `/data` (mounted bind/volume).
- Serves OpenAI chat completions + state endpoints over HTTP.
- Health check on `/health`.

### When to add a second backend
The router's `MODELS` env is a JSON dict. To add a second model:

```yaml
environment:
  MODELS: |
    {
      "kimi-k3": "http://k3:8080/v1",
      "kimi-linear": "http://linear:8080/v1"
    }
```

Add a second `model` service for the linear backend (or any other
OpenAI-compatible server), and the router will start dispatching
requests based on the `model:` field in the request body.

## What this is NOT

- Not a public cloud gateway. For multi-region / cloud-native, use a
  proper inference platform (Modal, RunPod, etc.) with their tooling.
- Not a replacement for vLLM. If you have a GPU and want to serve
  Qwen3, Llama, or Mistral at full speed, use vLLM directly. kimi-k3-lean
  is for **CPU-friendly OpenAI-compatible serving**, specifically the
  Kimi K3 engine which has no GPU path.
- Not certified for HIPAA / SOC2 / etc. The gateway stores no prompt
  content and the proxy is the only attack surface, but no audit
  trail is included. Add one if your deployment needs it.