# Architecture

The litmoe gateway is intentionally minimal. Every component earns its place.

```
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │                                                                                │
   │   CLIENTS                                                                     │
   │   ───────                                                                     │
   │   Hermes Agent · Claude Code · Open WebUI · aider · curl · Python              │
   │                                                                                │
   └─────────────────────────────────┬──────────────────────────────────────────────┘
                                     │ HTTP (OpenAI + Anthropic API)
                                     │ POST /v1/chat/completions
                                     │ POST /v1/messages
                                     │ GET  /v1/models
                                     ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │                              LITMOE GATEWAY                                    │
   │                              ────────────────                                  │
   │                                                                                │
   │   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
   │   │   /v1/models      │    │   /health        │    │   /v1/messages   │      │
   │   │  (list config)   │    │  (engine status) │    │  (Anthropic→OAI) │      │
   │   └──────────────────┘    └──────────────────┘    └──────────────────┘      │
   │                                                                                │
   │   ┌─────────────────────────────────────────────────────────────────────┐     │
   │   │              REQUEST ROUTER                                         │     │
   │   │   - parse JSON body                                                 │     │
   │   │   - read `model` field                                             │     │
   │   │   - look up engine by model name                                  │     │
   │   │   - if Anthropic endpoint: translate Messages→ChatCompletions      │     │
   │   └─────────────────────────────────────────────────────────────────────┘     │
   │                                                                                │
   │   litmoe/server.py   ·   ~215 lines FastAPI pass-through                         │
   │                                                                                │
   └────────────┬───────────────────────────────────────┬───────────────────────────┘
                │                                       │
                │ http (model name = deepseek-v3)   │ http (model name = kimi-k3)
                ▼                                       ▼
   ┌─────────────────────────────┐    ┌────────────────────────────────┐
   │   KTRANSFORMERS ENGINE      │    │   LLAMA.CPP ENGINE             │
   │   ────────────────────────  │    │   ─────────────────────────    │
   │                             │    │                                │
   │   spawned by: kt run        │    │   spawned by: llama-server     │
   │   listens on: :10002       │    │   listens on: :8081            │
   │                             │    │                                │
   │   ┌───────────────────────┐ │    │   ┌───────────────────────────┐ │
   │   │  GPU backend          │ │    │   │  CUDA / HIP / Metal /     │ │
   │   │  (sglang-kt)          │ │    │   │  Vulkan / SYCL backend    │ │
   │   │                       │ │    │   │                           │ │
   │   │  Hot experts on GPU   │ │    │   │  All weights in VRAM      │ │
   │   │  Cold experts on CPU  │ │    │   │  OR spillover to CPU      │ │
   │   │  Async offloading     │ │    │   │  per `--n-gpu-layers`     │ │
   │   └───────────────────────┘ │    │   └───────────────────────────┘ │
   │                             │    │                                │
   │   ┌───────────────────────┐ │    │   ┌───────────────────────────┐ │
   │   │  CPU backend          │ │    │   │  Quantized weights        │ │
   │   │  (kt-kernel AVX2)     │ │    │   │  1.5/2/3/4/5/6/8-bit      │ │
   │   │                       │ │    │   │  via GGUF format          │ │
   │   │  Auto-detects:        │ │    │   │                           │ │
   │   │  AMX / AVX-512 / AVX2 │ │    │   │  See llama.cpp quant      │ │
   │   │  broad CPU support    │ │    │   │  tools for conversion     │ │
   │   └───────────────────────┘ │    │   └───────────────────────────┘ │
   │                             │    │                                │
   │   via: litmoe/engines/      │    │   via: litmoe/engines/         │
   │        ktransformers.py     │    │        llamacpp.py             │
   │                             │    │                                │
   └─────────────────────────────┘    └────────────────────────────────┘
                │                                       │
                │ reads models.yaml                     │ reads models.yaml
                │                                       │
                └─────────────────┬─────────────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │     models.yaml              │
                   │     ───────────              │
                   │                              │
                   │   host: 127.0.0.1           │
                   │   port: 8080                │
                   │   api_key: null             │
                   │   models:                   │
                   │     - id: deepseek-v3       │
                   │       engine: ktransformers │
                   │       model_path: /data/ds3 │
                   │     - id: kimi-k3           │
                   │       engine: llamacpp      │
                   │       model_path: /data/k3  │
                   │                              │
                   └──────────────────────────────┘
```

## Data flow

1. Client sends `POST /v1/chat/completions` with `model: kimi-k3`.
2. Gateway parses the body, looks up `kimi-k3` in `models.yaml`.
3. Gateway forwards the request to the llama.cpp engine on port 8081.
4. llama.cpp runs the forward pass (GPU or CPU).
5. Gateway streams the response back to the client.

The gateway never touches the forward pass. It adds single-digit-millisecond
latency per request and zero compute.

## Engine lifecycle

- `litmoe serve` reads `models.yaml`, spawns each engine as a subprocess,
  waits for readiness, then starts the gateway.
- `litmoe stop` sends SIGTERM to all engine subprocesses.
- `litmoe status` polls `/health` on the gateway to show which engines are up.

## Ports

| Service | Default port | Configurable |
|---|---|---|
| Gateway | 8080 | `port` in models.yaml |
| ktransformers | 10002 | `default_port()` in ktransformers.py |
| llama.cpp | 8081 | `default_port()` in llamacpp.py |
| Open WebUI | 8080 (via Caddy) | `HTTP_PORT` in .env |
