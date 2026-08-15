# litMoE — install & verify

## Install

### Linux (glibc 2.31+, x86_64 or aarch64)

```bash
./scripts/setup-and-serve.sh --install-deps   # one-time
./scripts/setup-and-serve.sh
```

Manual:

```bash
LDFLAGS="-lm -pthread" make -j$(nproc)
sudo make install                              # or PREFIX=$HOME/.local make install
```

Note: conda's `LDFLAGS` env var breaks the Makefile's `LDFLAGS ?=` default.
Always pass `LDFLAGS="-lm -pthread"` explicitly on this host.

### macOS (Sonoma 14+ on M-series, Ventura 13+ on Intel)

```bash
./scripts/setup-and-serve.sh --install-deps   # installs Xcode CLI tools if missing
./scripts/setup-and-serve.sh
```

`apple/clang` does not have AVX2; the engine falls back to scalar kernels
on Intel macOS. Performance on M-series is similar to a Linux laptop.

### Windows (10/11, Server 2019+, x86_64)

PowerShell:

```powershell
.\scripts\setup-and-serve.ps1 -InstallDeps
.\scripts\setup-and-serve.ps1
```

Manually with CMake + MSVC:

```powershell
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --prefix C:\Users\you\k3lean
```

### Docker (any platform)

```bash
docker build -f Dockerfile -t litMoE .
docker run --rm -p 8080:8080 -v $PWD/checkpoints/k3:/data:ro litMoE
```

### Without admin (user-local install on any Unix)

```bash
PREFIX=$HOME/.local make install
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

---




## Build from source

The Makefile is POSIX (Linux + macOS). CMakeLists.txt works on every
platform including MSVC and MinGW on Windows.

```bash
# POSIX
LDFLAGS="-lm -pthread" make -j$(nproc) all

# CMake (everywhere)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Verify (3 GATEs)
LDFLAGS="-lm -pthread" make test
```

The build has two outputs:

| Output | Size | Purpose |
|---|---:|---|
| `bin/k3` | 167 KB | CLI for direct prompt/response |
| `bin/liblitmoe.so` | 162 KB | shared library, loaded by the Python server |

Both are reproducible from `src/` with the C engine unchanged.

### Convert a HuggingFace checkpoint

The C engine reads safetensors shards directly. `tools/convert.py`
verifies the layout and optionally re-shards:

```bash
# Verify only (no copy)
python3 tools/convert.py /path/to/hf/checkpoint --verify-only

# Convert in place
python3 tools/convert.py /path/to/hf/checkpoint /path/to/native

# Convert with fewer shards
python3 tools/convert.py /path/to/hf /path/to/native --shards 24
```

`tools/convert.py` uses `safetensors` directly — no PyTorch dependency.
If you don't have it: `pip install safetensors`, or run the Docker
fallback (`docker build -f Dockerfile.convert -t kimi-k3-convert .`).

If your HuggingFace download is already a complete safetensors
directory with `config.json` and `tokenizer.model` in the root,
`k3_open` loads it directly. The convert step is only needed when you
want a different shard count or want to pre-validate the layout.

---




## Verify it works

See the README's Quick Start section for the five real
demonstrations: `/v1/models`, blocking completion, streaming with
`curl -N`, auth, and `/v1/state/{save,load}`.

Expected response shape for the chat completion (truncated):

```
{
  "id": "chatcmpl-<random>",
  "object": "chat.completion",
  "created": 1755200000,
  "model": "kimi-k3",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "<real output from the engine, not a canned reply>"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": N,
    "completion_tokens": M,
    "total_tokens": N + M
  }
}
```

If `usage.prompt_tokens` reports 0, the engine didn't run — that's
the symptom of a missing or unreadable checkpoint. Check the server
startup log for `engine open failed`.

```json
{"id":"chatcmpl-...","object":"chat.completion","model":"kimi-k3",
 "choices":[{"index":0,"message":{"role":"assistant","content":"..."},
            "finish_reason":"stop"}],"usage":{"prompt_tokens":N,...}}
```

Engine-level verification (does not require a model on disk):

```bash
LDFLAGS="-lm -pthread" make test
```

Expected:

```
GATE 1  teacher forcing : 32/32 positions match tf_pred
GATE 2  greedy decode   : 20/20 generated tokens match full_ids
GATE 3  incremental    : 20/20 generated tokens match full_ids
VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
```

---




## Deployment

### Stand-alone (laptop / personal computer)

```bash
./scripts/setup-and-serve.sh --serve-only
# Now POST to http://127.0.0.1:8080/v1/chat/completions
```

### Shared LAN service (multi-user, browser-accessible, agent-CLI accessible)

After `bootstrap.sh` installs the launcher, the entire LAN stack comes
up with one command:

```bash
# Bare stack (Caddy + gateway + router, no chat UI)
litMoE stack up

# Full stack including Open WebUI for browser-based chats
litMoE stack up --webui

# Tear down / inspect
litMoE stack down
litMoE stack status
litMoE stack logs gateway
```

This wraps `docker compose -f deploy/compose.yml`, creating
`deploy/.env` from the example if missing.

What you get:

| URL | What it serves |
|---|---|
| `http://localhost/v1/models` | OpenAI endpoint (Caddy → gateway → router → model) |
| `http://localhost/llm/v1/...` | Same, with `/llm` prefix stripped |
| `http://localhost/healthz` | Liveness probe (public, no auth) |
| `http://localhost/workspace/...` | Model browser page |
| `http://localhost/client/...` | Laptop installer (`curl | bash` one-liner) |
| `http://localhost/` | Open WebUI chat (only with `--webui`) |

See **[deploy/README.md](../deploy/README.md)** for the full reference,
Caddyfile internals, and the gateway's auth/size-limit/streaming pattern.

For direct `docker compose` users (without the launcher):

```bash
cd deploy
cp .env.example .env           # set INTERNAL_API_KEY=$(openssl rand -hex 32)
docker compose --profile with-k3 --profile with-webui up -d --build
```

From any client on the LAN:

```bash
curl -fsSL -k https://<server>/client/install.sh | bash
source ~/llm-client/remote-env.sh
claude         # Claude Code, OpenCode, aider, Qwen — all use litMoE
```

Pattern matches the llm-server project but trimmed
for CPU-only / single-binary / cross-platform.

### Point Hermes at the deploy backend

Same as the Quick Start in the README — read
it once and apply the same three steps to the deploy LAN URL.
`$INTERNAL_API_KEY` is the key in `deploy/.env` (called
`INTERNAL_API_KEY`), not the per-host `LITMOE_API_KEY`. The `model.base_url`
becomes `http://<server>/llm/v1` (note the `/llm/` prefix the Caddy
router adds).

---



