# leanmoe — Multi-Model Lean Inference Engine

## Vision

One repo, one `curl | bash` install, one CLI. Run any trillion-parameter
MoE model on a personal computer — no GPU, no framework, no cloud.
Today: Kimi K3 (2.78T) and Qwen3 MoE (2.4T). Tomorrow: any MoE model
with disk-resident experts and a lean C engine.

## The User Experience

### Install (one command, any machine)

```bash
curl -fsSL https://raw.githubusercontent.com/chazhyseni/leanmoe/main/bootstrap.sh | bash
```

Builds `libmoe.so` + `bin/moe`, starts the HTTP server scaffold on
`http://127.0.0.1:8080`, installs the `leanmoe` CLI to `~/.local/bin`,
registers with Hermes if present. No model weights needed to install.

### Fetch a model

```bash
leanmoe fetch kimi-k3       # ~982 GB, ~4 hours, resumable
leanmoe fetch qwen3-moe     # ~XXX GB, ~X hours, resumable
leanmoe models available     # list downloadable models
```

Weights land at `~/.leanmoe/checkpoints/<model-name>/`.

### Serve

```bash
leanmoe serve kimi-k3                # start Kimi K3 on :8080
leanmoe serve qwen3-moe --port 8081  # start Qwen3 on :8081
leanmoe serve --auto                  # serve whatever weights are on disk
```

The engine auto-detects the architecture from `config.json` — no `--arch`
flag needed. One `libmoe.so` runs both.

### Chat

```bash
leanmoe chat -m "hello" --model kimi-k3
leanmoe chat -m "hello" --model qwen3-moe
```

Or through any harness (Hermes, Open WebUI, Claude Code, aider, etc.)
that speaks the OpenAI Chat Completions API at `http://127.0.0.1:8080/v1`.

### Multi-model status

```bash
leanmoe status
# kimi-k3:   running  PID 12345  http://127.0.0.1:8080
# qwen3-moe: running  PID 12346  http://127.0.0.1:8081
```

### LAN deployment

```bash
leanmoe stack up --webui --kimi-k3 --qwen3
# Caddy + gateway + router + Open WebUI + both model containers
# One endpoint: https://your-server/v1/chat/completions
# Router dispatches based on the "model" field
```

## Naming Convention

Everything shared is neutral (`leanmoe`, `libmoe`, `moe`, `MOE_*`).
Everything architecture-specific keeps its name (`kimi_ops`, `qwen_ops`).

| Old name (litMoE) | New name (leanmoe) | Scope |
|---|---|---|
| `litMoE` (CLI) | `leanmoe` | shared |
| `liblitmoe.so` | `libmoe.so` | shared |
| `bin/k3` | `bin/moe` | shared |
| `LITMOE_DIR`, `LITMOE_PORT`, etc. | `MOE_DIR`, `MOE_PORT`, etc. | shared |
| `k3_cache.c` | `cache.c` | shared |
| `k3_trunk.c` | `trunk.c` | shared |
| `k3_st.c` | `safetensors.c` | shared |
| `k3_engine.c` | `engine.c` | shared |
| `k3_api.c` | `api.c` | shared |
| `liblitmoe.h` | `moe.h` | shared |
| `k3_ops.c` | `kimi_ops.c` | arch-specific (Kimi) |
| `k3_bind.c` | `kimi_bind.c` | arch-specific (Kimi) |
| `k3.h` | `kimi.h` | arch-specific (Kimi) |
| `k3_cfg.h` | `kimi_cfg.h` | arch-specific (Kimi) |
| (new) | `qwen_ops.c` | arch-specific (Qwen3) |
| (new) | `qwen_bind.c` | arch-specific (Qwen3) |
| (new) | `qwen.h` | arch-specific (Qwen3) |
| (new) | `qwen_cfg.h` | arch-specific (Qwen3) |
| `fetch-model.sh` | `fetch-model.sh` | shared |
| `litmoe-doctor.sh` | `leanmoe-doctor.sh` | shared |

## Directory Structure

```
leanmoe/
├── README.md
├── LICENSE
├── NOTICE
├── Makefile
├── CMakeLists.txt
├── bootstrap.sh
├── bootstrap.ps1
├── leanmoe                     # CLI launcher
├── install.sh
├── install.ps1
├── Dockerfile
├── Dockerfile.convert
├── docker-compose.yml
├── .gitignore
│
├── include/
│   ├── moe.h                   # public C API (was liblitmoe.h)
│   ├── arch.h                  # ArchOps vtable
│   └── arch/
│       ├── kimi.h              # Kimi K3 config + structs
│       ├── kimi_cfg.h          # Kimi config reader
│       ├── qwen.h              # Qwen3 config + structs (new)
│       └── qwen_cfg.h          # Qwen3 config reader (new)
│
├── src/
│   ├── shared/                 # architecture-agnostic infrastructure
│   │   ├── cache.c             # LRU expert cache
│   │   ├── trunk.c             # trunk streaming
│   │   ├── safetensors.c       # safetensors reader
│   │   ├── loader.c            # checkpoint loader
│   │   ├── tokenizer.h         # BPE tokenizer
│   │   ├── engine.c            # dispatch + forward loop
│   │   └── api.c               # public C API impl
│   │
│   ├── arch/                   # architecture-specific kernels
│   │   ├── kimi_ops.c          # KDA, MLA, SiTU-GLU, ShortConv, AttnRes
│   │   ├── kimi_bind.c         # Kimi K3 tensor name mapping
│   │   ├── qwen_ops.c          # GQA, RoPE, SwiGLU, INT4 KV (new)
│   │   └── qwen_bind.c         # Qwen3 tensor name mapping (new)
│   │
│   └── cli/
│       └── moe_run.c           # CLI binary (was k3_run.c)
│
├── serve/                      # HTTP server (generic)
│   ├── __main__.py
│   ├── server.py
│   ├── engine.py
│   ├── api.py
│   └── chatfmt.py
│
├── scripts/
│   ├── fetch-model.sh          # generic downloader
│   ├── pack-trunk.sh
│   └── leanmoe-doctor.sh
│
├── tools/
│   ├── convert.py              # --arch flag for Kimi/Qwen
│   └── ...
│
├── tests/
│   ├── unit/
│   ├── fixtures/
│   │   ├── tiny_kimi/          # Kimi K3 test fixture (existing)
│   │   └── tiny_qwen/          # Qwen3 test fixture (new)
│   └── ...
│
├── deploy/                     # network deployment (generic)
│   ├── compose.yml
│   ├── caddy/Caddyfile
│   ├── gateway/
│   ├── router/
│   ├── workspace/
│   ├── model/Dockerfile
│   └── ...
│
└── docs/
    ├── MULTI_ARCH.md           # this design doc
    ├── INSTALL.md
    └── PERFORMANCE.md
```

## Architecture Diagram

```
                    ┌─────────────────────────────┐
                    │     Harness (any)            │
                    │  Hermes, Open WebUI,         │
                    │  Claude Code, aider, curl     │
                    └──────────┬──────────────────┘
                               │ OpenAI Chat Completions API
                               │ http://127.0.0.1:8080/v1
                               │
                    ┌──────────▼──────────────────┐
                    │    serve/server.py          │  ← generic HTTP
                    │    (stdlib, OpenAI shape)    │
                    └──────────┬──────────────────┘
                               │ ctypes
                               │
                    ┌──────────▼──────────────────┐
                    │    libmoe.so                 │  ← one library
                    │    (public C API in moe.h)    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    engine.c (dispatch)        │  ← reads config.json
                    │    detect_arch() → vtable     │     selects ArchOps
                    └──────────┬──────────────────┘
                    ┌──────────┴──────────────────┐
                    │                              │
           ┌───────▼───────┐              ┌───────▼───────┐
           │  kimi_ops.c   │              │  qwen_ops.c   │
           │               │              │               │
           │  KDA (69L)    │              │  GQA+RoPE     │
           │  MLA (24L)    │              │  (all layers) │
           │  SiTU-GLU     │              │  SwiGLU       │
           │  ShortConv    │              │  INT4 KV cache│
           │  AttnRes      │              │  Std residual │
           └───────┬───────┘              └───────┬───────┘
                   │                              │
           ┌───────▼──────────────────────────────▼───────┐
           │          SHARED INFRASTRUCTURE               │
           │                                             │
           │  cache.c    — LRU expert cache              │
           │  trunk.c    — trunk streaming (NVMe)        │
           │  safetensors.c — shard I/O                  │
           │  loader.c   — checkpoint index              │
           │  tokenizer.h — BPE tokenizer                │
           │                                             │
           │  NVMe: 1.45 TB experts (MXFP4)             │
           │  RAM:  trunk + expert cache + KV cache      │
           └─────────────────────────────────────────────┘
```

## How Architecture Detection Works

At `moe_open()` time (inside `engine.c`):

1. Read `config.json` from the model directory
2. Check for architecture markers:
   - `"linear_attn_config"` or `"kda_num_heads"` → `kimi_k3_ops`
   - `"num_key_value_heads"` → `qwen3_moe_ops`
   - else → error "unknown architecture"
3. Print: `"detected architecture: kimi-k3 (93 layers, 896 experts)"`
4. Select the matching `ArchOps` vtable
5. All subsequent forward calls go through the vtable

The host (Python server, CLI) never specifies the architecture.
It's auto-detected from the checkpoint's own config.

## INT4 KV Cache for Qwen3

Kimi K3's KDA is recurrent (O(1) state per layer) — no KV cache needed.
Qwen3's GQA needs a KV cache that grows with sequence length.

At BF16, 128K context, 80 layers, 8 GQA heads, 128 head_dim:
40 GB KV cache. Too much for a personal computer.

At INT4 (4-bit quantized KV):
10 GB KV cache. Fits the desktop preset (32 GB total RAM).

The INT4 KV cache:
- Quantize each K/V pair to 4-bit at write time (per-8-element FP8 scale)
- Dequantize on read during attention computation
- RAM-resident (no NVMe streaming — attention reads ALL KV per token)
- <1% quality degradation (proven in vLLM, SGLang, llama.cpp)

No context capping. No cache eviction. Full 128K context in 10 GB RAM.

## Multi-Model Server Management

The launcher manages multiple server instances (one per model):

```
~/.leanmoe/
├── checkpoints/
│   ├── kimi-k3/              # Kimi K3 weights
│   └── qwen3-moe/            # Qwen3 weights
├── servers/
│   ├── kimi-k3.pid           # PID of kimi-k3 server
│   ├── kimi-k3.env           # port, host, token, model-dir
│   ├── qwen3-moe.pid
│   └── qwen3-moe.env
├── server.log                # combined log
└── server.env                # default env (for scaffold mode)
```

`leanmoe status` shows all running servers.
`leanmoe stop kimi-k3` stops one. `leanmoe stop` stops all.

## Model Registry

Hardcoded in the launcher for now (can become a JSON file later):

```bash
# model_name → huggingface_repo
KIMI_LITMOE_REPO="moonshotai/Kimi-K3"
QWEN3_MOE_REPO="Qwen/Qwen3-235B-A22B"   # placeholder — verify actual repo

# model_name → display name
KIMI_K3_NAME="Kimi K3 (2.78T, MXFP4 experts)"
QWEN3_NAME="Qwen3 MoE (2.4T, GQA+RoPE)"
```

`leanmoe fetch <name>` looks up the repo, calls `scripts/fetch-model.sh`
with the right HF repo ID, downloads to `~/.leanmoe/checkpoints/<name>/`.

## Env Var Migration

All `K3_*` env vars become `MOE_*`:
`LITMOE_DIR` → `MOE_DIR`, `LITMOE_PORT` → `MOE_PORT`, etc.

The launcher reads both for backward compatibility:
```bash
# priority: shell env > MOE_* in conf > K3_* in conf > default
```

## Implementation Phases

### Phase 1: Rename (1-2 days)
Rename repo, CLI, env vars, lib, binary, scripts. Update all docs.
The `litMoE` GitHub repo redirects to `leanmoe`.

### Phase 2: Restructure source (1 day)
Move shared files to `src/shared/`, arch-specific to `src/arch/`.
Update all `#include` paths. Verify `make test` passes.

### Phase 3: ArchOps dispatch (1-2 days)
Create `kimi_ops.c` shim wrapping existing kernels.
Modify `engine.c` to dispatch through vtable.
Verify Kimi path is byte-identical (3 GATEs still pass).

### Phase 4: Qwen3 kernels (1-2 weeks)
`qwen_ops.c`: GQA + RoPE + SwiGLU + INT4 KV cache.
`qwen_bind.c`: Qwen3 tensor name mapping.
`qwen_cfg.h`: Qwen3 config reader.
Tiny Qwen3 fixture + oracle + 3-GATE tests.

### Phase 5: Multi-model launcher (2-3 days)
Multiple server instances, model registry, per-model PID files,
multi-model status, `leanmoe serve <model-name>`.

### Phase 6: Documentation + release (1-2 days)
New README, architecture diagram, updated INSTALL.md, deploy docs.
Tag release.

## Supported Models

| Model | Parameters | Architecture | Disk | RAM floor | Status |
|-------|-----------|-------------|------|-----------|--------|
| Kimi K3 | 2.78T | KDA + MLA + SiTU-GLU | 982 GB | 8 GB | Working |
| Qwen3 MoE | 2.4T | GQA + RoPE + SwiGLU | ~XXX GB | 8 GB | Planned |

## What Makes This Possible

The key innovation is **disk-resident expert streaming**. Both Kimi K3
and Qwen3 MoE activate only a small number of experts per token (16 for
Kimi, ~8 for Qwen3). The engine reads only those experts from NVMe per
token through an LRU cache it controls. The 1.45 TB expert pool never
enters RAM. This is what makes a trillion-parameter model runnable on
a personal computer, and it's architecture-agnostic — the cache doesn't
care whether the expert was quantized by Kimi's MXFP4 or Qwen3's BF16.