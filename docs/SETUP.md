# Setup Guide

This guide covers installing engines and downloading models for litmoe.
All sizes and requirements are verified against HuggingFace model repositories
and engine source code as of August 2026.

## Prerequisites

- Python 3.10+
- 50 GB free disk for engine binaries + logs (models need much more, see below)
- Linux or macOS. Windows via WSL2.

## Step 1: Install litmoe

```bash
git clone https://github.com/chazhyseni/litMoE
cd litMoE
pip install -e .
```

## Step 2: Install an engine

litmoe routes requests to inference engines. You need at least one.

### Option A: llama.cpp (recommended — supports all three models)

llama.cpp has native support for Kimi-K3, Qwen3.8-2.4T, and MiniMax-M3.
All three architectures are merged to master and CI-tested.

**Prebuilt binary (fastest):**

```bash
litmoe install --engine llamacpp
```

This downloads the latest release from github.com/ggml-org/llama.cpp/releases,
extracts `llama-server`, and symlinks it into `~/.local/bin/`.

**Build from source (for CUDA/HIP/Metal/Vulkan support):**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON    # or -DGGML_HIP=ON, -DGGML_METAL=ON, etc.
cmake --build build --config Release -j --target llama-server
# Binary is at build/bin/llama-server — add it to your PATH
```

### Option B: ktransformers (for DeepSeek-V3, Kimi-K2, GLM, Qwen3-30B)

ktransformers does NOT support Kimi-K3, Qwen3.8-2.4T, or MiniMax-M3.
It does support DeepSeek-V3/R1, Kimi-K2, GLM-5.x, MiniMax-M2.5, Qwen3-30B-A3B.

```bash
pip install kt-kernel
```

Requires CUDA toolkit 12.8+ for GPU backend. CPU-only mode works with AVX-512
or AMX (Intel Xeon 4th gen+). AVX2-only CPUs (AMD EPYC) work but slower.

## Step 3: Download model weights

### Kimi-K3 (2.78T params, 93B active per token)

```
litmoe install --model kimi-k3
```

| Quant | Size | RAM+VRAM needed | Notes |
|---|---|---|---|
| UD-IQ1_S | 594 GB | ~594 GB | Lightest viable. Default. |
| UD-IQ1_M | 649 GB | ~649 GB | Slightly better quality. |
| UD-Q2_K_XL | 861 GB | ~861 GB | Better quality, needs more RAM. |
| UD-Q4_K_XL | 1509 GB | ~1509 GB | High quality. Datacenter only. |

Engine: llama.cpp only. ktransformers does not support K3.

### Qwen3.8-2.4T (2.4T params, 95B active per token)

```
litmoe install --model qwen3.8
```

| Quant | Size | RAM+VRAM needed | Notes |
|---|---|---|---|
| UD-Q1_0 | 397 GB | ~397 GB | Lightest available. |
| UD-IQ1_S | 508 GB | ~508 GB | Default. |
| UD-IQ1_M | 564 GB | ~564 GB | Better quality. |
| UD-IQ2_XXS | 657 GB | ~657 GB | Higher quality. |

Engine: llama.cpp only. ktransformers does not support Qwen3.8-2.4T.

Note: Qwen3.8-27B (a smaller variant of the same architecture) has several
open bugs in llama.cpp (CUDA lockups, long-context crashes). The 2.4T variant
shares the same architecture (Qwen3_5MoeForCausalLM) but has no reported
issues specific to it. Test before relying on it in production.

### MiniMax-M3 (428B params, 23B active per token)

```
litmoe install --model minimax-m3
```

| Quant | Size | RAM+VRAM needed | Notes |
|---|---|---|---|
| UD-IQ1_M | 128 GB | ~128 GB | Lightest. Runs on 128 GB RAM machine. |
| UD-IQ2_M | 134 GB | ~134 GB | |
| UD-Q2_K_XL | 143 GB | ~143 GB | Good quality/size balance. |
| UD-Q4_K_M | 264 GB | ~264 GB | High quality. Needs 256+ GB RAM. |
| Q8_0 | 453 GB | ~453 GB | Near-lossless. |

Engine: llama.cpp (native, merged 2026-07-26, CI-tested).
ktransformers supports MiniMax-M2.5 but has no M3 tutorial or optimize rule.

## Hardware requirements by model

The total memory (RAM + VRAM) must be at least the GGUF size. VRAM holds GPU
layers (`-ngl`), RAM holds the rest. Add ~10% overhead for KV cache and OS.

### Kimi-K3 and Qwen3.8-2.4T

These are trillion-parameter models. IQ1_S is the practical quant.

| Hardware | What fits | RAM needed (IQ1_S) |
|---|---|---|
| CPU only, no GPU | 0 layers on GPU | 594 GB (K3) / 508 GB (Qwen3.8) |
| 1 GPU, 24 GB VRAM | ~4 layers on GPU | ~570 GB (K3) / ~484 GB (Qwen3.8) |
| 2 GPU, 160 GB VRAM | ~25 layers on GPU | ~434 GB (K3) / ~348 GB (Qwen3.8) |
| 4 GPU, 320 GB VRAM | ~50 layers on GPU | ~274 GB (K3) / ~188 GB (Qwen3.8) |
| 8 GPU, 640 GB VRAM | All layers on GPU | ~0 GB extra RAM (K3) |

These numbers are arithmetic (GGUF size minus VRAM), not measured. Real
throughput depends on RAM bandwidth, GPU model, and CPU speed. No benchmark
data exists yet for K3 or Qwen3.8-2.4T on these configs in litmoe.

The only measured speed data point we have: llama.cpp on this project's
EPYC 7B13 (24 cores, 377 GB DDR4, no GPU, 379 MB/s disk) ran Kimi-K3 IQ1_S
at 0.85 tokens/second. That was entirely disk-bound.

### MiniMax-M3

M3 is the most accessible trillion-scale model. At 128 GB (IQ1_M), it fits on
high-end workstations.

| Hardware | What fits | RAM needed (IQ1_M) |
|---|---|---|
| CPU only, 128 GB RAM | All layers on CPU | 128 GB |
| CPU only, 192 GB RAM | All layers on CPU + KV cache headroom | 128 GB |
| 1 GPU, 24 GB VRAM | ~11 layers on GPU | ~104 GB |
| 2 GPU, 48 GB VRAM | ~22 layers on GPU | ~80 GB |
| Mac Studio M3 Ultra, 192 GB unified | All layers via Metal | 128 GB |

M3 at 428B params is less than 1/5th the size of K3 (2.78T). The 23B active
parameter count means each token only computes 4 of 128 experts + 1 shared,
keeping per-token compute low.

No measured throughput data for M3 on CPU exists in this project. The
ktransformers MiniMax-M2.5 tutorial reports running on 2x RTX 4090 (48 GB
VRAM) + 200 GB RAM, but M3 is a different model.

## Step 4: Configure models.yaml

After `litmoe install --model <name>`, the model is added to `models.yaml`
automatically. You can also edit it manually:

```yaml
host: 127.0.0.1
port: 8080
api_key: null

models:
  - id: kimi-k3
    engine: llamacpp
    model_path: /home/user/.litmoe/models/kimi-k3/UD-IQ1_S
    n_gpu_layers: -1    # -1 = all layers to GPU, 0 = CPU only
    n_ctx: 4096

  - id: minimax-m3
    engine: llamacpp
    model_path: /home/user/.litmoe/models/minimax-m3/UD-IQ1_M
    n_gpu_layers: 0     # CPU only if no GPU
    n_ctx: 4096
```

`n_gpu_layers` controls GPU offload in llama.cpp:
- `-1`: try to put all layers on GPU (fails if not enough VRAM)
- `0`: CPU only
- `N`: put N layers on GPU, rest on CPU

## Step 5: Start and test

```bash
litmoe serve                          # starts gateway + all configured engines
curl http://127.0.0.1:8080/v1/models  # list available models
litmoe status                         # check engine health
litmoe stop                           # stop all engines
```

Test a request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "minimax-m3", "messages": [{"role": "user", "content": "hello"}]}'
```

## Quick reference: which engine for which model?

| Model | llama.cpp | ktransformers | Default quant | Size |
|---|---|---|---|---|
| Kimi-K3 | Yes (native) | No | UD-IQ1_S | 594 GB |
| Qwen3.8-2.4T | Yes (native) | No | UD-IQ1_S | 508 GB |
| MiniMax-M3 | Yes (native) | No (M2.5 yes) | UD-IQ1_M | 128 GB |
| DeepSeek-V3 | Yes | Yes (optimized) | varies | ~600 GB |
| Kimi-K2 | Yes | Yes (optimized, ~10 t/s) | Q4_K_M | ~600 GB |
| GLM-5.x | Yes | Yes (optimized) | varies | varies |
| Qwen3-30B-A3B | Yes | Yes (optimized) | varies | ~30 GB |

Sources: llama.cpp source (LLM_ARCH registrations, model .cpp files), ktransformers
optimize rules directory, HuggingFace model configs and GGUF repositories. All
verified August 2026.