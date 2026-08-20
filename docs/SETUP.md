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

### Option A: llama.cpp (recommended — supports all models below)

llama.cpp has native support for every model listed in this guide.
All architectures are merged to master and CI-tested.

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

### Option B: ktransformers (Linux + NVIDIA only)

ktransformers supports DeepSeek-V3/R1, Kimi-K2, GLM-5.x, MiniMax-M2.5/M3,
Qwen3-30B-A3B. Does NOT support Kimi-K3 or Qwen3.8-2.4T.

**Not available on macOS.** kt-kernel depends on triton, which requires
Linux + NVIDIA GPU. See [triton-lang/triton#3443](https://github.com/triton-lang/triton/issues/3443).
On macOS, use llama.cpp (Metal backend) instead.

```bash
litmoe install --engine ktransformers
```

This clones the repo and builds from source via `pip install ./kt-kernel`,
which bypasses the prebuilt wheel glibc requirement. Requires Python 3.11+
and a C++ compiler (gcc/clang). CUDA toolkit is needed for GPU backend.

Manual install (same thing):

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive
pip install ./kt-kernel    # builds C++/CUDA kernels
pip install .              # installs the ktransformers wrapper
```

CPU-only mode works with AVX-512 or AMX (Intel Xeon 4th gen+).
AVX2-only CPUs (AMD EPYC) work but slower.

## Step 3: Download model weights

### Smaller models (laptops, desktops, 16-96 GB RAM)

These run on CPU-only machines via llama.cpp. No GPU required.

#### Gemma-4-12B (12B dense, Aug 2026)

```
litmoe install --model gemma-4-12b
```

| Quant | Size | RAM needed | Fits |
|---|---|---|---|
| UD-IQ2_M | 4 GB | ~8 GB | 16 GB laptop |
| Q4_K_M | 7 GB | ~10 GB | 16 GB laptop |
| Q8_0 | 13 GB | ~16 GB | 16 GB laptop |
| BF16 | 24 GB | ~28 GB | 32 GB |

Google's latest 12B. Multimodal (text + image). Runs on any laptop.

#### Gemma-4-31B (31B dense, Aug 2026)

```
litmoe install --model gemma-4-31b
```

| Quant | Size | RAM needed | Fits |
|---|---|---|---|
| UD-IQ2_XXS | 9 GB | ~12 GB | 16 GB laptop |
| UD-Q4_K_XL | 19 GB | ~24 GB | 32 GB |
| Q8_0 | 33 GB | ~40 GB | 64 GB |
| BF16 | 61 GB | ~70 GB | 96 GB |

Google's latest 31B. Most capable model that fits a 16 GB laptop at IQ2_XXS.

#### Llama-4-Scout (109B total, 17B active MoE, 2026)

```
litmoe install --model llama-4-scout
```

| Quant | Size | RAM needed | Fits |
|---|---|---|---|
| Q3_K_M | 52 GB | ~60 GB | 64 GB |
| Q4_K_M | 65 GB | ~75 GB | 96 GB |
| Q6_K | 88 GB | ~96 GB | 96 GB (tight) |

Meta's latest MoE. 109B total but only 17B active per token — fast inference,
high capability. 16 experts.

#### DeepSeek-V4-Flash (MoE, 256 experts top-6, Jul 2026)

```
litmoe install --model deepseek-v4-flash
```

| Quant | Size | RAM needed | Fits |
|---|---|---|---|
| UD-IQ1_S | 83 GB | ~90 GB | 96 GB |
| UD-IQ1_M | 87 GB | ~95 GB | 96 GB (tight) |
| UD-Q2_K_XL | 97 GB | ~110 GB | 128 GB |
| UD-Q4_K_XL | 155 GB | ~170 GB | 192 GB |

DeepSeek's newest compact MoE. 43 layers, 4096 hidden, 256 experts.
Outperforms V4-Pro despite smaller size, per DeepSeek's benchmarks.

### Large models (128 GB+ RAM)

#### MiniMax-M3 (428B total, 23B active MoE, 2026)

```
litmoe install --model minimax-m3
```

| Quant | Size | RAM needed | Fits |
|---|---|---|---|
| UD-IQ1_M | 128 GB | ~140 GB | 128 GB (tight) / 192 GB |
| UD-Q2_K_XL | 143 GB | ~160 GB | 192 GB |
| UD-Q4_K_M | 264 GB | ~290 GB | 374 GB |
| Q8_0 | 453 GB | ~500 GB | 512 GB |

Engine: llama.cpp (native, merged 2026-07-26). ktransformers also supports M3
(via SGLang + KT-Kernel, requires SM90 GPU).

Most accessible trillion-scale model. At 128 GB IQ1_M, fits on a Mac Studio
with 128 GB unified memory or a workstation with 192 GB RAM.

#### Kimi-K3 (2.78T total, 93B active MoE, Aug 2026)

```
litmoe install --model kimi-k3
```

| Quant | Size | RAM needed | Fits |
|---|---|---|---|
| UD-IQ1_S | 594 GB | ~650 GB | 768 GB machine |
| UD-IQ1_M | 649 GB | ~700 GB | 768 GB machine |
| UD-Q2_K_XL | 861 GB | ~950 GB | 1 TB machine |

Engine: llama.cpp only. ktransformers does not support K3.

#### Qwen3.8-2.4T (2.4T total, 95B active MoE, 2026)

```
litmoe install --model qwen3.8
```

| Quant | Size | RAM needed | Fits |
|---|---|---|---|
| UD-Q1_0 | 397 GB | ~440 GB | 512 GB machine |
| UD-IQ1_S | 508 GB | ~560 GB | 768 GB machine |
| UD-IQ1_M | 564 GB | ~620 GB | 768 GB machine |

Engine: llama.cpp only. ktransformers does not support Qwen3.8-2.4T.

Note: Qwen3.8-27B (a smaller variant of the same architecture) has several
open bugs in llama.cpp (CUDA lockups, long-context crashes). The 2.4T variant
shares the same architecture (Qwen3_5MoeForCausalLM) but has no reported
issues specific to it. Test before relying on it in production.

## Hardware requirements and expected performance

Memory determines whether a model loads. Throughput determines whether
it's usable. These are different things — a model that "fits in RAM"
can still be too slow for interactive use.

### What "usable" means

- **Interactive chat**: >5 tokens/second. You can have a conversation.
- **Batch processing**: 0.5-5 tokens/second. Usable for one-shot
  completions, summaries, code generation — not back-and-forth chat.
- **Impractical**: <0.5 tokens/second. Each response takes minutes.
  Only viable for offline batch jobs.

### Measured throughput on this project's hardware

AMD EPYC 7B13 (24 physical cores, 48 SMT, AVX2 only, 377 GB DDR4-3200,
no GPU, Google Cloud PersistentDisk ~379 MB/s random / ~778 MB/s
sequential).

| Model | Type | Quant | Size | t/s | Usability |
|---|---|---|---|---|---|
| Kimi-K3 (2.78T) | MoE 93B active | UD-IQ1_S | 594 GB | 0.85 | Impractical |
| DeepSeek-V4-Flash (~150B) | MoE ~30B active | UD-IQ1_S | 83 GB | 0.33 | Impractical |
| Kimi-Linear-48B | MoE 3B active | Q4_K_M | 30 GB | 0.58 | Impractical |

All three measured via llama.cpp, CPU-only, no GPU, in Docker containers.

**Finding: no MoE model is usable for interactive chat on this hardware.**
Even Kimi-Linear-48B with only 3B active parameters is 0.58 t/s —
not dramatically faster than the 594 GB Kimi-K3. The bottleneck is not
active compute (which is small) but loading 256 expert weight matrices
through CPU memory bandwidth and cloud disk on every token.

Prefill (prompt processing) is faster: 16-28 t/s for Kimi-Linear-48B.
But generation is 0.58 t/s because each generated token requires loading
8 of 256 experts from memory.

### Measured throughput from ktransformers docs (GPU hardware)

These are from ktransformers' official tutorials, not measured by us.

| Model | Hardware | t/s | Source |
|---|---|---|---|
| Kimi-K2 | 1 GPU + 600 GB RAM | ~10 | ktransformers Kimi-K2 tutorial |
| DeepSeek-V3 | 1 GPU + 382 GB RAM | 10-16 | ktransformers tutorial |
| DeepSeek-R1 | 8xL20 + Xeon | 227 | ktransformers README |
| SmallThinker-21B | 1 GPU + 42 GB RAM | ~26 | ktransformers tutorial |

### What works for interactive chat

The only way to get >5 t/s on CPU-only hardware is a dense model
(no MoE expert routing). Based on llama.cpp community benchmarks
for similar hardware (not measured here):

| Model type | Size | t/s on THIS machine | Usability here |
|---|---|---|---|
| 7-9B dense | 5-10 GB | ~10-20 t/s (est.) | Interactive chat |
| 12B dense | 7-18 GB | ~5-10 t/s (est.) | Interactive chat |
| 27-32B dense | 16-55 GB | ~2-5 t/s (est.) | Slow chat |
| Any MoE >30B | varies | 0.3-0.85 t/s (measured) | Impractical |

The tradeoff: dense models have fewer total parameters than MoE models
of equivalent quality, so you get less capability per token but actual
conversation speed.

No dense model has been benchmarked on this machine yet — only estimates
from llama.cpp community benchmarks for similar hardware.

### Kimi-K3 and Qwen3.8-2.4T — hardware table

These are trillion-parameter models. They require datacenter-class
hardware for interactive use.

| Hardware | RAM needed (IQ1_S) | Est. t/s | Usability |
|---|---|---|---|
| CPU only (this machine) | 594 GB (K3) / 508 GB (Qwen3.8) | ~0.85 (measured) | Impractical |
| CPU only, local NVMe | same | ~5-10 (est.) | Batch processing |
| 1 GPU, 24 GB VRAM | ~570 GB (K3) | ~1-3 (est.) | Batch processing |
| 2 GPU, 160 GB VRAM | ~434 GB (K3) | ~3-8 (est.) | Batch / slow chat |
| 4 GPU, 320 GB VRAM | ~274 GB (K3) | ~5-15 (est.) | Interactive chat |
| 8 GPU, 640 GB VRAM | ~0 GB extra | ~15-50 (est.) | Interactive chat |

RAM numbers are arithmetic (GGUF size minus VRAM). t/s numbers are
estimates based on scaling from the measured 0.85 t/s data point.
No GPU measurements exist in this project.

### MiniMax-M3 — hardware table

| Hardware | RAM needed (IQ1_M) | Est. t/s | Usability |
|---|---|---|---|
| CPU only, 128 GB RAM | 128 GB | not measured | Unknown |
| CPU only, 192 GB RAM | 128 GB | not measured | Unknown |
| 1 GPU, 24 GB VRAM | ~104 GB | not measured | Unknown |
| 2 GPU, 48 GB VRAM | ~80 GB | not measured | Unknown |
| Mac Studio, 192 GB unified (Metal) | 128 GB | not measured | Unknown |

M3 has not been benchmarked in this project. At 428B params / 23B
active, it is 5x smaller than K3 — but whether that translates to
usable speed on CPU is unverified. The ktransformers M3 tutorial
targets 8x H20 GPUs.

### DeepSeek-V4-Flash — hardware table

| Hardware | RAM needed (IQ1_S) | t/s | Usability |
|---|---|---|---|
| CPU only (this machine) | 83 GB | 0.33 (measured) | Impractical |
| CPU only, local NVMe | 83 GB | not measured | Unknown |
| 1 GPU, 24 GB VRAM | ~60 GB | not measured | Unknown |
| 4 GPU, 320 GB VRAM | ~0 GB extra | not measured | Unknown |

V4-Flash at 0.33 t/s on this machine is not usable for chat despite
fitting comfortably in 378 GB RAM. The bottleneck is expert loading
from cloud disk, not RAM capacity.

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

| Model | llama.cpp | ktransformers | Default quant | Size | Measured t/s (CPU) |
|---|---|---|---|---|---|
| Gemma-4-12B | Yes (native) | No | Q4_K_M | 7 GB | not measured |
| Gemma-4-31B | Yes (native) | No | UD-Q4_K_XL | 19 GB | not measured |
| Llama-4-Scout | Yes (native) | No | Q4_K_M | 65 GB | not measured |
| DeepSeek-V4-Flash | Yes (native) | No | UD-IQ1_S | 83 GB | 0.33 (impractical) |
| MiniMax-M3 | Yes (native) | Yes (SM90 GPU) | UD-IQ1_M | 128 GB | not measured |
| Kimi-K3 | Yes (native) | No | UD-IQ1_S | 594 GB | 0.85 (impractical) |
| Qwen3.8-2.4T | Yes (native) | No | UD-IQ1_S | 508 GB | not measured |
| DeepSeek-V3 | Yes | Yes (Linux+NVIDIA) | varies | ~600 GB | not measured |
| Kimi-K2 | Yes | Yes (Linux+NVIDIA, ~10 t/s) | Q4_K_M | ~600 GB | not measured |

Measured on AMD EPYC 7B13, 24 cores, AVX2, 377 GB RAM, no GPU, cloud disk.
ktransformers requires Linux + NVIDIA GPU (triton dependency).
llama.cpp works on Linux, macOS (Metal), and Windows (Vulkan/DirectML).

Sources: llama.cpp source (LLM_ARCH registrations, model .cpp files), ktransformers
optimize rules and tutorials, HuggingFace model configs and GGUF repositories. All
verified August 2026.