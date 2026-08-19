# Methodology

This document explains why litmoe is structured the way it is: a thin
Python dispatcher over ktransformers and llama.cpp, with no custom inference code.

## The problem

Kimi K3 and other trillion-parameter MoE models are open-source but unusable
on most hardware. The HuggingFace checkpoint is 1.45 TB. To run interactively
you need either:

1. **A GPU machine**: A single H100 ($25k-$40k) gets you 5-15 tokens/second.
2. **More RAM than disk is cheap:** 1 TB DDR4 lets you hold the model in memory.
3. **Local NVMe** (7 GB/s vs 379 MB/s cloud disk) makes cold-cache fast.
4. **An engine that supports your hardware**: ktransformers for AMX/AVX-512/AVX2
   + CUDA, llama.cpp for CUDA/HIP/Metal/Vulkan.

The previous version of litmoe tried to be the fifth option: a custom
CPU-only C99 forward pass. It was 0.019 t/s on a 24-core EPYC. The math:

- 67 prompt tokens × 92 MoE layers × 16 experts = 98,496 expert lookups
- Each expert is 17.55 MB; ~50% dedup = ~859 GB to read from disk
- At 379 MB/s disk: 38 minutes minimum
- 24 cores × ~50 ms per expert compute: 82 minutes compute floor

No software optimization closes a 1000x gap to ktransformers and llama.cpp.
We tried AVX2 matmul, mmap, cross-layer prefetch, 2-bit quantization — all
shipped but all irrelevant. The bandwidth doesn't exist.

## What the dispatcher does instead

The dispatcher acknowledges that other people have spent years building inference
engines and uses them. Two open-source projects cover everything:

| Engine | Hardware | Strength |
|---|---|---|
| **ktransformers** (Tsinghua MADSys Lab, SOSP 2025) | CUDA + AMX + AVX-512 + AVX2 | Heterogeneous CPU+GPU MoE, expert offloading |
| **llama.cpp** | CUDA + HIP + Metal + Vulkan + SYCL | Mature cross-platform, every quant format |

litmoe is the front door: a Python package that:

1. Reads a `models.yaml` config.
2. Starts the chosen engine as a subprocess (`kt run` or `llama-server`).
3. Exposes a single OpenAI/Anthropic-compatible API on port 8080.
4. Routes `/v1/chat/completions` requests to the right engine by model name.

That's it. No custom forward pass. No CUDA kernels. No safetensors parsing.

## Why a dispatcher is the right shape

**Inference engines are mature.** ktransformers hit SOSP 2025 with a
heterogeneous-expert scheduler; llama.cpp ships 1.5-bit to 8-bit quantization
across every GPU vendor. The optimization space is enormous and competition
between these engines is healthy. Reimplementing kernels loses to both.

**Engines already speak HTTP.** Both `kt run` (sglang-kt backend) and
`llama-server` ship OpenAI-compatible servers. The gateway is a pass-through.

**Configuration is the hard part.** Users don't care which engine is running;
they care which model responds. The dispatcher lets a single `models.yaml`
mix engines: kimi-k3 → llama.cpp (native GGUF support), deepseek-v3 →
ktransformers (native AMX/AVX optimization), tiny test model → ktransformers
CPU. The user writes `model: kimi-k3` and gets a response.

**Inference is hardware-bound, not software-bound.** The previous "optimization"
work (cross-layer prefetch, 2-bit quantization, mmap advisor, fused matmul)
was a series of single-digit-percent improvements on a fundamentally
bandwidth-limited problem. The dispatcher makes that work unnecessary: pick
the right engine for the hardware and let it do what it's good at.

## What was measured

Hardware: AMD EPYC 7B13 (24 physical cores, 377 GB DDR4-3200, no GPU,
Google Cloud PersistentDisk at ~379 MB/s random / ~778 MB/s sequential read).

| Engine | Mode | Tokens/sec | Notes |
|---|---|---|---|
| Previous C99 AVX2 forward pass | CPU | 0.019 | 158s TTFT for 4-token prompt; thread stuck in DISK SLEEP |
| llama.cpp (BF16 trunk, IQ1_S experts) | CPU | 0.85 | 1.17 s/token measured in this environment |
| llama.cpp + Q2_0 expert quant | CPU | not measured in this env | the disk math says 1.4 t/s is a theoretical max |
| ktransformers AVX2 CPU backend | CPU | not measured (no installation) | ktransformers docs: comparable to llama.cpp |
| ktransformers sglang-kt GPU backend | GPU | 5-50 t/s typical | needs GPU machine; 8x L20 = 87.58 t/s concurrent |
| llama.cpp CUDA + Q2_K | GPU | 5-30 t/s typical | depends on VRAM size |

The previous engine was 45x slower than llama.cpp on the same hardware doing
the same thing. The gap to GPU is 100-1000x. There's no path from the custom
C engine to "interactive inference on this VM."

## What you get

- **CPU machine, AVX2 only:** ktransformers AVX2 backend, ~0.5-1 t/s for K3.
  Viable for batch processing, not for chat.
- **GPU machine:** ktransformers sglang-kt, 5-50 t/s. Viable for chat.
- **Anything else:** llama.cpp. Works on CUDA/HIP/Metal/Vulkan.

Pick the engine per model in `models.yaml`. Both engines speak OpenAI HTTP.
The dispatcher adds latency in the single-digit-millisecond range and never
touches the forward pass.

## What litmoe does NOT do

- It is not an inference engine. There are no model weights in this repo and
  no forward-pass code. The actual inference is done by subprocesses.
- It does not optimize for specific hardware. That's the engines' job.
- It does not quantize models. That's `kt quant`, `llama-quantize`, or
  third-party tools (Unsloth, MLX, etc.).
- It does not parallelize across machines. Single-node only.

## What's in this repo

```
litmoe/
├── pyproject.toml          # modern Python package
├── litmoe/
│   ├── config.py           # Pydantic models.yaml schema
│   ├── server.py           # FastAPI OpenAI/Anthropic gateway
│   ├── engines/
│   │   ├── base.py         # Engine abstract base
│   │   ├── ktransformers.py  # kt run subprocess adapter
│   │   └── llamacpp.py     # llama-server subprocess adapter
│   └── cli/
│       ├── main.py         # litmoe doctor|init|install|serve|status|stop
│       └── install.py      # one-command engine + model installer
├── examples/
│   └── models.yaml         # engine routes
├── deploy/
│   └── docker-compose.yml  # gateway + caddy + openwebui
├── docs/
│   ├── SETUP.md           # installation and hardware requirements
│   ├── METHODOLOGY.md      # this file
│   ├── ARCHITECTURE.md     # architecture diagram
│   └── architecture.svg    # rendered diagram
```

## What was learned along the way

Engineering lessons from building the previous C engine (kept for honesty, not for re-use):

1. **Don't compete with mature engines.** ktransformers and llama.cpp have
   teams of PhDs. Your CPU forward pass is not a viable alternative.
2. **Hardware bottlenecks don't yield to software.** A 50x compute gap to
   llama.cpp on identical hardware means your optimization is wrong, not
   the hardware.
3. **"Measure" beats "design".** The CPU floor we calculated (82 min/response)
   was confirmed by actual wall-clock measurement (158s TTFT) only after we'd
   shipped several rounds of unmeasured "optimizations".
4. **The real product is integration.** Users want OpenAI-format APIs over
   multiple models on multiple hardware. That is what this dispatcher does.

These lessons apply generally. The repo is the artifact that follows from them.
