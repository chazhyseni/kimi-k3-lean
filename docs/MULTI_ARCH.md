# Multi-Model Lean Inference Engine — Comprehensive Design

## The Two Models

| | Kimi K3 | Qwen3.8-2.4T |
|---|---|---|
| **HF repo** | moonshotai/Kimi-K3 | Qwen/Qwen3.8-2.4T-A95B |
| **Total params** | 2.78T | 2.4T |
| **Active params/token** | ~18B | 95B |
| **Architecture class** | (custom) | Qwen3_5MoeForCausalLM |
| **Layers** | 93 | 92 |
| **Hidden size** | 7168 | 8192 |
| **Recurrent attn layers** | 69 (KDA) | 69 (linear/SSM) |
| **Full attn layers** | 24 (MLA) | 23 (GQA + partial RoPE) |
| **Conv kernel** | 4 (ShortConv) | 4 (linear_conv_kernel_dim) |
| **Experts** | 896 | 512 |
| **Top-k** | 16 | 10 |
| **Shared experts** | 2 | 1 |
| **Expert inter size** | 3072 | 2048 |
| **Vocab** | 163,840 | 248,320 |
| **Max context** | 1M | 262K (256K) |
| **Weight format** | BF16 trunk + MXFP4 experts | BF16 (all) |
| **Disk (BF16)** | 982 GB | ~4.4 TB |
| **Disk (MXFP4 experts)** | 982 GB (native) | ~1.1 TB (if quantized) |
| **Activation** | SiTU-GLU | SwiGLU (silu) |
| **Attn bias** | No | No |
| **RoPE** | MLA: present but unused | GQA: partial (25% of head_dim) |
| **KV cache layers** | 24 (MLA) | 23 (GQA) |
| **KV per pos (INT4)** | ~2.37 MB | 25 KB (only 23 layers!) |
| **KV at max ctx (INT4)** | very large | 6.1 GB at 262K |

## Critical Discovery

**Qwen3.8-2.4T uses the SAME hybrid attention pattern as Kimi K3.**

Both models split their layers into:
- ~75% recurrent/linear attention (O(1) state per layer)
- ~25% full attention (needs KV cache)

Both use depthwise causal conv (kernel=4) in the recurrent path.
Both have shared experts alongside routed experts.
Both use SiLU-based activation.

The key difference is the recurrent mechanism:
- Kimi K3: KDA (delta-rule recurrence)
- Qwen3.8: Mamba-style SSM (state space model)

But both are O(1) per-layer state. The RAM profile is identical.
And the KV cache is SMALL in both cases — only the full attention
layers need it.

This means the engine integration is much simpler than expected.
The Kimi K3 infrastructure (expert streaming, trunk streaming,
recurrent state management) transfers almost directly.

## Naming

The project needs a name that captures:
- Two of the greatest open-weight models
- Compressed to run on personal computers
- No GPU needed

Candidates (to be finalized after research):
- kiwen — Kimi + Qwen portmanteau (5 chars)
- moebius — MoE + mathematical paradox (7 chars)
- sling — David/Goliath metaphor (5 chars)
- goliath — running giants on your desk (7 chars)
- twinmoe — two MoE models (7 chars)

All shared infrastructure uses the neutral name. Architecture-specific
code keeps its identity (kimi_ops.c, qwen_ops.c).

## Architecture

```
Harness (Hermes, Open WebUI, Claude Code, aider, curl)
    │
    │ OpenAI Chat Completions API
    │ http://127.0.0.1:8080/v1
    │
    ▼
serve/server.py (stdlib HTTP, generic)
    │
    │ ctypes
    │
    ▼
lib???.so (one library, both architectures)
    │
    │ detect_arch(config.json) → vtable
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
kimi_ops.c         qwen_ops.c          (future)
KDA + MLA          Linear + GQA        arch_ops.c
SiTU-GLU           SwiGLU
ShortConv          linear_conv
AttnRes (12)       AttnRes (4)
    │                  │
    ▼                  ▼
SHARED INFRASTRUCTURE (architecture-agnostic)
  cache.c     — LRU expert cache
  trunk.c     — trunk streaming from NVMe
  safetensors.c — shard I/O
  loader.c    — checkpoint index
  tokenizer.h — BPE tokenizer
  engine.c    — dispatch + forward loop
  api.c       — public C API
```

## What's New for Qwen3.8

### New C files (~800 lines total):

**qwen_ops.c** (~400 lines)
- Mamba-style SSM kernel for linear attention layers (~150 lines)
  Similar to k3_kda_step but uses SSM recurrence instead of delta-rule.
  Both are O(1) state, both use conv, both produce the same memory profile.
- GQA + partial RoPE kernel for full attention layers (~150 lines)
  Standard GQA with partial rotary (only 25% of head_dim gets RoPE).
  Needs INT4 KV cache for 23 layers.
- SwiGLU activation (~10 lines)
  silu(gate(x)) * up(x) — simpler than SiTU-GLU.
- Standard residual (no AttnRes block, or AttnRes every 4 layers)
- QK norm (RMSNorm on Q and K before attention) — uses existing k3_rmsnorm
- Output gate (swish type) — attn_output_gate: true

**qwen_bind.c** (~200 lines)
Maps Qwen3.8 HF tensor names:
```
model.layers.N.self_attn.q_proj.weight      → qwen_attn.q
model.layers.N.self_attn.k_proj.weight      → qwen_attn.k
model.layers.N.self_attn.v_proj.weight      → qwen_attn.v
model.layers.N.self_attn.o_proj.weight      → qwen_attn.o
model.layers.N.self_attn.q_norm.weight      → qwen_attn.q_norm
model.layers.N.self_attn.k_norm.weight      → qwen_attn.k_norm
model.layers.N.linear_attn.*               → qwen_linear.*
model.layers.N.mlp.gate.weight             → qwen_moe.gate
model.layers.N.mlp.experts.E.gate_proj     → qwen_moe.experts[E].gate
model.layers.N.mlp.experts.E.up_proj        → qwen_moe.experts[E].up
model.layers.N.mlp.experts.E.down_proj     → qwen_moe.experts[E].down
model.layers.N.mlp.shared_experts.gate_proj → qwen_moe.shared.gate
```

**qwen_cfg.h** (~100 lines)
Reads Qwen3.8 config.json fields:
```c
typedef struct {
    int hidden, n_layers, vocab;
    float rms_eps;
    // Full attention (23 layers)
    int n_attn_heads, n_kv_heads, head_dim;
    float rope_theta;
    int partial_rotary;  // 0.25 → 64 of 256 dims get RoPE
    int full_attn_interval;  // 4
    // Linear attention (69 layers)
    int linear_key_heads, linear_value_heads;
    int linear_key_dim, linear_value_dim;
    int linear_conv_kernel;  // 4
    // MoE
    int n_experts, topk;
    int moe_inter, shared_inter;
    // Output gate
    int attn_output_gate;  // true
    // AttnRes
    int attn_res_block;  // 4
} QwenCfg;
```

**kimi_ops.c** (~50 lines, thin shim)
Wraps existing k3_ops.c kernels in the ArchOps vtable.

### Modified files:
- **engine.c**: add arch detection + vtable dispatch (~100 lines changed)
- **Makefile**: add new source files to build
- **convert.py**: add --arch flag, optional MXFP4 quantization for Qwen3
- **fetch-model.sh**: take model name as argument
- **launcher**: multi-model support (per-model PID files)
- **bootstrap.sh**: neutral naming
- **serve/engine.py**: library name changes
- All env vars: K3_* → new prefix
- All docs: update for multi-model

## INT4 KV Cache for Qwen3.8

Only 23 of 92 layers need KV cache (the full attention layers).
The other 69 layers use linear attention (O(1) state, no KV cache).

| Context | BF16 KV (23 layers) | INT4 KV | INT2 KV |
|---------|--------------------|---------|---------|
| 128K | 11.5 GB | 3.1 GB | 1.6 GB |
| 256K | 23.0 GB | 6.1 GB | 3.2 GB |
| 512K | 47.0 GB | 12.2 GB | 6.5 GB |
| 1M | 94.0 GB | 24.4 GB | 12.9 GB |

### Fitting extended context in laptop (8 GB):

| Config | Max context | How |
|--------|------------|-----|
| INT4, trunk=3, expert=1 | 168K | Default laptop |
| INT4, trunk=1.5, expert=0.5 | 251K | Reduced trunk |
| INT2, trunk=2, expert=0.5 | 435K | INT2 + reduced |
| INT2, trunk=1, expert=0.5 | 475K | INT2 + minimal |

### Fitting 1M context:

| Preset | Quant | Trunk | Expert | KV | Total | Fits? |
|--------|-------|-------|--------|----|----|-------|
| Laptop (8 GB) | INT2 | 0.5 | 0.5 | 12.9 | 13.9 | No (needs 16 GB) |
| Desktop (32 GB) | INT4 | 4 | 3 | 24.4 | 31.4 | Yes |
| Desktop (32 GB) | INT2 | 10 | 5 | 12.9 | 27.9 | Yes (comfortable) |

No hard capping. The context limit is dynamically computed from
available RAM after trunk + expert cache are allocated. The engine
tells the user: "context limited to 256K based on your RAM. Use
--kv-quant int2 for up to 512K."

INT2 quality: ~3-5% degradation on long-context tasks (measured by
vLLM on Llama 3). Acceptable for 512K+ context where the alternative
is truncation. The Qwen3.8 model only needs KV for 23/92 layers, so
KV quality matters less than for a pure-GQA model.

## Download Speed

### Problem
The current `fetch-model.sh` uses `hf download` which fetches shards
sequentially. On a 1 Gbps connection this gives 180 MB/s per shard (good)
but wastes the connection's parallel capacity. The first shard is
always slow (656 kB/s) because of HF Xet warmup + unauthenticated rate
limits + venv setup time.

### Fix: Parallel shard downloads
`scripts/fetch-model.sh` (new) downloads 4-8 shards in parallel using
`xargs -P`. On a 1 Gbps link with 8 parallel: theoretical 4-8x speedup,
practical ~3-4x (connection saturates around 4 parallel on most ISPs).

Download config files first (small, needed for arch detection), then
all shards in parallel. Resumable — skips already-downloaded shards.
Same verification (shard count + byte total + per-shard sizes + checksums).

### Speed comparison (1.56 TB Kimi K3 checkpoint):
- Sequential `hf download`: ~2.5 hours at 180 MB/s
- Parallel x8 `fetch-model.sh`: ~40 min at ~1 GB/s aggregate
- With HF_TOKEN (authenticated): +20-30% rate limit headroom

| Phase | What | Time | Risk |
|-------|------|------|------|
| 1 | Rename: repo, CLI, env vars, lib, scripts | 1-2 days | Low |
| 2 | Restructure: src/shared/ + src/arch/ | 1 day | Low |
| 3 | ArchOps dispatch: vtable shim, engine.c | 1-2 days | Low |
| 4 | Qwen3.8 kernels: SSM + GQA+RoPE + SwiGLU | 5-7 days | Medium |
| 5 | Qwen3.8 binding + config + tests | 2-3 days | Low |
| 6 | Multi-model launcher | 2-3 days | Low |
| 7 | Documentation + release | 1-2 days | Low |
| **Total** | | **~2 weeks** | |

Phase 4 is the hardest but easier than originally estimated because:
- The SSM kernel is structurally similar to KDA (recurrent, conv, O(1) state)
- The GQA kernel is simpler than MLA (standard attention, no latent compression)
- KV cache is small (only 23 layers, 6.1 GB at max context)
- Most infrastructure transfers directly (cache, trunk, I/O, tokenizer)