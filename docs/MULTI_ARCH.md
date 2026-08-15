# Multi-Architecture Engine Design

## Goal

Support both Kimi K3 (2.8T) and Qwen3 MoE (2.4T) in one repo, sharing
the infrastructure that's already architecture-agnostic, with clean
separation for the parts that aren't.

## What's shared (zero changes needed)

These files have no architecture assumptions and work as-is for both models:

| File | What it does | Why it's generic |
|------|-------------|-----------------|
| `src/cache/k3_cache.c` | LRU expert cache | Caches "expert N at layer L" — no arch knowledge |
| `src/io/k3_trunk.c` | Trunk streaming | Streams + pins layers by index — no arch knowledge |
| `src/io/k3_st.c` | Safetensors reader | Reads any safetensors file — no arch knowledge |
| `src/io/k3_load.c` | Checkpoint loader | Loads shards + builds index — no arch knowledge |
| `src/tokenizer/k3_tok.h` | BPE tokenizer | tiktoken-format, works for both (both use BPE) |
| `serve/server.py` | HTTP server | OpenAI Chat Completions — model-agnostic |
| `serve/api.py` | Request/response shaping | OpenAI API — model-agnostic |
| `serve/chatfmt.py` | Chat formatting | Message flattening — model-agnostic |
| `serve/engine.py` | ctypes wrapper | Calls libk3.so — model-agnostic |
| `deploy/` | Caddy + gateway + router | HTTP routing — model-agnostic |
| `bootstrap.sh` | Install script | Clone + build + serve — model-agnostic |
| `kimi-k3-lean` | CLI launcher | Start/stop/chat — model-agnostic |

## What's architecture-specific

| Component | Kimi K3 | Qwen3 MoE | Shared kernel? |
|-----------|---------|-----------|---------------|
| Attention | KDA (69 layers) + MLA (24 layers) | GQA + RoPE (all layers) | No |
| Activation | SiTU-GLU | SwiGLU | No |
| KV cache | MLA: expanded latent cache. KDA: O(1) recurrent state | INT4 quantized KV cache | No |
| Residual | AttnRes block (every 12 layers) | Standard additive residual | No |
| Config | K3Cfg with kda/mla/situ fields | Q3Cfg with gqa/rope/swiglu fields | No (but same reader pattern) |
| Tensor names | `block_sparse_moe.gate.weight` etc. | `mlp.gate_proj.weight` etc. | No |
| Expert format | MXFP4 (4-bit packed) | BF16 (or quantize to MXFP4 at convert) | `k3_matmul_mxfp4` + `k3_matmul_bf16` both exist |
| Position encoding | KDA: none. MLA: NoPE (rope dim exists but unused) | RoPE (rotary) | No |
| Dense FFN | SiTU-GLU activation | SwiGLU activation | No |

## Architecture: shared interface + arch plugins

```
                   ┌──────────────────────────────────────┐
                   │         libk3 public API              │
                   │  k3_open, k3_step, k3_generate,      │
                   │  k3_tokenize, k3_save_state, etc.    │
                   │  (include/libk3/libk3.h — UNCHANGED)  │
                   └──────────────┬───────────────────────┘
                                  │
                   ┌──────────────▼───────────────────────┐
                   │       k3_engine.c (dispatch)         │
                   │  reads config.json → detects arch     │
                   │  selects ArchOps vtable               │
                   └──────────────┬───────────────────────┘
                          ┌────────┴────────┐
                   ┌──────▼──────┐  ┌───────▼───────┐
                   │  kimi_ops.c │  │  qwen_ops.c   │
                   │  (existing  │  │  (new)        │
                   │   k3_ops.c) │  │               │
                   │             │  │  GQA+RoPE     │
                   │  KDA, MLA   │  │  SwiGLU       │
                   │  SiTU-GLU   │  │  INT4 KV      │
                   │  ShortConv  │  │  Standard res │
                   │  AttnRes    │  │               │
                   └──────┬──────┘  └───────┬───────┘
                          │                 │
                   ┌──────▼─────────────────▼───────┐
                   │     SHARED INFRASTRUCTURE        │
                   │  k3_cache.c  (LRU expert cache)  │
                   │  k3_trunk.c  (trunk streaming)   │
                   │  k3_st.c     (safetensors I/O)   │
                   │  k3_load.c   (checkpoint loader) │
                   │  k3_tok.h    (BPE tokenizer)     │
                   └─────────────────────────────────┘
```

## The ArchOps vtable

```c
// include/k3/arch.h — NEW
typedef struct arch_ops {
    const char *name;                    // "kimi-k3" or "qwen3-moe"

    // Config
    int  (*cfg_load)(void *cfg, int *fa, int fa_max,
                     void *json_root, const char *path);
    size_t (*cfg_size)(void);            // sizeof(K3Cfg) or sizeof(Q3Cfg)

    // Tensor binding
    int64_t (*bind_layer_bytes)(const void *st, const void *cfg, int L);
    int     (*bind_layer)(const void *st, const void *cfg, int L,
                          void *layer_bind);
    size_t  (*layer_bind_size)(void);

    // Forward pass
    size_t  (*layer_scratch)(const void *cfg, int T);
    void    (*decoder_layer)(float *h, float *block_residual, int *n_blocks,
                             const void *layer_w, const void *cfg,
                             int layer_idx, int T,
                             float *state, float *scratch,
                             float *kvc, float *ropec,
                             int cached, int cap);

    // State
    size_t  (*state_per_layer)(const void *cfg);
    size_t  (*kv_per_pos)(const void *cfg);    // 0 for KDA, >0 for GQA/MLA

    // Introspection
    int     (*n_layers)(const void *cfg);
    int     (*vocab_size)(const void *cfg);
    int     (*ctx_size)(const void *cfg);
} ArchOps;

// One global per arch, selected at k3_open time:
extern const ArchOps kimi_k3_ops;
extern const ArchOps qwen3_moe_ops;
```

## Config detection

At `k3_open` time, read `config.json` and check for architecture markers:

```c
// In k3_engine.c:
const char *detect_arch(jval *config_root) {
    // Kimi K3: has "linear_attn_config" or "kda_num_heads"
    jval *lin = json_get(config_root, "linear_attn_config");
    jval *kda = json_get(config_root, "kda_num_heads");
    if (lin || kda) return "kimi-k3";

    // Kimi K3 nested: text_config.linear_attn_config
    jval *txt = json_get(config_root, "text_config");
    if (txt) {
        jval *lin2 = json_get(txt, "linear_attn_config");
        if (lin2) return "kimi-k3";
    }

    // Qwen3: has "num_key_value_heads" and no KDA fields
    jval *gqa = json_get(config_root, "num_key_value_heads");
    if (gqa) return "qwen3-moe";

    // Fallback: check for Kimi's block_sparse_moe in tensor names
    return NULL;  // unknown
}
```

## Qwen3 new files

```
src/
├── arch/
│   ├── arch.h              # ArchOps vtable (new)
│   ├── kimi_ops.c          # wraps existing k3_ops.c (thin shim, ~50 lines)
│   └── qwen_ops.c          # Qwen3 kernels (new, ~600 lines)
├── core/
│   └── k3_ops.c            # existing Kimi kernels (UNCHANGED)
├── model/
│   ├── k3_bind.c           # existing Kimi binding (UNCHANGED)
│   └── q3_bind.c           # Qwen3 tensor binding (new, ~200 lines)
├── io/
│   └── ...                 # all generic (UNCHANGED)
├── cache/
│   └── k3_cache.c          # generic (UNCHANGED)
└── lib/
    ├── k3_engine.c          # dispatch + forward (modified ~100 lines)
    └── k3_api.c             # public API (UNCHANGED)
```

## Qwen3 kernel implementation plan

### qwen_ops.c (~600 lines)

1. **GQA attention** (~250 lines)
   - Q/K/V projections via existing `k3_matmul_bf16`
   - RoPE: precompute sin/cos tables, apply to Q and K
   - Causal masked attention with INT4 KV cache
   - INT4 pack/unpack: 2 elements per byte, per-8-element scale (FP8)
   - Output projection via `k3_matmul_bf16`
   - KV cache grows per position, quantized to INT4 at write time

2. **SwiGLU** (~20 lines)
   ```c
   void q3_swiglu(float *out, const float *x, int n) {
       for (int i = 0; i < n; i++)
           out[i] = silu(x[i]) * x[i + n];  // gate * up
   }
   // then k3_mmw(out, result, down_proj, ...)
   ```

3. **Standard residual** (~10 lines)
   - No AttnRes block, no snapshot stack
   - Simple: `h = h + attn(h); h = h + mlp(h)`

4. **Layer dispatch** (~50 lines)
   ```c
   void q3_decoder_layer(float *h, const void *layer_w,
                         const void *cfg, int L, int T,
                         float *state, float *scratch,
                         float *kvc, float *ropec,
                         int cached, int cap) {
       // 1. RMSNorm (existing k3_rmsnorm)
       // 2. GQA attention (new, with INT4 KV cache)
       // 3. Residual add
       // 4. RMSNorm (existing k3_rmsnorm)
       // 5. MoE or dense SwiGLU (new SwiGLU + existing k3_moe)
       // 6. Residual add
   }
   ```

5. **INT4 KV cache** (~100 lines)
   - Pack: 2 KV values per byte + per-8-element FP8 scale
   - Unpack: dequantize on read during attention
   - Memory: 10 GB at 128K context (vs 40 GB BF16)
   - Allocation: part of the preset memory budget

### q3_bind.c (~200 lines)

Maps Qwen3 HF tensor names:
```
layers.N.self_attn.q_proj.weight     → q3_attn.q
layers.N.self_attn.k_proj.weight     → q3_attn.k  (n_kv_heads, not n_attn_heads)
layers.N.self_attn.v_proj.weight     → q3_attn.v
layers.N.self_attn.o_proj.weight     → q3_attn.o
layers.N.mlp.gate_proj.weight        → q3_mlp.gate (dense layers)
layers.N.mlp.up_proj.weight          → q3_mlp.up
layers.N.mlp.down_proj.weight        → q3_mlp.down
layers.N.block_sparse_moe.gate.weight → q3_moe.gate (MoE layers, if any)
```

### q3_cfg.h (~100 lines)

Qwen3 config reader:
```c
typedef struct {
    int hidden;              // e.g. 6144
    int n_layers;            // e.g. 80
    int vocab;               // e.g. 152064
    float rms_eps;
    int n_attn_heads;        // e.g. 64
    int n_kv_heads;          // e.g. 8 (GQA)
    int head_dim;            // e.g. 128
    float rope_theta;        // e.g. 1000000.0
    int max_position;        // e.g. 131072
    // MoE fields (shared structure with K3's MoE):
    int n_experts;
    int topk;
    int moe_inter;
    int first_dense;         // first_k_dense_replace
    int dense_inter;
} Q3Cfg;
```

## INT4 KV cache design

```
Per-position KV storage:
  ┌──────────────────────────────────────────┐
  │ scale[8]     FP8 scales (one per 8 vals) │
  │ packed[]     uint8, 2 INT4 values/byte   │
  └──────────────────────────────────────────┘

  K cache: [n_layers][n_kv_heads][head_dim][max_pos] packed
  V cache: [n_layers][n_kv_heads][head_dim][max_pos] packed

  Memory per position:
    2 (K+V) * n_kv_heads * head_dim * n_layers * 0.5 bytes (INT4)
    + small scale overhead (~1/16 of the data)
    ≈ 81,920 bytes/pos for 80 layers, 8 heads, 128 dim
    = 10 GB at 128K context

  Dequantize-on-read during attention:
    for each position p:
      for each kv_head h:
        unpack INT4 → FP32
        compute attention scores against Q

  The unpack is a hot loop but it's memory-bandwidth bound
  (reading from RAM, not NVMe), so it's fast.
```

## Memory budget per preset (Qwen3 2.4T)

| Preset | Trunk (GB) | Expert cache (GB) | KV cache (GB) @ 128K | Total |
|--------|-----------|-------------------|----------------------|-------|
| laptop | 3.0 | 1.0 | 2.5 (32K ctx) | 8.2 GB |
| desktop | 16.0 | 10.0 | 10.0 | 31.9 GB |
| workstation | 60.0 | 30.0 | 10.0 | ~95 GB |
| auto | fit | fit | fit | free RAM |

The preset system already trades trunk vs expert cache. KV cache
becomes a third budget item. The `auto` preset sizes it from free
RAM after trunk + expert cache are allocated.

## Build system changes

Makefile gains a new source group:

```makefile
ARCH_SRC := src/arch/kimi_ops.c src/arch/qwen_ops.c
ARCH_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(ARCH_SRC))

# q3_bind.c joins k3_bind.c
MODEL_SRC := src/model/k3_bind.c src/model/q3_bind.c
```

No #ifdefs. Both architectures are always compiled in. The vtable
dispatch at `k3_open` time selects which one runs.

## Convert tool changes

`tools/convert.py` gains a `--arch` flag:
- `--arch kimi-k3` (default, auto-detected): existing behavior
- `--arch qwen3-moe`: validates Qwen3 config fields, optional
  MXFP4 quantization of expert weights at convert time

The convert tool is the right place to do MXFP4 quantization if
we want the same 4-bit expert compression for Qwen3. The engine's
`k3_matmul_mxfp4` kernel works on any weights quantized to MXFP4
format, regardless of architecture.

## What changes in the public API

Nothing. `libk3.h` stays the same. `k3_open` detects the architecture
from config.json and selects the right vtable internally. The host
(Python server, CLI) doesn't know or care which architecture is running.

```python
# serve/engine.py — NO CHANGES
engine = Engine("/path/to/qwen3-checkpoint", preset="auto")
# k3_open reads config.json, sees num_key_value_heads, selects qwen3_moe_ops
# Everything else is identical from the Python side
```

## Testing

The 3-GATE oracle system already validates bit-identical output.
For Qwen3:

1. Build a tiny Qwen3 fixture (like `tests/fixtures/tiny_k3/`)
   with small hidden, few layers, same GQA structure
2. Write a `tools/make_q3_oracle.py` that produces reference logits
   using a Python reference implementation (PyTorch or a simple numpy GQA)
3. Test GATE 1: teacher forcing positions match
4. Test GATE 2: greedy decode matches
5. Test GATE 3: incremental decode with INT4 KV cache matches full recompute

The INT4 KV cache is the most critical thing to gate. The test must
prove that INT4-quantized KV produces the same tokens as BF16 KV
(within a tolerance, or ideally bit-identical for greedy decode since
INT4 quantization noise rarely changes the argmax).

## Effort estimate

| Component | Lines | Difficulty | Time |
|-----------|-------|------------|------|
| arch.h vtable | 80 | Easy | 2 hours |
| kimi_ops.c shim | 100 | Easy | 2 hours |
| qwen_ops.c (GQA+RoPE+SwiGLU+INT4 KV) | 600 | Medium | 3-5 days |
| q3_bind.c (tensor names) | 200 | Easy | 1 day |
| q3_cfg.h (config reader) | 100 | Easy | 2 hours |
| k3_engine.c dispatch | 100 | Medium | 1 day |
| tools/convert.py Qwen3 path | 100 | Easy | 2 hours |
| Tiny Qwen3 fixture + oracle | 300 | Medium | 2 days |
| 3-GATE tests for Qwen3 | 200 | Medium | 1 day |
| **Total** | **~1,800** | | **~2 weeks** |

The infrastructure (~4,000 lines) is reused with zero changes.
The new code is ~1,800 lines. The hardest part is the GQA+RoPE
kernel with INT4 KV cache, which is well-understood math with
proven implementations to reference (vLLM, llama.cpp, Flash Attention).