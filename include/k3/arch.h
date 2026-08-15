/* arch.h -- architecture dispatch interface for the multi-model engine.
 *
 * This is the seam between the shared infrastructure (cache, trunk, I/O,
 * HTTP server) and the architecture-specific kernels (attention, activation,
 * KV cache, tensor binding).
 *
 * At k3_open time, the engine reads config.json, detects the architecture
 * (Kimi K3 vs Qwen3 MoE vs future), and selects the matching ArchOps vtable.
 * All subsequent forward calls go through the vtable, so the shared code
 * never branches on architecture.
 *
 * ADDING A NEW ARCHITECTURE:
 *   1. Write arch_ops.c with the kernel implementations
 *   2. Write arch_bind.c with the tensor name mapping
 *   3. Write arch_cfg.h with the config reader
 *   4. Add an extern ArchOps to this file
 *   5. Add detection logic in k3_engine.c::detect_arch()
 *
 * The shared infrastructure (k3_cache.c, k3_trunk.c, k3_st.c, k3_load.c,
 * k3_tok.h) needs zero changes for a new architecture.
 */
#ifndef K3_ARCH_H
#define K3_ARCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------- ops */

typedef struct ArchOps {
    /* Human-readable name for /v1/models and logs. */
    const char *name;

    /* ---- config ---- */
    /* Load architecture config from a parsed JSON root. Returns 1 on
     * success, 0 on failure (with message to stderr). The config struct
     * is opaque to the caller; cfg_size() tells the caller how much to
     * allocate. */
    int    (*cfg_load)(void *cfg, int *layer_map, int map_max,
                       void *json_root, const char *whence);
    size_t (*cfg_size)(void);

    /* ---- tensor binding ---- */
    /* Compute bytes needed to bind one layer's weights from the checkpoint. */
    int64_t (*bind_layer_bytes)(const void *st, const void *cfg, int L);
    /* Bind one layer's weights. Fills the opaque layer_bind struct. */
    int    (*bind_layer)(const void *st, const void *cfg, int L, void *layer_bind);
    /* Size of the per-layer weight struct (K3LayerW, Q3LayerW, etc). */
    size_t (*layer_bind_size)(void);

    /* ---- forward pass ---- */
    /* Scratch buffer size for one layer's forward computation. */
    size_t (*layer_scratch)(const void *cfg, int T);
    /* One decoder layer. The block_residual / n_blocks parameters are
     * for Kimi's AttnRes; Qwen3 ignores them (passes NULL / 0). */
    void   (*decoder_layer)(float *h, float *block_residual, int *n_blocks,
                            const void *layer_w, const void *cfg,
                            int layer_idx, int T,
                            float *state, float *scratch,
                            float *kvc, float *ropec,
                            int cached, int cap);

    /* ---- state ---- */
    /* Per-layer recurrent state size (KDA state, GQA KV cache, etc).
     * For KDA this is the delta-rule matrix + conv history.
     * For GQA this is 0 (KV cache is separate, managed by the caller). */
    size_t (*state_per_layer)(const void *cfg);
    /* KV cache bytes per position per layer. 0 for recurrent attention
     * (KDA), >0 for cache-based attention (MLA, GQA). Used to size
     * the KV cache allocation. */
    size_t (*kv_per_pos)(const void *cfg);
    /* Whether this arch uses RoPE (Qwen3: yes, Kimi K3: no). */
    int    (*uses_rope)(const void *cfg);
    /* Whether this arch uses AttnRes block residuals. */
    int    (*uses_attn_res)(const void *cfg);

    /* ---- introspection ---- */
    int    (*n_layers)(const void *cfg);
    int    (*vocab_size)(const void *cfg);
    int    (*hidden_size)(const void *cfg);
    int    (*ctx_size)(const void *cfg);

    /* ---- model-level aggregator ---- */
    /* Some architectures have a final residual/aggregator beyond the
     * per-layer ones (Kimi K3's out_res). NULL if not used. */
    void   (*model_aggregator)(float *h, const void *model_w,
                               const void *cfg, int T, float *scratch);
} ArchOps;

/* ---- registered architectures ---- */
extern const ArchOps kimi_k3_ops;
extern const ArchOps qwen3_moe_ops;

/* Detect architecture from a parsed config.json. Returns the ArchOps
 * pointer, or NULL if unrecognized. */
const ArchOps *arch_detect(const void *json_root);

#ifdef __cplusplus
}
#endif

#endif /* K3_ARCH_H */