/* k3_internal.h -- private header that exposes the engine internals (formerly
 * static in cli/k3_run.c) to the libk3 shared library.
 *
 * This is internal: the public API is in libk3/libk3.h. Nothing outside
 * the libk3 shared library should include this header.
 *
 * The split is: cli/k3_run.c keeps the CLI's main() and arg parsing. The
 * rest of the model-level code (open, weights, forward, save/load, presets)
 * has been moved to lib/k3_engine.c and is exposed here.
 */
#ifndef LIBK3_K3_INTERNAL_H
#define LIBK3_K3_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include "libk3/libk3.h"  /* for k3_ctx */
#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"
#include "k3_trunk.h"
#include "k3_tok.h"
#include "k3_cfg.h"

/* The engine's Weights struct. Defined in k3_engine.c as `Weights`.
 * Defined here so k3_api.c can populate it. */
typedef struct {
    K3LayerBind *lay;
    K3ModelBind  mb;
    int          n_bound;
    K3Trunk     *trunk;
    float       *kvc, *ropec;
    int         *mla_slot;
    int          n_mla, kv_cap, cached;
    int          draft_mode;
} Weights;

/* K3Preset is the engine's preset table. Defined in k3_engine.c. */
typedef struct { const char *name; double trunk_gb, cache_gb; const char *note; } K3Preset;

/* K3StateHdr is the engine's state file header. */
typedef struct {
    char    magic[4];
    int32_t version;
    int32_t fp[12];
    int32_t n_bound, n_mla, cached, nseq;
    int64_t kper;
    int64_t kvpp, ropepp;
} K3StateHdr_inner;

/* ---- opaque handles ---- */
struct k3_ctx_inner {
    K3Cfg        c;
    K3St         st;
    int          fa[128];       /* layer map; sized for the released 24 MLA layers */
    int          want_layers;   /* -1 = bind all */
    int          layers_bound;  /* number actually bound */
    int          lm_head;       /* 1 if lm_head was loaded */
    K3LayerBind *lay;           /* per-layer bindings */
    K3ModelBind  mb;
    K3Trunk     *trunk;         /* may be NULL */
    K3Cache     *cache;         /* routed-expert cache */
    float       *kvc, *ropec;  /* KV cache + rope cache, only when incremental */
    int         *mla_slot;
    int          n_mla, kv_cap, cached;
    int          incremental;
    int          nseq;          /* current sequence length */
    int         *seq;           /* up to K3_MAX_PROMPT + K3_MAX_GEN */
    /* scratch buffers */
    float       *h, *br, *ks, *lg, *sc;
    /* tokenizer */
    Tok          tok;
    int          have_tok;
    /* state save/load header */
    int32_t      fp[12];
    int64_t      kper;
    int64_t      kvpp, ropepp;
    /* stats */
    k3_stats     stats;
    char         last_errmsg[512];
};

/* Forward declaration: the engine implementation in lib/k3_engine.c. */
int  k3_engine_open(k3_ctx *ctx, const k3_open_args *args);
int  k3_engine_step(k3_ctx *ctx, const int *prompt_ids, int prompt_len,
                    int *out_ids, int out_cap, int max_tokens);
int  k3_engine_save_state(k3_ctx *ctx, const char *path);
int  k3_engine_load_state(k3_ctx *ctx, const char *path);
void k3_engine_close(k3_ctx *ctx);
int  k3_engine_tokenize(k3_ctx *ctx, const char *text, int *out_ids, int out_cap);
int  k3_engine_detokenize(k3_ctx *ctx, const int *ids, int n_ids,
                          char *out_text, int out_cap);

/* K3_MAX_PROMPT and K3_MAX_GEN come from include/k3/k3.h */

#endif /* LIBK3_K3_INTERNAL_H */
