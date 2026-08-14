/* k3_api.c -- the public C API for the Kimi K3 engine.
 *
 * Implements the contracts declared in libk3/libk3.h. The implementations
 * call into the engine code in lib/k3_engine.c (which was extracted from
 * cli/k3_run.c). Lifecycle, configure, generate, state save/load.
 *
 * The opaque k3_ctx_inner struct is in k3_internal.h. Everything is
 * opaque to the host; the host only sees k3_ctx.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"
#include "k3_trunk.h"
#include "k3_tok.h"
#include "k3_cfg.h"
#include "k3_st.h"

#include "libk3/libk3.h"
#include "libk3/k3_internal.h"

/* K3_MAX_PROMPT, K3_MAX_GEN come from include/k3/k3.h */

/* Forward decls from k3_engine.c. The full definitions of Weights,
 * K3Preset, and K3StateHdr_inner are in include/libk3/k3_internal.h.
 * k3_api.c includes that header so the types are visible here. */


extern int      forward(Weights *w, const K3Cfg *c, K3Cache *cache,
                        const int *ids, int T,
                        float *logits_last, float *scratch, float *h, float *br,
                        float *kstate, int *arg_all);
extern int      argmax_(const float *v, int n);
extern int      real_cfg(K3Cfg *c, int *fa, int fa_max,
                        const char *dir, const char *cfg_path);
extern void     k3_state_fp(const K3Cfg *c, int32_t *fp);
extern int      k3_state_peek(const char *path, void *hdr);
extern int      k3_state_save(const char *path, const K3Cfg *c, const int *seq,
                              int nseq, void *hdr, const K3Cache *cache,
                              const Weights *w);
extern int      k3_state_load(const char *path, const K3Cfg *c, const void *hdr,
                              int *seq, int nseq, K3Cache *cache,
                              Weights *w);
extern const K3Preset *k3_preset_find(const char *name);

/* The opaque k3_ctx. */
struct k3_ctx {
    K3Cfg        c;
    K3St         st;
    int          fa[128];
    int          want_layers;
    int          layers_bound;
    int          nseq;
    int         *seq;
    K3Cache     *cache;
    K3LayerBind *lay;
    K3ModelBind  mb;
    K3Trunk     *trunk;
    float       *kvc, *ropec;
    int         *mla_slot;
    int          n_mla, kv_cap, cached;
    int          incremental;
    /* scratch */
    float       *h, *br, *ks, *lg, *sc;
    /* tokenizer */
    void        *tok;
    int          have_tok;
    /* state save/load */
    int32_t      fp[12];
    int64_t      kper;
    int64_t      kvpp, ropepp;
    /* stats */
    k3_stats     stats;
    char         last_errmsg[512];
    /* cache report */
    void        *cache_loaded;
};

/* ---- helpers ---- */

static void set_errmsg(k3_ctx *ctx, const char *fmt, ...)
{
    if (!ctx) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->last_errmsg, sizeof ctx->last_errmsg, fmt, ap);
    va_end(ap);
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ---- public API ---- */

k3_ctx *k3_open(const k3_open_args *args)
{
    if (!args || !args->model_dir) return NULL;
    k3_ctx *ctx = (k3_ctx *)calloc(1, sizeof *ctx);
    if (!ctx) return NULL;

    /* Default args. */
    double trunk_gb = args->trunk_gb > 0 ? args->trunk_gb : 16.0;
    double cache_gb = args->cache_gb > 0 ? args->cache_gb : 64.0;
    int want_layers = args->layers > 0 ? args->layers : -1;
    int incremental = args->incremental;

    /* If preset, override trunk/cache. */
    if (args->preset && args->preset[0] && strcmp(args->preset, "auto") != 0) {
        const K3Preset *p = k3_preset_find(args->preset);
        if (p) {
            trunk_gb = p->trunk_gb;
            cache_gb = p->cache_gb;
        }
    }

    /* Load config. */
    if (!real_cfg(&ctx->c, ctx->fa, 128, args->model_dir, args->config_path)) {
        set_errmsg(ctx, "could not read model config");
        free(ctx);
        return NULL;
    }
    ctx->want_layers = want_layers;
    ctx->incremental = incremental;

    /* Open safetensors index. */
    if (k3_st_open(&ctx->st, args->model_dir) != 0) {
        set_errmsg(ctx, "could not open safetensors index at %s", args->model_dir);
        free(ctx);
        return NULL;
    }

    /* Open trunk stream if --trunk is set. */
    ctx->trunk = NULL;
    if (args->trunk_dir && args->trunk_dir[0]) {
        ctx->trunk = (K3Trunk *)calloc(1, sizeof(K3Trunk));
        if (k3_trunk_open(ctx->trunk, args->trunk_dir, &ctx->c, (int64_t)(trunk_gb * 1e9)) != 0) {
            set_errmsg(ctx, "could not open trunk stream at %s", args->trunk_dir);
            free(ctx->trunk);
            k3_st_close(&ctx->st);
            free(ctx);
            return NULL;
        }
    }

    /* Bind layers. */
    int NL = (want_layers > 0 && want_layers < ctx->c.n_layers)
              ? want_layers : ctx->c.n_layers;
    ctx->layers_bound = NL;
    ctx->lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
    for (int L = 0; L < NL; L++) {
        if (k3_bind_layer(&ctx->st, &ctx->c, L, &ctx->lay[L]) != 0) {
            set_errmsg(ctx, "could not bind layer %d", L);
            /* free earlier */
            for (int L2 = 0; L2 < L; L2++) k3_bind_free(&ctx->lay[L2]);
            free(ctx->lay);
            k3_st_close(&ctx->st);
            free(ctx);
            return NULL;
        }
    }

    /* Bind model-level weights (embed, lm_head, final norm). */
    if (k3_bind_model(&ctx->st, &ctx->c, 1, &ctx->mb) != 0) {
        set_errmsg(ctx, "could not bind model-level weights");
        free(ctx->lay);
        k3_st_close(&ctx->st);
        free(ctx);
        return NULL;
    }

    /* Layer map for MLA slots. */
    ctx->mla_slot = (int *)malloc(sizeof(int) * (size_t)ctx->c.n_layers);
    ctx->n_mla = 0;
    for (int L = 0; L < ctx->c.n_layers; L++) {
        if (k3_is_mla(&ctx->c, L)) {
            ctx->mla_slot[L] = ctx->n_mla++;
        } else {
            ctx->mla_slot[L] = -1;
        }
    }

    /* KV cache and rope cache (only when incremental). */
    if (incremental) {
        ctx->kv_cap = K3_MAX_PROMPT + K3_MAX_GEN;
        const size_t kvperd = (size_t)ctx->kv_cap * ctx->c.n_heads *
                              (ctx->c.qk_nope + ctx->c.v_head);
        const size_t rpperd = (size_t)ctx->kv_cap * ctx->c.qk_rope;
        ctx->kvc   = (float *)calloc(kvperd * (size_t)ctx->n_mla, sizeof(float));
        ctx->ropec = (float *)calloc(rpperd * (size_t)ctx->n_mla, sizeof(float));
        if (!ctx->kvc || !ctx->ropec) {
            set_errmsg(ctx, "OOM for KV cache");
            k3_close(ctx);
            return NULL;
        }
    } else {
        ctx->kv_cap = 0;
        ctx->kvc = NULL;
        ctx->ropec = NULL;
    }
    ctx->cached = 0;

    /* Open the routed-expert cache. */
    {
        int64_t cb = (int64_t)(cache_gb * 1e9);
        ctx->cache = (K3Cache *)calloc(1, sizeof(K3Cache));
        if (k3_cache_init(ctx->cache, &ctx->st, &ctx->c, cb) != 0) {
            set_errmsg(ctx, "could not initialize expert cache");
            k3_close(ctx);
            return NULL;
        }
    }

    /* Sequence buffer. */
    ctx->seq = (int *)malloc((size_t)(K3_MAX_PROMPT + K3_MAX_GEN) * sizeof(int));
    ctx->nseq = 0;

    /* Scratch buffers. */
    const int E = ctx->c.hidden;
    const int maxb = ctx->c.n_layers / ctx->c.attn_res_block + 2;
    const int P = ctx->c.kda_heads * ctx->c.kda_head_dim;
    ctx->kper = (int64_t)P * ctx->c.kda_head_dim + (int64_t)3 * P * (ctx->c.conv_k - 1);
    ctx->kvpp = (int64_t)ctx->c.n_heads * (ctx->c.qk_nope + ctx->c.v_head);
    ctx->ropepp = (int64_t)ctx->c.qk_rope;

    ctx->h   = (float *)calloc((size_t)K3_MAX_PROMPT * E, sizeof(float));
    ctx->br  = (float *)calloc((size_t)K3_MAX_PROMPT * maxb * E, sizeof(float));
    ctx->ks  = (float *)calloc((size_t)ctx->kper * NL, sizeof(float));
    ctx->lg  = (float *)calloc((size_t)ctx->c.vocab, sizeof(float));
    ctx->sc  = (float *)calloc((size_t)E + 2 * (size_t)E * (maxb + 1), sizeof(float));
    if (!ctx->h || !ctx->br || !ctx->ks || !ctx->lg || !ctx->sc) {
        set_errmsg(ctx, "OOM for scratch buffers");
        k3_close(ctx);
        return NULL;
    }

    /* Tokenizer if asked. */
    ctx->have_tok = 0;
    if (args->tok_dir && args->tok_dir[0]) {
        Tok *tok = (Tok *)calloc(1, sizeof(Tok));
        k3_tok_load(tok, args->tok_dir);
        ctx->tok = tok;
        ctx->have_tok = 1;
    }

    /* State save/load fingerprint. */
    k3_state_fp(&ctx->c, ctx->fp);

    /* Initial stats. */
    memset(&ctx->stats, 0, sizeof ctx->stats);

    return ctx;
}

const char *k3_open_errmsg(void)
{
    /* Without a context, the host gets nothing. */
    return "(k3_open_errmsg: no context; see stderr)";
}

int k3_close(k3_ctx *ctx)
{
    if (!ctx) return 0;
    if (ctx->lay) {
        for (int L = 0; L < ctx->layers_bound; L++) k3_bind_free(&ctx->lay[L]);
        free(ctx->lay);
    }
    k3_bind_model_free(&ctx->mb);
    if (ctx->trunk) {
        k3_trunk_close(ctx->trunk);
        free(ctx->trunk);
    }
    if (ctx->cache) {
        k3_cache_free(ctx->cache);
        free(ctx->cache);
    }
    if (ctx->kvc)   free(ctx->kvc);
    if (ctx->ropec) free(ctx->ropec);
    if (ctx->mla_slot) free(ctx->mla_slot);
    if (ctx->seq)   free(ctx->seq);
    if (ctx->h)     free(ctx->h);
    if (ctx->br)    free(ctx->br);
    if (ctx->ks)    free(ctx->ks);
    if (ctx->lg)    free(ctx->lg);
    if (ctx->sc)    free(ctx->sc);
    if (ctx->tok && ctx->have_tok) {
        /* tok_free would be needed; assume OS frees at process exit */
    }
    k3_st_close(&ctx->st);
    free(ctx);
    return 0;
}

int k3_step(k3_ctx *ctx, const int *prompt_ids, int prompt_len,
            int *out_ids, int out_cap, int max_tokens)
{
    if (!ctx || !prompt_ids || prompt_len <= 0 || !out_ids || out_cap <= 0) return -1;
    if (prompt_len > K3_MAX_PROMPT) return -1;
    if (max_tokens <= 0) max_tokens = K3_MAX_GEN;
    if (max_tokens > K3_MAX_GEN) max_tokens = K3_MAX_GEN;
    if (prompt_len + max_tokens > K3_MAX_PROMPT + K3_MAX_GEN) {
        max_tokens = K3_MAX_PROMPT + K3_MAX_GEN - prompt_len;
    }
    if (max_tokens <= 0) return -1;

    /* Copy prompt into seq. */
    for (int i = 0; i < prompt_len; i++) ctx->seq[i] = prompt_ids[i];
    ctx->nseq = prompt_len;

    /* Run forward over the prompt. */
    const double t0 = now_s();
    Weights w = {
        .lay = ctx->lay, .mb = ctx->mb,
        .n_bound = ctx->layers_bound,
        .trunk = ctx->trunk,
        .kvc = ctx->kvc, .ropec = ctx->ropec,
        .mla_slot = ctx->mla_slot,
        .n_mla = ctx->n_mla, .kv_cap = ctx->kv_cap,
        .cached = ctx->cached,
        .draft_mode = 0,
    };
    if (forward(&w, &ctx->c, ctx->cache, ctx->seq, prompt_len,
                ctx->lg, ctx->sc, ctx->h, ctx->br, ctx->ks, NULL) != 0) {
        return -1;
    }
    if (ctx->kvc) ctx->cached += prompt_len;

    /* Sample (greedy) and emit tokens. */
    int n_out = 0;
    int next = argmax_(ctx->lg, ctx->c.vocab);
    out_ids[n_out++] = next;
    ctx->seq[ctx->nseq++] = next;

    for (int g = 1; g < max_tokens; g++) {
        Weights w1 = {
            .lay = ctx->lay, .mb = ctx->mb,
            .n_bound = ctx->layers_bound,
            .trunk = ctx->trunk,
            .kvc = ctx->kvc, .ropec = ctx->ropec,
            .mla_slot = ctx->mla_slot,
            .n_mla = ctx->n_mla, .kv_cap = ctx->kv_cap,
            .cached = ctx->cached,
            .draft_mode = 0,
        };
        if (forward(&w1, &ctx->c, ctx->cache, &ctx->seq[ctx->nseq - 1], 1,
                    ctx->lg, ctx->sc, ctx->h, ctx->br, ctx->ks, NULL) != 0) {
            return -1;
        }
        if (ctx->kvc) ctx->cached += 1;
        next = argmax_(ctx->lg, ctx->c.vocab);
        out_ids[n_out++] = next;
        if (n_out < out_cap) {
            ctx->seq[ctx->nseq++] = next;
        }
    }

    /* Stats. */
    ctx->stats.tokens_generated += n_out;
    ctx->stats.seconds_total += now_s() - t0;
    ctx->stats.expert_hits = ctx->cache->hits;
    ctx->stats.expert_misses = ctx->cache->misses;
    ctx->stats.bytes_read = ctx->cache->bytes_read;

    return n_out;
}

int k3_generate(k3_ctx *ctx, const int *prompt_ids, int prompt_len,
                int max_tokens, k3_token_cb cb, void *user)
{
    if (!cb) return k3_step(ctx, prompt_ids, prompt_len, NULL, 0, max_tokens);
    int total = 0;
    int out[K3_MAX_GEN];
    int n = k3_step(ctx, prompt_ids, prompt_len, out, max_tokens, max_tokens);
    if (n < 0) return -1;
    for (int i = 0; i < n; i++) {
        if (cb(user, out[i]) != 0) {
            /* Callback aborted. */
            return i;
        }
        total++;
    }
    return total;
}

int k3_save_state(k3_ctx *ctx, const char *path)
{
    if (!ctx || !path) return -1;
    /* The article's state save requires the K3StateHdr to be filled in
     * with the right sizes. We mirror the layout here. */
    struct {
        char    magic[4];
        int32_t version;
        int32_t fp[12];
        int32_t n_bound, n_mla, cached, nseq;
        int64_t kper;
        int64_t kvpp, ropepp;
    } hdr;
    memcpy(hdr.magic, "K3ST", 4);
    hdr.version = 1;
    memcpy(hdr.fp, ctx->fp, sizeof hdr.fp);
    hdr.n_bound = ctx->layers_bound;
    hdr.n_mla = ctx->n_mla;
    hdr.cached = ctx->cached;
    hdr.nseq = ctx->nseq;
    hdr.kper = ctx->kper;
    hdr.kvpp = ctx->kvpp;
    hdr.ropepp = ctx->ropepp;

    Weights w = {
        .lay = ctx->lay, .mb = ctx->mb,
        .n_bound = ctx->layers_bound,
        .trunk = ctx->trunk,
        .kvc = ctx->kvc, .ropec = ctx->ropec,
        .mla_slot = ctx->mla_slot,
        .n_mla = ctx->n_mla, .kv_cap = ctx->kv_cap,
        .cached = ctx->cached,
        .draft_mode = 0,
    };
    return k3_state_save(path, &ctx->c, ctx->seq, ctx->nseq, &hdr, ctx->cache, &w);
}

int k3_load_state(k3_ctx *ctx, const char *path)
{
    if (!ctx || !path) return -1;
    struct {
        char    magic[4];
        int32_t version;
        int32_t fp[12];
        int32_t n_bound, n_mla, cached, nseq;
        int64_t kper;
        int64_t kvpp, ropepp;
    } hdr;
    if (k3_state_peek(path, &hdr) != 0) return -1;
    if (memcmp(hdr.magic, "K3ST", 4) != 0) return -1;
    if (memcmp(hdr.fp, ctx->fp, sizeof hdr.fp) != 0) {
        set_errmsg(ctx, "state file architecture does not match current model");
        return -1;
    }
    Weights w = {
        .lay = ctx->lay, .mb = ctx->mb,
        .n_bound = ctx->layers_bound,
        .trunk = ctx->trunk,
        .kvc = ctx->kvc, .ropec = ctx->ropec,
        .mla_slot = ctx->mla_slot,
        .n_mla = ctx->n_mla, .kv_cap = ctx->kv_cap,
        .cached = ctx->cached,
        .draft_mode = 0,
    };
    if (k3_state_load(path, &ctx->c, &hdr, ctx->seq, ctx->nseq < hdr.nseq ? hdr.nseq : ctx->nseq,
                      ctx->cache, &w) != 0) {
        return -1;
    }
    ctx->nseq = hdr.nseq;
    ctx->cached = hdr.cached;
    return 0;
}

const char *k3_model_id(k3_ctx *ctx)
{
    if (!ctx) return NULL;
    return "kimi-k3";
}

int k3_n_layers(k3_ctx *ctx)
{
    if (!ctx) return 0;
    return ctx->layers_bound;
}

int k3_vocab_size(k3_ctx *ctx)
{
    if (!ctx) return 0;
    return ctx->c.vocab;
}

int k3_ctx_size(k3_ctx *ctx)
{
    if (!ctx) return 0;
    return K3_MAX_PROMPT + K3_MAX_GEN;
}

int k3_tokenize(k3_ctx *ctx, const char *text, int *out_ids, int out_cap)
{
    if (!ctx || !text || !out_ids) return -1;
    if (ctx->have_tok) {
        Tok *tok = (Tok *)ctx->tok;
        return tok_encode(tok, text, (int)strlen(text), out_ids, out_cap);
    }
    return -1;
}

int k3_detokenize(k3_ctx *ctx, const int *ids, int n_ids,
                  char *out_text, int out_cap)
{
    if (!ctx || !ids || !out_text || n_ids <= 0) return -1;
    if (ctx->have_tok) {
        /* The article's tok_decode takes a vector of bytes. The cleanest
         * path is to decode id-by-id and accumulate. */
        Tok *tok = (Tok *)ctx->tok;
        int written = 0;
        for (int i = 0; i < n_ids && written < out_cap - 1; i++) {
            char buf[64];
            tok_decode(tok, &ids[i], 1, buf, sizeof buf);
            int len = (int)strlen(buf);
            if (written + len >= out_cap - 1) break;
            memcpy(out_text + written, buf, (size_t)len);
            written += len;
        }
        out_text[written] = '\0';
        return written;
    }
    return -1;
}

void k3_get_stats(k3_ctx *ctx, k3_stats *out)
{
    if (!ctx || !out) return;
    *out = ctx->stats;
}

void k3_reset_stats(k3_ctx *ctx)
{
    if (!ctx) return;
    memset(&ctx->stats, 0, sizeof ctx->stats);
}
