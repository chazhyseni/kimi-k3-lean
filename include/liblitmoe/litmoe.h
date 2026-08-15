/* liblitmoe.h -- public C API for the article's Kimi K3 engine.
 *
 * WHAT THIS IS
 *   A small, stable surface for opening the model, running forward steps,
 *   generating tokens, and saving/loading state. The article's engine lives
 *   in src/core, src/io, src/cache, src/model, src/tokenizer; this header
 *   just exposes a few entry points so a host (the CLI in cli/k3_run.c, or
 *   a Python OpenAI server via ctypes) can drive it without depending on
 *   the internal layout.
 *
 * NOT IN THIS HEADER
 *   Anything kernel-level (`k3_matmul_mxfp4`, `k3_kda_layer`, etc.) is NOT
 *   here. The kernels are a private implementation detail and their signatures
 *   change. The model-level operations are here; they are the contract.
 *
 * DESIGN
 *   - The pointer is opaque. The host never sees the layer bindings, the
 *     cache, the trunk stream, or the recurrent state. That way the engine
 *     can change internally without breaking the host.
 *   - K3 step is one token in, one token out. The article's `forward()`
 *     takes an arbitrary prompt length and an argmax-channel, but the
 *     per-token interface is what an HTTP server actually needs. We support
 *     prefill (`k3_step` with prompt_len > 1) and incremental decode by
 *     caching state between calls.
 *   - State save/load maps to the article's `--save-state` / `--load-state`,
 *     which is verified bit-identical to full-recompute on the released
 *     checkpoint.
 *   - Streaming is via a callback. The host passes a function that gets called
 *     after each generated token; returning non-zero aborts generation.
 *     This is the same shape the article's main() uses internally.
 *
 * ERRORS
 *   Every function returns 0 on success and a negative number on failure.
 *   The failure path is sticky: once `k3_open` returns NULL or `k3_step`
 *   returns negative, subsequent calls on the same `k3_ctx` are no-ops and
 *   return negative. The host should check, then `k3_close`.
 *
 * THREAD SAFETY
 *   One `k3_ctx` per thread. The article's engine is single-caller inside
 *   one context; the OpenMP parallelism is inside the call. Two threads on
 *   the same `k3_ctx` will race; two threads on two `k3_ctx` is fine.
 */
#ifndef LIBK3_H
#define LIBK3_H

#include <stddef.h>
#include <stdint.h>

/* Engine version, mirrored from the article's CHANGELOG. */
#ifndef K3_VERSION
#define K3_VERSION "1.0.0"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* forward decl: the host never sees the contents. */
typedef struct k3_ctx k3_ctx;

/* Config knobs that affect how we open, not how we run. Mirrors the article's
 * --preset, --trunk, --cache-gb, --incremental, --tok. Defaults are sane. */
typedef struct k3_open_args {
    const char *model_dir;     /* required: path to the safetensors shards */
    const char *trunk_dir;     /* optional: packed trunk (scripts/pack-trunk.sh) */
    const char *config_path;   /* optional: config.json override */
    const char *tok_dir;       /* optional: tiktoken.model directory; needed for text-in */
    int         layers;        /* optional: bind only first N layers; -1 = all */
    double      cache_gb;      /* routed-expert cache budget; 0 = engine picks */
    double      trunk_gb;      /* trunk ring/pinned-layer budget; 0 = engine picks */
    const char *preset;        /* optional: laptop|desktop|workstation|server|max|auto */
    int         incremental;   /* 1 = carry KV cache + recurrent state */
} k3_open_args;

/* Callback for streaming tokens. Return 0 to continue, non-zero to abort. */
typedef int (*k3_token_cb)(void *user, int token_id);

/* ---- lifecycle ----
 *
 * k3_open reads the model config, opens the experts and the trunk stream
 * (if --trunk), binds the layers, and returns an opaque context. The
 * article's k3_cfg_load() refuses a config it cannot fully understand;
 * the host should distinguish that from a missing checkpoint by checking
 * stderr, since k3_open does not differentiate.
 *
 * Returns NULL on failure. The host can call k3_open_errmsg() after for
 * a static string. */
k3_ctx *k3_open(const k3_open_args *args);
const char *k3_open_errmsg(void);
int k3_close(k3_ctx *ctx);

/* ---- per-step ----
 *
 * k3_step feeds `prompt_ids[0..prompt_len-1]` and returns the next-token
 * sequence in `out_ids` (capacity out_cap). The first out token is the
 * argmax of the model after the prompt; successive tokens are generated
 * with greedy decode. Returns the number of tokens written, or negative
 * on failure.
 *
 * On both prefill and incremental this is the entry point. The article's
 * engine has a single forward() that does either path; the model-level
 * API does not have to expose that distinction.
 *
 * max_tokens: hard cap on generated tokens. 0 means "use the article's
 * default" (K3_MAX_GEN, very large). */
int k3_step(k3_ctx *ctx, const int *prompt_ids, int prompt_len,
            int *out_ids, int out_cap, int max_tokens);

/* ---- streaming ----
 *
 * k3_generate does the same as k3_step, but emits each generated token
 * via the callback. The callback's first token is the one after the
 * prompt; the prompt itself is not emitted. The callback may abort by
 * returning non-zero, which is the SSE stop signal.
 *
 * Returns the number of tokens generated, or negative on failure. */
int k3_generate(k3_ctx *ctx, const int *prompt_ids, int prompt_len,
                int max_tokens, k3_token_cb cb, void *user);

/* ---- state save / load ----
 *
 * Maps to --save-state / --load-state. The article says: "carries the
 * recurrent state and KV cache to disk so a second turn resumes instead
 * of re-reading the whole prompt." The state file is opaque; the article
 * guards against restore-from-different-architecture.
 *
 * Returns 0 on success, negative on failure. */
int k3_save_state(k3_ctx *ctx, const char *path);
int k3_load_state(k3_ctx *ctx, const char *path);

/* ---- introspection ----
 *
 * These let the host report model metadata in the OpenAI /v1/models
 * endpoint. The strings are static and persistent; the host should
 * copy them if it needs to keep them past the lifetime of the ctx. */
const char *k3_model_id(k3_ctx *ctx);
int         k3_n_layers(k3_ctx *ctx);
int         k3_vocab_size(k3_ctx *ctx);
int         k3_ctx_size(k3_ctx *ctx);

/* ---- tokenization ----
 *
 * Encoding is the article's `tok_encode` (BPE from tiktoken.model).
 * Decoding is the inverse. Both return the number of items written,
 * or negative on failure. `out` must be sized by the host; sizes are
 * the article's K3_MAX_PROMPT (8192) and K3_MAX_GEN (1024).
 *
 * k3_tokenize returns len(out_ids) on success, or negative. */
int k3_tokenize(k3_ctx *ctx, const char *text, int *out_ids, int out_cap);
int k3_detokenize(k3_ctx *ctx, const int *ids, int n_ids,
                  char *out_text, int out_cap);

/* ---- decode statistics ----
 *
 * After a forward step, the article's cache reports hits, misses, bytes
 * read. The host uses these for the OpenAI `waste`-extension telemetry
 * field. All counters are zeroed at k3_open time.
 *
 * Populated by k3_step / k3_generate. Reads are not synchronized across
 * threads; one host thread per k3_ctx. */
typedef struct k3_stats {
    uint64_t expert_hits;
    uint64_t expert_misses;
    uint64_t bytes_read;
    uint64_t tokens_generated;
    double   seconds_total;
} k3_stats;

void k3_get_stats(k3_ctx *ctx, k3_stats *out);
void k3_reset_stats(k3_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBK3_H */
