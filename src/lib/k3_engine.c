/* k3_run.c - run the REAL Kimi K3, all 93 layers, from the released checkpoint.
 *
 * WHAT THIS IS
 *   The full engine: safetensors index over 96 shards, resident trunk bound by name,
 *   routed experts streamed from disk through an LRU cache and multiplied straight out
 *   of MXFP4. Greedy decode. Token ids in, token ids out.
 *
 * MEMORY. The banner this program prints before allocating is a PLAN, not a measurement.
 *   It reports requested budgets rather than actual reservations, and in practice it
 *   OVERSTATES: across the 12-rung ladder in docs/data/ the planned total exceeded
 *   measured peak RSS by 0.13-1.84 GB, because both budgets round down to whole slots and
 *   that rounding outweighs the safetensors index it omits. Quote the "PEAK RSS" line
 *   instead, which comes from
 *   getrusage after the run. Fully resident, the weights are 108.81 GB of bf16 trunk plus
 *   4.70 GB of embed and lm_head, so 113.49 GB; streamed, the resident set is whatever
 *   budget is given, down to about 8.2 GB. The 1.45 TB of routed experts is never
 *   resident at any budget.
 *
 * THIS ENGINE IS I/O BOUND at small budgets and roughly balanced at large ones. The
 *   measured I/O share runs 40.9%-60.6% across the 12-rung ladder (docs/data/), dropping
 *   below 50% at 96 GB and above. The "I/O share" line printed at the end of every run
 *   reports it for that run. Going faster still means moving fewer bytes before it means
 *   computing less, which is why docs/TUNING.md is mostly about allocation.
 *
 * DECODE STRATEGY
 *   By default each step re-runs the whole prefix rather than carrying state forward.
 *   That is O(T^2), but it is the path the full-model oracle validates in
 *   tests/unit/k3_model.c. --incremental switches to prefill-then-one-token-at-a-time,
 *   carrying the KDA recurrent state and an MLA KV cache. GATE 3 of the tiny-model
 *   oracle requires it to produce the SAME tokens as full recompute, so the equivalence
 *   is tested rather than assumed. Context is limited by the MLA KV cache
 *   (~2.37 MB/position), not by array sizes; the engine computes the requirement up
 *   front and refuses the run if it will not fit.
 *
 * COMMAND LINE
 *   usage() below is the single source of truth for options and defaults; `k3 --help`
 *   prints it. It is not duplicated here, because a second copy is a second thing to
 *   keep correct and the copy is the one that goes stale.
 */
#define _POSIX_C_SOURCE 200809L
/* _POSIX_C_SOURCE alone hides the BSD rusage fields, ru_maxrss among them, from
 * <sys/resource.h> on Darwin. peak_rss_bytes() below needs it. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"
#include "k3_trunk.h"
#include "k3_tok.h"   /* text in/out; the --ids path never touches it */
#include "k3_cfg.h"   /* read the checkpoint's own config rather than assuming it */

#include "libk3/k3_internal.h"  /* Weights, K3Preset, K3StateHdr_inner */

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void human(double b, char *o, size_t n)
{
    const char *u[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0; while (b >= 1000.0 && i < 4) { b /= 1000.0; i++; }
    snprintf(o, n, "%.2f %s", b, u[i]);
}

/* The released constants, kept ONLY as a fallback for runs against a shard directory
 * that has no config.json (partial fixtures, hand-assembled trunks). Every value here
 * matches the released config.json, but a hardcoded table cannot notice a checkpoint
 * revision -- so k3_cfg_load_file() is preferred whenever a config is present, and this
 * path announces itself loudly rather than passing for the real thing. */
static void real_cfg_hardcoded(K3Cfg *c, int *fa)
{
    memset(c, 0, sizeof *c);
    c->hidden = 7168;  c->n_layers = 93;   c->vocab = 163840; c->rms_eps = 1e-5f;
    c->kda_heads = 96; c->kda_head_dim = 128; c->conv_k = 4;  c->gate_lb = -5.0f;
    c->n_heads = 96;   c->q_lora = 1536;   c->kv_lora = 512;
    c->qk_nope = 128;  c->qk_rope = 64;    c->v_head = 128;   c->mla_out_gate = 1;
    c->n_experts = 896; c->topk = 16;      c->n_shared = 2;
    c->latent = 3584;  c->moe_inter = 3072; c->routed_scale = 1.0f;
    c->moe_renorm = 1; c->latent_norm = 1;
    c->first_dense = 1; c->dense_inter = 33792;
    c->attn_res_block = 12;
    c->situ_b1 = 4.0f; c->situ_b2 = 25.0f;
    int n = 0;
    for (int i = 4; i <= 93; i += 4) fa[n++] = i;     /* config lists are ONE-based */
    fa[n++] = 93;
    c->n_full_attn = n; c->full_attn = fa;
}

/* Prefer the checkpoint's own config; fall back only when there is none.
 * cfg_path may be NULL, in which case <shard_dir>/config.json is tried.
 * Returns 1 on success, 0 if a config was found but could not be trusted -- and in
 * that case the caller MUST abort rather than fall back: a config that was found but
 * could not be parsed is evidence that the checkpoint is not what the fallback table
 * describes, which is exactly when the fallback is most dangerous. */
int real_cfg(K3Cfg *c, int *fa, int fa_max,
                    const char *shard_dir, const char *cfg_path)
{
    char guess[4096];
    if (!cfg_path) {
        snprintf(guess, sizeof guess, "%s/config.json", shard_dir);
        FILE *probe = fopen(guess, "rb");
        if (probe) { fclose(probe); cfg_path = guess; }
    }
    if (cfg_path) return k3_cfg_load_file(c, fa, fa_max, cfg_path);

    real_cfg_hardcoded(c, fa);
    printf("config: NO config.json found under %s\n"
           "        falling back to the built-in Kimi K3 constants (93 layers, 24 MLA).\n"
           "        These match the released checkpoint but are NOT read from it; pass\n"
           "        --config PATH to validate against the real file.\n", shard_dir);
    return 1;
}

int argmax_(const float *v, int n)
{ int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

/* ------------------------------------------------------- conversation state ----
 * Everything the engine carries between tokens, on disk. The point is turn two of a
 * conversation: without this, resuming re-reads the whole prompt through all 93 layers,
 * which on a streamed trunk costs minutes; with it, a resumed session pays only for the
 * tokens actually new.
 *
 * Three things are carried, and only three: the KDA recurrent matrices plus ShortConv
 * history (fixed size, independent of context), the MLA KV cache, and the shared rope
 * rows. The AttnRes block buffer is NOT carried because forward() clears it on entry and
 * rebuilds it from the layer outputs every pass; saving it would be saving scratch.
 *
 * The KV cache is stored position-major inside each MLA layer's slice, so only the
 * OCCUPIED positions are written and a resumed run may size its cache differently. The
 * header carries a config fingerprint: restoring state built by a different architecture
 * would produce fluent, wrong output with nothing to indicate it, which is the one
 * failure mode this engine refuses to have. */
#define K3_STATE_MAGIC "K3ST"
#define K3_STATE_VER   1

/* K3StateHdr_inner is in include/libk3/k3_internal.h */

void k3_state_fp(const K3Cfg *c, int32_t *fp)
{
    fp[0] = c->hidden;      fp[1] = c->n_layers;  fp[2]  = c->vocab;
    fp[3] = c->kda_heads;   fp[4] = c->kda_head_dim; fp[5] = c->conv_k;
    fp[6] = c->n_heads;     fp[7] = c->qk_nope;   fp[8]  = c->qk_rope;
    fp[9] = c->v_head;      fp[10] = c->n_experts; fp[11] = c->topk;
}

#define K3_SPEC_MAX 8
/* Longest-suffix n-gram drafting for --spec: if the last n ids (n=3, then 2) already
 * appeared earlier in the sequence, propose the ids that followed them there. Costs
 * nothing when it misses: no draft means the step runs exactly as without --spec. The
 * drafts are PROPOSALS only; batched greedy verification accepts precisely the prefix
 * the model itself would have emitted, so the output stream is identical to serial
 * decode by construction, and the A/B gate checks it. */
/* Reads only the header, so the caller can size buffers before committing to a load. */
int k3_state_peek(const char *path, K3StateHdr_inner *hd)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    const size_t got = fread(hd, 1, sizeof *hd, f);
    fclose(f);
    if (got != sizeof *hd || memcmp(hd->magic, K3_STATE_MAGIC, 4) != 0) {
        fprintf(stderr, "%s is not a k3 state file\n", path);
        return -1;
    }
    if (hd->version != K3_STATE_VER) {
        fprintf(stderr, "%s is state version %d, this build writes %d\n",
                path, hd->version, K3_STATE_VER);
        return -1;
    }
    return 0;
}

int k3_state_load(const char *path, const K3Cfg *c, const K3StateHdr_inner *hd,
                         int *seq, float *ks, float *kvc, float *ropec,
                         int n_bound, int n_mla, int kv_cap)
{
    int32_t fp[12];
    k3_state_fp(c, fp);
    if (memcmp(fp, hd->fp, sizeof fp) != 0) {
        fprintf(stderr, "REFUSING: %s was written by a different model architecture.\n"
                        "  Restoring it would produce fluent, wrong output.\n", path);
        return -1;
    }
    if (hd->n_bound != n_bound || hd->n_mla != n_mla) {
        fprintf(stderr, "REFUSING: %s holds %d bound layers and %d MLA layers, "
                        "this run has %d and %d\n",
                path, hd->n_bound, hd->n_mla, n_bound, n_mla);
        return -1;
    }
    if (hd->cached > kv_cap) {
        fprintf(stderr, "REFUSING: %s holds %d positions, this run's KV cache is %d.\n"
                        "  Raise --gen or shorten the prompt.\n", path, hd->cached, kv_cap);
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    if (fseek(f, (long)sizeof *hd, SEEK_SET) != 0) { fclose(f); return -1; }

    int rc = 0;
    if (fread(seq, sizeof(int), (size_t)hd->nseq, f) != (size_t)hd->nseq) rc = -1;
    if (!rc && fread(ks, sizeof(float), (size_t)hd->kper * n_bound, f)
               != (size_t)hd->kper * (size_t)n_bound) rc = -1;
    /* Position-major inside each layer slice, so a differently-sized destination cache
     * is written slice by slice rather than as one block. */
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = kvc + (size_t)mi * kv_cap * hd->kvpp;
        const size_t n = (size_t)hd->cached * hd->kvpp;
        if (fread(dst, sizeof(float), n, f) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = ropec + (size_t)mi * kv_cap * hd->ropepp;
        const size_t n = (size_t)hd->cached * hd->ropepp;
        if (fread(dst, sizeof(float), n, f) != n) rc = -1;
    }
    fclose(f);
    if (rc) fprintf(stderr, "%s is truncated\n", path);
    return rc;
}

int k3_state_save(const char *path, const K3Cfg *c, const int *seq, int nseq,
                         const float *ks, const float *kvc, const float *ropec,
                         int n_bound, int n_mla, int kv_cap, int cached,
                         int64_t kper, int64_t kvpp, int64_t ropepp)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    K3StateHdr_inner hd;
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, K3_STATE_MAGIC, 4);
    hd.version = K3_STATE_VER;
    k3_state_fp(c, hd.fp);
    hd.n_bound = n_bound; hd.n_mla = n_mla; hd.cached = cached; hd.nseq = nseq;
    hd.kper = kper; hd.kvpp = kvpp; hd.ropepp = ropepp;

    int rc = 0;
    if (fwrite(&hd, sizeof hd, 1, f) != 1) rc = -1;
    if (!rc && fwrite(seq, sizeof(int), (size_t)nseq, f) != (size_t)nseq) rc = -1;
    if (!rc && fwrite(ks, sizeof(float), (size_t)kper * n_bound, f)
               != (size_t)kper * (size_t)n_bound) rc = -1;
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = kvc + (size_t)mi * kv_cap * kvpp;
        const size_t n = (size_t)cached * kvpp;
        if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = ropec + (size_t)mi * kv_cap * ropepp;
        const size_t n = (size_t)cached * ropepp;
        if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
    }
    if (fclose(f) != 0) rc = -1;
    if (rc) fprintf(stderr, "failed writing %s\n", path);
    return rc;
}

static int spec_draft(const int *seq, int T, int cap, int *out)
{
    /* Evidence-gated: a draft only fires when the suffix n-gram's occurrences AGREE on
     * what follows. Measured on the released checkpoint, an eager most-recent-match
     * drafter went 0.91x on code: partial acceptances pay a replay sweep, so weak
     * drafts are worse than no drafts. Rules: match length 4 (then 3); if the n-gram
     * occurred more than once, every occurrence must propose the same next id, and the
     * draft stops at the first position where historical continuations diverge. */
    if (cap > K3_SPEC_MAX) cap = K3_SPEC_MAX;
    for (int n = 4; n >= 3; n--) {
        if (T < n + 1) continue;
        int m1 = -1, m2 = -1;                            /* two most recent matches */
        for (int j = T - n - 1; j >= 0; j--) {
            int hit = 1;
            for (int i = 0; i < n; i++)
                if (seq[j + i] != seq[T - n + i]) { hit = 0; break; }
            if (!hit) continue;
            if (m1 < 0) m1 = j;
            else { m2 = j; break; }
        }
        if (m1 < 0) continue;
        int nd = 0;
        for (int i = 0; nd < cap && m1 + n + i < T; i++) {
            const int cand = seq[m1 + n + i];
            if (m2 >= 0) {
                /* stop where the two histories stop agreeing */
                if (m2 + n + i >= m1 || seq[m2 + n + i] != cand) break;
            }
            out[nd++] = cand;
        }
        if (nd > 0) return nd;
    }
    return 0;
}

/* K3_VERSION is in libk3/libk3.h */

static void usage(FILE *f)
{
    fprintf(f,
"k3 " K3_VERSION ", Kimi K3 inference engine\n"
"\n"
"usage: k3 <model_dir> [options]\n"
"\n"
"prompt (exactly one):\n"
"  --prompt TEXT         tokenize TEXT and run it\n"
"  --prompt-file PATH    read the prompt from a file; use this for non-ASCII, since\n"
"                        argv is re-encoded by the shell\n"
"  --ids 1,2,3           raw token ids; the reproducible channel used by the tests\n"
"\n"
"memory:\n"
"  --preset NAME         auto | laptop | desktop | workstation | server | max\n"
"                        auto sizes both budgets from this machine's free RAM,\n"
"                        trunk-first; also spelled --trunk-gb auto\n"
"  --list-presets        show each preset's split and expected speed\n"
"  --trunk DIR           packed trunk directory; enables streaming (see scripts/)\n"
"  --trunk-gb X          trunk ring / pinned-layer budget\n"
"  --cache-gb X          routed-expert cache budget\n"
"\n"
"generation:\n"
"  --gen N               tokens to generate (default 8)\n"
"  --incremental         carry KV cache and recurrent state between tokens\n"
"  --save-state PATH     write the carried state after the run, so the next turn of a\n"
"                        conversation resumes instead of re-reading the whole prompt\n"
"  --load-state PATH     resume from a saved state; the prompt given now is treated as\n"
"                        the CONTINUATION of the saved sequence. Needs --incremental\n"
"  --draft-trunk DIR     hybrid decode: a second packed trunk (typically a quantized\n"
"                        derivation of the real one, see tools/qdq_trunk.py) DRAFTS\n"
"                        tokens which the exact model verifies in batched sweeps.\n"
"                        Output remains exactly the exact model's greedy decode; the\n"
"                        draft only proposes. Needs --incremental; implies --spec 4\n"
"  --draft-trunk-gb X    trunk budget for the draft model (default 6)\n"
"  --spec N              speculative decode: draft up to N tokens by n-gram lookup and\n"
"                        verify them in ONE batched sweep. Output is identical to\n"
"                        serial decode by construction; needs --incremental. An extra\n"
"                        verified position costs ~22%% of a serial token when the trunk\n"
"                        streams, so repetitive text decodes up to several times faster\n"
"  --tok DIR             directory with tiktoken.model and tokenizer_config.json\n"
"\n"
"diagnostics:\n"
"  --config PATH         model config; defaults to <model_dir>/config.json\n"
"  --layers N            bind only the first N layers (partial shard sets)\n"
"  --dump-logits PATH    write float32 logits for the first step\n"
"  --dump-cache-trace D  write expert_hist.json and expert_trace.bin into D, for\n"
"                        offline analysis with tools/sim_cache.py\n"
"  --out FILE            JSON results (default k3_run.json)\n"
"  --version, --help\n"
"\n"
"Memory is a dial, not a floor: the same model runs in 8 GB and in 224 GB and produces\n"
"identical output. Give memory to the trunk before the expert cache, see\n"
"docs/TUNING.md for why, and scripts/k3-doctor.sh to size this machine.\n");
}

/* ------------------------------------------------------------------- presets ----
 * Named memory budgets, so a user does not have to discover the trunk/cache split
 * empirically.
 *
 * The split is not arbitrary and it is not symmetric. Per token the engine re-reads the
 * ENTIRE 108.81 GB trunk but only ~25.8 GB of routed experts, so a gigabyte given to the
 * trunk removes roughly 1.17 GB/token of guaranteed traffic (one pinned layer) while a
 * gigabyte given to the expert cache removes, below about 36 GB of arena, nothing
 * measurable, K3's router is trained for flat expert usage, which defeats an LRU.
 *
 * Measured consequence: at a fixed 128 GB budget, trunk-first runs 1.69x faster than
 * cache-first. So every preset fills the trunk before it feeds the cache.
 * docs/PERFORMANCE.md carries the data and the noise floor that bounds it. */
/* K3Preset is defined in include/libk3/k3_internal.h */

/* The trunk/cache figures are BUDGETS passed to the two allocators. The description
 * quotes measured peak RSS for the whole process, which is the number that decides
 * whether a machine can run the preset, it includes the safetensors index, the KV
 * cache and scratch, none of which appear in either budget. Measured on the reference
 * machine in docs/PERFORMANCE.md; expect a little variation elsewhere. */
const K3Preset K3_PRESETS[] = {
    { "laptop",      3.0,   1.0,  "8.2 GB peak RSS. The floor. Runs, slowly." },
    { "desktop",    16.0,  10.0,  "31.9 GB peak RSS." },
    { "workstation", 60.0, 30.0,  "95.5 GB peak RSS; the expert cache starts to matter here." },
    { "server",     110.0, 13.0,  "~128 GB peak RSS; 90 of 93 trunk layers pinned. Fastest." },
    { "max",        110.0,109.0,  "~224 GB peak RSS; trunk pinned and a large expert cache." },
};
enum { K3_NPRESET = (int)(sizeof K3_PRESETS / sizeof K3_PRESETS[0]) };

const K3Preset *k3_preset_find(const char *name)
{
    for (int i = 0; i < K3_NPRESET; i++)
        if (!strcmp(name, K3_PRESETS[i].name)) return &K3_PRESETS[i];
    return NULL;
}

static void k3_preset_list(FILE *f)
{
    fprintf(f, "presets (trunk / expert-cache, in GB):\n");
    for (int i = 0; i < K3_NPRESET; i++)
        fprintf(f, "  %-12s %6.1f / %-6.1f  %s\n", K3_PRESETS[i].name,
                K3_PRESETS[i].trunk_gb, K3_PRESETS[i].cache_gb, K3_PRESETS[i].note);
    fprintf(f, "  %-12s %6s / %-6s  %s\n", "auto", "fit", "fit",
            "sizes both from this machine's free RAM, trunk-first. Recommended.");
    fprintf(f, "\nAll presets stream the trunk, so they need --trunk <packed_dir>.\n"
               "Run scripts/k3-doctor.sh to see which one this machine fits.\n");
}

/* PEAK resident set, in bytes. ru_maxrss is kilobytes on Linux and BYTES on Darwin, so
 * the scale factor differs by platform; applying the Linux one on macOS would overstate
 * the peak by 1024x.
 *
 * This is the authoritative memory figure. The banner printed before allocation is a
 * PLAN and understates: it omits the safetensors index (~78 MB at full scale), reports
 * requested budgets rather than actual reservations, and cannot observe fragmentation.
 * Quote this value, not the plan. */
static double peak_rss_bytes(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return (double)ru.ru_maxrss;            /* already bytes */
#else
    return (double)ru.ru_maxrss * 1024.0;   /* kilobytes */
#endif
}

/* MemAvailable, which is what the kernel thinks can actually be handed out, not
 * MemFree. Returns 0 if it cannot be read. */
static double mem_available_bytes(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double kb = 0.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemAvailable:", 13)) { kb = atof(line + 13); break; }
    fclose(f);
    return kb * 1024.0;
}

/* Weights is in include/libk3/k3_internal.h */

/* One full forward over T tokens, writing logits for the LAST position only. Every
 * step rebuilds state from scratch, matching the path the oracle validates.
 *
 * Returns 0 on success and -1 if the forward could not be completed. The caller MUST
 * check: on failure logits_last is left untouched, and argmaxing an untouched buffer
 * yields a token drawn from uninitialised memory, printed as though it were output. */
/* arg_all: when non-NULL, receives argmax(logits) for EVERY position 0..T-1, which is
 * what batched greedy verification consumes. logits_last still gets the final position's
 * full vector either way. The extra cost is one lm_head matmul per additional position,
 * pure RAM-resident compute; measured, an extra verified position costs ~22% of a serial
 * token at streamed-trunk budgets, which is the entire economics of --spec. */
int forward(Weights *w, const K3Cfg *c, K3Cache *cache, const int *ids, int T,
                   float *logits_last, float *scratch, float *h, float *br, float *kstate,
                   int *arg_all)
{
    const int E = c->hidden;
    const int maxb = c->n_layers / c->attn_res_block + 2;
    const int P = c->kda_heads * c->kda_head_dim;
    const size_t kper = (size_t)P * c->kda_head_dim + (size_t)3 * P * (c->conv_k - 1);

    for (int t = 0; t < T; t++)
        k3_embed_row(h + (size_t)t * E, w->mb.embed, w->mb.wdt, ids[t], E);

    memset(br, 0, (size_t)T * maxb * E * sizeof(float));
    /* Incremental decode carries the KDA recurrent matrix and ShortConv history across
     * steps, so it must NOT be cleared here; the full-recompute path rebuilds from
     * scratch every step and must be. */
    if (!w->kvc) memset(kstate, 0, kper * (size_t)w->n_bound * sizeof(float));
    int nb = 0;
    for (int L = 0; L < w->n_bound; L++) {
        /* Streaming: bring this layer in, and hint the next one so its read overlaps
         * this layer's arithmetic. The order is fixed 0..92 every token, so the hint is
         * never wrong. */
        if (w->trunk) {
            if (k3_trunk_bind(w->trunk, c, L, &w->lay[L]) != 0) {
                fprintf(stderr, "trunk bind failed at layer %d\n", L);
                return -1;
            }
            k3_trunk_prefetch(w->trunk, L + 1);
        }
        /* Point this layer's MoE at the cache before use. Doing it here rather than at
         * bind time keeps K3LayerBind independent of any particular cache. */
        if (w->lay[L].lay.moe) {
            w->lay[L].moe.src = &cache->src;
            w->lay[L].moe.layer = L;
            /* The draft routes only among resident experts, reading zero new expert bytes;
             * the exact model keeps true routing. This is what makes a draft step cheap. */
            w->lay[L].moe.cache_only = w->draft_mode;
        }
        if (w->kvc && w->mla_slot[L] >= 0) {
            const size_t kvper = (size_t)w->kv_cap * c->n_heads * (c->qk_nope + c->v_head);
            const size_t rpper = (size_t)w->kv_cap * c->qk_rope;
            const int mi = w->mla_slot[L];
            k3_decoder_layer_inc(h, br, &nb, &w->lay[L].lay, c, L, T,
                                 kstate + kper * (size_t)L, scratch,
                                 w->kvc + kvper * (size_t)mi,
                                 w->ropec + rpper * (size_t)mi,
                                 w->cached, w->kv_cap);
        } else {
            k3_decoder_layer_inc(h, br, &nb, &w->lay[L].lay, c, L, T,
                                 kstate + kper * (size_t)L, scratch,
                                 NULL, NULL, 0, 0);
        }
    }

    /* The model-level aggregator, beyond the two per layer. Exactly one pair exists in
     * the checkpoint; skipping it is silent. */
    if (w->mb.out_res_norm && w->mb.out_res_proj) {
        float *fold = scratch;
        float *src  = fold + E;
        for (int i = 0; i < E; i++) fold[i] = w->mb.out_res_norm[i] * w->mb.out_res_proj[i];
        for (int t = 0; t < T; t++) {
            for (int b = 0; b < nb; b++)
                memcpy(src + (size_t)b * E, br + ((size_t)t * maxb + b) * E,
                       (size_t)E * sizeof(float));
            memcpy(src + (size_t)nb * E, h + (size_t)t * E, (size_t)E * sizeof(float));
            k3_attn_res(h + (size_t)t * E, src, fold, nb + 1, E, c->rms_eps);
        }
    }

    float *nrm = scratch;
    if (arg_all) {
        for (int t = 0; t < T; t++) {
            k3_rmsnorm(nrm, h + (size_t)t * E, w->mb.norm, E, c->rms_eps);
            k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
            arg_all[t] = argmax_(logits_last, c->vocab);
        }
        /* logits_last now holds the FINAL position's vector, same as the plain path. */
        return 0;
    }
    k3_rmsnorm(nrm, h + (size_t)(T - 1) * E, w->mb.norm, E, c->rms_eps);
    k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
    return 0;
}
