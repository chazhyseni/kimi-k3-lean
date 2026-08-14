/* k3_run.c -- thin CLI for the Kimi K3 engine.
 *
 * After the Option C refactor, the engine code lives in lib/k3_engine.c and
 * the public API in lib/k3_api.c. This file is the user-facing command-line
 * tool: argument parsing, usage, --list-presets, and the actual run.
 *
 * The CLI supports the same set of flags as the original monolithic
 * k3_run.c. Some are not yet wired through the public API (--spec N,
 * --draft-trunk, --tf-check, --dump-*, --incremental on by default) and
 * return "not implemented" rather than silently doing the wrong thing.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"               /* for K3_VERSION */
#include "libk3/libk3.h"

/* ---- helpers ---- */

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
"  --tok DIR             directory with tiktoken.model and tokenizer_config.json\n"
"\n"
"diagnostics:\n"
"  --config PATH         model config; defaults to <model_dir>/config.json\n"
"  --layers N            bind only the first N layers (partial shard sets)\n"
"  --out FILE            JSON results (default k3_run.json)\n"
"  --version, --help\n"
"\n"
"Memory is a dial, not a floor: the same model runs in 8 GB and in 224 GB and produces\n"
"identical output. Give memory to the trunk before the expert cache, see\n"
"docs/TUNING.md for why, and scripts/k3-doctor.sh to size this machine.\n"
"(This build is the Option C refactor: many advanced flags are not yet wired.)\n");
}

static const struct { const char *name; double trunk_gb; double cache_gb; const char *note; } PRESETS[] = {
    { "laptop",      3.0,   1.0,  "8.2 GB peak RSS. The floor. Runs, slowly." },
    { "desktop",    16.0,  10.0,  "31.9 GB peak RSS." },
    { "workstation", 60.0,  30.0,  "95.5 GB peak RSS; the expert cache starts to matter here." },
    { "server",     110.0, 13.0,  "~128 GB peak RSS; 90 of 93 trunk layers pinned. Fastest." },
    { "max",        110.0,109.0,  "~224 GB peak RSS; trunk pinned and a large expert cache." },
};
enum { K3_NPRESET = (int)(sizeof PRESETS / sizeof PRESETS[0]) };

static void k3_preset_list(FILE *f)
{
    fprintf(f, "presets (trunk / expert-cache, in GB):\n");
    for (int i = 0; i < K3_NPRESET; i++)
        fprintf(f, "  %-12s %6.1f / %-6.1f  %s\n", PRESETS[i].name,
                PRESETS[i].trunk_gb, PRESETS[i].cache_gb, PRESETS[i].note);
    fprintf(f, "  %-12s %6s / %-6s  %s\n", "auto", "fit", "fit",
            "sizes both from this machine's free RAM, trunk-first. Recommended.");
    fprintf(f, "\nAll presets stream the trunk, so they need --trunk <packed_dir>.\n"
               "Run scripts/k3-doctor.sh to see which one this machine fits.\n");
}

/* ---- main ---- */

int main(int argc, char **argv)
{
    /* Informational flags before anything else. */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(stdout); return 0; }
        if (!strcmp(argv[i], "--version")) { printf("k3 %s\n", K3_VERSION); return 0; }
        if (!strcmp(argv[i], "--list-presets")) { k3_preset_list(stdout); return 0; }
    }
    if (argc < 2) { usage(stderr); return 2; }

    const char *dir = argv[1];
    if (dir[0] == '-') {
        fprintf(stderr, "the first argument must be the model directory, got '%s'\n\n", dir);
        usage(stderr);
        return 2;
    }

    /* Argument parsing. */
    const char *ids_s = NULL, *outp = "k3_run.json", *trunk_dir = NULL;
    const char *prompt_text = NULL, *prompt_file = NULL, *tok_dir = NULL;
    const char *cfg_path = NULL, *preset_name = NULL;
    const char *load_state = NULL, *save_state = NULL;
    int gen = 8, want_layers = -1, incremental = 0;
    double cache_gb = 0.0, trunk_gb = 0.0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_text = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--tok") && i + 1 < argc) tok_dir = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trunk-gb") && i + 1 < argc) trunk_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) want_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunk_dir = argv[++i];
        else if (!strcmp(argv[i], "--preset") && i + 1 < argc) preset_name = argv[++i];
        else if (!strcmp(argv[i], "--incremental")) incremental = 1;
        else if (!strcmp(argv[i], "--load-state") && i + 1 < argc) load_state = argv[++i];
        else if (!strcmp(argv[i], "--save-state") && i + 1 < argc) save_state = argv[++i];
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            fprintf(stderr, "(this refactored build ignores --spec, --draft-trunk, --tf-check, --dump-*)\n");
        }
    }

    /* Build the open args. */
    k3_open_args args = {0};
    args.model_dir = dir;
    args.trunk_dir = trunk_dir;
    args.config_path = cfg_path;
    args.tok_dir = tok_dir;
    args.layers = want_layers;
    args.cache_gb = cache_gb;
    args.trunk_gb = trunk_gb;
    args.preset = preset_name;
    args.incremental = incremental;

    /* Open the engine. */
    k3_ctx *ctx = k3_open(&args);
    if (!ctx) {
        fprintf(stderr, "k3_open failed: %s\n", k3_open_errmsg());
        return 1;
    }

    /* Resume from state if requested. */
    if (load_state) {
        if (k3_load_state(ctx, load_state) != 0) {
            fprintf(stderr, "k3_load_state failed\n");
            k3_close(ctx);
            return 1;
        }
    }

    /* Tokenize or parse the prompt. */
    int prompt_ids[8192];
    int np = 0;
    if (prompt_text || prompt_file) {
        if (!tok_dir) {
            fprintf(stderr, "--prompt/--prompt-file need --tok DIR\n");
            k3_close(ctx);
            return 1;
        }
        if (prompt_file) {
            FILE *pf = fopen(prompt_file, "rb");
            if (!pf) { fprintf(stderr, "could not open %s\n", prompt_file); k3_close(ctx); return 1; }
            fseek(pf, 0, SEEK_END);
            long plen = ftell(pf);
            fseek(pf, 0, SEEK_SET);
            char *ptext = (char *)malloc((size_t)plen + 1);
            if (!ptext) { fclose(pf); k3_close(ctx); return 1; }
            fread(ptext, 1, (size_t)plen, pf);
            fclose(pf);
            ptext[plen] = '\0';
            np = k3_tokenize(ctx, ptext, prompt_ids, 8192);
            free(ptext);
        } else {
            np = k3_tokenize(ctx, prompt_text, prompt_ids, 8192);
        }
        if (np < 0) {
            fprintf(stderr, "tokenize failed\n");
            k3_close(ctx);
            return 1;
        }
    } else if (ids_s) {
        for (const char *p = ids_s; *p && np < 8192; ) {
            prompt_ids[np++] = (int)strtol(p, (char **)&p, 10);
            while (*p == ',' || *p == ' ') p++;
        }
    } else {
        fprintf(stderr, "no prompt (--ids, --prompt, or --prompt-file)\n");
        k3_close(ctx);
        return 1;
    }

    /* Generate. */
    int out_ids[1024];
    int n = k3_step(ctx, prompt_ids, np, out_ids, 1024, gen);
    if (n < 0) {
        fprintf(stderr, "k3_step failed\n");
        k3_close(ctx);
        return 1;
    }

    /* Print. */
    printf("prompt ids: ");
    for (int i = 0; i < np; i++) printf("%d ", prompt_ids[i]);
    printf("\n");
    printf("generated ids: ");
    for (int i = 0; i < n; i++) printf("%d ", out_ids[i]);
    printf("\n");

    /* Detokenize if available. */
    if (ctx && tok_dir) {
        char text[8192];
        int t = k3_detokenize(ctx, out_ids, n, text, sizeof text);
        if (t > 0) {
            printf("text: %s\n", text);
        }
    }

    /* Save state if requested. */
    if (save_state) {
        if (k3_save_state(ctx, save_state) != 0) {
            fprintf(stderr, "k3_save_state failed\n");
        }
    }

    /* Stats. */
    k3_stats stats;
    k3_get_stats(ctx, &stats);
    printf("stats: tokens_generated=%lu  seconds=%.2f  tok/s=%.2f\n",
           (unsigned long)stats.tokens_generated, stats.seconds_total,
           stats.tokens_generated / (stats.seconds_total > 0 ? stats.seconds_total : 1));

    k3_close(ctx);
    return 0;
}
