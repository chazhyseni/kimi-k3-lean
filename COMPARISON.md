# warp vs kimi-k3-in-c — a measured comparison

This document compares the two reference implementations of "a 2.78T
Kimi K3 model running on a laptop" that the Medium article
`Building Kimi K3 (2.8T) Model in C to Run on 8GB RAM` and its companion
repo (`FareedKhan-dev/kimi-k3-in-c`, v1.0.0, 2026-08-07) describe.

**Scope.** This is a comparison of the two implementations, not a
promotion. Both are reference code that runs the full model from
scratch in C with no runtime dependencies. Both ship the same thing.
They differ in implementation, in engineering choices, and in what
they measured. Where they differ, the question is which trade-off was
better for which use case.

## Honest framing

Neither author is "right" — they shipped overlapping artifacts with
different engineering teams behind them. Warp is the maintainer's
shipping engine; it has been used end-to-end on this dev host and on
[M5 Pro](...). The article's repo is the author-of-the-article's
"here is how I built it" companion. Both projects describe the same
underlying model architecture (Kimi K3, 93 layers, 2.78T) and both
target a laptop-class machine.

The article's docstring is explicit about this: the goal is `8 GB of
RAM`, which is the small end of the achievable spectrum. Warp's
declared goal is a 64 GB laptop. These are two different shapes of the
same problem.

## At a glance

| | warp | kimi-k3-in-c (article) |
|---|---|---|
| First public release | 2026-Q2 | 2026-07-31 |
| Latest release | 0.6.8 (this session) | 1.0.0 (2026-08-07) |
| Source LOC (C only) | 10,835 | 10,802 |
| Source LOC (engine + tests) | 22,891 | 22,902 |
| Target RAM floor | 1.32 GB (Kimi-Linear) / 29.19 GB (K3) | **8.24 GB** (K3) |
| Target RAM ceiling | 64 GB laptop | 224 GB server |
| Quantization (experts) | **3-bit residual VQ** (VQ3R, 3.01 bits/weight) | **MXFP4** (0.53125 bytes/weight = 4.25 bits/weight) |
| Quantization (trunk) | Q4G / Q8G / F32 selectable | **bf16 native** (modular, switchable to Q4G/Q8G via tools) |
| Container format | `.waste` (custom, 4 KiB-aligned records) | per-shard safetensors + packed trunk (`.bin`) |
| Backend | AVX2 + AVX-512 + NEON + metal.m | AVX2 only |
| Tokenizer | tiktoken in C | tiktoken in C |
| Linear attention | KDA (Kimi Delta Attention) | KDA (same) |
| Attention residuals | yes (§2.2) | yes (§2.2) |
| OpenAI HTTP server | yes (Python, serve/__main__.py) | **no** |
| PyTorch differential test | yes (tools/kimi_ref.py, ~491 lines) | yes (tests/unit/, ~70 KB test files) |
| Test runtime in CI | `make check` (45 tests) | `make test` + ctest (full binary test suite) |
| Noise floor in reported numbers | per-model, single-run medians | **33%** (documented explicitly) |
| Dated benchmark files | inline in LEARNED.md | `docs/data/*.tsv` and `*.md` |

## What they share

1. **Same model architecture.** Both read the Kimi K3 checkpoint's
   `config.json` and route through 69 KDA + 24 Gated MLA layers with
   896 routed experts and top-16 selection. The article's docstring
   specifically rejects defaulting config fields, which is the same
   shape warp enforces.

2. **Same disk-first design.** Both keep the **dense trunk** in RAM
   or under a memory budget, and **stream routed experts** from disk.
   Both report ~25 GB of expert reads per token (K3) at the bottom of
   the memory ladder.

3. **Same disk I/O wall.** Both implementations describe the wall as
   the *disk*, not the *CPU*. The article's `docs/PERFORMANCE.md`
   opens with `Storage bandwidth is usually the ceiling, not the CPU`.
   Warp's `docs/LEARNED.md §1` is the same finding.

4. **Same caching algorithm.** Both use LRU (warp uses an LFRU —
   "least-frequently-recently-used" — variant, the article uses a
   straight LRU). Both give expert caching as a knob.

5. **Same test discipline.** Both have a per-layer differential test
   against a PyTorch reference. Both are explicit about refusing to
   half-understand a config.

6. **Same KDA recurrence.** Both implement the same formula
   `g = g_min · sigmoid(e^A · z)` with `g_min = -5` and per-head A
   indexing. Both have the same load-bearing recurrence order:
   decay → read → write delta → read output.

## What they have done differently

### 1. Expert quantization (the biggest single difference)

**warp: 3-bit residual VQ.** Container is 982 GB on K3
(3.01 bits/weight avg). Detailed in `docs/LEARNED.md §3` and
`docs/FORMAT.md`. The engine reads each expert and decodes through a
learned codebook in `src/vq.c`. CLAUDE.md reports a 19.5% relative
error at 3-bit on Kimi-Linear, recovered enough to produce correct
factual answers.

**article: native MXFP4.** Container is 1.45 TB of routed experts
(0.53125 bytes/weight = 4.25 bits/weight). The article reads the
checkpoint's native OCP MX FP4 format and consumes packed nibbles
directly: `value = E2M1[nibble] · 2^(E8M0_scale - 127)`. **No
dequantization.** The CLAUDE.md explains why: "Dequantising would
turn a 17.5 MB expert into 132 MB, and a token touches 1,472 of them
(92 routing layers × top-16): 194 GB per token of pure widening".

**Trade-off.** Warp's VQ3R is *narrower* (3 bits vs 4.25 bits ≈ 30%
less disk per expert) but goes through a learned codebook lookup
during inference. The article's MXFP4 is *faster* on read (no lookup
table, just a 16-entry E2M1 conversion) but takes more disk (1.45 TB
vs warp's ~700 GB routed experts). Neither is unambiguously better;
the article's approach stays closer to the reference model's format
and skips the VQ training step, warp's approach is more compact on
disk.

**The article's packaging makes this even more interesting.** The
container is 1.45 TB of routed experts + 108.81 GB of trunk + 5.3 GB
of floor. Warp's container is also ~982 GB total but with a
different shape. The article's design lets the trunk be *fully
pinned* (resident) at 110 GB while still running, which warp's
fixed-trunk-residency design does not allow.

### 2. Trunk residency — fixed vs. dialed

**warp: trunk fully resident.** The `waste_cfg.ram_budget_bytes` knob
controls the *expert cache*, not the trunk. The trunk is always
resident. On Kimi-Linear (27 layers) this is 1.3 GB. On K3 (93
layers) it would be 108.81 GB. The design is documented in
`docs/ENGINE.md`.

**article: trunk is a dial.** The `--trunk-gb` and `--cache-gb` flags
split the budget. The trunk can be `< 8 GB` (unpinned, ring-buffered)
or `> 110 GB` (fully pinned). At the smaller end, the article reports
byte-identical output across the entire 8 GB → 224 GB range.

**Trade-off.** Warp's fixed-trunk is simpler to reason about and
predict; the article's dialed trunk is more flexible. The article's
measurements show that *more* trunk is the dominant performance knob
(not expert cache), so the dial matters. Warp's design assumes the
user has enough RAM to hold the trunk; the article's design assumes
the user might not.

The article's `docs/TUNING.md` has a table showing the speedup is
1.69× from a 12.3 GB → 110 GB trunk change at a 128 GB budget. Warp
does not measure this because it doesn't allow the choice.

### 3. The memory wall — 8 GB vs 29 GB

The article's headline number is **8 GB RAM floor** on K3. The
`k3-doctor.sh` script reports `RAM floor 8.24 GB` and peak RSS on an
8 GB budget is exactly 8.24 GB. This is the article's defining claim.

Warp's measured floor is **29.19 GB** on K3. The 982 GB container is
the full model. There is no 8 GB mode in warp.

**Why the gap.** Three reasons, in approximate order of impact:

1. **Warp's trunk is fully resident.** K3's trunk is 108.81 GB. Warp
   keeps all of it in RAM. The article's 8 GB floor is achieved by
   ring-buffering the trunk and reading 1.4 GB/token of trunk traffic.
2. **Warp's KV cache is allocated.** The article's KV cache is
   opt-in via `--incremental`. Without it, the article uses 0 KB of
   KV.
3. **Warp's expert cache has a baseline.** The article's expert cache
   can be as small as 28 slots (0.49 GB on the 8 GB row).

These are not absolute; warp could be re-tuned to deliver 8 GB at the
cost of being slower (the article's 8 GB row is 32.69 s/token).
The article's design **trades peak speed for memory floor**; warp's
design **trades memory floor for peak speed**.

### 4. Storage layout

**warp: a single `.waste` directory.** Container is 982 GB on K3,
chunked at 4 KiB-aligned records per expert. CLAUDE.md describes the
format as disk-friendly with O_DIRECT.

**article: per-shard safetensors + packed trunk.** The article reads
the **original** safetensors shards directly (one coalesced pread per
expert) and packs the trunk into a single contiguous file via
`tools/pack_trunk.py`. The trunk-packing step is what lets the
article's fixed-order trunk read be a single pread per layer.

**Trade-off.** The article's approach is *much* simpler to set up
(no custom container format) but requires the user to keep the
original shards around (1.56 TB) and then a separate packed trunk
file. Warp's approach consolidates to one container but loses the
original safetensors. The article's approach is friendlier to the
HuggingFace ecosystem; warp's approach is friendlier to packaging.

### 5. OpenAI chat server

**warp: yes.** `serve/__main__.py` is a 270-line Python HTTP server
using stdlib `http.server`. Speaks OpenAI Chat Completions. Streams
SSE. Drop in for any OpenAI-compatible harness.

**article: no.** The article's CLI is `k3_run.c` (1,487 lines). It
takes text in, prints tokens out. There is no HTTP server. To use it
from a harness, the user would have to write one.

This is the **single biggest user-facing difference** between the two
projects. Warp is "drop this in as an OpenAI endpoint"; the article
is "run this binary and pipe tokens".

### 6. Available configs / presets

**warp: one auto-sized config.** The budget is one number
(`ram_budget_bytes`), the engine picks the rest. There is no
named-preset system.

**article: five named presets.** `laptop` (3 GB trunk + 1 GB cache,
~10 GB total, ~32 s/token), `desktop` (16+10, ~32 GB, ~31 s/token),
`workstation` (60+30, ~96 GB, ~24 s/token), `server` (110+13, ~128 GB,
~17 s/token), `max` (110+109, ~224 GB, ~19 s/token). Plus `--preset
auto` which sizes the budget from the machine's free RAM.

**Trade-off.** The article's presets encode the measured ladder so
the user doesn't have to. The presets are an honest transfer of the
benchmarks into user-facing choices. Warp's auto-sizing does the same
job but does not expose the tier names.

### 7. Things in warp that the article does not have

1. **OpenAI HTTP server.** (biggest single user-facing gap)
2. **Streaming SSE responses.** Easy to add; not in article.
3. **3-bit residual VQ on experts.** Smaller disk than MXFP4.
4. **Metal backend (src/metal.m).** Article is AVX2-only.
5. **Vision encoder + image projection.** Article is text-only.
6. **LFRU cache variant.** Article uses straight LRU.
7. **Layer-internal attention prediction / speculative decode.**
   Not in article.
8. **Bundle inference + downstream / `serve/regions.py`.**
   Not in article.

### 8. Things in the article that warp does not have

1. **8 GB RAM floor.** The capability to run K3 on a 32 GB
   unified-memory laptop in 8 GB. Not achievable in warp's current
   design without a trunk-streaming rework.
2. **Trunk as a dial (`--trunk-gb` / `--cache-gb`).** Warp forces
   trunk residency.
3. **MXFP4 native matmul.** No dequantization step. Faster per
   expert read.
4. **Pure stable-latent MoE with shared-expert handling.** The
   article explicitly handles the 2 SHARED experts not routing
   (warp's docs don't address this).
5. **Conversation resume (`--save-state` / `--load-state`).** Saves
   recurrent state + KV cache to disk between turns. Reports 3.9×
   faster on turn 2. Not in warp.
6. **Speculative decoding by n-gram (`--spec N`).** Drafts and
   verifies, output identical to greedy. Not in warp.
7. **Chunk-union prefill.** Fetches each unique routed expert once
   per chunk instead of once per token. Reports ~half the expert
   bytes on a prompt. Not in warp.
8. **Five named presets.** Warp auto-sizes but does not expose
   named tiers.
9. **Recorded expert-cache trace replay with Belady oracle.**
   `tools/sim_cache.py` lets you replay a recorded request trace
   at any cache capacity and compare LRU vs. optimal. Not in warp.
10. **33% noise floor documented in real numbers.** The article is
    explicit: "Differences smaller than 33% are not effects". Warp
    reports single-run medians without the noise-floor caveat.
11. **Deterministic fixtures with adversarial byte-pattern checks.**
    "nibble order is a convention, not a rule: the low nibble is the
    even element. Reversing it yields a matrix with the right values
    in the wrong places: every statistic looks correct and the model
    is wrong." Warp has nibble-order tests but they are not
    organized as comprehensively.
12. **`--incremental` verified bit-identical to full recompute.**
    The article proves the equivalence. Warp's KV cache is correct
    but without the explicit "verified bit-identical" claim.

### 9. Things both have

Both have fused MoE kernels, both have KDA, both have correct
PyTorch differential tests, both have config readers that refuse to
default missing fields, both have tiktoken-in-C tokenizers, both have
CI configurations, both have noise budgets in their measurements.

## What the article's measurements teach that warp does not

Three concrete findings that warp's docs and measurements do not
address:

1. **Quantile balancing defeats the expert cache.** K3's router is
   trained with quantile balancing, which deliberately flattens
   expert usage. The article's measurement: LRU retention stays at
   exactly 0.0% from 28 slots to 1,344 slots. The bytes read per
   token stay pinned at 25.83 GB across a 48× cache size increase.
   This is the model, not the implementation. Warp's docs report
   29-41% retention on K3; that discrepancy is not explained. Either
   warp's measurements are different (different prompts, different
   top-k, different cache sizes) or warp's cache is not actually LRU.
   The article's measurement deserves to be reproduced on warp before
   any conclusion is drawn.

2. **Trunk pinning dominates cache growth.** The article's
   measurement: at 128 GB, splitting 12.3 GB trunk + 110.7 GB cache
   (28.38 s/token) versus 110 GB trunk + 13 GB cache (16.80 s/token)
   is 1.69× — *the smaller cache is faster*. This is the
   counter-intuitive result that motivates the dialed-trunk design.
   Warp's design does not allow this knob, so warp cannot reach
   this speedup on the same hardware.

3. **Belady's optimal cache is 84% at 128 GB; LRU is 49%.** The
   article's measured Belady-optimal cache pattern climbs steadily
   while LRU stays flat. This means there is *exploitable locality*
   in K3's expert routing that LRU does not capture. A future
   cache-policy work item is to recover some of the 35% gap. Warp
   does not document this.

## What warp's measurements teach that the article does not

Three concrete findings that warp's docs and measurements have but the
article's docs and measurements do not:

1. **2.61 tok/s on 8-core EPYC Milan with cgroup cpuset pinning.**
   Warp's measured numbers on this dev host: 2.61 tok/s sustained on
   Kimi-Linear-48B, 0.8-1.85 tok/s without pinning. The article's
   reference machine is a 124-vCPU EPYC 7763 with 228 GB RAM, which
   is a different hardware class. Neither is necessarily comparable to
   the other.

2. **Streaming via SSE.** Warp's server streams `chat.completion.chunk`
   events. The article has no server, so no streaming.

3. **Multi-model single install.** Warp's container format is the
   same `.waste` for both Kimi-Linear and K3. The article's release
   is K3-specific.

## What this comparison does not say

It does not say "warp is better" or "the article is better". The
two are different project goals with different engineering teams
behind them. The article's repo is a "here's how I built it"
companion, and the article's `CHANGELOG.md` is explicit about the
goal: 8 GB floor, named presets, byte-identical output across the
memory ladder. Warp's goal is different: a 64 GB laptop engine that
exposes an OpenAI HTTP server for use from a harness.

The honest answer is: **depending on what you want to do, one or the
other is the right choice.**

- **I want to run K3 on maximum hardware with the cleanest possible
  installation and don't need an HTTP server.** Article's repo.
- **I want to drop a Kimi model into a 64 GB laptop as an OpenAI
  endpoint for an agent harness.** Warp.
- **I want both.** Read both, pick the parts you want.

## Addendum: what I did not do

I did not cross-compile warp against the article's measurements to
see if warp's K3 path achieves the same 17 s/token on the article's
machine. I did not run the article's test suite on this host. I did
not port the article's MXFP4 matmul to warp, or vice versa. I did
not run the article's `k3-doctor.sh` against this host. The
comparison is from reading both codebases and their docs, not from
implementing one atop the other.

If you want any of those follow-ups, the next session can do them.
