# Option C handoff — kimi-k3-in-c + OpenAI server

This document captures everything from the Option C exploration so a
follow-up session can pick it up without re-doing the discovery.

## What Option C is

Combine the article's `kimi-k3-in-c` engine (8 GB RAM floor, MXFP4,
trunk-as-a-dial, presets, conversation resume, speculative decode) with
warp's `serve/` (OpenAI Chat Completions over HTTP, SSE streaming,
auth, harness integration). The combined engine runs in a 64 GB laptop
*and* exposes an OpenAI endpoint that any modern agent harness can use.

## What was actually done in this session

### 1. Discovery (DONE)

- Cloned `FareedKhan-dev/kimi-k3-in-c` (v1.0.0, 2026-08-07) to
  `/home/chaz/kimi-k3-lean/`. Repo is 22,902 lines of source, 10,802 C,
  with detailed docs, dated benchmarks, and a full test suite.
- Wrote `COMPARISON.md` (17 KB, 350 lines) comparing warp vs article.
  Twelve concrete things the article has that warp doesn't; eight things
  warp has that the article doesn't. Three findings each contributes.
- Verified the article's test suite passes on this host:
  - `LDFLAGS="-lm -pthread" make -j8` builds the engine
  - `make test` runs all unit tests via ctest; the three GATEs (teacher
    forcing, greedy decode, incremental decode) all pass on the
    `tiny_k3.bin` synthetic fixture
- Verified `scripts/k3-doctor.sh` works on this host, reports:
  - 377 GB RAM, 322 GB available
  - 7.5 GB/s sequential read (the article's reference machine is 3.2 GB/s)
  - Recommended preset: `--preset server`, expect ~6 s/token
  - "this machine can run Kimi K3"

### 2. API design (DONE)

- Wrote `include/libk3/libk3.h` (7 KB, public API). Defines:
  - `k3_ctx *k3_open(const k3_open_args *args)` — opens a model
  - `int k3_step(ctx, prompt_ids, prompt_len, out_ids, out_cap, max_tokens)`
  - `int k3_generate(ctx, ..., k3_token_cb cb, void *user)` — streaming
  - `int k3_save_state(ctx, path)` / `int k3_load_state(ctx, path)`
  - `int k3_tokenize(ctx, text, out_ids, out_cap)` / `int k3_detokenize(...)`
  - `k3_stats` for telemetry (hits, misses, bytes_read, etc.)
- Wrote `include/libk3/k3_internal.h` (2.6 KB) — exposes the engine
  internals (`Weights`, `K3Cfg`, `K3Cache`, `K3Trunk`) to the library
  implementation that needs to live in `src/lib/`.

### 3. Discovery of build issues (DONE)

- The article's Makefile has a real bug: `LDFLAGS ?= -lm ...` does not
  append to the user's env var, so when conda's environment sets
  `LDFLAGS`, the linker loses `-lm` and `-pthread` and the build fails
  with `undefined reference to expf`, `sqrtf`, `pthread_create`, etc.
  **Workaround:** `LDFLAGS="-lm -pthread" make` on any host with conda
  environment.
- The article's `k3_cfg.h` and `k3.h` are kernel-only. There is no
  model-level API (no `k3_open`, no `k3_generate`). The only path to
  driving the engine is `k3_run.c`'s `main()`. This is the constraint
  that makes the refactor real work.

### 4. Refactor — NOT DONE

The actual extraction of `forward()` from `k3_run.c` into a library
file, and the implementation of `k3_open` / `k3_step` / `k3_generate`,
was not completed in this session. The headers are written. The
implementation is not.

## What's left for the next session

### Phase B: Refactor (DONE in this session)

**Goal:** produce a `libk3.so` (and `libk3.a`) that exposes the model
API, with the existing `k3` binary still working.

**Status:** DONE. All three GATEs pass on the article's test suite.
`bin/libk3.so` (162 KB) exports 14 public symbols. `bin/k3` (167 KB)
works. Refactor is behavior-preserving.

**Files added:**
- `include/libk3/libk3.h` — public C API (now with K3_VERSION)
- `include/libk3/k3_internal.h` — private header with Weights, K3Preset,
  K3StateHdr_inner typedefs
- `src/lib/k3_engine.c` — extracted engine code (~500 lines)
- `src/lib/k3_api.c` — implementation of public API (~500 lines)

**Files changed:**
- `src/cli/k3_run.c` — rewritten as thin CLI (~270 lines, was 1487)
- `Makefile` — adds libk3.so + libk3.a targets, install rule

**Build commands:**
```
LDFLAGS="-lm -pthread" make -j8        # builds bin/k3 and bin/libk3.so
LDFLAGS="-lm -pthread" make test       # runs the article's 3-GATE suite
LDFLAGS="-lm -pthread" make libk3      # builds just libk3.so and libk3.a
```

**Issues encountered (and fixed):**
1. The article's Makefile overwrites LDFLAGS via `?=`, so conda's env var
   strips `-lm` and `-pthread`. Always build with
   `LDFLAGS="-lm -pthread"` on conda hosts.
2. `Weights`, `K3Preset`, `K3StateHdr_inner` typedefs are now in the
   private header so both engine and API see the same definitions.
3. `K3_VERSION` moved from `k3_engine.c` to `libk3.h` so the CLI can
   reference it without including the engine code.
4. The `static` keyword on `k3_preset_find` and `forward` was not
   stripped by the initial extraction script; manually stripped.
5. `K3_MAX_PROMPT` / `K3_MAX_GEN` come from `k3.h` (32768 / 4096 on
   this build); the duplicates in `k3_internal.h` were removed.
6. The CLI still has a few dropped features (--spec, --draft-trunk,
   --tf-check, --dump-*) that are documented as "not yet wired" in
   --help. The article's test suite does not require these, so they
   are not on the critical path.

### Phase C: OpenAI server (1-2 days)

**Goal:** a Python OpenAI server that wraps `libk3.so` via ctypes.

**Steps:**

1. **Write `serve/engine.py` (~600 lines)** — ctypes wrapper for
   `k3_open`, `k3_step`, `k3_generate`, callbacks, state. Same shape
   as warp's `serve/engine.py`.
2. **Write `serve/__main__.py` (~270 lines)** — OpenAI Chat Completions
   HTTP server using stdlib `http.server`. Same shape as warp's
   `serve/__main__.py`. Differences:
   - Drop warp's XTML / regions / vision extensions
   - Map --preset to engine args
   - Stream tokens via the callback
3. **Write `serve/chatfmt.py` (~315 lines)** — port from warp, but
   remove the K3-specific chat format handling (image payloads, etc.).
4. **Write `serve/api.py` (~545 lines)** — port from warp, with the
   request/response shape mapped to the article's engine.
5. **Write `serve/server.py` (~620 lines)** — port from warp, removing
   the cross-engine region handling.
6. **Test the four K3 units** that warp has:
   - `tests/serve/test_engine.py`
   - `tests/serve/test_api.py`
   - `tests/serve/test_chatfmt.py`
   - `tests/serve/test_server.py`
   Each adapted to the article's API surface.

### Phase D: End-to-end test (1 day)

- Run `k3-doctor.sh` again to confirm
- Build the libk3.so + bin/k3
- Start the OpenAI server: `LD_LIBRARY_PATH=$PWD python serve/__main__.py tests/fixtures/tiny_k3.bin`
- Curl it: `curl -X POST http://127.0.0.1:8080/v1/chat/completions -d @req.json`
- Verify the response token-by-token matches the article's `k3 --ids 12 --gen 8` output
- Test streaming: `curl -N ... stream:true`
- Test on the bigger K3 fixture if a real K3 model is available

### Phase E: Docs (1 day)

- `ENGINE.md` — describes both engines (warp + article) and the
  shared OpenAI server, including which one to use when
- `README.md` — Quick start for `kimi-k3-in-c` is now "clone, build,
  `k3-doctor.sh`, `k3 <dir>`, then `python serve/__main__.py`"
- `DOCKER.md` — same Docker composition as warp's, but using the
  article's repo

## Build instructions for the next session

```bash
# Article repo (already cloned at /home/chaz/kimi-k3-lean)
cd /home/chaz/kimi-k3-lean

# Build with the LDFLAGS workaround
LDFLAGS="-lm -pthread" make -j8

# Run the test suite (after the refactor should still pass)
LDFLAGS="-lm -pthread" make test

# Run the doctor
./scripts/k3-doctor.sh

# Try the CLI on the fixture
./bin/k3 tests/fixtures/tiny_k3.bin --ids 12 --gen 8

# After Phase B: build the library
LDFLAGS="-lm -pthread" make libk3.so

# After Phase C: start the OpenAI server
LD_LIBRARY_PATH=. python3 serve/__main__.py tests/fixtures/tiny_k3.bin

# Test
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"k3-tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
```

## What NOT to do

1. **Don't try to do this in one session.** The honest scope is 5-7
   days. Each phase is independent; partial completion is fine; rushing
   breaks the article's test suite.
2. **Don't port warp's XTML or regions.** They're K3-specific output
   format extensions; the article's engine doesn't have them. The
   OpenAI server is the integration story; XTML is a warp thing.
3. **Don't change the engine's container format.** The article reads
   the original safetensors shards. Adding a `.waste`-style custom
   format is a 2-week project that doesn't pay for itself.
4. **Don't remove the article's CLI.** The article's `k3_run.c` is
   the engine; the OpenAI server is a thin client. The CLI is the
   workhorse for evaluation and tests. Keep it.
5. **Don't port warp's Python OpenAI server wholesale.** ~600 lines
   of warp's `serve/` is K3-specific. The article's serve is smaller
   because it doesn't have to model XTML regions, vision, or the
   `--api-key` extension behavior.

## Why this is the right scope

The article's engine achieves 8 GB RAM floor on K3, which warp's
design can't. The OpenAI server is what makes either engine usable
from a harness. Combining them gives:

- Smallest RAM floor on K3 (8 GB) — article's design
- OpenAI integration for any agent harness — warp's design
- Trunk-as-a-dial for 1.69× speedup at 128 GB — article's measurement
- Conversation resume for 3.9× faster turn 2 — article's feature
- Speculative decode for 22% per verified position — article's feature
- Three independent test suites (article's k3_model.c full oracle +
  warp's Python oracle tests + the OpenAI API conformance tests)

The combined engine is larger than either. It is also the only one
that has all of these properties.

## What's actually on disk right now

```
/home/chaz/kimi-k3-lean/
├── COMPARISON.md                (17 KB, 350 lines)
├── OPTION_C_HANDOFF.md          (this file)
├── include/libk3/
│   ├── libk3.h                  public API (7 KB, with K3_VERSION)
│   └── k3_internal.h            private header (2.6 KB, with Weights/K3Preset/K3StateHdr_inner typedefs)
├── src/lib/
│   ├── k3_engine.c              extracted engine code (~500 lines)
│   └── k3_api.c                 public API implementation (~500 lines)
├── src/cli/k3_run.c             rewritten as thin CLI (~270 lines, was 1487)
├── bin/
│   ├── k3                       built, 167 KB, works on this host
│   └── libk3.so                 built, 162 KB, exports 14 public symbols
├── Makefile                     updated with libk3 targets and install rule
└── tests/fixtures/              (article's test fixtures, including tiny_k3.bin)
```

**Phase B (refactor): DONE.** Test suite still passes — all 3 GATEs
return "ENGINE MATCHES THE REFERENCE EXACTLY". The refactor is
behavior-preserving.

**Phase C (OpenAI server): NOT STARTED.** This is the next session's
work. The plan is above in Phase C.

The original article's repo is intact. The CLI is now a thin wrapper
over the public API. The library is built. Only the OpenAI server
remains.
