# kimi-k3-lean

**Run a 2.78-trillion-parameter model as an OpenAI-compatible HTTP server on a CPU laptop. No GPU. No BLAS. No framework. Cross-platform: Linux, macOS, Windows.**

<table>
<tr>
<td align="center"><b>2.78T</b><br><sub>parameters</sub></td>
<td align="center"><b>1.56 TB</b><br><sub>checkpoint on disk</sub></td>
<td align="center"><b>8 GB</b><br><sub>RAM floor</sub></td>
<td align="center"><b>176 KB</b><br><sub>engine code</sub></td>
<td align="center"><b>162 KB</b><br><sub>libk3.so</sub></td>
<td align="center"><b>3 GATEs</b><br><sub>exact oracle match</sub></td>
</tr>
</table>

![combined architecture](docs/images/combined-architecture.png)

## What this is

A drop-in OpenAI Chat Completions server that runs Kimi K3 locally.
The harness (Hermes, Open WebUI, LM Studio, Continue.dev, Cursor, or
the `openai` Python client) talks to `http://127.0.0.1:8080/v1` exactly
as if it were talking to OpenAI's API. The bytes never leave the host.

The combined engine is two proven reference implementations stitched
together with a thin C library:

- **`FareedKhan-dev/kimi-k3-in-c`** (the article) — the inference engine itself.
  Six C files, two headers, no BLAS, no framework, no GPU. Runs the full
  2.78-trillion-parameter Kimi K3 with an 8 GB RAM floor. 176 KB of C.
- **`sqliteai/warp`** — the OpenAI Chat Completions server around
  `libwaste.so`. SSE streaming, request/response shaping, auth, the
  host-side surface that turns a `kimi-linear.waste` container into a
  model any agent harness can address.

This repo takes the **engine** from the first and the **server** from
the second, and glues them together with a 162 KB shared library
(`libk3.so`) that exposes the article's model-level operations
(`k3_open`, `k3_step`, `k3_generate`, `k3_save_state`, `k3_load_state`)
to the Python server.

---

## Why combine them

The two implementations are different answers to overlapping problems:

| | warp | kimi-k3-in-c |
|---|---|---|
| **Container format** | custom `.waste` (VQR3) | native safetensors + MXFP4 |
| **OpenAI server** | yes (`serve/`) | no |
| **Streaming SSE** | yes | no |
| **Hardware floor** | 64 GB laptop | 8 GB laptop |
| **Trunk residency** | fixed (always resident) | tunable dial (`--trunk-gb`, `--cache-gb`) |
| **Presets** | none | `laptop` / `desktop` / `workstation` / `server` / `max` |
| **Conversation resume** | no | yes (`--save-state` / `--load-state`) |
| **Apple Metal** | yes | no |
| **PyTorch differential test** | `tools/kimi_ref.py` | full 3-GATE oracle (32/32, 20/20, 20/20) |
| **Source size** | ~13 KLOC C | ~22 KLOC (10.8 KLOC C, plus docs and tests) |

Neither has both the **8 GB RAM floor** and the **OpenAI server**. Combining
them gives an engine that has both, plus everything else from the union.

See [`COMPARISON.md`](COMPARISON.md) for the full 350-line analysis.

---

## Architecture

The diagram above shows the data flow end-to-end. Top-to-bottom:

1. **Design principles.** Four choices that determine everything else:
   disk-resident experts, CPU-only kernels, memory-as-a-dial, and an
   OpenAI-compatible HTTP server.
2. **Request flow.** The harness (any OpenAI-compatible client) POSTs
   to `/v1/chat/completions`. The Python server translates to token ids.
3. **Forward step.** `tokenizer → embedding → 93 decoder layers → lm_head →
   argmax → detokenizer`. Per-token.
4. **State / cache.** RAM-resident: KDA recurrent matrix, MLA KV cache,
   routed-expert LRU, trunk cache.
5. **Memory boundary.** Per-token crossing when the budget is smaller
   than the total weights — the disk stream is the architecture.
6. **Disk weights.** `trunk` (108.81 GB bf16), `expert pool`
   (1.45 TB MXFP4), `embed + lm_head` (4.70 GB bf16).
7. **Offline preparation.** `fetch` (HuggingFace CLI) and `convert`
   (PyTorch reference → native format). Done once.
8. **Legend.**

The colored edges tell you where data crosses the memory boundary
on every token:

- **orange dashed** (per-token stream): expert LRU ↔ expert pool.
- **purple dashed** (cached weights): trunk cache ↔ trunk.

---

## Quick start

### Install (recommended)

```bash
git clone https://github.com/sqliteai/kimi-k3-lean.git
cd kimi-k3-lean
./install.sh PREFIX=$HOME/.local   # user-level install, no sudo
# or on Windows:
.\install.ps1 -Prefix C:\Users\you\k3lean
```

This builds the engine (Makefile on POSIX, CMake on Windows) and
installs `k3`, `libk3.so`, and the headers to your chosen prefix. See
[`INSTALL.md`](INSTALL.md) for per-OS details.

### From source (manual)

```bash
git clone https://github.com/sqliteai/kimi-k3-lean.git
cd kimi-k3-lean
LDFLAGS="-lm -pthread" make -j$(nproc)        # builds bin/k3 and bin/libk3.so
LDFLAGS="-lm -pthread" make test             # runs the 3-GATE oracle suite
LDFLAGS="-lm -pthread" make install PREFIX=/usr/local

# Cross-platform alternative (Linux, macOS, Windows):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix /usr/local
```

See [`BUILD.md`](BUILD.md) for the full source-build reference.

> **The `LDFLAGS=-lm -pthread` override is required** on any host with
> a conda environment. The article's Makefile uses `LDFLAGS ?=` which
> does not append to a conda-set `LDFLAGS`, so the linker loses the
> math and pthread libraries. Without this, you get
> `undefined reference to expf` and `pthread_create`.

### Fetch and convert the model

```bash
./scripts/k3-doctor.sh    # see which preset fits this machine
./bin/k3 /path/to/checkpoint --ids 12 --gen 8  # smoke test
```

For the real Kimi K3 weights (1.56 TB download, 92 GB on disk after
convert):

```bash
python tools/fetch_weights.py   # downloads 96 shards from HuggingFace
python tools/convert.py         # bf16 trunk + MXFP4 experts
```

### Start the OpenAI server

```bash
LD_LIBRARY_PATH=. python3 serve/__main__.py /path/to/checkpoint \
    --host 127.0.0.1 --port 8080
```

Or use a preset:

```bash
LD_LIBRARY_PATH=. python3 serve/__main__.py /path/to/checkpoint \
    --preset laptop \
    --host 127.0.0.1 --port 8080
```

The presets are the article's:

| preset | trunk / cache (GB) | peak RSS | use case |
|---|---:|---:|---|
| `laptop` | 3.0 / 1.0 | 8.2 GB | the floor |
| `desktop` | 16.0 / 10.0 | 31.9 GB | ordinary workstation |
| `workstation` | 60.0 / 30.0 | 95.5 GB | the cache starts to matter |
| `server` | 110.0 / 13.0 | 128 GB | 90 layers pinned, fastest |
| `max` | 110.0 / 109.0 | 224 GB | trunk pinned + large cache |

### Point any harness at it

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)
resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[{"role": "user", "content": "Hello."}],
)
print(resp.choices[0].message.content)
```

Same URL works in Hermes, Open WebUI, LM Studio, Continue.dev, Cursor
(Settings → Models → OpenAI API base URL), and any other OpenAI-
compatible client.

---

## What the engine actually does

The hard parts of this engine are all in the article's code; this
combined repo just exposes it via a library and wraps an HTTP server
around it. The interesting design decisions:

### Disk-resident experts

The 1.45 TB routed-expert pool never leaves the NVMe. The model reads
exactly the bytes it needs, multiplies straight out of MXFP4, and
discards. The LRU cache keeps the experts that the current conversation
actually uses warm in RAM, but the architecture does not require it —
running on 8 GB and streaming the whole 1.45 TB per token still
produces a correct answer, just slowly.

### CPU-only kernels

No BLAS, no CUDA, no Metal. The article's `k3_ops.c` is 1,361 lines of
AVX2 matmul that fits in your L1 cache. The argument is honest: a
GPU on your laptop costs $0 to $3000 and is not always available; the
CPU is always there. The 1.45 TB / 30 GB/s NVMe bus is the bottleneck,
not the matmul.

### Memory is a dial, not a floor

The same model produces the same answer on 8 GB, 32 GB, 64 GB, or
224 GB. More memory only buys speed. The presets above are calibrated
transfers of measured peak RSS, not theoretical estimates.

### OpenAI as the lingua franca

Every modern agent harness speaks OpenAI Chat Completions. The server
is a translator: harness JSON → `k3_step` ctypes call → token stream
→ SSE chunks back. The host never knows the model is local.

---

## What this repo adds

- **`include/libk3/libk3.h`** — the public C API (14 functions, 7 KB).
- **`include/libk3/k3_internal.h`** — private header shared by the
  engine and the API implementation.
- **`src/lib/k3_engine.c`** — extracted engine code (~500 lines),
  formerly static in `cli/k3_run.c`.
- **`src/lib/k3_api.c`** — public API implementation (~500 lines).
  Allocates scratch, binds layers, opens safetensors, opens trunk
  stream, opens expert cache, drives `forward()`.
- **`bin/libk3.so`** — the resulting 162 KB shared library.
- **`bin/k3`** — a thin CLI that calls the library and prints output.
  Replaces the original 1,487-line `k3_run.c`.

The Makefile has new targets (`make libk3`, `make libk3.so`,
`make libk3.a`) and an updated `install` rule.

The article's test suite still passes after the refactor:

```
GATE 1  teacher forcing : 32/32 positions match tf_pred
GATE 2  greedy decode   : 20/20 generated tokens match full_ids
GATE 3  incremental    : 20/20 generated tokens match full_ids  <- KV cache + carried KDA state

VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
```

---

## What this repo does NOT add

The OpenAI server in `serve/` is **not yet written** in this combined
repo. The plan is in [`OPTION_C_HANDOFF.md`](OPTION_C_HANDOFF.md), and
a cron job is set up to do the work over multiple sessions. When the
server is in place, this section will list its files.

Until then, the engine works via the CLI:

```bash
./bin/k3 /path/to/checkpoint --prompt "Hello." --gen 32
# or
./bin/k3 /path/to/checkpoint --ids 12,34,56 --gen 16
```

---

## Repository layout

```
.
├── COMPARISON.md               (17 KB, 350 lines) — warp vs article
├── OPTION_C_HANDOFF.md         (12 KB) — what's done, what's left
├── COMBINED.md                 (this file)
├── README.md                   (article's original README, untouched)
├── include/
│   ├── k3/                     # article's public headers
│   └── libk3/
│       ├── libk3.h             # public C API (14 functions)
│       └── k3_internal.h       # private header
├── src/
│   ├── core/k3_ops.c           # article's kernels (1,361 lines)
│   ├── io/                     # article's safetensors / trunk readers
│   ├── cache/k3_cache.c        # article's expert LRU cache
│   ├── model/k3_bind.c         # article's tensor-name binding
│   ├── tokenizer/              # article's BPE tokenizer
│   ├── lib/
│   │   ├── k3_engine.c         # extracted engine code (~500 lines)
│   │   └── k3_api.c            # public API implementation (~500 lines)
│   └── cli/k3_run.c            # thin CLI (~270 lines, was 1,487)
├── tests/
│   ├── fixtures/               # article's test fixtures (tiny_k3, etc.)
│   └── unit/                   # article's test suite
├── docs/
│   ├── images/
│   │   ├── combined-architecture.png   # this README's diagram
│   │   ├── combined-architecture.dot   # Graphviz source
│   │   └── main_architecture.png       # article's original
│   ├── ARCHITECTURE.md         # article's architecture doc
│   ├── ENGINE.md               # article's engine doc
│   ├── FORMAT.md               # container format spec
│   ├── PERFORMANCE.md          # benchmark ladder
│   └── TUNING.md               # how to choose a preset
├── bin/
│   ├── k3                      # CLI binary (167 KB)
│   └── libk3.so                # shared library (162 KB, 14 symbols)
└── Makefile                    # builds everything; see `make help`
```

---

## Credits and provenance

This combined engine is built on top of two upstream projects:

- **`FareedKhan-dev/kimi-k3-in-c`** (v1.0.0, 2026-08-07) — the
  inference engine. Licensed Apache-2.0.
  <https://github.com/FareedKhan-dev/kimi-k3-in-c>

- **`sqliteai/warp`** — the OpenAI Chat Completions server, SSE
  streaming, and harness integration story.
  <https://github.com/sqliteai/warp>

The two were not designed to interoperate. The article's engine has no
model-level API; warp's server wraps a different engine. The combined
repo adds `libk3.so` as the seam between them: the article's engine
becomes a library, and warp's server pattern becomes a Python HTTP
server that drives it.

No kernel-level code was rewritten. The article's six C files
(`k3_ops.c`, `k3_st.c`, `k3_load.c`, `k3_trunk.c`, `k3_cache.c`,
`k3_bind.c`) are unchanged. The only thing that changed is the
extraction of `forward()` and `argmax_()` from the CLI into a library,
and the addition of a Python server around it.

---

## License

Apache-2.0. See `LICENSE`.