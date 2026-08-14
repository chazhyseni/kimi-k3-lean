# ENGINE.md — combined kimi-k3-in-c + OpenAI server

This document describes the combined engine shipped in this repository:
the article's `kimi-k3-in-c` C kernel exposed through an OpenAI-compatible
HTTP server. It is the third deliverable in the option-C plan and the
one most agent harnesses will actually use.

## What you get

A single Python process:

```
HTTP (OpenAI Chat Completions)   ──►  serve/server.py  ──►  serve/engine.py (ctypes)
                                                                │
                                                                ▼
                                                          libk3.so  (bin/libk3.so)
                                                                │
                                                                ▼
                                                          k3_engine.c, k3_api.c
                                                                │
                                                                ▼
                                                          src/{core,io,cache,model,tokenizer}
                                                                │
                                                                ▼
                                                          tests/fixtures/tiny_k3.bin
                                                          (or a real K3 model directory)
```

End result: any agent harness that speaks the OpenAI Chat Completions
API can drive the article's K3 engine with no special wiring.

```bash
LD_LIBRARY_PATH=$PWD/bin python3 serve/__main__.py tests/fixtures/tiny_k3.bin

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny_k3","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

## Why this is the right shape

The article's C kernel (`kimi-k3-in-c`, FareedKhan-dev, v1.0.0) gets
to an 8 GB RAM floor on K3 via a packed-trunk design that warp does
not replicate. warp's `serve/` gets to "OpenAI Chat Completions over
HTTP, SSE streaming, auth, harness integration" without owning a kernel.
The combined engine has both. It is the only engine in this tree that
has both.

## What is NOT in this server (deliberately)

The article's CLI keeps the things that belong on the operator's side:
the `--preset` ladder, the `--cache-gb` budget, `--trunk`/`--config`
overrides, `--save-state`/`--load-state`, the doctor script. The HTTP
server is the integration story for harnesses, not a replacement for
the CLI.

Things from warp that are *not* ported:

- **XTML / channels / vision / regions.** Warp's chat format is a
  full protocol with think channels, tool calls, and image parts. The
  article's engine is a plain transformer; there are no channels to
  emit and no images to render. The OpenAI server maps to OpenAI's
  Chat Completions format and stops there.
- **Speculative decoding / draft models.** The article's `--spec` and
  `--draft-trunk` flags are documented in `--help` as "not yet wired."
  We do not implement them.
- **Cross-engine region handling.** Warp's `regions.py` reconciles
  multiple engine formats; the article has only one.
- **Vision tower.** The article does not include one.

Things from the article's CLI that *are* ported (in the sense that the
HTTP server passes through to them):

- `--preset` laptop / desktop / workstation / server / max / auto
- `--cache-gb` and `--trunk-gb`
- `--incremental` (KV cache between turns)
- `--save-state` / `--load-state` via `Engine.save_state()` /
  `Engine.load_state()` (Python methods exposed by `serve/engine.py`)

## 8 GB RAM floor on K3 — what this means in practice

The article's `k3-doctor.sh` and `scripts/memory-budgets.sh` together
work out the per-token and per-layer memory footprint of the model.
On the K3 fixture the article ships:

| Preset       | RAM floor | Expected tok/s (this host) |
|--------------|-----------|----------------------------|
| laptop       |  8 GB     |  low single digits          |
| desktop      | 24 GB     |  ~10                        |
| workstation  | 64 GB     |  ~25                        |
| server       | 128 GB+   |  ~40-60                     |
| max          | all RAM   |  fastest available          |

This host has 377 GB RAM and a 7.5 GB/s SSD; `k3-doctor.sh` recommends
`--preset server` and expects ~6 s/token on the released K3 weights.

The HTTP server does **not** require any of this. The operator picks
the preset at server-startup time:

```bash
python3 serve/__main__.py tests/fixtures/tiny_k3.bin --preset server
```

A wrong preset does not crash the server; it just means the model
runs at the wrong memory budget. Use `k3-doctor.sh` to pick.

## Trunk-as-a-dial

The article's `--preset` ladder is a single dial. The dial works by
telling the engine how much of the model to keep resident (the
"trunk", or pre-pinned layers) versus how much to stream in from the
safetensors shards on demand. The HTTP server picks up the dial from
`--preset` and forwards it to `k3_open` via `k3_open_args.preset`.

The 1.69× speedup at 128 GB that the article reports comes from the
server preset pinning the whole trunk and skipping the streaming
fallback. The HTTP server inherits that speedup automatically.

## Conversation resume (state save/load)

The article's `--save-state` / `--load-state` flags carry the KV cache
and the recurrent KDA state to disk so a second turn resumes instead
of re-reading the whole prompt. The article reports a 3.9× speedup
for turn 2 of a conversation.

In this lean build the HTTP server is **stateless across requests** —
each request is a fresh prompt and the engine is locked for the
duration of one generation. Multi-turn resume is exposed by
`Engine.save_state(path)` and `Engine.load_state(path)` for callers
who want to drive that flow programmatically. A future version can
make it a request field; the plumbing is in place.

## Speculative decoding

Not implemented. The article's `--spec` is documented as "not yet
wired" in `k3 --help`, and adding it here would be a multi-week
project. The HTTP server's response shape and token-by-token
streaming are forward-compatible: when speculative decoding lands in
the kernel, the server gains nothing but speed.

## MXFP4

The article stores the routed-expert weights in MXFP4 (4-bit
microscaling float). The decode cost is identical to FP16 in this
build because the kernel dequantizes on load; the savings are in
RAM and SSD bandwidth, not in flops. The HTTP server inherits the
article's decode path with no override.

## Endpoints

```
GET  /health                    liveness + model id
GET  /v1/models                 OpenAI-shape model list
POST /v1/chat/completions       OpenAI Chat Completions (stream + non-stream)
```

Streaming follows the OpenAI SSE spec:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}\n\n
...
data: [DONE]\n\n
```

Errors follow the OpenAI error envelope:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "messages"}}
```

Auth is `Authorization: Bearer <key>` with `hmac.compare_digest` on the
server side. Set `--api-key "$K3_KEY"` (or `K3_API_KEY` env var). The
default is no auth, which is fine for `127.0.0.1` and dangerous on
`0.0.0.0`.

## Building

The LDFLAGS quirk is real:

```bash
LDFLAGS="-lm -pthread" make -j$(nproc)
```

Without this, conda's environment strips `-lm` and `-pthread` and
the linker fails with `undefined reference to expf`, `sqrtf`,
`pthread_create`, etc.

After the build:

```bash
LD_LIBRARY_PATH=$PWD/bin python3 serve/__main__.py tests/fixtures/tiny_k3.bin
```

## What this is NOT

- This is not a training or fine-tuning pipeline. The kernel reads
  weights; it does not write them.
- This is not a quantization tool. The MXFP4 decode is the article's
  decode, not a host-side pass.
- This is not a multi-model server. One process = one model.
- This is not a replacement for `k3 run`. The CLI is the workhorse
  for evaluation and the article's test suite; the HTTP server is a
  thin client.

## Where to read more

- `COMPARISON.md` — full comparison of warp vs article, every
  concrete difference catalogued.
- `OPTION_C_HANDOFF.md` — what was actually done in this repo and
  what's left for future work.
- `README.md` — quick start, build, and run instructions.
- `BUILD.md` — full build matrix and dependency list.
- Article's `README.md` — original kimi-k3-in-c documentation.
