# Phases C/D/E — DONE

**Date:** 20260814_184413

All three phases from the Option C handoff are complete.

## Phase C: OpenAI server — DONE

Five files in `serve/`:

- `serve/__init__.py` — package init
- `serve/engine.py` (~470 lines) — ctypes wrapper around `libk3.so`.
  Maps the 14-function C API to a Python `Engine` class with
  `complete()` and `stream()` (matching what `server.py` calls) plus
  `step()` / `generate()` / `tokenize()` / `detokenize()` /
  `save_state()` / `load_state()` / `stats()`.
- `serve/chatfmt.py` — OpenAI Chat Completions message format converter
  (system / user / assistant roles; content arrays → token ids).
- `serve/api.py` — request/response shaping.
  `ChatCompletionRequest` → `Engine.complete()` / `Engine.stream()`.
  Streaming → SSE. `ChatCompletionResponse` with OpenAI-shape fields.
  `APIError(status, type_)` so the HTTP layer can return correct codes.
- `serve/server.py` (~375 lines) — `ThreadingHTTPServer` +
  `BaseHTTPRequestHandler`. Routes `/health`, `/v1/models`,
  `POST /v1/chat/completions`. Constant-time auth via
  `hmac.compare_digest`. Manual SSE chunked framing (`%X\r\n…\r\n0\r\n\r\n`).
  `data: <json>\n\n` per token; `data: [DONE]\n\n` sentinel.
- `serve/__main__.py` (~190 lines) — argparse. `--preset`, `--cache-gb`,
  `--trunk-gb`, `--trunk-dir`, `--config`, `--tok-dir`, `--layers`,
  `--api-key`, `--max-tokens`, `--host`, `--port`, `--log-requests`,
  `--dry-run`. SIGINT/SIGTERM handling. API-key warning when binding
  non-loopback.

**Added during Phase D:**

- `serve/fake_engine.py` — drop-in `FakeEngine` for HTTP-layer testing
  without the C engine or 982 GB of model weights. Implements only the
  methods the server calls; output is a fixed "hello from fake engine"
  string (one character per token). Used by `--dry-run`.

## Phase D: end-to-end test — DONE (dry-run)

Started the server with `--dry-run --port 8080` and curl-tested:

```
$ curl http://127.0.0.1:8080/health
{"status": "ok", "model": "fake", "engine_version": "1.0.0", "uptime_s": 0}

$ curl http://127.0.0.1:8080/v1/models
{"object": "list", "data": [{"id": "fake", "object": "model", "created": 1786732936,
  "owned_by": "kimi-k3-lean", "permission": []}]}

$ curl -X POST http://127.0.0.1:8080/v1/chat/completions     -H "Content-Type: application/json"     -d '{"model":"kimi-k3-lean","messages":[{"role":"user","content":"hi"}],
         "max_tokens":5}'
{"id": "chatcmpl-993b0e11c1f7447eb2d7665082bdb9e6", "object": "chat.completion",
  "created": 1786732938, "model": "kimi-k3-lean",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                 "finish_reason": "length", "logprobs": null}],
  "usage": {"prompt_tokens": 52, "completion_tokens": 5, "total_tokens": 57},
  "system_fingerprint": "fp_kimi_k3_lean"}

$ curl -N -X POST http://127.0.0.1:8080/v1/chat/completions     -H "Content-Type: application/json"     -d '{...,"stream":true}'
data: {"id": "...", "choices": [{"delta": {"role": "assistant", "content": ""}}]}
data: {"id": "...", "choices": [{"delta": {"content": "h"}}]}
data: {"id": "...", "choices": [{"delta": {"content": "e"}}]}
data: {"id": "...", "choices": [{"delta": {"content": "l"}}]}
data: {"id": "...", "choices": [{"delta": {"content": "l"}}]}
data: {"id": "...", "choices": [{"delta": {"content": "o"}}]}
data: {"id": "...", "choices": [{"delta": {}, "finish_reason": "length"}]}
[DONE]
```

Auth tests:

```
$ curl -X POST .../v1/chat/completions  (no Authorization)
{"error": {"message": "invalid API key", "type": "authentication_error"}}

$ curl -X POST .../v1/chat/completions  -H "Authorization: Bearer wrong-key"
{"error": {"message": "invalid API key", "type": "authentication_error"}}

$ curl -X POST .../v1/chat/completions  -H "Authorization: Bearer test-secret"
{... 200 OK, content "hel" ...}
```

Error envelope tests:

```
$ curl .../v1/chat/completions  -d 'not json'
{"error": {"message": "request body is not valid JSON: ...", "type": "invalid_request_error"}}

$ curl .../v1/chat/completions  -d '{"model":"kimi-k3-lean"}'
{"error": {"message": "messages must be a non-empty list",
            "type": "invalid_request_error", "param": "messages"}}

$ curl .../v1/foo
{"error": {"message": "no route for GET /v1/foo", "type": "not_found_error"}}
```

All OpenAI-shape responses. All SSE chunks correct. Auth works. Error
envelope matches the OpenAI spec.

**What's NOT verified by `--dry-run`:**

- The C engine actually loads real K3 weights and decodes tokens. That
  requires a real K3 checkpoint (1.56 TB download + 982 GB convert).
- The article's 3-GATE oracle still passes after Phase C (no C-side
  changes were made, but verified anyway):

```
GATE 1  teacher forcing : 32/32 positions match tf_pred
GATE 2  greedy decode   : 20/20 generated tokens match full_ids
GATE 3  incremental    : 20/20 generated tokens match full_ids
VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
```

So Phase D is complete at the HTTP / API-shape layer. The model-data
layer waits on real weights.

## Phase E: docs — DONE

- `ENGINE.md` (~8.5 KB) — describes the combined engine: warp's
  OpenAI server + the article's libk3. 8 GB RAM floor, trunk-as-a-dial,
  presets with the article's measured per-token times, conversation
  resume via `Engine.save_state` / `load_state`, MXFP4, and an explicit
  "what is NOT in this server" list.
- `README.md` — added a "Use this as an OpenAI-compatible server"
  section after the Quick-start verdict, with copy-pasteable curl
  examples for `/v1/chat/completions` (both blocking and streaming).

## End-to-end runbook (when you have real K3 data)

```bash
# Build (one-time).
LDFLAGS="-lm -pthread" make -j$(nproc)
LDFLAGS="-lm -pthread" make install PREFIX=$HOME/.local

# Run.
LD_LIBRARY_PATH=$HOME/.local/lib python3 -m serve /path/to/k3/checkpoint \
    --host 127.0.0.1 --port 8080 --preset server --api-key "$K3_KEY"

# Use.
curl http://127.0.0.1:8080/v1/models
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $K3_KEY" \
    -d '{"model":"kimi-k3-lean","messages":[{"role":"user","content":"hi"}]}'
```

Or via the OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.environ["K3_KEY"],
)
resp = client.chat.completions.create(
    model="kimi-k3-lean",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)
```

## What's NOT done

1. **End-to-end test against real K3 data** — see above.
2. **Homebrew formula, Windows MSI, .deb/.rpm** — install.sh and
   install.ps1 are the supported install paths.

## Repo at /home/chaz/kimi-k3-lean/

```
/home/chaz/kimi-k3-lean/
├── README.md, COMBINED.md, COMPARISON.md, OPTION_C_HANDOFF.md
├── ENGINE.md                  # Phase E
├── INSTALL.md, BUILD.md
├── CODE_OF_CONDUCT.md
├── Makefile, CMakeLists.txt
├── install.sh, install.ps1
├── Dockerfile, Dockerfile.convert
├── docker-compose.yml
├── .github/workflows/ci.yml, release.yml
├── include/libk3/libk3.h, k3_internal.h
├── src/lib/k3_engine.c, k3_api.c
├── src/cli/k3_run.c
├── bin/k3, bin/libk3.so
├── serve/
│   ├── engine.py
│   ├── fake_engine.py
│   ├── chatfmt.py
│   ├── api.py
│   ├── server.py
│   └── __main__.py
└── tests/                     # the article's test suite, 3 GATEs pass
```
