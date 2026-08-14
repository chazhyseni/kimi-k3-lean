"""
fake_engine.py — a stand-in Engine for HTTP-layer testing.

WHAT THIS IS
  A drop-in for serve.engine.Engine that returns canned responses
  without loading any model. Use with `--dry-run` to exercise the HTTP
  routing, request validation, SSE framing, and auth without the C
  engine or 982 GB of model data.

WHAT IT IS NOT
  A real model. The output is a fixed string ("hello from fake engine")
  regardless of input. Don't ship this in production; the production
  path is serve.engine.Engine.

USAGE

    python3 serve/__main__.py /path/to/anything --dry-run --port 8080

Then:

    curl http://127.0.0.1:8080/v1/models
    curl -X POST http://127.0.0.1:8080/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model":"kimi-k3-lean","messages":[{"role":"user","content":"hi"}]}'
"""

from __future__ import annotations

from typing import Iterator


# The same canned response every request. Deterministic; good for testing.
_FAKE_REPLY = "hello from fake engine"
_FAKE_STREAM_TOKENS = list(_FAKE_REPLY)  # one character per token


class FakeEngine:
    """A stand-in for Engine that returns canned responses.

    Implements only the methods the HTTP server calls. Other Engine
    methods (save_state, load_state, stats, etc.) raise NotImplementedError.
    """

    def __init__(
        self,
        model_path: str = "",
        *,
        model_id:  str = "kimi-k3-lean-fake",
        n_layers:   int = 93,
        vocab_size: int = 163840,
        ctx_size:   int = 32768,
    ) -> None:
        self._model_id   = model_id
        self._n_layers   = n_layers
        self._vocab_size = vocab_size
        self._ctx_size   = ctx_size
        self._model_path = model_path

        # Mimic the real Engine's lock so the server's threading code
        # still exercises the contention path.
        import threading
        self._lock = threading.Lock()

    # ----- introspection (mirrors Engine) -----

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def ctx_size(self) -> int:
        return self._ctx_size

    # ----- tokenization (deterministic hash → token ids) -----

    def tokenize(self, text: str, *, out_cap: int = 8192) -> list[int]:
        """Hash each character to a token id; deterministic."""
        if not text:
            return []
        return [ord(c) % self._vocab_size for c in text[:out_cap]]

    def detokenize(self, ids: list[int], *, out_cap: int = 4096) -> str:
        """Inverse of the deterministic tokenize above."""
        if not ids:
            return ""
        return "".join(chr(i % 256) for i in ids[:out_cap])

    # ----- step / generate / complete / stream -----

    def step(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
        out_cap: int = 4096,
    ) -> list[int]:
        """Return the deterministic stream token ids."""
        cap = max_tokens or len(_FAKE_REPLY)
        ids: list[int] = []
        for ch in _FAKE_REPLY[: min(cap, out_cap)]:
            ids.append(ord(ch) % self._vocab_size)
        return ids

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
    ) -> Iterator[str]:
        """Yield the canned reply, one token at a time."""
        for tid in self.stream(prompt_ids, max_tokens=max_tokens):
            yield self.detokenize([tid])

    def complete(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
    ) -> list[int]:
        """Same as step; named to match the server."""
        return self.step(prompt_ids, max_tokens=max_tokens)

    def stream(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
    ) -> Iterator[int]:
        """Yield the canned reply's character ids."""
        cap = max_tokens or len(_FAKE_STREAM_TOKENS)
        for ch in _FAKE_REPLY[:cap]:
            # Encode character back into the same token-id space tokenize() uses.
            yield ord(ch) % self._vocab_size

    # ----- state save / load -----

    def save_state(self, path) -> None:
        """Write a marker file. In the real engine, this persists the
        recurrent state + KV cache; for the fake, it's a smoke-test that
        the path goes through."""
        from pathlib import Path
        Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(path)).write_bytes(b"FAKE_STATE")

    def load_state(self, path) -> None:
        """Read a marker file. The real engine restores the recurrent state
        + KV cache; for the fake, we just verify the path is readable."""
        from pathlib import Path
        p = Path(str(path))
        if not p.exists():
            raise RuntimeError(f"state file not found: {p}")
        if p.read_bytes() != b"FAKE_STATE":
            raise RuntimeError(f"state file invalid: {p}")

    # ----- lifecycle -----

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakeEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass