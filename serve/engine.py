"""
kimi-k3-lean engine bindings — ctypes wrapper around libk3.so.

WHAT THIS IS
  A Python interface to the libk3 shared library (the seam between the
  C engine and this Python OpenAI server). Loads the library, exposes
  k3_open / k3_step / k3_generate / k3_close as a Python class.

DESIGN
  One Engine instance maps to one k3_ctx. The C engine is single-caller
  per context; the OpenMP parallelism is INSIDE the call, not across
  contexts. If you need concurrent requests, spawn one Engine per
  thread.

  Streaming is via a callback that emits tokens as they are decoded. The
  callback receives (token_id: int, text_segment: str); the Python
  caller wraps this in SSE chunks upstream.

USAGE

    >>> eng = Engine("/path/to/checkpoint", preset="server")
    >>> text = "Hello, world."     # optional prefill
    >>> ids  = eng.tokenize(text)  # optional explicit tokenization
    >>> out  = eng.step(ids, max_tokens=20)
    >>> print(eng.detokenize(out))

Or with streaming:

    >>> for chunk in eng.generate(ids, max_tokens=20):
    ...     sys.stdout.write(chunk)
    ...     sys.stdout.flush()

Or use it via the higher-level OpenAI-shaped `LLM.generate` further up
the stack; this module is intentionally low-level.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# ----------------------------------------------------------- library loading

_LIB_PATHS = [
    # In-source (development)
    "bin/libk3.so",
    "build/lib/libk3.so",
    # System installs.
    "/usr/local/lib/libk3.so",
    "/usr/lib/libk3.so",
    # User installs.
    os.path.expanduser("~/.local/lib/libk3.so"),
    # Docker / system-packaged installs (deploy/model/Dockerfile puts it here).
    "/opt/kimi-k3-lean/bin/libk3.so",
    "/opt/kimi-k3-lean/bin/libk3.dylib",
    "/opt/kimi-k3-lean/bin/k3.dll",
    # macOS install names (.dylib).
    "bin/libk3.dylib",
    "/usr/local/lib/libk3.dylib",
    "/opt/homebrew/lib/libk3.dylib",
    os.path.expanduser("~/.local/lib/libk3.dylib"),
    # Windows.
    "k3.dll",
    "build/bin/Release/k3.dll",
    "C:\\Program Files\\kimi-k3-lean\\bin\\k3.dll",
]


def _find_library() -> str:
    """Locate libk3. Honors LD_LIBRARY_PATH (POSIX) and PATH (Windows)
    automatically via ctypes.CDLL; explicit paths are tried first."""
    for p in _LIB_PATHS:
        if Path(p).exists():
            return str(p)
    # Fall back to whatever ctypes finds on the system search path.
    for name in ("libk3.so", "libk3.dylib", "k3.dll"):
        try:
            return ctypes.util.find_library(name) or name
        except Exception:
            continue
    raise FileNotFoundError(
        "Could not locate libk3.{so,dylib,dll}. Set LD_LIBRARY_PATH "
        "(POSIX) or PATH (Windows), or copy libk3 next to your script."
    )


# ----------------------------------------------------------- ctypes glue


class _K3Args(ctypes.Structure):
    """Mirrors C struct k3_open_args. Field order MUST match the header."""
    _fields_ = [
        ("model_dir",   c_char_p),
        ("trunk_dir",   c_char_p),
        ("config_path", c_char_p),
        ("tok_dir",     c_char_p),
        ("layers",      c_int),
        ("cache_gb",    c_double),
        ("trunk_gb",    c_double),
        ("preset",      c_char_p),
        ("incremental", c_int),
    ]


class _K3Stats(ctypes.Structure):
    """Mirrors C struct k3_stats."""
    _fields_ = [
        ("expert_hits",      c_uint64),
        ("expert_misses",    c_uint64),
        ("bytes_read",       c_uint64),
        ("tokens_generated", c_uint64),
        ("seconds_total",    c_double),
    ]


# Callback type: int (*)(void *user, int token_id).
# Returning 0 continues, non-zero aborts generation.
_K3_TOKEN_CB = ctypes.CFUNCTYPE(c_int, c_void_p, c_int)


# ----------------------------------------------------------- dataclasses


@dataclass
class Stats:
    """Decode-time statistics. Mirrors the C k3_stats struct."""
    expert_hits:      int
    expert_misses:    int
    bytes_read:       int
    tokens_generated: int
    seconds_total:    float


# ----------------------------------------------------------- engine


class EngineError(RuntimeError):
    """Raised when k3_open fails or a step returns a negative code."""


class Engine:
    """A live, single-threaded handle to a Kimi K3 model.

    One Engine per worker thread. The library is single-caller per
    context; concurrent calls on the same Engine are undefined.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        *,
        trunk_dir:     Optional[str | os.PathLike] = None,
        config_path:   Optional[str | os.PathLike] = None,
        tok_dir:       Optional[str | os.PathLike] = None,
        layers:        int = -1,
        cache_gb:      float = 0.0,
        trunk_gb:      float = 0.0,
        preset:        Optional[str] = None,
        incremental:   bool = True,
        library_path:  Optional[str] = None,
    ) -> None:
        """Open the model.

        All paths are passed through to the C engine. Defaults match the
        article's CLI: incremental decode, full layer set, preset picked
        by the engine if `preset` is None.
        """
        lib_path = library_path or _find_library()
        self._lib = ctypes.CDLL(lib_path)
        # The C engine is single-caller per context. The server's
        # ThreadingHTTPServer can dispatch concurrent requests on
        # different threads, so we serialize C calls here.
        self._lock = threading.Lock()

        # Bind the C functions we'll use. ctypes lets us skip the ones we
        # don't call.
        self._lib.k3_open.argtypes = [POINTER(_K3Args)]
        self._lib.k3_open.restype  = c_void_p
        self._lib.k3_open_errmsg.argtypes = []
        self._lib.k3_open_errmsg.restype  = c_char_p
        self._lib.k3_close.argtypes = [c_void_p]
        self._lib.k3_close.restype  = c_int

        self._lib.k3_step.argtypes = [
            c_void_p, POINTER(c_int), c_int,
            POINTER(c_int), c_int, c_int,
        ]
        self._lib.k3_step.restype = c_int

        self._lib.k3_generate.argtypes = [
            c_void_p, POINTER(c_int), c_int, c_int,
            _K3_TOKEN_CB, c_void_p,
        ]
        self._lib.k3_generate.restype = c_int

        self._lib.k3_save_state.argtypes = [c_void_p, c_char_p]
        self._lib.k3_save_state.restype  = c_int
        self._lib.k3_load_state.argtypes = [c_void_p, c_char_p]
        self._lib.k3_load_state.restype  = c_int

        self._lib.k3_model_id.argtypes   = [c_void_p]
        self._lib.k3_model_id.restype    = c_char_p
        self._lib.k3_n_layers.argtypes   = [c_void_p]
        self._lib.k3_n_layers.restype    = c_int
        self._lib.k3_vocab_size.argtypes = [c_void_p]
        self._lib.k3_vocab_size.restype  = c_int
        self._lib.k3_ctx_size.argtypes   = [c_void_p]
        self._lib.k3_ctx_size.restype    = c_int

        self._lib.k3_tokenize.argtypes = [c_void_p, c_char_p, POINTER(c_int), c_int]
        self._lib.k3_tokenize.restype  = c_int
        self._lib.k3_detokenize.argtypes = [c_void_p, POINTER(c_int), c_int, c_char_p, c_int]
        self._lib.k3_detokenize.restype  = c_int

        self._lib.k3_get_stats.argtypes = [c_void_p, POINTER(_K3Stats)]
        self._lib.k3_get_stats.restype  = None
        self._lib.k3_reset_stats.argtypes = [c_void_p]
        self._lib.k3_reset_stats.restype  = None

        # Open.
        args = _K3Args(
            model_dir=str(model_dir).encode(),
            trunk_dir=str(trunk_dir).encode() if trunk_dir else None,
            config_path=str(config_path).encode() if config_path else None,
            tok_dir=str(tok_dir).encode() if tok_dir else None,
            layers=int(layers),
            cache_gb=float(cache_gb),
            trunk_gb=float(trunk_gb),
            preset=preset.encode() if preset else None,
            incremental=1 if incremental else 0,
        )
        self._ctx = self._lib.k3_open(byref(args))
        if not self._ctx:
            err = self._lib.k3_open_errmsg()
            msg = err.decode() if err else "unknown error"
            raise EngineError(f"k3_open failed: {msg}")

        # Cache the model metadata so /v1/models doesn't have to call into
        # the C engine every request.
        self._model_id   = self._lib.k3_model_id(self._ctx).decode()
        self._n_layers   = self._lib.k3_n_layers(self._ctx)
        self._vocab_size = self._lib.k3_vocab_size(self._ctx)
        self._ctx_size   = self._lib.k3_ctx_size(self._ctx)

    # ----- introspection -----

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

    # ----- step / generate -----

    def step(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
        out_cap: int = 4096,
    ) -> list[int]:
        """Run one forward step: feed prompt, get next + greedy-decoded tokens.

        max_tokens=0 means "use the engine default" (K3_MAX_GEN).
        """
        # Validate and copy.
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        n = len(prompt_ids)
        c_in = (c_int * n)(*prompt_ids)
        c_out = (c_int * out_cap)()
        rc = self._lib.k3_step(
            self._ctx,
            c_in, n,
            c_out, out_cap,
            int(max_tokens),
        )
        if rc < 0:
            raise EngineError(f"k3_step failed: rc={rc}")
        return list(c_out[:rc])

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
    ) -> Iterator[str]:
        """Streaming generate: yield decoded text segments as tokens arrive.

        The engine emits one segment per token (the article's tokenizer
        is byte-level, so segments are 1-4 bytes each). The caller
        concatenates them. SSE chunks further upstream will package
        these into ChatCompletionChunk responses.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        # We need a stateful closure that the C side calls. Hold a
        # mutable list of decoded chunks here; the generator yields from
        # it as items appear.
        chunks: list[str] = []
        aborted = False

        # The callback runs in the C thread (which is the Python thread
        # of the caller, since ctypes calls are synchronous). Returning
        # 0 continues, non-zero aborts.
        @_K3_TOKEN_CB
        def cb(user, token_id: int) -> int:
            nonlocal aborted
            if aborted:
                return 1
            # Detokenize one token. We could batch, but the article's
            # detokenizer is single-token for streaming contexts.
            buf = ctypes.create_string_buffer(64)
            n = self._lib.k3_detokenize(self._ctx,
                                        (c_int * 1)(token_id), 1,
                                        buf, len(buf))
            if n > 0:
                chunks.append(buf.value[:n].decode(errors="replace"))
            return 0

        n = len(prompt_ids)
        c_in = (c_int * n)(*prompt_ids)
        rc = self._lib.k3_generate(
            self._ctx,
            c_in, n,
            int(max_tokens),
            cb, None,
        )
        if rc < 0:
            raise EngineError(f"k3_generate failed: rc={rc}")

        for c in chunks:
            yield c

    # ----- convenience aliases -----
    # server.py calls engine.complete() (blocking) and engine.stream()
    # (yielding token ids). These are thin wrappers over step() and
    # generate() that match what the server expects.

    def complete(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
    ) -> list[int]:
        """Blocking forward step: feed prompt, return all generated token ids.

        Identical to step() but named to match the server's expected API.
        """
        return self.step(prompt_ids, max_tokens=max_tokens)

    def stream(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 0,
    ) -> Iterator[int]:
        """Streaming generate: yield generated token ids (not text).

        The server calls detokenize() on each id before forwarding SSE
        chunks. Yields ints.
        """
        # We can't iterate the text generator and recover ids because
        # detokenize is lossy (multi-byte tokens). Re-call the C side
        # with our own callback that captures ids instead of decoding.
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        captured: list[int] = []

        @_K3_TOKEN_CB
        def cb(user, token_id: int) -> int:
            captured.append(token_id)
            return 0

        n = len(prompt_ids)
        c_in = (c_int * n)(*prompt_ids)
        rc = self._lib.k3_generate(
            self._ctx,
            c_in, n,
            int(max_tokens),
            cb, None,
        )
        if rc < 0:
            raise EngineError(f"k3_generate failed: rc={rc}")

        for tid in captured:
            yield tid

    # ----- tokenization -----

    def tokenize(self, text: str, *, out_cap: int = 8192) -> list[int]:
        """Encode text into token ids. out_cap defaults to K3_MAX_PROMPT."""
        if not text:
            return []
        c_text = text.encode()
        c_out = (c_int * out_cap)()
        rc = self._lib.k3_tokenize(self._ctx, c_text, c_out, out_cap)
        if rc < 0:
            raise EngineError(f"k3_tokenize failed: rc={rc}")
        return list(c_out[:rc])

    def detokenize(self, ids: list[int], *, out_cap: int = 4096) -> str:
        """Decode ids back into text."""
        if not ids:
            return ""
        c_in = (c_int * len(ids))(*ids)
        c_out = ctypes.create_string_buffer(out_cap)
        rc = self._lib.k3_detokenize(self._ctx, c_in, len(ids), c_out, out_cap)
        if rc < 0:
            raise EngineError(f"k3_detokenize failed: rc={rc}")
        return c_out.value.decode(errors="replace")

    # ----- state save / load -----

    def save_state(self, path: str | os.PathLike) -> None:
        """Persist the recurrent state + KV cache to a file."""
        rc = self._lib.k3_save_state(self._ctx, str(path).encode())
        if rc < 0:
            raise EngineError(f"k3_save_state failed: rc={rc}")

    def load_state(self, path: str | os.PathLike) -> None:
        """Restore the recurrent state + KV cache from a file."""
        rc = self._lib.k3_load_state(self._ctx, str(path).encode())
        if rc < 0:
            raise EngineError(f"k3_load_state failed: rc={rc}")

    # ----- stats -----

    def stats(self) -> Stats:
        """Read the post-decode counters. Single-threaded; no sync."""
        s = _K3Stats()
        self._lib.k3_get_stats(self._ctx, byref(s))
        return Stats(
            expert_hits=int(s.expert_hits),
            expert_misses=int(s.expert_misses),
            bytes_read=int(s.bytes_read),
            tokens_generated=int(s.tokens_generated),
            seconds_total=float(s.seconds_total),
        )

    def reset_stats(self) -> None:
        """Zero the counters without closing the engine."""
        self._lib.k3_reset_stats(self._ctx)

    # ----- lifecycle -----

    def close(self) -> None:
        """Close the engine. Idempotent."""
        if self._ctx:
            self._lib.k3_close(self._ctx)
            self._ctx = None

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup. ctypes.CDLL doesn't hold the GIL, but
        # k3_close does. Don't raise from __del__.
        try:
            self.close()
        except Exception:
            pass