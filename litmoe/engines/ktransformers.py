"""ktransformers engine adapter.

Heterogeneous CPU+GPU MoE inference with expert offloading.
Backends: CUDA (sglang-kt), AMX (Intel Xeon 4th gen+), AVX-512, AVX2.
Supported model families: DeepSeek-V3/R1, GLM-5.x, MiniMax-M2.x, Kimi-K2.x, Qwen3.

Reference: https://github.com/kvcache-ai/ktransformers
"""
from __future__ import annotations

import shutil
from pathlib import Path

from litmoe.engines.base import Engine
from litmoe.config import ModelEntry


class KtransformersEngine(Engine):
    """Adapter for ktransformers (kt run)."""

    def default_port(self) -> int:
        # Use assigned port if set by gateway, otherwise default
        return getattr(self, "_assigned_port", 10002)

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.default_port()}/v1/models"

    def build_command(self) -> list[str]:
        """Build ktransformers server command."""
        # ktransformers server is started via python -m ktransformers.server.main
        # or via the kt-kernel CLI: kt run
        # We use the server/main.py path for maximum compatibility
        cmd = ["python", "-m", "ktransformers.server.main"]

        # Model path (safetensors directory)
        if self.model.model_path:
            cmd.extend(["--model_path", self.model.model_path])
        else:
            cmd.extend(["--model_path", self.model.id])

        # GGUF path (for quantized models)
        if self.model.gguf_path:
            cmd.extend(["--gguf_path", self.model.gguf_path])

        # GPU offload
        if self.model.n_gpu_layers > 0:
            cmd.extend(["--n_gpu_layers", str(self.model.n_gpu_layers)])

        # Context size
        if self.model.n_ctx:
            cmd.extend(["--cache_lens", str(self.model.n_ctx)])

        # Port
        cmd.extend(["--port", str(self.default_port())])

        # Backend
        cmd.extend(["--backend_type", "balance_serve"])

        # Extra args
        cmd.extend(self.model.extra_args)

        return cmd


def is_installed() -> bool:
    """Check if ktransformers is installed."""
    return shutil.which("kt") is not None or _python_module_installed("ktransformers")


def _python_module_installed(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
