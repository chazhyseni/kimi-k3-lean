"""llama.cpp engine adapter.

Native support for Kimi-K3, Qwen3.5 MoE, and many other architectures via GGUF.
Backends: CUDA, HIP (AMD), Metal (Apple), Vulkan, SYCL, OpenCL, CANN (Ascend).
Quantization: 1.5/2/3/4/5/6/8-bit. SIMD: AVX, AVX2, AVX-512, AMX.

Reference: https://github.com/ggml-org/llama.cpp
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from litmoe.engines.base import Engine
from litmoe.config import ModelEntry


class LlamaCppEngine(Engine):
    """Adapter for llama.cpp server (llama-server)."""

    def default_port(self) -> int:
        # Assign unique ports per model to avoid conflicts when running
        # multiple models. Hash the model id to get a stable port.
        base = 8081
        if self.model.id:
            return base + (hash(self.model.id) % 100)
        return base

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.default_port()}/health"

    def build_command(self) -> list[str]:
        """Build llama-server command."""
        binary = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if not binary:
            raise FileNotFoundError(
                "llama-server not found. Build from https://github.com/ggml-org/llama.cpp "
                "or install a release binary."
            )
        cmd = [binary]

        # Model: GGUF path takes priority, then HF repo, then local path
        if self.model.gguf_path:
            cmd.extend(["-m", self.model.gguf_path])
        elif self.model.model_path and self.model.model_path.startswith(("http://", "https://")):
            # HF repo URL: use -hf flag
            cmd.extend(["-hf", self.model.model_path])
        elif self.model.model_path:
            # Local directory: could be HF safetensors or GGUF
            # llama.cpp auto-detects from the directory contents
            cmd.extend(["-m", self.model.model_path])
        else:
            raise ValueError(f"{self.model.id}: model_path or gguf_path required for llama.cpp")

        # GPU offload
        if self.model.n_gpu_layers == 0:
            cmd.extend(["-ngl", "0"])
        else:
            cmd.extend(["-ngl", str(self.model.n_gpu_layers)])

        # Context size
        if self.model.n_ctx:
            cmd.extend(["-c", str(self.model.n_ctx)])

        # Threads
        cmd.extend(["-t", str(os.cpu_count() or 8)])

        # Host/port
        cmd.extend(["--host", "127.0.0.1", "--port", str(self.default_port())])

        # Extra args
        cmd.extend(self.model.extra_args)

        return cmd


def is_installed() -> bool:
    """Check if llama-server is installed."""
    return shutil.which("llama-server") is not None or shutil.which("llama-server.exe") is not None
