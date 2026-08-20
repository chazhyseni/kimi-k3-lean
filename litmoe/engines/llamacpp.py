"""llama.cpp engine adapter.

Native support for Kimi-K3, Qwen3.5 MoE, and many other architectures via GGUF.
Backends: CUDA, HIP (AMD), Metal (Apple), Vulkan, SYCL, OpenCL, CANN (Ascend).
Quantization: 1.5/2/3/4/5/6/8-bit. SIMD: AVX, AVX2, AVX-512, AMX.

Reference: https://github.com/ggml-org/llama.cpp
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from litmoe.engines.base import Engine
from litmoe.config import ModelEntry
from litmoe.platform_utils import (
    is_macos, is_linux, get_library_path_env,
    find_llama_server_binary, fix_macos_dylib_paths,
)


class LlamaCppEngine(Engine):
    """Adapter for llama.cpp server (llama-server)."""

    def default_port(self) -> int:
        # Assign unique ports per model deterministically.
        # Use a simple ordinal: first model gets 8081, second 8082, etc.
        # The gateway sets _port_index before calling start().
        return getattr(self, "_assigned_port", 8081)

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.default_port()}/health"

    def _resolve_binary(self) -> tuple[str, Path | None]:
        """Resolve the llama-server binary path.

        Returns (binary_path, lib_dir) where lib_dir is the directory
        containing shared libraries (for env setup), or None if not needed.
        """
        # First check PATH (may be a wrapper script that handles env vars)
        found = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if found:
            # Check if it's a wrapper script or the actual binary
            p = Path(found)
            # If it's a symlink or script, the actual binary and libs are nearby
            prefix = Path(os.environ.get("LITMOE_PREFIX", Path.home() / ".local"))

            # Look for lib dirs
            for lib_subdir in ["lib/llama.cpp/local", "lib/llama.cpp/prebuilt"]:
                lib_dir = prefix / lib_subdir
                if lib_dir.exists():
                    return (found, lib_dir)

            # If the binary itself is in a directory with .so/.dylib files
            if any(p.parent.glob("lib*.so*")) or any(p.parent.glob("lib*.dylib*")):
                return (found, p.parent)

            return (found, None)

        # Not in PATH — search common install locations
        prefix = Path(os.environ.get("LITMOE_PREFIX", Path.home() / ".local"))
        binary = find_llama_server_binary(prefix)
        if binary:
            # The lib dir is where the binary lives (for source builds)
            # or its parent (for prebuilt extracts)
            lib_dir = binary.parent
            if any(lib_dir.glob("lib*.so*")) or any(lib_dir.glob("lib*.dylib*")):
                return (str(binary), lib_dir)
            return (str(binary), None)

        raise FileNotFoundError(
            "llama-server not found. Build from https://github.com/ggml-org/llama.cpp "
            "or install a release binary with: litmoe install --engine llamacpp"
        )

    def build_command(self) -> list[str]:
        """Build llama-server command."""
        binary, lib_dir = self._resolve_binary()
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

        # Store lib_dir for start() to set env vars
        self._lib_dir = lib_dir

        return cmd

    def start(self, log_dir: Path | None = None) -> None:
        """Start the engine, with library path env vars set for the subprocess.

        On macOS, sets BOTH DYLD_LIBRARY_PATH and DYLD_FALLBACK_LIBRARY_PATH.
        DYLD_FALLBACK_LIBRARY_PATH is NOT stripped by SIP for non-restricted
        (non-system) binaries, so it reliably makes shared libraries findable.

        Also runs fix_macos_dylib_paths() at serve time to patch any existing
        binary that was built before the fix was added to the install process.
        """
        cmd = self.build_command()
        env = os.environ.copy()
        env.update(self.model.env)

        # Set library path env vars so the subprocess can find shared libs
        lib_dir = getattr(self, "_lib_dir", None)
        if lib_dir:
            env.update(get_library_path_env(lib_dir))

            # On macOS, fix dylib paths at serve time too — handles binaries
            # that were built/installed before the fix was added
            from litmoe.platform_utils import is_macos
            if is_macos():
                binary_path = Path(cmd[0])
                fix_macos_dylib_paths(binary_path, Path(lib_dir))

        log_dir = log_dir or Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.model.id}.log"
        self._log_path = log_file

        with open(log_file, "w") as logf:
            self.process = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        self.base_url = f"http://127.0.0.1:{self.default_port()}"
        print(f"  {self.model.id}: started PID {self.process.pid}, logs: {log_file}")


def is_installed() -> bool:
    """Check if llama-server is installed."""
    return shutil.which("llama-server") is not None or shutil.which("llama-server.exe") is not None
