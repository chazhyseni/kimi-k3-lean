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

        Prefers the direct binary over wrapper scripts so that env vars
        set by start() are passed directly to the process (not through
        a bash wrapper that may overwrite or lose them).
        """
        prefix = Path(os.environ.get("LITMOE_PREFIX", Path.home() / ".local"))

        # First: look for the direct binary in known install locations
        # Prebuilt binaries are in subdirectories like prebuilt/llama-b10516/
        # Source builds are in local/
        for lib_subdir in ["lib/llama.cpp/local", "lib/llama.cpp/prebuilt"]:
            lib_dir = prefix / lib_subdir
            direct = lib_dir / "llama-server"
            if direct.exists():
                return (str(direct), lib_dir)
            # Search recursively for nested archives (e.g. prebuilt/llama-b10516/)
            if lib_dir.exists():
                for p in lib_dir.rglob("llama-server"):
                    if p.is_file():
                        return (str(p), p.parent)

        # Second: check PATH (may find a wrapper script)
        found = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if found:
            p = Path(found)
            for lib_subdir in ["lib/llama.cpp/local", "lib/llama.cpp/prebuilt"]:
                lib_dir = prefix / lib_subdir
                if lib_dir.exists():
                    return (found, lib_dir)
            # Check if shared libs are alongside the binary
            so_files = list(p.parent.glob("lib*.so*"))
            dylib_files = list(p.parent.glob("lib*.dylib*"))
            if so_files or dylib_files:
                return (found, p.parent)
            return (found, None)

        # Not found anywhere
        raise FileNotFoundError(
            "llama-server not found. Build from https://github.com/ggml-org/llama.cpp "
            "or install with: litmoe install --engine llamacpp"
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

        # Threads — cap at 8 for efficiency; more threads have diminishing
        # returns for LLM inference and waste CPU on scheduling overhead
        n_threads = min(os.cpu_count() or 8, 8)
        cmd.extend(["-t", str(n_threads)])

        # Host/port
        cmd.extend(["--host", "127.0.0.1", "--port", str(self.default_port())])

        # Extra args
        cmd.extend(self.model.extra_args)

        # Store lib_dir for start() to set env vars
        self._lib_dir = lib_dir

        return cmd

    def start(self, log_dir: Path | None = None) -> None:
        """Start the engine.

        On macOS, launches via /bin/bash to work around com.apple.provenance
        which blocks Python's execve() from running the binary directly.
        The wrapper script (rewritten at serve time with correct env vars)
        sets DYLD_FALLBACK_LIBRARY_PATH and execs the binary.
        On Linux, launches the binary directly with LD_LIBRARY_PATH.
        """
        cmd = self.build_command()
        env = os.environ.copy()
        env.update(self.model.env)

        # Prepare log file (used by all branches below)
        ld = Path(log_dir) if log_dir else Path("logs")
        ld.mkdir(parents=True, exist_ok=True)
        log_file = ld / f"{self.model.id}.log"
        self._log_path = log_file

        lib_dir = getattr(self, "_lib_dir", None)
        if lib_dir and is_macos():
            # macOS: com.apple.provenance blocks Python subprocess from
            # launching the binary. Strip it by copying binary over itself.
            # Then launch via /bin/bash as a belt-and-suspenders approach.
            actual_binary = lib_dir / "llama-server"
            if actual_binary.exists():
                try:
                    import shutil as _shutil
                    tmp = actual_binary.parent / ".llama-server.tmp"
                    _shutil.copy2(str(actual_binary), str(tmp))
                    _shutil.move(str(tmp), str(actual_binary))
                    os.chmod(str(actual_binary), 0o755)
                except Exception:
                    pass

            wrapper = Path(os.environ.get("LITMOE_PREFIX", Path.home() / ".local")) / "bin" / "llama-server"
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            is_prebuilt = "prebuilt" in str(lib_dir)
            if is_prebuilt:
                wrapper.write_text(f"#!/bin/bash\nexec {actual_binary} \"$@\"\n")
            else:
                wrapper.write_text(
                    f"#!/bin/bash\n"
                    f"export DYLD_FALLBACK_LIBRARY_PATH={lib_dir}:$DYLD_FALLBACK_LIBRARY_PATH\n"
                    f"export DYLD_LIBRARY_PATH={lib_dir}:$DYLD_LIBRARY_PATH\n"
                    f"exec {actual_binary} \"$@\"\n"
                )
            wrapper.chmod(0o755)

            # Launch via bash
            shell_cmd = f'"{wrapper}" ' + ' '.join(f'"{a}"' for a in cmd[1:])
            with open(log_file, "w") as logf:
                self.process = subprocess.Popen(
                    ['/bin/bash', '-c', shell_cmd],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
        elif lib_dir:
            # Linux: direct launch with LD_LIBRARY_PATH
            env.update(get_library_path_env(lib_dir))
            with open(log_file, "w") as logf:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
        else:
            # No lib_dir needed (prebuilt binary with system libs)
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
    """Check if llama-server is installed.

    Checks PATH first, then common install locations (~/.local, etc.).
    """
    # Check PATH
    if shutil.which("llama-server") or shutil.which("llama-server.exe"):
        return True

    # Check install prefix locations
    from litmoe.platform_utils import find_llama_server_binary
    binary = find_llama_server_binary()
    return binary is not None
