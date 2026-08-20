"""Cross-platform system utilities for litmoe.

Handles platform detection, memory detection, and library path setup
for both Linux and macOS.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def is_macos() -> bool:
    """True on macOS (Darwin)."""
    return platform.system() == "Darwin"


def is_linux() -> bool:
    """True on Linux."""
    return platform.system() == "Linux"


def get_total_memory_bytes() -> int | None:
    """Get total system RAM in bytes. Works on both Linux and macOS.

    Linux: sysconf("SC_PHYS_PAGES") * sysconf("SC_PAGE_SIZE")
    macOS: sysctl hw.memsize (sysconf doesn't work on macOS)
    """
    if is_macos():
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return int(r.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None

    # Linux
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass

    # Fallback: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024  # kB -> bytes
    except (FileNotFoundError, ValueError, IndexError):
        pass

    return None


def get_library_path_env(lib_dir: Path) -> dict[str, str]:
    """Build environment variables for shared library discovery.

    On Linux: sets LD_LIBRARY_PATH
    On macOS: sets BOTH DYLD_LIBRARY_PATH and DYLD_FALLBACK_LIBRARY_PATH.
      DYLD_FALLBACK_LIBRARY_PATH is NOT stripped by SIP for non-restricted
      binaries, so it serves as a reliable fallback when DYLD_LIBRARY_PATH
      gets stripped.
    """
    env: dict[str, str] = {}
    lib_dir_str = str(lib_dir)

    if is_macos():
        # DYLD_FALLBACK_LIBRARY_PATH is the key: it's not stripped by SIP
        # for non-restricted (non-system) binaries. DYLD_LIBRARY_PATH may
        # be stripped, but we set both for maximum compatibility.
        existing_fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if existing_fallback:
            env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib_dir_str}:{existing_fallback}"
        else:
            env["DYLD_FALLBACK_LIBRARY_PATH"] = lib_dir_str

        existing_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        if existing_dyld:
            env["DYLD_LIBRARY_PATH"] = f"{lib_dir_str}:{existing_dyld}"
        else:
            env["DYLD_LIBRARY_PATH"] = lib_dir_str
    else:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            env["LD_LIBRARY_PATH"] = f"{lib_dir_str}:{existing}"
        else:
            env["LD_LIBRARY_PATH"] = lib_dir_str

    return env


def fix_macos_dylib_paths(binary: Path, lib_dir: Path) -> bool:
    """Fix all @rpath/@loader_path references in a macOS binary and its dylibs.

    Uses install_name_tool to:
    1. Rewrite @rpath/ and @loader_path/ references in the main binary to
       absolute paths pointing to lib_dir.
    2. For each .dylib in lib_dir, fix its own @rpath references to absolute
       paths and set its install ID to its absolute path.
    3. Add an rpath entry pointing to lib_dir as a fallback.

    This is the nuclear option — after this, the binary and all its dylibs
    have absolute paths baked in and don't need any DYLD_* env vars at runtime.

    Returns True if any fixes were applied (or if not on macOS).
    """
    if not is_macos():
        return True

    if not binary.exists():
        return False

    lib_dir_abs = str(lib_dir.resolve())
    fixes_applied = False

    def get_otool_l(path: Path) -> list[str]:
        """Get shared library references via otool -L."""
        try:
            r = subprocess.run(
                ["otool", "-L", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                lines = r.stdout.strip().split("\n")[1:]  # skip first line (the binary itself)
                return [l.strip().split(" ")[0] for l in lines if l.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []

    def fix_binary_dylib_refs(path: Path) -> bool:
        """Fix @rpath and @loader_path references in a single binary/dylib."""
        changed = False
        refs = get_otool_l(path)
        for ref in refs:
            if ref.startswith("@rpath/"):
                lib_name = ref[len("@rpath/"):]
                target = Path(lib_dir_abs) / lib_name
                if target.exists():
                    try:
                        subprocess.run(
                            ["install_name_tool", "-change", ref, str(target), str(path)],
                            capture_output=True, timeout=10,
                        )
                        changed = True
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
            elif ref.startswith("@loader_path/"):
                lib_name = ref[len("@loader_path/"):]
                target = Path(lib_dir_abs) / lib_name
                if target.exists():
                    try:
                        subprocess.run(
                            ["install_name_tool", "-change", ref, str(target), str(path)],
                            capture_output=True, timeout=10,
                        )
                        changed = True
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
        return changed

    # 1. Fix the main binary's references
    if fix_binary_dylib_refs(binary):
        fixes_applied = True

    # 2. Fix each dylib in the lib_dir
    for dylib in lib_dir.glob("*.dylib*"):
        if dylib.is_file():
            # Fix its references to other dylibs
            if fix_binary_dylib_refs(dylib):
                fixes_applied = True
            # Set its install ID to absolute path
            try:
                subprocess.run(
                    ["install_name_tool", "-id", str(dylib.resolve()), str(dylib)],
                    capture_output=True, timeout=10,
                )
                fixes_applied = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    # 3. Add rpath as a fallback (belt and suspenders)
    try:
        # Check existing rpaths first to avoid duplicates
        r = subprocess.run(
            ["otool", "-l", str(binary)],
            capture_output=True, text=True, timeout=10,
        )
        existing_rpaths = []
        if r.returncode == 0:
            lines = r.stdout.split("\n")
            for i, line in enumerate(lines):
                if "LC_RPATH" in line:
                    # Next line(s) contain the path
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "path" in lines[j]:
                            p = lines[j].split("path")[1].split("(")[0].strip()
                            existing_rpaths.append(p)
                            break

        if lib_dir_abs not in existing_rpaths:
            subprocess.run(
                ["install_name_tool", "-add_rpath", lib_dir_abs, str(binary)],
                capture_output=True, timeout=10,
            )
            fixes_applied = True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return fixes_applied


def find_llama_server_binary(prefix: Path | None = None) -> Path | None:
    """Find the llama-server binary, checking PATH and common install locations.

    Returns the path to the binary (wrapper script or direct binary), or None.
    """
    # Check PATH first
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        return Path(found)

    # Check common install locations
    if prefix is None:
        prefix = Path(os.environ.get("LITMOE_PREFIX", Path.home() / ".local"))

    candidates = [
        prefix / "bin" / "llama-server",
        prefix / "lib" / "llama.cpp" / "prebuilt" / "llama-server",
        prefix / "lib" / "llama.cpp" / "local" / "llama-server",
    ]

    # Also search recursively in prebuilt/local dirs
    prebuilt_dir = prefix / "lib" / "llama.cpp" / "prebuilt"
    if prebuilt_dir.exists():
        for p in prebuilt_dir.rglob("llama-server"):
            if p.is_file():
                candidates.insert(0, p)

    local_dir = prefix / "lib" / "llama.cpp" / "local"
    if local_dir.exists():
        for p in local_dir.rglob("llama-server"):
            if p.is_file():
                candidates.insert(0, p)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None
