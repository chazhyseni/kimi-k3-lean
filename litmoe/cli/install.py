"""litmoe install - one-command engine + model installation.

Installs an inference engine (llama.cpp prebuilt binary or ktransformers pip package)
and downloads model weights, then writes the model entry into models.yaml.
"""
from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import click
import yaml

from litmoe.config import default_config_path

# ---------------------------------------------------------------------------
# Known models: friendly name -> (hf_repo, default_quant, engine)
# Quant availability verified against HuggingFace API, 2026-08.
# ---------------------------------------------------------------------------
KNOWN_MODELS = {
    "kimi-k3": {
        "hf_repo": "unsloth/Kimi-K3-GGUF",
        "engine": "llamacpp",
        "quants": ["UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-Q1_0", "UD-Q2_K_XL",
                   "UD-Q4_K_XL", "UD-Q8_K_XL", "UD-TQ1_0", "UD-TQ2_0"],
        "default_quant": "UD-IQ1_S",
        "size_gb": {"UD-IQ1_S": 594, "UD-IQ1_M": 649, "UD-Q2_K_XL": 861, "UD-Q4_K_XL": 1509},
    },
    "qwen3.8": {
        "hf_repo": "unsloth/Qwen3.8-2.4T-A95B-GGUF",
        "engine": "llamacpp",
        "quants": ["UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XS", "UD-IQ2_XXS", "UD-IQ3_XXS",
                   "UD-IQ4_XS", "UD-Q1_0", "Q8_0", "BF16"],
        "default_quant": "UD-IQ1_S",
        "size_gb": {"UD-IQ1_S": 508, "UD-IQ1_M": 564, "UD-Q1_0": 397, "UD-IQ2_XXS": 657},
    },
    "minimax-m3": {
        "hf_repo": "unsloth/MiniMax-M3-GGUF",
        "engine": "llamacpp",
        "quants": ["UD-IQ1_M", "UD-IQ2_M", "UD-IQ2_XXS", "UD-IQ3_S", "UD-IQ3_XXS",
                   "UD-IQ4_NL", "UD-IQ4_XS", "UD-Q2_K_XL", "UD-Q3_K_M", "UD-Q3_K_XL",
                   "UD-Q4_K_M", "UD-Q4_K_S", "UD-Q4_K_XL", "UD-Q5_K_M", "UD-Q5_K_S",
                   "UD-Q5_K_XL", "UD-Q6_K", "UD-Q6_K_XL", "UD-Q8_K_XL", "Q8_0", "BF16"],
        "default_quant": "UD-IQ1_M",
        "size_gb": {"UD-IQ1_M": 128, "UD-IQ2_M": 134, "UD-Q2_K_XL": 143, "UD-Q4_K_M": 264, "Q8_0": 453, "BF16": 852},
    },
}

LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _default_models_dir() -> Path:
    return Path(os.environ.get("LITMOE_MODELS_DIR", Path.home() / ".litmoe" / "models"))


def _default_prefix() -> Path:
    return Path(os.environ.get("LITMOE_PREFIX", Path.home() / ".local"))


# ---------------------------------------------------------------------------
# Engine installers
# ---------------------------------------------------------------------------

def install_llamacpp(prefix: Path) -> Path:
    """Download prebuilt llama.cpp binaries from GitHub releases.

    Returns the directory containing llama-server.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        asset_substr = "bin-ubuntu-x64.tar.gz"
    elif system == "linux" and machine in ("aarch64", "arm64"):
        asset_substr = "bin-ubuntu-arm64.tar.gz"
    elif system == "darwin" and machine == "arm64":
        asset_substr = "bin-macos-arm64.tar.gz"
    elif system == "darwin" and machine in ("x86_64", "amd64"):
        asset_substr = "bin-macos-x64.tar.gz"
    else:
        raise RuntimeError(
            f"No prebuilt llama.cpp for {system}/{machine}. "
            "Build from source: https://github.com/ggml-org/llama.cpp"
        )

    click.echo("  Resolving latest llama.cpp release...")
    import httpx
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        r = client.get(LLAMA_RELEASES_API)
        r.raise_for_status()
        data = r.json()
        tag = data["tag_name"]
        asset = next(a for a in data["assets"] if asset_substr in a["name"])
        url = asset["browser_download_url"]
        size_mb = asset["size"] / 1e6
        click.echo(f"  Downloading {asset['name']} ({size_mb:.0f} MB)...")

        dest_dir = prefix / "lib" / "llama.cpp"
        dest_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    tmp.write(chunk)

    click.echo(f"  Extracting to {dest_dir}...")
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(dest_dir)

    tmp_path.unlink()

    # Find the llama-server binary (could be in a nested directory or at top level)
    server = None
    for candidate in dest_dir.rglob("llama-server"):
        if candidate.is_file():
            server = candidate
            break
    if not server:
        raise RuntimeError(
            f"llama-server binary not found in extracted archive at {dest_dir}"
        )
    server.chmod(server.stat().st_mode | stat.S_IEXEC)

    # Symlink into prefix/bin for PATH discovery
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "llama-server"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(server)

    # llama-server needs its shared libs at runtime
    click.echo(f"  llama-server installed: {server}")
    click.echo(f"  Symlinked: {link}")
    click.echo(f"  NOTE: shared libraries live in {dest_dir}; llama-server finds them via rpath.")
    return dest_dir


def install_ktransformers() -> None:
    """Install ktransformers.

    Clones the repo and pip-installs from source. This bypasses the prebuilt
    wheel glibc requirement (manylinux_2_35) by building locally.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        click.echo(f"  Cloning ktransformers to {tmpdir}...")
        clone = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/kvcache-ai/ktransformers.git", tmpdir],
            capture_output=True, text=True, timeout=180,
        )
        if clone.returncode != 0:
            click.echo(f"  git clone failed: {clone.stderr.strip()[:200]}", err=True)
            click.echo("  Manual: https://github.com/kvcache-ai/ktransformers", err=True)
            sys.exit(1)

        # Initialize submodules (kt-kernel needs them for C++/CUDA code)
        click.echo("  Initializing submodules...")
        sub = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
            cwd=tmpdir, capture_output=True, text=True, timeout=180,
        )
        if sub.returncode != 0:
            click.echo(f"  submodule init warning: {sub.stderr.strip()[:200]}", err=True)

        # Install kt-kernel from local source (builds C++/CUDA via cmake+pybind11)
        click.echo("  Building kt-kernel from source (pip install ./kt-kernel)...")
        click.echo("  This compiles C++/CUDA kernels and may take several minutes.")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"{tmpdir}/kt-kernel"],
            timeout=600,
        )
        if result.returncode != 0:
            click.echo("  kt-kernel build failed. See output above.", err=True)
            click.echo("  Manual: https://github.com/kvcache-ai/ktransformers", err=True)
            sys.exit(1)

        # Install the ktransformers wrapper package
        click.echo("  Installing ktransformers package...")
        result2 = subprocess.run(
            [sys.executable, "-m", "pip", "install", tmpdir],
            timeout=120,
        )
        if result2.returncode != 0:
            click.echo("  ktransformers package install failed. See output above.", err=True)
            click.echo("  Manual: https://github.com/kvcache-ai/ktransformers", err=True)
            sys.exit(1)

    click.echo("  ktransformers installed from source.")


# ---------------------------------------------------------------------------
# Model downloader
# ---------------------------------------------------------------------------

def download_model(model_name: str, quant: str | None, models_dir: Path) -> Path:
    """Download model weights from HuggingFace. Returns the local directory."""
    info = KNOWN_MODELS[model_name]
    quant = quant or info["default_quant"]

    if quant not in info["quants"]:
        raise click.BadParameter(
            f"{model_name} quant must be one of: {', '.join(info['quants'])}"
        )

    size_note = info["size_gb"].get(quant)
    if size_note:
        click.echo(f"  NOTE: {model_name} {quant} is ~{size_note} GB. Ensure you have the disk space.")

    repo = info["hf_repo"]
    dest = models_dir / model_name / quant
    dest.mkdir(parents=True, exist_ok=True)

    click.echo(f"  Downloading {repo} [{quant}] -> {dest}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        click.echo("  Installing huggingface_hub...", err=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]"],
                       check=True)
        from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        allow_patterns=f"{quant}/*",
        local_dir=str(dest.parent),
    )
    # snapshot_download nests under <quant>/ ; normalize
    nested = dest.parent / quant
    if nested != dest and nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(dest))
        nested.rmdir()

    n_files = len(list(dest.glob("*.gguf")))
    click.echo(f"  {n_files} GGUF shards downloaded.")
    return dest


# ---------------------------------------------------------------------------
# models.yaml writer
# ---------------------------------------------------------------------------

def add_model_to_config(model_name: str, engine: str, model_path: Path,
                        n_ctx: int, config_path: Path) -> None:
    """Insert or replace a model entry in models.yaml."""
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {"host": "127.0.0.1", "port": 8080, "api_key": None, "models": []}

    cfg.setdefault("host", "127.0.0.1")
    cfg.setdefault("port", 8080)
    cfg.setdefault("api_key", None)
    models = cfg.setdefault("models", [])

    entry = {
        "id": model_name,
        "engine": engine,
        "model_path": str(model_path),
        "n_gpu_layers": -1,
        "n_ctx": n_ctx,
    }

    models[:] = [m for m in models if m.get("id") != model_name]
    models.append(entry)

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    click.echo(f"  models.yaml updated: {model_name} -> {engine} @ {model_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("install")
@click.argument("targets", nargs=-1)
@click.option("--model", "model_name", type=click.Choice(sorted(KNOWN_MODELS.keys())),
              default=None, help="Model to download")
@click.option("--quant", default=None, help="Quantization level (default: UD-IQ1_S)")
@click.option("--engine", type=click.Choice(["llamacpp", "ktransformers", "both", "none"]),
              default=None, help="Which engine(s) to install")
@click.option("--models-dir", type=click.Path(), default=None,
              help="Where to store model weights (default: ~/.litmoe/models)")
@click.option("--prefix", type=click.Path(), default=None,
              help="Install prefix for engine binaries (default: ~/.local)")
@click.option("--n-ctx", default=4096, type=int, help="Context size written to models.yaml")
@click.option("--config", "-c", type=click.Path(), default=None, help="Path to models.yaml")
@click.option("--yes", is_flag=True, help="Skip confirmation prompts")
def install_cmd(targets, model_name, quant, engine, models_dir, prefix, n_ctx, config, yes):
    """Install an engine and/or download model weights in one command.

    \b
    Examples:
      litmoe install                          # engine + pick a model interactively
      litmoe install --model minimax-m3       # llama.cpp + MiniMax-M3 (UD-IQ1_M, ~128 GB)
      litmoe install --model kimi-k3          # llama.cpp + Kimi-K3 (UD-IQ1_S, ~594 GB)
      litmoe install --model qwen3.8          # llama.cpp + Qwen3.8-2.4T (~508 GB)
      litmoe install --engine ktransformers   # ktransformers only (pip install kt-kernel)
      litmoe install --model kimi-k3 --quant UD-Q2_K_XL
    """
    models_dir = Path(models_dir) if models_dir else _default_models_dir()
    prefix = Path(prefix) if prefix else _default_prefix()
    config_path = Path(config) if config else default_config_path()

    # Positional targets act as shortcuts: "llamacpp", "ktransformers", "both", or a model name
    for t in targets:
        if t in ("llamacpp", "ktransformers", "both"):
            engine = t if engine is None else engine
        elif t in KNOWN_MODELS:
            model_name = t if model_name is None else model_name
        else:
            raise click.BadParameter(f"unknown target: {t}")

    # Default: engine matching the model, or both if nothing specified
    if engine is None:
        engine = "llamacpp" if model_name else "both"

    # ------------------------------------------------------------------
    # 1. Engine install
    # ------------------------------------------------------------------
    if engine in ("llamacpp", "both"):
        click.echo("Installing llama.cpp (prebuilt binaries)...")
        try:
            install_llamacpp(prefix)
        except Exception as e:
            click.echo(f"  llama.cpp install failed: {e}", err=True)
            sys.exit(1)

    if engine in ("ktransformers", "both"):
        click.echo("Installing ktransformers...")
        try:
            install_ktransformers()
        except Exception as e:
            click.echo(f"  ktransformers install failed: {e}", err=True)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Model download
    # ------------------------------------------------------------------
    if model_name:
        info = KNOWN_MODELS[model_name]
        quant_val = quant or info["default_quant"]
        size_note = info["size_gb"].get(quant_val)
        if size_note and not yes:
            click.echo(f"  WARNING: {model_name} {quant_val} is ~{size_note} GB.")
            click.confirm("Proceed with download?", abort=True)
        click.echo(f"Installing model: {model_name}")
        dest = download_model(model_name, quant, models_dir)
        engine_for_model = KNOWN_MODELS[model_name]["engine"]
        add_model_to_config(model_name, engine_for_model, dest, n_ctx, config_path)
        click.echo()
        click.echo("Done. Next:")
        click.echo(f"  litmoe serve")
    else:
        click.echo()
        click.echo("Engine installed. To add a model:")
        click.echo("  litmoe install --model kimi-k3")
        click.echo("  litmoe install --model qwen3.8")
