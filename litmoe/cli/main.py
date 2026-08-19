"""litmoe CLI - main entry point."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import click
import yaml

from litmoe import __version__
from litmoe.config import GatewayConfig, load_config, default_config_path
from litmoe.engines import kt_installed, llama_installed
from litmoe.cli.install import install_cmd


@click.group()
@click.version_option(version=__version__, prog_name="litmoe")
def cli():
    """litmoe - OpenAI-compatible gateway for ktransformers and llama.cpp"""
    pass


@cli.command()
def doctor():
    """Check hardware compatibility and engine availability."""
    click.echo(f"litmoe v{__version__}")
    click.echo()

    # CPU detection
    click.echo("=== CPU ===")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    click.echo(f"  {line.split(':')[1].strip()}")
                    break
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "flags" in line:
                    flags = line.split(":")[1].strip().split()
                    interesting = ["sse4_2", "avx", "avx2", "avx512f", "avx512_bf16",
                                   "avx512_vnni", "amx_tile", "amx_bf16", "amx_int8"]
                    supported = [flag for flag in interesting if flag in flags]
                    click.echo(f"  Instruction sets: {', '.join(supported)}")
                    break
    except FileNotFoundError:
        click.echo("  (not Linux)")

    # Memory
    click.echo()
    click.echo("=== Memory ===")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    click.echo(f"  Total: {kb / 1024 / 1024:.1f} GB")
                    break
    except FileNotFoundError:
        pass

    # GPU detection
    click.echo()
    click.echo("=== GPU ===")
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                click.echo(f"  {line}")
        else:
            click.echo("  nvidia-smi not working")
    else:
        click.echo("  No NVIDIA GPU detected")

    # Engine availability
    click.echo()
    click.echo("=== Engines ===")
    if kt_installed():
        click.echo("  ktransformers: installed")
    else:
        click.echo("  ktransformers: NOT installed (pip install kt-kernel)")
    if llama_installed():
        click.echo("  llama.cpp: installed")
    else:
        click.echo("  llama.cpp: NOT installed (install llama-server)")

    # Recommendation
    click.echo()
    click.echo("=== Recommended engine ===")
    if kt_installed():
        if shutil.which("nvidia-smi"):
            click.echo("  ktransformers (GPU + CUDA available)")
        else:
            click.echo("  ktransformers (CPU AMX/AVX2/AVX512)")
    elif llama_installed():
        click.echo("  llama.cpp")
    else:
        click.echo("  None - install ktransformers: pip install kt-kernel")
        click.echo("          or build llama.cpp: https://github.com/ggml-org/llama.cpp")


@cli.command()
def init():
    """Create a default models.yaml config file."""
    cfg_path = Path("models.yaml")
    if cfg_path.exists():
        click.echo(f"{cfg_path} already exists. Refusing to overwrite.")
        sys.exit(1)

    # Detect engines
    # For Kimi-K3 and Qwen3.8, llama.cpp is the engine (ktransformers doesn't support them yet)
    # For DeepSeek, MiniMax, GLM, etc., ktransformers is the engine
    if llama_installed():
        engine = "llamacpp"
    elif kt_installed():
        engine = "ktransformers"
    else:
        engine = "llamacpp"

    default_models = {
        "llamacpp": [
            {
                "id": "kimi-k3",
                "engine": "llamacpp",
                "model_path": "unsloth/Kimi-K3-GGUF:UD-IQ1_S",
                "n_gpu_layers": -1,
                "n_ctx": 4096,
            },
            {
                "id": "qwen3.8-2.4t",
                "engine": "llamacpp",
                "model_path": "unsloth/Qwen3.8-2.4T-A95B-GGUF:UD-IQ1_S",
                "n_gpu_layers": -1,
                "n_ctx": 4096,
            },
        ],
        "ktransformers": [
            {
                "id": "deepseek-v3",
                "engine": "ktransformers",
                "model_path": "/data/deepseek-v3",
                "n_gpu_layers": -1,
                "n_ctx": 4096,
            },
        ],
    }

    cfg = {
        "host": "127.0.0.1",
        "port": 8080,
        "api_key": None,
        "models": default_models.get(engine, default_models["ktransformers"]),
    }

    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    click.echo(f"Created {cfg_path}")
    click.echo("Edit model paths then run: litmoe serve")


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Path to models.yaml")
@click.option("--log-dir", default="logs", help="Directory for engine logs")
def serve(config, log_dir):
    """Start the gateway with all configured engines."""
    cfg_path = config or str(default_config_path())
    if not Path(cfg_path).exists():
        click.echo(f"Error: config not found: {cfg_path}", err=True)
        click.echo("Run 'litmoe init' to create one.", err=True)
        sys.exit(1)

    cfg = load_config(cfg_path)
    click.echo(f"litmoe v{__version__} starting gateway on {cfg.host}:{cfg.port}")
    model_ids = [m.id for m in cfg.models]
    click.echo(f"Models: {model_ids}")
    click.echo(f"Log dir: {log_dir}")
    click.echo()

    from litmoe.server import run as server_run
    server_run(cfg, log_dir=log_dir)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
def status(config):
    """Show running gateway and engines."""
    import httpx

    cfg_path = config or str(default_config_path())
    click.echo(f"Config: {cfg_path}")
    if not Path(cfg_path).exists():
        click.echo("  (no config)")
        return

    cfg = load_config(cfg_path)
    click.echo(f"Configured models: {[m.id for m in cfg.models]}")

    # Poll gateway health
    try:
        r = httpx.get(f"http://{cfg.host}:{cfg.port}/health", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            click.echo()
            click.echo(f"Gateway health: {data['status']}")
            for model_id, info in data.get("engines", {}).items():
                click.echo(f"  {model_id}: running={info['running']}, port={info['port']}")
    except Exception as e:
        click.echo()
        click.echo(f"Gateway not reachable: {e}")


@cli.command()
def stop():
    """Stop all engines (sends SIGTERM to running processes)."""
    patterns = ["kt run", "llama-server", "ktransformers.server.main"]
    click.echo(f"Looking for engine processes matching: {patterns}")
    found = False
    for pattern in patterns:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if result.stdout:
            found = True
            for pid in result.stdout.strip().split("\n"):
                click.echo(f"  Sending SIGTERM to {pid}")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
    if not found:
        click.echo("  No engine processes found")


cli.add_command(install_cmd)


def main():
    cli()


if __name__ == "__main__":
    main()
