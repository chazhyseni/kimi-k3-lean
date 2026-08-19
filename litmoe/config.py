"""Configuration loading and validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelEntry(BaseModel):
    """A model exposed via the OpenAI API."""
    id: str  # OpenAI model id (e.g. "kimi-k3")
    engine: Literal["ktransformers", "llamacpp"]
    model_path: str
    gguf_path: str | None = None
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    extra_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    """Top-level litmoe configuration."""
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str | None = None
    models: list[ModelEntry] = Field(default_factory=list)


def default_config_path() -> Path:
    """Path to the default config file."""
    return Path(os.environ.get("LITMOE_CONFIG", "models.yaml"))


def load_config(path: Path | str | None = None) -> GatewayConfig:
    """Load and validate litmoe config."""
    p = Path(path) if path else default_config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"litmoe config not found: {p}. "
            f"Create one with 'litmoe init' or copy examples/models.yaml."
        )
    with open(p) as f:
        data = yaml.safe_load(f)
    return GatewayConfig(**data)
