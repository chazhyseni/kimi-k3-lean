"""Base interface for engine adapters."""
from __future__ import annotations

import abc
import asyncio
import os
import signal
import subprocess
from pathlib import Path

from litmoe.config import ModelEntry


class Engine(abc.ABC):
    """Abstract base for engine adapters (ktransformers, llama.cpp, etc.).

    Engines are subprocesses that speak OpenAI-compatible HTTP. The gateway
    just forwards requests to them. This is the simplest possible architecture.
    """

    def __init__(self, model: ModelEntry):
        self.model = model
        self.process: subprocess.Popen | None = None
        self.base_url: str | None = None
        self._log_path: Path | None = None

    @abc.abstractmethod
    def build_command(self) -> list[str]:
        """Build the command to start this engine."""
        ...

    @abc.abstractmethod
    def health_url(self) -> str:
        """URL to poll for readiness."""
        ...

    @abc.abstractmethod
    def default_port(self) -> int:
        """Default port for this engine."""
        ...

    def start(self, log_dir: Path | None = None) -> None:
        """Start the engine as a subprocess."""
        cmd = self.build_command()
        env = os.environ.copy()
        env.update(self.model.env)

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

    def stop(self) -> None:
        """Stop the engine."""
        if self.process and self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=5)

    async def wait_ready(self, timeout: float = 300.0) -> bool:
        """Poll health endpoint until ready or timeout."""
        import httpx
        url = self.health_url()
        async with httpx.AsyncClient() as client:
            start = asyncio.get_event_loop().time()
            while True:
                if self.process and self.process.poll() is not None:
                    print(f"  {self.model.id}: process exited with code {self.process.returncode}")
                    return False
                try:
                    r = await client.get(url, timeout=5.0)
                    if r.status_code == 200:
                        return True
                except (httpx.RequestError, httpx.TimeoutException):
                    pass
                if asyncio.get_event_loop().time() - start > timeout:
                    print(f"  {self.model.id}: timeout waiting for {url}")
                    return False
                await asyncio.sleep(2.0)
