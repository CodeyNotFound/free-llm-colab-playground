from __future__ import annotations

import platform
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import requests


def generate_api_key() -> str:
    return f"colab-{secrets.token_urlsafe(24)}"


class CloudflareTunnel:
    def __init__(self, binary: str | Path = "cloudflared") -> None:
        self.binary = Path(binary)
        self.process: subprocess.Popen[str] | None = None
        self.logs: deque[str] = deque(maxlen=200)
        self.public_url: str | None = None

    def install(self, destination: str | Path = "cloudflared") -> Path:
        found = shutil.which("cloudflared")
        if found:
            self.binary = Path(found)
            return self.binary
        machine = platform.machine().lower()
        system = platform.system().lower()
        if system != "linux":
            raise RuntimeError("Automatic cloudflared installation is supported in Linux/Colab only.")
        arch = "amd64" if machine in {"x86_64", "amd64"} else "arm64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        target = Path(destination).resolve()
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with target.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
        self.binary = target
        return target

    def _capture(self) -> None:
        if not self.process or not self.process.stdout:
            return
        pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
        for line in self.process.stdout:
            self.logs.append(line.rstrip())
            match = pattern.search(line)
            if match:
                self.public_url = match.group(0)

    def start(self, port: int = 8080) -> str:
        self.stop()
        binary = self.binary if self.binary.exists() else self.install(self.binary)
        self.process = subprocess.Popen(
            [str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=self._capture, daemon=True).start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.public_url:
                return self.public_url
            if self.process.poll() is not None:
                break
            time.sleep(0.25)
        self.stop()
        raise RuntimeError("Cloudflare Quick Tunnel did not provide a public URL. Try again shortly.")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.public_url = None
