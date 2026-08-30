from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from .base import InferenceBackend


class LlamaCppError(RuntimeError):
    """Friendly wrapper for llama-server lifecycle and API failures."""


class LlamaCppBackend(InferenceBackend):
    def __init__(self, binary: str | Path = "llama-server", port: int = 8080) -> None:
        self.binary = str(binary)
        self.port = port
        self.process: subprocess.Popen[str] | None = None
        self.api_key = ""
        self.model_alias = "local-model"
        self.logs: deque[str] = deque(maxlen=1000)
        self.started_at: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _resolve_binary(self) -> str:
        candidate = Path(self.binary)
        if candidate.exists():
            return str(candidate)
        found = shutil.which(self.binary)
        if found:
            return found
        raise LlamaCppError(
            "llama-server was not found. Run scripts/install_llama_cpp.sh in Colab, "
            "or set LLAMA_SERVER_PATH to the binary."
        )

    def build_command(
        self,
        *,
        model_path: str | Path,
        gpu_layers: int,
        context_size: int,
        api_key: str,
        model_alias: str,
        threads: int,
        batch_size: int = 512,
        microbatch_size: int = 128,
        flash_attention: bool = True,
        kv_cache_type: str = "f16",
    ) -> list[str]:
        command = [
            self._resolve_binary(),
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--api-key",
            api_key,
            "--alias",
            model_alias,
            "--ctx-size",
            str(context_size),
            "--n-gpu-layers",
            str(gpu_layers),
            "--threads",
            str(threads),
            "--batch-size",
            str(batch_size),
            "--ubatch-size",
            str(microbatch_size),
            "--cache-type-k",
            kv_cache_type,
            "--cache-type-v",
            kv_cache_type,
            "--metrics",
        ]
        if flash_attention:
            command.extend(["--flash-attn", "on"])
        return command

    def _capture_logs(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {line.rstrip()}")

    def start(self, **kwargs: Any) -> None:
        self.stop()
        model_path = Path(kwargs["model_path"])
        if not model_path.is_file():
            raise LlamaCppError(f"The selected GGUF file does not exist: {model_path}")
        self.api_key = kwargs["api_key"]
        self.model_alias = kwargs["model_alias"]
        command = self.build_command(**kwargs)
        safe_command = ["<API_KEY>" if item == self.api_key else item for item in command]
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] Starting: {' '.join(safe_command)}")
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ},
        )
        threading.Thread(target=self._capture_logs, daemon=True).start()
        self.started_at = time.perf_counter()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = "\n".join(list(self.logs)[-20:])
                raise LlamaCppError(self._friendly_load_error(tail))
            if self.health():
                return
            time.sleep(1)
        self.stop()
        raise LlamaCppError(
            "The model did not become ready within 3 minutes. It may still be too large for this "
            "runtime. Reduce context, choose Q4, or lower GPU layers, then try once more."
        )

    @staticmethod
    def _friendly_load_error(logs: str) -> str:
        lowered = logs.lower()
        if "out of memory" in lowered or "cuda error" in lowered:
            return (
                "The model ran out of GPU memory while loading. Reduce context, lower GPU layers, "
                "or choose a smaller quantization. Technical logs are available in Nerd Mode."
            )
        return "llama-server exited before the model became ready. Open Logs in Nerd Mode for details."

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def health(self) -> bool:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=2)
            return response.ok
        except requests.RequestException:
            return False

    def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        payload = {
            "model": self.model_alias,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        if kwargs.get("stop"):
            payload["stop"] = kwargs["stop"]
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=(10, 600),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield f"<details><summary>🧠 Thinking</summary>\n\n{reasoning}\n\n</details>\n\n"
                    if content:
                        yield content
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise LlamaCppError(f"The model request failed: {exc}") from exc

    def log_text(self, lines: int = 200) -> str:
        return "\n".join(list(self.logs)[-lines:]) or "No server logs yet."
