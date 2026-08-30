from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class InferenceBackend(ABC):
    @abstractmethod
    def start(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def health(self) -> bool: ...

    @abstractmethod
    def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]: ...
