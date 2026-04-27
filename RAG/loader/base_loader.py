from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    SUPPORTED_TYPES = ["pdf"]

    @abstractmethod
    def load(self, file_path: str) -> str:
        raise NotImplementedError

