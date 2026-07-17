from __future__ import annotations

from abc import ABC, abstractmethod

from longrun_agent.protocol import ModelResponse


class ModelProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], tools: list[dict], tool_choice: dict | str | None = None) -> ModelResponse:
        raise NotImplementedError
