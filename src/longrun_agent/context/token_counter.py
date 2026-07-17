from __future__ import annotations

import json
import math
from typing import Any, Protocol

from longrun_agent.config import ContextTokenCounterConfig


class TokenCounter(Protocol):
    estimator_type: str

    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int: ...

    def count_tools(self, tools: list[dict[str, Any]]) -> int: ...


class ApproximateTokenCounter:
    estimator_type = "approximate"

    def __init__(self, config: ContextTokenCounterConfig):
        self.config = config

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.config.chars_per_token))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            total += self.config.per_message_overhead
            total += self.count_text(json.dumps(message, ensure_ascii=False, sort_keys=True))
        return total

    def count_tools(self, tools: list[dict[str, Any]]) -> int:
        total = 0
        for tool in tools:
            total += self.config.per_tool_overhead
            total += self.count_text(json.dumps(tool, ensure_ascii=False, sort_keys=True))
        return total


class TiktokenTokenCounter(ApproximateTokenCounter):
    estimator_type = "tiktoken"

    def __init__(self, config: ContextTokenCounterConfig):
        super().__init__(config)
        import tiktoken

        self._encoding = tiktoken.get_encoding(config.encoding)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(self._encoding.encode(text)))


def build_token_counter(config: ContextTokenCounterConfig) -> tuple[TokenCounter, bool]:
    if config.type != "tiktoken":
        return ApproximateTokenCounter(config), False
    try:
        return TiktokenTokenCounter(config), False
    except (ImportError, LookupError, ValueError):
        return ApproximateTokenCounter(config), True
