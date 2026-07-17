from __future__ import annotations

from typing import Any

from longrun_agent.config import ContextConfig
from longrun_agent.context.schema import ContextBudgetSnapshot
from longrun_agent.context.token_counter import TokenCounter


class ContextBudgetManager:
    def __init__(self, config: ContextConfig, counter: TokenCounter):
        self.config = config
        self.counter = counter

    def measure(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ContextBudgetSnapshot:
        message_tokens = self.counter.count_messages(messages)
        tool_tokens = self.counter.count_tools(tools)
        total = message_tokens + tool_tokens + self.config.reserve_output_tokens + self.config.safety_margin_tokens
        ratio = total / self.config.model_context_limit
        return ContextBudgetSnapshot(
            estimated_message_tokens=message_tokens,
            estimated_tool_schema_tokens=tool_tokens,
            reserved_output_tokens=self.config.reserve_output_tokens,
            safety_margin_tokens=self.config.safety_margin_tokens,
            estimated_total_tokens=total,
            context_limit=self.config.model_context_limit,
            usage_ratio=ratio,
            trigger_exceeded=ratio >= self.config.trigger_ratio,
            hard_stop_exceeded=ratio >= self.config.hard_stop_ratio,
            estimator_type=self.counter.estimator_type,
        )

    def should_prune(self, snapshot: ContextBudgetSnapshot) -> bool:
        return snapshot.trigger_exceeded and self.config.mode in {"recent_window", "deterministic_prune", "structured_reset"}

    def should_reset(self, snapshot: ContextBudgetSnapshot) -> bool:
        return snapshot.trigger_exceeded and self.config.mode == "structured_reset"

    def should_hard_stop(self, snapshot: ContextBudgetSnapshot) -> bool:
        return snapshot.hard_stop_exceeded
