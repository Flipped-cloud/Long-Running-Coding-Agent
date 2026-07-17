from pathlib import Path

import pytest

from longrun_agent.config import AppConfig, ModelConfig, WorkspaceConfig
from longrun_agent.context.budget import ContextBudgetManager
from longrun_agent.context.token_counter import ApproximateTokenCounter


def test_context_config_validation(tmp_path: Path):
    with pytest.raises(ValueError):
        AppConfig.model_validate(
            {
                "model": {"provider": "fake", "model_name": "fake"},
                "workspace": {"root": str(tmp_path)},
                "context": {"model_context_limit": 100, "reserve_output_tokens": 90, "safety_margin_tokens": 10},
            }
        )


def test_token_budget_counts_tools_and_reserved_output(tmp_path: Path):
    cfg = AppConfig(
        model=ModelConfig(provider="fake", model_name="fake"),
        workspace=WorkspaceConfig(root=tmp_path),
    )
    cfg.context.model_context_limit = 200
    cfg.context.reserve_output_tokens = 20
    cfg.context.safety_margin_tokens = 10
    manager = ContextBudgetManager(cfg.context, ApproximateTokenCounter(cfg.context.token_counter))

    snapshot = manager.measure([{"role": "user", "content": "x" * 100}], [{"type": "function", "function": {"name": "tool"}}])

    assert snapshot.estimated_tool_schema_tokens > 0
    assert snapshot.reserved_output_tokens == 20
    assert snapshot.safety_margin_tokens == 10
    assert snapshot.estimated_total_tokens > snapshot.estimated_message_tokens
    assert manager.should_prune(snapshot) == snapshot.trigger_exceeded


def test_budget_hard_stop_uses_estimated_total_tokens(tmp_path: Path):
    cfg = AppConfig(
        model=ModelConfig(provider="fake", model_name="fake"),
        workspace=WorkspaceConfig(root=tmp_path),
    )
    cfg.context.model_context_limit = 4096
    cfg.context.reserve_output_tokens = 1024
    cfg.context.safety_margin_tokens = 256
    cfg.context.trigger_ratio = 0.5
    cfg.context.hard_stop_ratio = 0.8
    manager = ContextBudgetManager(cfg.context, FixedCounter(message_tokens=8313, tool_tokens=0))

    snapshot = manager.measure([{"role": "user", "content": "ignored"}], [])

    assert snapshot.estimated_total_tokens == 9593
    assert snapshot.usage_ratio > 2
    assert snapshot.hard_stop_exceeded is True


class FixedCounter:
    estimator_type = "fixed"

    def __init__(self, *, message_tokens: int, tool_tokens: int):
        self.message_tokens = message_tokens
        self.tool_tokens = tool_tokens

    def count_text(self, text: str) -> int:
        return len(text)

    def count_messages(self, messages: list[dict]) -> int:
        return self.message_tokens

    def count_tools(self, tools: list[dict]) -> int:
        return self.tool_tokens
