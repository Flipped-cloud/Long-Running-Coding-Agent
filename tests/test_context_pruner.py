import json
from pathlib import Path

from longrun_agent.config import AppConfig, ModelConfig, WorkspaceConfig
from longrun_agent.context.budget import ContextBudgetManager
from longrun_agent.context.buffer import ContextBuffer
from longrun_agent.context.pruner import DeterministicContextPruner
from longrun_agent.context.stale_tracker import StaleTracker
from longrun_agent.context.token_counter import ApproximateTokenCounter
from longrun_agent.protocol import ToolResult


def _assistant(step: int, tool: str, call_id: str):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool, "arguments": "{}"}}],
    }


def _tool(call_id: str, tool: str, metadata: dict, output: str = "x" * 2000):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool,
        "content": ToolResult(
            tool_call_id=call_id,
            tool_name=tool,
            success=True,
            summary=f"{tool} ok",
            output=output,
            metadata=metadata,
        ).model_dump_json(),
    }


def _turn(buffer: ContextBuffer, step: int, tool: str, call_id: str, metadata: dict):
    buffer.add_assistant_tool_turn(_assistant(step, tool, call_id), step=step)
    buffer.add_tool_result(_tool(call_id, tool, metadata))
    buffer.finalize_turn()


def test_write_marks_old_read_and_old_test_stale(tmp_path: Path):
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    _turn(buffer, 1, "read_file", "r1", {"path": "a.py", "content_sha256": "old"})
    _turn(buffer, 2, "bash", "b1", {"normalized_command": "python -m pytest -q", "exit_code": 0, "verification_kind": "pytest"})
    _turn(buffer, 3, "write_file", "w1", {"path": "a.py", "status": "updated", "after_sha256": "new"})

    tracker = StaleTracker()
    tracker.refresh(buffer)

    assert {turn.step for turn in buffer.all_turns() if turn.status.value == "stale"} == {1, 2}


def test_repeated_bash_superseded_only_without_write_epoch(tmp_path: Path):
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    _turn(buffer, 1, "bash", "b1", {"normalized_command": "python -m pytest -q", "exit_code": 1, "code_epoch": 0})
    _turn(buffer, 2, "bash", "b2", {"normalized_command": "python -m pytest -q", "exit_code": 0, "code_epoch": 0})
    _turn(buffer, 3, "write_file", "w1", {"path": "a.py", "status": "updated"})
    _turn(buffer, 4, "bash", "b3", {"normalized_command": "python -m pytest -q", "exit_code": 0, "code_epoch": 1})
    tracker = StaleTracker()
    tracker.refresh(buffer)

    assert {turn.step for turn in buffer.all_turns() if turn.status.value == "superseded"} == {1}


def test_deterministic_pruner_compacts_without_orphan_tool_messages(tmp_path: Path):
    cfg = AppConfig(model=ModelConfig(provider="fake", model_name="fake"), workspace=WorkspaceConfig(root=tmp_path))
    cfg.context.recent_full_turns = 1
    budget = ContextBudgetManager(cfg.context, ApproximateTokenCounter(cfg.context.token_counter))
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    _turn(buffer, 1, "read_file", "r1", {"path": "a.py", "content_sha256": "old"})
    _turn(buffer, 2, "write_file", "w1", {"path": "a.py", "status": "updated", "after_sha256": "new"})
    StaleTracker().refresh(buffer)
    before = budget.measure(buffer.export_messages(), []).estimated_total_tokens

    outcome = DeterministicContextPruner(cfg.context, budget).prune(buffer, [], before_tokens=before)
    tool_messages = [message for message in outcome.messages if message["role"] == "tool"]
    assistant_calls = [
        call["id"] for message in outcome.messages if message["role"] == "assistant" for call in message.get("tool_calls", [])
    ]

    assert outcome.report.tokens_removed > 0
    assert sorted(message["tool_call_id"] for message in tool_messages) == sorted(assistant_calls)
    assert any(
        "context compacted" in json.loads(message["content"])["output"] or "Status: stale" in json.loads(message["content"])["output"]
        for message in tool_messages
    )
