from pathlib import Path

from longrun_agent.config import AppConfig, ModelConfig, WorkspaceConfig
from longrun_agent.context.buffer import ContextBuffer
from longrun_agent.context.compactor import StructuredContextCompactor
from longrun_agent.context.handoff import deterministic_handoff
from longrun_agent.context.schema import EvidenceReference, HandoffRecord, TaskContextSeed
from longrun_agent.context.token_counter import build_token_counter
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.protocol import ModelResponse, ToolCall, ToolResult


def seed() -> TaskContextSeed:
    return TaskContextSeed(
        project_id="p1",
        project_objective="ship",
        plan_version=1,
        task_id="t1",
        task_title="T1",
        task_objective="do it",
        task_status="in_progress",
        acceptance_criteria=["done"],
        attempts=1,
    )


def buffer_with_write() -> ContextBuffer:
    buffer = ContextBuffer.from_task(system_prompt="system", task="task")
    buffer.add_assistant_tool_turn(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "w1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}],
        },
        step=1,
    )
    buffer.add_tool_result(
        {
            "role": "tool",
            "tool_call_id": "w1",
            "name": "write_file",
            "content": ToolResult(
                tool_call_id="w1",
                tool_name="write_file",
                success=True,
                summary="write ok",
                metadata={"path": "a.py", "status": "updated", "after_sha256": "abc"},
            ).model_dump_json(),
        }
    )
    buffer.finalize_turn()
    return buffer


def test_deterministic_handoff_preserves_file_evidence(tmp_path: Path):
    record = deterministic_handoff(
        seed=seed(),
        buffer=buffer_with_write(),
        project_id="p1",
        session_id="s1",
        source_segment_id=1,
        target_segment_id=2,
        plan_version=1,
        workspace_root=tmp_path,
    )

    assert record.files_current[0].path == "a.py"
    assert record.files_current[0].sha256 == "abc"
    assert record.evidence_references


def test_model_compactor_accepts_native_tool_handoff(tmp_path: Path):
    record = HandoffRecord(
        handoff_id="h-model",
        project_id="p1",
        task_id="t1",
        session_id="s1",
        source_segment_id=1,
        target_segment_id=2,
        plan_version=1,
        task_objective="do it",
        evidence_references=[EvidenceReference(evidence_id="e1", type="tool", step=1)],
    )
    cfg = AppConfig(model=ModelConfig(provider="fake", model_name="fake"), workspace=WorkspaceConfig(root=tmp_path))
    compactor = StructuredContextCompactor(
        cfg.context,
        FakeModelProvider(
            [ModelResponse(tool_calls=[ToolCall(id="h", name="submit_context_handoff", arguments=record.model_dump(mode="json"))])]
        ),
    )

    compacted = compactor.compact(
        seed=seed(),
        buffer=buffer_with_write(),
        project_id="p1",
        session_id="s1",
        source_segment_id=1,
        target_segment_id=2,
        plan_version=1,
    )

    assert compacted.handoff_id == "h-model"
    assert compactor.fallback_count == 0


def test_model_compactor_protocol_failure_falls_back(tmp_path: Path):
    cfg = AppConfig(model=ModelConfig(provider="fake", model_name="fake"), workspace=WorkspaceConfig(root=tmp_path))
    cfg.context.structured_handoff.max_protocol_retries = 1
    compactor = StructuredContextCompactor(cfg.context, FakeModelProvider([ModelResponse()]))

    compacted = compactor.compact(
        seed=seed(),
        buffer=buffer_with_write(),
        project_id="p1",
        session_id="s1",
        source_segment_id=1,
        target_segment_id=2,
        plan_version=1,
    )

    assert compacted.generator == "deterministic"
    assert compactor.protocol_error_count == 1
    assert compactor.fallback_count == 1


def test_tiktoken_counter_falls_back_when_unavailable_encoding(tmp_path: Path):
    cfg = AppConfig(model=ModelConfig(provider="fake", model_name="fake"), workspace=WorkspaceConfig(root=tmp_path))
    cfg.context.token_counter.type = "tiktoken"
    cfg.context.token_counter.encoding = "missing-encoding"

    counter, fallback = build_token_counter(cfg.context.token_counter)

    assert fallback is True
    assert counter.estimator_type == "approximate"
