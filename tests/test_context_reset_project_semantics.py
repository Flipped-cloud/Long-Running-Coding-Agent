from pathlib import Path

from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import completion, config, submit_plan


def test_context_reset_stays_inside_same_project_task_session(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.agent.max_steps = 4
    cfg.context.mode = "structured_reset"
    cfg.context.model_context_limit = 1800
    cfg.context.reserve_output_tokens = 100
    cfg.context.safety_margin_tokens = 0
    cfg.context.trigger_ratio = 0.55
    cfg.context.hard_stop_ratio = 0.95
    cfg.context.recent_full_turns = 1
    cfg.context.structured_handoff.use_model = False
    (cfg.workspace.root / "big.txt").write_text("x" * 8000, encoding="utf-8")
    responses = [
        submit_plan(),
        ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "big.txt"})]),
        ModelResponse(tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "done.txt", "content": "ok"})]),
        completion("c1"),
    ]

    outcome = ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="ctx-reset").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("ctx-reset")
    session = store.read_sessions("ctx-reset")[0]
    task = state.task_by_id("ctx-reset:T1")
    assert outcome.sessions_run == 1
    assert state.session_count == 1
    assert task.attempts == 1
    assert len(task.session_ids) == 1
    assert session["context_reset_count"] == 1
    assert session["context_segment_count"] == 2
    assert session["latest_context_handoff_id"]
    assert store.load_handoff("ctx-reset", session["latest_context_handoff_id"]).task_id == task.id
    assert {event["event_type"] for event in store.read_context_events("ctx-reset")} >= {"context_handoff_created", "context_reset"}
