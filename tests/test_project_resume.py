import json
from pathlib import Path

from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.schema import ProjectStatus, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import completion, config, submit_plan


def test_project_resume_continues_without_regenerating_plan_or_repeating_completed_tasks(tmp_path: Path):
    first_cfg = config(tmp_path, mode="static", max_sessions=1)
    first_responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[
                ToolCall(id="p1", name="report_progress", arguments={"summary": "finished first task", "files_touched": ["first.py"]}),
                ToolCall(
                    id="c1",
                    name="request_task_completion",
                    arguments={"summary": "T1 complete", "acceptance_criteria_addressed": ["done"]},
                ),
            ]
        ),
    ]
    first = ProjectOrchestrator(first_cfg, FakeModelProvider(first_responses), project_id="resume-1").start("ship project")
    assert first.status == ProjectStatus.SESSION_LIMIT_REACHED.value

    store = ProjectStateStore(first_cfg.state.root, workspace_root=first_cfg.workspace.root)
    state = store.load("resume-1")
    state.task_by_id("resume-1:T1").blocker = "historical blocker note"
    store.save(state)
    assert state.session_count == 1
    assert state.task_by_id("resume-1:T1").status == TaskStatus.CANDIDATE_COMPLETE
    assert state.task_by_id("resume-1:T2").status == TaskStatus.PENDING

    second_cfg = config(tmp_path, mode="static", max_sessions=3)
    second = ProjectOrchestrator(
        second_cfg,
        FakeModelProvider([completion("c2")]),
    ).resume("resume-1")

    resumed = store.load("resume-1")
    sessions = store.read_sessions("resume-1")
    events = store.read_events("resume-1")
    assert second.status == ProjectStatus.CANDIDATE_COMPLETE.value
    assert len(resumed.revisions) == 1
    assert sum(1 for event in events if event["event_type"] == "initial_plan_generated") == 1
    assert any(event["event_type"] == "project_resumed" for event in events)
    assert [session["session_id"] for session in sessions] == ["resume-1-s1", "resume-1-s2"]
    assert len({session["session_id"] for session in sessions}) == 2
    assert resumed.task_by_id("resume-1:T1").session_ids == ["resume-1-s1"]
    assert resumed.task_by_id("resume-1:T1").attempts == 1
    assert resumed.task_by_id("resume-1:T2").attempts == 1
    assert resumed.task_by_id("resume-1:T1").progress_notes == ["finished first task"]
    assert resumed.task_by_id("resume-1:T1").files_touched == ["first.py"]
    assert resumed.task_by_id("resume-1:T1").blocker == "historical blocker note"
    assert resumed.task_by_id("resume-1:T1").completion_summary == "T1 complete"
    assert all(task.status == TaskStatus.CANDIDATE_COMPLETE for task in resumed.tasks)
    metrics = json.loads(store.metrics_path("resume-1").read_text(encoding="utf-8"))
    assert metrics["project_sessions"] == 2
