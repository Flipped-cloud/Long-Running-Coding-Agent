import json
from pathlib import Path

from longrun_agent.config import (
    AgentConfig,
    AppConfig,
    BashConfig,
    ModelConfig,
    PlanningConfig,
    PlanningExecutionConfig,
    StateConfig,
    TelemetryConfig,
    ToolsConfig,
    WorkspaceConfig,
)
from longrun_agent.control.channel import ControlSignal, ControlSignalType, TaskControlChannel
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.orchestration.session_trace import SessionTrace
from longrun_agent.protocol import FinalAnswer, ModelResponse, ToolCall
from longrun_agent.state.schema import ProjectState, ProjectStatus, TaskNode, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.verification.schema import TestCandidate as CandidateSchema
from longrun_agent.verification.store import VerificationStore


def config(tmp_path: Path, mode: str = "adaptive", max_sessions: int = 10) -> AppConfig:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AppConfig(
        model=ModelConfig(provider="fake", model_name="fake"),
        agent=AgentConfig(max_steps=5, max_consecutive_errors=2),
        workspace=WorkspaceConfig(root=workspace),
        tools=ToolsConfig(bash=BashConfig(timeout_seconds=5, max_output_chars=2000, shell=False)),
        telemetry=TelemetryConfig(run_root=tmp_path / "runs"),
        state=StateConfig(root=tmp_path / "projects", atomic_write=True),
        planning=PlanningConfig(
            mode=mode,
            execution=PlanningExecutionConfig(
                max_project_sessions=max_sessions,
                attempts_before_decomposition=1,
                final_verification_command=[],
            ),
        ),
    )


def submit_plan():
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id="plan",
                name="submit_plan",
                arguments={
                    "project_summary": "summary",
                    "tasks": [
                        {"key": "T1", "title": "T1", "objective": "first task", "acceptance_criteria": ["done"], "depends_on_keys": []},
                        {
                            "key": "T2",
                            "title": "T2",
                            "objective": "second task",
                            "acceptance_criteria": ["done"],
                            "depends_on_keys": ["T1"],
                        },
                    ],
                },
            )
        ]
    )


def completion(call_id: str):
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="request_task_completion",
                arguments={"summary": "done", "acceptance_criteria_addressed": ["done"]},
            )
        ]
    )


def final():
    return ModelResponse(final_answer=FinalAnswer(content="session done"))


def test_persisted_generated_test_is_inherited_by_decomposition_child(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    orchestrator = ProjectOrchestrator(cfg, FakeModelProvider([]), project_id="project")
    orchestrator.verification_store = VerificationStore(
        tmp_path / "verification",
        "project",
        workspace_root=cfg.workspace.root,
    )
    parent = TaskNode(
        id="parent",
        key="parent",
        title="parent",
        objective="fix",
        acceptance_criteria=["done"],
    )
    child = TaskNode(
        id="child",
        key="child",
        title="child",
        objective="finish fix",
        acceptance_criteria=["done"],
        parent_id=parent.id,
        depth=1,
    )
    candidate = CandidateSchema(
        task_id=parent.id,
        session_id="session-1",
        paths=["agent_tests/test_issue.py"],
        command_argv=["python", "agent_tests/test_issue.py"],
        issue_behavior="reproduce issue",
    )
    orchestrator.verification_store.save_test_candidate(candidate)
    state = ProjectState(project_id="project", objective="ship", tasks=[parent, child])

    assert orchestrator._existing_test_candidates(state, child) == [candidate]


def test_no_progress_does_not_force_decomposition_at_depth_limit(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    cfg.planning.decomposition.max_depth = 1
    orchestrator = ProjectOrchestrator(cfg, FakeModelProvider([]))
    task = TaskNode(
        id="child",
        key="child",
        title="child",
        objective="finish fix",
        acceptance_criteria=["done"],
        parent_id="parent",
        depth=1,
        consecutive_no_progress_sessions=cfg.planning.execution.max_no_progress_sessions,
    )

    assert not orchestrator._no_progress_decomposition_required(task)


def test_reported_files_are_not_progress_without_an_observed_write(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    orchestrator = ProjectOrchestrator(cfg, FakeModelProvider([]))
    channel = TaskControlChannel()
    channel.record(
        ControlSignal(
            type=ControlSignalType.PROGRESS,
            summary="planned edit",
            files_touched=["workflow_service/service.py"],
        )
    )

    assert orchestrator._substantive_progress_count(channel, SessionTrace()) == 0


def decomposition(parent_task_id: str):
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id="decompose",
                name="submit_decomposition",
                arguments={
                    "parent_task_id": parent_task_id,
                    "reason": "split stalled work",
                    "children": [
                        {
                            "key": "A",
                            "title": "A",
                            "objective": "specific child A",
                            "acceptance_criteria": ["done"],
                            "depends_on_child_keys": [],
                        },
                        {
                            "key": "B",
                            "title": "B",
                            "objective": "specific child B",
                            "acceptance_criteria": ["done"],
                            "depends_on_child_keys": ["A"],
                        },
                    ],
                },
            )
        ]
    )


def test_project_orchestrator_adaptive_decomposition_integration(tmp_path: Path):
    responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[ToolCall(id="p1", name="report_progress", arguments={"summary": "made progress", "files_touched": ["a.py"]})]
        ),
        completion("c1"),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="b1",
                    name="report_blocker",
                    arguments={"reason": "too broad", "attempted_actions": ["read"], "decomposition_recommended": True},
                )
            ]
        ),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="d1",
                    name="submit_decomposition",
                    arguments={
                        "parent_task_id": "project-1:T2",
                        "reason": "too broad",
                        "children": [
                            {
                                "key": "T2.1",
                                "title": "T2.1",
                                "objective": "specific child 1",
                                "acceptance_criteria": ["done"],
                                "depends_on_child_keys": [],
                            },
                            {
                                "key": "T2.2",
                                "title": "T2.2",
                                "objective": "specific child 2",
                                "acceptance_criteria": ["done"],
                                "depends_on_child_keys": ["T2.1"],
                            },
                        ],
                    },
                )
            ]
        ),
        completion("c21"),
        completion("c22"),
    ]
    cfg = config(tmp_path, mode="adaptive")
    outcome = ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="project-1").start("ship project")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("project-1")
    statuses = {task.key: task.status for task in state.tasks}
    assert outcome.status == ProjectStatus.CANDIDATE_COMPLETE.value
    assert statuses["T1"] == TaskStatus.CANDIDATE_COMPLETE
    assert statuses["T2"] == TaskStatus.CANDIDATE_COMPLETE
    assert statuses["T2.1"] == TaskStatus.CANDIDATE_COMPLETE
    assert statuses["T2.2"] == TaskStatus.CANDIDATE_COMPLETE
    events = [json.loads(line) for line in (cfg.state.root / "project-1" / "project_events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_types = {event["event_type"] for event in events}
    assert {
        "project_created",
        "initial_plan_generated",
        "task_progress",
        "task_blocked",
        "decomposition_generated",
        "parent_task_aggregated",
        "project_candidate_complete",
    }.issubset(event_types)


def test_consecutive_no_progress_forces_decomposition_instead_of_another_retry(tmp_path: Path):
    cfg = config(tmp_path, mode="adaptive_search", max_sessions=5)
    cfg.agent.max_steps = 1
    cfg.planning.execution.max_no_progress_sessions = 2
    responses = [
        submit_plan(),
        ModelResponse(tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "missing.py"})]),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="recovery",
                    name="submit_recovery_candidates",
                    arguments={
                        "task_id": "project-no-progress:T1",
                        "candidates": [
                            {
                                "id": "retry",
                                "kind": "retry_with_guidance",
                                "description": "try once more",
                                "rationale": "recover",
                                "expected_benefit": "progress",
                                "risks": "low",
                                "testability": "next session",
                                "child_tasks": [],
                            }
                        ],
                    },
                )
            ]
        ),
        ModelResponse(tool_calls=[ToolCall(id="r2", name="read_file", arguments={"path": "missing.py"})]),
        decomposition("project-no-progress:T1"),
        completion("a"),
        completion("b"),
        completion("t2"),
    ]

    outcome = ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="project-no-progress").start("ship")

    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("project-no-progress")
    events = store.read_events("project-no-progress")
    assert outcome.status == ProjectStatus.CANDIDATE_COMPLETE.value
    assert state.task_by_id("project-no-progress:T1").status == TaskStatus.CANDIDATE_COMPLETE
    assert sum(event["event_type"] == "recovery_candidates_generated" for event in events) == 1
    assert any(event["event_type"] == "no_progress_decomposition_forced" for event in events)


def test_project_orchestrator_final_answer_without_control_signal_does_not_complete(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    outcome = ProjectOrchestrator(cfg, FakeModelProvider([submit_plan(), final()]), project_id="project-2").start("ship project")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("project-2")
    assert outcome.status == ProjectStatus.SESSION_LIMIT_REACHED.value
    assert state.task_by_id("project-2:T1").status == TaskStatus.IN_PROGRESS
    events = (cfg.state.root / "project-2" / "project_events.jsonl").read_text(encoding="utf-8")
    assert "session_ended_without_task_signal" in events


def test_project_orchestrator_static_mode_blocks_instead_of_decomposing(tmp_path: Path):
    cfg = config(tmp_path, mode="static")
    responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="b1",
                    name="report_blocker",
                    arguments={"reason": "blocked", "attempted_actions": [], "decomposition_recommended": True},
                )
            ]
        ),
    ]
    outcome = ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="project-3").start("ship project")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("project-3")
    assert outcome.status == ProjectStatus.BLOCKED.value
    assert state.task_by_id("project-3:T1").status == TaskStatus.BLOCKED


def test_project_orchestrator_adaptive_search_applies_one_selected_candidate(tmp_path: Path):
    cfg = config(tmp_path, mode="adaptive_search", max_sessions=3)
    responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="b1",
                    name="report_blocker",
                    arguments={"reason": "blocked", "attempted_actions": [], "decomposition_recommended": True},
                )
            ]
        ),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="g1",
                    name="submit_recovery_candidates",
                    arguments={
                        "task_id": "project-4:T1",
                        "candidates": [
                            {
                                "id": "bad",
                                "kind": "decompose",
                                "description": "bad",
                                "rationale": "bad",
                                "expected_benefit": "none",
                                "risks": "high",
                                "testability": "none",
                                "child_tasks": [
                                    {"key": "X", "title": "X", "objective": "x", "acceptance_criteria": [], "depends_on_child_keys": []}
                                ],
                            },
                            {
                                "id": "good",
                                "kind": "decompose",
                                "description": "split",
                                "rationale": "recover",
                                "expected_benefit": "progress",
                                "risks": "low",
                                "testability": "checks",
                                "child_tasks": [
                                    {
                                        "key": "A",
                                        "title": "A",
                                        "objective": "specific A",
                                        "acceptance_criteria": ["done"],
                                        "depends_on_child_keys": [],
                                    },
                                    {
                                        "key": "B",
                                        "title": "B",
                                        "objective": "specific B",
                                        "acceptance_criteria": ["done"],
                                        "depends_on_child_keys": ["A"],
                                    },
                                ],
                            },
                        ],
                    },
                )
            ]
        ),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="s1",
                    name="select_recovery_candidate",
                    arguments={
                        "selected_candidate_id": "good",
                        "scores": [{"candidate_id": "good", "feasibility": 5, "testability": 5, "scope_control": 5, "recovery_value": 5}],
                        "selection_reason": "valid split",
                    },
                )
            ]
        ),
        completion("ca"),
        completion("cb"),
    ]
    outcome = ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="project-4").start("ship project")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("project-4")
    assert outcome.status == ProjectStatus.SESSION_LIMIT_REACHED.value
    assert {"A", "B"}.issubset({task.key for task in state.tasks})
    events = (cfg.state.root / "project-4" / "project_events.jsonl").read_text(encoding="utf-8")
    assert "recovery_candidates_generated" in events
    assert "recovery_candidate_rejected" in events
    assert "recovery_candidate_selected" in events
