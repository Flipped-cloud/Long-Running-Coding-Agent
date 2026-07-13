import json
from pathlib import Path

import pytest

from longrun_agent.config import InitialPlanConfig, PlanningConfig, PlanningExecutionConfig, load_config
from longrun_agent.exceptions import ConfigurationError
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import completion, config


def write_plan(path: Path, tasks: list[dict]) -> None:
    path.write_text(json.dumps({"project_summary": "summary", "tasks": tasks}), encoding="utf-8")


def task(key: str, deps=None):
    return {
        "key": key,
        "title": key,
        "objective": f"do {key}",
        "acceptance_criteria": ["done"],
        "depends_on_keys": deps or [],
    }


def test_file_plan_does_not_call_model_for_initial_plan(tmp_path: Path):
    plan = tmp_path / "plan.json"
    write_plan(plan, [task("T1")])
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning = PlanningConfig(
        mode="static",
        initial_plan=InitialPlanConfig(source="file", plan_file=plan, min_tasks=1, max_tasks=1),
        execution=PlanningExecutionConfig(max_project_sessions=1, final_verification_command=[]),
    )
    provider = FakeModelProvider([completion("c1")])
    outcome = ProjectOrchestrator(cfg, provider, project_id="file-plan-1").start("ship")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("file-plan-1")
    assert provider.calls == 1
    assert outcome.status == "candidate_complete"
    assert state.tasks[0].id == "file-plan-1:T1"
    events = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).read_events("file-plan-1")
    assert "initial_plan_loaded_from_file" in {event["event_type"] for event in events}


def test_file_plan_generates_task_dependencies(tmp_path: Path):
    plan = tmp_path / "plan.json"
    write_plan(plan, [task("A"), task("B", ["A"])])
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan = InitialPlanConfig(source="file", plan_file=plan, min_tasks=2, max_tasks=2)
    cfg.planning.execution.final_verification_command = []
    ProjectOrchestrator(cfg, FakeModelProvider([completion("c1")]), project_id="file-plan-2").start("ship")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("file-plan-2")
    assert state.task_by_id("file-plan-2:B").dependencies == ["file-plan-2:A"]


def test_invalid_file_plan_is_rejected(tmp_path: Path):
    plan = tmp_path / "plan.json"
    write_plan(plan, [task("A", ["missing"])])
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan = InitialPlanConfig(source="file", plan_file=plan, min_tasks=1, max_tasks=2)
    with pytest.raises(ConfigurationError, match="failed validation"):
        ProjectOrchestrator(cfg, FakeModelProvider([]), project_id="bad-plan").start("ship")


def test_resume_does_not_reload_file_plan(tmp_path: Path):
    plan = tmp_path / "plan.json"
    write_plan(plan, [task("T1"), task("T2", ["T1"])])
    cfg = config(tmp_path, mode="static", max_sessions=1)
    cfg.planning.initial_plan = InitialPlanConfig(source="file", plan_file=plan, min_tasks=2, max_tasks=2)
    cfg.planning.execution.final_verification_command = []
    ProjectOrchestrator(cfg, FakeModelProvider([completion("c1")]), project_id="resume-file-plan").start("ship")
    plan.unlink()

    resumed_cfg = config(tmp_path, mode="static", max_sessions=2)
    resumed_cfg.planning.initial_plan = InitialPlanConfig(source="file", plan_file=plan, min_tasks=2, max_tasks=2)
    resumed_cfg.planning.execution.final_verification_command = []
    outcome = ProjectOrchestrator(resumed_cfg, FakeModelProvider([completion("c2")])).resume("resume-file-plan")
    assert outcome.status == "candidate_complete"


def test_plan_file_path_resolves_relative_to_config_dir(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    plan = plan_dir / "plan.json"
    write_plan(plan, [task("T1")])
    cfg_path = config_dir / "planning.yaml"
    cfg_path.write_text(
        f"""
model:
  provider: fake
  model_name: fake
workspace:
  root: {tmp_path.as_posix()}
planning:
  mode: static
  initial_plan:
    source: file
    plan_file: ../plans/plan.json
    min_tasks: 1
    max_tasks: 1
""",
        encoding="utf-8",
    )
    assert load_config(cfg_path).planning.initial_plan.plan_file == plan.resolve()
