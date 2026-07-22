from __future__ import annotations

import json
import shutil
from pathlib import Path

from longrun_agent.config import load_config
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.store import ProjectStateStore
from longrun_agent.verification.store import VerificationStore


def test_full_system_config_and_fake_contract_e2e(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    config = load_config("configs/full_system.yaml")
    assert config.model.provider == "fake"
    assert config.planning.mode == "adaptive_search"
    assert config.context.mode == "structured_reset"
    assert config.knowledge.mode == "memory_skill"
    assert config.knowledge.skill.enabled is True
    assert config.verification.mode == "contract"
    assert config.verification.execution.isolation == "copy"
    assert config.state.atomic_write is True
    assert config.telemetry.save_prompts is True

    workspace = tmp_path / "workspace"
    shutil.copytree(Path("examples/verification_bench/full_fix"), workspace, ignore=shutil.ignore_patterns("__pycache__"))
    config.workspace.root = workspace
    config.state.root = tmp_path / "state"
    config.telemetry.run_root = tmp_path / "telemetry"
    config.knowledge.root = tmp_path / "knowledge"
    config.verification.store_root = config.state.root

    project_id = "full-system-integration"
    provider = FakeModelProvider(
        [
            _plan(),
            _complete("premature completion without implementation evidence"),
            _write_full_fix(),
            _pytest_public(),
            _complete("negative classification fixed and public pytest passes"),
        ]
    )
    orchestrator = ProjectOrchestrator(config, provider, project_id=project_id)

    first = orchestrator.start((workspace / "TASK.md").read_text(encoding="utf-8"))
    store = ProjectStateStore(config.state.root, workspace_root=workspace)
    first_state = store.load(project_id)
    reports_after_first = VerificationStore(config.state.root, project_id, workspace_root=workspace).list_reports()
    assert first.status == "active"
    assert first.verification_verdict == "reopened"
    assert first_state.tasks[0].status.value == "ready"
    assert [report.verdict.value for report in reports_after_first] == ["reopened"]

    second = orchestrator.resume(project_id)
    final_state = store.load(project_id)
    sessions = store.read_sessions(project_id)
    events = store.read_events(project_id)
    context_events = store.read_context_events(project_id)
    reports = VerificationStore(config.state.root, project_id, workspace_root=workspace).list_reports()
    knowledge_events = _read_jsonl(config.knowledge.root / "events.jsonl")

    assert second.status == "verified"
    assert final_state.status.value == "verified"
    assert final_state.tasks[0].status.value == "verified"
    assert [report.verdict.value for report in reports] == ["reopened", "verified"]
    assert sum(event["event_type"] == "task_completion_requested" for event in events) == 2
    assert any(event["event_type"] == "project_verified" for event in events)
    assert len(sessions) == 2
    assert any(session["terminal_signal"] == "completion_request" for session in sessions)
    assert any(session["changed_files"] == ["app.py"] for session in sessions)
    assert any(event["event_type"] == "context_budget_measured" and event["mode"] == "structured_reset" for event in context_events)
    assert any(event["event_type"] == "episode_created" for event in knowledge_events)
    assert (config.telemetry.run_root / f"{project_id}-s1" / "events.jsonl").exists()
    assert (config.telemetry.run_root / f"{project_id}-s2" / "events.jsonl").exists()
    assert provider.calls == 5


def _plan() -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id="plan",
                name="submit_plan",
                arguments={
                    "project_summary": "fix negative classification",
                    "tasks": [
                        {
                            "key": "fix",
                            "title": "Fix classifier",
                            "objective": "Make negative values classify as negative while preserving public behavior.",
                            "acceptance_criteria": ["contract verification passes"],
                            "depends_on_keys": [],
                        }
                    ],
                },
            )
        ]
    )


def _complete(summary: str) -> ModelResponse:
    call_id = "complete-" + "".join(char if char.isalnum() else "-" for char in summary.lower())[:48].strip("-")
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="request_task_completion",
                arguments={"summary": summary, "acceptance_criteria_addressed": ["contract verification passes"]},
            )
        ]
    )


def _write_full_fix() -> ModelResponse:
    content = (
        "def classify_number(value: int) -> str:\n"
        "    if value < 0:\n"
        '        return "negative"\n'
        "    if value == 0:\n"
        '        return "zero"\n'
        '    return "positive"\n'
    )
    return ModelResponse(tool_calls=[ToolCall(id="write-fix", name="write_file", arguments={"path": "app.py", "content": content})])


def _pytest_public() -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id="pytest-public", name="bash", arguments={"argv": ["python", "-m", "pytest", "-q", "tests/test_public.py"]})]
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
