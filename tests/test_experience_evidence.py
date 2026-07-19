from __future__ import annotations

from pathlib import Path

from longrun_agent.knowledge.evidence import build_experience_pack


def test_bash_observation_becomes_failed_test_evidence_with_output_excerpt(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)

    pack = build_experience_pack(
        project_id="p1",
        task_id="t1",
        task_objective="fix validation",
        acceptance_criteria=["pytest passes"],
        session_record=session,
        plan_version=1,
        starting_task_status="in_progress",
        ending_task_status="in_progress",
        workspace_root=workspace,
    )

    failed = next(item for item in pack.evidence_items if item.event_type == "test_command")
    assert failed.exit_code == 1
    assert failed.success is False
    assert failed.output_excerpt
    assert "test_validate_task_name_rejects_empty" in failed.output_excerpt
    assert "AssertionError" in failed.output_excerpt
    assert "assert True is False" in failed.output_excerpt
    assert failed.artifact_path == "tool-output.txt"


def test_files_touched_do_not_create_changed_file_or_write_evidence(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = _session(workspace)
    session["files_touched"] = [str(workspace / "task_app.py")]
    session["changed_files"] = []
    session["written_files"] = []

    pack = build_experience_pack(
        project_id="p1",
        task_id="t1",
        task_objective="learn from failed validation",
        acceptance_criteria=["pytest fails initially"],
        session_record=session,
        plan_version=1,
        starting_task_status="in_progress",
        ending_task_status="in_progress",
        workspace_root=workspace,
    )

    assert pack.files_touched == ["task_app.py"]
    assert pack.files_changed == []
    assert not any(item.event_type == "write_file" for item in pack.evidence_items)
    assert any(item.event_type == "read_file" and item.path == "task_app.py" for item in pack.evidence_items)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "task_app.py").write_text("def validate_task_name(name):\n    return bool(name)\n", encoding="utf-8")
    return workspace


def _session(workspace: Path) -> dict:
    return {
        "session_id": "s1",
        "run_id": "s1",
        "run_status": "max_steps_reached",
        "read_files": [str(workspace / "task_app.py")],
        "changed_files": [],
        "files_touched": [],
        "bash_commands": ["python -m pytest -q"],
        "bash_exit_codes": [1],
        "bash_observations": [
            {
                "command": "python -m pytest -q",
                "argv": ["python", "-m", "pytest", "-q"],
                "exit_code": 1,
                "success": True,
                "is_verification": True,
                "output_excerpt": (
                    "test_validate_task_name_rejects_empty\nAssertionError: assert True is False\nactual True expected False"
                ),
                "error_type": None,
                "artifact_path": "tool-output.txt",
                "operation_index": 2,
            }
        ],
        "successful_test_commands": [],
    }
