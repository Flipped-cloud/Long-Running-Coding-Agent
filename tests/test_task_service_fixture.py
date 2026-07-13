from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from longrun_agent.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "task_service_repo"
RESET_OUTPUT = "task_service_repo reset: 8 files restored"
CHECK_OUTPUT = "task_service_repo fixture valid"


def copy_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "task_service_repo"
    shutil.copytree(FIXTURE, repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return repo


def run_cmd(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(repo)}
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_reset(repo: Path) -> subprocess.CompletedProcess[str]:
    return run_cmd(repo, "reset_repo.py")


def run_check(repo: Path) -> subprocess.CompletedProcess[str]:
    return run_cmd(repo, "reset_repo.py", "--check")


def assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def test_reset_restores_gap_coverage_and_removes_stale_tests(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    run_reset(repo)
    (repo / "tests/test_gap_coverage.py").unlink()
    (repo / "tests/test_stale_generated.py").write_text("def test_stale():\n    assert True\n", encoding="utf-8")
    (repo / "tests/test_edge_cases.py").write_text("def test_edge():\n    assert True\n", encoding="utf-8")

    result = run_reset(repo)

    assert_success(result)
    assert RESET_OUTPUT in result.stdout
    assert (repo / "tests/test_gap_coverage.py").is_file()
    assert {path.name for path in (repo / "tests").glob("test_*.py")} == {
        "test_service.py",
        "test_gap_coverage.py",
    }


def test_reset_restores_model_and_removes_agent_leftovers(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    run_reset(repo)
    baseline_model = (repo / "task_service/model.py").read_text(encoding="utf-8")
    (repo / "task_service/model.py").write_text("BROKEN = True\n", encoding="utf-8")
    (repo / "INSPECTION_NOTES.md").write_text("stale\n", encoding="utf-8")
    (repo / "VALIDATION_IMPROVEMENTS.md").write_text("stale\n", encoding="utf-8")
    (repo / "tmp_validation").mkdir()

    result = run_reset(repo)

    assert_success(result)
    assert (repo / "task_service/model.py").read_text(encoding="utf-8") == baseline_model
    assert "expected a Task instance" not in baseline_model
    assert not (repo / "INSPECTION_NOTES.md").exists()
    assert not (repo / "VALIDATION_IMPROVEMENTS.md").exists()
    assert not (repo / "tmp_validation").exists()


def test_reset_preserves_protected_files(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    protected = [repo / "TASK.md", repo / "pyproject.toml", *repo.glob("scripted_project_*.json"), *repo.glob("scripted_resume_*.json")]
    before = {path.name: path.read_bytes() for path in protected}

    result = run_reset(repo)

    assert_success(result)
    assert {path.name: path.read_bytes() for path in protected} == before


def test_check_succeeds_for_valid_fixture_and_fails_for_missing_gap_test(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    assert_success(run_reset(repo))

    valid = run_check(repo)
    assert_success(valid)
    assert CHECK_OUTPUT in valid.stdout

    (repo / "tests/test_gap_coverage.py").unlink()
    invalid = run_check(repo)
    assert invalid.returncode != 0
    assert "missing test file: tests/test_gap_coverage.py" in invalid.stderr


def test_reset_is_idempotent_for_baseline_files(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    assert_success(run_reset(repo))
    baseline_paths = [
        "task_service/__init__.py",
        "task_service/model.py",
        "task_service/storage.py",
        "task_service/service.py",
        "task_service/cli.py",
        "tests/test_service.py",
        "tests/test_gap_coverage.py",
        "README.md",
    ]
    first = {path: (repo / path).read_text(encoding="utf-8") for path in baseline_paths}

    assert_success(run_reset(repo))

    second = {path: (repo / path).read_text(encoding="utf-8") for path in baseline_paths}
    assert second == first


def test_plan_model_validation_uses_behavioral_acceptance(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    plan = json.loads((repo / "plan_glm47_fast.json").read_text(encoding="utf-8"))
    task = next(item for item in plan["tasks"] if item["key"] == "model-validation")

    assert "If the focused validation test file is unexpectedly missing" in task["objective"]
    assert "create it from the stated behavioral requirements instead of reporting a blocker" in task["objective"]
    assert task["acceptance_criteria"][:3] == [
        "validate_task rejects a dict with ValueError('invalid task: expected a Task instance, got dict')",
        "validate_task rejects bool attempts with the required exact ValueError message",
        "validate_task rejects negative attempts with the required exact ValueError message",
    ]
    assert task["acceptance_criteria"][3] == "python -m pytest -q tests/test_gap_coverage.py passes"


def test_gap_coverage_collects_three_tests_after_reset(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    assert_success(run_reset(repo))

    result = run_cmd(repo, "-m", "pytest", "--collect-only", "-q", "tests/test_gap_coverage.py")

    assert_success(result)
    assert "test_validate_task_rejects_bool_attempts" in result.stdout
    assert "test_validate_task_rejects_negative_attempts" in result.stdout
    assert "test_validate_task_rejects_non_task_input" in result.stdout


def test_reset_baseline_test_behavior(tmp_path: Path):
    repo = copy_fixture(tmp_path)
    assert_success(run_reset(repo))

    service = run_cmd(repo, "-m", "pytest", "-q", "tests/test_service.py")
    gap = run_cmd(repo, "-m", "pytest", "-q", "tests/test_gap_coverage.py")

    assert_success(service)
    assert gap.returncode == 1, gap.stdout + gap.stderr
    assert gap.returncode != 4
    assert "3 failed" in gap.stdout


def test_glm47_config_points_to_existing_fixture_and_plan(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "glm-4.7-flash")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    cfg = load_config(ROOT / "configs/planning_static_glm47_30min.yaml")

    assert cfg.workspace.root.is_absolute()
    assert cfg.workspace.root.exists()
    assert cfg.workspace.root == FIXTURE.resolve()
    assert cfg.planning.mode == "static"
    assert cfg.planning.initial_plan.source == "file"
    assert cfg.planning.initial_plan.plan_file == (FIXTURE / "plan_glm47_fast.json").resolve()
    assert cfg.planning.initial_plan.plan_file.exists()
    assert cfg.planning.initial_plan.min_tasks == 4
    assert cfg.planning.initial_plan.max_tasks == 4
