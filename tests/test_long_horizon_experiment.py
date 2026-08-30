from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from longrun_agent.config import ConfigurationError, load_config
from longrun_agent.verification.contract import load_contract
from longrun_agent.verification.schema import CheckVisibility

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "long_horizon_real_api.yaml"
COMPREHENSIVE_CONFIG = REPO_ROOT / "configs" / "comprehensive_real_api.yaml"
PLAN = REPO_ROOT / "evals" / "long_horizon" / "plan.json"
COMPREHENSIVE_PLAN = REPO_ROOT / "evals" / "comprehensive" / "plan.json"
FIXTURE = REPO_ROOT / "examples" / "long_horizon_repo"
TASK_FILE = REPO_ROOT / "evals" / "long_horizon" / "TASK.md"
HIDDEN_TESTS = REPO_ROOT / "evals" / "long_horizon" / "hidden_assets" / "hidden_tests"
CONTRACT = REPO_ROOT / "evals" / "long_horizon" / "verification_contract.yaml"
COMPREHENSIVE_CONTRACT = REPO_ROOT / "evals" / "comprehensive" / "verification_contract.yaml"
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_long_horizon_real_api.sh"
VALIDATOR = REPO_ROOT / "scripts" / "validate_long_horizon_result.py"
SUMMARIZER = REPO_ROOT / "scripts" / "summarize_long_horizon_run.py"


def bash_executable() -> str:
    discovered = shutil.which("bash")
    candidates = [
        r"D:\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        discovered,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and Path(candidate).parent.name.lower() != "system32":
            return candidate
    pytest.skip("bash is not installed")


def configured_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "integration-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("LONGRUN_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("LONGRUN_PLAN_FILE", str(PLAN))


def test_long_horizon_config_loads_without_constructing_provider(tmp_path, monkeypatch):
    configured_environment(tmp_path, monkeypatch)
    config = load_config(CONFIG)
    assert config.model.provider == "openai_compatible"
    assert config.model.model_name == "integration-model"
    assert config.model.base_url == "https://example.invalid/v1"
    assert config.model.api_key_env == "OPENAI_API_KEY"
    assert config.workspace.root == (tmp_path / "workspace").resolve()
    assert config.planning.mode == "static"
    assert config.planning.initial_plan.plan_file == PLAN.resolve()
    assert config.planning.initial_plan.min_tasks == config.planning.initial_plan.max_tasks == 5
    assert config.planning.execution.max_project_sessions == 10
    assert config.planning.execution.max_sessions_per_task == 3
    assert config.planning.execution.max_no_progress_sessions == 2
    assert config.planning.execution.max_project_seconds == 3300
    assert config.agent.max_steps == 14
    assert config.context.mode == "structured_reset"
    assert config.context.model_context_limit == 32768
    assert config.context.trigger_ratio == 0.86
    assert not config.context.structured_handoff.use_model
    assert config.context.structured_handoff.fallback_deterministic
    assert config.knowledge.mode == "memory_skill"
    assert config.knowledge.reflection.enabled
    assert config.knowledge.skill.enabled
    assert not config.knowledge.skill.auto_execute
    assert config.state.atomic_write
    assert config.verification.mode == "contract"
    assert config.verification.contract.path == CONTRACT.resolve()
    assert config.verification.execution.isolation == "copy"
    assert config.verification.policy.require_project_contract


def test_long_horizon_config_rejects_missing_runtime_paths(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "integration-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.delenv("LONGRUN_WORKSPACE", raising=False)
    monkeypatch.delenv("LONGRUN_PLAN_FILE", raising=False)

    with pytest.raises(ConfigurationError, match="LONGRUN_PLAN_FILE, LONGRUN_WORKSPACE"):
        load_config(CONFIG)


def test_comprehensive_config_enables_generated_tests_and_adaptive_recovery(tmp_path, monkeypatch):
    configured_environment(tmp_path, monkeypatch)
    monkeypatch.setenv("LONGRUN_PLAN_FILE", str(COMPREHENSIVE_PLAN))

    config = load_config(COMPREHENSIVE_CONFIG)

    assert config.planning.mode == "adaptive_search"
    assert config.planning.initial_plan.min_tasks == config.planning.initial_plan.max_tasks == 5
    assert config.agent.max_steps == 28
    assert config.planning.execution.max_project_sessions == 15
    assert config.planning.execution.max_sessions_per_task == 3
    assert config.planning.decomposition.max_depth == 1
    assert config.planning.bounded_search.enabled
    assert config.context.mode == "structured_reset"
    assert config.knowledge.mode == "memory_skill"
    assert config.verification.contract.path == COMPREHENSIVE_CONTRACT.resolve()
    assert config.verification.generated_tests.enabled
    assert not config.verification.generated_tests.require_candidate_before_completion
    assert config.verification.generated_tests.minimum_valid_candidates == 1


def test_long_horizon_outputs_are_outside_agent_workspace(tmp_path, monkeypatch):
    configured_environment(tmp_path, monkeypatch)
    config = load_config(CONFIG)
    workspace = config.workspace.root.resolve()
    for output_root in (config.state.root, config.telemetry.run_root, config.knowledge.root):
        assert workspace != output_root.resolve()
        assert workspace not in output_root.resolve().parents
        assert output_root.resolve().is_relative_to(REPO_ROOT / ".runs" / "long_horizon_real_api")


def test_static_plan_has_expected_unique_dag():
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    assert len(tasks) == 5
    keys = [task["key"] for task in tasks]
    assert len(keys) == len(set(keys))
    assert all(task["objective"].strip() and task["acceptance_criteria"] for task in tasks)
    by_key = {task["key"]: task for task in tasks}
    assert all(dependency in by_key for task in tasks for dependency in task["depends_on_keys"])

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise AssertionError(f"cycle detected at {key}")
        if key in visited:
            return
        visiting.add(key)
        for dependency in by_key[key]["depends_on_keys"]:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in keys:
        visit(key)
    assert set(by_key["integration-docs"]["depends_on_keys"]) == set(keys) - {"integration-docs"}
    lifecycle_check = by_key["lifecycle-core"]["acceptance_criteria"][-1]
    assert "-k 'run_requires_satisfied_dependencies" in lifecycle_check
    cli_check = by_key["cli-execution"]["acceptance_criteria"][-1]
    assert "-k 'run_retry_cancel_paths or retry_failed_task'" in cli_check


def test_long_horizon_contract_keeps_hidden_oracle_outside_workspace():
    contract = load_contract(CONTRACT, workspace_root=FIXTURE)
    assert contract.scope == "project"
    assert contract.hidden_assets_root == (CONTRACT.parent / "hidden_assets").resolve()
    assert contract.hidden_assets_root.is_dir()
    assert [check.visibility for check in contract.checks] == [CheckVisibility.PUBLIC, CheckVisibility.HIDDEN]
    assert contract.integrity_rules.allowed_change_patterns == ["workflow_service/*.py", "README.md"]
    assert contract.integrity_rules.max_deleted_files == 0


def test_comprehensive_plan_and_contract_cover_five_generated_test_tasks():
    payload = json.loads(COMPREHENSIVE_PLAN.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    assert len(tasks) == 5
    assert "register" in tasks[0]["objective"].lower()
    assert all("register" not in task["objective"].lower() for task in tasks[1:])
    assert tasks[-1]["depends_on_keys"] == [task["key"] for task in tasks[:-1]]

    contract = load_contract(COMPREHENSIVE_CONTRACT, workspace_root=FIXTURE)
    assert contract.generated_test_policy.enabled
    assert contract.integrity_rules.allowed_test_directories == ["agent_tests"]
    assert "agent_tests/*.py" in contract.integrity_rules.allowed_change_patterns
    assert contract.hidden_assets_root == (REPO_ROOT / "evals" / "long_horizon" / "hidden_assets").resolve()


def write_project_artifacts(root: Path, project_id: str, status: str, *, include_optional: bool) -> tuple[Path, Path]:
    state_root = root / "projects"
    project_dir = state_root / project_id
    project_dir.mkdir(parents=True)
    state = {
        "project_id": project_id,
        "status": status,
        "session_count": 2,
        "tasks": [
            {"id": f"{project_id}:a", "status": "verified", "reopen_count": 1},
            {"id": f"{project_id}:b", "status": "failed", "reopen_count": 0},
        ],
        "revisions": [{"revision_id": "r1"}],
    }
    (project_dir / "project_state.json").write_text(json.dumps(state), encoding="utf-8")
    sessions = [
        {
            "steps": 3,
            "tool_call_count": 4,
            "total_tokens": 100,
            "input_tokens_total": 70,
            "output_tokens_total": 30,
            "context_reset_count": 1,
            "changed_files": ["workflow_service/models.py"],
            "successful_test_commands": ["python -m pytest -q tests/test_models.py"],
            "repeated_tool_calls": ["read_file:models.py"],
        },
        {"steps": 2, "tool_call_count": 1, "total_tokens": 50},
    ]
    (project_dir / "sessions.jsonl").write_text(
        "".join(json.dumps(session) + "\n" for session in sessions),
        encoding="utf-8",
    )
    if include_optional:
        (project_dir / "project_metrics.json").write_text(
            json.dumps(
                {
                    "total_tool_calls": 5,
                    "total_tokens": 150,
                    "total_context_resets": 1,
                    "repeated_tool_calls": 1,
                    "final_verification_passed": status == "candidate_complete",
                }
            ),
            encoding="utf-8",
        )
    return state_root, project_dir


@pytest.mark.parametrize(
    ("status", "include_optional", "expected_verification"),
    [
        ("candidate_complete", True, "passed"),
        ("failed", True, "failed"),
        ("time_limit_reached", False, "not_recorded"),
    ],
)
def test_summarizer_handles_terminal_states_and_missing_optional_metrics(
    tmp_path,
    status,
    include_optional,
    expected_verification,
):
    project_id = f"summary-{status}"
    state_root, _ = write_project_artifacts(tmp_path, project_id, status, include_optional=include_optional)
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"oracle_verified": False, "integrity_passed": True}), encoding="utf-8")
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARIZER),
            "--project-id",
            project_id,
            "--state-root",
            str(state_root),
            "--telemetry-root",
            str(tmp_path / "telemetry"),
            "--workspace",
            str(tmp_path / "workspace"),
            "--console-log",
            str(tmp_path / "console.log"),
            "--oracle",
            str(oracle),
            "--started-at",
            "2026-01-01T00:00:00Z",
            "--ended-at",
            "2026-01-01T00:31:00Z",
            "--elapsed-seconds",
            "1860",
            "--minimum-target-seconds",
            "1800",
            "--configured-project-limit-seconds",
            "3600",
            "--cli-exit-code",
            "1",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["project_status"] == status
    assert summary["duration_target_met"]
    assert summary["session_count"] == 2
    assert summary["total_model_steps"] == 5
    assert summary["final_verification_status"] == expected_verification
    assert summary["integrity_passed"]


def test_validator_detects_test_tampering_and_always_writes_json(tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(FIXTURE, workspace)
    shutil.copy2(TASK_FILE, workspace / "TASK.md")
    test_file = workspace / "tests" / "test_models.py"
    test_file.write_text(test_file.read_text(encoding="utf-8") + "\n# weakened\n", encoding="utf-8")
    (workspace / "conftest.py").write_text("collect_ignore = ['tests/test_models.py']\n", encoding="utf-8")
    output = tmp_path / "oracle.json"
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--workspace",
            str(workspace),
            "--fixture",
            str(FIXTURE),
            "--task-file",
            str(TASK_FILE),
            "--config",
            str(CONFIG),
            "--hidden-tests",
            str(HIDDEN_TESTS),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert not payload["integrity_passed"]
    assert "public_tests_unchanged" in payload["failed_checks"]
    assert "workspace_change_scope_respected" in payload["failed_checks"]
    assert payload["violations"]


def test_validator_allows_declared_generated_test_directory(tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(FIXTURE, workspace)
    shutil.copy2(TASK_FILE, workspace / "TASK.md")
    generated = workspace / "agent_tests" / "test_generated.py"
    generated.parent.mkdir()
    generated.write_text("def test_generated():\n    assert True\n", encoding="utf-8")
    output = tmp_path / "oracle.json"

    subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--workspace",
            str(workspace),
            "--fixture",
            str(FIXTURE),
            "--task-file",
            str(TASK_FILE),
            "--config",
            str(COMPREHENSIVE_CONFIG),
            "--hidden-tests",
            str(HIDDEN_TESTS),
            "--generated-test-dir",
            "agent_tests",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "workspace_change_scope_respected" in payload["passed_checks"]
    assert not any("agent_tests" in violation for violation in payload["violations"])


def test_run_script_fails_fast_without_api_environment(tmp_path):
    environment = os.environ.copy()
    for name in ("MODEL_NAME", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        environment.pop(name, None)
    environment["PROJECT_ID"] = "must-not-create-workspace"
    result = subprocess.run(
        [bash_executable(), str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert "missing required environment variable" in result.stderr
    assert not (REPO_ROOT / ".runs" / "long_horizon_real_api" / "workspaces" / environment["PROJECT_ID"]).exists()


def test_resume_mode_requires_explicit_project_id():
    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_NAME": "integration-model",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
            "OPENAI_API_KEY": "test-only-key",
            "RESUME_PROJECT": "1",
        }
    )
    environment.pop("PROJECT_ID", None)
    result = subprocess.run(
        [bash_executable(), str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert "PROJECT_ID is required when RESUME_PROJECT=1" in result.stderr


def test_long_horizon_shell_script_has_valid_bash_syntax():
    result = subprocess.run(
        [bash_executable(), "-n", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_run_script_creates_an_isolated_workspace_repository_before_execution():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'git -C "$WORKSPACE" init --quiet' in script
    assert script.index('git -C "$WORKSPACE" init --quiet') < script.index('echo "[baseline] $CASE_DESCRIPTION"')
