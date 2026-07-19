from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import app
from longrun_agent.evals.experience_learning.runner import run_experience_learning

CONFIG = Path("evals/experience_learning/config.yaml")


def _load_json(stdout: str) -> dict:
    return json.loads(stdout)


def test_packaged_experience_learning_runner_imports() -> None:
    assert callable(run_experience_learning)


def test_packaged_runner_returns_expected_shape() -> None:
    result = run_experience_learning(CONFIG, dry_run=True)
    modes = set(result["modes"])
    assert "backend" in result
    assert "model_provider" in result
    assert "cases" in result
    assert "output_root" in result
    assert modes == {"disabled", "raw_episode", "reflection", "verified_memory", "memory_skill"}


def test_experience_learning_cli_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "experience-learning", "--config", str(CONFIG), "--dry-run"])
    assert result.exit_code == 0, result.output
    payload = _load_json(result.output)
    assert "modes" in payload
    assert "backend" in payload


def test_root_compatibility_runner_runs() -> None:
    result = subprocess.run(
        [sys.executable, "evals/experience_learning/runner.py", "--config", str(CONFIG), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = _load_json(result.stdout)
    assert "modes" in payload


def test_packaged_module_runner_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "longrun_agent.evals.experience_learning.runner", "--config", str(CONFIG), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = _load_json(result.stdout)
    assert "modes" in payload


def test_three_experience_learning_entrypoints_return_same_payload() -> None:
    packaged_function_payload = run_experience_learning(CONFIG, dry_run=True)
    root_script = subprocess.run(
        [sys.executable, "evals/experience_learning/runner.py", "--config", str(CONFIG), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert root_script.returncode == 0, root_script.stderr
    root_script_payload = _load_json(root_script.stdout)

    runner = CliRunner()
    cli = runner.invoke(app, ["eval", "experience-learning", "--config", str(CONFIG), "--dry-run"])
    assert cli.exit_code == 0, cli.output
    cli_payload = _load_json(cli.output)

    assert root_script_payload == packaged_function_payload
    assert cli_payload == packaged_function_payload


def test_memory_skill_e2e_runs_real_orchestrator(tmp_path: Path) -> None:
    config = tmp_path / "experience.yaml"
    repo_a = Path("examples/knowledge_transfer/repo_a").resolve().as_posix()
    repo_b = Path("examples/knowledge_transfer/repo_b").resolve().as_posix()
    repo_c = Path("examples/knowledge_transfer/repo_c_negative").resolve().as_posix()
    output_root = (tmp_path / "runs").resolve().as_posix()
    config.write_text(
        f"""
seed: 7
backend: fake
repeats: 1
output_root: {output_root}
fail_fast_on_knowledge_error: true
modes:
  - memory_skill
cases:
  - case_id: repo_a_learn
    repository: {repo_a}
    task_file: TASK_LEARN.md
    reset_script: reset_repo.py
    role: learning_probe
    initial_verification_should_pass: false
    final_verification_should_pass: false
  - case_id: repo_a_reuse
    repository: {repo_a}
    task_file: TASK_FIX.md
    reset_script: reset_repo.py
    role: same_repository_reuse
    initial_verification_should_pass: false
    final_verification_should_pass: true
  - case_id: repo_b_transfer
    repository: {repo_b}
    task_file: TASK.md
    reset_script: reset_repo.py
    role: positive_transfer
    initial_verification_should_pass: false
    final_verification_should_pass: true
  - case_id: repo_c_negative
    repository: {repo_c}
    task_file: TASK.md
    reset_script: reset_repo.py
    role: negative_transfer
    initial_verification_should_pass: true
    final_verification_should_pass: true
verification:
  command:
    - python
    - -m
    - pytest
    - -q
  timeout_seconds: 30
""",
        encoding="utf-8",
    )

    report = run_experience_learning(config, mode="memory_skill", repeat=1)
    summary = report["results"][0]
    cases = {case["case_id"]: case for case in report["case_results"]}

    assert summary["episode_count"] == 4
    assert summary["reflection_candidate_count"] == 1
    assert summary["active_memory_count"] == 1
    assert summary["active_skill_count"] == 1
    assert summary["skill_retrieval_hit_rate"] == 1.0
    assert summary["negative_transfer_count"] == 0
    assert cases["repo_a_reuse"]["referenced_memory_ids"]
    assert cases["repo_b_transfer"]["referenced_skill_ids"]
    assert cases["repo_c_negative"]["exposed_skill_ids"] == []


def test_api_backend_model_config_validates_and_dry_run_is_safe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "api-smoke-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _write_minimal_config(tmp_path, backend="api", model=True)

    class ForbiddenProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dry-run must not instantiate a provider")

    monkeypatch.setattr("longrun_agent.evals.experience_learning.executor.OpenAICompatibleProvider", ForbiddenProvider)
    payload = run_experience_learning(config, dry_run=True)

    assert payload == {
        "backend": "api",
        "model_provider": "openai_compatible",
        "model_name": "api-smoke-model",
        "base_url_configured": True,
        "api_key_configured": True,
        "modes": ["verified_memory"],
        "repeats": [1],
        "cases": ["repo_a_learn"],
        "output_root": str(tmp_path / "runs"),
    }


def test_api_backend_requires_model(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path, backend="api", model=False)
    result = subprocess.run(
        [sys.executable, "-m", "longrun_agent.evals.experience_learning.runner", "--config", str(config), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires top-level model configuration" in result.stderr


def test_fake_backend_allows_missing_model(tmp_path: Path) -> None:
    payload = run_experience_learning(_write_minimal_config(tmp_path, backend="fake", model=False), dry_run=True)
    assert payload["backend"] == "fake"
    assert payload["model_provider"] == "fake"
    assert payload["model_name"] == "experience-learning-fake"


def test_api_backend_unset_environment_variable_is_clear(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.test/v1")
    config = _write_minimal_config(tmp_path, backend="api", model=True)
    result = subprocess.run(
        [sys.executable, "-m", "longrun_agent.evals.experience_learning.runner", "--config", str(config), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "model.model_name is required unless provider is fake" in result.stderr


def _write_minimal_config(tmp_path: Path, *, backend: str, model: bool) -> Path:
    repo_a = Path("examples/knowledge_transfer/repo_a").resolve().as_posix()
    model_block = (
        """
model:
  provider: openai_compatible
  model_name: "${MODEL_NAME}"
  base_url: "${OPENAI_BASE_URL}"
  api_key_env: OPENAI_API_KEY
  temperature: 0.0
  max_output_tokens: 1024
  request_timeout_seconds: 30
  max_api_retries: 1
"""
        if model
        else ""
    )
    config = tmp_path / f"{backend}.yaml"
    config.write_text(
        f"""
seed: 7
backend: {backend}
repeats: 1
output_root: runs
fail_fast_on_knowledge_error: true
modes:
  - verified_memory
cases:
  - case_id: repo_a_learn
    repository: {repo_a}
    task_file: TASK_LEARN.md
    reset_script: reset_repo.py
    role: learning_probe
    initial_verification_should_pass: false
    final_verification_should_pass: false
{model_block}
verification:
  command:
    - python
    - -m
    - pytest
    - -q
  timeout_seconds: 30
""",
        encoding="utf-8",
    )
    return config
