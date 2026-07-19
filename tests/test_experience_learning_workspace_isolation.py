from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import longrun_agent.evals.experience_learning.executor as experience_executor
from longrun_agent.evals.experience_learning.executor import _prepare_case_workspace
from longrun_agent.evals.experience_learning.runner import run_experience_learning
from longrun_agent.knowledge.evidence import RepositoryProfiler

TEMPLATE_REPOS = [
    Path("examples/knowledge_transfer/repo_a"),
    Path("examples/knowledge_transfer/repo_b"),
    Path("examples/knowledge_transfer/repo_c_negative"),
]


@pytest.fixture(autouse=True)
def forbid_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class ForbiddenProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("workspace isolation tests must not instantiate a real API provider")

    monkeypatch.setattr(experience_executor, "OpenAICompatibleProvider", ForbiddenProvider)


def test_template_repositories_are_not_modified_and_workspaces_are_isolated(tmp_path: Path) -> None:
    before = {repo: _tree_hash(repo) for repo in TEMPLATE_REPOS}
    config = _write_config(tmp_path)
    report = run_experience_learning(config)
    after = {repo: _tree_hash(repo) for repo in TEMPLATE_REPOS}

    assert after == before
    output_root = tmp_path / "runs"
    disabled_repo_a = output_root / "disabled" / "repeat_01" / "workspaces" / "repo_a_reuse" / "task_app.py"
    memory_skill_repo_a = output_root / "memory_skill" / "repeat_01" / "workspaces" / "repo_a_reuse" / "task_app.py"
    repeat_2_repo_a = output_root / "memory_skill" / "repeat_02" / "workspaces" / "repo_a_reuse" / "task_app.py"
    assert disabled_repo_a.exists()
    assert memory_skill_repo_a.exists()
    assert repeat_2_repo_a.exists()
    assert memory_skill_repo_a.read_text(encoding="utf-8") == repeat_2_repo_a.read_text(encoding="utf-8")
    assert "name.strip()" in memory_skill_repo_a.read_text(encoding="utf-8")
    assert "name.strip()" in disabled_repo_a.read_text(encoding="utf-8")
    assert "name.strip()" not in (Path("examples/knowledge_transfer/repo_a") / "task_app.py").read_text(encoding="utf-8")

    cases = {(case["mode"], case["repeat"], case["case_id"]): case for case in report["case_results"]}
    for key, case in cases.items():
        if key[2] == "repo_c_negative":
            assert case["initial_verification"]["passed"] is True
            assert case["final_verification"]["passed"] is True
            assert "2 passed" in case["initial_verification"]["stdout"]
            assert "2 passed" in case["final_verification"]["stdout"]
            assert case["modified_after_run"] is False
            assert Path(case["repository"]).is_relative_to(output_root / key[0] / f"repeat_{key[1]:02d}" / "workspaces")


def test_repository_fingerprint_uses_workspace_relative_skip_rules(tmp_path: Path) -> None:
    workspaces = tmp_path / ".runs" / "experience_learning" / "memory_skill" / "repeat_01" / "workspaces"
    repo_a_learn = _prepare_case_workspace(Path("examples/knowledge_transfer/repo_a"), workspaces / "repo_a_learn")
    repo_a_reuse = _prepare_case_workspace(Path("examples/knowledge_transfer/repo_a"), workspaces / "repo_a_reuse")
    repo_b = _prepare_case_workspace(Path("examples/knowledge_transfer/repo_b"), workspaces / "repo_b_transfer")
    repo_c = _prepare_case_workspace(Path("examples/knowledge_transfer/repo_c_negative"), workspaces / "repo_c_negative")

    a_learn = RepositoryProfiler(repo_a_learn).profile().repository_fingerprint
    a_reuse = RepositoryProfiler(repo_a_reuse).profile().repository_fingerprint
    b = RepositoryProfiler(repo_b).profile().repository_fingerprint
    c = RepositoryProfiler(repo_c).profile().repository_fingerprint

    assert a_learn != "unknown"
    assert a_learn == a_reuse
    assert len({a_learn, b, c}) == 3

    (repo_a_reuse / "task_app.py").write_text("def validate_task_name(name: str) -> bool:\n    return True\n", encoding="utf-8")
    assert RepositoryProfiler(repo_a_reuse).profile().repository_fingerprint == a_reuse

    (repo_a_reuse / "__pycache__").mkdir()
    (repo_a_reuse / "__pycache__" / "task_app.cpython-311.pyc").write_bytes(b"cache")
    (repo_a_reuse / ".runs").mkdir()
    (repo_a_reuse / ".runs" / "scratch.py").write_text("x = 1\n", encoding="utf-8")
    assert RepositoryProfiler(repo_a_reuse).profile().repository_fingerprint == a_reuse


def _write_config(tmp_path: Path) -> Path:
    repo_a = Path("examples/knowledge_transfer/repo_a").resolve().as_posix()
    repo_b = Path("examples/knowledge_transfer/repo_b").resolve().as_posix()
    repo_c = Path("examples/knowledge_transfer/repo_c_negative").resolve().as_posix()
    path = tmp_path / "experience.yaml"
    path.write_text(
        f"""
seed: 7
backend: fake
repeats: 2
output_root: {(tmp_path / "runs").as_posix()}
fail_fast_on_knowledge_error: true
modes:
  - disabled
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
    return path


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not _ignored(item)):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ignored(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {"__pycache__", ".pytest_cache", ".runs"}) or path.suffix == ".pyc"
