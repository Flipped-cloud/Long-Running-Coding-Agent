from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples" / "knowledge_transfer"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)


def _reset(repo: Path) -> None:
    result = _run([sys.executable, "reset_repo.py"], repo)
    assert result.returncode == 0, result.stderr


def _tracked_file_contents(repo: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in repo.glob("*.py") if path.name != "reset_repo.py"}


def test_repo_a_reset_is_repeatable_and_pytest_fails() -> None:
    repo = FIXTURES / "repo_a"
    _reset(repo)
    first = _tracked_file_contents(repo)
    _reset(repo)
    assert _tracked_file_contents(repo) == first
    result = _run([sys.executable, "-m", "pytest", "-q"], repo)
    assert result.returncode != 0
    assert "test_validate_task_name_rejects_empty" in result.stdout


def test_repo_b_reset_is_repeatable_and_pytest_fails() -> None:
    repo = FIXTURES / "repo_b"
    _reset(repo)
    first = _tracked_file_contents(repo)
    _reset(repo)
    assert _tracked_file_contents(repo) == first
    result = _run([sys.executable, "-m", "pytest", "-q"], repo)
    assert result.returncode != 0
    assert "test_normalize_command_trims_whitespace" in result.stdout


def test_repo_c_reset_is_repeatable_and_pytest_passes() -> None:
    repo = FIXTURES / "repo_c_negative"
    _reset(repo)
    first = _tracked_file_contents(repo)
    _reset(repo)
    assert _tracked_file_contents(repo) == first
    result = _run([sys.executable, "-m", "pytest", "-q"], repo)
    assert result.returncode == 0, result.stdout + result.stderr
