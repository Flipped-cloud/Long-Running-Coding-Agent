import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_get_found(tmp_path: Path):
    db = tmp_path / "tasks.json"
    # First add a task
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "add",
            "--db",
            str(db),
            "--id",
            "42",
            "--title",
            "The Answer",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Now get it
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "get",
            "--db",
            str(db),
            "--id",
            "42",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "42: The Answer completed=False attempts=0" in result.stdout


def test_cli_get_missing(tmp_path: Path):
    db = tmp_path / "tasks.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "get",
            "--db",
            str(db),
            "--id",
            "nonexistent",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "task not found: nonexistent" in result.stdout


def test_cli_add_and_list(tmp_path: Path):
    db = tmp_path / "tasks.json"
    # Add a task
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "add",
            "--db",
            str(db),
            "--id",
            "1",
            "--title",
            "First",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # List tasks
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "list",
            "--db",
            str(db),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "1: First completed=False attempts=0" in result.stdout


def test_cli_complete(tmp_path: Path):
    db = tmp_path / "tasks.json"
    # Add a task
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "add",
            "--db",
            str(db),
            "--id",
            "2",
            "--title",
            "Second",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Complete it
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "complete",
            "--db",
            str(db),
            "--id",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Verify via get
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_service.cli",
            "get",
            "--db",
            str(db),
            "--id",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "completed=True" in result.stdout
