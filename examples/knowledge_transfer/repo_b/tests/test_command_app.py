import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from command_app import is_valid_command, normalize_command


def test_normalize_command_trims_whitespace() -> None:
    assert normalize_command("  pytest -q  ") == "pytest -q"


def test_is_valid_command_accepts_regular_command() -> None:
    assert is_valid_command("pytest -q") is True


def test_is_valid_command_rejects_empty() -> None:
    assert not is_valid_command("")


def test_is_valid_command_rejects_empty_after_trim() -> None:
    assert not is_valid_command("   ")
