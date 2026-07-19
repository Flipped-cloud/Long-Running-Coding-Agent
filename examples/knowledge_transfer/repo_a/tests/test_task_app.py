import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_app import validate_task_name


def test_validate_task_name_rejects_empty() -> None:
    assert not validate_task_name("")
    assert not validate_task_name("   ")


def test_validate_task_name_accepts_text() -> None:
    assert validate_task_name("ship runtime") is True
