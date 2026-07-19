import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from format_app import preserve_command


def test_preserves_leading_spaces() -> None:
    assert preserve_command("  indented command") == "  indented command"


def test_preserves_exact_formatting() -> None:
    assert preserve_command("\t  exact") == "\t  exact"
