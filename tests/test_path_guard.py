from pathlib import Path

import pytest

from longrun_agent.exceptions import WorkspaceSecurityError
from longrun_agent.tools.path_guard import resolve_workspace_path


def test_path_guard_normal_relative_path(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("ok", encoding="utf-8")
    assert resolve_workspace_path(tmp_path, "a.txt", must_exist=True) == target.resolve()


def test_path_guard_rejects_parent_escape(tmp_path: Path):
    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_path_guard_rejects_absolute_path(tmp_path: Path):
    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, str((tmp_path.parent / "outside.txt").resolve()))


def test_path_guard_rejects_similar_prefix(tmp_path: Path):
    evil = tmp_path.parent / f"{tmp_path.name}_evil" / "x.txt"
    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, f"../{evil.parent.name}/x.txt")


def test_path_guard_rejects_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(WorkspaceSecurityError):
        resolve_workspace_path(tmp_path, "link.txt", must_exist=True)
