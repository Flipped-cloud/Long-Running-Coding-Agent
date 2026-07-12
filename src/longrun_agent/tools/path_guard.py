from __future__ import annotations

import os
from pathlib import Path

from longrun_agent.exceptions import WorkspaceSecurityError


def ensure_workspace_root(workspace_root: str | Path) -> Path:
    """Return an existing resolved workspace directory."""

    root = Path(workspace_root).resolve()
    if not root.exists() or not root.is_dir():
        raise WorkspaceSecurityError(f"workspace root does not exist: {workspace_root}")
    return root


def is_inside_path(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([str(child), str(parent)]) == str(parent)
    except ValueError:
        return False


def resolve_workspace_path(
    workspace_root: str | Path,
    requested_path: str,
    *,
    must_exist: bool = False,
    allow_create: bool = False,
) -> Path:
    """Resolve a model supplied relative path and reject workspace escapes."""

    if not requested_path or not requested_path.strip():
        raise WorkspaceSecurityError("empty paths are not allowed")
    root = ensure_workspace_root(workspace_root)
    raw = Path(requested_path)
    if raw.is_absolute():
        raise WorkspaceSecurityError("absolute paths are not allowed")

    candidate = root / raw
    if must_exist and not candidate.exists():
        raise FileNotFoundError(requested_path)

    parent = candidate.parent
    while not parent.exists() and parent != root:
        parent = parent.parent
    if not is_inside_path(parent.resolve(), root):
        raise WorkspaceSecurityError("path escapes workspace")

    if allow_create:
        for ancestor in candidate.parents:
            if ancestor == root.parent:
                break
            if ancestor.exists() and ancestor.is_symlink() and not is_inside_path(ancestor.resolve(), root):
                raise WorkspaceSecurityError("symlink parent escapes workspace")

    resolved = candidate.resolve(strict=False)
    if not is_inside_path(resolved, root):
        raise WorkspaceSecurityError("path escapes workspace")
    if candidate.exists() and candidate.is_symlink() and not is_inside_path(candidate.resolve(), root):
        raise WorkspaceSecurityError("symlink escapes workspace")
    return resolved


def relative_to_workspace(workspace_root: str | Path, path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(workspace_root).resolve())).replace("\\", "/")
