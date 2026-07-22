from __future__ import annotations

import os
from pathlib import Path

from longrun_agent.exceptions import WorkspaceSecurityError
from longrun_agent.tools.workspace_policy import WorkspaceAccessPolicy


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

    policy = WorkspaceAccessPolicy.for_workspace(ensure_workspace_root(workspace_root))
    if allow_create:
        return policy.resolve_write(requested_path)
    return policy.resolve_read(requested_path, must_exist=must_exist)


def relative_to_workspace(workspace_root: str | Path, path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(workspace_root).resolve())).replace("\\", "/")
