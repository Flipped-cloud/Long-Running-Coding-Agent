from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from longrun_agent.exceptions import WorkspaceSecurityError

ACCESS_DENIED_MESSAGE = "The requested path is outside the agent workspace."


class WorkspaceAccessDenied(WorkspaceSecurityError):
    pass


@dataclass
class WorkspaceAccessPolicy:
    workspace_root: Path
    readable_roots: list[Path] = field(default_factory=list)
    writable_roots: list[Path] = field(default_factory=list)
    denied_roots: list[Path] = field(default_factory=list)
    allow_absolute_paths: bool = False
    evaluation_isolation_enabled: bool = False
    private_markers: set[str] = field(default_factory=set)
    private_audit_path: Path | None = None

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()
        self.readable_roots = [path.resolve() for path in (self.readable_roots or [self.workspace_root])]
        self.writable_roots = [path.resolve() for path in (self.writable_roots or [self.workspace_root])]
        self.denied_roots = [path.resolve() for path in self.denied_roots]

    @classmethod
    def for_workspace(
        cls,
        workspace_root: Path,
        *,
        evaluation_isolation_enabled: bool = False,
        denied_roots: list[Path] | None = None,
        private_markers: set[str] | None = None,
        private_audit_path: Path | None = None,
    ) -> WorkspaceAccessPolicy:
        return cls(
            workspace_root=workspace_root,
            denied_roots=denied_roots or [],
            evaluation_isolation_enabled=evaluation_isolation_enabled,
            private_markers=private_markers or set(),
            private_audit_path=private_audit_path,
        )

    def record_private_block(self, *, tool_name: str, categories: list[str]) -> None:
        if self.private_audit_path is None:
            return
        self.private_audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event_type": "agent_visible_private_data_blocked",
            "timestamp": datetime.now(UTC).isoformat(),
            "tool_name": tool_name,
            "marker_categories": sorted(set(categories)),
        }
        with self.private_audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def resolve_read(self, requested: str, *, base: Path | None = None, must_exist: bool = False) -> Path:
        return self._resolve(requested, roots=self.readable_roots, base=base, must_exist=must_exist)

    def resolve_write(self, requested: str, *, base: Path | None = None) -> Path:
        return self._resolve(requested, roots=self.writable_roots, base=base, must_exist=False)

    def resolve_cwd(self, requested: str) -> Path:
        resolved = self.resolve_read(requested or ".", base=self.workspace_root, must_exist=True)
        if not resolved.is_dir():
            raise WorkspaceAccessDenied(ACCESS_DENIED_MESSAGE)
        return resolved

    def _resolve(
        self,
        requested: str,
        *,
        roots: list[Path],
        base: Path | None,
        must_exist: bool,
    ) -> Path:
        raw = requested.strip()
        if not raw or raw.lower().startswith("file://"):
            raise WorkspaceAccessDenied(ACCESS_DENIED_MESSAGE)
        if _is_absolute(raw) and not self.allow_absolute_paths:
            raise WorkspaceAccessDenied(ACCESS_DENIED_MESSAGE)
        if _contains_parent_escape(raw):
            raise WorkspaceAccessDenied(ACCESS_DENIED_MESSAGE)
        candidate = Path(raw) if _is_absolute(raw) else (base or self.workspace_root) / Path(raw)
        resolved = candidate.resolve(strict=False)
        if self._is_denied(resolved) or not any(_is_within(resolved, root) for root in roots):
            raise WorkspaceAccessDenied(ACCESS_DENIED_MESSAGE)
        if must_exist and not resolved.exists():
            raise FileNotFoundError(raw)
        self._reject_symlink_escape(candidate, roots)
        return resolved

    def _is_denied(self, path: Path) -> bool:
        return any(_is_within(path, root) for root in self.denied_roots)

    def _reject_symlink_escape(self, candidate: Path, roots: list[Path]) -> None:
        current = candidate
        while current != current.parent:
            if current.exists() and current.is_symlink():
                target = current.resolve()
                if self._is_denied(target) or not any(_is_within(target, root) for root in roots):
                    raise WorkspaceAccessDenied(ACCESS_DENIED_MESSAGE)
            if current == self.workspace_root:
                break
            current = current.parent


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PurePosixPath(value.replace("\\", "/")).is_absolute() or PureWindowsPath(value).is_absolute()


def _contains_parent_escape(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return ".." in PurePosixPath(normalized).parts


def sensitive_environment() -> dict[str, str]:
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "LANG", "LC_ALL", "PYTHONPATH"}
    return {name: value for name, value in os.environ.items() if name.upper() in allowed}
