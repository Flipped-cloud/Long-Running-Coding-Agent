from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import uuid
from fnmatch import fnmatch
from pathlib import Path

from longrun_agent.exceptions import ConfigurationError
from longrun_agent.verification.schema import FileManifestEntry, WorkspaceManifest

DEFAULT_EXCLUDES = {".git", ".runs", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


class SnapshotError(RuntimeError):
    pass


class CopySnapshotProvider:
    def __init__(self, workspace_root: Path, store_root: Path, *, cache_patterns: list[str] | None = None):
        self.workspace_root = workspace_root.resolve()
        self.store_root = store_root.resolve()
        if self.store_root == self.workspace_root or self.workspace_root in self.store_root.parents:
            raise ConfigurationError("snapshot store must be outside workspace")
        self.cache_patterns = cache_patterns or []

    @property
    def baseline_snapshot(self) -> Path:
        return self.store_root / "baseline" / "snapshot"

    @property
    def baseline_manifest_path(self) -> Path:
        return self.store_root / "baseline" / "manifest.json"

    def create_baseline(self) -> WorkspaceManifest:
        if self.baseline_snapshot.exists():
            shutil.rmtree(self.baseline_snapshot)
        self.baseline_snapshot.parent.mkdir(parents=True, exist_ok=True)
        self._copy_tree(self.workspace_root, self.baseline_snapshot)
        manifest = self.build_manifest(self.baseline_snapshot, "baseline")
        self.baseline_manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    def load_baseline_manifest(self) -> WorkspaceManifest:
        return WorkspaceManifest.model_validate_json(self.baseline_manifest_path.read_text(encoding="utf-8"))

    def create_candidate(self) -> tuple[Path, WorkspaceManifest]:
        destination = self.store_root / "c" / uuid.uuid4().hex[:12] / "s"
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._copy_tree(self.workspace_root, destination)
        return destination, self.build_manifest(destination, "candidate")

    def create_baseline_working_copy(self) -> Path:
        if not self.baseline_snapshot.exists():
            raise SnapshotError("baseline snapshot does not exist")
        destination = self.store_root / "r" / uuid.uuid4().hex[:12] / "s"
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._copy_tree(self.baseline_snapshot, destination)
        return destination

    def inject_hidden_assets(self, snapshot_root: Path, hidden_assets_root: Path) -> None:
        hidden_root = hidden_assets_root.resolve()
        if hidden_root == self.workspace_root or self.workspace_root in hidden_root.parents:
            raise SnapshotError("hidden assets root must be outside workspace")
        self._copy_tree(hidden_root, snapshot_root, include_root_contents=True)

    def cleanup(self, snapshot_root: Path) -> None:
        resolved = snapshot_root.resolve()
        if self.store_root not in resolved.parents:
            raise SnapshotError("refusing to clean snapshot outside verification store")
        shutil.rmtree(resolved.parent, ignore_errors=True)

    def build_manifest(self, root: Path, kind: str) -> WorkspaceManifest:
        entries = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            mode = stat.S_IMODE(path.stat().st_mode)
            entries.append(FileManifestEntry(relative_path=relative, sha256=digest, size=path.stat().st_size, mode=mode))
        canonical = json.dumps([item.model_dump(mode="json") for item in entries], sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return WorkspaceManifest(workspace_kind=kind, root=str(root), fingerprint=fingerprint, files=entries)

    def _copy_tree(self, source: Path, destination: Path, *, include_root_contents: bool = True) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if self._excluded(relative):
                continue
            target = destination / relative if include_root_contents else destination / source.name / relative
            if path.is_symlink():
                self._validate_symlink(path, source)
                continue
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def _excluded(self, relative: Path) -> bool:
        parts = set(relative.parts)
        if parts & DEFAULT_EXCLUDES:
            return True
        if any(part == ".env" or part.startswith(".env.") for part in relative.parts):
            return True
        normalized = relative.as_posix()
        return any(fnmatch(normalized, pattern) for pattern in self.cache_patterns)

    @staticmethod
    def _validate_symlink(path: Path, source: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise SnapshotError(f"invalid symlink in workspace: {path}") from exc
        if resolved != source and source not in resolved.parents:
            raise SnapshotError(f"symlink escapes workspace: {path}")


class SnapshotManager(CopySnapshotProvider):
    pass


class GitWorktreeSnapshotProvider(CopySnapshotProvider):
    def create_baseline(self) -> WorkspaceManifest:
        repository_root = self._git(["rev-parse", "--show-toplevel"], "Git repository root").strip()
        if Path(repository_root).resolve() != self.workspace_root:
            raise SnapshotError("git_worktree isolation requires workspace to be the Git repository root")
        baseline_commit = self._git(["rev-parse", "HEAD"], "baseline commit").strip()
        if not baseline_commit:
            raise SnapshotError("git_worktree isolation requires a resolvable baseline commit")
        dirty_entries = [
            line for line in self._git(["status", "--porcelain=v1", "--untracked-files=all"], "dirty state").splitlines() if line
        ]
        manifest = super().create_baseline()
        state_path = self.store_root / "baseline" / "git_state.json"
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "repository_root": str(self.workspace_root),
                    "baseline_commit": baseline_commit,
                    "dirty": bool(dirty_entries),
                    "dirty_entries": dirty_entries,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, state_path)
        return manifest

    def _git(self, arguments: list[str], label: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=self.workspace_root,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise SnapshotError(f"git_worktree isolation could not inspect {label}") from exc
        if result.returncode != 0:
            raise SnapshotError(f"git_worktree isolation could not resolve {label}")
        return result.stdout
