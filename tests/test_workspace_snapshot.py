from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from longrun_agent.verification.snapshot import CopySnapshotProvider, GitWorktreeSnapshotProvider, SnapshotError


def test_snapshot_excludes_caches_and_fingerprints_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / ".pytest_cache").mkdir()
    (workspace / ".pytest_cache" / "cache").write_text("ignored", encoding="utf-8")
    manager = CopySnapshotProvider(workspace, tmp_path / "store")

    baseline = manager.create_baseline()
    (workspace / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    candidate_root, candidate = manager.create_candidate()

    assert [item.relative_path for item in baseline.files] == ["app.py"]
    assert baseline.fingerprint != candidate.fingerprint
    assert (candidate_root / "app.py").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unsupported")
def test_snapshot_rejects_escaping_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (workspace / "escape").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(SnapshotError, match="escapes workspace"):
        CopySnapshotProvider(workspace, tmp_path / "store").create_baseline()


def test_git_worktree_snapshot_records_commit_and_dirty_state(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 0\n", encoding="utf-8")

    def fake_run(argv, **_kwargs):
        command = argv[1:]
        if command == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(returncode=0, stdout=str(workspace), stderr="")
        if command == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        return SimpleNamespace(returncode=0, stdout=" M app.py\n", stderr="")

    monkeypatch.setattr("longrun_agent.verification.snapshot.subprocess.run", fake_run)
    snapshots = GitWorktreeSnapshotProvider(workspace, tmp_path / "store")

    manifest = snapshots.create_baseline()
    git_state = json.loads((tmp_path / "store" / "baseline" / "git_state.json").read_text(encoding="utf-8"))

    assert manifest.files[0].relative_path == "app.py"
    assert git_state["baseline_commit"] == "abc123"
    assert git_state["dirty"] is True
    assert git_state["dirty_entries"] == [" M app.py"]
