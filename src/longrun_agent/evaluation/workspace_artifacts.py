from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

from longrun_agent.verification.snapshot import CopySnapshotProvider


def preserve_final_workspace_artifacts(trial_dir: Path, workspace: Path) -> list[Path]:
    snapshots = CopySnapshotProvider(workspace, trial_dir / "oracle" / "snapshots")
    baseline_manifest = snapshots.load_baseline_manifest()
    candidate_root, final_manifest = snapshots.create_candidate()
    try:
        baseline_files = {item.relative_path: item for item in baseline_manifest.files}
        final_files = {item.relative_path: item for item in final_manifest.files}
        changed = []
        for relative in sorted(set(baseline_files) | set(final_files)):
            before = baseline_files.get(relative)
            after = final_files.get(relative)
            if before is not None and after is not None and before.sha256 == after.sha256:
                continue
            changed.append(
                {
                    "path": relative,
                    "status": "added" if before is None else ("deleted" if after is None else "modified"),
                    "before_sha256": before.sha256 if before else None,
                    "after_sha256": after.sha256 if after else None,
                }
            )

        artifacts = trial_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        changed_path = artifacts / "changed_files.json"
        changed_path.write_text(json.dumps(changed, indent=2, sort_keys=True), encoding="utf-8")
        fingerprint_path = artifacts / "final_workspace_fingerprint.json"
        fingerprint_path.write_text(
            json.dumps(
                {
                    "fingerprint": final_manifest.fingerprint,
                    "files": [item.model_dump(mode="json") for item in final_manifest.files],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        patch_path = artifacts / "final_workspace_diff.patch"
        patch_path.write_text(
            _redact_environment_secrets(_render_patch(snapshots.baseline_snapshot, candidate_root, changed)),
            encoding="utf-8",
        )
        return [patch_path, changed_path, fingerprint_path]
    finally:
        snapshots.cleanup(candidate_root)


def _render_patch(baseline: Path, candidate: Path, changed: list[dict[str, str | None]]) -> str:
    chunks: list[str] = []
    for item in changed:
        relative = str(item["path"])
        before = _read_text(baseline / relative)
        after = _read_text(candidate / relative)
        if before is None or after is None:
            chunks.append(f"Binary files a/{relative} and b/{relative} differ\n")
            continue
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _redact_environment_secrets(text: str) -> str:
    values = {
        value
        for name, value in os.environ.items()
        if value and len(value) >= 8 and any(marker in name.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "[redacted]")
    return text
