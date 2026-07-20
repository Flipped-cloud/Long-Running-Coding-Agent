from __future__ import annotations

from pathlib import Path

from longrun_agent.verification.integrity import IntegrityValidator
from longrun_agent.verification.schema import IntegrityRules
from longrun_agent.verification.snapshot import CopySnapshotProvider


def test_integrity_validator_covers_change_boundaries_and_required_artifacts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "tests").mkdir()
    (candidate / "tests").mkdir()
    (baseline / "tests" / "test_trusted.py").write_text("assert True\n", encoding="utf-8")
    (candidate / "tests" / "test_trusted.py").write_text("assert False\n", encoding="utf-8")
    (baseline / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (candidate / "pytest.ini").write_text("[pytest]\naddopts = -k nothing\n", encoding="utf-8")
    (baseline / "required.txt").write_text("required\n", encoding="utf-8")
    (candidate / "forbidden.py").write_text("changed\n", encoding="utf-8")
    (candidate / "large.bin").write_bytes(b"\0binary")
    snapshots = CopySnapshotProvider(tmp_path / "workspace", tmp_path / "store")
    before = snapshots.build_manifest(baseline, "baseline")
    after = snapshots.build_manifest(candidate, "candidate")
    rules = IntegrityRules(
        protected_paths=["tests/test_trusted.py"],
        trusted_test_patterns=["tests/*.py"],
        forbidden_change_patterns=["forbidden.py"],
        allowed_change_patterns=["tests/*.py", "pytest.ini"],
        required_artifacts=["required.txt"],
        max_deleted_files=0,
        max_binary_file_bytes=1,
    )

    violations = IntegrityValidator().validate(before, after, rules, candidate)
    categories = {item.category for item in violations}

    assert {
        "protected_path_modified",
        "trusted_test_modified",
        "forbidden_change",
        "outside_allowed_changes",
        "excessive_deletion",
        "required_artifact_missing",
        "test_configuration_weakened",
        "large_binary_added",
    } <= categories
