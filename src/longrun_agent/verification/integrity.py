from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from longrun_agent.verification.schema import IntegrityRules, IntegrityViolation, WorkspaceManifest

TEST_CONFIGURATION_NAMES = {"pytest.ini", "tox.ini", "noxfile.py", "conftest.py", ".coveragerc"}


class IntegrityValidator:
    def validate(
        self,
        baseline: WorkspaceManifest,
        candidate: WorkspaceManifest,
        rules: IntegrityRules,
        candidate_root: Path,
    ) -> list[IntegrityViolation]:
        before = {entry.relative_path: entry for entry in baseline.files}
        after = {entry.relative_path: entry for entry in candidate.files}
        changed = {path for path in before.keys() & after.keys() if before[path].sha256 != after[path].sha256}
        deleted = set(before) - set(after)
        added = set(after) - set(before)
        violations: list[IntegrityViolation] = []
        for path in sorted(changed | deleted):
            if _matches(path, rules.protected_paths):
                violations.append(_violation("protected_path_modified", path, "A protected path was modified or deleted."))
            if _matches(path, rules.trusted_test_patterns):
                violations.append(_violation("trusted_test_modified", path, "A trusted test was modified or deleted."))
            if _matches(path, rules.contract_paths):
                violations.append(_violation("contract_modified", path, "A verification contract path was modified."))
        for path in sorted(changed | added | deleted):
            if _matches(path, rules.forbidden_change_patterns):
                violations.append(_violation("forbidden_change", path, "A forbidden change pattern was matched."))
            if rules.allowed_change_patterns and not _matches(path, rules.allowed_change_patterns):
                violations.append(_violation("outside_allowed_changes", path, "A changed path is outside allowed patterns."))
        if len(deleted) > rules.max_deleted_files:
            violations.append(_violation("excessive_deletion", "", "Too many files were deleted."))
        for required in rules.required_artifacts:
            if not (candidate_root / required).exists():
                violations.append(_violation("required_artifact_missing", required, "A required artifact is missing."))
        for marker in rules.hidden_asset_markers:
            if any(fnmatch(path, marker) for path in after):
                violations.append(_violation("hidden_asset_leak", marker, "Hidden verification assets leaked into the workspace."))
        if not rules.allow_project_config_changes:
            for path in sorted(changed | deleted):
                if Path(path).name in TEST_CONFIGURATION_NAMES:
                    violations.append(_violation("test_configuration_weakened", path, "A test configuration file was modified."))
                if Path(path).name in {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"}:
                    violations.append(_violation("project_config_modified", path, "Project configuration modification is not allowed."))
        for path in sorted(added):
            entry = after[path]
            if entry.size > rules.max_binary_file_bytes and _looks_binary(Path(candidate_root / path)):
                violations.append(_violation("large_binary_added", path, "An unexpectedly large binary file was added."))
            if _looks_like_test(path) and not any(
                path == root or path.startswith(root.rstrip("/") + "/") for root in rules.allowed_test_directories
            ):
                violations.append(_violation("test_outside_allowed_directory", path, "A generated test is outside allowed directories."))
        return violations


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch(path, pattern) or path == pattern.rstrip("/") for pattern in patterns)


def _looks_binary(path: Path) -> bool:
    try:
        return b"\0" in path.read_bytes()[:4096]
    except OSError:
        return False


def _looks_like_test(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("test_") or name.endswith("_test.py")


def _violation(category: str, path: str, summary: str) -> IntegrityViolation:
    return IntegrityViolation(category=category, path=path, description=summary, agent_visible_summary=summary)
