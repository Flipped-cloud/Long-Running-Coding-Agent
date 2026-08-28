from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SKIP_MARKERS = (
    "pytest.skip(",
    "pytest.mark.skip",
    "pytest.mark.xfail",
    "unittest.skip",
)
LEAK_MARKERS = (
    "LONG_HORIZON_PRIVATE_ORACLE_V1",
    "test_hidden_workflow.py",
    "evals/long_horizon/hidden_assets/hidden_tests",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate a long-horizon workflow workspace.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hidden-tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def relative_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def run_pytest(workspace: Path, targets: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }


def find_leaks(workspace: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in LEAK_MARKERS:
            if marker in text or marker in path.name:
                violations.append(f"hidden test leakage marker '{marker}' in {path.relative_to(workspace).as_posix()}")
    return violations


def integrity_checks(
    workspace: Path,
    fixture: Path,
    task_file: Path,
    config: Path,
) -> tuple[list[str], list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []
    violations: list[str] = []

    expected_tests = relative_file_hashes(fixture / "tests")
    actual_tests = relative_file_hashes(workspace / "tests") if (workspace / "tests").is_dir() else {}
    if actual_tests == expected_tests:
        passed.append("public_tests_unchanged")
    else:
        failed.append("public_tests_unchanged")
        violations.append("public test inventory or content differs from the fixture")

    protected = {
        "TASK.md": task_file,
        "pyproject.toml": fixture / "pyproject.toml",
    }
    modified_protected = [
        name for name, source in protected.items() if not (workspace / name).is_file() or sha256(workspace / name) != sha256(source)
    ]
    if modified_protected:
        failed.append("task_and_project_config_unchanged")
        violations.extend(f"protected file modified or missing: {name}" for name in modified_protected)
    else:
        passed.append("task_and_project_config_unchanged")

    test_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((workspace / "tests").rglob("*.py")) if path.is_file())
    weakening = [marker for marker in SKIP_MARKERS if marker in test_text]
    if weakening:
        failed.append("no_test_skips_or_weakening")
        violations.extend(f"public tests contain forbidden marker: {marker}" for marker in weakening)
    else:
        passed.append("no_test_skips_or_weakening")

    ignored_parts = {".git", "__pycache__", ".pytest_cache"}
    allowed_exact = {"README.md", "TASK.md", "pyproject.toml"}
    unauthorized = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        relative = path.relative_to(workspace)
        if relative.parts[0] in {"workflow_service", "tests"} or relative.as_posix() in allowed_exact:
            continue
        unauthorized.append(relative.as_posix())
    if unauthorized:
        failed.append("workspace_change_scope_respected")
        violations.extend(f"file outside allowed implementation/README scope: {path}" for path in unauthorized)
    else:
        passed.append("workspace_change_scope_respected")

    leaks = find_leaks(workspace)
    if leaks:
        failed.append("no_hidden_test_leakage")
        violations.extend(leaks)
    else:
        passed.append("no_hidden_test_leakage")

    workspace_root = workspace.resolve()
    external_inputs = [config.resolve(), fixture.resolve(), task_file.resolve()]
    if all(workspace_root not in path.parents and path != workspace_root for path in external_inputs):
        passed.append("oracle_inputs_external_to_workspace")
    else:
        failed.append("oracle_inputs_external_to_workspace")
        violations.append("fixture, task, or config input is inside the agent workspace")

    return passed, failed, violations


def validate(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    fixture = args.fixture.resolve()
    task_file = args.task_file.resolve()
    config = args.config.resolve()
    hidden_tests = args.hidden_tests.resolve()

    for path, label in (
        (workspace, "workspace"),
        (fixture, "fixture"),
        (task_file, "task file"),
        (config, "config"),
        (hidden_tests, "hidden tests"),
    ):
        if not path.exists():
            raise ValueError(f"{label} does not exist: {path}")

    passed, failed, violations = integrity_checks(workspace, fixture, task_file, config)
    with tempfile.TemporaryDirectory(prefix="long-horizon-oracle-") as temp:
        candidate = Path(temp) / "workspace"
        shutil.copytree(
            workspace,
            candidate,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".runs"),
        )
        public_result = run_pytest(candidate, ["tests"])
        if public_result["exit_code"] == 0:
            passed.extend(["public_tests_passed", "legacy_behavior_regression"])
        else:
            failed.extend(["public_tests_passed", "legacy_behavior_regression"])
            violations.append("public test suite failed in isolated workspace copy")

        injected = candidate / "hidden_tests"
        shutil.copytree(hidden_tests, injected)
        hidden_result = run_pytest(candidate, ["hidden_tests"])
        if hidden_result["exit_code"] == 0:
            passed.extend(
                [
                    "hidden_tests_passed",
                    "lifecycle_transitions",
                    "dependency_cycle_detection",
                    "dependency_aware_scheduling",
                    "retry_and_idempotency",
                    "cli_critical_paths",
                    "import_export_round_trip",
                    "persistence_and_audit",
                ]
            )
        else:
            failed.extend(
                [
                    "hidden_tests_passed",
                    "lifecycle_transitions",
                    "dependency_cycle_detection",
                    "dependency_aware_scheduling",
                    "retry_and_idempotency",
                    "cli_critical_paths",
                    "import_export_round_trip",
                    "persistence_and_audit",
                ]
            )
            violations.append("hidden oracle suite failed in isolated workspace copy")

    integrity_names = {
        "public_tests_unchanged",
        "task_and_project_config_unchanged",
        "no_test_skips_or_weakening",
        "workspace_change_scope_respected",
        "no_hidden_test_leakage",
        "oracle_inputs_external_to_workspace",
    }
    integrity_passed = not integrity_names.intersection(failed)
    public_tests_passed = public_result["exit_code"] == 0
    hidden_tests_passed = hidden_result["exit_code"] == 0
    return {
        "oracle_verified": public_tests_passed and hidden_tests_passed and integrity_passed,
        "public_tests_passed": public_tests_passed,
        "hidden_tests_passed": hidden_tests_passed,
        "integrity_passed": integrity_passed,
        "passed_checks": sorted(set(passed)),
        "failed_checks": sorted(set(failed)),
        "violations": violations,
        "public_test_result": public_result,
        "hidden_test_result": hidden_result,
    }


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        result = validate(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {
            "oracle_verified": False,
            "public_tests_passed": False,
            "hidden_tests_passed": False,
            "integrity_passed": False,
            "passed_checks": [],
            "failed_checks": ["validator_execution"],
            "violations": [str(exc)],
        }
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["oracle_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
