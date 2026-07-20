from __future__ import annotations

import sys
from pathlib import Path

from longrun_agent.verification.runner import VerificationRunner
from longrun_agent.verification.schema import CheckKind, ExecutionStatus, VerificationCheck


def test_runner_uses_argv_captures_artifacts_and_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = VerificationRunner(tmp_path / "artifacts")
    passed = VerificationCheck(
        check_id="pass",
        title="pass",
        kind=CheckKind.CANDIDATE_ONLY,
        argv=[sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
    )
    result = runner.run_check(passed, workspace, "candidate")
    assert result.status == ExecutionStatus.PASSED
    assert Path(result.stdout_artifact or "").read_text(encoding="utf-8").strip() == "out"
    assert Path(result.stderr_artifact or "").read_text(encoding="utf-8").strip() == "err"

    timeout = VerificationCheck(
        check_id="timeout",
        title="timeout",
        kind=CheckKind.CANDIDATE_ONLY,
        argv=[sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=1,
    )
    timed_out = runner.run_check(timeout, workspace, "candidate")
    assert timed_out.status == ExecutionStatus.TIMEOUT
    assert timed_out.infrastructure_error


def test_runner_classifies_missing_command_as_infrastructure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    check = VerificationCheck(check_id="missing", title="missing", kind=CheckKind.CANDIDATE_ONLY, argv=["missing-v05-command"])
    result = VerificationRunner(tmp_path / "artifacts").run_check(check, workspace, "candidate")
    assert result.status == ExecutionStatus.ERROR
    assert "could not start" in (result.infrastructure_error or "")
