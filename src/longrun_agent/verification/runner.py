from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from pathlib import Path

from longrun_agent.state.schema import utc_now
from longrun_agent.verification.schema import (
    CheckExecutionResult,
    CheckVisibility,
    ExecutionStatus,
    VerificationCheck,
)


def environment_fingerprint() -> str:
    payload = f"{platform.system()}|{platform.release()}|{platform.machine()}|{platform.python_version()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VerificationRunner:
    def __init__(self, artifacts_root: Path, *, max_output_chars: int = 20000):
        self.artifacts_root = artifacts_root.resolve()
        self.max_output_chars = max_output_chars
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def run_check(self, check: VerificationCheck, workspace: Path, workspace_kind: str) -> CheckExecutionResult:
        started_at = utc_now()
        started = time.monotonic()
        cwd = (workspace / check.cwd).resolve()
        workspace = workspace.resolve()
        if cwd != workspace and workspace not in cwd.parents:
            return self._infrastructure_result(check, workspace_kind, started_at, started, "check cwd escapes snapshot")
        if not cwd.exists() or not cwd.is_dir():
            return self._infrastructure_result(check, workspace_kind, started_at, started, "check cwd does not exist")
        env = os.environ.copy()
        env.update(check.environment)
        try:
            completed = subprocess.run(
                check.argv,
                cwd=_execution_cwd(cwd),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=check.timeout_seconds,
                env=env,
            )
            exit_code = int(completed.returncode)
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            status = ExecutionStatus.PASSED if exit_code == 0 else ExecutionStatus.FAILED
            timed_out = False
            infrastructure_error = None
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            status = ExecutionStatus.TIMEOUT
            timed_out = True
            infrastructure_error = "verification check timed out"
        except OSError as exc:
            return self._infrastructure_result(check, workspace_kind, started_at, started, f"command could not start: {exc}")
        stdout_artifact, stderr_artifact = self._save_output(check, workspace_kind, stdout, stderr)
        return CheckExecutionResult(
            check_id=check.check_id,
            kind=check.kind,
            visibility=check.visibility,
            workspace_kind=workspace_kind,
            started_at=started_at,
            finished_at=utc_now(),
            duration_seconds=max(0.0, time.monotonic() - started),
            exit_code=exit_code,
            timed_out=timed_out,
            stdout_excerpt=self._excerpt(stdout),
            stderr_excerpt=self._excerpt(stderr),
            stdout_artifact=str(stdout_artifact),
            stderr_artifact=str(stderr_artifact),
            status=status,
            infrastructure_error=infrastructure_error,
            environment_fingerprint=environment_fingerprint(),
        )

    def _save_output(self, check: VerificationCheck, workspace_kind: str, stdout: str, stderr: str) -> tuple[Path, Path]:
        visibility = "private" if check.visibility == CheckVisibility.HIDDEN else "public"
        root = self.artifacts_root / visibility
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"{check.check_id}-{workspace_kind}"
        stdout_path = root / f"{prefix}-stdout.txt"
        stderr_path = root / f"{prefix}-stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return stdout_path, stderr_path

    def _excerpt(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        half = self.max_output_chars // 2
        return value[:half] + "\n...[truncated]...\n" + value[-half:]

    def _infrastructure_result(
        self,
        check: VerificationCheck,
        workspace_kind: str,
        started_at: str,
        started: float,
        error: str,
    ) -> CheckExecutionResult:
        return CheckExecutionResult(
            check_id=check.check_id,
            kind=check.kind,
            visibility=check.visibility,
            workspace_kind=workspace_kind,
            started_at=started_at,
            finished_at=utc_now(),
            duration_seconds=max(0.0, time.monotonic() - started),
            status=ExecutionStatus.ERROR,
            infrastructure_error=error,
            environment_fingerprint=environment_fingerprint(),
        )


def _execution_cwd(path: Path) -> str | Path:
    value = str(path)
    if os.name == "nt" and len(value) >= 240 and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return path
