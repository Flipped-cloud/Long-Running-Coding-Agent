from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from longrun_agent.verification.runner import VerificationRunner
from longrun_agent.verification.schema import (
    CheckKind,
    CheckVisibility,
    TestCandidate,
    TestTransition,
    VerificationCheck,
    VerificationContract,
)
from longrun_agent.verification.snapshot import CopySnapshotProvider
from longrun_agent.verification.transitions import compute_transition


class TestCandidateRegistrationError(ValueError):
    pass


def register_test_candidate(
    *,
    workspace: Path,
    task_id: str,
    session_id: str,
    paths: list[str],
    command_argv: list[str],
    issue_behavior: str,
    expected_failure_reason: str,
    contract: VerificationContract | None = None,
) -> TestCandidate:
    protected = contract.integrity_rules.protected_paths if contract else []
    normalized = []
    for raw in paths:
        relative = PurePosixPath(raw.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise TestCandidateRegistrationError("test candidate path must stay inside workspace")
        path = (workspace / Path(*relative.parts)).resolve()
        root = workspace.resolve()
        if path != root and root not in path.parents:
            raise TestCandidateRegistrationError("test candidate path escapes workspace")
        if not path.is_file():
            raise TestCandidateRegistrationError(f"test candidate path does not exist: {raw}")
        if any(relative.match(pattern) for pattern in protected):
            raise TestCandidateRegistrationError(f"test candidate references protected path: {raw}")
        normalized.append(relative.as_posix())
    return TestCandidate(
        task_id=task_id,
        session_id=session_id,
        paths=normalized,
        command_argv=command_argv,
        issue_behavior=issue_behavior,
        expected_failure_reason=expected_failure_reason,
    )


class TestCandidateValidator:
    def __init__(self, snapshot_manager: CopySnapshotProvider, runner: VerificationRunner):
        self.snapshot_manager = snapshot_manager
        self.runner = runner

    def validate(self, candidate: TestCandidate, candidate_snapshot: Path) -> TestCandidate:
        baseline = self.snapshot_manager.create_baseline_working_copy()
        try:
            for relative in candidate.paths:
                source = candidate_snapshot / relative
                target = baseline / relative
                if not source.is_file():
                    candidate.rejection_reasons.append(f"candidate test missing: {relative}")
                    return candidate
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            check = VerificationCheck(
                check_id=candidate.candidate_id,
                title="Agent-generated test candidate",
                kind=CheckKind.GENERATED_TEST,
                visibility=CheckVisibility.PUBLIC,
                argv=candidate.command_argv,
            )
            candidate.baseline_result = self.runner.run_check(check, baseline, "baseline")
            candidate.candidate_result = self.runner.run_check(check, candidate_snapshot, "candidate")
            candidate.transition = compute_transition(candidate.baseline_result, candidate.candidate_result)
            if candidate.transition == TestTransition.F2P:
                candidate.valid = True
            elif candidate.transition == TestTransition.P2P:
                candidate.valid = True
                candidate.valid_but_irrelevant = True
                candidate.rejection_reasons.append("valid_but_irrelevant")
            elif candidate.transition == TestTransition.P2F:
                candidate.rejection_reasons.append("harmful_test")
            else:
                candidate.rejection_reasons.append("invalid_or_unfixed")
            return candidate
        finally:
            self.snapshot_manager.cleanup(baseline)
