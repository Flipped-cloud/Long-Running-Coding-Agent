from __future__ import annotations

from longrun_agent.state.schema import utc_now
from longrun_agent.verification.schema import (
    CheckExecutionResult,
    CheckKind,
    CheckVisibility,
    ExecutionStatus,
    IntegrityViolation,
    VerificationCheck,
    VerificationVerdict,
)
from longrun_agent.verification.schema import (
    TestTransition as Transition,
)
from longrun_agent.verification.transitions import build_transitions, compute_transition, summarize_and_decide


def _result(check_id: str, kind: CheckKind, status: ExecutionStatus, workspace: str):
    return CheckExecutionResult(
        check_id=check_id,
        kind=kind,
        visibility=CheckVisibility.PUBLIC,
        workspace_kind=workspace,
        started_at=utc_now(),
        finished_at=utc_now(),
        duration_seconds=0,
        status=status,
    )


def test_transition_matrix() -> None:
    failed = _result("r", CheckKind.RESOLUTION, ExecutionStatus.FAILED, "baseline")
    passed = _result("r", CheckKind.RESOLUTION, ExecutionStatus.PASSED, "candidate")
    assert compute_transition(failed, passed) == Transition.F2P
    assert compute_transition(failed, failed.model_copy(update={"workspace_kind": "candidate"})) == Transition.F2F
    assert compute_transition(passed.model_copy(update={"workspace_kind": "baseline"}), passed) == Transition.P2P
    assert compute_transition(passed.model_copy(update={"workspace_kind": "baseline"}), failed) == Transition.P2F


def test_verdicts_verified_partial_reopened_and_infrastructure() -> None:
    checks = [
        VerificationCheck(check_id="r1", title="r1", kind=CheckKind.RESOLUTION, argv=["pytest"]),
        VerificationCheck(check_id="r2", title="r2", kind=CheckKind.RESOLUTION, argv=["pytest"]),
        VerificationCheck(check_id="g", title="g", kind=CheckKind.REGRESSION, argv=["pytest"]),
    ]
    baseline = [
        _result("r1", CheckKind.RESOLUTION, ExecutionStatus.FAILED, "baseline"),
        _result("r2", CheckKind.RESOLUTION, ExecutionStatus.FAILED, "baseline"),
        _result("g", CheckKind.REGRESSION, ExecutionStatus.PASSED, "baseline"),
    ]
    candidate = [
        _result("r1", CheckKind.RESOLUTION, ExecutionStatus.PASSED, "candidate"),
        _result("r2", CheckKind.RESOLUTION, ExecutionStatus.PASSED, "candidate"),
        _result("g", CheckKind.REGRESSION, ExecutionStatus.PASSED, "candidate"),
    ]
    transitions = build_transitions(checks, baseline, candidate)
    summary, verdict = summarize_and_decide(checks, transitions, [], baseline, candidate)
    assert verdict == VerificationVerdict.VERIFIED
    assert summary.f2p_rate == summary.p2p_rate == 1

    candidate[1] = _result("r2", CheckKind.RESOLUTION, ExecutionStatus.FAILED, "candidate")
    summary, verdict = summarize_and_decide(checks, build_transitions(checks, baseline, candidate), [], baseline, candidate)
    assert verdict == VerificationVerdict.PARTIAL

    violation = IntegrityViolation(category="tampering", description="tampering", agent_visible_summary="integrity failed")
    _, verdict = summarize_and_decide(checks, build_transitions(checks, baseline, candidate), [violation], baseline, candidate)
    assert verdict == VerificationVerdict.REOPENED

    candidate[0].infrastructure_error = "missing dependency"
    _, verdict = summarize_and_decide(checks, build_transitions(checks, baseline, candidate), [], baseline, candidate)
    assert verdict == VerificationVerdict.INFRASTRUCTURE_ERROR

    candidate[0].infrastructure_error = None
    mismatched_baseline = list(baseline)
    mismatched_baseline[0] = _result("r1", CheckKind.RESOLUTION, ExecutionStatus.PASSED, "baseline")
    _, verdict = summarize_and_decide(
        checks,
        build_transitions(checks, mismatched_baseline, candidate),
        [],
        mismatched_baseline,
        candidate,
    )
    assert verdict == VerificationVerdict.INCONCLUSIVE
