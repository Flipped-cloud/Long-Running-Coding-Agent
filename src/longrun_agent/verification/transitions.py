from __future__ import annotations

from longrun_agent.verification.schema import (
    CheckExecutionResult,
    CheckKind,
    CheckTransitionResult,
    ExecutionStatus,
    ExpectedState,
    IntegrityViolation,
    TestTransition,
    VerificationCheck,
    VerificationSummary,
    VerificationVerdict,
)


def compute_transition(baseline: CheckExecutionResult | None, candidate: CheckExecutionResult | None) -> TestTransition:
    left = _state_letter(baseline)
    right = _state_letter(candidate)
    return TestTransition(f"{left}2{right}")


def build_transitions(
    checks: list[VerificationCheck],
    baseline_results: list[CheckExecutionResult],
    candidate_results: list[CheckExecutionResult],
) -> list[CheckTransitionResult]:
    baseline = {result.check_id: result for result in baseline_results}
    candidate = {result.check_id: result for result in candidate_results}
    return [
        CheckTransitionResult(
            check_id=check.check_id,
            kind=check.kind,
            required=check.required,
            transition=compute_transition(baseline.get(check.check_id), candidate.get(check.check_id)),
            baseline_status=baseline.get(check.check_id).status if check.check_id in baseline else ExecutionStatus.SKIPPED,
            candidate_status=candidate.get(check.check_id).status if check.check_id in candidate else ExecutionStatus.SKIPPED,
        )
        for check in checks
        if check.kind != CheckKind.INTEGRITY
    ]


def summarize_and_decide(
    checks: list[VerificationCheck],
    transitions: list[CheckTransitionResult],
    integrity_violations: list[IntegrityViolation],
    baseline_results: list[CheckExecutionResult],
    candidate_results: list[CheckExecutionResult],
) -> tuple[VerificationSummary, VerificationVerdict]:
    by_id = {item.check_id: item for item in transitions}
    infrastructure = [result for result in [*baseline_results, *candidate_results] if result.infrastructure_error]
    resolution = [check for check in checks if check.required and check.kind == CheckKind.RESOLUTION]
    regression = [check for check in checks if check.required and check.kind == CheckKind.REGRESSION]
    candidate_only = [check for check in checks if check.required and check.kind in {CheckKind.CANDIDATE_ONLY, CheckKind.STATIC}]
    resolution_passed = sum(
        by_id.get(check.check_id) is not None and by_id[check.check_id].transition == TestTransition.F2P for check in resolution
    )
    regression_passed = sum(
        by_id.get(check.check_id) is not None and by_id[check.check_id].transition == TestTransition.P2P for check in regression
    )
    candidate_only_passed = sum(
        by_id.get(check.check_id) is not None and by_id[check.check_id].candidate_status == ExecutionStatus.PASSED
        for check in candidate_only
    )
    integrity_passed = not any(item.severity in {"error", "critical"} for item in integrity_violations)
    required_total = len(resolution) + len(regression) + len(candidate_only)
    required_passed = resolution_passed + regression_passed + candidate_only_passed
    summary = VerificationSummary(
        resolution_total=len(resolution),
        resolution_passed=resolution_passed,
        f2p_rate=resolution_passed / len(resolution) if resolution else 1.0,
        regression_total=len(regression),
        regression_passed=regression_passed,
        p2p_rate=regression_passed / len(regression) if regression else 1.0,
        required_checks_passed=required_passed,
        required_checks_failed=required_total - required_passed,
        partial_resolution=0 < resolution_passed < len(resolution),
        integrity_passed=integrity_passed,
        hidden_checks_passed=sum(
            result.status == ExecutionStatus.PASSED for result in candidate_results if result.visibility.value == "hidden"
        ),
        hidden_checks_failed=sum(
            result.status != ExecutionStatus.PASSED for result in candidate_results if result.visibility.value == "hidden"
        ),
    )
    if infrastructure:
        return summary, VerificationVerdict.INFRASTRUCTURE_ERROR
    if not baseline_matches_contract(checks, baseline_results):
        return summary, VerificationVerdict.INCONCLUSIVE
    if required_total == 0:
        return summary, VerificationVerdict.INCONCLUSIVE
    if required_passed == required_total and integrity_passed:
        return summary, VerificationVerdict.VERIFIED
    if summary.partial_resolution and regression_passed == len(regression) and integrity_passed:
        return summary, VerificationVerdict.PARTIAL
    return summary, VerificationVerdict.REOPENED


def _state_letter(result: CheckExecutionResult | None) -> str:
    if result is None or result.status in {ExecutionStatus.ERROR, ExecutionStatus.SKIPPED, ExecutionStatus.TIMEOUT}:
        return "X"
    return "P" if result.status == ExecutionStatus.PASSED else "F"


def baseline_matches_contract(checks: list[VerificationCheck], results: list[CheckExecutionResult]) -> bool:
    by_id = {item.check_id: item for item in results}
    for check in checks:
        if check.baseline_expected in {ExpectedState.ANY, ExpectedState.NOT_RUN}:
            continue
        result = by_id.get(check.check_id)
        if result is None:
            return False
        expected_pass = check.baseline_expected == ExpectedState.PASS
        if (result.status == ExecutionStatus.PASSED) != expected_pass:
            return False
    return True
