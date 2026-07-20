from __future__ import annotations

from longrun_agent.verification.schema import CheckVisibility, VerificationContract, VerificationReport


def render_contract(contract: VerificationContract, *, include_hidden: bool = False) -> dict:
    payload = contract.model_dump(mode="json")
    if not include_hidden:
        payload["hidden_assets_root"] = None
        payload["checks"] = [check.model_dump(mode="json") for check in contract.checks if check.visibility == CheckVisibility.PUBLIC]
    return payload


def render_agent_feedback(report: VerificationReport) -> str:
    parts = [f"Verification report: {report.report_id}"]
    if report.summary.required_checks_failed:
        parts.append(f"Public or hidden acceptance categories still failing: {report.summary.required_checks_failed}.")
    if report.summary.regression_passed < report.summary.regression_total:
        parts.append("A regression category failed.")
    if not report.summary.integrity_passed:
        parts.append("Workspace integrity requirements were violated.")
    if report.verdict.value == "infrastructure_error":
        parts.append("Independent verification infrastructure was unavailable; implementation failure was not inferred.")
    parts.append("Inspect the public component named in the task specification and request completion again after repair.")
    return " ".join(parts)
