from __future__ import annotations

from datetime import datetime

from longrun_agent.evaluation.schema import ProgressSnapshot, VerificationMilestone
from longrun_agent.verification.schema import VerificationReport


def progress_snapshot(
    report: VerificationReport,
    milestones: list[VerificationMilestone],
    *,
    project_session: int = 0,
    step: int = 0,
) -> ProgressSnapshot:
    passing = {
        item.check_id for item in report.transitions if item.transition.value in {"F2P", "P2P"} or item.candidate_status.value == "passed"
    }
    passed = [milestone.milestone_id for milestone in milestones if milestone.check_ids and set(milestone.check_ids) <= passing]
    failed = [milestone.milestone_id for milestone in milestones if milestone.milestone_id not in passed]
    total_weight = sum(item.weight for item in milestones)
    passed_weight = sum(item.weight for item in milestones if item.milestone_id in passed)
    return ProgressSnapshot(
        project_session=project_session,
        step=step,
        passed_milestones=passed,
        failed_milestones=failed,
        score=passed_weight / total_weight if total_weight else 0.0,
        source_report_id=report.report_id,
    )


def progress_metrics(snapshots: list[ProgressSnapshot]) -> dict[str, float | int | None]:
    if not snapshots:
        return {
            "final_progress": 0.0,
            "maximum_progress": 0.0,
            "time_to_first_progress": None,
            "progress_auc": 0.0,
            "regression_count": 0,
            "longest_no_progress_interval": 0.0,
        }
    ordered = sorted(snapshots, key=lambda item: item.timestamp)
    start = _timestamp(ordered[0].timestamp)
    first_progress = next((_timestamp(item.timestamp) for item in ordered if item.score > 0), None)
    auc = 0.0
    longest = 0.0
    regressions = 0
    for previous, current in zip(ordered, ordered[1:], strict=False):
        interval = max(0.0, (_timestamp(current.timestamp) - _timestamp(previous.timestamp)).total_seconds())
        auc += interval * (previous.score + current.score) / 2
        if current.score < previous.score:
            regressions += 1
        if current.score == previous.score:
            longest = max(longest, interval)
    duration = max(0.0, (_timestamp(ordered[-1].timestamp) - start).total_seconds())
    return {
        "final_progress": ordered[-1].score,
        "maximum_progress": max(item.score for item in ordered),
        "time_to_first_progress": (first_progress - start).total_seconds() if first_progress else None,
        "progress_auc": auc / duration if duration else ordered[-1].score,
        "regression_count": regressions,
        "longest_no_progress_interval": longest,
    }


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)
