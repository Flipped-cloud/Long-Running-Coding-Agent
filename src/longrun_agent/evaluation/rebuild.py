from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from longrun_agent.evaluation.attribution import FailureAttributor
from longrun_agent.evaluation.coordinator import _collect_events
from longrun_agent.evaluation.failure_taxonomy import termination_reason_from_status
from longrun_agent.evaluation.reporting import read_trial_results, write_evaluation_report, write_trial_results_atomic
from longrun_agent.evaluation.schema import EVALUATION_SEMANTICS_VERSION, TerminationReason, TrialStatus
from longrun_agent.evaluation.trajectory import TrajectoryFeatureExtractor


def rebuild_evaluation_report(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    trials_path = root / "trials.jsonl"
    report_path = root / "report.json"
    if not trials_path.exists():
        raise ValueError(f"evaluation trials not found: {trials_path}")
    backup = root / "backups" / f"rebuild-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(trials_path, backup / "trials.jsonl")
    if report_path.exists():
        shutil.copy2(report_path, backup / "report.json")

    rebuilt = []
    for row in read_trial_results(trials_path):
        item = row.model_copy(deep=True)
        if item.descriptor.status == TrialStatus.COMPLETED and item.outcome is not None:
            canonical_trial_dir = root / "trials" / item.descriptor.trial_id
            trial_dir = canonical_trial_dir if canonical_trial_dir.exists() else item.descriptor.trial_dir
            events = _collect_events(trial_dir)
            item.outcome.termination_reason = reconstruct_termination_reason(item.outcome, trial_dir, events)
            features = TrajectoryFeatureExtractor().extract([events])
            item.attribution = FailureAttributor().attribute(
                case_id=item.descriptor.case_id,
                trial_id=item.descriptor.trial_id,
                termination_reason=item.outcome.termination_reason,
                features=features,
                events=events,
                oracle_verification_verdict=item.outcome.oracle_verification_verdict,
                integrity_passed=item.outcome.integrity_passed,
                runtime_verification_verdict=item.outcome.runtime_verification_verdict,
            )
            item.outcome.failure_attribution_id = item.attribution.attribution_id if item.attribution else None
        elif item.descriptor.status == TrialStatus.ERROR:
            item.attribution = FailureAttributor().attribute_error(
                case_id=item.descriptor.case_id,
                trial_id=item.descriptor.trial_id,
                explanation="Existing evaluation attempt ended in a harness error.",
            )
        item.metadata["evaluation_semantics_version"] = EVALUATION_SEMANTICS_VERSION
        rebuilt.append(item)

    write_trial_results_atomic(trials_path, rebuilt)
    report = write_evaluation_report(root, rebuilt)
    report["backup_path"] = str(backup)
    return report


def reconstruct_termination_reason(outcome, trial_dir: Path, events: list[dict[str, Any]]) -> TerminationReason:
    if outcome.oracle_verification_verdict == "verified" and outcome.integrity_passed:
        return TerminationReason.COMPLETED
    statuses = []
    for path in sorted((trial_dir / "state").glob("*/sessions.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                statuses.append(str(json.loads(line).get("run_status") or ""))
    for event in events:
        if event.get("event_type") == "run_finished":
            statuses.append(str(event.get("summary") or (event.get("payload") or {}).get("status") or ""))
    for status in reversed(statuses):
        reason = termination_reason_from_status(status)
        if reason != TerminationReason.UNKNOWN:
            return reason
    event_types = {str(event.get("event_type") or "") for event in events}
    if "task_completion_requested" in event_types and outcome.runtime_verification_verdict is not None:
        return TerminationReason.COMPLETED
    return outcome.termination_reason
