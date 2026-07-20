from __future__ import annotations

from longrun_agent.evaluation.progress import progress_metrics
from longrun_agent.evaluation.schema import ProgressSnapshot


def test_progress_auc_and_regression_are_evidence_derived() -> None:
    snapshots = [
        ProgressSnapshot(timestamp="2026-01-01T00:00:00+00:00", score=0.0, source_report_id="r1"),
        ProgressSnapshot(timestamp="2026-01-01T00:00:10+00:00", score=1.0, source_report_id="r2"),
        ProgressSnapshot(timestamp="2026-01-01T00:00:20+00:00", score=0.5, source_report_id="r3"),
    ]
    metrics = progress_metrics(snapshots)
    assert metrics["final_progress"] == 0.5
    assert metrics["maximum_progress"] == 1.0
    assert metrics["time_to_first_progress"] == 10
    assert metrics["progress_auc"] == 0.625
    assert metrics["regression_count"] == 1
