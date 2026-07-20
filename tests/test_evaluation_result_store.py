from __future__ import annotations

import os
from pathlib import Path

import pytest

from longrun_agent.evaluation import reporting
from longrun_agent.evaluation.reporting import (
    append_trial_attempt,
    normalize_trial_result_store,
    read_trial_attempts,
    read_trial_results,
    upsert_trial_result,
    write_evaluation_report,
)
from longrun_agent.evaluation.schema import TrialDescriptor, TrialResult, TrialStatus


def _result(trial_id: str, status: TrialStatus, *, error: str | None = None, marker: str = "") -> TrialResult:
    return TrialResult(
        descriptor=TrialDescriptor(
            evaluation_id="eval",
            case_id="case",
            config_id="config",
            trial_id=trial_id,
            trial_number=1,
            seed=0,
            trial_dir=Path("trials") / trial_id,
            status=status,
        ),
        error=error,
        metadata={"marker": marker},
    )


def _record(results_path: Path, attempts_path: Path, result: TrialResult, retry_reason: str | None = None) -> None:
    append_trial_attempt(
        attempts_path,
        result,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        retry_reason=retry_reason,
    )
    upsert_trial_result(results_path, result)


def test_error_then_completed_upserts_canonical_result_and_keeps_attempts(tmp_path: Path) -> None:
    results_path = tmp_path / "trials.jsonl"
    attempts_path = tmp_path / "trial_attempts.jsonl"

    _record(results_path, attempts_path, _result("trial-a", TrialStatus.ERROR, error="first"))
    _record(results_path, attempts_path, _result("trial-a", TrialStatus.COMPLETED, marker="latest"), "retry_after_error")

    results = read_trial_results(results_path)
    attempts = read_trial_attempts(attempts_path)
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 1
    assert results[0].descriptor.status == TrialStatus.COMPLETED
    assert results[0].metadata["marker"] == "latest"
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert attempts[1].retry_reason == "retry_after_error"


def test_four_failures_then_success_keeps_one_result_and_five_attempts(tmp_path: Path) -> None:
    results_path = tmp_path / "trials.jsonl"
    attempts_path = tmp_path / "trial_attempts.jsonl"

    for number in range(1, 5):
        _record(
            results_path,
            attempts_path,
            _result("trial-a", TrialStatus.ERROR, error=f"failure-{number}"),
            "retry_after_error" if number > 1 else None,
        )
    _record(results_path, attempts_path, _result("trial-a", TrialStatus.COMPLETED), "retry_after_error")

    assert len(read_trial_results(results_path)) == 1
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(read_trial_attempts(attempts_path)) == 5


def test_upsert_preserves_stable_order_for_different_trials(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    upsert_trial_result(path, _result("trial-a", TrialStatus.ERROR, error="old"))
    upsert_trial_result(path, _result("trial-b", TrialStatus.COMPLETED))
    upsert_trial_result(path, _result("trial-a", TrialStatus.COMPLETED, marker="new"))

    results = read_trial_results(path)
    assert [result.descriptor.trial_id for result in results] == ["trial-a", "trial-b"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert results[0].metadata["marker"] == "new"


def test_legacy_duplicate_results_load_latest_and_migrate_attempts(tmp_path: Path) -> None:
    results_path = tmp_path / "trials.jsonl"
    attempts_path = tmp_path / "trial_attempts.jsonl"
    rows = [
        _result("trial-a", TrialStatus.ERROR, error="first", marker="first"),
        _result("trial-b", TrialStatus.COMPLETED),
        _result("trial-a", TrialStatus.COMPLETED, marker="last"),
    ]
    results_path.write_text("".join(result.model_dump_json() + "\n" for result in rows), encoding="utf-8")

    loaded = read_trial_results(results_path)
    assert [result.descriptor.trial_id for result in loaded] == ["trial-a", "trial-b"]
    assert loaded[0].metadata["marker"] == "last"

    normalized = normalize_trial_result_store(results_path, attempts_path)
    attempts = read_trial_attempts(attempts_path)
    assert len(normalized) == 2
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(attempts) == 3
    assert [attempt.attempt_number for attempt in attempts if attempt.trial_id == "trial-a"] == [1, 2]
    assert attempts[-1].retry_reason == "legacy_retry"

    normalize_trial_result_store(results_path, attempts_path)
    assert len(read_trial_attempts(attempts_path)) == 3


def test_report_counts_only_latest_trial_state(tmp_path: Path) -> None:
    old = _result("trial-a", TrialStatus.ERROR, error="old")
    latest = _result("trial-a", TrialStatus.COMPLETED)
    other = _result("trial-b", TrialStatus.ERROR, error="current")

    report = write_evaluation_report(tmp_path, [old, latest, other])

    assert report["trial_count"] == 2
    assert report["completed_count"] == 1
    assert report["error_count"] == 1


def test_upsert_uses_temporary_file_and_os_replace(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "trials.jsonl"
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", replace)
    upsert_trial_result(path, _result("trial-a", TrialStatus.COMPLETED))

    assert len(calls) == 1
    source, destination = calls[0]
    assert source.parent == path.parent
    assert source != path
    assert destination == path
    assert not source.exists()


def test_corrupt_jsonl_fails_without_overwriting_existing_data(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    original = '{"descriptor": broken}\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid evaluation JSONL.*trials\.jsonl:1"):
        upsert_trial_result(path, _result("trial-a", TrialStatus.COMPLETED))

    assert path.read_text(encoding="utf-8") == original
