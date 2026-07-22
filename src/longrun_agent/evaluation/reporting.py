from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from longrun_agent.evaluation.aggregation import aggregate_results
from longrun_agent.evaluation.schema import TrialAttempt, TrialResult, TrialStatus, latest_trial_results
from longrun_agent.state.schema import utc_now


def write_evaluation_report(evaluation_dir: Path, results: list[TrialResult]) -> dict[str, Any]:
    results = latest_trial_results(results)
    aggregate = aggregate_results(results)
    payload = {
        "evaluation_id": results[0].descriptor.evaluation_id if results else evaluation_dir.name,
        "trial_count": len(results),
        "completed_count": sum(item.descriptor.status == TrialStatus.COMPLETED for item in results),
        "error_count": sum(item.descriptor.status == TrialStatus.ERROR for item in results),
        "aggregate": aggregate,
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(evaluation_dir / "report.json", json.dumps(payload, indent=2, sort_keys=True))
    return payload


def read_trial_results(path: Path) -> list[TrialResult]:
    return latest_trial_results(_read_jsonl(path, TrialResult, label="evaluation"))


def read_trial_attempts(path: Path) -> list[TrialAttempt]:
    return _read_jsonl(path, TrialAttempt, label="trial attempt")


def upsert_trial_result(path: Path, result: TrialResult) -> None:
    results = latest_trial_results([*read_trial_results(path), result])
    _write_trial_results(path, results)


def append_trial_attempt(
    path: Path,
    result: TrialResult,
    *,
    started_at: str,
    finished_at: str,
    retry_reason: str | None,
) -> TrialAttempt:
    attempts = read_trial_attempts(path)
    attempt_number = 1 + max(
        (attempt.attempt_number for attempt in attempts if attempt.trial_id == result.descriptor.trial_id),
        default=0,
    )
    attempt = _attempt_from_result(
        result,
        attempt_number=attempt_number,
        started_at=started_at,
        finished_at=finished_at,
        retry_reason=retry_reason,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(attempt.model_dump_json() + "\n")
        handle.flush()
    return attempt


def normalize_trial_result_store(results_path: Path, attempts_path: Path) -> list[TrialResult]:
    result_rows = _read_jsonl(results_path, TrialResult, label="evaluation")
    attempts = read_trial_attempts(attempts_path)
    _migrate_legacy_attempts(result_rows, attempts_path, attempts)
    canonical = latest_trial_results(result_rows)
    if len(canonical) != len(result_rows):
        _write_trial_results(results_path, canonical)
    return canonical


def _migrate_legacy_attempts(
    results: list[TrialResult],
    attempts_path: Path,
    existing_attempts: list[TrialAttempt],
) -> None:
    existing_counts = Counter((attempt.trial_id, attempt.result_fingerprint) for attempt in existing_attempts)
    observed_counts: Counter[tuple[str, str]] = Counter()
    next_numbers: dict[str, int] = defaultdict(int)
    for attempt in existing_attempts:
        next_numbers[attempt.trial_id] = max(next_numbers[attempt.trial_id], attempt.attempt_number)

    migration_time = utc_now()
    for result in results:
        trial_id = result.descriptor.trial_id
        fingerprint = _result_fingerprint(result)
        key = (trial_id, fingerprint)
        observed_counts[key] += 1
        if observed_counts[key] <= existing_counts[key]:
            continue
        next_numbers[trial_id] += 1
        append = _attempt_from_result(
            result,
            attempt_number=next_numbers[trial_id],
            started_at=str(result.metadata.get("started_at") or migration_time),
            finished_at=str(result.metadata.get("finished_at") or migration_time),
            retry_reason="legacy_retry" if next_numbers[trial_id] > 1 else None,
        )
        attempts_path.parent.mkdir(parents=True, exist_ok=True)
        with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(append.model_dump_json() + "\n")
            handle.flush()


def _attempt_from_result(
    result: TrialResult,
    *,
    attempt_number: int,
    started_at: str,
    finished_at: str,
    retry_reason: str | None,
) -> TrialAttempt:
    return TrialAttempt(
        evaluation_id=result.descriptor.evaluation_id,
        trial_id=result.descriptor.trial_id,
        attempt_number=attempt_number,
        status=result.descriptor.status,
        error=result.error,
        started_at=started_at,
        finished_at=finished_at,
        retry_reason=retry_reason,
        outcome_present=result.outcome is not None,
        result_fingerprint=_result_fingerprint(result),
        result=result,
    )


def _result_fingerprint(result: TrialResult) -> str:
    return hashlib.sha256(result.model_dump_json().encode("utf-8")).hexdigest()


def _read_jsonl(path: Path, model, *, label: str):
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid {label} JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _write_trial_results(path: Path, results: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for result in results:
                handle.write(result.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_trial_results_atomic(path: Path, results: list[TrialResult]) -> None:
    _write_trial_results(path, latest_trial_results(results))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
