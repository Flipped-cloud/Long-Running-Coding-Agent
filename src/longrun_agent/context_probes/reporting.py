from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from longrun_agent.context_probes.schemas import ProbeCase, ProbeCaseResult, ProbeMode


def write_dry_run_report(
    *,
    output_dir: Path,
    run_id: str,
    model: str,
    seed: int,
    cases: list[ProbeCase],
    modes: list[ProbeMode],
    lengths: list[int],
    started_at: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cases.jsonl", [case.model_dump(mode="json") for case in cases])
    (output_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    (output_dir / "results.jsonl").write_text("", encoding="utf-8")
    summary = {
        "run_id": run_id,
        "model": model,
        "seed": seed,
        "started_at": started_at,
        "finished_at": _now(),
        "requested_modes": modes,
        "requested_lengths": lengths,
        "sample_count": len(cases),
        "dry_run": True,
        "overall": {"case_count": len(cases), "prediction_count": 0},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(output_dir / "summary.csv", [])
    return summary


def write_run_report(
    *,
    output_dir: Path,
    run_id: str,
    model: str,
    seed: int,
    cases: list[ProbeCase],
    results: list[ProbeCaseResult],
    modes: list[ProbeMode],
    lengths: list[int],
    started_at: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cases.jsonl", [case.model_dump(mode="json") for case in cases])
    _write_jsonl(output_dir / "predictions.jsonl", [result.prediction.model_dump(mode="json") for result in results])
    _write_jsonl(output_dir / "results.jsonl", [result.model_dump(mode="json") for result in results])
    rows = [_flat_row(result) for result in results]
    summary = {
        "run_id": run_id,
        "model": model,
        "seed": seed,
        "started_at": started_at,
        "finished_at": _now(),
        "requested_modes": modes,
        "requested_lengths": lengths,
        "sample_count": len(cases),
        "overall": _group_summary(rows),
        "by_probe": _grouped(rows, ["probe"]),
        "by_mode": _grouped(rows, ["mode"]),
        "by_length": _grouped(rows, ["length"]),
        "by_position": _grouped(rows, ["position"]),
        "by_probe_mode": _grouped(rows, ["probe", "mode"]),
        "by_probe_mode_length": _grouped(rows, ["probe", "mode", "length"]),
        "by_probe_mode_length_position": _grouped(rows, ["probe", "mode", "length", "position"]),
        "position_probe": _position_summary(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(output_dir / "summary.csv", rows)
    return summary


def _flat_row(result: ProbeCaseResult) -> dict[str, Any]:
    prediction = result.prediction
    return {
        "case_id": result.case.case_id,
        "probe": result.case.probe,
        "mode": prediction.mode,
        "length": result.case.target_length_tokens,
        "position": result.case.position or "",
        "exact_match": result.exact_match,
        "provider_error": bool(prediction.provider_error),
        "protocol_error": bool(prediction.protocol_error) or not prediction.protocol_valid,
        "input_tokens": prediction.actual_input_tokens,
        "output_tokens": prediction.output_tokens,
        "compactor_tokens": prediction.compactor_input_tokens + prediction.compactor_output_tokens,
        "latency_seconds": prediction.latency_seconds,
        "context_resets": prediction.context_reset_count,
        "recent_window_trim_count": prediction.recent_window_trim_count,
        "prunes": prediction.deterministic_prune_count,
        "structured_reset_count": prediction.structured_reset_count,
        "structured_compaction_count": prediction.structured_compaction_count,
        "estimated_tokens_removed": prediction.estimated_tokens_removed,
        "context_budget_exhausted": prediction.context_budget_exhausted,
        "answer_provider_calls": prediction.answer_provider_calls,
        "protocol_retry_count": prediction.protocol_retry_count,
        "compactor_provider_calls": prediction.compactor_provider_calls,
        **result.metrics,
    }


def _grouped(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return {"|".join(str(part) for part in key): _group_summary(value) for key, value in sorted(groups.items())}


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(rows)
    exact = sum(1 for row in rows if row.get("exact_match"))
    provider_errors = sum(1 for row in rows if row.get("provider_error"))
    protocol_errors = sum(1 for row in rows if row.get("protocol_error"))
    total_latency = sum(float(row.get("latency_seconds") or 0.0) for row in rows)
    total_input = sum(int(row.get("input_tokens") or 0) for row in rows)
    total_output = sum(int(row.get("output_tokens") or 0) for row in rows)
    total_compactor = sum(int(row.get("compactor_tokens") or 0) for row in rows)
    budget_exhausted = sum(1 for row in rows if row.get("context_budget_exhausted"))
    return {
        "accuracy": _ratio(exact, denominator),
        "provider_error_rate": _ratio(provider_errors, denominator),
        "protocol_error_rate": _ratio(protocol_errors, denominator),
        "context_budget_exhaustion_count": budget_exhausted,
        "context_budget_exhaustion_rate": _ratio(budget_exhausted, denominator),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_compactor_tokens": total_compactor,
        "total_answer_provider_calls": sum(int(row.get("answer_provider_calls") or 0) for row in rows),
        "total_compactor_provider_calls": sum(int(row.get("compactor_provider_calls") or 0) for row in rows),
        "average_input_tokens": total_input / denominator if denominator else 0.0,
        "average_latency": total_latency / denominator if denominator else 0.0,
        "total_context_resets": sum(int(row.get("context_resets") or 0) for row in rows),
        "total_recent_window_trims": sum(int(row.get("recent_window_trim_count") or 0) for row in rows),
        "total_prunes": sum(int(row.get("prunes") or 0) for row in rows),
        "total_structured_resets": sum(int(row.get("structured_reset_count") or 0) for row in rows),
        "total_structured_compactions": sum(int(row.get("structured_compaction_count") or 0) for row in rows),
        "total_estimated_tokens_removed": sum(int(row.get("estimated_tokens_removed") or 0) for row in rows),
        "total_protocol_retries": sum(int(row.get("protocol_retry_count") or 0) for row in rows),
    }


def _position_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    position_rows = [row for row in rows if row.get("probe") == "position"]
    by_position = {
        position: _group_summary([row for row in position_rows if row.get("position") == position])
        for position in ["beginning", "middle", "end"]
    }
    rates = {
        position: by_position[position]["accuracy"]["rate"]
        for position in by_position
        if by_position[position]["accuracy"]["denominator"] > 0
    }
    return {
        "beginning_accuracy": by_position["beginning"]["accuracy"],
        "middle_accuracy": by_position["middle"]["accuracy"],
        "end_accuracy": by_position["end"]["accuracy"],
        "best_worst_gap": max(rates.values()) - min(rates.values()) if len(rates) > 1 else 0.0,
        "beginning_middle_gap": rates["beginning"] - rates["middle"] if {"beginning", "middle"} <= rates.keys() else 0.0,
        "end_middle_gap": rates["end"] - rates["middle"] if {"end", "middle"} <= rates.keys() else 0.0,
    }


def _ratio(numerator: int, denominator: int) -> dict[str, float | int]:
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else 0.0}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _now() -> str:
    return datetime.now(UTC).isoformat()
