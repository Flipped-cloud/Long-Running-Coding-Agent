from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_activation_report(predictions_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(predictions_path)
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode") or ""), []).append(row)
    return {
        "full_history": _full_history_report(by_mode.get("full_history", [])),
        "recent_window": _recent_window_report(by_mode.get("recent_window", [])),
        "deterministic_prune": _deterministic_prune_report(by_mode.get("deterministic_prune", [])),
        "structured_reset": _structured_reset_report(by_mode.get("structured_reset", [])),
    }


def activation_passed(report: dict[str, Any]) -> bool:
    return all(bool(section.get("activated")) for section in report.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = build_activation_report(args.predictions)
    output = args.output or args.predictions.with_name("activation_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not activation_passed(report):
        raise SystemExit(1)


def _full_history_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reset_count = sum(int(row.get("context_reset_count") or 0) for row in rows)
    prune_count = sum(int(row.get("deterministic_prune_count") or 0) for row in rows)
    return {
        "activated": bool(rows) and reset_count == 0 and prune_count == 0,
        "case_count": len(rows),
        "reset_count": reset_count,
        "prune_count": prune_count,
    }


def _recent_window_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trim_events = sum(int(row.get("recent_window_trim_count") or 0) for row in rows)
    prune_events = sum(int(row.get("deterministic_prune_count") or 0) for row in rows)
    reset_events = sum(int(row.get("context_reset_count") or 0) for row in rows)
    removed = sum(int(row.get("estimated_tokens_removed") or 0) for row in rows)
    activated = bool(rows) and (trim_events > 0 or removed > 0) and prune_events == 0 and reset_events == 0
    return {
        "activated": activated,
        "case_count": len(rows),
        "trim_events": trim_events,
        "prune_events": prune_events,
        "reset_events": reset_events,
        "removed_tokens": removed,
    }


def _deterministic_prune_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prune_events = sum(int(row.get("deterministic_prune_count") or 0) for row in rows)
    return {"activated": bool(rows) and prune_events > 0, "case_count": len(rows), "prune_events": prune_events}


def _structured_reset_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reset_events = sum(int(row.get("context_reset_count") or 0) for row in rows)
    structured_reset_events = sum(int(row.get("structured_reset_count") or 0) for row in rows)
    compaction_events = sum(int(row.get("structured_compaction_count") or 0) for row in rows)
    return {
        "activated": bool(rows) and reset_events > 0 and (structured_reset_events > 0 or compaction_events > 0),
        "case_count": len(rows),
        "reset_events": reset_events,
        "structured_reset_events": structured_reset_events,
        "compaction_events": compaction_events,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"predictions file not found: {path}", file=sys.stderr)
        raise SystemExit(1) from None
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"invalid JSON at {path}:{line_number}: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
    return rows


if __name__ == "__main__":
    main()
