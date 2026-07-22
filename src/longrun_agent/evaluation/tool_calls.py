from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_tool_calls(paths: list[Path]) -> list[dict[str, Any]]:
    calls: dict[tuple[str, int, str], dict[str, Any]] = {}
    order: list[tuple[str, int, str]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid telemetry JSONL at {path}:{line_number}: {exc}") from exc
            if event.get("event_type") not in {"tool_started", "tool_finished"}:
                continue
            tool_call_id = str(event.get("tool_call_id") or "")
            if not tool_call_id:
                continue
            key = (str(event.get("run_id") or ""), int(event.get("step") or 0), tool_call_id)
            if event["event_type"] == "tool_started":
                if key not in calls:
                    order.append(key)
                    calls[key] = {
                        "run_id": key[0],
                        "step": key[1],
                        "tool_call_id": key[2],
                        "tool_name": event.get("tool_name"),
                        "success": None,
                        "error_type": None,
                        "duration": None,
                    }
                continue
            if key not in calls:
                continue
            calls[key].update(
                success=event.get("success"),
                error_type=event.get("error_type"),
                duration=event.get("duration_seconds"),
            )
    return [calls[key] for key in order]
