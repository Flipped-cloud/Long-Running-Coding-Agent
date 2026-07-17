from __future__ import annotations

from collections.abc import Callable
from typing import Any

from longrun_agent.context.schema import ContextBudgetSnapshot, PruneReport

ContextEventSink = Callable[[str, dict[str, Any]], None]


def budget_payload(
    *,
    project_id: str | None,
    task_id: str | None,
    session_id: str | None,
    segment_id: int,
    step: int,
    mode: str,
    snapshot: ContextBudgetSnapshot,
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "task_id": task_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "step": step,
        "mode": mode,
        "estimated_tokens_before": snapshot.estimated_total_tokens,
        "estimated_tokens_after": snapshot.estimated_total_tokens,
        "usage_ratio": snapshot.usage_ratio,
        "actual_input_tokens": None,
        "items_removed": 0,
        "tokens_removed": 0,
        "handoff_id": None,
        "reason": None,
    }


def prune_payload(base: dict[str, Any], report: PruneReport) -> dict[str, Any]:
    payload = dict(base)
    payload.update(
        {
            "estimated_tokens_before": report.estimated_tokens_before,
            "estimated_tokens_after": report.estimated_tokens_after,
            "items_removed": report.items_removed,
            "tokens_removed": report.tokens_removed,
            "stale_items_removed": report.stale_items_removed,
            "superseded_items_removed": report.superseded_items_removed,
            "compacted_items": report.compacted_items,
        }
    )
    return payload
