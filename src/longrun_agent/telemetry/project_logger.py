from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from longrun_agent.telemetry.logger import sanitize_payload


class ProjectLogger:
    def __init__(self, events_path: Path):
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: str,
        *,
        project_id: str,
        task_id: str | None = None,
        session_id: str | None = None,
        plan_version: int = 0,
        old_status: str | None = None,
        new_status: str | None = None,
        trigger: str | None = None,
        reason: str | None = None,
        candidate_ids: list[str] | None = None,
        selected_candidate_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "project_id": project_id,
            "task_id": task_id,
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "plan_version": plan_version,
            "old_status": old_status,
            "new_status": new_status,
            "trigger": trigger,
            "reason": reason,
            "candidate_ids": candidate_ids or [],
            "selected_candidate_id": selected_candidate_id,
            "payload": sanitize_payload(payload or {}),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
