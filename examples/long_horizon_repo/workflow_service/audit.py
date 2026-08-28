from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    event_type: str
    task_id: str
    details: dict[str, Any]


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, event_type: str, task_id: str, details: dict[str, Any] | None = None) -> AuditEvent:
        events = self.read()
        event = AuditEvent(events[-1].sequence + 1 if events else 1, event_type, task_id, details or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        return event

    def read(self) -> list[AuditEvent]:
        if not self.path.exists():
            return []
        events: list[AuditEvent] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                event = AuditEvent(**payload)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid audit event on line {line_number}") from exc
            if (
                isinstance(event.sequence, bool)
                or not isinstance(event.sequence, int)
                or event.sequence < 1
                or (events and event.sequence <= events[-1].sequence)
            ):
                raise ValueError(f"invalid audit sequence on line {line_number}")
            if not isinstance(event.details, dict):
                raise ValueError(f"invalid audit details on line {line_number}")
            events.append(event)
        return events
