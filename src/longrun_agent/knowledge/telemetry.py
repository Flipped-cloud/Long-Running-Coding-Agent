from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longrun_agent.knowledge.schema import utc_now


def knowledge_event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {"event_type": event_type, "timestamp": utc_now(), **{key: value for key, value in payload.items() if value is not None}}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
