from __future__ import annotations

import json
from pathlib import Path

from workflow_service.models import Task
from workflow_service.serialization import task_from_dict, task_to_dict


class JsonTaskStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Task]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {item["id"]: task_from_dict(item) for item in payload}

    def save(self, tasks: dict[str, Task]) -> None:
        payload = [task_to_dict(task) for task in tasks.values()]
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def export_json(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else "[]"

    def import_json(self, payload: str) -> None:
        raw = json.loads(payload)
        tasks = {item["id"]: task_from_dict(item) for item in raw}
        self.save(tasks)
