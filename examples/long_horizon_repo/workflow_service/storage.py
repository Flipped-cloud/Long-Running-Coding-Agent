from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from workflow_service.graph import validate_graph
from workflow_service.models import Task
from workflow_service.serialization import task_from_dict, task_to_dict


def _decode_tasks(raw: object) -> dict[str, Task]:
    if not isinstance(raw, list):
        raise ValueError("task store payload must be a list")
    tasks: dict[str, Task] = {}
    for item in raw:
        task = task_from_dict(item)
        if task.id in tasks:
            raise ValueError(f"duplicate task id: {task.id}")
        tasks[task.id] = task
    return tasks


class JsonTaskStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Task]:
        if not self.path.exists():
            return {}
        return _decode_tasks(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, tasks: dict[str, Task]) -> None:
        payload = [task_to_dict(tasks[task_id]) for task_id in sorted(tasks)]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def export_json(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else "[]"

    def import_json(self, payload: str) -> None:
        tasks = _decode_tasks(json.loads(payload))
        validate_graph(tasks)
        self.save(tasks)
