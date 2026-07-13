from __future__ import annotations

import json
from pathlib import Path

from task_service.model import Task


def load_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    return [Task(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([task.__dict__ for task in tasks], indent=2), encoding="utf-8")
