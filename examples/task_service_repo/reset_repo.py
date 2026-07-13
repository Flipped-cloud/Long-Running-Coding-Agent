from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GAP_BOOL_MESSAGE = (
    "invalid task field 'attempts': expected an int, got bool (bool is not a valid attempt count, got True). "
    "Use 0 for no attempts, not True/False."
)
GAP_NEGATIVE_MESSAGE = "invalid task field 'attempts': must be a non-negative integer, got -1"
GAP_NON_TASK_MESSAGE = "invalid task: expected a Task instance, got dict"

BASELINE_FILES = {
    "task_service/__init__.py": "",
    "task_service/model.py": """from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    id: str
    title: str
    completed: bool = False
    attempts: int = 0


def validate_task(task: Task) -> None:
    if not task.id:
        raise ValueError("task id is required")
    if not task.title:
        raise ValueError("task title is required")
""",
    "task_service/storage.py": """from __future__ import annotations

import json
from pathlib import Path

from task_service.model import Task


def load_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    return [Task(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.write_text(json.dumps([task.__dict__ for task in tasks], indent=2), encoding="utf-8")
""",
    "task_service/service.py": """from __future__ import annotations

from task_service.model import Task, validate_task


class TaskService:
    def __init__(self, tasks: list[Task] | None = None):
        self.tasks = tasks or []

    def add(self, task: Task) -> None:
        validate_task(task)
        if any(existing.id == task.id for existing in self.tasks):
            raise ValueError("duplicate task id")
        self.tasks.append(task)

    def complete(self, task_id: str) -> bool:
        for task in self.tasks:
            if task.id == task_id:
                task.completed = True
                return True
        return False

    def retry(self, task_id: str) -> int:
        for task in self.tasks:
            if task.id == task_id:
                task.attempts += 1
                return task.attempts
        raise ValueError("unknown task")
""",
    "task_service/cli.py": """from __future__ import annotations

import argparse
from pathlib import Path

from task_service.model import Task
from task_service.service import TaskService
from task_service.storage import load_tasks, save_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["add", "list", "complete"])
    parser.add_argument("--db", default="tasks.json")
    parser.add_argument("--id")
    parser.add_argument("--title")
    args = parser.parse_args()

    db = Path(args.db)
    service = TaskService(load_tasks(db))
    if args.command == "add":
        service.add(Task(id=args.id or "", title=args.title or ""))
        save_tasks(db, service.tasks)
    elif args.command == "complete":
        service.complete(args.id or "")
        save_tasks(db, service.tasks)
    else:
        for task in service.tasks:
            print(f"{task.id}: {task.title} completed={task.completed} attempts={task.attempts}")


if __name__ == "__main__":
    main()
""",
    "tests/test_service.py": """from pathlib import Path

import pytest
from task_service.model import Task
from task_service.service import TaskService
from task_service.storage import load_tasks, save_tasks


def test_add_rejects_duplicate_id():
    service = TaskService()
    service.add(Task(id="1", title="one"))
    with pytest.raises(ValueError):
        service.add(Task(id="1", title="again"))


def test_json_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "tasks.json"
    save_tasks(path, [Task(id="1", title="one")])
    assert load_tasks(path)[0].title == "one"


def test_retry_counts_attempts():
    service = TaskService([Task(id="1", title="one")])
    assert service.retry("1") == 1
""",
    "tests/test_gap_coverage.py": f"""import pytest
from task_service.model import Task, validate_task


def test_validate_task_rejects_bool_attempts():
    with pytest.raises(ValueError) as exc_info:
        validate_task(Task(id="1", title="one", attempts=True))
    assert (
        str(exc_info.value)
        == {GAP_BOOL_MESSAGE!r}
    )


def test_validate_task_rejects_negative_attempts():
    with pytest.raises(ValueError) as exc_info:
        validate_task(Task(id="1", title="one", attempts=-1))
    assert str(exc_info.value) == {GAP_NEGATIVE_MESSAGE!r}


def test_validate_task_rejects_non_task_input():
    with pytest.raises(ValueError) as exc_info:
        validate_task({{"id": "1", "title": "one"}})
    assert str(exc_info.value) == "invalid task: expected a Task instance, got dict"
""",
    "README.md": """# Task Service Repo

A small multi-file Python repository for v0.2 planning experiments.

Run tests:

```bash
python -m pytest -q
```

The task in `TASK.md` has related requirements with clear dependencies, but the repository is intentionally small.
""",
}

EXPECTED_TEST_FILES = {"test_service.py", "test_gap_coverage.py"}
MUTABLE_PATHS = [
    "task_service/__pycache__",
    "tests/__pycache__",
    ".pytest_cache",
    "tmp_validation",
    "tasks.json",
    "INSPECTION_NOTES.md",
    "VALIDATION_IMPROVEMENTS.md",
]
GAP_REQUIRED_SNIPPETS = [
    "def test_validate_task_rejects_bool_attempts",
    'validate_task(Task(id="1", title="one", attempts=True))',
    GAP_BOOL_MESSAGE,
    "def test_validate_task_rejects_negative_attempts",
    'validate_task(Task(id="1", title="one", attempts=-1))',
    GAP_NEGATIVE_MESSAGE,
    "def test_validate_task_rejects_non_task_input",
    'validate_task({"id": "1", "title": "one"})',
    GAP_NON_TASK_MESSAGE,
]


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def reset_fixture() -> int:
    for relative_path in MUTABLE_PATHS:
        _remove_path(ROOT / relative_path)

    tests_dir = ROOT / "tests"
    if tests_dir.exists():
        for path in tests_dir.glob("test_*.py"):
            path.unlink()

    for relative_path, content in BASELINE_FILES.items():
        path = ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    failures = validate_fixture()
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)
    return len(BASELINE_FILES)


def validate_fixture() -> list[str]:
    failures: list[str] = []
    for relative_path in BASELINE_FILES:
        if not (ROOT / relative_path).is_file():
            failures.append(f"missing baseline file: {relative_path}")

    tests_dir = ROOT / "tests"
    actual_tests = {path.name for path in tests_dir.glob("test_*.py")} if tests_dir.exists() else set()
    missing_tests = sorted(EXPECTED_TEST_FILES - actual_tests)
    extra_tests = sorted(actual_tests - EXPECTED_TEST_FILES)
    for name in missing_tests:
        failures.append(f"missing test file: tests/{name}")
    for name in extra_tests:
        failures.append(f"extra test file: tests/{name}")

    if (ROOT / "tests/test_edge_cases.py").exists():
        failures.append("stale test file exists: tests/test_edge_cases.py")
    for relative_path in ("INSPECTION_NOTES.md", "VALIDATION_IMPROVEMENTS.md", "tmp_validation"):
        if (ROOT / relative_path).exists():
            failures.append(f"stale path exists: {relative_path}")

    gap_file = ROOT / "tests/test_gap_coverage.py"
    if gap_file.is_file():
        content = gap_file.read_text(encoding="utf-8")
        for snippet in GAP_REQUIRED_SNIPPETS:
            if snippet not in content:
                failures.append(f"missing gap coverage snippet: {snippet}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate the fixture without modifying files")
    args = parser.parse_args()

    if args.check:
        failures = validate_fixture()
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            raise SystemExit(1)
        print("task_service_repo fixture valid")
        return

    restored = reset_fixture()
    print(f"task_service_repo reset: {restored} files restored")


if __name__ == "__main__":
    main()
