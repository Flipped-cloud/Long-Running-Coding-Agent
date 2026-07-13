from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("examples/task_service_repo"))
    args = parser.parse_args()
    repo = args.repo.resolve()

    pytest_result = run([sys.executable, "-m", "pytest", "-q"], repo)
    if pytest_result.returncode != 0:
        print(pytest_result.stdout)
        print(pytest_result.stderr, file=sys.stderr)
        return 1

    sys.path.insert(0, str(repo))
    from task_service.cli import main as cli_main  # noqa: PLC0415
    from task_service.model import Task  # noqa: PLC0415
    from task_service.service import TaskService  # noqa: PLC0415
    from task_service.storage import load_tasks, save_tasks  # noqa: PLC0415

    nested = repo / "tmp_validation" / "nested" / "tasks.json"
    save_tasks(nested, [Task(id="v1", title="nested")])
    if load_tasks(nested)[0].id != "v1":
        raise AssertionError("nested JSON persistence failed")

    try:
        TaskService().retry("missing")
    except ValueError as exc:
        if "unknown task id: missing" not in str(exc):
            raise AssertionError("missing task retry error is not explicit") from exc
    else:
        raise AssertionError("missing task retry did not fail")

    db = repo / "tmp_validation" / "cli_tasks.json"
    run([sys.executable, "-m", "task_service.cli", "add", "--db", str(db), "--id", "cli-1", "--title", "from cli"], repo)
    query = run([sys.executable, "-m", "task_service.cli", "get", "--db", str(db), "--id", "cli-1"], repo)
    if query.returncode != 0 or "cli-1: from cli" not in query.stdout:
        raise AssertionError("CLI get by ID failed")

    invalid = run([sys.executable, "-m", "task_service.cli", "add", "--db", str(db), "--id", "../bad", "--title", "bad"], repo)
    empty_title = run([sys.executable, "-m", "task_service.cli", "add", "--db", str(db), "--id", "empty", "--title", ""], repo)
    if invalid.returncode == 0 or "task id must be a simple identifier" not in invalid.stderr:
        raise AssertionError("invalid ID did not produce a clear error")
    if empty_title.returncode == 0 or "task title is required" not in empty_title.stderr:
        raise AssertionError("empty title did not produce a clear error")

    # Keep the import referenced so linters know the CLI module is intentionally loaded above.
    assert callable(cli_main)
    print(json.dumps({"status": "ok", "pytest": pytest_result.stdout.strip()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
