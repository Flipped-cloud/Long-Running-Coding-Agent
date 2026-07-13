from __future__ import annotations

import argparse
import sys
from pathlib import Path

from task_service.model import Task
from task_service.service import TaskService
from task_service.storage import load_tasks, save_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["add", "list", "complete", "get"])
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
    elif args.command == "get":
        task_id = args.id or ""
        for task in service.tasks:
            if task.id == task_id:
                print(
                    f"{task.id}: {task.title} completed={task.completed} attempts={task.attempts}"
                )
                return
        print(f"task not found: {task_id}")
        sys.exit(1)
    else:
        for task in service.tasks:
            print(
                f"{task.id}: {task.title} completed={task.completed} attempts={task.attempts}"
            )


if __name__ == "__main__":
    main()
