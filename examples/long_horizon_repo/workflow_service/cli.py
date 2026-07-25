from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from workflow_service.audit import AuditLog
from workflow_service.metrics import task_metrics
from workflow_service.models import TaskStatus
from workflow_service.serialization import task_to_dict
from workflow_service.service import WorkflowService
from workflow_service.storage import JsonTaskStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workflow-service")
    parser.add_argument("--db", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--id", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--depends-on", action="append", default=[])
    create.add_argument("--max-attempts", type=int, default=3)

    get = subparsers.add_parser("get")
    get.add_argument("--id", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--status", choices=[status.value for status in TaskStatus])

    for command in ("run", "retry", "cancel"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--id", required=True)
        if command == "cancel":
            command_parser.add_argument("--reason", required=True)

    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--input", type=Path, required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = JsonTaskStore(args.db)
    service = WorkflowService(store, AuditLog(args.db.with_suffix(".events.jsonl")))
    try:
        if args.command == "create":
            task = service.create(
                args.id,
                args.name,
                dependencies=tuple(args.depends_on),
                max_attempts=args.max_attempts,
            )
            print(json.dumps(task_to_dict(task), sort_keys=True))
        elif args.command == "get":
            print(json.dumps(task_to_dict(service.get(args.id)), sort_keys=True))
        elif args.command == "list":
            status = TaskStatus(args.status) if args.status else None
            print(json.dumps([task_to_dict(task) for task in service.list(status)], sort_keys=True))
        elif args.command == "metrics":
            print(json.dumps(task_metrics(store.load()), sort_keys=True))
        else:
            raise ValueError(f"command not implemented: {args.command}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
