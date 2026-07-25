from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a long-horizon project run from persisted state.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--ended-at", required=True)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--minimum-target-seconds", type=int, required=True)
    parser.add_argument("--configured-project-limit-seconds", type=int, required=True)
    parser.add_argument("--cli-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing optional data file: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"could not read {path}: {exc}"
    return payload if isinstance(payload, dict) else {}, None


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"missing optional data file: {path}"
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                return rows, f"non-object JSONL row at {path}:{line_number}"
    except (OSError, json.JSONDecodeError) as exc:
        return rows, f"could not read {path}: {exc}"
    return rows, None


def unique_session_values(sessions: list[dict[str, Any]], field: str) -> list[str]:
    return sorted({str(value) for session in sessions for value in (session.get(field) or []) if value})


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = args.state_root.resolve() / args.project_id
    state_path = project_dir / "project_state.json"
    sessions_path = project_dir / "sessions.jsonl"
    metrics_path = project_dir / "project_metrics.json"
    state, state_warning = read_json(state_path)
    sessions, sessions_warning = read_jsonl(sessions_path)
    metrics, metrics_warning = read_json(metrics_path)
    oracle, oracle_warning = read_json(args.oracle)
    warnings = [warning for warning in (state_warning, sessions_warning, metrics_warning, oracle_warning) if warning]

    tasks = state.get("tasks") or []
    status_counts = Counter(str(task.get("status", "unknown")) for task in tasks if isinstance(task, dict))
    prompt_tokens = sum(int(session.get("input_tokens_total") or 0) for session in sessions)
    completion_tokens = sum(int(session.get("output_tokens_total") or 0) for session in sessions)
    total_tokens = int(metrics.get("total_tokens") or sum(int(session.get("total_tokens") or 0) for session in sessions))
    final_passed = metrics.get("final_verification_passed")
    if final_passed is True:
        final_verification_status = "passed"
    elif final_passed is False:
        final_verification_status = "failed"
    else:
        final_verification_status = "not_recorded"

    elapsed_seconds = max(0.0, args.elapsed_seconds)
    return {
        "project_id": args.project_id,
        "project_status": state.get("status", "state_missing"),
        "cli_exit_code": args.cli_exit_code,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_minutes": round(elapsed_seconds / 60.0, 3),
        "minimum_target_seconds": args.minimum_target_seconds,
        "duration_target_met": elapsed_seconds >= args.minimum_target_seconds,
        "configured_project_limit_seconds": args.configured_project_limit_seconds,
        "task_count": len(tasks),
        "task_status_counts": dict(sorted(status_counts.items())),
        "session_count": int(state.get("session_count") or len(sessions)),
        "context_reset_count": int(
            metrics.get("total_context_resets") or sum(int(session.get("context_reset_count") or 0) for session in sessions)
        ),
        "reopen_count": sum(int(task.get("reopen_count") or 0) for task in tasks if isinstance(task, dict)),
        "plan_revision_count": len(state.get("revisions") or []),
        "total_model_steps": sum(
            int(session.get("steps") or 0) + int(session.get("terminal_grace_turn_count") or 0) for session in sessions
        ),
        "total_tool_calls": int(metrics.get("total_tool_calls") or sum(int(session.get("tool_call_count") or 0) for session in sessions)),
        "repeated_action_count": int(
            metrics.get("repeated_tool_calls") or sum(len(session.get("repeated_tool_calls") or []) for session in sessions)
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "changed_files": unique_session_values(sessions, "changed_files"),
        "successful_test_commands": unique_session_values(sessions, "successful_test_commands"),
        "final_verification_status": final_verification_status,
        "oracle_verified": bool(oracle.get("oracle_verified", False)),
        "integrity_passed": bool(oracle.get("integrity_passed", False)),
        "state_path": str(state_path),
        "telemetry_path": str(args.telemetry_root.resolve()),
        "workspace_path": str(args.workspace.resolve()),
        "console_log_path": str(args.console_log.resolve()),
        "summary_warnings": warnings,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    write_summary(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
