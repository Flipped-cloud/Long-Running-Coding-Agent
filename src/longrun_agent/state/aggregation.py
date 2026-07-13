from __future__ import annotations

from datetime import datetime
from typing import Any

from longrun_agent.state.schema import ProjectState, TaskStatus, utc_now


def aggregate_candidate_complete_parents(state: ProjectState) -> list[str]:
    changed: list[str] = []
    while True:
        updated = False
        for task in state.tasks:
            if task.status != TaskStatus.DECOMPOSED:
                continue
            children = [candidate for candidate in state.tasks if candidate.parent_id == task.id]
            if children and all(child.status == TaskStatus.CANDIDATE_COMPLETE for child in children):
                task.status = TaskStatus.CANDIDATE_COMPLETE
                task.completion_summary = "All child tasks are candidate complete."
                task.updated_at = utc_now()
                changed.append(task.id)
                updated = True
        if not updated:
            break
    if changed:
        state.updated_at = utc_now()
    return changed


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float:
    if not started_at or not finished_at:
        return 0.0
    try:
        return max(0.0, (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds())
    except ValueError:
        return 0.0


def project_statistics(
    state: ProjectState,
    sessions: list[dict[str, Any]] | None = None,
    *,
    total_tool_calls: int | None = None,
    total_tokens: int | None = None,
    configured_max_project_seconds: int | None = None,
    wall_clock_seconds: float | None = None,
    final_verification_exit_code: int | None = None,
) -> dict[str, Any]:
    sessions = sessions or []
    derived_tool_calls = sum(int(session.get("tool_call_count", 0)) for session in sessions)
    derived_tokens = sum(int(session.get("total_tokens", 0)) for session in sessions)
    duration_seconds = sum(
        float(session.get("duration_seconds") or _duration_seconds(session.get("started_at"), session.get("finished_at")))
        for session in sessions
    )
    changed_files = {
        path for session in sessions for path in [*(session.get("changed_files") or []), *(session.get("files_touched") or [])] if path
    }
    return {
        "project_status": state.status.value,
        "candidate_completed_tasks": sum(1 for task in state.tasks if task.status == TaskStatus.CANDIDATE_COMPLETE),
        "blocked_tasks": sum(1 for task in state.tasks if task.status == TaskStatus.BLOCKED),
        "tasks_failed": sum(1 for task in state.tasks if task.status == TaskStatus.FAILED),
        "decomposition_count": sum(1 for revision in state.revisions if revision.trigger == "decomposition"),
        "max_task_depth": max((task.depth for task in state.tasks), default=0),
        "plan_revision_count": len(state.revisions),
        "recovery_candidate_count": sum(len(revision.candidate_ids) for revision in state.revisions),
        "sessions_without_terminal_signal": sum(1 for session in sessions if not session.get("terminal_signal")),
        "no_progress_sessions": sum(1 for session in sessions if session.get("no_progress")),
        "project_sessions": len(sessions),
        "total_tool_calls": derived_tool_calls if total_tool_calls is None else total_tool_calls,
        "total_tokens": derived_tokens if total_tokens is None else total_tokens,
        "duration_seconds": duration_seconds,
        "wall_clock_seconds": duration_seconds if wall_clock_seconds is None else wall_clock_seconds,
        "configured_max_project_seconds": configured_max_project_seconds,
        "time_budget_exhausted": state.status.value == "time_limit_reached",
        "repeated_tool_calls": sum(len(session.get("repeated_tool_calls") or []) for session in sessions),
        "changed_file_count": len(changed_files),
        "successful_test_command_count": sum(len(session.get("successful_test_commands") or []) for session in sessions),
        "final_verification_exit_code": final_verification_exit_code,
        "final_verification_passed": final_verification_exit_code == 0 if final_verification_exit_code is not None else None,
        "terminal_grace_turn_count": sum(int(session.get("terminal_grace_turn_count") or 0) for session in sessions),
        "terminal_signal_recovered_count": sum(1 for session in sessions if session.get("terminal_signal_recovered")),
        "unsupported_shell_syntax_count": sum(int(session.get("unsupported_shell_syntax_count") or 0) for session in sessions),
        "tool_argument_protocol_retry_count": sum(int(session.get("tool_argument_protocol_retry_count") or 0) for session in sessions),
        "tasks_completed_after_grace_turn": sum(
            1 for session in sessions if session.get("terminal_signal_recovered") and session.get("terminal_signal") == "completion_request"
        ),
        "auto_completion_recovered_count": sum(1 for session in sessions if session.get("auto_completion_recovered")),
    }
