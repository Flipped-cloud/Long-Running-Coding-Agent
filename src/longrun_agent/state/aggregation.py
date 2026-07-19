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
    total_input_tokens = sum(int(session.get("input_tokens_total") or 0) for session in sessions)
    total_output_tokens = sum(int(session.get("output_tokens_total") or 0) for session in sessions)
    total_compactor_tokens = sum(
        int(session.get("compactor_input_tokens") or 0) + int(session.get("compactor_output_tokens") or 0) for session in sessions
    )
    total_model_requests = sum(int(session.get("steps") or 0) + int(session.get("terminal_grace_turn_count") or 0) for session in sessions)
    sessions_with_reset = [session for session in sessions if int(session.get("context_reset_count") or 0) > 0]
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
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_compactor_tokens": total_compactor_tokens,
        "max_context_usage_ratio": max((float(session.get("max_context_usage_ratio") or 0.0) for session in sessions), default=0.0),
        "total_context_resets": sum(int(session.get("context_reset_count") or 0) for session in sessions),
        "total_context_prunes": sum(int(session.get("deterministic_prune_count") or 0) for session in sessions),
        "total_context_compactions": sum(int(session.get("structured_compaction_count") or 0) for session in sessions),
        "total_stale_items_removed": sum(int(session.get("stale_item_count") or 0) for session in sessions),
        "total_estimated_tokens_removed": sum(int(session.get("estimated_tokens_removed") or 0) for session in sessions),
        "average_tokens_per_model_request": derived_tokens / total_model_requests if total_model_requests else 0.0,
        "average_reset_recovery_steps": (
            sum(int(session.get("steps") or 0) for session in sessions_with_reset) / len(sessions_with_reset)
            if sessions_with_reset
            else 0.0
        ),
        "sessions_with_context_reset": len(sessions_with_reset),
        "context_budget_exhaustion_count": sum(1 for session in sessions if session.get("context_budget_exhausted")),
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
        "protocol_error_count": sum(int(session.get("protocol_error_count") or 0) for session in sessions),
        "recoverable_protocol_error_count": sum(int(session.get("recoverable_protocol_error_count") or 0) for session in sessions),
        "fatal_protocol_error_count": sum(int(session.get("fatal_protocol_error_count") or 0) for session in sessions),
        "provider_error_count": sum(int(session.get("provider_error_count") or 0) for session in sessions),
        "tool_argument_protocol_retry_count": sum(int(session.get("tool_argument_protocol_retry_count") or 0) for session in sessions),
        "tasks_completed_after_grace_turn": sum(
            1 for session in sessions if session.get("terminal_signal_recovered") and session.get("terminal_signal") == "completion_request"
        ),
        "auto_completion_recovered_count": sum(1 for session in sessions if session.get("auto_completion_recovered")),
        "episodes_created": sum(int(session.get("episodes_created") or 0) for session in sessions),
        "reflection_candidates": sum(int(session.get("reflection_candidates") or 0) for session in sessions),
        "active_memories_created": sum(int(session.get("active_memories_created") or 0) for session in sessions),
        "quarantined_memories": sum(int(session.get("quarantined_memories") or 0) for session in sessions),
        "memories_retrieved": sum(int(session.get("memories_retrieved") or 0) for session in sessions),
        "memories_exposed": sum(int(session.get("memories_exposed") or 0) for session in sessions),
        "memories_referenced": sum(int(session.get("memories_referenced") or 0) for session in sessions),
        "memory_helpful_use_count": sum(int(session.get("memories_helpful") or 0) for session in sessions),
        "memory_harmful_use_count": sum(int(session.get("memories_harmful") or 0) for session in sessions),
        "skills_created": sum(int(session.get("skills_created") or 0) for session in sessions),
        "skills_validated": sum(int(session.get("skills_validated") or 0) for session in sessions),
        "skills_retrieved": sum(int(session.get("skills_retrieved") or 0) for session in sessions),
        "skills_exposed": sum(int(session.get("skills_exposed") or 0) for session in sessions),
        "skills_referenced": sum(int(session.get("skills_referenced") or 0) for session in sessions),
        "skill_helpful_use_count": sum(int(session.get("skills_helpful") or 0) for session in sessions),
        "skill_harmful_use_count": sum(int(session.get("skills_harmful") or 0) for session in sessions),
        "knowledge_tokens_injected": sum(int(session.get("knowledge_tokens_injected") or 0) for session in sessions),
        "memory_conflicts": sum(int(session.get("memory_conflicts") or 0) for session in sessions),
        "negative_transfer_count": sum(int(session.get("negative_transfer_count") or 0) for session in sessions),
    }
