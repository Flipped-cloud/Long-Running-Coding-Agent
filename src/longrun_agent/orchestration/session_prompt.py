from __future__ import annotations

from longrun_agent.config import AppConfig
from longrun_agent.state.schema import ProjectState, TaskNode


def build_task_session_prompt(state: ProjectState, task: TaskNode, config: AppConfig | None = None) -> str:
    dependency_summaries = []
    for dependency_id in task.dependencies:
        dependency = state.task_by_id(dependency_id)
        dependency_summaries.append(f"- {dependency.id}: {dependency.completion_summary or dependency.status.value}")
    max_sessions = config.planning.execution.max_sessions_per_task if config else "unknown"
    max_steps = config.agent.max_steps if config else "unknown"
    max_session_seconds = config.agent.max_session_seconds if config else "unknown"
    no_progress_limit = config.planning.execution.max_no_progress_sessions if config else 1
    no_progress_warning = (
        task.consecutive_no_progress_sessions >= no_progress_limit
        if isinstance(no_progress_limit, int) and no_progress_limit > 0
        else False
    )
    return "\n".join(
        [
            f"Project objective: {state.objective}",
            f"Current task ID: {task.id}",
            f"Current objective: {task.objective}",
            f"Task objective: {task.objective}",
            "High-priority handoff / next action:",
            task.last_handoff_summary or "- none",
            "Acceptance criteria:",
            *[f"- {criterion}" for criterion in task.acceptance_criteria],
            "Completed dependency summary:",
            *(dependency_summaries or ["- none"]),
            "Existing progress notes:",
            *([f"- {note}" for note in task.progress_notes] or ["- none"]),
            "Previous handoff summary:",
            task.last_handoff_summary or "- none",
            "Previously read files:",
            *([f"- {path}" for path in task.read_files] or ["- none"]),
            "Previously changed files:",
            *([f"- {path}" for path in task.files_touched] or ["- none"]),
            f"Blocker: {task.blocker or 'none'}",
            f"Attempts: {task.attempts} / {max_sessions}",
            f"Remaining sessions for this task: {max(0, int(max_sessions) - task.attempts) if isinstance(max_sessions, int) else 'unknown'}",
            f"Session budget: max_steps={max_steps}, max_session_seconds={max_session_seconds}",
            f"Consecutive no-progress sessions: {task.consecutive_no_progress_sessions}",
            (
                "No-progress warning: do not repeat reads of unchanged files; this session must modify files, "
                "run tests, or report a blocker."
                if no_progress_warning
                else "No-progress warning: none"
            ),
            "Control tools: report_progress, report_blocker, request_task_completion, request_decomposition.",
            "Bash tool protocol:",
            '- Prefer argv, for example {"argv": ["python", "-m", "pytest", "-q"], "cwd": "."}.',
            "- Commands already run inside the workspace.",
            "- Do not use cd, &&, ||, pipes, redirection, semicolons, or shell built-ins.",
            "- Do not use absolute-path rm commands.",
            '- For CLI checks use argv, for example {"argv": ["python", "-m", "task_service.cli", "get", "--db", "tasks.json", "--id", "1"], "cwd": "."}.',
            "Execution rhythm:",
            "- Only read files named directly in the objective or acceptance criteria.",
            "- Complete necessary inspection within the first 3 successful tool calls.",
            "- Starting with the 4th successful tool call, modify files, run a test, report_blocker, or request_task_completion.",
            "- Last 2 model turns must be used for verification and a terminal control signal.",
            "- When complete, call request_task_completion.",
            "- When unable to complete, call report_blocker.",
            "Forbidden:",
            "- Do not consecutively read the same unchanged file.",
            "- Do not keep exploring after tests pass.",
            "- Do not use plain FinalAnswer instead of request_task_completion/report_blocker/request_decomposition.",
            "- Do not invent test results.",
        ]
    )
