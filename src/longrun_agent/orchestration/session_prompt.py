from __future__ import annotations

from longrun_agent.config import AppConfig
from longrun_agent.context.assembler import render_current_instruction, render_task_anchor
from longrun_agent.context.schema import TaskContextSeed
from longrun_agent.state.schema import ProjectState, TaskNode


def build_task_context_seed(
    state: ProjectState,
    task: TaskNode,
    *,
    knowledge_context: str | None = None,
    knowledge_retrieval_id: str | None = None,
    config: AppConfig | None = None,
) -> TaskContextSeed:
    dependency_summaries = []
    for dependency_id in task.dependencies:
        dependency = state.task_by_id(dependency_id)
        dependency_summaries.append(f"- {dependency.id}: {dependency.completion_summary or dependency.status.value}")
    recent_notes = []
    for index, note in enumerate(task.progress_notes[-5:], start=max(1, len(task.progress_notes) - 4)):
        recent_notes.append(f"- note {index}: {_short_note(note)}")
    return TaskContextSeed(
        project_id=state.project_id,
        project_objective=state.objective,
        plan_version=state.plan_version,
        task_id=task.id,
        task_title=task.title,
        task_objective=task.objective,
        task_status=task.status.value,
        acceptance_criteria=task.acceptance_criteria,
        dependency_summaries=dependency_summaries,
        blocker=task.blocker,
        attempts=task.attempts,
        progress_summary="\n".join(recent_notes) if recent_notes else None,
        files_touched=task.files_touched[-10:],
        latest_handoff_id=task.latest_context_handoff_id,
        knowledge_context=knowledge_context,
        knowledge_retrieval_id=knowledge_retrieval_id,
        pinned_protocol=_generated_test_protocol(config),
        final_protocol_reminders=_generated_test_final_checklist(config),
    )


def build_task_session_prompt(state: ProjectState, task: TaskNode, config: AppConfig | None = None) -> str:
    seed = build_task_context_seed(state, task, config=config)
    repeat = config.context.repeat_task_anchor_at_end if config else True
    return "\n\n".join([render_task_anchor(seed), render_current_instruction(seed, repeat_anchor=repeat)])


def _short_note(note: str) -> str:
    normalized = " ".join(note.split())
    if len(normalized) <= 240:
        return normalized
    return normalized[:200] + "... [truncated; full note available in project events]"


def _generated_test_protocol(config: AppConfig | None) -> list[str]:
    if config is None or config.verification.mode != "contract" or not config.verification.generated_tests.enabled:
        return []
    return [
        "Generated-test verification is enabled.",
        "Before requesting task completion, you must:",
        "1. Identify the faulty behavior described by the task.",
        "2. Add at least one focused issue-reproduction test.",
        "3. Execute that test against the current implementation.",
        "4. Register it with register_test_candidate.",
        "5. Inspect the registration result.",
        "6. Fix the implementation if necessary.",
        "7. Run the relevant regression tests.",
        "8. Call request_task_completion.",
        "Writing or running a test without calling register_test_candidate does not satisfy this workflow.",
        "register_test_candidate is not a completion signal.",
        "A generated test does not replace the frozen verification contract.",
    ]


def _generated_test_final_checklist(config: AppConfig | None) -> list[str]:
    if config is None or config.verification.mode != "contract" or not config.verification.generated_tests.enabled:
        return []
    return [
        "Create and run a focused issue-reproduction test.",
        "Call register_test_candidate and inspect its validation result.",
        "Run regression tests, then call request_task_completion.",
        "The frozen verification contract remains authoritative.",
    ]
