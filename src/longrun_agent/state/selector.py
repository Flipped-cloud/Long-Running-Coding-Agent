from __future__ import annotations

from longrun_agent.state.schema import ProjectState, ProjectStatus, TaskNode, TaskStatus, utc_now
from longrun_agent.state.transitions import StateTransitionController


class TaskSelector:
    def __init__(self, transitions: StateTransitionController | None = None):
        self.transitions = transitions or StateTransitionController()

    def update_ready_tasks(self, state: ProjectState) -> list[str]:
        ready: list[str] = []
        completed = {task.id for task in state.tasks if task.status == TaskStatus.CANDIDATE_COMPLETE}
        for task in state.tasks:
            if task.status == TaskStatus.PENDING and all(dependency in completed for dependency in task.dependencies):
                self.transitions.transition(state, task.id, TaskStatus.READY, reason="dependencies candidate complete", source="selector")
                ready.append(task.id)
        return ready

    def select_next(self, state: ProjectState) -> TaskNode | None:
        in_progress = [task for task in state.tasks if task.status == TaskStatus.IN_PROGRESS]
        if len(in_progress) > 1:
            raise ValueError("more than one task is IN_PROGRESS")
        if in_progress:
            state.active_task_id = in_progress[0].id
            return in_progress[0]
        self.update_ready_tasks(state)
        ready = sorted(
            [task for task in state.tasks if task.status == TaskStatus.READY],
            key=lambda task: (task.depth, task.created_at, task.id),
        )
        if ready:
            state.active_task_id = ready[0].id
            return ready[0]
        leaves = state.leaf_tasks()
        if leaves and all(task.status == TaskStatus.CANDIDATE_COMPLETE for task in leaves):
            state.status = ProjectStatus.CANDIDATE_COMPLETE
            state.updated_at = utc_now()
            return None
        if any(task.status == TaskStatus.BLOCKED for task in state.tasks):
            state.status = ProjectStatus.BLOCKED
            state.updated_at = utc_now()
            return None
        if any(task.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS} for task in state.tasks):
            raise ValueError("no reachable ready task; dependency cycle or unsatisfied dependency")
        return None
