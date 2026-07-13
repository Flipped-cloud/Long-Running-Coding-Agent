from __future__ import annotations

from pydantic import BaseModel

from longrun_agent.state.schema import ProjectState, TaskStatus, utc_now


class TransitionRecord(BaseModel):
    task_id: str
    old_status: TaskStatus
    new_status: TaskStatus
    reason: str
    source: str
    timestamp: str


LEGAL_TRANSITIONS = {
    TaskStatus.PENDING: {TaskStatus.READY},
    TaskStatus.READY: {TaskStatus.IN_PROGRESS, TaskStatus.FAILED},
    TaskStatus.IN_PROGRESS: {
        TaskStatus.READY,
        TaskStatus.BLOCKED,
        TaskStatus.CANDIDATE_COMPLETE,
        TaskStatus.DECOMPOSED,
        TaskStatus.FAILED,
    },
    TaskStatus.BLOCKED: {TaskStatus.READY, TaskStatus.DECOMPOSED},
}


class StateTransitionController:
    def transition(self, state: ProjectState, task_id: str, new_status: TaskStatus, *, reason: str, source: str) -> TransitionRecord:
        task = state.task_by_id(task_id)
        old_status = task.status
        if new_status not in LEGAL_TRANSITIONS.get(old_status, set()):
            raise ValueError(f"illegal task transition: {old_status} -> {new_status}")
        task.status = new_status
        task.updated_at = utc_now()
        if new_status == TaskStatus.IN_PROGRESS:
            state.active_task_id = task.id
        elif state.active_task_id == task.id:
            state.active_task_id = None
        state.updated_at = utc_now()
        return TransitionRecord(
            task_id=task.id,
            old_status=old_status,
            new_status=new_status,
            reason=reason,
            source=source,
            timestamp=utc_now(),
        )
