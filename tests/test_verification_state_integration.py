from __future__ import annotations

from longrun_agent.state.aggregation import aggregate_verified_parents
from longrun_agent.state.schema import ProjectState, TaskNode, TaskStatus
from longrun_agent.state.selector import TaskSelector
from longrun_agent.state.transitions import StateTransitionController


def _task(task_id: str, status: TaskStatus, dependencies=None, parent_id=None, depth=0):
    return TaskNode(
        id=task_id,
        key=task_id,
        title=task_id,
        objective=task_id,
        acceptance_criteria=["done"],
        status=status,
        dependencies=dependencies or [],
        parent_id=parent_id,
        depth=depth,
    )


def test_candidate_pending_verified_and_reopen_transitions() -> None:
    state = ProjectState(project_id="p", objective="o", tasks=[_task("t", TaskStatus.IN_PROGRESS)], active_task_id="t")
    transitions = StateTransitionController()
    transitions.transition(state, "t", TaskStatus.CANDIDATE_COMPLETE, reason="candidate", source="test")
    transitions.transition(state, "t", TaskStatus.VERIFICATION_PENDING, reason="verify", source="test")
    transitions.transition(state, "t", TaskStatus.REOPENED, reason="failed check", source="test")
    transitions.transition(state, "t", TaskStatus.READY, reason="retry", source="test")
    assert state.tasks[0].status == TaskStatus.READY


def test_verified_dependency_policy_and_parent_aggregation() -> None:
    parent = _task("parent", TaskStatus.DECOMPOSED)
    child = _task("child", TaskStatus.VERIFIED, parent_id="parent", depth=1)
    dependent = _task("dependent", TaskStatus.PENDING, dependencies=["child"])
    state = ProjectState(project_id="p", objective="o", tasks=[parent, child, dependent])
    TaskSelector(dependency_satisfaction="verified").update_ready_tasks(state)
    assert dependent.status == TaskStatus.READY
    assert aggregate_verified_parents(state) == ["parent"]
    assert parent.status == TaskStatus.VERIFIED


def test_old_project_state_loads_with_verification_defaults() -> None:
    state = ProjectState.model_validate(
        {
            "project_id": "p",
            "objective": "o",
            "tasks": [
                {
                    "id": "t",
                    "key": "t",
                    "title": "t",
                    "objective": "t",
                    "acceptance_criteria": ["done"],
                }
            ],
        }
    )
    assert state.project_verification_attempts == 0
    assert state.tasks[0].verification_attempts == 0
    assert state.tasks[0].reopen_count == 0
