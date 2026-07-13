import pytest

from longrun_agent.state.schema import ProjectState, ProjectStatus, TaskNode, TaskStatus
from longrun_agent.state.selector import TaskSelector


def node(task_id: str, key: str, status=TaskStatus.PENDING, dependencies=None, depth=0) -> TaskNode:
    return TaskNode(
        id=task_id,
        key=key,
        title=key,
        objective=key,
        acceptance_criteria=["done"],
        status=status,
        dependencies=dependencies or [],
        depth=depth,
    )


def test_dependency_transitions_to_ready_and_selector_order():
    state = ProjectState(
        project_id="p1",
        objective="ship",
        tasks=[node("t1", "T1", TaskStatus.CANDIDATE_COMPLETE), node("t2", "T2", dependencies=["t1"]), node("t3", "T3")],
    )
    selected = TaskSelector().select_next(state)
    assert selected.id == "t2"
    assert state.task_by_id("t2").status == TaskStatus.READY


def test_selector_all_leaf_candidate_complete_marks_project_complete():
    state = ProjectState(project_id="p1", objective="ship", tasks=[node("t1", "T1", TaskStatus.CANDIDATE_COMPLETE)])
    assert TaskSelector().select_next(state) is None
    assert state.status == ProjectStatus.CANDIDATE_COMPLETE


def test_selector_returns_existing_in_progress():
    state = ProjectState(project_id="p1", objective="ship", active_task_id="t1", tasks=[node("t1", "T1", TaskStatus.IN_PROGRESS)])
    assert TaskSelector().select_next(state).id == "t1"


def test_selector_marks_project_blocked_when_no_ready_and_blocked_exists():
    state = ProjectState(project_id="p1", objective="ship", tasks=[node("t1", "T1", TaskStatus.BLOCKED)])
    assert TaskSelector().select_next(state) is None
    assert state.status == ProjectStatus.BLOCKED


def test_selector_reports_unreachable_pending_task():
    state = ProjectState(
        project_id="p1",
        objective="ship",
        tasks=[node("t1", "T1", TaskStatus.FAILED), node("t2", "T2", dependencies=["t1"])],
    )
    with pytest.raises(ValueError, match="no reachable"):
        TaskSelector().select_next(state)


def test_selector_rejects_multiple_in_progress():
    with pytest.raises(ValueError):
        ProjectState(
            project_id="p1", objective="ship", tasks=[node("t1", "T1", TaskStatus.IN_PROGRESS), node("t2", "T2", TaskStatus.IN_PROGRESS)]
        )
