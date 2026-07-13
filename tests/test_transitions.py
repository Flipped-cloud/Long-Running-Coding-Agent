import pytest

from longrun_agent.state.schema import ProjectState, TaskNode, TaskStatus
from longrun_agent.state.transitions import StateTransitionController


def make_state() -> ProjectState:
    return ProjectState(
        project_id="p1",
        objective="ship",
        tasks=[TaskNode(id="t1", key="T1", title="T1", objective="do it", acceptance_criteria=["done"])],
    )


def test_legal_state_transition():
    state = make_state()
    record = StateTransitionController().transition(state, "t1", TaskStatus.READY, reason="deps", source="test")
    assert record.old_status == TaskStatus.PENDING
    assert record.new_status == TaskStatus.READY
    assert state.tasks[0].status == TaskStatus.READY


def test_illegal_state_transition():
    state = make_state()
    with pytest.raises(ValueError, match="illegal"):
        StateTransitionController().transition(state, "t1", TaskStatus.CANDIDATE_COMPLETE, reason="bad", source="test")
