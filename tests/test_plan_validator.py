import pytest

from longrun_agent.planning.protocol import ChildTaskDraft, TaskDraft
from longrun_agent.planning.validator import PlanValidator
from longrun_agent.state.schema import TaskNode


def draft(key: str, deps=None, criteria=None) -> TaskDraft:
    return TaskDraft(
        key=key,
        title=key,
        objective=f"do {key}",
        acceptance_criteria=["done"] if criteria is None else criteria,
        depends_on_keys=deps or [],
    )


def test_plan_validator_accepts_valid_plan():
    drafts = [draft("T1"), draft("T2", ["T1"])]
    assert PlanValidator().validate_initial(drafts, min_tasks=2, max_tasks=8) == drafts


def test_plan_validator_rejects_missing_acceptance_criteria():
    with pytest.raises(ValueError, match="acceptance"):
        PlanValidator().validate_initial([draft("T1", criteria=[]), draft("T2")], min_tasks=2, max_tasks=8)


def test_plan_validator_rejects_unknown_dependency():
    with pytest.raises(ValueError, match="unknown"):
        PlanValidator().validate_initial([draft("T1", ["missing"]), draft("T2")], min_tasks=2, max_tasks=8)


def test_plan_validator_rejects_dependency_cycle():
    with pytest.raises(ValueError, match="cycle"):
        PlanValidator().validate_initial([draft("T1", ["T2"]), draft("T2", ["T1"])], min_tasks=2, max_tasks=8)


def test_child_validator_rejects_max_depth_and_cycle():
    parent = TaskNode(id="t1", key="T1", title="T1", objective="parent", acceptance_criteria=["done"], depth=3)
    with pytest.raises(ValueError, match="depth"):
        PlanValidator().validate_children(
            parent,
            [ChildTaskDraft(key="C1", title="C1", objective="specific", acceptance_criteria=["done"])],
            min_children=1,
            max_children=2,
            max_depth=3,
        )

    parent.depth = 0
    with pytest.raises(ValueError, match="cycle"):
        PlanValidator().validate_children(
            parent,
            [
                ChildTaskDraft(key="C1", title="C1", objective="specific 1", acceptance_criteria=["done"], depends_on_child_keys=["C2"]),
                ChildTaskDraft(key="C2", title="C2", objective="specific 2", acceptance_criteria=["done"], depends_on_child_keys=["C1"]),
            ],
            min_children=2,
            max_children=3,
            max_depth=3,
        )
