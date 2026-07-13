import pytest

from longrun_agent.state.schema import ProjectState, TaskNode, TaskStatus


def task(task_id: str, key: str, **kwargs) -> TaskNode:
    return TaskNode(
        id=task_id,
        key=key,
        title=key,
        objective=f"do {key}",
        acceptance_criteria=["done"],
        **kwargs,
    )


def test_project_state_serialization_roundtrip():
    state = ProjectState(project_id="p1", objective="ship", tasks=[task("t1", "T1")])
    restored = ProjectState.model_validate_json(state.model_dump_json())
    assert restored.project_id == "p1"
    assert restored.tasks[0].status == TaskStatus.PENDING


def test_task_key_must_be_unique():
    with pytest.raises(ValueError, match="keys"):
        ProjectState(project_id="p1", objective="ship", tasks=[task("t1", "T1"), task("t2", "T1")])


def test_parent_depth_must_match():
    with pytest.raises(ValueError, match="depth"):
        ProjectState(project_id="p1", objective="ship", tasks=[task("t1", "T1"), task("t2", "T2", parent_id="t1", depth=0)])
