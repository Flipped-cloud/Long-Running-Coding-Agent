from longrun_agent.state.aggregation import aggregate_candidate_complete_parents, project_statistics
from longrun_agent.state.schema import ProjectState, TaskNode, TaskStatus


def test_parent_aggregates_when_children_candidate_complete():
    parent = TaskNode(id="t1", key="T1", title="T1", objective="parent", acceptance_criteria=["done"], status=TaskStatus.DECOMPOSED)
    child1 = TaskNode(
        id="t1-1",
        key="T1.1",
        title="T1.1",
        objective="child",
        acceptance_criteria=["done"],
        parent_id="t1",
        depth=1,
        status=TaskStatus.CANDIDATE_COMPLETE,
    )
    child2 = TaskNode(
        id="t1-2",
        key="T1.2",
        title="T1.2",
        objective="child",
        acceptance_criteria=["done"],
        parent_id="t1",
        depth=1,
        status=TaskStatus.CANDIDATE_COMPLETE,
    )
    state = ProjectState(project_id="p1", objective="ship", tasks=[parent, child1, child2])
    assert aggregate_candidate_complete_parents(state) == ["t1"]
    assert state.task_by_id("t1").status == TaskStatus.CANDIDATE_COMPLETE


def test_project_statistics_fields_are_present():
    state = ProjectState(project_id="p1", objective="ship", tasks=[])
    stats = project_statistics(state, total_tool_calls=2, total_tokens=3)
    assert stats["candidate_completed_tasks"] == 0
    assert stats["blocked_tasks"] == 0
    assert stats["decomposition_count"] == 0
    assert stats["max_task_depth"] == 0
    assert stats["plan_revision_count"] == 0
    assert stats["recovery_candidate_count"] == 0
    assert stats["sessions_without_terminal_signal"] == 0
    assert stats["project_sessions"] == 0
    assert stats["total_tool_calls"] == 2
    assert stats["total_tokens"] == 3
