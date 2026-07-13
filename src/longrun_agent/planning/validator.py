from __future__ import annotations

from longrun_agent.planning.protocol import ChildTaskDraft, TaskDraft
from longrun_agent.state.schema import TaskNode


def _assert_acyclic(keys: list[str], dependencies: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError("plan contains a dependency cycle")
        if key in visited:
            return
        visiting.add(key)
        for dependency in dependencies.get(key, []):
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in keys:
        visit(key)


class PlanValidator:
    def validate_initial(self, drafts: list[TaskDraft], *, min_tasks: int, max_tasks: int) -> list[TaskDraft]:
        if not min_tasks <= len(drafts) <= max_tasks:
            raise ValueError("initial plan task count is outside configured bounds")
        keys = [draft.key for draft in drafts]
        if len(keys) != len(set(keys)):
            raise ValueError("task keys must be unique")
        key_set = set(keys)
        dependencies: dict[str, list[str]] = {}
        for draft in drafts:
            if not draft.acceptance_criteria:
                raise ValueError("task acceptance_criteria must not be empty")
            for dependency in draft.depends_on_keys:
                if dependency not in key_set:
                    raise ValueError(f"depends_on_keys references unknown key: {dependency}")
            dependencies[draft.key] = list(draft.depends_on_keys)
        _assert_acyclic(keys, dependencies)
        return drafts

    def to_task_nodes(self, project_id: str, drafts: list[TaskDraft]) -> list[TaskNode]:
        id_by_key = {draft.key: f"{project_id}:{draft.key}" for draft in drafts}
        return [
            TaskNode(
                id=id_by_key[draft.key],
                key=draft.key,
                title=draft.title,
                objective=draft.objective,
                acceptance_criteria=draft.acceptance_criteria,
                dependencies=[id_by_key[key] for key in draft.depends_on_keys],
            )
            for draft in drafts
        ]

    def validate_children(
        self,
        parent: TaskNode,
        drafts: list[ChildTaskDraft],
        *,
        min_children: int,
        max_children: int,
        max_depth: int,
    ) -> list[ChildTaskDraft]:
        if parent.depth >= max_depth:
            raise ValueError("maximum decomposition depth reached")
        if not min_children <= len(drafts) <= max_children:
            raise ValueError("child task count is outside configured bounds")
        keys = [draft.key for draft in drafts]
        if len(keys) != len(set(keys)):
            raise ValueError("child task keys must be unique")
        key_set = set(keys)
        dependencies: dict[str, list[str]] = {}
        for draft in drafts:
            if draft.objective.strip() == parent.objective.strip():
                raise ValueError("child task must be more specific than parent")
            if not draft.acceptance_criteria:
                raise ValueError("child acceptance_criteria must not be empty")
            for dependency in draft.depends_on_child_keys:
                if dependency not in key_set:
                    raise ValueError(f"depends_on_child_keys references unknown key: {dependency}")
            dependencies[draft.key] = list(draft.depends_on_child_keys)
        _assert_acyclic(keys, dependencies)
        return drafts

    def children_to_task_nodes(self, parent: TaskNode, drafts: list[ChildTaskDraft]) -> list[TaskNode]:
        id_by_key = {draft.key: f"{parent.id}:{draft.key}" for draft in drafts}
        return [
            TaskNode(
                id=id_by_key[draft.key],
                key=draft.key,
                title=draft.title,
                objective=draft.objective,
                acceptance_criteria=draft.acceptance_criteria,
                dependencies=[id_by_key[key] for key in draft.depends_on_child_keys],
                parent_id=parent.id,
                depth=parent.depth + 1,
            )
            for draft in drafts
        ]
