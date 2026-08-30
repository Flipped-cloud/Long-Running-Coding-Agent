import pytest

from longrun_agent.config import DecompositionConfig
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.planning.decomposer import AsNeededDecomposer
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.schema import TaskNode


class CapturingProvider(FakeModelProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.messages = []

    def generate(self, messages, tools):
        self.messages.append(messages)
        return super().generate(messages, tools)


def child(key, deps=None):
    return {"key": key, "title": key, "objective": f"specific {key}", "acceptance_criteria": ["done"], "depends_on_child_keys": deps or []}


def decomposition_call(parent_id, children):
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id="d1", name="submit_decomposition", arguments={"parent_task_id": parent_id, "reason": "blocked", "children": children}
            )
        ]
    )


def test_decomposer_success():
    parent = TaskNode(id="t1", key="T1", title="T1", objective="parent", acceptance_criteria=["done"])
    provider = CapturingProvider([decomposition_call("t1", [child("C1"), child("C2", ["C1"])])])
    decomposer = AsNeededDecomposer(provider, DecompositionConfig(min_children=2, max_children=4, max_depth=3))
    children = decomposer.decompose(parent, "blocked")
    assert [task.parent_id for task in children] == ["t1", "t1"]
    assert children[1].dependencies == ["t1:C1"]
    assert "between 2 and 4 child tasks" in provider.messages[0][0]["content"]


def test_decomposer_rejects_child_cycle():
    parent = TaskNode(id="t1", key="T1", title="T1", objective="parent", acceptance_criteria=["done"])
    decomposer = AsNeededDecomposer(
        FakeModelProvider([decomposition_call("t1", [child("C1", ["C2"]), child("C2", ["C1"])])]),
        DecompositionConfig(max_protocol_retries=1),
    )
    with pytest.raises(ValueError, match="failed"):
        decomposer.decompose(parent, "blocked")
