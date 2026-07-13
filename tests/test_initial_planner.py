import pytest

from longrun_agent.config import InitialPlanConfig
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.planning.initial_planner import InitialPlanner
from longrun_agent.protocol import ModelResponse, ToolCall


def plan_call(tasks):
    return ModelResponse(tool_calls=[ToolCall(id="p1", name="submit_plan", arguments={"project_summary": "summary", "tasks": tasks})])


def task(key, deps=None, criteria=None):
    return {"key": key, "title": key, "objective": f"do {key}", "acceptance_criteria": criteria or ["done"], "depends_on_keys": deps or []}


def test_initial_planner_success():
    planner = InitialPlanner(FakeModelProvider([plan_call([task("T1"), task("T2", ["T1"])])]), InitialPlanConfig())
    tasks = planner.plan(project_id="p1", objective="ship")
    assert [task.key for task in tasks] == ["T1", "T2"]
    assert tasks[1].dependencies == ["p1:T1"]


def test_initial_planner_protocol_error_retries():
    planner = InitialPlanner(
        FakeModelProvider([ModelResponse(), plan_call([task("T1"), task("T2")])]),
        InitialPlanConfig(max_protocol_retries=2),
    )
    assert len(planner.plan(project_id="p1", objective="ship")) == 2


def test_initial_planner_fails_after_protocol_retries():
    planner = InitialPlanner(FakeModelProvider([ModelResponse()]), InitialPlanConfig(max_protocol_retries=1))
    with pytest.raises(ValueError, match="failed"):
        planner.plan(project_id="p1", objective="ship")


def test_initial_planner_prompt_uses_configured_task_range():
    provider = FakeModelProvider([plan_call([task("T1"), task("T2"), task("T3"), task("T4")])])
    planner = InitialPlanner(provider, InitialPlanConfig(min_tasks=4, max_tasks=4))
    assert len(planner.plan(project_id="p1", objective="ship")) == 4


def test_initial_planner_retry_message_includes_count_and_range():
    class CapturingProvider(FakeModelProvider):
        def __init__(self):
            super().__init__([plan_call([task("T1")]), plan_call([task("T1"), task("T2")])])
            self.messages = []

        def generate(self, messages, tools):
            self.messages.append(messages)
            return super().generate(messages, tools)

    provider = CapturingProvider()
    InitialPlanner(provider, InitialPlanConfig(min_tasks=2, max_tasks=3, max_protocol_retries=2)).plan(project_id="p1", objective="ship")
    assert "Submitted task count: 1; allowed range: 2-3" in provider.messages[-1][-1]["content"]
