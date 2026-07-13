from pathlib import Path

from longrun_agent.model.base import ModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.schema import ProjectStatus, TaskStatus
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import completion, config, submit_plan


class CapturingProvider(ModelProvider):
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.messages: list[list[dict]] = []

    def generate(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("responses exhausted")
        return self.responses.pop(0)


def test_retry_with_guidance_requeues_task_and_guides_next_session(tmp_path: Path):
    cfg = config(tmp_path, mode="adaptive_search", max_sessions=2)
    responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="b1",
                    name="report_blocker",
                    arguments={"reason": "needs retry", "attempted_actions": ["read"], "decomposition_recommended": True},
                )
            ]
        ),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="g1",
                    name="submit_recovery_candidates",
                    arguments={
                        "task_id": "T1",
                        "candidates": [
                            {
                                "id": "retry",
                                "kind": "retry_with_guidance",
                                "description": "Retry after reading the exact file first.",
                                "rationale": "recover",
                                "expected_benefit": "progress",
                                "risks": "low",
                                "testability": "control signal",
                                "child_tasks": [],
                            },
                            {
                                "id": "block",
                                "kind": "mark_blocked",
                                "description": "blocked",
                                "rationale": "stop",
                                "expected_benefit": "none",
                                "risks": "high",
                                "testability": "none",
                                "child_tasks": [],
                            },
                        ],
                    },
                )
            ]
        ),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="s1",
                    name="select_recovery_candidate",
                    arguments={
                        "selected_candidate_id": "retry",
                        "scores": [{"candidate_id": "retry", "feasibility": 5, "testability": 4, "scope_control": 5, "recovery_value": 5}],
                        "selection_reason": "retry is enough",
                    },
                )
            ]
        ),
        completion("c1"),
    ]
    provider = CapturingProvider(responses)
    outcome = ProjectOrchestrator(cfg, provider, project_id="retry-1").start("ship")
    state = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root).load("retry-1")
    task = state.task_by_id("retry-1:T1")
    assert outcome.status != ProjectStatus.BLOCKED.value
    assert task.status == TaskStatus.CANDIDATE_COMPLETE
    assert task.attempts == 2
    assert any("Recovery guidance: Retry after reading the exact file first." == note for note in task.progress_notes)
    assert any(
        "Recovery guidance: Retry after reading the exact file first." in (item.get("content") or "")
        for messages in provider.messages
        for item in messages
    )
