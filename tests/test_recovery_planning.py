import pytest

from longrun_agent.config import BoundedSearchConfig, DecompositionConfig
from longrun_agent.exceptions import ToolArgumentsProtocolError
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.planning.protocol import ChildTaskDraft, RecoveryCandidate
from longrun_agent.planning.recovery_evaluator import RecoveryCandidateEvaluator
from longrun_agent.planning.recovery_generator import RecoveryCandidateGenerator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.schema import TaskNode


def candidate(candidate_id: str, kind="retry_with_guidance", child_tasks=None):
    return {
        "id": candidate_id,
        "kind": kind,
        "description": "try narrower approach",
        "rationale": "recover",
        "expected_benefit": "progress",
        "risks": "low",
        "testability": "pytest",
        "child_tasks": child_tasks or [],
    }


class ProtocolThenResponseProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, messages, tools, tool_choice=None):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_recovery_generator_returns_candidates():
    response = ModelResponse(
        tool_calls=[
            ToolCall(
                id="g1", name="submit_recovery_candidates", arguments={"task_id": "t1", "candidates": [candidate("c1"), candidate("c2")]}
            )
        ]
    )
    generated = RecoveryCandidateGenerator(FakeModelProvider([response]), BoundedSearchConfig(candidate_count=2)).generate(
        TaskNode(id="t1", key="T1", title="T1", objective="task", acceptance_criteria=["done"]),
        "blocked",
    )
    assert [item.id for item in generated] == ["c1", "c2"]


def test_recovery_generator_retries_protocol_error():
    good = ModelResponse(
        tool_calls=[
            ToolCall(
                id="g1", name="submit_recovery_candidates", arguments={"task_id": "t1", "candidates": [candidate("c1"), candidate("c2")]}
            )
        ]
    )
    generated = RecoveryCandidateGenerator(
        FakeModelProvider([ModelResponse(), good]),
        BoundedSearchConfig(candidate_count=2, max_protocol_retries=2),
    ).generate(TaskNode(id="t1", key="T1", title="T1", objective="task", acceptance_criteria=["done"]), "blocked")
    assert len(generated) == 2


def test_recovery_generator_retries_malformed_tool_json_from_provider():
    good = ModelResponse(
        tool_calls=[
            ToolCall(
                id="g1", name="submit_recovery_candidates", arguments={"task_id": "t1", "candidates": [candidate("c1"), candidate("c2")]}
            )
        ]
    )
    provider = ProtocolThenResponseProvider(
        [ToolArgumentsProtocolError("submit_recovery_candidates", "Unterminated string", '{"candidates": ['), good]
    )

    generated = RecoveryCandidateGenerator(
        provider,
        BoundedSearchConfig(candidate_count=2, max_protocol_retries=2),
    ).generate(TaskNode(id="t1", key="T1", title="T1", objective="task", acceptance_criteria=["done"]), "blocked")

    assert [item.id for item in generated] == ["c1", "c2"]
    assert provider.calls == 2


def test_recovery_generator_fails_after_bad_protocol():
    bad = ModelResponse(tool_calls=[ToolCall(id="g1", name="submit_recovery_candidates", arguments={"task_id": "wrong", "candidates": []})])
    with pytest.raises(ValueError, match="generation failed"):
        RecoveryCandidateGenerator(FakeModelProvider([bad]), BoundedSearchConfig(max_protocol_retries=1)).generate(
            TaskNode(id="t1", key="T1", title="T1", objective="task", acceptance_criteria=["done"]),
            "blocked",
        )


def test_recovery_evaluator_filters_invalid_and_rejects_bad_selection():
    task = TaskNode(id="t1", key="T1", title="T1", objective="task", acceptance_criteria=["done"])
    valid = RecoveryCandidate.model_validate(candidate("c1"))
    second_valid = RecoveryCandidate.model_validate(candidate("c3"))
    invalid = RecoveryCandidate.model_validate(
        candidate("c2", "decompose", [ChildTaskDraft(key="C1", title="C1", objective="x", acceptance_criteria=[])])
    )
    evaluator = RecoveryCandidateEvaluator(
        FakeModelProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="s1",
                            name="select_recovery_candidate",
                            arguments={"selected_candidate_id": "c2", "scores": [], "selection_reason": "bad"},
                        )
                    ]
                )
            ]
        ),
        BoundedSearchConfig(max_protocol_retries=1),
        DecompositionConfig(),
    )
    filtered = evaluator.filter_candidates(task, [valid, invalid, second_valid])
    assert [candidate.id for candidate in filtered] == ["c1", "c3"]
    assert "c2" in evaluator.rejections
    with pytest.raises(ValueError, match="failed"):
        evaluator.select(filtered)


def test_recovery_evaluator_selects_valid_candidate():
    response = ModelResponse(
        tool_calls=[
            ToolCall(
                id="s1",
                name="select_recovery_candidate",
                arguments={
                    "selected_candidate_id": "c1",
                    "scores": [{"candidate_id": "c1", "feasibility": 5, "testability": 5, "scope_control": 4, "recovery_value": 5}],
                    "selection_reason": "best",
                },
            )
        ]
    )
    provider = FakeModelProvider([response])
    selection = RecoveryCandidateEvaluator(provider, BoundedSearchConfig(), DecompositionConfig()).select(
        [RecoveryCandidate.model_validate(candidate("c1"))]
    )
    assert selection.selected_candidate_id == "c1"
    assert provider.calls == 0


def test_recovery_evaluator_retries_malformed_tool_json_from_provider():
    good = ModelResponse(
        tool_calls=[
            ToolCall(
                id="s1",
                name="select_recovery_candidate",
                arguments={
                    "selected_candidate_id": "c1",
                    "scores": [{"candidate_id": "c1", "feasibility": 5, "testability": 5, "scope_control": 4, "recovery_value": 5}],
                    "selection_reason": "best",
                },
            )
        ]
    )
    provider = ProtocolThenResponseProvider(
        [ToolArgumentsProtocolError("select_recovery_candidate", "Unterminated string", '{"selected": '), good]
    )

    selection = RecoveryCandidateEvaluator(
        provider,
        BoundedSearchConfig(max_protocol_retries=2),
        DecompositionConfig(),
    ).select([RecoveryCandidate.model_validate(candidate("c1")), RecoveryCandidate.model_validate(candidate("c2"))])

    assert selection.selected_candidate_id == "c1"
    assert provider.calls == 2
