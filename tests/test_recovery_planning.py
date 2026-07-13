import pytest

from longrun_agent.config import BoundedSearchConfig, DecompositionConfig
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
    filtered = evaluator.filter_candidates(task, [valid, invalid])
    assert [candidate.id for candidate in filtered] == ["c1"]
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
    selection = RecoveryCandidateEvaluator(FakeModelProvider([response]), BoundedSearchConfig(), DecompositionConfig()).select(
        [RecoveryCandidate.model_validate(candidate("c1"))]
    )
    assert selection.selected_candidate_id == "c1"
