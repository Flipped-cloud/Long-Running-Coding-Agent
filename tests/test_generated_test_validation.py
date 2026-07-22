from __future__ import annotations

import sys
from pathlib import Path

import pytest

from longrun_agent.control.channel import ControlSignalType, TaskControlChannel
from longrun_agent.control.tools import (
    RegisterTestCandidateArgs,
    RegisterTestCandidateTool,
    RequestTaskCompletionArgs,
    RequestTaskCompletionTool,
)
from longrun_agent.protocol import ErrorType
from longrun_agent.state.schema import TaskNode, TaskStatus
from longrun_agent.tools.base import ToolContext
from longrun_agent.verification.generated_tests import TestCandidateValidator as CandidateValidator
from longrun_agent.verification.generated_tests import register_test_candidate
from longrun_agent.verification.runner import VerificationRunner
from longrun_agent.verification.schema import TestTransition as Transition
from longrun_agent.verification.snapshot import CopySnapshotProvider


def test_generated_test_f2p_is_valid_but_not_authoritative(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    snapshots = CopySnapshotProvider(workspace, tmp_path / "store")
    snapshots.create_baseline()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    test_path = workspace / "test_candidate.py"
    test_path.write_text("import app\nassert app.VALUE == 1\n", encoding="utf-8")
    candidate = register_test_candidate(
        workspace=workspace,
        task_id="t1",
        session_id="s1",
        paths=["test_candidate.py"],
        command_argv=[sys.executable, "test_candidate.py"],
        issue_behavior="VALUE must become one",
        expected_failure_reason="baseline value is zero",
    )
    candidate_root, _manifest = snapshots.create_candidate()
    result = CandidateValidator(snapshots, VerificationRunner(tmp_path / "artifacts")).validate(candidate, candidate_root)
    assert result.valid
    assert result.transition == Transition.F2P
    assert not result.valid_but_irrelevant


@pytest.mark.parametrize(
    ("predicate", "transition", "valid", "irrelevant", "rejection"),
    [
        ("app.VALUE == 1", Transition.F2P, True, False, None),
        ("app.VALUE >= 0", Transition.P2P, True, True, "valid_but_irrelevant"),
        ("app.VALUE == 2", Transition.F2F, False, False, "invalid_or_unfixed"),
        ("app.VALUE == 0", Transition.P2F, False, False, "harmful_test"),
    ],
)
def test_generated_test_transition_semantics(
    tmp_path: Path,
    predicate: str,
    transition: Transition,
    valid: bool,
    irrelevant: bool,
    rejection: str | None,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    snapshots = CopySnapshotProvider(workspace, tmp_path / "store")
    snapshots.create_baseline()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "test_candidate.py").write_text(
        f"import app\nraise SystemExit(0 if {predicate} else 1)\n",
        encoding="utf-8",
    )
    candidate = register_test_candidate(
        workspace=workspace,
        task_id="t1",
        session_id="s1",
        paths=["test_candidate.py"],
        command_argv=[sys.executable, "test_candidate.py"],
        issue_behavior="validate candidate behavior",
        expected_failure_reason="the baseline does not implement the behavior",
    )
    candidate_root, _manifest = snapshots.create_candidate()

    result = CandidateValidator(snapshots, VerificationRunner(tmp_path / "artifacts")).validate(candidate, candidate_root)

    assert result.transition == transition
    assert result.valid is valid
    assert result.valid_but_irrelevant is irrelevant
    assert (rejection in result.rejection_reasons) if rejection else not result.rejection_reasons


def test_completion_gate_rejects_missing_candidate_without_changing_task_status(tmp_path: Path) -> None:
    channel = TaskControlChannel(
        require_test_candidate_before_completion=True,
        minimum_registered_candidates=1,
        minimum_valid_candidates=1,
    )
    task = TaskNode(id="task", key="task", title="task", objective="fix", acceptance_criteria=["verified"])
    context = ToolContext(tmp_path, control_channel=channel)

    result = RequestTaskCompletionTool().execute(
        "complete",
        RequestTaskCompletionArgs(summary="done", acceptance_criteria_addressed=["verified"]),
        context,
    )

    assert not result.success
    assert result.error_type == ErrorType.GENERATED_TEST_REQUIREMENT_UNMET
    assert result.retryable
    assert result.metadata["registered_candidates"] == result.metadata["valid_candidates"] == 0
    assert channel.terminal_signal is None
    assert task.status == TaskStatus.PENDING


@pytest.mark.parametrize(("predicate", "valid"), [("app.VALUE == 2", False), ("app.VALUE == 1", True)])
def test_registered_candidate_returns_validation_feedback_and_gates_completion(
    tmp_path: Path,
    predicate: str,
    valid: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    snapshots = CopySnapshotProvider(workspace, tmp_path / "store")
    snapshots.create_baseline()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    test_path = workspace / "test_candidate.py"
    test_path.write_text(f"import app\nassert {predicate}\n", encoding="utf-8")

    def validate(candidate):
        candidate_root, _manifest = snapshots.create_candidate()
        try:
            return CandidateValidator(snapshots, VerificationRunner(tmp_path / "artifacts")).validate(candidate, candidate_root)
        finally:
            snapshots.cleanup(candidate_root)

    channel = TaskControlChannel(
        workspace=workspace,
        task_id="task",
        session_id="session",
        max_test_candidates=3,
        require_test_candidate_before_completion=True,
        minimum_registered_candidates=1,
        minimum_valid_candidates=1,
        candidate_validator=validate,
    )
    context = ToolContext(workspace, control_channel=channel)
    registration = RegisterTestCandidateTool().execute(
        "register",
        RegisterTestCandidateArgs(
            paths=["test_candidate.py"],
            command_argv=[sys.executable, "test_candidate.py"],
            issue_behavior="VALUE must become one",
            expected_failure_reason="baseline value is zero",
        ),
        context,
    )

    assert registration.success
    assert registration.metadata["valid"] is valid
    assert registration.metadata["transition"] == ("F2P" if valid else "F2F")
    assert registration.metadata["recommended_next_action"]

    completion = RequestTaskCompletionTool().execute(
        "complete",
        RequestTaskCompletionArgs(summary="done", acceptance_criteria_addressed=["verified"]),
        context,
    )
    assert completion.success is valid
    if valid:
        assert channel.terminal_signal is not None
        assert channel.terminal_signal.type == ControlSignalType.COMPLETION_REQUEST
    else:
        assert channel.terminal_signal is None
        assert completion.error_type == ErrorType.GENERATED_TEST_REQUIREMENT_UNMET
