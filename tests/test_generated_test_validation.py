from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
