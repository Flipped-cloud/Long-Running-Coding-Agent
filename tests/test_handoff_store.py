from pathlib import Path

import pytest

from longrun_agent.context.schema import ConfirmedFact, EvidenceReference, HandoffRecord
from longrun_agent.state.store import ProjectStateStore


def handoff() -> HandoffRecord:
    return HandoffRecord(
        handoff_id="h1",
        project_id="p1",
        task_id="t1",
        session_id="s1",
        source_segment_id=1,
        target_segment_id=2,
        plan_version=1,
        task_objective="do it",
        acceptance_criteria=["done"],
        evidence_references=[EvidenceReference(evidence_id="e1", type="tool", step=1)],
        confirmed_progress=[ConfirmedFact(statement="read file", evidence_ids=["e1"], source_step=1)],
    )


def test_handoff_rejects_missing_evidence_id():
    with pytest.raises(ValueError):
        HandoffRecord(
            handoff_id="bad",
            project_id="p1",
            task_id="t1",
            session_id="s1",
            source_segment_id=1,
            target_segment_id=2,
            plan_version=1,
            task_objective="do it",
            confirmed_progress=[ConfirmedFact(statement="unsupported", evidence_ids=["missing"], source_step=1)],
        )


def test_handoff_store_roundtrip(tmp_path: Path):
    store = ProjectStateStore(tmp_path / "projects", workspace_root=tmp_path / "workspace")
    record = handoff()
    store.save_handoff(record)

    loaded = store.load_handoff("p1", "h1")

    assert loaded == record
    assert store.handoff_path("p1", "h1").exists()
