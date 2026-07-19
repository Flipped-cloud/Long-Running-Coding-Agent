from __future__ import annotations

from pathlib import Path

import pytest

from longrun_agent.exceptions import KnowledgeMutationViolation
from longrun_agent.knowledge.schema import KnowledgeUseType, MemoryKind, MemoryRecord, MemoryScope, MemoryStatus, SkillRecord, SkillStep
from longrun_agent.knowledge.store import KnowledgeStore


def test_frozen_store_blocks_records_but_allows_usage_and_retrieval_telemetry(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    writable = KnowledgeStore(root)
    memory = _memory()
    skill = _skill(memory.memory_id)
    writable.save_memory(memory)
    writable.save_skill(skill)
    before = writable.records_hash()
    frozen = KnowledgeStore(root, record_mutation_policy="frozen_records")

    with pytest.raises(KnowledgeMutationViolation):
        frozen.save_memory(memory)
    with pytest.raises(KnowledgeMutationViolation):
        frozen.save_skill(skill)

    frozen.add_memory_usage(memory.memory_id, KnowledgeUseType.REFERENCED, project_id="p", task_id="t", session_id="s")
    frozen.add_skill_usage(skill.skill_id, KnowledgeUseType.HELPFUL, project_id="p", task_id="t", session_id="s")
    frozen.append_retrieval({"retrieval_id": "r1", "memory_ids": [memory.memory_id], "skill_ids": [skill.skill_id]})

    assert frozen.records_hash() == before
    assert len(frozen.read_jsonl(frozen.uses_path)) == 2
    assert len(frozen.read_jsonl(frozen.retrievals_path)) == 1


def test_snapshot_hash_changes_only_when_records_change(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    store.save_memory(_memory())
    snapshot = store.write_snapshot(tmp_path / "knowledge_snapshot.json")

    assert snapshot["records_hash"] == store.records_hash()
    assert snapshot["memory_ids"]
    assert snapshot["skill_ids"] == []


def _memory() -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.PORTABLE,
        status=MemoryStatus.ACTIVE,
        title="Input validation",
        lesson="Normalize textual input.",
        applicability_conditions=["Text input"],
        anti_conditions=["Preserve whitespace"],
    )


def _skill(memory_id: str) -> SkillRecord:
    return SkillRecord(
        title="Normalize input",
        summary="Normalize boundaries before validation.",
        scope=MemoryScope.PORTABLE,
        preconditions=["Text input"],
        anti_conditions=["Preserve whitespace"],
        procedure=[SkillStep(description="Normalize input", tool_name="write_file")],
        verification=["pytest -q"],
        source_memory_ids=[memory_id],
        confidence=0.9,
    )
