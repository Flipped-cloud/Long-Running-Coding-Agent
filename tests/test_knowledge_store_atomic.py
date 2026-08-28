import os
from pathlib import Path

from longrun_agent.knowledge.schema import MemoryKind, MemoryRecord, MemoryScope, MemoryStatus
from longrun_agent.knowledge.store import KnowledgeStore


def test_knowledge_store_retries_transient_permission_error(tmp_path: Path, monkeypatch) -> None:
    store = KnowledgeStore(tmp_path / "knowledge")
    actual_replace = os.replace
    calls = 0

    def transient_replace(source, target):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient Windows file lock")
        actual_replace(source, target)

    monkeypatch.setattr("longrun_agent.filesystem.os.replace", transient_replace)
    memory = MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.PORTABLE,
        status=MemoryStatus.ACTIVE,
        title="Retry atomic replace",
        lesson="Retry transient sharing violations.",
    )

    store.save_memory(memory)

    assert calls == 3
    assert store.load_memory(memory.memory_id).title == "Retry atomic replace"
