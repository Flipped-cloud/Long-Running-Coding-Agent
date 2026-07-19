from __future__ import annotations

from longrun_agent.knowledge.schema import KnowledgeConflictRecord, MemoryRecord, ReflectionCandidate


class MemoryConflictDetector:
    def __init__(self, store):
        self.store = store

    def find_conflicts_for_candidate(self, candidate: ReflectionCandidate) -> list[KnowledgeConflictRecord]:
        conflicts: list[KnowledgeConflictRecord] = []
        candidate_actions = {item.lower() for item in candidate.recommended_actions}
        candidate_avoid = {item.lower() for item in candidate.avoid_actions}
        for memory in self.store.list_memories({"status": "active"}):
            memory_actions = {item.lower() for item in memory.recommended_actions}
            memory_avoid = {item.lower() for item in memory.avoid_actions}
            if candidate_actions & memory_avoid or candidate_avoid & memory_actions:
                record = KnowledgeConflictRecord(
                    left_id=candidate.candidate_id,
                    right_id=memory.memory_id,
                    conflict_type="contradictory_recommendation",
                    severity="high",
                    reason="recommended action conflicts with existing avoid action",
                )
                self.store.append_conflict(record)
                conflicts.append(record)
        return conflicts

    def conflict_risk(self, memory: MemoryRecord) -> float:
        conflicts = self.store.read_jsonl(self.store.conflicts_path)
        return min(1.0, sum(1 for item in conflicts if memory.memory_id in {item.get("left_id"), item.get("right_id")}) / 3)
