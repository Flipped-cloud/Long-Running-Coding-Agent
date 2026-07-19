from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from longrun_agent.exceptions import ConfigurationError, KnowledgeMutationViolation
from longrun_agent.knowledge.schema import (
    ExperienceEvidencePack,
    KnowledgeConflictRecord,
    KnowledgeUseRecord,
    KnowledgeUseType,
    MemoryRecord,
    MemoryStatus,
    SkillRecord,
    SkillStatus,
    utc_now,
)
from longrun_agent.knowledge.telemetry import append_jsonl, knowledge_event
from longrun_agent.tools.path_guard import is_inside_path


class KnowledgeUseWriteResult:
    def __init__(self, record: KnowledgeUseRecord, updated_record: MemoryRecord | SkillRecord, *, created: bool):
        self.record = record
        self.use_record = record
        self.updated_record = updated_record
        self.created = created

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.record, name):
            return getattr(self.record, name)
        return getattr(self.updated_record, name)


class KnowledgeStore:
    def __init__(
        self,
        root: Path,
        *,
        workspace_root: Path | None = None,
        atomic_write: bool = True,
        record_mutation_policy: str = "read_write",
    ):
        self.root = root.resolve()
        self.atomic_write = atomic_write
        self.record_mutation_policy = record_mutation_policy
        if workspace_root is not None and is_inside_path(self.root, workspace_root.resolve()):
            raise ConfigurationError("knowledge root must not be inside the agent workspace")
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def memories_dir(self) -> Path:
        return self.root / "memories"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def retrievals_path(self) -> Path:
        return self.root / "retrievals.jsonl"

    @property
    def uses_path(self) -> Path:
        return self.root / "uses.jsonl"

    @property
    def conflicts_path(self) -> Path:
        return self.root / "conflicts.jsonl"

    @property
    def rejections_dir(self) -> Path:
        return self.root / "rejections"

    def memory_path(self, memory_id: str) -> Path:
        return self.memories_dir / f"{memory_id}.json"

    def skill_path(self, skill_id: str) -> Path:
        return self.skills_dir / skill_id / "skill.json"

    def episode_path(self, project_dir: Path, episode_id: str) -> Path:
        return project_dir / "knowledge" / "episodes" / f"{episode_id}.json"

    def save_episode(self, project_dir: Path, pack: ExperienceEvidencePack) -> Path:
        pack = ExperienceEvidencePack.model_validate(pack.model_dump())
        path = self.episode_path(project_dir, pack.episode_id)
        self._atomic_write_json(path, pack.model_dump(mode="json"))
        self.append_event(
            "episode_created", project_id=pack.project_id, task_id=pack.task_id, session_id=pack.session_id, episode_id=pack.episode_id
        )
        return path

    def save_reflection_rejection(self, candidate, pack: ExperienceEvidencePack, result) -> Path:
        evidence = pack.evidence_by_id()
        payload = {
            "candidate_id": candidate.candidate_id,
            "project_id": pack.project_id,
            "task_id": pack.task_id,
            "session_id": pack.session_id,
            "candidate": candidate.model_dump(mode="json"),
            "cited_evidence_ids": list(candidate.evidence_ids),
            "cited_evidence": [
                {
                    "evidence_id": evidence_id,
                    "event_type": evidence[evidence_id].event_type,
                    "command": evidence[evidence_id].command,
                    "path": evidence[evidence_id].path,
                    "success": evidence[evidence_id].success,
                    "exit_code": evidence[evidence_id].exit_code,
                    "output_excerpt": evidence[evidence_id].output_excerpt,
                    "artifact_path": evidence[evidence_id].artifact_path,
                }
                for evidence_id in candidate.evidence_ids
                if evidence_id in evidence
            ],
            "rejection_reasons": list(result.rejection_reasons),
            "final_scope": result.final_scope.value if result.final_scope else None,
            "timestamp": utc_now(),
        }
        path = self.rejections_dir / f"{candidate.candidate_id}.json"
        self._atomic_write_json(path, payload)
        return path

    def save_memory(self, record: MemoryRecord) -> None:
        self._guard_record_mutation("save_memory", record.memory_id)
        record = MemoryRecord.model_validate(record.model_dump())
        self._atomic_write_json(self.memory_path(record.memory_id), record.model_dump(mode="json"))
        self.append_event(
            "memory_activated"
            if record.status == MemoryStatus.ACTIVE
            else "memory_quarantined"
            if record.status == MemoryStatus.QUARANTINED
            else "memory_saved",
            memory_id=record.memory_id,
            status_after=record.status.value,
            evidence_ids=record.evidence_ids,
        )

    def load_memory(self, memory_id: str) -> MemoryRecord:
        return MemoryRecord.model_validate_json(self.memory_path(memory_id).read_text(encoding="utf-8"))

    def list_memories(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for path in sorted(self.memories_dir.glob("*.json")):
            try:
                record = MemoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, ValidationError):
                continue
            if _matches(record, filters or {}):
                records.append(record)
        return records

    def update_memory_status(self, memory_id: str, status: MemoryStatus, *, reason: str = "") -> MemoryRecord:
        self._guard_record_mutation("update_memory_status", memory_id)
        record = self.load_memory(memory_id)
        before = record.status
        record.status = status
        record.updated_at = utc_now()
        self.save_memory(record)
        self.append_event(
            f"memory_{status.value}",
            memory_id=memory_id,
            status_before=before.value,
            status_after=status.value,
            reason=reason,
        )
        return record

    def find_duplicate_memory(self, candidate: MemoryRecord) -> MemoryRecord | None:
        signature = _memory_signature(candidate)
        for record in self.list_memories({"status": MemoryStatus.ACTIVE.value}):
            if _memory_signature(record) == signature:
                return record
        return None

    def add_memory_usage(
        self,
        memory_id: str,
        use_type: KnowledgeUseType,
        *,
        project_id: str = "",
        task_id: str = "",
        session_id: str = "",
        retrieval_id: str | None = None,
        reason: str = "",
    ) -> KnowledgeUseWriteResult:
        record = self.load_memory(memory_id)
        use_record = KnowledgeUseRecord(
            project_id=project_id or record.project_id or "",
            task_id=task_id,
            session_id=session_id,
            retrieval_id=retrieval_id,
            memory_id=memory_id,
            use_type=use_type,
            reason=reason,
        )
        if self._use_exists(use_record):
            return KnowledgeUseWriteResult(use_record, record, created=False)
        if self.record_mutation_policy != "frozen_records":
            _apply_usage(record, use_type)
            self.save_memory(record)
        self.append_use(use_record)
        return KnowledgeUseWriteResult(use_record, record, created=True)

    def save_skill(self, record: SkillRecord) -> None:
        self._guard_record_mutation("save_skill", record.skill_id)
        record = SkillRecord.model_validate(record.model_dump())
        self._atomic_write_json(self.skill_path(record.skill_id), record.model_dump(mode="json"))
        self.append_event(
            "skill_promoted" if record.status == SkillStatus.ACTIVE else "skill_saved",
            skill_id=record.skill_id,
            status_after=record.status.value,
        )

    def load_skill(self, skill_id: str) -> SkillRecord:
        return SkillRecord.model_validate_json(self.skill_path(skill_id).read_text(encoding="utf-8"))

    def list_skills(self, filters: dict[str, Any] | None = None) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        for path in sorted(self.skills_dir.glob("*/skill.json")):
            try:
                record = SkillRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, ValidationError):
                continue
            if _matches(record, filters or {}):
                records.append(record)
        return records

    def update_skill_status(self, skill_id: str, status: SkillStatus, *, reason: str = "") -> SkillRecord:
        self._guard_record_mutation("update_skill_status", skill_id)
        record = self.load_skill(skill_id)
        before = record.status
        record.status = status
        record.updated_at = utc_now()
        self.save_skill(record)
        self.append_event(
            f"skill_{status.value}",
            skill_id=skill_id,
            status_before=before.value,
            status_after=status.value,
            reason=reason,
        )
        return record

    def add_skill_usage(
        self,
        skill_id: str,
        use_type: KnowledgeUseType,
        *,
        project_id: str = "",
        task_id: str = "",
        session_id: str = "",
        retrieval_id: str | None = None,
        reason: str = "",
    ) -> KnowledgeUseWriteResult:
        record = self.load_skill(skill_id)
        use_record = KnowledgeUseRecord(
            project_id=project_id,
            task_id=task_id,
            session_id=session_id,
            retrieval_id=retrieval_id,
            skill_id=skill_id,
            use_type=use_type,
            reason=reason,
        )
        if self._use_exists(use_record):
            return KnowledgeUseWriteResult(use_record, record, created=False)
        if self.record_mutation_policy != "frozen_records":
            _apply_usage(record, use_type)
            self.save_skill(record)
        self.append_use(use_record)
        return KnowledgeUseWriteResult(use_record, record, created=True)

    def find_skill_by_derivation(
        self,
        *,
        source_memory_ids: list[str],
        statuses: set[SkillStatus] | None = None,
    ) -> SkillRecord | None:
        from longrun_agent.knowledge.schema import skill_derivation_key

        key = skill_derivation_key(source_memory_ids)
        if not key:
            return None
        allowed = statuses or {SkillStatus.DRAFT, SkillStatus.VALIDATED, SkillStatus.ACTIVE}
        return next(
            (skill for skill in self.list_skills() if skill.derivation_key == key and skill.status in allowed),
            None,
        )

    def records_hash(self) -> str:
        import hashlib

        records = [{"kind": "memory", "id": record.memory_id, "record": record.model_dump(mode="json")} for record in self.list_memories()]
        records.extend({"kind": "skill", "id": record.skill_id, "record": record.model_dump(mode="json")} for record in self.list_skills())
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write_snapshot(self, path: Path) -> dict[str, Any]:
        skills = self.list_skills()
        payload = {
            "records_hash": self.records_hash(),
            "memory_ids": sorted(memory.memory_id for memory in self.list_memories()),
            "skill_ids": sorted(skill.skill_id for skill in skills),
            "skills": [
                {
                    "skill_id": skill.skill_id,
                    "derivation_key": skill.derivation_key,
                    "semantic_signature": skill.semantic_signature,
                    "source_memory_ids": skill.source_memory_ids,
                    "status": skill.status.value,
                    "scope": skill.scope.value,
                }
                for skill in skills
            ],
            "timestamp": utc_now(),
        }
        self._atomic_write_json(path, payload)
        self.append_event("knowledge_snapshot_created", snapshot_path=str(path), records_hash=payload["records_hash"])
        return payload

    def append_event(self, event_type: str, **payload: Any) -> None:
        append_jsonl(self.events_path, knowledge_event(event_type, **payload))

    def append_retrieval(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.retrievals_path, knowledge_event("knowledge_retrieval", **payload))

    def append_use(self, record: KnowledgeUseRecord) -> None:
        append_jsonl(self.uses_path, record.model_dump(mode="json"))

    def append_conflict(self, record: KnowledgeConflictRecord) -> None:
        append_jsonl(self.conflicts_path, record.model_dump(mode="json"))
        self.append_event("knowledge_conflict_detected", **record.model_dump(mode="json"))

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if self.atomic_write:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        else:
            path.write_text(text, encoding="utf-8")

    def _guard_record_mutation(self, operation: str, record_id: str) -> None:
        if self.record_mutation_policy != "frozen_records":
            return
        self.append_event("knowledge_mutation_violation", operation=operation, record_id=record_id)
        raise KnowledgeMutationViolation(f"record mutation forbidden while frozen: {operation}({record_id})")

    def _use_exists(self, candidate: KnowledgeUseRecord) -> bool:
        if not (candidate.project_id and candidate.task_id and candidate.session_id):
            return False
        item_id = candidate.memory_id or candidate.skill_id
        for row in self.read_jsonl(self.uses_path):
            if (
                row.get("project_id") == candidate.project_id
                and row.get("task_id") == candidate.task_id
                and row.get("session_id") == candidate.session_id
                and (row.get("memory_id") or row.get("skill_id")) == item_id
                and row.get("use_type") == candidate.use_type.value
            ):
                return True
        return False


def _matches(record: Any, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        value = getattr(record, key, None)
        if hasattr(value, "value"):
            value = value.value
        if expected is None:
            continue
        if isinstance(expected, (set, list, tuple)):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def _memory_signature(record: MemoryRecord) -> tuple[str, str, tuple[str, ...]]:
    return (record.kind.value, " ".join(record.lesson.lower().split()), tuple(sorted(record.recommended_actions)))


def _apply_usage(record: Any, use_type: KnowledgeUseType) -> None:
    now = utc_now()
    if use_type == KnowledgeUseType.RETRIEVED:
        record.retrieval_count += 1
        record.last_retrieved_at = now
    elif use_type == KnowledgeUseType.EXPOSED:
        record.exposure_count += 1
    elif use_type == KnowledgeUseType.REFERENCED:
        record.reference_count += 1
        record.last_used_at = now
    elif use_type == KnowledgeUseType.HELPFUL:
        record.helpful_count += 1
        record.last_used_at = now
    elif use_type == KnowledgeUseType.HARMFUL:
        record.harmful_count += 1
        record.last_used_at = now
