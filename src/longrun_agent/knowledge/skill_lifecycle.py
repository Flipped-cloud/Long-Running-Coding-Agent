from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.schema import (
    ExperienceEvidencePack,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    SkillCandidateDraft,
    SkillRecord,
    SkillStatus,
    utc_now,
)
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.model.base import ModelProvider

KNOWN_TOOLS = {"read_file", "write_file", "bash", "report_progress", "report_blocker", "request_task_completion", "request_decomposition"}

SKILL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_skill_candidate",
        "description": "Submit one verified procedural skill candidate grounded in the supplied evidence.",
        "parameters": SkillCandidateDraft.model_json_schema(),
    },
}


class SkillGenerationResult(BaseModel):
    draft: SkillCandidateDraft | None = None
    success: bool = False
    attempt_count: int = 0
    failure_stage: (
        Literal[
            "provider_request",
            "missing_tool_call",
            "argument_decode",
            "json_extract",
            "schema_validation",
            "portability_validation",
            "protocol_exhausted",
        ]
        | None
    ) = None
    exception_type: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    raw_response_excerpt: str = ""
    generation_origin: Literal["tool_call", "content_json", "deterministic_repair"] | None = None
    repair_actions: list[str] = Field(default_factory=list)


class SkillCandidateGenerator:
    def __init__(self, model: ModelProvider, config):
        self.model = model
        self.config = config

    def generate(
        self,
        pack: ExperienceEvidencePack,
        *,
        source_memory_ids: list[str],
        source_memories: list[MemoryRecord] | None = None,
    ) -> SkillCandidateDraft | None:
        return self.generate_result(pack, source_memory_ids=source_memory_ids, source_memories=source_memories).draft

    def generate_result(
        self,
        pack: ExperienceEvidencePack,
        *,
        source_memory_ids: list[str],
        source_memories: list[MemoryRecord] | None = None,
    ) -> SkillGenerationResult:
        payload = {
            "task_objective": pack.task_objective,
            "acceptance_criteria": pack.acceptance_criteria,
            "successful_verification": pack.successful_verifications,
            "changed_files": pack.files_changed,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "event_type": item.event_type,
                    "summary": item.summary,
                    "tool_name": item.tool_name,
                    "command": item.command,
                    "success": item.success,
                    "exit_code": item.exit_code,
                }
                for item in pack.evidence_items[:20]
            ],
            "source_memory_ids": source_memory_ids,
            "source_memories": [
                {
                    "memory_id": memory.memory_id,
                    "lesson": memory.lesson,
                    "recommended_actions": memory.recommended_actions,
                    "applicability_conditions": memory.applicability_conditions,
                    "anti_conditions": memory.anti_conditions,
                    "tags": memory.tags,
                }
                for memory in source_memories or []
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Create a conservative reusable skill only from successful verification evidence. "
                    "Source memories may be repository-scoped. Abstract repository-specific experience into a "
                    "parameterized portable procedure. Do not copy source filenames, function names, project IDs, "
                    "or paths. Preserve source_memory_ids for provenance. The output scope should be portable when "
                    "the procedure passes the portability checks."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        tool_choice = {"type": "function", "function": {"name": "submit_skill_candidate"}}
        last_result = SkillGenerationResult(success=False)
        max_protocol_retries = int(getattr(self.config, "max_protocol_retries", 2))
        require_portable_output = bool(source_memories) and bool(getattr(self.config, "require_active_helpful_source_for_candidate", False))
        for retry in range(max_protocol_retries + 1):
            try:
                response = self.model.generate(messages, [SKILL_TOOL_SCHEMA], tool_choice=tool_choice)
            except Exception as exc:
                stage = "argument_decode" if exc.__class__.__name__ == "ToolArgumentsProtocolError" else "provider_request"
                last_result = SkillGenerationResult(
                    success=False,
                    attempt_count=retry + 1,
                    failure_stage=stage,
                    exception_type=exc.__class__.__name__,
                    validation_errors=[str(exc)],
                    raw_response_excerpt=_sanitize_excerpt(getattr(exc, "raw_arguments", "") or str(exc)),
                )
                if retry >= max_protocol_retries:
                    return last_result.model_copy(update={"failure_stage": "protocol_exhausted" if stage == "argument_decode" else stage})
                messages.append({"role": "user", "content": "Call submit_skill_candidate with valid JSON arguments."})
                continue

            response_excerpt = _response_excerpt(response)
            if _finish_reason(response) == "length":
                return SkillGenerationResult(
                    success=False,
                    attempt_count=retry + 1,
                    failure_stage="protocol_exhausted",
                    raw_response_excerpt=response_excerpt,
                )
            parsed, origin, stage, errors, repair_actions = _parse_skill_candidate_response(
                response,
                minimum_confidence=float(getattr(self.config, "minimum_candidate_confidence", 0.65)),
                require_portable_output=require_portable_output,
            )
            if parsed is not None:
                return SkillGenerationResult(
                    draft=parsed,
                    success=True,
                    attempt_count=retry + 1,
                    raw_response_excerpt=response_excerpt,
                    generation_origin=origin,
                    repair_actions=repair_actions,
                )
            last_result = SkillGenerationResult(
                success=False,
                attempt_count=retry + 1,
                failure_stage=stage,
                validation_errors=errors,
                raw_response_excerpt=response_excerpt,
                generation_origin=origin,
                repair_actions=repair_actions,
            )
            if retry >= max_protocol_retries:
                return last_result.model_copy(update={"failure_stage": "protocol_exhausted" if stage == "schema_validation" else stage})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The skill candidate did not validate. Resubmit exactly one complete submit_skill_candidate object. "
                        "When supplied source memories are active helpful evidence, return a portable parameterized procedure "
                        "and avoid repository-specific function names, file names, project IDs, or paths. "
                        f"Validation errors: {json.dumps(errors[:8], ensure_ascii=False)}"
                    ),
                }
            )
        return last_result


class SkillValidator:
    def __init__(self, store: KnowledgeStore | None = None, config: KnowledgeConfig | None = None):
        self.store = store
        self.config = config or KnowledgeConfig(mode="memory_skill")

    def validate(self, skill: SkillRecord, pack: ExperienceEvidencePack) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if pack.verification_verdict and pack.verification_verdict != "verified":
            reasons.append("formal VERIFIED evidence required")
        if pack.integrity_violations:
            reasons.append("integrity violations prevent skill validation")
        if pack.infrastructure_error:
            reasons.append("infrastructure error is not skill success evidence")
        if not pack.successful_verifications:
            reasons.append("successful verification evidence required")
        if pack.failed_verifications and not pack.successful_verifications:
            reasons.append("stale or failed verification cannot validate skill")
        if not skill.preconditions:
            reasons.append("preconditions required")
        if not skill.anti_conditions:
            reasons.append("anti-conditions required")
        if skill.confidence < self.config.skill.minimum_candidate_confidence:
            reasons.append("confidence below threshold")
        evidence = pack.evidence_by_id()
        missing_evidence = [item for item in skill.evidence_ids if item not in evidence]
        if missing_evidence:
            reasons.append(f"unknown evidence ids: {missing_evidence}")
        if not any(item.success and item.event_type in {"successful_verification", "test_command"} for item in evidence.values()):
            reasons.append("successful verification evidence item required")
        for step in skill.procedure:
            if step.tool_name and step.tool_name not in KNOWN_TOOLS:
                reasons.append(f"unknown tool in skill: {step.tool_name}")
            if _contains_leak(step.command or ""):
                reasons.append("skill command contains absolute path or secret-like text")
        if skill.scope == MemoryScope.PORTABLE:
            if not skill.source_memory_ids:
                reasons.append("portable skill requires source memory")
            if self.store is not None:
                for memory_id in skill.source_memory_ids:
                    try:
                        memory = self.store.load_memory(memory_id)
                    except FileNotFoundError:
                        reasons.append(f"unknown source memory: {memory_id}")
                        continue
                    if memory.status != MemoryStatus.ACTIVE:
                        reasons.append(f"portable skill source memory is not active: {memory_id}")
            if _skill_contains_source_specific_text(skill):
                reasons.append("portable skill contains source-specific file, function, path, or secret-like text")
        return not reasons, reasons


class SkillLifecycleManager:
    def __init__(self, config: KnowledgeConfig, store: KnowledgeStore):
        self.config = config
        self.store = store

    def candidate_from_success(
        self, pack: ExperienceEvidencePack, *, draft: SkillCandidateDraft | None = None, source_memory_ids: list[str] | None = None
    ) -> SkillRecord | None:
        if self.config.mode != "memory_skill" or not self.config.skill.enabled:
            return None
        source_memory_ids = source_memory_ids or []
        has_verified_success = bool(pack.successful_verifications) and (pack.verification_verdict in {None, "verified"})
        has_code_change = bool(pack.files_changed)
        has_active_source = self._active_sources(source_memory_ids)
        if not has_verified_success:
            return None
        if self.config.skill.require_active_helpful_source_for_candidate:
            if not has_active_source:
                self.store.append_event(
                    "skill_candidate_skipped",
                    episode_id=pack.episode_id,
                    reason="source_memory_not_active",
                )
                return None
        elif not has_code_change:
            self.store.append_event(
                "skill_candidate_skipped",
                episode_id=pack.episode_id,
                reason="no_changed_files",
            )
            return None
        if draft is None:
            return None
        source_anti_conditions = self._source_anti_conditions(source_memory_ids)
        draft_source_ids = draft.source_memory_ids or source_memory_ids
        draft = draft.model_copy(
            update={
                "source_memory_ids": draft_source_ids,
                "anti_conditions": _dedupe([*draft.anti_conditions, *source_anti_conditions]),
            }
        )
        if draft.scope == MemoryScope.PORTABLE and _draft_contains_source_specific_text(draft):
            self.store.append_event(
                "skill_validation_failed",
                episode_id=pack.episode_id,
                reason="portable skill contains source-specific file, function, path, or secret-like text",
            )
            return None
        skill = SkillRecord(
            status=SkillStatus.DRAFT,
            title=draft.title,
            summary=draft.summary,
            scope=draft.scope,
            preconditions=draft.preconditions,
            anti_conditions=draft.anti_conditions,
            parameters=draft.parameters,
            procedure=draft.procedure,
            verification=draft.verification,
            stop_conditions=draft.stop_conditions,
            tags=draft.tags,
            repository_fingerprint=pack.repository_fingerprint,
            source_memory_ids=draft.source_memory_ids,
            source_episode_ids=[pack.episode_id],
            source_task_ids=[pack.task_id],
            evidence_ids=draft.evidence_ids,
            confidence=draft.confidence,
            success_count=1,
            distinct_task_ids=[pack.task_id],
            distinct_repository_fingerprints=[pack.repository_fingerprint],
        )
        valid, reasons = SkillValidator(self.store, self.config).validate(skill, pack)
        if not valid:
            self.store.append_event("skill_validation_failed", episode_id=pack.episode_id, reason="; ".join(reasons))
            return None
        duplicate = self._find_duplicate(skill)
        if duplicate is not None:
            duplicate.source_episode_ids = _dedupe([*duplicate.source_episode_ids, *skill.source_episode_ids])
            duplicate.source_task_ids = _dedupe([*duplicate.source_task_ids, *skill.source_task_ids])
            duplicate.evidence_ids = _dedupe([*duplicate.evidence_ids, *skill.evidence_ids])
            duplicate.updated_at = utc_now()
            self.store.save_skill(duplicate)
            self.store.append_event(
                "skill_derivation_upserted",
                skill_id=duplicate.skill_id,
                derivation_key=duplicate.derivation_key,
                semantic_signature=duplicate.semantic_signature,
                candidate_semantic_signature=skill.semantic_signature,
            )
            return duplicate
        skill.status = SkillStatus.VALIDATED
        self.store.save_skill(skill)
        self.store.append_event("skill_candidate_created", episode_id=pack.episode_id, skill_id=skill.skill_id)
        self.store.append_event("skill_validated", episode_id=pack.episode_id, skill_id=skill.skill_id)
        return skill

    def record_success(self, skill_id: str, *, task_id: str, repository_fingerprint: str) -> SkillRecord:
        skill = self.store.load_skill(skill_id)
        if task_id in skill.distinct_task_ids:
            return skill
        skill.success_count += 1
        if task_id not in skill.distinct_task_ids:
            skill.distinct_task_ids.append(task_id)
        if repository_fingerprint not in skill.distinct_repository_fingerprints:
            skill.distinct_repository_fingerprints.append(repository_fingerprint)
        if (
            skill.status == SkillStatus.VALIDATED
            and skill.success_count >= self.config.skill.promotion_min_successes
            and len(skill.distinct_task_ids) >= self.config.skill.promotion_min_distinct_tasks
            and len(skill.distinct_repository_fingerprints) >= self.config.skill.promotion_min_distinct_repositories
        ):
            skill.status = SkillStatus.ACTIVE
            self.store.append_event("skill_promoted", skill_id=skill.skill_id, status_after=skill.status.value)
        skill.updated_at = utc_now()
        self.store.save_skill(skill)
        return skill

    def record_harmful_use(self, skill_id: str, *, reason: str) -> SkillRecord:
        skill = self.store.load_skill(skill_id)
        skill.harmful_count += 1
        skill.failure_count += 1
        if skill.status == SkillStatus.ACTIVE:
            skill.status = SkillStatus.DEPRECATED
            self.store.append_event("skill_deprecated", skill_id=skill_id, reason=reason)
        skill.updated_at = utc_now()
        self.store.save_skill(skill)
        return skill

    def _active_sources(self, source_memory_ids: list[str]) -> bool:
        if not source_memory_ids:
            return False
        for memory_id in source_memory_ids:
            try:
                memory = self.store.load_memory(memory_id)
            except FileNotFoundError:
                return False
            if memory.status != MemoryStatus.ACTIVE:
                return False
        return True

    def _source_anti_conditions(self, source_memory_ids: list[str]) -> list[str]:
        anti_conditions: list[str] = []
        for memory_id in source_memory_ids:
            try:
                memory = self.store.load_memory(memory_id)
            except FileNotFoundError:
                continue
            anti_conditions.extend(memory.anti_conditions)
        return _dedupe(anti_conditions)

    def _find_duplicate(self, candidate: SkillRecord) -> SkillRecord | None:
        return self.store.find_skill_by_derivation(source_memory_ids=candidate.source_memory_ids)


def _parse_skill_candidate_response(
    response, *, minimum_confidence: float, require_portable_output: bool
) -> tuple[SkillCandidateDraft | None, str | None, str, list[str], list[str]]:
    calls = [call for call in response.tool_calls if call.name == "submit_skill_candidate"]
    if len(calls) == 1:
        return _parse_candidate_data(
            calls[0].arguments,
            origin="tool_call",
            minimum_confidence=minimum_confidence,
            require_portable_output=require_portable_output,
        )
    if len(calls) > 1:
        return None, "tool_call", "missing_tool_call", ["submit_skill_candidate must be called exactly once"], []
    content = _response_content(response)
    if not content:
        return None, None, "missing_tool_call", ["no submit_skill_candidate tool call and no content JSON"], []
    data, errors, repair_actions = _extract_json_data(content)
    if data is None:
        return None, "content_json", "json_extract", errors, repair_actions
    return _parse_candidate_data(
        data,
        origin="content_json",
        repair_actions=repair_actions,
        minimum_confidence=minimum_confidence,
        require_portable_output=require_portable_output,
    )


def _parse_candidate_data(
    data: Any,
    *,
    origin: Literal["tool_call", "content_json"],
    minimum_confidence: float,
    require_portable_output: bool,
    repair_actions: list[str] | None = None,
) -> tuple[SkillCandidateDraft | None, str, str, list[str], list[str]]:
    repair_actions = list(repair_actions or [])
    if isinstance(data, str):
        try:
            data = json.loads(data)
            repair_actions.append("decoded_json_string_arguments")
        except json.JSONDecodeError as exc:
            return None, origin, "argument_decode", [f"{exc.__class__.__name__}: {exc}"], repair_actions
    if not isinstance(data, dict):
        return None, origin, "argument_decode", [f"expected JSON object, got {type(data).__name__}"], repair_actions
    repaired = _repair_format_only(data, repair_actions)
    result_origin = "deterministic_repair" if repair_actions else origin
    try:
        draft = SkillCandidateDraft.model_validate(repaired)
    except ValidationError as exc:
        return None, result_origin, "schema_validation", _validation_errors(exc), repair_actions
    if draft.confidence < minimum_confidence:
        return (
            None,
            result_origin,
            "schema_validation",
            [f"confidence: expected >= {minimum_confidence}, got {draft.confidence}"],
            repair_actions,
        )
    if require_portable_output:
        portability_errors: list[str] = []
        if draft.scope != MemoryScope.PORTABLE:
            portability_errors.append(f"scope: expected portable for active helpful source memory, got {draft.scope.value}")
        if _draft_contains_source_specific_text(draft):
            portability_errors.append("portable skill contains source-specific file, function, path, or secret-like text")
        if portability_errors:
            return None, result_origin, "portability_validation", portability_errors, repair_actions
    return draft, result_origin, "", [], repair_actions


def _repair_format_only(data: dict[str, Any], repair_actions: list[str]) -> dict[str, Any]:
    repaired = dict(data)
    list_fields = {
        "preconditions",
        "anti_conditions",
        "parameters",
        "procedure",
        "verification",
        "stop_conditions",
        "tags",
        "evidence_ids",
        "source_memory_ids",
    }
    for field_name in list_fields:
        if field_name in repaired and repaired[field_name] is None:
            repaired[field_name] = []
            repair_actions.append(f"null_to_empty_list:{field_name}")
    if isinstance(repaired.get("scope"), str):
        normalized_scope = repaired["scope"].strip().lower()
        if normalized_scope != repaired["scope"]:
            repaired["scope"] = normalized_scope
            repair_actions.append("normalized_scope_case")
    return repaired


def _extract_json_data(content: str) -> tuple[Any | None, list[str], list[str]]:
    repair_actions: list[str] = []
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
        repair_actions.append("removed_markdown_fence")
    try:
        return json.loads(text), [], repair_actions
    except json.JSONDecodeError:
        pass
    extracted = _extract_unique_json_object(text)
    if extracted is None:
        return None, ["could not extract a unique complete JSON object"], repair_actions
    repair_actions.append("extracted_json_object_from_text")
    try:
        return json.loads(extracted), [], repair_actions
    except json.JSONDecodeError as exc:
        return None, [f"{exc.__class__.__name__}: {exc}"], repair_actions


def _extract_unique_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _response_content(response) -> str:
    if response.final_answer is not None:
        return response.final_answer.content or ""
    message = response.raw_metadata.get("message") if isinstance(response.raw_metadata, dict) else None
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""


def _response_excerpt(response) -> str:
    payload = {
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments} for call in response.tool_calls if call.name == "submit_skill_candidate"
        ],
        "content": _response_content(response),
        "finish_reason": _finish_reason(response),
    }
    return _sanitize_excerpt(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def _finish_reason(response) -> str | None:
    if isinstance(response.raw_metadata, dict):
        value = response.raw_metadata.get("finish_reason")
        if value:
            return str(value)
    return None


def _sanitize_excerpt(text: str, *, limit: int = 2000) -> str:
    sanitized = text or ""
    sanitized = re.sub(r"(?i)(api[_-]?key|authorization|bearer|token)\s*[:=]\s*['\"]?[^'\"\s,}]+", r"\1=[redacted]", sanitized)
    sanitized = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", sanitized)
    sanitized = re.sub(r"[A-Za-z]:\\[^\s\"',}]+", "<path>", sanitized)
    sanitized = re.sub(r"/(?:Users|home|tmp|var)/[^\s\"',}]+", "<path>", sanitized)
    if len(sanitized) <= limit:
        return sanitized
    half = limit // 2
    return sanitized[:half] + "\n...[truncated]...\n" + sanitized[-half:]


def _validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for item in exc.errors()[:12]:
        loc = ".".join(str(part) for part in item.get("loc", ()))
        errors.append(f"{loc}: {item.get('msg', '')}")
    return errors or [str(exc)]


def _draft_contains_source_specific_text(draft: SkillCandidateDraft) -> bool:
    text = " ".join(
        [
            draft.title,
            draft.summary,
            *draft.preconditions,
            *draft.anti_conditions,
            *draft.parameters,
            *[step.description for step in draft.procedure],
            *[step.command or "" for step in draft.procedure],
            *draft.verification,
            *draft.stop_conditions,
        ]
    )
    return _contains_source_specific_text(text)


def _skill_contains_source_specific_text(skill: SkillRecord) -> bool:
    text = " ".join(
        [
            skill.title,
            skill.summary,
            *skill.preconditions,
            *skill.anti_conditions,
            *skill.parameters,
            *(step.description for step in skill.procedure),
            *(step.command or "" for step in skill.procedure),
            *skill.verification,
            *skill.stop_conditions,
        ]
    )
    return _contains_source_specific_text(text)


def _contains_source_specific_text(text: str) -> bool:
    function_names = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(\)", text))
    source_specific_functions = function_names - {"strip", "bool", "str", "len", "lower", "upper", "isinstance", "pytest"}
    return _contains_leak(text) or bool(re.search(r"\b[\w.-]+\.py\b", text) or source_specific_functions)


def _contains_leak(text: str) -> bool:
    lowered = text.lower()
    return (
        "api_key" in lowered or "token" in lowered or "secret" in lowered or "password" in lowered or ":\\" in text or text.startswith("/")
    )


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
