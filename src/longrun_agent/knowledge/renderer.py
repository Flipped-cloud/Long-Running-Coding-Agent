from __future__ import annotations

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.schema import RetrievedKnowledgeBundle


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def render_memory(record) -> str:
    return "\n".join(
        [
            f"Memory ID: {record.memory_id}",
            f"Scope: {record.scope.value}",
            "Use when:",
            *[f"- {item}" for item in record.applicability_conditions[:5]],
            "Do not use when:",
            *[f"- {item}" for item in record.anti_conditions[:5]],
            f"Lesson: {record.lesson}",
            "Recommended actions:",
            *[f"- {item}" for item in record.recommended_actions[:5]],
            f"Evidence strength: confidence={record.confidence:.2f}; evidence_ids={', '.join(record.evidence_ids[:5])}",
        ]
    )


def render_skill(record) -> str:
    return "\n".join(
        [
            f"Skill ID: {record.skill_id}",
            f"Version: {record.version}",
            f"Status: {record.status.value}",
            "Use when:",
            *[f"- {item}" for item in record.preconditions[:5]],
            "Do not use when:",
            *[f"- {item}" for item in record.anti_conditions[:5]],
            "Parameters:",
            *([f"- {item}" for item in record.parameters[:5]] or ["- none"]),
            "Procedure:",
            *[f"- {step.description}" for step in record.procedure[:8]],
            "Verification:",
            *[f"- {item}" for item in record.verification[:5]],
            "Stop conditions:",
            *[f"- {item}" for item in record.stop_conditions[:5]],
        ]
    )


def render_bundle(bundle: RetrievedKnowledgeBundle, config: KnowledgeConfig) -> tuple[str, int]:
    sections: list[str] = []
    memory_budget = config.memory.max_context_tokens
    skill_budget = config.skill.max_context_tokens
    memory_lines = []
    used = 0
    arbitrated = bool(bundle.primary_skill_ids or bundle.shadowed_memory_ids or bundle.provenance_links)
    primary_memory_ids = set(
        bundle.primary_memory_ids if arbitrated else (bundle.primary_memory_ids or [memory.memory_id for memory in bundle.memories])
    )
    primary_skill_ids = set(bundle.primary_skill_ids or [skill.skill_id for skill in bundle.skills])
    for memory in bundle.memories:
        if memory.memory_id not in primary_memory_ids:
            continue
        rendered = render_memory(memory)
        tokens = estimate_tokens(rendered)
        if used + tokens > memory_budget:
            continue
        memory_lines.append(rendered)
        used += tokens
    skill_lines = []
    skill_used = 0
    for skill in bundle.skills:
        if skill.skill_id not in primary_skill_ids:
            continue
        rendered = render_skill(skill)
        tokens = estimate_tokens(rendered)
        if skill_used + tokens > skill_budget:
            continue
        skill_lines.append(rendered)
        skill_used += tokens
    if memory_lines:
        sections.append("Relevant Past Experiences:\n" + "\n\n".join(memory_lines))
    if skill_lines:
        sections.append("Relevant Procedural Skills:\n" + "\n\n".join(skill_lines))
    if bundle.shadowed_memory_ids:
        sections.append(
            "Skill provenance:\n" + "\n".join(f"- Derived from Memory ID: {memory_id}" for memory_id in bundle.shadowed_memory_ids)
        )
    if sections:
        sections.insert(
            0,
            (
                "Knowledge Decision required: report referenced Memory ID or Skill ID values before editing, "
                "verification, or completion; if none apply, report empty ID lists with a reason."
            ),
        )
    text = "\n\n".join(sections)
    return text, estimate_tokens(text) if text else 0
