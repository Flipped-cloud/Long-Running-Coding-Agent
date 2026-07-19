from __future__ import annotations

from longrun_agent.knowledge.schema import RetrievedKnowledgeBundle, SkillStatus


class KnowledgeArbitrator:
    def arbitrate(self, bundle: RetrievedKnowledgeBundle, *, memory_skill_mode: bool) -> RetrievedKnowledgeBundle:
        memory_ids = [memory.memory_id for memory in bundle.memories]
        if not memory_skill_mode:
            return bundle.model_copy(
                update={
                    "primary_memory_ids": memory_ids,
                    "primary_skill_ids": [],
                    "shadowed_memory_ids": [],
                    "provenance_links": {},
                }
            )

        primary_skill_ids: list[str] = []
        shadowed_memory_ids: list[str] = []
        provenance_links: dict[str, list[str]] = {}
        selected_memory_ids = set(memory_ids)
        for skill in bundle.skills:
            if skill.status not in {SkillStatus.VALIDATED, SkillStatus.ACTIVE}:
                continue
            primary_skill_ids.append(skill.skill_id)
            parents = sorted(selected_memory_ids & set(skill.source_memory_ids))
            if parents:
                provenance_links[skill.skill_id] = parents
                shadowed_memory_ids.extend(parents)

        shadowed = sorted(set(shadowed_memory_ids))
        primary_memory_ids = [memory_id for memory_id in memory_ids if memory_id not in set(shadowed)]
        return bundle.model_copy(
            update={
                "primary_memory_ids": primary_memory_ids,
                "primary_skill_ids": primary_skill_ids,
                "shadowed_memory_ids": shadowed,
                "provenance_links": provenance_links,
            }
        )
