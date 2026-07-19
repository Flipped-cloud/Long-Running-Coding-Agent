from __future__ import annotations

import argparse
import json
from pathlib import Path

from longrun_agent.evals.experience_learning.generator import load_experience_config
from longrun_agent.knowledge.evidence import build_experience_pack
from longrun_agent.knowledge.schema import KnowledgeUseType
from longrun_agent.knowledge.skill_lifecycle import SkillCandidateGenerator, SkillLifecycleManager
from longrun_agent.knowledge.store import KnowledgeStore
from longrun_agent.model.openai_compatible import OpenAICompatibleProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-source-scope", choices=["repository", "portable", "any"], default="any")
    args = parser.parse_args()

    config = load_experience_config(args.config)
    if config.model is None:
        raise SystemExit("config must contain model for API replay")

    run_root = args.run_root.resolve()
    project_dir = run_root / "projects" / args.project_id
    sessions_path = project_dir / "sessions.jsonl"
    sessions = _read_jsonl(sessions_path)
    if not sessions:
        raise SystemExit(f"no sessions found: {sessions_path}")
    session = sessions[-1]
    workspace = Path(session.get("repository") or "")
    if not workspace.exists():
        workspace = Path(str(session.get("changed_files", [""])[0])).parent if session.get("changed_files") else run_root
    knowledge_store = KnowledgeStore(run_root / "knowledge")
    source_memory_ids = _helpful_memory_ids(run_root / "knowledge" / "uses.jsonl", args.project_id)
    source_memories = [knowledge_store.load_memory(memory_id) for memory_id in source_memory_ids]
    if args.require_source_scope != "any" and any(memory.scope.value != args.require_source_scope for memory in source_memories):
        raise SystemExit(f"source memory scope does not match --require-source-scope={args.require_source_scope}")
    pack = build_experience_pack(
        project_id=args.project_id,
        task_id=str(session.get("task_id") or f"{args.project_id}:main"),
        task_objective=str(session.get("task_objective") or args.project_id),
        acceptance_criteria=list(session.get("acceptance_criteria") or ["verification passes"]),
        session_record=session,
        plan_version=int(session.get("plan_version") or 1),
        starting_task_status="in_progress",
        ending_task_status="candidate_complete",
        workspace_root=workspace if workspace.exists() else run_root,
    )
    provider = OpenAICompatibleProvider(config.model)
    from longrun_agent.config import KnowledgeConfig

    knowledge_config = KnowledgeConfig(mode="memory_skill", skill={"require_active_helpful_source_for_candidate": True})
    generation = SkillCandidateGenerator(provider, knowledge_config.skill).generate_result(
        pack, source_memory_ids=source_memory_ids, source_memories=source_memories
    )
    skill = None
    if generation.draft is not None:
        skill = SkillLifecycleManager(knowledge_config, knowledge_store).candidate_from_success(
            pack, draft=generation.draft, source_memory_ids=source_memory_ids
        )
    payload = {
        "source_memory_ids": source_memory_ids,
        "source_memories": [
            {"memory_id": memory.memory_id, "status": memory.status.value, "scope": memory.scope.value} for memory in source_memories
        ],
        "provider_response_type": generation.generation_origin,
        "attempt_count": generation.attempt_count,
        "failure_stage": generation.failure_stage,
        "validation_errors": generation.validation_errors,
        "raw_response_excerpt": generation.raw_response_excerpt,
        "candidate": generation.draft.model_dump(mode="json") if generation.draft else None,
        "final_scope": skill.scope.value if skill else "",
        "candidate_id": skill.skill_id if skill else "",
        "repair_actions": generation.repair_actions,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if skill is not None else 1)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _helpful_memory_ids(uses_path: Path, project_id: str) -> list[str]:
    ids: list[str] = []
    for use in _read_jsonl(uses_path):
        if use.get("project_id") != project_id or use.get("use_type") != KnowledgeUseType.HELPFUL.value:
            continue
        memory_id = use.get("memory_id")
        if memory_id and memory_id not in ids:
            ids.append(memory_id)
    return ids


if __name__ == "__main__":
    main()
