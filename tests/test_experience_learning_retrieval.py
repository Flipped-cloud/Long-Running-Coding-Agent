from __future__ import annotations

import json
from pathlib import Path

from longrun_agent.config import KnowledgeConfig
from longrun_agent.knowledge.retrieval import retrieve_bundle
from longrun_agent.knowledge.schema import KnowledgeRetrievalQuery, MemoryScope, SkillRecord, SkillStatus, SkillStep
from longrun_agent.knowledge.store import KnowledgeStore


def test_retrieval_logs_portable_skill_acceptance_and_rejection_reasons(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = KnowledgeStore(tmp_path / "knowledge", workspace_root=workspace)
    skill = SkillRecord(
        status=SkillStatus.VALIDATED,
        scope=MemoryScope.PORTABLE,
        title="Whitespace validation repair",
        summary="Repair validators that reject whitespace-only textual input by stripping before boolean validation.",
        preconditions=[
            "A validator accepts textual names or commands.",
            "Tests require whitespace-only input to be rejected.",
        ],
        anti_conditions=["preserve leading whitespace", "indentation is semantically meaningful"],
        procedure=[SkillStep(description="Apply strip-before-bool validation.", tool_name="write_file")],
        verification=["python -m pytest -q"],
        tags=["pytest", "python", "whitespace", "validation"],
        evidence_ids=["e1", "e2", "e3"],
    )
    store.save_skill(skill)
    config = KnowledgeConfig(mode="memory_skill")

    repo_b_query = KnowledgeRetrievalQuery(
        task_objective="Fix shell command validation so whitespace-only commands are rejected after trimming.",
        acceptance_criteria=["pytest passes"],
        repository_fingerprint="repo-b",
        language_tags=["python"],
        tool_tags=["pytest"],
    )
    bundle, _scores = retrieve_bundle(config, store, repo_b_query)
    assert [record.skill_id for record in bundle.skills] == [skill.skill_id]
    accepted = _last_retrieval(store.root)["skill_candidates"][0]
    assert accepted["skill_id"] == skill.skill_id
    assert accepted["status"] == "validated"
    assert accepted["scope"] == "portable"
    assert accepted["repository_match"] is False
    assert accepted["accepted"] is True
    assert accepted["selected"] is True
    assert accepted["rejection_reasons"] == []
    assert accepted["precondition_score"] > 0

    repo_c_query = repo_b_query.model_copy(
        update={"task_objective": "Preserve leading whitespace because indentation is semantically meaningful."}
    )
    rejected_bundle, _scores = retrieve_bundle(config, store, repo_c_query)
    assert rejected_bundle.skills == []
    rejected = _last_retrieval(store.root)["skill_candidates"][0]
    assert rejected["accepted"] is False
    assert rejected["selected"] is False
    assert "anti_condition_matched" in rejected["rejection_reasons"]
    assert rejected["anti_condition_score"] >= config.skill.anti_condition_match_threshold


def test_repository_scope_skill_does_not_cross_repository_fingerprint(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = KnowledgeStore(tmp_path / "knowledge", workspace_root=workspace)
    skill = SkillRecord(
        status=SkillStatus.VALIDATED,
        scope=MemoryScope.REPOSITORY,
        repository_fingerprint="repo-a",
        title="Repository specific validation repair",
        summary="Repair validation in this repository only.",
        preconditions=["validator rejects whitespace-only input"],
        anti_conditions=["preserve leading whitespace"],
        procedure=[SkillStep(description="Patch repository validator.", tool_name="write_file")],
        verification=["python -m pytest -q"],
        tags=["pytest", "python", "validation"],
        evidence_ids=["e1"],
    )
    store.save_skill(skill)

    bundle, _scores = retrieve_bundle(
        KnowledgeConfig(mode="memory_skill"),
        store,
        KnowledgeRetrievalQuery(
            task_objective="Fix validation so whitespace-only input is rejected.",
            repository_fingerprint="repo-b",
            language_tags=["python"],
            tool_tags=["pytest"],
        ),
    )

    assert bundle.skills == []
    rejected = _last_retrieval(store.root)["skill_candidates"][0]
    assert "repository_fingerprint_mismatch" in rejected["rejection_reasons"]


def _last_retrieval(root: Path) -> dict:
    rows = [json.loads(line) for line in (root / "retrievals.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1]
