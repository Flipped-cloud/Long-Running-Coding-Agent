from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_experience_pilot import validate


def test_experience_readiness_goes_on_structural_evidence_with_zero_behavioral_usage(tmp_path: Path) -> None:
    report = _report(tmp_path)

    ok, failures = validate(report)

    assert ok
    assert failures == []
    payload = json.loads(report.read_text(encoding="utf-8"))
    memory_skill = next(item for item in payload["results"] if item["mode"] == "memory_skill")
    assert memory_skill["skill_reference_count"] == 0
    assert memory_skill["skill_helpful_count"] == 0
    assert memory_skill["cross_repository_skill_helpful_count"] == 0
    assert memory_skill["preexisting_skill_reference_count"] == 0
    assert memory_skill["preexisting_skill_helpful_count"] == 0
    assert memory_skill["knowledge_assisted_transfer_success_rate"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("snapshot_not_exposed", "no Frozen Snapshot Skill was exposed"),
        ("repo_b_created_skill", "target-side knowledge creation detected"),
        ("repo_b_hash_changed", "records hash changed"),
        ("repo_b_mutation_violation", "mutation_violation_count expected 0"),
        ("repo_b_verification_failed", "final verification did not pass"),
        ("repo_c_modified", "workspace modified"),
        ("provider_error", "provider_error_count expected 0"),
        ("knowledge_error", "knowledge_error_count expected 0"),
        ("fatal_error", "fatal_protocol_error_count expected 0"),
    ],
)
def test_experience_readiness_rejects_broken_structural_invariants(
    tmp_path: Path,
    mutation: str,
    expected_failure: str,
) -> None:
    report = _report(tmp_path, mutation=mutation)

    ok, failures = validate(report)

    assert not ok
    assert any(expected_failure in item for item in failures)


def test_experience_readiness_validator_single_mode_does_not_require_verified_memory(tmp_path: Path) -> None:
    report = _report(tmp_path, modes=["memory_skill"])

    ok, failures = validate(report, required_modes=["memory_skill"])

    assert ok
    assert failures == []


def _report(tmp_path: Path, *, mutation: str = "", modes: list[str] | None = None) -> Path:
    selected = set(modes or ["verified_memory", "memory_skill"])
    results = []
    if "verified_memory" in selected:
        results.append(
            {
                "mode": "verified_memory",
                "active_memory_count": 1,
                "memory_exposure_count": 1,
                "memory_reference_count": 1,
                "memory_helpful_count": 1,
                "knowledge_error_count": 0,
                "provider_error_count": 0,
                "fatal_protocol_error_count": 0,
                "frozen_store_mutation_count": 0,
                "duplicate_derivation_key_count": 0,
                "negative_transfer_count": 0,
            },
        )
    if "memory_skill" in selected:
        results.append(
            {
                "mode": "memory_skill",
                "skill_reference_count": 0,
                "skill_helpful_count": 0,
                "cross_repository_skill_helpful_count": 0,
                "preexisting_skill_reference_count": 0,
                "preexisting_skill_helpful_count": 0,
                "knowledge_assisted_transfer_success_rate": 0.0,
                "target_side_skill_creation_count": 0,
                "frozen_store_mutation_count": 0,
                "duplicate_derivation_key_count": 0,
                "knowledge_error_count": 1 if mutation == "knowledge_error" else 0,
                "provider_error_count": 1 if mutation == "provider_error" else 0,
                "fatal_protocol_error_count": 1 if mutation == "fatal_error" else 0,
                "negative_transfer_count": 0,
            },
        )
    case_results = [
        {
            "mode": "memory_skill",
            "repeat": 1,
            "case_id": "repo_a_learn",
            "knowledge_phase": "bootstrap_learning",
            "created_memory_ids": ["MEM-1"],
            "created_skill_ids": [],
            "available_memory_ids": ["MEM-1"],
            "available_skill_ids": [],
            "knowledge_error": False,
            "provider_error_count": 0,
            "fatal_protocol_error_count": 0,
        },
        {
            "mode": "memory_skill",
            "repeat": 1,
            "case_id": "repo_a_reuse",
            "knowledge_phase": "bootstrap_consolidation",
            "created_memory_ids": [],
            "created_skill_ids": ["SKILL-1"],
            "available_memory_ids": ["MEM-1"],
            "available_skill_ids": ["SKILL-1"],
        },
        {
            "mode": "memory_skill",
            "repeat": 1,
            "case_id": "repo_b_transfer",
            "knowledge_phase": "frozen_transfer",
            "initial_verification": {"passed": False},
            "final_verification": {"passed": mutation != "repo_b_verification_failed"},
            "created_memory_ids": [],
            "created_skill_ids": ["SKILL-2"] if mutation == "repo_b_created_skill" else [],
            "available_memory_ids": ["MEM-1"],
            "available_skill_ids": ["SKILL-1", "SKILL-2"] if mutation == "repo_b_created_skill" else ["SKILL-1"],
            "knowledge_records_hash_before": "hash",
            "knowledge_records_hash_after": "changed" if mutation == "repo_b_hash_changed" else "hash",
            "snapshot_skill_ids": ["SKILL-1"],
            "exposed_skill_ids": [] if mutation == "snapshot_not_exposed" else ["SKILL-1"],
            "referenced_skill_ids": [],
            "helpful_skill_ids": [],
            "mutation_violation_count": 1 if mutation == "repo_b_mutation_violation" else 0,
        },
        {
            "mode": "memory_skill",
            "repeat": 1,
            "case_id": "repo_c_negative",
            "knowledge_phase": "frozen_negative_control",
            "initial_verification": {"passed": True},
            "final_verification": {"passed": True},
            "modified_after_run": mutation == "repo_c_modified",
            "created_memory_ids": [],
            "created_skill_ids": [],
            "knowledge_records_hash_before": "hash",
            "knowledge_records_hash_after": "hash",
            "mutation_violation_count": 0,
            "harmful_memory_ids": [],
            "harmful_skill_ids": [],
        },
    ]
    snapshot_path = tmp_path / "memory_skill" / "repeat_01" / "knowledge_snapshot.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "skill_ids": ["SKILL-1"],
                "skills": [{"skill_id": "SKILL-1", "scope": "portable"}],
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"results": results, "case_results": case_results}), encoding="utf-8")
    return path
