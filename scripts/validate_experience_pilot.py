from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from longrun_agent.knowledge.evidence import RepositoryProfiler


def validate(report_path: Path, *, required_modes: list[str] | None = None) -> tuple[bool, list[str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = {item["mode"]: item for item in report.get("results", [])}
    modes = set(required_modes or results)
    cases = report.get("case_results", [])
    failures: list[str] = []

    for mode in sorted(modes):
        if mode not in results:
            failures.append(f"{mode}: missing from report results")

    if "verified_memory" in modes:
        verified = results.get("verified_memory", {})
        _at_least(failures, verified, "verified_memory", "active_memory_count", 1)
        _at_least(failures, verified, "verified_memory", "memory_exposure_count", 1)
        _at_least(failures, verified, "verified_memory", "memory_reference_count", 1)
        _at_least(failures, verified, "verified_memory", "memory_helpful_count", 1)

    if "memory_skill" in modes:
        memory_skill = results.get("memory_skill", {})
        _equals(failures, memory_skill, "memory_skill", "target_side_skill_creation_count", 0)
        _equals(failures, memory_skill, "memory_skill", "frozen_store_mutation_count", 0)
        _equals(failures, memory_skill, "memory_skill", "duplicate_derivation_key_count", 0)
        _check_memory_skill_phases(cases, failures, report_path)

    for result in results.values():
        mode = str(result.get("mode"))
        _equals(failures, result, mode, "knowledge_error_count", 0)
        _equals(failures, result, mode, "provider_error_count", 0)
        _equals(failures, result, mode, "fatal_protocol_error_count", 0)
        _equals(failures, result, mode, "frozen_store_mutation_count", 0)
        _equals(failures, result, mode, "duplicate_derivation_key_count", 0)
        _equals(failures, result, mode, "negative_transfer_count", 0)

    for case in cases:
        if case.get("case_id") == "repo_c_negative":
            prefix = f"{case.get('mode')}/repo_c_negative"
            if not case.get("initial_verification", {}).get("passed"):
                failures.append(f"{prefix}: initial verification did not pass")
            if not case.get("final_verification", {}).get("passed"):
                failures.append(f"{prefix}: final verification did not pass")
            if case.get("modified_after_run"):
                failures.append(f"{prefix}: modified_after_run is true")
            if case.get("harmful_memory_ids") or case.get("harmful_skill_ids"):
                failures.append(f"{prefix}: harmful knowledge IDs are not empty")
        for field in ("created_memory_ids", "created_skill_ids"):
            created = set(case.get(field) or [])
            available_field = field.replace("created_", "available_")
            available = set(case.get(available_field) or [])
            if not created <= available:
                failures.append(f"{case.get('mode')}/{case.get('case_id')}: {field} is not a subset of {available_field}")

    _check_workspace_fingerprints(cases, failures)

    return not failures, failures


def _at_least(failures: list[str], item: dict[str, Any], mode: str, field: str, expected: int) -> None:
    actual = int(item.get(field) or 0)
    if actual < expected:
        failures.append(f"{mode}: {field} expected >= {expected}, got {actual}")


def _equals(failures: list[str], item: dict[str, Any], mode: str, field: str, expected: int) -> None:
    actual = int(item.get(field) or 0)
    if actual != expected:
        failures.append(f"{mode}: {field} expected {expected}, got {actual}")


def _check_workspace_fingerprints(cases: list[dict[str, Any]], failures: list[str]) -> None:
    memory_skill = [case for case in cases if case.get("mode") == "memory_skill"]
    paths = [Path(str(case.get("repository") or "")) for case in memory_skill if case.get("repository")]
    existing_paths = [path for path in paths if path.exists()]
    if len(existing_paths) != len(set(existing_paths)):
        failures.append("memory_skill: case workspaces are not isolated")
    profiles = {
        case.get("case_id"): RepositoryProfiler(Path(case["repository"])).profile().repository_fingerprint
        for case in memory_skill
        if case.get("repository") and Path(case["repository"]).exists()
    }
    required = {"repo_a_learn", "repo_a_reuse", "repo_b_transfer", "repo_c_negative"}
    if required <= set(profiles):
        if profiles["repo_a_learn"] != profiles["repo_a_reuse"]:
            failures.append("fingerprint: repo_a learn/reuse should match")
        distinct = {profiles["repo_a_learn"], profiles["repo_b_transfer"], profiles["repo_c_negative"]}
        if len(distinct) != 3:
            failures.append("fingerprint: repo A/B/C should be pairwise different")


def _check_memory_skill_phases(cases: list[dict[str, Any]], failures: list[str], report_path: Path) -> None:
    memory_skill = [case for case in cases if case.get("mode") == "memory_skill"]
    by_phase = {case.get("knowledge_phase"): case for case in memory_skill}
    learning = by_phase.get("bootstrap_learning")
    consolidation = by_phase.get("bootstrap_consolidation")
    transfer = by_phase.get("frozen_transfer")
    negative = by_phase.get("frozen_negative_control")
    for phase, case in (
        ("bootstrap_learning", learning),
        ("bootstrap_consolidation", consolidation),
        ("frozen_transfer", transfer),
        ("frozen_negative_control", negative),
    ):
        if case is None:
            failures.append(f"memory_skill: missing phase {phase}")
    if learning is not None and not learning.get("created_memory_ids"):
        failures.append("memory_skill/bootstrap_learning: no Memory created")
    if learning is not None:
        if learning.get("knowledge_error"):
            failures.append("memory_skill/bootstrap_learning: knowledge_error is true")
        for field in ("provider_error_count", "fatal_protocol_error_count"):
            if int(learning.get(field) or 0) != 0:
                failures.append(f"memory_skill/bootstrap_learning: {field} expected 0")
    if consolidation is not None:
        created_skill_ids = set(consolidation.get("created_skill_ids") or [])
        if len(created_skill_ids) != 1:
            failures.append("memory_skill/bootstrap_consolidation: expected exactly one created Skill")
        snapshot = _load_snapshot(report_path, int(consolidation.get("repeat") or 1))
        snapshot_skill_ids = set(snapshot.get("skill_ids") or [])
        portable_skill_ids = {
            item.get("skill_id") for item in snapshot.get("skills") or [] if item.get("scope") == "portable" and item.get("skill_id")
        }
        if not created_skill_ids <= snapshot_skill_ids:
            failures.append("memory_skill/bootstrap_consolidation: created Skill is missing from Frozen Snapshot")
        if not created_skill_ids <= portable_skill_ids:
            failures.append("memory_skill/bootstrap_consolidation: created Skill is not Portable")
    if transfer is not None:
        if not transfer.get("final_verification", {}).get("passed"):
            failures.append("memory_skill/frozen_transfer: final verification did not pass")
        if transfer.get("created_memory_ids") or transfer.get("created_skill_ids"):
            failures.append("memory_skill/frozen_transfer: target-side knowledge creation detected")
        if transfer.get("knowledge_records_hash_before") != transfer.get("knowledge_records_hash_after"):
            failures.append("memory_skill/frozen_transfer: records hash changed")
        if int(transfer.get("mutation_violation_count") or 0) != 0:
            failures.append("memory_skill/frozen_transfer: mutation_violation_count expected 0")
        snapshot_ids = set(transfer.get("snapshot_skill_ids") or [])
        exposed = set(transfer.get("exposed_skill_ids") or [])
        if not snapshot_ids or not (snapshot_ids & exposed):
            failures.append("memory_skill/frozen_transfer: no Frozen Snapshot Skill was exposed")
    if negative is not None:
        if not negative.get("initial_verification", {}).get("passed"):
            failures.append("memory_skill/frozen_negative_control: initial verification did not pass")
        if not negative.get("final_verification", {}).get("passed"):
            failures.append("memory_skill/frozen_negative_control: final verification did not pass")
        if negative.get("created_memory_ids") or negative.get("created_skill_ids"):
            failures.append("memory_skill/frozen_negative_control: knowledge creation detected")
        if negative.get("knowledge_records_hash_before") != negative.get("knowledge_records_hash_after"):
            failures.append("memory_skill/frozen_negative_control: records hash changed")
        if int(negative.get("mutation_violation_count") or 0) != 0:
            failures.append("memory_skill/frozen_negative_control: mutation_violation_count expected 0")
        if negative.get("modified_after_run"):
            failures.append("memory_skill/frozen_negative_control: workspace modified")
        if negative.get("harmful_memory_ids") or negative.get("harmful_skill_ids"):
            failures.append("memory_skill/frozen_negative_control: harmful knowledge IDs are not empty")


def _load_snapshot(report_path: Path, repeat: int) -> dict[str, Any]:
    path = report_path.parent / "memory_skill" / f"repeat_{repeat:02d}" / "knowledge_snapshot.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--required-modes", default="")
    args = parser.parse_args()
    required_modes = [item.strip() for item in args.required_modes.split(",") if item.strip()] or None
    ok, failures = validate(args.report, required_modes=required_modes)
    print("GO" if ok else "NO-GO")
    for failure in failures:
        print(f"- {failure}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
