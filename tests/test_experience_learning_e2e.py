from __future__ import annotations

import json
from pathlib import Path

import pytest

import longrun_agent.evals.experience_learning.executor as experience_executor
import longrun_agent.orchestration.orchestrator as orchestrator_module
from longrun_agent.evals.experience_learning.executor import _app_config, _task_with_preflight
from longrun_agent.evals.experience_learning.fake_provider import ProtocolAwareExperienceFakeProvider
from longrun_agent.evals.experience_learning.generator import load_experience_config
from longrun_agent.evals.experience_learning.runner import run_experience_learning
from longrun_agent.evals.experience_learning.schemas import VerificationResult
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.store import ProjectStateStore
from scripts.validate_experience_pilot import validate

MODES = ["disabled", "raw_episode", "reflection", "verified_memory", "memory_skill"]
CASE_IDS = ["repo_a_learn", "repo_a_reuse", "repo_b_transfer", "repo_c_negative"]


@pytest.fixture(autouse=True)
def forbid_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class ForbiddenApiProvider:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Experience Learning E2E tests must not instantiate a real API provider")

    monkeypatch.setattr(experience_executor, "OpenAICompatibleProvider", ForbiddenApiProvider)


def test_all_modes_real_e2e_semantics_and_artifacts(tmp_path: Path) -> None:
    config_path = _write_experience_config(tmp_path, modes=MODES, repeats=1)
    report = run_experience_learning(config_path)
    output_root = tmp_path / "runs"

    assert len(report["results"]) == 5
    assert len(report["case_results"]) == 20
    assert (output_root / "report.json").exists()
    _assert_repeat_artifacts(output_root, MODES, repeats=[1])

    summaries = {(item["mode"], item["repeat"]): item for item in report["results"]}
    assert (
        summaries[("disabled", 1)]
        | {
            "episode_count": 0,
            "reflection_candidate_count": 0,
            "active_memory_count": 0,
            "active_skill_count": 0,
            "memory_reference_count": 0,
            "skill_reference_count": 0,
            "negative_transfer_count": 0,
        }
        == summaries[("disabled", 1)]
    )
    assert summaries[("raw_episode", 1)]["episode_count"] == 4
    assert summaries[("raw_episode", 1)]["reflection_candidate_count"] == 0
    assert summaries[("raw_episode", 1)]["active_memory_count"] == 0
    assert summaries[("reflection", 1)]["episode_count"] == 4
    assert summaries[("reflection", 1)]["reflection_candidate_count"] == 1
    assert summaries[("reflection", 1)]["active_memory_count"] == 0
    assert summaries[("verified_memory", 1)]["episode_count"] == 4
    assert summaries[("verified_memory", 1)]["active_memory_count"] == 1
    assert summaries[("verified_memory", 1)]["memory_exposure_count"] == 2
    assert summaries[("verified_memory", 1)]["memory_reference_count"] == 2
    assert summaries[("verified_memory", 1)]["memory_helpful_count"] == 2
    assert summaries[("verified_memory", 1)]["active_skill_count"] == 0
    assert summaries[("memory_skill", 1)]["episode_count"] == 4
    assert summaries[("memory_skill", 1)]["active_memory_count"] == 1
    assert summaries[("memory_skill", 1)]["active_skill_count"] == 0
    assert summaries[("memory_skill", 1)]["validated_skill_count"] == 1
    assert summaries[("memory_skill", 1)]["portable_skill_count"] == 1
    assert summaries[("memory_skill", 1)]["skill_exposure_count"] == 1
    assert summaries[("memory_skill", 1)]["skill_reference_count"] == 1
    assert summaries[("memory_skill", 1)]["skill_helpful_count"] == 1
    assert summaries[("memory_skill", 1)]["cross_repository_skill_helpful_count"] == 1
    assert summaries[("memory_skill", 1)]["knowledge_assisted_transfer_success_rate"] == 1.0
    assert summaries[("memory_skill", 1)]["preexisting_skill_reference_count"] == 1
    assert summaries[("memory_skill", 1)]["preexisting_skill_helpful_count"] == 1
    assert summaries[("memory_skill", 1)]["target_side_skill_creation_count"] == 0
    assert summaries[("memory_skill", 1)]["frozen_store_mutation_count"] == 0
    assert summaries[("memory_skill", 1)]["duplicate_derivation_key_count"] == 0
    assert summaries[("memory_skill", 1)]["skills_created_without_source_memory_count"] == 0
    assert summaries[("memory_skill", 1)]["negative_transfer_count"] == 0
    for summary in summaries.values():
        assert "unsupported_shell_syntax_count" in summary
        assert "protocol_error_count" in summary
        assert summary["fatal_protocol_error_count"] == 0
        assert "provider_error_count" in summary

    cases = {(item["mode"], item["case_id"]): item for item in report["case_results"]}
    for mode in MODES:
        assert cases[(mode, "repo_a_learn")]["initial_verification"]["passed"] is False
        assert cases[(mode, "repo_a_learn")]["final_verification"]["passed"] is False
        assert cases[(mode, "repo_a_reuse")]["initial_verification"]["passed"] is False
        assert cases[(mode, "repo_a_reuse")]["final_verification"]["passed"] is True
        assert cases[(mode, "repo_b_transfer")]["initial_verification"]["passed"] is False
        assert cases[(mode, "repo_b_transfer")]["final_verification"]["passed"] is True
        assert cases[(mode, "repo_c_negative")]["initial_verification"]["passed"] is True
        assert cases[(mode, "repo_c_negative")]["final_verification"]["passed"] is True
        assert cases[(mode, "repo_c_negative")]["modified_after_run"] is False

    assert cases[("verified_memory", "repo_a_reuse")]["referenced_memory_ids"]
    assert cases[("memory_skill", "repo_a_reuse")]["helpful_memory_ids"]
    assert cases[("memory_skill", "repo_a_learn")]["created_memory_ids"]
    assert cases[("memory_skill", "repo_a_reuse")]["created_memory_ids"] == []
    assert cases[("memory_skill", "repo_a_reuse")]["created_skill_ids"]
    assert cases[("memory_skill", "repo_b_transfer")]["created_skill_ids"] == []
    assert cases[("memory_skill", "repo_b_transfer")]["created_memory_ids"] == []
    assert (
        cases[("memory_skill", "repo_b_transfer")]["knowledge_records_hash_before"]
        == cases[("memory_skill", "repo_b_transfer")]["knowledge_records_hash_after"]
    )
    assert cases[("memory_skill", "repo_b_transfer")]["mutation_violation_count"] == 0
    assert cases[("memory_skill", "repo_b_transfer")]["referenced_skill_ids"]
    assert cases[("memory_skill", "repo_b_transfer")]["helpful_skill_ids"]
    assert cases[("memory_skill", "repo_b_transfer")]["created_skill_ids"] == []
    assert cases[("memory_skill", "repo_c_negative")]["exposed_skill_ids"] == []
    assert cases[("memory_skill", "repo_c_negative")]["referenced_skill_ids"] == []
    assert cases[("memory_skill", "repo_c_negative")]["created_memory_ids"] == []
    assert cases[("memory_skill", "repo_c_negative")]["created_skill_ids"] == []
    assert (
        cases[("memory_skill", "repo_c_negative")]["knowledge_records_hash_before"]
        == cases[("memory_skill", "repo_c_negative")]["knowledge_records_hash_after"]
    )
    for case_id in CASE_IDS:
        repository = Path(cases[("memory_skill", case_id)]["repository"])
        assert repository == output_root / "memory_skill" / "repeat_01" / "workspaces" / case_id

    memory_skill_root = output_root / "memory_skill" / "repeat_01"
    snapshot = json.loads((memory_skill_root / "knowledge_snapshot.json").read_text(encoding="utf-8"))
    uses = _read_jsonl(memory_skill_root / "knowledge" / "uses.jsonl")
    assert any(use["use_type"] == "referenced" and use["memory_id"] for use in uses)
    assert any(use["use_type"] == "helpful" and use["skill_id"] for use in uses)
    assert all(use.get("project_id") and use.get("task_id") and use.get("session_id") for use in uses if use["use_type"] != "retrieved")
    repo_a_learn_session = _read_jsonl(memory_skill_root / "projects" / "memory_skill-r1-repo_a_learn" / "sessions.jsonl")[0]
    assert repo_a_learn_session["changed_files"] == []
    assert repo_a_learn_session["bash_observations"][0]["output_excerpt"]
    episode = json.loads(
        next((memory_skill_root / "projects" / "memory_skill-r1-repo_a_learn" / "knowledge" / "episodes").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    failed_test = next(item for item in episode["evidence_items"] if item["event_type"] == "test_command")
    assert "AssertionError" in failed_test["output_excerpt"]
    skills = list((memory_skill_root / "knowledge" / "skills").glob("*/skill.json"))
    assert len(skills) == 1
    skill = json.loads(skills[0].read_text(encoding="utf-8"))
    assert skill["scope"] == "portable"
    assert skill["source_memory_ids"] == cases[("memory_skill", "repo_a_reuse")]["referenced_memory_ids"]
    assert snapshot["skill_ids"] == [skill["skill_id"]]
    assert snapshot["skills"][0]["derivation_key"] == skill["derivation_key"]
    memories = list((memory_skill_root / "knowledge" / "memories").glob("*.json"))
    assert any(
        "Call strip() before checking whether the input is empty." in json.loads(path.read_text(encoding="utf-8"))["recommended_actions"]
        and json.loads(path.read_text(encoding="utf-8"))["scope"] == "portable"
        for path in memories
    )
    events = _read_jsonl(memory_skill_root / "knowledge" / "events.jsonl")
    repo_a_skill_event = next(
        event
        for event in events
        if event.get("event_type") == "skill_candidate_evaluated" and event.get("project_id") == "memory_skill-r1-repo_a_reuse"
    )
    assert repo_a_skill_event["candidate_created"] is True
    assert repo_a_skill_event["failure_stage"] == ""
    repo_b_skill_event = next(
        event
        for event in events
        if event.get("event_type") == "skill_candidate_evaluated" and event.get("project_id") == "memory_skill-r1-repo_b_transfer"
    )
    assert repo_b_skill_event["rejection_reasons"] == ["existing_helpful_skill_reused"]
    assert (
        cases[("memory_skill", "repo_b_transfer")]["referenced_skill_ids"]
        == cases[("memory_skill", "repo_b_transfer")]["helpful_skill_ids"]
    )
    assert (
        cases[("memory_skill", "repo_b_transfer")]["referenced_skill_ids"] == cases[("memory_skill", "repo_a_reuse")]["created_skill_ids"]
    )
    retrievals = _read_jsonl(memory_skill_root / "knowledge" / "retrievals.jsonl")
    repo_b_retrieval = next(
        item for item in retrievals if item.get("primary_skill_ids") == cases[("memory_skill", "repo_b_transfer")]["referenced_skill_ids"]
    )
    derived = next(item for item in repo_b_retrieval["retrieval_audit"] if item["item_id"] == skill["skill_id"])
    assert derived["origin"] == "derived_from_selected_memory"
    assert repo_b_retrieval["shadowed_memory_ids"] == skill["source_memory_ids"]


def test_snapshot_transfer_is_ready_when_model_reports_empty_knowledge_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_reports: list[str] = []

    class EmptyUsageExperienceProvider(ProtocolAwareExperienceFakeProvider):
        def _session_response(self, prompt: str) -> ModelResponse:
            if self.case.role != "positive_transfer":
                return super()._session_response(prompt)
            self.session_step += 1
            sequence = [
                (
                    "report_knowledge_use",
                    {
                        "memory_ids": [],
                        "skill_ids": [],
                        "reason": "The task was completed without claiming use of the exposed Skill.",
                    },
                ),
                ("read_file", {"path": "command_app.py"}),
                (
                    "write_file",
                    {
                        "path": "command_app.py",
                        "content": (
                            "def normalize_command(command: str) -> str:\n"
                            "    return command.strip()\n\n\n"
                            "def is_valid_command(command: str) -> bool:\n"
                            "    return bool(normalize_command(command))\n"
                        ),
                    },
                ),
                ("bash", {"argv": ["python", "-m", "pytest", "-q"]}),
                (
                    "request_task_completion",
                    {
                        "summary": "Command whitespace validation is fixed and pytest passes.",
                        "acceptance_criteria_addressed": ["pytest passes", "whitespace-only commands are rejected"],
                    },
                ),
            ]
            name, arguments = sequence[min(self.session_step - 1, len(sequence) - 1)]
            if name == "report_knowledge_use":
                empty_reports.append(self.case.case_id)
            return ModelResponse(
                tool_calls=[ToolCall(id=f"{name}-{self.session_step}", name=name, arguments=arguments)],
                usage={"input_tokens": 1, "output_tokens": 1},
            )

    monkeypatch.setattr(experience_executor, "ProtocolAwareExperienceFakeProvider", EmptyUsageExperienceProvider)
    config_path = _write_experience_config(tmp_path, modes=["memory_skill"], repeats=1)

    report = run_experience_learning(config_path)

    cases = {(item["mode"], item["case_id"]): item for item in report["case_results"]}
    transfer = cases[("memory_skill", "repo_b_transfer")]
    summary = report["results"][0]
    assert empty_reports == ["repo_b_transfer"]
    assert transfer["snapshot_skill_exposed"] is True
    assert transfer["frozen_transfer_completed"] is True
    assert transfer["final_verification"]["passed"] is True
    assert transfer["created_memory_ids"] == []
    assert transfer["created_skill_ids"] == []
    assert transfer["knowledge_records_hash_before"] == transfer["knowledge_records_hash_after"]
    assert transfer["mutation_violation_count"] == 0
    assert transfer["referenced_skill_ids"] == []
    assert transfer["helpful_skill_ids"] == []
    assert summary["skill_reference_count"] == 0
    assert summary["skill_helpful_count"] == 0
    assert summary["preexisting_skill_reference_count"] == 0
    assert summary["preexisting_skill_helpful_count"] == 0
    assert summary["knowledge_assisted_transfer_success_rate"] == 0.0
    assert summary["snapshot_skill_exposure_count"] >= 1
    assert summary["frozen_transfer_completion_rate"] == 1.0
    assert summary["frozen_negative_control_preservation_rate"] == 1.0
    assert not any(
        use.get("project_id") == transfer["project_id"] and use.get("use_type") in {"referenced", "helpful"}
        for use in _read_jsonl(tmp_path / "runs" / "memory_skill" / "repeat_01" / "knowledge" / "uses.jsonl")
    )
    ok, failures = validate(tmp_path / "runs" / "report.json", required_modes=["memory_skill"])
    assert ok
    assert failures == []


def test_mode_repeat_isolation_and_resume_do_not_duplicate_counts(tmp_path: Path) -> None:
    config_path = _write_experience_config(tmp_path, modes=["memory_skill"], repeats=2)
    report = run_experience_learning(config_path)
    output_root = tmp_path / "runs"
    _assert_repeat_artifacts(output_root, ["memory_skill"], repeats=[1, 2])

    summaries = {(item["mode"], item["repeat"]): item for item in report["results"]}
    assert summaries[("memory_skill", 1)]["active_skill_count"] == 0
    assert summaries[("memory_skill", 2)]["active_skill_count"] == 0
    assert summaries[("memory_skill", 1)]["validated_skill_count"] == 1
    assert summaries[("memory_skill", 2)]["validated_skill_count"] == 1
    assert summaries[("memory_skill", 1)]["skill_reference_count"] == 1
    assert summaries[("memory_skill", 2)]["skill_reference_count"] == 1

    repeat_1_root = output_root / "memory_skill" / "repeat_01"
    repeat_2_root = output_root / "memory_skill" / "repeat_02"
    assert _ids(repeat_1_root / "knowledge" / "memories", "*.json") != _ids(repeat_2_root / "knowledge" / "memories", "*.json")
    assert _ids(repeat_1_root / "knowledge" / "skills", "*/skill.json") != _ids(repeat_2_root / "knowledge" / "skills", "*/skill.json")
    assert all(use["project_id"].startswith("memory_skill-r1-") for use in _contextual_uses(repeat_1_root))
    assert all(use["project_id"].startswith("memory_skill-r2-") for use in _contextual_uses(repeat_2_root))

    loaded = load_experience_config(config_path)
    case = next(item for item in loaded.cases if item.case_id == "repo_b_transfer")
    case = case.model_copy(update={"repository": repeat_1_root / "workspaces" / "repo_b_transfer"})
    app_config = _app_config(
        case=case,
        mode="memory_skill",
        repeat_root=repeat_1_root,
        verification=loaded.verification,
        fail_fast=True,
        backend="fake",
    )
    project_id = "memory_skill-r1-repo_b_transfer"
    state_store = ProjectStateStore(app_config.state.root, workspace_root=app_config.workspace.root)
    before_sessions = len(state_store.read_sessions(project_id))
    before_uses = _read_jsonl(repeat_1_root / "knowledge" / "uses.jsonl")

    outcome = ProjectOrchestrator(app_config, ProtocolAwareExperienceFakeProvider(case)).resume(project_id)

    after_sessions = len(state_store.read_sessions(project_id))
    after_uses = _read_jsonl(repeat_1_root / "knowledge" / "uses.jsonl")
    assert outcome.project_id == project_id
    assert after_sessions == before_sessions
    assert after_uses == before_uses


def test_strict_errors_true_fails_on_knowledge_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_experience_config(tmp_path, modes=["verified_memory"], repeats=1, case_ids=["repo_a_reuse"])

    def fail_retrieval(*_args, **_kwargs):
        raise RuntimeError("injected knowledge failure")

    monkeypatch.setattr(orchestrator_module, "retrieve_bundle", fail_retrieval)
    with pytest.raises(RuntimeError, match="injected knowledge failure"):
        run_experience_learning(config_path)

    events = _read_jsonl(tmp_path / "runs" / "verified_memory" / "repeat_01" / "knowledge" / "events.jsonl")
    assert any(event.get("event_type") == "knowledge_error" for event in events)


def test_preflight_evidence_is_rendered_with_passed_state_and_without_workspace_path(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    passed = VerificationResult(
        command=["python", "-m", "pytest", "-q"],
        exit_code=0,
        stdout=f"{workspace}\\n2 passed",
        stderr="",
        passed=True,
    )
    failed = passed.model_copy(update={"exit_code": 1, "stdout": "1 failed", "passed": False})

    passed_task = _task_with_preflight("Task body", passed, workspace)
    failed_task = _task_with_preflight("Task body", failed, workspace)

    assert "passed: true" in passed_task
    assert "passed: false" in failed_task
    assert str(workspace) not in passed_task
    assert "do not edit without contradictory evidence" in passed_task


def _write_experience_config(
    tmp_path: Path,
    *,
    modes: list[str],
    repeats: int,
    case_ids: list[str] | None = None,
) -> Path:
    repos = {
        "repo_a_learn": Path("examples/knowledge_transfer/repo_a").resolve().as_posix(),
        "repo_a_reuse": Path("examples/knowledge_transfer/repo_a").resolve().as_posix(),
        "repo_b_transfer": Path("examples/knowledge_transfer/repo_b").resolve().as_posix(),
        "repo_c_negative": Path("examples/knowledge_transfer/repo_c_negative").resolve().as_posix(),
    }
    cases = {
        "repo_a_learn": ("TASK_LEARN.md", "learning_probe", "bootstrap_learning", "false", "false"),
        "repo_a_reuse": ("TASK_FIX.md", "same_repository_reuse", "bootstrap_consolidation", "false", "true"),
        "repo_b_transfer": ("TASK.md", "positive_transfer", "frozen_transfer", "false", "true"),
        "repo_c_negative": ("TASK.md", "negative_transfer", "frozen_negative_control", "true", "true"),
    }
    selected_cases = case_ids or CASE_IDS
    mode_lines = "\n".join(f"  - {mode}" for mode in modes)
    case_lines = []
    for case_id in selected_cases:
        task_file, role, knowledge_phase, initial, final = cases[case_id]
        case_lines.append(
            f"""
  - case_id: {case_id}
    repository: {repos[case_id]}
    task_file: {task_file}
    reset_script: reset_repo.py
    role: {role}
    knowledge_phase: {knowledge_phase}
    initial_verification_should_pass: {initial}
    final_verification_should_pass: {final}"""
        )
    path = tmp_path / "experience.yaml"
    path.write_text(
        f"""
seed: 7
backend: fake
repeats: {repeats}
output_root: {(tmp_path / "runs").as_posix()}
fail_fast_on_knowledge_error: true
modes:
{mode_lines}
cases:{"".join(case_lines)}
verification:
  command:
    - python
    - -m
    - pytest
    - -q
  timeout_seconds: 30
""",
        encoding="utf-8",
    )
    return path


def _assert_repeat_artifacts(output_root: Path, modes: list[str], *, repeats: list[int]) -> None:
    for mode in modes:
        for repeat in repeats:
            repeat_root = output_root / mode / f"repeat_{repeat:02d}"
            assert (repeat_root / "summary.json").exists()
            assert (repeat_root / "knowledge_snapshot.json").exists()
            assert (repeat_root / "projects").is_dir()
            assert (repeat_root / "telemetry").is_dir()
            assert (repeat_root / "knowledge" / "memories").is_dir()
            assert (repeat_root / "knowledge" / "skills").is_dir()
            assert (repeat_root / "knowledge" / "events.jsonl").exists()
            assert (repeat_root / "knowledge" / "uses.jsonl").exists()
            for case_id in CASE_IDS:
                assert (repeat_root / "cases" / case_id / "case_result.json").exists()
            summary = json.loads((repeat_root / "summary.json").read_text(encoding="utf-8"))
            episode_paths = list((repeat_root / "projects").glob("*/knowledge/episodes/*.json"))
            assert len(episode_paths) == summary["episode_count"]
            if mode in {"verified_memory", "memory_skill"}:
                assert list((repeat_root / "knowledge" / "memories").glob("*.json"))
            if mode == "memory_skill":
                assert list((repeat_root / "knowledge" / "skills").glob("*/skill.json"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(root: Path, pattern: str) -> set[str]:
    return {path.parent.name if path.name == "skill.json" else path.stem for path in root.glob(pattern)}


def _contextual_uses(repeat_root: Path) -> list[dict]:
    return [use for use in _read_jsonl(repeat_root / "knowledge" / "uses.jsonl") if use.get("project_id")]
