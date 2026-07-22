from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from longrun_agent.config import load_config
from longrun_agent.evaluation.adapter import TaskAdapter
from longrun_agent.evaluation.attribution import FailureAttributor
from longrun_agent.evaluation.reporting import (
    append_trial_attempt,
    normalize_trial_result_store,
    upsert_trial_result,
    write_evaluation_report,
)
from longrun_agent.evaluation.schema import (
    EVALUATION_SEMANTICS_VERSION,
    EvaluationManifest,
    EvaluationTaskCase,
    TrialDescriptor,
    TrialResult,
    TrialStatus,
)
from longrun_agent.evaluation.trajectory import TrajectoryFeatureExtractor


class EvaluationCoordinator:
    def __init__(
        self,
        manifest: EvaluationManifest,
        adapters: dict[str, TaskAdapter],
        *,
        continue_on_case_error: bool = True,
        preserve_workspaces: bool = False,
    ):
        self.manifest = manifest
        self.adapters = adapters
        self.continue_on_case_error = continue_on_case_error
        self.preserve_workspaces = preserve_workspaces
        self.evaluation_dir = manifest.output_root / manifest.evaluation_id
        self.results_path = self.evaluation_dir / "trials.jsonl"
        self.attempts_path = self.evaluation_dir / "trial_attempts.jsonl"
        self.events_path = self.evaluation_dir / "events.jsonl"

    def expand_trials(self) -> list[tuple[EvaluationTaskCase, Path, TrialDescriptor]]:
        trials = []
        for case in self.manifest.task_cases:
            for config in self.manifest.agent_configs:
                for trial_number in range(1, self.manifest.trial_count + 1):
                    for seed in self.manifest.seeds:
                        trial_id = f"{case.case_id}-{config.config_id}-t{trial_number:02d}-s{seed}"
                        descriptor = TrialDescriptor(
                            evaluation_id=self.manifest.evaluation_id,
                            case_id=case.case_id,
                            config_id=config.config_id,
                            trial_id=trial_id,
                            trial_number=trial_number,
                            seed=seed,
                            trial_dir=self.evaluation_dir / "trials" / trial_id,
                            config_mode=config.mode,
                            shared_knowledge_root=(self.evaluation_dir / "shared_knowledge" if self.manifest.shared_knowledge else None),
                        )
                        trials.append((case, config.path, descriptor))
        return trials

    def run(self) -> dict[str, Any]:
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self._event("evaluation_started", evaluation_id=self.manifest.evaluation_id)
        _assert_semantics_compatible(self.results_path)
        canonical = normalize_trial_result_store(self.results_path, self.attempts_path)
        existing = {item.descriptor.trial_id: item for item in canonical}
        incompatible = [
            item.descriptor.trial_id
            for item in existing.values()
            if item.descriptor.status == TrialStatus.COMPLETED
            and item.metadata.get("evaluation_semantics_version") != EVALUATION_SEMANTICS_VERSION
        ]
        if incompatible:
            raise ValueError(
                "Existing evaluation results use incompatible semantics. Use a new evaluation_id or remove the old evaluation directory."
            )
        results = list(existing.values())
        for case, config_path, descriptor in self.expand_trials():
            prior = existing.get(descriptor.trial_id)
            if prior and prior.descriptor.status == TrialStatus.COMPLETED:
                self._event("trial_resumed", trial_id=descriptor.trial_id, status="already_completed")
                continue
            from longrun_agent.state.schema import utc_now

            started_at = utc_now()
            result = self._run_trial(case, config_path, descriptor)
            finished_at = utc_now()
            append_trial_attempt(
                self.attempts_path,
                result,
                started_at=started_at,
                finished_at=finished_at,
                retry_reason="retry_after_error" if prior and prior.descriptor.status == TrialStatus.ERROR else None,
            )
            upsert_trial_result(self.results_path, result)
            results = [item for item in results if item.descriptor.trial_id != descriptor.trial_id]
            results.append(result)
            if result.error and not self.continue_on_case_error:
                break
        report = write_evaluation_report(self.evaluation_dir, results)
        self._event("aggregate_report_created", evaluation_id=self.manifest.evaluation_id)
        return report

    def _run_trial(self, case: EvaluationTaskCase, config_path: Path, descriptor: TrialDescriptor) -> TrialResult:
        adapter = self.adapters.get(case.adapter)
        if adapter is None:
            descriptor.status = TrialStatus.ERROR
            return TrialResult(descriptor=descriptor, error=f"unknown task adapter: {case.adapter}")
        descriptor.status = TrialStatus.RUNNING
        self._event("trial_started", trial_id=descriptor.trial_id, case_id=case.case_id)
        try:
            adapter.prepare(case, descriptor)
            adapter.reset(case, descriptor)
            raw_outcome = adapter.run_agent(case, config_path, descriptor.seed, descriptor)
            verification_result = adapter.verify(case, raw_outcome, descriptor)
            outcome = adapter.collect_artifacts(case, raw_outcome, verification_result, descriptor)
            for snapshot in outcome.progress_snapshots:
                self._event(
                    "progress_snapshot_created",
                    project_id=outcome.project_id,
                    trial_id=descriptor.trial_id,
                    case_id=case.case_id,
                    report_id=snapshot.source_report_id,
                    evidence_ids=snapshot.passed_milestones,
                )
            events = _collect_events(descriptor.trial_dir)
            features = TrajectoryFeatureExtractor().extract([events])
            self._event(
                "trajectory_features_extracted",
                project_id=outcome.project_id,
                trial_id=descriptor.trial_id,
                case_id=case.case_id,
                evidence_ids=[item for item in [features.first_observable_symptom, features.first_causal_divergence] if item is not None],
            )
            attribution = FailureAttributor().attribute(
                case_id=case.case_id,
                trial_id=descriptor.trial_id,
                termination_reason=outcome.termination_reason,
                features=features,
                events=events,
                oracle_verification_verdict=outcome.oracle_verification_verdict,
                integrity_passed=outcome.integrity_passed,
                runtime_verification_verdict=outcome.runtime_verification_verdict,
            )
            if attribution is not None:
                outcome.failure_attribution_id = attribution.attribution_id
                self._event(
                    "failure_attribution_created",
                    project_id=outcome.project_id,
                    trial_id=descriptor.trial_id,
                    case_id=case.case_id,
                    sanitized_reason=attribution.primary_code,
                    evidence_ids=attribution.evidence_event_ids,
                )
            descriptor.status = TrialStatus.COMPLETED
            result = TrialResult(
                descriptor=descriptor,
                outcome=outcome,
                attribution=attribution,
                metadata={
                    "seed": descriptor.seed,
                    "config_path": str(config_path),
                    "contract_hash": verification_result.oracle_contract_hash,
                    "oracle_contract_id": verification_result.oracle_contract_id,
                    "oracle_contract_hash": verification_result.oracle_contract_hash,
                    "oracle_baseline_fingerprint": verification_result.oracle_baseline_fingerprint,
                    "oracle_candidate_fingerprint": verification_result.oracle_candidate_fingerprint,
                    "oracle_report_private_path": verification_result.oracle_report_private_path,
                    "evaluation_semantics_version": EVALUATION_SEMANTICS_VERSION,
                    **_config_metadata(config_path, descriptor.config_mode),
                },
            )
            self._event("trial_finished", trial_id=descriptor.trial_id, case_id=case.case_id)
            return result
        except Exception as exc:
            descriptor.status = TrialStatus.ERROR
            self._event("trial_error", trial_id=descriptor.trial_id, case_id=case.case_id, sanitized_reason=str(exc))
            error = f"{type(exc).__name__}: {exc}"
            attribution = FailureAttributor().attribute_error(
                case_id=case.case_id,
                trial_id=descriptor.trial_id,
                explanation=f"Evaluation harness failed with {type(exc).__name__}.",
            )
            return TrialResult(descriptor=descriptor, attribution=attribution, error=error)
        finally:
            adapter.cleanup(case, descriptor)
            if not self.preserve_workspaces and descriptor.status == TrialStatus.COMPLETED:
                shutil.rmtree(descriptor.trial_dir / "workspace", ignore_errors=True)

    def _event(self, event_type: str, **payload: Any) -> None:
        from longrun_agent.state.schema import utc_now

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        defaults = {
            "project_id": None,
            "task_id": None,
            "session_id": None,
            "contract_id": None,
            "contract_hash": None,
            "report_id": None,
            "check_id": None,
            "trial_id": None,
            "case_id": None,
            "verdict": None,
            "sanitized_reason": "",
            "evidence_ids": [],
            "artifact_paths": [],
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event_type": event_type, "timestamp": utc_now(), **defaults, **payload}, sort_keys=True) + "\n")


def _collect_events(trial_dir: Path) -> list[dict[str, Any]]:
    events = []
    for path in _allowed_event_paths(trial_dir):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evaluation event JSONL at {path}:{line_number}: {exc}") from exc
    return events


def _allowed_event_paths(trial_dir: Path) -> list[Path]:
    allowed = [
        *sorted((trial_dir / "state").glob("*/project_events.jsonl")),
        *sorted((trial_dir / "state").glob("*/verification/events.jsonl")),
        trial_dir / "knowledge" / "events.jsonl",
        trial_dir / "knowledge" / "uses.jsonl",
        trial_dir / "oracle" / "public_events.jsonl",
    ]
    telemetry = trial_dir / "telemetry"
    if telemetry.exists():
        allowed.extend(sorted(telemetry.rglob("events.jsonl")))
    return list(dict.fromkeys(path for path in allowed if path.exists() and path.is_file()))


def _assert_semantics_compatible(path: Path) -> None:
    if not path.exists():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid evaluation JSONL at {path}:{line_number}: {exc}") from exc
        status = (payload.get("descriptor") or {}).get("status")
        version = (payload.get("metadata") or {}).get("evaluation_semantics_version")
        if status == TrialStatus.COMPLETED.value and version != EVALUATION_SEMANTICS_VERSION:
            raise ValueError(
                "Existing evaluation results use incompatible semantics. Use a new evaluation_id or remove the old evaluation directory."
            )


def _config_metadata(config_path: Path, fallback_mode: str) -> dict[str, str]:
    if not config_path.exists():
        return {
            "mode": fallback_mode,
            "context_mode": "unknown",
            "planning_mode": "unknown",
            "knowledge_mode": "unknown",
            "verification_mode": "unknown",
        }
    config = load_config(config_path)
    return {
        "mode": fallback_mode,
        "context_mode": config.context.mode,
        "planning_mode": config.planning.mode,
        "knowledge_mode": config.knowledge.mode,
        "verification_mode": config.verification.mode,
    }
