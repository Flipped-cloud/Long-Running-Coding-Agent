from __future__ import annotations

import json
import subprocess
from pathlib import Path

from longrun_agent.evaluation.schema import EvaluationOutcome, TerminationReason


class SWEbenchExportAdapter:
    def export_patch(self, *, instance_id: str, model_name_or_path: str, workspace: Path, output_path: Path) -> dict[str, str]:
        result = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=workspace,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
        payload = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": result.stdout,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def parse_external_report(
        self,
        path: Path,
        *,
        evaluation_id: str,
        config_id: str,
        trial_id: str,
        seed: int,
    ) -> EvaluationOutcome:
        payload = json.loads(path.read_text(encoding="utf-8"))
        resolved = bool(payload.get("resolved"))
        return EvaluationOutcome(
            evaluation_id=evaluation_id,
            case_id=str(payload.get("instance_id") or "unknown"),
            config_id=config_id,
            trial_id=trial_id,
            seed=seed,
            project_id=str(payload.get("instance_id") or "unknown"),
            project_status="verified" if resolved else "candidate_complete",
            verification_verdict="verified" if resolved else "reopened",
            runtime_verification_verdict=None,
            runtime_verification_report_id=None,
            oracle_verification_verdict="verified" if resolved else "reopened",
            oracle_verification_report_id=f"external-{payload['instance_id']}",
            oracle_total_requirements=2,
            oracle_verified_requirements=int(float(payload.get("f2p_rate", 0)) == 1) + int(float(payload.get("p2p_rate", 0)) == 1),
            termination_reason=TerminationReason.COMPLETED,
            full_resolution=resolved,
            f2p_rate=float(payload.get("f2p_rate") or int(resolved)),
            p2p_rate=float(payload.get("p2p_rate") or int(resolved)),
            integrity_passed=bool(payload.get("integrity_passed", resolved)),
            artifact_paths=[str(path)],
        )
