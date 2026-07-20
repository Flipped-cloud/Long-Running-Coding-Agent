from __future__ import annotations

from pathlib import Path

from longrun_agent.evaluation.adapters.local_project import LocalProjectAdapter
from longrun_agent.evaluation.coordinator import EvaluationCoordinator
from longrun_agent.evaluation.fake_provider import verification_bench_fake_provider
from longrun_agent.evaluation.reporting import read_trial_results
from longrun_agent.evaluation.schema import AgentConfigReference, EvaluationManifest, EvaluationTaskCase


def test_fake_system_e2e_all_conditions_have_oracle_requirements(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = EvaluationManifest(
        evaluation_id="fake-system-e2e",
        task_cases=[
            EvaluationTaskCase(
                case_id="full_fix",
                fixture=root / "examples" / "verification_bench" / "full_fix",
                task_file=Path("TASK.md"),
                contract_path=root / "examples" / "verification_bench" / "contracts" / "full_fix.yaml",
            )
        ],
        agent_configs=[
            AgentConfigReference(config_id=name, path=root / "configs" / f"{name}.yaml", mode=name)
            for name in (
                "verification_disabled",
                "legacy_command",
                "contract_verification",
                "contract_plus_generated_tests",
            )
        ],
        output_root=tmp_path / "evaluations",
    )
    coordinator = EvaluationCoordinator(
        manifest,
        {"local_project": LocalProjectAdapter(verification_bench_fake_provider)},
        preserve_workspaces=True,
    )

    report = coordinator.run()

    assert report["completed_count"] == 4
    assert report["error_count"] == 0
    overall = report["aggregate"]["overall"]["all"]
    assert overall["metrics"]["verified"]["mean"] == 1
    assert overall["metrics"]["f2p_rate"]["mean"] == 1
    assert overall["metrics"]["p2p_rate"]["mean"] == 1
    assert overall["metrics"]["verified_completion_count"]["mean"] == 1
    assert overall["success_at_k"] == 1
    rows = read_trial_results(coordinator.results_path)
    assert all(row.outcome is not None and row.outcome.oracle_total_requirements > 0 for row in rows)
    assert all(row.outcome is not None and row.outcome.verification_verdict == row.outcome.oracle_verification_verdict for row in rows)
    disabled = next(row.outcome for row in rows if row.descriptor.config_id == "verification_disabled")
    assert disabled is not None and disabled.runtime_verification_verdict is None
    assert len({row.metadata["oracle_contract_hash"] for row in rows}) == 1
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (coordinator.evaluation_dir / "report.json", coordinator.results_path, coordinator.events_path)
    )
    assert "hidden-negative-resolution" not in public_text
    assert "hidden_tests/test_resolution.py" not in public_text
