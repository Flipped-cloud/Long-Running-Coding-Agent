from __future__ import annotations

import json
from pathlib import Path

from longrun_agent.evaluation.adapters.local_project import LocalProjectAdapter
from longrun_agent.evaluation.coordinator import EvaluationCoordinator
from longrun_agent.evaluation.fake_provider import verification_bench_fake_provider
from longrun_agent.evaluation.reporting import read_trial_attempts, read_trial_results
from longrun_agent.evaluation.schema import AgentConfigReference, EvaluationManifest, EvaluationTaskCase
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.protocol import ModelResponse, ToolCall


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
    attempts = read_trial_attempts(coordinator.attempts_path)
    assert len(coordinator.results_path.read_text(encoding="utf-8").splitlines()) == 4
    assert len(rows) == 4
    assert len(attempts) == 4
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

    resumed_report = coordinator.run()
    assert resumed_report["trial_count"] == 4
    assert len(coordinator.results_path.read_text(encoding="utf-8").splitlines()) == 4
    assert len(read_trial_attempts(coordinator.attempts_path)) == 4


def test_numeric_bash_argv_completes_local_project_evaluation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = EvaluationManifest(
        evaluation_id="numeric-argv-e2e",
        task_cases=[
            EvaluationTaskCase(
                case_id="full_fix",
                fixture=root / "examples" / "verification_bench" / "full_fix",
                task_file=Path("TASK.md"),
                contract_path=root / "examples" / "verification_bench" / "contracts" / "full_fix.yaml",
            )
        ],
        agent_configs=[
            AgentConfigReference(
                config_id="contract_verification",
                path=root / "configs" / "contract_verification.yaml",
                mode="contract_verification",
            )
        ],
        output_root=tmp_path / "evaluations",
    )

    def provider_factory(config, case, seed):
        original = verification_bench_fake_provider(config, case, seed)
        responses = list(original._responses)
        responses.insert(
            1,
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="real-regression-response",
                        name="bash",
                        arguments={"argv": ["find", ".", "-type", "f", "-maxdepth", 3], "cwd": "."},
                    )
                ]
            ),
        )
        return FakeModelProvider(responses)

    coordinator = EvaluationCoordinator(
        manifest,
        {"local_project": LocalProjectAdapter(provider_factory)},
        preserve_workspaces=True,
    )

    report = coordinator.run()

    assert report["completed_count"] == 1
    assert report["error_count"] == 0
    row = read_trial_results(coordinator.results_path)[0]
    assert row.error is None
    assert row.outcome is not None
    assert row.outcome.full_resolution is True
    assert row.metadata["oracle_report_private_path"]
    assert Path(row.metadata["oracle_report_private_path"]).exists()
    telemetry_events = [
        line
        for path in row.descriptor.trial_dir.joinpath("telemetry").rglob("events.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    normalized = [event for line in telemetry_events if (event := json.loads(line))["event_type"] == "tool_arguments_normalized"]
    assert len(normalized) == 1
    assert normalized[0]["tool_call_id"] == "real-regression-response"
    assert normalized[0]["payload"]["index"] == 5
