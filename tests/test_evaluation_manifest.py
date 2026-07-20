from __future__ import annotations

from pathlib import Path

from longrun_agent.evaluation.coordinator import EvaluationCoordinator
from longrun_agent.evaluation.schema import AgentConfigReference, EvaluationManifest, EvaluationTaskCase


def test_manifest_expands_task_config_trial_seed_product(tmp_path: Path) -> None:
    manifest = EvaluationManifest(
        evaluation_id="eval",
        task_cases=[EvaluationTaskCase(case_id="a"), EvaluationTaskCase(case_id="b")],
        agent_configs=[AgentConfigReference(config_id="c1", path=tmp_path / "c1.yaml")],
        trial_count=2,
        seeds=[0, 1],
        output_root=tmp_path,
    )
    trials = EvaluationCoordinator(manifest, {}).expand_trials()
    assert len(trials) == 8
    assert len({item[2].trial_id for item in trials}) == 8
