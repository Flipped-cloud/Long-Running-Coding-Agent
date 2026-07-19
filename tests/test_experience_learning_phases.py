from __future__ import annotations

from pathlib import Path

from longrun_agent.evals.experience_learning.executor import _app_config
from longrun_agent.evals.experience_learning.generator import load_experience_config


def test_phase_field_drives_mutation_policy_without_case_id_branching(tmp_path: Path) -> None:
    config_path = tmp_path / "phases.yaml"
    repo = Path("examples/knowledge_transfer/repo_a").resolve().as_posix()
    config_path.write_text(
        f"""
backend: fake
output_root: {(tmp_path / "runs").as_posix()}
modes: [memory_skill]
cases:
  - case_id: arbitrary-bootstrap-name
    repository: {repo}
    task_file: TASK_LEARN.md
    reset_script: reset_repo.py
    role: learning_probe
    knowledge_phase: bootstrap_learning
    initial_verification_should_pass: false
    final_verification_should_pass: false
  - case_id: arbitrary-transfer-name
    repository: {repo}
    task_file: TASK_FIX.md
    reset_script: reset_repo.py
    role: positive_transfer
    knowledge_phase: frozen_transfer
    initial_verification_should_pass: false
    final_verification_should_pass: true
""",
        encoding="utf-8",
    )

    config = load_experience_config(config_path)
    bootstrap = _app_config(
        case=config.cases[0],
        mode="memory_skill",
        repeat_root=tmp_path / "bootstrap",
        verification=config.verification,
        fail_fast=True,
        backend="fake",
        record_mutation_policy="read_write",
    )
    transfer = _app_config(
        case=config.cases[1],
        mode="memory_skill",
        repeat_root=tmp_path / "transfer",
        verification=config.verification,
        fail_fast=True,
        backend="fake",
        record_mutation_policy="frozen_records",
    )

    assert config.cases[0].knowledge_phase == "bootstrap_learning"
    assert config.cases[1].knowledge_phase == "frozen_transfer"
    assert bootstrap.knowledge.record_mutation_policy == "read_write"
    assert transfer.knowledge.record_mutation_policy == "frozen_records"
