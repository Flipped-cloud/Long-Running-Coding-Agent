from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from longrun_agent.evaluation.adapters.local_project import LocalProjectAdapter
from longrun_agent.evaluation.oracle import OfflineOracleEvaluator
from longrun_agent.evaluation.schema import EvaluationTaskCase, TrialDescriptor
from longrun_agent.tools.sandbox import EvaluationSandboxUnavailable


def _trial(tmp_path: Path) -> tuple[EvaluationTaskCase, TrialDescriptor, Path]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "TASK.md").write_text("Set VALUE to 1.", encoding="utf-8")
    (fixture / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    hidden = tmp_path / "hidden_assets"
    (hidden / "hidden_tests").mkdir(parents=True)
    (hidden / "hidden_tests" / "secret_resolution.py").write_text(
        "from app import VALUE\nraise SystemExit(0 if VALUE == 1 else 1)\n",
        encoding="utf-8",
    )
    contract = tmp_path / "contracts" / "contract.yaml"
    contract.parent.mkdir()
    python = Path(sys.executable).as_posix()
    contract.write_text(
        f"""
contract_id: private-oracle-contract
project_id: project
hidden_assets_root: {hidden.as_posix()}
checks:
  - check_id: public-resolution
    title: Public resolution
    kind: resolution
    visibility: public
    argv: [{python}, -c, \"from app import VALUE; raise SystemExit(0 if VALUE == 1 else 1)\"]
  - check_id: hidden-resolution
    title: Private oracle resolution
    kind: resolution
    visibility: hidden
    argv: [{python}, -c, \"from app import VALUE; raise SystemExit(0 if VALUE == 1 else 1)\"]
""",
        encoding="utf-8",
    )
    case = EvaluationTaskCase(case_id="case", fixture=fixture, task_file=Path("TASK.md"), contract_path=contract)
    descriptor = TrialDescriptor(
        evaluation_id="evaluation",
        case_id="case",
        config_id="config",
        trial_id="trial",
        trial_number=1,
        seed=0,
        trial_dir=tmp_path / "trials" / "trial",
    )
    return case, descriptor, contract


def test_oracle_private_contract_executes_public_and_hidden_checks(tmp_path: Path) -> None:
    case, descriptor, _ = _trial(tmp_path)
    adapter = LocalProjectAdapter(lambda *_: None)
    adapter.prepare(case, descriptor)
    adapter.reset(case, descriptor)
    workspace = adapter.workspace(case, descriptor)
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = OfflineOracleEvaluator().evaluate(
        case=case,
        descriptor=descriptor,
        project_id="project",
        final_workspace=workspace,
    )

    assert result.oracle_verdict == "verified"
    assert result.oracle_total_requirements == 2
    assert result.oracle_verified_requirements == 2
    public_contract = json.loads((workspace / ".longrun" / "agent_contract.json").read_text(encoding="utf-8"))
    assert [item["check_id"] for item in public_contract["checks"]] == ["public-resolution"]
    private_report = json.loads(Path(result.oracle_report_private_path).read_text(encoding="utf-8"))
    assert private_report["summary"]["hidden_checks_passed"] == 1


def test_formal_adapter_fails_closed_before_provider_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, descriptor, contract = _trial(tmp_path)
    config = tmp_path / "formal.yaml"
    config.write_text(
        f"""
model:
  provider: openai_compatible
  model_name: no-request-model
  base_url: https://example.invalid/v1
  api_key_env: NEVER_USED_API_KEY
workspace:
  root: {(tmp_path / "initial-workspace").as_posix()}
state:
  root: {(tmp_path / "state").as_posix()}
telemetry:
  run_root: {(tmp_path / "telemetry").as_posix()}
verification:
  mode: contract
  contract:
    path: {contract.as_posix()}
""",
        encoding="utf-8",
    )
    provider_calls = 0

    def provider_factory(*_):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider must not be created")

    def unavailable(_policy):
        raise EvaluationSandboxUnavailable("EVALUATION_SANDBOX_UNAVAILABLE")

    adapter = LocalProjectAdapter(provider_factory)
    adapter.prepare(case, descriptor)
    adapter.reset(case, descriptor)
    monkeypatch.setattr("longrun_agent.evaluation.adapters.local_project.build_subprocess_sandbox", unavailable)

    with pytest.raises(EvaluationSandboxUnavailable, match="EVALUATION_SANDBOX_UNAVAILABLE"):
        adapter.run_agent(case, config, 0, descriptor)
    assert provider_calls == 0
