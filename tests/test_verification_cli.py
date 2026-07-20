from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import app


def test_contract_validate_does_not_show_hidden_details(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    hidden = tmp_path / "hidden"
    workspace.mkdir()
    hidden.mkdir()
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        f"""
contract_id: cli-contract
project_id: p1
scope: project
hidden_assets_root: {hidden.as_posix()}
checks:
  - check_id: hidden-secret-name
    title: Secret assertion title
    kind: resolution
    visibility: hidden
    argv: [python, -V]
""",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
model: {{provider: fake, model_name: fake}}
workspace: {{root: {workspace.as_posix()}}}
state: {{root: {(tmp_path / "state").as_posix()}}}
verification:
  mode: contract
  contract: {{path: {contract.as_posix()}, freeze_on_project_start: true}}
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["verify", "contract", "validate", "--config", str(config)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["hidden_checks"] == 1
    assert "Secret assertion title" not in result.stdout


def test_eval_report_command_reads_existing_report(tmp_path: Path) -> None:
    report = tmp_path / "eval" / "report.json"
    report.parent.mkdir()
    report.write_text('{"evaluation_id": "eval", "completed_count": 1}', encoding="utf-8")
    result = CliRunner().invoke(app, ["eval", "report", "--evaluation-id", "eval", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["completed_count"] == 1
