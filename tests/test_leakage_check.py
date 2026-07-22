from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import app
from longrun_agent.evaluation.leakage import check_evaluation_leakage
from longrun_agent.verification.contract import split_contract
from longrun_agent.verification.schema import CheckKind, CheckVisibility, VerificationCheck, VerificationContract


def _run_root(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "evaluation"
    trial = root / "trials" / "trial-1"
    private_path = trial / "oracle" / "private" / "contract.json"
    private_path.parent.mkdir(parents=True)
    marker = "private-secret-resolution"
    contract = VerificationContract(
        contract_id="contract",
        project_id="project",
        checks=[
            VerificationCheck(
                check_id=marker,
                title="Private resolution title",
                kind=CheckKind.RESOLUTION,
                visibility=CheckVisibility.HIDDEN,
                argv=["python", "hidden_tests/private_resolution.py"],
            )
        ],
    )
    _, private = split_contract(contract)
    private_path.write_text(private.model_dump_json(indent=2), encoding="utf-8")
    telemetry = trial / "telemetry" / "run" / "prompt.json"
    telemetry.parent.mkdir(parents=True)
    return root, telemetry, marker


def test_leakage_check_clean_fixture_is_go(tmp_path: Path) -> None:
    root, telemetry, _ = _run_root(tmp_path)
    telemetry.write_text(json.dumps({"messages": [{"content": "public task context"}]}), encoding="utf-8")

    report = check_evaluation_leakage(root)

    assert report["status"] == "GO"
    assert report["leak_count"] == 0
    result = CliRunner().invoke(app, ["eval", "leakage-check", "--run-root", str(root)])
    assert result.exit_code == 0


def test_leakage_check_real_agent_visible_leak_is_nonzero_and_sanitized(tmp_path: Path) -> None:
    root, telemetry, marker = _run_root(tmp_path)
    telemetry.write_text(
        json.dumps({"messages": [{"role": "tool", "content": f"leaked {marker}"}]}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["eval", "leakage-check", "--run-root", str(root)])
    report = json.loads((root / "leakage_report.json").read_text(encoding="utf-8"))

    assert result.exit_code == 2
    assert report["status"] == "NO_GO"
    assert report["leak_count"] == 1
    assert report["affected_trials"] == ["trial-1"]
    assert report["findings"][0]["json_pointer"] == "/messages/0/content"
    assert marker not in json.dumps(report)
