import json

from typer.testing import CliRunner

from longrun_agent.cli import app


def test_context_eval_cli_dry_run_writes_summary(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "context",
            "--config",
            "evals/context_lifecycle/config.yaml",
            "--probe",
            "all",
            "--lengths",
            "2048",
            "--samples",
            "1",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["probe"] == "all"
    assert payload["case_count"] == 4
    assert payload["prediction_count"] == 0
    assert (tmp_path / "summary.json").exists()
