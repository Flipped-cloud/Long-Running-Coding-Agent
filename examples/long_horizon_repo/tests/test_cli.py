import json
import subprocess
import sys


def run_cli(tmp_path, *args):
    db = tmp_path / "state" / "tasks.json"
    return subprocess.run(
        [sys.executable, "-m", "workflow_service.cli", "--db", str(db), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_create_get_list(tmp_path):
    created = run_cli(tmp_path, "create", "--id", "a", "--name", "A")
    assert created.returncode == 0
    assert json.loads(created.stdout)["id"] == "a"
    fetched = run_cli(tmp_path, "get", "--id", "a")
    assert json.loads(fetched.stdout)["name"] == "A"
    listed = run_cli(tmp_path, "list")
    assert [item["id"] for item in json.loads(listed.stdout)] == ["a"]


def test_cli_get_missing_is_nonzero(tmp_path):
    result = run_cli(tmp_path, "get", "--id", "missing")
    assert result.returncode != 0
    assert "unknown task id: missing" in result.stderr


def test_cli_run_retry_cancel_paths(tmp_path):
    run_cli(tmp_path, "create", "--id", "a", "--name", "A")
    ran = run_cli(tmp_path, "run", "--id", "a")
    assert ran.returncode == 0
    assert json.loads(ran.stdout)["status"] == "succeeded"
    cancelled = run_cli(tmp_path, "cancel", "--id", "a", "--reason", "stop")
    assert cancelled.returncode != 0
    assert "cannot cancel task from status: succeeded" in cancelled.stderr


def test_cli_retry_failed_task(tmp_path):
    run_cli(tmp_path, "create", "--id", "a", "--name", "A")
    failed = run_cli(tmp_path, "run", "--id", "a", "--fail", "boom")
    assert json.loads(failed.stdout)["status"] == "failed"
    retried = run_cli(tmp_path, "retry", "--id", "a")
    assert retried.returncode == 0
    assert json.loads(retried.stdout)["status"] == "pending"


def test_cli_import_export_round_trip(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('[{"id": "a", "name": "A"}]', encoding="utf-8")
    imported = run_cli(tmp_path, "import", "--input", str(source))
    assert imported.returncode == 0
    exported = tmp_path / "output.json"
    result = run_cli(tmp_path, "export", "--output", str(exported))
    assert result.returncode == 0
    assert json.loads(exported.read_text(encoding="utf-8"))[0]["id"] == "a"


def test_cli_metrics(tmp_path):
    run_cli(tmp_path, "create", "--id", "a", "--name", "A")
    result = run_cli(tmp_path, "metrics")
    assert result.returncode == 0
    assert json.loads(result.stdout)["total"] == 1
