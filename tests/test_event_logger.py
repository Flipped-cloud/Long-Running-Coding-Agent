import json
from pathlib import Path

from longrun_agent.protocol import RunResult, RunStatus
from longrun_agent.telemetry.logger import EventLogger


def test_event_logger_writes_valid_jsonl_and_run_json(tmp_path: Path):
    logger = EventLogger("r1", tmp_path / "r1", "fake")
    logger.log(1, "run_started", sanitized_arguments={"API_KEY": "secret"})
    result = RunResult(
        run_id="r1",
        status=RunStatus.COMPLETED,
        final_answer="done",
        steps=1,
        consecutive_errors=0,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        event_log_path=str(logger.events_path),
        run_json_path=str(logger.run_json_path),
    )
    logger.save_run(result)
    lines = logger.events_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["event_type"] == "run_started"
    assert "secret" not in logger.events_path.read_text(encoding="utf-8")
    assert logger.run_json_path.exists()
