import json
from pathlib import Path

from longrun_agent.protocol import RunResult, RunStatus
from longrun_agent.telemetry.logger import EventLogger, is_sensitive_key, sanitize_payload


def run_result(logger: EventLogger, total_tokens: int = 120) -> RunResult:
    return RunResult(
        run_id=logger.run_id,
        status=RunStatus.COMPLETED,
        final_answer="done",
        steps=1,
        consecutive_errors=0,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        event_log_path=str(logger.events_path),
        run_json_path=str(logger.run_json_path),
        total_tokens=total_tokens,
    )


def test_sensitive_key_detection_does_not_redact_token_metrics():
    assert is_sensitive_key("api_key")
    assert is_sensitive_key("openai_api_key")
    assert is_sensitive_key("access_token")
    assert is_sensitive_key("refresh_token")
    assert is_sensitive_key("id_token")
    assert is_sensitive_key("authorization")
    assert is_sensitive_key("password")
    assert is_sensitive_key("client_secret")
    assert is_sensitive_key("secret")
    assert not is_sensitive_key("input_tokens")
    assert not is_sensitive_key("output_tokens")
    assert not is_sensitive_key("total_tokens")
    assert not is_sensitive_key("prompt_tokens")
    assert not is_sensitive_key("completion_tokens")
    assert not is_sensitive_key("max_output_tokens")
    assert not is_sensitive_key("token_count")


def test_sanitize_payload_redacts_nested_values_but_keeps_token_counts():
    payload = {
        "api_key": "secret-a",
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "max_output_tokens": 4096,
        "nested": {"password": "secret-b", "token_count": 7},
        "items": [{"access_token": "secret-c", "completion_tokens": 20}],
    }
    sanitized = sanitize_payload(payload)
    assert sanitized["api_key"] == "[redacted]"
    assert sanitized["nested"]["password"] == "[redacted]"
    assert sanitized["items"][0]["access_token"] == "[redacted]"
    assert sanitized["input_tokens"] == 100
    assert sanitized["output_tokens"] == 20
    assert sanitized["total_tokens"] == 120
    assert sanitized["max_output_tokens"] == 4096
    assert sanitized["nested"]["token_count"] == 7
    assert sanitized["items"][0]["completion_tokens"] == 20


def test_sanitize_payload_truncates_long_plain_strings():
    sanitized = sanitize_payload({"message": "x" * 5000})
    assert len(sanitized["message"]) < 5000
    assert "[truncated]" in sanitized["message"]


def test_event_logger_writes_valid_jsonl_run_json_and_redacts_secrets(tmp_path: Path):
    logger = EventLogger("r1", tmp_path / "r1", "fake")
    logger.log(
        1,
        "model_response",
        sanitized_arguments={"api_key": "secret-a", "input_tokens": 100},
        input_tokens=100,
        output_tokens=20,
        payload={
            "openai_api_key": "secret-b",
            "nested": {"password": "secret-c", "total_tokens": 120},
            "list": [{"access_token": "secret-d", "max_output_tokens": 4096}],
        },
    )
    logger.save_prompt(1, {"messages": [{"role": "user", "content": "hello"}], "model_api_key": "secret-e"})
    logger.save_run(
        run_result(logger, total_tokens=120),
        stats={"total_tokens": 120, "client_secret": "secret-f"},
    )

    event_text = logger.events_path.read_text(encoding="utf-8")
    prompt_text = (logger.prompts_dir / "model-turn-0001.json").read_text(encoding="utf-8")
    run_text = logger.run_json_path.read_text(encoding="utf-8")
    for secret in ["secret-a", "secret-b", "secret-c", "secret-d", "secret-e", "secret-f"]:
        assert secret not in event_text
        assert secret not in prompt_text
        assert secret not in run_text

    event = json.loads(event_text.splitlines()[0])
    assert event["input_tokens"] == 100
    assert event["output_tokens"] == 20
    assert event["sanitized_arguments"]["input_tokens"] == 100
    assert event["payload"]["nested"]["total_tokens"] == 120
    assert event["payload"]["list"][0]["max_output_tokens"] == 4096

    run_json = json.loads(run_text)
    assert run_json["total_tokens"] == 120
    assert run_json["stats"]["total_tokens"] == 120
    assert isinstance(run_json["total_tokens"], int)
    assert isinstance(run_json["stats"]["total_tokens"], int)
