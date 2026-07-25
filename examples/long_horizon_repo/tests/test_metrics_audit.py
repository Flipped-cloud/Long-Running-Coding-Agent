import json

from workflow_service.audit import AuditLog
from workflow_service.metrics import task_metrics
from workflow_service.models import Task, TaskStatus


def test_audit_sequence_survives_reopen(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"
    AuditLog(path).append("created", "a")
    AuditLog(path).append("started", "a")
    assert [event.sequence for event in AuditLog(path).read()] == [1, 2]


def test_audit_log_is_json_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    AuditLog(path).append("created", "a", {"source": "cli"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["details"] == {"source": "cli"}


def test_metrics_include_terminal_rate_and_retry_count():
    tasks = {
        "a": Task("a", "A", status=TaskStatus.SUCCEEDED, attempts=1),
        "b": Task("b", "B", status=TaskStatus.FAILED, attempts=3),
        "c": Task("c", "C"),
    }
    metrics = task_metrics(tasks)
    assert metrics["terminal"] == 2
    assert metrics["terminal_rate"] == 2 / 3
    assert metrics["retry_count"] == 2


def test_empty_metrics_are_defined():
    assert task_metrics({}) == {
        "total": 0,
        "status_counts": {},
        "attempts": 0,
        "terminal": 0,
        "terminal_rate": 0.0,
        "retry_count": 0,
    }
