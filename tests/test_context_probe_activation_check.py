import json

from evals.context_lifecycle.activation_check import activation_passed, build_activation_report


def test_activation_check_reports_all_modes(tmp_path):
    rows = [
        {"mode": "full_history", "context_reset_count": 0, "deterministic_prune_count": 0, "estimated_tokens_removed": 0},
        {
            "mode": "recent_window",
            "context_reset_count": 0,
            "recent_window_trim_count": 1,
            "deterministic_prune_count": 0,
            "estimated_tokens_removed": 120,
        },
        {"mode": "deterministic_prune", "context_reset_count": 0, "deterministic_prune_count": 2, "estimated_tokens_removed": 300},
        {
            "mode": "structured_reset",
            "context_reset_count": 1,
            "structured_reset_count": 1,
            "structured_compaction_count": 1,
            "estimated_tokens_removed": 400,
        },
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = build_activation_report(path)

    assert activation_passed(report)
    assert report["recent_window"]["trim_events"] == 1
    assert report["recent_window"]["removed_tokens"] == 120
    assert report["deterministic_prune"]["prune_events"] == 2
    assert report["structured_reset"]["reset_events"] == 1


def test_activation_check_fails_when_mode_does_not_trigger(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"mode": "recent_window", "estimated_tokens_removed": 0}), encoding="utf-8")

    report = build_activation_report(path)

    assert not activation_passed(report)
