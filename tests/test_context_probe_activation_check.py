import importlib.util
import json
from pathlib import Path

_ACTIVATION_CHECK_PATH = Path(__file__).resolve().parents[1] / "evals" / "context_lifecycle" / "activation_check.py"
_SPEC = importlib.util.spec_from_file_location("context_lifecycle_activation_check", _ACTIVATION_CHECK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
activation_passed = _MODULE.activation_passed
build_activation_report = _MODULE.build_activation_report


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
