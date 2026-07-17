import json

import yaml

from longrun_agent.config import AppConfig
from longrun_agent.context_probes.evaluator import evaluate_prediction
from longrun_agent.context_probes.generator import generate_cases
from longrun_agent.context_probes.reporting import write_run_report
from longrun_agent.context_probes.runner import run_probe
from longrun_agent.context_probes.schemas import ProbePrediction


def test_reporting_writes_jsonl_and_grouped_summary(tmp_path):
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=16)[0]
    prediction = ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode="full_history",
        predicted=case.expected,
        protocol_valid=True,
        actual_input_tokens=2048,
        output_tokens=12,
    )
    result = evaluate_prediction(case, prediction)

    summary = write_run_report(
        output_dir=tmp_path,
        run_id="run1",
        model="fake",
        seed=16,
        cases=[case],
        results=[result],
        modes=["full_history"],
        lengths=[2048],
        started_at="2026-01-01T00:00:00+00:00",
    )

    assert summary["by_mode"]["full_history"]["accuracy"]["numerator"] == 1
    for name in ["cases.jsonl", "predictions.jsonl", "results.jsonl"]:
        rows = [json.loads(line) for line in (tmp_path / name).read_text(encoding="utf-8").splitlines()]
        assert rows
    assert (tmp_path / "summary.csv").exists()


def test_reporting_counts_context_budget_exhaustion(tmp_path):
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=21)[0]
    prediction = ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode="full_history",
        protocol_valid=False,
        error_type="context_budget_exhausted",
        protocol_error="context_budget_exhausted",
        context_budget_exhausted=True,
        runtime_context_config={"model_context_limit": 4096},
    )
    result = evaluate_prediction(case, prediction)

    summary = write_run_report(
        output_dir=tmp_path,
        run_id="run-budget",
        model="fake",
        seed=21,
        cases=[case],
        results=[result],
        modes=["full_history"],
        lengths=[2048],
        started_at="2026-01-01T00:00:00+00:00",
    )

    assert summary["overall"]["accuracy"]["numerator"] == 0
    assert summary["overall"]["accuracy"]["denominator"] == 1
    assert summary["overall"]["provider_error_rate"]["numerator"] == 0
    assert summary["overall"]["context_budget_exhaustion_count"] == 1


def test_position_empty_groups_do_not_affect_gap(tmp_path):
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=22, position_ratios=[0.1])[0]
    prediction = ProbePrediction(
        case_id=case.case_id,
        probe=case.probe,
        mode="full_history",
        predicted=case.expected,
        protocol_valid=True,
    )
    result = evaluate_prediction(case, prediction)

    summary = write_run_report(
        output_dir=tmp_path,
        run_id="run-position",
        model="fake",
        seed=22,
        cases=[case],
        results=[result],
        modes=["full_history"],
        lengths=[2048],
        started_at="2026-01-01T00:00:00+00:00",
    )

    assert summary["position_probe"]["middle_accuracy"]["denominator"] == 0
    assert summary["position_probe"]["end_accuracy"]["denominator"] == 0
    assert summary["position_probe"]["best_worst_gap"] == 0.0


def test_summary_model_name_uses_resolved_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "DeepSeek-V4-Flash")
    app_config_path = tmp_path / "mode.yaml"
    payload = AppConfig(
        model={"provider": "openai_compatible", "model_name": "${MODEL_NAME}", "api_key_env": "OPENAI_API_KEY"},
        workspace={"root": tmp_path / "workspace"},
        state={"root": tmp_path / "state"},
        context={"mode": "full_history"},
    ).model_dump(mode="json")
    payload["model"]["model_name"] = "${MODEL_NAME}"
    app_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    eval_config_path = tmp_path / "eval.yaml"
    eval_config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "samples": 1,
                "lengths": [2048],
                "modes": ["full_history"],
                "mode_configs": {"full_history": str(app_config_path)},
                "output_root": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )

    result = run_probe(config_path=eval_config_path, probe="position", dry_run=True)

    assert result["summary"]["model"] == "DeepSeek-V4-Flash"
