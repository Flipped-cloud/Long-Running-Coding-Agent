from __future__ import annotations

from longrun_agent.config import AppConfig
from longrun_agent.evaluation.schema import EvaluationTaskCase
from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.protocol import ModelResponse, ToolCall


def verification_bench_fake_provider(config: AppConfig, case: EvaluationTaskCase, seed: int) -> FakeModelProvider:
    del seed
    responses = [_plan(case)]
    write_path, content = _implementation(case.case_id)
    responses.append(_call("write_file", {"path": write_path, "content": content}))
    if config.verification.mode == "contract" and config.verification.generated_tests.enabled:
        test_path = "tests/test_generated_candidate.py"
        responses.extend(
            [
                _call(
                    "write_file",
                    {
                        "path": test_path,
                        "content": "from app import classify_number\n\n\ndef test_negative_issue():\n    assert classify_number(-1) == 'negative'\n",
                    },
                ),
                _call(
                    "register_test_candidate",
                    {
                        "paths": [test_path],
                        "command_argv": ["python", "-m", "pytest", "-q", test_path],
                        "issue_behavior": "negative numbers must be classified as negative",
                        "expected_failure_reason": "the baseline does not classify negative numbers",
                    },
                ),
            ]
        )
    responses.extend(
        [
            _call("bash", {"argv": ["python", "-m", "pytest", "-q"]}),
            _call(
                "request_task_completion",
                {
                    "summary": "Applied the deterministic fixture change and ran the public test suite.",
                    "acceptance_criteria_addressed": ["public tests executed"],
                },
            ),
        ]
    )
    return FakeModelProvider(responses)


def _plan(case: EvaluationTaskCase) -> ModelResponse:
    return _call(
        "submit_plan",
        {
            "project_summary": f"Run local verification fixture {case.case_id}",
            "tasks": [
                {
                    "key": "fixture-change",
                    "title": "Apply fixture change",
                    "objective": f"Implement the {case.case_id} fixture behavior",
                    "acceptance_criteria": ["public tests executed"],
                    "depends_on_keys": [],
                }
            ],
        },
    )


def _implementation(case_id: str) -> tuple[str, str]:
    if case_id == "test_tampering":
        return "tests/test_trusted.py", "from app import value\n\n\ndef test_value():\n    assert value() == 0\n"
    if case_id == "partial_fix":
        return (
            "app.py",
            "def classify_number(value: int) -> str:\n"
            "    if value < 0 and value % 2:\n"
            "        return 'negative'\n"
            "    return 'nonnegative'\n",
        )
    if case_id == "regression":
        return (
            "app.py",
            "def classify_number(value: int) -> str:\n    if value < 0:\n        return 'negative'\n    return 'nonnegative'\n",
        )
    return (
        "app.py",
        "def classify_number(value: int) -> str:\n"
        "    if value < 0:\n"
        "        return 'negative'\n"
        "    if value == 0:\n"
        "        return 'zero'\n"
        "    return 'positive'\n",
    )


def _call(name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(id=f"{name}-{len(str(arguments))}", name=name, arguments=arguments)])
