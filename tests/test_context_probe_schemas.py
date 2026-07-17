import pytest
from pydantic import ValidationError

from longrun_agent.context_probes.generator import generate_cases
from longrun_agent.context_probes.schemas import ProbeCase


def test_probe_case_does_not_accept_mode_field():
    case = generate_cases(probe="position", lengths=[2048], samples=1, seed=1)[0]
    payload = case.model_dump()
    payload["mode"] = "full_history"

    with pytest.raises(ValidationError):
        ProbeCase.model_validate(payload)
