from pathlib import Path

import pytest

from longrun_agent.config import ConfigurationError, expand_env_vars, get_api_key, load_config


def write_config(path: Path, workspace: Path) -> None:
    path.write_text(
        f"""
model:
  provider: fake
  model_name: fake
  base_url: null
  api_key_env: MODEL_API_KEY
  temperature: 0.0
  max_output_tokens: 100
  request_timeout_seconds: 30
  max_api_retries: 1
agent:
  max_steps: 3
  max_consecutive_errors: 2
workspace:
  root: {workspace.as_posix()}
tools:
  read_file:
    max_lines: 50
    max_chars: 1000
  write_file:
    max_chars: 1000
    save_diff: true
  bash:
    timeout_seconds: 5
    max_output_chars: 1000
    shell: false
telemetry:
  run_root: .runs
  save_prompts: true
  save_full_tool_outputs: true
""",
        encoding="utf-8",
    )


def test_env_var_expansion(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "fake-model")
    assert expand_env_vars({"model": "${MODEL_NAME}"})["model"] == "fake-model"


def test_load_config_with_env_override(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    write_config(cfg, tmp_path)
    monkeypatch.setenv("LONGRUN_MODEL", "new")
    assert load_config(cfg).model.model_name == "new"


def test_fake_provider_does_not_require_api_key(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    write_config(cfg, tmp_path)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert get_api_key(load_config(cfg).model) == ""


def test_real_provider_requires_api_key(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    write_config(cfg, tmp_path)
    config = load_config(cfg)
    config.model.provider = "openai_compatible"
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        get_api_key(config.model)
