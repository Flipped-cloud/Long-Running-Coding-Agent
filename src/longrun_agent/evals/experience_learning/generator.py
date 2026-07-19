from __future__ import annotations

from pathlib import Path

import yaml

from longrun_agent.config import expand_env_vars
from longrun_agent.evals.experience_learning.schemas import ExperienceLearningCase, ExperienceLearningConfig
from longrun_agent.exceptions import ConfigurationError


def load_experience_config(config_path: Path) -> ExperienceLearningConfig:
    resolved = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    raw = expand_env_vars(raw)
    base = resolved.parent
    raw["output_root"] = _resolve(base, raw.get("output_root", ".runs/experience_learning"))
    raw["cases"] = [_normalize_case(base, item) for item in raw.get("cases", [])]
    try:
        return ExperienceLearningConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"invalid experience learning config {resolved}: {exc}") from exc


def generate_cases(config_path: Path) -> list[ExperienceLearningCase]:
    return load_experience_config(config_path).cases


def _normalize_case(base: Path, item) -> dict:
    if isinstance(item, str):
        repo_name = item
        role = "negative_transfer" if "negative" in repo_name else "positive_transfer"
        task_file = "TASK.md"
        if repo_name == "repo_a":
            task_file = "TASK_FIX.md"
        return {
            "case_id": repo_name,
            "repository": _resolve(base, f"../../examples/knowledge_transfer/{repo_name}"),
            "task_file": task_file,
            "reset_script": "reset_repo.py",
            "role": role,
            "initial_verification_should_pass": role == "negative_transfer",
            "final_verification_should_pass": True,
            "is_negative_transfer": role == "negative_transfer",
        }
    normalized = dict(item)
    normalized["repository"] = _resolve(base, normalized["repository"])
    normalized["task_file"] = Path(normalized["task_file"])
    normalized["reset_script"] = Path(normalized["reset_script"])
    normalized["is_negative_transfer"] = normalized.get("role") == "negative_transfer"
    task_path = normalized["repository"] / normalized["task_file"]
    normalized["task"] = task_path.read_text(encoding="utf-8") if task_path.exists() else normalized.get("case_id", "")
    return normalized


def _resolve(base: Path, value) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()
