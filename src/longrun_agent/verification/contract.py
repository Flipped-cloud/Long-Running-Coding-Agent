from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from longrun_agent.exceptions import ConfigurationError
from longrun_agent.verification.schema import VerificationContract


def load_contract(path: Path, *, workspace_root: Path | None = None) -> VerificationContract:
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        contract = VerificationContract.model_validate(payload)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid verification contract {path}: {exc}") from exc
    if contract.hidden_assets_root is not None and not contract.hidden_assets_root.is_absolute():
        contract.hidden_assets_root = (path.parent / contract.hidden_assets_root).resolve()
    validate_contract_paths(contract, workspace_root=workspace_root)
    return contract


def validate_contract_paths(contract: VerificationContract, *, workspace_root: Path | None) -> None:
    if contract.hidden_assets_root is None:
        return
    hidden_root = contract.hidden_assets_root.resolve()
    if workspace_root is not None:
        workspace = workspace_root.resolve()
        if hidden_root == workspace or workspace in hidden_root.parents:
            raise ConfigurationError("hidden assets root must be outside workspace")
    if not hidden_root.exists() or not hidden_root.is_dir():
        raise ConfigurationError(f"hidden assets root does not exist: {hidden_root}")


def freeze_contract(contract: VerificationContract) -> VerificationContract:
    return contract.freeze()
