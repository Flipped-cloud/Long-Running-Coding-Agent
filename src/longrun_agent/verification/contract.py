from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from longrun_agent.exceptions import ConfigurationError
from longrun_agent.verification.schema import (
    AgentVisibleContract,
    AgentVisibleIntegrityRules,
    CheckVisibility,
    OraclePrivateContract,
    VerificationContract,
)


def load_contract(path: Path, *, workspace_root: Path | None = None) -> VerificationContract:
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if payload.get("contract_visibility") == "agent":
            contract = AgentVisibleContract.model_validate(payload).to_runtime_contract()
        else:
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


def split_contract(contract: VerificationContract) -> tuple[AgentVisibleContract, OraclePrivateContract]:
    frozen = contract.freeze()
    rules = frozen.integrity_rules
    public = AgentVisibleContract(
        contract_id=frozen.contract_id,
        contract_version=frozen.contract_version,
        project_id=frozen.project_id,
        task_id=frozen.task_id,
        task_key=frozen.task_key,
        scope=frozen.scope,
        source=frozen.source,
        checks=[check for check in frozen.checks if check.visibility == CheckVisibility.PUBLIC],
        integrity_rules=AgentVisibleIntegrityRules(
            protected_paths=rules.protected_paths,
            trusted_test_patterns=rules.trusted_test_patterns,
            forbidden_change_patterns=rules.forbidden_change_patterns,
            allowed_change_patterns=rules.allowed_change_patterns,
            required_artifacts=rules.required_artifacts,
            max_deleted_files=rules.max_deleted_files,
            max_binary_file_bytes=rules.max_binary_file_bytes,
            allow_project_config_changes=rules.allow_project_config_changes,
            allowed_test_directories=rules.allowed_test_directories,
        ),
        generated_test_policy=frozen.generated_test_policy,
        completion_requirements=[],
        opaque_contract_fingerprint=frozen.contract_hash,
        inconclusive_action=frozen.inconclusive_action,
    )
    private = OraclePrivateContract(contract=frozen, private_fingerprint=frozen.contract_hash)
    return public, private


def private_marker_registry(private: OraclePrivateContract) -> set[str]:
    contract = private.contract
    markers = {"hidden_assets_root", "oracle_private", "private_fingerprint"}
    public_values = {
        value
        for check in contract.checks
        if check.visibility == CheckVisibility.PUBLIC
        for value in [check.check_id, check.title, *check.argv, *check.artifact_patterns, *check.tags]
    }
    if contract.hidden_assets_root:
        markers.add(str(contract.hidden_assets_root))
    markers.update(contract.integrity_rules.hidden_asset_markers)
    for check in contract.checks:
        if check.visibility != CheckVisibility.HIDDEN:
            continue
        markers.update({check.check_id, check.title})
        markers.update(item for item in check.argv if len(item) >= 6 and _private_argv_marker(item))
    return {marker for marker in markers if marker and len(marker.strip()) >= 6 and marker not in public_values}


def _private_argv_marker(value: str) -> bool:
    return "/" in value or "\\" in value or any(character.isspace() for character in value)
