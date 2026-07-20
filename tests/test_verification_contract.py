from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from longrun_agent.config import AppConfig
from longrun_agent.exceptions import ConfigurationError
from longrun_agent.verification.contract import load_contract
from longrun_agent.verification.schema import CheckKind, VerificationCheck, VerificationContract


def _check(check_id: str = "resolution") -> VerificationCheck:
    return VerificationCheck(check_id=check_id, title="resolution", kind=CheckKind.RESOLUTION, argv=["python", "-m", "pytest"])


def test_contract_hash_is_canonical_and_freeze_is_stable() -> None:
    first = VerificationContract(project_id="p1", checks=[_check()])
    second = VerificationContract.model_validate(first.model_dump())

    frozen = first.freeze()

    assert frozen.contract_hash == second.canonical_hash()
    assert frozen.freeze().contract_hash == frozen.contract_hash
    assert frozen.frozen_at


def test_contract_rejects_duplicate_check_ids() -> None:
    with pytest.raises(ValidationError, match="check IDs must be unique"):
        VerificationContract(project_id="p1", checks=[_check(), _check()])


def test_check_rejects_shell_syntax_and_escaping_cwd() -> None:
    with pytest.raises(ValidationError, match="forbidden shell syntax"):
        VerificationCheck(check_id="bad", title="bad", kind=CheckKind.RESOLUTION, argv=["pytest", "&&", "echo"])
    with pytest.raises(ValidationError, match="cwd must stay inside"):
        VerificationCheck(check_id="bad", title="bad", kind=CheckKind.RESOLUTION, argv=["pytest"], cwd="../outside")


def test_old_app_config_defaults_to_disabled_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = AppConfig.model_validate(
        {
            "model": {"provider": "fake"},
            "workspace": {"root": workspace},
            "state": {"root": tmp_path / "state"},
            "config_dir": tmp_path,
        }
    )
    assert config.verification.mode == "disabled"


def test_contract_mode_requires_path_and_frozen_contract() -> None:
    with pytest.raises(ValidationError, match="contract.path is required"):
        AppConfig.model_validate(
            {
                "model": {"provider": "fake"},
                "workspace": {"root": "workspace"},
                "verification": {"mode": "contract"},
            }
        )


def test_hidden_assets_root_must_exist_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    hidden = workspace / "hidden"
    hidden.mkdir(parents=True)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        f"project_id: p\nhidden_assets_root: {hidden.as_posix()}\nchecks: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="outside workspace"):
        load_contract(contract_path, workspace_root=workspace)

    contract_path.write_text(
        f"project_id: p\nhidden_assets_root: {(tmp_path / 'missing').as_posix()}\nchecks: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_contract(contract_path, workspace_root=workspace)
