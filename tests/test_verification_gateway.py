from __future__ import annotations

import sys
from pathlib import Path

from longrun_agent.verification.gateway import VerificationGateway
from longrun_agent.verification.runner import VerificationRunner
from longrun_agent.verification.schema import (
    CheckKind,
    IntegrityRules,
    VerificationCheck,
    VerificationContract,
    VerificationPurpose,
    VerificationVerdict,
)
from longrun_agent.verification.snapshot import CopySnapshotProvider
from longrun_agent.verification.store import VerificationStore


def _gateway(tmp_path: Path, workspace: Path, contract: VerificationContract):
    store = VerificationStore(tmp_path / "state", contract.project_id, workspace_root=workspace)
    frozen = store.load_contract(contract.contract_id) if list(store.contracts_dir.glob("*.json")) else contract.freeze()
    if not list(store.contracts_dir.glob("*.json")):
        store.save_contract(frozen)
        frozen = store.load_contract(contract.contract_id)
    snapshots = CopySnapshotProvider(workspace, store.root)
    runner = VerificationRunner(store.root / "artifacts")
    return VerificationGateway(store=store, snapshot_manager=snapshots, runner=runner), snapshots, frozen


def test_gateway_verifies_f2p_and_p2p_in_isolated_snapshots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = workspace / "app.py"
    app.write_text("VALUE = 0\nREGRESSION = True\n", encoding="utf-8")
    contract = VerificationContract(
        contract_id="c1",
        project_id="p1",
        checks=[
            VerificationCheck(
                check_id="resolution",
                title="resolve issue",
                kind=CheckKind.RESOLUTION,
                visibility="hidden",
                argv=[sys.executable, "-c", "import app; raise SystemExit(0 if app.VALUE == 1 else 1)"],
            ),
            VerificationCheck(
                check_id="regression",
                title="preserve behavior",
                kind=CheckKind.REGRESSION,
                argv=[sys.executable, "-c", "import app; raise SystemExit(0 if app.REGRESSION else 1)"],
            ),
        ],
    )
    gateway, snapshots, frozen = _gateway(tmp_path, workspace, contract)
    snapshots.create_baseline()
    app.write_text("VALUE = 1\nREGRESSION = True\n", encoding="utf-8")

    report = gateway.verify(frozen)

    assert report.verdict == VerificationVerdict.VERIFIED
    assert report.purpose == VerificationPurpose.RUNTIME
    assert report.summary.f2p_rate == report.summary.p2p_rate == 1
    assert len(report.artifact_paths) == 8
    assert app.read_text(encoding="utf-8") == "VALUE = 1\nREGRESSION = True\n"
    public_events = (tmp_path / "state" / "p1" / "verification" / "events.jsonl").read_text(encoding="utf-8")
    private_events = (tmp_path / "state" / "p1" / "verification" / "private_events.jsonl").read_text(encoding="utf-8")
    assert '"check_id": "resolution"' not in public_events
    assert "resolution-baseline-stdout" not in public_events
    assert '"check_id": "resolution"' in private_events


def test_gateway_integrity_blocks_test_tampering(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_file = workspace / "test_trusted.py"
    test_file.write_text("TRUSTED = True\n", encoding="utf-8")
    contract = VerificationContract(
        contract_id="c2",
        project_id="p2",
        checks=[
            VerificationCheck(
                check_id="candidate",
                title="candidate",
                kind=CheckKind.CANDIDATE_ONLY,
                argv=[sys.executable, "-c", "raise SystemExit(0)"],
            )
        ],
        integrity_rules=IntegrityRules(trusted_test_patterns=["test_*.py"]),
    )
    gateway, snapshots, frozen = _gateway(tmp_path, workspace, contract)
    snapshots.create_baseline()
    test_file.write_text("TRUSTED = False\n", encoding="utf-8")

    report = gateway.verify(frozen)

    assert report.verdict == VerificationVerdict.REOPENED
    assert {item.category for item in report.integrity_violations} == {"trusted_test_modified"}


def test_gateway_contract_hash_mismatch_is_contract_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contract = VerificationContract(contract_id="c3", project_id="p3").freeze()
    store = VerificationStore(tmp_path / "state", "p3", workspace_root=workspace)
    snapshots = CopySnapshotProvider(workspace, store.root)
    gateway = VerificationGateway(store=store, snapshot_manager=snapshots, runner=VerificationRunner(store.root / "artifacts"))
    contract.project_id = "tampered"
    assert gateway.verify(contract).verdict == VerificationVerdict.CONTRACT_INVALID
