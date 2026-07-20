from __future__ import annotations

from pathlib import Path

import pytest

from longrun_agent.exceptions import StateStoreError
from longrun_agent.verification.schema import (
    CheckKind,
    VerificationCheck,
    VerificationContract,
    VerificationPurpose,
    VerificationReport,
    VerificationVerdict,
)
from longrun_agent.verification.store import VerificationStore


def test_verification_store_freezes_contract_and_skips_corrupt_report(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = VerificationStore(tmp_path / "state", "p1", workspace_root=workspace)
    contract = VerificationContract(
        contract_id="c1",
        project_id="p1",
        checks=[VerificationCheck(check_id="r", title="r", kind=CheckKind.RESOLUTION, argv=["python", "-V"])],
    )
    path = store.save_contract(contract)
    frozen = store.load_contract("c1")
    assert path.exists()
    assert store.verify_contract_hash(frozen)

    changed = frozen.model_copy(deep=True)
    changed.checks[0].title = "changed"
    with pytest.raises(StateStoreError, match="cannot be modified"):
        store.save_contract(changed)

    report = VerificationReport(project_id="p1", contract_id="c1", contract_hash=frozen.contract_hash, verdict=VerificationVerdict.VERIFIED)
    store.save_report(report)
    oracle_report = report.model_copy(update={"report_id": "oracle-report", "purpose": VerificationPurpose.ORACLE})
    store.save_report(oracle_report)
    (store.reports_dir / "corrupt.json").write_text("not-json", encoding="utf-8")
    assert {item.report_id for item in store.list_reports()} == {report.report_id, oracle_report.report_id}
    assert [item.report_id for item in store.list_reports(purpose=VerificationPurpose.RUNTIME)] == [report.report_id]
    assert [item.report_id for item in store.list_reports(purpose=VerificationPurpose.ORACLE)] == [oracle_report.report_id]
