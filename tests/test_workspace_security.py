from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from longrun_agent.evaluation.adapters.local_project import LocalProjectAdapter
from longrun_agent.evaluation.schema import EvaluationTaskCase, TrialDescriptor
from longrun_agent.protocol import ErrorType, ToolCall, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext
from longrun_agent.tools.read_file import ReadFileTool
from longrun_agent.tools.result_guard import project_agent_visible_tool_result, sanitize_agent_visible_tool_result
from longrun_agent.tools.router import ToolRouter
from longrun_agent.tools.workspace_policy import ACCESS_DENIED_MESSAGE, WorkspaceAccessPolicy
from longrun_agent.tools.write_file import WriteFileTool
from longrun_agent.verification.contract import load_contract, private_marker_registry, split_contract
from longrun_agent.verification.schema import AgentVisibleContract, OraclePrivateContract


def _contract(path: Path, hidden_root: Path) -> None:
    path.write_text(
        f"""
contract_id: security-contract
project_id: project
hidden_assets_root: {hidden_root.as_posix()}
checks:
  - check_id: public-check
    title: Public check
    kind: regression
    visibility: public
    argv: [python, -c, \"print('public')\"]
  - check_id: secret-resolution-id
    title: Secret hidden requirement
    kind: resolution
    visibility: hidden
    argv: [python, hidden_tests/secret_resolution.py]
integrity_rules:
  hidden_asset_markers: [secret_resolution_marker]
""",
        encoding="utf-8",
    )


def test_file_tools_share_policy_and_reject_original_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original_contract = tmp_path / "contracts" / "full.yaml"
    original_contract.parent.mkdir()
    original_contract.write_text("private", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, denied_roots=[original_contract.parent])
    context = ToolContext(workspace, workspace_policy=policy)

    read = ToolRouter([ReadFileTool()]).execute(
        ToolCall(id="read", name="read_file", arguments={"path": str(original_contract)}),
        context,
    )
    write = ToolRouter([WriteFileTool()]).execute(
        ToolCall(id="write", name="write_file", arguments={"path": str(original_contract), "content": "changed"}),
        context,
    )

    assert context.workspace_policy is policy
    assert read.error_type == write.error_type == ErrorType.WORKSPACE_ACCESS_DENIED
    assert read.output == write.output == ACCESS_DENIED_MESSAGE
    assert original_contract.read_text(encoding="utf-8") == "private"


def test_adapter_projects_public_contract_and_freezes_private_contract(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "TASK.md").write_text("Fix it.", encoding="utf-8")
    hidden_root = tmp_path / "hidden_assets"
    hidden_root.mkdir()
    contract_path = tmp_path / "contracts" / "contract.yaml"
    contract_path.parent.mkdir()
    _contract(contract_path, hidden_root)
    descriptor = TrialDescriptor(
        evaluation_id="evaluation",
        case_id="case",
        config_id="config",
        trial_id="trial",
        trial_number=1,
        seed=0,
        trial_dir=tmp_path / "trials" / "trial",
    )
    case = EvaluationTaskCase(case_id="case", fixture=fixture, contract_path=contract_path)

    LocalProjectAdapter(lambda *_: None).prepare(case, descriptor)

    public_path = descriptor.trial_dir / "workspace" / ".longrun" / "agent_contract.json"
    private_path = descriptor.trial_dir / "oracle" / "private" / "contract.json"
    public_text = public_path.read_text(encoding="utf-8")
    public = AgentVisibleContract.model_validate_json(public_text)
    private = OraclePrivateContract.model_validate_json(private_path.read_text(encoding="utf-8"))
    assert [check.check_id for check in public.checks] == ["public-check"]
    assert public.opaque_contract_fingerprint == private.private_fingerprint
    for marker in (
        "hidden_assets_root",
        "secret-resolution-id",
        "Secret hidden requirement",
        "hidden_tests/secret_resolution.py",
        "secret_resolution_marker",
    ):
        assert marker not in public_text
    assert private.contract.hidden_assets_root == hidden_root.resolve()
    assert len(private.contract.checks) == 2


def test_public_contract_is_safe_for_prompt_snapshot(tmp_path: Path) -> None:
    hidden_root = tmp_path / "hidden"
    hidden_root.mkdir()
    path = tmp_path / "contract.yaml"
    _contract(path, hidden_root)
    public, _ = split_contract(load_contract(path, workspace_root=tmp_path / "workspace"))
    prompt_snapshot = json.dumps(
        {"messages": [{"role": "user", "content": public.model_dump_json()}]},
        sort_keys=True,
    )
    assert "secret-resolution-id" not in prompt_snapshot
    assert "hidden_assets_root" not in prompt_snapshot
    assert "secret_resolution_marker" not in prompt_snapshot


def test_tool_result_private_marker_is_blocked_and_privately_audited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    audit_path = tmp_path / "trial" / "oracle" / "private" / "tool_result_blocks.jsonl"
    marker = "secret_resolution_marker"
    policy = WorkspaceAccessPolicy.for_workspace(
        workspace,
        private_markers={marker},
        private_audit_path=audit_path,
    )
    raw = ToolResult(
        tool_call_id="call",
        tool_name="custom",
        success=True,
        summary="read succeeded",
        output=f"result contains {marker}",
    )

    guarded = sanitize_agent_visible_tool_result(raw, policy)

    assert guarded.error_type == ErrorType.PRIVATE_DATA_BLOCKED
    assert marker not in guarded.model_dump_json()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["event_type"] == "agent_visible_private_data_blocked"
    assert marker not in audit_path.read_text(encoding="utf-8")


def test_tool_result_metadata_private_paths_are_projected_without_blocking(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private_root = tmp_path / "oracle" / "private"
    policy = WorkspaceAccessPolicy.for_workspace(workspace, denied_roots=[private_root])
    raw = ToolResult(
        tool_call_id="call",
        tool_name="bash",
        success=True,
        summary="bash finished with exit code 0",
        output="[False, False, False]",
        metadata={
            "command": f"python -c inspect {private_root}",
            "normalized_command": f"python -c inspect {private_root}",
            "argv": ["python", "-c", str(private_root)],
            "cwd": str(tmp_path),
            "exit_code": 0,
            "output_artifact": str(private_root / "output.txt"),
        },
        artifact_path=str(private_root / "output.txt"),
    )

    visible = project_agent_visible_tool_result(raw, policy)

    assert visible.success is True
    assert visible.output == "[False, False, False]"
    assert visible.error_type is None
    assert visible.metadata == {"exit_code": 0}
    assert visible.artifact_path is None
    visible_json = visible.model_dump_json()
    assert str(private_root) not in visible_json
    assert "python -c inspect" not in visible_json
    assert raw.metadata["command"] == f"python -c inspect {private_root}"
    assert raw.metadata["argv"] == ["python", "-c", str(private_root)]
    assert raw.artifact_path == str(private_root / "output.txt")


def test_tool_result_workspace_artifact_is_projected_to_relative_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifact = workspace / ".runs" / "tool-output.txt"
    policy = WorkspaceAccessPolicy.for_workspace(workspace)
    raw = ToolResult(
        tool_call_id="call",
        tool_name="bash",
        success=True,
        summary="ok",
        artifact_path=str(artifact),
    )

    visible = project_agent_visible_tool_result(raw, policy)

    assert visible.artifact_path == ".runs/tool-output.txt"
    assert raw.artifact_path == str(artifact)


def test_tool_result_output_private_path_is_blocked(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private_root = tmp_path / "oracle" / "private"
    policy = WorkspaceAccessPolicy.for_workspace(workspace, denied_roots=[private_root])
    raw = ToolResult(
        tool_call_id="call",
        tool_name="bash",
        success=True,
        summary="ok",
        output=f"read {private_root / 'contract.json'}",
    )

    visible = project_agent_visible_tool_result(raw, policy)

    assert visible.error_type == ErrorType.PRIVATE_DATA_BLOCKED
    assert str(private_root) not in visible.model_dump_json()


def test_tool_result_error_message_private_marker_is_blocked(tmp_path: Path) -> None:
    marker = "hidden_tests/test_resolution.py"
    policy = WorkspaceAccessPolicy.for_workspace(tmp_path, private_markers={marker})
    raw = ToolResult(
        tool_call_id="call",
        tool_name="bash",
        success=False,
        summary="command failed",
        error_message=f"failed while loading {marker}",
    )

    visible = project_agent_visible_tool_result(raw, policy)

    assert visible.error_type == ErrorType.PRIVATE_DATA_BLOCKED
    assert marker not in visible.model_dump_json()


class _Args(BaseModel):
    pass


class _LeakingTool(BaseTool):
    name = "leak"
    description = "test"
    args_model = _Args

    def execute(self, call_id, arguments, context):
        return ToolResult(
            tool_call_id=call_id,
            tool_name=self.name,
            success=True,
            summary="ok",
            output="secret_resolution_marker",
        )


def test_router_applies_final_tool_result_guard(tmp_path: Path) -> None:
    policy = WorkspaceAccessPolicy.for_workspace(tmp_path, private_markers={"secret_resolution_marker"})
    context = ToolContext(tmp_path, workspace_policy=policy)
    result = ToolRouter([_LeakingTool()]).execute(ToolCall(id="leak", name="leak", arguments={}), context)
    assert result.error_type == ErrorType.PRIVATE_DATA_BLOCKED
    assert "secret_resolution_marker" not in result.output


def test_private_marker_registry_is_contract_driven(tmp_path: Path) -> None:
    hidden_root = tmp_path / "hidden"
    hidden_root.mkdir()
    path = tmp_path / "contract.yaml"
    _contract(path, hidden_root)
    _, private = split_contract(load_contract(path, workspace_root=tmp_path / "workspace"))
    markers = private_marker_registry(private)
    assert "secret-resolution-id" in markers
    assert "secret_resolution_marker" in markers
