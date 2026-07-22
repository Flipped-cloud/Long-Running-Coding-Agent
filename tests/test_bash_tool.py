from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from longrun_agent.protocol import ErrorType, ToolCall
from longrun_agent.tools.base import ToolContext
from longrun_agent.tools.bash import BashTool
from longrun_agent.tools.router import ToolRouter
from longrun_agent.tools.sandbox import (
    SANDBOX_BASE_ROOT,
    SANDBOX_RUNTIME_UNAVAILABLE,
    SANDBOX_VENV_ROOT,
    EvaluationSandbox,
    EvaluationSandboxRuntimeUnavailable,
    EvaluationSandboxUnavailable,
    RuntimeSymlinkHop,
    SandboxProcessFailure,
    SubprocessSandbox,
    TrustedRuntimeMountPlan,
)
from longrun_agent.tools.workspace_policy import ACCESS_DENIED_MESSAGE, WorkspaceAccessPolicy


def test_runtime_symlink_hop_path_type_is_derived_from_relative_flag(tmp_path: Path) -> None:
    common = {
        "link_path": tmp_path / "python",
        "raw_target": "python3",
        "absolute_target": tmp_path / "python3",
        "resolved_target": tmp_path / "python3",
        "target_exists": True,
        "sandbox_link_path": None,
        "sandbox_target_path": None,
    }

    relative = RuntimeSymlinkHop(target_is_relative=True, **common)
    absolute = RuntimeSymlinkHop(target_is_relative=False, **common)

    assert relative.path_type == "relative_symlink"
    assert absolute.path_type == "absolute_symlink"


class RecordingSandbox(SubprocessSandbox):
    def __init__(self) -> None:
        self.calls = 0

    def popen(self, argv, *, cwd, shell, **kwargs):
        self.calls += 1
        raise AssertionError("denied command started a subprocess")


def _execute(workspace: Path, sandbox: RecordingSandbox, arguments: dict):
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    context = ToolContext(workspace, workspace_policy=policy, subprocess_sandbox=sandbox)
    return ToolRouter([BashTool()]).execute(ToolCall(id="bash-1", name="bash", arguments=arguments), context)


@pytest.mark.parametrize(
    "arguments",
    [
        {"argv": ["cat", "{outside}"]},
        {"argv": ["cat", "../outside.txt"]},
        {"argv": ["cat", "file:///outside.txt"]},
        {"argv": ["python", "-c", "print('no')"], "cwd": ".."},
    ],
)
def test_bash_rejects_workspace_escape_without_starting_subprocess(tmp_path: Path, arguments: dict) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    arguments = {
        key: [item.format(outside=outside) for item in value] if isinstance(value, list) else value for key, value in arguments.items()
    }
    sandbox = RecordingSandbox()

    result = _execute(workspace, sandbox, arguments)

    assert result.success is False
    assert result.error_type == ErrorType.WORKSPACE_ACCESS_DENIED
    assert result.output == ACCESS_DENIED_MESSAGE
    assert str(outside) not in result.output
    assert sandbox.calls == 0


def test_bash_rejects_symlink_escape_without_starting_subprocess(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    sandbox = RecordingSandbox()

    result = _execute(workspace, sandbox, {"argv": ["cat", "link.txt"]})

    assert result.error_type == ErrorType.WORKSPACE_ACCESS_DENIED
    assert sandbox.calls == 0


def test_bash_rejects_other_trial_and_oracle_private_store(tmp_path: Path) -> None:
    current = tmp_path / "trials" / "current" / "workspace"
    current.mkdir(parents=True)
    other = tmp_path / "trials" / "other" / "workspace" / "secret.txt"
    private = tmp_path / "trials" / "current" / "oracle" / "private" / "contract.json"
    other.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    other.write_text("other", encoding="utf-8")
    private.write_text("private", encoding="utf-8")

    for path in (other, private):
        sandbox = RecordingSandbox()
        result = _execute(current, sandbox, {"argv": ["cat", str(path)]})
        assert result.error_type == ErrorType.WORKSPACE_ACCESS_DENIED
        assert sandbox.calls == 0


def test_evaluation_sandbox_blocks_python_code_host_read_or_fails_closed(tmp_path: Path) -> None:
    from longrun_agent.tools.sandbox import build_subprocess_sandbox

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "host-secret.txt"
    outside.write_text("private", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    try:
        sandbox = build_subprocess_sandbox(policy)
    except EvaluationSandboxRuntimeUnavailable:
        raise
    except EvaluationSandboxUnavailable as exc:
        assert str(exc) == "EVALUATION_SANDBOX_UNAVAILABLE"
        return

    context = ToolContext(workspace, workspace_policy=policy, subprocess_sandbox=sandbox)
    code = f"from pathlib import Path; print(Path({str(outside)!r}).exists())"
    result = ToolRouter([BashTool()]).execute(
        ToolCall(id="bash-1", name="bash", arguments={"argv": [os.fspath(Path(os.sys.executable)), "-c", code]}),
        context,
    )
    assert result.success
    assert "False" in result.output


def test_runtime_mount_plan_covers_interpreter_prefix_and_base_prefix(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)

    plan = TrustedRuntimeMountPlan.from_current_process(policy, system_roots=())

    assert any(plan.executable.is_relative_to(root) for root in plan.runtime_roots)
    assert any(plan.resolved_executable.is_relative_to(root) for root in plan.runtime_roots)
    assert any(plan.prefix == root or plan.prefix.is_relative_to(root) for root in plan.runtime_roots)
    assert any(plan.base_prefix == root or plan.base_prefix.is_relative_to(root) for root in plan.runtime_roots)
    assert str(plan.canonical_executable).startswith(str(SANDBOX_VENV_ROOT))
    assert str(plan.canonical_resolved_executable).startswith((str(SANDBOX_VENV_ROOT), str(SANDBOX_BASE_ROOT)))


def test_runtime_mount_plan_preserves_symlink_entry_and_real_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "venv"
    base_prefix = tmp_path / "trusted-python"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    target = base_prefix / "bin" / "python3"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    executable = prefix / "bin" / "python"
    try:
        executable.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)

    plan = TrustedRuntimeMountPlan.from_current_process(
        policy,
        executable=executable,
        prefix=prefix,
        base_prefix=base_prefix,
        system_roots=(),
    )

    assert plan.executable == executable.absolute()
    assert plan.resolved_executable == target.resolve()
    assert set(plan.runtime_roots) == {prefix.resolve(), base_prefix.resolve()}


def test_bwrap_command_mounts_only_frozen_runtime_roots_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    plan = TrustedRuntimeMountPlan.from_current_process(
        WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True),
        system_roots=(),
    )
    sandbox = EvaluationSandbox(
        WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True),
        runtime_plan=plan,
        bwrap_path=fake_bwrap,
    )

    command = sandbox.build_command([str(plan.executable), "-c", "print('ok')"], cwd=workspace)

    triples = [command[index : index + 3] for index in range(len(command) - 2)]
    assert ["--ro-bind", str(plan.prefix), str(SANDBOX_VENV_ROOT)] in triples
    assert ["--ro-bind", str(plan.base_prefix), str(SANDBOX_BASE_ROOT)] in triples
    assert ["--ro-bind", "/", "/"] not in [command[index : index + 3] for index in range(len(command) - 2)]
    assert str(workspace.parent) not in command
    assert command[command.index("--") + 1] == str(plan.canonical_executable)
    assert str(plan.executable) not in command[command.index("--") + 1 :]


def test_absolute_runtime_executable_rewrites_to_canonical_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "venv"
    base_prefix = tmp_path / "trusted-python"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    target = base_prefix / "bin" / "python3.12"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    python312 = prefix / "bin" / "python3.12"
    python = prefix / "bin" / "python"
    try:
        python312.symlink_to(target)
        python.symlink_to("python3.12")
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    plan = TrustedRuntimeMountPlan.from_current_process(
        policy,
        executable=python,
        prefix=prefix,
        base_prefix=base_prefix,
        system_roots=(),
    )
    sandbox = EvaluationSandbox(policy, runtime_plan=plan, bwrap_path=fake_bwrap)

    resolved = sandbox.resolve_allowed_executable(str(python), cwd=workspace)
    command = sandbox.build_command([str(python), "-c", "print('ok')"], cwd=workspace)

    assert resolved.host_requested_path == python.absolute()
    assert resolved.host_resolved_path == target.resolve()
    assert resolved.sandbox_executable_path == SANDBOX_VENV_ROOT / "bin/python"
    assert resolved.runtime_kind == "venv"
    assert command[command.index("--") + 1] == str(SANDBOX_VENV_ROOT / "bin/python")
    assert str(python) not in command[command.index("--") + 1 :]
    assert any(item.path_type == "relative_symlink" for item in resolved.interpreter_chain)
    assert any(item.path_type == "absolute_symlink" for item in resolved.interpreter_chain)


def test_console_script_and_bare_python_resolve_through_canonical_runtime_path(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "venv"
    base_prefix = tmp_path / "trusted-python"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    target = base_prefix / "bin" / "python3.12"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    python = prefix / "bin" / "python"
    try:
        python.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    pytest_script = prefix / "bin" / "pytest"
    pytest_script.write_text(f"#!{python}\nprint('pytest')\n", encoding="utf-8")
    pytest_script.chmod(0o755)
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    monkeypatch.setenv("PATH", str(prefix / "bin"))
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    plan = TrustedRuntimeMountPlan.from_current_process(
        policy,
        executable=python,
        prefix=prefix,
        base_prefix=base_prefix,
        system_roots=(),
    )
    sandbox = EvaluationSandbox(policy, runtime_plan=plan, bwrap_path=fake_bwrap)

    python_command = sandbox.build_command(["python", "-m", "pytest", "-q"], cwd=workspace)
    pytest_command = sandbox.build_command(["pytest", "-q"], cwd=workspace)
    direct_pytest_command = sandbox.build_command([str(pytest_script), "-q"], cwd=workspace)

    assert python_command[python_command.index("--") + 1] == str(SANDBOX_VENV_ROOT / "bin/python")
    assert pytest_command[pytest_command.index("--") + 1] == str(SANDBOX_VENV_ROOT / "bin/pytest")
    assert direct_pytest_command[direct_pytest_command.index("--") + 1] == str(SANDBOX_VENV_ROOT / "bin/pytest")


def test_multihop_absolute_symlink_creates_exact_intermediate_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "repo" / ".venv"
    home = tmp_path / "home" / "user"
    base_prefix = home / ".local" / "share" / "uv" / "python" / "cpython-3.12.13-linux-x86_64-gnu"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (home / ".local" / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    target = base_prefix / "bin" / "python3.12"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    local_alias = home / ".local" / "bin" / "python3.12"
    python312 = prefix / "bin" / "python3.12"
    python = prefix / "bin" / "python"
    try:
        local_alias.symlink_to(target)
        python312.symlink_to(local_alias)
        python.symlink_to("python3.12")
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)

    plan = TrustedRuntimeMountPlan.from_current_process(
        policy,
        executable=python,
        prefix=prefix,
        base_prefix=base_prefix,
        system_roots=(),
    )

    assert [hop.raw_target for hop in plan.symlink_chain] == ["python3.12", str(local_alias), str(target)]
    assert [hop.target_is_relative for hop in plan.symlink_chain] == [True, False, False]
    intermediate_aliases = [alias for alias in plan.aliases if alias.alias_kind == "intermediate_symlink_alias"]
    assert len(intermediate_aliases) == 1
    assert intermediate_aliases[0].host_alias_path == local_alias.absolute()
    assert intermediate_aliases[0].sandbox_target_path == SANDBOX_BASE_ROOT / "bin/python3.12"
    assert intermediate_aliases[0].source_hop_index == 1


def test_intermediate_symlink_target_outside_trusted_runtime_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "repo" / ".venv"
    base_prefix = tmp_path / "trusted-python"
    untrusted = tmp_path / "untrusted" / "bin"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    untrusted.mkdir(parents=True)
    target = untrusted / "python3.12"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    python = prefix / "bin" / "python"
    try:
        python.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)

    with pytest.raises(EvaluationSandboxRuntimeUnavailable, match=SANDBOX_RUNTIME_UNAVAILABLE):
        TrustedRuntimeMountPlan.from_current_process(
            policy,
            executable=python,
            prefix=prefix,
            base_prefix=base_prefix,
            system_roots=(),
        )


def test_symlink_loop_is_rejected_fail_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "repo" / ".venv"
    base_prefix = tmp_path / "trusted-python"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    try:
        python.symlink_to("python")
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)

    with pytest.raises(EvaluationSandboxRuntimeUnavailable, match=SANDBOX_RUNTIME_UNAVAILABLE):
        TrustedRuntimeMountPlan.from_current_process(
            policy,
            executable=python,
            prefix=prefix,
            base_prefix=base_prefix,
            system_roots=(),
        )


def test_mount_plan_uses_file_alias_without_mounting_intermediate_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "repo" / ".venv"
    home = tmp_path / "home" / "user"
    base_prefix = home / ".local" / "share" / "uv" / "python" / "cpython-3.12.13-linux-x86_64-gnu"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (home / ".local" / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    target = base_prefix / "bin" / "python3.12"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    local_alias = home / ".local" / "bin" / "python3.12"
    python = prefix / "bin" / "python"
    try:
        local_alias.symlink_to(target)
        python.symlink_to(local_alias)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    plan = TrustedRuntimeMountPlan.from_current_process(
        policy,
        executable=python,
        prefix=prefix,
        base_prefix=base_prefix,
        system_roots=(),
    )
    sandbox = EvaluationSandbox(policy, runtime_plan=plan, bwrap_path=fake_bwrap)

    command = sandbox.build_command([str(python), "-c", "print('ok')"], cwd=workspace)
    triples = [command[index : index + 3] for index in range(len(command) - 2)]

    assert ["--symlink", str(SANDBOX_BASE_ROOT / "bin/python3.12"), local_alias.as_posix()] in triples
    assert ["--ro-bind", str(home), home.as_posix()] not in triples
    assert ["--ro-bind", str(home / ".local"), (home / ".local").as_posix()] not in triples
    assert ["--ro-bind", str(home / ".local" / "bin"), (home / ".local" / "bin").as_posix()] not in triples
    assert ["--ro-bind", str(tmp_path), tmp_path.as_posix()] not in triples


def test_sandbox_path_contains_only_canonical_runtime_and_system_paths(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    prefix = tmp_path / "repo" / ".venv"
    base_prefix = tmp_path / "trusted-python"
    outside_bin = tmp_path / "home" / "user" / ".local" / "bin"
    workspace.mkdir()
    (prefix / "bin").mkdir(parents=True)
    (base_prefix / "bin").mkdir(parents=True)
    outside_bin.mkdir(parents=True)
    target = base_prefix / "bin" / "python"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(outside_bin), "/usr/local/bin", "/usr/bin", "/bin"]))
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    plan = TrustedRuntimeMountPlan.from_current_process(
        policy,
        executable=target,
        prefix=base_prefix,
        base_prefix=base_prefix,
        system_roots=(),
    )
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    sandbox = EvaluationSandbox(policy, runtime_plan=plan, bwrap_path=fake_bwrap)

    assert sandbox.environment["PATH"] == (f"{SANDBOX_VENV_ROOT / 'bin'}:{SANDBOX_BASE_ROOT / 'bin'}:/usr/bin:/bin")


def test_absolute_untrusted_executable_is_denied_without_process_start(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-program"
    outside.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    outside.chmod(0o755)
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    plan = TrustedRuntimeMountPlan.from_current_process(policy, system_roots=())
    sandbox = EvaluationSandbox(policy, runtime_plan=plan, bwrap_path=fake_bwrap)

    def forbidden(*args, **kwargs):
        raise AssertionError("untrusted executable started a subprocess")

    monkeypatch.setattr("longrun_agent.tools.sandbox.subprocess.Popen", forbidden)
    context = ToolContext(workspace, workspace_policy=policy, subprocess_sandbox=sandbox)
    result = ToolRouter([BashTool()]).execute(
        ToolCall(id="bash-1", name="bash", arguments={"argv": [str(outside)]}),
        context,
    )

    assert result.error_type == ErrorType.WORKSPACE_ACCESS_DENIED
    assert result.output == ACCESS_DENIED_MESSAGE


def test_missing_bwrap_and_runtime_fail_closed_with_specific_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)

    with pytest.raises(EvaluationSandboxUnavailable, match="EVALUATION_SANDBOX_UNAVAILABLE"):
        EvaluationSandbox(policy, bwrap_path=tmp_path / "missing-bwrap")
    with pytest.raises(EvaluationSandboxRuntimeUnavailable, match=SANDBOX_RUNTIME_UNAVAILABLE):
        TrustedRuntimeMountPlan.from_current_process(policy, executable=tmp_path / "missing-python", system_roots=())


def test_preflight_maps_bwrap_execvp_failure_to_runtime_unavailable(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    sandbox = EvaluationSandbox(
        policy,
        runtime_plan=TrustedRuntimeMountPlan.from_current_process(policy, system_roots=()),
        bwrap_path=fake_bwrap,
    )

    class FailedProcess:
        returncode = 127

        def communicate(self, timeout):
            return "", "bwrap: execvp trusted-python: No such file or directory"

    monkeypatch.setattr(sandbox, "popen", lambda *args, **kwargs: FailedProcess())

    with pytest.raises(EvaluationSandboxRuntimeUnavailable, match=SANDBOX_RUNTIME_UNAVAILABLE):
        sandbox.preflight()


def test_bash_launcher_failure_is_not_reported_as_nonzero_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class LauncherFailureSandbox(SubprocessSandbox):
        def popen(self, argv, *, cwd, shell, **kwargs):
            class FailedProcess:
                returncode = 127

                def communicate(self, timeout):
                    return "", "bwrap: execvp /opt/longrun/runtime/venv/bin/python: No such file or directory"

            return FailedProcess()

        def classify_process_result(self, returncode, stdout, stderr):
            return SandboxProcessFailure(ErrorType.SANDBOX_RUNTIME_ERROR, SANDBOX_RUNTIME_UNAVAILABLE)

    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    context = ToolContext(workspace, workspace_policy=policy, subprocess_sandbox=LauncherFailureSandbox())

    result = ToolRouter([BashTool()]).execute(
        ToolCall(id="bash-1", name="bash", arguments={"argv": ["python", "-c", "print('no')"]}),
        context,
    )

    assert result.success is False
    assert result.error_type == ErrorType.SANDBOX_RUNTIME_ERROR
    assert result.output == SANDBOX_RUNTIME_UNAVAILABLE
    assert result.metadata["exit_code"] == 127


def test_preflight_phase1_failure_does_not_enter_phase2(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_bwrap = tmp_path / "bwrap"
    fake_bwrap.write_text("test", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(workspace, evaluation_isolation_enabled=True)
    sandbox = EvaluationSandbox(
        policy,
        runtime_plan=TrustedRuntimeMountPlan.from_current_process(policy, system_roots=()),
        bwrap_path=fake_bwrap,
    )

    class Phase1Missing:
        returncode = 0

        def communicate(self, timeout):
            return '{"ok": false, "missing_path_category": "resolved_executable"}\n', ""

    monkeypatch.setattr(sandbox, "_run_probe_command", lambda *args, **kwargs: Phase1Missing())
    monkeypatch.setattr(
        sandbox,
        "_phase2_probe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("phase2 should not run")),
    )

    with pytest.raises(EvaluationSandboxRuntimeUnavailable, match=SANDBOX_RUNTIME_UNAVAILABLE):
        sandbox.preflight()
    assert sandbox.last_private_diagnostics["missing_path_category"] == "resolved_executable"


def test_real_evaluation_sandbox_runs_python_and_pytest_with_isolation(tmp_path: Path) -> None:
    if os.name == "nt" or shutil.which("bwrap") is None:
        pytest.skip("native bwrap is unavailable")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_runtime.py").write_text("def test_runtime():\n    assert True\n", encoding="utf-8")
    outside = tmp_path / ".env"
    contract = tmp_path / "contracts" / "private.yaml"
    oracle = tmp_path / "trial" / "oracle" / "private" / "contract.json"
    contract.parent.mkdir()
    oracle.parent.mkdir(parents=True)
    for path in (outside, contract, oracle):
        path.write_text("PRIVATE", encoding="utf-8")
    policy = WorkspaceAccessPolicy.for_workspace(
        workspace,
        evaluation_isolation_enabled=True,
        denied_roots=[contract.parent, oracle.parent.parent],
    )
    sandbox = EvaluationSandbox(policy)
    doctor = sandbox.preflight()
    context = ToolContext(workspace, workspace_policy=policy, subprocess_sandbox=sandbox)
    router = ToolRouter([BashTool()])

    python_result = router.execute(
        ToolCall(id="python", name="bash", arguments={"argv": [sys.executable, "-c", "print('PYTHON_RUNTIME_GO')"]}),
        context,
    )
    hidden_code = f"from pathlib import Path; print([Path(item).exists() for item in {list(map(str, (outside, contract, oracle)))!r}])"
    hidden_result = router.execute(
        ToolCall(id="hidden", name="bash", arguments={"argv": [sys.executable, "-c", hidden_code]}),
        context,
    )
    module_pytest = router.execute(
        ToolCall(id="module", name="bash", arguments={"argv": [sys.executable, "-m", "pytest", "-q"]}),
        context,
    )
    pytest_executable = shutil.which("pytest")
    assert pytest_executable is not None
    console_pytest = router.execute(
        ToolCall(id="console", name="bash", arguments={"argv": [pytest_executable, "-q"]}),
        context,
    )
    readonly_code = (
        "from pathlib import Path; import sys; p=Path(sys.prefix)/'.sandbox-write'; "
        "\ntry: p.write_text('x'); print('WRITABLE')\nexcept OSError: print('READ_ONLY')"
    )
    readonly = router.execute(
        ToolCall(id="readonly", name="bash", arguments={"argv": [sys.executable, "-c", readonly_code]}),
        context,
    )

    assert python_result.success and "PYTHON_RUNTIME_GO" in python_result.output
    assert doctor["status"] == "GO"
    assert hidden_result.success and "[False, False, False]" in hidden_result.output
    assert module_pytest.success and module_pytest.metadata["exit_code"] == 0
    assert console_pytest.success and console_pytest.metadata["exit_code"] == 0
    assert readonly.success and "READ_ONLY" in readonly.output
