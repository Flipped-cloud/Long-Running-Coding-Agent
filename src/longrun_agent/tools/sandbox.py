from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from longrun_agent.protocol import ErrorType
from longrun_agent.tools.workspace_policy import WorkspaceAccessDenied, WorkspaceAccessPolicy, sensitive_environment

SANDBOX_UNAVAILABLE = "EVALUATION_SANDBOX_UNAVAILABLE"
SANDBOX_RUNTIME_UNAVAILABLE = "EVALUATION_SANDBOX_RUNTIME_UNAVAILABLE"
SANDBOX_RUNTIME_ROOT = PurePosixPath("/opt/longrun/runtime")
SANDBOX_VENV_ROOT = SANDBOX_RUNTIME_ROOT / "venv"
SANDBOX_BASE_ROOT = SANDBOX_RUNTIME_ROOT / "python-base"
SYSTEM_RUNTIME_ROOTS = (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/etc"))
ELF_LOADER_CANDIDATES = (
    PurePosixPath("/lib64/ld-linux-x86-64.so.2"),
    PurePosixPath("/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"),
)


class EvaluationSandboxUnavailable(RuntimeError):
    pass


class EvaluationSandboxRuntimeUnavailable(EvaluationSandboxUnavailable):
    pass


@dataclass(frozen=True)
class RuntimePathMapping:
    host_path: Path
    sandbox_path: PurePosixPath
    path_type: str
    purpose: str


@dataclass(frozen=True)
class RuntimeSymlinkHop:
    link_path: Path
    raw_target: str
    absolute_target: Path
    resolved_target: Path
    target_is_relative: bool
    target_exists: bool
    sandbox_link_path: PurePosixPath | None
    sandbox_target_path: PurePosixPath | None

    @property
    def path_type(self) -> str:
        return "relative_symlink" if self.target_is_relative else "absolute_symlink"


@dataclass(frozen=True)
class RuntimeAlias:
    host_alias_path: Path
    sandbox_target_path: PurePosixPath
    alias_kind: str
    source_hop_index: int | None = None


@dataclass(frozen=True)
class ResolvedSandboxExecutable:
    host_requested_path: Path
    host_resolved_path: Path
    sandbox_executable_path: PurePosixPath
    runtime_kind: str
    interpreter_chain: tuple[RuntimeSymlinkHop, ...]


@dataclass(frozen=True)
class SandboxMount:
    source: str | None
    target: PurePosixPath
    type: str
    read_only: bool
    purpose: str
    phase: str


@dataclass(frozen=True)
class SandboxMountPlan:
    entries: tuple[SandboxMount, ...]

    def validate(self) -> None:
        seen_bind_targets: set[PurePosixPath] = set()
        seen_alias_targets: set[PurePosixPath] = set()
        for entry in self.entries:
            if entry.target == PurePosixPath("/"):
                raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE)
            if entry.type in {"ro-bind", "bind"}:
                if entry.target in seen_bind_targets:
                    raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE)
                seen_bind_targets.add(entry.target)
            elif entry.type == "symlink":
                if entry.target in seen_alias_targets or entry.target in seen_bind_targets:
                    raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
                if entry.source is None or entry.source == str(entry.target):
                    raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
                seen_alias_targets.add(entry.target)
        if SANDBOX_VENV_ROOT in seen_bind_targets and SANDBOX_BASE_ROOT in seen_bind_targets:
            return
        raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)

    def bwrap_args(self) -> list[str]:
        args: list[str] = []
        for entry in self.entries:
            if entry.type == "dir":
                args.extend(["--dir", str(entry.target)])
            elif entry.type == "ro-bind":
                assert entry.source is not None
                args.extend(["--ro-bind", entry.source, str(entry.target)])
            elif entry.type == "bind":
                assert entry.source is not None
                args.extend(["--bind", entry.source, str(entry.target)])
            elif entry.type == "symlink":
                assert entry.source is not None
                args.extend(["--symlink", entry.source, str(entry.target)])
            elif entry.type == "proc":
                args.extend(["--proc", str(entry.target)])
            elif entry.type == "dev":
                args.extend(["--dev", str(entry.target)])
            elif entry.type == "tmpfs":
                args.extend(["--tmpfs", str(entry.target)])
            else:
                raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE)
        return args


@dataclass(frozen=True)
class SandboxProcessFailure:
    error_type: ErrorType
    public_message: str
    retryable: bool = False


@dataclass(frozen=True)
class TrustedRuntimeMountPlan:
    executable: Path
    resolved_executable: Path
    prefix: Path
    base_prefix: Path
    runtime_roots: tuple[Path, ...]
    system_roots: tuple[Path, ...]
    path_mappings: tuple[RuntimePathMapping, ...] = field(default_factory=tuple)
    symlink_chain: tuple[RuntimeSymlinkHop, ...] = field(default_factory=tuple)
    aliases: tuple[RuntimeAlias, ...] = field(default_factory=tuple)

    @classmethod
    def from_current_process(
        cls,
        policy: WorkspaceAccessPolicy,
        *,
        executable: Path | None = None,
        prefix: Path | None = None,
        base_prefix: Path | None = None,
        system_roots: tuple[Path, ...] | None = None,
    ) -> TrustedRuntimeMountPlan:
        executable_path = Path(executable or sys.executable).absolute()
        try:
            resolved_executable = executable_path.resolve(strict=True)
            prefix_path = Path(prefix or sys.prefix).resolve(strict=True)
            base_prefix_path = Path(base_prefix or sys.base_prefix).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE) from exc
        if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)

        fixed_roots = tuple(path.absolute() for path in (system_roots or SYSTEM_RUNTIME_ROOTS) if path.exists() and path.is_dir())
        runtime_roots = _minimal_roots((prefix_path, base_prefix_path))
        for root in runtime_roots:
            _validate_runtime_root(root, policy)
        if not _covered_by(executable_path, [*runtime_roots, *fixed_roots]):
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
        if not _covered_by(resolved_executable, [*runtime_roots, *fixed_roots]):
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)

        plan = cls(
            executable=executable_path,
            resolved_executable=resolved_executable,
            prefix=prefix_path,
            base_prefix=base_prefix_path,
            runtime_roots=runtime_roots,
            system_roots=fixed_roots,
        )
        symlink_chain = plan.resolve_symlink_chain(plan.executable)
        aliases = plan._runtime_aliases(symlink_chain)
        return cls(
            executable=plan.executable,
            resolved_executable=plan.resolved_executable,
            prefix=plan.prefix,
            base_prefix=plan.base_prefix,
            runtime_roots=plan.runtime_roots,
            system_roots=plan.system_roots,
            path_mappings=plan._default_mappings(),
            symlink_chain=symlink_chain,
            aliases=aliases,
        )

    @property
    def allowed_executable_roots(self) -> tuple[Path, ...]:
        return (*self.runtime_roots, *self.system_roots)

    @property
    def canonical_executable(self) -> PurePosixPath:
        mapped = self.map_host_to_sandbox(self.executable)
        if mapped is None:
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
        return mapped

    @property
    def canonical_resolved_executable(self) -> PurePosixPath:
        mapped = self.map_host_to_sandbox(self.resolved_executable)
        if mapped is None:
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
        return mapped

    def map_host_to_sandbox(self, path: Path, *, workspace_root: Path | None = None) -> PurePosixPath | None:
        resolved = Path(path).absolute()
        mapping_roots = (
            (self.prefix, SANDBOX_VENV_ROOT, "venv"),
            (self.base_prefix, SANDBOX_BASE_ROOT, "python-base"),
        )
        for host_root, sandbox_root, _kind in mapping_roots:
            if _is_within(resolved, host_root):
                return sandbox_root / resolved.relative_to(host_root).as_posix()
        if workspace_root is not None and _is_within(resolved, workspace_root):
            return PurePosixPath("/workspace") / resolved.relative_to(workspace_root).as_posix()
        for root in self.system_roots:
            if _is_within(resolved, root):
                return PurePosixPath(resolved.as_posix())
        return None

    def map_sandbox_to_host(self, path: str) -> Path | None:
        sandbox_path = PurePosixPath(path)
        for host_root, sandbox_root in ((self.prefix, SANDBOX_VENV_ROOT), (self.base_prefix, SANDBOX_BASE_ROOT)):
            try:
                relative = sandbox_path.relative_to(sandbox_root)
            except ValueError:
                continue
            return host_root / Path(*relative.parts)
        return None

    def _default_mappings(self) -> tuple[RuntimePathMapping, ...]:
        mappings = [
            RuntimePathMapping(self.prefix, SANDBOX_VENV_ROOT, "directory", "python virtual environment"),
            RuntimePathMapping(self.base_prefix, SANDBOX_BASE_ROOT, "directory", "python base runtime"),
            RuntimePathMapping(self.executable, self.canonical_executable, "executable", "requested python executable"),
            RuntimePathMapping(
                self.resolved_executable,
                self.canonical_resolved_executable,
                "executable",
                "resolved python executable",
            ),
        ]
        return tuple(dict.fromkeys(mappings))

    def resolve_symlink_chain(self, start_path: Path, *, max_depth: int = 32) -> tuple[RuntimeSymlinkHop, ...]:
        chain: list[RuntimeSymlinkHop] = []
        current = Path(start_path).absolute()
        seen: set[Path] = set()
        for _depth in range(max_depth):
            try:
                os.lstat(current)
            except OSError as exc:
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE) from exc
            if not current.is_symlink():
                return tuple(chain)
            if current in seen:
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
            seen.add(current)
            try:
                raw_target = os.readlink(current)
            except OSError as exc:
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE) from exc
            target = Path(raw_target)
            host_target = target if target.is_absolute() else current.parent / target
            absolute_target = host_target.absolute()
            target_exists = absolute_target.exists()
            try:
                resolved_target = absolute_target.resolve(strict=True)
            except (OSError, RuntimeError):
                resolved_target = absolute_target
            sandbox_link = self.map_host_to_sandbox(current)
            sandbox_target = self.map_host_to_sandbox(absolute_target)
            if sandbox_target is None and target_exists:
                sandbox_target = self.map_host_to_sandbox(resolved_target)
            chain.append(
                RuntimeSymlinkHop(
                    link_path=current,
                    raw_target=raw_target,
                    absolute_target=absolute_target,
                    resolved_target=resolved_target,
                    target_is_relative=not target.is_absolute(),
                    target_exists=target_exists,
                    sandbox_link_path=sandbox_link,
                    sandbox_target_path=sandbox_target,
                )
            )
            if not target_exists:
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
            if not _covered_by(resolved_target, self.allowed_executable_roots):
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
            current = absolute_target
        raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)

    def _runtime_aliases(self, symlink_chain: tuple[RuntimeSymlinkHop, ...]) -> tuple[RuntimeAlias, ...]:
        aliases = [RuntimeAlias(self.prefix, SANDBOX_VENV_ROOT, "prefix_alias")]
        if self.base_prefix != self.prefix:
            aliases.append(RuntimeAlias(self.base_prefix, SANDBOX_BASE_ROOT, "base_prefix_alias"))
        for index, hop in enumerate(symlink_chain):
            if hop.target_is_relative or self.map_host_to_sandbox(hop.absolute_target) is not None:
                continue
            sandbox_target = self.map_host_to_sandbox(hop.resolved_target)
            if sandbox_target is None:
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
            aliases.append(
                RuntimeAlias(
                    hop.absolute_target,
                    sandbox_target,
                    "intermediate_symlink_alias",
                    source_hop_index=index,
                )
            )
        deduped: dict[Path, RuntimeAlias] = {}
        for alias in aliases:
            existing = deduped.get(alias.host_alias_path)
            if existing is not None and existing.sandbox_target_path != alias.sandbox_target_path:
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
            deduped[alias.host_alias_path] = alias
        return tuple(deduped.values())


class SubprocessSandbox(ABC):
    def preflight(self) -> dict[str, Any]:
        return {"status": "GO", "sandbox": "local_guarded"}

    def classify_process_result(self, returncode: int, stdout: str, stderr: str) -> SandboxProcessFailure | None:
        return None

    @abstractmethod
    def popen(self, argv: list[str] | str, *, cwd: Path, shell: bool, **kwargs: Any) -> subprocess.Popen:
        raise NotImplementedError


class LocalGuardedSandbox(SubprocessSandbox):
    def __init__(self, policy: WorkspaceAccessPolicy):
        self.policy = policy

    def popen(self, argv: list[str] | str, *, cwd: Path, shell: bool, **kwargs: Any) -> subprocess.Popen:
        return subprocess.Popen(argv, cwd=cwd, shell=shell, env=sensitive_environment(), **kwargs)


class EvaluationSandbox(SubprocessSandbox):
    def __init__(
        self,
        policy: WorkspaceAccessPolicy,
        *,
        runtime_plan: TrustedRuntimeMountPlan | None = None,
        bwrap_path: Path | None = None,
    ):
        self.policy = policy
        discovered_bwrap = bwrap_path or (Path(value) if os.name != "nt" and (value := shutil.which("bwrap")) else None)
        if discovered_bwrap is None or not discovered_bwrap.exists():
            raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE)
        self.bwrap = discovered_bwrap.resolve()
        self.runtime_plan = runtime_plan or TrustedRuntimeMountPlan.from_current_process(policy)
        self.environment = _trusted_environment(self.runtime_plan, self.policy.workspace_root)
        self.mount_plan = self._build_mount_plan()
        self.mount_plan.validate()
        self.last_private_diagnostics: dict[str, Any] = {}

    def popen(self, argv: list[str] | str, *, cwd: Path, shell: bool, **kwargs: Any) -> subprocess.Popen:
        if shell or isinstance(argv, str):
            raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE)
        command = self.build_command(argv, cwd=cwd)
        return subprocess.Popen(command, cwd=cwd, shell=False, env=self.environment, **kwargs)

    def build_command(self, argv: list[str], *, cwd: Path) -> list[str]:
        cwd = cwd.resolve()
        relative_cwd = cwd.relative_to(self.policy.workspace_root)
        sandbox_argv = list(argv)
        executable = self.resolve_allowed_executable(sandbox_argv[0], cwd=cwd)
        sandbox_argv[0] = str(executable.sandbox_executable_path)
        command = [
            str(self.bwrap),
            "--die-with-parent",
            "--unshare-all",
            "--new-session",
            *self.mount_plan.bwrap_args(),
            "--chdir",
            (PurePosixPath("/workspace") / relative_cwd.as_posix()).as_posix(),
            "--",
            *sandbox_argv,
        ]
        self.last_private_diagnostics["last_bwrap_argv"] = command
        return command

    def resolve_allowed_executable(self, requested: str, *, cwd: Path) -> ResolvedSandboxExecutable:
        if not requested:
            raise WorkspaceAccessDenied("The requested path is outside the agent workspace.")
        host_requested = self._resolve_requested_host_executable(requested, cwd=cwd)
        try:
            host_absolute = host_requested.absolute()
            host_resolved = host_absolute.resolve(strict=True)
        except OSError as exc:
            raise WorkspaceAccessDenied("The requested path is outside the agent workspace.") from exc
        roots = [self.policy.workspace_root, *self.runtime_plan.allowed_executable_roots]
        if not _covered_by(host_absolute, roots) or not _covered_by(host_resolved, roots):
            raise WorkspaceAccessDenied("The requested path is outside the agent workspace.")
        if not host_resolved.is_file() or not os.access(host_resolved, os.X_OK):
            raise WorkspaceAccessDenied("The requested path is outside the agent workspace.")
        sandbox_executable = self.runtime_plan.map_host_to_sandbox(host_absolute, workspace_root=self.policy.workspace_root)
        if sandbox_executable is None:
            raise WorkspaceAccessDenied("The requested path is outside the agent workspace.")
        runtime_kind = _runtime_kind(sandbox_executable)
        return ResolvedSandboxExecutable(
            host_requested_path=host_absolute,
            host_resolved_path=host_resolved,
            sandbox_executable_path=sandbox_executable,
            runtime_kind=runtime_kind,
            interpreter_chain=self.runtime_plan.symlink_chain,
        )

    def classify_process_result(self, returncode: int, stdout: str, stderr: str) -> SandboxProcessFailure | None:
        if returncode == 0:
            return None
        lowered = stderr.casefold()
        if _is_bwrap_platform_error(stderr):
            return SandboxProcessFailure(ErrorType.EVALUATION_SANDBOX_UNAVAILABLE, SANDBOX_UNAVAILABLE)
        runtime_markers = (
            "bwrap: execvp",
            "can't bind mount",
            "can't make parent dir",
            "no such file or directory",
            "runtime alias missing",
            "loader missing",
        )
        if any(marker in lowered for marker in runtime_markers):
            return SandboxProcessFailure(ErrorType.SANDBOX_RUNTIME_ERROR, SANDBOX_RUNTIME_UNAVAILABLE)
        return None

    def preflight(self) -> dict[str, Any]:
        workspace_probe = self.policy.workspace_root / ".longrun" / f"sandbox-probe-{uuid.uuid4().hex}.txt"
        workspace_probe.parent.mkdir(parents=True, exist_ok=True)
        workspace_probe.write_text("READ", encoding="utf-8")
        runtime_probe = SANDBOX_VENV_ROOT / f".sandbox-runtime-write-{uuid.uuid4().hex}"
        outside_handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        outside_path = Path(outside_handle.name)
        outside_handle.write("PRIVATE")
        outside_handle.close()
        denied_targets = [str(path) for path in self.policy.denied_roots if path.exists()]
        try:
            phase1 = self._phase1_probe()
            self.last_private_diagnostics["phase1"] = phase1
            if not phase1.get("ok"):
                self.last_private_diagnostics["missing_path_category"] = phase1.get("missing_path_category")
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
            phase2 = self._phase2_probe(workspace_probe, outside_path, runtime_probe, denied_targets)
            self.last_private_diagnostics["phase2"] = phase2
            if (
                not phase2.get("python_runtime_available")
                or not phase2.get("pytest_importable")
                or not phase2.get("workspace_ok")
                or workspace_probe.read_text(encoding="utf-8") != "WRITE"
                or phase2.get("outside_visible")
                or phase2.get("private_visible")
                or phase2.get("runtime_writable")
                or phase2.get("prefix") != str(SANDBOX_VENV_ROOT)
            ):
                raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
            return {
                "status": "GO",
                "bwrap_available": True,
                "user_namespace_available": True,
                "python_runtime_available": True,
                "workspace_read_write": True,
                "outside_workspace_hidden": True,
                "private_roots_hidden": True,
                "trusted_runtime_read_only": True,
                "canonical_runtime_root": str(SANDBOX_RUNTIME_ROOT),
                "canonical_venv_root": str(SANDBOX_VENV_ROOT),
                "canonical_base_root": str(SANDBOX_BASE_ROOT),
                "runtime_executable": str(self.runtime_plan.canonical_executable),
                "runtime_resolved_executable": str(self.runtime_plan.canonical_resolved_executable),
            }
        except subprocess.TimeoutExpired as exc:
            self.last_private_diagnostics["failure_stage"] = "timeout"
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE) from exc
        finally:
            workspace_probe.unlink(missing_ok=True)
            outside_path.unlink(missing_ok=True)

    def _build_mount_plan(self) -> SandboxMountPlan:
        mounts: list[SandboxMount] = []
        mount_targets = [
            SANDBOX_VENV_ROOT,
            SANDBOX_BASE_ROOT,
            PurePosixPath("/workspace"),
            *[PurePosixPath(root.as_posix()) for root in self.runtime_plan.system_roots],
            PurePosixPath("/proc"),
            PurePosixPath("/dev"),
            PurePosixPath("/tmp"),
            *[PurePosixPath(alias.host_alias_path.as_posix()) for alias in self.runtime_plan.aliases],
        ]
        for parent in _mount_parent_directories(mount_targets):
            mounts.append(SandboxMount(None, parent, "dir", False, "mount parent skeleton", "setup"))
        for root in self.runtime_plan.system_roots:
            mounts.append(SandboxMount(str(root), PurePosixPath(root.as_posix()), "ro-bind", True, "system runtime", "setup"))
        mounts.append(SandboxMount(str(self.runtime_plan.prefix), SANDBOX_VENV_ROOT, "ro-bind", True, "canonical venv", "setup"))
        mounts.append(
            SandboxMount(str(self.runtime_plan.base_prefix), SANDBOX_BASE_ROOT, "ro-bind", True, "canonical python base", "setup")
        )
        for alias in self.runtime_plan.aliases:
            mounts.append(
                SandboxMount(
                    str(alias.sandbox_target_path),
                    PurePosixPath(alias.host_alias_path.as_posix()),
                    "symlink",
                    True,
                    alias.alias_kind,
                    "setup",
                )
            )
        mounts.extend(
            [
                SandboxMount(None, PurePosixPath("/proc"), "proc", False, "proc", "setup"),
                SandboxMount(None, PurePosixPath("/dev"), "dev", False, "dev", "setup"),
                SandboxMount(None, PurePosixPath("/tmp"), "tmpfs", False, "temporary filesystem", "setup"),
                SandboxMount(str(self.policy.workspace_root), PurePosixPath("/workspace"), "bind", False, "trial workspace", "setup"),
            ]
        )
        return SandboxMountPlan(tuple(mounts))

    def _resolve_requested_host_executable(self, requested: str, *, cwd: Path) -> Path:
        raw = Path(requested)
        mapped_canonical = self.runtime_plan.map_sandbox_to_host(requested)
        if mapped_canonical is not None:
            return mapped_canonical
        if raw.is_absolute():
            return raw
        if "/" in requested or "\\" in requested:
            return cwd / raw
        resolved = _which_host_executable(requested, self.runtime_plan, self.policy.workspace_root)
        if resolved is None:
            raise WorkspaceAccessDenied("The requested path is outside the agent workspace.")
        return resolved

    def _phase1_probe(self) -> dict[str, Any]:
        hop_checks = []
        for index, hop in enumerate(self.runtime_plan.symlink_chain):
            if hop.target_is_relative:
                check_path = hop.sandbox_target_path
            else:
                check_path = PurePosixPath(hop.absolute_target.as_posix())
            if check_path is not None:
                hop_checks.append((index, check_path))
        intermediate_aliases = [alias for alias in self.runtime_plan.aliases if alias.alias_kind == "intermediate_symlink_alias"]
        hop_script = ""
        for index, path in hop_checks:
            hop_script += (
                f"hop_{index}_target_exists=false\n"
                f"if [ -e {_shell_quote(str(path))} ]; then hop_{index}_target_exists=true; "
                f'elif [ "$failed_hop_index" = null ]; then failed_hop_index={index}; fi\n'
            )
        alias_script = ""
        for alias in intermediate_aliases:
            alias_script += (
                f"if ! [ -L {_shell_quote(alias.host_alias_path.as_posix())} ] "
                f"|| ! readlink -e {_shell_quote(alias.host_alias_path.as_posix())} >/dev/null 2>&1; "
                "then intermediate_aliases_resolve=false; fi\n"
            )
        script = (
            "set -eu\n"
            f"venv={str(SANDBOX_VENV_ROOT)!r}\n"
            f"base={str(SANDBOX_BASE_ROOT)!r}\n"
            f"alias_exe={_shell_quote(self.runtime_plan.executable.as_posix())}\n"
            f"canonical_python={_shell_quote(str(self.runtime_plan.canonical_executable))}\n"
            f"loaders={json.dumps([str(item) for item in ELF_LOADER_CANDIDATES])!r}\n"
            "canonical_venv_exists=false\n"
            '[ -d "$venv" ] && canonical_venv_exists=true\n'
            "canonical_base_exists=false\n"
            '[ -d "$base" ] && canonical_base_exists=true\n'
            "original_alias_is_symlink=false\n"
            '[ -L "$alias_exe" ] && original_alias_is_symlink=true\n'
            "original_alias_target_resolves=false\n"
            'readlink -e "$alias_exe" >/dev/null 2>&1 && original_alias_target_resolves=true\n'
            "canonical_python_exists=false\n"
            '[ -e "$canonical_python" ] && canonical_python_exists=true\n'
            "canonical_python_executable=false\n"
            '[ -x "$canonical_python" ] && canonical_python_executable=true\n'
            "intermediate_aliases_resolve=true\n"
            "failed_hop_index=null\n"
            f"{hop_script}"
            f"{alias_script}"
            "loader_ok=false\n"
            'for loader in $(printf \'%s\' "$loaders" | tr -d \'[]",\' ); do [ -e "$loader" ] && loader_ok=true; done\n'
            "missing=''\n"
            "[ \"$canonical_venv_exists\" = true ] || missing='canonical_venv_root'\n"
            '[ -n "$missing" ] || [ "$canonical_base_exists" = true ] || missing=\'canonical_base_root\'\n'
            '[ -n "$missing" ] || [ "$original_alias_is_symlink" = true ] || missing=\'original_executable_alias\'\n'
            'if [ -z "$missing" ] && [ "$original_alias_target_resolves" != true ]; then '
            "if [ \"$failed_hop_index\" != null ]; then missing='intermediate_absolute_symlink_target'; "
            "else missing='original_executable_alias'; fi; fi\n"
            '[ -n "$missing" ] || [ "$canonical_python_exists" = true ] || missing=\'canonical_python\'\n'
            '[ -n "$missing" ] || [ "$canonical_python_executable" = true ] || missing=\'executable_bit\'\n'
            '[ -n "$missing" ] || [ "$intermediate_aliases_resolve" = true ] || missing=\'intermediate_absolute_symlink_target\'\n'
            '[ -n "$missing" ] || [ "$loader_ok" = true ] || missing=\'elf_loader\'\n'
            "ok=false\n"
            '[ -z "$missing" ] && ok=true\n'
            'printf \'{"ok":%s,"checks":{"original_alias_is_symlink":%s,'
            '"original_alias_target_resolves":%s,"canonical_venv_exists":%s,'
            '"canonical_base_exists":%s,"canonical_python_exists":%s,'
            '"canonical_python_executable":%s,"intermediate_aliases_resolve":%s,'
            '"elf_loader_exists":%s},"missing_path_category":"%s",'
            '"failed_hop_index":%s,"alias_readlink":"%s"}\\n\' '
            '"$ok" "$original_alias_is_symlink" "$original_alias_target_resolves" '
            '"$canonical_venv_exists" "$canonical_base_exists" "$canonical_python_exists" '
            '"$canonical_python_executable" "$intermediate_aliases_resolve" "$loader_ok" '
            '"$missing" "$failed_hop_index" "$(readlink "$alias_exe" 2>/dev/null || true)"\n'
        )
        process = self._run_probe_command(["/bin/sh", "-c", script], cwd=self.policy.workspace_root)
        stdout, stderr = process.communicate(timeout=30)
        self.last_private_diagnostics.update({"phase1_returncode": process.returncode, "phase1_stdout": stdout, "phase1_stderr": stderr})
        if process.returncode != 0:
            failure = self.classify_process_result(int(process.returncode or 0), stdout or "", stderr or "")
            if failure and failure.error_type == ErrorType.EVALUATION_SANDBOX_UNAVAILABLE:
                raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE)
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
        try:
            return json.loads((stdout or "").strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE) from exc

    def _phase2_probe(
        self,
        workspace_probe: Path,
        outside_path: Path,
        runtime_probe: PurePosixPath,
        denied_targets: list[str],
    ) -> dict[str, Any]:
        code = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "import pytest\n"
            f"workspace = Path({workspace_probe.name!r})\n"
            f"outside = Path({str(outside_path)!r})\n"
            f"runtime = Path({str(runtime_probe)!r})\n"
            f"denied = {denied_targets!r}\n"
            "runtime_writable = True\n"
            "try:\n"
            "    runtime.write_text('MUTATION', encoding='utf-8')\n"
            "except OSError:\n"
            "    runtime_writable = False\n"
            "workspace_ok = workspace.read_text(encoding='utf-8') == 'READ'\n"
            "workspace.write_text('WRITE', encoding='utf-8')\n"
            "print('PYTHON_RUNTIME_GO')\n"
            "print(json.dumps({'python_runtime_available': True, 'pytest_importable': True, 'prefix': sys.prefix, "
            "'base_prefix': sys.base_prefix, 'workspace_ok': workspace_ok, 'outside_visible': outside.exists(), "
            "'private_visible': any(Path(item).exists() for item in denied), 'runtime_writable': runtime_writable}))\n"
        )
        process = self._run_probe_command([str(self.runtime_plan.canonical_executable), "-c", code], cwd=workspace_probe.parent)
        stdout, stderr = process.communicate(timeout=30)
        self.last_private_diagnostics.update({"phase2_returncode": process.returncode, "phase2_stdout": stdout, "phase2_stderr": stderr})
        if process.returncode != 0:
            failure = self.classify_process_result(int(process.returncode or 0), stdout or "", stderr or "")
            if failure and failure.error_type == ErrorType.EVALUATION_SANDBOX_UNAVAILABLE:
                raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE)
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
        try:
            return json.loads((stdout or "").strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE) from exc

    def _run_probe_command(self, argv: list[str], *, cwd: Path) -> subprocess.Popen:
        try:
            return self.popen(
                argv,
                cwd=cwd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise EvaluationSandboxUnavailable(SANDBOX_UNAVAILABLE) from exc


def build_subprocess_sandbox(policy: WorkspaceAccessPolicy) -> SubprocessSandbox:
    if policy.evaluation_isolation_enabled:
        return EvaluationSandbox(policy)
    return LocalGuardedSandbox(policy)


def _minimal_roots(paths) -> tuple[Path, ...]:
    roots: list[Path] = []
    for candidate in sorted({Path(path).resolve() for path in paths}, key=lambda path: (len(path.parts), str(path))):
        if not _covered_by(candidate, roots):
            roots.append(candidate)
    return tuple(roots)


def _validate_runtime_root(root: Path, policy: WorkspaceAccessPolicy) -> None:
    if not root.exists() or not root.is_dir() or root == Path(root.anchor):
        raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
    if root == policy.workspace_root or _is_within(policy.workspace_root, root):
        raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)
    if any(root == denied or _is_within(denied, root) or _is_within(root, denied) for denied in policy.denied_roots):
        raise EvaluationSandboxRuntimeUnavailable(SANDBOX_RUNTIME_UNAVAILABLE)


def _covered_by(path: Path, roots) -> bool:
    return any(_is_within(path, root) for root in roots)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _runtime_kind(path: PurePosixPath) -> str:
    if _posix_within(path, SANDBOX_VENV_ROOT):
        return "venv"
    if _posix_within(path, SANDBOX_BASE_ROOT):
        return "python-base"
    if _posix_within(path, PurePosixPath("/workspace")):
        return "workspace"
    return "system"


def _posix_within(path: PurePosixPath, root: PurePosixPath) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mount_parent_directories(targets: list[PurePosixPath]) -> tuple[PurePosixPath, ...]:
    parents: set[PurePosixPath] = set()
    destinations = set(targets)
    for target in targets:
        for parent in target.parents:
            if parent == PurePosixPath("/") or parent in destinations:
                continue
            parents.add(parent)
    return tuple(sorted(parents, key=lambda path: (len(path.parts), str(path))))


def _trusted_environment(plan: TrustedRuntimeMountPlan, workspace_root: Path) -> dict[str, str]:
    environment = sensitive_environment()
    _ = (plan, workspace_root)
    environment["PATH"] = ":".join(
        [
            str(SANDBOX_VENV_ROOT / "bin"),
            str(SANDBOX_BASE_ROOT / "bin"),
            "/usr/bin",
            "/bin",
        ]
    )
    environment["VIRTUAL_ENV"] = str(SANDBOX_VENV_ROOT)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _which_host_executable(requested: str, plan: TrustedRuntimeMountPlan, workspace_root: Path) -> Path | None:
    path_dirs = [
        plan.prefix / "bin",
        plan.base_prefix / "bin",
        workspace_root,
    ]
    for value in sensitive_environment().get("PATH", "").split(os.pathsep):
        if value:
            path_dirs.append(Path(value).absolute())
    for directory in dict.fromkeys(path_dirs):
        mapped = plan.map_host_to_sandbox(directory, workspace_root=workspace_root)
        if mapped is None:
            continue
        candidate = directory / requested
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _is_bwrap_platform_error(stderr: str) -> bool:
    lowered = stderr.casefold()
    return any(
        marker in lowered
        for marker in (
            "creating new namespace failed",
            "operation not permitted",
            "user namespaces are not enabled",
            "permission denied",
            "namespace setup",
        )
    )
