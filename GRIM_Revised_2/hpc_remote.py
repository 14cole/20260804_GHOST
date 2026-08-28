"""Credential-safe remote HPC operations for the GRIM Runs tab.

The module intentionally has no Qt or third-party dependencies.  It launches
the system OpenSSH clients (``ssh``/``scp``) or PuTTY clients
(``plink``/``pscp``) with argument vectors and never invokes a local shell.
Authentication is delegated to an SSH agent, an OpenSSH key/config entry, or
a PuTTY saved session; passwords are neither accepted nor stored here.

Remote commands still pass through the login host's POSIX shell, as required
by SSH.  Every command token is therefore quoted with :func:`shlex.quote`, and
paths and Slurm identifiers receive additional validation before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Protocol, Sequence


DEFAULT_CONNECT_TIMEOUT = 15.0
DEFAULT_COMMAND_TIMEOUT = 60.0
DEFAULT_TRANSFER_TIMEOUT = 1800.0
MAX_TIMEOUT = 86400.0

_PUTTY_BATCH_PROMPT_MARKER = "cannot answer interactive prompts in batch mode"
_PUTTY_BATCH_PROMPT_GUIDANCE = (
    "PuTTY needs an interactive response. Run the saved PuTTY session "
    "interactively to identify and resolve the prompt. Save its auto-login "
    "username and cache only a verified host key; use an approved key loaded "
    "in Pageant or keep an authenticated PuTTY connection open with connection "
    "sharing enabled. If site policy requires a prompt for every connection, "
    "use Runs > Export Bundle and transfer/submit it manually."
)


class Transport(str, Enum):
    """Supported Windows-side SSH client families."""

    OPENSSH = "openssh"
    PUTTY = "putty"


class RemoteError(RuntimeError):
    """Base class for remote transport failures."""


class ConfigurationError(RemoteError, ValueError):
    """Raised before execution when connection or operation input is unsafe."""


class CommandTimeoutError(RemoteError, TimeoutError):
    """Raised when a local SSH client exceeds its bounded timeout."""

    def __init__(self, operation: str, timeout: float, argv: Sequence[str]) -> None:
        super().__init__(f"{operation} timed out after {timeout:g} seconds")
        self.operation = operation
        self.timeout = timeout
        self.argv = tuple(argv)


class CommandFailedError(RemoteError):
    """Raised when an SSH, transfer, or remote command returns non-zero."""

    def __init__(
        self,
        operation: str,
        result: "CommandResult",
        *,
        guidance: str | None = None,
    ) -> None:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        if len(detail) > 2000:
            detail = detail[:1997] + "..."
        message = f"{operation} failed with exit code {result.returncode}: {detail}"
        if guidance:
            message += f"\n\n{guidance}"
        super().__init__(message)
        self.operation = operation
        self.result = result


class ProtocolError(RemoteError):
    """Raised when successful remote output does not follow its stated format."""


class HpcBundleFailedError(RemoteError):
    """Raised for a structured, unsuccessful ``hpc_bundle`` response."""

    def __init__(self, bundle_result: "HpcBundleResult") -> None:
        payload = bundle_result.payload
        detail: Any = payload.get("error") or payload.get("message")
        if isinstance(detail, Mapping):
            detail = detail.get("message") or detail.get("detail") or detail
        detail_text = str(detail or "remote hpc_bundle reported failure")
        if len(detail_text) > 2000:
            detail_text = detail_text[:1997] + "..."
        super().__init__(
            f"remote hpc_bundle failed with exit code "
            f"{bundle_result.result.returncode}: {detail_text}"
        )
        self.bundle_result = bundle_result
        self.result = bundle_result.result


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one local SSH-client process."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass(frozen=True)
class ConnectionInfo:
    """Verified remote endpoint information."""

    transport: Transport
    hostname: str
    username: str
    result: CommandResult


@dataclass(frozen=True)
class TransferResult:
    """Completed upload or download."""

    direction: str
    local_path: Path
    remote_path: str
    result: CommandResult


@dataclass(frozen=True)
class HpcBundleResult:
    """Machine-readable result returned by the remote ``hpc_bundle`` CLI."""

    payload: Mapping[str, Any]
    run_id: str | None
    job_ids: tuple[str, ...]
    result: CommandResult
    schema: str | None = None
    ok: bool = True
    bundle_id: str | None = None


@dataclass(frozen=True)
class JobStatus:
    """Normalized status row from Slurm's ``squeue`` or ``sacct``."""

    job_id: str
    state: str
    job_name: str
    elapsed: str
    detail: str
    source: str


class CompletedProcessLike(Protocol):
    """Small subset of :class:`subprocess.CompletedProcess` used by the client."""

    returncode: int
    stdout: str | bytes | None
    stderr: str | bytes | None


class ProcessRunner(Protocol):
    """Injectable process boundary used by tests and application integrations."""

    def __call__(
        self, argv: Sequence[str], *, timeout: float
    ) -> CompletedProcessLike: ...


def _default_process_runner(
    argv: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        shell=False,
        creationflags=creation_flags,
    )


_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_USER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_JOB_ID_RE = re.compile(r"^[0-9][0-9A-Za-z_.+\-\[\]%]*$")
_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._+@%=/, \-]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)


def _normalize_slurm_state(value: str) -> str:
    """Return the stable Slurm state name used by the GUI."""

    state = str(value).strip().upper()
    if not state:
        return "UNKNOWN"
    # sacct may append '+' for a truncated value and may append a qualifier,
    # such as "CANCELLED by 1234".  A wider query column avoids ordinary
    # truncation; this also keeps older cluster output predictable.
    state = state.split(None, 1)[0].rstrip("+")
    return {"OUT_OF_ME": "OUT_OF_MEMORY"}.get(state, state)


def _require_clean_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{label} must be a non-empty string")
    if _CONTROL_RE.search(value):
        raise ConfigurationError(f"{label} may not contain control characters")
    return value


def _validate_timeout(value: float, label: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be a number") from exc
    if not math.isfinite(timeout) or timeout <= 0 or timeout > MAX_TIMEOUT:
        raise ConfigurationError(
            f"{label} must be greater than zero and at most {MAX_TIMEOUT:g} seconds"
        )
    return timeout


def _validate_remote_path(value: str, label: str = "remote path") -> str:
    value = _require_clean_text(value, label)
    if not _REMOTE_PATH_RE.fullmatch(value):
        raise ConfigurationError(
            f"{label} must be an absolute POSIX path without shell metacharacters"
        )
    raw_parts = value.split("/")
    if ".." in raw_parts:
        raise ConfigurationError(f"{label} may not contain '..' components")
    path = PurePosixPath(value)
    if not path.is_absolute() or not path.name:
        raise ConfigurationError(f"{label} must identify an absolute file or directory")
    return path.as_posix()


def _join_remote_path(directory: str, name: str) -> str:
    directory = _validate_remote_path(directory, "remote directory")
    name = _require_clean_text(name, "bundle filename")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ConfigurationError("bundle filename is not a safe path component")
    return _validate_remote_path(
        (PurePosixPath(directory) / name).as_posix(), "remote bundle path"
    )


def _validate_job_ids(job_ids: Sequence[str | int]) -> tuple[str, ...]:
    if isinstance(job_ids, (str, bytes)):
        job_ids = [job_ids.decode() if isinstance(job_ids, bytes) else job_ids]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in job_ids:
        job_id = str(value)
        if len(job_id) > 128 or not _JOB_ID_RE.fullmatch(job_id):
            raise ConfigurationError(f"invalid Slurm job ID: {job_id!r}")
        if job_id not in seen:
            normalized.append(job_id)
            seen.add(job_id)
    if not normalized:
        raise ConfigurationError("at least one Slurm job ID is required")
    return tuple(normalized)


def _job_row_belongs_to(requested_id: str, returned_id: str) -> bool:
    """Match a Slurm array task/step row to its submitted parent job ID."""

    return returned_id == requested_id or any(
        returned_id.startswith(requested_id + separator)
        for separator in ("_", ".", "+")
    )


def _aggregate_job_rows(job_id: str, rows: Sequence["JobStatus"]) -> "JobStatus":
    """Collapse Slurm array-task and step rows into one authoritative job row."""

    if len(rows) == 1 and rows[0].job_id == job_id:
        return rows[0]
    states = [_normalize_slurm_state(row.state) for row in rows]
    active = [state for state in states if state not in _TERMINAL_SLURM_STATES]
    if active:
        if "RUNNING" in active:
            state = "RUNNING"
        elif "PENDING" in active:
            state = "PENDING"
        else:
            state = active[0]
    elif states and all(value == "COMPLETED" for value in states):
        state = "COMPLETED"
    else:
        state = next(
            (value for value in states if value != "COMPLETED"),
            "UNKNOWN",
        )
    counts: dict[str, int] = {}
    for value in states:
        counts[value] = counts.get(value, 0) + 1
    summary = ", ".join(
        f"{count} {value}" for value, count in sorted(counts.items())
    )
    exact = next((row for row in rows if row.job_id == job_id), rows[0])
    return JobStatus(
        job_id=job_id,
        state=state,
        job_name=exact.job_name,
        elapsed=exact.elapsed,
        detail=f"Aggregated {len(rows)} Slurm row(s): {summary}",
        source=exact.source,
    )


@dataclass(frozen=True)
class ConnectionConfig:
    """Validated SSH client configuration with no password-bearing fields.

    Prefer one of the class methods instead of constructing this class
    directly.  OpenSSH direct mode accepts ``host`` (and optionally user, port,
    and key); alias mode accepts only ``ssh_config_alias``.  PuTTY mode accepts
    only a saved ``putty_session`` and optional pinned ``host_key`` fingerprint.
    """

    transport: Transport | str = Transport.OPENSSH
    host: str | None = None
    port: int | None = None
    username: str | None = None
    identity_file: Path | str | None = None
    ssh_config_alias: str | None = None
    ssh_config_file: Path | str | None = None
    known_hosts_file: Path | str | None = None
    putty_session: str | None = None
    host_key: str | None = None
    ssh_executable: str = "ssh"
    scp_executable: str = "scp"
    plink_executable: str = "plink"
    pscp_executable: str = "pscp"
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT
    transfer_timeout: float = DEFAULT_TRANSFER_TIMEOUT

    def __post_init__(self) -> None:
        try:
            transport = Transport(self.transport)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"unsupported transport: {self.transport!r}"
            ) from exc
        object.__setattr__(self, "transport", transport)

        for field_name in (
            "ssh_executable",
            "scp_executable",
            "plink_executable",
            "pscp_executable",
        ):
            _require_clean_text(getattr(self, field_name), field_name)
        for field_name in ("connect_timeout", "command_timeout", "transfer_timeout"):
            object.__setattr__(
                self,
                field_name,
                _validate_timeout(getattr(self, field_name), field_name),
            )

        for field_name in ("identity_file", "ssh_config_file", "known_hosts_file"):
            value = getattr(self, field_name)
            if value is not None:
                path = Path(value).expanduser()
                _require_clean_text(str(path), field_name)
                object.__setattr__(self, field_name, path)

        if transport is Transport.OPENSSH:
            self._validate_openssh()
        else:
            self._validate_putty()

    def _validate_openssh(self) -> None:
        direct = self.host is not None
        alias = self.ssh_config_alias is not None
        if direct == alias:
            raise ConfigurationError(
                "OpenSSH requires exactly one of host or ssh_config_alias"
            )
        if self.putty_session is not None or self.host_key is not None:
            raise ConfigurationError(
                "putty_session and host_key are only valid for PuTTY mode"
            )
        if direct:
            assert self.host is not None
            if not _ENDPOINT_RE.fullmatch(self.host) or self.host.startswith("-"):
                raise ConfigurationError("host is not a valid SSH endpoint")
            if self.ssh_config_file is not None:
                raise ConfigurationError(
                    "ssh_config_file is only valid with ssh_config_alias"
                )
            port = 22 if self.port is None else self.port
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise ConfigurationError("port must be an integer from 1 through 65535")
            object.__setattr__(self, "port", port)
            if self.username is not None and not _USER_RE.fullmatch(self.username):
                raise ConfigurationError("username contains unsupported characters")
        else:
            assert self.ssh_config_alias is not None
            if not _ALIAS_RE.fullmatch(self.ssh_config_alias):
                raise ConfigurationError("ssh_config_alias contains unsupported characters")
            if self.host is not None or self.port is not None:
                raise ConfigurationError("host and port cannot override an SSH config alias")
            if self.username is not None or self.identity_file is not None:
                raise ConfigurationError(
                    "username and identity_file must be defined by the SSH config alias"
                )

    def _validate_putty(self) -> None:
        if not self.putty_session:
            raise ConfigurationError("PuTTY mode requires a saved putty_session")
        session = _require_clean_text(self.putty_session, "putty_session")
        if session.startswith("-") or ":" in session:
            raise ConfigurationError(
                "putty_session may not start with '-' or contain ':'"
            )
        forbidden = {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "identity_file": self.identity_file,
            "ssh_config_alias": self.ssh_config_alias,
            "ssh_config_file": self.ssh_config_file,
            "known_hosts_file": self.known_hosts_file,
        }
        present = [name for name, value in forbidden.items() if value is not None]
        if present:
            raise ConfigurationError(
                "PuTTY saved-session mode cannot also set " + ", ".join(present)
            )
        if self.host_key is not None:
            host_key = _require_clean_text(self.host_key, "host_key")
            if host_key.startswith("-"):
                raise ConfigurationError("host_key may not start with '-'")

    @classmethod
    def openssh_host(
        cls,
        host: str,
        *,
        username: str | None = None,
        port: int = 22,
        identity_file: Path | str | None = None,
        known_hosts_file: Path | str | None = None,
        ssh_executable: str = "ssh",
        scp_executable: str = "scp",
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        transfer_timeout: float = DEFAULT_TRANSFER_TIMEOUT,
    ) -> "ConnectionConfig":
        return cls(
            transport=Transport.OPENSSH,
            host=host,
            port=port,
            username=username,
            identity_file=identity_file,
            known_hosts_file=known_hosts_file,
            ssh_executable=ssh_executable,
            scp_executable=scp_executable,
            connect_timeout=connect_timeout,
            command_timeout=command_timeout,
            transfer_timeout=transfer_timeout,
        )

    @classmethod
    def openssh_alias(
        cls,
        alias: str,
        *,
        ssh_config_file: Path | str | None = None,
        known_hosts_file: Path | str | None = None,
        ssh_executable: str = "ssh",
        scp_executable: str = "scp",
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        transfer_timeout: float = DEFAULT_TRANSFER_TIMEOUT,
    ) -> "ConnectionConfig":
        return cls(
            transport=Transport.OPENSSH,
            ssh_config_alias=alias,
            ssh_config_file=ssh_config_file,
            known_hosts_file=known_hosts_file,
            ssh_executable=ssh_executable,
            scp_executable=scp_executable,
            connect_timeout=connect_timeout,
            command_timeout=command_timeout,
            transfer_timeout=transfer_timeout,
        )

    @classmethod
    def putty_session_config(
        cls,
        session: str,
        *,
        host_key: str | None = None,
        plink_executable: str = "plink",
        pscp_executable: str = "pscp",
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        transfer_timeout: float = DEFAULT_TRANSFER_TIMEOUT,
    ) -> "ConnectionConfig":
        return cls(
            transport=Transport.PUTTY,
            putty_session=session,
            host_key=host_key,
            plink_executable=plink_executable,
            pscp_executable=pscp_executable,
            connect_timeout=connect_timeout,
            command_timeout=command_timeout,
            transfer_timeout=transfer_timeout,
        )

    # More natural spelling for callers; kept as an alias so the dataclass
    # field can retain the descriptive ``putty_session`` name.
    putty_saved_session = putty_session_config


class HpcRemoteClient:
    """Synchronous, UI-agnostic SSH/SCP primitives for a headless Slurm host.

    GUI callers should invoke these methods in their existing worker/thread
    mechanism.  Submission only waits for the remote ``hpc_bundle`` process to
    return its run and job identifiers.  The scheduler owns submitted jobs, so
    they continue after SSH disconnects or GRIM exits.
    """

    def __init__(
        self,
        config: ConnectionConfig,
        runner: ProcessRunner | None = None,
    ) -> None:
        if not isinstance(config, ConnectionConfig):
            raise ConfigurationError("config must be a ConnectionConfig")
        self.config = config
        self._runner: ProcessRunner = runner or _default_process_runner

    def run_remote(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None = None,
        operation: str = "remote command",
    ) -> CommandResult:
        """Run tokenized arguments on the login host and require success."""

        return self._run_remote(
            arguments, timeout=timeout, operation=operation, check=True
        )

    def _run_remote(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None,
        operation: str,
        check: bool,
    ) -> CommandResult:
        tokens = self._validate_command_tokens(arguments)
        command = " ".join(shlex.quote(token) for token in tokens)
        argv = self._ssh_argv(command)
        return self._execute(
            argv,
            timeout=self._operation_timeout(timeout, self.config.command_timeout),
            operation=operation,
            check=check,
        )

    def test_connection(self, *, timeout: float | None = None) -> ConnectionInfo:
        """Verify non-interactive login and return hostname/remote username."""

        probe = "printf 'GRIM_HPC_OK\\n'; hostname; id -un"
        result = self.run_remote(
            ("sh", "-c", probe), timeout=timeout, operation="SSH connection test"
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) < 3 or lines[0] != "GRIM_HPC_OK":
            raise ProtocolError(
                "SSH connection test succeeded but returned an unexpected response"
            )
        return ConnectionInfo(
            transport=self.config.transport,
            hostname=lines[1],
            username=lines[2],
            result=result,
        )

    def upload_bundle(
        self,
        local_path: Path | str,
        remote_directory: str,
        *,
        timeout: float | None = None,
    ) -> TransferResult:
        """Create ``remote_directory`` and upload a portable file or directory."""

        local = Path(local_path).expanduser()
        if not local.exists():
            raise ConfigurationError(f"local bundle does not exist: {local}")
        if not local.is_file() and not local.is_dir():
            raise ConfigurationError(f"local bundle is not a file or directory: {local}")
        remote_directory = _validate_remote_path(remote_directory, "remote directory")
        remote_path = _join_remote_path(remote_directory, local.name)

        self.run_remote(
            ("mkdir", "-p", "--", remote_directory),
            operation="create remote bundle directory",
        )
        argv = self._copy_argv(
            local_operand=str(local.resolve()),
            remote_operand=remote_directory,
            upload=True,
            recursive=local.is_dir(),
        )
        result = self._execute(
            argv,
            timeout=self._operation_timeout(timeout, self.config.transfer_timeout),
            operation="upload portable bundle",
        )
        return TransferResult("upload", local, remote_path, result)

    def invoke_hpc_bundle(
        self,
        remote_cli_path: str,
        arguments: Sequence[str] = (),
        *,
        python_executable: str = "python3",
        timeout: float | None = None,
    ) -> HpcBundleResult:
        """Invoke a remote ``hpc_bundle.py`` CLI and parse its JSON response.

        ``arguments`` should select the CLI operation.  The entire stdout may
        be JSON, or the final non-empty line may be a JSON object.
        Common ``run_id``/``job_id``/``job_ids`` shapes are normalized for the
        Runs tab while the complete payload remains available.
        """

        remote_cli_path = _validate_remote_path(remote_cli_path, "remote CLI path")
        python_executable = _require_clean_text(python_executable, "python_executable")
        if python_executable.startswith("-"):
            raise ConfigurationError("python_executable may not start with '-'")
        result = self._run_remote(
            (python_executable, remote_cli_path, *arguments),
            timeout=timeout,
            operation="invoke remote hpc_bundle CLI",
            check=False,
        )
        payload = self._parse_json_payload(result.stdout)
        run_id = self._extract_run_id(payload)
        job_ids = self._extract_job_ids(payload)
        schema_value = payload.get("schema")
        schema = str(schema_value) if schema_value is not None else None
        ok_value = payload.get("ok", result.returncode == 0)
        ok = bool(ok_value)
        bundle_value = payload.get("bundle_id")
        bundle_id = str(bundle_value) if bundle_value is not None else None
        bundle_result = HpcBundleResult(
            payload=payload,
            run_id=run_id,
            job_ids=job_ids,
            result=result,
            schema=schema,
            ok=ok,
            bundle_id=bundle_id,
        )
        if result.returncode != 0 or not ok:
            raise HpcBundleFailedError(bundle_result)
        return bundle_result

    def stage_hpc_bundle(
        self,
        remote_cli_path: str,
        remote_bundle_directory: str,
        remote_workspace_root: str,
        *,
        run_driver: bool = False,
        submit: bool = False,
        python_executable: str = "python3",
        timeout: float | None = None,
    ) -> HpcBundleResult:
        """Stage and optionally submit the finalized GRIM bundle contract.

        This runs ``hpc_bundle.py stage BUNDLE --workspace-root WORKSPACE``
        with optional ``--run-driver`` and ``--submit`` flags.  A successful
        response must use schema ``ghost.hpc.stage-result.v1``.  Any submitted
        jobs belong to Slurm and continue after SSH or GRIM disconnects.
        """

        remote_bundle_directory = _validate_remote_path(
            remote_bundle_directory, "remote bundle directory"
        )
        remote_workspace_root = _validate_remote_path(
            remote_workspace_root, "remote workspace root"
        )
        arguments = [
            "stage",
            remote_bundle_directory,
            "--workspace-root",
            remote_workspace_root,
        ]
        if run_driver:
            arguments.append("--run-driver")
        if submit:
            arguments.append("--submit")
        result = self.invoke_hpc_bundle(
            remote_cli_path,
            arguments,
            python_executable=python_executable,
            timeout=(self.config.transfer_timeout if timeout is None else timeout),
        )
        if result.schema != "ghost.hpc.stage-result.v1":
            raise ProtocolError(
                "remote hpc_bundle returned unsupported schema "
                f"{result.schema!r}; expected 'ghost.hpc.stage-result.v1'"
            )
        return result

    def query_squeue(
        self, job_ids: Sequence[str | int], *, timeout: float | None = None
    ) -> tuple[JobStatus, ...]:
        """Return active/pending Slurm rows from ``squeue``."""

        normalized = _validate_job_ids(job_ids)
        result = self._run_remote(
            (
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(normalized),
                "--format=%i|%T|%j|%M|%R",
            ),
            timeout=timeout,
            operation="query Slurm queue",
            check=False,
        )
        if result.returncode != 0:
            # A purged completed job commonly makes squeue exit 1. Treat that
            # as the expected hand-off to sacct, not as a transport failure.
            missing_job = (
                not result.stdout.strip()
                and "invalid job id" in result.stderr.casefold()
            )
            if missing_job:
                return ()
            raise CommandFailedError("query Slurm queue", result)
        return self._parse_status_rows(result.stdout, source="squeue")

    def query_sacct(
        self, job_ids: Sequence[str | int], *, timeout: float | None = None
    ) -> tuple[JobStatus, ...]:
        """Return completed/historical Slurm rows from ``sacct``."""

        normalized = _validate_job_ids(job_ids)
        base_arguments = (
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(normalized),
            "--format=JobID,State%32,JobName,Elapsed,ExitCode",
        )
        # Slurm added sacct's --array expansion switch in 23.02.  Prefer it
        # because it exposes every task for correct parent-state aggregation,
        # but retain compatibility with older cluster clients.  Retry only
        # when the command explicitly says this option is unsupported; SSH,
        # authentication and accounting failures must remain visible.
        array_arguments = base_arguments[:3] + ("--array",) + base_arguments[3:]
        result = self._run_remote(
            array_arguments,
            timeout=timeout,
            operation="query Slurm accounting",
            check=False,
        )
        if result.returncode != 0:
            error = f"{result.stderr}\n{result.stdout}".casefold()
            unsupported_array_option = (
                "--array" in error
                and any(
                    marker in error
                    for marker in (
                        "unrecognized option",
                        "unknown option",
                        "invalid option",
                        "illegal option",
                        "not recognized",
                    )
                )
            )
            if not unsupported_array_option:
                raise CommandFailedError("query Slurm accounting", result)
            result = self.run_remote(
                base_arguments,
                timeout=timeout,
                operation="query Slurm accounting (legacy Slurm fallback)",
            )
        return self._parse_status_rows(result.stdout, source="sacct")

    def query_jobs(
        self, job_ids: Sequence[str | int], *, timeout: float | None = None
    ) -> tuple[JobStatus, ...]:
        """Query ``squeue`` first and use ``sacct`` for jobs no longer queued."""

        normalized = _validate_job_ids(job_ids)
        active = self.query_squeue(normalized, timeout=timeout)
        active_by_id = {
            job_id: tuple(
                row
                for row in active
                if _job_row_belongs_to(job_id, row.job_id)
            )
            for job_id in normalized
        }
        missing = tuple(job_id for job_id in normalized if not active_by_id[job_id])
        historical = self.query_sacct(missing, timeout=timeout) if missing else ()
        historical_by_id = {
            job_id: tuple(
                row
                for row in historical
                if _job_row_belongs_to(job_id, row.job_id)
            )
            for job_id in missing
        }
        ordered = []
        for job_id in normalized:
            matching_rows = active_by_id.get(job_id) or historical_by_id.get(job_id, ())
            if matching_rows:
                row = _aggregate_job_rows(job_id, matching_rows)
            else:
                row = JobStatus(
                    job_id=job_id,
                    state="UNKNOWN",
                    job_name="",
                    elapsed="",
                    detail="No squeue or sacct record returned",
                    source="unavailable",
                )
            ordered.append(row)
        statuses = list(ordered)
        known = {row.job_id for row in statuses}
        statuses.extend(
            row for row in (*active, *historical) if row.job_id not in known
        )
        return tuple(statuses)

    def tail_log(
        self,
        remote_path: str,
        *,
        lines: int = 200,
        timeout: float | None = None,
    ) -> str:
        """Return the last ``lines`` of a validated remote log path."""

        remote_path = _validate_remote_path(remote_path, "remote log path")
        if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= 100000:
            raise ConfigurationError("lines must be an integer from 1 through 100000")
        result = self.run_remote(
            ("tail", "--lines", str(lines), "--", remote_path),
            timeout=timeout,
            operation="tail remote log",
        )
        return result.stdout

    def tail_run_logs(
        self,
        remote_run_directory: str,
        *,
        lines: int = 80,
        files: int = 8,
        timeout: float | None = None,
    ) -> str:
        """Tail recent Slurm task logs without interpolating remote paths."""

        run_directory = _validate_remote_path(
            remote_run_directory, "remote run directory"
        )
        if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= 10000:
            raise ConfigurationError("lines must be an integer from 1 through 10000")
        if isinstance(files, bool) or not isinstance(files, int) or not 1 <= files <= 100:
            raise ConfigurationError("files must be an integer from 1 through 100")
        script = (
            'logs="$1/logs"; lines="$2"; files="$3"; '
            '[ -d "$logs" ] || exit 0; '
            'find "$logs" -maxdepth 1 -type f '
            '\\( -name "*.out" -o -name "*.err" \\) -print | '
            'sort | tail -n "$files" | while IFS= read -r file; do '
            'printf "\\n===== %s =====\\n" "$file"; '
            'tail -n "$lines" -- "$file"; done'
        )
        result = self.run_remote(
            ("sh", "-c", script, "grim-tail-logs", run_directory, str(lines), str(files)),
            timeout=timeout,
            operation="tail Slurm task logs",
        )
        return result.stdout

    def read_stage_result(
        self,
        remote_stage_result: str,
        *,
        timeout: float | None = None,
    ) -> HpcBundleResult:
        """Read and validate a previously written stage-result document."""

        remote_path = _validate_remote_path(
            remote_stage_result, "remote stage-result path"
        )
        command_result = self.run_remote(
            ("cat", "--", remote_path),
            timeout=timeout,
            operation="read HPC stage result",
        )
        payload = self._parse_json_payload(command_result.stdout)
        schema = str(payload.get("schema") or "")
        if schema != "ghost.hpc.stage-result.v1":
            raise ProtocolError(
                "remote stage result has unsupported schema "
                f"{schema!r}; expected 'ghost.hpc.stage-result.v1'"
            )
        bundle_value = payload.get("bundle_id")
        return HpcBundleResult(
            payload=payload,
            run_id=self._extract_run_id(payload),
            job_ids=self._extract_job_ids(payload),
            result=command_result,
            schema=schema,
            ok=bool(payload.get("ok", False)),
            bundle_id=str(bundle_value) if bundle_value is not None else None,
        )

    def recover_hpc_bundle(
        self,
        remote_cli_path: str,
        remote_stage_directory: str,
        *,
        python_executable: str = "python3",
        timeout: float | None = None,
    ) -> HpcBundleResult:
        """Ask the trusted bundle CLI to reconstruct an interrupted stage."""

        stage_directory = _validate_remote_path(
            remote_stage_directory, "remote stage directory"
        )
        try:
            result = self.invoke_hpc_bundle(
                remote_cli_path,
                ("recover", stage_directory),
                python_executable=python_executable,
                timeout=timeout,
            )
        except HpcBundleFailedError as exc:
            # A recovered partial/running stage correctly reports ok=false;
            # its validated job IDs are still the information the GUI needs.
            result = exc.bundle_result
        if result.schema != "ghost.hpc.stage-result.v1":
            raise ProtocolError(
                "remote hpc_bundle recovery returned unsupported schema "
                f"{result.schema!r}; expected 'ghost.hpc.stage-result.v1'"
            )
        return result

    def cancel(
        self, job_ids: Sequence[str | int], *, timeout: float | None = None
    ) -> CommandResult:
        """Cancel one or more validated Slurm job IDs with ``scancel``."""

        normalized = _validate_job_ids(job_ids)
        return self.run_remote(
            ("scancel", "--", *normalized),
            timeout=timeout,
            operation="cancel Slurm job",
        )

    def download_results(
        self,
        remote_path: str,
        local_directory: Path | str,
        *,
        recursive: bool = True,
        timeout: float | None = None,
    ) -> TransferResult:
        """Download a validated remote result file or tree using SCP/PSCP."""

        remote_path = _validate_remote_path(remote_path, "remote result path")
        local_directory = Path(local_directory).expanduser()
        if local_directory.exists() and not local_directory.is_dir():
            raise ConfigurationError(
                f"local result destination is not a directory: {local_directory}"
            )
        local_directory.mkdir(parents=True, exist_ok=True)
        local_path = local_directory / PurePosixPath(remote_path).name
        if local_path.exists():
            raise ConfigurationError(
                "local result target already exists; choose another parent folder: "
                f"{local_path}"
            )
        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{local_path.name}.grim-download-",
                dir=str(local_directory.resolve()),
            )
        )
        try:
            argv = self._copy_argv(
                local_operand=str(staging_directory),
                remote_operand=remote_path,
                upload=False,
                recursive=recursive,
            )
            result = self._execute(
                argv,
                timeout=self._operation_timeout(timeout, self.config.transfer_timeout),
                operation="download HPC results",
            )
            staged_path = staging_directory / local_path.name
            if not staged_path.exists():
                raise ProtocolError(
                    "download command succeeded but the expected result was not "
                    f"created: {staged_path.name}"
                )
            # The staging directory is a private sibling of the final target, so
            # this publication is an atomic same-filesystem rename.
            staged_path.replace(local_path)
        finally:
            shutil.rmtree(staging_directory, ignore_errors=True)
        return TransferResult("download", local_path, remote_path, result)

    def _operation_timeout(self, value: float | None, default: float) -> float:
        return default if value is None else _validate_timeout(value, "timeout")

    @staticmethod
    def _validate_command_tokens(arguments: Sequence[str]) -> tuple[str, ...]:
        if isinstance(arguments, (str, bytes)) or not arguments:
            raise ConfigurationError("remote command must be a non-empty token sequence")
        tokens: list[str] = []
        for argument in arguments:
            token = str(argument)
            _require_clean_text(token, "remote command token")
            tokens.append(token)
        if tokens[0].startswith("-"):
            raise ConfigurationError("remote executable may not start with '-'")
        return tuple(tokens)

    def _openssh_common(self) -> list[str]:
        config = self.config
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ForwardX11=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "RemoteCommand=none",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            f"ConnectTimeout={max(1, math.ceil(config.connect_timeout))}",
        ]
        if config.known_hosts_file is not None:
            options.extend(["-o", f"UserKnownHostsFile={config.known_hosts_file}"])
        if config.ssh_config_file is not None:
            options.extend(["-F", str(config.ssh_config_file)])
        if config.identity_file is not None:
            options.extend(["-i", str(config.identity_file)])
        return options

    def _target(self) -> str:
        config = self.config
        if config.transport is Transport.PUTTY:
            assert config.putty_session is not None
            return config.putty_session
        if config.ssh_config_alias is not None:
            return config.ssh_config_alias
        assert config.host is not None
        return f"{config.username}@{config.host}" if config.username else config.host

    def _copy_target(self) -> str:
        """Return an SCP-compatible target, including IPv6 brackets."""

        config = self.config
        if config.transport is Transport.PUTTY or config.ssh_config_alias is not None:
            return self._target()
        assert config.host is not None
        host = f"[{config.host}]" if ":" in config.host else config.host
        return f"{config.username}@{host}" if config.username else host

    def _ssh_argv(self, command: str) -> list[str]:
        config = self.config
        if config.transport is Transport.OPENSSH:
            argv = [config.ssh_executable, "-T", *self._openssh_common()]
            if config.ssh_config_alias is None:
                assert config.port is not None
                argv.extend(["-p", str(config.port)])
            argv.extend(["--", self._target(), command])
            return argv

        argv = [
            config.plink_executable,
            "-batch",
            "-T",
            "-load",
            self._target(),
        ]
        if config.host_key is not None:
            argv.extend(["-hostkey", config.host_key])
        argv.append(command)
        return argv

    def _copy_argv(
        self,
        *,
        local_operand: str,
        remote_operand: str,
        upload: bool,
        recursive: bool,
    ) -> list[str]:
        config = self.config
        # OpenSSH 9 uses SFTP for SCP by default, where shell quotes become
        # literal filename characters.  argv already preserves spaces.
        remote_spec = f"{self._copy_target()}:{remote_operand}"
        if config.transport is Transport.OPENSSH:
            argv = [config.scp_executable, "-B", *self._openssh_common()]
            if config.ssh_config_alias is None:
                assert config.port is not None
                argv.extend(["-P", str(config.port)])
            if recursive:
                argv.append("-r")
            argv.append("--")
        else:
            argv = [
                config.pscp_executable,
                "-batch",
                "-load",
                self._target(),
            ]
            if config.host_key is not None:
                argv.extend(["-hostkey", config.host_key])
            if recursive:
                argv.append("-r")
        if upload:
            argv.extend([local_operand, remote_spec])
        else:
            argv.extend([remote_spec, local_operand])
        return argv

    def _execute(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        operation: str,
        check: bool = True,
    ) -> CommandResult:
        started = time.monotonic()
        try:
            completed = self._runner(tuple(argv), timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise CommandTimeoutError(operation, timeout, argv) from exc
        except FileNotFoundError as exc:
            executable = argv[0] if argv else "SSH client"
            raise ConfigurationError(
                f"{operation} could not start because {executable!r} was not found"
            ) from exc
        except OSError as exc:
            raise RemoteError(f"{operation} could not start: {exc}") from exc
        elapsed = time.monotonic() - started
        result = CommandResult(
            argv=tuple(str(part) for part in argv),
            returncode=int(completed.returncode),
            stdout=self._decode_output(completed.stdout),
            stderr=self._decode_output(completed.stderr),
            elapsed_seconds=elapsed,
        )
        if check and result.returncode != 0:
            guidance = None
            if (
                self.config.transport is Transport.PUTTY
                and _PUTTY_BATCH_PROMPT_MARKER in result.stderr.casefold()
            ):
                guidance = _PUTTY_BATCH_PROMPT_GUIDANCE
            raise CommandFailedError(operation, result, guidance=guidance)
        return result

    @staticmethod
    def _decode_output(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _parse_json_payload(stdout: str) -> Mapping[str, Any]:
        text = stdout.strip()
        candidates = [text] if text else []
        candidates.extend(
            line.strip() for line in reversed(text.splitlines()) if line.strip()
        )
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                return value
        excerpt = text if len(text) <= 500 else text[:497] + "..."
        raise ProtocolError(
            "remote hpc_bundle did not return a JSON object"
            + (f": {excerpt}" if excerpt else "")
        )

    @classmethod
    def _extract_run_id(cls, payload: Mapping[str, Any]) -> str | None:
        for key, value in cls._walk_items(payload):
            if key in {"run_id", "runId"} and value is not None:
                return str(value)
        bundle_id = payload.get("bundle_id")
        if bundle_id is not None:
            return str(bundle_id)
        return None

    @classmethod
    def _extract_job_ids(cls, payload: Mapping[str, Any]) -> tuple[str, ...]:
        values: list[str | int] = []
        for key, value in cls._walk_items(payload):
            if key in {"job_id", "jobId"} and isinstance(value, (str, int)):
                values.append(value)
            elif key in {"job_ids", "jobIds"} and isinstance(value, (list, tuple)):
                values.extend(item for item in value if isinstance(item, (str, int)))
        if not values:
            return ()
        try:
            return _validate_job_ids(values)
        except ConfigurationError as exc:
            raise ProtocolError(f"remote hpc_bundle returned an invalid job ID: {exc}") from exc

    @classmethod
    def _walk_items(cls, value: Any):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key), child
                yield from cls._walk_items(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from cls._walk_items(child)

    @staticmethod
    def _parse_status_rows(stdout: str, *, source: str) -> tuple[JobStatus, ...]:
        rows: list[JobStatus] = []
        for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split("|", 4)]
            if len(fields) != 5 or not fields[0]:
                raise ProtocolError(
                    f"{source} returned a malformed row at line {line_number}: {raw_line!r}"
                )
            rows.append(
                JobStatus(
                    job_id=fields[0],
                    state=_normalize_slurm_state(fields[1]),
                    job_name=fields[2],
                    elapsed=fields[3],
                    detail=fields[4],
                    source=source,
                )
            )
        return tuple(rows)


__all__ = [
    "CommandFailedError",
    "CommandResult",
    "CommandTimeoutError",
    "ConfigurationError",
    "ConnectionConfig",
    "ConnectionInfo",
    "HpcBundleFailedError",
    "HpcBundleResult",
    "HpcRemoteClient",
    "JobStatus",
    "ProcessRunner",
    "ProtocolError",
    "RemoteError",
    "TransferResult",
    "Transport",
]
