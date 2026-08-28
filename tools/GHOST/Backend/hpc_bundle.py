#!/usr/bin/env python3
"""Portable GRIM/GHOST HPC request bundles.

The portable bundle is deliberately *not* an HPC run.  Windows writes only
relative input files and a declarative request.  On the Linux login node this
module verifies those bytes, derives Linux paths, and configures a fresh copy
of the canonical HPC driver.  The driver itself then creates the final run
manifest and records the Linux solver/runtime provenance.

This distinction matters: a final manifest created on Windows would contain
Windows paths and a Windows numerical-runtime fingerprint, neither of which
describes the solver that runs under SLURM.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # Linux/POSIX login-node coordination.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - exercised by Windows bundle creation
    fcntl = None

try:  # Keep direct lease tests functional on Windows.
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None

from geometry_io import material_sidecar_paths
from hpc_common import BOR_DRIVER, TWOD_DRIVER, configure_driver


REQUEST_SCHEMA = "ghost.hpc.portable-request.v1"
STAGE_RESULT_SCHEMA = "ghost.hpc.stage-result.v1"
README_NAME = "README.txt"
REQUEST_NAME = "request.json"

_BUNDLE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLURM_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@,+-]*$")
_JOB_ID_RE = re.compile(r"\bSubmitted\s+batch\s+job\s+([0-9]+)\b", re.I)
_STAGE_LEASE_STALE_SECONDS = 300.0
_STAGE_LEASE_HEARTBEAT_SECONDS = 10.0

_COMMON_SETTINGS = {
    "FREQUENCIES_GHZ",
    "AZIMUTHS_DEG",
    "GEOMETRY_UNITS",
    "N_NODES",
    "N_JOBS",
    "SLURM_PARTITION",
    "SLURM_ACCOUNT",
    "SLURM_QOS",
    "SLURM_TIME",
    "CORES_PER_NODE",
    "MEM_PER_NODE",
    "MAX_WORKERS_PER_NODE",
    "SLURM_MAIL_TYPE",
    "SLURM_MAIL_USER",
    "MESH_CERTIFICATION",
    "BLAS_THREADS_PER_WORKER",
    "MEMORY_HEADROOM",
    "TASKS_PER_CHILD",
    "CLAIM_STALE_SECONDS",
}

_SETTINGS_BY_SOLVER = {
    "2d": _COMMON_SETTINGS | {
        "ARRAY_THROTTLE",
        "MEMORY_SAFETY",
        "MAX_SOLVE_GB",
        "MAX_PANELS",
        "ASSEMBLY_THREADS",
    },
    "bor": _COMMON_SETTINGS | {
        "ELEVATIONS_DEG",
        "BODY_AXIS_AZ_DEG",
        "BODY_AXIS_EL_DEG",
        "BODY_ROLL_DEG",
        "CFIE_ALPHA",
        "N_MODES",
        "MODE_TOL",
        "MAX_ELEMENTS",
        "ASSEMBLY",
        "TABLE_PRECISION",
        "STREAM_BUDGET_GB",
        "WORKERS_PER_UNIT",
    },
}

_POSITIVE_INTS = {
    "N_NODES",
    "N_JOBS",
    "MAX_PANELS",
    "MAX_ELEMENTS",
    "WORKERS_PER_UNIT",
    "BLAS_THREADS_PER_WORKER",
    "TASKS_PER_CHILD",
    "CLAIM_STALE_SECONDS",
}
_OPTIONAL_POSITIVE_INTS = {
    "ARRAY_THROTTLE",
    "CORES_PER_NODE",
    "MAX_WORKERS_PER_NODE",
    "N_MODES",
}
_FINITE_NUMBERS = {
    "BODY_AXIS_AZ_DEG",
    "BODY_AXIS_EL_DEG",
    "BODY_ROLL_DEG",
}
_POSITIVE_NUMBERS = {
    "MAX_SOLVE_GB",
    "MODE_TOL",
    "STREAM_BUDGET_GB",
}


class BundleError(ValueError):
    """A portable request is unsafe, corrupt, or internally inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence barrier for a completed directory update."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class _StageLease:
    """Exclusive, heartbeat-backed lease for one bundle stage operation.

    The lease lives in the workspace rather than inside the staged payload, so
    it also serializes first-time creation and never alters the verified file
    inventory.  A kernel-owned acquisition guard closes the stale
    check/replace race without permitting time-based theft from a paused
    process; owner token and generation checks prevent an old process from
    heartbeating or deleting a successor's lease after takeover.
    """

    def __init__(
        self,
        path: Path,
        *,
        stale_seconds: float = _STAGE_LEASE_STALE_SECONDS,
        heartbeat_seconds: float = _STAGE_LEASE_HEARTBEAT_SECONDS,
    ) -> None:
        self.path = Path(path)
        self.guard_path = self.path.with_name(self.path.name + ".guard")
        self.stale_seconds = float(stale_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        if not math.isfinite(self.stale_seconds) or self.stale_seconds <= 0.0:
            raise ValueError("Stage lease stale_seconds must be positive and finite.")
        if not math.isfinite(self.heartbeat_seconds) or self.heartbeat_seconds <= 0.0:
            raise ValueError("Stage lease heartbeat_seconds must be positive and finite.")
        self.owner_token = uuid.uuid4().hex
        self.generation = 0
        self.recovered_stale = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _read_document(path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.is_symlink() or path.stat().st_size > 64 * 1024:
                return None
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return document if isinstance(document, dict) else None

    @staticmethod
    def _identity(document: Optional[Mapping[str, Any]]) -> Tuple[str, int]:
        if not document:
            return "", 0
        token = str(document.get("owner_token") or "")
        generation = document.get("generation", 0)
        if isinstance(generation, bool):
            generation = 0
        try:
            generation = int(generation)
        except (TypeError, ValueError, OverflowError):
            generation = 0
        return token, max(generation, 0)

    @staticmethod
    def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def _matches(self) -> bool:
        return self._identity(self._read_document(self.path)) == (
            self.owner_token,
            self.generation,
        )

    def require_current(self) -> None:
        try:
            current = self._matches()
        except OSError as exc:
            raise BundleError(
                "Could not verify ownership of the bundle stage lease; no new "
                "external action was started."
            ) from exc
        if not current:
            raise BundleError(
                "This stage process no longer owns the bundle lease; a newer "
                "generation may have recovered it. No stage journals were changed."
            )

    def _guard_payload(self, token: str) -> Dict[str, Any]:
        return {
            "schema": "ghost.hpc.stage-lease-guard.v1",
            "owner_token": token,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "created_utc": _utc_now(),
        }

    @staticmethod
    def _try_lock_file(path: Path) -> Optional[int]:
        if path.is_symlink():
            raise BundleError(f"Stage lease guard cannot be a symbolic link: {path}")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(str(path), flags, 0o600)
        except OSError as exc:
            if exc.errno in {
                getattr(errno, "ELOOP", -1), getattr(errno, "EMLINK", -1)
            }:
                raise BundleError(
                    f"Stage lease guard cannot be a symbolic link: {path}"
                ) from exc
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise BundleError(
                    f"Stage lease guard is not a regular file: {path}"
                )
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in {errno.EACCES, errno.EAGAIN}:
                        os.close(fd)
                        return None
                    raise
            elif msvcrt is not None:  # pragma: no cover - Windows-only branch
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if exc.errno in {
                        errno.EACCES,
                        errno.EAGAIN,
                        getattr(errno, "EDEADLK", -1),
                    }:
                        os.close(fd)
                        return None
                    raise
            else:  # pragma: no cover - every supported platform has one API
                raise BundleError("No supported advisory file-lock API is available.")
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _unlock_file(fd: int) -> None:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows-only branch
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)

    def _acquire_guard(self) -> Tuple[int, str]:
        token = uuid.uuid4().hex
        fd = self._try_lock_file(self.guard_path)
        if fd is None:
            raise BundleError(
                "Another process is acquiring this bundle's stage lease; retry "
                "after that stage command finishes."
            )
        payload = (
            json.dumps(
                self._guard_payload(token),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
            os.fsync(fd)
            return fd, token
        except BaseException:
            self._unlock_file(fd)
            raise

    def _release_guard(self, guard: Tuple[int, str]) -> None:
        try:
            self._unlock_file(guard[0])
        except OSError:
            pass

    @classmethod
    def _release_owned_file(cls, path: Path, token: str, generation: Optional[int] = None) -> None:
        try:
            document = cls._read_document(path)
        except OSError:
            return
        if document is None or str(document.get("owner_token") or "") != token:
            return
        if generation is not None and cls._identity(document) != (token, generation):
            return
        try:
            path.unlink()
        except OSError:
            pass

    def _lease_payload(self, generation: int) -> Dict[str, Any]:
        return {
            "schema": "ghost.hpc.stage-lease.v1",
            "owner_token": self.owner_token,
            "generation": int(generation),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "acquired_utc": _utc_now(),
        }

    def __enter__(self) -> "_StageLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise BundleError(f"Stage lease cannot be a symbolic link: {self.path}")
        guard = self._acquire_guard()
        temporary: Optional[Path] = None
        try:
            prior = self._read_document(self.path)
            if self.path.exists():
                try:
                    age = time.time() - self.path.stat().st_mtime
                except OSError as exc:
                    raise BundleError("Could not inspect the existing stage lease.") from exc
                if age <= self.stale_seconds:
                    owner = prior or {}
                    raise BundleError(
                        "This bundle already has an active stage/submit lease "
                        f"(host={owner.get('host', 'unknown')}, "
                        f"pid={owner.get('pid', 'unknown')}); retry after it finishes."
                    )
                self.generation = self._identity(prior)[1] + 1
                self.recovered_stale = True
                temporary = self.path.with_name(
                    f".{self.path.name}.{self.owner_token}.tmp"
                )
                self._write_exclusive(
                    temporary, self._lease_payload(self.generation)
                )
                # The guard tells the old owner's heartbeat to stop.  Recheck
                # immediately before replacement so a heartbeat already in
                # flight cannot be mistaken for a dead stage process.
                if time.time() - self.path.stat().st_mtime <= self.stale_seconds:
                    raise BundleError(
                        "The prior stage lease resumed heartbeating during stale recovery; "
                        "the running stage command was left untouched."
                    )
                os.replace(str(temporary), str(self.path))
                temporary = None
            else:
                self.generation = 1
                self._write_exclusive(
                    self.path, self._lease_payload(self.generation)
                )
            if not self._matches():
                raise BundleError("Stage lease ownership could not be verified.")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            self._release_guard(guard)

        self._stop.clear()

        def _beat() -> None:
            while not self._stop.wait(self.heartbeat_seconds):
                try:
                    heartbeat_guard = self._acquire_guard()
                except (BundleError, OSError):
                    continue
                try:
                    try:
                        current = self._matches()
                    except OSError:
                        continue
                    if not current:
                        return
                    try:
                        now = time.time()
                        os.utime(str(self.path), (now, now))
                    except OSError:
                        continue
                finally:
                    self._release_guard(heartbeat_guard)

        self._thread = threading.Thread(
            target=_beat, name="stage-lease-heartbeat", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.heartbeat_seconds * 2.0))
            self._thread = None
        # Use the same acquisition guard as stale takeover so the token check
        # and unlink cannot straddle a successor's atomic replacement.
        try:
            release_guard = self._acquire_guard()
        except (BundleError, OSError):
            return
        try:
            self._release_owned_file(
                self.path, self.owner_token, self.generation
            )
        finally:
            self._release_guard(release_guard)


def _safe_relative_path(raw_value: Any, *, label: str) -> str:
    if not isinstance(raw_value, str):
        raise BundleError(f"{label} must be a relative POSIX path string.")
    value = raw_value
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise BundleError(f"{label} is not a safe relative POSIX path: {value!r}.")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise BundleError(f"{label} contains an empty or dot path component.")
    return "/".join(parts)


def _resolved_bundle_file(root: Path, relative: str, *, label: str) -> Path:
    safe = _safe_relative_path(relative, label=label)
    candidate = root.joinpath(*safe.split("/"))
    if candidate.is_symlink():
        raise BundleError(f"{label} may not be a symbolic link: {safe}.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"{label} is missing or unreadable: {safe}.") from exc
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise BundleError(f"{label} escapes the bundle root: {safe}.") from exc
    if not resolved.is_file():
        raise BundleError(f"{label} is not a regular file: {safe}.")
    return resolved


def _require_number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise BundleError(f"{name} must be {qualifier}.")
    return number


def _require_positive_int(value: Any, *, name: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        suffix = " or null" if optional else ""
        raise BundleError(f"{name} must be a positive integer{suffix}.")


def _validate_numeric_list(
    value: Any,
    *,
    name: str,
    positive: bool = False,
    unique: bool = True,
) -> List[float]:
    if not isinstance(value, list) or not value:
        raise BundleError(f"{name} must be a non-empty JSON array.")
    numbers = [
        _require_number(item, name=f"{name}[{index}]", positive=positive)
        for index, item in enumerate(value)
    ]
    if unique and len(set(numbers)) != len(numbers):
        raise BundleError(f"{name} values must be unique.")
    return numbers


def _validate_slurm_text(value: Any, *, name: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value or not _SLURM_VALUE_RE.fullmatch(value):
        suffix = " or null" if optional else ""
        raise BundleError(
            f"{name} must be one conservative SLURM token{suffix}; spaces, "
            "control characters, and shell syntax are not accepted."
        )


def _validate_settings(solver: str, raw_settings: Any) -> Dict[str, Any]:
    if solver not in _SETTINGS_BY_SOLVER:
        raise BundleError("solver must be exactly '2d' or 'bor'.")
    if not isinstance(raw_settings, dict):
        raise BundleError("settings must be a JSON object.")
    settings = dict(raw_settings)
    unknown = sorted(set(settings) - _SETTINGS_BY_SOLVER[solver])
    if unknown:
        raise BundleError(
            "Unsupported or execution-owned setting(s): " + ", ".join(unknown)
        )

    for name in _POSITIVE_INTS:
        if name in settings:
            _require_positive_int(settings[name], name=name)
    for name in _OPTIONAL_POSITIVE_INTS:
        if name in settings:
            _require_positive_int(settings[name], name=name, optional=True)
    for name in _FINITE_NUMBERS:
        if name in settings:
            _require_number(settings[name], name=name)
    for name in _POSITIVE_NUMBERS:
        if name in settings and settings[name] is not None:
            _require_number(settings[name], name=name, positive=True)

    for name in ("FREQUENCIES_GHZ", "AZIMUTHS_DEG", "ELEVATIONS_DEG"):
        if name in settings:
            values = _validate_numeric_list(
                settings[name], name=name, positive=(name == "FREQUENCIES_GHZ")
            )
            if name == "FREQUENCIES_GHZ" and len(
                {f"{value:.3f}" for value in values}
            ) != len(values):
                raise BundleError(
                    "FREQUENCIES_GHZ values must remain distinct at 0.001 GHz."
                )

    if "GEOMETRY_UNITS" in settings and settings["GEOMETRY_UNITS"] not in {
        "inches",
        "meters",
    }:
        raise BundleError("GEOMETRY_UNITS must be 'inches' or 'meters'.")
    if "MESH_CERTIFICATION" in settings and not isinstance(
        settings["MESH_CERTIFICATION"], bool
    ):
        raise BundleError("MESH_CERTIFICATION must be true or false.")
    if "ASSEMBLY_THREADS" in settings:
        value = settings["ASSEMBLY_THREADS"]
        if value != "auto":
            _require_positive_int(value, name="ASSEMBLY_THREADS")
    if "ASSEMBLY" in settings and settings["ASSEMBLY"] not in {
        "auto",
        "tables",
        "streaming",
    }:
        raise BundleError("ASSEMBLY must be auto, tables, or streaming.")
    if "TABLE_PRECISION" in settings and settings["TABLE_PRECISION"] not in {
        "auto",
        "single",
        "double",
    }:
        raise BundleError("TABLE_PRECISION must be auto, single, or double.")
    if "CFIE_ALPHA" in settings:
        alpha = _require_number(settings["CFIE_ALPHA"], name="CFIE_ALPHA")
        if not 0.0 < alpha < 1.0:
            raise BundleError("CFIE_ALPHA must lie strictly between 0 and 1.")
    for name in ("MEMORY_HEADROOM",):
        if name in settings:
            value = _require_number(settings[name], name=name)
            if not 0.0 < value <= 1.0:
                raise BundleError(f"{name} must lie in (0, 1].")
    if "MEMORY_SAFETY" in settings:
        value = _require_number(settings["MEMORY_SAFETY"], name="MEMORY_SAFETY")
        if value < 1.0:
            raise BundleError("MEMORY_SAFETY must be at least 1.")
    if "CLAIM_STALE_SECONDS" in settings and settings["CLAIM_STALE_SECONDS"] < 60:
        raise BundleError("CLAIM_STALE_SECONDS must be at least 60.")

    for name in ("SLURM_PARTITION",):
        if name in settings:
            _validate_slurm_text(settings[name], name=name)
    for name in (
        "SLURM_ACCOUNT",
        "SLURM_QOS",
        "SLURM_TIME",
        "MEM_PER_NODE",
        "SLURM_MAIL_TYPE",
        "SLURM_MAIL_USER",
    ):
        if name in settings:
            _validate_slurm_text(settings[name], name=name, optional=True)

    try:
        json.dumps(settings, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BundleError("settings must contain only finite JSON values.") from exc
    return settings


def _bundle_readme() -> str:
    return (
        "GRIM / GHOST portable HPC request\n"
        "=================================\n\n"
        "This folder is a declarative request, not a completed HPC run. It has\n"
        "no Windows paths or Windows solver/runtime provenance. Keep the whole\n"
        "folder together when uploading it to the Linux login node.\n\n"
        "From the Linux login node, with the matching GHOST checkout available,\n"
        "the one-command submit form is:\n\n"
        "  python3 /path/to/GHOST/Backend/hpc_bundle.py stage /path/to/THIS_FOLDER --workspace-root /scratch/$USER/grim --run-driver --submit\n\n"
        "Replace both example paths for your cluster. Omit --submit to build the\n"
        "run folder and SLURM scripts without calling sbatch. Omit --run-driver\n"
        "as well to verify and stage inputs only. The command prints one JSON\n"
        "object containing the stage path, run path, log path, and any detected\n"
        "SLURM job IDs. Compute jobs continue after the SSH session closes.\n\n"
        "Do not edit request.json or payload files after export; the Linux\n"
        "stager verifies their exact sizes and SHA-256 hashes.\n"
    )


def _copy_geometry_payload(
    temporary_root: Path,
    geometries: Sequence[Mapping[str, Any]],
    solver: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    allowed_roles = {"FRD", "OPN"} if solver == "2d" else {"BOR"}
    if not isinstance(geometries, Sequence) or isinstance(geometries, (str, bytes)):
        raise BundleError("geometries must be a non-empty sequence.")
    if not geometries:
        raise BundleError("At least one geometry is required.")

    geometry_records: List[Dict[str, Any]] = []
    file_records: List[Dict[str, Any]] = []
    stems = set()
    source_paths = set()
    for index, raw_record in enumerate(geometries):
        if not isinstance(raw_record, Mapping):
            raise BundleError(f"Geometry entry {index} must be an object.")
        unknown = set(raw_record) - {"role", "path"}
        if unknown:
            raise BundleError(
                f"Geometry entry {index} has unsupported field(s): "
                + ", ".join(sorted(str(value) for value in unknown))
            )
        role = str(raw_record.get("role", "")).strip().upper()
        if role not in allowed_roles:
            raise BundleError(
                f"Geometry entry {index} role must be one of "
                f"{sorted(allowed_roles)} for solver {solver}."
            )
        source = Path(str(raw_record.get("path", ""))).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".geo":
            raise BundleError(f"Geometry entry {index} is not a readable .geo file.")
        if source in source_paths:
            raise BundleError(f"Geometry file appears more than once: {source.name}.")
        source_paths.add(source)
        normalized_stem = source.stem.casefold()
        if normalized_stem in stems:
            raise BundleError(
                f"Geometry stems must be unique; duplicate stem {source.stem!r}."
            )
        if source.stem in ("", ".", ".."):
            raise BundleError(f"Geometry has an unsafe output stem: {source.name!r}.")
        stems.add(normalized_stem)

        try:
            sidecars = [
                Path(value).resolve()
                for value in material_sidecar_paths(str(source))
            ]
        except (OSError, ValueError) as exc:
            raise BundleError(
                f"Geometry {source.name!r} or its material sidecars are invalid: {exc}"
            ) from exc
        folder_name = f"{index:04d}_{source.stem}"
        relative_folder = f"payload/{role}/{folder_name}"
        _safe_relative_path(relative_folder, label=f"Geometry entry {index} folder")
        destination_folder = temporary_root.joinpath(*relative_folder.split("/"))
        destination_folder.mkdir(parents=True, exist_ok=False)

        geometry_relative = f"{relative_folder}/{source.name}"
        _safe_relative_path(geometry_relative, label=f"Geometry entry {index} path")
        geometry_destination = destination_folder / source.name
        shutil.copy2(source, geometry_destination)
        sidecar_relatives: List[str] = []
        for sidecar in sidecars:
            relative = f"{relative_folder}/{sidecar.name}"
            _safe_relative_path(relative, label=f"Geometry entry {index} sidecar")
            shutil.copy2(sidecar, destination_folder / sidecar.name)
            sidecar_relatives.append(relative)
        geometry_records.append(
            {
                "role": role,
                "path": geometry_relative,
                "sidecars": sidecar_relatives,
            }
        )

        for relative, kind in [(geometry_relative, "geometry")] + [
            (relative, "material") for relative in sidecar_relatives
        ]:
            path = temporary_root.joinpath(*relative.split("/"))
            file_records.append(
                {
                    "path": relative,
                    "kind": kind,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return geometry_records, file_records


def create_portable_bundle(
    bundle_dir: os.PathLike[str] | str,
    *,
    solver: str,
    geometries: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    bundle_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a portable directory bundle and return its verified request."""

    normalized_solver = str(solver).strip().lower()
    if not isinstance(settings, Mapping):
        raise BundleError("settings must be a mapping.")
    checked_settings = _validate_settings(normalized_solver, dict(settings))
    identifier = str(bundle_id or uuid.uuid4().hex).lower()
    if not _BUNDLE_ID_RE.fullmatch(identifier):
        raise BundleError("bundle_id must be 32 lowercase hexadecimal characters.")

    target = Path(bundle_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise BundleError(f"Bundle destination already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent)))
    try:
        geometry_records, file_records = _copy_geometry_payload(
            temporary, geometries, normalized_solver
        )
        readme_path = temporary / README_NAME
        readme_path.write_text(_bundle_readme(), encoding="utf-8", newline="\n")
        file_records.append(
            {
                "path": README_NAME,
                "kind": "documentation",
                "size": readme_path.stat().st_size,
                "sha256": _sha256_file(readme_path),
            }
        )
        request = {
            "schema": REQUEST_SCHEMA,
            "bundle_id": identifier,
            "created_utc": _utc_now(),
            "solver": normalized_solver,
            "settings": checked_settings,
            "geometries": geometry_records,
            "files": sorted(file_records, key=lambda item: str(item["path"])),
        }
        _write_json_atomic(temporary / REQUEST_NAME, request)
        # Validate the exact temporary tree before publishing it.  A malformed
        # source filename must never leave a destination that merely *looks*
        # like a completed bundle.
        verify_portable_bundle(temporary)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_portable_bundle(target)


def _read_request(bundle_root: Path) -> Tuple[Dict[str, Any], bytes]:
    request_path = bundle_root / REQUEST_NAME
    if request_path.is_symlink() or not request_path.is_file():
        raise BundleError(f"Bundle has no regular {REQUEST_NAME} file.")
    raw = request_path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise BundleError("request.json exceeds the 4 MiB safety limit.")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError("request.json is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise BundleError("request.json must contain one JSON object.")
    return value, raw


def verify_portable_bundle(bundle_dir: os.PathLike[str] | str) -> Dict[str, Any]:
    """Verify every declared byte and the complete portable request contract."""

    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir():
        raise BundleError(f"Bundle directory does not exist: {root}")
    request, _raw = _read_request(root)
    if request.get("schema") != REQUEST_SCHEMA:
        raise BundleError(
            f"Unsupported request schema {request.get('schema')!r}; expected "
            f"{REQUEST_SCHEMA!r}."
        )
    expected_request_fields = {
        "schema",
        "bundle_id",
        "created_utc",
        "solver",
        "settings",
        "geometries",
        "files",
    }
    if set(request) != expected_request_fields:
        raise BundleError("request.json contains missing or unsupported top-level fields.")
    identifier = request.get("bundle_id")
    if not isinstance(identifier, str) or not _BUNDLE_ID_RE.fullmatch(identifier):
        raise BundleError("request.json has no valid lowercase hexadecimal bundle_id.")
    solver = request.get("solver")
    if solver not in _SETTINGS_BY_SOLVER:
        raise BundleError("request.json solver must be exactly '2d' or 'bor'.")
    _validate_settings(str(solver), request.get("settings"))

    raw_files = request.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BundleError("request.json files must be a non-empty array.")
    file_by_path: Dict[str, Dict[str, Any]] = {}
    for index, raw_record in enumerate(raw_files):
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "path",
            "kind",
            "size",
            "sha256",
        }:
            raise BundleError(f"File inventory entry {index} has invalid fields.")
        relative = _safe_relative_path(
            raw_record["path"], label=f"File inventory entry {index} path"
        )
        if relative in file_by_path:
            raise BundleError(f"File inventory path appears twice: {relative}.")
        if raw_record["kind"] not in {"geometry", "material", "documentation"}:
            raise BundleError(f"File inventory entry {relative} has invalid kind.")
        size = raw_record["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleError(f"File inventory entry {relative} has invalid size.")
        expected_hash = raw_record["sha256"]
        if not isinstance(expected_hash, str) or not _HEX_SHA256_RE.fullmatch(
            expected_hash
        ):
            raise BundleError(f"File inventory entry {relative} has invalid SHA-256.")
        path = _resolved_bundle_file(root, relative, label="Bundle payload file")
        if path.stat().st_size != size or _sha256_file(path) != expected_hash:
            raise BundleError(f"Bundle payload differs from its inventory: {relative}.")
        file_by_path[relative] = raw_record

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_files = set(file_by_path) | {REQUEST_NAME}
    if actual_files != expected_files:
        raise BundleError(
            "Bundle file set differs from its exact inventory "
            f"(missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)})."
        )
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise BundleError("Portable bundles may not contain symbolic links.")

    raw_geometries = request.get("geometries")
    if not isinstance(raw_geometries, list) or not raw_geometries:
        raise BundleError("request.json geometries must be a non-empty array.")
    allowed_roles = {"FRD", "OPN"} if solver == "2d" else {"BOR"}
    geometry_paths = set()
    material_paths = set()
    stems = set()
    for index, raw_geometry in enumerate(raw_geometries):
        if not isinstance(raw_geometry, dict) or set(raw_geometry) != {
            "role",
            "path",
            "sidecars",
        }:
            raise BundleError(f"Geometry inventory entry {index} has invalid fields.")
        role = raw_geometry["role"]
        if role not in allowed_roles:
            raise BundleError(
                f"Geometry role {role!r} is incompatible with solver {solver}."
            )
        relative = _safe_relative_path(
            raw_geometry["path"], label=f"Geometry entry {index} path"
        )
        if relative in geometry_paths or not relative.lower().endswith(".geo"):
            raise BundleError(f"Geometry path is duplicate or not .geo: {relative}.")
        if file_by_path.get(relative, {}).get("kind") != "geometry":
            raise BundleError(f"Geometry is not declared as a geometry file: {relative}.")
        geometry_paths.add(relative)
        stem = Path(relative).stem
        normalized_stem = stem.casefold()
        if normalized_stem in stems:
            raise BundleError(f"Geometry stems must be unique; duplicate {stem!r}.")
        stems.add(normalized_stem)
        raw_sidecars = raw_geometry["sidecars"]
        if not isinstance(raw_sidecars, list) or not all(
            isinstance(value, str) for value in raw_sidecars
        ):
            raise BundleError(f"Geometry entry {relative} has an invalid sidecar list.")
        if len(set(raw_sidecars)) != len(raw_sidecars):
            raise BundleError(f"Geometry entry {relative} has duplicate sidecars.")
        declared_sidecars = {
            _safe_relative_path(value, label=f"Sidecar for {relative}")
            for value in raw_sidecars
        }
        geometry_path = _resolved_bundle_file(root, relative, label="Geometry file")
        try:
            parsed_sidecars = {
                Path(value).resolve().relative_to(root).as_posix()
                for value in material_sidecar_paths(str(geometry_path))
            }
        except (OSError, ValueError) as exc:
            raise BundleError(
                f"Geometry {relative!r} or its material sidecars are invalid: {exc}"
            ) from exc
        if declared_sidecars != parsed_sidecars:
            raise BundleError(
                f"Geometry sidecars do not match its parsed material references: {relative}."
            )
        for sidecar in declared_sidecars:
            if file_by_path.get(sidecar, {}).get("kind") != "material":
                raise BundleError(f"Sidecar is not declared as material: {sidecar}.")
        material_paths.update(declared_sidecars)

    if geometry_paths != {
        path for path, record in file_by_path.items() if record["kind"] == "geometry"
    }:
        raise BundleError("Geometry records do not cover the exact geometry inventory.")
    if material_paths != {
        path for path, record in file_by_path.items() if record["kind"] == "material"
    }:
        raise BundleError("Sidecar records do not cover the exact material inventory.")
    if set(file_by_path) - geometry_paths - material_paths != {README_NAME}:
        raise BundleError(f"The only documentation file must be {README_NAME}.")
    return request


def _linux_staging_available() -> bool:
    return sys.platform.startswith("linux")


def _copy_verified_files(bundle_root: Path, stage_dir: Path, request: Mapping[str, Any]) -> None:
    for record in request["files"]:
        relative = str(record["path"])
        source = _resolved_bundle_file(bundle_root, relative, label="Bundle payload file")
        destination = stage_dir.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _verify_staged_files(stage_dir: Path, request: Mapping[str, Any]) -> None:
    for record in request["files"]:
        relative = str(record["path"])
        path = _resolved_bundle_file(stage_dir, relative, label="Staged payload file")
        if path.stat().st_size != int(record["size"]) or _sha256_file(path) != record[
            "sha256"
        ]:
            raise BundleError(f"Staged payload differs from request: {relative}.")
    expected_payload = {
        str(record["path"])
        for record in request["files"]
        if str(record["path"]).startswith("payload/")
    }
    payload_root = stage_dir / "payload"
    actual_payload: set[str] = set()
    if payload_root.exists():
        for path in payload_root.rglob("*"):
            if path.is_symlink():
                raise BundleError(f"Staged payload may not contain symlinks: {path}.")
            if path.is_file():
                actual_payload.add(path.relative_to(stage_dir).as_posix())
            elif not path.is_dir():
                raise BundleError(f"Unsupported staged payload entry: {path}.")
    if actual_payload != expected_payload:
        extra = sorted(actual_payload - expected_payload)
        missing = sorted(expected_payload - actual_payload)
        detail = []
        if extra:
            detail.append("unexpected: " + ", ".join(extra))
        if missing:
            detail.append("missing: " + ", ".join(missing))
        raise BundleError(
            "Staged payload does not match the exact request inventory"
            + (" (" + "; ".join(detail) + ")" if detail else "")
            + "."
        )


def _parse_job_ids(output: str) -> List[str]:
    found: List[str] = []
    for value in _JOB_ID_RE.findall(output):
        if value not in found:
            found.append(value)
    return found


def _read_submitted_job_ids(run_dir: Optional[str]) -> List[str]:
    """Recover IDs persisted after each successful sbatch invocation."""

    if not run_dir:
        return []
    path = Path(run_dir) / "submitted_jobs.json"
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return []
        if document.get("schema") != "ghost.hpc.submitted-jobs.v1":
            return []
        values = document.get("job_ids", ())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).isdigit()]


def _read_all_submitted_job_ids(output_root: Path) -> List[str]:
    """Return every journaled job ID under a stage, in deterministic order."""

    found: List[str] = []
    if not output_root.is_dir() or output_root.is_symlink():
        return found
    for run_path in sorted(output_root.glob("run_*"), key=lambda path: path.name):
        if not run_path.is_dir() or run_path.is_symlink():
            continue
        for job_id in _read_submitted_job_ids(str(run_path)):
            if job_id not in found:
                found.append(job_id)
    return found


def recover_staged_bundle(stage_directory: os.PathLike[str] | str) -> Dict[str, Any]:
    """Reconstruct stage state without submitting or executing anything.

    This is the reconnect path for a lost SSH session.  A running state and
    durable job journals take precedence over an older stage-only result.
    Otherwise a matching finalized ``stage_result.json`` is returned.
    """

    stage_input = Path(stage_directory).expanduser()
    if not stage_input.is_absolute():
        raise BundleError("Recovery stage directory must be an absolute path.")
    if stage_input.is_symlink():
        raise BundleError("Recovery stage directory cannot be a symbolic link.")
    try:
        stage_dir = stage_input.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"Recovery stage directory is missing: {stage_input}") from exc
    match = re.fullmatch(r"grim_([0-9a-f]{32})", stage_dir.name)
    if match is None or not stage_dir.is_dir() or stage_dir.is_symlink():
        raise BundleError(
            "Recovery target must be a GRIM stage directory named grim_<bundle-id>."
        )
    bundle_id = match.group(1)
    metadata_path = stage_dir / "stage_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BundleError("Recovery target has no readable stage metadata.") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != "ghost.hpc.stage-metadata.v1"
        or metadata.get("bundle_id") != bundle_id
    ):
        raise BundleError("Recovery stage metadata does not match its directory.")

    state: Dict[str, Any] = {}
    state_path = stage_dir / "stage_state.json"
    if state_path.is_file():
        try:
            parsed_state = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(parsed_state, dict):
                state = parsed_state
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            state = {}

    output_root = stage_dir / "runs"
    candidates = [
        path
        for path in output_root.glob("run_*")
        if path.is_dir() and not path.is_symlink()
    ]
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    run_path = candidates[-1] if candidates else None
    run_dir = str(run_path) if run_path is not None else None
    job_ids = _read_all_submitted_job_ids(output_root)
    status = str(state.get("status") or "unknown")
    returncode = state.get("returncode")
    state_job_ids = state.get("job_ids", ())
    if isinstance(state_job_ids, (list, tuple)):
        for value in state_job_ids:
            job_id = str(value)
            if job_id.isdigit() and job_id not in job_ids:
                job_ids.append(job_id)
    complete = status == "complete" and returncode == 0
    log_path = stage_dir / "driver_submit.log"
    if log_path.is_file() and not log_path.is_symlink():
        try:
            for job_id in _parse_job_ids(
                log_path.read_text(encoding="utf-8", errors="replace")
            ):
                if job_id not in job_ids:
                    job_ids.append(job_id)
        except OSError:
            pass

    result_path = stage_dir / "stage_result.json"
    if result_path.is_file():
        try:
            finalized = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BundleError("Recovery stage result is unreadable.") from exc
        if (
            not isinstance(finalized, dict)
            or finalized.get("schema") != STAGE_RESULT_SCHEMA
            or finalized.get("bundle_id") != bundle_id
        ):
            raise BundleError("Recovery stage result does not match its directory.")
        finalized_ids_value = finalized.get("job_ids", ())
        finalized_ids = (
            {str(value) for value in finalized_ids_value if str(value).isdigit()}
            if isinstance(finalized_ids_value, (list, tuple))
            else set()
        )
        # A running state was durably published before the prior result was
        # invalidated.  It therefore always outranks that result.  Journals are
        # likewise monotonic evidence that a nominally terminal result is old.
        terminal_state_matches = (
            status in {"complete", "failed"}
            and bool(finalized.get("driver_ran"))
            and finalized.get("returncode") == returncode
            and bool(finalized.get("submission_requested"))
            == bool(state.get("submission_requested"))
        )
        if (
            status not in {"running", "complete", "failed"}
            or (terminal_state_matches and set(job_ids).issubset(finalized_ids))
        ):
            return finalized

    result = {
        "schema": STAGE_RESULT_SCHEMA,
        "ok": complete,
        "bundle_id": bundle_id,
        "solver": str(metadata.get("solver") or ""),
        "bundle_sha256": str(metadata.get("bundle_sha256") or ""),
        "stage_dir": str(stage_dir),
        "driver_path": str(stage_dir / "driver_configured.py"),
        "output_root": str(output_root),
        "driver_ran": bool(state),
        "submission_requested": bool(state.get("submission_requested", False)),
        "submitted": bool(job_ids),
        "run_dir": run_dir,
        "run_id": run_path.name if run_path is not None else None,
        "job_ids": job_ids,
        "log_path": str(log_path) if log_path.is_file() else None,
        "returncode": returncode,
        "recovered": True,
        "stage_state": status,
    }
    if not complete:
        result["error"] = (
            "The login-side stage command did not finalize. Recovered job IDs "
            "remain authoritative; inspect scheduler state before resubmitting."
        )
    return result


def stage_portable_bundle(
    bundle_dir: os.PathLike[str] | str,
    workspace_root: os.PathLike[str] | str,
    *,
    run_driver: bool = False,
    submit: bool = False,
) -> Dict[str, Any]:
    """Verify and stage a request on Linux, optionally building/submitting it.

    ``submit=True`` implies ``run_driver=True``.  The returned mapping is also
    written to ``stage_result.json`` so a dropped SSH connection can reconnect
    and inspect the same result without blindly resubmitting.
    """

    if not _linux_staging_available():
        raise BundleError(
            "Portable HPC requests must be staged by this command on the Linux "
            "login node; Windows may create or verify bundles only."
        )
    run_driver = bool(run_driver or submit)
    bundle_root = Path(bundle_dir).expanduser().resolve()
    request = verify_portable_bundle(bundle_root)
    reread_request, request_bytes = _read_request(bundle_root)
    if reread_request != request:
        raise BundleError("request.json changed while it was being verified.")
    bundle_hash = _sha256_bytes(request_bytes)

    workspace = Path(workspace_root).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    stage_dir = workspace / f"grim_{request['bundle_id']}"
    if (
        stage_dir == bundle_root
        or bundle_root in stage_dir.parents
        or stage_dir in bundle_root.parents
    ):
        raise BundleError(
            "The derived stage directory and uploaded bundle directory may not overlap."
        )
    lease_path = workspace / f".grim_{request['bundle_id']}.stage-lease.json"
    with _StageLease(lease_path) as stage_lease:
        return _stage_portable_bundle_with_lease(
            bundle_root=bundle_root,
            workspace=workspace,
            request=request,
            bundle_hash=bundle_hash,
            run_driver=run_driver,
            submit=submit,
            stage_lease=stage_lease,
        )


def _stage_portable_bundle_with_lease(
    *,
    bundle_root: Path,
    workspace: Path,
    request: Mapping[str, Any],
    bundle_hash: str,
    run_driver: bool,
    submit: bool,
    stage_lease: _StageLease,
) -> Dict[str, Any]:
    """Complete one verified stage operation while its workspace lease is held."""

    stage_lease.require_current()
    stage_dir = workspace / f"grim_{request['bundle_id']}"
    metadata_path = stage_dir / "stage_metadata.json"
    result_path = stage_dir / "stage_result.json"
    state_path = stage_dir / "stage_state.json"
    prior_result: Optional[Dict[str, Any]] = None

    if stage_dir.exists():
        if not metadata_path.is_file():
            raise BundleError(
                f"Stage target already exists without GRIM metadata: {stage_dir}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("bundle_sha256") != bundle_hash:
            raise BundleError(
                "The bundle ID already exists in this workspace with different bytes."
            )
        _verify_staged_files(stage_dir, request)
        if result_path.is_file():
            parsed_prior = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(parsed_prior, dict):
                raise BundleError("The existing stage result is not a JSON object.")
            prior_result = parsed_prior
    else:
        private_stage = workspace / (
            f".{stage_dir.name}.{stage_lease.owner_token}.stage-tmp"
        )
        private_stage.mkdir(parents=False, exist_ok=False)
        try:
            _copy_verified_files(bundle_root, private_stage, request)
            _write_json_atomic(
                private_stage / metadata_path.name,
                {
                    "schema": "ghost.hpc.stage-metadata.v1",
                    "bundle_id": request["bundle_id"],
                    "bundle_sha256": bundle_hash,
                    "solver": request["solver"],
                    "staged_utc": _utc_now(),
                },
            )
            stage_lease.require_current()
            os.replace(str(private_stage), str(stage_dir))
            _fsync_directory(workspace)
        except BaseException:
            # Only this lease's unobservable private sibling is removed.  The
            # final grim_<id> path is published by one atomic rename and can
            # therefore never expose an interrupted partial copy.
            shutil.rmtree(private_stage, ignore_errors=True)
            raise

    payload_root = stage_dir / "payload"
    output_root = stage_dir / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    settings = dict(request["settings"])
    if request["solver"] == "2d":
        frd_root = payload_root / "FRD"
        opn_root = payload_root / "OPN"
        frd_root.mkdir(parents=True, exist_ok=True)
        opn_root.mkdir(parents=True, exist_ok=True)
        settings.update({"FRD_DIR": str(frd_root), "OPN_DIR": str(opn_root)})
        canonical_driver = TWOD_DRIVER
    else:
        bor_root = payload_root / "BOR"
        bor_root.mkdir(parents=True, exist_ok=True)
        settings["GEOMETRY_DIRS"] = [str(bor_root)]
        canonical_driver = BOR_DRIVER
    settings.update(
        {
            "OUTPUT_DIR": str(output_root),
            "PYTHON_EXE": sys.executable,
            "SUBMIT": bool(submit),
        }
    )
    driver_path = configure_driver(
        canonical_driver, stage_dir / "driver_configured.py", settings
    )

    base_result: Dict[str, Any] = {
        "schema": STAGE_RESULT_SCHEMA,
        "ok": True,
        "bundle_id": request["bundle_id"],
        "solver": request["solver"],
        "bundle_sha256": bundle_hash,
        "stage_dir": str(stage_dir),
        "driver_path": str(driver_path),
        "output_root": str(output_root),
        "driver_ran": False,
        "submission_requested": bool(submit),
        "submitted": False,
        "run_dir": None,
        "run_id": None,
        "job_ids": [],
        "log_path": None,
        "returncode": None,
    }
    state: Dict[str, Any] = {}
    if state_path.is_file():
        parsed_state = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(parsed_state, dict):
            state = parsed_state

    recovered_job_ids = _read_all_submitted_job_ids(output_root)
    state_ids_value = state.get("job_ids", ())
    if isinstance(state_ids_value, (list, tuple)):
        for value in state_ids_value:
            job_id = str(value)
            if job_id.isdigit() and job_id not in recovered_job_ids:
                recovered_job_ids.append(job_id)
    prior_log = stage_dir / "driver_submit.log"
    if prior_log.is_file() and not prior_log.is_symlink():
        try:
            for job_id in _parse_job_ids(
                prior_log.read_text(encoding="utf-8", errors="replace")
            ):
                if job_id not in recovered_job_ids:
                    recovered_job_ids.append(job_id)
        except OSError:
            pass

    state_is_running = state.get("status") == "running"
    state_status = str(state.get("status") or "")
    if prior_result is not None and prior_result.get("driver_ran"):
        prior_ids_value = prior_result.get("job_ids", ())
        prior_ids = (
            {str(value) for value in prior_ids_value if str(value).isdigit()}
            if isinstance(prior_ids_value, (list, tuple))
            else set()
        )
        if not state:
            prior_is_current = set(recovered_job_ids).issubset(prior_ids)
        elif state_status in {"complete", "failed"}:
            prior_is_current = (
                prior_result.get("returncode") == state.get("returncode")
                and bool(prior_result.get("submission_requested"))
                == bool(state.get("submission_requested"))
                and set(recovered_job_ids).issubset(prior_ids)
            )
        else:
            prior_is_current = False
        if prior_is_current:
            if bool(prior_result.get("submission_requested")) == bool(submit):
                return prior_result
            raise BundleError(
                "This bundle already built a run with a different submission mode; "
                "use a new exported bundle rather than creating a duplicate run."
            )

    recovered_running_state = bool(state_is_running)
    if state_is_running and bool(state.get("submission_requested")):
        evidence = (
            " Recorded SLURM job ID(s): " + ", ".join(recovered_job_ids) + "."
            if recovered_job_ids
            else " SLURM may have accepted a job before its ID could be journaled."
        )
        raise BundleError(
            "A prior submit-requested stage invocation did not finalize."
            + evidence
            + " Recover and reconcile that attempt with the scheduler; automatic "
            "resubmission is disabled to prevent duplicate jobs."
        )

    if state and not state_is_running:
        if bool(state.get("submission_requested")) != bool(submit):
            raise BundleError(
                "This bundle already ran with a different submission mode; use a "
                "new exported bundle rather than creating another run."
            )
        if state_status == "complete" and state.get("returncode") == 0:
            # A terminal state is itself a durable completion record. If the
            # companion result was lost after publication, reconstruct it
            # rather than rerunning a driver that may already have submitted.
            return recover_staged_bundle(stage_dir)
        if bool(state.get("submission_requested")):
            evidence = (
                " Recorded SLURM job ID(s): " + ", ".join(recovered_job_ids) + "."
                if recovered_job_ids
                else " The scheduler outcome cannot be proven from local journals."
            )
            raise BundleError(
                "A prior submit-requested stage attempt has no trustworthy final "
                "result."
                + evidence
                + " Recover and reconcile it with Slurm; automatic resubmission "
                "is disabled to prevent duplicate jobs."
            )

    if not run_driver:
        if state_is_running:
            raise BundleError(
                "A prior non-submitting driver invocation did not finalize; rerun with "
                "--run-driver to recover it rather than masking its running state."
            )
        _write_json_atomic(result_path, base_result)
        return base_result
    # Recheck exact staged bytes and inventory immediately before allowing the
    # configured driver to execute them.
    _verify_staged_files(stage_dir, request)
    stage_lease.require_current()
    _write_json_atomic(
        state_path,
        {
            "schema": "ghost.hpc.stage-state.v1",
            "status": "running",
            "started_utc": _utc_now(),
            "submission_requested": bool(submit),
            "lease_owner_token": stage_lease.owner_token,
            "lease_generation": stage_lease.generation,
            "recovered_prior_running_state": recovered_running_state,
        },
    )
    # Publishing running state first makes it authoritative to reconnecting
    # readers; removing the older result then prevents it from being mistaken
    # for this attempt if the login session dies before the driver returns.
    try:
        result_path.unlink()
    except FileNotFoundError:
        pass
    else:
        _fsync_directory(result_path.parent)
    before_runs = {path.resolve() for path in output_root.glob("run_*") if path.is_dir()}
    inherited_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    backend_dir = str(Path(__file__).resolve().parent)
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = (
        backend_dir
        if not inherited_pythonpath
        else os.pathsep.join((backend_dir, inherited_pythonpath))
    )
    log_path = stage_dir / "driver_submit.log"
    try:
        completed = subprocess.run(
            [sys.executable, str(driver_path)],
            cwd=str(stage_dir),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        stage_lease.require_current()
        message = f"Could not start the configured Linux HPC driver: {exc}"
        log_path.write_text(message + "\n", encoding="utf-8", newline="\n")
        failed_result = dict(base_result)
        failed_result.update(
            {
                "ok": False,
                "log_path": str(log_path),
                "error": message,
            }
        )
        _write_json_atomic(result_path, failed_result)
        _write_json_atomic(
            state_path,
            {
                "schema": "ghost.hpc.stage-state.v1",
                "status": "failed",
                "finished_utc": _utc_now(),
                "submission_requested": bool(submit),
                "returncode": None,
                "run_dir": None,
                "job_ids": [],
                "lease_owner_token": stage_lease.owner_token,
                "lease_generation": stage_lease.generation,
                "recovered_prior_running_state": recovered_running_state,
            },
        )
        return failed_result
    stage_lease.require_current()
    log_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    after_runs = {path.resolve() for path in output_root.glob("run_*") if path.is_dir()}
    new_runs = sorted(after_runs - before_runs, key=lambda path: path.name)
    run_dir = str(new_runs[-1]) if new_runs else None
    run_id = Path(run_dir).name if run_dir else None
    job_ids = _parse_job_ids(completed.stdout)
    for job_id in _read_submitted_job_ids(run_dir):
        if job_id not in job_ids:
            job_ids.append(job_id)
    result = dict(base_result)
    result.update(
        {
            "ok": completed.returncode == 0,
            "driver_ran": True,
            "submitted": bool(job_ids),
            "run_dir": run_dir,
            "run_id": run_id,
            "job_ids": job_ids,
            "log_path": str(log_path),
            "returncode": int(completed.returncode),
        }
    )
    if completed.returncode:
        result["error"] = "The Linux HPC driver failed; inspect log_path."
    _write_json_atomic(result_path, result)
    _write_json_atomic(
        state_path,
        {
            "schema": "ghost.hpc.stage-state.v1",
            "status": "complete" if completed.returncode == 0 else "failed",
            "finished_utc": _utc_now(),
            "submission_requested": bool(submit),
            "returncode": int(completed.returncode),
            "run_dir": run_dir,
            "job_ids": job_ids,
            "lease_owner_token": stage_lease.owner_token,
            "lease_generation": stage_lease.generation,
            "recovered_prior_running_state": recovered_running_state,
        },
    )
    return result


def _parse_geometry_argument(value: str) -> Dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("geometry must be ROLE=/path/to/file.geo")
    role, path = value.split("=", 1)
    if not role.strip() or not path.strip():
        raise argparse.ArgumentTypeError("geometry must be ROLE=/path/to/file.geo")
    return {"role": role.strip(), "path": path.strip()}


def _json_output(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a portable bundle.")
    create_parser.add_argument("--solver", required=True, choices=("2d", "bor"))
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument("--settings", required=True, help="JSON settings file.")
    create_parser.add_argument(
        "--geometry",
        action="append",
        required=True,
        type=_parse_geometry_argument,
        metavar="ROLE=PATH",
    )

    verify_parser = subparsers.add_parser("verify", help="Verify a portable bundle.")
    verify_parser.add_argument("bundle")

    stage_parser = subparsers.add_parser("stage", help="Stage a request on Linux.")
    stage_parser.add_argument("bundle")
    stage_parser.add_argument("--workspace-root", required=True)
    stage_parser.add_argument("--run-driver", action="store_true")
    stage_parser.add_argument("--submit", action="store_true")

    recover_parser = subparsers.add_parser(
        "recover", help="Read a stage result or recover its submitted job IDs."
    )
    recover_parser.add_argument("stage_directory")

    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
            request = create_portable_bundle(
                args.output,
                solver=args.solver,
                geometries=args.geometry,
                settings=settings,
            )
            payload: Dict[str, Any] = {
                "ok": True,
                "schema": request["schema"],
                "bundle_id": request["bundle_id"],
                "bundle_dir": str(Path(args.output).expanduser().resolve()),
                "solver": request["solver"],
            }
        elif args.command == "verify":
            request = verify_portable_bundle(args.bundle)
            payload = {
                "ok": True,
                "schema": request["schema"],
                "bundle_id": request["bundle_id"],
                "bundle_dir": str(Path(args.bundle).expanduser().resolve()),
                "solver": request["solver"],
                "n_geometries": len(request["geometries"]),
            }
        elif args.command == "stage":
            payload = stage_portable_bundle(
                args.bundle,
                args.workspace_root,
                run_driver=args.run_driver,
                submit=args.submit,
            )
        else:
            payload = recover_staged_bundle(args.stage_directory)
        _json_output(payload)
        return 0 if payload.get("ok", True) else 1
    except (ValueError, OSError) as exc:
        _json_output({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
