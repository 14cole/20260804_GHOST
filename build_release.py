"""Build a complete, traceable GRIM source release from reviewed source.

The release is intentionally a source-tree distribution: GRIM discovers the
authoritative GHOST and FREDDY implementations below ``tools`` at runtime.  A
build produces an extracted folder, a deterministic ZIP of that folder, an
internal payload checksum manifest, and an external manifest that also covers
the ZIP itself.

Normal builds require an exactly clean Git checkout and package only tracked
files.  A source tree exported without Git can instead be built from an
explicit file inventory.  The builder itself uses only the Python standard
library, but the acceptance gate verifies the installed GRIM dependencies and
runs the project's test suites before publication.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from typing import Iterable, Sequence

from GRIM_Revised_2.grim_diagnostics import (
    FREDDY_SENTINELS,
    GHOST_SENTINELS,
    GRIM_STARTUP_FILES,
    collect_diagnostics,
    startup_exit_code,
)


PRODUCT_NAME = "GRIM"
MANIFEST_NAME = "SHA256SUMS.txt"
BUILD_INFO_NAME = "BUILD-INFO.json"
CONSTRAINTS_PATH = "requirements/constraints-windows-py312.txt"
COPY_CHUNK_SIZE = 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SUPPORTED_RELEASE_PYTHON = (3, 12)
SUPPORTED_RELEASE_PLATFORMS = frozenset({"win32"})
SUPPORTED_RELEASE_MACHINES = frozenset({"amd64", "x86_64"})
NATIVE_POLICIES = frozenset({"ignore", "warn", "require"})

# Construct these tokens so the release builder does not itself contain a
# forbidden product/name string that it is required to detect in the payload.
FORBIDDEN_RELEASE_TERMS = tuple(
    "".join(parts)
    for parts in (
        ("clau", "de"),
        ("em", "ery"),
        ("co", "dex"),
        ("m", "ac"),
        ("ma", "cos"),
        ("os", "x"),
        ("dar", "win"),
        ("ap", "ple"),
    )
)

TEXT_SUFFIXES = frozenset(
    {
        ".bat", ".c", ".cc", ".cfg", ".cmake", ".conf", ".cpp", ".csv",
        ".f", ".f03", ".f08", ".f77", ".f90", ".f95", ".geo", ".h",
        ".hpp", ".ini", ".ipynb", ".js", ".json", ".m", ".md", ".mk",
        ".ps1", ".py", ".pyi", ".pyw", ".rst", ".sbatch", ".sh",
        ".slurm", ".svg", ".toml", ".tsv", ".txt", ".xml", ".yaml",
        ".yml",
    }
)
TEXT_FILE_NAMES = frozenset(
    {".editorconfig", ".gitattributes", ".gitignore", "license", "makefile", "readme"}
)

# These sentinels make an incomplete manual copy fail before any release
# artifact is created.  They cover the host, assets, embedded tools, and the
# standalone Windows launch paths.
REQUIRED_FILES = (
    "pyproject.toml",
    CONSTRAINTS_PATH,
    "requirements/windows-py312.txt",
    "requirements/README.md",
    "requirements/wheelhouse_manifest.py",
    "requirements/test_wheelhouse_manifest.py",
    "README.md",
    "build_release.py",
    "clean_utf8.py",
    "Build_GRIM_Release.bat",
    "Launch_GRIM_GUI.bat",
    "Launch_GRIM_Diagnostics.bat",
    "Launch_PowerPoint_Image_Imprinter.bat",
    "GRIM_Revised_2/GRIM.png",
    "GRIM_Revised_2/ppt_image_imprinter.py",
    "GRIM_Revised_2/grim_csv_schema.py",
    "GRIM_Revised_2/ptm_io.py",
    "GRIM_Revised_2/read_ss.py",
    "GRIM_Revised_2/templates/GRIM_Report_Template.pptx",
    *(
        f"GRIM_Revised_2/{Path(relative).as_posix()}"
        for relative in GRIM_STARTUP_FILES
    ),
    "tools/GHOST/Launch_GHOST_GUI.bat",
    *(
        f"tools/GHOST/Backend/{Path(relative).as_posix()}"
        for relative in GHOST_SENTINELS
    ),
    "tools/GHOST/Backend/build_bor_stream_kernel.py",
    "tools/GHOST/Backend/create_feature_manifest.py",
    "tools/GHOST/Backend/grim_compat.py",
    "tools/GHOST/Backend/grim_naming.py",
    "tools/GHOST/Backend/import_3d_reference.py",
    "tools/GHOST/Backend/mesh_quality.py",
    "tools/GHOST/Backend/place_features.py",
    "tools/GHOST/Backend/run_local_bor.py",
    "tools/GHOST/Backend/run_local_monostatic.py",
    "tools/GHOST/Backend/validate_feature_reconstruction.py",
    "tools/GHOST/tests/test_assembly_workload.py",
    "tools/GHOST/CEM_Tools/README.md",
    "tools/GHOST/CEM_Tools/pyproject.toml",
    "tools/GHOST/CEM_Tools/requirements.txt",
    "tools/GHOST/CEM_Tools/run_gui.py",
    *(
        f"tools/GHOST/CEM_Tools/cem_tools/{name}.py"
        for name in (
            "__init__",
            "__main__",
            "cli",
            "errors",
            "grim_bridge",
            "grim_native",
            "gui",
            "naming",
            "operations",
            "registry",
            "solver_pairing",
        )
    ),
    "tools/GHOST/point_features_template.csv",
    "tools/GHOST/line_features_template.csv",
    "tools/FREDDY/Launch_FREDDY_GUI.bat",
    "tools/FREDDY/impedance_gui.py",
    *(
        f"tools/FREDDY/{Path(relative).as_posix()}"
        for relative in FREDDY_SENTINELS
    ),
    "tools/FREDDY/materials/air_reference.csv",
)

# An explicit-inventory build is used only for a reviewed export without Git.
# These globs bind that allowlist to every runtime Python module and acceptance
# test that is actually present in the export, rather than merely testing the
# fuller source tree while silently omitting a lazy module or test from the
# published source distribution.  Static REQUIRED_FILES still detects absence
# of the core contract files themselves.
REQUIRED_INVENTORY_GLOBS = (
    "GRIM_Revised_2/*.py",
    "GRIM_Revised_2/plot_modes/*.py",
    "GRIM_Revised_2/templates/*",
    "requirements/test*.py",
    "tools/GHOST/Backend/*.py",
    "tools/GHOST/tests/test*.py",
    "tools/GHOST/CEM_Tools/cem_tools/*.py",
    "tools/GHOST/CEM_Tools/tests/test*.py",
    "tools/FREDDY/ibc/*.py",
    "tools/FREDDY/tests/test*.py",
    "tools/FREDDY/materials/*.csv",
)

EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        ".cache",
        ".agents",
        ".idea",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".vscode",
        "htmlcov",
        "build",
        "dist",
    }
)
EXCLUDED_FILE_NAMES = frozenset(
    {
        ".git",
        ".coverage",
        ".ds_store",
        "coverage.xml",
        "thumbs.db",
    }
)
EXCLUDED_FILE_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".tmp",
    ".temp",
    ".swp",
    ".swo",
    ".bak",
    ".orig",
)


class ReleaseBuildError(RuntimeError):
    """A release could not be built safely."""


@dataclass(frozen=True)
class FileRecord:
    """A copied payload file and its digest."""

    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseArtifacts:
    """Paths and summary information returned by :func:`build_release`."""

    release_name: str
    version: str
    release_directory: Path
    archive_path: Path
    external_manifest_path: Path
    archive_sha256: str
    payload_file_count: int
    build_id: str
    source_revision: str
    source_tag: str


@dataclass(frozen=True)
class SourceInventory:
    """Reviewed source identity and the exact files eligible for packaging."""

    kind: str
    revision: str
    tag: str
    timestamp: str
    relative_files: tuple[Path, ...]


@dataclass(frozen=True)
class AcceptanceReport:
    """Deterministic summary of the release checks run before publication."""

    tests: tuple[str, ...]
    diagnostics: str
    utf8: str
    forbidden_terms: str
    dependency_lock: str
    native_policy: str
    native_status: str


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_version(version: str) -> str:
    value = version.strip()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._+-]*[A-Za-z0-9])?", value):
        raise ReleaseBuildError(
            "Release version must contain only letters, numbers, '.', '_', '+', "
            "or '-', and must start and end with a letter or number."
        )
    return value


def read_project_version(source_root: Path) -> str:
    """Read ``project.version`` while retaining Python 3.10 compatibility."""

    pyproject_path = source_root / "pyproject.toml"
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - exercised only on Python 3.10
        tomllib = None

    if tomllib is not None:
        try:
            with pyproject_path.open("rb") as stream:
                value = tomllib.load(stream)["project"]["version"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ReleaseBuildError(
                f"Cannot read [project].version from {pyproject_path}: {exc}"
            ) from exc
        return _safe_version(str(value))

    # ``tomllib`` was added after Python 3.10.  This deliberately narrow
    # fallback understands only the scalar version key inside [project].
    in_project = False
    try:
        lines = pyproject_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - platform-specific I/O failure
        raise ReleaseBuildError(f"Cannot read {pyproject_path}: {exc}") from exc
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        match = re.fullmatch(r"version\s*=\s*(['\"])([^'\"]+)\1\s*(?:#.*)?", stripped)
        if match:
            return _safe_version(match.group(2))
    raise ReleaseBuildError(f"Cannot find [project].version in {pyproject_path}")


def validate_source_tree(source_root: Path) -> None:
    """Fail closed unless the complete integrated source tree is present."""

    if not source_root.is_dir():
        raise ReleaseBuildError(f"Source root does not exist: {source_root}")

    missing: list[str] = []
    unsafe: list[str] = []
    for relative_name in REQUIRED_FILES:
        candidate = source_root.joinpath(*PurePosixPath(relative_name).parts)
        if not candidate.is_file():
            missing.append(relative_name)
        elif candidate.is_symlink():
            unsafe.append(relative_name)

    if missing:
        listed = "\n  - ".join(missing)
        raise ReleaseBuildError(
            "The source tree is incomplete; no release was created. "
            f"Missing required file(s):\n  - {listed}"
        )
    if unsafe:
        listed = "\n  - ".join(unsafe)
        raise ReleaseBuildError(
            "Required release files must be regular files inside the source "
            f"tree, not symbolic links:\n  - {listed}"
        )


def _run_git(source_root: Path, arguments: Sequence[str]) -> bytes:
    command = ("git", "-C", str(source_root), *arguments)
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseBuildError(
            "Git is required for a normal release build. If this is an exported "
            "source tree, supply --source-inventory with an explicitly reviewed "
            f"file list. Git error: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseBuildError(
            "Cannot inspect source control for the release. If this is an exported "
            "source tree, supply --source-inventory with an explicitly reviewed "
            f"file list. Git reported: {detail or 'unknown error'}"
        )
    return completed.stdout


def _normalise_inventory_paths(values: Iterable[str | Path]) -> tuple[Path, ...]:
    relative_files: list[Path] = []
    seen_portable_paths: dict[str, str] = {}
    for raw_value in values:
        raw_text = str(raw_value).strip().replace("\\", "/")
        if not raw_text or raw_text.startswith("#"):
            continue
        pure = PurePosixPath(raw_text)
        if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
            raise ReleaseBuildError(f"Unsafe source-inventory path: {raw_text!r}")
        posix_name = pure.as_posix()
        portable_key = unicodedata.normalize("NFC", posix_name).casefold()
        previous = seen_portable_paths.get(portable_key)
        if previous is not None:
            raise ReleaseBuildError(
                "Source-inventory paths that differ only by case or Unicode "
                f"normalization are not portable: {previous!r} and {posix_name!r}"
            )
        seen_portable_paths[portable_key] = posix_name
        relative_files.append(Path(*pure.parts))
    relative_files.sort(key=lambda path: path.as_posix())
    if not relative_files:
        raise ReleaseBuildError("The reviewed source inventory contains no files.")
    return tuple(relative_files)


def read_source_inventory(path: Path | str) -> tuple[Path, ...]:
    """Read a newline-delimited, source-relative release allowlist."""

    inventory_path = Path(path).expanduser().resolve()
    try:
        text = inventory_path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ReleaseBuildError(f"Cannot read source inventory {inventory_path}: {exc}") from exc
    return _normalise_inventory_paths(text.splitlines())


def discover_source_inventory(
    source_root: Path,
    *,
    explicit_files: Iterable[str | Path] | None = None,
    expected_version: str | None = None,
) -> SourceInventory:
    """Resolve an exact, reviewed set of packageable source files."""

    if explicit_files is not None:
        git_metadata_root = next(
            (
                candidate
                for candidate in (source_root.resolve(), *source_root.resolve().parents)
                if (candidate / ".git").exists()
            ),
            None,
        )
        if git_metadata_root is not None:
            raise ReleaseBuildError(
                "--source-inventory is only for a reviewed source export that "
                "is not inside a Git worktree. A Git-controlled tree must use "
                "the normal clean, version-tagged release path; the inventory "
                "path cannot bypass source-control acceptance. Git metadata was "
                f"found at {git_metadata_root}."
            )
        relative_files = _normalise_inventory_paths(explicit_files)
        digest = hashlib.sha256()
        for relative in relative_files:
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
        return SourceInventory(
            kind="explicit-inventory",
            revision=f"inventory-{digest.hexdigest()}",
            tag="",
            timestamp="",
            relative_files=relative_files,
        )

    top_level = Path(
        _run_git(source_root, ("rev-parse", "--show-toplevel"))
        .decode("utf-8", errors="strict")
        .strip()
    ).resolve()
    if top_level != source_root.resolve():
        raise ReleaseBuildError(
            f"Release source must be the Git worktree root ({top_level}), not {source_root}."
        )

    status = _run_git(
        source_root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=no"),
    )
    if status:
        entries = [
            item.decode("utf-8", errors="replace")
            for item in status.split(b"\0")
            if item
        ]
        shown = "\n  - ".join(entries[:20])
        remainder = max(0, len(entries) - 20)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise ReleaseBuildError(
            "The source checkout is not exactly clean. Releases package only "
            "reviewed Git-tracked content; commit, remove, or intentionally "
            f"ignore every change first:\n  - {shown}{suffix}"
        )

    relative_files = _normalise_inventory_paths(
        value.decode("utf-8", errors="strict")
        for value in _run_git(source_root, ("ls-files", "--cached", "-z")).split(b"\0")
        if value
    )
    revision = _run_git(source_root, ("rev-parse", "HEAD")).decode("ascii", errors="strict").strip()
    version = expected_version if expected_version is not None else read_project_version(source_root)
    tags = tuple(
        value.strip()
        for value in _run_git(source_root, ("tag", "--points-at", "HEAD", "--list"))
        .decode("utf-8", errors="strict")
        .splitlines()
        if value.strip()
    )
    accepted_tags = (f"v{version}", version)
    matching_tags = [tag for tag in accepted_tags if tag in tags]
    if not matching_tags:
        present = ", ".join(tags) if tags else "none"
        raise ReleaseBuildError(
            f"Release commit {revision[:12]} is not tagged for project version "
            f"{version}. Create tag v{version!s} (or {version!s}); tags "
            f"currently at HEAD: {present}."
        )
    release_tag = matching_tags[0]
    timestamp = _run_git(source_root, ("show", "-s", "--format=%cI", "HEAD")).decode(
        "ascii", errors="strict"
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ReleaseBuildError(f"Git returned an invalid source revision: {revision!r}")
    return SourceInventory(
        kind="git",
        revision=revision,
        tag=release_tag,
        timestamp=timestamp,
        relative_files=relative_files,
    )


def _excluded_file_name(name: str) -> bool:
    lowered = name.casefold()
    if lowered in EXCLUDED_FILE_NAMES:
        return True
    if lowered.endswith(EXCLUDED_FILE_SUFFIXES):
        return True
    if (
        lowered.endswith("~")
        or lowered.startswith("~$")
        or lowered.startswith(".~lock.")
        or lowered.startswith(".nfs")
    ):
        return True
    return False


def _excluded_directory_name(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered.startswith(".")
        or lowered in EXCLUDED_DIRECTORY_NAMES
        or lowered.endswith(".egg-info")
    )


def collect_payload_files(
    source_root: Path,
    output_root: Path,
    allowed_files: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Validate and return the reviewed packageable files in stable order."""

    source_root = source_root.resolve()
    output_root = output_root.resolve(strict=False)
    if source_root == output_root:
        raise ReleaseBuildError("The output directory cannot be the source root.")

    relative_files: list[Path] = []
    for relative in _normalise_inventory_paths(allowed_files):
        if relative.as_posix() in {MANIFEST_NAME, BUILD_INFO_NAME}:
            raise ReleaseBuildError(
                f"Source inventory uses reserved generated release path: {relative.as_posix()}"
            )
        if any(_excluded_directory_name(part) for part in relative.parts[:-1]):
            continue
        if _excluded_file_name(relative.name):
            continue
        candidate = source_root / relative
        if candidate.is_symlink():
            raise ReleaseBuildError(
                f"Symbolic-link file is not allowed in a release: {relative.as_posix()}"
            )
        try:
            mode = candidate.stat().st_mode
        except OSError as exc:
            raise ReleaseBuildError(
                f"Reviewed source file is missing or unreadable: {relative.as_posix()}: {exc}"
            ) from exc
        if not stat.S_ISREG(mode):
            raise ReleaseBuildError(
                f"Only regular files can be packaged: {relative.as_posix()}"
            )
        if _is_relative_to(candidate.resolve(strict=True), output_root):
            raise ReleaseBuildError(
                f"A reviewed source file points into the output directory: {relative.as_posix()}"
            )
        relative_files.append(relative)

    required = set(REQUIRED_FILES)
    for pattern in REQUIRED_INVENTORY_GLOBS:
        for candidate in source_root.glob(pattern):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(source_root)
            if any(_excluded_directory_name(part) for part in relative.parts[:-1]):
                continue
            if _excluded_file_name(relative.name):
                continue
            required.add(relative.as_posix())
    present = {path.as_posix() for path in relative_files}
    missing_required = sorted(required - present)
    if missing_required:
        listed = "\n  - ".join(missing_required)
        raise ReleaseBuildError(
            "Required files are absent from the reviewed source inventory:\n"
            f"  - {listed}"
        )
    return tuple(relative_files)


def _is_text_payload(relative: Path) -> bool:
    return (
        relative.suffix.casefold() in TEXT_SUFFIXES
        or relative.name.casefold() in TEXT_FILE_NAMES
    )


def _validate_utf8_payload(source_root: Path, relative_files: Sequence[Path]) -> None:
    failures: list[str] = []
    for relative in relative_files:
        if not _is_text_payload(relative):
            continue
        path = source_root / relative
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{relative.as_posix()}: {exc}")
            continue
        if "\0" in text:
            failures.append(
                f"{relative.as_posix()}: NUL bytes are not valid in a release text file"
            )
    if failures:
        listed = "\n  - ".join(failures[:30])
        suffix = f"\n  ... and {len(failures) - 30} more" if len(failures) > 30 else ""
        raise ReleaseBuildError(
            f"Strict UTF-8 release gate failed:\n  - {listed}{suffix}"
        )


def _validate_forbidden_terms(source_root: Path, relative_files: Sequence[Path]) -> None:
    findings: list[str] = []
    patterns = {
        term: re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        for term in FORBIDDEN_RELEASE_TERMS
    }

    def inspect_text(label: str, text: str) -> None:
        for term, pattern in patterns.items():
            match = pattern.search(text)
            if match is not None:
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{label}:{line}: forbidden term {term!r}")

    for relative in relative_files:
        if relative.suffix.casefold() == ".command":
            findings.append(f"{relative.as_posix()}: unsupported platform launcher")
            continue
        path = source_root / relative
        if _is_text_payload(relative):
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeError):
                continue  # The UTF-8 gate reports this with the more useful error.
            inspect_text(relative.as_posix(), text)
            continue
        if relative.suffix.casefold() not in {".docx", ".pptx", ".xlsx"}:
            continue
        try:
            with zipfile.ZipFile(path) as package:
                members = sorted(
                    name
                    for name in package.namelist()
                    if name.casefold().endswith((".xml", ".rels"))
                )
                for member in members:
                    text = package.read(member).decode("utf-8", errors="strict")
                    inspect_text(f"{relative.as_posix()}!{member}", text)
        except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
            findings.append(
                f"{relative.as_posix()}: cannot inspect Office package metadata: {exc}"
            )
    if findings:
        listed = "\n  - ".join(findings[:30])
        suffix = f"\n  ... and {len(findings) - 30} more" if len(findings) > 30 else ""
        raise ReleaseBuildError(
            f"Forbidden-term release gate failed:\n  - {listed}{suffix}"
        )


def _normalise_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _declared_project_dependencies(
    source_root: Path,
) -> tuple[set[str], set[str], set[str]]:
    try:
        import tomllib

        with (source_root / "pyproject.toml").open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ReleaseBuildError(f"Cannot inspect pyproject dependency metadata: {exc}") from exc

    def names(requirements: Iterable[object]) -> set[str]:
        result: set[str] = set()
        for raw in requirements:
            match = re.match(r"\s*([A-Za-z0-9_.-]+)", str(raw))
            if match is None:
                raise ReleaseBuildError(f"Cannot parse project dependency {raw!r}.")
            result.add(_normalise_distribution_name(match.group(1)))
        return result

    project = document.get("project", {})
    if not isinstance(project, dict):
        raise ReleaseBuildError("pyproject.toml [project] must be a table.")
    required = names(project.get("dependencies", ()))
    optional: set[str] = set()
    optional_groups = project.get("optional-dependencies", {})
    if not isinstance(optional_groups, dict):
        raise ReleaseBuildError("pyproject optional-dependencies must be a table.")
    for requirements in optional_groups.values():
        optional.update(names(requirements))
    build_system = document.get("build-system", {})
    if not isinstance(build_system, dict):
        raise ReleaseBuildError("pyproject build-system must be a table.")
    build = names(build_system.get("requires", ()))
    return required, optional, build


def _read_exact_constraints(source_root: Path) -> dict[str, str]:
    path = source_root.joinpath(*PurePosixPath(CONSTRAINTS_PATH).parts)
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReleaseBuildError(f"Cannot read dependency lock {path}: {exc}") from exc
    constraints: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        requirement = stripped.split(";", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", requirement)
        if match is None:
            raise ReleaseBuildError(
                f"Dependency lock entry must use an exact 'name==version' pin: "
                f"{CONSTRAINTS_PATH}:{line_number}: {line!r}"
            )
        name = _normalise_distribution_name(match.group(1))
        if name in constraints:
            raise ReleaseBuildError(
                f"Duplicate dependency lock entry for {name!r} at "
                f"{CONSTRAINTS_PATH}:{line_number}."
            )
        constraints[name] = match.group(2)
    required, optional, build = _declared_project_dependencies(source_root)
    missing = sorted((required | optional | build) - set(constraints))
    if missing:
        raise ReleaseBuildError(
            f"Dependency lock is missing required project package(s): {', '.join(missing)}"
        )
    return constraints


def _validate_dependency_lock(source_root: Path) -> None:
    if sys.platform not in SUPPORTED_RELEASE_PLATFORMS:
        raise ReleaseBuildError(
            "The verified release must be built and tested on 64-bit Windows; "
            f"this builder is running on {sys.platform!r}."
        )
    machine = platform.machine().strip().casefold()
    pointer_bits = struct.calcsize("P") * 8
    if machine not in SUPPORTED_RELEASE_MACHINES or pointer_bits != 64:
        raise ReleaseBuildError(
            "The verified release must be built and tested on 64-bit x86 Windows; "
            f"platform.machine() reported {platform.machine()!r} and the Python "
            f"process is {pointer_bits}-bit."
        )
    if sys.version_info[:2] != SUPPORTED_RELEASE_PYTHON:
        expected = ".".join(map(str, SUPPORTED_RELEASE_PYTHON))
        raise ReleaseBuildError(
            f"The verified Windows release stack requires Python {expected}; "
            f"this builder is running Python {sys.version_info.major}.{sys.version_info.minor}."
        )
    constraints = _read_exact_constraints(source_root)
    mismatches: list[str] = []
    for name, expected in constraints.items():
        try:
            installed = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            mismatches.append(f"{name}: required {expected}, not installed")
            continue
        if installed != expected:
            mismatches.append(f"{name}: required {expected}, installed {installed}")
    if mismatches:
        raise ReleaseBuildError(
            "Installed release dependencies do not match the reviewed lock:\n  - "
            + "\n  - ".join(mismatches)
        )


def _run_test_suite(name: str, cwd: Path, arguments: Sequence[str]) -> None:
    print(f"Release gate: running {name} ...", flush=True)
    try:
        completed = subprocess.run(
            (sys.executable, *arguments),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60 * 60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseBuildError(f"Cannot run {name}: {exc}") from exc
    if completed.returncode != 0:
        tail = completed.stdout[-12000:].strip()
        raise ReleaseBuildError(
            f"{name} failed with exit code {completed.returncode}. Last output:\n{tail}"
        )
    print(f"Release gate: {name} passed.", flush=True)


def run_acceptance_gates(
    source_root: Path,
    relative_files: Sequence[Path],
    *,
    native_policy: str,
) -> AcceptanceReport:
    """Run all release-blocking source, dependency, diagnostic, and test checks."""

    if native_policy not in NATIVE_POLICIES:
        raise ReleaseBuildError(
            f"Unknown native acceleration policy {native_policy!r}; expected one of "
            f"{', '.join(sorted(NATIVE_POLICIES))}."
        )

    print("Release gate: checking strict UTF-8 ...", flush=True)
    _validate_utf8_payload(source_root, relative_files)
    print("Release gate: checking prohibited release terms ...", flush=True)
    _validate_forbidden_terms(source_root, relative_files)
    print("Release gate: checking the exact Windows dependency lock ...", flush=True)
    _validate_dependency_lock(source_root)

    print("Release gate: running startup diagnostics ...", flush=True)
    diagnostic_results = collect_diagnostics(source_root)
    if startup_exit_code(diagnostic_results):
        blockers = [
            f"{result.name}: {result.summary}"
            for result in diagnostic_results
            if result.blocks_startup
        ]
        raise ReleaseBuildError(
            "Startup diagnostics failed:\n  - " + "\n  - ".join(blockers)
        )
    expected_native_keys = {"native_bor"}
    native_by_key = {
        result.key: result
        for result in diagnostic_results
        if result.key in expected_native_keys
    }
    native_parts = [
        f"{key}={native_by_key[key].status} ({native_by_key[key].summary})"
        if key in native_by_key
        else f"{key}=MISSING (diagnostic result absent)"
        for key in sorted(expected_native_keys)
    ]
    native_summary = "; ".join(native_parts)
    native_ready = (
        set(native_by_key) == expected_native_keys
        and all(result.status == "PASS" for result in native_by_key.values())
    )
    if native_policy == "require" and not native_ready:
        raise ReleaseBuildError(
            "Native acceleration is required by release policy, but diagnostics "
            f"did not pass every native check: {native_summary}"
        )
    if native_policy == "warn" and not native_ready:
        print(f"Release gate warning: {native_summary}", flush=True)

    suites = (
        (
            "UTF-8 cleaner tests",
            source_root,
            ("-W", "error", "-m", "unittest", "-v", "test_clean_utf8.py"),
        ),
        (
            "offline wheelhouse tests",
            source_root / "requirements",
            ("-m", "unittest", "discover", "-s", ".", "-p", "test*.py", "-v"),
        ),
        (
            "GRIM tests",
            source_root,
            ("-m", "unittest", "discover", "-s", "GRIM_Revised_2", "-p", "test*.py", "-v"),
        ),
        (
            "GHOST tests",
            source_root / "tools" / "GHOST",
            ("-m", "unittest", "discover", "-s", "tests", "-p", "test*.py", "-v"),
        ),
        (
            "GHOST CEM tools tests",
            source_root / "tools" / "GHOST" / "CEM_Tools",
            ("-m", "unittest", "discover", "-s", "tests", "-p", "test*.py", "-v"),
        ),
        (
            "GHOST HPC scheduling integration",
            source_root / "tools" / "GHOST",
            ("tests/test_hpc_scheduling.py",),
        ),
        (
            "GHOST local-driver integration",
            source_root / "tools" / "GHOST",
            ("tests/test_local_drivers.py",),
        ),
        (
            "GHOST ASCII-transfer compatibility",
            source_root / "tools" / "GHOST",
            ("tests/test_source_is_ascii.py",),
        ),
        (
            "FREDDY tests",
            source_root / "tools" / "FREDDY",
            ("-m", "unittest", "discover", "-s", "tests", "-p", "test*.py", "-v"),
        ),
    )
    for name, cwd, command in suites:
        _run_test_suite(name, cwd, command)

    return AcceptanceReport(
        tests=tuple(name for name, _cwd, _command in suites),
        diagnostics="pass",
        utf8="pass",
        forbidden_terms="pass",
        dependency_lock="pass",
        native_policy=native_policy,
        native_status=native_summary,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(COPY_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _snapshot_source_files(
    source_root: Path,
    relative_files: Sequence[Path],
) -> tuple[FileRecord, ...]:
    records: list[FileRecord] = []
    for relative in relative_files:
        try:
            sha256, size = _hash_file(source_root / relative)
        except OSError as exc:
            raise ReleaseBuildError(
                f"Cannot snapshot reviewed source file {relative.as_posix()}: {exc}"
            ) from exc
        records.append(FileRecord(relative.as_posix(), size, sha256))
    return tuple(records)


def _copy_payload(
    source_root: Path, relative_files: Iterable[Path], destination_root: Path
) -> tuple[FileRecord, ...]:
    records: list[FileRecord] = []
    resolved_source_root = source_root.resolve()
    for relative in relative_files:
        source = source_root / relative
        try:
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise ReleaseBuildError(f"Source file disappeared during build: {source}") from exc
        if source.is_symlink() or not _is_relative_to(resolved_source, resolved_source_root):
            raise ReleaseBuildError(
                f"Source file escaped the release root during build: {relative.as_posix()}"
            )

        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                while True:
                    chunk = input_stream.read(COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    output_stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            shutil.copystat(source, destination, follow_symlinks=False)
        except OSError as exc:
            raise ReleaseBuildError(
                f"Cannot copy {relative.as_posix()} into the release: {exc}"
            ) from exc
        records.append(FileRecord(relative.as_posix(), size, digest.hexdigest()))
    return tuple(records)


def _payload_manifest_text(release_name: str, records: Sequence[FileRecord]) -> str:
    lines = [
        f"# SHA-256 payload manifest for {release_name}",
        f"# Paths are relative to {release_name} and use '/' separators.",
    ]
    lines.extend(f"{record.sha256}  {record.relative_path}" for record in records)
    return "\n".join(lines) + "\n"


def _source_tree_digest(records: Sequence[FileRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_path):
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record.size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_info_text(
    *,
    release_name: str,
    version: str,
    source: SourceInventory,
    source_records: Sequence[FileRecord],
    acceptance: AcceptanceReport,
    constraints_sha256: str,
) -> tuple[str, str]:
    tree_sha256 = _source_tree_digest(source_records)
    identity = source.revision if source.kind == "git" else tree_sha256
    build_id = f"{PRODUCT_NAME.lower()}-{version}-{identity[:12]}"
    payload = {
        "acceptance": {
            "dependency_lock": acceptance.dependency_lock,
            "diagnostics": acceptance.diagnostics,
            "forbidden_terms": acceptance.forbidden_terms,
            "native_policy": acceptance.native_policy,
            "native_status": acceptance.native_status,
            "test_suites": list(acceptance.tests),
            "utf8": acceptance.utf8,
        },
        "build": {
            "builder_python": platform.python_version(),
            "constraints_path": CONSTRAINTS_PATH,
            "constraints_sha256": constraints_sha256,
            "id": build_id,
            "target": "windows-x86_64-python312",
        },
        "product": PRODUCT_NAME,
        "release_name": release_name,
        "schema_version": 1,
        "source": {
            "file_count": len(source_records),
            "inventory_kind": source.kind,
            "revision": source.revision,
            "tag": source.tag or None,
            "source_timestamp": source.timestamp or None,
            "tree_sha256": tree_sha256,
        },
        "version": version,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n", build_id


def _external_manifest_text(
    release_name: str,
    archive_name: str,
    archive_sha256: str,
    extracted_records: Sequence[FileRecord],
) -> str:
    lines = [
        f"# SHA-256 release manifest for {release_name}",
        "# The first entry verifies the ZIP; remaining paths verify an extracted folder.",
        f"{archive_sha256}  {archive_name}",
    ]
    lines.extend(
        f"{record.sha256}  {release_name}/{record.relative_path}"
        for record in extracted_records
    )
    return "\n".join(lines) + "\n"


def _write_text_file(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ReleaseBuildError(f"Cannot write {path}: {exc}") from exc


def _zip_member_mode(path: Path) -> int:
    if path.suffix.casefold() == ".sh":
        return 0o755
    return 0o644


def _create_deterministic_zip(release_root: Path, archive_path: Path) -> None:
    release_name = release_root.name
    members = sorted(
        (path for path in release_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(release_root).as_posix(),
    )
    try:
        with zipfile.ZipFile(
            archive_path,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for member in members:
                relative = member.relative_to(release_root)
                archive_name = PurePosixPath(release_name, *relative.parts).as_posix()
                info = zipfile.ZipInfo(archive_name, date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | _zip_member_mode(member)) << 16
                with member.open("rb") as source, archive.open(
                    info, mode="w", force_zip64=True
                ) as destination:
                    shutil.copyfileobj(source, destination, length=COPY_CHUNK_SIZE)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseBuildError(f"Cannot create release ZIP {archive_path}: {exc}") from exc


def _publish_file_atomic(source: Path, destination: Path) -> None:
    """Expose one complete staged file without an overwrite or partial name."""

    try:
        # The staging directory is deliberately created below output_root, so a
        # hard link is same-volume.  Link creation is atomic and fails if the
        # public name already exists; unlike copying to the final name, a file
        # watcher can never observe a truncated ZIP or manifest.
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        raise ReleaseBuildError(f"Cannot publish {destination}: {exc}") from exc


def _publish_release_directory(staged: Path, destination: Path) -> None:
    """Atomically expose a complete staged directory on the Windows target."""

    try:
        # Both paths share output_root.  On the supported Windows build target,
        # os.rename is atomic and refuses an existing destination, preserving
        # the builder's no-overwrite contract while avoiding a visible partial
        # extracted tree.
        os.rename(staged, destination)
    except OSError as exc:
        raise ReleaseBuildError(f"Cannot publish release folder {destination}: {exc}") from exc


def build_release(
    source_root: Path | str,
    output_directory: Path | str | None = None,
    *,
    version: str | None = None,
    source_inventory: Iterable[str | Path] | None = None,
    native_policy: str = "warn",
    run_acceptance: bool = True,
) -> ReleaseArtifacts:
    """Build and publish a complete release folder, ZIP, and checksum manifest.

    Existing release artifacts are never overwritten.  ``ReleaseBuildError``
    is raised before output creation for an incomplete source tree.
    """

    source_root = Path(source_root).expanduser().resolve()
    validate_source_tree(source_root)
    project_version = read_project_version(source_root)
    if version is not None and _safe_version(version) != project_version:
        raise ReleaseBuildError(
            f"Release version override {version!r} disagrees with pyproject.toml "
            f"version {project_version!r}. Update the reviewed project version first."
        )
    release_version = project_version
    release_name = f"{PRODUCT_NAME}-{release_version}"
    output_root = (
        Path(output_directory).expanduser().resolve(strict=False)
        if output_directory is not None
        else source_root / "dist"
    )

    reviewed_source = discover_source_inventory(
        source_root,
        explicit_files=source_inventory,
        expected_version=project_version,
    )
    relative_files = collect_payload_files(
        source_root,
        output_root,
        reviewed_source.relative_files,
    )
    if not relative_files:
        raise ReleaseBuildError("No payload files were found; no release was created.")
    explicit_snapshot = (
        _snapshot_source_files(source_root, relative_files)
        if reviewed_source.kind == "explicit-inventory"
        else None
    )

    if run_acceptance:
        acceptance = run_acceptance_gates(
            source_root,
            relative_files,
            native_policy=native_policy,
        )
    else:
        # Deliberately private-looking API behavior for isolated builder unit
        # tests. The CLI never exposes a switch that can bypass acceptance.
        acceptance = AcceptanceReport(
            tests=("not-run",),
            diagnostics="not-run",
            utf8="not-run",
            forbidden_terms="not-run",
            dependency_lock="not-run",
            native_policy=native_policy,
            native_status="not-run",
        )

    if reviewed_source.kind == "git":
        after_gates = discover_source_inventory(
            source_root,
            expected_version=project_version,
        )
        if (
            after_gates.revision != reviewed_source.revision
            or after_gates.tag != reviewed_source.tag
            or after_gates.relative_files != reviewed_source.relative_files
        ):
            raise ReleaseBuildError(
                "The reviewed Git source changed while release acceptance checks "
                "were running. Re-run the build from the now-stable checkout."
            )

    constraints_path = source_root.joinpath(*PurePosixPath(CONSTRAINTS_PATH).parts)
    constraints_sha256, _constraints_size = _hash_file(constraints_path)

    final_release_directory = output_root / release_name
    final_archive = output_root / f"{release_name}.zip"
    final_manifest = output_root / f"{release_name}-SHA256SUMS.txt"
    targets = (final_release_directory, final_archive, final_manifest)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        listed = "\n  - ".join(existing)
        raise ReleaseBuildError(
            "Release targets already exist and will not be overwritten:\n"
            f"  - {listed}\nMove or remove them, or choose another output directory."
        )

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReleaseBuildError(f"Cannot create output directory {output_root}: {exc}") from exc
    if not output_root.is_dir():
        raise ReleaseBuildError(f"Output path is not a directory: {output_root}")

    published_files: list[Path] = []
    published_directory = False
    try:
        with tempfile.TemporaryDirectory(prefix=".grim-release-", dir=output_root) as temp_name:
            temporary_root = Path(temp_name)
            staged_release = temporary_root / release_name
            staged_release.mkdir()
            source_records = _copy_payload(source_root, relative_files, staged_release)
            if explicit_snapshot is not None and source_records != explicit_snapshot:
                raise ReleaseBuildError(
                    "The reviewed inventory source changed after acceptance began. "
                    "No release was published; review and re-run the build."
                )
            if reviewed_source.kind == "git":
                after_copy = discover_source_inventory(
                    source_root,
                    expected_version=project_version,
                )
                if (
                    after_copy.revision != reviewed_source.revision
                    or after_copy.tag != reviewed_source.tag
                    or after_copy.relative_files != reviewed_source.relative_files
                ):
                    raise ReleaseBuildError(
                        "The reviewed Git source changed while files were being "
                        "copied. No release was published; re-run from a stable checkout."
                    )

            build_info_text, build_id = _build_info_text(
                release_name=release_name,
                version=release_version,
                source=reviewed_source,
                source_records=source_records,
                acceptance=acceptance,
                constraints_sha256=constraints_sha256,
            )
            build_info_path = staged_release / BUILD_INFO_NAME
            _write_text_file(build_info_path, build_info_text)
            build_info_sha256, build_info_size = _hash_file(build_info_path)
            records = tuple(
                sorted(
                    (
                        *source_records,
                        FileRecord(BUILD_INFO_NAME, build_info_size, build_info_sha256),
                    ),
                    key=lambda record: record.relative_path,
                )
            )

            internal_manifest = staged_release / MANIFEST_NAME
            _write_text_file(
                internal_manifest, _payload_manifest_text(release_name, records)
            )
            manifest_sha256, manifest_size = _hash_file(internal_manifest)
            extracted_records = tuple(
                sorted(
                    (*records, FileRecord(MANIFEST_NAME, manifest_size, manifest_sha256)),
                    key=lambda record: record.relative_path,
                )
            )

            staged_archive = temporary_root / f"{release_name}.zip"
            _create_deterministic_zip(staged_release, staged_archive)
            archive_sha256, _archive_size = _hash_file(staged_archive)

            staged_external_manifest = temporary_root / final_manifest.name
            _write_text_file(
                staged_external_manifest,
                _external_manifest_text(
                    release_name,
                    final_archive.name,
                    archive_sha256,
                    extracted_records,
                ),
            )

            # Publish each artifact atomically.  The checksum manifest is last
            # and therefore acts as the completion marker for consumers.  If a
            # later publication step fails, only names created by this
            # invocation are removed; pre-existing user files are never touched.
            _publish_release_directory(staged_release, final_release_directory)
            published_directory = True
            _publish_file_atomic(staged_archive, final_archive)
            published_files.append(final_archive)
            _publish_file_atomic(staged_external_manifest, final_manifest)
            published_files.append(final_manifest)
    except Exception:
        if published_directory:
            shutil.rmtree(final_release_directory, ignore_errors=True)
        for path in reversed(published_files):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise

    return ReleaseArtifacts(
        release_name=release_name,
        version=release_version,
        release_directory=final_release_directory,
        archive_path=final_archive,
        external_manifest_path=final_manifest,
        archive_sha256=archive_sha256,
        payload_file_count=len(records),
        build_id=build_id,
        source_revision=reviewed_source.revision,
        source_tag=reviewed_source.tag,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned, checksum-manifested GRIM/GHOST/FREDDY source "
            "folder and ZIP from a clean Git checkout or reviewed inventory."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="complete integrated source root (default: folder containing this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: SOURCE/dist)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "consistency assertion; when supplied it must equal [project].version "
            "in pyproject.toml"
        ),
    )
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=None,
        help=(
            "reviewed newline-delimited source-relative file allowlist for an "
            "exported source tree without Git"
        ),
    )
    parser.add_argument(
        "--native-policy",
        choices=sorted(NATIVE_POLICIES),
        default="warn",
        help=(
            "handling when Windows native GHOST BoR acceleration is unavailable "
            "(default: warn and record in BUILD-INFO.json)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        inventory = (
            read_source_inventory(args.source_inventory)
            if args.source_inventory is not None
            else None
        )
        result = build_release(
            args.source,
            args.output,
            version=args.version,
            source_inventory=inventory,
            native_policy=args.native_policy,
        )
    except ReleaseBuildError as exc:
        print(f"Release build failed: {exc}", file=sys.stderr)
        return 2

    print(f"Built {result.release_name} ({result.payload_file_count} payload files)")
    print(f"Folder:   {result.release_directory}")
    print(f"ZIP:      {result.archive_path}")
    print(f"Manifest: {result.external_manifest_path}")
    print(f"ZIP SHA-256: {result.archive_sha256}")
    print(f"Build ID: {result.build_id}")
    print(f"Source revision: {result.source_revision}")
    if result.source_tag:
        print(f"Source tag: {result.source_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
