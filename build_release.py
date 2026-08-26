"""Build a complete, portable GRIM source release without using Git.

The release is intentionally a source-tree distribution: GRIM discovers the
authoritative GHOST and FREDDY implementations below ``tools`` at runtime.  A
build produces an extracted folder, a deterministic ZIP of that folder, an
internal payload checksum manifest, and an external manifest that also covers
the ZIP itself.

Only the Python standard library is used so this script can run before GRIM's
scientific and GUI dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
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
)


PRODUCT_NAME = "GRIM"
MANIFEST_NAME = "SHA256SUMS.txt"
COPY_CHUNK_SIZE = 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

# These sentinels make an incomplete manual copy fail before any release
# artifact is created.  They cover the host, assets, embedded tools, and the
# standalone launch paths on both supported desktop platforms.
REQUIRED_FILES = (
    "pyproject.toml",
    "README.md",
    "build_release.py",
    "clean_utf8.py",
    "Build_GRIM_Release.bat",
    "Launch_GRIM_GUI.bat",
    "Launch_GRIM_GUI.command",
    "Launch_GRIM_Diagnostics.bat",
    "Launch_PowerPoint_Image_Imprinter.bat",
    "GRIM_Revised_2/GRIM.png",
    "GRIM_Revised_2/ppt_image_imprinter.py",
    "GRIM_Revised_2/templates/GRIM_Report_Template.pptx",
    *(
        f"GRIM_Revised_2/{Path(relative).as_posix()}"
        for relative in GRIM_STARTUP_FILES
    ),
    "tools/GHOST/Launch_GHOST_GUI.bat",
    "tools/GHOST/Launch_GHOST_GUI.command",
    *(
        f"tools/GHOST/Backend/{Path(relative).as_posix()}"
        for relative in GHOST_SENTINELS
    ),
    "tools/GHOST/point_features_template.csv",
    "tools/GHOST/line_features_template.csv",
    "tools/FREDDY/Launch_FREDDY_GUI.bat",
    "tools/FREDDY/Launch_FREDDY_GUI.command",
    "tools/FREDDY/impedance_gui.py",
    *(
        f"tools/FREDDY/{Path(relative).as_posix()}"
        for relative in FREDDY_SENTINELS
    ),
    "tools/FREDDY/materials/air_reference.csv",
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
        ".codex",
        ".codex-tmp",
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
    return lowered in EXCLUDED_DIRECTORY_NAMES or lowered.endswith(".egg-info")


def collect_payload_files(source_root: Path, output_root: Path) -> tuple[Path, ...]:
    """Return included source-relative files in deterministic POSIX order."""

    source_root = source_root.resolve()
    output_root = output_root.resolve(strict=False)
    if source_root == output_root:
        raise ReleaseBuildError("The output directory cannot be the source root.")

    relative_files: list[Path] = []
    seen_portable_paths: dict[str, str] = {}

    for directory, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names, key=lambda item: item.casefold()):
            candidate = directory_path / name
            if _excluded_directory_name(name):
                continue
            if candidate.resolve(strict=False) == output_root:
                continue
            if candidate.is_symlink():
                relative = candidate.relative_to(source_root).as_posix()
                raise ReleaseBuildError(
                    f"Symbolic-link directory is not allowed in a release: {relative}"
                )
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names, key=lambda item: item.casefold()):
            if _excluded_file_name(name):
                continue
            candidate = directory_path / name
            relative = candidate.relative_to(source_root)
            if candidate.is_symlink():
                raise ReleaseBuildError(
                    f"Symbolic-link file is not allowed in a release: {relative.as_posix()}"
                )
            try:
                mode = candidate.stat().st_mode
            except OSError as exc:
                raise ReleaseBuildError(f"Cannot inspect {candidate}: {exc}") from exc
            if not stat.S_ISREG(mode):
                raise ReleaseBuildError(
                    f"Only regular files can be packaged: {relative.as_posix()}"
                )

            posix_name = relative.as_posix()
            portable_key = unicodedata.normalize("NFC", posix_name).casefold()
            previous = seen_portable_paths.get(portable_key)
            if previous is not None:
                raise ReleaseBuildError(
                    "Paths that differ only by case or Unicode normalization are "
                    f"not portable: {previous!r} and {posix_name!r}"
                )
            seen_portable_paths[portable_key] = posix_name
            relative_files.append(relative)

    relative_files.sort(key=lambda path: path.as_posix())
    return tuple(relative_files)


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
    if path.suffix.casefold() in {".command", ".sh"}:
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


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    created = False
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            created = True
            shutil.copyfileobj(input_stream, output_stream, length=COPY_CHUNK_SIZE)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except OSError as exc:
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise ReleaseBuildError(f"Cannot publish {destination}: {exc}") from exc


def _publish_release_directory(staged: Path, destination: Path) -> None:
    try:
        destination.mkdir()
    except OSError as exc:
        raise ReleaseBuildError(f"Cannot reserve release folder {destination}: {exc}") from exc
    try:
        for source in sorted(staged.rglob("*"), key=lambda path: path.as_posix()):
            relative = source.relative_to(staged)
            target = destination / relative
            if source.is_dir():
                target.mkdir(exist_ok=False)
            else:
                _copy_file_exclusive(source, target)
                shutil.copystat(source, target, follow_symlinks=False)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def build_release(
    source_root: Path | str,
    output_directory: Path | str | None = None,
    *,
    version: str | None = None,
) -> ReleaseArtifacts:
    """Build and publish a complete release folder, ZIP, and checksum manifest.

    Existing release artifacts are never overwritten.  ``ReleaseBuildError``
    is raised before output creation for an incomplete source tree.
    """

    source_root = Path(source_root).expanduser().resolve()
    validate_source_tree(source_root)
    release_version = _safe_version(version) if version is not None else read_project_version(source_root)
    release_name = f"{PRODUCT_NAME}-{release_version}"
    output_root = (
        Path(output_directory).expanduser().resolve(strict=False)
        if output_directory is not None
        else source_root / "dist"
    )

    relative_files = collect_payload_files(source_root, output_root)
    if not relative_files:
        raise ReleaseBuildError("No payload files were found; no release was created.")

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
            records = _copy_payload(source_root, relative_files, staged_release)

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

            # Each destination is opened or created exclusively.  If any later
            # publication fails, only artifacts created by this invocation are
            # removed; pre-existing user files are never touched.
            _copy_file_exclusive(staged_archive, final_archive)
            published_files.append(final_archive)
            _copy_file_exclusive(staged_external_manifest, final_manifest)
            published_files.append(final_manifest)
            _publish_release_directory(staged_release, final_release_directory)
            published_directory = True
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
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned, checksum-manifested GRIM/GHOST/FREDDY source "
            "folder and ZIP without using Git."
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
        help="release version override (default: [project].version in pyproject.toml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        result = build_release(args.source, args.output, version=args.version)
    except ReleaseBuildError as exc:
        print(f"Release build failed: {exc}", file=sys.stderr)
        return 2

    print(f"Built {result.release_name} ({result.payload_file_count} payload files)")
    print(f"Folder:   {result.release_directory}")
    print(f"ZIP:      {result.archive_path}")
    print(f"Manifest: {result.external_manifest_path}")
    print(f"ZIP SHA-256: {result.archive_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
