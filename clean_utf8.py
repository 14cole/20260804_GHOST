#!/usr/bin/env python3
"""Repair copied GRIM text files by converting them to strict UTF-8.

The script is intentionally standard-library-only and ASCII source so it can
be copied to either Windows or Linux even when the destination tree already
contains files damaged by a text-mode transfer.

Run without ``--apply`` for a preview.  Applying a repair creates a ZIP with
the exact original bytes before any file is atomically replaced.
"""

from __future__ import annotations

import argparse
import codecs
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Iterable, Sequence
import zipfile


CHUNK_SIZE = 1024 * 1024

# Keep this an allowlist.  Several GRIM formats, including .grim, .ptm, .ss,
# .pio, and .cmplx_di, are binary even though a UTF-8 exception elsewhere can
# make an encoding repair seem tempting.
TEXT_SUFFIXES = frozenset(
    {
        ".asy",
        ".bat",
        ".c",
        ".cc",
        ".cfg",
        ".cmake",
        ".conf",
        ".cpp",
        ".cst_data",
        ".csv",
        ".f",
        ".f03",
        ".f08",
        ".f77",
        ".f90",
        ".f95",
        ".fish",
        ".geo",
        ".h",
        ".hpp",
        ".ini",
        ".ipynb",
        ".js",
        ".json",
        ".log",
        ".m",
        ".md",
        ".mk",
        ".out",
        ".pbs",
        ".ps1",
        ".py",
        ".pyi",
        ".pyw",
        ".rst",
        ".sbatch",
        ".sh",
        ".slurm",
        ".svg",
        ".toml",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
        ".zsh",
    }
)

TEXT_FILE_NAMES = frozenset(
    {
        ".dockerignore",
        ".editorconfig",
        ".gitattributes",
        ".gitignore",
        ".gitmodules",
        "copying",
        "dockerfile",
        "license",
        "makefile",
        "readme",
    }
)

SKIP_DIRECTORY_NAMES = frozenset(
    {
        ".agents",
        ".cache",
        ".git",
        ".hg",
        ".hypothesis",
        ".idea",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "dist",
        "env",
        "htmlcov",
        "node_modules",
        "venv",
    }
)

PYTHON_SUFFIXES = frozenset({".py", ".pyi", ".pyw"})
XML_SUFFIXES = frozenset({".svg", ".xml"})

PYTHON_CODING_RE = re.compile(
    r"(?i)^([ \t\f]*#.*?coding[ \t]*[:=][ \t]*)([-_.a-z0-9]+)"
)
XML_ENCODING_RE = re.compile(
    r"(?i)(<\?xml[^>]*\bencoding\s*=\s*['\"])([^'\"]+)(['\"])"
)
UTF8_CP1252_ERROR_HANDLER = "grim_utf8_with_windows_1252"


def _decode_windows_1252_at_utf8_error(
    error: UnicodeError,
) -> tuple[str, int]:
    """Preserve valid UTF-8 while decoding only invalid spans as cp1252."""

    if not isinstance(error, UnicodeDecodeError):
        raise error
    damaged = error.object[error.start : error.end]
    try:
        replacement = damaged.decode("cp1252", errors="strict")
    except UnicodeDecodeError:
        raise error
    return replacement, error.end


codecs.register_error(
    UTF8_CP1252_ERROR_HANDLER, _decode_windows_1252_at_utf8_error
)


@dataclass(frozen=True)
class Conversion:
    """One file that can be losslessly converted to UTF-8."""

    path: Path
    relative_path: Path
    scope_root: Path
    resolved_path: Path
    source_encoding: str
    source_errors: str
    display_encoding: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    sha256: str


@dataclass(frozen=True)
class ScanIssue:
    """A candidate that could not be inspected or safely converted."""

    relative_path: Path
    message: str


@dataclass(frozen=True)
class ScanResult:
    """Result of inspecting all recognized text files below a path."""

    root: Path
    base: Path
    candidate_count: int
    clean_count: int
    conversions: tuple[Conversion, ...]
    issues: tuple[ScanIssue, ...]


def _normalise_extra_suffix(value: str) -> str:
    suffix = value.strip().casefold()
    if not suffix:
        raise argparse.ArgumentTypeError("an extension cannot be empty")
    if not suffix.startswith("."):
        suffix = "." + suffix
    if any(character in suffix for character in ("/", "\\", "*", "?")):
        raise argparse.ArgumentTypeError(
            "an extension must not contain a path separator or wildcard"
        )
    return suffix


def _is_candidate(path: Path, suffixes: frozenset[str]) -> bool:
    name = path.name.casefold()
    return name in TEXT_FILE_NAMES or path.suffix.casefold() in suffixes


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_link_like(path: Path) -> bool:
    """Return True for symlinks and, on modern Python, Windows junctions."""

    if path.is_symlink():
        return True
    if os.name == "nt":
        try:
            attributes = os.lstat(path).st_file_attributes
        except (AttributeError, OSError):
            attributes = 0
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _bad_control_character(text: str) -> str | None:
    for character in text:
        value = ord(character)
        if value < 32 and character not in "\t\n\r\f":
            return f"U+{value:04X}"
        if 0x7F <= value <= 0x9F:
            return f"U+{value:04X}"
    return None


def _validate_decoding(
    path: Path,
    encoding: str,
    *,
    errors: str = "strict",
) -> tuple[bool, str]:
    """Decode a file incrementally and reject controls typical of binary data."""

    decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                try:
                    text = decoder.decode(chunk, final=False)
                except UnicodeDecodeError as exc:
                    return False, str(exc)
                control = _bad_control_character(text)
                if control is not None:
                    return False, f"contains binary/control character {control}"
            try:
                tail = decoder.decode(b"", final=True)
            except UnicodeDecodeError as exc:
                return False, str(exc)
    except OSError as exc:
        return False, str(exc)

    control = _bad_control_character(tail)
    if control is not None:
        return False, f"contains binary/control character {control}"
    return True, ""


def _guess_bomless_utf16(sample: bytes) -> str | None:
    """Recognize ordinary ASCII-heavy UTF-16 without guessing arbitrary data."""

    pair_count = len(sample) // 2
    if pair_count < 2:
        return None
    paired = sample[: pair_count * 2]
    even_nuls = paired[0::2].count(0)
    odd_nuls = paired[1::2].count(0)
    even_ratio = even_nuls / pair_count
    odd_ratio = odd_nuls / pair_count
    if odd_ratio >= 0.30 and even_ratio <= 0.05:
        return "utf-16-le"
    if even_ratio >= 0.30 and odd_ratio <= 0.05:
        return "utf-16-be"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stat_matches_conversion(info: os.stat_result, item: Conversion) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_dev == item.device
        and info.st_ino == item.inode
        and info.st_size == item.size
        and info.st_mtime_ns == item.mtime_ns
    )


def _assert_source_unchanged(item: Conversion) -> os.stat_result:
    """Fail if a planned source moved or changed, including during hashing."""

    if _is_link_like(item.path):
        raise RuntimeError("source became a symbolic link or junction")
    try:
        resolved_before = item.path.resolve(strict=True)
        before = os.stat(item.path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"cannot re-check source: {exc}") from exc
    if (
        resolved_before != item.resolved_path
        or not _is_relative_to(resolved_before, item.scope_root)
        or not _stat_matches_conversion(before, item)
    ):
        raise RuntimeError("source changed after it was scanned; left untouched")

    try:
        digest = _sha256(item.path)
        after = os.stat(item.path, follow_symlinks=False)
        resolved_after = item.path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"cannot finish source re-check: {exc}") from exc
    if (
        _is_link_like(item.path)
        or resolved_after != resolved_before
        or not _stat_matches_conversion(after, item)
        or digest != item.sha256
    ):
        raise RuntimeError("source changed after it was scanned; left untouched")
    return after


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence for a completed rename on POSIX filesystems."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = -1
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
    except OSError:
        # Some network and virtual filesystems do not support directory fsync.
        pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _conversion_for(
    path: Path,
    relative_path: Path,
    scope_root: Path,
    source_encoding: str,
    display_encoding: str,
    *,
    source_errors: str = "strict",
) -> Conversion:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise OSError("path is no longer a regular file")
    resolved_path = path.resolve(strict=True)
    if not _is_relative_to(resolved_path, scope_root):
        raise OSError("resolved path escapes the requested scan folder")
    digest = _sha256(path)
    final_info = os.stat(path, follow_symlinks=False)
    final_resolved_path = path.resolve(strict=True)
    if (
        not stat.S_ISREG(final_info.st_mode)
        or final_info.st_dev != info.st_dev
        or final_info.st_ino != info.st_ino
        or final_info.st_size != info.st_size
        or final_info.st_mtime_ns != info.st_mtime_ns
        or final_resolved_path != resolved_path
    ):
        raise OSError("file changed while it was being fingerprinted")
    return Conversion(
        path=path,
        relative_path=relative_path,
        scope_root=scope_root,
        resolved_path=resolved_path,
        source_encoding=source_encoding,
        source_errors=source_errors,
        display_encoding=display_encoding,
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        device=info.st_dev,
        inode=info.st_ino,
        sha256=digest,
    )


def inspect_file(
    path: Path,
    relative_path: Path,
    scope_root: Path,
) -> Conversion | ScanIssue | None:
    """Return a conversion, an issue, or None when the file is clean UTF-8."""

    try:
        resolved_path = path.resolve(strict=True)
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        return ScanIssue(relative_path, f"cannot inspect file: {exc}")
    if (
        _is_link_like(path)
        or not stat.S_ISREG(info.st_mode)
        or not _is_relative_to(resolved_path, scope_root)
    ):
        return ScanIssue(
            relative_path,
            "file is not a regular in-scope path and will not be modified",
        )

    try:
        with path.open("rb") as stream:
            sample = stream.read(64 * 1024)
    except OSError as exc:
        return ScanIssue(relative_path, f"cannot read file: {exc}")

    # UTF-32 BOMs must be tested before UTF-16 because their prefixes overlap.
    bom_encodings = (
        (codecs.BOM_UTF32_LE, "utf-32", "UTF-32 LE with BOM"),
        (codecs.BOM_UTF32_BE, "utf-32", "UTF-32 BE with BOM"),
        (codecs.BOM_UTF16_LE, "utf-16", "UTF-16 LE with BOM"),
        (codecs.BOM_UTF16_BE, "utf-16", "UTF-16 BE with BOM"),
    )
    for bom, encoding, label in bom_encodings:
        if not sample.startswith(bom):
            continue
        valid, reason = _validate_decoding(path, encoding)
        if not valid:
            return ScanIssue(relative_path, f"invalid {label}: {reason}")
        try:
            return _conversion_for(path, relative_path, scope_root, encoding, label)
        except OSError as exc:
            return ScanIssue(relative_path, f"cannot fingerprint file: {exc}")

    utf16_encoding = _guess_bomless_utf16(sample)
    if utf16_encoding is not None:
        valid, reason = _validate_decoding(path, utf16_encoding)
        if not valid:
            return ScanIssue(
                relative_path,
                f"looks like {utf16_encoding} but cannot be decoded: {reason}",
            )
        try:
            return _conversion_for(
                path,
                relative_path,
                scope_root,
                utf16_encoding,
                utf16_encoding.upper() + " without BOM",
            )
        except OSError as exc:
            return ScanIssue(relative_path, f"cannot fingerprint file: {exc}")

    # A UTF-8 BOM is valid UTF-8 and is deliberately preserved.  Some checked-
    # in validation inputs are hash-pinned with that exact byte sequence.
    utf8_valid, utf8_reason = _validate_decoding(path, "utf-8")
    if utf8_valid:
        return None

    mixed_valid, mixed_reason = _validate_decoding(
        path,
        "utf-8",
        errors=UTF8_CP1252_ERROR_HANDLER,
    )
    if mixed_valid:
        try:
            return _conversion_for(
                path,
                relative_path,
                scope_root,
                "utf-8",
                "UTF-8 with Windows-1252 bytes",
                source_errors=UTF8_CP1252_ERROR_HANDLER,
            )
        except OSError as exc:
            return ScanIssue(relative_path, f"cannot fingerprint file: {exc}")

    return ScanIssue(
        relative_path,
        f"invalid UTF-8 ({utf8_reason}); Windows-1252 recovery failed "
        f"({mixed_reason})",
    )


def _walk_candidates(
    root: Path,
    suffixes: frozenset[str],
) -> tuple[list[Path], list[ScanIssue]]:
    issues: list[ScanIssue] = []
    if root.is_file():
        if _is_link_like(root):
            return [], [ScanIssue(Path(root.name), "symbolic links are not modified")]
        if not _is_candidate(root, suffixes):
            return [], [
                ScanIssue(
                    Path(root.name),
                    "unrecognized text extension; use --include-extension if this "
                    "is known to be text",
                )
            ]
        return [root], issues

    def on_walk_error(exc: OSError) -> None:
        filename = Path(exc.filename) if exc.filename else root
        try:
            relative = filename.relative_to(root)
        except ValueError:
            relative = filename
        issues.append(ScanIssue(relative, f"cannot scan directory: {exc}"))

    candidates: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=on_walk_error
    ):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names, key=str.casefold):
            child = directory_path / name
            lowered_name = name.casefold()
            try:
                resolved_child = child.resolve(strict=True)
            except OSError as exc:
                relative = child.relative_to(root)
                issues.append(ScanIssue(relative, f"cannot resolve directory: {exc}"))
                continue
            if (
                lowered_name.startswith(".")
                or lowered_name in SKIP_DIRECTORY_NAMES
                or lowered_name.endswith(".egg-info")
                or _is_link_like(child)
                or not _is_relative_to(resolved_child, root)
            ):
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names, key=str.casefold):
            candidate = directory_path / name
            if _is_link_like(candidate) or not _is_candidate(candidate, suffixes):
                continue
            try:
                resolved_candidate = candidate.resolve(strict=True)
            except OSError as exc:
                relative = candidate.relative_to(root)
                issues.append(ScanIssue(relative, f"cannot resolve file: {exc}"))
                continue
            if not _is_relative_to(resolved_candidate, root):
                relative = candidate.relative_to(root)
                issues.append(
                    ScanIssue(relative, "resolved path escapes the requested folder")
                )
                continue
            try:
                candidate_info = os.stat(candidate, follow_symlinks=False)
            except OSError as exc:
                relative = candidate.relative_to(root)
                issues.append(ScanIssue(relative, f"cannot inspect file: {exc}"))
                continue
            if not stat.S_ISREG(candidate_info.st_mode):
                relative = candidate.relative_to(root)
                issues.append(ScanIssue(relative, "path is not a regular file"))
                continue
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.relative_to(root).as_posix().casefold())
    return candidates, issues


def scan_tree(
    root: Path,
    *,
    extra_suffixes: Iterable[str] = (),
) -> ScanResult:
    """Inspect a file or directory without changing it."""

    root = root.expanduser()
    if _is_link_like(root):
        raise ValueError(f"symbolic links are not modified: {root}")
    root = root.resolve()
    if not root.exists():
        raise ValueError(f"path does not exist: {root}")
    if not root.is_file() and not root.is_dir():
        raise ValueError(f"path is not a regular file or directory: {root}")

    suffixes = frozenset(TEXT_SUFFIXES.union(extra_suffixes))
    base = root if root.is_dir() else root.parent
    candidates, walk_issues = _walk_candidates(root, suffixes)
    conversions: list[Conversion] = []
    issues = list(walk_issues)
    clean_count = 0

    for path in candidates:
        relative = path.relative_to(base)
        result = inspect_file(path, relative, base)
        if result is None:
            clean_count += 1
        elif isinstance(result, Conversion):
            conversions.append(result)
        else:
            issues.append(result)

    return ScanResult(
        root=root,
        base=base,
        candidate_count=len(candidates),
        clean_count=clean_count,
        conversions=tuple(conversions),
        issues=tuple(issues),
    )


def _default_backup_path(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    stem = root.name or "files"
    parent = root.parent
    candidate = parent / f"{stem}-utf8-backup-{stamp}.zip"
    index = 2
    while candidate.exists():
        candidate = parent / f"{stem}-utf8-backup-{stamp}-{index}.zip"
        index += 1
    return candidate


def create_backup(
    result: ScanResult,
    destination: Path | None = None,
) -> Path:
    """Create an atomic ZIP containing the exact bytes that will be replaced."""

    destination = (
        _default_backup_path(result.root)
        if destination is None
        else destination.expanduser().resolve(strict=False)
    )
    if destination.exists():
        raise FileExistsError(f"backup already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"backup parent directory does not exist: {destination.parent}"
        )

    manifest = {
        "format": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(result.root),
        "files": [
            {
                "path": item.relative_path.as_posix(),
                "source_encoding": item.display_encoding,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in result.conversions
        ],
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                "UTF8_CLEANUP_MANIFEST.json",
                json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            )
            for item in result.conversions:
                _assert_source_unchanged(item)
                archive_name = PurePosixPath(
                    "originals", *item.relative_path.parts
                ).as_posix()
                archive.write(item.path, archive_name)
                _assert_source_unchanged(item)
        # Windows requires a writable descriptor for FlushFileBuffers, which
        # is what os.fsync uses there.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def _rewrite_declared_encoding(path: Path, lines: list[str]) -> list[str]:
    suffix = path.suffix.casefold()
    if suffix in PYTHON_SUFFIXES:
        return [PYTHON_CODING_RE.sub(r"\1utf-8", line) for line in lines]
    if suffix in XML_SUFFIXES and lines:
        lines[0] = XML_ENCODING_RE.sub(r"\1utf-8\3", lines[0])
    return lines


def _write_utf8_temporary(item: Conversion) -> Path:
    _assert_source_unchanged(item)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{item.path.name}.utf8-", suffix=".tmp", dir=item.path.parent
    )
    temporary = Path(temporary_name)
    try:
        with item.path.open(
            "r",
            encoding=item.source_encoding,
            errors=item.source_errors,
            newline="",
        ) as source, os.fdopen(
            descriptor, "w", encoding="utf-8", errors="strict", newline=""
        ) as destination:
            descriptor = -1
            suffix = item.path.suffix.casefold()
            if suffix in PYTHON_SUFFIXES:
                header_line_count = 2
            elif suffix in XML_SUFFIXES:
                header_line_count = 1
            else:
                header_line_count = 0
            header: list[str] = []
            for _ in range(header_line_count):
                line = source.readline()
                if not line:
                    break
                header.append(line)
            destination.writelines(_rewrite_declared_encoding(item.path, header))
            shutil.copyfileobj(source, destination, length=CHUNK_SIZE)
            destination.flush()
            os.fsync(destination.fileno())

        shutil.copystat(item.path, temporary, follow_symlinks=False)
        valid, reason = _validate_decoding(temporary, "utf-8")
        if not valid:
            raise RuntimeError(f"converted temporary file failed verification: {reason}")
        return temporary
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def apply_conversion(item: Conversion) -> None:
    """Convert one planned file and replace it only after verification."""

    temporary = _write_utf8_temporary(item)
    try:
        # Re-check immediately before replacement so a save that happened
        # during a long conversion is never overwritten.
        _assert_source_unchanged(item)
        os.replace(temporary, item.path)
        _fsync_directory(item.path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find invalid UTF-8 in known GRIM text files and safely convert "
            "Windows-1252 or UTF-16 text to UTF-8. Preview is the default."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        type=Path,
        help="file or folder to scan (default: current folder)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the planned conversions; without this flag nothing changes",
    )
    backup_group = parser.add_mutually_exclusive_group()
    backup_group.add_argument(
        "--backup",
        type=Path,
        help="recovery ZIP path (default: beside the scanned folder)",
    )
    backup_group.add_argument(
        "--no-backup",
        action="store_true",
        help="apply without a recovery ZIP (not recommended)",
    )
    parser.add_argument(
        "--include-extension",
        action="append",
        default=[],
        metavar="EXT",
        type=_normalise_extra_suffix,
        help=(
            "also treat EXT as text (repeatable); do not use this for .stl, "
            ".grim, .ptm, .ss, or another binary/mixed format"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print the count of already-clean files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = scan_tree(
            args.path,
            extra_suffixes=args.include_extension,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Scanned: {result.root}")
    for item in result.conversions:
        action = "CONVERT" if args.apply else "WOULD CONVERT"
        print(
            f"[{action}] {item.relative_path.as_posix()} "
            f"({item.display_encoding} -> UTF-8)"
        )
    for issue in result.issues:
        print(
            f"[UNRESOLVED] {issue.relative_path.as_posix()}: {issue.message}",
            file=sys.stderr,
        )

    if args.verbose:
        print(f"Already clean UTF-8: {result.clean_count}")
    print(
        "Summary: "
        f"{result.candidate_count} text file(s), "
        f"{len(result.conversions)} conversion(s), "
        f"{len(result.issues)} unresolved"
    )

    if not args.apply:
        if result.conversions:
            print("Preview only; re-run the same command with --apply to repair them.")
        else:
            print("No recognized text files need conversion.")
        return 1 if result.issues else 0

    if not result.conversions:
        print("No files changed.")
        return 1 if result.issues else 0

    backup_path: Path | None = None
    if not args.no_backup:
        try:
            backup_path = create_backup(result, args.backup)
        except (OSError, RuntimeError) as exc:
            print(f"ERROR: could not create recovery ZIP: {exc}", file=sys.stderr)
            print("No files were changed.", file=sys.stderr)
            return 1
        print(f"Recovery ZIP: {backup_path}")

    applied = 0
    failed = 0
    for item in result.conversions:
        try:
            apply_conversion(item)
        except (OSError, RuntimeError, UnicodeError) as exc:
            failed += 1
            print(
                f"[FAILED] {item.relative_path.as_posix()}: {exc}",
                file=sys.stderr,
            )
        else:
            applied += 1
            print(f"[REPAIRED] {item.relative_path.as_posix()}")

    print(
        f"Applied: {applied}; failed: {failed}; "
        f"unresolved during scan: {len(result.issues)}"
    )
    if (result.base / "SHA256SUMS.txt").is_file() and applied:
        print(
            "NOTICE: This folder has SHA256SUMS.txt. Repaired files no longer "
            "match the original release checksums; keep the recovery ZIP."
        )
    if backup_path is not None:
        print(f"Original bytes can be recovered from: {backup_path}")
    return 1 if failed or result.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
