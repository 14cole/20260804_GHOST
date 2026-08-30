#!/usr/bin/env python3
"""Generate or verify a locked offline wheelhouse SHA-256 manifest.

This utility is deliberately standard-library-only. It never downloads a
package and never imports a wheel. The wheelhouse must contain exactly one
valid wheel for every exact pin in the reviewed Windows constraints file.
"""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Iterable, Sequence
import zipfile


MANIFEST_NAME = "WHEELHOUSE-SHA256SUMS.txt"
CHUNK_SIZE = 1024 * 1024
TARGET_CPYTHON_MINOR = 12
TARGET_WINDOWS_PLATFORM = "win_amd64"


class WheelhouseError(RuntimeError):
    """The offline dependency bundle is incomplete, corrupt, or inconsistent."""


def _normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def read_constraints(path: Path | str) -> dict[str, str]:
    constraints_path = Path(path).expanduser().resolve()
    try:
        lines = constraints_path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WheelhouseError(f"Cannot read constraints {constraints_path}: {exc}") from exc

    pins: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        requirement = stripped.split(";", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", requirement)
        if match is None:
            raise WheelhouseError(
                f"Constraints must contain exact pins: {constraints_path}:{line_number}: {line!r}"
            )
        name = _normalise_name(match.group(1))
        if name in pins:
            raise WheelhouseError(f"Duplicate constraint for {name!r} at line {line_number}.")
        pins[name] = match.group(2)
    if not pins:
        raise WheelhouseError(f"Constraints contain no package pins: {constraints_path}")
    return pins


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise WheelhouseError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _supported_windows_cp312_tag(
    python_tag: str,
    abi_tag: str,
    platform_tag: str,
) -> bool:
    """Return whether one wheel tag runs on 64-bit CPython 3.12/Windows."""

    if platform_tag not in {"any", TARGET_WINDOWS_PLATFORM}:
        return False
    if python_tag == "py3":
        return abi_tag == "none"
    generic_match = re.fullmatch(r"py3(\d+)", python_tag)
    if generic_match is not None:
        return abi_tag == "none" and int(generic_match.group(1)) <= TARGET_CPYTHON_MINOR
    match = re.fullmatch(r"cp3(\d+)", python_tag)
    if match is None:
        return False
    minor = int(match.group(1))
    if abi_tag == "none":
        return minor == TARGET_CPYTHON_MINOR
    # Platform-independent wheels cannot carry a CPython or stable extension
    # ABI.  These tags are not in pip's supported tag set for this target even
    # though their individual components look plausible.
    if platform_tag == "any":
        return False
    if minor == TARGET_CPYTHON_MINOR:
        return abi_tag in {"cp312", "abi3"}
    # A limited-API extension compiled for an older CPython 3.x release is
    # intentionally compatible with newer CPython releases.
    return 2 <= minor <= TARGET_CPYTHON_MINOR and abi_tag == "abi3"


def _expanded_filename_tags(
    python_tags: str,
    abi_tags: str,
    platform_tags: str,
) -> set[str]:
    return {
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in python_tags.split(".")
        for abi_tag in abi_tags.split(".")
        for platform_tag in platform_tags.split(".")
    }


def _wheel_identity(path: Path) -> tuple[str, str]:
    parts = path.stem.split("-")
    if len(parts) not in {5, 6}:
        raise WheelhouseError(f"Invalid wheel filename: {path.name}")
    name = _normalise_name(parts[0])
    version = parts[1].replace("_", ".")
    if len(parts) == 5:
        python_tags, abi_tags, platform_tags = parts[2:]
    else:
        build_tag, python_tags, abi_tags, platform_tags = parts[2:]
        if re.fullmatch(r"[0-9][0-9A-Za-z_.]*", build_tag) is None:
            raise WheelhouseError(
                f"Wheel {path.name} has an invalid build tag: {build_tag!r}"
            )
    filename_tags = _expanded_filename_tags(
        python_tags, abi_tags, platform_tags
    )
    if not any(
        _supported_windows_cp312_tag(*tag.split("-", 2))
        for tag in filename_tags
    ):
        raise WheelhouseError(
            f"Wheel {path.name} is not compatible with 64-bit CPython 3.12 "
            "on Windows."
        )
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise WheelhouseError(
                    f"Wheel {path.name} has a corrupt member: {bad_member}"
                )
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise WheelhouseError(
                    f"Wheel {path.name} contains duplicate ZIP member names."
                )
            for item in names:
                pure = PurePosixPath(item)
                if (
                    not item
                    or "\\" in item
                    or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                ):
                    raise WheelhouseError(
                        f"Wheel {path.name} contains an unsafe ZIP member: {item!r}"
                    )
            dist_info_roots = {
                PurePosixPath(item).parts[0]
                for item in names
                if PurePosixPath(item).parts
                and PurePosixPath(item).parts[0].endswith(".dist-info")
            }
            if len(dist_info_roots) != 1:
                raise WheelhouseError(
                    f"Wheel {path.name} must contain exactly one .dist-info directory."
                )
            dist_info_root = next(iter(dist_info_roots))
            required_metadata = {
                "WHEEL": f"{dist_info_root}/WHEEL",
                "METADATA": f"{dist_info_root}/METADATA",
                "RECORD": f"{dist_info_root}/RECORD",
            }
            missing = [
                label for label, member in required_metadata.items() if member not in names
            ]
            if missing:
                raise WheelhouseError(
                    f"Wheel {path.name} is missing required metadata: {', '.join(missing)}"
                )
            wheel_document = BytesParser(policy=policy.compat32).parsebytes(
                archive.read(required_metadata["WHEEL"])
            )
            package_document = BytesParser(policy=policy.compat32).parsebytes(
                archive.read(required_metadata["METADATA"])
            )
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise WheelhouseError(f"Wheel is not a valid ZIP archive: {path.name}: {exc}") from exc

    dist_identity = dist_info_root.removesuffix(".dist-info").rsplit("-", 1)
    if len(dist_identity) != 2:
        raise WheelhouseError(
            f"Wheel {path.name} has an invalid .dist-info directory name: {dist_info_root!r}"
        )
    dist_name, dist_version = dist_identity
    metadata_names = package_document.get_all("Name", failobj=[])
    metadata_versions = package_document.get_all("Version", failobj=[])
    if len(metadata_names) != 1 or len(metadata_versions) != 1:
        raise WheelhouseError(
            f"Wheel {path.name} METADATA must declare exactly one Name and Version."
        )
    identities = (
        ("filename", name, version),
        (".dist-info", _normalise_name(dist_name), dist_version.replace("_", ".")),
        (
            "METADATA",
            _normalise_name(str(metadata_names[0]).strip()),
            str(metadata_versions[0]).strip(),
        ),
    )
    if any(item_name != name or item_version != version for _label, item_name, item_version in identities):
        detail = "; ".join(
            f"{label}={item_name}=={item_version}"
            for label, item_name, item_version in identities
        )
        raise WheelhouseError(
            f"Wheel {path.name} has contradictory package identity: {detail}"
        )

    metadata_tags = {
        str(value).strip()
        for value in wheel_document.get_all("Tag", failobj=[])
        if str(value).strip()
    }
    if not metadata_tags:
        raise WheelhouseError(
            f"Wheel {path.name} has no compatibility Tag in .dist-info/WHEEL."
        )
    for tag in metadata_tags:
        components = tag.split("-")
        if len(components) != 3:
            raise WheelhouseError(
                f"Wheel {path.name} has a malformed metadata Tag: {tag!r}"
            )
    shared_tags = filename_tags.intersection(metadata_tags)
    if not shared_tags:
        raise WheelhouseError(
            f"Wheel {path.name} filename tags contradict its WHEEL metadata."
        )
    if not any(
        _supported_windows_cp312_tag(*tag.split("-", 2))
        for tag in shared_tags
    ):
        raise WheelhouseError(
            f"Wheel {path.name} has no Windows CPython 3.12-compatible tag "
            "shared by its filename and WHEEL metadata."
        )
    return name, version


def _wheel_inventory(
    wheelhouse: Path,
    constraints: dict[str, str],
) -> tuple[Path, ...]:
    if not wheelhouse.is_dir():
        raise WheelhouseError(f"Wheelhouse is not a directory: {wheelhouse}")
    unsupported = sorted(
        path.name
        for path in wheelhouse.iterdir()
        if path.name != MANIFEST_NAME and (not path.is_file() or path.suffix.casefold() != ".whl")
    )
    if unsupported:
        raise WheelhouseError(
            "Wheelhouse must be flat and contain only .whl files plus its manifest: "
            + ", ".join(unsupported)
        )
    wheels = tuple(sorted(wheelhouse.glob("*.whl"), key=lambda path: path.name.casefold()))
    if not wheels:
        raise WheelhouseError(f"Wheelhouse contains no wheels: {wheelhouse}")

    by_name: dict[str, tuple[Path, str]] = {}
    for wheel in wheels:
        if wheel.is_symlink():
            raise WheelhouseError(f"Symbolic-link wheel is not allowed: {wheel.name}")
        name, version = _wheel_identity(wheel)
        if name in by_name:
            raise WheelhouseError(
                f"Multiple wheels were supplied for {name!r}: "
                f"{by_name[name][0].name}, {wheel.name}"
            )
        by_name[name] = (wheel, version)

    missing = sorted(set(constraints) - set(by_name))
    extra = sorted(set(by_name) - set(constraints))
    wrong = sorted(
        f"{name}: locked {constraints[name]}, wheel {by_name[name][1]}"
        for name in set(constraints) & set(by_name)
        if constraints[name] != by_name[name][1]
    )
    problems: list[str] = []
    if missing:
        problems.append("missing locked package(s): " + ", ".join(missing))
    if extra:
        problems.append("unlocked package(s): " + ", ".join(extra))
    if wrong:
        problems.append("version mismatch(es): " + "; ".join(wrong))
    if problems:
        raise WheelhouseError("Wheelhouse does not match constraints: " + " | ".join(problems))
    return wheels


def _manifest_text(wheels: Iterable[Path], constraints_sha256: str) -> str:
    lines = [
        "# SHA-256 manifest for the locked GRIM Windows CPython 3.12 wheelhouse",
        f"# Constraints-SHA256: {constraints_sha256}",
        "# Verify before every offline installation.",
    ]
    lines.extend(f"{_sha256(path)}  {path.name}" for path in wheels)
    return "\n".join(lines) + "\n"


def generate_manifest(
    wheelhouse: Path | str,
    constraints_path: Path | str,
) -> Path:
    root = Path(wheelhouse).expanduser().resolve()
    resolved_constraints = Path(constraints_path).expanduser().resolve()
    constraints = read_constraints(resolved_constraints)
    wheels = _wheel_inventory(root, constraints)
    text = _manifest_text(wheels, _sha256(resolved_constraints))
    descriptor, temporary_name = tempfile.mkstemp(prefix=".wheelhouse-manifest-", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        destination = root / MANIFEST_NAME
        os.replace(temporary, destination)
        return destination
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_manifest(path: Path) -> tuple[dict[str, str], str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WheelhouseError(f"Cannot read wheelhouse manifest {path}: {exc}") from exc
    entries: dict[str, str] = {}
    constraints_sha256 = ""
    for line_number, line in enumerate(lines, 1):
        if line.startswith("# Constraints-SHA256: "):
            candidate = line.removeprefix("# Constraints-SHA256: ")
            if constraints_sha256 or re.fullmatch(r"[0-9a-f]{64}", candidate) is None:
                raise WheelhouseError(
                    f"Invalid constraints digest at manifest line {line_number}."
                )
            constraints_sha256 = candidate
            continue
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+\.whl)", line)
        if match is None:
            raise WheelhouseError(f"Invalid manifest entry at line {line_number}: {line!r}")
        name = match.group(2)
        portable = name.casefold()
        if any(existing.casefold() == portable for existing in entries):
            raise WheelhouseError(f"Duplicate manifest wheel name at line {line_number}: {name}")
        entries[name] = match.group(1)
    if not entries:
        raise WheelhouseError(f"Wheelhouse manifest contains no entries: {path}")
    if not constraints_sha256:
        raise WheelhouseError(f"Wheelhouse manifest lacks its constraints digest: {path}")
    return entries, constraints_sha256


def verify_manifest(
    wheelhouse: Path | str,
    constraints_path: Path | str,
) -> int:
    root = Path(wheelhouse).expanduser().resolve()
    resolved_constraints = Path(constraints_path).expanduser().resolve()
    constraints = read_constraints(resolved_constraints)
    wheels = _wheel_inventory(root, constraints)
    entries, manifest_constraints_sha256 = _read_manifest(root / MANIFEST_NAME)
    actual_constraints_sha256 = _sha256(resolved_constraints)
    if manifest_constraints_sha256 != actual_constraints_sha256:
        raise WheelhouseError(
            "Constraints SHA-256 does not match the wheelhouse manifest; use the "
            "lock that generated this dependency bundle."
        )
    wheel_names = {path.name for path in wheels}
    if set(entries) != wheel_names:
        missing = sorted(wheel_names - set(entries))
        extra = sorted(set(entries) - wheel_names)
        raise WheelhouseError(
            "Manifest inventory mismatch; missing entries: "
            f"{missing or 'none'}; extra entries: {extra or 'none'}"
        )
    failures = [
        path.name
        for path in wheels
        if _sha256(path) != entries[path.name]
    ]
    if failures:
        raise WheelhouseError("Wheel SHA-256 mismatch: " + ", ".join(failures))
    return len(wheels)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the locked offline GRIM wheelhouse manifest."
    )
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument(
        "--constraints",
        type=Path,
        default=Path(__file__).resolve().with_name("constraints-windows-py312.txt"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "generate":
            path = generate_manifest(args.wheelhouse, args.constraints)
            print(f"Wrote {path}")
        else:
            count = verify_manifest(args.wheelhouse, args.constraints)
            print(f"Verified {count} locked wheel(s) in {Path(args.wheelhouse).resolve()}")
    except WheelhouseError as exc:
        print(f"Wheelhouse verification failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
