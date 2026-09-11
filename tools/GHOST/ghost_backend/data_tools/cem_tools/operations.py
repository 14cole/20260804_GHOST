"""Public headless operations used by both the CLI and GUI."""

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable
import uuid

import numpy as np

from .errors import CemToolError
from .grim_bridge import INPUT_EXTENSIONS, OUTPUT_EXTENSIONS, convert_dataset
from .grim_native import join_payloads, load_grim, save_grim_atomic, subtract_payloads
from .naming import group_stem
from .solver_pairing import pairing_module


ProgressCallback = Callable[[int, int, str], None]


def _report(progress, completed: 'int', total: 'int', message: 'str') -> 'None':
    """Report completed batch items; total zero means discovery is underway."""
    if progress is not None:
        progress(completed, total, message)


@dataclass(frozen=True)
class BatchResult:
    written: 'tuple[Path, ...]'
    skipped: 'tuple[Path, ...]' = ()
    warnings: 'tuple[str, ...]' = ()

    def summary(self) -> 'str':
        text = f"Wrote {len(self.written)} file(s)"
        if self.skipped:
            text += f"; skipped {len(self.skipped)} unchanged file(s)"
        if self.warnings:
            text += f"; {len(self.warnings)} warning(s)"
        return text


def _directory(path: 'str | os.PathLike[str]', *, create: 'bool' = False) -> 'Path':
    result = Path(path).expanduser().resolve()
    if create:
        result.mkdir(parents=True, exist_ok=True)
    if not result.is_dir():
        raise CemToolError(f"not a directory: {result}")
    return result


def _files(folder: 'Path', extensions: 'tuple[str, ...]') -> 'list[Path]':
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def _require_separate_output(output: 'Path', *inputs: 'Path') -> 'None':
    if any(output == source for source in inputs):
        raise CemToolError(
            "input and output folders must be different; only Rename Files "
            "supports in-place operation"
        )


def _group_grim(folder: 'Path', remove: 'str') -> 'dict[str, list[Path]]':
    files = _files(folder, (".grim",))
    if not files:
        raise CemToolError(f"no .grim files found in {folder}")
    groups: 'dict[str, list[Path]]' = defaultdict(list)
    for path in files:
        groups[group_stem(path, remove=remove)].append(path)
    return dict(groups)


def _join_files(paths: 'list[Path]', axis: 'str', status=None) -> 'dict':
    payloads = []
    for index, path in enumerate(paths, start=1):
        if status is not None:
            status(f"Reading {index}/{len(paths)}: {path.name}")
        payloads.append(load_grim(path))
    if status is not None:
        status("Combining samples")
    return join_payloads(payloads, axis=axis, labels=[str(path) for path in paths])


def _join_library_group(paths: 'list[Path]', status=None) -> 'dict':
    """Join arbitrary single- or multi-pol/frequency files into one grid."""
    payloads = []
    for index, path in enumerate(paths, start=1):
        if status is not None:
            status(f"Reading {index}/{len(paths)}: {path.name}")
        payloads.append((path, load_grim(path)))
    if status is not None:
        status("Combining input samples")
    frequency_buckets: 'dict[tuple[float, ...], list[tuple[Path, dict]]]' = defaultdict(list)
    for path, payload in payloads:
        key = tuple(np.asarray(payload["frequencies"], dtype=float).tolist())
        frequency_buckets[key].append((path, payload))
    by_frequency: 'list[tuple[str, dict]]' = []
    for key, parts in frequency_buckets.items():
        labels = [str(path) for path, _ in parts]
        joined = join_payloads(
            [payload for _, payload in parts],
            axis="polarizations",
            labels=labels,
        )
        by_frequency.append(("+".join(labels), joined))
    if len(by_frequency) == 1:
        return by_frequency[0][1]
    return join_payloads(
        [payload for _, payload in by_frequency],
        axis="frequencies",
        labels=[label for label, _ in by_frequency],
    )


def _variation_groups(folder: 'Path', required_role: 'str') -> 'dict[str, list[Path]]':
    pairing = pairing_module()
    files = _files(folder, (".grim",))
    if not files:
        raise CemToolError(f"no .grim files found in {folder}")
    groups: 'dict[str, list[Path]]' = defaultdict(list)
    for path in files:
        variation = group_stem(path, remove="axes")
        try:
            _base, role = pairing.parse_variation(variation)
            pairing.parse_base(variation)
        except ValueError as exc:
            raise CemToolError(str(exc)) from exc
        if role != required_role:
            raise CemToolError(
                f"{path.name}: expected final _{required_role} role marker"
            )
        groups[variation].append(path)
    return dict(groups)


def _concatenate(
    input_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    *,
    axis: 'str',
    remove: 'str',
    overwrite: 'bool',
    progress: 'ProgressCallback | None' = None,
) -> 'BatchResult':
    _report(progress, 0, 0, "Scanning input files")
    source = _directory(input_dir)
    destination = _directory(output_dir, create=True)
    _require_separate_output(destination, source)
    written = []
    groups = _group_grim(source, remove)
    total = len(groups)
    for completed, (stem, paths) in enumerate(groups.items()):
        def status(message):
            _report(progress, completed, total, f"{stem}: {message}")
        payload = _join_files(paths, axis, status=status)
        status(f"Writing {stem}.grim")
        written.append(
            save_grim_atomic(payload, destination / f"{stem}.grim", overwrite=overwrite)
        )
        _report(progress, completed + 1, total, f"Saved {stem}.grim")
    return BatchResult(tuple(written))


def concatenate_polarizations(
    input_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    *,
    overwrite: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
) -> 'BatchResult':
    return _concatenate(
        input_dir, output_dir, axis="polarizations",
        remove="polarization", overwrite=overwrite, progress=progress,
    )


def concatenate_frequencies(
    input_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    *,
    overwrite: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
) -> 'BatchResult':
    return _concatenate(
        input_dir, output_dir, axis="frequencies",
        remove="frequency", overwrite=overwrite, progress=progress,
    )


def subtract_datasets(
    opn_dir: 'str | os.PathLike[str]',
    frd_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    *,
    overwrite: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
) -> 'BatchResult':
    """Coherently subtract OPN - FRD using solver raw far-field amplitudes."""
    _report(progress, 0, 0, "Scanning OPN and FRD libraries")
    opn_path = _directory(opn_dir)
    frd_path = _directory(frd_dir)
    opn = _variation_groups(opn_path, "OPN")
    frd = _variation_groups(frd_path, "FRD")
    pairing = pairing_module()
    virtual_root = Path("/CEM_Tools_pairing")
    virtual_paths = [
        str(virtual_root / f"{variation}.grim")
        for variation in sorted(set(opn) | set(frd))
    ]
    try:
        pairs, unmatched = pairing.pair_variants(virtual_paths)
    except ValueError as exc:
        raise CemToolError(str(exc)) from exc
    if not pairs:
        raise CemToolError("no OPN case has a compatible FRD baseline")
    destination = _directory(output_dir, create=True)
    _require_separate_output(destination, opn_path, frd_path)
    written = []
    total = len(pairs)
    for completed, pair in enumerate(pairs):
        featured_variation = Path(pair["featured"]).stem
        clean_variation = Path(pair["clean"]).stem
        def status(message):
            _report(progress, completed, total, f"{pair['delta_name']}: {message}")
        featured = _join_library_group(opn[featured_variation], status=status)
        clean = _join_library_group(frd[clean_variation], status=status)
        status("Subtracting complex fields")
        delta = subtract_payloads(
            featured, clean,
            featured_label=f"OPN/{featured_variation}",
            clean_label=f"FRD/{clean_variation}",
        )
        status("Writing delta")
        written.append(
            save_grim_atomic(
                delta, destination / pair["delta_name"], overwrite=overwrite
            )
        )
        _report(progress, completed + 1, total, f"Saved {pair['delta_name']}")
    warnings = tuple(
        f"{Path(item['path']).name}: {item['reason']}" for item in unmatched
    )
    return BatchResult(tuple(written), warnings=warnings)


def rename_files(
    input_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str] | None',
    keyword: 'str',
    replacement: 'str',
    *,
    in_place: 'bool' = False,
    overwrite: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
) -> 'BatchResult':
    _report(progress, 0, 0, "Checking filenames and destinations")
    source = _directory(input_dir)
    if not keyword:
        raise CemToolError("rename keyword cannot be empty")
    if not in_place and not output_dir:
        raise CemToolError("an output folder is required unless rename-in-place is selected")
    destination = source if in_place else _directory(output_dir or "", create=True)
    files = sorted(path for path in source.iterdir() if path.is_file())
    mappings = [(path, destination / path.name.replace(keyword, replacement)) for path in files]
    mappings = [(source_path, target) for source_path, target in mappings if source_path.name != target.name]
    targets = [target for _, target in mappings]
    if len(set(targets)) != len(targets):
        raise CemToolError("rename replacements would create duplicate filenames")
    source_set = {path.resolve() for path, _ in mappings}
    for _, target in mappings:
        if target.exists() and target.resolve() not in source_set and not overwrite:
            raise CemToolError(f"output exists: {target}")

    total = len(mappings)
    _report(progress, 0, total, f"{total} matching files")
    written: 'list[Path]' = []
    if not in_place:
        for completed, (source_path, target) in enumerate(mappings):
            _report(progress, completed, total, f"Copying {source_path.name} to {target.name}")
            shutil.copy2(source_path, target)
            written.append(target)
            _report(progress, completed + 1, total, f"Saved {target.name}")
    else:
        staged: 'list[tuple[Path, Path]]' = []
        try:
            for index, (source_path, target) in enumerate(mappings, start=1):
                _report(progress, 0, total, f"Preparing rename {index}/{total}: {source_path.name}")
                temporary = source / f".cemtools-rename-{uuid.uuid4().hex}"
                os.replace(source_path, temporary)
                staged.append((temporary, target))
            for completed, (temporary, target) in enumerate(staged):
                _report(progress, completed, total, f"Renaming to {target.name}")
                if target.exists() and overwrite:
                    target.unlink()
                os.replace(temporary, target)
                written.append(target)
                _report(progress, completed + 1, total, f"Renamed {target.name}")
        except Exception:
            for temporary, target in staged:
                if temporary.exists():
                    original = next((old for old, new in mappings if new == target), None)
                    if original is not None and not original.exists():
                        os.replace(temporary, original)
            raise
    skipped = tuple(path for path in files if keyword not in path.name)
    return BatchResult(tuple(written), skipped)


def convert_files(
    input_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    extension: 'str',
    *,
    overwrite: 'bool' = False,
    progress: 'ProgressCallback | None' = None,
) -> 'BatchResult':
    _report(progress, 0, 0, "Scanning input datasets")
    source = _directory(input_dir)
    destination = _directory(output_dir, create=True)
    _require_separate_output(destination, source)
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = "." + normalized
    if normalized not in OUTPUT_EXTENSIONS:
        if normalized == ".ss":
            raise CemToolError(".ss is currently read-only and cannot be an output")
        raise CemToolError(f"unsupported output extension: {normalized}")
    files = _files(source, INPUT_EXTENSIONS)
    if not files:
        raise CemToolError(f"no supported dataset files found in {source}")
    written: 'list[Path]' = []
    total = len(files)
    for completed, path in enumerate(files):
        _report(progress, completed, total, f"Converting {path.name} to {normalized}")
        written.extend(convert_dataset(path, destination, normalized, overwrite=overwrite))
        _report(progress, completed + 1, total, f"Converted {path.name}")
    return BatchResult(tuple(written))
