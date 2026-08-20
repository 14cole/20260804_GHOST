"""Public headless operations used by both the CLI and GUI."""

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from .errors import CemToolError
from .grim_bridge import INPUT_EXTENSIONS, OUTPUT_EXTENSIONS, convert_dataset
from .grim_native import join_payloads, load_grim, save_grim_atomic, subtract_payloads
from .naming import group_stem
from .solver_pairing import pairing_module


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


def _join_files(paths: 'list[Path]', axis: 'str') -> 'dict':
    payloads = [load_grim(path) for path in paths]
    return join_payloads(payloads, axis=axis, labels=[str(path) for path in paths])


def _join_library_group(paths: 'list[Path]') -> 'dict':
    """Join arbitrary single- or multi-pol/frequency files into one grid."""
    payloads = [(path, load_grim(path)) for path in paths]
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
) -> 'BatchResult':
    source = _directory(input_dir)
    destination = _directory(output_dir, create=True)
    _require_separate_output(destination, source)
    written = []
    for stem, paths in _group_grim(source, remove).items():
        payload = _join_files(paths, axis)
        written.append(
            save_grim_atomic(payload, destination / f"{stem}.grim", overwrite=overwrite)
        )
    return BatchResult(tuple(written))


def concatenate_polarizations(
    input_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    *,
    overwrite: 'bool' = False,
) -> 'BatchResult':
    return _concatenate(
        input_dir, output_dir, axis="polarizations",
        remove="polarization", overwrite=overwrite,
    )


def concatenate_frequencies(
    input_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    *,
    overwrite: 'bool' = False,
) -> 'BatchResult':
    return _concatenate(
        input_dir, output_dir, axis="frequencies",
        remove="frequency", overwrite=overwrite,
    )


def subtract_datasets(
    opn_dir: 'str | os.PathLike[str]',
    frd_dir: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    *,
    overwrite: 'bool' = False,
) -> 'BatchResult':
    """Coherently subtract OPN - FRD using solver raw far-field amplitudes."""
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
    for pair in pairs:
        featured_variation = Path(pair["featured"]).stem
        clean_variation = Path(pair["clean"]).stem
        featured = _join_library_group(opn[featured_variation])
        clean = _join_library_group(frd[clean_variation])
        delta = subtract_payloads(
            featured, clean,
            featured_label=f"OPN/{featured_variation}",
            clean_label=f"FRD/{clean_variation}",
        )
        written.append(
            save_grim_atomic(
                delta, destination / pair["delta_name"], overwrite=overwrite
            )
        )
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
) -> 'BatchResult':
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

    written: 'list[Path]' = []
    if not in_place:
        for source_path, target in mappings:
            shutil.copy2(source_path, target)
            written.append(target)
    else:
        staged: 'list[tuple[Path, Path]]' = []
        try:
            for source_path, target in mappings:
                temporary = source / f".cemtools-rename-{uuid.uuid4().hex}"
                os.replace(source_path, temporary)
                staged.append((temporary, target))
            for temporary, target in staged:
                if target.exists() and overwrite:
                    target.unlink()
                os.replace(temporary, target)
                written.append(target)
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
) -> 'BatchResult':
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
    for path in files:
        written.extend(convert_dataset(path, destination, normalized, overwrite=overwrite))
    return BatchResult(tuple(written))
