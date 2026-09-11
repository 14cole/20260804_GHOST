"""Dataset and CSV batch staging, compression, publication, and rollback."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import copy
import os
import tempfile
import zlib

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.csv import write_flat_csv


def _target_path_key(path: str | os.PathLike) -> str:
    """Return an absolute, normalized, case-insensitive output-path key."""

    return os.path.abspath(os.path.normpath(os.fspath(path))).casefold()


def _duplicate_target_groups(paths: list[str]) -> list[list[str]]:
    """Return case-insensitive duplicate output groups, in plan order."""

    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for raw_path in paths:
        path = os.fspath(raw_path)
        key = _target_path_key(path)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(path)
    return [grouped[key] for key in order if len(grouped[key]) > 1]


def _ensure_grim_output_path(path: str | os.PathLike) -> str:
    output = os.fspath(path)
    return output if output.casefold().endswith(".grim") else output + ".grim"


class _GrimBatchRollbackError(RuntimeError):
    """A batch failed and at least one prior artifact could not be restored."""


_GRIM_COMPRESSION_SAMPLE_BYTES = 1024**2


_GRIM_SMALL_ARCHIVE_BYTES = 8 * 1024**2


_GRIM_LARGE_MINIMUM_SAVINGS = 0.20


def _representative_contiguous_bytes(
    array: np.ndarray,
    byte_budget: int,
) -> bytes:
    """Return at most ``byte_budget`` bytes without copying the full array."""

    value = np.asarray(array)
    budget = max(0, int(byte_budget))
    if budget == 0 or value.nbytes == 0 or not value.flags.c_contiguous:
        return b""
    raw = memoryview(value).cast("B")
    if len(raw) <= budget:
        return bytes(raw)


    block = max(1, budget // 3)
    middle = max(0, (len(raw) - block) // 2)
    end = max(0, len(raw) - block)
    sample = bytes(raw[:block]) + bytes(raw[middle : middle + block])
    remaining = budget - len(sample)
    if remaining > 0:
        sample += bytes(raw[end : end + remaining])
    return sample[:budget]


def _grim_save_compression_decision(dataset: RcsGrid) -> dict[str, object]:
    """Choose compact versus fast NPZ storage from a bounded byte sample."""

    arrays: list[np.ndarray] = [dataset.rcs_power, dataset.rcs_phase]
    arrays.extend(
        np.asarray(value)
        for value in dataset._extra_to_write().values()
        if isinstance(value, np.ndarray)
        and not np.asarray(value).dtype.hasobject
        and np.asarray(value).ndim > 0
    )


    core = arrays[:2]
    ancillary = sorted(arrays[2:], key=lambda value: value.nbytes, reverse=True)[:6]
    sampled_arrays = [
        value
        for value in core + ancillary
        if value.nbytes and value.flags.c_contiguous
    ]
    total_payload_bytes = sum(int(value.nbytes) for value in arrays)
    per_array_budget = max(
        1,
        _GRIM_COMPRESSION_SAMPLE_BYTES // max(1, len(sampled_arrays)),
    )
    sample_parts: list[bytes] = []
    remaining = _GRIM_COMPRESSION_SAMPLE_BYTES
    for value in sampled_arrays:
        if remaining <= 0:
            break
        part = _representative_contiguous_bytes(
            value, min(per_array_budget, remaining)
        )
        sample_parts.append(part)
        remaining -= len(part)
    sample = b"".join(sample_parts)
    if sample:
        compressed_sample_bytes = len(zlib.compress(sample, level=1))
        compression_ratio = compressed_sample_bytes / len(sample)
    else:
        compression_ratio = 1.0
    estimated_savings = max(0.0, 1.0 - float(compression_ratio))
    minimum_savings = (
        0.02
        if total_payload_bytes <= _GRIM_SMALL_ARCHIVE_BYTES
        else _GRIM_LARGE_MINIMUM_SAVINGS
    )
    return {
        "compressed": bool(estimated_savings >= minimum_savings),
        "estimated_savings_fraction": estimated_savings,
        "sample_bytes": len(sample),
        "payload_bytes": total_payload_bytes,
        "minimum_savings_fraction": minimum_savings,
    }


def _stage_and_publish_grim_batch(
    entries: list[tuple[RcsGrid, str, str]],
    *,
    compression_log: list[dict[str, object]] | None = None,
) -> list[str]:
    """Stage each grid and publish completed files using same-directory backups.

    On publication failure, restore existing targets and retain any backup whose
    restoration fails.
    """

    targets = [
        _ensure_grim_output_path(path) for _dataset, path, _history in entries
    ]
    duplicates = _duplicate_target_groups(targets)
    if duplicates:
        names = ", ".join(os.path.basename(group[0]) for group in duplicates)
        raise ValueError(f"multiple datasets resolve to the same output: {names}")

    staged: list[tuple[str, str]] = []
    backups: dict[str, str | None] = {}
    publication_complete = False
    try:
        for (dataset, _raw_target, row_history), target in zip(entries, targets):
            directory = os.path.dirname(os.path.abspath(target)) or os.curdir
            if os.path.lexists(target) and not os.path.isfile(target):
                raise OSError(
                    f"output target exists but is not a regular file: {target}"
                )
            fd, stage_path = tempfile.mkstemp(
                prefix=".grim-stage-",
                suffix=".staging.grim",
                dir=directory,
            )
            os.close(fd)
            try:


                snapshot = copy.copy(dataset)
                snapshot.history = str(row_history or "").strip()
                compression = _grim_save_compression_decision(snapshot)
                snapshot.save(
                    stage_path,
                    compressed=bool(compression["compressed"]),
                )
                if compression_log is not None:
                    compression_log.append(
                        {**compression, "target": target}
                    )
            except Exception:
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
                raise
            staged.append((stage_path, target))

        for stage_path, target in staged:
            backup_path: str | None = None
            if os.path.lexists(target):
                directory = os.path.dirname(os.path.abspath(target)) or os.curdir
                fd, backup_path = tempfile.mkstemp(
                    prefix=".grim-backup-",
                    suffix=".backup",
                    dir=directory,
                )
                os.close(fd)
                try:
                    os.replace(target, backup_path)
                except BaseException:
                    try:
                        os.unlink(backup_path)
                    except OSError:
                        pass
                    raise
            backups[target] = backup_path
            os.replace(stage_path, target)
        publication_complete = True

    except Exception as original_error:


        rollback_errors: list[str] = []
        for target in reversed(list(backups)):
            backup_path = backups[target]
            try:
                if backup_path and os.path.lexists(backup_path):
                    os.replace(backup_path, target)
                    backups[target] = None
                elif backup_path is None and os.path.lexists(target):
                    os.unlink(target)
            except OSError as exc:
                rollback_errors.append(f"{target}: {exc}")
        if rollback_errors:
            raise _GrimBatchRollbackError(
                "Save publication failed and rollback could not restore every "
                "prior dataset. Retained .grim-backup file(s): "
                + "; ".join(rollback_errors)
            ) from original_error
        raise
    finally:
        for stage_path, _target in staged:
            if os.path.lexists(stage_path):
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass

    if publication_complete:
        for backup_path in backups.values():
            if backup_path and os.path.lexists(backup_path):
                try:
                    os.unlink(backup_path)
                except OSError:
                    pass
    return targets


def _write_dataset_csv(
    dataset: "RcsGrid",
    path: str,
    *,
    scale: str = "linear",
    sep: str = ",",
    include_phase: bool = True,
) -> None:
    """Atomically write the authoritative versioned flat-RCS interchange."""

    output_path = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(output_path) or os.curdir
    fd, stage_path = tempfile.mkstemp(
        prefix=".grim-csv-", suffix=".staging", dir=directory
    )
    os.close(fd)
    try:
        write_flat_csv(
            dataset,
            stage_path,
            scale=scale,
            delimiter=sep,
            include_phase=bool(include_phase),
        )
        os.replace(stage_path, output_path)
    finally:
        if os.path.lexists(stage_path):
            try:
                os.unlink(stage_path)
            except OSError:
                pass


class _CsvBatchRollbackError(RuntimeError):
    """CSV publication failed and at least one prior target was not restored."""


def _stage_and_publish_csv_batch(
    entries: list[tuple["RcsGrid", str]],
    *,
    scale: str,
    include_phase: bool,
) -> list[str]:
    """Write every CSV first, then transactionally publish the whole batch."""

    targets = [os.path.abspath(os.fspath(path)) for _dataset, path in entries]
    duplicates = _duplicate_target_groups(targets)
    if duplicates:
        names = ", ".join(os.path.basename(group[0]) for group in duplicates)
        raise ValueError(f"multiple datasets resolve to the same CSV output: {names}")

    staged: list[tuple[str, str]] = []
    backups: dict[str, str | None] = {}
    publication_complete = False
    try:
        for (dataset, _raw_target), target in zip(entries, targets):
            directory = os.path.dirname(target) or os.curdir
            if os.path.lexists(target) and not os.path.isfile(target):
                raise OSError(f"CSV output target is not a regular file: {target}")
            fd, stage_path = tempfile.mkstemp(
                prefix=".grim-csv-stage-", suffix=".csv", dir=directory
            )
            os.close(fd)
            try:
                write_flat_csv(
                    dataset,
                    stage_path,
                    scale=scale,
                    delimiter=",",
                    include_phase=include_phase,
                )
            except Exception:
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
                raise
            staged.append((stage_path, target))

        for stage_path, target in staged:
            backup_path: str | None = None
            if os.path.lexists(target):
                directory = os.path.dirname(target) or os.curdir
                fd, backup_path = tempfile.mkstemp(
                    prefix=".grim-csv-backup-", suffix=".backup", dir=directory
                )
                os.close(fd)
                try:
                    os.replace(target, backup_path)
                except BaseException:
                    try:
                        os.unlink(backup_path)
                    except OSError:
                        pass
                    raise
            backups[target] = backup_path
            os.replace(stage_path, target)
        publication_complete = True
    except Exception as original_error:
        rollback_errors: list[str] = []
        for target in reversed(list(backups)):
            backup_path = backups[target]
            try:
                if backup_path and os.path.lexists(backup_path):
                    os.replace(backup_path, target)
                    backups[target] = None
                elif backup_path is None and os.path.lexists(target):
                    os.unlink(target)
            except OSError as exc:
                rollback_errors.append(f"{target}: {exc}")
        if rollback_errors:
            raise _CsvBatchRollbackError(
                "CSV publication failed and rollback could not restore every "
                "prior target. Retained backup file(s): "
                + "; ".join(rollback_errors)
            ) from original_error
        raise
    finally:
        for stage_path, _target in staged:
            if os.path.lexists(stage_path):
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass

    if publication_complete:
        for backup_path in backups.values():
            if backup_path and os.path.lexists(backup_path):
                try:
                    os.unlink(backup_path)
                except OSError:
                    pass
    return targets


class DatasetBatchRollbackError(RuntimeError):
    """A batch publication failed and an earlier artifact was not restored."""


def _grim_output_path(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path)
    return value if value.casefold().endswith(".grim") else value + ".grim"


def _replace_file(source: str, destination: str) -> None:
    """Replace the destination file with the staged source file."""

    os.replace(source, destination)


def save_dataset_batch(
    entries: Sequence[tuple[RcsGrid, str | os.PathLike[str]]],
) -> tuple[Path, ...]:
    """Stage and transactionally publish an ordered batch of GRIM datasets."""

    normalized = [(dataset, _grim_output_path(path)) for dataset, path in entries]
    if not normalized:
        raise ValueError("save_dataset_batch needs at least one dataset and path")
    keys: set[str] = set()
    for _dataset, target in normalized:
        key = os.path.abspath(os.path.normpath(target)).casefold()
        if key in keys:
            raise ValueError(f"multiple datasets resolve to the same output: {target}")
        keys.add(key)

    staged: list[tuple[str, str]] = []
    backups: dict[str, str | None] = {}
    publication_complete = False
    try:
        for dataset, target in normalized:
            if not isinstance(dataset, RcsGrid):
                raise TypeError("save_dataset_batch entries must contain RcsGrid objects")
            absolute_target = os.path.abspath(target)
            directory = os.path.dirname(absolute_target) or os.curdir
            if os.path.lexists(absolute_target) and not os.path.isfile(absolute_target):
                raise OSError(
                    f"output target exists but is not a regular file: {absolute_target}"
                )
            fd, stage_path = tempfile.mkstemp(
                prefix=".grim-stage-",
                suffix=".staging.grim",
                dir=directory,
            )
            os.close(fd)
            try:
                dataset.save(stage_path)
            except Exception:
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
                raise
            staged.append((stage_path, absolute_target))

        for stage_path, target in staged:
            backup_path: str | None = None
            if os.path.lexists(target):
                directory = os.path.dirname(target) or os.curdir
                fd, backup_path = tempfile.mkstemp(
                    prefix=".grim-backup-",
                    suffix=".backup",
                    dir=directory,
                )
                os.close(fd)
                try:
                    _replace_file(target, backup_path)
                except BaseException:
                    try:
                        os.unlink(backup_path)
                    except OSError:
                        pass
                    raise
            backups[target] = backup_path
            _replace_file(stage_path, target)
        publication_complete = True
    except Exception as original_error:
        rollback_errors: list[str] = []
        for target in reversed(list(backups)):
            backup_path = backups[target]
            try:
                if backup_path and os.path.lexists(backup_path):
                    _replace_file(backup_path, target)
                    backups[target] = None
                elif backup_path is None and os.path.lexists(target):
                    os.unlink(target)
            except OSError as exc:
                rollback_errors.append(f"{target}: {exc}")
        if rollback_errors:
            raise DatasetBatchRollbackError(
                "Dataset publication failed and rollback could not restore every "
                "prior artifact. Retained .grim-backup file(s): "
                + "; ".join(rollback_errors)
            ) from original_error
        raise
    finally:
        for stage_path, _target in staged:
            if os.path.lexists(stage_path):
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass

    if publication_complete:
        for backup_path in backups.values():
            if backup_path and os.path.lexists(backup_path):
                try:
                    os.unlink(backup_path)
                except OSError:
                    pass
    return tuple(Path(target).resolve() for _dataset, target in normalized)
