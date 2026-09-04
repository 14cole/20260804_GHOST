from __future__ import annotations

import copy
import ctypes
import json
import math
import os
import re
import shutil
import tempfile
import uuid
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PySide6.QtCore import QItemSelectionModel, QObject, QThread, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidgetItem,
    QVBoxLayout,
)

from grim_dataset import (
    C0,
    GRIM_GC_CONVENTION,
    LEGACY_PTM_GC_CONVENTION,
    RcsGrid,
    canonical_angular_coordinate_system,
    wedge_to_conic_geometry_deg,
)
from grim_headless import (
    SUPPORTED_EXTENSIONS,
    is_supported_path,
    load_flat_csv as load_flat_csv_headless,
    load_dataset as load_dataset_headless,
)
from grim_csv_schema import write_flat_csv
from grim_python import (
    DatasetReference,
    _derived_response_extra,
    coherent_divide,
    convert_extrusion,
    crop_dataset,
    decimate_axis,
    medianize_azimuth,
    offset_db,
    regrid_axis,
    shift_dataset,
)

# Characters forbidden in filenames on Windows (and `/` on POSIX). Replaced
# with `_` so dataset names with op symbols like `|`, `÷`, etc. still save.
_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# Stable identity used by consumers such as the PPT report workspace.  Row
# numbers and display names can both change, while one dataset can also appear
# in more than one row, so neither is a safe persistent selection key.
DATASET_ID_ROLE = Qt.UserRole + 32
DATASET_DIRTY_ROLE = Qt.UserRole + 33
DATASET_PATH_ROLE = Qt.UserRole + 34

# Explicit output limits keep a typo such as a 1e-9 degree step from allocating
# an axis (and then a dense four-dimensional result) before the user can react.
# The byte preflight below normally trips first; this independent count limit
# also protects small grids and Python-recorder script generation.
_MAX_EXPLICIT_AXIS_POINTS = 1_000_000
_MAX_DERIVED_PEAK_BYTES_FALLBACK = 2 * 1024**3


def _canonical_angle_unit(value: object, *, default: str = "deg") -> str:
    text = str(value or default).strip().lower()
    aliases = {
        "degree": "deg",
        "degrees": "deg",
        "deg": "deg",
        "radian": "rad",
        "radians": "rad",
        "rad": "rad",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            f"unsupported angular unit {value!r}; use deg or rad"
        ) from exc


def _angle_axis_degrees(dataset: "RcsGrid", axis_name: str) -> np.ndarray:
    values = np.asarray(dataset.get_axis(axis_name), dtype=float)
    unit = _canonical_angle_unit((dataset.units or {}).get(axis_name, "deg"))
    return np.rad2deg(values) if unit == "rad" else values


def _degrees_to_angle_axis(
    dataset: "RcsGrid", axis_name: str, values_degrees
) -> np.ndarray:
    values = np.asarray(values_degrees, dtype=float)
    unit = _canonical_angle_unit((dataset.units or {}).get(axis_name, "deg"))
    return np.deg2rad(values) if unit == "rad" else values


_FREQUENCY_TO_HZ = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}


def _canonical_frequency_unit(value: object, *, default: str = "GHz") -> str:
    text = str(value or default).strip().lower()
    if text not in _FREQUENCY_TO_HZ:
        raise ValueError(
            f"unsupported frequency unit {value!r}; use Hz, kHz, MHz, or GHz"
        )
    return {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}[text]


def _frequency_axis_hz(dataset: "RcsGrid", values=None) -> np.ndarray:
    native = dataset.frequencies if values is None else values
    unit = _canonical_frequency_unit((dataset.units or {}).get("frequency", "GHz"))
    return np.asarray(native, dtype=float) * _FREQUENCY_TO_HZ[unit.lower()]


def _hz_to_frequency_axis(dataset: "RcsGrid", values_hz) -> np.ndarray:
    unit = _canonical_frequency_unit((dataset.units or {}).get("frequency", "GHz"))
    return np.asarray(values_hz, dtype=float) / _FREQUENCY_TO_HZ[unit.lower()]


def _assert_same_angular_frame(reference: "RcsGrid", dataset: "RcsGrid") -> None:
    """Reject transferring angle coordinates between different frames."""

    reference_system = reference.angular_coordinate_system()
    dataset_system = dataset.angular_coordinate_system()
    if reference_system != dataset_system:
        raise ValueError(
            "angular coordinate system differs from the active reference "
            f"({dataset_system} != {reference_system})"
        )
    if reference_system != "great_circle":
        return
    reference_convention = reference.great_circle_coordinate_convention()
    dataset_convention = dataset.great_circle_coordinate_convention()
    if reference_convention != dataset_convention:
        raise ValueError(
            "great-circle convention differs from the active reference "
            f"({dataset_convention} != {reference_convention})"
        )
    if not np.allclose(
        reference.angular_frame_orientation_deg(),
        dataset.angular_frame_orientation_deg(),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError(
            "great-circle roll/tilt differs from the active reference"
        )


def _derived_grid_peak_bytes(dataset: "RcsGrid", shape) -> int:
    cells = math.prod(int(value) for value in shape)
    itemsize = max(
        np.dtype(dataset.rcs_power.dtype).itemsize,
        np.dtype(dataset.rcs_phase.dtype).itemsize,
    )
    # Retained power+phase plus interpolation/constructor scratch.  This is a
    # deliberately conservative guard, not an exact allocator model.
    return int(cells * itemsize * 6)


def _derived_grid_memory_limit() -> int:
    available = _available_memory_bytes()
    if available is None:
        return _MAX_DERIVED_PEAK_BYTES_FALLBACK
    return max(64 * 1024**2, int(available * 0.5))


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or suffix == "TiB":
            return f"{size:.1f} {suffix}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _compact_item_summary(items, *, limit: int = 5) -> str:
    """Keep status-bar text bounded while still reporting the total count."""

    values = [str(item) for item in items]
    shown = values[: max(0, int(limit))]
    text = ", ".join(shown)
    remaining = len(values) - len(shown)
    if remaining:
        text += f", …and {remaining} more"
    return text


def _target_path_key(path: str | os.PathLike) -> str:
    """Return a platform-independent collision key for an output path.

    GRIM release folders are commonly prepared on Windows and then copied to
    another machine.  Always case-fold here rather than relying on the host
    filesystem so names such as ``Body`` and ``body`` cannot silently replace
    one another in a batch on a case-insensitive destination.
    """

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


def _append_provenance(existing: object, event: object) -> str:
    """Append one operation to durable history without duplicating it."""

    previous = str(existing or "").strip()
    addition = str(event or "").strip()
    if not addition:
        return previous
    if not previous:
        return addition
    if previous == addition or previous.endswith("\n" + addition):
        return previous
    return previous + "\n" + addition


_COHERENT_METADATA_LABELS = {
    "phase_reference": "phase reference / phase center",
    "time_convention": "phasor time convention",
    "polarization_basis": "polarization basis",
}


def _missing_coherent_metadata_keys(datasets) -> set[str]:
    """Return coherent declarations absent from any selected input."""

    missing: set[str] = set()
    for dataset in datasets:
        getter = getattr(dataset, "_declared_scalar_metadata", None)
        for key in _COHERENT_METADATA_LABELS:
            if callable(getter):
                value = getter(key)
            else:
                value = (dataset.extra or {}).get(
                    key, (dataset.units or {}).get(key, "")
                )
            if not str(value or "").strip():
                missing.add(key)
    return missing


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
    # Beginning/middle/end blocks avoid classifying a file from one unusually
    # smooth local region while keeping the sample strictly bounded.
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
    # Power and phase always participate. Add only the largest ancillary
    # payloads so hundreds of scalar metadata arrays cannot dilute their
    # representative sample.
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
    """Write every grid first, then atomically publish each completed file.

    Existing targets are moved to same-directory backups during publication so
    a late failure can restore the entire batch.  Callers must still obtain one
    explicit overwrite confirmation before invoking this helper.
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
                # A single RcsGrid may intentionally appear in multiple rows.
                # Serialize the provenance belonging to this row from a shallow
                # snapshot.  The large numerical arrays remain shared/read-only
                # for the duration of the save, while the live GUI object's
                # history is never mutated from the worker thread.
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
        # Restore every target whose original was moved, including the target
        # whose publication itself failed.  A backup that cannot be restored
        # remains beside the target and is reported instead of being deleted.
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


def _sanitize_filename(name: str | None) -> str:
    """Return a filesystem-safe version of `name` (UI display name unchanged)."""
    cleaned = _BAD_FILENAME_CHARS.sub("_", name or "").strip().strip(".")
    return cleaned or "dataset"


POLARIZATION_DISPLAY_ORDER = ("VV", "TE", "HH", "TM", "VH", "HV")
_POLARIZATION_DISPLAY_RANK = {
    polarization: index for index, polarization in enumerate(POLARIZATION_DISPLAY_ORDER)
}


def _polarization_display_sort_key(value: object, original_index: int) -> tuple[int, int]:
    label = str(value).strip().upper()
    rank = _POLARIZATION_DISPLAY_RANK.get(label, len(POLARIZATION_DISPLAY_ORDER))
    return rank, original_index


def _sorted_polarization_indices(values, indices) -> list[int]:
    return sorted(
        (int(idx) for idx in indices),
        key=lambda idx: _polarization_display_sort_key(values[idx], idx),
    )


def _sorted_polarization_values(values) -> list:
    return [values[idx] for idx in _sorted_polarization_indices(values, range(len(values)))]


def _wedge_to_conic_deg(phi_deg: np.ndarray, tau_deg: np.ndarray):
    """Compatibility wrapper around the tested dataset geometry kernel."""

    return wedge_to_conic_geometry_deg(phi_deg, tau_deg)


class AlignDialog(QDialog):
    """Choose alignment mode when aligning datasets to a reference."""

    def __init__(self, ref_name: str, n_others: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Align Datasets")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"Reference: <b>{ref_name}</b>  —  aligning {n_others} other dataset(s) to it."
        ))

        grp = QGroupBox("Alignment Mode")
        grp_layout = QVBoxLayout(grp)
        self._btn_group = QButtonGroup(self)
        self._radio_intersect = QRadioButton(
            "Intersect — keep only axis values present in both datasets (exact match, no interpolation)"
        )
        self._radio_interp = QRadioButton(
            "Interpolate — linearly interpolate to the reference axes (no extrapolation)"
        )
        self._radio_intersect.setChecked(True)
        self._btn_group.addButton(self._radio_intersect, 0)
        self._btn_group.addButton(self._radio_interp, 1)
        grp_layout.addWidget(self._radio_intersect)
        grp_layout.addWidget(self._radio_interp)
        layout.addWidget(grp)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_mode(self) -> str:
        return "interp" if self._radio_interp.isChecked() else "intersect"


class CropDialog(QDialog):
    """Choose selected-value slicing or physical ranges with exact strides."""

    def __init__(self, reference: "RcsGrid", *, has_selected_values: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop / Slice")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Create a smaller dataset without interpolation. Use parameter-list "
            "selections, or crop physical ranges and retain every Nth source sample."
        ))

        self._rb_selected = QRadioButton("Use values selected in the parameter lists")
        self._rb_ranges = QRadioButton("Use numeric ranges and exact source-sample strides")
        self._rb_selected.setEnabled(bool(has_selected_values))
        self._rb_selected.setChecked(bool(has_selected_values))
        self._rb_ranges.setChecked(not bool(has_selected_values))
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._rb_selected)
        mode_group.addButton(self._rb_ranges)
        layout.addWidget(self._rb_selected)
        layout.addWidget(self._rb_ranges)

        range_group = QGroupBox("Ranges")
        range_layout = QGridLayout(range_group)
        range_layout.addWidget(QLabel("Axis"), 0, 0)
        range_layout.addWidget(QLabel("Minimum"), 0, 1)
        range_layout.addWidget(QLabel("Maximum"), 0, 2)
        range_layout.addWidget(QLabel("Stride"), 0, 3)
        self._range_controls: dict[str, tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox, QSpinBox]] = {}

        az = _angle_axis_degrees(reference, "azimuth")
        el = _angle_axis_degrees(reference, "elevation")
        freq = np.asarray(reference.frequencies, dtype=float)
        frequency_unit = _canonical_frequency_unit(
            (reference.units or {}).get("frequency", "GHz")
        )
        if reference.angular_coordinate_system() == "great_circle":
            azimuth_label, elevation_label = "Aspect", "Pitch"
        else:
            azimuth_label, elevation_label = "Azimuth", "Elevation"
        specs = (
            ("azimuth", f"{azimuth_label} (deg)", az),
            ("elevation", f"{elevation_label} (deg)", el),
            ("frequency", f"Frequency ({frequency_unit})", freq),
        )
        for row, (axis, label, values) in enumerate(specs, start=1):
            enabled = QCheckBox(label)
            enabled.setChecked(True)
            minimum = QDoubleSpinBox()
            maximum = QDoubleSpinBox()
            for spin in (minimum, maximum):
                spin.setDecimals(9)
                spin.setRange(-1.0e300, 1.0e300)
                spin.setKeyboardTracking(False)
            minimum.setValue(float(np.min(values)))
            maximum.setValue(float(np.max(values)))
            stride = QSpinBox()
            stride.setRange(1, max(1, int(values.size)))
            stride.setValue(1)
            enabled.toggled.connect(minimum.setEnabled)
            enabled.toggled.connect(maximum.setEnabled)
            enabled.toggled.connect(stride.setEnabled)
            range_layout.addWidget(enabled, row, 0)
            range_layout.addWidget(minimum, row, 1)
            range_layout.addWidget(maximum, row, 2)
            range_layout.addWidget(stride, row, 3)
            self._range_controls[axis] = (enabled, minimum, maximum, stride)

        self._selected_polarizations = QCheckBox(
            "Limit output to polarizations selected in the parameter list"
        )
        self._selected_polarizations.setChecked(False)
        range_layout.addWidget(self._selected_polarizations, 4, 0, 1, 4)
        layout.addWidget(range_group)
        self._range_group = range_group
        range_group.setEnabled(self._rb_ranges.isChecked())
        self._rb_ranges.toggled.connect(range_group.setEnabled)

        note = QLabel(
            "Stride selects existing samples; it does not filter or invent values. "
            "Use Regrid when a specific coordinate grid is required."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict[str, object]:
        ranges: dict[str, tuple[float, float] | None] = {}
        strides: dict[str, int] = {}
        for axis, (enabled, minimum, maximum, stride) in self._range_controls.items():
            ranges[axis] = (
                (float(minimum.value()), float(maximum.value()))
                if enabled.isChecked()
                else None
            )
            strides[axis] = int(stride.value()) if enabled.isChecked() else 1
        return {
            "mode": "selected" if self._rb_selected.isChecked() else "ranges",
            "ranges": ranges,
            "strides": strides,
            "selected_polarizations": self._selected_polarizations.isChecked(),
        }


class RegridDialog(QDialog):
    """Pick one numeric axis and an explicit, uniformly spaced target grid."""

    _AXIS_LABELS = {
        "azimuth": "Azimuth",
        "elevation": "Elevation",
        "frequency": "Frequency",
    }

    def __init__(self, reference: "RcsGrid", parent=None) -> None:
        super().__init__(parent)
        self._reference = reference
        self.setWindowTitle("Regrid")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Linearly interpolate the complex field onto one new axis. GRIM never "
            "extrapolates; every selected dataset must cover the requested range."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        grid = QGridLayout()
        grid.addWidget(QLabel("Axis:"), 0, 0)
        self._axis = QComboBox()
        axis_labels = dict(self._AXIS_LABELS)
        if reference.angular_coordinate_system() == "great_circle":
            axis_labels.update(azimuth="Aspect", elevation="Pitch")
        for key in ("azimuth", "elevation", "frequency"):
            self._axis.addItem(axis_labels[key], key)
        grid.addWidget(self._axis, 0, 1)

        self._label_start = QLabel()
        self._label_stop = QLabel()
        self._label_step = QLabel()
        self._spin_start = QDoubleSpinBox()
        self._spin_stop = QDoubleSpinBox()
        self._spin_step = QDoubleSpinBox()
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.setDecimals(9)
            spin.setRange(-1.0e300, 1.0e300)
            spin.setKeyboardTracking(False)
        self._spin_step.setMinimum(1.0e-12)
        grid.addWidget(self._label_start, 1, 0)
        grid.addWidget(self._spin_start, 1, 1)
        grid.addWidget(self._label_stop, 2, 0)
        grid.addWidget(self._spin_stop, 2, 1)
        grid.addWidget(self._label_step, 3, 0)
        grid.addWidget(self._spin_step, 3, 1)
        layout.addLayout(grid)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: gray;")
        layout.addWidget(self._summary)
        self._axis.currentIndexChanged.connect(self._load_axis_defaults)
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.valueChanged.connect(self._update_summary)
        self._load_axis_defaults()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _axis_values_and_unit(self, axis: str) -> tuple[np.ndarray, str]:
        if axis in {"azimuth", "elevation"}:
            return _angle_axis_degrees(self._reference, axis), "deg"
        unit = _canonical_frequency_unit(
            (self._reference.units or {}).get("frequency", "GHz")
        )
        return np.asarray(self._reference.frequencies, dtype=float), unit

    def _load_axis_defaults(self, *_args) -> None:
        axis = str(self._axis.currentData())
        values, unit = self._axis_values_and_unit(axis)
        step = float(np.median(np.diff(values))) if values.size > 1 else 1.0
        step = abs(step) if np.isfinite(step) and step != 0.0 else 1.0
        for label, stem in (
            (self._label_start, "Start"),
            (self._label_stop, "Stop"),
            (self._label_step, "Step"),
        ):
            label.setText(f"{stem} ({unit}):")
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.blockSignals(True)
        self._spin_start.setValue(float(np.min(values)))
        self._spin_stop.setValue(float(np.max(values)))
        self._spin_step.setValue(step)
        for spin in (self._spin_start, self._spin_stop, self._spin_step):
            spin.blockSignals(False)
        self._update_summary()

    def _update_summary(self, *_args) -> None:
        start, stop, step = self.get_values()
        if step <= 0.0 or stop < start:
            self._summary.setText("Enter an increasing finite range and positive step.")
            return
        count = int(np.floor((stop - start) / step + 1.0e-9)) + 1
        resolved = start + max(0, count - 1) * step
        self._summary.setText(
            f"Resolved grid: {count:,} samples; final coordinate {resolved:.9g}. "
            "The stop value is not exceeded."
        )

    def get_values(self) -> tuple[float, float, float]:
        return (
            float(self._spin_start.value()),
            float(self._spin_stop.value()),
            float(self._spin_step.value()),
        )

    def get_params(self) -> dict[str, object]:
        start, stop, step = self.get_values()
        return {
            "axis": str(self._axis.currentData()),
            "start": start,
            "stop": stop,
            "step": step,
            "unit": self._axis_values_and_unit(str(self._axis.currentData()))[1],
        }


# Compatibility alias for extensions that imported the former dialog class.
InterpolateDialog = RegridDialog


class StitchDialog(QDialog):
    """Choose an explicit overlap policy for a union-grid merge."""

    def __init__(self, operand_names, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Overlapping Datasets")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        operands = QLabel("Operand order: " + " → ".join(map(str, operand_names)))
        operands.setWordWrap(True)
        layout.addWidget(operands)
        layout.addWidget(QLabel("When finite samples share the same grid cell:"))

        self._policy = QComboBox()
        self._policy.addItem("Priority: first operand wins", "priority-first")
        self._policy.addItem("Priority: last operand wins", "priority-last")
        self._policy.addItem(
            "Average linear power (overlap phase removed)", "power-mean"
        )
        self._policy.addItem("Average coherent complex field", "coherent-mean")
        layout.addWidget(self._policy)

        self._policy_help = QLabel()
        self._policy_help.setWordWrap(True)
        self._policy_help.setStyleSheet("color: gray;")
        layout.addWidget(self._policy_help)
        self._policy.currentIndexChanged.connect(self._update_help)
        self._update_help()

        tolerance_row = QHBoxLayout()
        tolerance_row.addWidget(QLabel("Native-axis matching tolerance:"))
        self._tolerance = QDoubleSpinBox()
        self._tolerance.setDecimals(12)
        self._tolerance.setRange(0.0, 1.0)
        self._tolerance.setValue(1.0e-6)
        self._tolerance.setSingleStep(1.0e-6)
        tolerance_row.addWidget(self._tolerance)
        tolerance_row.addStretch(1)
        layout.addLayout(tolerance_row)

        tolerance_help = QLabel(
            "One unitless number is applied independently to azimuth, elevation, "
            "and frequency in their declared native axis units. Selected datasets "
            "must therefore use the same units (for example, all degrees and GHz)."
        )
        tolerance_help.setWordWrap(True)
        tolerance_help.setStyleSheet("color: gray;")
        layout.addWidget(tolerance_help)
        self._tolerance_help = tolerance_help

        preview = QLabel(
            "GRIM computes the merged result and overlap report in the background, "
            "adds one new unsaved dataset, and reports resolved/equal/conflicting "
            "counts. The complete report is retained in dataset provenance."
        )
        preview.setWordWrap(True)
        layout.addWidget(preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_help(self, *_args) -> None:
        policy = str(self._policy.currentData())
        descriptions = {
            "priority-first": (
                "Conflicting overlaps use the first selected operand. Missing cells "
                "are still filled by later operands."
            ),
            "priority-last": (
                "Conflicting overlaps use the last selected operand. Selection order "
                "is therefore physically significant."
            ),
            "power-mean": (
                "Repeated measurements are averaged in linear power. Phase remains "
                "available in single-source cells and is marked unknown wherever "
                "multiple contributors were averaged."
            ),
            "coherent-mean": (
                "Complex fields are averaged. Axes and all declared phase, time, and "
                "polarization conventions must agree."
            ),
        }
        self._policy_help.setText(descriptions[policy])

    def get_params(self) -> dict[str, object]:
        return {
            "policy": str(self._policy.currentData()),
            "tol": float(self._tolerance.value()),
        }


class DatasetAuditDialog(QDialog):
    """Scrollable, copyable presentation of one or more audit reports."""

    def __init__(self, reports, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Audit")
        self.resize(760, 600)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Read-only audit: no samples or metadata were changed. FAIL indicates an "
            "invalid dataset; WARN identifies a condition worth reviewing."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setPlainText(self._format_reports(reports))
        layout.addWidget(text, 1)
        self.report_text = text
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _format_reports(reports) -> str:
        status_labels = {
            "ok": "PASS",
            "pass": "PASS",
            "warning": "WARN",
            "warn": "WARN",
            "error": "FAIL",
            "fail": "FAIL",
        }

        def render_value(value) -> str:
            if isinstance(value, float):
                return f"{value:.9g}"
            if isinstance(value, bool):
                return "yes" if value else "no"
            if value is None:
                return "not available"
            return str(value)

        def append_mapping(lines, mapping, indent: int) -> None:
            prefix = " " * indent
            for key in sorted(mapping):
                value = mapping[key]
                label = str(key).replace("_", " ")
                if isinstance(value, dict):
                    lines.append(f"{prefix}{label}:")
                    append_mapping(lines, value, indent + 2)
                else:
                    lines.append(f"{prefix}{label}: {render_value(value)}")

        blocks: list[str] = []
        for name, report in reports:
            raw_status = str(report.get("status", "unknown")).strip().lower()
            status = status_labels.get(raw_status, "WARN")
            lines = [f"{name}", f"Status: {status}"]
            for key, heading in (
                ("errors", "Errors"),
                ("warnings", "Warnings"),
                ("info", "Information"),
            ):
                values = report.get(key) or []
                if values:
                    lines.append(f"{heading}:")
                    for value in values:
                        if not isinstance(value, dict):
                            lines.append(f"  - {value}")
                            continue
                        code = str(value.get("code", "issue")).replace("_", " ")
                        message = str(value.get("message", ""))
                        details = {
                            detail_key: detail_value
                            for detail_key, detail_value in value.items()
                            if detail_key not in {"code", "message"}
                        }
                        suffix = ""
                        if details:
                            suffix = "; " + ", ".join(
                                f"{str(detail_key).replace('_', ' ')}="
                                f"{render_value(detail_value)}"
                                for detail_key, detail_value in sorted(details.items())
                            )
                        lines.append(f"  - [{code}] {message}{suffix}")
            metrics = report.get("metrics") or {}
            if metrics:
                lines.append("Metrics:")
                append_mapping(lines, metrics, 2)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


class DatasetCompatibilityDialog(QDialog):
    """Copyable preflight report for selected multi-dataset operations."""

    def __init__(self, report_text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Compatibility")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Read-only preflight against operand 1. PASS means the tested "
            "contract is satisfied; WARN requires a recorded assumption."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setPlainText(str(report_text))
        layout.addWidget(text, 1)
        self.report_text = text
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DatasetProvenanceDialog(QDialog):
    """Copyable, bounded view of lineage and metadata for selected rows."""

    def __init__(self, report_text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Provenance")
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Large metadata arrays are described by shape, type, and size rather "
            "than expanded. JSON provenance records are formatted for review."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        text.setPlainText(str(report_text))
        layout.addWidget(text, 1)
        self.report_text = text
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ShiftDialog(QDialog):
    """Pick which axes (and/or RCS phase) to shift and by what amount.

    Azimuth/Elevation translate the corresponding axis values (degrees).
    Phase rotates every complex sample by exp(j·θ) (degrees) — it doesn't
    move axis values, but it lives here as the sole "shift the data
    instead of an axis" option to keep the UI consolidated.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shift")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select what to shift:"))

        def make_row(label_text: str, checked: bool, suffix: str = " °") -> tuple:
            row = QHBoxLayout()
            chk = QCheckBox(label_text)
            chk.setChecked(checked)
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(1.0)
            spin.setValue(0.0)
            spin.setSuffix(suffix)
            spin.setEnabled(checked)
            chk.toggled.connect(spin.setEnabled)
            row.addWidget(chk)
            row.addWidget(spin)
            layout.addLayout(row)
            return chk, spin

        self._chk_az,    self._spin_az    = make_row("Azimuth",   True)
        self._chk_el,    self._spin_el    = make_row("Elevation", False)
        self._chk_phase, self._spin_phase = make_row("Phase",     False)
        # Phase is bounded to one full rotation since shift is mod 360°.
        self._spin_phase.setRange(-360.0, 360.0)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "azimuth":   (self._chk_az.isChecked(),    float(self._spin_az.value())),
            "elevation": (self._chk_el.isChecked(),    float(self._spin_el.value())),
            "phase":     (self._chk_phase.isChecked(), float(self._spin_phase.value())),
        }


class RangeCalibrationDialog(QDialog):
    """Assign loaded grids to the measured/exact calibration roles."""

    def __init__(self, dataset_entries, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Range Cal — Complex Substitution")
        self._entries = list(dataset_entries)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Selected table rows are the DUT measurement(s) to calibrate. "
            "Choose a measured calibration target and its trusted complex "
            "exact/reference response from the loaded datasets."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QGridLayout()
        self.combo_measured = QComboBox()
        self.combo_exact = QComboBox()
        for row_index, (name, _dataset) in enumerate(self._entries, start=1):
            label = f"[{row_index}] {name}"
            self.combo_measured.addItem(label)
            self.combo_exact.addItem(label)
        if len(self._entries) > 1:
            self.combo_exact.setCurrentIndex(1)
        form.addWidget(QLabel("Measured calibration target:"), 0, 0)
        form.addWidget(self.combo_measured, 0, 1)
        form.addWidget(QLabel("Exact/reference response:"), 1, 0)
        form.addWidget(self.combo_exact, 1, 1)

        self.spin_offset_m = QDoubleSpinBox()
        self.spin_offset_m.setDecimals(9)
        self.spin_offset_m.setRange(-1.0e6, 1.0e6)
        self.spin_offset_m.setSingleStep(0.001)
        self.spin_offset_m.setSuffix(" m")
        self.spin_offset_m.setToolTip(
            "Enter the one-way physical displacement. Positive means the "
            "measured calibration target is farther from radar than the "
            "DUT/reference plane; GRIM applies the monostatic two-way phase."
        )
        form.addWidget(QLabel("Signed calibrator range offset ΔR:"), 2, 0)
        form.addWidget(self.spin_offset_m, 2, 1)

        gain_row = QHBoxLayout()
        self.chk_gain_limit = QCheckBox("Limit correction gain")
        self.chk_gain_limit.setChecked(True)
        self.spin_gain_limit_db = QDoubleSpinBox()
        self.spin_gain_limit_db.setDecimals(1)
        self.spin_gain_limit_db.setRange(0.0, 300.0)
        self.spin_gain_limit_db.setValue(60.0)
        self.spin_gain_limit_db.setSuffix(" dB")
        self.spin_gain_limit_db.setToolTip(
            "Mask calibration bins whose |Aexact/Ameasured| correction exceeds "
            "this level. Other usable bins are still calibrated."
        )
        self.chk_gain_limit.toggled.connect(self.spin_gain_limit_db.setEnabled)
        gain_row.addWidget(self.chk_gain_limit)
        gain_row.addWidget(self.spin_gain_limit_db)
        form.addWidget(QLabel("Calibration bin masking:"), 3, 0)
        form.addLayout(gain_row, 3, 1)
        layout.addLayout(form)

        phase_law = QLabel(
            "Positive ΔR is away from radar. GRIM applies "
            "Aout = Adut · Aexact · exp(−j4πfΔR/c) / Ameasured."
        )
        phase_law.setWordWrap(True)
        layout.addWidget(phase_law)

        self.chk_broadcast = QCheckBox(
            "Broadcast singleton calibration azimuth/elevation across DUT angles"
        )
        self.chk_broadcast.setToolTip(
            "No angular averaging or interpolation is performed. Enable only "
            "when one frequency/polarization correction applies to every DUT look."
        )
        layout.addWidget(self.chk_broadcast)

        assumption_note = QLabel(
            "Selecting these roles requests complex calibration. Missing acquisition "
            "or phase-center declarations are recorded as assumptions; explicit "
            "incompatible units, axes, quantities, or phase signs still stop the job."
        )
        assumption_note.setWordWrap(True)
        layout.addWidget(assumption_note)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)

        warning = QLabel(
            "The exact response must be complex sigma₃D/dBsm data. A finite "
            "cylinder's 3-D reference must be supplied; GRIM will not substitute "
            "GHOST's infinite 2-D cylinder solution. Invalid/null correction bins "
            "are masked and reported."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.combo_measured.currentIndexChanged.connect(self._update_validity)
        self.combo_exact.currentIndexChanged.connect(self._update_validity)
        self._update_validity()

    def _update_validity(self, *_args) -> None:
        same_reference = (
            self.combo_measured.currentIndex() == self.combo_exact.currentIndex()
        )
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        ok_button.setEnabled(not same_reference)
        if same_reference:
            self.validation_label.setText(
                "Choose different datasets for measured calibration and exact reference."
            )
        else:
            self.validation_label.setText(
                "Ready. Missing provenance will be recorded as assumed, not blocked."
            )

    def get_params(self) -> dict:
        measured_index = int(self.combo_measured.currentIndex())
        exact_index = int(self.combo_exact.currentIndex())
        return {
            "measured": self._entries[measured_index],
            "exact": self._entries[exact_index],
            "range_offset_m": float(self.spin_offset_m.value()),
            "allow_singleton_angular_broadcast": self.chk_broadcast.isChecked(),
            "convention_attested": False,
            "maximum_correction_gain_db": (
                float(self.spin_gain_limit_db.value())
                if self.chk_gain_limit.isChecked()
                else None
            ),
        }


class SupportReferenceDifferenceDialog(QDialog):
    """Assign the two physical roles for guided support-reference subtraction."""

    def __init__(self, entries, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Support-Referenced Difference")
        self.setMinimumWidth(640)
        self._entries = list(entries)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Create an unsaved derived dataset using the exact complex operation "
            "A(target + support) - A(support-only). Select the physical roles "
            "below; GRIM will not interpolate, regrid, or change phase."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        role_box = QGroupBox("1. Assign acquisition roles")
        role_layout = QGridLayout(role_box)
        self.combo_target = QComboBox()
        self.combo_support = QComboBox()
        for row_index, (name, _dataset) in enumerate(self._entries, start=1):
            label = f"[{row_index}] {name}"
            self.combo_target.addItem(label)
            self.combo_support.addItem(label)
        if len(self._entries) > 1:
            self.combo_support.setCurrentIndex(1)
        role_layout.addWidget(QLabel("Target + support acquisition:"), 0, 0)
        role_layout.addWidget(self.combo_target, 0, 1)
        role_layout.addWidget(QLabel("Support-only reference:"), 1, 0)
        role_layout.addWidget(self.combo_support, 1, 1)
        layout.addWidget(role_box)

        self.compatibility_label = QLabel("")
        self.compatibility_label.setWordWrap(True)
        self.compatibility_label.setObjectName("supportReferenceCompatibility")
        layout.addWidget(self.compatibility_label)

        interpretation = QLabel(
            "The selected roles request exact complex subtraction. Missing acquisition "
            "metadata is recorded as assumed. The result is support-referenced, not a "
            "reconstructed free-space target: coupling, shadowing, multiple bounce, "
            "and acquisition drift cannot be recovered from two files."
        )
        interpretation.setWordWrap(True)
        layout.addWidget(interpretation)

        note = QLabel(
            "The output is added as a new unsaved row after calculation. Neither "
            "input is modified. QA metrics and assumptions are stored with the "
            "result; they do not prove physical support removal."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.combo_target.currentIndexChanged.connect(self._update_validity)
        self.combo_support.currentIndexChanged.connect(self._update_validity)
        self._update_validity()

    def _selected_entries(self):
        if not self._entries:
            return None, None
        target_index = int(self.combo_target.currentIndex())
        support_index = int(self.combo_support.currentIndex())
        if target_index < 0 or support_index < 0:
            return None, None
        return (
            self._entries[target_index],
            self._entries[support_index],
        )

    def _update_validity(self, *_args) -> None:
        target_entry, support_entry = self._selected_entries()
        valid = False
        message = "Select two different datasets."
        if target_entry is not None and support_entry is not None:
            if self.combo_target.currentIndex() == self.combo_support.currentIndex():
                message = (
                    "Target + support and support-only must be different rows."
                )
            else:
                target = target_entry[1]
                support = support_entry[1]
                missing = _missing_coherent_metadata_keys((target, support))
                acquisition_contract = None
                try:
                    target._assert_compatible(
                        support,
                        coherent=True,
                        coherent_metadata_attested=False,
                        _scan_phase_samples=False,
                    )
                    acquisition_contract = target._assert_support_reference_metadata_compatible(
                        support
                    )
                except (TypeError, ValueError) as exc:
                    message = (
                        "Not compatible for exact complex subtraction: " + str(exc)
                    )
                else:
                    acquisition_missing = dict(
                        acquisition_contract.get(
                            "missing_declarations_by_role", {}
                        )
                    )
                    valid = True
                    if missing or acquisition_missing:
                        coherent_labels = ", ".join(
                            _COHERENT_METADATA_LABELS[key]
                            for key in _COHERENT_METADATA_LABELS
                            if key in missing
                        )
                        semantic_families = acquisition_contract.get(
                            "semantic_families", {}
                        )
                        acquisition_labels = sorted(
                            {
                                str(
                                    semantic_families.get(
                                        fact.split(".", 1)[0], {}
                                    ).get("label", fact.split(".", 1)[0])
                                )
                                for fact in acquisition_missing
                            }
                        )
                        missing_sections = []
                        if coherent_labels:
                            missing_sections.append(
                                "coherent conventions: " + coherent_labels
                            )
                        if acquisition_labels:
                            missing_sections.append(
                                "acquisition/setup declarations: "
                                + ", ".join(acquisition_labels)
                            )
                        message = (
                            "Exact axes are compatible and no explicit declaration "
                            "contradicts the other input. "
                            "The full finite-phase sample scan will run in the "
                            "background before subtraction. "
                            "The following missing declarations will be recorded "
                            "as operation assumptions: "
                            + "; ".join(missing_sections)
                            + "."
                        )
                    else:
                        message = (
                            "Ready: axes, units, coherent metadata, coordinate frame, "
                            "phase reference, time convention, and polarization "
                            "basis are compatible. Full finite-phase sample QA will "
                            "run in the background before subtraction."
                        )
        self.compatibility_label.setText(message)
        ok = self.buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled(valid)

    def get_params(self) -> dict:
        target_entry, support_entry = self._selected_entries()
        return {
            "target": target_entry,
            "support": support_entry,
            "metadata_attested": False,
            "assumptions_attested": False,
        }


class RoundDialog(QDialog):
    """Pick which axes to round and at what decimal precision."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Round Axes")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select axes to round:"))
        self._chk_az = QCheckBox("Azimuths")
        self._chk_el = QCheckBox("Elevations")
        self._chk_fr = QCheckBox("Frequencies")
        self._chk_az.setChecked(True)
        self._chk_el.setChecked(True)
        self._chk_fr.setChecked(True)
        layout.addWidget(self._chk_az)
        layout.addWidget(self._chk_el)
        layout.addWidget(self._chk_fr)

        decimals_row = QHBoxLayout()
        decimals_row.addWidget(QLabel("Decimal places:"))
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(0)
        self._spin.setRange(0, 9)
        self._spin.setValue(1)
        self._spin.setSingleStep(1)
        decimals_row.addWidget(self._spin)
        decimals_row.addStretch(1)
        layout.addLayout(decimals_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "azimuths": self._chk_az.isChecked(),
            "elevations": self._chk_el.isChecked(),
            "frequencies": self._chk_fr.isChecked(),
            "decimals": int(self._spin.value()),
        }


class WrapDialog(QDialog):
    """Wrap the azimuth coordinate, stored phase, or both."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wrap")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose what to wrap:"))

        self._wrap_azimuth = QCheckBox("Azimuth axis")
        self._wrap_phase = QCheckBox("Phase values")
        self._wrap_azimuth.setChecked(True)
        self._wrap_phase.setChecked(False)
        layout.addWidget(self._wrap_azimuth)
        layout.addWidget(self._wrap_phase)

        layout.addWidget(QLabel("Target interval:"))

        self._rb_0_360 = QRadioButton("[0°, 360°)")
        self._rb_pm180 = QRadioButton("[-180°, 180°)")
        self._rb_0_360.setChecked(True)
        layout.addWidget(self._rb_0_360)
        layout.addWidget(self._rb_pm180)

        group = QButtonGroup(self)
        group.addButton(self._rb_0_360)
        group.addButton(self._rb_pm180)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_mode(self) -> str:
        return "0_360" if self._rb_0_360.isChecked() else "-180_180"

    def get_params(self) -> dict[str, object]:
        return {
            "azimuth": self._wrap_azimuth.isChecked(),
            "phase": self._wrap_phase.isChecked(),
            "mode": self.get_mode(),
        }


class DecimateDialog(QDialog):
    """Configure boxcar-prefiltered integer-factor downsampling."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Decimate with Prefilter")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Average adjacent bins on a uniformly sampled source axis before "
            "retaining one output bin. This avoids the unfiltered point-sampling "
            "behavior of a coarse Regrid."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        self._axis = QComboBox()
        self._axis.addItem("Azimuth / Aspect", "azimuth")
        self._axis.addItem("Elevation / Pitch", "elevation")
        self._axis.addItem("Frequency", "frequency")
        self._factor = QSpinBox()
        self._factor.setRange(2, 1_000_000)
        self._factor.setValue(2)
        self._mode = QComboBox()
        self._mode.addItem("Linear-power mean (phase becomes unknown)", "power")
        self._mode.addItem("Coherent complex-field mean", "coherent")
        grid.addWidget(QLabel("Axis:"), 0, 0)
        grid.addWidget(self._axis, 0, 1)
        grid.addWidget(QLabel("Integer factor:"), 1, 0)
        grid.addWidget(self._factor, 1, 1)
        grid.addWidget(QLabel("Filter domain:"), 2, 0)
        grid.addWidget(self._mode, 2, 1)
        layout.addLayout(grid)
        warning = QLabel(
            "The final partial bin is retained using its actual sample count. "
            "A coherent mean requires finite phase and represents filtered field, "
            "not averaged RCS power."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict[str, object]:
        return {
            "axis": str(self._axis.currentData()),
            "factor": int(self._factor.value()),
            "mode": str(self._mode.currentData()),
        }


class MedianizeDialog(QDialog):
    """Pick the sliding-window parameters for a median smoothing pass along
    the azimuth axis.

    Window = full azimuth width of each window (degrees), centred on each
    output sample. Slide = step between adjacent window centres (degrees).
    Slide < window gives overlap (heavier smoothing, denser output); slide =
    window gives non-overlapping bins; slide > window subsamples the input.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Medianize")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Sliding median over azimuth — replaces samples within each "
            "window with the median linear σ inside it."
        ))

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Window (deg):"))
        self._spin_window = QDoubleSpinBox()
        self._spin_window.setDecimals(4)
        self._spin_window.setRange(1.0e-4, 360.0)
        self._spin_window.setSingleStep(0.1)
        self._spin_window.setValue(5.0)
        win_row.addWidget(self._spin_window)
        win_row.addStretch(1)
        layout.addLayout(win_row)

        slide_row = QHBoxLayout()
        slide_row.addWidget(QLabel("Slide (deg):"))
        self._spin_slide = QDoubleSpinBox()
        self._spin_slide.setDecimals(4)
        self._spin_slide.setRange(1.0e-4, 360.0)
        self._spin_slide.setSingleStep(0.1)
        self._spin_slide.setValue(1.0)
        slide_row.addWidget(self._spin_slide)
        slide_row.addStretch(1)
        layout.addLayout(slide_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "window_deg": float(self._spin_window.value()),
            "slide_deg": float(self._spin_slide.value()),
        }


class ExtrusionLengthDialog(QDialog):
    """Ask the user for the extrusion length L (with units) when converting a
    3D dBsm measurement into the 2D scattering-width dBke representation.

    The conversion assumes broadside illumination of a uniform extruded body
    and uses the textbook relation  σ_3D = (2 L² / λ) · σ_2D , so the linear-
    sigma scale applied per frequency bin is λ_f / (2 L²) (c / (2 L² f) in Hz).
    """

    _UNIT_TO_M = {"m": 1.0, "in": 0.0254, "ft": 0.3048}

    def __init__(
        self,
        parent=None,
        *,
        title: str = "Convert dBsm → dBke",
        formula: str = (
            "σ_2D = σ_3D · λ / (2 L²) → dBke = dBsm + 10·log₁₀(π / L²) "
            "(frequency-independent offset)."
        ),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Extrusion length L (assumes broadside illumination of a uniform extruded body):"
        ))

        row = QHBoxLayout()
        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(6)
        self._spin.setRange(1.0e-6, 1.0e6)
        self._spin.setSingleStep(1.0)
        self._spin.setValue(24.0)
        row.addWidget(self._spin)
        self._combo = QComboBox()
        self._combo.addItems(["in", "ft", "m"])
        self._combo.setCurrentText("in")
        row.addWidget(self._combo)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QLabel(formula))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def length_m(self) -> float:
        unit = self._combo.currentText().strip().lower()
        factor = self._UNIT_TO_M.get(unit, 1.0)
        return float(self._spin.value()) * factor

    def display_text(self) -> str:
        return f"{float(self._spin.value()):g} {self._combo.currentText()}"


class AxisUnitsDialog(QDialog):
    """Choose equivalent storage units for all three numeric axes."""

    def __init__(self, reference: RcsGrid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert Axis Units")
        layout = QVBoxLayout(self)
        note = QLabel(
            "This is an exact unit conversion. Physical coordinates and all "
            "RCS samples remain unchanged; no interpolation is performed."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        self._azimuth = QComboBox()
        self._elevation = QComboBox()
        self._frequency = QComboBox()
        for combo in (self._azimuth, self._elevation):
            combo.addItems(["deg", "rad"])
        self._frequency.addItems(["Hz", "kHz", "MHz", "GHz"])
        current_az = _canonical_angle_unit(
            (reference.units or {}).get("azimuth", "deg")
        )
        current_el = _canonical_angle_unit(
            (reference.units or {}).get("elevation", "deg")
        )
        current_frequency = _canonical_frequency_unit(
            (reference.units or {}).get("frequency", "GHz")
        )
        self._azimuth.setCurrentText(current_az)
        self._elevation.setCurrentText(current_el)
        self._frequency.setCurrentText(current_frequency)
        for row, (label, combo) in enumerate((
            ("Azimuth / Aspect:", self._azimuth),
            ("Elevation / Pitch:", self._elevation),
            ("Frequency:", self._frequency),
        )):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(combo, row, 1)
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> dict[str, str]:
        return {
            "azimuth": self._azimuth.currentText(),
            "elevation": self._elevation.currentText(),
            "frequency": self._frequency.currentText(),
        }


class CoordinateSystemDialog(QDialog):
    """Let the user correct an imported coordinate interpretation."""

    def __init__(self, reference: RcsGrid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Coordinates")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Choose what the stored angles represent, regardless of file format. "
            "Creates selected copies with unchanged angle values, sample order, "
            "power, and phase. This does not perform a coordinate conversion."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self._system = QComboBox()
        self._system.addItem("Azimuth / Elevation (Conic)", "conic")
        self._system.addItem("Aspect / Pitch (Great Circle)", "great_circle")
        self._system.setCurrentIndex(
            max(0, self._system.findData(reference.angular_coordinate_system()))
        )
        layout.addWidget(self._system)
        self._gc_options = QGroupBox("Great-circle frame")
        frame_layout = QGridLayout(self._gc_options)
        self._convention = QComboBox()
        self._convention.addItem("PTM / unspecified", LEGACY_PTM_GC_CONVENTION)
        self._convention.addItem("GRIM convention", GRIM_GC_CONVENTION)
        self._convention.setCurrentIndex(max(
            0, self._convention.findData(reference.great_circle_coordinate_convention())
        ))
        frame_layout.addWidget(QLabel("Convention"), 0, 0)
        frame_layout.addWidget(self._convention, 0, 1)
        roll, tilt = reference.angular_frame_orientation_deg()
        self._roll, self._tilt = QDoubleSpinBox(), QDoubleSpinBox()
        for row, label, spin, value in (
            (1, "Roll (deg)", self._roll, roll),
            (2, "Tilt (deg)", self._tilt, tilt),
        ):
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(6)
            spin.setValue(value)
            frame_layout.addWidget(QLabel(label), row, 0)
            frame_layout.addWidget(spin, row, 1)
        layout.addWidget(self._gc_options)
        self._system.currentIndexChanged.connect(self._update_frame_options)
        self._update_frame_options()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_frame_options(self, *_args) -> None:
        self._gc_options.setVisible(self._system.currentData() == "great_circle")

    def get_params(self) -> dict:
        params = {"coordinate_system": self._system.currentData()}
        if params["coordinate_system"] == "great_circle":
            params.update(
                gc_convention=self._convention.currentData(),
                roll_deg=self._roll.value(), tilt_deg=self._tilt.value(),
            )
        return params


class ConicGCDialog(QDialog):
    """Expose only GRIM's exact, convention-tagged equatorial relabel.

    A general conic/great-circle conversion changes both the sampling path and
    polarization basis.  GRIM does not yet implement the full scattering-matrix
    basis rotation, and a fixed-pitch PTM cut maps to a curved conic path rather
    than a rectangular :class:`RcsGrid`.  The one exact exception is an
    unrotated, zero-pitch, co-polar great-circle cut, which can be relabeled as
    the same conic equatorial cut without changing VV or HH.
    """

    def __init__(
        self,
        source_coordinate_system=None,
        source_gc_convention=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert Conic ↔ Great-Circle (Equator)")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "This is an exact tag change only: one 0° elevation/pitch, stored "
            "roll=tilt=0°, and VV/HH. GRIM defines signed GC aspect equal to "
            "conic azimuth on this plane; no field interpolation occurs. General "
            "GC cuts need a curved-path representation and Jones-basis rotation."
        ))

        dir_group = QGroupBox("Direction")
        dir_layout = QVBoxLayout(dir_group)
        self._radio_c2g = QRadioButton(
            "Conic → Great-Circle (create a GRIM_GC_V1-tagged equatorial cut)"
        )
        self._radio_g2c = QRadioButton(
            "Great-Circle → Conic (same exact equatorial convention)"
        )
        if canonical_angular_coordinate_system(source_coordinate_system) == "great_circle":
            self._radio_g2c.setChecked(True)
        else:
            self._radio_c2g.setChecked(True)
        dir_layout.addWidget(self._radio_c2g)
        dir_layout.addWidget(self._radio_g2c)
        layout.addWidget(dir_group)

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        self._radio_relabel = QRadioButton("Exact equatorial relabel (no interpolation)")
        self._radio_regrid = QRadioButton(
            "General re-grid (unavailable until polarization-basis rotation is implemented)"
        )
        self._radio_relabel.setChecked(True)
        self._radio_regrid.setEnabled(False)
        mode_layout.addWidget(self._radio_relabel)
        mode_layout.addWidget(self._radio_regrid)
        layout.addWidget(mode_group)

        convention_note = QLabel(
            "For an unmarked legacy PTM, choosing GC→Conic records GRIM_GC_V1 "
            "as an operation assumption. Explicitly incompatible convention tags "
            "remain unsupported."
        )
        convention_note.setWordWrap(True)
        layout.addWidget(convention_note)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "direction": (
                "conic_to_gc" if self._radio_c2g.isChecked() else "gc_to_conic"
            ),
            "mode": "relabel",
            "attest_legacy_ptm_convention": False,
        }


class WedgeConicDialog(QDialog):
    """Confirm the physical conventions for a wedge-to-conic re-grid.

    Geometry: vertical-axis turntable (axis = world-z, fixed), target tilted
    by a foam wedge with ridge along body-y (pitch wedge). The current
    `azimuths` axis holds the turntable angle φ; `elevations` holds the wedge
    tilt τ. Output (azimuths, elevations) become true conic (longitude φ',
    latitude θ') on the body sphere.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wedge → Conic")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Input axes: azimuth = turntable angle φ, elevation = wedge tilt τ.\n"
            "Output axes: normal-range conic azimuth/elevation. The converter "
            "uses the full complex Jones matrix, rotates V/H, and leaves "
            "unsupported normal-range looks as NaN.\n\n"
            "A single wedge tilt cannot produce a normal constant-elevation "
            "azimuth sweep; two or more measured tilts are required."
        ))

        axes_note = QLabel(
            "Choosing this operation treats azimuth as rotation about fixed world "
            "+z and elevation as article pitch about body +y. If the source lacks "
            "an axis tag, that assumption is stored with the converted dataset."
        )
        axes_note.setWordWrap(True)
        layout.addWidget(axes_note)
        self._chk_cross_zero = QCheckBox(
            "Assume missing VH/HV is exactly zero (only use when justified)."
        )
        layout.addWidget(self._chk_cross_zero)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "mode": "regrid",
            "attest_wedge_axes": False,
            "assume_missing_cross_pol_zero": self._chk_cross_zero.isChecked(),
        }


class ExportCsvDialog(QDialog):
    """Options for exporting RCS data to a CSV file."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export to CSV")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel("Magnitude:"), 0, 0)
        self._combo_scale = QComboBox()
        self._combo_scale.addItem("Linear", "linear")
        self._combo_scale.addItem("dB (dimensionless ratio)", "db")
        self._combo_scale.addItem("dBsm", "dbsm")
        self._combo_scale.addItem("dBke", "dbke")
        self._combo_scale.addItem("Both (Linear + dataset's physical dB unit)", "both")
        grid.addWidget(self._combo_scale, 0, 1)

        layout.addLayout(grid)

        self._chk_phase = QCheckBox("Include phase column (degrees)")
        self._chk_phase.setChecked(True)
        layout.addWidget(self._chk_phase)

        layout.addWidget(QLabel(
            "Writes versioned GRIM flat RCS CSV with explicit axis units, physical "
            "quantity, coordinate convention, and coherent metadata.\n"
            "For dBke export, frequency-dependent conversion uses the dataset frequency axis.\n"
            "One row per sample — all combinations of dataset axes."
        ))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_options(self) -> tuple[str, bool]:
        """Return (scale, include_phase)."""
        return (
            self._combo_scale.currentData(),
            self._chk_phase.isChecked(),
        )


class StatisticsDialog(QDialog):
    """Single dialog for statistics dataset: all options in one place."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistics Dataset")
        layout = QVBoxLayout(self)

        params_grid = QGridLayout()

        params_grid.addWidget(QLabel("Statistic:"), 0, 0)
        self.combo_stat = QComboBox()
        self.combo_stat.addItems(["mean", "median", "min", "max", "std", "percentile"])
        params_grid.addWidget(self.combo_stat, 0, 1)

        params_grid.addWidget(QLabel("Percentile:"), 0, 2)
        self.spin_pct = QDoubleSpinBox()
        self.spin_pct.setRange(0.0, 100.0)
        self.spin_pct.setDecimals(1)
        self.spin_pct.setSingleStep(5.0)
        self.spin_pct.setValue(50.0)
        self.spin_pct.setEnabled(False)
        self.spin_pct.setToolTip("Only used when Statistic = percentile")
        params_grid.addWidget(self.spin_pct, 0, 3)

        layout.addLayout(params_grid)

        axes_group = QGroupBox("Axes to Reduce")
        axes_row = QHBoxLayout(axes_group)
        self.chk_az = QCheckBox("Azimuth")
        self.chk_az.setChecked(True)
        self.chk_el = QCheckBox("Elevation")
        self.chk_el.setChecked(True)
        self.chk_freq = QCheckBox("Frequency")
        self.chk_freq.setChecked(True)
        self.chk_pol = QCheckBox("Polarization")
        self.chk_pol.setChecked(False)
        for chk in (self.chk_az, self.chk_el, self.chk_freq, self.chk_pol):
            axes_row.addWidget(chk)
        axes_row.addStretch(1)
        layout.addWidget(axes_group)

        domain_note = QLabel(
            "Statistics are computed on linear power, not on displayed dB values. "
            "The reduced result has undefined coherent phase."
        )
        domain_note.setWordWrap(True)
        domain_note.setToolTip(
            "For example, converting the mean linear power to dB is generally not "
            "the same as averaging sample values after converting each one to dB."
        )
        layout.addWidget(domain_note)

        self.chk_broadcast = QCheckBox(
            "Repeat the reduced value across the original grid (larger file)"
        )
        self.chk_broadcast.setChecked(False)
        self.chk_broadcast.setToolTip(
            "Normally a reduction produces singleton axes and a compact dataset. "
            "Enable this only when a downstream workflow requires the statistic "
            "to be repeated at every original coordinate."
        )
        layout.addWidget(self.chk_broadcast)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.combo_stat.currentTextChanged.connect(
            lambda t: self.spin_pct.setEnabled(t == "percentile")
        )

    def get_params(self) -> tuple[str, float, list[str], bool]:
        """Return (statistic, percentile, axes, broadcast_reduced)."""
        statistic = self.combo_stat.currentText()
        percentile = self.spin_pct.value()
        axes = [
            name
            for chk, name in (
                (self.chk_az, "azimuth"),
                (self.chk_el, "elevation"),
                (self.chk_freq, "frequency"),
                (self.chk_pol, "polarization"),
            )
            if chk.isChecked()
        ]
        return statistic, percentile, axes, self.chk_broadcast.isChecked()



def _dataset_with_rcs(
    dataset: "RcsGrid",
    rcs,
    *,
    rcs_power=None,
    rcs_domain: str | None = None,
) -> "RcsGrid":
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs,
        rcs_power=rcs_power,
        rcs_domain=(dataset.rcs_domain if rcs_domain is None else rcs_domain),
        source_path=dataset.source_path,
        history=dataset.history,
        units=dict(dataset.units or {}),
        extra=_derived_response_extra(dataset),
    )


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


def _load_dataset_csv(path: str) -> "RcsGrid":
    """Compatibility name for the one authoritative flat-CSV parser."""

    return load_flat_csv_headless(path)


def _load_dataset_from_dropped_text(path: str) -> tuple["RcsGrid", str]:
    """Compatibility wrapper around the authoritative headless dispatcher."""

    dataset = load_dataset_headless(path)
    history = str(getattr(dataset, "history", "") or "").strip()
    if not history:
        history = f"Imported dataset: {path}"
    return dataset, history


def _is_supported_dataset_path(path: str) -> bool:
    return is_supported_path(path)


def _available_memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError):
        pass

    # GRIM is commonly copied to a clean workstation where psutil is not yet
    # installed.  Retain a real memory budget on the two primary deployment
    # families instead of falling back immediately to CPU-count concurrency.
    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = (
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                )

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages >= 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None


_LOADER_MIN_WORKING_BYTES = 64 * 1024**2
_LOADER_UNKNOWN_MEMORY_BUDGET_BYTES = 512 * 1024**2
_GRIM_LOAD_PEAK_FACTOR = 3.5
_TEXT_LOADER_EXTENSIONS = (".csv", ".txt", ".cst_data", ".out", ".ss")


def _grim_archive_uncompressed_bytes(path: str) -> int | None:
    """Return declared uncompressed NPZ bytes without extracting the archive."""

    if not str(path).casefold().endswith(".grim"):
        return None
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if not members:
                return 0
            return sum(max(0, int(member.file_size)) for member in members)
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        # The authoritative loader will report a malformed archive.  Keep a
        # conservative file-size fallback here so planning itself remains
        # non-destructive and does not mask that parse error.
        return None


def _dataset_load_memory_estimate(path: str) -> tuple[int, int]:
    """Return conservative ``(retained, per-load peak)`` byte estimates."""

    try:
        stored_bytes = max(0, int(os.path.getsize(path)))
    except OSError:
        stored_bytes = 0
    expanded_bytes = _grim_archive_uncompressed_bytes(path)
    if expanded_bytes is not None:
        retained = max(stored_bytes, int(expanded_bytes))
        # RcsGrid validates and cleans power/phase into new arrays while the
        # NPZ members are still live.  Include those transient copies instead
        # of treating a highly compressed archive as its on-disk byte count.
        peak = max(
            _LOADER_MIN_WORKING_BYTES,
            int(math.ceil(_GRIM_LOAD_PEAK_FACTOR * retained)),
        )
        return retained, peak

    lower = str(path).casefold()
    # Delimited readers create Python strings/dicts, coordinate-key maps, dense
    # output arrays, and duplicate-validation state concurrently. Real SENTRi
    # imports have measured around 77x their file size at peak, so the old 4x
    # rule was unsafe by more than an order of magnitude.
    if lower.endswith((".csv", ".txt")):
        retained = max(stored_bytes, 16 * stored_bytes)
        peak = max(_LOADER_MIN_WORKING_BYTES, 96 * stored_bytes)
        return retained, peak
    if lower.endswith((".cst_data", ".out", ".ss")):
        retained = max(stored_bytes, 8 * stored_bytes)
        peak = max(_LOADER_MIN_WORKING_BYTES, 40 * stored_bytes)
        return retained, peak

    # Binary single-cut formats still materialize complex, power, and phase
    # arrays but avoid the large per-cell Python parser overhead.
    retained = max(stored_bytes, 2 * stored_bytes)
    peak = max(_LOADER_MIN_WORKING_BYTES, 6 * stored_bytes)
    return retained, peak


def _recommended_loader_workers(tasks) -> int:
    if isinstance(tasks, int):
        task_count = int(tasks)
        paths = []
    else:
        paths = [str(task[1]) for task in tasks]
        task_count = len(paths)
    cpu_total = os.cpu_count() or 1
    if cpu_total <= 2:
        target = cpu_total
    else:
        target = cpu_total - 1
    target = max(1, min(int(task_count), int(target)))
    if not paths:
        return target
    estimates = [_dataset_load_memory_estimate(path) for path in paths]
    retained_total = sum(retained for retained, _peak in estimates)
    transient_extras = sorted(
        (max(0, peak - retained) for retained, peak in estimates),
        reverse=True,
    )
    available = _available_memory_bytes()
    budget = (
        int(available * 0.5)
        if available is not None
        else _LOADER_UNKNOWN_MEMORY_BUDGET_BYTES
    )

    # Every successfully loaded grid remains in the result batch.  Reserve
    # that final retained footprint, then admit only as many simultaneous
    # parse/clean workspaces as fit in the remaining budget.  This prevents a
    # tiny, highly compressed .grim file from spawning CPU-count workers that
    # each expand into a large in-memory grid.
    safe_workers = 0
    for worker_count in range(1, target + 1):
        planned_peak = retained_total + sum(transient_extras[:worker_count])
        if planned_peak <= budget:
            safe_workers = worker_count
        else:
            break
    if safe_workers < 1:
        budget_mib = budget / 1024**2
        required_mib = (
            retained_total + (transient_extras[0] if transient_extras else 0)
        ) / 1024**2
        source = (
            "available memory"
            if available is not None
            else "the conservative fallback budget"
        )
        raise MemoryError(
            f"This dataset batch needs an estimated {required_mib:.0f} MiB "
            f"but only {budget_mib:.0f} MiB of {source} is reserved for "
            "loading. Load fewer or smaller dataset files at a time."
        )
    # Python's delimited parsers are memory-heavy and mostly GIL-bound. Running
    # several simultaneously increases peak memory without a reliable speedup.
    if any(path.casefold().endswith(_TEXT_LOADER_EXTENSIONS) for path in paths):
        return 1
    return safe_workers


def _load_dataset_path_task(task: tuple[int, str]) -> dict[str, object]:
    index, path = task
    file_name = os.path.basename(path)
    dataset_name = os.path.splitext(file_name)[0]
    lower = path.lower()
    try:
        if not _is_supported_dataset_path(path):
            return {
                "status": "ignored",
                "index": index,
                "path": path,
                "file_name": file_name,
                "error": "Unsupported file extension",
            }
        dataset = load_dataset_headless(path)
        history = str(getattr(dataset, "history", "") or path)
    except Exception as exc:
        return {
            "status": "error",
            "index": index,
            "path": path,
            "file_name": file_name,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "index": index,
        "path": path,
        "file_name": file_name,
        "name": dataset_name,
        "history": history,
        "dataset": dataset,
    }


def _join_many_with_progress(
    grids: list[RcsGrid],
    *,
    tol: float = 1e-6,
    progress_cb=None,
) -> RcsGrid:
    checked = RcsGrid._ensure_grids(grids)
    total = len(checked)
    available = _available_memory_bytes()
    limit = int(available * 0.5) if available is not None else None
    result = RcsGrid.join_many(
        *checked,
        tol=tol,
        overlap="error",
        max_output_bytes=limit,
    )
    if progress_cb is not None:
        progress_cb(total, total)
    return result


class _DatasetLoadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, tasks: list[tuple[int, str]], ignored_count: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._tasks = list(tasks)
        self._ignored_count = int(ignored_count)

    def run(self) -> None:
        total = len(self._tasks)
        loaded: list[dict[str, object]] = []
        failed: list[str] = []
        used_parallel = False

        def _consume(result: dict[str, object], done_count: int) -> None:
            status = str(result.get("status", "error"))
            file_name = str(result.get("file_name", "dataset"))
            if status == "ok":
                loaded.append(result)
                self.progress.emit(done_count, total, f"Loaded {file_name}")
                return
            error_text = str(result.get("error", "Unknown error"))
            failed.append(f"{file_name} ({error_text})")
            self.progress.emit(done_count, total, f"Failed {file_name}")

        try:
            worker_count = _recommended_loader_workers(self._tasks) if total else 1
            if total == 1:
                _consume(_load_dataset_path_task(self._tasks[0]), 1)
            elif total > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = {
                        pool.submit(_load_dataset_path_task, task): task
                        for task in self._tasks
                    }
                    done_count = 0
                    for future in as_completed(futures):
                        result = future.result()
                        done_count += 1
                        _consume(result, done_count)
                used_parallel = worker_count > 1
        except Exception as exc:
            # Individual parse failures normally arrive as result mappings.
            # This catches pool setup/submission/future faults so the owning
            # QThread still receives exactly one terminal signal and can quit.
            failed.append(f"Dataset loader ({type(exc).__name__}: {exc})")
        finally:
            self.finished.emit(
                {
                    "loaded": loaded,
                    "failed": failed,
                    "ignored": self._ignored_count,
                    "used_parallel": used_parallel,
                    "total_supported": total,
                }
            )


class _CsvExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(
        self,
        entries: list[tuple[RcsGrid, str]],
        *,
        scale: str,
        include_phase: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._entries = list(entries)
        self._scale = str(scale)
        self._include_phase = bool(include_phase)

    def run(self) -> None:
        total = len(self._entries)
        try:
            self.progress.emit(0, total, "Writing staged CSV files")
            paths = _stage_and_publish_csv_batch(
                self._entries,
                scale=self._scale,
                include_phase=self._include_phase,
            )
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.progress.emit(total, total, "Published CSV files")
        self.finished.emit({"ok": True, "paths": paths, "total": total})


class _BackgroundJobCancelled(RuntimeError):
    """Internal cooperative-cancellation sentinel."""


class _BackgroundCallableWorker(QObject):
    """Run one pure-Python/NumPy callable away from Qt's GUI thread."""

    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, function, *, reports_progress: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._function = function
        self._reports_progress = bool(reports_progress)
        self.supports_cancellation = self._reports_progress

    def run(self) -> None:
        try:
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()
            if self._reports_progress:
                def report_progress(done_count, total_count, detail=""):
                    if QThread.currentThread().isInterruptionRequested():
                        raise _BackgroundJobCancelled()
                    self.progress.emit(done_count, total_count, detail)

                result = self._function(report_progress)
            else:
                result = self._function()
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()
        except _BackgroundJobCancelled:
            self.finished.emit({"ok": False, "cancelled": True})
            return
        except Exception as exc:
            self.finished.emit(
                {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            )
            return
        self.finished.emit({"ok": True, "result": result})


class _JoinDatasetsWorker(QObject):
    supports_cancellation = False
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, grids: list[RcsGrid], tol: float = 1e-6, parent=None) -> None:
        super().__init__(parent)
        self._grids = list(grids)
        self._tol = float(tol)

    def run(self) -> None:
        total = max(1, len(self._grids))
        try:
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()

            def _emit_progress(done_count: int, total_count: int) -> None:
                if QThread.currentThread().isInterruptionRequested():
                    raise _BackgroundJobCancelled()
                self.progress.emit(done_count, total_count, "Joining datasets")

            merged = _join_many_with_progress(self._grids, tol=self._tol, progress_cb=_emit_progress)
        except _BackgroundJobCancelled:
            self.finished.emit({"ok": False, "cancelled": True, "total": total})
            return
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.finished.emit({"ok": True, "merged": merged, "total": total})


class _RangeCalibrationWorker(QObject):
    """Apply one calibration definition to DUT grids off the GUI thread."""

    supports_cancellation = True

    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(
        self,
        targets: list[tuple[str, RcsGrid]],
        measured_entry: tuple[str, RcsGrid],
        exact_entry: tuple[str, RcsGrid],
        params: dict[str, object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._targets = list(targets)
        self._measured_name, self._measured = measured_entry
        self._exact_name, self._exact = exact_entry
        self._params = dict(params)

    def run(self) -> None:
        total = len(self._targets)
        results: list[dict[str, object]] = []
        failed: list[str] = []
        try:
            offset_m = float(self._params["range_offset_m"])
        except (KeyError, TypeError, ValueError) as exc:
            self.finished.emit(
                {
                    "results": results,
                    "failed": [f"invalid range-calibration parameters ({exc})"],
                    "total": total,
                }
            )
            return
        exact_display = str(self._exact_name)
        if len(exact_display) > 48:
            exact_display = exact_display[:45] + "..."

        for index, (target_name, target) in enumerate(self._targets, start=1):
            if QThread.currentThread().isInterruptionRequested():
                failed.append("cancelled by user")
                break
            try:
                calibrated = target.range_calibrate(
                    self._measured,
                    self._exact,
                    offset_m,
                    allow_singleton_angular_broadcast=bool(
                        self._params.get(
                            "allow_singleton_angular_broadcast", False
                        )
                    ),
                    convention_attested=False,
                    measured_label=self._measured_name,
                    exact_label=self._exact_name,
                    maximum_correction_gain_db=self._params.get(
                        "maximum_correction_gain_db", 60.0
                    ),
                )
            except Exception as exc:
                failed.append(f"{target_name} ({exc})")
                self.progress.emit(index, total, f"Skipped {target_name}")
                continue

            results.append(
                {
                    "dataset": calibrated,
                    "source_dataset": target,
                    "name": (
                        f"{target_name} [Range Cal: {exact_display}; "
                        f"ΔR {offset_m:+.6g} m]"
                    ),
                    "history": (
                        f"Range Cal: {target_name}; measured={self._measured_name}; "
                        f"exact={self._exact_name}; ΔR={offset_m:+.12g} m "
                        "(positive away)"
                    ),
                }
            )
            self.progress.emit(index, total, f"Calibrated {target_name}")

        self.finished.emit(
            {
                "results": results,
                "failed": failed,
                "total": total,
            }
        )


class DatasetOpsMixin:
    def _preflight_derived_outputs(
        self,
        operation_name: str,
        plans,
        *,
        extra_bytes: int = 0,
    ) -> bool:
        """Reject a derived-result batch whose conservative peak is unsafe.

        ``plans`` contains ``(dataset, output_shape)`` pairs. The common
        estimator includes retained power/phase plus constructor and ufunc
        scratch; callers add operation-specific tensors through ``extra_bytes``.
        """

        try:
            estimated_peak = int(extra_bytes) + sum(
                _derived_grid_peak_bytes(dataset, shape)
                for dataset, shape in plans
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.status.showMessage(
                f"{operation_name} blocked: invalid output-size estimate ({exc})"
            )
            return False
        limit = _derived_grid_memory_limit()
        if estimated_peak <= limit:
            return True
        self.status.showMessage(
            f"{operation_name} blocked before allocation: estimated working set "
            f"{_format_bytes(estimated_peak)} exceeds the current safety limit "
            f"{_format_bytes(limit)}. Process fewer or smaller datasets."
        )
        return False

    def _ensure_background_worker_state(self) -> None:
        if hasattr(self, "_background_worker_thread"):
            return
        self._background_worker_thread: QThread | None = None
        self._background_worker: QObject | None = None
        self._background_worker_name = ""
        self._pending_join_names: list[str] | None = None
        self._pending_join_references: list[DatasetReference] | None = None
        self._pending_range_record: dict[str, object] | None = None
        self._pending_callable_completion = None
        self._pending_import_batches: list[tuple[tuple[str, ...], int]] = []
        self._queued_import_keys: set[str] = set()
        self._active_import_keys: set[str] = set()
        self._import_cycle_results: list[tuple[str, bool]] = []
        self._last_import_summary = ""

    def _background_job_active(self) -> bool:
        self._ensure_background_worker_state()
        thread = self._background_worker_thread
        return isinstance(thread, QThread) and thread.isRunning()

    def _set_background_progress(
        self,
        done_count: int | None = None,
        total_count: int | None = None,
        detail: str = "",
    ) -> None:
        """Keep long dataset work visible even when status text is replaced."""

        progress = getattr(self, "dataset_job_progress", None)
        if progress is None:
            return
        total = int(total_count or 0)
        if total > 0:
            done = min(max(int(done_count or 0), 0), total)
            progress.setRange(0, total)
            progress.setValue(done)
            progress.setFormat(f"%v / %m  {str(detail).strip()}".rstrip())
        else:
            progress.setRange(0, 0)
            progress.setFormat(str(detail).strip() or "Working…")
        progress.setVisible(True)

    def _clear_background_progress(self) -> None:
        progress = getattr(self, "dataset_job_progress", None)
        if progress is None:
            return
        progress.setVisible(False)
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat("%p%")

    def _try_start_background_job(self, job_name: str, worker: QObject) -> bool:
        self._ensure_background_worker_state()
        if bool(getattr(self, "_isar_busy", False)):
            self.status.showMessage(
                "An ISAR reconstruction is still running. Please wait before "
                f"starting {job_name.lower()}."
            )
            return False
        if self._background_job_active():
            active_name = self._background_worker_name or "Another background job"
            self.status.showMessage(f"{active_name} is still running. Please wait.")
            return False

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_background_thread_finished)

        self._background_worker_thread = thread
        self._background_worker = worker
        self._background_worker_name = job_name
        self._set_background_progress(detail=job_name)
        cancel_button = getattr(self, "btn_dataset_cancel", None)
        if cancel_button is not None:
            cancel_button.setEnabled(True)
            cancel_button.setVisible(
                bool(getattr(worker, "supports_cancellation", False))
            )
        self._update_dataset_action_states(force_busy=True)
        thread.start()
        return True

    def _cancel_background_job(self) -> None:
        """Request a cooperative stop at the next safe worker boundary."""

        self._ensure_background_worker_state()
        thread = self._background_worker_thread
        if not isinstance(thread, QThread) or not thread.isRunning():
            self.status.showMessage("No dataset job is currently running.")
            return
        thread.requestInterruption()
        cancel_button = getattr(self, "btn_dataset_cancel", None)
        if cancel_button is not None:
            cancel_button.setEnabled(False)
        self.status.showMessage(
            f"Cancellation requested for "
            f"{self._background_worker_name or 'the dataset job'}; "
            "finishing the current safe block…"
        )

    def _start_background_callable(
        self,
        job_name: str,
        function,
        completion,
        *,
        reports_progress: bool = False,
    ) -> bool:
        """Run computation off-thread and publish its result on Qt's thread."""

        self._ensure_background_worker_state()
        if self._background_job_active():
            active_name = self._background_worker_name or "Another background job"
            self.status.showMessage(f"{active_name} is still running. Please wait.")
            return False
        worker = _BackgroundCallableWorker(
            function, reports_progress=reports_progress
        )
        if reports_progress:
            worker.progress.connect(self._on_background_callable_progress)
        worker.finished.connect(self._on_background_callable_finished)
        self._pending_callable_completion = completion
        if not self._try_start_background_job(job_name, worker):
            self._pending_callable_completion = None
            return False
        return True

    def _on_background_callable_progress(
        self, done_count: int, total_count: int, detail: str
    ) -> None:
        job_name = self._background_worker_name or "Dataset operation"
        detail_text = str(detail).strip()
        suffix = f" ({detail_text})" if detail_text else ""
        self._set_background_progress(done_count, total_count, detail_text)
        self.status.showMessage(
            f"{job_name}... {int(done_count)}/{int(total_count)}{suffix}"
        )

    def _start_dataset_map_job(
        self,
        job_name: str,
        datasets: list[tuple[str, RcsGrid]],
        operation,
        completion,
        *,
        start_message: str,
    ) -> bool:
        """Run an independent full-grid transform for each selected row."""

        launch_items = tuple(datasets)
        if not self._preflight_derived_outputs(
            job_name,
            [
                (dataset, tuple(int(value) for value in dataset.rcs_power.shape))
                for _name, dataset in launch_items
            ],
        ):
            return False

        def compute(progress):
            results = []
            skipped = []
            total = len(launch_items)
            for index, (name, dataset) in enumerate(launch_items, start=1):
                try:
                    result = operation(index - 1, name, dataset)
                except Exception as exc:
                    skipped.append(f"{name} ({exc})")
                else:
                    results.append((index - 1, name, result))
                progress(index, total, name)
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            completion(results, skipped)

        started = self._start_background_callable(
            job_name,
            compute,
            publish,
            reports_progress=True,
        )
        if started:
            self.status.showMessage(start_message)
        return started

    def _on_background_callable_finished(self, payload: dict[str, object]) -> None:
        completion = self._pending_callable_completion
        self._pending_callable_completion = None
        if bool(payload.get("cancelled", False)):
            self.status.showMessage(
                f"{self._background_worker_name or 'Dataset operation'} "
                "cancelled; no unfinished result was published."
            )
            return
        if not bool(payload.get("ok", False)):
            self.status.showMessage(
                f"{self._background_worker_name or 'Dataset operation'} failed: "
                + str(payload.get("error", "Unknown error"))
            )
            return
        if callable(completion):
            completion(payload.get("result"))

    def _on_background_thread_finished(self) -> None:
        completed_import = bool(self._active_import_keys)
        self._background_worker_thread = None
        self._background_worker = None
        self._background_worker_name = ""
        self._active_import_keys.clear()
        self._clear_background_progress()
        cancel_button = getattr(self, "btn_dataset_cancel", None)
        if cancel_button is not None:
            cancel_button.setVisible(False)
            cancel_button.setEnabled(True)
        self._update_dataset_action_states()

        if self._start_next_pending_import_batch():
            return

        if completed_import and self._import_cycle_results:
            cycle_messages = [
                message for message, _failed in self._import_cycle_results
            ]
            details = " ".join(cycle_messages[:3])
            if len(cycle_messages) > 3:
                details += f" …and {len(cycle_messages) - 3} more import batches."
            prefix = (
                "Dataset imports completed with errors."
                if any(failed for _message, failed in self._import_cycle_results)
                else "Dataset imports completed."
            )
            summary = f"{prefix} {details}".strip()
            self._last_import_summary = summary
            self.status.setToolTip(summary)
            self.status.showMessage(summary)
            self._import_cycle_results.clear()

        pending_isar = getattr(self, "_isar_pending", None)
        if pending_isar is not None and not bool(getattr(self, "_isar_busy", False)):
            self._isar_pending = None
            submit = getattr(self, "_isar_submit", None)
            if callable(submit):
                submit(pending_isar)

    def _start_next_pending_import_batch(self) -> bool:
        """Drain one queued import when both dataset and ISAR workers are idle."""

        self._ensure_background_worker_state()
        if (
            self._background_job_active()
            or bool(getattr(self, "_isar_busy", False))
            or not self._pending_import_batches
        ):
            return False
        paths, ignored = self._pending_import_batches.pop(0)
        for path in paths:
            self._queued_import_keys.discard(_target_path_key(path))
        if self._start_dataset_import_batch(list(paths), ignored_count=ignored):
            return True
        # Preserve the user's request if another operation started in the
        # narrow window between the idle check and QThread creation.
        self._pending_import_batches.insert(0, (paths, ignored))
        self._queued_import_keys.update(_target_path_key(path) for path in paths)
        return False

    def _on_load_worker_progress(self, done_count: int, total_count: int, detail: str) -> None:
        self._set_background_progress(done_count, total_count, detail)
        detail_text = str(detail).strip()
        if detail_text:
            self.status.showMessage(
                f"Loading datasets... {done_count}/{total_count} ({detail_text})"
            )
            return
        self.status.showMessage(f"Loading datasets... {done_count}/{total_count}")

    def _on_load_worker_finished(self, summary: dict[str, object]) -> None:
        self._ensure_background_worker_state()
        loaded_entries_raw = summary.get("loaded", [])
        failed_entries_raw = summary.get("failed", [])
        ignored = int(summary.get("ignored", 0) or 0)
        used_parallel = bool(summary.get("used_parallel", False))
        total_supported = int(summary.get("total_supported", 0) or 0)

        loaded_entries = [entry for entry in loaded_entries_raw if isinstance(entry, dict)]
        loaded_entries.sort(key=lambda item: int(item.get("index", 0)))
        failed = [str(item) for item in failed_entries_raw]

        loaded = 0
        for entry in loaded_entries:
            dataset = entry.get("dataset")
            if not isinstance(dataset, RcsGrid):
                file_name = str(entry.get("file_name", "dataset"))
                failed.append(f"{file_name} (worker returned invalid dataset)")
                continue
            name = str(entry.get("name", "dataset"))
            history = str(entry.get("history", ""))
            file_name = str(entry.get("file_name", ""))
            container_path = str(entry.get("path", "") or file_name)
            dataset_id = self._add_dataset_row(
                dataset,
                name,
                history,
                file_name=container_path,
                notify=False,
            )
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None:
                recorder.bind_loaded(
                    DatasetReference(dataset_id, name, container_path)
                )
            loaded += 1

        if loaded:
            notify = getattr(self, "_notify_dataset_catalog_changed", None)
            if callable(notify):
                notify()

        if failed:
            msg = f"Loaded {loaded} dataset(s)." if loaded else "No datasets loaded."
            msg += f" Failed: {_compact_item_summary(failed, limit=5)}"
        elif loaded:
            msg = f"Loaded {loaded} dataset(s)."
        else:
            msg = "No datasets loaded."

        if ignored:
            msg += f" Ignored {ignored} unsupported file(s)."
        if used_parallel and total_supported > 1:
            msg += " Loaded in parallel."
        self._import_cycle_results.append((msg, bool(failed)))
        self._last_import_summary = msg
        if failed:
            tooltip_failures = _compact_item_summary(failed, limit=20)
            self.status.setToolTip(
                (f"Loaded {loaded} dataset(s)." if loaded else "No datasets loaded.")
                + f" Failed: {tooltip_failures}"
            )
        else:
            self.status.setToolTip(msg)
        self.status.showMessage(msg)

    def _on_csv_export_progress(
        self, done_count: int, total_count: int, detail: str
    ) -> None:
        self._set_background_progress(done_count, total_count, detail)
        suffix = f" ({str(detail).strip()})" if str(detail).strip() else ""
        self.status.showMessage(
            f"Exporting CSV... {done_count}/{total_count}{suffix}"
        )

    def _on_csv_export_finished(self, payload: dict[str, object]) -> None:
        if not bool(payload.get("ok", False)):
            self.status.showMessage(
                "CSV export failed; no partial batch was kept. "
                + str(payload.get("error", "Unknown error"))
            )
            return
        paths = [str(path) for path in payload.get("paths", [])]
        self.status.showMessage(f"Exported {len(paths)} dataset(s) to CSV.")

    def _on_join_worker_progress(self, done_count: int, total_count: int, _: str) -> None:
        self._set_background_progress(done_count, total_count, "Join")
        self.status.showMessage(f"Join... {done_count}/{total_count}")

    def _on_join_worker_finished(self, payload: dict[str, object]) -> None:
        names = self._pending_join_names or []
        self._pending_join_names = None
        input_refs = self._pending_join_references
        self._pending_join_references = None

        ok = bool(payload.get("ok", False))
        if not ok:
            self.status.showMessage(str(payload.get("error", "Join failed.")))
            return

        merged = payload.get("merged")
        if not isinstance(merged, RcsGrid):
            self.status.showMessage("Join failed: worker produced invalid output.")
            return

        if not names:
            names = ["Dataset"]
        new_name = " | ".join(names)
        history = f"Join (equal/complementary overlaps merged; conflicts rejected): {new_name}"
        output_name = f"Join[{new_name}]"
        output_id = self._add_dataset_row(merged, output_name, history, file_name="")
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None and input_refs:
            recorder.record_function(
                self._python_output_reference(output_id, output_name),
                "join_datasets",
                input_refs,
                kwargs={"tol": 1.0e-6},
                comment="Join datasets on their union axes; reject conflicting overlaps",
            )
        self.status.showMessage(
            "Join created. Equal or complementary overlaps were merged; "
            "conflicting finite samples would have stopped the operation."
        )

    def _on_range_cal_worker_progress(
        self, done_count: int, total_count: int, detail: str
    ) -> None:
        self._set_background_progress(done_count, total_count, detail)
        detail_text = str(detail).strip()
        suffix = f" ({detail_text})" if detail_text else ""
        self.status.showMessage(
            f"Range Cal... {done_count}/{total_count}{suffix}"
        )

    def _on_range_cal_worker_finished(self, payload: dict[str, object]) -> None:
        record_spec = self._pending_range_record
        self._pending_range_record = None
        raw_results = payload.get("results", [])
        failed = [str(value) for value in payload.get("failed", [])]
        produced = 0
        for entry in raw_results:
            if not isinstance(entry, dict):
                failed.append("worker returned a malformed result")
                continue
            dataset = entry.get("dataset")
            if not isinstance(dataset, RcsGrid):
                failed.append("worker returned an invalid calibrated dataset")
                continue
            output_name = str(entry.get("name", "Range Cal result"))
            output_id = self._add_dataset_row(
                dataset,
                output_name,
                str(entry.get("history", "Range Cal")),
                file_name="",
            )
            recorder = getattr(self, "python_recorder", None)
            source_dataset = entry.get("source_dataset")
            if recorder is not None and record_spec is not None:
                targets_by_identity = record_spec.get("targets", {})
                target_ref = (
                    targets_by_identity.get(id(source_dataset))
                    if isinstance(targets_by_identity, dict)
                    else None
                )
                measured_ref = record_spec.get("measured")
                exact_ref = record_spec.get("exact")
                if all(
                    isinstance(value, DatasetReference)
                    for value in (target_ref, measured_ref, exact_ref)
                ):
                    offset_m = float(record_spec["range_offset_m"])
                    allow_broadcast = bool(
                        record_spec.get("allow_singleton_angular_broadcast", False)
                    )
                    gain_limit = record_spec.get("maximum_correction_gain_db", 60.0)
                    measured_label = str(record_spec.get("measured_label", ""))
                    exact_label = str(record_spec.get("exact_label", ""))
                    recorder.record_expression(
                        self._python_output_reference(output_id, output_name),
                        [target_ref, measured_ref, exact_ref],
                        lambda variables,
                        offset_m=offset_m,
                        allow_broadcast=allow_broadcast,
                        gain_limit=gain_limit,
                        measured_label=measured_label,
                        exact_label=exact_label: (
                            f"{variables[0]}.range_calibrate(\n"
                            f"    {variables[1]},\n"
                            f"    {variables[2]},\n"
                            f"    {offset_m!r},\n"
                            f"    allow_singleton_angular_broadcast={allow_broadcast!r},\n"
                            f"    measured_label={measured_label!r},\n"
                            f"    exact_label={exact_label!r},\n"
                            f"    maximum_correction_gain_db={gain_limit!r},\n"
                            f")"
                        ),
                        comment="Complex range calibration with resolved references",
                    )
            produced += 1

        message = f"Range Cal created {produced} dataset(s)."
        if failed:
            message += f" Skipped: {', '.join(failed)}"
        self.status.showMessage(message)

    def _start_dataset_import_batch(
        self, paths: list[str], *, ignored_count: int = 0
    ) -> bool:
        """Start one already-filtered import batch.

        This is deliberately separate from ``_handle_files_dropped`` so a
        batch queued behind a join or calibration can be resumed verbatim.
        """

        tasks = [(index, path) for index, path in enumerate(paths)]
        if not tasks:
            return False
        worker = _DatasetLoadWorker(tasks, ignored_count=ignored_count)
        worker.progress.connect(self._on_load_worker_progress)
        worker.finished.connect(self._on_load_worker_finished)
        self._active_import_keys = {_target_path_key(path) for path in paths}
        if not self._try_start_background_job("Dataset loading", worker):
            self._active_import_keys.clear()
            return False
        self.status.showMessage(f"Loading datasets... 0/{len(tasks)}")
        return True

    def _handle_files_dropped(self, paths: list[str]) -> None:
        self._ensure_background_worker_state()
        accepted: list[str] = []
        ignored = 0
        already_pending = set(self._active_import_keys) | set(self._queued_import_keys)
        batch_keys: set[str] = set()
        duplicate_count = 0
        for raw_path in paths:
            path = os.fspath(raw_path)
            if _is_supported_dataset_path(path):
                key = _target_path_key(path)
                if key in already_pending or key in batch_keys:
                    duplicate_count += 1
                    continue
                batch_keys.add(key)
                accepted.append(path)
            else:
                ignored += 1

        if not accepted:
            if ignored:
                self.status.showMessage(
                    "No supported dropped files. Supported: "
                    + ", ".join(SUPPORTED_EXTENSIONS)
                )
            elif duplicate_count:
                self.status.showMessage(
                    f"Skipped {duplicate_count} dataset import(s) already loading or queued."
                )
            return

        if self._background_job_active() or self._pending_import_batches:
            batch = tuple(accepted)
            self._pending_import_batches.append((batch, ignored))
            self._queued_import_keys.update(batch_keys)
            message = (
                f"Queued {len(batch)} dataset import(s) as batch "
                f"{len(self._pending_import_batches)}; they will load automatically."
            )
            if duplicate_count:
                message += f" Skipped {duplicate_count} duplicate(s)."
            self.status.showMessage(message)
            return

        self._import_cycle_results.clear()
        if not self._start_dataset_import_batch(accepted, ignored_count=ignored):
            # A job can begin between the active check and thread startup. Keep
            # the user's files instead of losing that race.
            batch = tuple(accepted)
            self._pending_import_batches.insert(0, (batch, ignored))
            self._queued_import_keys.update(batch_keys)
            self.status.showMessage(
                f"Queued {len(batch)} dataset import(s); they will load automatically."
            )

    def _load_dataset_files(self) -> None:
        """Choose dataset files and route them through the drop/headless loader."""

        patterns = " ".join(f"*{extension}" for extension in SUPPORTED_EXTENSIONS)
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Load GRIM datasets",
            "",
            f"Supported datasets ({patterns});;All files (*)",
        )
        if paths:
            self._handle_files_dropped([str(path) for path in paths])

    def _add_dataset_row(
        self,
        dataset: RcsGrid,
        name: str,
        history: str,
        file_name: str | None = None,
        *,
        dirty: bool | None = None,
        notify: bool = True,
    ) -> str:
        """Add a dataset and keep its artifact history authoritative.

        ``file_name`` is non-empty only for an artifact that already exists on
        disk.  Derived rows therefore begin dirty even when their RcsGrid
        inherited the source dataset's ``source_path`` metadata.
        """

        durable_history = _append_provenance(dataset.history, history)
        # A single in-memory RcsGrid may be published into more than one row
        # (for example, two Assembly branches).  Row provenance must not leak
        # from the first row into the second merely because both callers hand
        # us the same Python object.  A shallow grid copy keeps the large,
        # effectively read-only sample arrays shared while giving each row its
        # own scalar history and metadata dictionaries.
        row_dataset = copy.copy(dataset)
        row_dataset.units = dict(dataset.units or {})
        row_dataset.extra = dict(dataset.extra or {})
        row_dataset.history = durable_history
        is_dirty = not bool(file_name) if dirty is None else bool(dirty)
        row = self.table.rowCount()
        signals_were_blocked = self.table.blockSignals(True)
        try:
            self.table.insertRow(row)
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, row_dataset)
            dataset_id = uuid.uuid4().hex
            name_item.setData(DATASET_ID_ROLE, dataset_id)
            name_item.setData(DATASET_DIRTY_ROLE, is_dirty)
            name_font = name_item.font()
            name_font.setBold(is_dirty)
            name_item.setFont(name_font)
            name_item.setToolTip(
                "Unsaved derived dataset" if is_dirty else "Saved or loaded dataset"
            )

            source_path = ""
            if not is_dirty:
                # file_name is the container GRIM/PTM/CSV path selected by the
                # user. Solver metadata may instead name its originating .geo,
                # so it is only a fallback for legacy callers without a path.
                source_path = str(file_name or dataset.source_path or "")
            file_text = "Unsaved" if is_dirty else os.path.basename(file_name or source_path)
            file_item = QTableWidgetItem(file_text)
            file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
            file_item.setData(DATASET_PATH_ROLE, source_path)
            file_item.setToolTip(source_path or "Not saved yet")
            history_item = QTableWidgetItem(durable_history)
            history_item.setFlags(history_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, file_item)
            self.table.setItem(row, 2, history_item)
        finally:
            self.table.blockSignals(signals_were_blocked)
        if notify:
            catalog_notify = getattr(self, "_notify_dataset_catalog_changed", None)
            if callable(catalog_notify):
                catalog_notify()
        return dataset_id

    def _python_reference_for_dataset(
        self, dataset: RcsGrid
    ) -> DatasetReference | None:
        """Resolve an in-memory row through its stable UUID for script output."""

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            if name_item is None or name_item.data(Qt.UserRole) is not dataset:
                continue
            path_item = self.table.item(row, 1)
            return DatasetReference(
                dataset_id=str(name_item.data(DATASET_ID_ROLE) or ""),
                name=name_item.text(),
                path=(
                    str(path_item.data(DATASET_PATH_ROLE) or "")
                    if path_item is not None
                    else ""
                ),
            )
        return None

    @staticmethod
    def _python_output_reference(dataset_id: str, name: str) -> DatasetReference:
        return DatasetReference(dataset_id=str(dataset_id), name=str(name), path="")

    def _python_input_references(
        self, datasets: list[tuple[str, RcsGrid]]
    ) -> list[DatasetReference] | None:
        references = [
            self._python_reference_for_dataset(dataset) for _name, dataset in datasets
        ]
        if any(reference is None for reference in references):
            return None
        return [reference for reference in references if reference is not None]

    def _dataset_row_is_dirty(self, row: int) -> bool:
        item = self.table.item(int(row), 0)
        return bool(item is not None and item.data(DATASET_DIRTY_ROLE))

    def _dirty_dataset_rows(self) -> list[int]:
        return [
            row
            for row in range(self.table.rowCount())
            if self._dataset_row_is_dirty(row)
        ]

    def _set_dataset_row_saved(self, row: int, output_path: str) -> None:
        """Mark one successfully published artifact clean without touching history."""

        name_item = self.table.item(row, 0)
        if name_item is None:
            return
        signals_were_blocked = self.table.blockSignals(True)
        try:
            name_item.setData(DATASET_DIRTY_ROLE, False)
            font = name_item.font()
            font.setBold(False)
            name_item.setFont(font)
            name_item.setToolTip("Saved dataset")

            file_item = self.table.item(row, 1)
            if file_item is None:
                file_item = QTableWidgetItem()
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, file_item)
            file_item.setText(os.path.basename(output_path))
            file_item.setData(DATASET_PATH_ROLE, output_path)
            file_item.setToolTip(output_path)
        finally:
            self.table.blockSignals(signals_were_blocked)
        notify = getattr(self, "_notify_dataset_catalog_changed", None)
        if callable(notify):
            notify()

    def _on_dataset_selection_changed(self) -> None:
        previous_active = getattr(self, "active_dataset", None)

        def commit_active_dataset(dataset) -> None:
            self.active_dataset = dataset
            if dataset is previous_active:
                return
            invalidate_isar = getattr(self, "_invalidate_isar_result", None)
            if callable(invalidate_isar):
                invalidate_isar()

        selected = self.table.selectionModel().selectedRows()
        self._update_dataset_selection_order([idx.row() for idx in selected])
        self._update_dataset_action_states()
        summary_label = getattr(self, "lbl_dataset_selection_summary", None)
        if not selected:
            commit_active_dataset(None)
            self._clear_param_lists()
            if summary_label is not None:
                summary_label.setText(
                    "Select a row to inspect it. Ctrl-click rows in operand order."
                )
            return

        selected_rows = {idx.row() for idx in selected}
        current_row = int(self.table.currentRow())
        # The current row is what Qt visually presents as active.  Using the
        # first selected row made the parameter lists silently describe a
        # different dataset after Ctrl-clicking another selected row.
        row = current_row if current_row in selected_rows else selected[0].row()
        item = self.table.item(row, 0)
        dataset = item.data(Qt.UserRole) if item else None
        if not isinstance(dataset, RcsGrid):
            commit_active_dataset(None)
            self._clear_param_lists()
            return
        active_changed = dataset is not previous_active
        commit_active_dataset(dataset)
        if active_changed:
            self._populate_params(dataset)
        if summary_label is not None:
            active_name = item.text() if item is not None else f"Row {row + 1}"
            shown_active_name = (
                active_name if len(active_name) <= 80 else active_name[:77] + "…"
            )
            order = [
                operand_row
                for operand_row in getattr(self, "_dataset_selection_order", [])
                if operand_row in selected_rows
            ]
            operand_names = []
            for operand_index, operand_row in enumerate(order, start=1):
                operand_item = self.table.item(operand_row, 0)
                operand_name = (
                    operand_item.text()
                    if operand_item is not None
                    else f"Row {operand_row + 1}"
                )
                operand_names.append(f"{operand_index}: {operand_name}")
            order_text = "  →  ".join(operand_names)
            if len(operand_names) > 1:
                shown_names = [
                    value if len(value) <= 56 else value[:53] + "…"
                    for value in operand_names[:4]
                ]
                shown_order = "  →  ".join(shown_names)
                if len(operand_names) > len(shown_names):
                    shown_order += f"  →  … +{len(operand_names) - len(shown_names)} more"
                summary_label.setText(
                    f"Active parameters: {shown_active_name}    Operand order: {shown_order}"
                )
                summary_label.setToolTip(
                    f"Active parameters: {active_name}\nOperand order: {order_text}"
                )
            else:
                summary_label.setText(f"Active parameters: {shown_active_name}")
                summary_label.setToolTip(f"Active parameters: {active_name}")

    def _update_dataset_action_states(self, *, force_busy: bool = False) -> None:
        """Disable actions whose operand-count or job-state contract is unmet."""

        table = getattr(self, "table", None)
        if table is None:
            return
        selection_model = table.selectionModel()
        selected_count = (
            len(selection_model.selectedRows())
            if selection_model is not None
            else 0
        )
        row_count = int(table.rowCount())
        busy = bool(force_busy or self._background_job_active())

        def enable(names, condition) -> None:
            state = bool(condition) and not busy
            for name in names:
                button = getattr(self, name, None)
                if button is not None:
                    button.setEnabled(state)

        enable(
            (
                "btn_slice", "btn_stats", "btn_percentile", "btn_interpolate",
                "btn_decimate", "btn_mirror", "btn_wrap", "btn_shift",
                "btn_round", "btn_offset", "btn_medianize", "btn_duplicate",
                "btn_audit", "btn_provenance", "btn_set_coordinates",
                "btn_axis_units", "btn_el_to_az360", "btn_swap_el_az",
                "btn_sentri_elevation", "btn_to_dbke", "btn_to_dbsm",
                "btn_conic_gc", "btn_wedge_to_conic",
            ),
            selected_count >= 1,
        )
        enable(
            (
                "btn_coherent_add", "btn_coherent_sub", "btn_incoherent_add",
                "btn_incoherent_sub", "btn_join", "btn_stitch", "btn_overlap",
                "btn_align", "btn_compatibility", "btn_support_reference",
            ),
            selected_count >= 2,
        )
        enable(("btn_coherent_div", "btn_dbdiff"), selected_count == 2)
        enable(
            ("btn_range_cal",),
            selected_count >= 1 and row_count - selected_count >= 2,
        )
        enable(
            ("btn_dataset_save", "btn_dataset_export", "btn_dataset_delete"),
            selected_count >= 1,
        )
        enable(("btn_dataset_save_all",), row_count >= 1)
        enable(("btn_dataset_save_dirty",), bool(self._dirty_dataset_rows()))
        enable(
            ("btn_dataset_undo_delete",),
            bool(getattr(self, "_last_deleted_dataset_rows", ())),
        )

    def _update_dataset_selection_order(self, selected_rows: list[int]) -> None:
        selected_set = set(selected_rows)
        previous_order = getattr(self, "_dataset_selection_order", [])
        order = [row for row in previous_order if row in selected_set]
        current_row = self.table.currentRow()

        for row in selected_rows:
            if row not in order:
                order.append(row)

        # Use the active row as the most-recent selection.
        if current_row in selected_set and current_row in order:
            order = [row for row in order if row != current_row] + [current_row]

        self._dataset_selection_order = order

    def _on_dataset_rows_reordered(self) -> None:
        self._dataset_selection_order = []
        self._update_dataset_selection_order(
            [idx.row() for idx in self.table.selectionModel().selectedRows()]
        )
        notify = getattr(self, "_notify_dataset_catalog_changed", None)
        if callable(notify):
            notify()

    def _populate_params(self, dataset: RcsGrid) -> None:
        self._update_parameter_headers(dataset)
        self._fill_list(self.list_pol, dataset.polarizations)
        self._fill_list(self.list_freq, dataset.frequencies)
        self._fill_list(self.list_elev, dataset.elevations)
        self._fill_list(self.list_az, dataset.azimuths)
        self._apply_default_param_selection()

    def _update_parameter_headers(self, dataset: RcsGrid | None) -> None:
        """Label selectors from the active grid's actual coordinate metadata."""

        if dataset is None:
            labels = ("Polarization", "Frequency", "Elevation", "Azimuth")
        else:
            units = dataset.units or {}

            def _unit(key: str, default: str) -> str:
                value = str(units.get(key, default) or default).strip()
                return value or default

            frequency_unit = _unit("frequency", "GHz")
            elevation_unit = _unit("elevation", "deg")
            azimuth_unit = _unit("azimuth", "deg")
            if dataset.angular_coordinate_system() == "great_circle":
                elevation_name, azimuth_name = "Pitch", "Aspect"
            else:
                elevation_name, azimuth_name = "Elevation", "Azimuth"
            labels = (
                "Polarization",
                f"Frequency ({frequency_unit})",
                f"{elevation_name} ({elevation_unit})",
                f"{azimuth_name} ({azimuth_unit})",
            )

        for attribute, text in zip(
            ("lbl_pol", "lbl_freq", "lbl_elev", "lbl_az"), labels
        ):
            label = getattr(self, attribute, None)
            if label is not None:
                label.setText(text)

    @staticmethod
    def _select_first_item(widget: QListWidget) -> None:
        if widget.count() <= 0:
            return
        widget.clearSelection()
        first = widget.item(0)
        if first is None:
            return
        first.setSelected(True)
        widget.setCurrentItem(first)

    def _apply_default_param_selection(self) -> None:
        widgets = (self.list_pol, self.list_freq, self.list_elev, self.list_az)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._select_first_item(self.list_pol)
            self._select_first_item(self.list_freq)
            self._select_first_item(self.list_elev)
            if self.list_az.count() > 0:
                self.list_az.selectAll()
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        # Refresh availability masks from selected polarization and trigger one autoplot update.
        self._on_polarization_selection_changed()

    def _fill_list(self, widget: QListWidget, values, indices=None) -> None:
        widget.setUpdatesEnabled(False)
        widget.blockSignals(True)
        try:
            widget.clear()
            if indices is None:
                indices = list(range(len(values)))
            else:
                indices = [int(idx) for idx in indices]
            if widget is getattr(self, "list_pol", None):
                indices = _sorted_polarization_indices(values, indices)
            for idx in indices:
                value = values[idx]
                item = QListWidgetItem(str(value))
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                item.setData(Qt.UserRole, value)
                item.setData(Qt.UserRole + 1, int(idx))
                widget.addItem(item)
        finally:
            widget.blockSignals(False)
            widget.setUpdatesEnabled(True)

    def _clear_param_lists(self) -> None:
        for widget in (self.list_pol, self.list_freq, self.list_elev, self.list_az):
            widget.clear()
        self._update_parameter_headers(None)

    def _on_param_item_changed(self, item: QListWidgetItem, axis_name: str, widget: QListWidget) -> None:
        """Validate and transactionally commit one inline parameter edit."""

        dataset = self.active_dataset
        if dataset is None:
            return
        axis_arr = dataset.get_axis(axis_name)
        idx = item.data(Qt.UserRole + 1)
        if idx is None:
            return
        idx = int(idx)
        if idx < 0 or idx >= len(axis_arr):
            return
        old_value = axis_arr[idx]
        entered_text = item.text()

        def restore_item() -> None:
            signals_were_blocked = widget.blockSignals(True)
            try:
                item.setText(str(old_value))
                item.setData(Qt.UserRole, old_value)
            finally:
                widget.blockSignals(signals_were_blocked)

        owning_row = None
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            if name_item is not None and name_item.data(Qt.UserRole) is dataset:
                owning_row = row
                break
        if owning_row is None:
            restore_item()
            self.status.showMessage(
                "Parameter edit rejected: the active dataset row is no longer available."
            )
            return

        source_reference = self._python_reference_for_dataset(dataset)
        try:
            edited = dataset.edit_axis_value(axis_name, idx, entered_text)
        except (IndexError, TypeError, ValueError) as exc:
            restore_item()
            self.status.showMessage(f"Parameter edit rejected: {exc}")
            return

        if edited is dataset:
            restore_item()
            self.status.showMessage(f"{axis_name.capitalize()} value is unchanged.")
            return

        name_item = self.table.item(owning_row, 0)
        if name_item is None:
            restore_item()
            return
        dataset_id = str(name_item.data(DATASET_ID_ROLE) or "")
        dataset_name = name_item.text().strip() or f"Dataset {owning_row + 1}"
        source_item = self.table.item(owning_row, 1)
        source_path = (
            str(source_item.data(DATASET_PATH_ROLE) or "")
            if source_item is not None
            else ""
        )

        table_signals_were_blocked = self.table.blockSignals(True)
        try:
            name_item.setData(Qt.UserRole, edited)
            name_item.setData(DATASET_DIRTY_ROLE, True)
            name_font = name_item.font()
            name_font.setBold(True)
            name_item.setFont(name_font)
            name_item.setToolTip("Unsaved parameter edits")

            if source_item is not None:
                source_item.setText("Unsaved")
                source_item.setToolTip(
                    "Unsaved parameter edits"
                    + (f"; original source: {source_path}" if source_path else "")
                )

            history_item = self.table.item(owning_row, 2)
            if history_item is None:
                history_item = QTableWidgetItem()
                history_item.setFlags(history_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(owning_row, 2, history_item)
            history_item.setText(str(edited.history or ""))
        finally:
            self.table.blockSignals(table_signals_were_blocked)

        self.active_dataset = edited
        clear_plot = getattr(self, "_clear_plot", None)
        if callable(clear_plot):
            clear_plot()
            canvas = getattr(self, "plot_canvas", None)
            if canvas is not None:
                canvas.draw_idle()
        self._populate_params(edited)
        notify = getattr(self, "_notify_dataset_catalog_changed", None)
        if callable(notify):
            notify()

        recorder = getattr(self, "python_recorder", None)
        if (
            recorder is not None
            and source_reference is not None
            and dataset_id
        ):
            if axis_name == "polarization":
                recorded_value = entered_text.strip()
            else:
                # The edited coordinate may have moved to another stored index,
                # but replay uses the original index and entered value; the
                # model performs the same stable sort deterministically.
                recorded_value = float(entered_text)
            recorder.record_method(
                self._python_output_reference(dataset_id, dataset_name),
                source_reference,
                "edit_axis_value",
                args=(axis_name, idx, recorded_value),
                comment=f"Edit {axis_name} parameter for {dataset_name}",
            )
        self.status.showMessage(
            f"Edited {axis_name} parameter for {dataset_name}; dataset is unsaved."
        )

    def _selected_indices(self, widget: QListWidget) -> set[int]:
        indices = set()
        for item in widget.selectedItems():
            idx = item.data(Qt.UserRole + 1)
            if idx is not None:
                indices.add(int(idx))
        return indices

    def _displayed_indices(self, widget: QListWidget) -> set[int]:
        indices = set()
        for row in range(widget.count()):
            item = widget.item(row)
            if item is None:
                continue
            idx = item.data(Qt.UserRole + 1)
            if idx is not None:
                indices.add(int(idx))
        return indices

    def _selected_values(self, widget: QListWidget) -> list:
        values = []
        for item in widget.selectedItems():
            values.append(item.data(Qt.UserRole))
        return values

    def _indices_for_values(self, axis_arr, values, tol=1e-6) -> list[int] | None:
        return RcsGrid._indices_for_axis_values(axis_arr, values, tol=tol)

    def _selected_datasets(self) -> list[tuple[str, RcsGrid]]:
        datasets: list[tuple[str, RcsGrid]] = []
        selected = self.table.selectionModel().selectedRows()
        for model_index in selected:
            row = model_index.row()
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if isinstance(dataset, RcsGrid):
                datasets.append((item.text(), dataset))
        if not datasets and isinstance(self.active_dataset, RcsGrid):
            datasets.append(("Dataset", self.active_dataset))
        return datasets

    def _selected_datasets_ordered(
        self,
        *,
        use_selection_order: bool = False,
        empty_message: str = "Select two or more datasets to combine.",
    ) -> list[tuple[str, RcsGrid]] | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage(empty_message)
            return None

        selected_rows = [idx.row() for idx in selected]
        if use_selection_order:
            ordered_rows = [
                row for row in getattr(self, "_dataset_selection_order", []) if row in selected_rows
            ]
            for row in selected_rows:
                if row not in ordered_rows:
                    ordered_rows.append(row)
            selected_rows = ordered_rows
        else:
            selected_rows = sorted(selected_rows)

        datasets: list[tuple[str, RcsGrid]] = []
        for row in selected_rows:
            item = self.table.item(row, 0)
            if item is None:
                return None
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                return None
            datasets.append((item.text(), dataset))
        return datasets

    def _confirm_coherent_metadata(
        self,
        datasets: list[tuple[str, RcsGrid]],
        operation_name: str,
        *,
        independent: bool = False,
    ) -> bool | None:
        """Require explicit confirmation when coherent declarations are missing."""

        labels = _COHERENT_METADATA_LABELS
        missing = _missing_coherent_metadata_keys(
            [dataset for _name, dataset in datasets]
        )

        if not missing:
            return False

        missing_text = ", ".join(labels[key] for key in labels if key in missing)
        affected = []
        for name, dataset in datasets:
            absent = []
            getter = getattr(dataset, "_declared_scalar_metadata", None)
            for key, label in labels.items():
                value = getter(key) if callable(getter) else ""
                if not str(value or "").strip():
                    absent.append(label)
            if absent:
                affected.append(f"• {name}: {', '.join(absent)}")
        details = "\n".join(affected[:12])
        if len(affected) > 12:
            details += f"\n• …and {len(affected) - 12} more"
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        physical_statement = (
            "For each dataset, coherent filtering is physically meaningful only "
            "when the phase reference/center, phasor time convention, and "
            "polarization basis apply consistently across its filtered samples. "
            if independent
            else "A coherent result is physically meaningful only when phase "
            "center, phasor time convention, and polarization basis are "
            "compatible across the inputs. "
        )
        answer = QMessageBox.question(
            self,
            f"Confirm {operation_name} Assumptions",
            f"The selected datasets do not fully declare {missing_text}.\n\n"
            f"{details}\n\n"
            + physical_statement
            + "Proceed under that explicit assumption and record it in provenance?",
            buttons.Yes | buttons.No,
            buttons.No,
        )
        if answer != buttons.Yes:
            self.status.showMessage(
                f"{operation_name} cancelled: coherent metadata assumptions "
                "were not confirmed."
            )
            return None
        self.status.showMessage(
            f"{operation_name}: missing coherent declarations explicitly "
            "accepted; the assumption will be recorded."
        )
        return True

    def _combine_datasets_add(
        self,
        op_label: str,
        op_symbol: str,
        func_add: str,
        func_add_many: str,
        *,
        coherent: bool = False,
    ) -> None:
        datasets = self._selected_datasets_ordered()
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        base = datasets[0][1]
        # Coherent addition may hold two complex operands plus the retained
        # power/phase result. The common six-real-array estimate is adequate
        # for ordinary power arithmetic; reserve one more output estimate for
        # complex-field reconstruction.
        combine_extra = (
            _derived_grid_peak_bytes(base, base.rcs_power.shape)
            if coherent else 0
        )
        if not self._preflight_derived_outputs(
            op_label,
            [(base, base.rcs_power.shape)],
            extra_bytes=combine_extra,
        ):
            return
        metadata_attested = False
        if coherent:
            attestation = self._confirm_coherent_metadata(datasets, op_label)
            if attestation is None:
                return
            metadata_attested = attestation
        names = [name for name, _ in datasets]
        input_refs = self._python_input_references(datasets)
        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"

        def _calculate_result():
            if len(datasets) == 2:
                return getattr(base, func_add)(
                    datasets[1][1], metadata_attested=metadata_attested
                ) if coherent else getattr(base, func_add)(datasets[1][1])
            others = [ds for _, ds in datasets[1:]]
            return getattr(base, func_add_many)(
                *others, metadata_attested=metadata_attested
            ) if coherent else getattr(base, func_add_many)(*others)

        def _publish_result(result):
            output_id = self._add_dataset_row(
                result, new_name, history, file_name=""
            )
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and input_refs is not None:
                method = func_add if len(datasets) == 2 else func_add_many
                recorder.record_expression(
                    self._python_output_reference(output_id, new_name),
                    input_refs,
                    lambda variables, method=method, attested=metadata_attested: (
                        f"{variables[0]}.{method}({', '.join(variables[1:])}"
                        + (", metadata_attested=True" if attested else "")
                        + ")"
                    ),
                    comment=op_label,
                )
            self.status.showMessage(f"{op_label} created: {new_name}")

        if self._start_background_callable(
            op_label, _calculate_result, _publish_result
        ):
            self.status.showMessage(f"{op_label} is running in the background...")

    def _combine_datasets_sub(
        self,
        op_label: str,
        op_symbol: str,
        func_sub: str,
        *,
        coherent: bool = False,
        required_count: int | None = None,
    ) -> None:
        datasets = self._selected_datasets_ordered(use_selection_order=True)
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        if required_count is not None and len(datasets) != int(required_count):
            self.status.showMessage(
                f"{op_label}: select exactly {int(required_count)} datasets."
            )
            return
        base = datasets[0][1]
        combine_extra = (
            _derived_grid_peak_bytes(base, base.rcs_power.shape)
            if coherent else 0
        )
        if not self._preflight_derived_outputs(
            op_label,
            [(base, base.rcs_power.shape)],
            extra_bytes=combine_extra,
        ):
            return
        metadata_attested = False
        if coherent:
            attestation = self._confirm_coherent_metadata(datasets, op_label)
            if attestation is None:
                return
            metadata_attested = attestation
        names = [name for name, _ in datasets]
        input_refs = self._python_input_references(datasets)
        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"

        def _calculate_result():
            result = datasets[0][1]
            for _, ds in datasets[1:]:
                result = getattr(result, func_sub)(
                    ds, metadata_attested=metadata_attested
                ) if coherent else getattr(result, func_sub)(ds)
            return result

        def _publish_result(result):
            output_id = self._add_dataset_row(
                result, new_name, history, file_name=""
            )
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and input_refs is not None:
                recorder.record_expression(
                    self._python_output_reference(output_id, new_name),
                    input_refs,
                    lambda variables, method=func_sub, attested=metadata_attested: (
                        ".".join(
                            [variables[0]]
                            + [
                                f"{method}({variable}"
                                + (", metadata_attested=True" if attested else "")
                                + ")"
                                for variable in variables[1:]
                            ]
                        )
                    ),
                    comment=op_label,
                )
            self.status.showMessage(f"{op_label} created: {new_name}")

        if self._start_background_callable(
            op_label, _calculate_result, _publish_result
        ):
            self.status.showMessage(f"{op_label} is running in the background...")

    def _coherent_add_selected(self) -> None:
        self._combine_datasets_add(
            "Coherent +",
            "+",
            "coherent_add",
            "coherent_add_many",
            coherent=True,
        )

    def _coherent_sub_selected(self) -> None:
        self._combine_datasets_sub(
            "Coherent -", "-", "coherent_subtract", coherent=True
        )

    def _support_reference_difference_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message=(
                "Select target+support and support-only datasets, then choose "
                "Support Ref -."
            ),
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage(
                "Support Ref -: select at least two datasets to assign the "
                "target+support and support-only roles."
            )
            return

        dialog = SupportReferenceDifferenceDialog(datasets, parent=self)
        if dialog.exec() != QDialog.Accepted:
            dialog.deleteLater()
            return
        params = dialog.get_params()
        dialog.deleteLater()
        target_name, target = params["target"]
        support_name, support = params["support"]
        metadata_attested = bool(params["metadata_attested"])
        assumptions_attested = bool(params["assumptions_attested"])
        inputs = [(target_name, target), (support_name, support)]
        input_refs = self._python_input_references(inputs)
        output_name = f"SupportRef[{target_name} - {support_name}]"

        def compute():
            return target.support_referenced_difference(
                support,
                metadata_attested=metadata_attested,
                assumptions_attested=assumptions_attested,
                target_label=target_name,
                support_label=support_name,
            )

        def publish(result) -> None:
            raw_provenance = (result.extra or {}).get(
                "support_reference_difference_json", ""
            )
            if isinstance(raw_provenance, np.ndarray):
                raw_provenance = raw_provenance.reshape(()).item()
            provenance = json.loads(str(raw_provenance))
            qa = provenance["qa"]
            energies = qa["energy_sum_linear"]

            def _metric(value, *, suffix=""):
                if value is None:
                    return "not defined"
                return f"{float(value):.6g}{suffix}"

            common = int(qa["common_finite_sample_count"])
            total = int(qa["total_sample_count"])
            excluded = int(qa["excluded_sample_count"])
            coherence = qa.get("complex_coherence")
            coherence_phase = qa.get("complex_coherence_phase_deg")
            coherence_text = (
                f"{float(coherence):.6f} at "
                f"{float(coherence_phase):.3f} deg"
                if coherence is not None and coherence_phase is not None
                else "not meaningful (fewer than two common samples or zero energy)"
            )
            output_id = self._add_dataset_row(
                result, output_name, "", file_name=""
            )
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and input_refs is not None:
                recorder.record_expression(
                    self._python_output_reference(output_id, output_name),
                    input_refs,
                    lambda variables: (
                        f"{variables[0]}.support_referenced_difference("
                        f"{variables[1]}, "
                        f"target_label={target_name!r}, "
                        f"support_label={support_name!r})"
                    ),
                    comment=(
                        "Support-referenced exact complex difference; not a "
                        "reconstructed free-space target"
                    ),
                )
            self.status.showMessage(
                f"Support-referenced difference created: {output_name} (unsaved); "
                f"usable {common:,}/{total:,}, masked {excluded:,}, "
                f"after/before {_metric(qa.get('post_to_pre_energy_db'), suffix=' dB')}, "
                f"coherence {coherence_text}."
            )

        if self._start_background_callable(
            "Support-referenced difference", compute, publish
        ):
            self.status.showMessage(
                "Support-referenced exact complex subtraction and QA are "
                "running in the background..."
            )

    def _incoherent_add_selected(self) -> None:
        self._combine_datasets_add("Incoherent +", "+", "incoherent_add", "incoherent_add_many")

    def _incoherent_sub_selected(self) -> None:
        self._combine_datasets_sub("Incoherent -", "-", "incoherent_subtract")

    def _dbdiff_selected(self) -> None:
        self._combine_datasets_sub(
            "Δ dB", "Δ", "arithmetic_db_subtract", required_count=2
        )

    def _audit_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to audit.",
        )
        if datasets is None:
            return

        def compute():
            reports = []
            for name, dataset in datasets:
                try:
                    report = dataset.audit()
                except Exception as exc:
                    report = {
                        "status": "error",
                        "errors": [f"audit could not inspect this dataset: {exc}"],
                        "warnings": [],
                        "info": [],
                        "metrics": {},
                    }
                reports.append((name, report))
            return reports

        def publish(reports) -> None:
            dialog = DatasetAuditDialog(reports, parent=self)
            dialog.exec()
            dialog.deleteLater()
            counts = {"pass": 0, "warn": 0, "fail": 0}
            status_keys = {
                "ok": "pass",
                "pass": "pass",
                "warning": "warn",
                "warn": "warn",
                "error": "fail",
                "fail": "fail",
            }
            for _name, report in reports:
                status = str(report.get("status", "error")).strip().lower()
                counts[status_keys.get(status, "warn")] += 1
            self.status.showMessage(
                "Dataset audit complete: "
                f"{counts['pass']} pass, {counts['warn']} warning, {counts['fail']} fail."
            )

        if self._start_background_callable("Dataset audit", compute, publish):
            self.status.showMessage(f"Auditing {len(datasets)} dataset(s)...")

    def _compatibility_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets to compare compatibility.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage(
                "Compatibility needs at least two selected datasets."
            )
            return
        reference_name, reference = datasets[0]

        def compute():
            blocks = [
                "Operand 1 (reference): " + reference_name,
                "Selection order: " + " -> ".join(name for name, _grid in datasets),
            ]
            pass_count = 0
            warning_count = 0
            fail_count = 0
            for operand_index, (name, dataset) in enumerate(datasets[1:], start=2):
                lines = [f"Operand {operand_index}: {name}"]
                axis_ok = False
                physical_ok = False
                exact_ok = False
                coherent_ok = False
                try:
                    reference._assert_axis_metadata_compatible(dataset)
                except Exception as exc:
                    lines.append(f"  FAIL axis frame/units: {exc}")
                    fail_count += 1
                else:
                    axis_ok = True
                    lines.append("  PASS axis frame/units")
                    pass_count += 1
                try:
                    reference._assert_physical_metadata_compatible(dataset)
                except Exception as exc:
                    lines.append(f"  FAIL physical quantity: {exc}")
                    fail_count += 1
                else:
                    physical_ok = True
                    lines.append("  PASS physical quantity/log unit")
                    pass_count += 1
                try:
                    reference._assert_compatible(dataset)
                except Exception as exc:
                    lines.append(f"  FAIL exact element-wise grid: {exc}")
                    fail_count += 1
                else:
                    exact_ok = True
                    lines.append("  PASS exact element-wise grid")
                    pass_count += 1
                try:
                    reference._assert_compatible(
                        dataset,
                        coherent=True,
                        coherent_metadata_attested=False,
                        _scan_phase_samples=False,
                    )
                except Exception as exc:
                    lines.append(f"  FAIL coherent declarations: {exc}")
                    fail_count += 1
                else:
                    coherent_ok = True
                    missing = _missing_coherent_metadata_keys(
                        (reference, dataset)
                    )
                    if missing:
                        rendered = ", ".join(
                            _COHERENT_METADATA_LABELS[key]
                            for key in _COHERENT_METADATA_LABELS
                            if key in missing
                        )
                        lines.append(
                            "  WARN coherent declarations missing: " + rendered
                        )
                        warning_count += 1
                    else:
                        lines.append("  PASS coherent declarations")
                        pass_count += 1
                available = []
                if axis_ok:
                    available.extend(("Align", "Overlap"))
                if physical_ok:
                    available.extend(("Join", "Merge Overlaps"))
                if exact_ok:
                    available.extend(("Incoherent +/-", "Delta dB"))
                if coherent_ok and exact_ok:
                    available.extend(("Coherent +/-", "Coherent divide"))
                lines.append(
                    "  Compatible operation families: "
                    + (", ".join(available) if available else "none without repair")
                )
                blocks.append("\n".join(lines))
            summary = (
                f"Checks: {pass_count} pass, {warning_count} warning, "
                f"{fail_count} fail"
            )
            return summary + "\n\n" + "\n\n".join(blocks)

        def publish(report_text) -> None:
            dialog = DatasetCompatibilityDialog(report_text, parent=self)
            dialog.exec()
            dialog.deleteLater()
            first_line = str(report_text).splitlines()[0]
            self.status.showMessage("Dataset compatibility complete: " + first_line)

        if self._start_background_callable(
            "Dataset compatibility", compute, publish
        ):
            self.status.showMessage(
                f"Comparing {len(datasets)} datasets against operand 1..."
            )

    @staticmethod
    def _provenance_value_text(key: str, value) -> str:
        """Format metadata without materializing or printing large arrays."""

        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return repr(value)
        if array.size > 16:
            return (
                f"<array shape={tuple(int(v) for v in array.shape)!r}, "
                f"dtype={array.dtype}, bytes={int(array.nbytes):,}>"
            )
        if array.size == 1:
            try:
                scalar = array.reshape(()).item()
            except ValueError:
                scalar = value
            if str(key).endswith("_json") and isinstance(scalar, str):
                try:
                    return json.dumps(
                        json.loads(scalar), indent=2, sort_keys=True
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            return repr(scalar)
        return repr(array.tolist())

    def _provenance_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to inspect provenance.",
        )
        if datasets is None:
            return
        blocks = []
        for operand_index, (name, dataset) in enumerate(datasets, start=1):
            reference = self._python_reference_for_dataset(dataset)
            source = reference.path if reference is not None else dataset.source_path
            lines = [
                f"Operand {operand_index}: {name}",
                f"Source: {source or 'unsaved / in-memory'}",
                f"Shape: {tuple(int(value) for value in dataset.rcs_power.shape)!r}",
                f"Coordinate system: {dataset.angular_coordinate_system()}",
                "Units:",
                json.dumps(dict(dataset.units or {}), indent=2, sort_keys=True),
                "History:",
                str(dataset.history or "(none)"),
                "Extra metadata:",
            ]
            if dataset.extra:
                for key in sorted(dataset.extra, key=str):
                    rendered = self._provenance_value_text(
                        str(key), dataset.extra[key]
                    )
                    indented = rendered.replace("\n", "\n    ")
                    lines.append(f"  {key}: {indented}")
            else:
                lines.append("  (none)")
            blocks.append("\n".join(lines))
        dialog = DatasetProvenanceDialog("\n\n".join(blocks), parent=self)
        dialog.exec()
        dialog.deleteLater()
        self.status.showMessage(
            f"Displayed provenance for {len(datasets)} dataset(s)."
        )

    def _join_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets to join.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to join.")
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        worker = _JoinDatasetsWorker(grids, tol=1e-6)
        worker.progress.connect(self._on_join_worker_progress)
        worker.finished.connect(self._on_join_worker_finished)
        if not self._try_start_background_job("Dataset join", worker):
            return
        self._pending_join_names = names
        self._pending_join_references = self._python_input_references(datasets)
        self.status.showMessage(f"Join... 0/{len(grids)}")

    def _stitch_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets with overlaps to merge.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to merge.")
            return

        dialog = StitchDialog([name for name, _dataset in datasets], parent=self)
        if dialog.exec() != QDialog.Accepted:
            dialog.deleteLater()
            return
        params = dialog.get_params()
        dialog.deleteLater()
        policy = str(params["policy"])
        tolerance = float(params["tol"])
        metadata_attested = False
        if policy == "coherent-mean":
            attestation = self._confirm_coherent_metadata(
                datasets, "Coherent overlap merge"
            )
            if attestation is None:
                return
            metadata_attested = attestation

        names = [name for name, _dataset in datasets]
        grids = [dataset for _name, dataset in datasets]
        references = self._python_input_references(datasets)
        memory_limit = _derived_grid_memory_limit()

        def compute():
            return RcsGrid.stitch_many(
                *grids,
                policy=policy,
                tol=tolerance,
                metadata_attested=metadata_attested,
                max_output_bytes=memory_limit,
                return_report=True,
            )

        def publish(payload) -> None:
            stitched, report = payload
            overlap = int(report.get("overlap_count", 0) or 0)
            equal = int(report.get("equal_count", 0) or 0)
            conflicting = int(report.get("conflict_count", 0) or 0)
            contributors = int(report.get("contributing_count", 0) or 0)
            finite_output = int(report.get("output_finite_count", 0) or 0)
            missing_output = int(report.get("missing_count", 0) or 0)
            max_contributors = int(report.get("max_contributors", 0) or 0)
            output_name = " ⊕ ".join(names) + f" [Merge {policy}]"
            history = (
                f"Merge Overlaps ({policy}, tol={tolerance:g}, overlap={overlap}, "
                f"conflicts={conflicting}): " + " -> ".join(names)
            )
            output_id = self._add_dataset_row(
                stitched, output_name, history, file_name=""
            )
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and references is not None:
                recorder.record_function(
                    self._python_output_reference(output_id, output_name),
                    "stitch_datasets",
                    references,
                    kwargs={
                        "policy": policy,
                        "tol": tolerance,
                    },
                    comment=f"Merge {len(datasets)} overlapping datasets using {policy}",
                )
            self.status.showMessage(
                f"Overlap merge created 1 dataset from {len(datasets)} operands; "
                f"{overlap:,} overlap cell(s), {conflicting:,} conflict(s) "
                f"resolved by {policy}, {missing_output:,} missing output cell(s), "
                f"maximum {max_contributors:,} contributor(s) per cell."
            )

        if self._start_background_callable("Dataset overlap merge", compute, publish):
            self.status.showMessage(
                f"Merging {len(datasets)} datasets and analyzing overlaps..."
            )

    def _overlap_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets for overlap.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets for overlap.")
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        input_refs = self._python_input_references(datasets)

        upper_shape = tuple(
            min(len(grid.get_axis(axis_name)) for grid in grids)
            for axis_name in (
                "azimuth",
                "elevation",
                "frequency",
                "polarization",
            )
        )
        overlap_peak = sum(
            _derived_grid_peak_bytes(grid, upper_shape) for grid in grids
        ) + math.prod(upper_shape)
        memory_limit = _derived_grid_memory_limit()
        if overlap_peak > memory_limit:
            self.status.showMessage(
                "Overlap blocked before allocation: the common-grid upper-bound "
                f"working set {_format_bytes(overlap_peak)} exceeds the current "
                f"safety limit {_format_bytes(memory_limit)}. Select fewer or "
                "smaller datasets."
            )
            return

        def publish(overlap_grids) -> None:
            if not isinstance(overlap_grids, (tuple, list)) or len(overlap_grids) != len(datasets):
                self.status.showMessage("Overlap failed: worker returned invalid outputs.")
                return
            output_refs: list[DatasetReference] = []
            for (name, _), overlap_grid in zip(datasets, overlap_grids):
                if not isinstance(overlap_grid, RcsGrid):
                    self.status.showMessage("Overlap failed: worker returned an invalid grid.")
                    return
                history = f"Overlap with [{', '.join(names)}]: {name}"
                output_name = f"{name} [Overlap]"
                output_id = self._add_dataset_row(
                    overlap_grid, output_name, history, file_name=""
                )
                output_refs.append(
                    self._python_output_reference(output_id, output_name)
                )
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and input_refs is not None:
                recorder.record_multi_function(
                    output_refs,
                    "RcsGrid.overlap_many",
                    input_refs,
                    kwargs={"tol": 1.0e-6},
                    comment="Crop datasets to their common finite overlap",
                )
            self.status.showMessage(
                f"Overlap created {len(output_refs)} dataset(s)."
            )

        if self._start_background_callable(
            "Dataset overlap",
            lambda: RcsGrid.overlap_many(*grids, tol=1.0e-6),
            publish,
        ):
            self.status.showMessage(
                f"Finding the common finite overlap for {len(grids)} datasets..."
            )


    def _prompt_choice(self, title: str, label: str, choices: list[str], default_idx: int = 0) -> str | None:
        value, ok = QInputDialog.getItem(self, title, label, choices, default_idx, False)
        if not ok:
            return None
        return str(value)

    def _slice_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to crop or slice.",
        )
        if datasets is None:
            return

        reference = (
            self.active_dataset
            if isinstance(getattr(self, "active_dataset", None), RcsGrid)
            else datasets[0][1]
        )
        sel_az = self._selected_values(self.list_az)
        sel_el = self._selected_values(self.list_elev)
        sel_freq = self._selected_values(self.list_freq)
        sel_pol = self._selected_values(self.list_pol)
        has_selected = bool(sel_az or sel_el or sel_freq or sel_pol)
        try:
            dialog = CropDialog(
                reference,
                has_selected_values=has_selected,
                parent=self,
            )
        except (TypeError, ValueError) as exc:
            self.status.showMessage(
                f"Crop / Slice blocked: active reference metadata is invalid ({exc})"
            )
            return
        if dialog.exec() != QDialog.Accepted:
            dialog.deleteLater()
            return
        params = dialog.get_params()
        dialog.deleteLater()
        mode = str(params["mode"])
        if mode == "selected" and not has_selected:
            self.status.showMessage(
                "Crop / Slice: select at least one parameter value or use numeric ranges."
            )
            return

        range_params = params["ranges"]
        stride_params = params["strides"]
        try:
            selected_az_values = np.asarray(sel_az, dtype=float)
            selected_el_values = np.asarray(sel_el, dtype=float)
            if _canonical_angle_unit(
                (reference.units or {}).get("azimuth", "deg")
            ) == "rad":
                selected_az_values = np.rad2deg(selected_az_values)
            if _canonical_angle_unit(
                (reference.units or {}).get("elevation", "deg")
            ) == "rad":
                selected_el_values = np.rad2deg(selected_el_values)
            selected_az_deg = selected_az_values.tolist()
            selected_el_deg = selected_el_values.tolist()
            selected_freq_hz = (
                _frequency_axis_hz(reference, sel_freq).tolist()
                if sel_freq
                else []
            )
            ref_frequency_unit = _canonical_frequency_unit(
                (reference.units or {}).get("frequency", "GHz")
            )
            ref_frequency_scale = _FREQUENCY_TO_HZ[ref_frequency_unit.lower()]
            frequency_range_ref = range_params.get("frequency")
            frequency_range_hz = (
                tuple(
                    float(value) * ref_frequency_scale
                    for value in frequency_range_ref
                )
                if frequency_range_ref is not None
                else None
            )
        except (TypeError, ValueError, IndexError) as exc:
            self.status.showMessage(
                f"Crop / Slice blocked: selected reference values are invalid ({exc})"
            )
            return
        selected_pols = (
            list(sel_pol) if params.get("selected_polarizations") else None
        )
        if params.get("selected_polarizations") and not selected_pols:
            self.status.showMessage(
                "Crop / Slice: select at least one polarization or disable the "
                "polarization limit."
            )
            return

        plans: list[tuple[str, RcsGrid, dict[str, object], tuple[int, int, int, int]]] = []
        plan_errors: list[str] = []
        estimated_peak = 0
        for name, dataset in datasets:
            try:
                transfers_angular_values = (
                    bool(selected_az_deg or selected_el_deg)
                    if mode == "selected"
                    else bool(
                        range_params.get("azimuth") is not None
                        or range_params.get("elevation") is not None
                    )
                )
                if transfers_angular_values:
                    _assert_same_angular_frame(reference, dataset)
                if mode == "selected":
                    kwargs = {
                        "azimuths": (
                            _degrees_to_angle_axis(dataset, "azimuth", selected_az_deg).tolist()
                            if selected_az_deg else None
                        ),
                        "elevations": (
                            _degrees_to_angle_axis(dataset, "elevation", selected_el_deg).tolist()
                            if selected_el_deg else None
                        ),
                        "frequencies": (
                            _hz_to_frequency_axis(dataset, selected_freq_hz).tolist()
                            if selected_freq_hz else None
                        ),
                        "polarizations": list(sel_pol) if sel_pol else None,
                        "azimuth_stride": 1,
                        "elevation_stride": 1,
                        "frequency_stride": 1,
                    }
                    shape = (
                        len(selected_az_deg) if selected_az_deg else len(dataset.azimuths),
                        len(selected_el_deg) if selected_el_deg else len(dataset.elevations),
                        len(selected_freq_hz) if selected_freq_hz else len(dataset.frequencies),
                        len(sel_pol) if sel_pol else len(dataset.polarizations),
                    )
                else:
                    az_range = range_params.get("azimuth")
                    el_range = range_params.get("elevation")
                    native_az_range = (
                        tuple(_degrees_to_angle_axis(dataset, "azimuth", az_range).tolist())
                        if az_range is not None else None
                    )
                    native_el_range = (
                        tuple(_degrees_to_angle_axis(dataset, "elevation", el_range).tolist())
                        if el_range is not None else None
                    )
                    native_freq_range = (
                        tuple(_hz_to_frequency_axis(dataset, frequency_range_hz).tolist())
                        if frequency_range_hz is not None else None
                    )
                    kwargs = {
                        "azimuth_range": native_az_range,
                        "elevation_range": native_el_range,
                        "frequency_range": native_freq_range,
                        "azimuth_stride": int(stride_params["azimuth"]),
                        "elevation_stride": int(stride_params["elevation"]),
                        "frequency_stride": int(stride_params["frequency"]),
                        "polarizations": selected_pols,
                    }

                    def retained(axis_values, bounds, stride):
                        values = np.asarray(axis_values, dtype=float)
                        if bounds is None:
                            count = values.size
                        else:
                            lo, hi = sorted(map(float, bounds))
                            # Match RcsGrid.axis_crop's native-axis tolerance
                            # so the memory plan cannot reject a boundary bin
                            # that the actual crop would retain.
                            count = int(
                                np.count_nonzero(
                                    (values >= lo - 1.0e-6)
                                    & (values <= hi + 1.0e-6)
                                )
                            )
                        return (count + int(stride) - 1) // int(stride)

                    shape = (
                        retained(dataset.azimuths, native_az_range, kwargs["azimuth_stride"]),
                        retained(dataset.elevations, native_el_range, kwargs["elevation_stride"]),
                        retained(dataset.frequencies, native_freq_range, kwargs["frequency_stride"]),
                        len(selected_pols) if selected_pols else len(dataset.polarizations),
                    )
                if any(int(size) < 1 for size in shape):
                    raise ValueError("requested crop leaves an empty axis")
                estimated_peak += _derived_grid_peak_bytes(dataset, shape)
                plans.append((name, dataset, kwargs, shape))
            except (TypeError, ValueError) as exc:
                plan_errors.append(f"{name} ({exc})")

        if plan_errors:
            self.status.showMessage(
                "Crop / Slice blocked: " + _compact_item_summary(plan_errors)
            )
            return
        memory_limit = _derived_grid_memory_limit()
        if estimated_peak > memory_limit:
            self.status.showMessage(
                "Crop / Slice blocked before allocation: estimated working set "
                f"{_format_bytes(estimated_peak)} exceeds the current safety limit "
                f"{_format_bytes(memory_limit)}. Tighten the ranges, increase stride, "
                "or process fewer datasets."
            )
            return

        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset, _kwargs, _shape in plans
        ]

        def compute():
            results = []
            skipped = []
            for plan_index, (name, dataset, kwargs, _shape) in enumerate(plans):
                try:
                    result = crop_dataset(dataset, **kwargs)
                except (TypeError, ValueError) as exc:
                    skipped.append(f"{name} ({exc})")
                    continue
                results.append((plan_index, name, result, kwargs))
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            recorder = getattr(self, "python_recorder", None)
            for plan_index, name, result, kwargs in results:
                history = (
                    f"Crop / Slice ({mode}): {name} | az={len(result.azimuths)}, "
                    f"el={len(result.elevations)}, freq={len(result.frequencies)}, "
                    f"pol={len(result.polarizations)}"
                )
                output_name = f"{name} [Crop]"
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[plan_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "crop_dataset",
                        [source_ref],
                        kwargs=kwargs,
                        comment=f"Crop / Slice {name}",
                    )
            produced = len(results)
            if produced == 0:
                self.status.showMessage("Crop / Slice created 0 datasets.")
            elif skipped:
                self.status.showMessage(
                    f"Crop / Slice created {produced} dataset(s). Skipped: "
                    + _compact_item_summary(skipped)
                )
            else:
                self.status.showMessage(
                    f"Crop / Slice created {produced} dataset(s)."
                )

        if self._start_background_callable("Dataset crop", compute, publish):
            self.status.showMessage(
                f"Cropping {len(plans)} dataset(s) in the background..."
            )

    def _percentile_selected(self) -> None:
        """Replace every azimuth sample with its azimuth-percentile value."""

        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets for percentile reduction.",
        )
        if datasets is None:
            return
        percentile, accepted = QInputDialog.getDouble(
            self,
            "Azimuth Percentile",
            "Linear-power percentile across azimuth (0-100):",
            90.0,
            0.0,
            100.0,
            1,
        )
        if not accepted:
            return
        self._create_statistics_datasets(
            datasets,
            statistic="percentile",
            percentile=float(percentile),
            axes=("azimuth",),
            broadcast_reduced=True,
            operation_title="Percentile",
            output_qualifier=" az",
        )

    def _statistics_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets for statistics.",
        )
        if datasets is None:
            return

        dlg = StatisticsDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        params = dlg.get_params()
        if len(params) == 3:
            statistic, percentile, axes = params
            broadcast_reduced = False
        else:
            statistic, percentile, axes, broadcast_reduced = params
        if not axes:
            self.status.showMessage("Select at least one axis for statistics reduction.")
            return
        self._create_statistics_datasets(
            datasets,
            statistic=statistic,
            percentile=percentile,
            axes=axes,
            broadcast_reduced=bool(broadcast_reduced),
        )

    def _create_statistics_datasets(
        self,
        datasets,
        *,
        statistic,
        percentile,
        axes,
        broadcast_reduced,
        operation_title="Statistics",
        output_qualifier="",
    ) -> None:
        """Preflight and run a linear-power statistics reduction."""

        stat_label = f"p{percentile:g}" if statistic == "percentile" else statistic

        axis_numbers = {
            "azimuth": 0,
            "elevation": 1,
            "frequency": 2,
            "polarization": 3,
        }
        reduce_indices = {axis_numbers[name] for name in axes}
        retained_output_bytes = 0
        per_dataset_workspace_bytes = 0
        for _name, dataset in datasets:
            source_shape = tuple(int(value) for value in dataset.rcs_power.shape)
            output_shape = (
                source_shape
                if broadcast_reduced
                else tuple(
                    1 if index in reduce_indices else length
                    for index, length in enumerate(source_shape)
                )
            )
            output_cells = math.prod(output_shape)
            source_cells = math.prod(source_shape)
            working_itemsize = max(
                8,
                np.dtype(dataset.rcs_power.dtype).itemsize,
                np.dtype(dataset.rcs_phase.dtype).itemsize,
            )
            # Final power and phase arrays from earlier datasets remain in the
            # result list. Median/percentile may partition several input-sized
            # buffers; output construction may simultaneously hold a broadcast
            # value plus sanitized power/phase arrays. Those phases are not
            # concurrent, so use their maximum rather than double-counting.
            retained_output_bytes += int(output_cells * working_itemsize * 2)
            reduction_workspace = int(
                source_cells
                * working_itemsize
                * (4 if statistic in {"median", "percentile"} else 2)
            )
            construction_workspace = int(output_cells * working_itemsize * 4)
            per_dataset_workspace_bytes = max(
                per_dataset_workspace_bytes,
                reduction_workspace,
                construction_workspace,
            )
        estimated_peak = retained_output_bytes + per_dataset_workspace_bytes
        memory_limit = _derived_grid_memory_limit()
        if estimated_peak > memory_limit:
            self.status.showMessage(
                f"{operation_title} blocked before allocation: estimated working set "
                f"{_format_bytes(estimated_peak)} exceeds the current safety "
                f"limit {_format_bytes(memory_limit)}. Reduce fewer datasets at "
                "once or keep compact output enabled."
            )
            return

        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def compute():
            results = []
            skipped = []
            for dataset_index, (name, dataset) in enumerate(datasets):
                try:
                    result = dataset.statistics_dataset(
                        statistic=statistic,
                        axes=axes,
                        domain="magnitude",
                        percentile=percentile,
                        broadcast_reduced=bool(broadcast_reduced),
                    )
                except (ValueError, TypeError) as exc:
                    skipped.append(f"{name} ({exc})")
                    continue
                results.append((dataset_index, name, result))
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            for dataset_index, name, stat_grid in results:
                history = (
                    f"{operation_title} ({stat_label}, linear power, axes={axes}): "
                    f"{name}"
                )
                output_name = f"{name} [{stat_label}{output_qualifier}]"
                output_id = self._add_dataset_row(
                    stat_grid, output_name, history, file_name=""
                )
                source_ref = source_references[dataset_index]
                recorder = getattr(self, "python_recorder", None)
                if recorder is not None and source_ref is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        source_ref,
                        "statistics_dataset",
                        kwargs={
                            "statistic": statistic,
                            "axes": axes,
                            "domain": "magnitude",
                            "percentile": percentile,
                            "broadcast_reduced": bool(broadcast_reduced),
                        },
                        comment=(
                            f"Reduce {name} to {stat_label} statistics on linear power"
                        ),
                    )
            produced = len(results)
            if produced == 0:
                self.status.showMessage(f"{operation_title} created 0 datasets.")
            elif skipped:
                self.status.showMessage(
                    f"{operation_title} created {produced} dataset(s). Skipped: "
                    + ", ".join(skipped)
                )
            else:
                self.status.showMessage(
                    f"{operation_title} created {produced} dataset(s)."
                )

        if self._start_background_callable(
            f"Dataset {operation_title.lower()}", compute, publish
        ):
            self.status.showMessage(
                f"Computing {stat_label} linear-power statistics for "
                f"{len(datasets)} dataset(s)..."
            )


    def _delete_selected_datasets(self) -> None:
        if self._background_job_active():
            self.status.showMessage(
                "Wait for the active dataset job to finish before deleting rows."
            )
            return
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select one or more datasets to delete.")
            return
        rows = sorted((idx.row() for idx in selected), reverse=True)
        dirty_rows = [row for row in rows if self._dataset_row_is_dirty(row)]
        if dirty_rows:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            names = []
            for row in dirty_rows[:10]:
                item = self.table.item(row, 0)
                names.append(item.text() if item is not None else f"Dataset {row + 1}")
            details = "\n".join(f"• {name}" for name in names)
            if len(dirty_rows) > len(names):
                details += f"\n• …and {len(dirty_rows) - len(names)} more"
            answer = QMessageBox.question(
                self,
                "Delete Unsaved Datasets?",
                f"{len(dirty_rows)} selected dataset(s) have never been saved:\n\n"
                f"{details}\n\nDelete them? The next Undo Delete action can "
                "restore this batch.",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                self.status.showMessage("Delete cancelled; unsaved datasets were kept.")
                return
        deleted_rows = []
        for row in sorted(rows):
            items = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                items.append(item.clone() if item is not None else None)
            deleted_rows.append((row, items))
        for row in rows:
            self.table.removeRow(row)
        self._last_deleted_dataset_rows = deleted_rows
        self.active_dataset = None
        self._clear_param_lists()
        notify = getattr(self, "_notify_dataset_catalog_changed", None)
        if callable(notify):
            notify()
        self.status.showMessage(
            f"Deleted {len(rows)} dataset(s). Undo Delete can restore this batch."
        )

    def _undo_last_deleted_datasets(self) -> None:
        if self._background_job_active():
            self.status.showMessage(
                "Wait for the active dataset job to finish before restoring rows."
            )
            return
        deleted_rows = list(getattr(self, "_last_deleted_dataset_rows", ()))
        if not deleted_rows:
            self.status.showMessage("There is no deleted dataset batch to restore.")
            return
        signals_were_blocked = self.table.blockSignals(True)
        restored_rows = []
        try:
            for original_row, items in sorted(deleted_rows, key=lambda entry: entry[0]):
                row = min(max(int(original_row), 0), self.table.rowCount())
                self.table.insertRow(row)
                for column, item in enumerate(items):
                    if item is not None:
                        self.table.setItem(row, column, item)
                restored_rows.append(row)
        finally:
            self.table.blockSignals(signals_were_blocked)
        self._last_deleted_dataset_rows = []
        notify = getattr(self, "_notify_dataset_catalog_changed", None)
        if callable(notify):
            notify()
        self.table.clearSelection()
        if restored_rows:
            self.table.selectRow(restored_rows[0])
        self.status.showMessage(
            f"Restored {len(restored_rows)} dataset(s) from the last deletion."
        )

    def _save_selected_datasets(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select one or more datasets to save.")
            return

        rows = sorted(idx.row() for idx in selected)
        if len(rows) == 1:
            row = rows[0]
            item = self.table.item(row, 0)
            if item is None:
                return
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                return
            name = item.text().strip() or "dataset"
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Dataset",
                f"{_sanitize_filename(name)}.grim",
                "GRIM Files (*.grim)",
            )
            if not path:
                return
            self._save_dataset_plan(
                [(row, dataset, _ensure_grim_output_path(path))],
                dialog_title="Save Dataset",
            )
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Save Selected Datasets"
        )
        if directory:
            self._save_rows_to_directory(
                rows, directory, dialog_title="Save Selected Datasets"
            )

    def _save_all_datasets(self) -> None:
        if self.table.rowCount() == 0:
            self.status.showMessage("No datasets to save.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Save All Datasets")
        if not directory:
            return
        self._save_rows_to_directory(
            list(range(self.table.rowCount())),
            directory,
            dialog_title="Save All Datasets",
        )

    def _save_dirty_datasets(self) -> None:
        """Save only rows whose current in-memory state is not on disk."""

        rows = self._dirty_dataset_rows()
        if not rows:
            self.status.showMessage("No dirty dataset rows to save.")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Save Dirty Datasets"
        )
        if not directory:
            return
        self._save_rows_to_directory(
            rows, directory, dialog_title="Save Dirty Datasets"
        )

    def _save_rows_to_directory(
        self, rows: list[int], directory: str, *, dialog_title: str
    ) -> bool:
        plan: list[tuple[int, RcsGrid, str]] = []
        for row in rows:
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                continue
            name = item.text().strip() or f"dataset_{row + 1}"
            filename = f"{_sanitize_filename(name)}.grim"
            plan.append((row, dataset, os.path.join(directory, filename)))
        return self._save_dataset_plan(plan, dialog_title=dialog_title)

    def _save_dataset_plan(
        self,
        plan: list[tuple[int, RcsGrid, str]],
        *,
        dialog_title: str,
    ) -> bool:
        """Preflight, stage, and publish one save plan without silent replacement."""

        if not plan:
            self.status.showMessage("No valid datasets to save.")
            return False

        targets = [_ensure_grim_output_path(path) for _row, _dataset, path in plan]
        duplicate_groups = _duplicate_target_groups(targets)
        if duplicate_groups:
            details = "\n".join(
                f"• {os.path.basename(group[0])} ({len(group)} datasets)"
                for group in duplicate_groups
            )
            QMessageBox.critical(
                self,
                "Duplicate Output Names",
                "Multiple dataset names resolve to the same output after "
                "filename sanitizing and case-folding. Rename them before saving:\n\n"
                + details,
            )
            self.status.showMessage("Save cancelled: duplicate output names.")
            return False

        directory_targets = [path for path in targets if os.path.isdir(path)]
        if directory_targets:
            QMessageBox.critical(
                self,
                "Invalid Output Target",
                "A planned dataset output is an existing directory:\n\n"
                + "\n".join(directory_targets),
            )
            self.status.showMessage("Save cancelled: an output target is a directory.")
            return False

        existing = [path for path in targets if os.path.lexists(path)]
        if existing:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            shown = "\n".join(f"• {os.path.basename(path)}" for path in existing[:12])
            if len(existing) > 12:
                shown += f"\n• …and {len(existing) - 12} more"
            answer = QMessageBox.question(
                self,
                "Replace Existing Dataset Files?",
                f"{len(existing)} existing file(s) will be replaced:\n\n{shown}\n\n"
                "Replace all listed files?",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                self.status.showMessage("Save cancelled; no files were changed.")
                return False

        save_entries: list[tuple[RcsGrid, str, str]] = []
        row_snapshots: list[tuple[str, RcsGrid, str]] = []
        for (row, dataset, _raw_target), target in zip(plan, targets):
            name_item = self.table.item(row, 0)
            if name_item is None:
                self.status.showMessage(
                    "Save cancelled: a planned dataset row is no longer available."
                )
                return False
            dataset_id = str(name_item.data(DATASET_ID_ROLE) or "")
            if not dataset_id:
                dataset_id = uuid.uuid4().hex
                name_item.setData(DATASET_ID_ROLE, dataset_id)
            history_item = self.table.item(row, 2)
            row_history = (
                history_item.text()
                if history_item is not None
                else str(dataset.history or "")
            )
            save_entries.append((dataset, target, row_history))
            row_snapshots.append((dataset_id, dataset, target))

        def compute_save():
            compression_log: list[dict[str, object]] = []
            try:
                return {
                    "published": _stage_and_publish_grim_batch(
                        save_entries,
                        compression_log=compression_log,
                    ),
                    "compression": compression_log,
                    "error": None,
                }
            except Exception as exc:
                return {
                    "published": [],
                    "compression": compression_log,
                    "error": exc,
                }

        def publish_save(payload) -> None:
            error = payload.get("error") if isinstance(payload, dict) else None
            if error is not None:
                failure_text = (
                    str(error)
                    if isinstance(error, _GrimBatchRollbackError)
                    else "No partial batch was kept. " + str(error)
                )
                QMessageBox.critical(
                    self,
                    f"{dialog_title} Failed",
                    failure_text,
                )
                self.status.showMessage(f"Save failed: {error}")
                return

            published = list(payload.get("published", []))
            if len(published) != len(row_snapshots):
                self.status.showMessage(
                    "Save failed: worker returned an incomplete publication list."
                )
                return

            recorded_saves: list[tuple[DatasetReference, str]] = []
            rows_not_marked = 0
            for (dataset_id, saved_dataset, _target), output_path in zip(
                row_snapshots, published
            ):
                found_row = None
                for candidate in range(self.table.rowCount()):
                    candidate_item = self.table.item(candidate, 0)
                    if (
                        candidate_item is not None
                        and str(candidate_item.data(DATASET_ID_ROLE) or "")
                        == dataset_id
                    ):
                        found_row = candidate
                        break
                if found_row is None:
                    rows_not_marked += 1
                    continue
                name_item = self.table.item(found_row, 0)
                if (
                    name_item is None
                    or name_item.data(Qt.UserRole) is not saved_dataset
                ):
                    # The user edited/replaced this row while compression was
                    # running. The published file is the launch-time snapshot,
                    # so the newer in-memory row must remain visibly unsaved.
                    rows_not_marked += 1
                    continue
                self._set_dataset_row_saved(found_row, output_path)
                recorded_saves.append(
                    (
                        DatasetReference(
                            dataset_id,
                            name_item.text(),
                            output_path,
                        ),
                        output_path,
                    )
                )

            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and len(recorded_saves) == 1:
                recorder.record_save(*recorded_saves[0])
            elif recorder is not None and recorded_saves:
                recorder.record_save_batch(recorded_saves)
            message = (
                f"Saved {len(published)} dataset(s) to "
                f"{os.path.dirname(os.path.abspath(published[0]))}."
            )
            compression = list(payload.get("compression", []))
            if compression:
                shown_modes = []
                for decision in compression[:3]:
                    mode = (
                        "compact"
                        if bool(decision.get("compressed", False))
                        else "fast uncompressed"
                    )
                    saving = 100.0 * float(
                        decision.get("estimated_savings_fraction", 0.0)
                    )
                    shown_modes.append(
                        f"{os.path.basename(str(decision.get('target', 'dataset')))}: "
                        f"{mode} ({saving:.0f}% sampled saving)"
                    )
                if len(compression) > len(shown_modes):
                    shown_modes.append(
                        f"and {len(compression) - len(shown_modes)} more"
                    )
                message += " Storage mode: " + "; ".join(shown_modes) + "."
            if rows_not_marked:
                message += (
                    f" {rows_not_marked} row(s) changed or were removed while "
                    "saving and remain unsaved in the GUI."
                )
            self.status.showMessage(message)

        started = self._start_background_callable(
            "Native dataset save", compute_save, publish_save
        )
        if started:
            self.status.showMessage(
                f"Saving {len(save_entries)} dataset(s) in the background…"
            )
        return started

    def _export_plot(self) -> None:
        if self.last_plot_mode == "isar_image":
            if getattr(self, "_isar_busy", False):
                self.status.showMessage(
                    "ISAR reconstruction is still running; wait for the latest "
                    "image before exporting."
                )
                return
            figure_is_current = getattr(self, "_isar_figure_is_current", None)
            if not callable(figure_is_current) or not figure_is_current():
                self.status.showMessage(
                    "The current ISAR settings have no completed image to export."
                )
                return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf)",
        )
        if not path:
            return
        root, ext = os.path.splitext(path)
        if not ext:
            if "PDF" in selected_filter:
                path = f"{path}.pdf"
            else:
                path = f"{path}.png"
        self.plot_figure.savefig(path, dpi=200, bbox_inches="tight")
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None:
            emit_plot = getattr(self, "_emit_last_successful_python_plot", None)
            if callable(emit_plot):
                emit_plot()
            # Plot wrappers freeze their resolved semantic spec only after a
            # successful render. Selector edits that fail validation therefore
            # cannot replace the export target with an invalid or stale spec.
            recorder.record_plot_save(path, dpi=200)
        self.status.showMessage(f"Plot exported: {os.path.basename(path)}")

    def _export_isar_result(self) -> None:
        """Save the latest source-bound numerical ISAR result off the GUI thread."""

        if getattr(self, "_isar_busy", False):
            self.status.showMessage(
                "ISAR reconstruction is still running; wait before exporting its result."
            )
            return
        payload = getattr(self, "_last_isar_artifact", None)
        result_is_current = getattr(self, "_isar_numerical_result_is_current", None)
        if (
            payload is None
            or not callable(result_is_current)
            or not result_is_current()
        ):
            self.status.showMessage(
                "No current completed ISAR result is available; render the image first."
            )
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Numerical ISAR Result",
            "isar-result.isar.npz",
            "GRIM ISAR Result (*.isar.npz)",
        )
        if not path:
            return
        band_results, manifest = payload

        def compute_export():
            from isar_artifact import save_isar_artifact

            return save_isar_artifact(path, band_results, manifest)

        def publish_export(saved_path):
            self.status.showMessage(
                f"ISAR result exported: {os.path.basename(str(saved_path))}"
            )

        if self._start_background_callable(
            "ISAR result export", compute_export, publish_export
        ):
            self.status.showMessage("Saving numerical ISAR result in the background…")

    def _on_plot_context_menu(self, pos) -> None:
        line = self._dataset_line_at_canvas_position(pos)
        if line is not None:
            self._show_dataset_plot_style_menu(line, self.plot_canvas.mapToGlobal(pos))
            return
        menu = QMenu(self)
        action_copy = menu.addAction("Copy Plot")
        action_fit_both = menu.addAction("Fit Both (Reset View)")
        action_zoom_box = menu.addAction("Zoom Box")
        action_zoom_box.setCheckable(True)
        action_zoom_box.setChecked(self._button_checked(getattr(self, "btn_zoom_box", None)))
        menu.addSeparator()
        pbp_menu = menu.addMenu("PBP Fill Mode")
        action_pbp_gray = pbp_menu.addAction("Gray")
        action_pbp_gray.setCheckable(True)
        action_pbp_gray.setChecked(self.pbp_fill_mode == "gray")
        action_pbp_rcs = pbp_menu.addAction("Heatmap (RCS Value)")
        action_pbp_rcs.setCheckable(True)
        action_pbp_rcs.setChecked(self.pbp_fill_mode == "heatmap_rcs")
        action_pbp_density = pbp_menu.addAction("Heatmap (Overlap Density)")
        action_pbp_density.setCheckable(True)
        action_pbp_density.setChecked(self.pbp_fill_mode == "heatmap_density")
        action = menu.exec(self.plot_canvas.mapToGlobal(pos))
        if action == action_copy:
            pixmap = self.plot_canvas.grab()
            QApplication.clipboard().setPixmap(pixmap)
            self.status.showMessage("Plot copied to clipboard.")
        elif action == action_fit_both:
            self._fit_both()
        elif action == action_zoom_box:
            btn_zoom_box = getattr(self, "btn_zoom_box", None)
            if btn_zoom_box is not None:
                btn_zoom_box.setChecked(not btn_zoom_box.isChecked())
        elif action in (action_pbp_gray, action_pbp_rcs, action_pbp_density):
            if action == action_pbp_gray:
                self.pbp_fill_mode = "gray"
            elif action == action_pbp_rcs:
                self.pbp_fill_mode = "heatmap_rcs"
            else:
                self.pbp_fill_mode = "heatmap_density"
            if self.last_plot_mode == "azimuth_rect":
                self._plot_azimuth_rect()
            elif self.last_plot_mode == "azimuth_polar":
                self._plot_azimuth_polar()
            elif self.last_plot_mode == "frequency":
                self._plot_frequency()
            elif self.last_plot_mode == "isar_image":
                self._plot_isar_image()

    def _on_dataset_header_double_clicked(self, section: int) -> None:
        if section != 0:
            return
        self.table.selectAll()

    def _on_dataset_context_menu(self, pos) -> None:
        if not self.table.selectionModel().selectedRows():
            index = self.table.indexAt(pos)
            if index.isValid():
                self.table.selectRow(index.row())
            else:
                return
        menu = QMenu(self)
        action_save = menu.addAction("Save")
        export_menu = menu.addMenu("Export as…")
        action_export_pio = export_menu.addAction("Pioneer (.pio)…")
        action_export_ptm = export_menu.addAction("PTM (.ptm)…")
        action_export_csv = export_menu.addAction("CSV…")
        action_coordinates = menu.addAction("Set Coordinates…")
        action_delete = menu.addAction("Delete")
        menu.addSeparator()
        action_color = menu.addAction("Text Color…")
        action_reset_color = menu.addAction("Reset Text Color")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == action_save:
            self._save_selected_datasets()
        elif action == action_export_pio:
            self._export_pio_selected()
        elif action == action_export_ptm:
            self._export_ptm_selected()
        elif action == action_export_csv:
            self._export_csv_selected()
        elif action == action_coordinates:
            self._set_coordinates_selected()
        elif action == action_delete:
            self._delete_selected_datasets()
        elif action == action_color:
            self._set_dataset_text_color()
        elif action == action_reset_color:
            self._reset_dataset_text_color()

    def _set_dataset_text_color(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if not rows:
            return
        initial = self.table.item(rows[0], 0)
        initial_color = initial.foreground().color() if initial else QColor()
        color = QColorDialog.getColor(initial_color, self, "Choose Text Color")
        if not color.isValid():
            return
        brush = QBrush(color)
        for row in rows:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setForeground(brush)

    def _reset_dataset_text_color(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        for row in rows:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setForeground(QBrush())

    def _align_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets to align (first = reference).",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to align (first = reference).")
            return

        ref_name, ref_grid = datasets[0]
        others = datasets[1:]
        dlg = AlignDialog(ref_name, len(others), parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        mode = dlg.get_mode()
        align_plans = []
        for _name, dataset in others:
            if mode == "interp":
                output_shape = tuple(int(value) for value in ref_grid.rcs_power.shape)
            elif mode == "intersect":
                output_shape = tuple(
                    min(int(left), int(right))
                    for left, right in zip(
                        dataset.rcs_power.shape, ref_grid.rcs_power.shape
                    )
                )
            else:
                output_shape = tuple(int(value) for value in dataset.rcs_power.shape)
            align_plans.append((dataset, output_shape))
        if not self._preflight_derived_outputs("Align", align_plans):
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in others
        ]
        reference_ref = self._python_reference_for_dataset(ref_grid)

        def compute(progress):
            results = []
            skipped = []
            total = len(others)
            for index, (name, dataset) in enumerate(others, start=1):
                try:
                    aligned = dataset.align_to(ref_grid, mode=mode)
                except (ValueError, TypeError) as exc:
                    skipped.append(f"{name} ({exc})")
                else:
                    results.append((index - 1, name, aligned))
                progress(index, total, name)
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, aligned in results:
                history = f"Align ({mode}) to {ref_name}: {name}"
                output_name = f"{name} [Aligned]"
                output_id = self._add_dataset_row(
                    aligned, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if (
                    recorder is not None
                    and source_ref is not None
                    and reference_ref is not None
                ):
                    recorder.record_expression(
                        self._python_output_reference(output_id, output_name),
                        [source_ref, reference_ref],
                        lambda variables, mode=mode: (
                            f"{variables[0]}.align_to({variables[1]}, mode={mode!r})"
                        ),
                        comment=f"Align {name} to {ref_name}",
                    )

            produced = len(results)
            if produced == 0:
                message = "Align created 0 datasets."
            else:
                message = f"Align created {produced} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        if self._start_background_callable(
            "Dataset alignment", compute, publish, reports_progress=True
        ):
            self.status.showMessage(
                f"Aligning {len(others)} dataset(s) to {ref_name}..."
            )

    def _interpolate_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to regrid.",
        )
        if datasets is None:
            return

        reference = (
            self.active_dataset
            if isinstance(getattr(self, "active_dataset", None), RcsGrid)
            else datasets[0][1]
        )
        try:
            dlg = RegridDialog(reference, parent=self)
        except (TypeError, ValueError) as exc:
            self.status.showMessage(
                f"Regrid blocked: active reference metadata is invalid ({exc})"
            )
            return
        if dlg.exec() != QDialog.Accepted:
            dlg.deleteLater()
            return
        params = dlg.get_params()
        dlg.deleteLater()
        axis = str(params["axis"])
        start = float(params["start"])
        stop = float(params["stop"])
        step = float(params["step"])
        display_unit = str(params["unit"])
        if not all(np.isfinite(value) for value in (start, stop, step)):
            self.status.showMessage("Regrid: start, stop, and step must be finite.")
            return
        if step <= 0.0 or stop < start:
            self.status.showMessage(
                "Regrid: step must be positive and stop must be greater than or equal to start."
            )
            return

        n_float = np.floor((stop - start) / step + 1e-9) + 1.0
        if not np.isfinite(n_float) or n_float < 1.0:
            self.status.showMessage("Regrid: the requested grid is not finite.")
            return
        if n_float > _MAX_EXPLICIT_AXIS_POINTS:
            self.status.showMessage(
                "Regrid blocked before allocation: the requested grid has "
                f"{int(n_float):,} points; the safety limit is "
                f"{_MAX_EXPLICIT_AXIS_POINTS:,}. Increase the step size."
            )
            return
        n = int(n_float)
        resolved_stop = float(start + step * max(0, n - 1))

        axis_index = {"azimuth": 0, "elevation": 1, "frequency": 2}[axis]
        estimated_peak = 0
        for _name, dataset in datasets:
            shape = list(dataset.rcs_power.shape)
            shape[axis_index] = n
            estimated_peak += _derived_grid_peak_bytes(dataset, shape)
        memory_limit = _derived_grid_memory_limit()
        if estimated_peak > memory_limit:
            self.status.showMessage(
                "Regrid blocked before allocation: estimated working set "
                f"{_format_bytes(estimated_peak)} exceeds the current safety "
                f"limit {_format_bytes(memory_limit)}. Increase the step size "
                "or process fewer datasets at once."
            )
            return

        # Keep one compact native-unit start/step pair per dataset. The full
        # n-point target is constructed only while that dataset is processed
        # in the worker, rather than retaining one large array per selection.
        native_specs: list[tuple[float, float]] = []
        downsampled: list[str] = []
        if axis == "frequency":
            try:
                ref_frequency_unit = _canonical_frequency_unit(
                    (reference.units or {}).get("frequency", "GHz")
                )
            except (TypeError, ValueError) as exc:
                self.status.showMessage(
                    f"Regrid blocked: active reference metadata is invalid ({exc})"
                )
                return
            reference_frequency_scale = _FREQUENCY_TO_HZ[
                ref_frequency_unit.lower()
            ]
            physical_step = step * reference_frequency_scale
        else:
            reference_frequency_scale = None
            physical_step = step

        for name, dataset in datasets:
            try:
                if axis in {"azimuth", "elevation"}:
                    _assert_same_angular_frame(reference, dataset)
                    angle_unit = _canonical_angle_unit(
                        (dataset.units or {}).get(axis, "deg")
                    )
                    native_scale = np.pi / 180.0 if angle_unit == "rad" else 1.0
                    native_start = start * native_scale
                    native_step = step * native_scale
                    source_physical = _angle_axis_degrees(dataset, axis)
                else:
                    dataset_frequency_unit = _canonical_frequency_unit(
                        (dataset.units or {}).get("frequency", "GHz")
                    )
                    dataset_frequency_scale = _FREQUENCY_TO_HZ[
                        dataset_frequency_unit.lower()
                    ]
                    conversion = reference_frequency_scale / dataset_frequency_scale
                    native_start = start * conversion
                    native_step = step * conversion
                    source_physical = _frequency_axis_hz(dataset)
                source_step = (
                    float(np.median(np.diff(source_physical)))
                    if source_physical.size > 1 else float("inf")
                )
                if (
                    np.isfinite(source_step)
                    and source_step > 0.0
                    and physical_step > source_step * 1.01
                ):
                    downsampled.append(name)
                native_specs.append((float(native_start), float(native_step)))
            except (TypeError, ValueError) as exc:
                self.status.showMessage(f"Regrid blocked: {name} ({exc})")
                return

        if downsampled:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            answer = QMessageBox.question(
                self,
                "Confirm Point-Sampled Downsampling",
                "The requested grid is coarser than the source sampling for:\n\n"
                + "\n".join(f"• {name}" for name in downsampled[:12])
                + (
                    f"\n• …and {len(downsampled) - 12} more"
                    if len(downsampled) > 12 else ""
                )
                + "\n\nRegrid performs complex point interpolation and does not "
                "apply an anti-alias filter. Continue with point-sampled "
                "downsampling?",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                self.status.showMessage(
                    "Regrid cancelled: coarser point sampling was not confirmed."
                )
                return
        downsampled_set = set(downsampled)

        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def compute():
            results = []
            skipped = []
            for dataset_index, ((name, dataset), native_spec) in enumerate(
                zip(datasets, native_specs)
            ):
                native_start, native_step = native_spec
                try:
                    native_values = (
                        native_start
                        + native_step * np.arange(n, dtype=float)
                    )
                    interpolated = regrid_axis(
                        dataset,
                        axis,
                        values=native_values,
                    )
                except (ValueError, TypeError) as exc:
                    skipped.append(f"{name} ({exc})")
                    continue
                results.append(
                    (
                        dataset_index,
                        name,
                        interpolated,
                        native_start,
                        native_step,
                    )
                )
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            for (
                dataset_index,
                name,
                interpolated,
                native_start,
                native_step,
            ) in results:
                history = (
                    f"Regrid {axis} [{start:g}..{resolved_stop:g} {display_unit}, "
                    f"step {step:g} {display_unit}, no extrapolation]: {name}"
                )
                if name in downsampled_set:
                    history += "; coarser point sampling; no anti-alias filter"
                output_name = f"{name} [Regrid {axis}]"
                output_id = self._add_dataset_row(
                    interpolated, output_name, history, file_name=""
                )
                source_ref = source_references[dataset_index]
                recorder = getattr(self, "python_recorder", None)
                if recorder is not None and source_ref is not None:
                    recorder.record_expression(
                        self._python_output_reference(output_id, output_name),
                        [source_ref],
                        lambda variables, selected_axis=axis, first=native_start, increment=native_step, count=n: (
                            f"regrid_axis({variables[0]}, {selected_axis!r}, "
                            f"values={first!r} + {increment!r} * "
                            f"np.arange({count}, dtype=float))"
                        ),
                        comment=f"Regrid {name} on a resolved {axis} grid",
                    )
            produced = len(results)
            if produced == 0:
                self.status.showMessage(
                    f"Regrid created 0 datasets. Skipped: {_compact_item_summary(skipped)}"
                    if skipped else "Regrid created 0 datasets."
                )
                return
            message = f"Regrid created {produced} dataset(s) on {axis}."
            if downsampled:
                message += (
                    " Point-sampled without anti-alias filtering: "
                    + _compact_item_summary(downsampled)
                    + "."
                )
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        if self._start_background_callable(
            "Dataset regrid", compute, publish
        ):
            self.status.showMessage(
                f"Regridding {len(datasets)} dataset(s) onto {n:,} {axis} samples..."
            )

    def _decimate_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to decimate.",
        )
        if datasets is None:
            return

        dlg = DecimateDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            dlg.deleteLater()
            return
        params = dlg.get_params()
        dlg.deleteLater()
        axis = str(params["axis"])
        factor = int(params["factor"])
        mode = str(params["mode"])
        metadata_attested = False
        if mode == "coherent":
            attestation = self._confirm_coherent_metadata(
                datasets,
                "Coherent Decimation",
                independent=True,
            )
            if attestation is None:
                return
            metadata_attested = attestation

        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return decimate_axis(
                dataset,
                axis=axis,
                factor=factor,
                mode=mode,
                metadata_attested=metadata_attested,
            )

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, decimated in results:
                output_name = f"{name} [Decimate {axis} x{factor}]"
                history = (
                    f"Decimate {axis} by {factor}, {mode} boxcar prefilter: {name}"
                )
                output_id = self._add_dataset_row(
                    decimated, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "decimate_axis",
                        [source_ref],
                        kwargs={
                            "axis": axis,
                            "factor": factor,
                            "mode": mode,
                            **(
                                {"metadata_attested": True}
                                if metadata_attested
                                else {}
                            ),
                        },
                        comment=f"Prefilter and decimate {name}",
                    )
            message = f"Decimate created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Dataset decimation",
            datasets,
            operation,
            publish,
            start_message=(
                f"Prefiltering and decimating {len(datasets)} dataset(s) "
                f"along {axis} by {factor}..."
            ),
        )


    def _mirror_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to mirror.",
        )
        if datasets is None:
            return

        default_about = 0.0
        ref = self.active_dataset if self.active_dataset is not None else datasets[0][1]
        if isinstance(ref, RcsGrid) and len(ref.azimuths) > 0:
            try:
                az_vals = _angle_axis_degrees(ref, "azimuth")
            except (TypeError, ValueError) as exc:
                self.status.showMessage(f"Mirror blocked: {exc}")
                return
            finite = az_vals[np.isfinite(az_vals)]
            if finite.size > 0:
                default_about = float(np.mean(finite))

        about, ok = QInputDialog.getDouble(
            self,
            "Mirror Dataset",
            "Mirror about azimuth (degrees):",
            default_about,
            -1e9,
            1e9,
            6,
        )
        if not ok:
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return dataset.mirror_about_azimuth(about)

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, mirrored in results:
                history = f"Mirror about az={about:.6g} deg: {name}"
                output_name = f"{name} [Mirror {about:.6g}°]"
                output_id = self._add_dataset_row(
                    mirrored, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        source_ref,
                        "mirror_about_azimuth",
                        args=(float(about),),
                        comment=f"Mirror {name} about azimuth {about:g} degrees",
                    )
            message = f"Mirror created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Dataset mirror",
            datasets,
            operation,
            publish,
            start_message=f"Mirroring {len(datasets)} dataset(s)...",
        )

    def _wrap_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to wrap.",
        )
        if datasets is None:
            return

        dlg = WrapDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            dlg.deleteLater()
            return
        params = dlg.get_params()
        dlg.deleteLater()
        mode = str(params["mode"])
        wrap_azimuth = bool(params["azimuth"])
        wrap_phase_values = bool(params["phase"])
        if not (wrap_azimuth or wrap_phase_values):
            self.status.showMessage("Wrap: select azimuth, phase, or both.")
            return
        suffix = "0–360°" if mode == "0_360" else "-180–180°"
        target_label = (
            "azimuth and phase" if wrap_azimuth and wrap_phase_values
            else "azimuth" if wrap_azimuth else "phase"
        )

        estimated_peak = sum(
            _derived_grid_peak_bytes(dataset, dataset.rcs_power.shape)
            for _name, dataset in datasets
        )
        memory_limit = _derived_grid_memory_limit()
        if estimated_peak > memory_limit:
            self.status.showMessage(
                "Wrap blocked before allocation: estimated working set "
                f"{_format_bytes(estimated_peak)} exceeds the current safety limit "
                f"{_format_bytes(memory_limit)}. Process fewer datasets at once."
            )
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def compute():
            results = []
            skipped = []
            for dataset_index, (name, dataset) in enumerate(datasets):
                try:
                    wrapped = dataset
                    if wrap_azimuth:
                        wrapped = wrapped.wrap_azimuth(mode)
                    if wrap_phase_values:
                        wrapped = wrapped.wrap_phase(mode)
                except (TypeError, ValueError) as exc:
                    skipped.append(f"{name} ({exc})")
                    continue
                dropped = len(dataset.azimuths) - len(wrapped.azimuths)
                results.append((dataset_index, name, wrapped, dropped))
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            dropped_total = 0
            recorder = getattr(self, "python_recorder", None)
            for dataset_index, name, wrapped, dropped in results:
                dropped_total += int(dropped)
                drop_note = (
                    f" (merged {dropped} seam-alias azimuth coordinate(s))"
                    if dropped else ""
                )
                history = f"Wrap {target_label} to {suffix}{drop_note}: {name}"
                output_name = f"{name} [Wrap {target_label} {suffix}]"
                output_id = self._add_dataset_row(
                    wrapped, output_name, history, file_name=""
                )
                source_ref = source_references[dataset_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_expression(
                        self._python_output_reference(output_id, output_name),
                        [source_ref],
                        lambda variables, az=wrap_azimuth, phase=wrap_phase_values, selected_mode=mode: (
                            variables[0]
                            + (f".wrap_azimuth({selected_mode!r})" if az else "")
                            + (f".wrap_phase({selected_mode!r})" if phase else "")
                        ),
                        comment=f"Wrap {name} {target_label} to {suffix}",
                    )
            produced = len(results)
            if produced == 0:
                self.status.showMessage(
                    "Wrap created 0 datasets."
                    + (f" Skipped: {_compact_item_summary(skipped)}" if skipped else "")
                )
                return
            message = f"Wrap created {produced} dataset(s)."
            if dropped_total:
                message += (
                    f" Merged {dropped_total} equivalent duplicate azimuth sample(s)."
                )
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        if self._start_background_callable("Dataset wrap", compute, publish):
            self.status.showMessage(
                f"Wrapping {target_label} for {len(datasets)} dataset(s)..."
            )

    def _shift_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to shift.",
        )
        if datasets is None:
            return

        dlg = ShiftDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        params = dlg.get_params()
        az_on, az_delta = params["azimuth"]
        el_on, el_delta = params["elevation"]
        ph_on, ph_delta = params["phase"]
        if not (az_on or el_on or ph_on):
            self.status.showMessage("Shift: no axes selected.")
            return

        suffix_parts = []
        history_parts = []
        if az_on:
            suffix_parts.append(f"Az{az_delta:+.6g}°")
            history_parts.append(f"Az {az_delta:+.6g} deg")
        if el_on:
            suffix_parts.append(f"El{el_delta:+.6g}°")
            history_parts.append(f"El {el_delta:+.6g} deg")
        if ph_on:
            suffix_parts.append(f"Ph{ph_delta:+.6g}°")
            history_parts.append(f"Phase {ph_delta:+.6g} deg")
        suffix = " ".join(suffix_parts)
        history_axes = ", ".join(history_parts)
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return shift_dataset(
                dataset,
                azimuth_degrees=float(az_delta) if az_on else None,
                elevation_degrees=float(el_delta) if el_on else None,
                phase_degrees=float(ph_delta) if ph_on else None,
            )

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, shifted in results:
                history = f"Shift ({history_axes}): {name}"
                output_name = f"{name} [Shift {suffix}]"
                output_id = self._add_dataset_row(
                    shifted, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "shift_dataset",
                        [source_ref],
                        kwargs={
                            "azimuth_degrees": float(az_delta) if az_on else None,
                            "elevation_degrees": float(el_delta) if el_on else None,
                            "phase_degrees": float(ph_delta) if ph_on else None,
                        },
                        comment=f"Shift {name}: {history_axes}",
                    )
            message = f"Shift created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Dataset shift",
            datasets,
            operation,
            publish,
            start_message=f"Shifting {len(datasets)} dataset(s)...",
        )

    def _round_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to round.",
        )
        if datasets is None:
            return

        dlg = RoundDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        params = dlg.get_params()
        if not (params["azimuths"] or params["elevations"] or params["frequencies"]):
            self.status.showMessage("Round: no axes selected.")
            return
        decimals = params["decimals"]
        axes_label = ",".join(
            ax[:2] for ax, key in (("Az", "azimuths"), ("El", "elevations"), ("Fq", "frequencies"))
            if params[key]
        )
        enabled_methods = tuple(
            method
            for enabled, method in (
                (params["azimuths"], "round_azimuths"),
                (params["elevations"], "round_elevations"),
                (params["frequencies"], "round_frequencies"),
            )
            if enabled
        )
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            rounded = dataset
            for method in enabled_methods:
                rounded = getattr(rounded, method)(decimals)
            return rounded

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, rounded in results:
                history = f"Round {axes_label} to {decimals} dp: {name}"
                output_name = f"{name} [Round {decimals}dp]"
                output_id = self._add_dataset_row(
                    rounded, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_expression(
                        self._python_output_reference(output_id, output_name),
                        [source_ref],
                        lambda variables, methods=enabled_methods, decimals=decimals: (
                            variables[0]
                            + "".join(
                                f".{method}({int(decimals)})" for method in methods
                            )
                        ),
                        comment=(
                            f"Round {name} axes {axes_label} to {decimals} decimals"
                        ),
                    )
            message = f"Round created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Dataset rounding",
            datasets,
            operation,
            publish,
            start_message=f"Rounding {len(datasets)} dataset(s)...",
        )

    def _swap_elevation_azimuth_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to swap elevation and azimuth.",
        )
        if datasets is None:
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return dataset.swap_elevation_azimuth()

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, swapped in results:
                history = f"Swap El/Az: {name}"
                output_name = f"{name} [Swap El/Az]"
                output_id = self._add_dataset_row(
                    swapped, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        source_ref,
                        "swap_elevation_azimuth",
                        comment=f"Swap elevation and azimuth for {name}",
                    )
            message = f"Swap El/Az created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Elevation/azimuth swap",
            datasets,
            operation,
            publish,
            start_message=f"Swapping axes for {len(datasets)} dataset(s)...",
        )

    def _convert_sentri_elevation_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message=(
                "Select one or more native SENTRi datasets to convert to "
                "GRIM elevation."
            ),
        )
        if datasets is None:
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return dataset.convert_sentri_elevation_to_grim()

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, converted in results:
                history = (
                    "SENTRi elevation to GRIM: elevation=90-theta; "
                    f"no interpolation or phase change: {name}"
                )
                output_name = f"{name} [SENTRi El→GRIM]"
                output_id = self._add_dataset_row(
                    converted, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        source_ref,
                        "convert_sentri_elevation_to_grim",
                        comment=(
                            f"Convert native SENTRi theta to GRIM signed "
                            f"elevation for {name}"
                        ),
                    )
            message = f"SENTRi El→GRIM created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "SENTRi elevation conversion",
            datasets,
            operation,
            publish,
            start_message=f"Converting {len(datasets)} SENTRi dataset(s)...",
        )

    def _elevation_to_azimuth_360_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert elevation pair into 360 azimuth.",
        )
        if datasets is None:
            return

        reference = (
            self.active_dataset
            if isinstance(getattr(self, "active_dataset", None), RcsGrid)
            else datasets[0][1]
        )
        selected_el_values = self._selected_values(self.list_elev)
        selected_pair_deg: tuple[float, float] | None = None
        if len(selected_el_values) == 2:
            try:
                pair_native = np.asarray(
                    sorted(float(v) for v in selected_el_values), dtype=float
                )
                if _canonical_angle_unit(
                    (reference.units or {}).get("elevation", "deg")
                ) == "rad":
                    pair_native = np.rad2deg(pair_native)
                selected_pair_deg = (
                    float(pair_native[0]),
                    float(pair_native[1]),
                )
            except (TypeError, ValueError):
                selected_pair_deg = None
        pair_text = (
            "the equal-and-opposite minimum/maximum elevation in each dataset"
            if selected_pair_deg is None
            else f"{selected_pair_deg[0]:.6g}/{selected_pair_deg[1]:.6g} deg"
        )
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        answer = QMessageBox.question(
            self,
            "Confirm Elevation-Pair Relabel",
            "El→Az360 is not a general spherical-coordinate conversion. It "
            "combines equal-and-opposite elevation cuts by shifting the second "
            "half 180° in azimuth, without interpolation or polarization-basis "
            f"rotation. This run will use {pair_text}.\n\n"
            "Confirm that the acquisition geometry and polarization convention "
            "make that relabel physically valid?",
            buttons.Yes | buttons.No,
            buttons.No,
        )
        if answer != buttons.Yes:
            self.status.showMessage(
                "El→Az360 cancelled: the acquisition-specific relabel was not confirmed."
            )
            return

        native_pairs: list[tuple[float, float] | None] = []
        try:
            for _name, dataset in datasets:
                _assert_same_angular_frame(reference, dataset)
                if selected_pair_deg is None:
                    native_pairs.append(None)
                    continue
                native = _degrees_to_angle_axis(
                    dataset, "elevation", selected_pair_deg
                )
                native_pairs.append((float(native[0]), float(native[1])))
        except (TypeError, ValueError) as exc:
            self.status.showMessage(f"El→Az360 blocked: {exc}")
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(index, _name, dataset):
            selected_pair = native_pairs[index]
            if selected_pair is None:
                return dataset.combine_elevation_pair_to_azimuth_360(
                    azimuth_shift_deg=180.0,
                    assumptions_attested=True,
                )
            return dataset.combine_elevation_pair_to_azimuth_360(
                selected_pair[0],
                selected_pair[1],
                azimuth_shift_deg=180.0,
                assumptions_attested=True,
            )

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, result in results:
                history = f"El->Az360 (shift +180 deg, pair={pair_text}): {name}"
                output_name = f"{name} [El->Az360]"
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        source_ref,
                        "combine_elevation_pair_to_azimuth_360",
                        args=native_pairs[source_index] or (),
                        kwargs={
                            "azimuth_shift_deg": 180.0,
                            "assumptions_attested": True,
                        },
                        comment=(
                            f"Convert {name} elevation pair to 360-degree azimuth"
                        ),
                    )
            message = f"El->Az360 created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Elevation-to-azimuth conversion",
            datasets,
            operation,
            publish,
            start_message=f"Converting {len(datasets)} dataset(s) to 360-degree azimuth...",
        )

    def _range_cal_selected(self) -> None:
        targets = self._selected_datasets_ordered(
            empty_message="Select one or more measured DUT datasets to range-calibrate.",
        )
        if targets is None:
            return

        loaded_entries: list[tuple[str, RcsGrid]] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if isinstance(dataset, RcsGrid):
                loaded_entries.append((item.text(), dataset))
        target_ids = {id(dataset) for _name, dataset in targets}
        reference_entries = [
            entry for entry in loaded_entries if id(entry[1]) not in target_ids
        ]
        if len(reference_entries) < 2:
            self.status.showMessage(
                "Range Cal needs two unselected reference datasets in addition "
                "to the selected DUT row(s): measured calibration and complex exact."
            )
            return

        dialog = RangeCalibrationDialog(reference_entries, parent=self)
        if dialog.exec() != QDialog.Accepted:
            dialog.deleteLater()
            return
        params = dialog.get_params()
        dialog.deleteLater()
        measured_name, measured = params["measured"]
        exact_name, exact = params["exact"]
        if measured is exact:
            self.status.showMessage(
                "Range Cal: measured calibration and exact reference must be "
                "different datasets."
            )
            return
        if id(measured) in target_ids or id(exact) in target_ids:
            self.status.showMessage(
                "Range Cal: select only DUT rows as targets; choose measured and "
                "exact references in the dialog without selecting their table rows."
            )
            return
        target_refs = self._python_input_references(targets) or []
        measured_ref = self._python_reference_for_dataset(measured)
        exact_ref = self._python_reference_for_dataset(exact)
        self._pending_range_record = None
        if measured_ref is not None and exact_ref is not None:
            self._pending_range_record = {
                "targets": {
                    id(dataset): reference
                    for (_name, dataset), reference in zip(targets, target_refs)
                },
                "measured": measured_ref,
                "exact": exact_ref,
                "range_offset_m": float(params["range_offset_m"]),
                "allow_singleton_angular_broadcast": bool(
                    params.get("allow_singleton_angular_broadcast", False)
                ),
                "maximum_correction_gain_db": params.get(
                    "maximum_correction_gain_db", 60.0
                ),
                "measured_label": measured_name,
                "exact_label": exact_name,
            }

        worker = _RangeCalibrationWorker(
            targets,
            (measured_name, measured),
            (exact_name, exact),
            params,
        )
        worker.progress.connect(self._on_range_cal_worker_progress)
        worker.finished.connect(self._on_range_cal_worker_finished)
        self.status.showMessage(f"Range Cal... 0/{len(targets)}")
        if not self._try_start_background_job("Range Cal", worker):
            self._pending_range_record = None

    def _offset_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to offset.",
        )
        if datasets is None:
            return

        value, ok = QInputDialog.getDouble(
            self, "Offset", "Offset (dB) — shifts all displayed values by this amount:",
            0.0, -300.0, 300.0, 4,
        )
        if not ok:
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return offset_db(dataset, float(value))

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, result in results:
                history = f"Offset ({value:+.6g}): {name}"
                output_name = f"{name} [Offset {value:+.6g}]"
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "offset_db",
                        [source_ref],
                        args=(float(value),),
                        comment=f"Offset {name} by {value:+g} dB",
                    )
            message = f"Offset created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Dataset offset",
            datasets,
            operation,
            publish,
            start_message=f"Applying offset to {len(datasets)} dataset(s)...",
        )

    def _convert_to_dbke_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert to dBke.",
        )
        if datasets is None:
            return

        dlg = ExtrusionLengthDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        length_m = dlg.length_m()
        length_label = dlg.display_text()
        if length_m <= 0.0 or not np.isfinite(length_m):
            self.status.showMessage("Convert to dBke: length must be positive.")
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return convert_extrusion(
                dataset, to="dbke", length_m=float(length_m)
            )

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, result in results:
                history = (
                    f"Convert to dBke (extruded L={length_label}, "
                    f"{length_m:.6g} m): {name}"
                )
                output_name = f"{name} [→ dBke L={length_label}]"
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "convert_extrusion",
                        [source_ref],
                        kwargs={"to": "dbke", "length_m": float(length_m)},
                        comment=f"Convert {name} from dBsm to dBke",
                    )
            conversion_offset_db = 10.0 * np.log10(
                np.pi / (length_m * length_m)
            )
            message = (
                f"Convert to dBke created {len(results)} dataset(s) "
                f"(L={length_label} → constant offset "
                f"{conversion_offset_db:+.2f} dB)."
            )
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "dBsm-to-dBke conversion",
            datasets,
            operation,
            publish,
            start_message=f"Converting {len(datasets)} dataset(s) to dBke...",
        )

    def _convert_to_dbsm_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert to dBsm.",
        )
        if datasets is None:
            return

        dlg = ExtrusionLengthDialog(
            parent=self,
            title="Convert dBke → dBsm",
            formula=(
                "σ_3D = σ_2D · (2 L² / λ) → dBsm = dBke + 20·log₁₀(L) − "
                "10·log₁₀(π) (frequency-independent offset)."
            ),
        )
        if dlg.exec() != QDialog.Accepted:
            return
        length_m = dlg.length_m()
        length_label = dlg.display_text()
        if length_m <= 0.0 or not np.isfinite(length_m):
            self.status.showMessage("Convert to dBsm: length must be positive.")
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return convert_extrusion(
                dataset, to="dbsm", length_m=float(length_m)
            )

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, result in results:
                history = (
                    f"Convert to dBsm (extruded L={length_label}, "
                    f"{length_m:.6g} m): {name}"
                )
                output_name = f"{name} [→ dBsm L={length_label}]"
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "convert_extrusion",
                        [source_ref],
                        kwargs={"to": "dbsm", "length_m": float(length_m)},
                        comment=f"Convert {name} from dBke to dBsm",
                    )
            conversion_offset_db = (
                20.0 * np.log10(length_m) - 10.0 * np.log10(np.pi)
            )
            message = (
                f"Convert to dBsm created {len(results)} dataset(s) "
                f"(L={length_label} → constant offset "
                f"{conversion_offset_db:+.2f} dB)."
            )
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "dBke-to-dBsm conversion",
            datasets,
            operation,
            publish,
            start_message=f"Converting {len(datasets)} dataset(s) to dBsm...",
        )

    def _set_coordinates_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to set their coordinates.",
        )
        if datasets is None:
            return
        reference = next(
            (grid for _name, grid in datasets if grid is self.active_dataset),
            datasets[0][1],
        )
        dialog = CoordinateSystemDialog(reference, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        params = dialog.get_params()
        label = "Az/El" if params["coordinate_system"] == "conic" else "Aspect/Pitch"
        references = [self._python_reference_for_dataset(grid) for _name, grid in datasets]

        def operation(_index, _name, dataset):
            return dataset.set_angular_coordinate_system(**params)

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            new_rows = []
            for source_index, name, result in results:
                output_name = f"{name} [{label}]"
                new_rows.append(self.table.rowCount())
                output_id = self._add_dataset_row(result, output_name, "", file_name="")
                if recorder is not None and references[source_index] is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        references[source_index], "set_angular_coordinate_system",
                        kwargs=params, comment=f"Declare {label} coordinates for {name}",
                    )
            if new_rows:
                selection = self.table.selectionModel()
                self.table.clearSelection()
                for row in new_rows:
                    selection.select(
                        self.table.model().index(row, 0),
                        QItemSelectionModel.Select | QItemSelectionModel.Rows,
                    )
                selection.setCurrentIndex(
                    self.table.model().index(new_rows[0], 0), QItemSelectionModel.NoUpdate
                )
            message = f"Set Coordinates created and selected {len(results)} {label} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Set Coordinates", datasets, operation, publish,
            start_message=f"Setting coordinates for {len(datasets)} dataset(s)...",
        )

    def _convert_axis_units_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert axis units.",
        )
        if datasets is None:
            return
        reference = next(
            (grid for _name, grid in datasets if grid is self.active_dataset),
            datasets[0][1],
        )
        try:
            dialog = AxisUnitsDialog(reference, parent=self)
        except (TypeError, ValueError) as exc:
            self.status.showMessage(f"Axis Units blocked: {exc}")
            return
        if dialog.exec() != QDialog.Accepted:
            dialog.deleteLater()
            return
        params = dialog.get_params()
        dialog.deleteLater()
        references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            return dataset.convert_axis_units(**params)

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, result in results:
                unit_label = (
                    f"{params['azimuth']}/{params['elevation']}/"
                    f"{params['frequency']}"
                )
                output_name = f"{name} [Units {unit_label}]"
                output_id = self._add_dataset_row(
                    result,
                    output_name,
                    f"Axis Units ({unit_label}): {name}",
                    file_name="",
                )
                source_ref = references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        source_ref,
                        "convert_axis_units",
                        kwargs=params,
                        comment=f"Convert storage-axis units for {name}",
                    )
            message = f"Axis Units created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Axis-unit conversion",
            datasets,
            operation,
            publish,
            start_message=f"Converting axis units for {len(datasets)} dataset(s)...",
        )

    def _convert_conic_gc_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert.",
        )
        if datasets is None:
            return

        first_grid = datasets[0][1]
        first_coordinate_system = first_grid.angular_coordinate_system()
        dlg = ConicGCDialog(
            source_coordinate_system=first_coordinate_system,
            source_gc_convention=(
                first_grid.great_circle_coordinate_convention()
                if first_coordinate_system == "great_circle" else None
            ),
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        params = dlg.get_params()
        direction = params["direction"]
        mode = params["mode"]
        attest_legacy = bool(params.get("attest_legacy_ptm_convention", False))
        arrow = "Conic→GC" if direction == "conic_to_gc" else "GC→Conic"
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def operation(_index, _name, dataset):
            if mode != "relabel":
                raise ValueError(
                    "general conic/great-circle conversion is unavailable "
                    "until full polarization-basis rotation is implemented"
                )
            return self._conic_gc_relabel(
                dataset,
                direction,
                attest_legacy_ptm_convention=attest_legacy,
            )

        def publish(results, skipped) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, payload in results:
                result, suffix, hist_extra = payload
                history = f"{arrow} {mode}: {name}{hist_extra}"
                output_name = f"{name} [{arrow} {suffix}]"
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_method(
                        self._python_output_reference(output_id, output_name),
                        source_ref,
                        "convert_equatorial_conic_gc",
                        args=(direction,),
                        kwargs={"attest_legacy_ptm_convention": attest_legacy},
                        comment=f"{arrow} exact zero-plane relabel for {name}",
                    )
            message = f"{arrow} ({mode}) created {len(results)} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        self._start_dataset_map_job(
            "Conic/Great-Circle conversion",
            datasets,
            operation,
            publish,
            start_message=f"Converting {len(datasets)} dataset(s): {arrow}...",
        )

    def _conic_gc_relabel(
        self,
        dataset: "RcsGrid",
        direction: str,
        *,
        attest_legacy_ptm_convention=False,
    ):
        """Qt-facing compatibility wrapper around the tested grid operation."""

        result = dataset.convert_equatorial_conic_gc(
            direction,
            attest_legacy_ptm_convention=attest_legacy_ptm_convention,
        )
        note = (
            "; exact zero-plane relabel; no interpolation; "
            "GRIM_GC_V1 convention"
        )
        return result, "equator", note

    def _convert_wedge_to_conic_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert.",
        )
        if datasets is None:
            return

        dlg = WedgeConicDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        params = dlg.get_params()
        mode = params["mode"]
        assume_cross_zero = params["assume_missing_cross_pol_zero"]
        wedge_workspace = 0
        wedge_plans = []
        for _name, dataset in datasets:
            shape = tuple(int(value) for value in dataset.rcs_power.shape)
            wedge_plans.append((dataset, shape))
            query_frequency_cells = int(
                shape[0] * shape[1] * shape[2]
            )
            source_cells = int(math.prod(shape))
            # Full-complex source plus interpolated/source Jones matrices,
            # basis-change matrices, query coordinates, and einsum output.
            wedge_workspace = max(
                wedge_workspace,
                source_cells * np.dtype(np.complex128).itemsize
                + query_frequency_cells * 192,
            )
        if not self._preflight_derived_outputs(
            "Wedge→Conic",
            wedge_plans,
            extra_bytes=wedge_workspace,
        ):
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def compute(progress):
            results = []
            skipped = []
            total = len(datasets)
            for index, (name, dataset) in enumerate(datasets, start=1):
                try:
                    az_in = np.asarray(dataset.azimuths, dtype=float)
                    el_in = np.asarray(dataset.elevations, dtype=float)
                    if az_in.size < 2 or el_in.size < 1:
                        raise ValueError("need at least 2 azimuths and 1 elevation")
                    result, suffix, hist_extra = self._wedge_to_conic_regrid(
                        dataset,
                        assume_missing_cross_pol_zero=assume_cross_zero,
                    )
                except Exception as exc:
                    skipped.append(f"{name} ({exc})")
                else:
                    results.append((index - 1, name, result, suffix, hist_extra))
                progress(index, total, name)
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, result, suffix, hist_extra in results:
                history = f"Wedge→Conic {mode}: {name}{hist_extra}"
                output_name = f"{name} [Wedge→Conic {suffix}]"
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "wedge_to_conic",
                        [source_ref],
                        kwargs={
                            "mode": mode,
                            "assume_missing_cross_pol_zero": assume_cross_zero,
                        },
                        comment=f"Wedge-to-Conic {mode} for {name}",
                    )

            produced = len(results)
            message = f"Wedge→Conic ({mode}) created {produced} dataset(s)."
            if skipped:
                message += f" Skipped: {_compact_item_summary(skipped)}"
            self.status.showMessage(message)

        if self._start_background_callable(
            "Wedge-to-Conic conversion",
            compute,
            publish,
            reports_progress=True,
        ):
            self.status.showMessage(
                f"Converting {len(datasets)} wedge dataset(s) to conic coordinates..."
            )

    def _wedge_to_conic_relabel(self, dataset: "RcsGrid"):
        raise ValueError(
            "Wedge samples have paired longitude/latitude coordinates and "
            "cannot be represented by a one-dimensional RcsGrid relabel. Use "
            "the physical re-grid with at least two measured wedge tilts."
        )

    def _wedge_to_conic_regrid(
        self,
        dataset: "RcsGrid",
        *,
        assume_missing_cross_pol_zero=False,
    ):
        """Run the tested physical direction/Jones conversion."""

        result = dataset.convert_wedge_to_conic(
            attest_wedge_axes=False,
            assume_missing_cross_pol_zero=assume_missing_cross_pol_zero,
        )
        return result, "normal conic", "; inverse-mapped complex Jones re-grid"

    def _medianize_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to medianize.",
        )
        if datasets is None:
            return

        dlg = MedianizeDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        params = dlg.get_params()
        window_deg = params["window_deg"]
        slide_deg = params["slide_deg"]
        if window_deg <= 0.0 or slide_deg <= 0.0:
            self.status.showMessage("Medianize: window and slide must be positive.")
            return

        preflight_errors: list[str] = []
        peak_estimate = 0
        window_workspace = 0
        for name, dataset in datasets:
            try:
                az_deg = _angle_axis_degrees(dataset, "azimuth")
                if az_deg.size < 2:
                    raise ValueError("need at least two azimuth samples")
                typical_step = float(np.median(np.diff(az_deg)))
                span = float(az_deg[-1] - az_deg[0])
                periodic = span >= 360.0 - max(1.5 * typical_step, 1.0e-6)
                if periodic:
                    count = int(np.ceil(360.0 / slide_deg - 1.0e-12))
                elif span < window_deg:
                    count = 1
                else:
                    count = int(np.floor((span - window_deg) / slide_deg + 1e-9)) + 1
                if count > _MAX_EXPLICIT_AXIS_POINTS:
                    raise ValueError(
                        f"would create {count:,} azimuths; safety limit is "
                        f"{_MAX_EXPLICIT_AXIS_POINTS:,}"
                    )
                shape = (
                    count,
                    len(dataset.elevations),
                    len(dataset.frequencies),
                    len(dataset.polarizations),
                )
                peak_estimate += _derived_grid_peak_bytes(dataset, shape)
                source_cells = int(math.prod(dataset.rcs_power.shape))
                source_itemsize = max(
                    np.dtype(dataset.rcs_power.dtype).itemsize,
                    np.dtype(dataset.rcs_phase.dtype).itemsize,
                )
                # A widest-window advanced-index copy and nanmedian partition
                # scratch can coexist with all previously retained outputs.
                window_workspace = max(
                    window_workspace,
                    source_cells * source_itemsize * 3,
                )
            except (TypeError, ValueError) as exc:
                preflight_errors.append(f"{name} ({exc})")
        if preflight_errors:
            self.status.showMessage(
                "Medianize blocked before allocation: " + "; ".join(preflight_errors)
            )
            return
        memory_limit = _derived_grid_memory_limit()
        peak_estimate += window_workspace
        if peak_estimate > memory_limit:
            self.status.showMessage(
                "Medianize blocked before allocation: estimated working set "
                f"{_format_bytes(peak_estimate)} exceeds the current safety "
                f"limit {_format_bytes(memory_limit)}. Increase the slide size "
                "or process fewer datasets at once."
            )
            return

        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def compute():
            results = []
            skipped = []
            for dataset_index, (name, dataset) in enumerate(datasets):
                try:
                    result = medianize_azimuth(
                        dataset,
                        window_degrees=float(window_deg),
                        slide_degrees=float(slide_deg),
                    )
                except Exception as exc:
                    skipped.append(f"{name} ({exc})")
                    continue
                results.append((dataset_index, name, result))
            return results, skipped

        def publish(payload) -> None:
            results, skipped = payload
            for dataset_index, name, result in results:
                history = (
                    f"Medianize (window={window_deg:g}°, "
                    f"slide={slide_deg:g}°): {name}"
                )
                output_name = (
                    f"{name} [Median w={window_deg:g}° s={slide_deg:g}°]"
                )
                output_id = self._add_dataset_row(
                    result, output_name, history, file_name=""
                )
                source_ref = source_references[dataset_index]
                recorder = getattr(self, "python_recorder", None)
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "medianize_azimuth",
                        [source_ref],
                        kwargs={
                            "window_degrees": float(window_deg),
                            "slide_degrees": float(slide_deg),
                        },
                        comment=f"Medianize {name} over azimuth",
                    )
            produced = len(results)
            if produced == 0:
                self.status.showMessage("Medianize created 0 datasets.")
                return
            message = (
                f"Medianize created {produced} dataset(s) "
                f"(window={window_deg:g}°, slide={slide_deg:g}°)."
            )
            if skipped:
                message += f" Skipped: {', '.join(skipped)}"
            self.status.showMessage(message)

        if self._start_background_callable(
            "Dataset medianization", compute, publish
        ):
            self.status.showMessage(
                f"Medianizing {len(datasets)} dataset(s) in the background..."
            )


    def _duplicate_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to duplicate.",
        )
        if datasets is None:
            return
        metadata_array_bytes = 0
        for _name, dataset in datasets:
            for value in (dataset.extra or {}).values():
                if isinstance(value, np.ndarray):
                    metadata_array_bytes += int(value.nbytes)
        if not self._preflight_derived_outputs(
            "Duplicate",
            [
                (dataset, tuple(int(value) for value in dataset.rcs_power.shape))
                for _name, dataset in datasets
            ],
            # deepcopy owns another metadata-array set while the source stays
            # resident. The row publication itself only shallow-copies extras.
            extra_bytes=2 * metadata_array_bytes,
        ):
            return
        source_references = [
            self._python_reference_for_dataset(dataset)
            for _name, dataset in datasets
        ]

        def compute(progress):
            copies = []
            total = len(datasets)
            for index, (name, dataset) in enumerate(datasets, start=1):
                duplicate = RcsGrid(
                    dataset.azimuths.copy(),
                    dataset.elevations.copy(),
                    dataset.frequencies.copy(),
                    dataset.polarizations.copy(),
                    rcs_power=dataset.rcs_power.copy(),
                    rcs_phase=dataset.rcs_phase.copy(),
                    rcs_domain=dataset.rcs_domain,
                    source_path=dataset.source_path,
                    history=dataset.history,
                    units=copy.deepcopy(dataset.units or {}),
                    extra=copy.deepcopy(dataset.extra or {}),
                )
                copies.append((index - 1, name, duplicate))
                progress(index, total, name)
            return copies

        def publish(copies) -> None:
            recorder = getattr(self, "python_recorder", None)
            for source_index, name, duplicate in copies:
                output_name = f"{name} [Copy]"
                output_id = self._add_dataset_row(
                    duplicate,
                    output_name,
                    f"Duplicate of: {name}",
                    file_name="",
                )
                source_ref = source_references[source_index]
                if recorder is not None and source_ref is not None:
                    recorder.record_function(
                        self._python_output_reference(output_id, output_name),
                        "duplicate_dataset",
                        [source_ref],
                        comment=f"Duplicate {name}",
                    )
            self.status.showMessage(f"Duplicated {len(copies)} dataset(s).")

        if self._start_background_callable(
            "Dataset duplication", compute, publish, reports_progress=True
        ):
            self.status.showMessage(
                f"Duplicating {len(datasets)} dataset(s) in the background..."
            )

    def _iter_pio_slices(self, dataset: RcsGrid, base_name: str):
        """Yield filenames and indices for single-cut complex file formats.

        Pioneer and PTM files each hold one 2-D (azimuth, frequency) complex
        slice, so a larger grid is split into one file per (elevation,
        polarization) combination.  The historical method name is retained for
        compatibility with existing GUI automation.
        """
        safe = _sanitize_filename(base_name)
        n_el = len(dataset.elevations)
        n_pol = len(dataset.polarizations)
        for ei in range(n_el):
            for pi in range(n_pol):
                parts = [safe]
                if n_pol > 1:
                    pol_label = str(dataset.polarizations[pi]).strip() or f"pol{pi}"
                    parts.append(f"p{pi:03d}_{_sanitize_filename(pol_label)}")
                if n_el > 1:
                    # The index guarantees uniqueness even when formatted
                    # floating values are identical or the axis uses radians.
                    parts.append(f"el{ei:04d}_{float(dataset.elevations[ei]):.12g}")
                yield "_".join(parts), ei, pi

    @staticmethod
    def _write_pio_batch(
        directory: str,
        plans,
        *,
        precision: str = "single",
        progress_cb=None,
    ) -> int:
        """Validate and stage a Pioneer fan-out before publishing any target."""

        prepared = []
        seen_targets: dict[str, str] = {}
        for name, dataset, stem, el_idx, pol_idx in plans:
            target = os.path.abspath(os.path.join(directory, f"{stem}.pio"))
            target_key = _target_path_key(target)
            prior = seen_targets.get(target_key)
            if prior is not None:
                raise ValueError(
                    "Pioneer export would create the same file more than once: "
                    f"{os.path.basename(target)} (from {prior!r} and {name!r})"
                )
            if os.path.lexists(target):
                raise FileExistsError(
                    f"Pioneer target already exists: {target}. Choose an empty "
                    "folder or rename/remove the existing file."
                )
            seen_targets[target_key] = str(name)
            prepared.append((dataset, stem, int(el_idx), int(pol_idx), target))

        work_total = max(1, len(prepared) * 2)
        with tempfile.TemporaryDirectory(prefix=".grim_pio_", dir=directory) as stage:
            staged = []
            for index, (dataset, stem, el_idx, pol_idx, target) in enumerate(
                prepared, start=1
            ):
                stage_path = os.path.join(stage, f"{stem}.pio")
                saved = dataset.save_pio(
                    stage_path,
                    el_idx=el_idx,
                    pol_idx=pol_idx,
                    precision=precision,
                )
                staged.append((saved, target))
                if progress_cb is not None:
                    progress_cb(index, work_total, f"Staged {os.path.basename(target)}")
            published: list[str] = []
            try:
                for index, (stage_path, target) in enumerate(staged, start=1):
                    if os.path.lexists(target):
                        raise FileExistsError(
                            f"Pioneer target appeared during export: {target}"
                        )
                    os.replace(stage_path, target)
                    published.append(target)
                    if progress_cb is not None:
                        progress_cb(
                            len(prepared) + index,
                            work_total,
                            f"Published {os.path.basename(target)}",
                        )
            except BaseException as original_error:
                cleanup_errors = []
                for target in reversed(published):
                    try:
                        if os.path.lexists(target):
                            os.unlink(target)
                    except OSError as exc:
                        cleanup_errors.append(f"{target}: {exc}")
                if cleanup_errors:
                    raise RuntimeError(
                        "Pioneer publication failed and rollback could not remove "
                        "every newly published file: " + "; ".join(cleanup_errors)
                    ) from original_error
                raise
        return len(prepared)

    def _export_pio_selected(self) -> None:
        try:
            self._export_pio_selected_impl()
        except (OSError, TypeError, ValueError) as exc:
            self.status.showMessage(f"Pioneer export failed: {exc}")

    def _export_pio_selected_impl(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to export.",
        )
        if datasets is None:
            return

        prefers_double = any(
            np.dtype(dataset.rcs_power.dtype).itemsize > 4
            or dataset._complete_authoritative_raw_arrays() is not None
            for _name, dataset in datasets
        )
        precision_choices = [
            "Double precision (64-bit real/imag)",
            "Single precision (32-bit real/imag; smaller legacy files)",
        ]
        precision_label, accepted = QInputDialog.getItem(
            self,
            "Pioneer Export Precision",
            "On-disk complex-sample precision:",
            precision_choices,
            0 if prefers_double else 1,
            False,
        )
        if not accepted:
            return
        precision = (
            "double" if str(precision_label).startswith("Double") else "single"
        )

        if len(datasets) == 1:
            name, dataset = datasets[0]
            slices = list(self._iter_pio_slices(dataset, name))
            if len(slices) == 1:
                stem, el_idx, pol_idx = slices[0]
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    f"Export {name} as Pioneer .pio",
                    f"{stem}.pio",
                    "Pioneer Files (*.pio);;All Files (*)",
                )
                if not path:
                    return
                def compute_single(progress):
                    progress(0, 1, f"Writing {os.path.basename(path)}")
                    saved = dataset.save_pio(
                        path,
                        el_idx=el_idx,
                        pol_idx=pol_idx,
                        precision=precision,
                    )
                    progress(1, 1, f"Published {os.path.basename(saved)}")
                    return saved

                def publish_single(saved) -> None:
                    self.status.showMessage(
                        f"Exported {os.path.basename(saved)}."
                    )

                if self._start_background_callable(
                    "Pioneer export",
                    compute_single,
                    publish_single,
                    reports_progress=True,
                ):
                    self.status.showMessage("Exporting 1 Pioneer file...")
                return
            directory = QFileDialog.getExistingDirectory(
                self,
                f"Export {name} ({len(slices)} slices) as .pio",
            )
            if not directory:
                return
            plans = [(name, dataset, *item) for item in slices]
        else:
            directory = QFileDialog.getExistingDirectory(
                self, "Export Selected Datasets as .pio"
            )
            if not directory:
                return
            plans = [
                (name, dataset, stem, el_idx, pol_idx)
                for dataset_index, (name, dataset) in enumerate(datasets, start=1)
                for stem, el_idx, pol_idx in self._iter_pio_slices(
                    dataset, f"d{dataset_index:03d}_{name}"
                )
            ]

        def compute_batch(progress):
            return self._write_pio_batch(
                directory,
                plans,
                precision=precision,
                progress_cb=progress,
            )

        def publish_batch(produced) -> None:
            self.status.showMessage(
                f"Exported {produced} .pio file(s) to {directory}."
            )

        if self._start_background_callable(
            "Pioneer export",
            compute_batch,
            publish_batch,
            reports_progress=True,
        ):
            self.status.showMessage(
                f"Exporting {len(plans)} Pioneer file(s) in the background..."
            )

    @staticmethod
    def _write_ptm_batch(directory: str, plans, *, progress_cb=None) -> int:
        """Validate and stage a PTM fan-out before publishing any target."""

        prepared = []
        seen_targets: dict[str, str] = {}
        for _name, dataset, stem, el_idx, pol_idx in plans:
            target = os.path.abspath(os.path.join(directory, f"{stem}.ptm"))
            target_key = os.path.normcase(target).casefold()
            prior = seen_targets.get(target_key)
            if prior is not None:
                raise ValueError(
                    "PTM export would create the same file more than once: "
                    f"{os.path.basename(target)} (from {prior!r} and {_name!r})"
                )
            if os.path.lexists(target):
                raise FileExistsError(
                    f"PTM target already exists: {target}. Choose an empty "
                    "folder or rename/remove the existing file."
                )
            seen_targets[target_key] = str(_name)
            prepared.append((dataset, stem, int(el_idx), int(pol_idx), target))

        # Validate/write every slice into a sibling staging folder first. A
        # publication failure removes every new target already moved, so this
        # empty-folder fan-out is all-or-nothing.
        work_total = max(1, len(prepared) * 2)
        with tempfile.TemporaryDirectory(prefix=".grim_ptm_", dir=directory) as stage:
            staged = []
            for index, (dataset, stem, el_idx, pol_idx, target) in enumerate(
                prepared, start=1
            ):
                stage_path = os.path.join(stage, f"{stem}.ptm")
                saved = dataset.save_ptm(
                    stage_path, el_idx=el_idx, pol_idx=pol_idx
                )
                staged.append((saved, target))
                if progress_cb is not None:
                    progress_cb(index, work_total, f"Staged {os.path.basename(target)}")
            published: list[str] = []
            try:
                for index, (stage_path, target) in enumerate(staged, start=1):
                    if os.path.lexists(target):
                        raise FileExistsError(
                            f"PTM target appeared during export: {target}"
                        )
                    os.replace(stage_path, target)
                    published.append(target)
                    if progress_cb is not None:
                        progress_cb(
                            len(prepared) + index,
                            work_total,
                            f"Published {os.path.basename(target)}",
                        )
            except BaseException as original_error:
                cleanup_errors = []
                for target in reversed(published):
                    try:
                        if os.path.lexists(target):
                            os.unlink(target)
                    except OSError as exc:
                        cleanup_errors.append(f"{target}: {exc}")
                if cleanup_errors:
                    raise RuntimeError(
                        "PTM publication failed and rollback could not remove "
                        "every newly published file: " + "; ".join(cleanup_errors)
                    ) from original_error
                raise
        return len(prepared)

    def _export_ptm_selected(self) -> None:
        """Export selected grids as one legacy PTM per elevation/polarization."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to export.",
        )
        if datasets is None:
            return

        try:
            if len(datasets) == 1:
                name, dataset = datasets[0]
                slices = list(self._iter_pio_slices(dataset, name))
                if len(slices) == 1:
                    stem, el_idx, pol_idx = slices[0]
                    path, _ = QFileDialog.getSaveFileName(
                        self,
                        f"Export {name} as PTM",
                        f"{stem}.ptm",
                        "PTM Files (*.ptm);;All Files (*)",
                    )
                    if not path:
                        return

                    def compute_single(progress):
                        progress(0, 1, f"Writing {os.path.basename(path)}")
                        saved = dataset.save_ptm(
                            path, el_idx=el_idx, pol_idx=pol_idx
                        )
                        progress(1, 1, f"Published {os.path.basename(saved)}")
                        return saved

                    def publish_single(saved) -> None:
                        self.status.showMessage(
                            f"Exported {os.path.basename(saved)}."
                        )

                    if self._start_background_callable(
                        "PTM export",
                        compute_single,
                        publish_single,
                        reports_progress=True,
                    ):
                        self.status.showMessage("Exporting 1 PTM file...")
                    return
                directory = QFileDialog.getExistingDirectory(
                    self,
                    f"Export {name} ({len(slices)} slices) as .ptm",
                )
                if not directory:
                    return
                plans = [(name, dataset, *item) for item in slices]
            else:
                directory = QFileDialog.getExistingDirectory(
                    self, "Export Selected Datasets as .ptm"
                )
                if not directory:
                    return
                plans = [
                    (name, dataset, stem, el_idx, pol_idx)
                    for dataset_index, (name, dataset) in enumerate(datasets, start=1)
                    for stem, el_idx, pol_idx in self._iter_pio_slices(
                        dataset, f"d{dataset_index:03d}_{name}"
                    )
                ]

            def compute_batch(progress):
                return self._write_ptm_batch(
                    directory, plans, progress_cb=progress
                )

            def publish_batch(produced) -> None:
                self.status.showMessage(
                    f"Exported {produced} .ptm file(s) to {directory}."
                )

            if self._start_background_callable(
                "PTM export",
                compute_batch,
                publish_batch,
                reports_progress=True,
            ):
                self.status.showMessage(
                    f"Exporting {len(plans)} PTM file(s) in the background..."
                )
        except (OSError, TypeError, ValueError) as exc:
            self.status.showMessage(f"PTM export failed: {exc}")

    def _export_csv_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to export.",
        )
        if datasets is None:
            return

        dlg = ExportCsvDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        scale, include_phase = dlg.get_options()

        entries: list[tuple[RcsGrid, str]] = []
        if len(datasets) == 1:
            name, dataset = datasets[0]
            safe_name = _sanitize_filename(name)
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Export {name}",
                f"{safe_name}.csv",
                "CSV Files (*.csv);;All Files (*)",
            )
            if not path:
                return
            if not path.casefold().endswith(".csv"):
                path += ".csv"
            entries.append((dataset, os.path.abspath(path)))
        else:
            directory = QFileDialog.getExistingDirectory(
                self, "Export Selected Datasets as CSV"
            )
            if not directory:
                return
            for dataset_index, (name, dataset) in enumerate(datasets, start=1):
                filename = f"d{dataset_index:03d}_{_sanitize_filename(name)}.csv"
                entries.append((dataset, os.path.abspath(os.path.join(directory, filename))))

        targets = [path for _dataset, path in entries]
        duplicate_groups = _duplicate_target_groups(targets)
        if duplicate_groups:
            self.status.showMessage(
                "CSV export cancelled: multiple datasets resolve to the same filename."
            )
            return
        invalid_targets = [path for path in targets if os.path.isdir(path)]
        if invalid_targets:
            self.status.showMessage(
                "CSV export cancelled: an output target is an existing directory."
            )
            return

        existing = [path for path in targets if os.path.lexists(path)]
        if existing:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            shown = "\n".join(f"• {os.path.basename(path)}" for path in existing[:12])
            if len(existing) > 12:
                shown += f"\n• …and {len(existing) - 12} more"
            answer = QMessageBox.question(
                self,
                "Replace Existing CSV Files?",
                f"{len(existing)} existing file(s) will be replaced:\n\n{shown}\n\n"
                "Replace all listed files transactionally?",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                self.status.showMessage("CSV export cancelled; no files were changed.")
                return

        row_counts = [math.prod(dataset.rcs_power.shape) for dataset, _ in entries]
        total_rows = sum(row_counts)
        # V1 repeats explicit physical metadata per row for robust standalone
        # interchange. This conservative estimate is used only for time/disk UI.
        entry_estimated_bytes = []
        try:
            for (dataset, _target), count in zip(entries, row_counts):
                metadata_chars = sum(
                    len(str(dataset._declared_scalar_metadata(key) or ""))
                    for key in (
                        "phase_reference",
                        "time_convention",
                        "polarization_basis",
                    )
                )
                entry_estimated_bytes.append(
                    count
                    * (
                        300
                        + max(
                            (len(str(value)) for value in dataset.polarizations),
                            default=0,
                        )
                        + metadata_chars
                    )
                )
        except (TypeError, ValueError) as exc:
            self.status.showMessage(f"CSV export blocked: {exc}")
            return
        estimated_bytes = sum(entry_estimated_bytes)
        if total_rows > 5_000_000 or estimated_bytes > 1024**3:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            answer = QMessageBox.question(
                self,
                "Large CSV Export",
                f"This export contains {total_rows:,} rows and may use about "
                f"{_format_bytes(estimated_bytes)}. CSV is portable but much larger "
                "and slower than .grim. Continue in the background?",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                self.status.showMessage("Large CSV export cancelled.")
                return

        estimates_by_directory: dict[str, int] = {}
        for (_dataset, target), estimate in zip(entries, entry_estimated_bytes):
            directory = os.path.dirname(target) or os.curdir
            estimates_by_directory[directory] = (
                estimates_by_directory.get(directory, 0) + estimate
            )
        for directory, estimate in estimates_by_directory.items():
            try:
                free = int(shutil.disk_usage(directory).free)
            except OSError:
                continue
            if estimate > int(free * 0.9):
                self.status.showMessage(
                    "CSV export blocked before writing: estimated staged output "
                    f"{_format_bytes(estimate)} exceeds safe free space "
                    f"{_format_bytes(free)} in {directory}."
                )
                return

        worker = _CsvExportWorker(
            entries,
            scale=scale,
            include_phase=include_phase,
        )
        worker.progress.connect(self._on_csv_export_progress)
        worker.finished.connect(self._on_csv_export_finished)
        if not self._try_start_background_job("CSV export", worker):
            return
        self.status.showMessage(
            f"Exporting {len(entries)} dataset(s) to CSV in the background..."
        )


    def _reselect_indices(self, widget: QListWidget, indices: set[int]) -> None:
        if not indices:
            return
        widget.blockSignals(True)
        for row in range(widget.count()):
            item = widget.item(row)
            idx = item.data(Qt.UserRole + 1)
            if idx in indices:
                item.setSelected(True)
        widget.blockSignals(False)

    # ── RCS-specific processing ───────────────────────────────────────────────

    def _coherent_div_selected(self) -> None:
        """Divide numerator dataset by denominator (complex, element-wise)."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select exactly 2 datasets (numerator first, then denominator).",
        )
        if datasets is None:
            return
        if len(datasets) != 2:
            self.status.showMessage("Coherent ÷: select exactly 2 datasets.")
            return
        name_a, ds_a = datasets[0]
        name_b, ds_b = datasets[1]

        if not self._preflight_derived_outputs(
            "Coherent ÷",
            [(ds_a, ds_a.rcs_power.shape)],
            extra_bytes=_derived_grid_peak_bytes(ds_a, ds_a.rcs_power.shape),
        ):
            return

        attestation = self._confirm_coherent_metadata(datasets, "Coherent ÷")
        if attestation is None:
            return
        input_refs = self._python_input_references(datasets)
        out_name = f"{name_a} ÷ {name_b}"

        def compute():
            return coherent_divide(
                ds_a, ds_b, metadata_attested=bool(attestation)
            )

        def publish(result) -> None:
            output_id = self._add_dataset_row(
                result,
                out_name,
                f"Coherent ÷: {name_a} / {name_b}",
                file_name="",
            )
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and input_refs is not None:
                recorder.record_function(
                    self._python_output_reference(output_id, out_name),
                    "coherent_divide",
                    input_refs,
                    kwargs={"metadata_attested": True} if attestation else None,
                    comment=f"Coherently divide {name_a} by {name_b}",
                )
            self.status.showMessage(f"Coherent ÷ produced: {out_name}")

        if self._start_background_callable("Coherent ÷", compute, publish):
            self.status.showMessage("Coherent ÷ is running in the background...")
