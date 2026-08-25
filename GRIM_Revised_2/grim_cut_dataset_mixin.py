from __future__ import annotations

import copy
import csv
import os
import re
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PySide6.QtCore import QObject, QThread, Qt, Signal
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
    QPushButton,
    QRadioButton,
    QTableWidgetItem,
    QVBoxLayout,
)

from grim_dataset import C0, RcsGrid, canonical_angular_coordinate_system
from grim_headless import (
    SUPPORTED_EXTENSIONS,
    is_supported_path,
    load_dataset as load_dataset_headless,
)
from grim_python import DatasetReference

# Characters forbidden in filenames on Windows (and `/` on POSIX). Replaced
# with `_` so dataset names with op symbols like `|`, `÷`, etc. still save.
_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# Stable identity used by consumers such as the PPT report workspace.  Row
# numbers and display names can both change, while one dataset can also appear
# in more than one row, so neither is a safe persistent selection key.
DATASET_ID_ROLE = Qt.UserRole + 32
DATASET_DIRTY_ROLE = Qt.UserRole + 33
DATASET_PATH_ROLE = Qt.UserRole + 34


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


def _ensure_grim_output_path(path: str | os.PathLike) -> str:
    output = os.fspath(path)
    return output if output.casefold().endswith(".grim") else output + ".grim"


class _GrimBatchRollbackError(RuntimeError):
    """A batch failed and at least one prior artifact could not be restored."""


def _stage_and_publish_grim_batch(
    entries: list[tuple[RcsGrid, str, str]],
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
            prior_history = dataset.history
            try:
                # A single RcsGrid may intentionally appear in multiple rows.
                # Serialize the provenance belonging to this row, then restore
                # the shared in-memory object before moving to the next entry.
                dataset.history = str(row_history or "").strip()
                dataset.save(stage_path)
            except Exception:
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
                raise
            finally:
                dataset.history = prior_history
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
    """Forward map: turntable angle φ + wedge tilt τ → conic (longitude, latitude).

    Physical setup: vertical-axis turntable, target tilted by a foam wedge
    with ridge along body-y (pitch wedge). LOS in body frame is
        r̂_body = (cos τ cos φ, −sin φ, sin τ cos φ)
    Conic output: φ' = atan2(r̂_y, r̂_x), θ' = arcsin(r̂_z).
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    tau = np.deg2rad(np.asarray(tau_deg, dtype=float))
    ct, st = np.cos(tau), np.sin(tau)
    cp, sp = np.cos(phi), np.sin(phi)
    rx = ct * cp
    ry = -sp
    rz = np.clip(st * cp, -1.0, 1.0)
    lat = np.arcsin(rz)
    lon = np.arctan2(ry, rx)
    return np.rad2deg(lon), np.rad2deg(lat)


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


class InterpolateDialog(QDialog):
    """Pick a target azimuth grid (start/stop/step) for interpolation."""

    def __init__(self, hint: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Interpolate")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Resample selected dataset(s) onto a new azimuth grid."))
        if hint:
            hint_label = QLabel(hint)
            hint_label.setStyleSheet("color: gray;")
            layout.addWidget(hint_label)

        grid = QGridLayout()
        grid.addWidget(QLabel("Start (°):"), 0, 0)
        self._spin_start = QDoubleSpinBox()
        self._spin_start.setDecimals(6)
        self._spin_start.setRange(-1e9, 1e9)
        self._spin_start.setValue(0.0)
        grid.addWidget(self._spin_start, 0, 1)

        grid.addWidget(QLabel("Stop (°):"), 1, 0)
        self._spin_stop = QDoubleSpinBox()
        self._spin_stop.setDecimals(6)
        self._spin_stop.setRange(-1e9, 1e9)
        self._spin_stop.setValue(0.0)
        grid.addWidget(self._spin_stop, 1, 1)

        grid.addWidget(QLabel("Step (°):"), 2, 0)
        self._spin_step = QDoubleSpinBox()
        self._spin_step.setDecimals(6)
        self._spin_step.setRange(1e-6, 1e6)
        self._spin_step.setValue(1.0)
        grid.addWidget(self._spin_step, 2, 1)
        layout.addLayout(grid)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def set_defaults(self, start: float, stop: float, step: float) -> None:
        self._spin_start.setValue(float(start))
        self._spin_stop.setValue(float(stop))
        self._spin_step.setValue(float(step))

    def get_values(self) -> tuple[float, float, float]:
        return (
            float(self._spin_start.value()),
            float(self._spin_stop.value()),
            float(self._spin_step.value()),
        )


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
            "Reject calibration bins whose |Aexact/Ameasured| correction exceeds "
            "this level. This catches measured-calibration nulls/noise-floor bins."
        )
        self.chk_gain_limit.toggled.connect(self.spin_gain_limit_db.setEnabled)
        gain_row.addWidget(self.chk_gain_limit)
        gain_row.addWidget(self.spin_gain_limit_db)
        form.addWidget(QLabel("Calibration validity gate:"), 3, 0)
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

        self.chk_attest = QCheckBox(
            "I confirm the acquisition, phase-center, and background assumptions."
        )
        self.chk_attest.setToolTip(
            "DUT and measured calibration share one acquisition/phase convention; "
            "the exact response uses the intended phase center; additive "
            "background/support scattering has already been removed."
        )
        layout.addWidget(self.chk_attest)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)

        warning = QLabel(
            "The exact response must be complex sigma₃D/dBsm data. A finite "
            "cylinder's 3-D reference must be supplied; GRIM will not substitute "
            "GHOST's infinite 2-D cylinder solution. Calibration nulls are rejected."
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
        self.chk_attest.toggled.connect(self._update_validity)
        self._update_validity()

    def _update_validity(self, *_args) -> None:
        same_reference = (
            self.combo_measured.currentIndex() == self.combo_exact.currentIndex()
        )
        attested = self.chk_attest.isChecked()
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        ok_button.setEnabled(not same_reference and attested)
        if same_reference:
            self.validation_label.setText(
                "Choose different datasets for measured calibration and exact reference."
            )
        elif not attested:
            self.validation_label.setText(
                "Confirm the calibration assumptions to enable Range Cal."
            )
        else:
            self.validation_label.setText("Ready to apply complex Range Cal.")

    def get_params(self) -> dict:
        measured_index = int(self.combo_measured.currentIndex())
        exact_index = int(self.combo_exact.currentIndex())
        return {
            "measured": self._entries[measured_index],
            "exact": self._entries[exact_index],
            "range_offset_m": float(self.spin_offset_m.value()),
            "allow_singleton_angular_broadcast": self.chk_broadcast.isChecked(),
            "convention_attested": self.chk_attest.isChecked(),
            "maximum_correction_gain_db": (
                float(self.spin_gain_limit_db.value())
                if self.chk_gain_limit.isChecked()
                else None
            ),
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
    """Pick the azimuth wrap range: [0, 360) or [-180, 180)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wrap Azimuth")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Wrap azimuth axis into:"))

        self._rb_0_360 = QRadioButton("0° to 360°")
        self._rb_pm180 = QRadioButton("-180° to 180°")
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

        self._chk_attest_legacy = QCheckBox(
            "For an unmarked legacy PTM, I confirm aspect +90° is conic "
            "azimuth +90° and its V/H basis follows GRIM_GC_V1"
        )
        self._chk_attest_legacy.setToolTip(
            "The legacy PTM bytes do not define aspect sign/origin or the H/V "
            "basis. Leave this clear unless the producing tool's convention is known."
        )
        source_is_unmarked_gc = (
            canonical_angular_coordinate_system(source_coordinate_system)
            == "great_circle"
            and str(source_gc_convention or "").strip().lower()
            in {"", "legacy_ptm_unspecified"}
        )
        self._chk_attest_legacy.setEnabled(
            self._radio_g2c.isChecked() and source_is_unmarked_gc
        )
        self._radio_g2c.toggled.connect(
            lambda checked: self._chk_attest_legacy.setEnabled(
                bool(checked) and source_is_unmarked_gc
            )
        )
        layout.addWidget(self._chk_attest_legacy)

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
            "attest_legacy_ptm_convention": self._chk_attest_legacy.isChecked(),
        }


class WedgeConicDialog(QDialog):
    """Pick mode (relabel / re-grid) for converting a turntable+wedge dataset
    to conic coordinates. Bounds are derived from the input dataset.

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
            "Output axes: azimuth = conic longitude φ', elevation = conic latitude θ'."
        ))

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        self._radio_relabel = QRadioButton(
            "Relabel (flatten to 1D scatter on φ' — preserves σ exactly, loses grid structure)"
        )
        self._radio_regrid = QRadioButton(
            "Re-grid (interpolate onto a uniform conic grid, bounds auto-derived)"
        )
        self._radio_regrid.setChecked(True)
        mode_layout.addWidget(self._radio_relabel)
        mode_layout.addWidget(self._radio_regrid)
        layout.addWidget(mode_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict:
        return {
            "mode": "relabel" if self._radio_relabel.isChecked() else "regrid",
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
        self._combo_scale.addItem("Both (Linear + dBsm + dBke)", "both")
        grid.addWidget(self._combo_scale, 0, 1)

        layout.addLayout(grid)

        self._chk_phase = QCheckBox("Include phase column (degrees)")
        self._chk_phase.setChecked(False)
        layout.addWidget(self._chk_phase)

        layout.addWidget(QLabel(
            "Columns: azimuth, elevation, frequency, polarization, [magnitude], [phase].\n"
            "For dBke export, frequency-dependent conversion uses the dataset frequency axis.\n"
            "One row per sample — all combinations of selected axes."
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

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.combo_stat.currentTextChanged.connect(
            lambda t: self.spin_pct.setEnabled(t == "percentile")
        )

    def get_params(self) -> tuple[str, float, list[str]]:
        """Return (statistic, percentile, axes)."""
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
        return statistic, percentile, axes



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
        units=dataset.units,
        extra={
            "phase_reference": dataset.extra["phase_reference"]
            for _ in (0,)
            if "phase_reference" in dataset.extra
        },
    )


_FREQUENCY_UNIT_FACTORS = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}


def _canonical_frequency_unit(value: object, *, default: str = "GHz") -> str:
    """Return a CSV-safe frequency unit understood by :class:`RcsGrid`."""

    text = str(value or "").strip()
    if not text:
        return default
    aliases = {
        "hz": "Hz",
        "khz": "kHz",
        "mhz": "MHz",
        "ghz": "GHz",
    }
    canonical = aliases.get(text.lower())
    if canonical is None:
        raise ValueError(
            f"unsupported frequency unit {value!r}; use Hz, kHz, MHz, or GHz"
        )
    return canonical


def _canonical_rcs_log_unit(value: object, *, default: str = "dBsm") -> str:
    """Return the preferred logarithmic RCS unit for CSV round trips."""

    text = str(value or "").strip().lower()
    if not text:
        return default
    if text == "dbsm":
        return "dBsm"
    if text == "dbke":
        return "dBke"
    if text == "db":
        return "dB"
    raise ValueError(
        f"unsupported RCS log unit {value!r}; use dBsm, dBke, or dB"
    )


def _infer_legacy_frequency_unit(values) -> str:
    """Infer units for older CSVs that predate the frequency_unit column."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    typical = float(np.nanmedian(np.abs(finite))) if finite.size else 0.0
    if typical >= 1.0e6:
        return "Hz"
    if typical >= 1.0e3:
        return "MHz"
    return "GHz"


def _csv_number(value: object, format_spec: str) -> str:
    """Format a finite number, leaving missing data blank for spreadsheets."""

    number = float(value)
    if not np.isfinite(number):
        return ""
    return format(number, format_spec)


def _write_dataset_csv(
    dataset: "RcsGrid",
    path: str,
    *,
    scale: str = "linear",
    sep: str = ",",
    include_phase: bool = False,
) -> None:
    """Write a flat az×el×freq×pol CSV from a dataset.

    Magnitude is authoritative linear power and does not require phase. This
    matters for statistics, incoherent arithmetic, and magnitude-only imports,
    whose phase is intentionally unknown.
    """

    scale = str(scale).strip().lower()
    if scale not in {"linear", "db", "dbsm", "dbke", "both"}:
        raise ValueError(
            "CSV magnitude scale must be linear, db, dbsm, dbke, or both"
        )
    is_ratio = dataset.linear_quantity() == "power_ratio"
    if is_ratio and scale not in {"linear", "db"}:
        raise ValueError("dimensionless power ratios can be exported only as Linear or dB")
    if not is_ratio and scale == "db":
        raise ValueError("dimensionless dB export is only valid for power-ratio datasets")
    az = dataset.azimuths
    el = dataset.elevations
    fr = dataset.frequencies
    pol = dataset.polarizations
    power = dataset.rcs_power
    phase = dataset.rcs_phase
    frequency_unit = _canonical_frequency_unit(
        (dataset.units or {}).get("frequency", "GHz")
    )
    rcs_log_unit = _canonical_rcs_log_unit(
        (dataset.units or {}).get("rcs_log_unit", dataset.default_log_unit())
    )
    angular_coordinate_system = dataset.angular_coordinate_system()
    great_circle_convention = (
        dataset.great_circle_coordinate_convention()
        if angular_coordinate_system == "great_circle" else ""
    )
    angular_roll_deg, angular_tilt_deg = dataset.angular_frame_orientation_deg()
    header = [
        "azimuth",
        "elevation",
        "frequency",
        "frequency_unit",
        "polarization",
        "rcs_log_unit",
        "angular_coordinate_system",
        "great_circle_coordinate_convention",
        "angular_roll_deg",
        "angular_tilt_deg",
    ]
    if scale in ("linear", "both"):
        header.append("magnitude_linear")
    if scale == "db":
        header.append("magnitude_db")
    if scale in ("dbsm", "both"):
        header.append("magnitude_dbsm")
    if scale in ("dbke", "both"):
        header.append("magnitude_dbke")
    if include_phase:
        header.append("phase_deg")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerow(header)
        for ai, az_v in enumerate(az):
            for ei, el_v in enumerate(el):
                for fi, fr_v in enumerate(fr):
                    for pi, pol_v in enumerate(pol):
                        mag = float(power[ai, ei, fi, pi])
                        phase_rad = float(phase[ai, ei, fi, pi])
                        row = [
                            str(az_v),
                            str(el_v),
                            str(fr_v),
                            frequency_unit,
                            str(pol_v),
                            rcs_log_unit,
                            angular_coordinate_system,
                            great_circle_convention,
                            format(angular_roll_deg, ".10g"),
                            format(angular_tilt_deg, ".10g"),
                        ]
                        if scale in ("linear", "both"):
                            row.append(_csv_number(mag, ".10g"))
                        if scale == "db":
                            row.append(_csv_number(dataset.linear_to_dbsm(mag), ".6f"))
                        if scale in ("dbsm", "both"):
                            row.append(_csv_number(
                                dataset.linear_to_dbsm(mag), ".6f"
                            ))
                        if scale in ("dbke", "both"):
                            row.append(_csv_number(
                                dataset.linear_to_dbke(mag, fr_v), ".6f"
                            ))
                        if include_phase:
                            phase_deg = (
                                np.degrees(np.angle(np.exp(1j * phase_rad)))
                                if np.isfinite(phase_rad)
                                else np.nan
                            )
                            row.append(_csv_number(phase_deg, ".6f"))
                        writer.writerow(row)


def _load_dataset_csv(path: str) -> "RcsGrid":
    """Load a dataset from a delimited text file exported by _write_dataset_csv()."""
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("missing CSV header row")

        field_map: dict[str, str] = {}
        for raw_name in reader.fieldnames:
            if raw_name is None:
                continue
            key = str(raw_name).strip().lower()
            if key and key not in field_map:
                field_map[key] = raw_name

        required = ["azimuth", "elevation", "frequency", "polarization"]
        missing = [name for name in required if name not in field_map]
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(missing)}")

        has_linear = "magnitude_linear" in field_map
        has_db = "magnitude_db" in field_map
        has_dbsm = "magnitude_dbsm" in field_map
        has_dbke = "magnitude_dbke" in field_map
        if not has_linear and not has_db and not has_dbsm and not has_dbke:
            raise ValueError("missing magnitude column (need magnitude_linear, magnitude_db, magnitude_dbsm, or magnitude_dbke)")
        has_phase = "phase_deg" in field_map
        has_frequency_unit = "frequency_unit" in field_map
        has_rcs_log_unit = "rcs_log_unit" in field_map
        has_angular_coordinates = "angular_coordinate_system" in field_map
        has_gc_convention = "great_circle_coordinate_convention" in field_map
        has_angular_roll = "angular_roll_deg" in field_map
        has_angular_tilt = "angular_tilt_deg" in field_map

        def _cell(row: dict[str, str], key: str) -> str:
            source = field_map[key]
            raw = row.get(source, "")
            return str(raw).strip() if raw is not None else ""

        records: list[
            tuple[float, float, float, str, float | None, float | None, float]
        ] = []
        frequency_units_seen: set[str] = set()
        rcs_log_units_seen: set[str] = set()
        angular_coordinates_seen: set[str] = set()
        gc_conventions_seen: set[str] = set()
        angular_rolls_seen: set[float] = set()
        angular_tilts_seen: set[float] = set()
        pol_order: list[str] = []
        for line_no, row in enumerate(reader, start=2):
            az_text = _cell(row, "azimuth")
            el_text = _cell(row, "elevation")
            fr_text = _cell(row, "frequency")
            pol_text = _cell(row, "polarization")
            if not (az_text or el_text or fr_text or pol_text):
                continue
            if not pol_text:
                raise ValueError(f"line {line_no}: polarization is blank")
            try:
                az = float(az_text)
                el = float(el_text)
                fr = float(fr_text)
            except ValueError as exc:
                raise ValueError(f"line {line_no}: invalid axis value ({exc})") from exc
            if not all(np.isfinite(value) for value in (az, el, fr)):
                raise ValueError(f"line {line_no}: axis values must be finite")
            if has_frequency_unit:
                unit_text = _cell(row, "frequency_unit")
                if not unit_text:
                    raise ValueError(
                        f"line {line_no}: frequency_unit is blank"
                    )
                try:
                    frequency_units_seen.add(
                        _canonical_frequency_unit(unit_text)
                    )
                except ValueError as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
            if has_rcs_log_unit:
                log_unit_text = _cell(row, "rcs_log_unit")
                if not log_unit_text:
                    raise ValueError(
                        f"line {line_no}: rcs_log_unit is blank"
                    )
                try:
                    rcs_log_units_seen.add(
                        _canonical_rcs_log_unit(log_unit_text)
                    )
                except ValueError as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
            if has_angular_coordinates:
                angular_text = _cell(row, "angular_coordinate_system")
                if not angular_text:
                    raise ValueError(
                        f"line {line_no}: angular_coordinate_system is blank"
                    )
                angular_coordinates_seen.add(
                    canonical_angular_coordinate_system(angular_text)
                )
                if has_gc_convention:
                    convention_text = _cell(
                        row, "great_circle_coordinate_convention"
                    )
                    if angular_text and canonical_angular_coordinate_system(
                        angular_text
                    ) == "great_circle":
                        gc_conventions_seen.add(
                            convention_text or "legacy_ptm_unspecified"
                        )
            for present, key, target in (
                (has_angular_roll, "angular_roll_deg", angular_rolls_seen),
                (has_angular_tilt, "angular_tilt_deg", angular_tilts_seen),
            ):
                if not present:
                    continue
                value_text = _cell(row, key)
                if not value_text:
                    raise ValueError(f"line {line_no}: {key} is blank")
                try:
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_no}: invalid {key} ({exc})"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"line {line_no}: {key} must be finite")
                target.add(value)

            lin_value: float | None = None
            dbke_value: float | None = None
            if has_linear:
                linear_text = _cell(row, "magnitude_linear")
                if linear_text:
                    try:
                        lin_value = float(linear_text)
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid magnitude_linear ({exc})") from exc
                    if not np.isfinite(lin_value):
                        lin_value = None
            if lin_value is None and has_dbsm:
                db_text = _cell(row, "magnitude_dbsm")
                if db_text:
                    try:
                        db_value = float(db_text)
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid magnitude_dbsm ({exc})") from exc
                    if np.isfinite(db_value):
                        lin_value = float(10.0 ** (db_value / 10.0))
            if lin_value is None and has_db:
                db_text = _cell(row, "magnitude_db")
                if db_text:
                    try:
                        db_value = float(db_text)
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid magnitude_db ({exc})") from exc
                    if np.isfinite(db_value):
                        lin_value = float(10.0 ** (db_value / 10.0))
            if lin_value is None and has_dbke:
                db_text = _cell(row, "magnitude_dbke")
                if db_text:
                    try:
                        dbke_value = float(db_text)
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid magnitude_dbke ({exc})") from exc
                    if not np.isfinite(dbke_value):
                        dbke_value = None

            phase_rad = float("nan")
            if has_phase:
                phase_text = _cell(row, "phase_deg")
                if phase_text:
                    try:
                        phase_rad = float(np.deg2rad(float(phase_text)))
                    except ValueError as exc:
                        raise ValueError(f"line {line_no}: invalid phase_deg ({exc})") from exc

            if pol_text not in pol_order:
                pol_order.append(pol_text)
            records.append((
                az, el, fr, pol_text, lin_value, dbke_value, phase_rad
            ))

    if not records:
        raise ValueError("CSV contains no data rows")
    if len(frequency_units_seen) > 1:
        raise ValueError(
            "CSV contains multiple frequency units; one RCS grid requires "
            "a single frequency unit"
        )
    frequency_unit = (
        next(iter(frequency_units_seen))
        if frequency_units_seen
        else _infer_legacy_frequency_unit([record[2] for record in records])
    )
    if len(rcs_log_units_seen) > 1:
        raise ValueError(
            "CSV contains multiple RCS log units; one RCS grid requires "
            "a single preferred log unit"
        )
    rcs_log_unit = (
        next(iter(rcs_log_units_seen))
        if rcs_log_units_seen
        else "dB" if has_db and not has_dbsm and not has_dbke
        else "dBke" if has_dbke and not has_dbsm else "dBsm"
    )
    if len(angular_coordinates_seen) > 1:
        raise ValueError(
            "CSV contains multiple angular coordinate systems; one RCS grid "
            "requires a single convention"
        )
    angular_coordinate_system = (
        next(iter(angular_coordinates_seen))
        if angular_coordinates_seen else "conic"
    )
    if len(gc_conventions_seen) > 1:
        raise ValueError(
            "CSV contains multiple great-circle coordinate conventions"
        )
    gc_convention = (
        next(iter(gc_conventions_seen))
        if gc_conventions_seen else "legacy_ptm_unspecified"
    )
    if len(angular_rolls_seen) > 1 or len(angular_tilts_seen) > 1:
        raise ValueError(
            "CSV contains multiple angular frame orientations; one RCS grid "
            "requires one roll/tilt pair"
        )
    angular_roll_deg = next(iter(angular_rolls_seen)) if angular_rolls_seen else 0.0
    angular_tilt_deg = next(iter(angular_tilts_seen)) if angular_tilts_seen else 0.0

    az_values = np.asarray(sorted({r[0] for r in records}), dtype=float)
    el_values = np.asarray(sorted({r[1] for r in records}), dtype=float)
    fr_values = np.asarray(sorted({r[2] for r in records}), dtype=float)
    pol_values = np.asarray(pol_order, dtype=object)

    az_index = {float(v): i for i, v in enumerate(az_values.tolist())}
    el_index = {float(v): i for i, v in enumerate(el_values.tolist())}
    fr_index = {float(v): i for i, v in enumerate(fr_values.tolist())}
    pol_index = {str(v): i for i, v in enumerate(pol_values.tolist())}

    shape = (len(az_values), len(el_values), len(fr_values), len(pol_values))
    power = np.full(shape, np.nan, dtype=np.float32)
    phase = np.full(shape, np.nan, dtype=np.float32)

    for az, el, fr, pol, lin_value, dbke_value, phase_rad in records:
        if lin_value is None and dbke_value is not None:
            freq_hz = (
                float(fr) * _FREQUENCY_UNIT_FACTORS[frequency_unit]
            )
            if np.isfinite(freq_hz) and freq_hz > 0.0:
                lin_value = float(
                    (C0 / (2.0 * np.pi * freq_hz))
                    * (10.0 ** (dbke_value / 10.0))
                )
        if lin_value is None:
            lin_value = float("nan")
        elif np.isfinite(lin_value):
            lin_value = max(lin_value, 0.0)
        ai = az_index[float(az)]
        ei = el_index[float(el)]
        fi = fr_index[float(fr)]
        pi = pol_index[str(pol)]
        power[ai, ei, fi, pi] = np.float32(lin_value)
        phase[ai, ei, fi, pi] = np.float32(phase_rad)

    if not np.isfinite(power).any():
        raise ValueError("CSV contains no finite magnitude values")

    units = {
        "azimuth": "deg",
        "elevation": "deg",
        "frequency": frequency_unit,
        "rcs_log_unit": rcs_log_unit,
        "angular_coordinate_system": angular_coordinate_system,
        "angular_roll_deg": angular_roll_deg,
        "angular_tilt_deg": angular_tilt_deg,
        "rcs_linear_quantity": (
            "power_ratio" if rcs_log_unit == "dB"
            else "sigma_2d" if rcs_log_unit == "dBke"
            else "sigma_3d"
        ),
    }
    if angular_coordinate_system == "great_circle":
        units["great_circle_coordinate_convention"] = gc_convention
    return RcsGrid(
        az_values,
        el_values,
        fr_values,
        pol_values,
        rcs_power=power,
        rcs_phase=phase,
        rcs_domain="power_phase",
        source_path=path,
        units=units,
    )


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
        return None


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
    # npz parsing and RcsGrid construction can transiently need several times
    # the archive size. Bound concurrent loads by half of currently available
    # RAM, and keep very large archives serial even without psutil.
    largest = max((os.path.getsize(path) for path in paths if os.path.isfile(path)), default=0)
    per_worker = max(64 * 1024**2, largest * 4)
    available = _available_memory_bytes()
    if available is not None:
        target = min(target, max(1, int((available * 0.5) // per_worker)))
    elif largest >= 512 * 1024**2:
        target = 1
    elif largest >= 128 * 1024**2:
        target = min(target, 2)
    return max(1, target)


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
            if total == 1:
                _consume(_load_dataset_path_task(self._tasks[0]), 1)
            elif total > 1:
                worker_count = _recommended_loader_workers(self._tasks)
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
                used_parallel = True
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


class _JoinDatasetsWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, grids: list[RcsGrid], tol: float = 1e-6, parent=None) -> None:
        super().__init__(parent)
        self._grids = list(grids)
        self._tol = float(tol)

    def run(self) -> None:
        total = max(1, len(self._grids))
        try:
            def _emit_progress(done_count: int, total_count: int) -> None:
                self.progress.emit(done_count, total_count, "Joining datasets")

            merged = _join_many_with_progress(self._grids, tol=self._tol, progress_cb=_emit_progress)
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.finished.emit({"ok": True, "merged": merged, "total": total})


class _RangeCalibrationWorker(QObject):
    """Apply one calibration definition to DUT grids off the GUI thread."""

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
                    convention_attested=True,
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
    def _ensure_background_worker_state(self) -> None:
        if hasattr(self, "_background_worker_thread"):
            return
        self._background_worker_thread: QThread | None = None
        self._background_worker: QObject | None = None
        self._background_worker_name = ""
        self._pending_join_names: list[str] | None = None
        self._pending_join_references: list[DatasetReference] | None = None
        self._pending_range_record: dict[str, object] | None = None
        self._pending_import_batches: list[tuple[tuple[str, ...], int]] = []
        self._queued_import_keys: set[str] = set()
        self._active_import_keys: set[str] = set()
        self._import_cycle_results: list[tuple[str, bool]] = []
        self._last_import_summary = ""

    def _background_job_active(self) -> bool:
        self._ensure_background_worker_state()
        thread = self._background_worker_thread
        return isinstance(thread, QThread) and thread.isRunning()

    def _try_start_background_job(self, job_name: str, worker: QObject) -> bool:
        self._ensure_background_worker_state()
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
        thread.start()
        return True

    def _on_background_thread_finished(self) -> None:
        completed_import = bool(self._active_import_keys)
        self._background_worker_thread = None
        self._background_worker = None
        self._background_worker_name = ""
        self._active_import_keys.clear()

        if self._pending_import_batches:
            paths, ignored = self._pending_import_batches.pop(0)
            for path in paths:
                self._queued_import_keys.discard(_target_path_key(path))
            self._start_dataset_import_batch(list(paths), ignored_count=ignored)
            return

        if completed_import and self._import_cycle_results:
            details = " ".join(message for message, _failed in self._import_cycle_results)
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

    def _on_load_worker_progress(self, done_count: int, total_count: int, detail: str) -> None:
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
            msg += f" Failed: {', '.join(failed)}"
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
        self.status.setToolTip(msg)
        self.status.showMessage(msg)

    def _on_join_worker_progress(self, done_count: int, total_count: int, _: str) -> None:
        self.status.showMessage(f"Joining datasets... {done_count}/{total_count}")

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
        history = f"Join (last selected wins overlap): {new_name}"
        output_name = f"Join[{new_name}]"
        output_id = self._add_dataset_row(merged, output_name, history, file_name="")
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None and input_refs:
            recorder.record_function(
                self._python_output_reference(output_id, output_name),
                "join_datasets",
                input_refs,
                kwargs={"tol": 1.0e-6},
                comment="Join datasets on their union axes",
            )
        self.status.showMessage(f"Join created. Overlap winner: {names[-1]}.")

    def _on_range_cal_worker_progress(
        self, done_count: int, total_count: int, detail: str
    ) -> None:
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
                            f"    convention_attested=True,\n"
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
        selected = self.table.selectionModel().selectedRows()
        self._update_dataset_selection_order([idx.row() for idx in selected])
        if not selected:
            self.active_dataset = None
            self._clear_param_lists()
            return
        row = selected[0].row()
        item = self.table.item(row, 0)
        dataset = item.data(Qt.UserRole) if item else None
        if not isinstance(dataset, RcsGrid):
            self.active_dataset = None
            self._clear_param_lists()
            return
        self.active_dataset = dataset
        self._populate_params(dataset)

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
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
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
        """Reject legacy programmatic inline edits without mutating the grid."""

        if self.active_dataset is None:
            return
        axis_arr = self.active_dataset.get_axis(axis_name)
        idx = item.data(Qt.UserRole + 1)
        if idx is None:
            return
        if idx < 0 or idx >= len(axis_arr):
            return
        old_value = axis_arr[idx]
        widget.blockSignals(True)
        try:
            item.setText(str(old_value))
            item.setData(Qt.UserRole, old_value)
        finally:
            widget.blockSignals(False)
        self.status.showMessage(
            "Parameter axes are read-only. Use a validated dataset operation "
            "to transform coordinates."
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

    def _combine_datasets_add(
        self,
        op_label: str,
        op_symbol: str,
        func_add: str,
        func_add_many: str,
    ) -> None:
        datasets = self._selected_datasets_ordered()
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        names = [name for name, _ in datasets]
        base = datasets[0][1]
        try:
            if len(datasets) == 2:
                result = getattr(base, func_add)(datasets[1][1])
            else:
                others = [ds for _, ds in datasets[1:]]
                result = getattr(base, func_add_many)(*others)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"
        output_id = self._add_dataset_row(result, new_name, history, file_name="")
        input_refs = self._python_input_references(datasets)
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None and input_refs is not None:
            method = func_add if len(datasets) == 2 else func_add_many
            recorder.record_expression(
                self._python_output_reference(output_id, new_name),
                input_refs,
                lambda variables, method=method: (
                    f"{variables[0]}.{method}({', '.join(variables[1:])})"
                ),
                comment=op_label,
            )
        self.status.showMessage(f"{op_label} created: {new_name}")

    def _combine_datasets_sub(self, op_label: str, op_symbol: str, func_sub: str) -> None:
        datasets = self._selected_datasets_ordered(use_selection_order=True)
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        names = [name for name, _ in datasets]
        result = datasets[0][1]
        try:
            for _, ds in datasets[1:]:
                result = getattr(result, func_sub)(ds)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"
        output_id = self._add_dataset_row(result, new_name, history, file_name="")
        input_refs = self._python_input_references(datasets)
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None and input_refs is not None:
            recorder.record_expression(
                self._python_output_reference(output_id, new_name),
                input_refs,
                lambda variables, method=func_sub: (
                    ".".join(
                        [variables[0]]
                        + [f"{method}({variable})" for variable in variables[1:]]
                    )
                ),
                comment=op_label,
            )
        self.status.showMessage(f"{op_label} created: {new_name}")

    def _coherent_add_selected(self) -> None:
        self._combine_datasets_add("Coherent +", "+", "coherent_add", "coherent_add_many")

    def _coherent_sub_selected(self) -> None:
        self._combine_datasets_sub("Coherent -", "-", "coherent_subtract")

    def _incoherent_add_selected(self) -> None:
        self._combine_datasets_add("Incoherent +", "+", "incoherent_add", "incoherent_add_many")

    def _incoherent_sub_selected(self) -> None:
        self._combine_datasets_sub("Incoherent -", "-", "incoherent_subtract")

    def _dbdiff_selected(self) -> None:
        self._combine_datasets_sub("Δ dB", "Δ", "arithmetic_db_subtract")

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
        self.status.showMessage(f"Joining datasets... 0/{len(grids)}")

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
        try:
            overlap_grids = RcsGrid.overlap_many(*grids, tol=1e-6)
            produced = 0
            output_refs: list[DatasetReference] = []
            for (name, _), overlap_grid in zip(datasets, overlap_grids):
                history = f"Overlap with [{', '.join(names)}]: {name}"
                output_name = f"{name} [Overlap]"
                output_id = self._add_dataset_row(
                    overlap_grid, output_name, history, file_name=""
                )
                output_refs.append(
                    self._python_output_reference(output_id, output_name)
                )
                produced += 1
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        if produced == 0:
            self.status.showMessage("No overlap outputs were created.")
            return
        input_refs = self._python_input_references(datasets)
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None and input_refs is not None:
            recorder.record_multi_function(
                output_refs,
                "RcsGrid.overlap_many",
                input_refs,
                kwargs={"tol": 1.0e-6},
                comment="Crop datasets to their common finite overlap",
            )
        self.status.showMessage(f"Overlap created {produced} dataset(s).")

    def _prompt_choice(self, title: str, label: str, choices: list[str], default_idx: int = 0) -> str | None:
        value, ok = QInputDialog.getItem(self, title, label, choices, default_idx, False)
        if not ok:
            return None
        return str(value)

    def _slice_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to slice.",
        )
        if datasets is None:
            return

        sel_az = self._selected_values(self.list_az)
        sel_el = self._selected_values(self.list_elev)
        sel_freq = self._selected_values(self.list_freq)
        sel_pol = self._selected_values(self.list_pol)

        if not (sel_az or sel_el or sel_freq or sel_pol):
            self.status.showMessage(
                "Select parameter values (azimuth/elevation/frequency/polarization) to slice."
            )
            return

        crop_params = {
            "azimuths": sel_az or None,
            "elevations": sel_el or None,
            "frequencies": sel_freq or None,
            "polarizations": sel_pol or None,
        }

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                sliced = dataset.axis_crop(**crop_params)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = (
                "Slice (selected params): "
                f"{name} | az={len(sliced.azimuths)}, el={len(sliced.elevations)}, "
                f"freq={len(sliced.frequencies)}, pol={len(sliced.polarizations)}"
            )
            output_name = f"{name} [Slice]"
            output_id = self._add_dataset_row(
                sliced, output_name, history, file_name=""
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_method(
                    self._python_output_reference(output_id, output_name),
                    source_ref,
                    "axis_crop",
                    kwargs=crop_params,
                    comment=f"Slice {name} by selected physical values",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Slice created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Slice created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Slice created {produced} dataset(s).")

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

        statistic, percentile, axes = dlg.get_params()
        if not axes:
            self.status.showMessage("Select at least one axis for statistics reduction.")
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                stat_grid = dataset.statistics_dataset(
                    statistic=statistic,
                    axes=axes,
                    domain="magnitude",
                    percentile=percentile,
                    broadcast_reduced=True,
                )
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue

            if statistic == "percentile":
                stat_label = f"p{percentile:g}"
            else:
                stat_label = statistic
            history = f"Statistics ({stat_label}, axes={axes}): {name}"
            output_name = f"{name} [{stat_label}]"
            output_id = self._add_dataset_row(
                stat_grid, output_name, history, file_name=""
            )
            source_ref = self._python_reference_for_dataset(dataset)
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
                        "broadcast_reduced": True,
                    },
                    comment=f"Reduce {name} to {stat_label} statistics",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Statistics created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Statistics created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Statistics created {produced} dataset(s).")

    def _delete_selected_datasets(self) -> None:
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
                f"{details}\n\nDelete them permanently?",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                self.status.showMessage("Delete cancelled; unsaved datasets were kept.")
                return
        for row in rows:
            self.table.removeRow(row)
        self.active_dataset = None
        self._clear_param_lists()
        notify = getattr(self, "_notify_dataset_catalog_changed", None)
        if callable(notify):
            notify()
        self.status.showMessage(f"Deleted {len(rows)} dataset(s).")

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

        try:
            published = _stage_and_publish_grim_batch(
                [
                    (
                        dataset,
                        target,
                        (
                            self.table.item(row, 2).text()
                            if self.table.item(row, 2) is not None
                            else str(dataset.history or "")
                        ),
                    )
                    for row, dataset, target in plan
                ]
            )
        except Exception as exc:
            if isinstance(exc, _GrimBatchRollbackError):
                failure_text = str(exc)
            else:
                failure_text = "No partial batch was kept. " + str(exc)
            QMessageBox.critical(
                self,
                f"{dialog_title} Failed",
                failure_text,
            )
            self.status.showMessage(f"Save failed: {exc}")
            return False

        recorded_saves: list[tuple[DatasetReference, str]] = []
        for (row, _dataset, _target), output_path in zip(plan, published):
            self._set_dataset_row_saved(row, output_path)
            name_item = self.table.item(row, 0)
            if name_item is not None:
                recorded_saves.append(
                    (
                        DatasetReference(
                            str(name_item.data(DATASET_ID_ROLE) or ""),
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
        self.status.showMessage(
            f"Saved {len(published)} dataset(s) to "
            f"{os.path.dirname(os.path.abspath(published[0]))}."
        )
        return True

    def _export_plot(self) -> None:
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

    def _on_plot_context_menu(self, pos) -> None:
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
        produced = 0
        skipped: list[str] = []
        for name, dataset in others:
            try:
                aligned = dataset.align_to(ref_grid, mode=mode)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Align ({mode}) to {ref_name}: {name}"
            output_name = f"{name} [Aligned]"
            output_id = self._add_dataset_row(
                aligned, output_name, history, file_name=""
            )
            source_ref = self._python_reference_for_dataset(dataset)
            reference_ref = self._python_reference_for_dataset(ref_grid)
            recorder = getattr(self, "python_recorder", None)
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
            produced += 1

        if produced == 0:
            self.status.showMessage("Align created 0 datasets.")
            return
        msg = f"Align created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _interpolate_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to interpolate.",
        )
        if datasets is None:
            return

        hint = None
        default_start, default_stop, default_step = -180.0, 179.0, 1.0
        if len(datasets) == 1:
            az = np.asarray(datasets[0][1].azimuths, dtype=float)
            if az.size:
                az_min, az_max = float(az.min()), float(az.max())
                az_step = float(np.median(np.diff(az))) if az.size > 1 else 1.0
                hint = f"Current azimuths: {az_min:g}° to {az_max:g}° ({az.size} samples, ~{az_step:g}° step)"
                default_start, default_stop, default_step = az_min, az_max, az_step

        dlg = InterpolateDialog(hint=hint, parent=self)
        dlg.set_defaults(default_start, default_stop, default_step)
        if dlg.exec() != QDialog.Accepted:
            return

        start, stop, step = dlg.get_values()
        if step <= 0.0:
            self.status.showMessage("Step must be positive.")
            return
        if stop < start:
            self.status.showMessage("Stop must be ≥ start.")
            return

        n = int(np.floor((stop - start) / step + 1e-9)) + 1
        new_az = start + step * np.arange(n, dtype=float)

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                interpolated = dataset.interpolate_axis("azimuth", new_az)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = (
                f"Interpolate azimuth [{start:g}°..{stop:g}° step {step:g}°]: {name}"
            )
            output_name = f"{name} [Interp]"
            output_id = self._add_dataset_row(
                interpolated, output_name, history, file_name=""
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_method(
                    self._python_output_reference(output_id, output_name),
                    source_ref,
                    "interpolate_axis",
                    args=("azimuth", new_az.tolist()),
                    comment=f"Interpolate {name} on a resolved azimuth grid",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage(
                f"Interpolate created 0 datasets. Skipped: {', '.join(skipped)}"
                if skipped
                else "Interpolate created 0 datasets."
            )
            return
        msg = f"Interpolate created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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
            az_vals = np.asarray(ref.azimuths, dtype=float)
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

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                mirrored = dataset.mirror_about_azimuth(about)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Mirror about az={about:.6g} deg: {name}"
            output_name = f"{name} [Mirror {about:.6g}°]"
            output_id = self._add_dataset_row(
                mirrored,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_method(
                    self._python_output_reference(output_id, output_name),
                    source_ref,
                    "mirror_about_azimuth",
                    args=(float(about),),
                    comment=f"Mirror {name} about azimuth {about:g} degrees",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Mirror created 0 datasets.")
            return
        msg = f"Mirror created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _wrap_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to wrap.",
        )
        if datasets is None:
            return

        dlg = WrapDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        mode = dlg.get_mode()
        suffix = "0–360°" if mode == "0_360" else "-180–180°"

        produced = 0
        dropped_total = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                wrapped = dataset.wrap_azimuth(mode)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            dropped = len(dataset.azimuths) - len(wrapped.azimuths)
            dropped_total += dropped
            drop_note = f" (dropped {dropped} duplicate az)" if dropped else ""
            history = f"Wrap az to {suffix}{drop_note}: {name}"
            output_name = f"{name} [Wrap {suffix}]"
            output_id = self._add_dataset_row(
                wrapped,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_method(
                    self._python_output_reference(output_id, output_name),
                    source_ref,
                    "wrap_azimuth",
                    args=(mode,),
                    comment=f"Wrap {name} to {suffix}",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Wrap created 0 datasets.")
            return
        msg = f"Wrap created {produced} dataset(s)."
        if dropped_total:
            msg += f" Dropped {dropped_total} duplicate azimuth sample(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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

        phasor = np.exp(1j * np.deg2rad(ph_delta)) if ph_on else None

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                shifted = dataset
                if az_on:
                    shifted = shifted.shift_azimuth(az_delta)
                if el_on:
                    shifted = shifted.shift_elevation(el_delta)
                if ph_on:
                    shifted = _dataset_with_rcs(
                        shifted,
                        shifted.rcs * phasor,
                        rcs_power=shifted.rcs_power,
                        rcs_domain="complex_amplitude",
                    )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Shift ({history_axes}): {name}"
            output_name = f"{name} [Shift {suffix}]"
            output_id = self._add_dataset_row(
                shifted,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
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
            produced += 1

        if produced == 0:
            self.status.showMessage("Shift created 0 datasets.")
            return
        msg = f"Shift created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                rounded = dataset
                if params["azimuths"]:
                    rounded = rounded.round_azimuths(decimals)
                if params["elevations"]:
                    rounded = rounded.round_elevations(decimals)
                if params["frequencies"]:
                    rounded = rounded.round_frequencies(decimals)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = (
                f"Round {axes_label} to {decimals} dp: {name}"
            )
            output_name = f"{name} [Round {decimals}dp]"
            output_id = self._add_dataset_row(
                rounded,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                enabled_methods = [
                    method
                    for enabled, method in (
                        (params["azimuths"], "round_azimuths"),
                        (params["elevations"], "round_elevations"),
                        (params["frequencies"], "round_frequencies"),
                    )
                    if enabled
                ]
                recorder.record_expression(
                    self._python_output_reference(output_id, output_name),
                    [source_ref],
                    lambda variables, methods=tuple(enabled_methods), decimals=decimals: (
                        variables[0]
                        + "".join(f".{method}({int(decimals)})" for method in methods)
                    ),
                    comment=f"Round {name} axes {axes_label} to {decimals} decimals",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Round created 0 datasets.")
            return
        msg = f"Round created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _swap_elevation_azimuth_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to swap elevation and azimuth.",
        )
        if datasets is None:
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                swapped = dataset.swap_elevation_azimuth()
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Swap El/Az: {name}"
            output_name = f"{name} [Swap El/Az]"
            output_id = self._add_dataset_row(
                swapped,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_method(
                    self._python_output_reference(output_id, output_name),
                    source_ref,
                    "swap_elevation_azimuth",
                    comment=f"Swap elevation and azimuth for {name}",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Swap El/Az created 0 datasets.")
            return
        msg = f"Swap El/Az created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _elevation_to_azimuth_360_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to convert elevation pair into 360 azimuth.",
        )
        if datasets is None:
            return

        selected_el_values = self._selected_values(self.list_elev)
        selected_pair: tuple[float, float] | None = None
        if len(selected_el_values) == 2:
            try:
                pair = tuple(sorted(float(v) for v in selected_el_values))
                selected_pair = (pair[0], pair[1])
            except (TypeError, ValueError):
                selected_pair = None

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                if selected_pair is None:
                    result = dataset.combine_elevation_pair_to_azimuth_360(azimuth_shift_deg=180.0)
                    pair_text = "min/max elevation"
                else:
                    result = dataset.combine_elevation_pair_to_azimuth_360(
                        selected_pair[0],
                        selected_pair[1],
                        azimuth_shift_deg=180.0,
                    )
                    pair_text = f"{selected_pair[0]:.6g}/{selected_pair[1]:.6g} deg"
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue

            history = f"El->Az360 (shift +180 deg, pair={pair_text}): {name}"
            output_name = f"{name} [El->Az360]"
            output_id = self._add_dataset_row(
                result,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                args = selected_pair or ()
                recorder.record_method(
                    self._python_output_reference(output_id, output_name),
                    source_ref,
                    "combine_elevation_pair_to_azimuth_360",
                    args=args,
                    kwargs={"azimuth_shift_deg": 180.0},
                    comment=f"Convert {name} elevation pair to 360-degree azimuth",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("El->Az360 created 0 datasets.")
            return
        msg = f"El->Az360 created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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
        if not params["convention_attested"]:
            self.status.showMessage(
                "Range Cal: confirm the acquisition, phase-center, and background "
                "statement before applying complex calibration."
            )
            return

        worker = _RangeCalibrationWorker(
            targets,
            (measured_name, measured),
            (exact_name, exact),
            params,
        )
        worker.progress.connect(self._on_range_cal_worker_progress)
        worker.finished.connect(self._on_range_cal_worker_finished)
        self.status.showMessage(f"Range Cal... 0/{len(targets)}")
        if self._try_start_background_job("Range Cal", worker):
            target_refs = self._python_input_references(targets) or []
            measured_ref = self._python_reference_for_dataset(measured)
            exact_ref = self._python_reference_for_dataset(exact)
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

        linear_scale = 10.0 ** (value / 10.0)
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                if dataset.rcs_domain == "complex_amplitude":
                    result_rcs = dataset.rcs * np.sqrt(linear_scale)
                else:
                    result_rcs = dataset.rcs * linear_scale
                result = _dataset_with_rcs(
                    dataset,
                    result_rcs,
                    rcs_power=dataset.rcs_power * linear_scale,
                    rcs_domain=dataset.rcs_domain,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Offset ({value:+.6g}): {name}"
            output_name = f"{name} [Offset {value:+.6g}]"
            output_id = self._add_dataset_row(
                result, output_name, history, file_name=""
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_function(
                    self._python_output_reference(output_id, output_name),
                    "offset_db",
                    [source_ref],
                    args=(float(value),),
                    comment=f"Offset {name} by {value:+g} dB",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Offset created 0 datasets.")
            return
        msg = f"Offset created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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

        c0 = 299_792_458.0
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            current_unit = str((dataset.units or {}).get("rcs_log_unit", "dBsm")).strip().lower()
            if current_unit == "dbke":
                skipped.append(f"{name} (already dBke)")
                continue
            try:
                # Per-frequency extrusion conversion: σ_2D = σ_3D · λ_f / (2 L²)
                # = σ_3D · c / (2 L² f).  Shape the (n_freq,) factor so it
                # broadcasts over (n_az, n_el, n_freq, n_pol).
                freq_hz = np.asarray(
                    dataset._frequency_value_to_hz(dataset.frequencies), dtype=float
                )
                scale_per_f = np.where(
                    np.isfinite(freq_hz) & (freq_hz > 0.0),
                    c0 / (2.0 * length_m * length_m * freq_hz),
                    np.nan,
                )
                scale_4d = scale_per_f.reshape(1, 1, -1, 1)
                new_power = dataset.rcs_power * scale_4d
                new_units = dict(dataset.units or {})
                new_units["rcs_log_unit"] = "dBke"
                new_units["rcs_linear_quantity"] = "sigma_2d"
                if dataset.rcs_domain == "complex_amplitude":
                    amp_scale_4d = np.sqrt(np.maximum(scale_4d, 0.0))
                    new_rcs = dataset.rcs * amp_scale_4d
                    result = RcsGrid(
                        dataset.azimuths,
                        dataset.elevations,
                        dataset.frequencies,
                        dataset.polarizations,
                        new_rcs,
                        rcs_power=new_power,
                        rcs_domain=dataset.rcs_domain,
                        units=new_units,
                    )
                else:
                    result = RcsGrid(
                        dataset.azimuths,
                        dataset.elevations,
                        dataset.frequencies,
                        dataset.polarizations,
                        rcs=None,
                        rcs_power=new_power,
                        rcs_phase=dataset.rcs_phase,
                        rcs_domain=dataset.rcs_domain,
                        units=new_units,
                    )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Convert to dBke (extruded L={length_label}, {length_m:.6g} m): {name}"
            output_name = f"{name} [→ dBke L={length_label}]"
            output_id = self._add_dataset_row(
                result,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_function(
                    self._python_output_reference(output_id, output_name),
                    "convert_extrusion",
                    [source_ref],
                    kwargs={"to": "dbke", "length_m": float(length_m)},
                    comment=f"Convert {name} from dBsm to dBke",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Convert to dBke created 0 datasets.")
            return
        # Frequency-independent dB offset (extrusion approximation) for the status line.
        offset_db = 10.0 * np.log10(np.pi / (length_m * length_m))
        msg = (
            f"Convert to dBke created {produced} dataset(s) "
            f"(L={length_label} → constant offset {offset_db:+.2f} dB)."
        )
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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

        c0 = 299_792_458.0
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            current_unit = str((dataset.units or {}).get("rcs_log_unit", "dBsm")).strip().lower()
            if current_unit == "dbsm":
                skipped.append(f"{name} (already dBsm)")
                continue
            try:
                # Inverse of the dBke conversion: σ_3D = σ_2D · 2L²/λ
                # = σ_2D · 2L²·f/c. Per-frequency factor broadcast over
                # (n_az, n_el, n_freq, n_pol).
                freq_hz = np.asarray(
                    dataset._frequency_value_to_hz(dataset.frequencies), dtype=float
                )
                scale_per_f = np.where(
                    np.isfinite(freq_hz) & (freq_hz > 0.0),
                    2.0 * length_m * length_m * freq_hz / c0,
                    np.nan,
                )
                scale_4d = scale_per_f.reshape(1, 1, -1, 1)
                new_power = dataset.rcs_power * scale_4d
                new_units = dict(dataset.units or {})
                new_units["rcs_log_unit"] = "dBsm"
                new_units["rcs_linear_quantity"] = "sigma_3d"
                if dataset.rcs_domain == "complex_amplitude":
                    amp_scale_4d = np.sqrt(np.maximum(scale_4d, 0.0))
                    new_rcs = dataset.rcs * amp_scale_4d
                    result = RcsGrid(
                        dataset.azimuths,
                        dataset.elevations,
                        dataset.frequencies,
                        dataset.polarizations,
                        new_rcs,
                        rcs_power=new_power,
                        rcs_domain=dataset.rcs_domain,
                        units=new_units,
                    )
                else:
                    result = RcsGrid(
                        dataset.azimuths,
                        dataset.elevations,
                        dataset.frequencies,
                        dataset.polarizations,
                        rcs=None,
                        rcs_power=new_power,
                        rcs_phase=dataset.rcs_phase,
                        rcs_domain=dataset.rcs_domain,
                        units=new_units,
                    )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Convert to dBsm (extruded L={length_label}, {length_m:.6g} m): {name}"
            output_name = f"{name} [→ dBsm L={length_label}]"
            output_id = self._add_dataset_row(
                result,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_function(
                    self._python_output_reference(output_id, output_name),
                    "convert_extrusion",
                    [source_ref],
                    kwargs={"to": "dbsm", "length_m": float(length_m)},
                    comment=f"Convert {name} from dBke to dBsm",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Convert to dBsm created 0 datasets.")
            return
        # The extrusion offset is the exact negative of the forward direction:
        # dBsm − dBke = 20·log10(L) − 10·log10(π).
        offset_db = 20.0 * np.log10(length_m) - 10.0 * np.log10(np.pi)
        msg = (
            f"Convert to dBsm created {produced} dataset(s) "
            f"(L={length_label} → constant offset {offset_db:+.2f} dB)."
        )
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                if mode != "relabel":
                    raise ValueError(
                        "general conic/great-circle conversion is unavailable "
                        "until full polarization-basis rotation is implemented"
                    )
                result, suffix, hist_extra = self._conic_gc_relabel(
                    dataset,
                    direction,
                    attest_legacy_ptm_convention=attest_legacy,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue

            arrow = "Conic→GC" if direction == "conic_to_gc" else "GC→Conic"
            history = f"{arrow} {mode}: {name}{hist_extra}"
            output_name = f"{name} [{arrow} {suffix}]"
            output_id = self._add_dataset_row(
                result,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_method(
                    self._python_output_reference(output_id, output_name),
                    source_ref,
                    "convert_equatorial_conic_gc",
                    args=(direction,),
                    kwargs={"attest_legacy_ptm_convention": attest_legacy},
                    comment=f"{arrow} exact zero-plane relabel for {name}",
                )
            produced += 1

        if produced == 0:
            msg = "Conic↔GC created 0 datasets."
            if skipped:
                msg += f" Skipped: {', '.join(skipped)}"
            self.status.showMessage(msg)
            return
        arrow = "Conic→GC" if direction == "conic_to_gc" else "GC→Conic"
        msg = f"{arrow} ({mode}) created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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
        mode = dlg.get_params()["mode"]

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                az_in = np.asarray(dataset.azimuths, dtype=float)
                el_in = np.asarray(dataset.elevations, dtype=float)
                if az_in.size < 2 or el_in.size < 1:
                    skipped.append(f"{name} (need ≥2 azimuths and ≥1 elevation)")
                    continue

                if mode == "relabel":
                    result, suffix, hist_extra = self._wedge_to_conic_relabel(dataset)
                else:
                    result, suffix, hist_extra = self._wedge_to_conic_regrid(dataset)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue

            history = f"Wedge→Conic {mode}: {name}{hist_extra}"
            output_name = f"{name} [Wedge→Conic {suffix}]"
            output_id = self._add_dataset_row(
                result,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_function(
                    self._python_output_reference(output_id, output_name),
                    "wedge_to_conic",
                    [source_ref],
                    kwargs={"mode": mode},
                    comment=f"Wedge-to-Conic {mode} for {name}",
                )
            produced += 1

        if produced == 0:
            self.status.showMessage("Wedge→Conic created 0 datasets.")
            return
        msg = f"Wedge→Conic ({mode}) created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _wedge_to_conic_relabel(self, dataset: "RcsGrid"):
        """Flatten the (φ, τ) grid to a 1-D scatter on conic longitude φ'.

        Each input sample carries a unique (φ', θ') pair. Sort by φ' and store
        θ' per sample in history (the result isn't on a rectangular conic
        grid; this preserves σ exactly without interpolation).
        """
        az_in = np.asarray(dataset.azimuths, dtype=float)
        el_in = np.asarray(dataset.elevations, dtype=float)
        n_az, n_el = az_in.size, el_in.size

        phi_grid, tau_grid = np.meshgrid(az_in, el_in, indexing="ij")
        new_lon, new_lat = _wedge_to_conic_deg(phi_grid.ravel(), tau_grid.ravel())

        order = np.argsort(new_lon, kind="stable")
        flat_power = dataset.rcs_power.reshape(
            n_az * n_el, dataset.frequencies.size, dataset.polarizations.size
        )
        flat_phase = dataset.rcs_phase.reshape(
            n_az * n_el, dataset.frequencies.size, dataset.polarizations.size
        )
        sorted_lon = new_lon[order]
        sorted_lat = new_lat[order]
        sorted_power = flat_power[order][:, None, :, :]
        sorted_phase = flat_phase[order][:, None, :, :]

        result = RcsGrid(
            sorted_lon,
            np.array([0.0]),
            dataset.frequencies,
            dataset.polarizations,
            rcs=None,
            rcs_power=sorted_power,
            rcs_phase=sorted_phase,
            rcs_domain=dataset.rcs_domain,
            units=dict(dataset.units or {}),
        )
        lat_preview = ", ".join(f"{v:.3g}" for v in sorted_lat[: min(8, sorted_lat.size)])
        if sorted_lat.size > 8:
            lat_preview += f", … ({sorted_lat.size} total)"
        hist_extra = (
            f"; relabeled axis 0 to conic longitude φ' (sorted asc); "
            f"θ' per sample = [{lat_preview}]; "
            f"θ' ∈ [{sorted_lat.min():.3g}, {sorted_lat.max():.3g}]"
        )
        return result, "φ'-scatter", hist_extra

    def _wedge_to_conic_regrid(self, dataset: "RcsGrid"):
        """Interpolate the (φ, τ) scatter onto a uniform conic (φ', θ') grid.

        The forward map (φ, τ) → (φ', θ') isn't bijective, so we can't
        back-solve like the conic↔GC re-grid does. Instead, forward-map every
        input sample, then use `LinearNDInterpolator` (Delaunay triangulation
        on the scattered output points) to fill the output grid. Phase uses
        nearest-neighbour (`NearestNDInterpolator`) for the same wrap reasons
        as the conic↔GC path.

        Output bounds: hull of the forward-mapped longitude/latitude (no user
        inputs). Output sample count: input N_φ × N_τ.
        """
        from scipy.interpolate import LinearNDInterpolator

        az_in = np.asarray(dataset.azimuths, dtype=float)
        el_in = np.asarray(dataset.elevations, dtype=float)
        n_az, n_el = az_in.size, el_in.size

        phi_grid, tau_grid = np.meshgrid(az_in, el_in, indexing="ij")
        lon_in, lat_in = _wedge_to_conic_deg(phi_grid.ravel(), tau_grid.ravel())

        lon_lo, lon_hi = float(lon_in.min()), float(lon_in.max())
        lat_lo, lat_hi = float(lat_in.min()), float(lat_in.max())
        if not (lon_hi > lon_lo) or not (lat_hi > lat_lo):
            raise ValueError("forward-mapped hull is degenerate")

        n_lon = max(int(n_az), 2)
        n_lat = max(int(n_el), 2)
        lon_grid = np.linspace(lon_lo, lon_hi, n_lon, dtype=float)
        lat_grid = np.linspace(lat_lo, lat_hi, n_lat, dtype=float)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid, indexing="ij")
        query = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
        points = np.column_stack([lon_in, lat_in])

        n_f = dataset.frequencies.size
        n_pol = dataset.polarizations.size
        phase_complete = not np.any(
            np.isfinite(dataset.rcs_power) & ~np.isfinite(dataset.rcs_phase)
        )
        if phase_complete:
            flat_complex = dataset.rcs.reshape(n_az * n_el, n_f * n_pol)
            real_out = LinearNDInterpolator(
                points, flat_complex.real, fill_value=np.nan
            )(query)
            imag_out = LinearNDInterpolator(
                points, flat_complex.imag, fill_value=np.nan
            )(query)
            complex_out = real_out + 1j * imag_out
            power_out = np.abs(complex_out) ** 2
            phase_out = np.angle(complex_out)
        else:
            flat_power = dataset.rcs_power.reshape(n_az * n_el, n_f * n_pol)
            power_out = LinearNDInterpolator(
                points, flat_power, fill_value=np.nan
            )(query)
            phase_out = np.full(power_out.shape, np.nan, dtype=power_out.dtype)

        new_shape = (n_lon, n_lat, n_f, n_pol)
        power_out = power_out.reshape(new_shape).astype(dataset.rcs_power.dtype)
        phase_out = phase_out.reshape(new_shape).astype(dataset.rcs_phase.dtype)

        result = RcsGrid(
            lon_grid,
            lat_grid,
            dataset.frequencies,
            dataset.polarizations,
            rcs=None,
            rcs_power=power_out,
            rcs_phase=phase_out,
            rcs_domain=dataset.rcs_domain,
            units=dict(dataset.units or {}),
        )
        in_bounds = int(np.sum(np.isfinite(power_out[..., 0, 0])))
        total = n_lon * n_lat
        coverage = 100.0 * in_bounds / max(total, 1)
        hist_extra = (
            f"; output axes φ'=[{lon_grid[0]:g}..{lon_grid[-1]:g}/{lon_grid.size}], "
            f"θ'=[{lat_grid[0]:g}..{lat_grid[-1]:g}/{lat_grid.size}]; "
            f"coverage {coverage:.1f}%"
        )
        return result, "φ'×θ'", hist_extra

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

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                az = np.asarray(dataset.azimuths, dtype=float)
                if az.size < 2:
                    skipped.append(f"{name} (need ≥2 azimuth samples)")
                    continue
                az_min = float(az.min())
                az_max = float(az.max())
                if az_max - az_min < window_deg * 0.5:
                    skipped.append(f"{name} (az span < window/2)")
                    continue

                # Output azimuth grid: window centres stepped by `slide`,
                # restricted to centres whose full window stays inside the
                # data so each output sample is supported by real samples
                # on both sides.
                half_w = window_deg * 0.5
                first_centre = az_min + half_w
                last_centre = az_max - half_w
                if last_centre < first_centre:
                    # Window wider than the data — fall back to a single
                    # centre at the midpoint so the user still gets one row.
                    centres = np.array([0.5 * (az_min + az_max)], dtype=float)
                else:
                    n_steps = int(np.floor((last_centre - first_centre) / slide_deg + 1e-9)) + 1
                    centres = first_centre + np.arange(n_steps, dtype=float) * slide_deg

                # Compute the median of the linear power in each window and
                # A power median has no physically defined coherent phase. Mark
                # it unknown so this statistical result cannot later be added
                # as if it were a measured complex field.
                n_el = dataset.elevations.size
                n_f = dataset.frequencies.size
                n_pol = dataset.polarizations.size
                new_power = np.empty(
                    (centres.size, n_el, n_f, n_pol), dtype=dataset.rcs_power.dtype
                )
                new_phase = np.full_like(new_power, np.nan)
                for i, c in enumerate(centres):
                    in_window = np.where(np.abs(az - c) <= half_w)[0]
                    if in_window.size == 0:
                        # Fall back to nearest single sample.
                        in_window = np.array([int(np.argmin(np.abs(az - c)))])
                    window_power = dataset.rcs_power[in_window, :, :, :]
                    new_power[i] = np.nanmedian(window_power, axis=0)

                result = RcsGrid(
                    centres,
                    dataset.elevations,
                    dataset.frequencies,
                    dataset.polarizations,
                    rcs=None,
                    rcs_power=new_power,
                    rcs_phase=new_phase,
                    rcs_domain=dataset.rcs_domain,
                    units=dict(dataset.units or {}),
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue

            history = (
                f"Medianize (window={window_deg:g}°, slide={slide_deg:g}°): {name}"
            )
            output_name = f"{name} [Median w={window_deg:g}° s={slide_deg:g}°]"
            output_id = self._add_dataset_row(
                result,
                output_name,
                history,
                file_name="",
            )
            source_ref = self._python_reference_for_dataset(dataset)
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
            produced += 1

        if produced == 0:
            self.status.showMessage("Medianize created 0 datasets.")
            return
        msg = (
            f"Medianize created {produced} dataset(s) "
            f"(window={window_deg:g}°, slide={slide_deg:g}°)."
        )
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _duplicate_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to duplicate.",
        )
        if datasets is None:
            return

        for name, dataset in datasets:
            dup = RcsGrid(
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
            output_name = f"{name} [Copy]"
            output_id = self._add_dataset_row(
                dup, output_name, f"Duplicate of: {name}", file_name=""
            )
            source_ref = self._python_reference_for_dataset(dataset)
            recorder = getattr(self, "python_recorder", None)
            if recorder is not None and source_ref is not None:
                recorder.record_function(
                    self._python_output_reference(output_id, output_name),
                    "duplicate_dataset",
                    [source_ref],
                    comment=f"Duplicate {name}",
                )
        self.status.showMessage(f"Duplicated {len(datasets)} dataset(s).")

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
                    parts.append(pol_label)
                if n_el > 1:
                    parts.append(f"el{float(dataset.elevations[ei]):g}")
                yield "_".join(parts), ei, pi

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
                saved = dataset.save_pio(path, el_idx=el_idx, pol_idx=pol_idx)
                self.status.showMessage(f"Exported {os.path.basename(saved)}.")
                return
            directory = QFileDialog.getExistingDirectory(
                self,
                f"Export {name} ({len(slices)} slices) as .pio",
            )
            if not directory:
                return
            produced = 0
            for stem, el_idx, pol_idx in slices:
                dataset.save_pio(
                    os.path.join(directory, f"{stem}.pio"),
                    el_idx=el_idx,
                    pol_idx=pol_idx,
                )
                produced += 1
            self.status.showMessage(
                f"Exported {produced} .pio file(s) to {directory}."
            )
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Export Selected Datasets as .pio"
        )
        if not directory:
            return
        produced = 0
        for name, dataset in datasets:
            for stem, el_idx, pol_idx in self._iter_pio_slices(dataset, name):
                dataset.save_pio(
                    os.path.join(directory, f"{stem}.pio"),
                    el_idx=el_idx,
                    pol_idx=pol_idx,
                )
                produced += 1
        self.status.showMessage(f"Exported {produced} .pio file(s) to {directory}.")

    @staticmethod
    def _write_ptm_batch(directory: str, plans) -> int:
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

        # Validate/write every slice into a sibling staging folder first, so a
        # format/validation failure publishes nothing. Final filesystem moves
        # are necessarily sequential; a rare move failure can leave a partial
        # published set and is reported to the user.
        with tempfile.TemporaryDirectory(prefix=".grim_ptm_", dir=directory) as stage:
            staged = []
            for dataset, stem, el_idx, pol_idx, target in prepared:
                stage_path = os.path.join(stage, f"{stem}.ptm")
                saved = dataset.save_ptm(
                    stage_path, el_idx=el_idx, pol_idx=pol_idx
                )
                staged.append((saved, target))
            for stage_path, target in staged:
                os.replace(stage_path, target)
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
                    saved = dataset.save_ptm(
                        path, el_idx=el_idx, pol_idx=pol_idx
                    )
                    self.status.showMessage(
                        f"Exported {os.path.basename(saved)}."
                    )
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
                    for name, dataset in datasets
                    for stem, el_idx, pol_idx in self._iter_pio_slices(dataset, name)
                ]

            produced = self._write_ptm_batch(directory, plans)
            self.status.showMessage(
                f"Exported {produced} .ptm file(s) to {directory}."
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
        produced = 0
        for name, dataset in datasets:
            safe_name = _sanitize_filename(name)
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Export {name}",
                f"{safe_name}.csv",
                "CSV Files (*.csv);;All Files (*)",
            )
            if not path:
                continue
            if not path.lower().endswith(".csv"):
                path = f"{path}.csv"
            _write_dataset_csv(dataset, path, scale=scale, sep=",", include_phase=include_phase)
            produced += 1

        if produced:
            self.status.showMessage(f"Exported {produced} dataset(s) to CSV.")
        else:
            self.status.showMessage("Export cancelled.")

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

        try:
            ds_a._assert_compatible(ds_b, coherent=True)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(f"Coherent ÷: {exc}")
            return

        denom = ds_b.rcs.copy()
        numerator = ds_a.rcs
        result_rcs = np.full(
            numerator.shape, np.nan + 1j * np.nan,
            dtype=np.result_type(numerator.dtype, denom.dtype),
        )
        valid = np.isfinite(numerator) & np.isfinite(denom) & (denom != 0)
        np.divide(numerator, denom, out=result_rcs, where=valid)
        ratio_units = dict(ds_a.units or {})
        ratio_units["rcs_log_unit"] = "dB"
        ratio_units["rcs_linear_quantity"] = "power_ratio"
        result = RcsGrid(
            ds_a.azimuths, ds_a.elevations, ds_a.frequencies,
            ds_a.polarizations, result_rcs,
            rcs_power=np.abs(result_rcs) ** 2,
            rcs_domain="complex_amplitude",
            units=ratio_units,
        )
        out_name = f"{name_a} ÷ {name_b}"
        output_id = self._add_dataset_row(
            result, out_name, f"Coherent ÷: {name_a} / {name_b}", file_name=""
        )
        input_refs = self._python_input_references(datasets)
        recorder = getattr(self, "python_recorder", None)
        if recorder is not None and input_refs is not None:
            recorder.record_function(
                self._python_output_reference(output_id, out_name),
                "coherent_divide",
                input_refs,
                comment=f"Coherently divide {name_a} by {name_b}",
            )
        self.status.showMessage(f"Coherent ÷ produced: {out_name}")
