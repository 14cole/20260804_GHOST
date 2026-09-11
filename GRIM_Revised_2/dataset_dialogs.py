"""Dataset operation dialogs and their input-unit presentation helpers."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QRadioButton, QSpinBox, QVBoxLayout
from grim_backend.datasets.constants import GRIM_GC_CONVENTION, LEGACY_PTM_GC_CONVENTION
from grim_backend.datasets.grid import RcsGrid
from grim_backend.datasets.coordinates import canonical_angular_coordinate_system


_FREQUENCY_TO_HZ = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}


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

def _canonical_frequency_unit(value: object, *, default: str = "GHz") -> str:
    text = str(value or default).strip().lower()
    if text not in _FREQUENCY_TO_HZ:
        raise ValueError(
            f"unsupported frequency unit {value!r}; use Hz, kHz, MHz, or GHz"
        )
    return {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}[text]

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

        self.spin_offset_in = QDoubleSpinBox()
        self.spin_offset_in.setDecimals(12)
        self.spin_offset_in.setRange(-1.0e6 / 0.0254, 1.0e6 / 0.0254)
        self.spin_offset_in.setSingleStep(0.001)
        self.spin_offset_in.setSuffix(" in")
        self.spin_offset_in.setToolTip(
            "Enter the one-way physical displacement. Positive means the "
            "measured calibration target is farther from radar than the "
            "DUT/reference plane; GRIM applies the monostatic two-way phase."
        )
        form.addWidget(QLabel("Signed calibrator range offset ΔR:"), 2, 0)
        form.addWidget(self.spin_offset_in, 2, 1)

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
            "range_offset_m": float(self.spin_offset_in.value()) * 0.0254,
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
