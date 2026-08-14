from __future__ import annotations

import csv
import os
import re
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
    QPushButton,
    QRadioButton,
    QTableWidgetItem,
    QVBoxLayout,
)

from grim_dataset import C0, RcsGrid

# Characters forbidden in filenames on Windows (and `/` on POSIX). Replaced
# with `_` so dataset names with op symbols like `|`, `÷`, etc. still save.
_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


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


def _conic_to_gc_deg(phi_deg: np.ndarray, theta_deg: np.ndarray):
    """Forward map: spherical (φ, θ) → great-circle (α, ψ) angles in degrees.

    Convention: r̂_conic = (cos θ cos φ, cos θ sin φ, sin θ)
                r̂_gc   = (cos α, cos ψ sin α, sin ψ sin α)
    So α = arccos(cos θ cos φ) ∈ [0°, 180°],
       ψ = atan2(sin θ, cos θ sin φ) ∈ (-180°, 180°].
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    # Clip for numerical safety before arccos.
    x = np.clip(ct * cp, -1.0, 1.0)
    alpha = np.arccos(x)
    psi = np.arctan2(st, ct * sp)
    return np.rad2deg(alpha), np.rad2deg(psi)


def _gc_to_conic_deg(alpha_deg: np.ndarray, psi_deg: np.ndarray):
    """Inverse map: great-circle (α, ψ) → spherical (φ, θ) angles in degrees."""
    alpha = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    psi = np.deg2rad(np.asarray(psi_deg, dtype=float))
    sa, ca = np.sin(alpha), np.cos(alpha)
    sp, cp = np.sin(psi), np.cos(psi)
    z = np.clip(sp * sa, -1.0, 1.0)
    theta = np.arcsin(z)
    phi = np.arctan2(cp * sa, ca)
    return np.rad2deg(phi), np.rad2deg(theta)


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
    """Pick direction (conic↔great-circle) and mode (relabel / re-grid). Output
    grid bounds and sample counts are derived from the input dataset.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert Conic ↔ Great-Circle")
        layout = QVBoxLayout(self)

        dir_group = QGroupBox("Direction")
        dir_layout = QVBoxLayout(dir_group)
        self._radio_c2g = QRadioButton("Conic → Great-Circle (input axes are φ, θ; output α, ψ)")
        self._radio_g2c = QRadioButton("Great-Circle → Conic (input axes are α, ψ; output φ, θ)")
        self._radio_c2g.setChecked(True)
        dir_layout.addWidget(self._radio_c2g)
        dir_layout.addWidget(self._radio_g2c)
        layout.addWidget(dir_group)

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        self._radio_relabel = QRadioButton(
            "Relabel (flatten to 1D scatter — preserves σ exactly, loses grid structure)"
        )
        self._radio_regrid = QRadioButton(
            "Re-grid (bilinear interpolation onto a uniform output grid, bounds auto-derived)"
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
            "direction": "conic_to_gc" if self._radio_c2g.isChecked() else "gc_to_conic",
            "mode": "relabel" if self._radio_relabel.isChecked() else "regrid",
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
    raise ValueError(
        f"unsupported RCS log unit {value!r}; use dBsm or dBke"
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
    if scale not in {"linear", "dbsm", "dbke", "both"}:
        raise ValueError(
            "CSV magnitude scale must be linear, dbsm, dbke, or both"
        )
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
    header = [
        "azimuth",
        "elevation",
        "frequency",
        "frequency_unit",
        "polarization",
        "rcs_log_unit",
    ]
    if scale in ("linear", "both"):
        header.append("magnitude_linear")
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
                        ]
                        if scale in ("linear", "both"):
                            row.append(_csv_number(mag, ".10g"))
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
        has_dbsm = "magnitude_dbsm" in field_map
        has_dbke = "magnitude_dbke" in field_map
        if not has_linear and not has_dbsm and not has_dbke:
            raise ValueError("missing magnitude column (need magnitude_linear and/or magnitude_dbsm and/or magnitude_dbke)")
        has_phase = "phase_deg" in field_map
        has_frequency_unit = "frequency_unit" in field_map
        has_rcs_log_unit = "rcs_log_unit" in field_map

        def _cell(row: dict[str, str], key: str) -> str:
            source = field_map[key]
            raw = row.get(source, "")
            return str(raw).strip() if raw is not None else ""

        records: list[
            tuple[float, float, float, str, float | None, float | None, float]
        ] = []
        frequency_units_seen: set[str] = set()
        rcs_log_units_seen: set[str] = set()
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
        else "dBke" if has_dbke and not has_dbsm else "dBsm"
    )

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

    return RcsGrid(
        az_values,
        el_values,
        fr_values,
        pol_values,
        rcs_power=power,
        rcs_phase=phase,
        rcs_domain="power_phase",
        source_path=path,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": frequency_unit,
            "rcs_log_unit": rcs_log_unit,
        },
    )


def _load_dataset_from_dropped_text(path: str) -> tuple["RcsGrid", str]:
    """Load dropped delimited files, including theta/phi text variants."""
    lower = path.lower()
    attempts = []
    if lower.endswith(".out"):
        attempts = [
            ("OUT", lambda: RcsGrid.load_out(path)),
        ]
    elif lower.endswith(".txt"):
        attempts = [
            ("theta/phi TXT", lambda: RcsGrid.load_theta_phi_txt(path)),
            ("delimited table", lambda: _load_dataset_csv(path)),
        ]
    elif lower.endswith(".csv"):
        attempts = [
            ("delimited table", lambda: _load_dataset_csv(path)),
            ("theta/phi CSV", lambda: RcsGrid.load_theta_phi_csv(path)),
        ]
    elif lower.endswith(".pio") or lower.endswith(".cmplx_di"):
        attempts = [
            ("Pioneer", lambda: RcsGrid.load_pio(path)),
        ]
    elif lower.endswith(".ss"):
        attempts = [
            ("Xpatch SS", lambda: RcsGrid.load_ss(path)),
        ]
    else:
        attempts = [("delimited table", lambda: _load_dataset_csv(path))]

    errors: list[str] = []
    for label, loader in attempts:
        try:
            dataset = loader()
            history = str(getattr(dataset, "history", "") or "").strip()
            if not history:
                history = f"Imported delimited text: {path}"
            return dataset, history
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise ValueError("; ".join(errors))


def _is_supported_dataset_path(path: str) -> bool:
    lower = str(path).lower()
    return (
        lower.endswith(".grim")
        or lower.endswith(".csv")
        or lower.endswith(".txt")
        or lower.endswith(".out")
        or lower.endswith(".pio")
        or lower.endswith(".cmplx_di")
        or lower.endswith(".ss")
    )


def _recommended_loader_workers(task_count: int) -> int:
    cpu_total = os.cpu_count() or 1
    if cpu_total <= 2:
        target = cpu_total
    else:
        target = cpu_total - 1
    return max(1, min(int(task_count), int(target)))


def _load_dataset_path_task(task: tuple[int, str]) -> dict[str, object]:
    index, path = task
    file_name = os.path.basename(path)
    dataset_name = os.path.splitext(file_name)[0]
    lower = path.lower()
    try:
        if lower.endswith(".grim"):
            dataset = RcsGrid.load(path)
            history = path
        elif (
            lower.endswith(".csv")
            or lower.endswith(".txt")
            or lower.endswith(".out")
            or lower.endswith(".pio")
            or lower.endswith(".cmplx_di")
            or lower.endswith(".ss")
        ):
            dataset, history = _load_dataset_from_dropped_text(path)
        else:
            return {
                "status": "ignored",
                "index": index,
                "path": path,
                "file_name": file_name,
                "error": "Unsupported file extension",
            }
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
    if total == 1:
        grid = checked[0]
        if progress_cb is not None:
            progress_cb(1, 1)
        return grid._new_grid(
            np.array(grid.azimuths, copy=True),
            np.array(grid.elevations, copy=True),
            np.array(grid.frequencies, copy=True),
            np.array(grid.polarizations, copy=True),
            rcs_power=np.array(grid.rcs_power, copy=True),
            rcs_phase=np.array(grid.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    az_union = RcsGrid._axis_union([grid.azimuths for grid in checked], tol=tol)
    el_union = RcsGrid._axis_union([grid.elevations for grid in checked], tol=tol)
    f_union = RcsGrid._axis_union([grid.frequencies for grid in checked], tol=tol)
    p_union = RcsGrid._axis_union([grid.polarizations for grid in checked], tol=0.0)

    shape = (len(az_union), len(el_union), len(f_union), len(p_union))
    joined_power = np.full(shape, np.nan, dtype=np.float32)
    joined_phase = np.full(shape, np.nan, dtype=np.float32)

    for idx, grid in enumerate(checked, start=1):
        az_idx = RcsGrid._indices_for_axis_values(az_union, grid.azimuths, tol=tol)
        el_idx = RcsGrid._indices_for_axis_values(el_union, grid.elevations, tol=tol)
        f_idx = RcsGrid._indices_for_axis_values(f_union, grid.frequencies, tol=tol)
        p_idx = RcsGrid._indices_for_axis_values(p_union, grid.polarizations, tol=0.0)
        if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
            raise ValueError("failed to align a dataset during join")
        joined_power[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs_power
        joined_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs_phase
        if progress_cb is not None:
            progress_cb(idx, total)

    last = checked[-1]
    return RcsGrid(
        az_union,
        el_union,
        f_union,
        p_union,
        rcs_power=joined_power,
        rcs_phase=joined_phase,
        rcs_domain="power_phase",
        source_path=last.source_path,
        history=last.history,
        units=dict(last.units),
    )


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

        if total == 0:
            self.finished.emit(
                {
                    "loaded": loaded,
                    "failed": failed,
                    "ignored": self._ignored_count,
                    "used_parallel": used_parallel,
                    "total_supported": total,
                }
            )
            return

        if total == 1:
            _consume(_load_dataset_path_task(self._tasks[0]), 1)
        else:
            worker_count = _recommended_loader_workers(total)
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


class DatasetOpsMixin:
    def _ensure_background_worker_state(self) -> None:
        if hasattr(self, "_background_worker_thread"):
            return
        self._background_worker_thread: QThread | None = None
        self._background_worker: QObject | None = None
        self._background_worker_name = ""
        self._pending_join_names: list[str] | None = None

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
        self._background_worker_thread = None
        self._background_worker = None
        self._background_worker_name = ""

    def _on_load_worker_progress(self, done_count: int, total_count: int, detail: str) -> None:
        detail_text = str(detail).strip()
        if detail_text:
            self.status.showMessage(
                f"Loading datasets... {done_count}/{total_count} ({detail_text})"
            )
            return
        self.status.showMessage(f"Loading datasets... {done_count}/{total_count}")

    def _on_load_worker_finished(self, summary: dict[str, object]) -> None:
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
            self._add_dataset_row(dataset, name, history, file_name=file_name)
            loaded += 1

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
        self.status.showMessage(msg)

    def _on_join_worker_progress(self, done_count: int, total_count: int, _: str) -> None:
        self.status.showMessage(f"Joining datasets... {done_count}/{total_count}")

    def _on_join_worker_finished(self, payload: dict[str, object]) -> None:
        names = self._pending_join_names or []
        self._pending_join_names = None

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
        self._add_dataset_row(merged, f"Join[{new_name}]", history, file_name="")
        self.status.showMessage(f"Join created. Overlap winner: {names[-1]}.")

    def _handle_files_dropped(self, paths: list[str]) -> None:
        tasks: list[tuple[int, str]] = []
        ignored = 0
        for index, raw_path in enumerate(paths):
            path = str(raw_path)
            if _is_supported_dataset_path(path):
                tasks.append((index, path))
            else:
                ignored += 1

        if not tasks:
            if ignored:
                self.status.showMessage(
                    "No supported dropped files. Supported: .grim, .csv, .txt, .out, .pio, .cmplx_di, .ss"
                )
            return

        worker = _DatasetLoadWorker(tasks, ignored_count=ignored)
        worker.progress.connect(self._on_load_worker_progress)
        worker.finished.connect(self._on_load_worker_finished)
        if not self._try_start_background_job("Dataset loading", worker):
            return
        self.status.showMessage(f"Loading datasets... 0/{len(tasks)}")

    def _add_dataset_row(self, dataset: RcsGrid, name: str, history: str, file_name: str | None = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.UserRole, dataset)
        file_text = file_name or ""
        file_item = QTableWidgetItem(file_text)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        history_item = QTableWidgetItem(history)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, file_item)
        self.table.setItem(row, 2, history_item)

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

    def _populate_params(self, dataset: RcsGrid) -> None:
        self._fill_list(self.list_pol, dataset.polarizations)
        self._fill_list(self.list_freq, dataset.frequencies)
        self._fill_list(self.list_elev, dataset.elevations)
        self._fill_list(self.list_az, dataset.azimuths)
        self._apply_default_param_selection()

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

    def _on_param_item_changed(self, item: QListWidgetItem, axis_name: str, widget: QListWidget) -> None:
        if self.active_dataset is None:
            return
        axis_arr = self.active_dataset.get_axis(axis_name)
        idx = item.data(Qt.UserRole + 1)
        if idx is None:
            return
        if idx < 0 or idx >= len(axis_arr):
            return
        old_value = axis_arr[idx]
        new_text = item.text()
        if axis_name == "polarization":
            new_value = new_text
        else:
            try:
                new_value = float(new_text)
            except ValueError:
                widget.blockSignals(True)
                item.setText(str(old_value))
                widget.blockSignals(False)
                return
        axis_arr[idx] = new_value
        item.setData(Qt.UserRole, new_value)

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
        self._add_dataset_row(result, new_name, history, file_name="")
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
        self._add_dataset_row(result, new_name, history, file_name="")
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
            for (name, _), overlap_grid in zip(datasets, overlap_grids):
                history = f"Overlap with [{', '.join(names)}]: {name}"
                self._add_dataset_row(overlap_grid, f"{name} [Overlap]", history, file_name="")
                produced += 1
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        if produced == 0:
            self.status.showMessage("No overlap outputs were created.")
            return
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
            self._add_dataset_row(sliced, f"{name} [Slice]", history, file_name="")
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
            self._add_dataset_row(stat_grid, f"{name} [{stat_label}]", history, file_name="")
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
        for row in rows:
            self.table.removeRow(row)
        self.active_dataset = None
        self._clear_param_lists()
        self.status.showMessage(f"Deleted {len(rows)} dataset(s).")

    def _save_selected_datasets(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select one or more datasets to save.")
            return

        rows = sorted(idx.row() for idx in selected)

        if len(rows) == 1:
            # Single dataset — let the user pick the exact file path.
            row = rows[0]
            item = self.table.item(row, 0)
            if item is None:
                return
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                return
            name = item.text().strip() or "dataset"
            file_item = self.table.item(row, 1)
            prev_file = file_item.text() if file_item else ""
            prev_stem = os.path.splitext(prev_file)[0] if prev_file else ""
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Dataset",
                f"{_sanitize_filename(name)}.grim",
                "GRIM Files (*.grim)",
            )
            if not path:
                return
            saved_path = dataset.save(path)
            file_name = os.path.basename(saved_path)
            if file_item is None:
                file_item = QTableWidgetItem(file_name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, file_item)
            else:
                file_item.setText(file_name)
            history_item = self.table.item(row, 2)
            if history_item is None:
                history_item = QTableWidgetItem(saved_path)
                self.table.setItem(row, 2, history_item)
            else:
                history_item.setText(saved_path)
            new_stem = os.path.splitext(file_name)[0]
            if prev_stem and item.text().strip() == prev_stem:
                item.setText(new_stem)
            elif not item.text().strip():
                item.setText(new_stem)
            self.status.showMessage("Save completed.")
        else:
            # Multiple datasets — pick a folder once, save each using its table name.
            directory = QFileDialog.getExistingDirectory(self, "Save Selected Datasets")
            if not directory:
                return
            saved = 0
            for row in rows:
                item = self.table.item(row, 0)
                if item is None:
                    continue
                dataset = item.data(Qt.UserRole)
                if not isinstance(dataset, RcsGrid):
                    continue
                name = item.text().strip() or f"dataset_{row + 1}"
                path = os.path.join(directory, f"{_sanitize_filename(name)}.grim")
                saved_path = dataset.save(path)
                file_name = os.path.basename(saved_path)
                file_item = self.table.item(row, 1)
                if file_item is None:
                    file_item = QTableWidgetItem(file_name)
                    file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, 1, file_item)
                else:
                    file_item.setText(file_name)
                history_item = self.table.item(row, 2)
                if history_item is None:
                    history_item = QTableWidgetItem(saved_path)
                    self.table.setItem(row, 2, history_item)
                else:
                    history_item.setText(saved_path)
                saved += 1
            self.status.showMessage(f"Saved {saved} dataset(s) to {directory}.")

    def _save_all_datasets(self) -> None:
        if self.table.rowCount() == 0:
            self.status.showMessage("No datasets to save.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Save All Datasets")
        if not directory:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                continue
            name = item.text().strip() or f"dataset_{row + 1}"
            filename = f"{_sanitize_filename(name)}.grim"
            path = os.path.join(directory, filename)
            saved_path = dataset.save(path)
            file_name = os.path.basename(saved_path)
            file_item = self.table.item(row, 1)
            if file_item is None:
                file_item = QTableWidgetItem(file_name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, file_item)
            else:
                file_item.setText(file_name)
            history_item = self.table.item(row, 2)
            if history_item is None:
                history_item = QTableWidgetItem(saved_path)
                self.table.setItem(row, 2, history_item)
            else:
                history_item.setText(saved_path)
        self.status.showMessage("Save all completed.")

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
            self._add_dataset_row(aligned, f"{name} [Aligned]", history, file_name="")
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
            self._add_dataset_row(
                interpolated, f"{name} [Interp]", history, file_name=""
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
            self._add_dataset_row(
                mirrored,
                f"{name} [Mirror {about:.6g}°]",
                history,
                file_name="",
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
            self._add_dataset_row(
                wrapped,
                f"{name} [Wrap {suffix}]",
                history,
                file_name="",
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
            self._add_dataset_row(
                shifted,
                f"{name} [Shift {suffix}]",
                history,
                file_name="",
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
            self._add_dataset_row(
                rounded,
                f"{name} [Round {decimals}dp]",
                history,
                file_name="",
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
            self._add_dataset_row(
                swapped,
                f"{name} [Swap El/Az]",
                history,
                file_name="",
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
            self._add_dataset_row(
                result,
                f"{name} [El->Az360]",
                history,
                file_name="",
            )
            produced += 1

        if produced == 0:
            self.status.showMessage("El->Az360 created 0 datasets.")
            return
        msg = f"El->Az360 created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

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
            self._add_dataset_row(result, f"{name} [Offset {value:+.6g}]", history, file_name="")
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
                if dataset.rcs_domain == "complex_amplitude":
                    amp_scale_4d = np.sqrt(np.maximum(scale_4d, 0.0)).astype(np.complex64)
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
            self._add_dataset_row(
                result,
                f"{name} [→ dBke L={length_label}]",
                history,
                file_name="",
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
                if dataset.rcs_domain == "complex_amplitude":
                    amp_scale_4d = np.sqrt(np.maximum(scale_4d, 0.0)).astype(np.complex64)
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
            self._add_dataset_row(
                result,
                f"{name} [→ dBsm L={length_label}]",
                history,
                file_name="",
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

        dlg = ConicGCDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        params = dlg.get_params()
        direction = params["direction"]
        mode = params["mode"]

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
                    result, suffix, hist_extra = self._conic_gc_relabel(
                        dataset, direction
                    )
                else:
                    result, suffix, hist_extra = self._conic_gc_regrid(
                        dataset, direction
                    )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue

            arrow = "Conic→GC" if direction == "conic_to_gc" else "GC→Conic"
            history = f"{arrow} {mode}: {name}{hist_extra}"
            self._add_dataset_row(
                result,
                f"{name} [{arrow} {suffix}]",
                history,
                file_name="",
            )
            produced += 1

        if produced == 0:
            self.status.showMessage("Conic↔GC created 0 datasets.")
            return
        arrow = "Conic→GC" if direction == "conic_to_gc" else "GC→Conic"
        msg = f"{arrow} ({mode}) created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _conic_gc_relabel(self, dataset: "RcsGrid", direction: str):
        """Flatten the 2-D (az, el) grid into a 1-D scatter and replace the
        primary axis with the transformed coordinate. ψ (or θ) varies per
        sample and is logged into history rather than stored as an axis,
        because the result is by construction not on a rectangular grid.
        """
        az_in = np.asarray(dataset.azimuths, dtype=float)
        el_in = np.asarray(dataset.elevations, dtype=float)
        n_az, n_el = az_in.size, el_in.size
        # Build flat (φ, θ) or (α, ψ) pairs.
        phi_grid, theta_grid = np.meshgrid(az_in, el_in, indexing="ij")
        if direction == "conic_to_gc":
            new_pri, new_sec = _conic_to_gc_deg(phi_grid.ravel(), theta_grid.ravel())
            pri_label, sec_label = "α", "ψ"
        else:
            new_pri, new_sec = _gc_to_conic_deg(phi_grid.ravel(), theta_grid.ravel())
            pri_label, sec_label = "φ", "θ"

        order = np.argsort(new_pri, kind="stable")
        flat_power = dataset.rcs_power.reshape(n_az * n_el, dataset.frequencies.size, dataset.polarizations.size)
        flat_phase = dataset.rcs_phase.reshape(n_az * n_el, dataset.frequencies.size, dataset.polarizations.size)
        sorted_pri = new_pri[order]
        sorted_sec = new_sec[order]
        sorted_power = flat_power[order][:, None, :, :]
        sorted_phase = flat_phase[order][:, None, :, :]

        result = RcsGrid(
            sorted_pri,
            np.array([0.0]),
            dataset.frequencies,
            dataset.polarizations,
            rcs=None,
            rcs_power=sorted_power,
            rcs_phase=sorted_phase,
            rcs_domain=dataset.rcs_domain,
            units=dict(dataset.units or {}),
        )
        # Log the secondary coordinate trajectory (truncated for readability).
        sec_preview = ", ".join(f"{v:.3g}" for v in sorted_sec[: min(8, sorted_sec.size)])
        if sorted_sec.size > 8:
            sec_preview += f", … ({sorted_sec.size} total)"
        hist_extra = (
            f"; relabeled axis 0 to {pri_label} (sorted asc); "
            f"{sec_label} per sample = [{sec_preview}]; "
            f"{sec_label} ∈ [{sorted_sec.min():.3g}, {sorted_sec.max():.3g}]"
        )
        suffix = f"{pri_label}-scatter"
        return result, suffix, hist_extra

    def _conic_gc_regrid(
        self,
        dataset: "RcsGrid",
        direction: str,
    ):
        """Bilinearly interpolate the dataset onto a uniform output grid.

        Output bounds and sample counts are auto-derived: forward-map every
        input sample to compute the (pri, sec) hull, snap to natural domain
        edges when the input wraps the full sphere, and preserve the input's
        per-axis sample count.

        For Conic→GC: input axes are (φ, θ), output is (α, ψ). For each
        output cell we back-solve to the corresponding (φ, θ) via
        `_gc_to_conic_deg`, normalise into the input φ range (using periodic
        wrap when the input spans ≥359°), then call `scipy.interpolate.interpn`
        once on the multi-channel `rcs_power` and once on `rcs_phase` (nearest
        for phase — bilinear-interpolating a wrapped angle introduces fake
        ridges near ±π).
        """
        from scipy.interpolate import interpn

        az_in = np.asarray(dataset.azimuths, dtype=float)
        el_in = np.asarray(dataset.elevations, dtype=float)
        if np.any(np.diff(az_in) <= 0) or np.any(np.diff(el_in) <= 0):
            raise ValueError("input axes must be strictly increasing for re-grid")

        # Forward-map every input sample to determine the output hull, then
        # snap to natural domain edges for full-sphere inputs.
        in_az_mesh, in_el_mesh = np.meshgrid(az_in, el_in, indexing="ij")
        if direction == "conic_to_gc":
            fwd_pri, fwd_sec = _conic_to_gc_deg(in_az_mesh.ravel(), in_el_mesh.ravel())
            pri_label, sec_label = "α", "ψ"
        else:
            fwd_pri, fwd_sec = _gc_to_conic_deg(in_az_mesh.ravel(), in_el_mesh.ravel())
            pri_label, sec_label = "φ", "θ"

        az_span = float(az_in.max() - az_in.min())
        el_span = float(el_in.max() - el_in.min())
        full_sphere = az_span >= 359.0 and el_span >= 179.0
        if full_sphere and direction == "conic_to_gc":
            pri_lo, pri_hi = 0.0, 180.0
            sec_lo, sec_hi = -180.0, 180.0
        elif full_sphere and direction == "gc_to_conic":
            pri_lo, pri_hi = -180.0, 180.0
            sec_lo, sec_hi = -90.0, 90.0
        else:
            pri_lo, pri_hi = float(fwd_pri.min()), float(fwd_pri.max())
            sec_lo, sec_hi = float(fwd_sec.min()), float(fwd_sec.max())

        n_pri = max(int(az_in.size), 2)
        n_sec = max(int(el_in.size), 2)
        pri_grid = np.linspace(pri_lo, pri_hi, n_pri, dtype=float)
        sec_grid = np.linspace(sec_lo, sec_hi, n_sec, dtype=float)

        # Build the output (pri, sec) mesh and back-solve to (φ_query, θ_query).
        pri_mesh, sec_mesh = np.meshgrid(pri_grid, sec_grid, indexing="ij")
        if direction == "conic_to_gc":
            phi_q, theta_q = _gc_to_conic_deg(pri_mesh.ravel(), sec_mesh.ravel())
        else:
            # GC→Conic: the input axes carry (α, ψ); the output (pri, sec) =
            # (φ, θ); we back-solve via _conic_to_gc_deg.
            phi_q, theta_q = _conic_to_gc_deg(pri_mesh.ravel(), sec_mesh.ravel())

        # Wrap φ_query into the input range when the input is periodic in φ.
        if az_span >= 359.0:
            phi_q = ((phi_q - az_in[0]) % 360.0) + az_in[0]
        if el_span >= 359.0:
            theta_q = ((theta_q - el_in[0]) % 360.0) + el_in[0]

        query = np.column_stack([phi_q, theta_q])

        power_out = interpn(
            (az_in, el_in),
            dataset.rcs_power,
            query,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        phase_out = interpn(
            (az_in, el_in),
            dataset.rcs_phase,
            query,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        new_shape = (
            pri_grid.size,
            sec_grid.size,
            dataset.frequencies.size,
            dataset.polarizations.size,
        )
        power_out = power_out.reshape(new_shape).astype(np.float32)
        phase_out = phase_out.reshape(new_shape).astype(np.float32)

        result = RcsGrid(
            pri_grid,
            sec_grid,
            dataset.frequencies,
            dataset.polarizations,
            rcs=None,
            rcs_power=power_out,
            rcs_phase=phase_out,
            rcs_domain=dataset.rcs_domain,
            units=dict(dataset.units or {}),
        )
        in_bounds = np.sum(np.isfinite(power_out[..., 0, 0]))
        total = pri_grid.size * sec_grid.size
        coverage = 100.0 * in_bounds / max(total, 1)
        hist_extra = (
            f"; output axes {pri_label}=[{pri_grid[0]:g}..{pri_grid[-1]:g}/{pri_grid.size}], "
            f"{sec_label}=[{sec_grid[0]:g}..{sec_grid[-1]:g}/{sec_grid.size}]; "
            f"coverage {coverage:.1f}%"
        )
        suffix = f"{pri_label}×{sec_label}"
        return result, suffix, hist_extra

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
            self._add_dataset_row(
                result,
                f"{name} [Wedge→Conic {suffix}]",
                history,
                file_name="",
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
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

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
        flat_power = dataset.rcs_power.reshape(n_az * n_el, n_f * n_pol)
        flat_phase = dataset.rcs_phase.reshape(n_az * n_el, n_f * n_pol)

        power_interp = LinearNDInterpolator(points, flat_power, fill_value=np.nan)
        phase_interp = NearestNDInterpolator(points, flat_phase)
        power_out = power_interp(query)
        phase_out = phase_interp(query)
        # NearestNDInterpolator has no fill_value option — mask cells outside
        # the convex hull (where the linear interp returned NaN) so phase and
        # power agree about which cells are valid.
        outside = ~np.isfinite(power_out)
        phase_out = np.where(outside, np.nan, phase_out)

        new_shape = (n_lon, n_lat, n_f, n_pol)
        power_out = power_out.reshape(new_shape).astype(np.float32)
        phase_out = phase_out.reshape(new_shape).astype(np.float32)

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
                # keep the phase of the input sample nearest the centre — a
                # circular-statistic median isn't well-defined, and "nearest
                # to centre" gives a phase that is at least locally
                # consistent. Power-domain median doesn't change under the
                # log → result is identical to a median in dB.
                n_el = dataset.elevations.size
                n_f = dataset.frequencies.size
                n_pol = dataset.polarizations.size
                new_power = np.empty(
                    (centres.size, n_el, n_f, n_pol), dtype=np.float32
                )
                new_phase = np.empty_like(new_power)
                for i, c in enumerate(centres):
                    in_window = np.where(np.abs(az - c) <= half_w)[0]
                    if in_window.size == 0:
                        # Fall back to nearest single sample.
                        in_window = np.array([int(np.argmin(np.abs(az - c)))])
                    window_power = dataset.rcs_power[in_window, :, :, :]
                    new_power[i] = np.nanmedian(window_power, axis=0)
                    centre_idx = int(in_window[np.argmin(np.abs(az[in_window] - c))])
                    new_phase[i] = dataset.rcs_phase[centre_idx, :, :, :]

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
            self._add_dataset_row(
                result,
                f"{name} [Median w={window_deg:g}° s={slide_deg:g}°]",
                history,
                file_name="",
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
                list(dataset.polarizations),
                dataset.rcs.copy(),
                rcs_power=dataset.rcs_power.copy(),
                rcs_domain=dataset.rcs_domain,
            )
            self._add_dataset_row(dup, f"{name} [Copy]", f"Duplicate of: {name}", file_name="")
        self.status.showMessage(f"Duplicated {len(datasets)} dataset(s).")

    def _iter_pio_slices(self, dataset: RcsGrid, base_name: str):
        """Yield (filename_stem, el_idx, pol_idx) for every (el, pol) slice.

        A .pio file holds a 2-D (azimuth, frequency) complex slice, so any grid
        with multiple elevations or polarizations must be split into one file
        per (el, pol) combination. The stem is suffixed only on the axes that
        actually have multiple values.
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
            ds_a._assert_compatible(ds_b)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(f"Coherent ÷: {exc}")
            return

        denom = ds_b.rcs.copy()
        denom[denom == 0] = 1e-30 + 0j
        result_rcs = ds_a.rcs / denom
        result = RcsGrid(
            ds_a.azimuths, ds_a.elevations, ds_a.frequencies,
            ds_a.polarizations, result_rcs,
            rcs_power=np.abs(result_rcs) ** 2,
            rcs_domain="complex_amplitude",
            units=ds_a.units,
        )
        out_name = f"{name_a} ÷ {name_b}"
        self._add_dataset_row(result, out_name, f"Coherent ÷: {name_a} / {name_b}", file_name="")
        self.status.showMessage(f"Coherent ÷ produced: {out_name}")
