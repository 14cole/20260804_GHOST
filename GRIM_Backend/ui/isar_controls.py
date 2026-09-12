"""ISAR control synchronization without touching the numerical worker."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QToolButton, QPlainTextEdit, QSizePolicy,
)


def sync_frequency_controls(context, dataset):
    """Display native frequency values with explicit units and acquired bounds."""
    if context is None or dataset is None:
        return
    values = np.asarray(dataset.frequencies, dtype=float)
    if not values.size or not np.all(np.isfinite(values)):
        return
    unit = str((dataset.units or {}).get("frequency", "")).strip()
    lo, hi = float(values.min()), float(values.max())
    key = (id(dataset), unit, lo, hi)
    if getattr(context, "_isar_frequency_bounds", None) == key:
        return
    context._isar_frequency_bounds = key
    for spin, value in ((context.spin_isar_freq_min, lo), (context.spin_isar_freq_max, hi)):
        old = spin.blockSignals(True)
        try:
            spin.setDecimals(9)
            spin.setRange(lo, hi)
            spin.setSuffix(f" {unit}" if unit else "")
            spin.setSingleStep(max((hi - lo) / 100.0, 1e-9))
            spin.setValue(value)
            spin.setToolTip(f"Frequency limit in the dataset's {unit or 'declared'} units.")
        finally:
            spin.blockSignals(old)


def sync_reconstruction_controls(context):
    sparse = context.combo_isar_recon.currentText().lower().startswith("sparse")
    context.spin_isar_l1_strength.setEnabled(sparse)
    context.spin_isar_l1_iters.setEnabled(sparse)
    context.combo_isar_window.setEnabled(not sparse)
    advanced = getattr(context, 'isar_advanced', None)
    if advanced is not None:
        advanced.native.setEnabled(sparse)


class AdvancedIsarControls(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        for label, value in (("Automatic aperture policy", "auto"), ("Single coherent image", "coherent"),
                             ("Qualitative max-look composite", "composite")):
            self.mode.addItem(label, value)
        self.mode.setToolTip("Automatic retains the 20° composite switch. A coherent image requires <90° and should be checked against the requested scene.")
        layout.addRow("Image mode", self.mode)
        self.scene = QCheckBox("Use scene bounds and crop result")
        layout.addRow(self.scene)
        self.x_half, self.y_half = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.x_half, self.y_half):
            spin.setRange(.000001, 100000.)
            spin.setDecimals(6)
            spin.setSuffix(" m")
            spin.setValue(1.)
            spin.setEnabled(False)
            spin.setToolTip("Occupied scene half extent about the fixed phase origin. Cropping changes retained image size, not the acquired resolution.")
        layout.addRow("Cross-range half extent", self.x_half)
        layout.addRow("Range half extent", self.y_half)
        self.side = QSpinBox()
        self.side.setRange(32, 4096)
        self.side.setSingleStep(256)
        self.side.setValue(1024)
        self.side.setToolTip("Composite grid pixels per side. More pixels do not improve physical resolution.")
        layout.addRow("Composite pixels per side", self.side)
        self.native = QCheckBox("Check sparse image against acquired polar samples")
        self.native.setChecked(True)
        self.native.setToolTip("Bounded direct point prediction. Reports selected-sample count and any omitted image support.")
        layout.addRow(self.native)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.scene.toggled.connect(self._scene_changed)
        for spin in (self.x_half, self.y_half):
            spin.valueChanged.connect(lambda _=None: self.changed.emit() if self.scene.isChecked() else None)
        self.side.valueChanged.connect(lambda: self.changed.emit() if self.side.isEnabled() else None)
        self.native.toggled.connect(lambda: self.changed.emit() if self.native.isEnabled() else None)

    def _mode_changed(self):
        self.side.setEnabled(self.mode.currentData() != 'coherent')
        self.changed.emit()

    def _scene_changed(self, enabled):
        self.x_half.setEnabled(enabled)
        self.y_half.setEnabled(enabled)
        self.changed.emit()

    def options(self):
        return {"aperture_mode": self.mode.currentData(),
                "scene_half_extent_m": (self.x_half.value(), self.y_half.value()) if self.scene.isChecked() else None,
                "composite_side": self.side.value(), "native_diagnostics": self.native.isChecked()}


class _QualityPanel(QPlainTextEdit):
    def minimumSizeHint(self):
        return QSize(0, 0)

    def sizeHint(self):
        return QSize(400, 110)


class IsarTools(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        for index, (name, label) in enumerate((("plan", "Plan image"), ("cancel", "Cancel"), ("open", "Open result"),
                            ("compare", "Compare result"), ("save_recipe", "Save recipe"),
                            ("load_recipe", "Load recipe"), ("guide", "Workflow"), ("quality", "Quality"))):
            if index == 4:
                bar.addStretch(1)
                bar = QHBoxLayout()
                layout.addLayout(bar)
            button = QToolButton(text=label)
            setattr(self, name, button)
            bar.addWidget(button)
        bar.addStretch(1)
        self.cancel.setEnabled(False)
        self.quality.setCheckable(True)
        self.quality.setChecked(True)
        self.details = _QualityPanel()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(150)
        self.details.setMinimumHeight(64)
        self.details.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.details.setPlainText("Select the acquisition band and angular sector, then use Plan image to review sampling, focus, and memory before formation.")
        self.quality.toggled.connect(self.details.setVisible)
        layout.addWidget(self.details)

    def show_quality(self, text):
        self.details.setPlainText(text)


def quality_text(bands, unit="m"):
    lines = []
    for i, band in enumerate(bands, 1):
        plan = band.get("accuracy_plan", {})
        lines.append(f"Image {i}: {plan.get('image_kind', 'saved result')}; {band.get('resolved_reconstruction', '')}")
        if plan:
            lines.append(f"Aperture {plan['aperture_degrees']:.4g}°; frequencies {plan['frequency_min_hz']/1e9:.6g}–{plan['frequency_max_hz']/1e9:.6g} GHz; "
                f"horizontal plane, elevation {plan['elevation_degrees']:.4g}°. "
                f"Fast curvature {plan['fast_range_curvature_edge_rad']:.3g} rad; "
                f"native phase steps az/f {plan['max_native_azimuth_phase_step_rad']:.3g}/{plan['max_native_frequency_phase_step_rad']:.3g} rad.")
            lines.extend(plan.get("warnings", []))
        if "phase_coverage" in band:
            lines.append(f"Measured gridded coverage {band['phase_coverage']:.1%}; gaps az/f {band.get('az_gap_count', 0)}/{band.get('freq_gap_count', 0)}; "
                f"largest sample spacing {band.get('az_largest_gap', 0):.4g}° / {band.get('freq_largest_gap', 0):.4g} Hz.")
        psf = band.get("psf", {})
        lines.append(psf.get("definition", "No PSF diagnostics in this artifact."))
        for key, label in (("cross_range", "Cross-range"), ("range", "Range")):
            p = psf.get(key, {})
            if p.get("status") == "computed":
                width = p.get('power_fwhm')
                width_text = f"{width:.4g} {unit}" if width is not None else "unresolved"
                lines.append(f"{label} PSF: power FWHM {width_text}; PSLR {p['pslr_db']:.2f} dB; cut ISLR {p['islr_db']:.2f} dB.")
        if "sampling" in band and band['sampling']:
            s = band['sampling']
            if 'cross_resolution' in s:
                lines.append(f"Nominal resolution cross/range {s['cross_resolution']:.4g}/{s['range_resolution']:.4g} {unit}; pixel spacing is not resolution.")
        n = band.get("native_residual", {})
        if n.get("status") == "computed":
            lines.append(f"Native polar residual {n['relative_complex_l2_residual']:.2%} on {n['sample_count']}/{n['source_sample_count']} source positions "
                f"({n['support_pixels']} image points; omitted image energy {n['omitted_image_energy_fraction']:.3g}).")
            if n.get('warning'):
                lines.append(n['warning'])
            if n.get('scope'):
                lines.append('Residual scope: ' + n['scope'] + '.')
        elif n.get('reason') and 'sparse' in str(band.get('resolved_reconstruction', '')):
            lines.append("Native residual unavailable: " + n['reason'])
        if 'sparse_converged' in band:
            lines.append(f"Gridded LASSO converged: {band['sparse_converged']}; output residual {band.get('sparse_output_relative_residual_norm', 0):.2%}.")
        looks = band.get('composite_sublooks', [])
        if looks:
            lines.append(f"{len(looks)} sublooks; realized widths {min(p['span_degrees'] for p in looks):.3g}–{max(p['span_degrees'] for p in looks):.3g}°; maximum magnitude aggregation.")
        memory = band.get('memory_budget', {})
        if memory:
            lines.append(f"Memory: source power/phase {memory.get('source_power_phase_bytes', 0)/1024**2:.1f} MiB, "
                f"retained prior results {memory.get('retained_previous_results_bytes', 0)/1024**2:.1f} MiB, "
                f"preparation cache {memory.get('preparation_cache_bytes', 0)/1024**2:.1f} MiB, "
                f"geometry cache {memory.get('geometry_plan_cache_bytes', 0)/1024**2:.1f} MiB. Formation budget is additional working memory.")
        assumptions = band.get('isar_contract_undeclared_fields', [])
        if assumptions:
            lines.append("Assumed undeclared conventions: " + "; ".join(assumptions))
        lines.append("")
    return "\n".join(lines).strip()
