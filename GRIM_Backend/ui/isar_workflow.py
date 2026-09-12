"""Numerical result viewing, compatible comparisons, and ISAR recipe workflow."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QToolButton, QFileDialog, QMessageBox, QPlainTextEdit,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from GRIM_Backend.isar.artifact import load_isar_artifact
from GRIM_Backend.isar.comparison import hydrate_band, compare_images, roi_statistics
from GRIM_Backend.isar.geometry import image_extent
from GRIM_Backend.isar import recipes
from GRIM_Backend.plotting.modes.isar_mode import _result_array_bytes, _decimate_display_max
from .isar_controls import quality_text

_VIEW_BYTES = 512 * 1024**2


class IsarResultDialog(QDialog):
    """View artifacts without replacing an active acquisition or its recipe."""
    def __init__(self, first, second=None, *, first_name='A', second_name='B', parent=None):
        super().__init__(parent)
        self.setWindowTitle('ISAR numerical results')
        self.resize(1100, 800)
        self.first, self.second = first, second
        self.names = (first_name, second_name)
        self.comparison = None
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.band_a, self.band_b = QComboBox(), QComboBox()
        for combo, payload, label in ((self.band_a, first, 'A'), (self.band_b, second, 'B')):
            if payload is not None:
                for i, info in enumerate(payload[0]['bands']):
                    az = info.get('azimuth_values_degrees', [])
                    span = f" {min(az):g}–{max(az):g}°" if az else ''
                    combo.addItem(f'{label} image {i+1}{span}', i)
            else:
                combo.setVisible(False)
            bar.addWidget(combo)
        self.resample = QCheckBox('Resample B intensity onto A overlap')
        self.resample.setVisible(second is not None)
        self.resample.setToolTip('Explicitly enables bilinear interpolation of linear intensity. Comparison uses only overlapping physical coordinates.')
        bar.addWidget(self.resample)
        self.roi_button = QToolButton(text='Statistics in current view')
        self.profiles = QToolButton(text='Peak profiles')
        self.profiles.setCheckable(True)
        bar.addWidget(self.roi_button)
        bar.addWidget(self.profiles)
        layout.addLayout(bar)
        self.figure = Figure(layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(160)
        layout.addWidget(self.details)
        self.band_a.currentIndexChanged.connect(self.render)
        self.band_b.currentIndexChanged.connect(self.render)
        self.resample.toggled.connect(self.render)
        self.profiles.toggled.connect(self.render)
        self.roi_button.clicked.connect(self.roi)
        self.render()

    def render(self):
        self.figure.clear()
        self.comparison = None
        ai = self.band_a.currentIndex()
        am, arrays = self.first
        a = hydrate_band(arrays[ai], am, ai)
        if self.second is None:
            ax = self.figure.add_subplot(111)
            self.axes = [ax]
            display = 20 * np.log10(np.maximum(_decimate_display_max(a['magnitude'], 1024), 1e-6))
            try:
                extent = image_extent(a)
            except ValueError as exc:
                self.details.setPlainText(f'Image cannot be displayed on a physical grid: {exc}. Numerical arrays remain available through load_isar_artifact.')
                self.canvas.draw_idle()
                return
            mesh = ax.imshow(display.T, extent=extent, origin='lower', aspect='equal', cmap='viridis')
            unit = am.get('formation', {}).get('length_unit', 'm')
            ax.set(xlabel=f'Cross-range ({unit})', ylabel=f'Range ({unit})', title=self.names[0])
            self.figure.colorbar(mesh, ax=ax, label='Image intensity (dB)')
            self.details.setPlainText(quality_text([a], unit))
            self.roi_button.setEnabled(False)
            self.profiles.setEnabled(False)
        else:
            bm, arrays_b = self.second
            bi = self.band_b.currentIndex()
            b = hydrate_band(arrays_b[bi], bm, bi)
            try:
                comparison = compare_images(am, a, bm, b, allow_resample=self.resample.isChecked(), maximum_cells=1_000_000)
            except (ValueError, KeyError, TypeError) as exc:
                self.details.setPlainText('Comparison unavailable: ' + str(exc))
                self.canvas.draw_idle()
                return
            self.comparison = comparison
            extent = image_extent(comparison)
            rows = 2 if self.profiles.isChecked() else 1
            axes = self.figure.subplots(rows, 3, squeeze=False)
            self.axes = list(axes[0])
            for ax in self.axes[1:]:
                ax.sharex(self.axes[0])
                ax.sharey(self.axes[0])
            lo = min(float(comparison['a_db'].min()), float(comparison['b_db'].min()))
            hi = max(float(comparison['a_db'].max()), float(comparison['b_db'].max()))
            delta_limit = max(float(abs(comparison['delta_db']).max()), 1e-6)
            for ax, key, label in zip(self.axes, ('a_db', 'b_db', 'delta_db'), (*self.names, 'A − B')):
                delta = key == 'delta_db'
                image = _decimate_display_max(comparison[key], 1024) if not delta else comparison[key]
                mesh = ax.imshow(image.T, extent=extent, origin='lower', aspect='equal',
                    cmap='RdBu_r' if delta else 'viridis', vmin=-delta_limit if delta else lo,
                    vmax=delta_limit if delta else hi)
                ax.set(title=label, xlabel='Cross-range (m)', ylabel='Range (m)')
                self.figure.colorbar(mesh, ax=ax, label='Δ intensity (dB)' if delta else 'Intensity (dB)')
            if rows == 2:
                ix, iy = np.unravel_index(np.argmax(comparison['a_db']), comparison['a_db'].shape)
                for key, label in (('a_db', 'A'), ('b_db', 'B')):
                    axes[1, 0].plot(comparison['x_range'], comparison[key][:, iy], label=label)
                    axes[1, 1].plot(comparison['y_range'], comparison[key][ix, :], label=label)
                axes[1, 0].set(xlabel='Cross-range (m)', ylabel='Intensity (dB)', title='Cut through A peak')
                axes[1, 1].set(xlabel='Range (m)', ylabel='Intensity (dB)', title='Cut through A peak')
                axes[1, 0].legend()
                axes[1, 1].legend()
                axes[1, 2].axis('off')
            self.details.setPlainText(f"A: {self.names[0]}\nB: {self.names[1]}\nResampling: {comparison['resampling']}. Floor: −120 dB.\n"
                + self._statistics_text(comparison['statistics']) + '\n' + comparison['quality_note']
                + '\n\n' + quality_text([a], am.get('formation', {}).get('length_unit', 'm'))
                + '\n\n' + quality_text([b], bm.get('formation', {}).get('length_unit', 'm')))
            self.roi_button.setEnabled(True)
        self.canvas.draw_idle()

    @staticmethod
    def _statistics_text(stats):
        return f"{stats['pixels']:,} pixels; mean A−B {stats['mean_delta_db']:.3f} dB; RMS {stats['rms_delta_db']:.3f} dB; " \
               f"min/max {stats['minimum_delta_db']:.3f}/{stats['maximum_delta_db']:.3f} dB."

    def roi(self):
        if self.comparison is None:
            return
        xlo, xhi = sorted(self.axes[-1].get_xlim())
        ylo, yhi = sorted(self.axes[-1].get_ylim())
        c = self.comparison
        ix = np.flatnonzero((c['x_range'] >= xlo) & (c['x_range'] <= xhi))
        iy = np.flatnonzero((c['y_range'] >= ylo) & (c['y_range'] <= yhi))
        try:
            stats = roi_statistics(c['delta_db'][np.ix_(ix, iy)])
            self.details.appendPlainText(f'\nROI in A−B view: x [{xlo:.4g}, {xhi:.4g}] m, y [{ylo:.4g}, {yhi:.4g}] m.\n' + self._statistics_text(stats))
        except ValueError as exc:
            self.details.appendPlainText(str(exc))


def open_result(window, *, compare=False):
    if getattr(window, '_isar_busy', False):
        window.status.showMessage('Finish or cancel ISAR formation before loading a numerical result.')
        return
    current = getattr(window, '_last_isar_artifact', None)
    if current is not None and not window._isar_numerical_result_is_current():
        current = None
    use_current = compare and current is not None
    if compare and not use_current:
        paths, _ = QFileDialog.getOpenFileNames(window, 'Choose two ISAR results', '', 'ISAR results (*.isar.npz)')
        if not paths:
            return
        if len(paths) != 2:
            window.status.showMessage('Choose exactly two numerical ISAR results to compare.')
            return
    else:
        path, _ = QFileDialog.getOpenFileName(window, 'Compare against result' if compare else 'Open ISAR result', '', 'ISAR results (*.isar.npz)')
        if not path:
            return
        paths = [path]
    # Include retained live arrays and an old viewer when budgeting additional loads.
    resident = _result_array_bytes(current[0]) if current is not None else 0
    old = getattr(window, '_isar_result_dialog', None)
    if old is not None:
        for payload in (old.first, old.second):
            if payload is not None:
                resident += _result_array_bytes(payload[1])

    def read():
        remaining = _VIEW_BYTES - resident - 128 * 1024**2
        loaded = []
        for path in paths:
            item = load_isar_artifact(path, maximum_working_bytes=remaining)
            loaded.append(item)
            remaining -= _result_array_bytes(item[1])
        return loaded

    def publish(loaded):
        first = (current[1], current[0]) if use_current else loaded[0]
        second = loaded[0] if use_current else (loaded[1] if len(loaded) == 2 else None)
        first_name = 'Current completed image' if use_current else Path(paths[0]).name
        second_name = Path(paths[-1]).name if second is not None else 'B'
        dialog = IsarResultDialog(first, second, first_name=first_name, second_name=second_name, parent=window)
        previous = getattr(window, '_isar_result_dialog', None)
        if previous is not None:
            previous.close()
            previous.deleteLater()
        window._isar_result_dialog = dialog
        dialog.show()
        window.status.showMessage('Numerical ISAR result opened with its saved acquisition and quality diagnostics.')
    window._start_background_callable('Open ISAR result', read, publish)


def save_recipe(window):
    recipe = getattr(window, '_last_isar_recipe', None)
    if recipe is None or not window._isar_numerical_result_is_current() or getattr(window, '_isar_busy', False):
        window.status.showMessage('Form the current ISAR image before saving its accepted recipe.')
        return
    path, _ = QFileDialog.getSaveFileName(window, 'Save ISAR recipe', 'image.isar.json', 'ISAR recipe (*.isar.json)')
    if not path:
        return
    try:
        recipes.save_recipe(path, recipe)
    except (OSError, ValueError) as exc:
        window.status.showMessage('Recipe save failed: ' + str(exc))
        return
    window.status.showMessage('Saved the completed image recipe with physical source selectors.')


def load_recipe(window):
    if window.active_dataset is None or getattr(window, '_isar_busy', False):
        window.status.showMessage('Select a dataset and finish or cancel ISAR before loading a recipe.')
        return
    path, _ = QFileDialog.getOpenFileName(window, 'Load ISAR recipe', '', 'ISAR recipe (*.isar.json)')
    if not path:
        return
    try:
        arguments = recipes.recipe_arguments(window.active_dataset, recipes.load_recipe(path))
        apply_recipe_controls(window, arguments)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        window.status.showMessage('Recipe could not be applied: ' + str(exc))
        return
    window._invalidate_isar_result()
    window.status.showMessage('Recipe loaded. Use Plan image to inspect it, then Apply ISAR Settings.')


def apply_recipe_controls(window, arguments):
    """Validate representation before changing any controls or selections."""
    target = arguments.get('azimuth_target_degrees')
    if target is not None:
        target = np.asarray(target)
        step = float(np.mean(np.diff(target)))
        if not np.allclose(np.diff(target), step, atol=1e-9, rtol=1e-10):
            raise ValueError('This interpolation axis is not uniform; use the headless recipe API for it')
        for spin, value in ((window.spin_isar_az_min, target[0]), (window.spin_isar_az_max, target[-1]), (window.spin_isar_az_step, step)):
            if not spin.minimum() <= value <= spin.maximum() or abs(round(float(value), spin.decimals()) - value) > 1e-9:
                raise ValueError('The interpolation axis exceeds GUI precision or limits; use the headless recipe API')
    recon = arguments.get('reconstruction', 'fast')
    recon_index = {'fast': 0, 'fft': 0, 'accurate': 1, 'sparse': 2, 'auto': 3}[recon]
    advanced = window.isar_advanced
    center = arguments.get('aperture_center_degrees')
    half = arguments.get('scene_half_extent_m')
    numeric = [(window.spin_isar_l1_strength, arguments.get('l1_strength', .05)),
               (window.spin_isar_l1_iters, arguments.get('l1_iterations', 300))]
    if half is not None:
        numeric.extend(zip((advanced.x_half, advanced.y_half), half))
    if center is not None:
        from GRIM_Backend.plotting.modes.isar_mode import _angle_values_to_degrees
        center = float(center) % 360.
        numeric.append((window.spin_isar_ap_center, center))
        az = _angle_values_to_degrees(window.active_dataset, 'azimuth',
            window.active_dataset.azimuths[arguments['azimuth_indices']])
        width = min(360., max(.01, np.ceil(2 * np.max(abs((az - center + 180) % 360 - 180)) * 1000) / 1000))
    for spin, value in numeric:
        digits = spin.decimals() if hasattr(spin, 'decimals') else 0
        if not spin.minimum() <= value <= spin.maximum() or abs(round(float(value), digits) - value) > 1e-9:
            raise ValueError('A recipe value exceeds GUI precision or limits; use the headless recipe API')
    from GRIM_Backend.plotting.actions import _selected_polarization_axis_availability
    availability = _selected_polarization_axis_availability(window.active_dataset,
        [arguments['polarization_index']], require_phase=True)
    axis_specs = [(window.list_freq, window.active_dataset.frequencies, arguments['frequency_indices'], availability[0]),
                  (window.list_elev, window.active_dataset.elevations, [arguments['elevation_index']], availability[1]),
                  (window.list_az, window.active_dataset.azimuths, arguments['azimuth_indices'], availability[2])]
    for _, _, indices, mask in axis_specs:
        if not np.all(mask[indices]):
            raise ValueError('Recipe requests unavailable samples for this polarization')
    widgets = [window.combo_isar_window, window.combo_isar_units, window.combo_isar_recon,
        window.spin_isar_l1_strength, window.spin_isar_l1_iters, window.chk_isar_flip_x, window.chk_isar_flip_y,
        window.chk_isar_freq_band, window.chk_isar_az_interp, window.chk_isar_aperture,
        window.spin_isar_az_min, window.spin_isar_az_max, window.spin_isar_az_step,
        window.spin_isar_ap_center, window.spin_isar_ap_width,
        window.list_az, window.list_freq, window.list_elev, window.list_pol,
        advanced, advanced.mode, advanced.scene, advanced.x_half, advanced.y_half, advanced.side, advanced.native]
    blockers = [QSignalBlocker(w) for w in widgets]
    try:
        window.combo_isar_recon.setCurrentIndex(recon_index)
        window.combo_isar_window.setCurrentText(arguments.get('window', 'Hanning'))
        window.combo_isar_units.setCurrentText(arguments.get('length_unit', 'm'))
        window.spin_isar_l1_strength.setValue(arguments.get('l1_strength', .05))
        window.spin_isar_l1_iters.setValue(arguments.get('l1_iterations', 300))
        window.chk_isar_flip_x.setChecked(arguments.get('flip_x', False))
        window.chk_isar_flip_y.setChecked(arguments.get('flip_y', False))
        # Exact selectors are visible in the lists. A visible aperture center
        # retains wrap semantics; its width includes every selected sample.
        window.chk_isar_freq_band.setChecked(False)
        window.chk_isar_aperture.setChecked(center is not None)
        if center is not None:
            window.spin_isar_ap_center.setValue(center)
            window.spin_isar_ap_width.setValue(width)
        window.chk_isar_az_interp.setChecked(target is not None)
        if target is not None:
            window.spin_isar_az_min.setValue(float(target[0]))
            window.spin_isar_az_max.setValue(float(target[-1]))
            window.spin_isar_az_step.setValue(step)
        advanced.mode.setCurrentIndex(advanced.mode.findData(arguments.get('aperture_mode', 'auto')))
        advanced.side.setEnabled(advanced.mode.currentData() != 'coherent')
        advanced.scene.setChecked(half is not None)
        if half is not None:
            advanced.x_half.setValue(half[0])
            advanced.y_half.setValue(half[1])
        advanced.x_half.setEnabled(half is not None)
        advanced.y_half.setEnabled(half is not None)
        advanced.side.setValue(arguments.get('composite_side', 1024))
        advanced.native.setChecked(arguments.get('native_diagnostics', True))
        for widget, values, _, mask in axis_specs:
            window._fill_list(widget, values, np.flatnonzero(mask))
            widget.blockSignals(True)
        for widget, indices in ((window.list_az, arguments['azimuth_indices']), (window.list_freq, arguments['frequency_indices']),
                                (window.list_elev, [arguments['elevation_index']]), (window.list_pol, [arguments['polarization_index']])):
            widget.clearSelection()
            for index in indices:
                item = next((widget.item(i) for i in range(widget.count()) if widget.item(i).data(Qt.UserRole + 1) == index), None)
                if item is not None:
                    item.setSelected(True)
    finally:
        del blockers
    from .isar_controls import sync_reconstruction_controls
    sync_reconstruction_controls(window)


def show_guide(window):
    QMessageBox.information(window, 'ISAR workflow',
        '1. Inspect the complex dataset and its frequency, angular, phase, and acquisition conventions. ISAR assumes a stationary far-field monostatic scene and a horizontal image plane.\n\n'
        '2. Use Dataset Operations → Range Cal for complex reference calibration. For matched target-plus-support and support-only acquisitions, use Support Ref −. Subtraction does not recover coupling or shadowing.\n\n'
        '3. Select one polarization, one elevation, an acquired frequency band, and an angular sector. Enter the occupied scene half extents in ISAR Settings.\n\n'
        '4. Use Plan image. Inspect native sampling and curvature warnings. Recommended PFA chooses Fast or Accurate for the scene. Select coherent or qualitative composite explicitly when needed. Interpolating or adding pixels does not add resolution.\n\n'
        '5. Apply ISAR Settings. Review measured coverage, gaps, origin PSF cuts, and sparse native-data residuals in Quality. Sparse convergence certifies its gridded optimization, not a physical source classification. Cancel stops at a bounded processing block.\n\n'
        '6. Export ISAR Result and Save recipe. Open result works without the original dataset. Compare result checks frame, acquisition and normalization compatibility; use shared intensity scales, peak profiles, and statistics in the zoomed A−B view. Differences are generic image-intensity dB.\n\n'
        'Height cannot be recovered independently from one azimuth cut. Complex images retain a documented spatial-frequency phase origin for headless coefficient conversion.')
