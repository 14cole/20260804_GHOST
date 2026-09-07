"""Search setup and provenance for FREDDY's existing inverse-design engine."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import threading

from .compute import build_uncertainty_scales, make_frequency_sweep, make_sweep, validate_incidence_angle
from .io import read_material_table, save_project_file
from .compute import validate_sweep_coverage
from .inverse_grid import DesignGrid, NumericChoices


class StopInverseSearch(Exception):
    """Cooperative stop at a boundary where no partial score is published."""


def configure_layers(layers, rows):
    """Validate every row before replacing any active layer."""
    if len(rows) != len(layers):
        raise ValueError('Layer list changed. Reopen search setup.')
    result = copy.deepcopy(layers)
    for index, (layer, row) in enumerate(zip(result, rows), 1):
        prefix = 'inv_rs_' if layer.is_sheet else 'inv_t_'
        suffixes = ('min', 'max', 'accuracy') if layer.is_sheet else ('min_in', 'max_in', 'accuracy_in')
        values = [None, None, None]
        if row['vary']:
            try:
                values = [float(row['minimum']), float(row['maximum']),
                          float(row['step']) if str(row['step']).strip() else None]
            except (TypeError, ValueError):
                raise ValueError(f'Layer {index}: enter numeric minimum, maximum, and step.') from None
            if any(v is not None and (not math.isfinite(v) or v <= 0) for v in values):
                raise ValueError(f'Layer {index}: bounds and step must be finite and positive.')
            if values[1] < values[0]:
                raise ValueError(f'Layer {index}: maximum must be at least minimum.')
            NumericChoices.build(*values, f'Layer {index}')
        for suffix, value in zip(suffixes, values):
            setattr(layer, prefix + suffix, value)
    return result


def search_identity(layers, frequencies, angles, polarization, uncertainty, score_mode):
    """Hash physics inputs, not run budget, display choices, or output paths."""
    sources = {}
    for index, layer in enumerate(layers,1):
        if layer.is_sheet:
            continue
        for raw in [layer.file_0deg] + ([layer.file_90deg] if layer.anisotropic else []):
            path = Path(raw).resolve()
            digest = hashlib.sha256()
            try:
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
            except OSError as exc:
                raise ValueError(f'Layer {index}: cannot read material file {path}: {exc}') from exc
            sources[str(path)] = digest.hexdigest()
    value = {'layers': [asdict(l) for l in layers], 'frequencies': frequencies,
             'angles': angles, 'polarization': polarization.strip().lower(), 'uncertainty': asdict(uncertainty),
             'score_mode': score_mode, 'sources': sources,
             'method': 'all-combinations-v1'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_layers(layers, frequencies, *, materials=True):
    """Return actionable errors without modifying ranges or data coverage."""
    errors = []
    if not layers:
        errors.append('Add at least one layer.')
    for index, layer in enumerate(layers, 1):
        nominal = layer.sheet_resistance if layer.is_sheet else layer.thickness_in
        if not math.isfinite(nominal) or nominal <= 0:
            errors.append(f'Layer {index}: nominal value must be finite and positive.')
        lo, hi, step = ((layer.inv_rs_min, layer.inv_rs_max, layer.inv_rs_accuracy)
                        if layer.is_sheet else (layer.inv_t_min_in, layer.inv_t_max_in, layer.inv_t_accuracy_in))
        try:
            if (lo is None) != (hi is None):
                raise ValueError('set both bounds, or select Fixed.')
            if lo is not None:
                configure_layers([layer], [{'vary': True, 'minimum': lo, 'maximum': hi, 'step': '' if step is None else step}])
        except ValueError as exc:
            errors.append(f'Layer {index}: {str(exc).removeprefix("Layer 1: ")}')
        if materials and not layer.is_sheet:
            for axis, raw in [('0 deg / isotropic', layer.file_0deg)] + ([('90 deg', layer.file_90deg)] if layer.anisotropic else []):
                try:
                    table = read_material_table(Path(raw))
                    validate_sweep_coverage(frequencies, table, f'layer {index} {axis}')
                except Exception as exc:
                    errors.append(f'Layer {index}, {axis}, {Path(raw).name or "missing material"}: {exc}')
    if errors:
        raise ValueError('\n'.join(errors))


class InverseWorkflowMixin:
    def _build_inverse_workflow(self, layout):
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QLabel
        self._inverse_stop_event = threading.Event()
        self._inverse_checkpoint = None
        self._inverse_result_identity = None
        self._inverse_active = False
        self._inverse_progress = None
        row = QHBoxLayout()
        self.inv_setup_btn = QPushButton('Fixed / variable layers…')
        self.inv_check_btn = QPushButton('Check setup')
        for b in (self.inv_setup_btn, self.inv_check_btn):
            row.addWidget(b)
        layout.addLayout(row)
        self.inv_setup_status = QLabel('All allowed design combinations are analyzed. Tolerance corners check systematic changes; they are not manufacturing yield.')
        self.inv_setup_status.setWordWrap(True)
        layout.addWidget(self.inv_setup_status)
        self.inv_setup_btn.clicked.connect(self._edit_inverse_layers)
        self.inv_check_btn.clicked.connect(self._check_inverse_setup)
        self.inv_work_count = QLabel()
        self.inv_work_count.setWordWrap(True)
        layout.addWidget(self.inv_work_count)
        from PySide6.QtCore import QTimer
        self._inverse_count_timer = QTimer(self)
        self._inverse_count_timer.setSingleShot(True)
        self._inverse_count_timer.setInterval(150)
        self._inverse_count_timer.timeout.connect(self._refresh_inverse_work_count)
        for name in ('inv_freq_mode', 'inv_freq_list', 'inv_target_start', 'inv_target_stop',
                     'inv_target_step', 'inv_angle_start', 'inv_angle_stop', 'inv_angle_step',
                     'inv_uncertainty', 'inv_unc_t_pct', 'inv_unc_eps_pct', 'inv_unc_mu_pct'):
            getattr(self, name + '_var').valueChanged.connect(self._schedule_inverse_work_count)
        self._schedule_inverse_work_count()

    def _build_inverse_continue_actions(self, layout):
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QLineEdit, QLabel
        row = QHBoxLayout()
        self.inv_stop_btn = QPushButton('Stop and keep best')
        self.inv_extend_btn = QPushButton('Resume remaining')
        self.inv_save_candidate_btn = QPushButton('Save selected stack…')
        for button in (self.inv_stop_btn, self.inv_extend_btn, self.inv_save_candidate_btn):
            row.addWidget(button)
        self.inv_stop_btn.setEnabled(False)
        self.inv_extend_btn.setEnabled(False)
        self.inv_stop_btn.clicked.connect(self._stop_inverse_search)
        self.inv_extend_btn.clicked.connect(self._resume_inverse_analysis)
        self.inv_save_candidate_btn.clicked.connect(self._save_inverse_candidate)
        self.inv_extend_btn.setToolTip('Finish an interrupted analysis without repeating completed combinations. Inputs and material files must still match.')
        layout.addLayout(row)
        recovery = QHBoxLayout()
        recovery.addWidget(QLabel('Recovery file'))
        self.inverse_recovery_path = QLineEdit()
        self.inverse_recovery_path.setPlaceholderText('Optional: choose a file before a long search')
        recovery.addWidget(self.inverse_recovery_path, 1)
        self.inv_checkpoint_save_btn = QPushButton('Choose / save...')
        self.inv_checkpoint_load_btn = QPushButton('Load checkpoint...')
        self.inv_checkpoint_save_btn.clicked.connect(self._choose_inverse_checkpoint)
        self.inv_checkpoint_load_btn.clicked.connect(self._load_inverse_checkpoint)
        self.inv_checkpoint_load_btn.setToolTip('Open the matching project first. Resume validates the setup and material contents before reusing scores.')
        recovery.addWidget(self.inv_checkpoint_save_btn)
        recovery.addWidget(self.inv_checkpoint_load_btn)
        layout.addLayout(recovery)

    def _choose_inverse_checkpoint(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from .search_checkpoint import save_checkpoint
        if self.job_is_running():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Choose search recovery file',
                                             self.inverse_recovery_path.text() or 'search.fsearch',
                                             'FREDDY search checkpoint (*.fsearch)')
        if not path:
            return
        if not path.lower().endswith('.fsearch'):
            path += '.fsearch'
        try:
            if self._inverse_checkpoint is not None:
                # A loaded checkpoint can be copied before Resume has rebuilt
                # its plots. It already passed the checkpoint reader's checks.
                if self._inverse_result_identity is not None:
                    self._ensure_inverse_result_current()
                save_checkpoint(path, self._inverse_checkpoint)
            self.inverse_recovery_path.setText(str(Path(path).resolve()))
            self.status_var.set('Recovery file selected. Complete scores are saved every 30 seconds and when the search stops.')
        except Exception as exc:
            QMessageBox.warning(self, 'Search checkpoint', str(exc))

    def _load_inverse_checkpoint(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from .search_checkpoint import load_checkpoint
        if self.job_is_running():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Load search checkpoint', '',
                                             'FREDDY search checkpoint (*.fsearch)')
        if not path:
            return
        try:
            checkpoint = load_checkpoint(path)
        except Exception as exc:
            QMessageBox.warning(self, 'Search checkpoint', str(exc))
            return
        self._inverse_checkpoint = checkpoint
        self._inverse_result_identity = None
        self.inverse_candidates = []
        self.inverse_plot_freqs = []
        self.inverse_plot_samples = []
        self.inverse_result_metadata = {}
        self._inverse_summary = ''
        self._inverse_page_index = 0
        self._refresh_inverse_results_list()
        self.inverse_recovery_path.setText(str(Path(path).resolve()))
        self.inv_extend_btn.setEnabled(True)
        self.inv_setup_status.setText(f"Loaded {checkpoint['next_index']:,} completed combinations. Resume checks the current setup and material files, then rebuilds the comparison plots.")

    def _edit_inverse_layers(self):
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
                                      QComboBox, QDialogButtonBox, QMessageBox, QHeaderView)
        from PySide6.QtCore import Qt
        if self.job_is_running():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Allowed layer values — fixed or variable')
        dialog.resize(810, 380)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(self.layers), 6)
        table.setHorizontalHeaderLabels(['Layer / material', 'Mode', 'Current', 'Minimum', 'Maximum', 'Step'])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i, layer in enumerate(self.layers):
            lo, hi, step = ((layer.inv_rs_min, layer.inv_rs_max, layer.inv_rs_accuracy) if layer.is_sheet
                            else (layer.inv_t_min_in, layer.inv_t_max_in, layer.inv_t_accuracy_in))
            label = f'{i+1}. ' + ('Sheet resistance (Ω/sq)' if layer.is_sheet else Path(layer.file_0deg).name + ' — thickness (in)')
            nominal = layer.sheet_resistance if layer.is_sheet else layer.thickness_in
            for col, value in [(0, label), (2, f'{nominal:g}')]:
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, col, item)
            mode = QComboBox()
            mode.addItems(['Fixed', 'Vary'])
            mode.setCurrentText('Vary' if lo is not None or hi is not None else 'Fixed')
            table.setCellWidget(i, 1, mode)
            for col, value in zip((3,4,5), (lo,hi,step)):
                table.setItem(i, col, QTableWidgetItem('' if value is None else f'{value:g}'))
        layout.addWidget(table)
        from PySide6.QtWidgets import QLabel
        help_text = QLabel('Vary: set Minimum, Maximum, and Step. Values start at Minimum and advance by Step.\nAn off-step Maximum is not added. Fixed layers use their current value. Equal limits define one value.')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        def accept():
            try:
                rows = [{'vary': table.cellWidget(i,1).currentText() == 'Vary',
                         'minimum': table.item(i,3).text(), 'maximum': table.item(i,4).text(),
                         'step': table.item(i,5).text()} for i in range(table.rowCount())]
                revised = configure_layers(self.layers, rows)
            except ValueError as exc:
                QMessageBox.warning(dialog, 'Search parameters', str(exc))
                return
            self.layers = revised
            self._refresh_layers()
            dialog.accept()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def _inverse_setup_values(self):
        if self.inv_freq_mode_var.get().lower().startswith('discrete'):
            freqs = self._parse_inverse_discrete_freqs(self.inv_freq_list_var.get())
        else:
            freqs = make_frequency_sweep(float(self.inv_target_start_var.get()), float(self.inv_target_stop_var.get()), float(self.inv_target_step_var.get()))
        start, stop = map(validate_incidence_angle, (float(self.inv_angle_start_var.get()), float(self.inv_angle_stop_var.get())))
        if stop < start:
            raise ValueError('Angle stop must be at least start.')
        angles = [start] if start == stop else make_sweep(start, stop, float(self.inv_angle_step_var.get()))
        return self._snapshot_layers(), freqs, angles, self._read_inverse_uncertainty_config()

    def _check_inverse_setup(self):
        from PySide6.QtWidgets import QMessageBox
        if self.job_is_running():
            return
        try:
            layers, freqs, angles, cfg = self._inverse_setup_values()
            if int(self.inv_top_n_var.get()) <= 0:
                raise ValueError('Keep best must be a positive integer.')
        except Exception as exc:
            self.inv_setup_status.setText(str(exc))
            return
        def worker():
            try:
                check_layers(layers, freqs)
            except Exception as exc:
                return f'Correct setup:\n{exc}'
            return 'Setup ready. ' + self._inverse_work_description(layers, len(freqs), len(angles), cfg)
        def success(text):
            self.inv_setup_status.setText(text)
        self._run_background_task('Check analysis setup', worker, success, 'Analysis setup — open Fixed / variable layers to correct the listed layer')

    def _schedule_inverse_work_count(self, *_args):
        if hasattr(self, '_inverse_count_timer'):
            self._inverse_count_timer.start()

    @staticmethod
    def _inverse_work_description(layers, frequency_count, angle_count, cfg):
        grid = DesignGrid(layers)
        cases = len(build_uncertainty_scales(cfg))
        return (f'{grid.total:,} combinations × {frequency_count:,} frequencies × '
                f'{angle_count:,} angles × {cases} nominal/tolerance cases = '
                f'{grid.total*frequency_count*angle_count*cases:,} response points. '
                'Comparison plots add work.\n' + grid.description())

    def _refresh_inverse_work_count(self):
        try:
            # Editing a tiny step must not allocate a huge frequency/angle
            # vector just to display the work count. Match make_sweep's count.
            def count(start, stop, step):
                if not all(math.isfinite(v) for v in (start, stop, step)) or step <= 0 or stop < start:
                    raise ValueError('Sweep limits must be finite, stop ≥ start, and step > 0.')
                return math.floor((stop - start) / step + 1e-12) + 1
            if self.inv_freq_mode_var.get().lower().startswith('discrete'):
                frequency_count = len(self._parse_inverse_discrete_freqs(self.inv_freq_list_var.get()))
            else:
                start = float(self.inv_target_start_var.get())
                if start <= 0:
                    raise ValueError('Frequency start must be greater than zero.')
                frequency_count = count(start, float(self.inv_target_stop_var.get()), float(self.inv_target_step_var.get()))
            start, stop = map(validate_incidence_angle, (float(self.inv_angle_start_var.get()), float(self.inv_angle_stop_var.get())))
            angle_count = 1 if start == stop else count(start, stop, float(self.inv_angle_step_var.get()))
            text = self._inverse_work_description(self._snapshot_layers(), frequency_count, angle_count,
                                                  self._read_inverse_uncertainty_config())
        except Exception as exc:
            text = 'Complete setup: ' + str(exc)
        self.inv_work_count.setText(text)

    def _inverse_can_resume(self):
        checkpoint = self._inverse_checkpoint
        return bool(checkpoint and (checkpoint['next_index'] < checkpoint['total']
                                   or not checkpoint['plots_complete']))

    def _show_inverse_progress(self):
        if self._inverse_progress is None:
            return
        done, total, phase = self._inverse_progress
        text = f'{phase}: {done:,} / {total:,} combinations ({100 * done / total:.1f}%)'
        self.status_var.set(text)
        self.inv_setup_status.setText(text)
        self.status_progress.setRange(0, 1000)
        self.status_progress.setValue(int(1000 * done / total))

    def _stop_inverse_search(self):
        if self._inverse_active:
            self._inverse_stop_event.set()
            self.status_var.set('Stopping at the next safe boundary; keeping complete candidates…')
            self.inv_stop_btn.setEnabled(False)

    def _resume_inverse_analysis(self):
        self._run_inverse_design(resume=True)

    def _save_inverse_candidate(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        if self.job_is_running() or not self.inverse_candidates:
            return
        row = self._selected_inverse_index()
        if row < 0:
            return
        try:
            self._ensure_inverse_result_current()
            candidate = self.inverse_candidates[row]
            state = copy.deepcopy(self._collect_project_state())
            for layer, thickness, resistance in zip(state['layers'], candidate.thickness_in, candidate.sheet_resistance_ohm):
                layer['thickness_in'] = thickness
                layer['sheet_resistance'] = resistance
            path, _ = QFileDialog.getSaveFileName(self, 'Save candidate as a separate FREDDY project', 'candidate.json', 'FREDDY project (*.json)')
            if not path:
                return
            if not path.lower().endswith('.json'):
                path += '.json'
            if self.project_path and Path(path).resolve() == self.project_path.resolve():
                raise ValueError('Choose a different file to preserve the original stack.')
            # Reopening the saved candidate must not inherit the original
            # project's nominal/sweep destinations either.
            target = Path(path).resolve()
            base = target.stem
            serial = 1
            while any((target.parent / (base + suffix)).exists() for suffix in ('_impedance.csv','_angle.csv','_thickness.csv','_ibc')):
                serial += 1
                base = f'{target.stem}_{serial}'
            for key, suffix in [('output','_impedance.csv'), ('angle_output','_angle.csv'),
                                ('thk_output','_thickness.csv'), ('ibc_batch_output_dir','_ibc')]:
                state['controls'][key] = str(target.parent / (base + suffix))
            save_project_file(Path(path), state)
            self.status_var.set(f'Saved candidate to {path}. The current stack is unchanged.')
        except Exception as exc:
            QMessageBox.warning(self, 'Save candidate', str(exc))

    def _ensure_inverse_result_current(self):
        if self._inverse_result_identity is None:
            raise ValueError('Run the search before applying or saving a candidate.')
        layers, freqs, angles, cfg = self._inverse_setup_values()
        identity = search_identity(layers, freqs, angles, self.inv_wave_pol_var.get(), cfg,
                                   self.inv_score_mode_var.get())
        if identity != self._inverse_result_identity:
            raise ValueError('Stack, targets, tolerances, or material files changed. Analyze the updated setup before using these candidates.')
