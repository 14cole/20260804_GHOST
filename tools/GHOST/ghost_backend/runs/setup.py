"""Portable 2D and BOR physics setups for GHOST and batch drivers."""
import json
import math
import os
from pathlib import Path
import tempfile

DEFAULT_QUALITY = {'residual_norm_max': 1e-6, 'condition_est_max': 1e6, 'warnings_max': 10}


def validate_setup(value):
    if isinstance(value, dict) and value.get('schema') == 'grim.bor-run-setup':
        from ghost_backend.runs.bor_setup import validate_bor_setup
        return validate_bor_setup(value)
    if not isinstance(value, dict) or value.get('schema') != 'grim.2d-run-setup' or type(value.get('version')) is not int or value.get('version') not in (1, 2):
        raise ValueError('Choose a supported GRIM 2D run setup (version 1 or 2).')
    allowed = {'schema', 'version', 'frequencies_ghz', 'angles_deg', 'units', 'mesh_certification',
               'accuracy', 'lu_precision', 'scattering', 'observation_angles_deg', 'quality'}
    if set(value) - {'solver_method', 'execution_options'} != allowed:
        raise ValueError('Run setup has missing or unsupported fields. No settings were applied.')
    result = dict(value)
    if value['version'] == 2 and 'execution_options' not in value:
        raise ValueError('Version 2 setups require execution settings.')
    from ghost_backend.execution.options import validate_for_run
    result['version'] = 2
    result.setdefault('solver_method', 'direct')
    if result['solver_method'] not in ('auto', 'direct', 'experimental_cpu'):
        raise ValueError('Unsupported solver method.')
    if result['solver_method'] == 'experimental_cpu' and (value['scattering'] != 'monostatic' or value['lu_precision'] != 'double'):
        raise ValueError('Experimental CPU requires monostatic scattering and double LU precision.')
    for key in ('frequencies_ghz', 'angles_deg', 'observation_angles_deg'):
        entries = value[key]
        if not isinstance(entries, list) or (key != 'observation_angles_deg' and not entries):
            raise ValueError(f'{key}: supply a nonempty numeric list.')
        if len(entries) > 100000:
            raise ValueError(f'{key}: setup exceeds 100,000 samples.')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in entries):
            raise ValueError(f'{key}: every sample must be finite and numeric.')
        if len(entries) != len(set(entries)):
            raise ValueError(f'{key}: duplicate samples are not supported in a saved setup.')
        if key == 'frequencies_ghz' and any(x <= 0 for x in entries):
            raise ValueError('Frequencies must be positive GHz values.')
        result[key] = list(entries)
    for key, choices in [('units', ('inches', 'meters')), ('accuracy', ('standard', 'tight')),
                         ('lu_precision', ('double', 'mixed')), ('scattering', ('monostatic', 'bistatic'))]:
        if value[key] not in choices:
            raise ValueError(f'Unsupported {key}: {value[key]!r}.')
    if type(value['mesh_certification']) is not bool:
        raise ValueError('mesh_certification must be true or false.')
    if value['scattering'] == 'bistatic' and not value['observation_angles_deg']:
        raise ValueError('Bistatic setup requires observation angles.')
    quality = value['quality']
    if not isinstance(quality, dict) or set(quality) != set(DEFAULT_QUALITY):
        raise ValueError('Invalid quality thresholds.')
    for key, x in quality.items():
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or (key != 'warnings_max' and x == 0):
            raise ValueError(f'Invalid quality threshold: {key}.')
    if int(quality['warnings_max']) != quality['warnings_max']:
        raise ValueError('Warning threshold must be a whole number.')
    result['quality'] = dict(quality)
    result['execution_options'] = validate_for_run(value.get('execution_options', {}),
        result['solver_method'], value['lu_precision'], value['scattering'])
    return result


def save_setup(path, value):
    value = validate_setup(value)
    path = Path(path)
    if not path.name.endswith('.run.json'):
        raise ValueError('Save run setups with the .run.json suffix.')
    fd, temporary = tempfile.mkstemp(prefix='.run-setup-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_setup(path):
    return validate_setup(json.loads(Path(path).read_text(encoding='utf-8')))


def geometry_dimensions(snapshot, units):
    points = [(float(pair[x]), float(pair[y])) for seg in snapshot.get('segments', [])
              for pair in seg.get('point_pairs', []) for x,y in [('x1','y1'), ('x2','y2')]]
    if not points or any(not math.isfinite(v) for p in points for v in p):
        raise ValueError('Load geometry with finite coordinates to inspect its dimensions.')
    scale = .0254 if units == 'inches' else 1.
    width = (max(x for x,y in points) - min(x for x,y in points)) * scale
    height = (max(y for x,y in points) - min(y for x,y in points)) * scale
    return f'X span {width/.0254:g} in \u00d7 Y span {height/.0254:g} in'


class RunSetupMixin:
    def _build_run_setup_controls(self, form, advanced_form=None):
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QComboBox, QLabel, QWidget, QHBoxLayout, QPushButton
        except ImportError:
            from PySide2.QtCore import Qt
            from PySide2.QtWidgets import QComboBox, QLabel, QWidget, QHBoxLayout, QPushButton
        details = advanced_form if advanced_form is not None else form
        from ghost_backend.ui.execution import ExecutionOptionsWidget
        self.execution_options_widget = ExecutionOptionsWidget()
        self.execution_options_widget.changed.connect(self._sync_execution_options)
        details.addRow(self.execution_options_widget)
        self.geometry_preset_combo = QComboBox()
        self.geometry_preset_combo.setPlaceholderText('Custom (Advanced Settings)')
        self.geometry_preset_combo.setMinimumContentsLength(24)
        self.geometry_preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.geometry_preset_notice = QLabel()
        self.geometry_preset_notice.setWordWrap(True)
        for name, label, description in (
            ('small', 'Small Geometry (No RAM Optimization)',
             'Dense solve without sweep compression. Use when the geometry fits comfortably in RAM.'),
            ('large', 'Large Geometry (RAM Optimization)',
             'Reduces RAM for large geometries and wide sweeps. May take longer for small geometries.'),
            ('adaptive', 'Automatic (RAM-aware)',
             'Chooses dense or compressed after forecasting both certification meshes.'),
            ('balanced', 'Balanced',
             'Reuses work across angles with a dense solve. Uses more RAM than Large Geometry.'),
        ):
            self.geometry_preset_combo.addItem(label, name)
            self.geometry_preset_combo.setItemData(self.geometry_preset_combo.count() - 1, description, Qt.ToolTipRole)
        form.insertRow(2, 'Geometry preset', self.geometry_preset_combo)
        form.insertRow(3, self.geometry_preset_notice)
        self.geometry_preset_combo.currentIndexChanged.connect(self._apply_geometry_preset)
        method = self.cmb_solver_method
        precision = self.cmb_lu_precision
        method.currentIndexChanged.connect(self._sync_geometry_preset)
        precision.currentIndexChanged.connect(self._sync_geometry_preset)
        row = QWidget()
        buttons = QHBoxLayout(row)
        buttons.setContentsMargins(0,0,0,0)
        self.save_run_setup_button = QPushButton('Save run setup\u2026')
        self.load_run_setup_button = QPushButton('Load run setup\u2026')
        buttons.addWidget(self.save_run_setup_button)
        buttons.addWidget(self.load_run_setup_button)
        self.save_run_setup_button.clicked.connect(self._save_run_setup)
        self.load_run_setup_button.clicked.connect(self._load_run_setup)
        details.addRow(row)
        self.run_setup_notice = QLabel()
        self.run_setup_notice.setWordWrap(True)
        form.addRow(self.run_setup_notice)
        self.run_output_notice = QLabel('Output: automatic unique GRIM file after solving.')
        self.run_output_notice.setWordWrap(True)
        self.run_preflight_button = QPushButton('Check geometry and run setup')
        self.run_preflight_button.clicked.connect(self._check_run_setup)
        form.addRow(self.run_preflight_button)
        self.cmb_units.currentTextChanged.connect(self._update_run_dimensions)
        self.edit_geo_path.textChanged.connect(self._update_run_dimensions)

    def _apply_geometry_preset(self, *_):
        """Apply a complete performance preset while retaining host paths and RAM budgets."""
        name = self.geometry_preset_combo.currentData()
        if name is None or self._setup_busy():
            return
        solver = self.cmb_solver_kind
        if solver.currentData() != '2d' or self.cmb_scatter_mode.currentData() != 'monostatic':
            self._sync_geometry_preset()
            return
        from PySide6.QtCore import QSignalBlocker
        from ghost_backend.execution.options import geometry_preset
        value = geometry_preset(name)
        try:
            current = self.execution_options_widget.value()
        except ValueError as exc:
            self._sync_geometry_preset()
            self.run_setup_notice.setText(str(exc))
            return
        for key in ('ram_budget_gib', 'temporary_directory'):
            value['execution_options'][key] = current[key]
        method = self.cmb_solver_method
        precision = self.cmb_lu_precision
        blockers = [QSignalBlocker(widget) for widget in (method, precision, self.execution_options_widget)]
        method.setCurrentIndex(method.findData(value['solver_method']))
        precision.setCurrentIndex(precision.findData(value['lu_precision']))
        self.execution_options_widget.set_value(value['execution_options'])
        del blockers
        self._sync_execution_options()

    def _sync_geometry_preset(self, *_):
        """Display the preset matching active settings, or a non-selectable custom state."""
        if not hasattr(self, 'geometry_preset_combo'):
            return
        from PySide6.QtCore import QSignalBlocker, Qt
        from ghost_backend.execution.options import geometry_preset
        solver = self.cmb_solver_kind
        available = solver.currentData() == '2d' and self.cmb_scatter_mode.currentData() == 'monostatic'
        self.geometry_preset_combo.setEnabled(available and not self._setup_busy())
        try:
            current = self.execution_options_widget.value()
        except ValueError:
            current = None
        method = self.cmb_solver_method.currentData()
        precision = self.cmb_lu_precision.currentData()
        match = -1
        for index in range(self.geometry_preset_combo.count()):
            value = geometry_preset(self.geometry_preset_combo.itemData(index))
            ignored = {'ram_budget_gib', 'temporary_directory'}
            if current is None:
                break
            if current['factorization'] == 'dense':
                ignored.add('compressed_storage_mib')
            if (value['solver_method'] == method and value['lu_precision'] == precision
                    and all(current[key] == expected for key, expected in value['execution_options'].items() if key not in ignored)):
                match = index
                break
        blocker = QSignalBlocker(self.geometry_preset_combo)
        self.geometry_preset_combo.setCurrentIndex(match if available else -1)
        del blocker
        self.geometry_preset_combo.setPlaceholderText('Custom (Advanced Settings)' if available else 'Not applicable')
        if not available:
            description = 'Geometry presets apply to 2D monostatic solves.'
        elif match < 0:
            description = 'Using custom performance settings. Choose a preset to reset them, or edit Advanced Settings.'
        else:
            description = self.geometry_preset_combo.itemData(match, Qt.ToolTipRole)
        self.geometry_preset_notice.setText(description)
        self.geometry_preset_combo.setToolTip(description)

    def _setup_busy(self):
        return self._job_is_active()

    def _capture_run_setup(self):
        solver = self.cmb_solver_kind
        is_bor = solver.currentData() == 'bor'
        freq, angles = self._collect_frequency_values(), self._collect_elevation_values()
        if is_bor:
            return validate_setup(dict(schema='grim.bor-run-setup', version=1,
                frequencies_ghz=freq, aspects_deg=angles, radar_grid=None,
                units=self.cmb_units.currentText(),
                mesh_certification=self.chk_mesh_certification.isChecked(),
                accuracy=self.cmb_accuracy_target.currentData(),
                cfie_alpha=float(self.edit_cfie_alpha.text()), bor_options=self.bor_options_widget.value()))
        return validate_setup(dict(schema='grim.2d-run-setup', version=2,
            execution_options=self.execution_options_widget.value(),
            frequencies_ghz=freq, angles_deg=angles,
            units=self.cmb_units.currentText(),
            mesh_certification=self.chk_mesh_certification.isChecked(),
            accuracy=self.cmb_accuracy_target.currentData(),
            lu_precision=self.cmb_lu_precision.currentData(),
            solver_method=self.cmb_solver_method.currentData(),
            scattering=self.cmb_scatter_mode.currentData(),
            observation_angles_deg=[] if self.cmb_scatter_mode.currentData() == 'monostatic' else self._parse_list(self.edit_obs_angles.text(), 'Observation angles'),
            quality=dict(
                residual_norm_max=float(self.edit_quality_residual_max.text()),
                condition_est_max=float(self.edit_quality_condition_max.text()),
                warnings_max=float(self.edit_quality_warnings_max.text()))))

    def _sync_execution_options(self, *_):
        if not hasattr(self, 'execution_options_widget'):
            return
        method = self.cmb_solver_method
        precision = self.cmb_lu_precision
        factor = self.execution_options_widget.factor_combo.currentData()
        solver = self.cmb_solver_kind
        if solver.currentData() == '2d' and factor in ('compressed', 'adaptive'):
            method.setCurrentIndex(method.findData('experimental_cpu'))
        if factor != 'dense':
            precision.setCurrentIndex(precision.findData('double'))
        if hasattr(self, 'btn_advanced_settings'):
            self._apply_job_state()
        self._sync_geometry_preset()

    def _apply_saved_run_setup(self, raw):
        value = validate_setup(raw)
        if value['schema'] == 'grim.bor-run-setup':
            return self._apply_saved_bor_setup(value)
        solver = self.cmb_solver_kind
        from PySide6.QtCore import QSignalBlocker
        method = self.cmb_solver_method
        precision = self.cmb_lu_precision
        blocked = [solver, method, precision, self.execution_options_widget]
        blocked.append(self.cmb_scatter_mode)
        blockers = [QSignalBlocker(widget) for widget in blocked]
        solver.setCurrentIndex(solver.findData('2d'))
        def text(values):
            return ', '.join(format(v, '.17g') for v in values)
        self.cmb_freq_mode.setCurrentIndex(0)
        self.cmb_elev_mode.setCurrentIndex(0)
        self.edit_freq_list.setText(text(value['frequencies_ghz']))
        self.edit_elev_list.setText(text(value['angles_deg']))
        self.cmb_units.setCurrentText(value['units'])
        self.cmb_scatter_mode.setCurrentIndex(self.cmb_scatter_mode.findData(value['scattering']))
        self.edit_obs_angles.setText(text(value['observation_angles_deg']))
        for field,key in [(self.edit_quality_residual_max,'residual_norm_max'), (self.edit_quality_condition_max,'condition_est_max'), (self.edit_quality_warnings_max,'warnings_max')]:
            field.setText(format(value['quality'][key], '.17g'))
        self.chk_mesh_certification.setChecked(value['mesh_certification'])
        for combo,key in [(self.cmb_accuracy_target,'accuracy'), (self.cmb_lu_precision,'lu_precision')]:
            combo.setCurrentIndex(combo.findData(value[key]))
        method = self.cmb_solver_method
        method.setCurrentIndex(method.findData('direct' if value['solver_method'] == 'auto' else value['solver_method']))
        self.execution_options_widget.set_value(value['execution_options'])
        del blockers
        self._sync_execution_options()
        self._on_solver_kind_changed()
        self.run_setup_notice.setText('2D setup loaded. Check geometry, dimensions, and output before running.')

    def _apply_saved_bor_setup(self, value):
        from PySide6.QtCore import QSignalBlocker
        solver = self.cmb_solver_kind
        blockers = [QSignalBlocker(solver)]
        blockers.append(QSignalBlocker(self.cmb_scatter_mode))
        self.cmb_scatter_mode.setCurrentIndex(self.cmb_scatter_mode.findData('monostatic'))
        solver.setCurrentIndex(solver.findData('bor'))
        text = lambda samples: ', '.join(format(x, '.17g') for x in samples)
        self.cmb_freq_mode.setCurrentIndex(0)
        self.cmb_elev_mode.setCurrentIndex(0)
        self.edit_freq_list.setText(text(value['frequencies_ghz']))
        self.edit_elev_list.setText(text(value['aspects_deg']))
        self.cmb_units.setCurrentText(value['units'])
        self.chk_mesh_certification.setChecked(value['mesh_certification'])
        accuracy = self.cmb_accuracy_target
        accuracy.setCurrentIndex(accuracy.findData(value['accuracy']))
        self.edit_cfie_alpha.setText(format(value['cfie_alpha'], '.17g'))
        self.bor_options_widget.set_value(value['bor_options'])
        del blockers
        self._on_solver_kind_changed()
        note = ('Radar grid reduced to its unique body aspects. GHOST solves and plots in body coordinates.'
                if value.get('radar_grid') is not None else
                'Original angle convention retained.')
        self.run_setup_notice.setText('BOR setup loaded. ' + note + ' Check geometry and output before running.')

    def _save_run_setup(self):
        from PySide6.QtWidgets import QFileDialog
        if self._setup_busy(): return
        try:
            value = self._capture_run_setup()
            path,_ = QFileDialog.getSaveFileName(self, 'Save run setup', 'setup.run.json', 'Run setup (*.run.json)')
            if path:
                if not path.endswith('.run.json'): path += '.run.json'
                save_setup(path,value)
                self.run_setup_notice.setText(f'Setup saved: {path}')
        except Exception as exc:
            self.run_setup_notice.setText(str(exc))

    def _load_run_setup(self):
        from PySide6.QtWidgets import QFileDialog
        if self._setup_busy(): return
        path,_ = QFileDialog.getOpenFileName(self, 'Load run setup', '', 'Run setup (*.run.json)')
        if path:
            try:
                self._apply_saved_run_setup(read_setup(path))
            except Exception as exc:
                self.run_setup_notice.setText(str(exc))

    def _update_run_dimensions(self, *_):
        try:
            snapshot,_,_ = self._load_geometry_for_solver()
            self.lbl_run_dimensions.setText(geometry_dimensions(snapshot,self.cmb_units.currentText()))
        except Exception as exc:
            self.lbl_run_dimensions.setText(str(exc))

    def _run_setup_summary(self, snapshot, base_dir, value, checkpoint=None):
        if value['schema'] == 'grim.bor-run-setup':
            from ghost_backend.runs.bor_setup import resource_summary
            return resource_summary(snapshot, base_dir, value, checkpoint)
        from ghost_backend.twod.preparation import prepare_geometry
        _, result, library, _ = prepare_geometry(snapshot, base_dir, value['units'])
        for freq in value['frequencies_ghz']:
            if checkpoint is not None:
                checkpoint()
            from ghost_backend.twod.formulations.thin_layer import (
                ThinLayerDefinition,
                validate_thin_layer,
            )
            used_ibcs={int(seg['properties'][2]) for seg in snapshot['segments'] if int(seg['properties'][2])>0}
            used_media={int(seg['properties'][i]) for seg in snapshot['segments'] for i in (3,4) if int(seg['properties'][i])>0}
            for flag in used_ibcs:
                model=library.impedance_models[flag]
                if isinstance(model,ThinLayerDefinition):
                    eps,mu=library.get_medium(model.dielectric_flag,freq)
                    validate_thin_layer(eps,mu,model.thickness_m,2*math.pi*freq*1e9/299792458.)
                else:
                    library.get_impedance(flag, freq, arc_s=0.)
                    library.get_impedance(flag, freq, arc_s=1.)
            for flag in used_media:
                library.get_medium(flag, freq)
        selection_note = ''
        if value['execution_options']['factorization'] == 'adaptive':
            from ghost_backend.execution.selection import select_backend
            from ghost_backend.runs.quality import accuracy_target_policy
            selection = select_backend(dict(geometry_snapshot=snapshot, material_base_dir=base_dir,
                geometry_units=value['units'], frequencies_ghz=value['frequencies_ghz'],
                elevations_deg=value['angles_deg'], solver_method=value['solver_method'], max_panels=100000,
                mesh_convergence_policy=accuracy_target_policy(value['accuracy'])), value['execution_options'], value['mesh_certification'], checkpoint)
            selection_note = 'Planned backend: {}. Dense peak forecast {:.2f} GiB; admission budget {:.2f} GiB.\n'.format(
                selection['selected'], selection['dense_peak_gib'], selection['admission_budget_gib'])
        warnings = list(result['warnings']) + list(library.warnings)
        count = len(value['frequencies_ghz'])*len(value['angles_deg'])
        if value['scattering']=='bistatic': count *= len(value['observation_angles_deg'])
        return (f"{result['segment_count']} segments; {result['primitive_count']} primitives. "
                + geometry_dimensions(snapshot,value['units']) + '\n'
                + f"{len(value['frequencies_ghz'])} frequencies \u00d7 {len(value['angles_deg'])} incident angles; {count} samples per channel, VV + HH. "
                + ('Base/fine mesh comparison' if value['mesh_certification'] else 'Survey; no mesh certificate')
                + f"; {value['accuracy']} target; {value['solver_method']} method; {value['lu_precision']} LU.\n"
                + 'Execution: ' + value['execution_options']['factorization'] + '; RAM budget ' + str(value['execution_options']['ram_budget_gib'] or 'available memory') + '; compressed payload cap ' + str(value['execution_options']['compressed_storage_mib']) + ' MiB.\n'
                + selection_note
                + ('Warnings: ' + '; '.join(warnings) if warnings else 'Geometry and material checks passed.')
                + '\nSolver quality and convergence are evaluated during the run.')

    def _check_run_setup(self):
        if self._setup_busy(): return
        try:
            value = self._capture_run_setup()
            snapshot,_,base_dir = self._load_geometry_for_solver()
            self._start_setup_check(snapshot,base_dir,value)
            self._update_run_dimensions()
            self._update_run_output_note()
        except Exception as exc:
            self.run_setup_notice.setText(f'Correct before running: {exc}')

    def _update_run_output_note(self):
        if not self.chk_export_after_solve.isChecked():
            self.run_output_notice.setText('Output: retained in GHOST; use Export Last Result when ready.')
            return
        raw=self.edit_output.text().strip()
        if raw:
            path=Path(raw).expanduser()
            if not path.is_absolute():
                _,source,base=self._load_geometry_for_solver()
                path=(Path(base) if source else self._documents_output_dir())/path
            if path.suffix.lower()!='.grim': path=Path(str(path)+'.grim')
            details=str(path.resolve())
            note=f'Output: {path.name} in {path.parent.name}.'
            if path.exists(): note+=' Existing output: replacement will require review at export.'
        else:
            _,source,base=self._load_geometry_for_solver()
            folder=Path(base) if source else self._documents_output_dir()
            details=str(folder)
            note=f'Output: a unique timestamped GRIM file in {folder.name}. Hover here for the full folder path.'
        if self.cmb_scatter_mode.currentData()=='bistatic':
            note+=' Bistatic runs write a separate file for each incident angle.'
        self.run_output_notice.setText(note)
        self.run_output_notice.setToolTip(details)

    def _start_setup_check(self, snapshot, base_dir, value):
        from ghost_backend.ui.solver import _SolveWorker, QThread
        import threading
        self._abort_event=threading.Event()
        self._solve_run_serial += 1
        run_id=self._solve_run_serial
        self._active_solve_run_id=run_id
        self._pending_solve_context=None
        self._set_solving_state(True)
        self.run_setup_notice.setText('Checking the current geometry and setup\u2026')
        thread=QThread(self)
        is_bor = value['schema'] == 'grim.bor-run-setup'
        worker=_SolveWorker(snapshot,'',base_dir,value['frequencies_ghz'],value['aspects_deg'] if is_bor else value['angles_deg'],value['units'],value.get('quality', DEFAULT_QUALITY),
                            abort_event=self._abort_event,preflight_setup=value,preflight_only=True,
                            execution_options=value.get('execution_options'), solver_method=value.get('solver_method', 'direct'),
                            lu_precision=value.get('lu_precision', 'double'), solver_kind='bor' if is_bor else '2d',
                            bor_options=value.get('bor_options'))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_solver_progress)
        worker.setup_checked.connect(self.run_setup_notice.setText)
        worker.telemetry.connect(self._on_execution_progress)
        worker.finished.connect(self._setup_check_finished)
        worker.error.connect(self._setup_check_failed)
        worker.canceled.connect(self._on_solver_canceled)
        for signal in (worker.finished,worker.error,worker.canceled):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._on_solver_thread_finished(run_id))
        self._solve_thread,self._solve_worker=thread,worker
        thread.start()

    def _setup_check_finished(self, *_):
        self._set_solving_state(False)
        self.lbl_status.setText('Setup check complete. Review the summary, then run. Geometry and materials are checked again when solving.')

    def _setup_check_failed(self, message):
        self._set_solving_state(False)
        self.run_setup_notice.setText('Correct before running: '+message)
