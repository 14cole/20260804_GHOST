"""Controls for saved CPU factorization and per-run resource settings."""
try:
    from PySide6.QtCore import Signal, QSignalBlocker
    from PySide6.QtWidgets import (
        QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
        QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QWidget,
    )
except ImportError:
    from PySide2.QtCore import Signal, QSignalBlocker
    from PySide2.QtWidgets import (
        QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
        QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QWidget,
    )
from ghost_backend.execution.options import DEFAULTS, efficient_defaults, from_environment, validate_options


class ExecutionOptionsWidget(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__('2D execution and resources', parent)
        form = QFormLayout(self)
        self.factor_combo = QComboBox()
        for label, value in [('Dense LU', 'dense'), ('Hierarchical factor (dense assembly)', 'hierarchical'),
                             ('Compressed assembly (low RAM)', 'compressed'),
                             ('RAM-aware dense / compressed', 'adaptive'),
                             ('Hierarchical with dense fallback', 'auto')]:
            self.factor_combo.addItem(label, value)
        self.factor_combo.setToolTip('Compressed assembly avoids global dense matrices. Hierarchical factorization still assembles a dense operator.')
        form.addRow('Factorization', self.factor_combo)
        self.ram_spin = QDoubleSpinBox()
        self.ram_spin.setRange(0, 1048576)
        self.ram_spin.setDecimals(3)
        self.ram_spin.setSpecialValueText('Available memory')
        self.ram_spin.setSuffix(' GiB')
        self.ram_spin.setToolTip('Per-solve admission budget, checked against estimated total RAM and current availability. This is not an operating-system memory cap.')
        form.addRow('RAM budget per solve', self.ram_spin)
        self.storage_spin = QSpinBox()
        self.storage_spin.setRange(16, 1048576)
        self.storage_spin.setSuffix(' MiB')
        self.storage_spin.setToolTip('Retained compressed operator and inverse payload, including partner polarization. 8192 MiB is 8 GiB. This does not preallocate RAM or cap total process memory. Too small a cap stops the run; workspaces require additional RAM.')
        form.addRow('Compressed storage cap', self.storage_spin)
        self.assembly_spin = QSpinBox()
        self.assembly_spin.setRange(0, 1024)
        self.assembly_spin.setSpecialValueText('Auto')
        self.assembly_spin.setToolTip('Workers building matrix tiles. More workers can speed assembly but require extra workspace RAM. Auto follows batch allocation; desktop Auto uses one thread. Start with 4 on a suitable CPU.')
        form.addRow('Assembly threads', self.assembly_spin)
        self.blas_spin = QSpinBox()
        self.blas_spin.setRange(1, 1024)
        self.blas_spin.setToolTip('Threads used by native matrix operations during this solve. Start with 2; larger dense factorizations may benefit from more. High assembly and BLAS counts can compete for CPUs. Prior limits are restored afterward.')
        form.addRow('BLAS threads per solve', self.blas_spin)
        self.temp_edit = QLineEdit()
        self.temp_edit.setPlaceholderText('System temporary directory on the execution host')
        self.temp_edit.setToolTip('Directory for compressed partner-polarization spooling. For HPC, use a directory available on the compute node, or leave empty.')
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.temp_edit)
        browse = QPushButton('Browse...')
        browse.clicked.connect(self._browse)
        layout.addWidget(browse)
        form.addRow('Temporary disk directory', row)
        self.mesh_combo = QComboBox()
        self.mesh_combo.addItem('Global material wavelength', 'global')
        self.mesh_combo.addItem('Local material sizing (experimental)', 'local')
        self.mesh_combo.setToolTip('Global uses the shortest relevant material wavelength. Local can reduce unknowns on less demanding material segments while protecting corners, junctions and nearby boundaries. Keep certification enabled; failed local convergence retries global sizing.')
        form.addRow('Mesh sizing', self.mesh_combo)
        self.rhs_combo = QComboBox()
        for label, value in [('Automatic', 'auto'), ('Disabled', 'off'), ('Enabled', 'on')]:
            self.rhs_combo.addItem(label, value)
        self.rhs_combo.setToolTip('Solve an independent incident-field basis and reconstruct all requested angles, subject to numerical checks. Automatic avoids overhead on small systems; Enabled attempts reuse more broadly; Disabled solves every right-hand side. A bounded basis uses RAM.')
        form.addRow('Sweep basis reuse', self.rhs_combo)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setToolTip('Simultaneous incident-angle workspaces, not angle spacing. 256 favors long-sweep throughput; 64 or 128 can reduce workspace RAM. The frequency matrix/factor is reused between batches.')
        form.addRow('Angles per batch', self.batch_spin)
        self.defaults_button = QPushButton('Use efficient defaults')
        self.defaults_button.clicked.connect(lambda: self.set_value(efficient_defaults()))
        form.addRow(self.defaults_button)
        self.notice = QLabel('RAM estimates, temporary disk usage, and the numeric storage cap are separate quantities.')
        self.notice.setWordWrap(True)
        form.addRow(self.notice)
        self._retained = dict(DEFAULTS)
        try:
            initial = from_environment(efficient_defaults())
        except (ValueError, TypeError):
            initial = efficient_defaults()
            self.notice.setText('Invalid launch settings. Default execution settings are shown; review them before running.')
        self.set_value(initial)
        for widget in (self.factor_combo, self.rhs_combo, self.mesh_combo):
            widget.currentIndexChanged.connect(self._changed)
        self.ram_spin.valueChanged.connect(self._ram_changed)
        for widget in (self.ram_spin, self.storage_spin, self.assembly_spin, self.blas_spin, self.batch_spin):
            widget.valueChanged.connect(self._changed)
        self.temp_edit.textChanged.connect(self._changed)
        self._changed()

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Compressed temporary directory', self.temp_edit.text())
        if path:
            self.temp_edit.setText(path)

    def _changed(self, *_):
        self.storage_spin.setEnabled(self.factor_combo.currentData() in ('compressed', 'adaptive'))
        self.changed.emit()

    def _ram_changed(self, *_):
        self._ram_edited = True

    def value(self):
        result = dict(self._retained)
        result.update(factorization=self.factor_combo.currentData(),
                      mesh_strategy=self.mesh_combo.currentData(),
                      ram_budget_gib=(self.ram_spin.value() or None) if self._ram_edited else self._retained['ram_budget_gib'],
                      compressed_storage_mib=self.storage_spin.value(),
                      temporary_directory=self.temp_edit.text().strip(),
                      assembly_threads=self.assembly_spin.value() or 'auto',
                      blas_threads=self.blas_spin.value(), rhs_compression=self.rhs_combo.currentData(),
                      angle_batch_size=self.batch_spin.value())
        return validate_options(result)

    def set_value(self, raw):
        value = validate_options(raw)
        widgets = (self.factor_combo, self.rhs_combo, self.mesh_combo, self.ram_spin, self.storage_spin,
                   self.assembly_spin, self.blas_spin, self.batch_spin, self.temp_edit)
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self._retained = value
        self._ram_edited = False
        self.factor_combo.setCurrentIndex(self.factor_combo.findData(value['factorization']))
        self.mesh_combo.setCurrentIndex(self.mesh_combo.findData(value['mesh_strategy']))
        self.rhs_combo.setCurrentIndex(self.rhs_combo.findData(value['rhs_compression']))
        self.ram_spin.setValue(value['ram_budget_gib'] or 0)
        self.storage_spin.setValue(value['compressed_storage_mib'])
        self.assembly_spin.setValue(0 if value['assembly_threads'] == 'auto' else value['assembly_threads'])
        self.blas_spin.setValue(value['blas_threads'])
        self.batch_spin.setValue(value['angle_batch_size'])
        self.temp_edit.setText(value['temporary_directory'])
        del blockers
        self._changed()
