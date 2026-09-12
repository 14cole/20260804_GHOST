"""Large, run-owned result views for FREDDY's analysis and export modes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)

from .analysis_data import band_metrics, passing_sample_ranges, write_comparison_report
from .material_explorer import extrema_preserving_indices
from .plot import style_axis, style_colorbar

METRICS = {
    'PEC reflection (dB)': 'metal_loss_db', 'PEC reflection phase (deg)': 'metal_phase_deg',
    'PEC absorbed power (dB)': 'metal_absorption_db', 'Air reflection (dB)': 'air_loss_db',
    'Air reflection phase (deg)': 'air_phase_deg', 'Air absorbed power (dB)': 'air_absorption_db',
    'Transmission (dB)': 'insertion_loss_db', 'Transmission phase (deg)': 'insertion_phase_deg',
}
VIEWS = {
    'Impedance': ['Resistance', 'Reactance', 'Frequency curves', 'Tolerance envelope', 'Power balance'],
    'IBC Batch': ['Frequency curves', 'Heatmap', 'Bandwidth', 'Band coverage', 'Resistance', 'Reactance', 'Frequency of minimum'],
    'Thickness': ['Heatmap', 'Frequency curves', 'Bandwidth', 'Band coverage', 'Tolerance envelope', 'Frequency of minimum', 'At selected frequency', 'Power balance'],
    'Off Angle': ['Heatmap', 'Frequency curves', 'TE/TM comparison', 'Bandwidth', 'Band coverage', 'Tolerance envelope', 'At selected frequency', 'Power balance'],
}
PALETTE = ['#479de0', '#eaa33e', '#49b88a', '#c278c9', '#e07070', '#a5a84d', '#9994ed']


class NumericItem(QTableWidgetItem):
    def __lt__(self, other):
        return self.data(Qt.UserRole + 1) < other.data(Qt.UserRole + 1)


class SweepResultsPanel(QWidget):
    def __init__(self, host, mode):
        super().__init__(host)
        self.host, self.mode = host, mode
        self.result = None
        self.coating = None
        self.coating_context = ''
        self.checked = set()
        self.selected = 0
        self.figure = self.canvas = None
        self._band_cache = None
        self._table_key = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(5)
        top = QHBoxLayout()
        self.status = QLabel('Run from Setup to populate these results.')
        self.status.setWordWrap(True)
        top.addWidget(self.status, 1)
        details = QPushButton('Run details…')
        details.clicked.connect(self.show_details)
        top.addWidget(details)
        explain = QPushButton('Explain view')
        explain.clicked.connect(lambda: host._show_guide('plots-analysis'))
        top.addWidget(explain)
        layout.addLayout(top)
        controls = QHBoxLayout()
        self.view = QComboBox()
        self.view.addItems(VIEWS[mode])
        self.metric = QComboBox()
        self.metric.addItems(METRICS)
        self.bound = QComboBox()
        self.bound.addItems(['Nominal', 'Upper analyzed bound', 'Lower analyzed bound', 'Tolerance span'])
        self.bound.setToolTip('Controls maps, curves, and comparison-table values. Bandwidth charts show nominal and worst reflection together; span views use nominal reflection in the comparison table.')
        self.polarization = QComboBox()
        self.polarization.addItem('TE')
        for label, widget in [('View', self.view), ('Metric', self.metric), ('', self.bound), ('Pol', self.polarization)]:
            if label:
                controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        controls.addStretch(1)
        self.table_toggle = QPushButton('Show comparison table')
        self.table_toggle.setCheckable(True)
        controls.addWidget(self.table_toggle)
        layout.addLayout(controls)
        band = QHBoxLayout()
        self.target = QDoubleSpinBox()
        self.target.setRange(-200, 0)
        self.target.setValue(-10)
        self.target.setSuffix(' dB')
        self.start = QDoubleSpinBox()
        self.stop = QDoubleSpinBox()
        self.frequency = QDoubleSpinBox()
        for box in (self.start, self.stop, self.frequency):
            box.setRange(.000001, 1e8)
            box.setDecimals(6)
            box.setMaximumWidth(120)
            box.setKeyboardTracking(False)
        self.target.setKeyboardTracking(False)
        self.selected_picker = QComboBox()
        self.selected_picker.setMinimumWidth(115)
        self.selected_picker.setMaximumWidth(210)
        for label, widget in [('Target', self.target), ('Band GHz', self.start), ('to', self.stop),
                              ('Selected', self.selected_picker), ('Slice GHz', self.frequency)]:
            band.addWidget(QLabel(label))
            band.addWidget(widget)
        band.addStretch(1)
        self.use_ibc = QPushButton('Use selected IBC for GHOST')
        self.use_ibc.clicked.connect(lambda: host._use_batch_result(self.result, self.selected))
        self.use_ibc.setVisible(mode == 'IBC Batch')
        band.addWidget(self.use_ibc)
        layout.addLayout(band)
        self.split = QSplitter(Qt.Vertical)
        layout.addWidget(self.split, 1)
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            class Toolbar(NavigationToolbar2QT):
                toolitems = tuple(item for item in NavigationToolbar2QT.toolitems
                                  if item[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom', None))
            self.figure = Figure(figsize=(10, 4), dpi=100, layout='constrained')
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumSize(200, 200)
            self.toolbar = Toolbar(self.canvas, self)
            self.toolbar.setMaximumHeight(32)
            self.toolbar.addAction('Save plot…').triggered.connect(host._save_plot)
            self.export_action = self.toolbar.addAction('Export comparison CSV…')
            self.export_action.setToolTip('Export all sampled reflection choices, band metrics, passing margin, and captured run context.')
            self.export_action.triggered.connect(self.export_comparison)
            plot_layout.addWidget(self.toolbar)
            plot_layout.addWidget(self.canvas, 1)
            self.canvas.mpl_connect('button_press_event', self.plot_clicked)
        except ImportError:
            plot_layout.addWidget(QLabel('Install matplotlib to view plots.'))
        self.split.addWidget(plot)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['Plot', 'Selection', 'Widest band (GHz)', 'Coverage (%)', 'Worst (dB)', 'Null at (GHz)', 'Output file', 'Margin (dB)'])
        self.table.horizontalHeaderItem(5).setToolTip('Sampled reflection minimum over the entire completed frequency sweep, not just the comparison band.')
        self.table.horizontalHeaderItem(7).setToolTip('Target minus worst reflection in the selected band. Nonnegative means the entire band passes under the displayed bound.')
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(25)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(90)
        self.table.setSortingEnabled(True)
        self.table.sortItems(1, Qt.AscendingOrder)
        self.split.addWidget(self.table)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([460, 115])
        self.note = QLabel('Band metrics apply to reflection. Tolerance bounds describe analyzed cases, not manufacturing yield.')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.table.itemSelectionChanged.connect(self.table_selected)
        self.table.itemChanged.connect(self.table_checked)
        self.selected_picker.currentIndexChanged.connect(self.picker_selected)
        for box in (self.metric, self.bound, self.polarization):
            box.currentIndexChanged.connect(self.refresh)
        for box in (self.start, self.stop, self.target):
            box.valueChanged.connect(self.refresh)
        self.frequency.valueChanged.connect(self.draw)
        self.view.currentIndexChanged.connect(self.draw)
        self.table_toggle.toggled.connect(self.draw)
        self.draw()

    def set_result(self, result):
        self.result = result
        self.table.horizontalHeaderItem(1).setText(result.axis_label)
        self._band_cache = None
        self._table_key = None
        self.selected = 0
        n = len(result.values)
        self.checked = set(np.linspace(0, n - 1, min(5, n), dtype=int).tolist())
        blockers = [QSignalBlocker(w) for w in (self.view, self.metric, self.polarization, self.selected_picker, self.start, self.stop, self.frequency)]
        self.view.setCurrentText(VIEWS[self.mode][0])
        self.selected_picker.clear()
        for index, value in enumerate(result.values):
            self.selected_picker.addItem(self.selection_label(index), index)
            if index < len(result.files):
                self.selected_picker.setItemData(index, str(result.files[index]), Qt.ToolTipRole)
        self.polarization.clear()
        self.polarization.addItems(list(result.metrics))
        self.polarization.setCurrentText(result.polarization)
        keys = result.metrics[result.polarization]
        if self.mode == 'Impedance':
            keys = {key for key in keys if key.startswith('metal_') == (result.backing == 'pec')}
        self.metric.clear()
        self.metric.addItems([name for name, key in METRICS.items() if key in keys])
        self.metric.setCurrentText('Air reflection (dB)' if result.backing == 'air' else 'PEC reflection (dB)')
        for box in (self.start, self.stop, self.frequency):
            box.setRange(result.frequencies[0], result.frequencies[-1])
        self.start.setValue(result.frequencies[0])
        self.stop.setValue(result.frequencies[-1])
        self.frequency.setValue(result.frequencies[len(result.frequencies) // 2])
        del blockers
        self.refresh()

    def selection_label(self, index):
        r = self.result
        return 'Nominal stack' if self.mode == 'Impedance' else f'{r.values[index]:g} {r.axis_label.split("(")[-1].rstrip(")")}'

    def reflection_key(self):
        key = METRICS.get(self.metric.currentText(), 'metal_loss_db')
        return 'air_loss_db' if key.startswith(('air_', 'insertion_')) else 'metal_loss_db'

    def grids(self, key=None, pol=None):
        key = key or METRICS.get(self.metric.currentText(), 'metal_loss_db')
        pol = pol or self.polarization.currentText()
        nominal = np.asarray(self.result.metrics[pol][key])
        lower = self.result.lower.get(pol, {}).get(key, nominal)
        upper = self.result.upper.get(pol, {}).get(key, nominal)
        return nominal, np.asarray(lower), np.asarray(upper)

    def display_grid(self, nominal, lower, upper):
        index = self.bound.currentIndex()
        return upper - lower if index == 3 else (nominal, upper, lower)[index]

    def comparison_index(self):
        return self.bound.currentIndex() if self.bound.currentIndex() < 3 else 0

    def performance(self):
        key = (id(self.result), self.polarization.currentText(), self.reflection_key(), self.start.value(), self.stop.value(), self.target.value())
        if self._band_cache is None or self._band_cache[0] != key:
            nominal, lower, upper = self.grids(self.reflection_key())
            options = dict(low=self.start.value(), high=self.stop.value())
            nom = band_metrics(self.result.frequencies, nominal, self.target.value(), **options)
            worst = nom if upper is nominal else band_metrics(self.result.frequencies, upper, self.target.value(), **options)
            best = nom if lower is nominal else band_metrics(self.result.frequencies, lower, self.target.value(), **options)
            self._band_cache = key, nom, worst, best
        return self._band_cache[1:]

    def refresh(self, *_):
        if self.result is None:
            self.draw()
            return
        try:
            metrics = self.performance()[self.comparison_index()]
        except ValueError as exc:
            self.note.setText(str(exc))
            self._table_key = None
            self.table.setRowCount(0)
            self.draw()
            return
        self._populate_table(metrics)
        self.draw()

    def _populate_table(self, metrics):
        table_key = (self._band_cache[0], self.comparison_index())
        if self._table_key == table_key:
            return
        blocker = QSignalBlocker(self.table)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(metrics))
        nominal, lower, upper = self.grids(self.reflection_key())
        grid = (nominal, upper, lower)[self.comparison_index()]
        for index, metric in enumerate(metrics):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            check.setCheckState(Qt.Checked if index in self.checked else Qt.Unchecked)
            self.table.setItem(index, 0, check)
            null_freq = self.result.frequencies[int(np.argmin(grid[:, index]))]
            values = [self.result.values[index], metric.widest_ghz, metric.coverage_pct, metric.worst_db, null_freq]
            for col, value in enumerate(values, 1):
                item = NumericItem('—' if value is None else f'{value:.5g}')
                item.setData(Qt.UserRole, index)
                item.setData(Qt.UserRole + 1, float('inf') if value is None else value)
                self.table.setItem(index, col, item)
            path = self.result.files[index] if index < len(self.result.files) else ''
            item = QTableWidgetItem(Path(path).name)
            item.setToolTip(str(path))
            self.table.setItem(index, 6, item)
            margin = self.target.value() - metric.worst_db
            item = NumericItem(f'{margin:+.5g}')
            item.setData(Qt.UserRole + 1, margin)
            item.setToolTip('Passes the whole selected band' if margin >= 0 else 'Misses the whole-band requirement')
            self.table.setItem(index, 7, item)
        self.table.setSortingEnabled(True)
        self.select_table_row()
        del blocker
        self._table_key = table_key

    def select_table_row(self):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).data(Qt.UserRole) == self.selected:
                self.table.setCurrentCell(row, 1)
                return

    def picker_selected(self, index):
        if self.result is None or index < 0:
            return
        self.selected = index
        blocker = QSignalBlocker(self.table)
        self.select_table_row()
        del blocker
        self.draw()

    def table_selected(self):
        row = self.table.currentRow()
        if row >= 0 and self.table.item(row, 1) is not None:
            self.selected_picker.setCurrentIndex(self.table.item(row, 1).data(Qt.UserRole))

    def table_checked(self, *_):
        self.checked = {int(self.table.item(row, 1).data(Qt.UserRole)) for row in range(self.table.rowCount())
                        if self.table.item(row, 0).checkState() == Qt.Checked}
        self.draw()

    def show_details(self):
        from .ghost_coating import coating_report_text
        if self.view.currentText().startswith('Coating') and self.coating:
            text = self.coating_context + '\n\n' + coating_report_text(self.coating)
        elif self.result:
            text = self.result.context + '\n\n' + self.result.summary
            text += '\n\nStack at run time:\n' + '\n'.join(self.result.layers)
        else:
            text = 'No completed run yet.'
        QMessageBox.information(self, 'Run details', text)

    def export_comparison(self):
        if self.result is None or self.view.currentText().startswith('Coating') or self.host.job_is_running():
            return
        try:
            metrics = self.performance()[self.comparison_index()]
        except ValueError as exc:
            QMessageBox.warning(self, 'Export comparison', str(exc))
            return
        # Capture every display choice before opening a file dialog or worker.
        result = self.result
        nominal, lower, upper = self.grids(self.reflection_key())
        grid = (nominal, upper, lower)[self.comparison_index()]
        options = dict(polarization=self.polarization.currentText(), reflection_key=self.reflection_key(),
                       bound=('nominal', 'upper analyzed', 'lower analyzed')[self.comparison_index()],
                       target=self.target.value(), low=self.start.value(), high=self.stop.value())
        filename, _filter = QFileDialog.getSaveFileName(self, 'Export reflection comparison',
                                                       'freddy_comparison.csv', 'CSV files (*.csv)')
        if not filename:
            return
        self.host._run_background_task('Comparison export',
            lambda: write_comparison_report(filename, result, grid, metrics, **options),
            lambda count: self.host.status_var.set(f'Exported {count:,} comparison rows to {filename}'),
            'Comparison export')

    def set_coating(self, report, context):
        self.coating, self.coating_context = report, context
        if self.view.findText('Coating error vs angle') < 0:
            self.view.addItems(['Coating error vs angle', 'Coating error map'])
        self.view.setCurrentText('Coating error vs angle')
        self.draw()

    def plot_clicked(self, event):
        if self.result is None or event.inaxes not in self.figure.axes or event.button != 1:
            return
        if event.inaxes.get_xlabel() != self.result.axis_label:
            return  # a colorbar is not a selectable thickness/angle axis
        if self.toolbar.mode or self.view.currentText() not in ('Heatmap', 'Bandwidth', 'Band coverage', 'Frequency of minimum', 'At selected frequency', 'TE/TM comparison'):
            return
        if event.xdata is not None:
            self.selected_picker.setCurrentIndex(int(np.argmin(abs(np.asarray(self.result.values) - event.xdata))))
        if self.view.currentText() in ('Heatmap', 'TE/TM comparison') and event.ydata is not None:
            self.frequency.setValue(self.result.frequencies[int(np.argmin(abs(np.asarray(self.result.frequencies) - event.ydata)))])

    def draw(self, *_):
        view = self.view.currentText()
        coating = view.startswith('Coating')
        if self.bound.currentIndex() == 3 and view not in ('Heatmap', 'Frequency curves', 'At selected frequency', 'TE/TM comparison'):
            blocker = QSignalBlocker(self.bound)
            self.bound.setCurrentIndex(0)
            del blocker
        self.table.setVisible(self.table_toggle.isChecked() and not coating)
        self.table_toggle.setText('Hide comparison table' if self.table_toggle.isChecked() else 'Show comparison table')
        self.metric.setEnabled(not coating and view in ('Heatmap', 'Frequency curves', 'At selected frequency', 'Power balance'))
        for control in (self.target, self.start, self.stop, self.selected_picker, self.table_toggle):
            control.setEnabled(not coating)
        self.polarization.setEnabled(not coating and view != 'TE/TM comparison')
        self.frequency.setEnabled(view in ('Heatmap', 'At selected frequency', 'TE/TM comparison'))
        self.bound.setEnabled(not coating and view not in ('Power balance', 'Resistance', 'Reactance', 'Tolerance envelope'))
        self.use_ibc.setEnabled(self.result is not None and not self.host.job_is_running())
        if hasattr(self, 'export_action'):
            self.export_action.setEnabled(self.result is not None and not coating and not self.host.job_is_running())
        self.status.setText(self.coating_context if coating else self.result.context if self.result else 'Run from Setup to populate these results.')
        if self.figure is None:
            return
        self.figure.clear()
        colors = self.host._colors
        self.figure.patch.set_facecolor(colors['plot_bg'])
        import textwrap
        self.figure.suptitle(textwrap.fill(self.status.text(), 150), fontsize=8, color=colors['plot_text'])
        ax = self.figure.add_subplot(111)
        try:
            if coating:
                self.draw_coating(ax)
                self.note.setText('Absolute complex-reflection difference for a planar scalar IBC. This does not certify finite-body RCS accuracy.')
            elif self.result is None:
                self.message(ax, 'Run from Setup to see the results here.')
            else:
                self.draw_result(ax, view)
                basis = 'Air' if self.reflection_key() == 'air_loss_db' else 'PEC'
                metrics = self.performance()[self.comparison_index()]
                # A view may reset a span bound above; keep visible table values
                # and CSV interpretation in step with that effective bound.
                self._populate_table(metrics)
                passing = passing_sample_ranges(self.result.values, metrics, self.target.value())
                unit = self.result.axis_label.split('(')[-1].rstrip(')')
                passed = '; '.join((f'{low:g}' if low == high else f'{low:g}–{high:g}') + f' {unit}' for low, high in passing[:5])
                if self.mode == 'Impedance' and passing:
                    passed = 'nominal stack'
                if len(passing) > 5:
                    passed += ' …'
                self.note.setText(f'{basis} reflection ≤ {self.target.value():g} dB over {self.start.value():g}–{self.stop.value():g} GHz: '
                                  + (f'passing sampled ranges: {passed}.' if passing else 'no sampled choice passes the entire band.')
                                  + '\nRanges group passing samples; verify between them. Bandwidth interpolates frequencies; tolerance bounds are not yield.'
                                  + (' Span view: comparison uses nominal reflection.' if self.bound.currentIndex() == 3 else '')
                                  + (' Curves are reduced for display only.' if len(self.result.frequencies) > 5000 else ''))
        except (ValueError, KeyError, IndexError) as exc:
            ax.clear()
            self.message(ax, str(exc))
            self.note.setText('Choose an available view and a band within the completed run.')
        for axis in self.figure.axes:
            style_axis(axis, colors)
            axis.title.set_fontsize(11)
            axis.xaxis.label.set_fontsize(10)
            axis.yaxis.label.set_fontsize(10)
            axis.tick_params(labelsize=9)
            legend = axis.get_legend()
            if legend:
                legend.get_frame().set_facecolor(colors['plot_axes_bg'])
                legend.get_frame().set_edgecolor(colors['plot_spine'])
                for text in legend.get_texts():
                    text.set_color(colors['plot_text'])
        self.toolbar.update()
        self.canvas.draw_idle()

    def message(self, ax, text):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(.5, .5, text, transform=ax.transAxes, ha='center', va='center', color=self.host._colors['text'], wrap=True)

    def heatmap(self, ax, data, title, *, limits=None, threshold=False):
        r = self.result
        if len(r.frequencies) < 2 or len(r.values) < 2:
            raise ValueError('A heatmap needs at least two frequencies and two sweep values. Use Frequency curves or At selected frequency.')
        cmap = 'magma' if self.bound.currentIndex() == 3 else 'twilight' if 'phase' in title.lower() else 'viridis'
        mesh = ax.pcolormesh(r.values, r.frequencies, data, shading='nearest', cmap=cmap,
                             vmin=limits[0] if limits else None, vmax=limits[1] if limits else None)
        colorbar = self.figure.colorbar(mesh, ax=ax, pad=.02)
        style_colorbar(colorbar, self.host._colors)
        if threshold and np.min(data) < self.target.value() < np.max(data):
            contour = ax.contour(r.values, r.frequencies, data, levels=[self.target.value()], colors=['#ffffff'], linewidths=1.2)
            ax.clabel(contour, fmt=lambda v: f'{v:g} dB', fontsize=8)
        ax.axvline(r.values[self.selected], color='#eaa33e', linestyle='--', linewidth=1)
        ax.set(title=title, xlabel=r.axis_label, ylabel='Frequency (GHz)',
               xlim=(r.values[0], r.values[-1]), ylim=(r.frequencies[0], r.frequencies[-1]))

    def draw_result(self, ax, view):
        r = self.result
        f = r.frequencies
        nominal, lower, upper = self.grids()
        grid = self.display_grid(nominal, lower, upper)
        key = METRICS.get(self.metric.currentText(), 'metal_loss_db')
        pol = self.polarization.currentText()
        if view == 'Heatmap':
            self.heatmap(ax, grid, f'{self.metric.currentText()} · {pol} · {self.bound.currentText()}', threshold=key in ('metal_loss_db', 'air_loss_db') and self.bound.currentIndex() != 3)
        elif view == 'TE/TM comparison':
            if 'TM comparison unavailable' in r.summary:
                raise ValueError('TM comparison unavailable for this directional stack at oblique angles. Use TE or a full tensor material model.')
            if not all(p in r.metrics for p in ('TE', 'TM')):
                raise ValueError('Enable “Compute TE and TM comparison” on Setup and run again.')
            if len(r.frequencies) < 2 or len(r.values) < 2:
                raise ValueError('TE/TM maps need at least two frequencies and two angles. Select a polarization and use Frequency curves.')
            self.figure.clear()
            self.figure.suptitle(r.context, fontsize=8, color=self.host._colors['plot_text'])
            axes = self.figure.subplots(1, 2, sharex=True, sharey=True)
            key = self.reflection_key()
            grids = [self.display_grid(*self.grids(key, p)) for p in ('TE', 'TM')]
            limits = min(np.min(g) for g in grids), max(np.max(g) for g in grids)
            for axis, data, label in zip(axes, grids, ('TE', 'TM')):
                self.heatmap(axis, data, f'{label} · {"PEC" if key.startswith("metal") else "Air"} reflection (dB) · {self.bound.currentText()}', limits=limits, threshold=self.bound.currentIndex() != 3)
        elif view in ('Bandwidth', 'Band coverage'):
            nom, worst = self.performance()[:2]
            for data, label, color in [(nom, 'Nominal', PALETTE[0]), (worst, 'Worst analyzed reflection', PALETTE[1])]:
                y = [m.widest_ghz if view == 'Bandwidth' else m.coverage_pct for m in data]
                if view == 'Bandwidth' and any(v is None for v in y):
                    raise ValueError('A single frequency has no bandwidth. Use Band coverage for passing-point coverage.')
                ax.plot(r.values, y, label=label, color=color, lw=2, marker='.' if len(y) < 80 else None)
                if label == 'Nominal' and not r.upper:
                    break
            ax.set(xlabel=r.axis_label, ylabel='Widest passing band (GHz)' if view == 'Bandwidth' else 'Band coverage (%)',
                   title=f'Reflection ≤ {self.target.value():g} dB · {self.start.value():g}–{self.stop.value():g} GHz · {pol}')
            ax.legend(fontsize=8)
        elif view == 'Frequency of minimum':
            nominal, lower, upper = self.grids(self.reflection_key())
            grid = (nominal, upper, lower)[self.comparison_index()]
            locations = np.asarray(f)[np.argmin(grid, axis=0)]
            ax.plot(r.values, locations, color=PALETTE[0], marker='.', lw=1.5)
            ax.set(xlabel=r.axis_label, ylabel='Frequency of sampled minimum (GHz)', title='Track the reflection minimum; compare bandwidth before choosing a design')
        elif view == 'At selected frequency':
            index = int(np.argmin(abs(np.asarray(f) - self.frequency.value())))
            ax.plot(r.values, grid[index, :], color=PALETTE[0], marker='.', lw=2)
            ax.set(xlabel=r.axis_label, ylabel=self.metric.currentText(), title=f'{self.metric.currentText()} at {f[index]:g} GHz · {pol}')
            if key in ('metal_loss_db', 'air_loss_db') and self.bound.currentIndex() != 3:
                ax.axhline(self.target.value(), linestyle='--', color=self.host._colors['plot_text'])
        elif view == 'Tolerance envelope':
            nominal, lower, upper = self.grids(self.reflection_key())
            ax.fill_between(f, lower[:, self.selected], upper[:, self.selected], color=PALETTE[0], alpha=.2, label='Analyzed tolerance range')
            ax.plot(f, nominal[:, self.selected], color=PALETTE[0], lw=2, label='Nominal')
            ax.plot(f, upper[:, self.selected], color=PALETTE[1], lw=1.5, label='Worst analyzed reflection')
            ax.axhline(self.target.value(), linestyle='--', color=self.host._colors['plot_text'])
            ax.set(xlabel='Frequency (GHz)', ylabel='Reflection (dB)', title=f'{self.selection_label(self.selected)} · {pol} · tolerance envelope')
            ax.legend(fontsize=8)
        elif view in ('Resistance', 'Reactance'):
            if r.impedance is None:
                raise ValueError('No impedance values were retained for this run.')
            data = np.real(r.impedance) if view == 'Resistance' else np.imag(r.impedance)
            self.curves(ax, data, f'{view} (Ω)', view)
            component = 'real' if view == 'Resistance' else 'imag'
            if r.impedance_bounds:
                ax.fill_between(f, r.impedance_bounds[component + '_min'], r.impedance_bounds[component + '_max'], color=PALETTE[0], alpha=.2)
        elif view == 'Power balance':
            metric = r.metrics[pol]
            backing = r.backing if self.mode == 'Impedance' else 'air' if self.reflection_key() == 'air_loss_db' else 'pec'
            prefix = 'metal' if backing == 'pec' else 'air'
            reflection = 100 * 10 ** (metric[prefix + '_loss_db'][:, self.selected] / 10)
            absorption = 100 * 10 ** (metric[prefix + '_absorption_db'][:, self.selected] / 10)
            transmission = np.zeros(len(f)) if backing == 'pec' else 100 * 10 ** (metric['insertion_loss_db'][:, self.selected] / 10)
            for values, label, color in zip((reflection, absorption, transmission), ('Reflected', 'Absorbed', 'Transmitted'), PALETTE):
                ax.plot(f, values, label=label, color=color, lw=2)
            ax.set(xlabel='Frequency (GHz)', ylabel='Incident power (%)', ylim=(-1, 101), title=f'{backing.upper()} backing · {pol} · nominal power balance')
            ax.legend(fontsize=8)
        else:
            self.curves(ax, grid, self.metric.currentText(), f'{self.metric.currentText()} · {pol} · {self.bound.currentText()}')
            if key in ('metal_loss_db', 'air_loss_db') and self.bound.currentIndex() != 3:
                ax.axhline(self.target.value(), linestyle='--', color=self.host._colors['plot_text'], linewidth=1)
                performance = self.performance()[self.comparison_index()][self.selected]
                for low, high in performance.passing_bands:
                    ax.axvspan(low, high, color=PALETTE[self.selected % len(PALETTE)], alpha=.12)
        ax.grid(True, alpha=.15)

    def curves(self, ax, grid, ylabel, title):
        for index in sorted(self.checked | {self.selected}):
            indices = extrema_preserving_indices(grid[:, index])
            ax.plot(np.asarray(self.result.frequencies)[indices], grid[indices, index], color=PALETTE[index % len(PALETTE)],
                    lw=2.5 if index == self.selected else 1.3, alpha=1 if index == self.selected else .7,
                    label=self.selection_label(index) + (' selected' if index == self.selected else ''))
        ax.set(xlabel='Frequency (GHz)', ylabel=ylabel, title=title)
        ax.legend(fontsize=8, ncol=min(5, max(1, len(self.checked) // 3)))

    def draw_coating(self, ax):
        if not self.coating:
            raise ValueError('Run Check GHOST coating from Impedance Setup first.')
        report = self.coating
        if self.view.currentText() == 'Coating error vs angle':
            for pol, color in zip(('TE', 'TM'), PALETTE):
                rows = [r for r in report['angles'] if r['polarization'] == pol]
                ax.plot([r['angle_deg'] for r in rows], [r['max_absolute_complex_reflection_error'] for r in rows], label=pol, color=color, marker='o')
            ax.set(xlabel='Incidence angle (deg)', ylabel='Maximum |ΔΓ| across frequency', title='Scalar IBC versus full planar stack')
            ax.legend(fontsize=8)
        else:
            freqs, angles = report['frequencies_ghz'], report['sample_angles_deg']
            if len(freqs) < 2 or len(angles) < 2:
                raise ValueError('Use the error-versus-angle view for a single-frequency check.')
            self.figure.clear()
            self.figure.suptitle(self.coating_context, fontsize=8, color=self.host._colors['plot_text'])
            axes = self.figure.subplots(1, 2, sharex=True, sharey=True)
            vmax = max(np.max(report['error_grids'][p]) for p in ('TE', 'TM'))
            for axis, pol in zip(axes, ('TE', 'TM')):
                image = axis.pcolormesh(angles, freqs, report['error_grids'][pol], shading='nearest', vmin=0, vmax=vmax, cmap='magma')
                cb = self.figure.colorbar(image, ax=axis, pad=.02)
                style_colorbar(cb, self.host._colors)
                axis.set(xlabel='Incidence angle (deg)', ylabel='Frequency (GHz)', title=f'{pol} · absolute |ΔΓ|')
