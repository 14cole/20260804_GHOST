"""Full-width inverse-design results and candidate comparison controls."""
from __future__ import annotations

from pathlib import Path

from .inverse_performance import band_performance, response_curve, search_progress


class InverseResultsMixin:
    def _create_inverse_candidate_table(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget

        class CandidateTable(QTableWidget):
            def setCurrentRow(self, candidate_index):
                # Keep original candidate identities when the user sorts columns.
                for row in range(self.rowCount()):
                    if self.item(row, 1).data(Qt.UserRole) == candidate_index:
                        self.setCurrentCell(row, 1)
                        return

        table = CandidateTable(0, 9)
        table.setHorizontalHeaderLabels(['Plot', 'Candidate', 'Score (dB)', 'Coverage (%)',
                                          'Widest band (GHz)', 'Deepest (dB)', 'Worst point (dB)', 'Thickness (in)', 'Req. margin (dB)'])
        table.setColumnHidden(8, True)
        table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(25)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.setMinimumHeight(90)
        table.setSortingEnabled(True)
        table.sortItems(1, Qt.AscendingOrder)
        table.itemSelectionChanged.connect(self._update_inverse_plot)
        table.itemChanged.connect(self._update_inverse_plot)
        return table

    def _build_inverse_results_workspace(self, work_split, root_layout):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                                      QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget, QStackedWidget)
        self.inverse_result_metadata = {}
        self._inverse_metrics = []
        self._inverse_curves = []
        self._inverse_summary = ''
        self._inverse_page_index = 0
        self.inverse_workspace_tabs = QTabWidget()
        self.inverse_workspace_tabs.setDocumentMode(True)
        self.inverse_workspace_tabs.addTab(work_split, 'Setup')
        panel = QWidget()
        self.result_pages = QStackedWidget()
        self.result_pages.addWidget(panel)
        self.inverse_workspace_tabs.addTab(self.result_pages, 'Results')
        root_layout.addWidget(self.inverse_workspace_tabs, 1)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(5)

        top = QHBoxLayout()
        self.inv_result_status = QLabel('Analyze all combinations from Setup to compare candidates here.')
        self.inv_result_status.setWordWrap(True)
        top.addWidget(self.inv_result_status, 1)
        details = QPushButton('Run details…')
        details.clicked.connect(self._show_inverse_run_details)
        top.addWidget(details)
        explain = QPushButton('Explain view')
        explain.clicked.connect(lambda: self._show_guide('plots-inverse'))
        top.addWidget(explain)
        layout.addLayout(top)
        self.inv_requirement_status = QLabel()
        self.inv_requirement_status.setWordWrap(True)
        layout.addWidget(self.inv_requirement_status)

        controls = QHBoxLayout()
        self.inv_plot_view = QComboBox()
        self.inv_plot_view.addItems(['Reflection curves', 'Depth vs bandwidth', 'Analysis history',
                                    'Selected candidate angle map', 'Selected candidate tolerance'])
        self.inv_inspect_angle = QComboBox()
        self.inv_inspect_angle.setToolTip('Angle from the completed run, used by the selected candidate tolerance view.')
        self.inv_inspect_angle.currentIndexChanged.connect(self._update_inverse_plot)
        self.inv_curve_mode = QComboBox()
        self.inv_curve_mode.addItems(['Worst analyzed case', 'Point percentile'])
        self.inv_curve_mode.setToolTip('Worst analyzed case uses the highest reflection at each frequency across the run’s angles and tolerance corners.')
        self.inv_target_db = QDoubleSpinBox()
        self.inv_target_db.setRange(-200, 0)
        self.inv_target_db.setDecimals(1)
        self.inv_target_db.setValue(-10)
        self.inv_target_db.setSuffix(' dB')
        self.inv_target_db.setToolTip('Passing means reflection is at or below this target. This changes the comparison, not the search objective.')
        self.inv_percentile_entry.setMaximumWidth(50)
        self.inv_percentile_entry.setToolTip('Percentile across analyzed angles and tolerance corners. Lower percentiles are more optimistic; this is not manufacturing yield.')
        for widget in (QLabel('View'), self.inv_plot_view, self.inv_curve_mode,
                       self.inv_percentile_entry, self.inv_inspect_angle, QLabel('Target'), self.inv_target_db):
            controls.addWidget(widget)
        controls.addStretch(1)
        self.inv_table_toggle = QPushButton('Show candidate table')
        self.inv_table_toggle.setCheckable(True)
        self.inv_table_toggle.setToolTip('Compare metrics, sort candidates, and choose which curves to overlay. Hide the table for a larger plot.')
        controls.addWidget(self.inv_table_toggle)
        layout.addLayout(controls)
        self.inv_results_split = QSplitter(Qt.Vertical)
        layout.addWidget(self.inv_results_split, 1)
        plot = QWidget()
        plot_layout = QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        self.inv_figure = self.inv_canvas = self.inv_axis = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            class ResultsToolbar(NavigationToolbar2QT):
                toolitems = tuple(item for item in NavigationToolbar2QT.toolitems
                                  if item[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom', None))

            self.inv_figure = Figure(figsize=(10, 4), dpi=100, layout='constrained')
            self.inv_axis = self.inv_figure.add_subplot(111)
            self.inv_canvas = FigureCanvasQTAgg(self.inv_figure)
            self.inv_canvas.setMinimumSize(200, 200)
            toolbar = ResultsToolbar(self.inv_canvas, self)
            toolbar.setMaximumHeight(32)
            save = toolbar.addAction('Save plot…')
            save.triggered.connect(self._save_plot)
            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(self.inv_canvas, 1)
            self.inv_canvas.mpl_connect('pick_event', self._pick_inverse_candidate)
        except ImportError:
            plot_layout.addWidget(QLabel('Install matplotlib to enable plots. Candidate scores remain available below.'))
        self.inv_results_split.addWidget(plot)
        self.inv_results_split.addWidget(self.inv_results_list)
        self.inv_results_split.setStretchFactor(0, 1)
        self.inv_results_split.setStretchFactor(1, 0)
        self.inv_results_split.setSizes([460, 115])
        self.inv_result_note = QLabel()
        self.inv_result_note.setWordWrap(True)
        layout.addWidget(self.inv_result_note)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel('Selected'))
        self.inv_candidate_picker = QComboBox()
        self.inv_candidate_picker.setMinimumWidth(65)
        self.inv_candidate_picker.setToolTip('Choose the highlighted candidate to inspect, apply, or save. Open the candidate table to change plot overlays.')
        bottom.addWidget(self.inv_candidate_picker)
        self.inv_selected_description = QLabel('Select a row to inspect a candidate; check Plot to overlay it.')
        self.inv_selected_description.setMinimumWidth(0)
        from PySide6.QtWidgets import QSizePolicy
        self.inv_selected_description.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bottom.addWidget(self.inv_selected_description, 1)
        for button in (self.inv_apply_btn, self.inv_save_candidate_btn, self.inv_extend_btn):
            bottom.addWidget(button)
        layout.addLayout(bottom)
        self.inverse_workspace_tabs.currentChanged.connect(self._inverse_workspace_changed)
        self.inv_plot_view.currentIndexChanged.connect(self._update_inverse_plot)
        self.inv_curve_mode.currentIndexChanged.connect(self._refresh_inverse_results_list)
        self.inv_target_db.valueChanged.connect(self._refresh_inverse_results_list)
        self.inv_table_toggle.toggled.connect(self._update_inverse_plot)
        self.inv_candidate_picker.currentIndexChanged.connect(self._select_inverse_from_picker)

    def _inverse_workspace_changed(self, index):
        if self._is_inverse_tab_active():
            self._inverse_page_index = index
            if index == 1:
                self._update_inverse_plot()
        else:
            self._analysis_workspace_changed(index)

    def _selected_inverse_index(self):
        from PySide6.QtCore import Qt
        table = self.inv_results_list
        row = table.currentRow() if table is not None else -1
        return int(table.item(row, 1).data(Qt.UserRole)) if row >= 0 and table.item(row, 1) else -1

    def _inverse_checked_indices(self):
        from PySide6.QtCore import Qt
        table = self.inv_results_list
        return {int(table.item(row, 1).data(Qt.UserRole)) for row in range(table.rowCount())
                if table.item(row, 0).checkState() == Qt.Checked}

    def _refresh_inverse_results_list(self, *_args):
        from PySide6.QtCore import Qt, QSignalBlocker
        from PySide6.QtWidgets import QTableWidgetItem

        class NumericItem(QTableWidgetItem):
            def __lt__(self, other):
                return self.data(Qt.UserRole + 1) < other.data(Qt.UserRole + 1)

        if not hasattr(self, 'inv_target_db'):
            return
        table = self.inv_results_list
        selected = self._selected_inverse_index()
        checked = self._inverse_checked_indices() if table.rowCount() else set(range(min(5, len(self.inverse_candidates))))
        blocker = QSignalBlocker(table)
        table.setSortingEnabled(False)
        table.setRowCount(len(self.inverse_candidates))
        requirement = self.inverse_result_metadata.get('requirement_db')
        table.horizontalHeaderItem(2).setText('Gap (dB)' if requirement is not None else 'Score (dB)')
        table.setColumnHidden(8, requirement is None)
        percentile = self._current_inverse_percentile() if self.inv_curve_mode.currentIndex() else 100.
        self._inverse_curves, self._inverse_metrics = [], []
        continuous = self.inverse_result_metadata.get('band_sweep', False)
        available = len(self.inverse_plot_samples) == len(self.inverse_candidates) and bool(self.inverse_plot_freqs)
        for index, candidate in enumerate(self.inverse_candidates):
            curve = metric = None
            if available:
                try:
                    curve = response_curve(self.inverse_plot_samples[index], percentile)
                    metric = band_performance(self.inverse_plot_freqs, curve, self.inv_target_db.value(), continuous=continuous)
                except ValueError:
                    curve = None
            self._inverse_curves.append(curve)
            self._inverse_metrics.append(metric)
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            check.setCheckState(Qt.Checked if index in checked else Qt.Unchecked)
            table.setItem(index, 0, check)
            values = [index + 1, candidate.score_db, metric.coverage_pct if metric else None,
                      metric.widest_ghz if metric else None, metric.deepest_db if metric else None,
                      metric.worst_db if metric else None, sum(candidate.thickness_in),
                      -candidate.score_db if requirement is not None else None]
            for col, value in enumerate(values, 1):
                text = ('—' if value is None else f'#{value}' if col == 1 else
                        f'{value:.4g}' if col in (4, 7) else f'{value:.2f}')
                item = NumericItem(text)
                item.setData(Qt.UserRole, index)
                item.setData(Qt.UserRole + 1, float('inf') if value is None else value)
                if col == 1:
                    item.setToolTip(self._inverse_candidate_description(index))
                elif col == 2:
                    item.setToolTip(f'Captured search gap = worst PEC reflection − ({requirement:g} dB) across every analyzed frequency/angle/tolerance case. Lower is better; ≤ 0 passes.'
                                    if requirement is not None else 'Original search objective: mean dB across frequency and angle, then worst or average across tolerance corners. Lower is better.')
                elif col == 8:
                    item.setToolTip('Captured search requirement margin = −gap. Nonnegative passes every analyzed condition. This does not change with Results target or percentile.')
                elif col == 3:
                    item.setToolTip('Estimated fraction of the sampled band at/below target.' if continuous and len(self.inverse_plot_freqs) > 1
                                    else 'Fraction of discrete target frequencies at/below target; no bandwidth is inferred.')
                elif col == 4:
                    item.setToolTip(f'Estimated passing interval: {metric.widest_band[0]:.5g}–{metric.widest_band[1]:.5g} GHz'
                                    if metric and metric.widest_ghz else 'No positive passing bandwidth, or unavailable for discrete targets.')
                table.setItem(index, col, item)
        table.setSortingEnabled(True)
        if self.inverse_candidates:
            table.setCurrentRow(max(0, min(selected, len(self.inverse_candidates) - 1)))
        del blocker
        blocker = QSignalBlocker(self.inv_candidate_picker)
        self.inv_candidate_picker.clear()
        for index in range(len(self.inverse_candidates)):
            self.inv_candidate_picker.addItem(f'#{index + 1}', index)
        self.inv_candidate_picker.setCurrentIndex(self._selected_inverse_index())
        del blocker
        blocker = QSignalBlocker(self.inv_inspect_angle)
        previous = self.inv_inspect_angle.currentData()
        self.inv_inspect_angle.clear()
        for angle in self.inverse_result_metadata.get('angles', []):
            self.inv_inspect_angle.addItem(f'{angle:g}°', angle)
        found = self.inv_inspect_angle.findData(previous)
        if found >= 0:
            self.inv_inspect_angle.setCurrentIndex(found)
        del blocker
        self._update_inverse_plot()

    def _select_inverse_from_picker(self, index):
        if index >= 0:
            self.inv_results_list.setCurrentRow(index)

    def _inverse_candidate_description(self, index):
        candidate = self.inverse_candidates[index]
        layers = []
        for i, thickness in enumerate(candidate.thickness_in):
            resistance = candidate.sheet_resistance_ohm[i]
            labels = self.inverse_result_metadata.get('layer_labels', [])
            name = labels[i] if i < len(labels) else Path(candidate.material_files[i]).name if candidate.material_files[i] else 'Sheet' if resistance > 0 else 'Constant material'
            layers.append(f'{name}: {resistance:g} Ω/sq' if resistance > 0 else f'{name}: {thickness:.4g} in')
        requirement = self.inverse_result_metadata.get('requirement_db')
        verdict = (f'{"PASS" if candidate.score_db <= 0 else "MISS"} · margin {-candidate.score_db:+.3f} dB · '
                   if requirement is not None else '')
        return f'#{index + 1} · ' + verdict + '; '.join(layers)

    def _show_inverse_run_details(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, 'Inverse Design — run details', self._inverse_summary or 'No completed run yet.')

    def _pick_inverse_candidate(self, event):
        indices = getattr(event.artist, '_inverse_indices', ())
        picked = getattr(event, 'ind', ())
        if len(picked) and len(indices) > picked[0]:
            self.inv_results_list.setCurrentRow(indices[picked[0]])

    def _update_inverse_plot(self, *_args):
        from PySide6.QtCore import QSignalBlocker
        if not hasattr(self, 'inv_plot_view'):
            return
        view = self.inv_plot_view.currentIndex()
        is_history = view == 2
        self.inv_inspect_angle.setVisible(view == 4)
        self.inv_results_list.setVisible(self.inv_table_toggle.isChecked() and not is_history)
        self.inv_table_toggle.setEnabled(not is_history)
        self.inv_table_toggle.setText('Hide candidate table' if self.inv_table_toggle.isChecked() else 'Show candidate table')
        self.inv_curve_mode.setEnabled(view in (0, 1))
        self.inv_target_db.setEnabled(not is_history)
        self.inv_percentile_entry.setVisible(self.inv_curve_mode.currentIndex() == 1 and view in (0, 1))
        selected = self._selected_inverse_index()
        blocker = QSignalBlocker(self.inv_candidate_picker)
        self.inv_candidate_picker.setCurrentIndex(selected)
        del blocker
        self.inv_selected_description.setText(self._inverse_candidate_description(selected) if selected >= 0 else 'Select a candidate.')
        self.inv_selected_description.setToolTip(self.inv_selected_description.text())
        requirement = self.inverse_result_metadata.get('requirement_db')
        self.inv_requirement_status.setVisible(requirement is not None)
        self.inv_requirement_status.setText(
            f'Captured search requirement: PEC reflection ≤ {requirement:g} dB at every analyzed frequency, angle, and tolerance case. Gap ≤ 0 passes. Results target/percentile only changes comparison plots.'
            if requirement is not None else '')
        continuous = self.inverse_result_metadata.get('band_sweep', False) and len(self.inverse_plot_freqs) > 1
        note = ('Band widths use linear interpolation between samples; use a finer sweep to verify narrow features.' if continuous else
                'Discrete targets: coverage counts passing frequencies; no bandwidth between targets is inferred.')
        ranking = 'worst-point requirement gap' if requirement is not None else 'mean reflection dB'
        self.inv_result_note.setText(f'History includes each completed combination. A completed run minimizes {ranking} on the configured grid; untested values between steps are not covered.' if is_history else
                                    note + f'\nComparison covers retained candidates. Analysis ranks {ranking}; increase Keep best to inspect more alternatives.')
        if self.inv_axis is None:
            return
        axis, colors = self.inv_axis, self._colors
        # The angle map has a colorbar; recreate the axis to remove it when
        # changing views and avoid shrinking the plotting area on every draw.
        self.inv_figure.clear()
        axis = self.inv_axis = self.inv_figure.add_subplot(111)
        self.inv_figure.patch.set_facecolor(colors['plot_bg'])
        self._style_plot_axis(axis)
        axis.grid(True, color=colors['plot_grid'], alpha=.3)
        if is_history:
            scores, best = search_progress(list(self.inverse_result_metadata.get('scores', [])))
            if scores:
                x = list(range(1, len(scores) + 1))
                axis.scatter(x, scores, s=12, alpha=.3, color=colors['plot_line_freq'], label='Completed designs')
                axis.plot(x, best, color=colors['plot_line_angle'], linewidth=2, label='Best so far')
                axis.set_title(f'Combination scores · {len(scores):,} unique completed designs')
                axis.set_xlabel('Completed combination (fixed grid order)')
                axis.set_ylabel('Worst-point gap (dB; ≤ 0 passes)' if requirement is not None else 'Scoring objective (dB; lower is better)')
                if requirement is not None:
                    axis.axhline(0., color=colors['plot_text'], linestyle='--', linewidth=1, label='Requirement boundary')
                axis.legend(fontsize=9)
            else:
                self._inverse_plot_message('Analyze the configured combinations to see their scores.')
        elif not self.inverse_candidates:
            self._inverse_plot_message('Run inverse design from Setup to compare candidate stacks.')
        elif not any(m is not None for m in self._inverse_metrics):
            self._inverse_plot_message('Response curves are unavailable for this run.\nScores remain in the table and Analysis history.\nResume the analysis to compute complete curves.')
        elif view in (3, 4):
            self._plot_inverse_cases(selected, view)
        elif view == 0:
            self._plot_inverse_response(selected, continuous)
        else:
            self._plot_inverse_bandwidth(selected, continuous)
        axis.title.set_fontsize(12)
        axis.xaxis.label.set_fontsize(10)
        axis.yaxis.label.set_fontsize(10)
        axis.tick_params(labelsize=9)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(colors['plot_axes_bg'])
            legend.get_frame().set_edgecolor(colors['plot_spine'])
            for text in legend.get_texts():
                text.set_color(colors['plot_text'])
        self.inv_canvas.draw_idle()

    def _plot_inverse_cases(self, selected, view):
        import numpy as np
        from .analysis_data import inverse_case_grid
        from .plot import style_colorbar
        angles = self.inverse_result_metadata.get('angles', [])
        scales = self.inverse_result_metadata.get('scales', [])
        try:
            cases = inverse_case_grid(self.inverse_plot_samples[selected], angles, scales)
            axis = self.inv_axis
            frequencies = self.inverse_plot_freqs
            if view == 3:
                if len(angles) < 2 or len(frequencies) < 2:
                    raise ValueError('An angle map needs at least two frequencies and angles.\nUse Selected candidate tolerance for this run.')
                grid = cases.max(axis=1)
                mesh = axis.pcolormesh(angles, frequencies, grid, shading='nearest', cmap='viridis')
                colorbar = self.inv_figure.colorbar(mesh, ax=axis, pad=.02)
                style_colorbar(colorbar, self._colors)
                if np.min(grid) < self.inv_target_db.value() < np.max(grid):
                    contour = axis.contour(angles, frequencies, grid, levels=[self.inv_target_db.value()], colors=['white'])
                    axis.clabel(contour, fmt=lambda value: f'{value:g} dB', fontsize=8)
                axis.set(xlabel='Incidence angle (deg)', ylabel='Frequency (GHz)',
                         title=f'Candidate #{selected + 1} · PEC reflection (dB) · worst analyzed tolerance at each angle',
                         xlim=(angles[0], angles[-1]), ylim=(frequencies[0], frequencies[-1]))
                self.inv_result_note.setText('Uses the angles and tolerance cases captured with this run. Threshold boundaries interpolate the sampled grid.')
            else:
                angle_index = max(0, self.inv_inspect_angle.currentIndex())
                nominal_index = next(i for i, scale in enumerate(scales) if all(abs(v - 1) < 1e-12 for v in scale))
                selected_cases = cases[:, :, angle_index]
                axis.fill_between(frequencies, selected_cases.min(axis=1), selected_cases.max(axis=1), color=self._colors['plot_line_freq'], alpha=.2, label='Analyzed tolerance range')
                axis.plot(frequencies, selected_cases[:, nominal_index], color=self._colors['plot_line_freq'], lw=2, label='Nominal')
                axis.plot(frequencies, selected_cases.max(axis=1), color=self._colors['plot_line_angle'], lw=1.5, label='Worst analyzed reflection')
                axis.axhline(self.inv_target_db.value(), color=self._colors['plot_text'], ls='--', linewidth=1)
                axis.set(xlabel='Frequency (GHz)', ylabel='PEC reflection (dB)', title=f'Candidate #{selected + 1} · {angles[angle_index]:g}° · tolerance envelope')
                axis.legend(fontsize=8)
                self.inv_result_note.setText('Systematic tolerance cases at the selected angle from this run. This is not a manufacturing-yield or confidence interval.')
        except (ValueError, IndexError, StopIteration) as exc:
            self._inverse_plot_message(str(exc) or 'Labeled nominal tolerance data are unavailable. Analyze this setup to generate them.')

    def _inverse_plot_message(self, text):
        self.inv_axis.set_xticks([])
        self.inv_axis.set_yticks([])
        self.inv_axis.text(.5, .5, text, ha='center', va='center', transform=self.inv_axis.transAxes,
                          color=self._colors['text'], fontsize=11)

    def _plot_inverse_response(self, selected, continuous):
        axis = self.inv_axis
        shown = sorted(self._inverse_checked_indices() | ({selected} if selected >= 0 else set()))
        palette = ['#4ba3ed', '#f0a34a', '#55bf9a', '#d889d3', '#e57373', '#adac52', '#9d9af2', '#61c8d8']
        for index in shown:
            curve = self._inverse_curves[index]
            if curve is None:
                continue
            color = palette[index % len(palette)]
            axis.plot(self.inverse_plot_freqs, curve, color=color, linewidth=2.5 if index == selected else 1.4,
                      alpha=1 if index == selected else .8, marker='.' if continuous else 'o',
                      linestyle='-' if continuous else 'None', markersize=3 if continuous else 6,
                      label=f'#{index + 1}' + (' selected' if index == selected else ''))
            if index == selected and continuous:
                metric = self._inverse_metrics[index]
                for lo, hi in metric.passing_bands:
                    axis.axvspan(lo, hi, color=color, alpha=.12)
        target = self.inv_target_db.value()
        axis.axhline(target, linestyle='--', color=self._colors['text'], alpha=.75, linewidth=1, label=f'Target {target:g} dB')
        mode = 'Worst analyzed case' if self.inv_curve_mode.currentIndex() == 0 else f'P{self._current_inverse_percentile():g} across analyzed cases'
        axis.set_title(f'PEC reflection · {mode}')
        axis.set_xlabel('Frequency (GHz)' + (' · shaded bands meet target for selected candidate' if continuous else ' · discrete targets'))
        axis.set_ylabel('PEC reflection |Γ| (dB)')
        axis.legend(loc='best', ncol=min(5, max(1, (len(shown) + 1) // 2)), fontsize=8)

    def _plot_inverse_bandwidth(self, selected, continuous):
        axis = self.inv_axis
        indices = [i for i, metric in enumerate(self._inverse_metrics) if metric is not None]
        xs = [self._inverse_metrics[i].widest_ghz if continuous else self._inverse_metrics[i].coverage_pct for i in indices]
        ys = [self._inverse_metrics[i].deepest_db for i in indices]
        points = axis.scatter(xs, ys, s=70, color=self._colors['plot_line_freq'], alpha=.75, picker=6)
        points._inverse_indices = indices
        for i, x, y in zip(indices, xs, ys):
            axis.annotate(f'#{i + 1}', (x, y), xytext=(5, 7), textcoords='offset points', fontsize=8, color=self._colors['text'])
        if selected in indices:
            j = indices.index(selected)
            axis.scatter([xs[j]], [ys[j]], s=160, marker='*', color=self._colors['plot_line_angle'], zorder=4)
        axis.margins(x=.15, y=.18)
        axis.set_title(f'Null depth vs useful {"bandwidth" if continuous else "target coverage"} · target {self.inv_target_db.value():g} dB')
        axis.set_xlabel('Widest continuous passing band (GHz; farther right is wider)' if continuous else 'Discrete targets meeting threshold (%; farther right is more coverage)')
        axis.set_ylabel('Deepest reflection (dB)')
