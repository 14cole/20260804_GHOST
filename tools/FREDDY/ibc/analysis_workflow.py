"""Connect run-owned sweep results to FREDDY navigation and validated exports."""
from pathlib import Path


def stack_description(layers):
    return [f'Layer {i}: sheet {layer.sheet_resistance:g} Ω/sq' if layer.is_sheet else
            f'Layer {i}: {layer.thickness_in:g} in; {layer.file_0deg}' + (f'; 90° {layer.file_90deg}' if layer.anisotropic else '')
            for i, layer in enumerate(layers, 1)]


def tolerance_description(config):
    return (f'tolerances T ±{config.thickness_pct:g}%, ε ±{config.eps_pct:g}%, μ ±{config.mu_pct:g}%'
            if config.enabled else 'nominal')


class AnalysisWorkflowMixin:
    def _build_analysis_workspaces(self):
        from .sweep_results import SweepResultsPanel
        self.analysis_panels = {}
        self._analysis_page_indices = {}
        for mode in ('Impedance', 'IBC Batch', 'Thickness', 'Off Angle'):
            panel = SweepResultsPanel(self, mode)
            self.result_pages.addWidget(panel)
            self.analysis_panels[mode] = panel
            self._analysis_page_indices[mode] = 0

    def _active_analysis_panel(self):
        return getattr(self, 'analysis_panels', {}).get(self._active_left_tab_label())

    def _sync_analysis_chrome(self):
        from PySide6.QtCore import QSignalBlocker
        panel = self._active_analysis_panel()
        if panel is None:
            if hasattr(self, 'result_pages'):
                self.result_pages.setCurrentIndex(0)
            return False
        self.result_pages.setCurrentWidget(panel)
        blocker = QSignalBlocker(self.inverse_workspace_tabs)
        self.inverse_workspace_tabs.setTabVisible(1, True)
        self.inverse_workspace_tabs.tabBar().show()
        self.inverse_workspace_tabs.setCurrentIndex(self._analysis_page_indices[self._active_left_tab_label()])
        del blocker
        if not self.results_pane.isHidden():
            self._solver_split_sizes = self.work_split.sizes()
        self.results_pane.hide()
        self.layers_group.show()
        return True

    def _analysis_workspace_changed(self, index):
        panel = self._active_analysis_panel()
        if panel is not None:
            self._analysis_page_indices[panel.mode] = index
            if index == 1:
                panel.draw()

    def _show_analysis_result(self, mode, result):
        panel = self.analysis_panels[mode]
        panel.set_result(result)
        self._open_analysis_results(mode)

    def _open_analysis_results(self, mode):
        self._analysis_page_indices[mode] = 1
        if self._active_left_tab_label() == mode:
            self._sync_mode_chrome()

    def _clear_analysis_results(self):
        for mode, panel in self.analysis_panels.items():
            panel.result = panel.coating = None
            panel._band_cache = None
            panel.table.setRowCount(0)
            panel.selected_picker.clear()
            panel.draw()
            self._analysis_page_indices[mode] = 0

    def _use_batch_result(self, result, index):
        from PySide6.QtWidgets import QMessageBox
        from .analysis_data import verified_batch_file
        if self.job_is_running() or result is None:
            return
        def worker():
            return verified_batch_file(result, index)
        def success(path):
            try:
                self._publish_nominal_artifact('ibc', path)
                self.status_var.set(f'Selected {Path(path).name}. Use Export and attach to current GHOST geometry to attach it.')
            except Exception as exc:
                QMessageBox.warning(self, 'Selected IBC', str(exc))
        self._run_background_task('Validate selected IBC', worker, success, 'Selected IBC changed or unavailable')
