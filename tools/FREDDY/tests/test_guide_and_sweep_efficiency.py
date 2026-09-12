from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QApplication

from ibc import compute
from ibc.analysis_data import (SweepResult, accumulate_grid_bounds, band_metrics,
                               write_comparison_report)
from ibc.compute import LayerConfig, LoadedLayer, MaterialTable, UncertaintyConfig
from ibc.guide import ANALYSIS_PLOTS, MODE_TOPICS
from ibc.io import write_material_table
from ibc.sweep_results import VIEWS
from ibc.ui import ImpedanceGui, HEATMAP_METRIC_KEYS


class SweepEfficiencyTests(unittest.TestCase):
    def test_array_metrics_match_public_lists_and_scalar_physics(self):
        table = MaterialTable([1., 20.], [6 - .8j, 4 - .5j], [1 - .1j, 1 - .02j])
        stack = [LoadedLayer(.001, False, 0., table, None),
                 LoadedLayer(0., False, 0., None, None, True, 240.)]
        frequencies = [2., 5., 9., 16.]
        for pol in ('te', 'tm'):
            arrays = compute.compute_angle_metrics_many(frequencies, 47., stack, pol, return_arrays=True)
            lists = compute.compute_angle_metrics_many(frequencies, 47., stack, pol)
            for key in arrays:
                self.assertIsInstance(arrays[key], np.ndarray)
                self.assertIsInstance(lists[key], list)
                np.testing.assert_array_equal(arrays[key], lists[key])
            with mock.patch.object(compute, 'NUMPY_AVAILABLE', False):
                scalar = compute.compute_angle_metrics_many(frequencies, 47., stack, pol, return_arrays=True)
            for key in arrays:
                np.testing.assert_allclose(arrays[key], scalar[key], atol=2e-11, rtol=2e-11)

    def test_envelopes_align_wrapped_phase_without_mutating_inputs(self):
        nominal = {'metal_phase_deg': np.array([[179., -179.]]), 'metal_loss_db': np.array([[-10., -20.]])}
        changed = {'metal_phase_deg': np.array([[-179., 179.]]), 'metal_loss_db': np.array([[-8., -25.]])}
        before = {k: v.copy() for k, v in changed.items()}
        low = {k: v.copy() for k, v in nominal.items()}
        high = {k: v.copy() for k, v in nominal.items()}
        identities = {k: (id(low[k]), id(high[k])) for k in low}
        accumulate_grid_bounds(nominal, low, high, changed)
        np.testing.assert_array_equal(low['metal_phase_deg'], [[179., -181.]])
        np.testing.assert_array_equal(high['metal_phase_deg'], [[181., -179.]])
        np.testing.assert_array_equal(nominal['metal_phase_deg'], [[179., -179.]])
        np.testing.assert_array_equal(low['metal_loss_db'], [[-10., -25.]])
        for key in changed:
            np.testing.assert_array_equal(changed[key], before[key])
            self.assertEqual((id(low[key]), id(high[key])), identities[key])

    def test_report_uses_band_metrics_but_full_sweep_null_and_quotes_context(self):
        frequencies = [1., 2., 3.]
        grid = np.array([[-40., -1.], [-12., -12.], [-12., -8.]])
        result = SweepResult(frequencies, [.01, .02], 'Thickness (in)', 'Captured, run\nTE', {'TE': {'metal_loss_db': grid}},
                             layers=['Measured "blend", 0.01 in'])
        metrics = band_metrics(frequencies, grid, -10, 2, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'comparison.csv'
            self.assertEqual(write_comparison_report(path, result, grid, metrics, polarization='TE',
                reflection_key='metal_loss_db', bound='nominal', target=-10, low=2, high=3), 2)
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['run_context'], result.context)
        self.assertEqual(rows[0]['stack_at_run'], result.layers[0])
        self.assertEqual(float(rows[0]['margin_db']), 2)
        self.assertEqual(rows[0]['passes_entire_band'], 'True')
        self.assertEqual(float(rows[0]['sampled_null_full_sweep_ghz']), 1)
        self.assertEqual(float(rows[1]['margin_db']), -2)
        self.assertEqual(rows[1]['passes_entire_band'], 'False')
        self.assertEqual(float(rows[1]['coverage_pct']), 50)

    def test_report_preserves_destination_on_failure_and_single_point_has_no_bandwidth(self):
        result = SweepResult([1.], [0.], 'Normal incidence', 'run', {})
        grid = np.array([[-10.]])
        metrics = band_metrics([1.], grid, -10)
        options = dict(polarization='TE', reflection_key='metal_loss_db', bound='nominal', target=-10, low=1, high=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'comparison.csv'
            write_comparison_report(path, result, grid, metrics, **options)
            with path.open(newline='', encoding='utf-8') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['widest_band_ghz'], '')
            self.assertEqual(row['passes_entire_band'], 'True')
            original = path.read_bytes()
            with mock.patch('ibc.io.os.replace', side_effect=OSError('publication failed')):
                with self.assertRaisesRegex(OSError, 'publication failed'):
                    write_comparison_report(path, result, grid, metrics, **options)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])


class GuideWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.ui = ImpedanceGui()

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def test_guide_covers_every_analysis_view_and_searches_body_text(self):
        documented = {name for name, _description in ANALYSIS_PLOTS}
        self.assertTrue({v for views in VIEWS.values() for v in views}.issubset(documented))
        self.ui.guide.search.setText('dBsm')
        self.assertGreater(self.ui.guide.topic_list.count(), 0)
        self.assertIn('dBsm', self.ui.guide.browser.toPlainText())
        self.ui.guide.search.setText('no_such_topic_123')
        self.assertEqual(self.ui.guide.topic_list.count(), 0)
        self.assertIn('No matching topics', self.ui.guide.browser.toPlainText())
        self.ui.guide.open_topic('plots-analysis')
        self.assertIn('Tolerance envelope', self.ui.guide.browser.toPlainText())

    def test_workflow_links_open_setup_without_running_or_editing_project(self):
        state = self.ui._collect_project_state()
        with mock.patch.object(self.ui, '_run_background_task') as run:
            for mode, topic in MODE_TOPICS.items():
                self.ui._select_mode(self.ui._mode_labels.index(mode))
                self.ui._show_guide()
                self.assertEqual(self.ui.guide.topic_list.currentItem().data(Qt.UserRole), topic)
                self.ui.guide._follow_link(QUrl('mode:' + mode))
                self.assertEqual(self.ui._active_left_tab_label(), mode)
                if mode in self.ui._analysis_page_indices or mode == 'Inverse Design':
                    self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 0)
            self.ui.guide._follow_link(QUrl('https://example.com'))
            self.ui.guide._follow_link(QUrl('mode:unknown'))
            run.assert_not_called()
        self.assertEqual(self.ui._collect_project_state(), state)

    def test_material_tables_are_shared_per_run_and_refreshed_on_next_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'material.csv'
            first = MaterialTable([1., 20.], [4 - .5j] * 2, [1 + 0j] * 2)
            write_material_table(path, first)
            layers = [LayerConfig(.02, False, str(path), '', 0.)] * 3
            from ibc.ui import read_material_table
            with mock.patch('ibc.ui.read_material_table', wraps=read_material_table) as read:
                loaded = self.ui._load_layers(layers)
                self.assertEqual(read.call_count, 1)
            self.assertIs(loaded[0].table_0deg, loaded[2].table_0deg)
            changed = MaterialTable([1., 20.], [8 - 1j] * 2, [1 + 0j] * 2)
            write_material_table(path, changed)
            reloaded = self.ui._load_layers(layers)
            self.assertIsNot(reloaded[0].table_0deg, loaded[0].table_0deg)
            self.assertNotEqual(reloaded[0].table_0deg, loaded[0].table_0deg)

    def test_sweep_envelopes_match_scalar_fallback_and_keep_nominal_arrays(self):
        table = MaterialTable([1., 20.], [6 - .8j] * 2, [1 - .1j] * 2)
        layers = [LoadedLayer(.001, False, 0., table, None)]
        cfg = UncertaintyConfig(True, 5., 5., 5.)
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('angle', 'thickness'):
                def run(name):
                    path = Path(directory) / name
                    if mode == 'angle':
                        return self.ui._compute_angle_mode(path, layers, cfg, [0., 45.], [2., 8., 16.], 'te')
                    return self.ui._compute_thickness_mode(path, layers, 0, cfg, [.02, .04], [2., 8., 16.], 'te', 45.)
                vector = run(mode + '_vector.csv')
                with mock.patch('ibc.ui.NUMPY_AVAILABLE', False), mock.patch('ibc.compute.NUMPY_AVAILABLE', False):
                    scalar = run(mode + '_scalar.csv')
                for fast, slow in zip(vector[1:4], scalar[1:4]):
                    for key in HEATMAP_METRIC_KEYS:
                        self.assertIsInstance(fast[key], np.ndarray)
                        np.testing.assert_allclose(fast[key], slow[key], atol=2e-10, rtol=2e-10)
                nominal = vector[1]['metal_loss_db']
                self.assertFalse(np.shares_memory(nominal, vector[2]['metal_loss_db']))
                self.assertFalse(np.shares_memory(nominal, vector[3]['metal_loss_db']))

    def test_nominal_display_avoids_span_allocation_and_reuses_band_calculations(self):
        panel = self.ui.analysis_panels['Thickness']
        grid = np.array([[-15., -12.], [-15., -8.]])
        panel.set_result(SweepResult([2., 3.], [.01, .02], 'Thickness (in)', 'run', {'TE': {'metal_loss_db': grid}}))
        class NoSubtract(np.ndarray):
            def __sub__(self, other):
                raise AssertionError('Unrequested tolerance span allocated')
        bound = grid.view(NoSubtract)
        self.assertIs(panel.display_grid(grid, bound, bound), grid)
        a, b, c = panel.performance()
        self.assertIs(a, b)
        self.assertIs(a, c)
        item = panel.table.item(0, 1)
        panel.refresh()
        self.assertIs(panel.table.item(0, 1), item)
        panel.table.sortItems(7, Qt.DescendingOrder)
        self.assertEqual(panel.table.item(0, 1).data(Qt.UserRole), 0)
        self.assertEqual(float(panel.table.item(0, 7).text()), 5)

    def test_export_captures_display_choices_before_dialog_and_preserves_selection(self):
        panel = self.ui.analysis_panels['Thickness']
        nominal = np.array([[-15., -12.], [-15., -8.]])
        upper = nominal + 3
        panel.set_result(SweepResult([2., 3.], [.01, .02], 'Thickness (in)', 'Completed run',
            {'TE': {'metal_loss_db': nominal}}, upper={'TE': {'metal_loss_db': upper}}))
        panel.bound.setCurrentIndex(1)
        panel.selected_picker.setCurrentIndex(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.csv'
            def choose_file(*args):
                panel.target.setValue(-5)  # a later edit must not relabel captured data
                return str(path), 'CSV files (*.csv)'
            def immediate(_name, worker, success, _error):
                success(worker())
            with mock.patch('ibc.sweep_results.QFileDialog.getSaveFileName', side_effect=choose_file), \
                 mock.patch.object(self.ui, '_run_background_task', side_effect=immediate):
                panel.export_comparison()
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(panel.selected, 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['bound'], 'upper analyzed')
        self.assertEqual(float(rows[0]['target_db']), -10)
        self.assertEqual(float(rows[0]['worst_reflection_db']), -12)
        self.assertEqual(rows[0]['run_context'], 'Completed run')


if __name__ == '__main__':
    unittest.main()
