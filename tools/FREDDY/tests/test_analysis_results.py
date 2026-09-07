import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from ibc.analysis_data import band_metrics, inverse_case_grid, verified_batch_file, passing_sample_ranges
from ibc.compute import LayerConfig, MaterialTable
from ibc.io import write_material_table
from ibc.ui import ImpedanceGui


class AnalysisDataTests(unittest.TestCase):
    def test_band_endpoints_interpolate_and_do_not_extrapolate(self):
        m = band_metrics([1., 2., 10.], np.array([[-20.], [-20.], [0.]]), -10., 2., 8.)[0]
        self.assertAlmostEqual(m.widest_ghz, 4.)
        self.assertAlmostEqual(m.coverage_pct, 100 * 4 / 6)
        with self.assertRaises(ValueError):
            band_metrics([1., 2.], np.array([[-20.], [-20.]]), -10., .5, 2.)
        with self.assertRaises(ValueError):
            band_metrics([1., 2.], np.array([[-20.], [-20.]]), -10., 2., 1.)

    def test_case_labels_preserve_frequency_tolerance_angle_axes(self):
        samples = [[10, 11, 20, 21], [30, 31, 40, 41]]
        grid = inverse_case_grid(samples, [0., 45.], [(1, 1, 1), (.95, 1, 1)])
        np.testing.assert_array_equal(grid[:, 0, 1], [11, 31])
        np.testing.assert_array_equal(grid.max(axis=1), [[20, 21], [40, 41]])
        with self.assertRaises(ValueError):
            inverse_case_grid(samples, [0.], [(1, 1, 1)])

    def test_passing_samples_group_without_bridging_failed_choice(self):
        metrics = band_metrics([1., 2.], np.array([[-20, -20, 0, -20], [-20, -20, 0, -20]]), -10)
        self.assertEqual(passing_sample_ranges([.01, .02, .03, .04], metrics, -10), [(.01, .02), (.04, .04)])


class AnalysisWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        material = self.folder / 'test-material.csv'
        write_material_table(material, MaterialTable([1., 20.], [6 - .8j] * 2, [1 - .1j] * 2))
        self.ui = ImpedanceGui()
        self.ui.layers = [LayerConfig(.08, False, str(material), '', 0., inv_t_min_in=.05, inv_t_max_in=.12, inv_t_accuracy_in=.01)]
        self.ui._refresh_layers()
        values = {
            'f_start': '2', 'f_stop': '6', 'f_step': '1',
            'angle_f_start': '2', 'angle_f_stop': '6', 'angle_f_step': '1',
            'angle_start': '0', 'angle_stop': '45', 'angle_step': '45',
            'thk_f_start': '2', 'thk_f_stop': '6', 'thk_f_step': '1',
            'thk_start': '.04', 'thk_stop': '.12', 'thk_step': '.04',
            'thk_angle': '0', 'output': str(self.folder / 'imp.csv'),
            'thk_output': str(self.folder / 'thickness.csv'), 'angle_output': str(self.folder / 'angle.csv'),
            'ibc_batch_output_dir': str(self.folder), 'ibc_batch_start': '.04', 'ibc_batch_stop': '.12',
            'ibc_batch_step': '.04', 'ibc_batch_unit': 'in',
            'inv_freq_mode': 'Band sweep', 'inv_target_start': '2', 'inv_target_stop': '6',
            'inv_target_step': '1', 'inv_angle_start': '0', 'inv_angle_stop': '45', 'inv_angle_step': '45',
            'inv_max_evals': '4', 'inv_top_n': '2',
        }
        for name, value in values.items():
            getattr(self.ui, name + '_var').set(value)
        self.ui.inv_refine_var.set(False)
        self.errors = []
        def now(_name, worker, success, _error):
            success(worker())
        for patch in [mock.patch.object(self.ui, '_run_background_task', side_effect=now),
                      mock.patch.object(self.ui, '_confirm_output_replacements', return_value=True),
                      mock.patch('ibc.ui.messagebox.showinfo'), mock.patch('ibc.ui.messagebox.showwarning'),
                      mock.patch('ibc.ui.messagebox.showerror', side_effect=lambda *a, **k: self.errors.append(a))]:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def test_impedance_retains_off_angle_and_air_power_balance_is_physical(self):
        self.ui._select_mode(2)
        self.ui._compute_off_angle()
        previous = self.ui.last_heatmap_results
        panel = self.ui.analysis_panels['Off Angle']
        old_result = panel.result
        self.assertEqual(set(old_result.metrics), {'TE', 'TM'})
        self.ui._select_mode(0)
        self.ui.backing_var.set('air')
        self.ui._compute_impedance()
        self.assertIs(self.ui.last_heatmap_results, previous)
        self.assertIs(panel.result, old_result)
        impedance = self.ui.analysis_panels['Impedance']
        self.assertEqual(impedance.result.backing, 'air')
        impedance.view.setCurrentText('Power balance')
        lines = impedance.figure.axes[0].lines
        np.testing.assert_allclose(sum(np.asarray(line.get_ydata()) for line in lines), 100., atol=1e-10)
        self.assertTrue(np.any(lines[2].get_ydata() > 1))
        self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 1)
        self.assertFalse(self.errors)

    def test_batch_curves_match_exported_data_and_selection_is_explicit_after_sort(self):
        from ibc.analysis_data import reflection_from_impedance
        published = []
        self.ui.nominal_artifact_exported.connect(lambda *args: published.append(args))
        self.ui._select_mode(1)
        self.ui._export_ibc_batch()
        panel = self.ui.analysis_panels['IBC Batch']
        result = panel.result
        self.assertEqual(published, [])
        self.assertEqual(len(result.files), 3)
        for index, file in enumerate(result.files):
            csv = np.loadtxt(file, delimiter=',', skiprows=1)
            z = csv[:, 1] + 1j * csv[:, 2]
            np.testing.assert_allclose(result.impedance[:, index], z, rtol=1e-11)
            np.testing.assert_allclose(result.metrics['TE']['metal_loss_db'][:, index], reflection_from_impedance(z), atol=1e-9)
        panel.table.sortItems(1, Qt.DescendingOrder)
        panel.table.setCurrentCell(0, 1)
        self.assertEqual(panel.selected, 2)
        self.ui._use_batch_result(result, panel.selected)
        self.assertEqual(published, [('ibc', str(result.files[2]))])
        result.files[2].write_text(result.files[2].read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verified_batch_file(result, 2)
        self.assertFalse(self.errors)

    def test_failed_batch_does_not_publish_a_partial_results_set(self):
        self.ui._select_mode(1)
        self.ui._export_ibc_batch()
        previous = self.ui.analysis_panels['IBC Batch'].result
        with mock.patch('ibc.ui.export_pec_ibc_thickness_batch', side_effect=RuntimeError('write failed')):
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                self.ui._export_ibc_batch()
        self.assertIs(self.ui.analysis_panels['IBC Batch'].result, previous)

    def test_batch_rejects_file_replacement_between_export_and_result_publication(self):
        from ibc.batch import export_pec_ibc_thickness_batch
        def changed(items, *args, **kwargs):
            count = export_pec_ibc_thickness_batch(items, *args, **kwargs)
            items[0].path.write_text('different contents')
            return count
        with mock.patch('ibc.ui.export_pec_ibc_thickness_batch', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed during publication'):
                self.ui._export_ibc_batch()
        self.assertIsNone(self.ui.analysis_panels['IBC Batch'].result)

    def test_thickness_views_use_frozen_grid_and_tolerance_envelopes(self):
        self.ui._select_mode(3)
        self.ui.thk_uncertainty_var.set(True)
        self.ui._compute_thickness()
        panel = self.ui.analysis_panels['Thickness']
        self.assertIn('TE', panel.result.upper)
        before = panel.result.context
        original = panel.result.metrics['TE']['metal_loss_db'].copy()
        self.ui.thk_angle_var.set('60')
        self.ui.thk_f_stop_var.set('18')
        for view in ['Frequency curves', 'Bandwidth', 'Band coverage', 'Tolerance envelope', 'Frequency of minimum', 'At selected frequency', 'Power balance']:
            panel.view.setCurrentText(view)
            self.assertTrue(panel.figure.axes[0].lines, view)
        self.assertEqual(panel.result.context, before)
        np.testing.assert_array_equal(panel.result.metrics['TE']['metal_loss_db'], original)
        panel.bound.setCurrentIndex(1)
        panel.target.setValue(-8)
        self.assertFalse(self.errors)

    def test_lower_and_span_views_preserve_existing_uncertainty_display_capabilities(self):
        self.ui._select_mode(3)
        self.ui.thk_uncertainty_var.set(True)
        self.ui._compute_thickness()
        panel = self.ui.analysis_panels['Thickness']
        panel.view.setCurrentText('Frequency curves')
        panel.bound.setCurrentIndex(2)
        low = panel.result.lower['TE']['metal_loss_db']
        high = panel.result.upper['TE']['metal_loss_db']
        np.testing.assert_allclose(panel.figure.axes[0].lines[0].get_ydata(), low[:, 0])
        panel.bound.setCurrentIndex(3)
        np.testing.assert_allclose(panel.figure.axes[0].lines[0].get_ydata(), (high - low)[:, 0])
        self.assertEqual(len(panel.figure.axes[0].lines), 3)  # no absolute -10 dB line on a span
        self.assertIn('comparison uses nominal', panel.note.text())

    def test_clicking_colorbar_does_not_select_a_thickness_but_map_click_does(self):
        from types import SimpleNamespace
        self.ui._select_mode(3)
        self.ui._compute_thickness()
        panel = self.ui.analysis_panels['Thickness']
        panel.plot_clicked(SimpleNamespace(inaxes=panel.figure.axes[1], button=1, xdata=0.12, ydata=4.))
        self.assertEqual(panel.selected, 0)
        panel.plot_clicked(SimpleNamespace(inaxes=panel.figure.axes[0], button=1, xdata=0.12, ydata=4.))
        self.assertEqual(panel.selected, 2)
        self.assertEqual(panel.frequency.value(), 4.)

    def test_large_frequency_overlay_keeps_narrow_null_and_full_metric_data(self):
        from ibc.analysis_data import SweepResult
        f = np.linspace(1., 2., 20000)
        y = np.full((len(f), 1), -2.)
        y[12345, 0] = -70.
        panel = self.ui.analysis_panels['IBC Batch']
        panel.set_result(SweepResult(f.tolist(), [.04], 'Thickness (in)', 'Large example', {'TE': {'metal_loss_db': y}}))
        line = panel.figure.axes[0].lines[0]
        self.assertLessEqual(len(line.get_xdata()), 5000)
        self.assertEqual(min(line.get_ydata()), -70.)
        self.assertEqual(panel.result.metrics['TE']['metal_loss_db'].shape, (20000, 1))

    def test_both_polarizations_have_shared_scale_and_option_persists(self):
        self.ui._select_mode(2)
        self.ui._compute_off_angle()
        panel = self.ui.analysis_panels['Off Angle']
        panel.view.setCurrentText('TE/TM comparison')
        axes = [axis for axis in panel.figure.axes if axis.get_title().startswith(('TE', 'TM'))]
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].collections[0].get_clim(), axes[1].collections[0].get_clim())
        self.ui.angle_compare_both.setChecked(False)
        state = self.ui._collect_project_state()
        self.assertFalse(state['controls']['angle_compare_both'])
        self.ui._compute_off_angle()
        self.assertEqual(list(panel.result.metrics), ['TE'])
        panel.view.setCurrentText('TE/TM comparison')
        self.assertIn('Enable', panel.figure.axes[0].texts[0].get_text())

    def test_directional_te_run_survives_unsupported_optional_tm_comparison(self):
        self.ui.layers[0].anisotropic = True
        self.ui.layers[0].file_90deg = self.ui.layers[0].file_0deg
        self.ui._select_mode(2)
        self.ui._compute_off_angle()
        panel = self.ui.analysis_panels['Off Angle']
        self.assertEqual(list(panel.result.metrics), ['TE'])
        self.assertIn('TM comparison unavailable', panel.result.context)
        panel.view.setCurrentText('TE/TM comparison')
        self.assertIn('directional stack', panel.figure.axes[0].texts[0].get_text())
        self.assertFalse(self.errors)

    def test_single_point_axes_have_actionable_view_instead_of_false_heatmap_extent(self):
        self.ui._select_mode(3)
        self.ui.thk_f_stop_var.set('2')
        self.ui.thk_stop_var.set('.04')
        self.ui._compute_thickness()
        panel = self.ui.analysis_panels['Thickness']
        self.assertIn('at least two', panel.figure.axes[0].texts[0].get_text())
        panel.view.setCurrentText('Band coverage')
        self.assertTrue(panel.figure.axes[0].lines)
        panel.view.setCurrentText('Bandwidth')
        self.assertIn('single frequency', panel.figure.axes[0].texts[0].get_text())

    def test_coating_map_matches_report_maxima_and_exports_no_ibc(self):
        published = []
        self.ui.nominal_artifact_exported.connect(lambda *a: published.append(a))
        self.ui._check_ghost_coating()
        panel = self.ui.analysis_panels['Impedance']
        report = panel.coating
        for pol in ('TE', 'TM'):
            rows = [r for r in report['angles'] if r['polarization'] == pol]
            np.testing.assert_allclose(np.max(report['error_grids'][pol], axis=0),
                                       [r['max_absolute_complex_reflection_error'] for r in rows])
        panel.view.setCurrentText('Coating error map')
        self.assertEqual(len(panel.figure.axes), 4)
        self.assertEqual(published, [])

    def test_inverse_angle_map_and_tolerance_use_run_labels_after_setup_edits(self):
        self.ui._select_mode(4)
        self.ui.inv_uncertainty_var.set(True)
        self.ui._run_inverse_design()
        metadata = copy.deepcopy(self.ui.inverse_result_metadata)
        self.ui.inv_angle_stop_var.set('80')
        self.ui.inv_uncertainty_var.set(False)
        self.ui.inv_plot_view.setCurrentIndex(3)
        self.assertEqual(self.ui.inv_axis.get_xlim(), (0., 45.))
        self.ui.inv_plot_view.setCurrentIndex(4)
        self.ui.inv_inspect_angle.setCurrentIndex(1)
        self.assertIn('45°', self.ui.inv_axis.get_title())
        self.assertEqual(self.ui.inverse_result_metadata, metadata)
        self.assertEqual(len(self.ui.inv_figure.axes), 1)  # old colorbar removed
        self.assertFalse(self.errors)

    def test_project_load_clears_run_owned_results_and_save_uses_current_figure(self):
        self.ui._select_mode(3)
        self.ui._compute_thickness()
        panel = self.ui.analysis_panels['Thickness']
        with mock.patch('ibc.ui.filedialog.asksaveasfilename', return_value=str(self.folder / 'plot.png')), \
                mock.patch.object(panel.figure, 'savefig') as save, mock.patch.object(self.ui.fig, 'savefig') as old:
            self.ui._save_plot()
        save.assert_called_once()
        old.assert_not_called()
        self.ui._apply_project_state(self.ui._collect_project_state())
        self.assertTrue(all(p.result is None for p in self.ui.analysis_panels.values()))
        self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 0)
