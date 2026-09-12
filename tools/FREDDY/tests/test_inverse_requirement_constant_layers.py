from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import copy
import tempfile
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from ibc import compute
from ibc.compute import (ConstantMaterial, LayerConfig, LoadedLayer, MaterialTable,
                         UncertaintyConfig, compute_angle_metrics_many, build_uncertainty_scales)
from ibc.design_search import InverseSearchRequest, run_inverse_search, score_inverse_candidate
from ibc.inverse_grid import DesignGrid
from ibc.inverse_workflow import search_identity, check_layers
from ibc.io import (constant_material_from_layer, layer_config_to_dict, layer_config_from_dict,
                    load_project_file, save_project_file)
from ibc.search_checkpoint import load_checkpoint, save_checkpoint
from ibc.ui import ImpedanceGui
from ibc.ui_dialogs import LayerDialog
from ibc.ui_options import INVERSE_SCORE_MODE_OPTIONS, INVERSE_SCORE_WHOLE_BAND


def constant_layer(**changes):
    return replace(LayerConfig(.08, False, '', '', 0., material_source='constant',
                               constant_eps_real=6., constant_eps_imag=-.8,
                               constant_mu_real=1., constant_mu_imag=-.1), **changes)


def statistics(values):
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


class RequirementObjectiveTests(unittest.TestCase):
    def score(self, values, mode):
        return score_inverse_candidate([1., 2.], [0.], [], 'te', [(1., 1., 1.)], mode,
            stop_requested=lambda: False, statistics=statistics, requirement_db=-10,
            compute_metrics=lambda *a, **k: {'metal_loss_db': values})

    def test_whole_band_rejects_deep_narrow_null_that_wins_on_mean(self):
        mean_narrow = self.score([-40., -1.], INVERSE_SCORE_MODE_OPTIONS[0])
        mean_broad = self.score([-11., -11.], INVERSE_SCORE_MODE_OPTIONS[0])
        self.assertLess(mean_narrow[0], mean_broad[0])
        narrow = self.score([-40., -1.], INVERSE_SCORE_WHOLE_BAND)
        broad = self.score([-11., -11.], INVERSE_SCORE_WHOLE_BAND)
        self.assertEqual(narrow[0], 9.)
        self.assertEqual(broad[0], -1.)
        self.assertLess(broad[0], narrow[0])
        self.assertEqual(narrow[1:], mean_narrow[1:])
        self.assertEqual(self.score([-10., -10.], INVERSE_SCORE_WHOLE_BAND)[0], 0.)

    def test_objective_uses_worst_frequency_angle_and_tolerance_corner(self):
        scales = [(1., 1., 1.), (.95, .95, .95)]
        def metrics(_frequencies, angle, *args, **kwargs):
            return {'metal_loss_db': [-30., -4.] if angle == 60 and kwargs['eps_scale'] == .95 else [-20., -20.]}
        score = score_inverse_candidate([1., 2.], [0., 60.], [], 'te', scales, INVERSE_SCORE_WHOLE_BAND,
            stop_requested=lambda: False, statistics=statistics, compute_metrics=metrics, requirement_db=-10)
        self.assertEqual(score[0], 6)
        self.assertEqual(score[1], -20)

    def test_constant_search_resume_preserves_scores_and_rejects_changed_requirement(self):
        layer = constant_layer(inv_t_min_in=.02, inv_t_max_in=.1, inv_t_accuracy_in=.02)
        request = InverseSearchRequest([layer], [2., 4., 8.], [0., 45.], 'te', UncertaintyConfig(True, 5, 5, 5),
            INVERSE_SCORE_WHOLE_BAND, None, DesignGrid([layer]), 3, '2, 4, 8 GHz', 0., 45., True, -10.)
        count = [0]
        def progress(done, total, phase):
            count[0] = done
        def scorer(*args, **kwargs):
            return score_inverse_candidate(*args, **kwargs, stop_requested=lambda: False, statistics=statistics)
        with mock.patch('ibc.design_search.read_material_table', side_effect=AssertionError('No files needed')):
            _partial, checkpoint = run_inverse_search(request, stop_requested=lambda: count[0] >= 2,
                                                     progress=progress, score_candidate=scorer)
        self.assertEqual(checkpoint['next_index'], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recovery.fsearch'
            save_checkpoint(path, checkpoint)
            restored = load_checkpoint(path)
            resumed_request = replace(request, checkpoint=restored)
            tracked = mock.Mock(wraps=scorer)
            resumed, finished = run_inverse_search(resumed_request, stop_requested=lambda: False,
                                                   progress=lambda *a: None, score_candidate=tracked)
            self.assertEqual(tracked.call_count, 3)
            complete, fresh = run_inverse_search(request, stop_requested=lambda: False,
                                                 progress=lambda *a: None, score_candidate=scorer)
            self.assertEqual(finished['score_rows'], fresh['score_rows'])
            self.assertEqual(resumed[0], complete[0])
            for changed in (replace(resumed_request, requirement_db=-12.),
                            replace(resumed_request, score_mode=INVERSE_SCORE_MODE_OPTIONS[0]),
                            replace(resumed_request, layer_snapshot=[replace(layer, constant_eps_real=7.)])):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    run_inverse_search(changed, stop_requested=lambda: False, progress=lambda *a: None, score_candidate=scorer)


class ConstantMaterialTests(unittest.TestCase):
    def test_constant_properties_match_flat_measured_table_for_all_solver_paths(self):
        constant = constant_material_from_layer(constant_layer())
        self.assertIsInstance(constant, ConstantMaterial)
        flat = MaterialTable([1e-6, 1000.], [constant.eps_r]*2, [constant.mu_r]*2)
        layers = [LoadedLayer(.001, False, 0., constant, None),
                  LoadedLayer(0., False, 0., None, None, True, 320.)]
        reference = [replace(layers[0], table_0deg=flat), layers[1]]
        frequencies = [1e-6, 2., 18., 1000.]
        compute.validate_sweep_coverage([1e-9, 1e6], constant, 'constant')
        for pol in ('te', 'tm'):
            for angle in (0., 55.):
                for numpy in (True, False):
                    with mock.patch.object(compute, 'NUMPY_AVAILABLE', numpy):
                        actual = compute_angle_metrics_many(frequencies, angle, layers, pol, eps_scale=1.05, mu_scale=.95)
                        expected = compute_angle_metrics_many(frequencies, angle, reference, pol, eps_scale=1.05, mu_scale=.95)
                    for key in actual:
                        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=1e-10)

    def test_constant_project_roundtrip_needs_no_files_and_validates_values(self):
        layer = constant_layer()
        state = {'layers': [layer_config_to_dict(layer)], 'controls': {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'project.json'
            save_project_file(path, state)
            loaded = load_project_file(path)
            self.assertEqual(layer_config_from_dict(loaded['layers'][0]), layer)
            self.assertEqual(list(Path(directory).iterdir()), [path])
        for invalid in ({'constant_eps_real': float('nan')}, {'constant_mu_imag': float('inf')},
                        {'constant_eps_imag': .1}, {'constant_mu_imag': .1},
                        {'constant_mu_real': 0., 'constant_mu_imag': 0.},
                        {'anisotropic': True}, {'is_sheet': True}, {'material_source': 'unknown'}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                layer_config_from_dict(dict(state['layers'][0], **invalid))
        check_layers([layer], [1e-6, 1000.])

    def test_identity_tracks_constant_components_and_only_active_requirement(self):
        layer = constant_layer()
        cfg = UncertaintyConfig(False, 0, 0, 0)
        def identity(mode, target=-10, layers=None):
            return search_identity(layers or [layer], [2., 8.], [0.], 'te', cfg, mode, target)
        self.assertNotEqual(identity(INVERSE_SCORE_WHOLE_BAND), identity(INVERSE_SCORE_WHOLE_BAND, -12))
        self.assertEqual(identity(INVERSE_SCORE_MODE_OPTIONS[0]), identity(INVERSE_SCORE_MODE_OPTIONS[0], -12))
        self.assertNotEqual(identity(INVERSE_SCORE_WHOLE_BAND), identity(INVERSE_SCORE_WHOLE_BAND, layers=[replace(layer, constant_mu_imag=-.2)]))
        with self.assertRaisesRegex(ValueError, 'target'):
            identity(INVERSE_SCORE_WHOLE_BAND, float('nan'))


class NewFeaturesUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.ui = ImpedanceGui()
        self.ui.layers = [constant_layer(inv_t_min_in=.04, inv_t_max_in=.12, inv_t_accuracy_in=.04)]
        self.ui._refresh_layers()
        self.errors = []
        def immediate(_name, worker, success, _error):
            success(worker())
        for patch in (mock.patch.object(self.ui, '_run_background_task', side_effect=immediate),
                      mock.patch.object(self.ui, '_confirm_output_replacements', return_value=True),
                      mock.patch('ibc.ui.messagebox.showinfo'), mock.patch('ibc.ui.messagebox.showwarning'),
                      mock.patch('ibc.ui.messagebox.showerror', side_effect=lambda *a, **k: self.errors.append(a))):
            patch.start()
            self.addCleanup(patch.stop)
        for name, value in {'f_start':'2', 'f_stop':'6', 'f_step':'2', 'output':str(self.folder/'ibc.csv'),
                            'angle_f_start':'2', 'angle_f_stop':'6', 'angle_f_step':'2', 'angle_start':'0', 'angle_stop':'45', 'angle_step':'45', 'angle_output':str(self.folder/'angle.csv'),
                            'thk_f_start':'2', 'thk_f_stop':'6', 'thk_f_step':'2', 'thk_start':'.04', 'thk_stop':'.12', 'thk_step':'.04', 'thk_output':str(self.folder/'thickness.csv'),
                            'ibc_batch_start':'.04', 'ibc_batch_stop':'.12', 'ibc_batch_step':'.04', 'ibc_batch_output_dir':str(self.folder),
                            'inv_target_start':'2', 'inv_target_stop':'6', 'inv_target_step':'2', 'inv_angle_start':'0', 'inv_angle_stop':'45', 'inv_angle_step':'45', 'inv_top_n':'3'}.items():
            getattr(self.ui, name + '_var').set(value)

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def test_layer_editor_adds_and_edits_constants_and_rejects_gain(self):
        dialog = LayerDialog(self.ui)
        dialog.source_var.set('Constant εr / μr (all frequencies)')
        dialog.constant_vars['constant_eps_real'].set('4')
        dialog.constant_vars['constant_eps_imag'].set('-0.3')
        dialog._on_ok()
        self.assertEqual(dialog.result.constant_eps_real, 4.)
        self.assertEqual(dialog.result.file_0deg, '')
        edited = LayerDialog(self.ui, initial=dialog.result)
        self.assertTrue(edited.source_var.get().startswith('Constant'))
        edited.constant_vars['constant_mu_imag'].set('0.2')
        with mock.patch('ibc.ui_dialogs.messagebox.showerror') as error:
            edited._on_ok()
            error.assert_called_once()
            self.assertIn('gain-sign', error.call_args.args[1])
        self.assertIsNone(edited.result)
        dialog.deleteLater()
        edited.deleteLater()

    def test_constants_work_in_impedance_off_angle_thickness_and_ibc_batch(self):
        with mock.patch('ibc.ui.read_material_table', side_effect=AssertionError('Constant layer read a CSV')):
            self.ui._compute_impedance()
            self.ui._compute_off_angle()
            self.ui._compute_thickness()
            self.ui._export_ibc_batch()
        for mode in ('Impedance', 'Off Angle', 'Thickness', 'IBC Batch'):
            result = self.ui.analysis_panels[mode].result
            self.assertIsNotNone(result, mode)
            self.assertIn('Constant', result.layers[0])
        self.assertEqual(len(self.ui.analysis_panels['IBC Batch'].result.files), 3)
        self.assertEqual(set(self.ui.analysis_panels['Off Angle'].result.metrics), {'TE', 'TM'})
        self.assertEqual(self.ui._material_explorer_stack_sources(), [])
        self.assertFalse(self.errors)

    def test_requirement_scores_margins_and_project_are_independent_of_plot_controls(self):
        self.ui.inv_score_mode_var.set(INVERSE_SCORE_WHOLE_BAND)
        self.ui.inv_requirement_db_var.set('-10')
        self.ui.inv_uncertainty_var.set(True)
        self.ui._run_inverse_design()
        self.assertFalse(self.errors)
        self.assertEqual(len(self.ui.inverse_candidates), 3)
        for index, candidate in enumerate(self.ui.inverse_candidates):
            self.assertAlmostEqual(candidate.score_db, np.max(self.ui.inverse_plot_samples[index]) + 10., places=11)
        scores = [c.score_db for c in self.ui.inverse_candidates]
        self.assertEqual(scores, sorted(scores))
        original_margin = float(self.ui.inv_results_list.item(0, 8).text())
        self.ui.inv_curve_mode.setCurrentIndex(1)
        self.ui.inv_target_db.setValue(-2)
        self.assertEqual(float(self.ui.inv_results_list.item(0, 8).text()), original_margin)
        self.ui._ensure_inverse_result_current()
        self.assertEqual([c.score_db for c in self.ui.inverse_candidates], scores)
        self.ui.inv_results_list.sortItems(8, Qt.DescendingOrder)
        self.assertEqual(self.ui.inv_results_list.item(0, 1).data(Qt.UserRole), 0)
        self.assertIn('Constant', self.ui._inverse_candidate_description(0))
        state = self.ui._collect_project_state()
        self.ui._apply_project_state(state)
        self.assertEqual(self.ui.inv_requirement_db_var.get(), '-10')
        self.assertEqual(self.ui.inv_score_mode_var.get(), INVERSE_SCORE_WHOLE_BAND)
        self.assertTrue(self.ui.layers[0].is_constant)

    def test_changed_search_target_or_constants_block_candidate_use(self):
        self.ui.inv_score_mode_var.set(INVERSE_SCORE_WHOLE_BAND)
        self.ui._run_inverse_design()
        self.ui.inv_requirement_db_var.set('-12')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.ui._ensure_inverse_result_current()
        self.ui.inv_requirement_db_var.set('-10')
        self.ui.layers[0].constant_eps_real = 7.
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.ui._ensure_inverse_result_current()

    def test_apply_and_save_candidate_preserve_constant_material_values(self):
        self.ui.inv_score_mode_var.set(INVERSE_SCORE_WHOLE_BAND)
        self.ui._run_inverse_design()
        selected = self.ui.inverse_candidates[0]
        path = self.folder / 'candidate.json'
        with mock.patch('PySide6.QtWidgets.QFileDialog.getSaveFileName', return_value=(str(path), '')):
            self.ui._save_inverse_candidate()
        saved = layer_config_from_dict(load_project_file(path)['layers'][0])
        self.assertTrue(saved.is_constant)
        self.assertEqual(saved.constant_eps_imag, -.8)
        self.assertEqual(saved.thickness_in, selected.thickness_in[0])
        self.ui._apply_inverse_candidate()
        self.assertTrue(self.ui.layers[0].is_constant)
        self.assertEqual(self.ui.layers[0].constant_eps_real, 6.)
        self.assertEqual(self.ui.layers[0].file_0deg, '')
        self.assertEqual(self.ui.layers[0].thickness_in, selected.thickness_in[0])
        self.assertFalse(self.errors)

    def test_invalid_constant_project_or_requirement_leaves_live_state_unchanged(self):
        state = self.ui._collect_project_state()
        changed = copy.deepcopy(state)
        changed['layers'][0]['constant_eps_real'] = float('nan')
        with self.assertRaises(ValueError):
            self.ui._apply_project_state(changed)
        self.assertEqual(self.ui._collect_project_state(), state)
        changed = copy.deepcopy(state)
        changed['controls'].update(inv_score_mode=INVERSE_SCORE_WHOLE_BAND, inv_requirement_db='nan')
        with self.assertRaises(ValueError):
            self.ui._apply_project_state(changed)
        self.assertEqual(self.ui._collect_project_state(), state)


if __name__ == '__main__':
    unittest.main()
