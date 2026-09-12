from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ibc.compute import (LayerConfig, LoadedLayer, ConstantMaterial, MaterialTable, INCH_TO_M,
                         compute_angle_metrics, UncertaintyConfig)
from ibc.io import layer_config_from_dict, layer_config_to_dict, save_project_file, load_project_file
from ibc.inverse_workflow import search_identity
from ibc.tolerance_config import DEFAULT_SPEC, SETUP_DEFAULTS, MODES, validate_setup, validate_tolerances
from ibc.tolerance_analysis import (PreparedStudy, parameters_from_layers, sampled_deviations,
                                    crossing_brackets, run_tolerance_study, StopToleranceAnalysis,
                                    export_tolerance_report, study_workload)
from ibc.ui import ImpedanceGui
from ibc.ui_dialogs import LayerDialog, SheetDialog
from ibc.ui_options import INVERSE_SCORE_MODE_OPTIONS


def spec(bound, **kwargs):
    return {**DEFAULT_SPEC, 'bound': bound, **kwargs}


def example():
    configs = [LayerConfig(0, False, '', '', 0, is_sheet=True, sheet_resistance=377,
                           tolerances={'sheet_resistance': spec(10)}),
               LayerConfig(.31, False, '', '', 0, material_source='constant',
                           tolerances={'thickness': spec(15)})]
    layers = [LoadedLayer(0, False, 0, None, None, True, 377),
              LoadedLayer(.31*INCH_TO_M, False, 0, ConstantMaterial(1+0j, 1+0j), None)]
    return configs, layers


def setup(**kwargs):
    return {**SETUP_DEFAULTS, 'f_start': '9', 'f_stop': '11', 'f_step': '.5',
            'a_step': '15', 'points': '5', 'samples': '64', **kwargs}


class ToleranceBackendTests(unittest.TestCase):
    def test_per_layer_component_perturbations_match_independent_scalar_solver(self):
        freqs = [2., 7., 12.]
        material = MaterialTable(freqs, [6-.8j, 5-.7j, 4-.6j], [1.3-.2j, 1.2-.15j, 1.1-.1j])
        configs, layers = example()
        configs[1].tolerances = {key: spec(5) for key in ('thickness', 'eps_real', 'eps_imag', 'mu_real', 'mu_imag')}
        layers[1].table_0deg = material
        params = parameters_from_layers(configs)
        before = copy.deepcopy(layers)
        study = PreparedStudy(layers, freqs, [0., 40.], ['te', 'tm'], params)
        offsets = np.array([-.4, .2, -.6, .8, -.3, .7])
        perturbed = copy.deepcopy(layers)
        for p, x in zip(params, offsets):
            layer = perturbed[p.layer]
            if p.key == 'sheet_resistance': layer.sheet_resistance *= 1 + x*.1
            elif p.key == 'thickness': layer.thickness_m *= 1 + x*.05
            else:
                values = layer.table_0deg.eps_r if p.key.startswith('eps') else layer.table_0deg.mu_r
                for i, value in enumerate(values):
                    values[i] += abs(value.real if p.key.endswith('real') else value.imag) * x * .05 * (1 if p.key.endswith('real') else 1j)
        expected = np.array([[[compute_angle_metrics(f, angle, perturbed, pol)['metal_loss_db'] for f in freqs]
                              for angle in [0., 40.]] for pol in ['te', 'tm']])
        np.testing.assert_allclose(study.response(offsets), expected, atol=2e-12, rtol=0)
        self.assertEqual(layers, before)
        np.testing.assert_array_equal(study.properties[1][0], material.eps_r)

    def test_absolute_thickness_and_material_bounds(self):
        configs, layers = example()
        configs[0].tolerances = {}
        configs[1].tolerances = {'thickness': spec(.01, units='Absolute'), 'eps_real': spec(.1, units='Absolute')}
        params = parameters_from_layers(configs)
        result = PreparedStudy(layers, [10.], [0.], ['te'], params).response([1., -1.])
        direct = copy.deepcopy(layers)
        direct[1].thickness_m = .32 * INCH_TO_M
        direct[1].table_0deg = ConstantMaterial(.9+0j, 1+0j)
        self.assertAlmostEqual(result[0, 0, 0], compute_angle_metrics(10, 0, direct, 'te')['metal_loss_db'], places=11)

    def test_sampler_reproducible_bounded_and_group_correlated(self):
        configs, _ = example()
        configs[0].tolerances['sheet_resistance'] = spec(10, group='lot', loading=1)
        configs[1].tolerances['thickness'] = spec(15, group='lot', loading=-1)
        params = parameters_from_layers(configs)
        a = np.vstack(list(sampled_deviations(params, 4096, 77, 64)))
        b = np.vstack(list(sampled_deviations(params, 4096, 77, 128)))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a[:, 0], -a[:, 1], atol=2e-15)
        self.assertTrue(np.all(np.abs(a) <= 1))
        self.assertAlmostEqual(a[:, 0].mean(), 0, places=3)
        self.assertAlmostEqual(a[:, 0].var(), 1/3, places=3)
        params = [replace(p, group='', distribution='Truncated normal (±3σ)') for p in params]
        c = np.vstack(list(sampled_deviations(params, 4096, 77)))
        self.assertLess(abs(np.corrcoef(c.T)[0, 1]), .02)
        self.assertAlmostEqual(c[:, 0].std(), .3289, places=3)
        self.assertTrue(np.all(np.abs(c) < 1))
        self.assertFalse(np.array_equal(c, np.vstack(list(sampled_deviations(params, 4096, 78)))))

    def test_pointwise_failures_are_distinct_from_whole_region_yield(self):
        configs, layers = example()
        configs[0].tolerances = {}
        def metrics(freqs, angle, stack, pol, **kwargs):
            delta = stack[1].thickness_m / layers[1].thickness_m - 1
            # Each simulated stack misses one of two frequencies; each point
            # individually fails in about half the trials.
            return {'metal_loss_db': [-10 + delta, -10 - delta]}
        result = run_tolerance_study(layers, configs, setup(f_step='2', a_stop='0', mode=MODES[1]), compute_metrics=metrics)
        self.assertEqual(result['passing'], 0)
        np.testing.assert_allclose(result['failure_counts'], 32, atol=1)
        self.assertEqual(result['evaluated'], 69)
        self.assertEqual(result['nominal_margin_db'], 0)
        self.assertEqual(result['convergence'][-1], [64, 0.])

    def test_partial_group_loadings_define_latent_correlation(self):
        from scipy.special import ndtri
        configs, _ = example()
        configs[0].tolerances['sheet_resistance'] = spec(10, group='shared', loading=.8)
        configs[1].tolerances['thickness'] = spec(15, group='shared', loading=.5)
        draws = np.vstack(list(sampled_deviations(parameters_from_layers(configs), 4096, 123)))
        latent = ndtri((draws + 1)/2)
        self.assertAlmostEqual(np.corrcoef(latent.T)[0, 1], .8*.5, delta=.01)

    def test_sampling_parameter_order_survives_sorted_json_project_save(self):
        configs, layers = example()
        configs[1].tolerances = {'thickness': spec(10), 'eps_real': spec(5)}
        restored = [layer_config_from_dict(json.loads(json.dumps(layer_config_to_dict(layer), sort_keys=True))) for layer in configs]
        self.assertEqual(parameters_from_layers(configs), parameters_from_layers(restored))
        a = run_tolerance_study(layers, configs, setup(mode=MODES[1]))
        b = run_tolerance_study(layers, restored, setup(mode=MODES[1]))
        np.testing.assert_array_equal(a['trial_margins_db'], b['trial_margins_db'])

    def test_combined_property_support_rejects_singularity(self):
        configs, layers = example()
        layers[1].table_0deg = ConstantMaterial(1-.5j, 1+0j)
        configs[1].tolerances = {'eps_real': spec(1, units='Absolute'), 'eps_imag': spec(.5, units='Absolute')}
        with self.assertRaisesRegex(ValueError, 'singular'):
            PreparedStudy(layers, [10.], [0.], ['te'], parameters_from_layers(configs))

    def test_report_captures_numeric_results_and_never_reinterpolates_per_trial(self):
        configs, layers = example()
        from ibc.compute import prepare_layer_properties_many
        with mock.patch('ibc.tolerance_analysis.prepare_layer_properties_many', wraps=prepare_layer_properties_many) as prepare:
            result = run_tolerance_study(layers, configs, setup(mode=MODES[1]))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(result['trial_margins_db'].shape, (64,))
        self.assertEqual(result['failure_counts'].shape, (2, 3, 5))
        self.assertEqual(result['passing'], int(np.sum(result['trial_margins_db'] >= 0)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'study.json'
            export_tolerance_report(path, result)
            restored = json.loads(path.read_text(encoding='utf-8'))
            np.testing.assert_array_equal(restored['failure_counts'], result['failure_counts'])
            self.assertEqual(restored['setup'], setup(mode=MODES[1]))
            original = path.read_bytes()
            invalid = dict(result, bad=float('nan'))
            with self.assertRaises(ValueError): export_tolerance_report(path, invalid)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_pass_miss_brackets_handle_nonmonotonicity_and_nominal_failure(self):
        offsets = [-1, -.5, 0, .5, 1]
        result = crossing_brackets(offsets, [2, -1, 1, -2, 3])
        self.assertEqual(result['negative'], [0., -.5])
        self.assertEqual(result['positive'], [0., .5])
        self.assertFalse(crossing_brackets(offsets, [1, 1, -1, 1, 1])['nominal_pass'])
        self.assertIsNone(crossing_brackets(offsets, [1]*5)['positive'])

    def test_invalid_physical_support_and_missing_material_coverage(self):
        configs, layers = example()
        for key, value, message in [('thickness', spec(100), 'positive'),
                                     ('eps_imag', spec(.1, units='Absolute'), 'gain'),
                                     ('eps_imag', spec(5), 'zero'),
                                     ('eps_real', spec(100), 'singular')]:
            configs[1].tolerances = {key: value}
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                PreparedStudy(layers, [10.], [0.], ['te'], parameters_from_layers(configs))
        configs[1].tolerances = {'thickness': spec(5)}
        layers[1].table_0deg = MaterialTable([1., 2.], [1+0j]*2, [1+0j]*2)
        with self.assertRaises(ValueError):
            PreparedStudy(layers, [10.], [0.], ['te'], parameters_from_layers(configs))

    def test_input_validation_and_bounded_memory_estimate(self):
        for change in [dict(samples='17'), dict(points='6'), dict(target='nan'), dict(a_stop='90'),
                       dict(f_step='0'), dict(seed='-1'), dict(f_step='1e-100')]:
            with self.subTest(change=change), self.assertRaises(ValueError): validate_setup(setup(**change))
        with self.assertRaises(ValueError): validate_tolerances({'thickness': spec(1, loading=1.1)}, False)
        with self.assertRaises(ValueError): validate_tolerances({'eps_real': spec(1)}, True)
        a = study_workload(setup(mode=MODES[1], samples='16'), 2, 2)
        b = study_workload(setup(mode=MODES[1], samples='65536'), 2, 2)
        self.assertLess(b['array_mib'] - a['array_mib'], 1)

    def test_cancellation_between_angles_and_during_trials_publishes_nothing(self):
        configs, layers = example()
        for stop_after in (0, 4, 12):
            progress = [0]
            def update(done, *_): progress[0] = done
            with self.assertRaises(StopToleranceAnalysis):
                run_tolerance_study(layers, configs, setup(mode=MODES[1]),
                                    stop_requested=lambda: progress[0] >= stop_after, progress=update)


class ToleranceUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.ui = ImpedanceGui()
        self.ui.layers, _ = example()
        self.ui._refresh_layers()
        self.panel = self.ui.tolerance_workspace
        self.panel.restore_setup(setup(mode=MODES[1]))
        self.errors = []
        self.patcher = mock.patch('ibc.ui.messagebox.showerror', side_effect=lambda *args, **kwargs: self.errors.append(args))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.ui.deleteLater()
        self.app.processEvents()

    def wait_for_task(self):
        deadline = time.monotonic() + 10
        while self.ui.job_is_running() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertFalse(self.ui.job_is_running())

    def test_real_background_job_captures_results_and_all_plots(self):
        self.ui._select_mode(self.ui._mode_labels.index('Sensitivity & Yield'))
        self.panel.run()
        self.assertTrue(self.ui.job_is_running())
        self.assertFalse(self.panel.parameters.isEnabled())
        self.assertTrue(self.panel.stop_button.isEnabled())
        self.wait_for_task()
        self.assertFalse(self.errors)
        self.assertEqual(self.panel.result['trials'], 64)
        self.assertEqual(self.panel.tabs.currentIndex(), 1)
        self.assertTrue(self.panel.export_button.isEnabled())
        captured = self.panel.result['target_db']
        self.panel.fields['target'].setText('-4')
        self.assertEqual(self.panel.result['target_db'], captured)
        for i in range(self.panel.view.count()):
            self.panel.view.setCurrentIndex(i)
            self.panel.canvas.draw()
        self.panel.summary.sortItems(1, Qt.AscendingOrder)
        self.panel.summary.selectRow(0)
        self.assertEqual(self.panel.parameter_choice.currentIndex(), self.panel.summary.item(0, 0).data(Qt.UserRole))
        with tempfile.TemporaryDirectory() as directory, mock.patch('ibc.tolerance_ui.filedialog.asksaveasfilename', return_value=str(Path(directory)/'study.json')):
            self.panel.export()
            self.wait_for_task()
            exported = json.loads((Path(directory)/'study.json').read_text(encoding='utf-8'))
            self.assertEqual(exported['target_db'], captured)
            self.assertEqual(exported['layers'][1]['tolerances']['thickness']['bound'], 15)

    def test_project_roundtrip_and_invalid_load_preserve_live_state(self):
        state = self.ui._collect_project_state()
        for bad in [dict(state, tolerance_setup=dict(state['tolerance_setup'], target='nan')),
                    dict(state, layers=[dict(state['layers'][0], tolerances={'eps_real': spec(5)})])]:
            with self.assertRaises(ValueError): self.ui._apply_project_state(bad)
            self.assertEqual(self.ui._collect_project_state(), state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'project.json'
            save_project_file(path, state)
            restored = load_project_file(path)
            self.ui._apply_project_state(restored)
        self.assertEqual(self.ui.layers[1].tolerances['thickness']['bound'], 15)
        self.assertEqual(self.panel.capture_setup(), setup(mode=MODES[1]))
        self.assertIsNone(self.panel.result)
        legacy = copy.deepcopy(state)
        legacy.pop('tolerance_setup')
        for layer in legacy['layers']: layer.pop('tolerances', None)
        self.ui._apply_project_state(legacy)
        self.assertEqual(self.panel.capture_setup(), SETUP_DEFAULTS)
        self.assertTrue(all(not layer.tolerances for layer in self.ui.layers))

    def test_layer_edit_move_and_inverse_identity_preserve_tolerances(self):
        for layer, dialog_class in zip(self.ui.layers, (SheetDialog, LayerDialog)):
            dialog = dialog_class(self.ui, initial=layer)
            dialog._on_ok()
            self.assertEqual(dialog.result.tolerances, layer.tolerances)
            dialog.deleteLater()
        self.ui.layer_list.setCurrentRow(1)
        self.ui._move_up()
        self.assertIn('thickness', self.ui.layers[0].tolerances)
        serialized = layer_config_to_dict(self.ui.layers[0])
        self.assertEqual(layer_config_from_dict(serialized).tolerances, self.ui.layers[0].tolerances)
        cfg = UncertaintyConfig(False, 0, 0, 0)
        identity = search_identity(self.ui.layers, [9., 11.], [0.], 'te', cfg, INVERSE_SCORE_MODE_OPTIONS[0])
        self.ui.layers[0].tolerances['thickness']['bound'] = 25
        self.assertEqual(search_identity(self.ui.layers, [9., 11.], [0.], 'te', cfg, INVERSE_SCORE_MODE_OPTIONS[0]), identity)

    def test_stop_keeps_previous_result_and_releases_busy_state(self):
        self.panel.run()
        self.wait_for_task()
        previous = self.panel.result
        entered = __import__('threading').Event()
        def paused(*args, stop_requested, **kwargs):
            entered.set()
            while not stop_requested(): time.sleep(.005)
            raise StopToleranceAnalysis()
        with mock.patch('ibc.tolerance_ui.run_tolerance_study', side_effect=paused):
            self.panel.run()
            self.assertTrue(entered.wait(2))
            self.panel.stop_button.click()
            self.wait_for_task()
        self.assertIs(self.panel.result, previous)
        self.assertFalse(self.panel.stop_button.isEnabled())
        self.assertTrue(self.panel.run_button.isEnabled())
        self.assertIn('previous results retained', self.ui.status_var.get())
        self.assertFalse(self.errors)


if __name__ == '__main__':
    unittest.main()
