"""Result integrity, transactional restoration and cancellable recipe searches."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from PySide6.QtWidgets import QApplication
from ibc import design_search as search
from ibc.compute import MaterialTable, UncertaintyConfig
from ibc.io import read_material_table, write_material_table
from ibc.ui import ImpedanceGui
from ibc.ui_options import MIX_OBJECTIVE_PROPERTY


class ReleaseReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / name for name in ('a.csv', 'b.csv')]
        for path, epsilon in zip(self.paths, (2., 6.)):
            self.write_material(path, epsilon)
        self.ui = ImpedanceGui()
        self.ui.mix_components = [self.ui._coerce_mix_component({'file':str(p), 'parts':1., 'units':'volume_percent'}) for p in self.paths]
        self.ui.mix_rule_var.set('linear')
        self.ui.mix_freq_mode_var.set('Discrete list')
        self.ui.mix_freq_list_var.set('1,2')
        self.ui.mix_max_evals_var.set('5')
        self.ui.mix_top_n_var.set('2')
        self.ui.mix_refine_var.set(False)
        self.ui.mix_uncertainty_var.set(False)
        self.addCleanup(self.ui.deleteLater)

    def tearDown(self):
        self.app.processEvents()

    def write_material(self, path, epsilon):
        write_material_table(path, MaterialTable([1., 2.], [complex(epsilon, -.1)]*2, [1+0j]*2))

    def request(self, **changes):
        request = search.MixSearchRequest(
            UncertaintyConfig(False, 0., 0., 0.),
            [{'file':str(p), 'parts':1., 'density':1.} for p in self.paths],
            [1., 2.], 'linear', True, False,
            {'freqs':[1., 2.], 'eps':[4.-.1j]*2, 'mu':[1+0j]*2, 'w_eps':1., 'w_mu':1.},
            .125, None, 'Worst-case', 123, [.5, .5], [.5, .5], 200, 1,
            False, 'constant target', '1 and 2 GHz', True)
        return replace(request, **changes)

    def test_preview_export_and_apply_use_captured_values_after_file_change(self):
        self.ui._preview_mix()
        self.write_material(self.paths[1], 18.)
        for index, action in enumerate((self.ui._export_mix_material, self.ui._apply_mix_as_layer)):
            destination = self.root / f'export{index}.csv'
            with mock.patch('ibc.ui.filedialog.asksaveasfilename', return_value=str(destination)), \
                 mock.patch('ibc.ui.messagebox.showinfo'), mock.patch('ibc.ui.messagebox.showerror') as error:
                action()
            error.assert_not_called()
            self.assertEqual([z.real for z in read_material_table(destination).eps_r], [4., 4.])
        self.assertEqual(self.ui.mix_preview['eps_re'], [4., 4.])

    def test_search_candidate_export_is_independent_of_mutable_source_files(self):
        candidates, displays, _ = search.run_mix_search(self.request())
        self.ui.mix_candidates, self.ui.mix_plot_data = candidates, displays
        self.write_material(self.paths[1], 18.)
        table, thickness, _ = self.ui._current_mix_material()
        self.assertEqual([z.real for z in table.eps_r], [4., 4.])
        self.assertEqual(thickness, displays[0]['thickness_in'])
        self.ui.mix_components.reverse()
        # A captured result remains the same even if a caller changes the list
        # directly. GUI component-edit actions additionally invalidate results.
        self.assertEqual(self.ui._current_mix_material()[0].eps_r, table.eps_r)
        self.ui._invalidate_mix_results()
        with self.assertRaisesRegex(ValueError, 'Preview'):
            self.ui._current_mix_material()

    def test_invalid_project_leaves_controls_layers_and_results_unchanged(self):
        self.ui._preview_mix()
        before = self.ui._collect_project_state()
        preview = self.ui.mix_preview
        for mutation in ('amount', 'component_type', 'controls'):
            state = copy.deepcopy(before)
            state['controls']['f_start'] = '999'
            if mutation == 'amount':
                state['mixes']['components'][1]['parts'] = 'invalid'
            elif mutation == 'component_type':
                state['mixes']['components'][1] = 123
            else:
                state['controls']['wave_pol'] = 'invalid'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.ui._apply_project_state(state)
            self.assertEqual(self.ui._collect_project_state(), before)
            self.assertIs(self.ui.mix_preview, preview)

    def test_fixed_recipe_is_evaluated_once_and_reports_progress(self):
        progress = mock.Mock()
        with mock.patch.object(search, 'project_bounded_fractions', wraps=search.project_bounded_fractions) as project:
            candidates, _, message = search.run_mix_search(self.request(refine=True), progress=progress)
        self.assertEqual(project.call_count, 1)
        self.assertEqual(candidates[0].fractions, [.5, .5])
        self.assertIn('Evaluated 1 bounded recipe', message)
        self.assertIn('refinement evaluations 0', message)
        progress.assert_called_with(1, 1, 'Sampling')

    def test_stop_during_sampling_does_not_generate_the_whole_search(self):
        stop = threading.Event()
        def progress(done, total, phase):
            if done == 3:
                stop.set()
        with mock.patch.object(search, 'project_bounded_fractions', wraps=search.project_bounded_fractions) as project:
            with self.assertRaises(search.StopMixSearch):
                search.run_mix_search(self.request(lower=[0.,0.], upper=[1.,1.], max_evals=1000000),
                                      stop_requested=stop.is_set, progress=progress)
        self.assertLess(project.call_count, 20)

    def test_refinement_budget_is_bounded_and_stop_propagates_through_optimizer(self):
        class EndlessOptimizer:
            @staticmethod
            def minimize(objective, x, **kwargs):
                while True:
                    objective(x)
        req = self.request(lower=[0.,0.], upper=[1.,1.], max_evals=1, refine=True)
        with mock.patch.object(search, 'MIX_REFINE_MAX_EVALS', 4):
            _, _, message = search.run_mix_search(req, optimizer=EndlessOptimizer)
        self.assertIn('refinement evaluations 4', message)
        stop = threading.Event()
        def progress(done, total, phase):
            if phase == 'Refining':
                stop.set()
        with self.assertRaises(search.StopMixSearch):
            search.run_mix_search(req, optimizer=EndlessOptimizer,
                                  stop_requested=stop.is_set, progress=progress)

    def test_ui_does_not_publish_search_after_inputs_change(self):
        self.ui.mix_objective_var.set(MIX_OBJECTIVE_PROPERTY)
        def run(_name, worker, success, _error):
            result = worker()
            self.ui.mix_thickness_var.set('.25')
            success(result)
        with mock.patch.object(self.ui, '_run_background_task', side_effect=run), \
             mock.patch('ibc.ui.messagebox.showwarning') as warning:
            self.ui._run_mix_design()
        self.assertIn('Inputs changed', warning.call_args.args[1])
        self.assertFalse(self.ui.mix_candidates)

    def test_stop_button_controls_only_the_material_search(self):
        self.ui._mix_active = True
        self.ui._set_task_state(True, 'Searching')
        self.assertTrue(self.ui.mix_stop_btn.isEnabled())
        self.assertFalse(self.ui.inv_stop_btn.isEnabled())
        self.ui.mix_stop_btn.click()
        self.assertTrue(self.ui._mix_stop_event.is_set())
        self.ui._mix_active = False
        self.ui._set_task_state(False, 'Ready')
        self.assertFalse(self.ui.mix_stop_btn.isEnabled())
