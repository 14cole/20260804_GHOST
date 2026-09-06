import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock
from PySide6.QtWidgets import QApplication
from ibc.ui import ImpedanceGui
from ibc.compute import LayerConfig, MaterialTable
from ibc.io import write_material_table, load_project_file
from ibc.inverse_workflow import configure_layers, check_layers, StopInverseSearch


class InverseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ui=ImpedanceGui()
        self.ui.layers=[LayerConfig(0.,False,'','',0.,is_sheet=True,sheet_resistance=100.,inv_rs_min=100.,inv_rs_max=500.,inv_rs_accuracy=100.)]
        for var,value in [('inv_freq_mode_var','Discrete list'),('inv_freq_list_var','1,2'),('inv_angle_start_var','0'),('inv_angle_stop_var','30'),('inv_angle_step_var','30'),('inv_max_evals_var','5'),('inv_top_n_var','3')]:
            getattr(self.ui,var).set(value)
        self.ui.inv_refine_var.set(False)
        self.ui.inv_uncertainty_var.set(False)
        self.results=[]
        def now(_name,worker,success,_error):
            result=worker()
            self.results.append(result)
            success(result)
        for patch in [mock.patch.object(self.ui,'_run_background_task',side_effect=now),mock.patch('ibc.ui.messagebox.showinfo')]:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def test_stop_keeps_only_completed_scores_and_resume_finishes_exact_grid(self):
        original=self.ui._score_inverse_candidate
        calls=[]
        def score(*args,**kwargs):
            key=args[2][0].sheet_resistance
            calls.append(key)
            result=original(*args,**kwargs)
            if len(calls)==2: self.ui._inverse_stop_event.set()
            return result
        with mock.patch.object(self.ui,'_score_inverse_candidate',side_effect=score):
            self.ui._run_inverse_design()
            self.assertEqual(self.ui._inverse_checkpoint['next_index'],2)
            self.assertEqual(len(self.ui.inverse_candidates),2)
            self.assertIn('incomplete',self.results[-1][1])
            self.assertTrue(self.ui.inv_extend_btn.isEnabled())
            self.ui._run_inverse_design(resume=True)
        self.assertEqual(calls,[100.,200.,300.,400.,500.])
        self.assertEqual(self.ui._inverse_checkpoint['next_index'],5)
        self.assertEqual(len(self.ui._inverse_checkpoint['score_rows']),25)
        self.assertFalse(self.ui.inv_extend_btn.isEnabled())
        self.assertIn('Evaluated: 5 of 5 combinations',self.results[-1][1])
        self.ui.inv_freq_list_var.set('1,3')
        with self.assertRaisesRegex(ValueError,'Inputs or material files changed'):
            self.ui._run_inverse_design(resume=True)

    def test_partial_score_is_not_retained(self):
        with mock.patch.object(self.ui,'_score_inverse_candidate',side_effect=StopInverseSearch):
            self.ui._run_inverse_design()
        self.assertEqual(self.ui.inverse_candidates,[])
        self.assertEqual(len(self.ui._inverse_checkpoint['score_rows']),0)
        self.assertEqual(self.ui._inverse_checkpoint['next_index'],0)
        self.assertTrue(self.ui.inv_extend_btn.isEnabled())

    def test_stop_during_plots_resumes_without_repeating_any_scores(self):
        original=self.ui._score_inverse_candidate
        with mock.patch('ibc.ui.compute_angle_metrics_many',side_effect=StopInverseSearch):
            # Patch scoring because it shares the response function with plotting.
            with mock.patch.object(self.ui,'_score_inverse_candidate',return_value=(-1,)*5) as score:
                self.ui._run_inverse_design()
                self.assertEqual(score.call_count,5)
        self.assertEqual(self.ui._inverse_checkpoint['next_index'],5)
        self.assertFalse(self.ui._inverse_checkpoint['plots_complete'])
        self.assertTrue(self.ui.inv_extend_btn.isEnabled())
        with mock.patch.object(self.ui,'_score_inverse_candidate',side_effect=original) as score:
            self.ui._run_inverse_design(resume=True)
            score.assert_not_called()
        self.assertTrue(self.ui._inverse_checkpoint['plots_complete'])
        self.assertFalse(self.ui.inv_extend_btn.isEnabled())

    def test_legacy_seed_budget_and_refinement_cannot_limit_or_change_the_grid(self):
        self.ui.inv_max_evals_var.set('1')
        self.ui.inv_seed_var.set('not a seed')
        self.ui.inv_refine_var.set(True)
        with mock.patch('ibc.ui._scipy_optimize.minimize') as refine, mock.patch('ibc.ui.random.Random') as random:
            self.ui._run_inverse_design()
            refine.assert_not_called()
            random.assert_not_called()
        first=copy.deepcopy(self.ui.inverse_candidates)
        self.ui.inv_seed_var.set('9')
        self.ui._ensure_inverse_result_current()
        self.ui._run_inverse_design()
        self.assertEqual(self.ui.inverse_candidates,first)
        self.assertEqual(self.ui._inverse_checkpoint['next_index'],5)
        self.assertEqual(len(self.ui.inverse_candidates),3)

    def test_every_multi_layer_combination_is_analyzed_even_when_keep_best_is_one(self):
        from itertools import product
        self.ui.layers.append(LayerConfig(0.,False,'','',0.,is_sheet=True,sheet_resistance=10.,
                                         inv_rs_min=10.,inv_rs_max=30.,inv_rs_accuracy=10.))
        self.ui.inv_top_n_var.set('1')
        calls=[]
        def score(_f,_a,layers,*args):
            values=tuple(l.sheet_resistance for l in layers)
            calls.append(values)
            score=-sum(values)
            return (score,)*5
        with mock.patch.object(self.ui,'_score_inverse_candidate',side_effect=score):
            self.ui._run_inverse_design()
        self.assertEqual(calls,list(product([100.,200.,300.,400.,500.],[10.,20.,30.])))
        self.assertEqual(len(self.ui.inverse_candidates),1)
        self.assertEqual(self.ui.inverse_candidates[0].sheet_resistance_ohm,[500.,30.])
        self.assertEqual(self.ui._inverse_checkpoint['next_index'],15)

    def test_missing_step_is_actionable_and_live_count_includes_tolerance_cases(self):
        self.ui._refresh_inverse_work_count()
        self.assertIn('5 combinations × 2 frequencies × 2 angles × 1',self.ui.inv_work_count.text())
        self.ui.layers[0].inv_rs_accuracy=None
        self.ui._refresh_inverse_work_count()
        self.assertIn('Layer 1: set a step',self.ui.inv_work_count.text())
        with mock.patch('ibc.ui.messagebox.showerror') as error:
            self.ui._run_inverse_design()
        self.assertIn('set a step',error.call_args.args[1])
        self.assertEqual(self.results,[])

    def test_live_count_never_allocates_the_frequency_sweep(self):
        self.ui.inv_freq_mode_var.set('Band sweep')
        self.ui.inv_target_start_var.set('2')
        self.ui.inv_target_stop_var.set('4')
        self.ui.inv_target_step_var.set('0.00000001')
        with mock.patch('ibc.inverse_workflow.make_frequency_sweep') as sweep:
            self.ui._refresh_inverse_work_count()
            sweep.assert_not_called()
        self.assertIn('200,000,001 frequencies',self.ui.inv_work_count.text())

    def test_loading_project_clears_interrupted_resume_state(self):
        with mock.patch.object(self.ui,'_score_inverse_candidate',side_effect=StopInverseSearch):
            self.ui._run_inverse_design()
        self.assertTrue(self.ui.inv_extend_btn.isEnabled())
        self.ui._apply_project_state(self.ui._collect_project_state())
        self.assertIsNone(self.ui._inverse_checkpoint)
        self.assertIsNone(self.ui._inverse_progress)
        self.assertFalse(self.ui.inv_extend_btn.isEnabled())

    def test_fixed_bulk_and_material_provenance_and_separate_save(self):
        material=Path(self.temp.name)/'bulk.csv'
        write_material_table(material,MaterialTable([1.,3.],[3-.1j]*2,[1.]*2))
        self.ui.layers.append(LayerConfig(.125,False,str(material),'',0.))
        self.ui._run_inverse_design()
        self.assertTrue(all(c.thickness_in[1]==.125 for c in self.ui.inverse_candidates))
        before=copy.deepcopy(self.ui.layers)
        destination=Path(self.temp.name)/'candidate.json'
        self.ui.inv_results_list.setCurrentRow(0)
        with mock.patch('PySide6.QtWidgets.QFileDialog.getSaveFileName',return_value=(str(destination),'')):
            self.ui._save_inverse_candidate()
        self.assertEqual(self.ui.layers,before)
        self.assertTrue(destination.is_file())
        saved=load_project_file(destination)
        self.assertEqual(saved['layers'][1]['thickness_in'],.125)
        self.assertEqual(Path(saved['controls']['output']).parent, destination.parent)
        self.assertEqual(Path(saved['controls']['output']).name,'candidate_impedance.csv')
        material.write_text(material.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'material files changed'):
            self.ui._ensure_inverse_result_current()

    def test_layer_setup_atomic_and_actionable(self):
        before=copy.deepcopy(self.ui.layers)
        with self.assertRaisesRegex(ValueError,'Layer 1'):
            configure_layers(before,[dict(vary=True,minimum=10,maximum=1,step='')])
        self.assertEqual(self.ui.layers,before)
        fixed=configure_layers(before,[dict(vary=False,minimum='',maximum='',step='')])
        self.assertIsNone(fixed[0].inv_rs_min)
        self.ui.layers.append(LayerConfig(.1,False,str(Path(self.temp.name)/'missing.csv'),'',0.))
        with self.assertRaisesRegex(ValueError,'Layer 2'):
            check_layers(self.ui.layers,[1.])
