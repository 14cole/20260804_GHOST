import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from ibc.compute import InverseCandidate, LayerConfig
from ibc.ui import ImpedanceGui


class InverseResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.ui = ImpedanceGui()
        self.ui._select_mode(4)
        self.ui.inverse_candidates = [
            InverseCandidate(-12, -12, -12, -12, -12, [0], [''], [100]),
            InverseCandidate(-11, -11, -11, -11, -11, [0], [''], [200]),
        ]
        self.ui.inverse_plot_freqs = [1, 2, 3, 4, 5]
        self.ui.inverse_plot_samples = [[[v] for v in curve] for curve in [
            [-2, -2, -50, -2, -2], [-2, -15, -15, -15, -2]]]
        self.ui.inverse_result_metadata = {'band_sweep': True, 'scores': [-3, -6, -12, -11]}
        self.ui._refresh_inverse_results_list()

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def test_results_take_full_workspace_and_other_modes_restore_their_plots(self):
        self.assertTrue(self.ui.results_pane.isHidden())
        self.ui.inverse_workspace_tabs.setCurrentIndex(1)
        self.assertFalse(self.ui.work_split.isVisibleTo(self.ui.inverse_workspace_tabs))
        self.ui._select_mode(2)
        self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 0)
        self.assertTrue(self.ui.results_pane.isHidden())
        self.assertTrue(self.ui.inverse_workspace_tabs.isTabVisible(1))
        self.assertIs(self.ui.result_pages.currentWidget(), self.ui.analysis_panels['Off Angle'])
        self.ui._select_mode(4)
        self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 1)
        self.ui.inverse_workspace_tabs.setCurrentIndex(0)
        self.assertFalse(self.ui.layers_group.isHidden())
        self.assertTrue(self.ui.results_pane.isHidden())
        self.ui._select_mode(5)
        self.assertFalse(self.ui.results_pane.isHidden())
        self.assertFalse(self.ui.inverse_workspace_tabs.isTabVisible(1))

    def test_sorting_and_threshold_changes_keep_candidate_identity_for_apply_and_save(self):
        table = self.ui.inv_results_list
        table.sortItems(4, Qt.DescendingOrder)
        table.setCurrentCell(0, 1)
        self.assertEqual(self.ui._selected_inverse_index(), 1)
        self.ui.inv_target_db.setValue(-12)
        self.assertEqual(self.ui._selected_inverse_index(), 1)
        self.ui.layers = [LayerConfig(0, False, '', '', 0, is_sheet=True, sheet_resistance=300)]
        with mock.patch.object(self.ui, '_ensure_inverse_result_current'), mock.patch('ibc.ui.messagebox.showinfo'):
            self.ui._apply_inverse_candidate()
        self.assertEqual(self.ui.layers[0].sheet_resistance, 200)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'selected.json')
            with mock.patch.object(self.ui, '_ensure_inverse_result_current'), \
                    mock.patch('PySide6.QtWidgets.QFileDialog.getSaveFileName', return_value=(path, '')):
                self.ui._save_inverse_candidate()
            from ibc.io import load_project_file
            self.assertEqual(load_project_file(Path(path))['layers'][0]['sheet_resistance'], 200)

    def test_display_changes_do_not_alter_search_scores_and_discrete_widths_are_unavailable(self):
        original = copy.deepcopy(self.ui.inverse_candidates)
        checkpoint = {'scores': {'test': (-12, -12, -12, -12, -12)}}
        self.ui._inverse_checkpoint = copy.deepcopy(checkpoint)
        self.ui.inv_target_db.setValue(-18)
        self.ui.inv_curve_mode.setCurrentIndex(1)
        self.ui.inv_percentile_var.set('50')
        self.ui._on_inverse_percentile_changed()
        self.assertEqual(self.ui.inverse_candidates, original)
        self.assertEqual(self.ui._inverse_checkpoint, checkpoint)
        self.ui.inverse_result_metadata['band_sweep'] = False
        self.ui._refresh_inverse_results_list()
        self.assertTrue(all(m.widest_ghz is None for m in self.ui._inverse_metrics))
        self.ui.inv_plot_view.setCurrentIndex(1)
        self.assertIn('Discrete targets', self.ui.inv_axis.get_xlabel())

    def test_plot_views_history_and_save_use_the_dedicated_figure(self):
        self.ui.inverse_workspace_tabs.setCurrentIndex(1)
        self.assertTrue(self.ui.inv_results_list.isHidden())
        self.ui.inv_table_toggle.setChecked(True)
        self.assertFalse(self.ui.inv_results_list.isHidden())
        self.assertIn('Worst analyzed case', self.ui.inv_axis.get_title())
        self.assertEqual(len(self.ui.inv_axis.lines), 3)  # two candidates plus threshold
        self.ui.inv_plot_view.setCurrentIndex(1)
        self.assertIn('Null depth', self.ui.inv_axis.get_title())
        from types import SimpleNamespace
        self.ui._pick_inverse_candidate(SimpleNamespace(artist=self.ui.inv_axis.collections[0], ind=[1]))
        self.assertEqual(self.ui._selected_inverse_index(), 1)
        self.ui.inv_plot_view.setCurrentIndex(2)
        self.assertTrue(self.ui.inv_results_list.isHidden())
        self.assertEqual(list(self.ui.inv_axis.lines[0].get_ydata()), [-3, -6, -12, -12])
        with mock.patch('ibc.ui.filedialog.asksaveasfilename', return_value='example.png'), \
                mock.patch('ibc.ui.messagebox.showinfo'), mock.patch.object(self.ui.inv_figure, 'savefig') as save, \
                mock.patch.object(self.ui.fig, 'savefig') as old_save:
            self.ui._save_plot()
        save.assert_called_once()
        old_save.assert_not_called()

    def test_run_auto_opens_results_and_resume_keeps_complete_history(self):
        self.ui.layers = [LayerConfig(0, False, '', '', 0, is_sheet=True, sheet_resistance=100,
                                     inv_rs_min=100, inv_rs_max=500, inv_rs_accuracy=100)]
        for key, value in [('inv_freq_mode_var', 'Band sweep'), ('inv_target_start_var', '1'),
                           ('inv_target_stop_var', '2'), ('inv_target_step_var', '.5'),
                           ('inv_angle_start_var', '0'), ('inv_angle_stop_var', '0'),
                           ('inv_max_evals_var', '3'), ('inv_top_n_var', '2')]:
            getattr(self.ui, key).set(value)
        self.ui.inv_refine_var.set(False)
        self.ui.inv_uncertainty_var.set(False)
        def now(_name, worker, success, _error):
            self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 0)
            success(worker())
        original = self.ui._score_inverse_candidate
        calls = []
        def score(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            if len(calls) == 2:
                self.ui._inverse_stop_event.set()
            return result
        with mock.patch.object(self.ui, '_run_background_task', side_effect=now), \
                mock.patch.object(self.ui, '_score_inverse_candidate', side_effect=score):
            self.ui._run_inverse_design()
            self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 1)
            self.assertEqual(len(self.ui.inverse_result_metadata['scores']), 2)
            first = self.ui.inverse_result_metadata['scores'][:]
            self.ui._run_inverse_design(resume=True)
        self.assertEqual(self.ui.inverse_result_metadata['scores'][:2], first)
        self.assertEqual(len(self.ui.inverse_result_metadata['scores']), 5)
        self.assertEqual(self.ui.inverse_result_metadata['scores'], self.ui._inverse_checkpoint['score_rows'][::5])
        self.assertTrue(self.ui.inverse_result_metadata['complete'])
        self.assertTrue(self.ui.inverse_result_metadata['band_sweep'])
        self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 1)

    def test_stopped_run_without_curves_keeps_scores_and_history_visible(self):
        self.ui.inverse_plot_samples = []
        self.ui._refresh_inverse_results_list()
        self.assertEqual(self.ui.inv_results_list.rowCount(), 2)
        self.assertIn('unavailable', self.ui.inv_axis.texts[0].get_text())
        self.ui.inv_plot_view.setCurrentIndex(2)
        self.assertEqual(len(self.ui.inv_axis.lines[0].get_ydata()), 4)

    def test_picker_and_plot_checkboxes_follow_original_candidates_after_sort(self):
        table = self.ui.inv_results_list
        table.sortItems(4, Qt.DescendingOrder)
        self.ui.inv_candidate_picker.setCurrentIndex(1)
        self.assertEqual(self.ui._selected_inverse_index(), 1)
        # The sorted first row is #2. Uncheck #1 and retain only #2 on the plot.
        table.item(1, 0).setCheckState(Qt.Unchecked)
        self.assertEqual(self.ui._inverse_checked_indices(), {1})
        self.assertEqual([line.get_label() for line in self.ui.inv_axis.lines], ['#2 selected', 'Target -10 dB'])
        self.ui.inv_target_db.setValue(-12)
        self.assertEqual(self.ui._inverse_checked_indices(), {1})
        self.assertEqual(self.ui.inv_candidate_picker.currentIndex(), 1)

    def test_project_load_clears_previous_results_and_history(self):
        state = self.ui._collect_project_state()
        self.ui._apply_project_state(state)
        self.assertEqual(self.ui.inverse_result_metadata, {})
        self.assertEqual(self.ui.inv_results_list.rowCount(), 0)
        self.assertEqual(self.ui.inverse_workspace_tabs.currentIndex(), 0)

    def test_nonfinite_percentile_is_rejected_instead_of_becoming_worst_case(self):
        self.ui.inv_percentile_var.set('nan')
        with mock.patch('ibc.ui.messagebox.showerror') as error:
            self.ui._on_inverse_percentile_changed()
        error.assert_called_once()
        self.assertEqual(self.ui._current_inverse_percentile(), 10)
