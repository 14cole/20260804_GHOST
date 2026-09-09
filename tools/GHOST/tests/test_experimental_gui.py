"""Selection, recipe round trips and real Qt worker execution."""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Backend'))
from PySide6.QtWidgets import QApplication
from ghost_gui import GhostWorkspace
from solver_tab import _SolveWorker
from run_setup import DEFAULT_QUALITY
from test_experimental_cpu import fixture


class ExperimentalGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selection_and_recipe_roundtrip(self):
        workspace = GhostWorkspace()
        try:
            tab = workspace.solver_tab
            self.assertEqual(tab.cmb_solver_method.currentData(), 'direct')
            tab.cmb_lu_precision.setCurrentIndex(tab.cmb_lu_precision.findData('mixed'))
            tab.cmb_solver_method.setCurrentIndex(tab.cmb_solver_method.findData('experimental_cpu'))
            self.assertEqual(tab.cmb_lu_precision.currentData(), 'double')
            self.assertFalse(tab.cmb_lu_precision.isEnabled())
            recipe = tab._capture_run_setup()
            self.assertEqual(recipe['solver_method'], 'experimental_cpu')
            tab.cmb_solver_method.setCurrentIndex(0)
            tab._apply_saved_run_setup(recipe)
            self.assertEqual(tab.cmb_solver_method.currentData(), 'experimental_cpu')
            tab.cmb_scatter_mode.setCurrentIndex(tab.cmb_scatter_mode.findData('bistatic'))
            self.assertEqual(tab.cmb_solver_method.currentData(), 'direct')
            self.assertFalse(tab.cmb_solver_method.isEnabled())
        finally:
            workspace.close()

    def test_worker_returns_experimental_both_channel_fields(self):
        worker = _SolveWorker(snapshot=fixture('pec', 32), source_path='', base_dir='',
            frequencies=[.6], elevations=list(range(519)), units='meters',
            quality_thresholds=DEFAULT_QUALITY, solver_method='experimental_cpu')
        progress = []
        worker.progress.connect(lambda percent, message: progress.append(percent))
        result = worker._run_2d(worker.snapshot, worker._on_progress)
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertEqual(result['metadata']['solver_method_requested'], 'experimental_cpu')
        self.assertEqual(set(result['co_solved_samples']), {'VV', 'HH'})
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
