"""Selection, recipe round trips and real Qt worker execution."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.ui.solver import _SolveWorker
from ghost_backend.runs.setup import DEFAULT_QUALITY
from test_experimental_cpu import fixture


class ExperimentalGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_worker_run_applies_profile_and_bor_ignores_2d_environment(self):
        from ghost_backend.execution.options import current_options, validate_options
        profile = validate_options(dict(factorization='compressed', blas_threads=1, assembly_threads=2))
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION': 'invalid-for-2d'}):
            worker = _SolveWorker(snapshot=fixture('pec', 24), source_path='', base_dir='',
                frequencies=[.6], elevations=[0.,90.], units='meters',
                quality_thresholds=DEFAULT_QUALITY, solver_method='experimental_cpu',
                mesh_certification=False, execution_options=profile)
            completed, errors, events = [], [], []
            worker.finished.connect(lambda result,path: completed.append(result))
            worker.error.connect(errors.append)
            worker.telemetry.connect(events.append)
            worker.run()
            self.assertFalse(errors)
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]['metadata']['execution_options'], profile)
            self.assertIsNone(current_options())
            self.assertTrue(events)
            bor = _SolveWorker(snapshot={}, source_path='', base_dir='', frequencies=[.6],
                elevations=[0.], units='meters', quality_thresholds=DEFAULT_QUALITY, solver_kind='bor')
            with mock.patch.object(bor, '_execute_run') as run:
                bor.run()
                run.assert_called_once_with()

    def test_selection_and_recipe_roundtrip(self):
        workspace = GhostWorkspace()
        try:
            tab = workspace.solver_tab
            self.assertEqual(tab.cmb_solver_method.currentData(), 'experimental_cpu')
            self.assertEqual(tab.execution_options_widget.value()['factorization'], 'compressed')
            tab.execution_options_widget.set_value(dict(factorization='dense'))
            tab.cmb_solver_method.setCurrentIndex(tab.cmb_solver_method.findData('direct'))
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
        if os.environ.get('GHOST_CPU_FACTORIZATION') == 'compressed':
            self.assertEqual(result['metadata']['solver_method'], 'compressed_experimental_cpu')
            self.assertTrue(result['metadata']['compressed_factors'])
        self.assertEqual(set(result['co_solved_samples']), {'VV', 'HH'})
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 100)

    def test_compressed_gui_admits_fine_mesh_above_legacy_panel_cap(self):
        import ghost_backend.twod.solver as rcs
        from ghost_backend.runs.quality import scale_snapshot_panel_density
        value = fixture('pec', 14000)
        worker = _SolveWorker(snapshot=value, source_path='', base_dir='',
            frequencies=[.6], elevations=[0.], units='meters',
            quality_thresholds=DEFAULT_QUALITY, solver_method='experimental_cpu')
        def build_actual_fine_mesh(**kwargs):
            fine = scale_snapshot_panel_density(kwargs['geometry_snapshot'], 1.5)
            fine['_2d_certification_refinement_factor'] = 1.5
            fine['_2d_certification_base_segment_n'] = [s['properties'][1] for s in value['segments']]
            return len(rcs._build_panels(fine, 1., rcs.C0/.6e9,
                max_panels=kwargs.get('max_panels', rcs.MAX_PANELS_DEFAULT)))
        with mock.patch('ghost_backend.ui.solver.solve_monostatic_rcs_2d_certified', side_effect=build_actual_fine_mesh):
            with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION': 'dense'}):
                with self.assertRaisesRegex(ValueError, 'limit is 20000'):
                    worker._run_2d(value, None)
            with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION': 'compressed'}):
                # Every already-discretized primitive receives at least two
                # panels in the certificate, so this fixture refines to 28k.
                self.assertEqual(worker._run_2d(value, None), 28000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
