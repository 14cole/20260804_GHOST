"""Desktop BOR controls and worker option handoff."""
import os,sys,unittest
from pathlib import Path
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.ui.solver import _SolveWorker
from ghost_backend.runs.setup import DEFAULT_QUALITY


class BorExecutionGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_options_select_and_reach_worker(self):
        workspace=GhostWorkspace()
        try:
            tab=workspace.solver_tab
            tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('bor'))
            widget=tab.bor_options_widget
            self.assertFalse(widget.isHidden())
            widget.factorization.setCurrentIndex(widget.factorization.findData('compressed'))
            widget.batch.setValue(17)
            widget.storage.setValue(512)
            options=widget.value()
            self.assertTrue(widget.storage.isEnabled())
            worker=_SolveWorker(snapshot={},source_path='',base_dir='',frequencies=[1.],
                elevations=[0.],units='meters',quality_thresholds=DEFAULT_QUALITY,
                solver_kind='bor',mesh_certification=False,bor_options=options)
            with mock.patch('ghost_backend.ui.solver.solve_monostatic_rcs_bor_survey',return_value={}) as solve:
                worker._run_bor()
                self.assertEqual(solve.call_args[1]['bor_options'],options)
            tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('2d'))
            self.assertTrue(widget.isHidden())
        finally:workspace.close()


if __name__=='__main__':unittest.main()
