"""Captured validation, new controls, and real checkpointed desktop workers."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QFontDatabase, QFont
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.ui.geometry import GeometryTab
from ghost_backend.ui.solver import _SolveWorker
from ghost_backend.geometry.io import Segment
from ghost_backend.geometry.validation import GeometryAudit
from ghost_backend.runs.setup import DEFAULT_QUALITY, RunSetupMixin
from test_execution_options import setup_record
from test_experimental_cpu import fixture


class PipelineGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        font=Path('C:/Windows/Fonts/segoeui.ttf')
        if font.exists():
            identity=QFontDatabase.addApplicationFont(str(font))
            cls.app.setFont(QFont(QFontDatabase.applicationFontFamilies(identity)[0],9))

    def drain(self, tab):
        limit=time.monotonic()+5
        while tab._validation_worker is not None and time.monotonic()<limit:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(tab._validation_worker)

    def test_worker_validation_and_stale_result_discard(self):
        tab=GeometryTab()
        tab.segments=[Segment('sheet','1',['1','1','0','0','0'],[0.,1.],[0.,0.])]
        try:
            with mock.patch.object(QMessageBox,'warning') as warning, mock.patch.object(QMessageBox,'information'):
                tab.validate_geometry()
                self.assertIsNotNone(tab._validation_worker)
                self.drain(tab)
                self.assertEqual(tab.btn_validate.text(),'Validate')
                self.assertTrue(warning.called)  # Open-chain advisory preserved.
            tab.issue_rows=set()
            tab._validation_version=7
            with mock.patch.object(QMessageBox,'warning') as warning:
                tab._validation_ready(6,([('ERROR',0,'stale')],{0}))
                warning.assert_not_called()
            self.assertEqual(tab.issue_rows,set())
            self.assertIn('changed',tab.lbl_status.text())
        finally:
            tab.close()

    def test_validation_checks_crossings_and_cancels_without_widgets(self):
        segments=[Segment('a','1',['1','1','0','0','0'],[0.,1.],[0.,1.]),
                  Segment('b','1',['1','1','0','0','0'],[0.,1.],[1.,0.])]
        findings,rows=GeometryAudit(segments,[],[],'').run()
        self.assertEqual(rows,{0,1})
        self.assertTrue(any('intersection' in message for _,_,message in findings))
        def abort():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):
            GeometryAudit(segments,[],[],'',abort).run()

    def test_normals_use_collections_and_bound_labels(self):
        tab=GeometryTab()
        try:
            tab.segments=[Segment(str(i),'1',['1','1','0','0','0'],[float(i),float(i)+.5],[0.,0.]) for i in range(1000)]
            tab.chk_show_normals.setChecked(True)
            self.assertEqual(len(tab.normal_artists),2)
            tab._selected_row=10
            tab._render_normals()
            self.assertEqual(len(tab.normal_artists),3)
        finally:tab.close()

    def test_new_settings_roundtrip_and_actual_preflight(self):
        workspace=GhostWorkspace()
        try:
            tab=workspace.solver_tab
            preset=tab.geometry_preset_combo
            preset.setCurrentIndex(preset.findData('adaptive'))
            self.assertEqual(tab.execution_options_widget.value()['factorization'],'adaptive')
            self.assertFalse(tab.cmb_solver_method.isEnabled())
            tab.execution_options_widget.mesh_combo.setCurrentIndex(1)
            record=tab._capture_run_setup()
            self.assertEqual(record['execution_options']['mesh_strategy'],'local')
            tab._apply_saved_run_setup(record)
            self.assertEqual(tab.execution_options_widget.value()['mesh_strategy'],'local')
            note=RunSetupMixin._run_setup_summary(None,fixture('pec',24),'',setup_record({'factorization':'adaptive'}))
            self.assertIn('Planned backend: dense',note)
            tab.cmb_scatter_mode.setCurrentIndex(tab.cmb_scatter_mode.findData('bistatic'))
            self.assertEqual(tab.execution_options_widget.value()['mesh_strategy'],'global')
            self.assertFalse(tab.chk_frequency_checkpoints.isEnabled())
        finally:workspace.close()

    def test_real_desktop_sweep_checkpoint_resume_and_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs=dict(snapshot=fixture('pec',48),source_path='',base_dir='',frequencies=[.6,.8],
                elevations=[0.,90.],units='meters',quality_thresholds=DEFAULT_QUALITY,
                solver_method='experimental_cpu',mesh_certification=True,
                execution_options={'factorization':'adaptive'},checkpoint_directory=directory)
            for expected_reused in (0,2):
                worker=_SolveWorker(**kwargs)
                completed,errors=[],[]
                worker.finished.connect(lambda result,path:completed.append(result))
                worker.error.connect(errors.append)
                worker.run()
                self.assertFalse(errors,errors)
                self.assertEqual(len(completed),1)
                result=completed[0]
                self.assertTrue(result['metadata']['mesh_convergence_certified'])
                self.assertEqual(result['metadata']['frequency_checkpoints']['reused'],expected_reused)
                if expected_reused:
                    self.assertEqual(result['metadata']['runtime_profile']['stage_seconds'],{})
                    self.assertIsNone(result['metadata']['runtime_profile']['sampled_peak_process_rss_bytes'])


if __name__=='__main__':unittest.main()
