"""Independent expected-behavior tests. Failures intentionally expose defects."""
import copy
import csv
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM']='offscreen'
AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/FREDDY')]
import numpy as np
from GRIM_Backend.plotting.modes.compare_mode import rf_agreement_statistics
from GRIM_Backend.ui.table_import import create_table_import_dialog
from PySide6.QtWidgets import QApplication, QMessageBox
from ibc.ui import ImpedanceGui
from ibc.compute import LayerConfig

class NumericalInvariants(unittest.TestCase):
    def test_identical_weak_signals_are_a_positive_control(self):
        x=np.arange(30.)
        a=(1+np.sin(x/5)**2)*1e-12
        self.assertAlmostEqual(rf_agreement_statistics(x,a,a).score,100.,places=10)

    def test_rf_agreement_is_invariant_under_common_power_scaling(self):
        x=np.arange(30.)
        a=1+np.sin(x/5)**2
        b=a*1.01
        original=rf_agreement_statistics(x,a,b)
        for factor in (1e-4,1e-8,1e-10,1e-12):
            with self.subTest(power_scale=factor):
                scaled=rf_agreement_statistics(x,a*factor,b*factor)
                print('RF SCALE',factor,'scores',original.score,scaled.score,'CCC',original.linear_ccc,scaled.linear_ccc)
                self.assertAlmostEqual(original.score,scaled.score,places=8)

class InputAndOutputEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_mapped_import_preserves_available_phase_and_missing_phase(self):
        from GRIM_Backend.io.mapped_table import grid_from_mapped_rows
        control=grid_from_mapped_rows([(2,dict(frequency=9,azimuth=0,elevation=0,polarization='VV',magnitude=2,phase=90)),
            (3,dict(frequency=10,azimuth=0,elevation=0,polarization='VV',magnitude=3,phase=''))],
            'control.csv','RCS power (m²)')
        self.assertAlmostEqual(control.rcs_phase.ravel()[0],math.pi/2)
        self.assertTrue(np.isnan(control.rcs_phase.ravel()[1]))
        with tempfile.TemporaryDirectory(dir=AUDIT) as temp:
            source=Path(temp)/'partial.csv'
            source.write_text('frequency_GHz,azimuth_deg,elevation_deg,magnitude,phase_deg\n9,0,0,2,90\n10,0,0,3,\n',encoding='utf-8')
            from GRIM_Backend.execution.dataset_jobs import _load_dataset_path_task
            self.assertEqual(_load_dataset_path_task((0,str(source)))['status'],'mapping_required')
            dialog=create_table_import_dialog(None,source)
            try:
                dialog.magnitude_format.setCurrentText('RCS power (m²)')
                dialog.submit()
                deadline=time.monotonic()+10
                while dialog.job is not None and time.monotonic()<deadline:
                    self.app.processEvents(); time.sleep(.005)
                self.assertIsNone(dialog.job, 'Import worker did not finish within ten seconds')
                self.assertIsNotNone(dialog.dataset,dialog.status.text())
                np.testing.assert_allclose(dialog.dataset.rcs_power.ravel(),[2,3])
                phase=dialog.dataset.rcs_phase.ravel()
                self.assertAlmostEqual(phase[0],math.pi/2)
                self.assertTrue(np.isnan(phase[1]))
            finally: dialog.deleteLater(); self.app.processEvents()

    def test_freddy_cannot_replace_an_active_material_input(self):
        with tempfile.TemporaryDirectory(dir=AUDIT) as temp:
            source=Path(temp)/'material.csv'
            original='frequency_hz,eps_real,eps_imag,mu_real,mu_imag\n9000000000,4,-0.4,1,0\n11000000000,4,-0.4,1,0\n'
            source.write_text(original,encoding='utf-8')
            ui=ImpedanceGui()
            try:
                ui.layers=[LayerConfig(.12,False,str(source),'',0)]
                ui._refresh_layers()
                ui.f_start_var.set('9');ui.f_stop_var.set('11');ui.f_step_var.set('1')
                ui.output_var.set(str(source))
                with patch('ibc.ui.messagebox.askyesno',return_value=True),patch.object(QMessageBox,'critical',return_value=QMessageBox.Ok):
                    ui._compute_impedance()
                    deadline=time.monotonic()+15
                    while ui.job_is_running() and time.monotonic()<deadline:
                        self.app.processEvents();time.sleep(.005)
                self.assertFalse(ui.job_is_running(), 'Impedance worker did not finish within fifteen seconds')
                print('SOURCE AFTER IMPEDANCE',source.read_text(encoding='utf-8'))
                self.assertEqual(source.read_text(encoding='utf-8'),original,'Analysis output overwrote its five-column material input')
            finally: ui.hide();ui.deleteLater();self.app.processEvents()

if __name__=='__main__': unittest.main(verbosity=2)
