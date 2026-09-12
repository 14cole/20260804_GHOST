"""BOR saved setups and explicit radar/body coordinates for batch drivers."""
import copy, os, sys, tempfile, unittest
from pathlib import Path
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/GHOST'))
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.bor.options import validate_options
from ghost_backend.runs.setup import read_setup, save_setup
from ghost_backend.runs.bor_setup import driver_settings


class BorProfilesUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_exchange_persistence_request_and_busy_controls(self):
        ghost=GhostWorkspace()
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path=Path(temporary.name)/'body.run.json'
        try:
            tab=ghost.solver_tab
            original_2d=tab._capture_run_setup()
            tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('bor'))
            tab.edit_freq_list.setText('1, 2')
            tab.edit_elev_list.setText('0, 31, 90, 180')
            tab.edit_cfie_alpha.setText('.65')
            options=validate_options(dict(factorization='compressed',angle_batch_size=17,
                compressed_storage_mib=256,compression_tile=24,tile_cache_mib=3))
            tab.bor_options_widget.set_value(options)
            recipe=tab._capture_run_setup()
            self.assertTrue(tab.save_run_setup_button.isEnabled())
            self.assertTrue(tab.run_preflight_button.isEnabled())
            save_setup(path,recipe)
            saved=read_setup(path)
            self.assertEqual(saved,recipe)
            settings=driver_settings(saved)
            self.assertEqual(settings['BOR_EXECUTION_OPTIONS'],options)
            self.assertEqual(settings['AZIMUTHS_DEG'],[0.])
            self.assertEqual(settings['ELEVATIONS_DEG'],[90.,59.,0.,-90.])
            self.assertEqual(settings['CFIE_ALPHA'],.65)
            self.assertEqual(settings['BODY_AXIS_EL_DEG'],90.)
            # Existing setups with a radar grid remain loadable in GHOST.
            saved['radar_grid']=dict(azimuths_deg=[0.],elevations_deg=[90.,59.,0.,-90.],
                axis_az_deg=0.,axis_el_deg=90.,roll_deg=0.)
            save_setup(path,saved)
            tab._apply_saved_run_setup(read_setup(path))
            self.assertIn('body coordinates',tab.run_setup_notice.text())
            actual=tab._capture_run_setup()
            np.testing.assert_allclose(actual['aspects_deg'],recipe['aspects_deg'],atol=1e-10,rtol=0)
            self.assertEqual(actual['bor_options'],options)
            self.assertIsNone(actual['radar_grid'])
            bad=copy.deepcopy(saved)
            bad['bor_options']['angle_batch_size']=0
            with self.assertRaises(ValueError):tab._apply_saved_run_setup(bad)
            self.assertEqual(tab._capture_run_setup(),actual)
            tab._set_solving_state(True)
            self.assertFalse(tab.bor_options_widget.isEnabled())
            self.assertFalse(tab.load_run_setup_button.isEnabled())
            tab._set_solving_state(False)
            tab._apply_saved_run_setup(original_2d)
            self.assertEqual(tab._capture_run_setup(),original_2d)
            self.assertTrue(tab.bor_options_widget.isHidden())
        finally:
            tab._set_solving_state(False)
            ghost.close()
            ghost.deleteLater()
            self.app.processEvents()
