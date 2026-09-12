"""GHOST saved 2D profiles preserve physics and execution resources."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/GHOST/ghost_backend'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/GHOST'))
from PySide6.QtWidgets import QApplication
from ghost_backend.execution.options import validate_options, efficient_defaults, geometry_preset
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.runs.setup import read_setup, save_setup


class ExecutionProfilesUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'profile.run.json'

    def test_presets_switch_atomically_and_preserve_physics_and_host_constraints(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ghost = GhostWorkspace()
            receiver_window = GhostWorkspace()
            receiver = receiver_window.solver_tab
            restored_window = None
            try:
                tab = ghost.solver_tab
                self.assertEqual(tab.geometry_preset_combo.currentData(), 'large')
                self.assertEqual(receiver.geometry_preset_combo.currentData(), 'large')
                tab.cmb_accuracy_target.setCurrentIndex(tab.cmb_accuracy_target.findData('tight'))
                tab.chk_mesh_certification.setChecked(False)
                tab.execution_options_widget.set_value(dict(efficient_defaults(),
                    ram_budget_gib=4.250000017, temporary_directory=str(Path.cwd())))
                for name in ('small', 'large', 'balanced', 'large', 'small'):
                    tab.geometry_preset_combo.setCurrentIndex(tab.geometry_preset_combo.findData(name))
                    recipe = tab._capture_run_setup()
                    expected = geometry_preset(name)
                    expected['execution_options'].update(ram_budget_gib=4.250000017,
                                                         temporary_directory=str(Path.cwd()))
                    for key, value in expected.items():
                        self.assertEqual(recipe[key], value)
                    self.assertEqual(recipe['accuracy'], 'tight')
                    self.assertFalse(recipe['mesh_certification'])
                    receiver._apply_saved_run_setup(recipe)
                    self.assertEqual(receiver.geometry_preset_combo.currentData(), name)
                    self.assertEqual(receiver._capture_run_setup(), recipe)
                save_setup(self.path, receiver._capture_run_setup())
                restored_window = GhostWorkspace()
                restored = restored_window.solver_tab
                restored._apply_saved_run_setup(read_setup(self.path))
                self.assertEqual(restored.geometry_preset_combo.currentData(), 'small')
                self.assertEqual(restored._capture_run_setup(), recipe)
                restored.geometry_preset_combo.setCurrentIndex(restored.geometry_preset_combo.findData('balanced'))
                tab._apply_saved_run_setup(restored._capture_run_setup())
                self.assertEqual(tab.geometry_preset_combo.currentData(), 'balanced')
                for item in (tab, restored):
                    item.execution_options_widget.batch_spin.setValue(37)
                    self.assertEqual(item.geometry_preset_combo.currentIndex(), -1)
                    self.assertIn('Custom', item.geometry_preset_combo.placeholderText())
                custom = tab._capture_run_setup()
                receiver._apply_saved_run_setup(custom)
                self.assertEqual(receiver.geometry_preset_combo.currentIndex(), -1)
                self.assertEqual(receiver._capture_run_setup(), custom)
                tab.execution_options_widget.temp_edit.setText('relative/path')
                self.assertEqual(tab.geometry_preset_combo.currentIndex(), -1)
                with self.assertRaises(ValueError):
                    tab._capture_run_setup()
            finally:
                for widget in (ghost, receiver_window, restored_window):
                    if widget is not None:
                        widget.close()
                        widget.deleteLater()
                self.app.processEvents()

    def test_presets_respect_busy_bistatic_and_bor_modes(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ghost = GhostWorkspace()
            receiver_window = GhostWorkspace()
            receiver = receiver_window.solver_tab
            try:
                tab = ghost.solver_tab
                tab._set_solving_state(True)
                self.assertFalse(tab.geometry_preset_combo.isEnabled())
                tab._set_solving_state(False)
                self.assertTrue(tab.geometry_preset_combo.isEnabled())
                tab.cmb_scatter_mode.setCurrentIndex(tab.cmb_scatter_mode.findData('bistatic'))
                self.assertFalse(tab.geometry_preset_combo.isEnabled())
                self.assertEqual(tab.geometry_preset_combo.currentIndex(), -1)
                self.assertEqual(tab._capture_run_setup()['execution_options']['factorization'], 'dense')
                bistatic = tab._capture_run_setup()
                tab.cmb_scatter_mode.setCurrentIndex(0)
                self.assertTrue(tab.geometry_preset_combo.isEnabled())
                tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('bor'))
                tab._apply_saved_run_setup(bistatic)
                self.assertIn('Azimuth', tab.lbl_angle_list.text())
                self.assertFalse(tab.edit_obs_angles.isHidden())
                self.assertFalse(tab.geometry_preset_combo.isEnabled())
                self.assertEqual(tab._capture_run_setup(), bistatic)
                tab.cmb_scatter_mode.setCurrentIndex(0)
                for item, solver in ((tab, tab.cmb_solver_kind), (receiver, receiver.cmb_solver_kind)):
                    item.geometry_preset_combo.setCurrentIndex(item.geometry_preset_combo.findData('large'))
                    solver.setCurrentIndex(solver.findData('bor'))
                    self.assertFalse(item.geometry_preset_combo.isEnabled())
                    self.assertEqual(item.geometry_preset_combo.currentIndex(), -1)
                    self.assertIn('2D monostatic', item.geometry_preset_notice.text())
                    solver.setCurrentIndex(solver.findData('2d'))
                    self.assertTrue(item.geometry_preset_combo.isEnabled())
                    self.assertEqual(item.geometry_preset_combo.currentData(), 'large')
                    self.assertEqual(item._capture_run_setup()['solver_method'], 'experimental_cpu')
            finally:
                ghost.close()
                receiver_window.close()
                ghost.deleteLater()
                receiver_window.deleteLater()
                self.app.processEvents()

    def test_advanced_controls_are_collapsed_without_changing_the_setup(self):
        ghost = GhostWorkspace()
        receiver_window = GhostWorkspace()
        receiver = receiver_window.solver_tab
        try:
            tab = ghost.solver_tab
            for item in (tab, receiver):
                before = item._capture_run_setup()
                advanced = item.advanced_settings_widget
                self.assertTrue(advanced.isHidden())
                self.assertTrue(advanced.isAncestorOf(item.execution_options_widget))
                self.assertTrue(advanced.isAncestorOf(item.save_run_setup_button))
                self.assertFalse(advanced.isAncestorOf(item.geometry_preset_combo))
                item.btn_advanced_settings.setChecked(True)
                self.assertFalse(advanced.isHidden())
                item.btn_advanced_settings.setChecked(False)
                self.assertTrue(advanced.isHidden())
                self.assertEqual(item._capture_run_setup(), before)
            for control in (tab.cmb_lu_precision, tab.cmb_solver_method,
                            tab.cmb_accuracy_target, tab.chk_mesh_certification, tab.edit_cfie_alpha,
                            tab.edit_quality_residual_max, tab.btn_currents, tab.btn_solver_report):
                self.assertTrue(tab.advanced_settings_widget.isAncestorOf(control))
            for control in (tab.edit_geo_path, tab.cmb_solver_kind, tab.cmb_units, tab.cmb_freq_mode, tab.cmb_elev_mode,
                            tab.cmb_scatter_mode, tab.run_preflight_button, tab.edit_output, tab.btn_run):
                self.assertFalse(tab.advanced_settings_widget.isAncestorOf(control))
            tab.cmb_elev_mode.setCurrentIndex(1)
            tab.edit_elev_start.setText('0')
            tab.edit_elev_stop.setText('360')
            tab.edit_elev_step.setText('1')
            self.assertEqual(tab._capture_run_setup()['angles_deg'], list(range(361)))
            self.assertTrue(tab.edit_elev_list.isHidden())
            self.assertFalse(tab.elev_sweep_row.isHidden())
            self.assertIn('Azimuth', tab.lbl_angle_sweep.text())
            tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData('bor'))
            self.assertIn('Aspect', tab.lbl_angle_sweep.text())
        finally:
            ghost.close()
            receiver_window.close()
            ghost.deleteLater()
            receiver_window.deleteLater()
            self.app.processEvents()

    def test_new_windows_use_efficient_defaults_and_saved_dense_can_be_restored(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            ghost = GhostWorkspace()
            receiver_window = GhostWorkspace()
            receiver = receiver_window.solver_tab
            restored_window = None
            try:
                expected = efficient_defaults()
                for tab in (ghost.solver_tab,receiver):
                    self.assertEqual(tab.execution_options_widget.value(), expected)
                    recipe = tab._capture_run_setup()
                    self.assertEqual(recipe['solver_method'], 'experimental_cpu')
                    self.assertEqual(recipe['lu_precision'], 'double')
                    self.assertTrue(recipe['mesh_certification'])
                recipe = receiver._capture_run_setup()
                recipe.update(solver_method='direct', lu_precision='mixed',
                              execution_options=validate_options(dict(factorization='dense', blas_threads=1)))
                receiver._apply_saved_run_setup(recipe)
                save_setup(self.path, receiver._capture_run_setup())
                restored_window = GhostWorkspace()
                restored = restored_window.solver_tab
                restored._apply_saved_run_setup(read_setup(self.path))
                self.assertEqual(restored._capture_run_setup(), recipe)
                restored.execution_options_widget.defaults_button.click()
                self.assertEqual(restored.execution_options_widget.value(), expected)
                self.assertEqual(restored.cmb_solver_method.currentData(), 'experimental_cpu')
                self.assertEqual(restored.cmb_lu_precision.currentData(), 'double')
            finally:
                for widget in (ghost, receiver_window, restored_window):
                    if widget is not None:
                        widget.close()
                        widget.deleteLater()
                self.app.processEvents()

    def test_saved_file_preserves_complete_profile(self):
        ghost = GhostWorkspace()
        receiver_window = GhostWorkspace()
        receiver = receiver_window.solver_tab
        restored_window = None
        try:
            tab = ghost.solver_tab
            profile = validate_options(dict(factorization='compressed', ram_budget_gib=4.250000017,
                compressed_storage_mib=256, assembly_threads='auto', blas_threads=2,
                angle_batch_size=37, rhs_compression='on', assembly_tile=96,
                far_quadrature_order=12, far_grading=False))
            tab.execution_options_widget.set_value(profile)
            recipe = tab._capture_run_setup()
            self.assertEqual(recipe['solver_method'], 'experimental_cpu')
            self.assertEqual(recipe['lu_precision'], 'double')
            receiver._apply_saved_run_setup(recipe)
            self.assertEqual(receiver._capture_run_setup(), recipe)
            save_setup(self.path, receiver._capture_run_setup())
            with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION': 'dense'}):
                restored_window = GhostWorkspace()
                restored = restored_window.solver_tab
                restored._apply_saved_run_setup(read_setup(self.path))
            self.assertEqual(restored.execution_options_widget.value(), profile)
            self.assertEqual(restored.cmb_solver_method.currentData(), 'experimental_cpu')
            before = tab._capture_run_setup()
            invalid = copy.deepcopy(recipe)
            invalid['execution_options']['blas_threads'] = 0
            with self.assertRaises(ValueError):
                tab._apply_saved_run_setup(invalid)
            self.assertEqual(tab._capture_run_setup(), before)
            dense = copy.deepcopy(recipe)
            dense['execution_options']['factorization'] = 'dense'
            dense.update(solver_method='direct', lu_precision='mixed')
            tab._apply_saved_run_setup(dense)
            receiver._apply_saved_run_setup(dense)
            self.assertEqual(tab._capture_run_setup(), dense)
            self.assertEqual(receiver._capture_run_setup(), dense)
        finally:
            for widget in (ghost, receiver_window, restored_window):
                if widget is not None:
                    widget.close()
                    widget.deleteLater()
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
