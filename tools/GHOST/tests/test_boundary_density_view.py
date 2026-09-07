"""Boundary densities stay in the solver view with captured units and inputs."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Backend"))
import solver_tab
from PySide6.QtWidgets import QApplication


def density_result():
    channels = {}
    for label, real, imag in (("VV", [3., 0.], [4., 0.]), ("HH", [0., -2.], [1., 0.])):
        channels[label] = dict(
            element_count=2, coordinate_units="meters", formulation=f"{label} SLP density",
            centers_x=[.0254, .254], centers_y=[0., 0.],
            normals_x=[0., 0.], normals_y=[1., 1.], lengths=[.0254, .0254],
            density_real=real, density_imag=imag,
        )
    return dict(frequency_ghz=3., cut_angle_deg=12., geometry_units_in="inches", channels=channels)


class BoundaryDensityViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.ui = solver_tab.SolverTab()
        self.ui._active_density_run_id = 1
        self.ui._density_abort_event = threading.Event()
        self.ui._pending_density_context = {"uses_geometry_tab": True, "geometry_stale": False, "input_sha256": {}}

    def tearDown(self):
        self.ui.deleteLater()
        self.app.processEvents()

    def test_completed_result_plots_elements_and_complex_values_in_captured_units(self):
        result = density_result()
        self.ui.cmb_units.setCurrentText("meters")
        self.ui._on_density_finished(1, result)
        self.assertIs(self.ui.last_density_result, result)
        self.assertEqual(self.ui.cmb_result_view.currentData(), "density_abs")
        self.assertEqual(len(self.ui.canvas.fig.axes), 4)
        vv, hh = self.ui.canvas.fig.axes[:2]
        self.assertEqual(vv.get_xlabel(), "X (in)")
        np.testing.assert_allclose(vv.collections[1].get_array(), [5., 0.])
        np.testing.assert_allclose(hh.collections[1].get_array(), [1., 2.])
        np.testing.assert_allclose(np.sort(vv.collections[1].get_segments(), axis=1),
                                   [[[.5, 0.], [1.5, 0.]], [[9.5, 0.], [10.5, 0.]]])
        self.assertEqual(self.ui.table_results.rowCount(), 4)
        self.assertEqual(self.ui.table_results.item(0, 6).text(), "5")
        self.assertEqual(self.ui.table_results.item(1, 7).text(), "undefined")
        self.assertIn("3 GHz | 12 deg", self.ui.lbl_result_details.text())
        self.assertFalse(self.ui._is_computing_density)

    def test_phase_view_and_rcs_switch_preserve_results_and_reset_colorbars(self):
        rcs = {"samples": [{"frequency_ghz": 3., "theta_scat_deg": 0.,
                            "polarization": "VV", "rcs_linear": 1.}], "metadata": {}}
        self.ui.last_result = rcs
        self.ui._on_density_finished(1, density_result())
        self.ui.cmb_result_view.setCurrentIndex(2)
        vv, hh = self.ui.canvas.fig.axes[:2]
        self.assertAlmostEqual(float(vv.collections[1].get_array()[0]), np.degrees(np.arctan2(4, 3)))
        self.assertTrue(np.ma.getmaskarray(vv.collections[1].get_array())[1])
        np.testing.assert_allclose(hh.collections[1].get_array(), [90., 180.])
        self.ui.cmb_result_view.setCurrentIndex(0)
        self.assertIs(self.ui.last_result, rcs)
        self.assertEqual(len(self.ui.canvas.fig.axes), 1)
        self.assertEqual(len(self.ui.canvas.ax.lines), 1)
        self.assertEqual(self.ui.table_results.rowCount(), 1)
        self.ui.cmb_result_view.setCurrentIndex(1)
        self.assertEqual(len(self.ui.canvas.fig.axes), 4)
        self.assertEqual(self.ui.table_results.rowCount(), 4)

    def test_input_replacement_and_queued_cancellation_leave_prior_density_intact(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "body.geo"
                path.write_text("original")
                self.ui._pending_density_context = {
                    "input_sha256": {str(path): solver_tab._stable_sha256(str(path))},
                }
                prior = density_result()
                self.ui.last_density_result = prior
                if cancel:
                    self.ui._density_abort_event.set()
                else:
                    path.write_text("changed")
                with mock.patch.object(self.ui, "_plot_boundary_densities") as plot, \
                     mock.patch.object(solver_tab.QMessageBox, "warning") as warning:
                    self.ui._on_density_finished(1, density_result())
                plot.assert_not_called()
                self.assertIs(self.ui.last_density_result, prior)
                self.assertEqual(warning.call_count, 0 if cancel else 1)

    def test_invalid_arrays_do_not_replace_prior_results(self):
        prior = density_result()
        self.ui.last_density_result = prior
        bad = copy.deepcopy(prior)
        bad["channels"]["HH"]["density_real"] = [float("nan"), 0.]
        with mock.patch.object(solver_tab.QMessageBox, "warning") as warning:
            self.ui._on_density_finished(1, bad)
        warning.assert_called_once()
        self.assertIs(self.ui.last_density_result, prior)

    def test_geometry_edit_marks_displayed_density_as_out_of_date(self):
        self.ui._on_density_finished(1, density_result())
        self.ui._mark_geometry_dependent_results_stale()
        self.assertIn("Geometry changed", self.ui.lbl_result_details.text())


if __name__ == "__main__":
    unittest.main()
