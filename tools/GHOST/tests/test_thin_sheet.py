"""Thin-layer phase, transparency, reciprocity, and explicit-bulk references."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Backend"))
import rcs_solver as rcs
from thin_sheet import solve_thin_layer_fields, validate_thin_layer
from mie_reference import two_layer_dielectric_cylinder_amplitude
from geometry_io import build_geometry_text, parse_geometry
from unittest import mock


def sheet_snapshot(points, panels=1):
    return {"segments": [{"name": "sheet", "seg_type": 1,
        "properties": ["1", str(panels), "1", "0", "0"],
        "point_pairs": [dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1])
                        for a, b in zip(points[:-1], points[1:])]}],
        "ibcs": [["1", "constant", "75", "0", "0", "0"]], "dielectrics": []}


def sheet_mesh(points, panels=1):
    snapshot = sheet_snapshot(points, panels)
    materials = rcs.MaterialLibrary.from_entries(snapshot["ibcs"], [], ".")
    panels = rcs._build_panels(snapshot, 1., rcs.C0/1e9)
    infos = rcs._build_coupled_panel_info(panels, materials, 1., "TM", 2*np.pi*1e9/rcs.C0)
    return rcs._build_linear_mesh_interface_aware(panels, infos)[0]


class ThinSheetPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radius, cls.frequency = .08, 1e9
        cls.k = 2*np.pi*cls.frequency/rcs.C0
        angle = np.linspace(0, 2*np.pi, 81)
        cls.mesh = sheet_mesh(np.column_stack((cls.radius*np.cos(angle), cls.radius*np.sin(angle))))

    def test_complex_shell_field_matches_independent_bessel_boundary_match(self):
        for eps, mu in ((3-.02j, 1), (2.1-.05j, 1.3-.02j)):
            for pol in ("TM", "TE"):
                with self.subTest(epsilon=eps, mu=mu, polarization=pol):
                    d = .001
                    width, amplitude, residual, evidence = solve_thin_layer_fields(
                        self.mesh, self.k, [0., 37., 90.], pol, eps, mu, d)
                    truth = two_layer_dielectric_cylinder_amplitude(
                        self.radius-d/2, self.radius+d/2, eps, mu, 1, 1,
                        self.frequency, pol)
                    np.testing.assert_allclose(amplitude, truth, rtol=.005, atol=1e-6)
                    np.testing.assert_allclose(width, abs(amplitude)**2/(4*self.k), rtol=1e-14)
                    self.assertLess(residual, 1e-11)
                    self.assertTrue(evidence["normal_polarization_terms"])
                    self.assertFalse(evidence["approximation_error_certified"])

    def test_air_layer_is_transparent_including_phase(self):
        for pol in ("TM", "TE"):
            _, field, _, _ = solve_thin_layer_fields(self.mesh, self.k, [0, 90], pol, 1, 1, .001)
            np.testing.assert_array_equal(field, [0j, 0j])

    def test_open_strip_orientation_and_bistatic_reciprocity(self):
        points = np.column_stack((np.linspace(-.05, .05, 25), np.zeros(25)))
        forward, reverse = sheet_mesh(points), sheet_mesh(points[::-1])
        for pol in ("TM", "TE"):
            args = (self.k, [12., 48., 86.], pol, 3-.05j, 1, .0005)
            _, field, _, _ = solve_thin_layer_fields(forward, *args, observation_angles_deg=[12., 48., 86.])
            _, reversed_field, _, _ = solve_thin_layer_fields(reverse, *args, observation_angles_deg=[12., 48., 86.])
            np.testing.assert_allclose(field, reversed_field, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(field, field.T, rtol=.005, atol=2e-6)

    def test_thick_layer_and_active_medium_are_rejected(self):
        for args in ((3, 1, .1, self.k), (3+.1j, 1, .001, self.k),
                     (3, 1, float("nan"), self.k)):
            with self.assertRaises(ValueError):
                validate_thin_layer(*args)

    def test_public_monostatic_and_bistatic_routes_preserve_complex_fields(self):
        snapshot = sheet_snapshot([[-.05, 0.], [.05, 0.]], 24)
        snapshot["ibcs"] = [["1", "thin_dielectric", ".0005", "2"]]
        snapshot["dielectrics"] = [["2", "3", "-.05", "1", "0"]]
        mono = rcs.solve_monostatic_rcs_2d(snapshot, [1.], [12., 48.], geometry_units="meters", compute_condition_number=True)
        bi = rcs.solve_bistatic_rcs_2d(snapshot, [1.], [12., 48.], [12., 48.], geometry_units="meters", compute_condition_number=True)
        for pol in ("VV", "HH"):
            diagonal = [row for row in bi["co_solved_samples"][pol] if row["theta_inc_deg"] == row["theta_scat_deg"]]
            for a, b in zip(mono["co_solved_samples"][pol], diagonal):
                self.assertAlmostEqual(a["rcs_amp_real"], b["rcs_amp_real"], places=10)
                self.assertAlmostEqual(a["rcs_amp_imag"], b["rcs_amp_imag"], places=10)
            self.assertTrue(mono["metadata"]["thin_layer"][pol])
            self.assertTrue(bi["metadata"]["thin_layer"][pol])
        self.assertIn("runtime_profile", mono["metadata"])

    def test_file_and_editor_roundtrip_do_not_coerce_layer_to_impedance(self):
        ibcs, dielectrics = [["1", "thin_dielectric", ".001", "2"]], [["2", "3", "0", "1", "0"]]
        text = build_geometry_text("film", [], ibcs, dielectrics)
        self.assertEqual(parse_geometry(text)[2:], (ibcs, dielectrics))
        from PySide6.QtWidgets import QApplication
        from geometry_tab import GeometryTab
        app = QApplication.instance() or QApplication([])
        tab = GeometryTab()
        try:
            tab._populate_small_table(tab.table_ibc, ibcs, tab.lbl_ibc, "IBCS/Resistances")
            self.assertEqual(tab._read_small_table(tab.table_ibc), ibcs)
            self.assertEqual(tab._ibcs_lookup()[1]["kind"], "thin_dielectric")
        finally:
            tab.close()
            tab.deleteLater()
            app.processEvents()

    def test_memory_gate_precedes_operator_allocation(self):
        with mock.patch.object(rcs, "_solve_memory_limit_gb", return_value=1e-12), \
             mock.patch.object(rcs, "_assemble_linear_mass_matrix") as assembly:
            with self.assertRaises(MemoryError):
                solve_thin_layer_fields(self.mesh, self.k, [0], "TM", 3, 1, .001)
            assembly.assert_not_called()


if __name__ == "__main__":
    unittest.main()
