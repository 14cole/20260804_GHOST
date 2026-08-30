"""Focused regressions for the experimental sparse ISAR reconstruction."""

import unittest

import numpy as np

from grim_dataset import RcsGrid
from plot_modes import isar_mode


class TestSparseIsarSolver(unittest.TestCase):
    @staticmethod
    def _unit_point_problem(sample_count=32):
        theta = np.deg2rad(np.linspace(-2.0, 2.0, sample_count))
        frequency = np.linspace(8.0e9, 9.0e9, sample_count)
        samples = np.ones((sample_count, sample_count), dtype=np.complex64)
        return theta, frequency, samples

    def test_default_budget_debiases_trivial_on_grid_point(self):
        theta, frequency, samples = self._unit_point_problem()
        result = isar_mode._compute_band_sparse_l1(
            samples,
            theta,
            frequency,
            float(np.mean(np.diff(frequency))),
            1.0,
            0.05,
            300,
        )
        self.assertNotIsInstance(result, str)
        image, _x, _y, diagnostics = result

        self.assertAlmostEqual(float(np.abs(image).max()), 1.0, places=3)
        self.assertTrue(diagnostics["sparse_debias_applied"])
        self.assertLess(
            diagnostics["sparse_output_residual_norm"],
            diagnostics["sparse_lasso_residual_norm"],
        )
        self.assertIn("sparse_converged", diagnostics)
        self.assertIn("sparse_relative_duality_gap", diagnostics)
        self.assertIn("sparse_objective", diagnostics)
        self.assertIn("sparse_status", diagnostics)

    def test_primal_dual_gap_certifies_convergence(self):
        theta, frequency, samples = self._unit_point_problem()
        result = isar_mode._compute_band_sparse_l1(
            samples,
            theta,
            frequency,
            float(np.mean(np.diff(frequency))),
            1.0,
            0.05,
            900,
        )
        self.assertNotIsInstance(result, str)
        image, _x, _y, diagnostics = result

        self.assertTrue(diagnostics["sparse_converged"])
        self.assertLessEqual(diagnostics["sparse_relative_duality_gap"], 1.0e-4)
        self.assertLess(diagnostics["sparse_iterations"], 900)
        self.assertEqual(diagnostics["sparse_support_size"], 1)
        self.assertAlmostEqual(float(np.abs(image).max()), 1.0, places=5)


class TestSparseIsarPublicValidation(unittest.TestCase):
    def setUp(self):
        azimuth = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
        frequency = np.asarray([8.0, 8.25, 8.5, 8.75, 9.0])
        self.grid = RcsGrid(
            azimuth,
            [0.0],
            frequency,
            ["VV"],
            rcs=np.ones((5, 1, 5, 1), dtype=np.complex64),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "angular_coordinate_system": "conic",
                "time_convention": "exp(+jwt)",
            },
            extra={
                "phase_reference": "fixed origin",
                "measurement_geometry": "far-field monostatic",
                "motion_compensation": "stable",
                "range_phase_convention": "S~exp(-j*2*k*R)",
            },
        )

    def test_unknown_window_is_not_silently_hanning(self):
        with self.assertRaisesRegex(ValueError, "unsupported ISAR window"):
            isar_mode.form_isar(self.grid, window="Hannig")
        self.assertEqual(
            isar_mode._window_name("Kaiser beta=15"), "Kaiser β=15"
        )

    def test_unknown_length_unit_is_not_silently_meters(self):
        for invalid_unit in ("yards", "", None):
            with self.subTest(length_unit=invalid_unit):
                with self.assertRaisesRegex(ValueError, "unsupported ISAR length unit"):
                    isar_mode.form_isar(self.grid, length_unit=invalid_unit)

    def test_invalid_sparse_controls_are_rejected_without_coercion(self):
        invalid_cases = (
            ({"l1_strength": 0.0}, "l1_strength"),
            ({"l1_strength": 1.0}, "l1_strength"),
            ({"l1_strength": float("nan")}, "l1_strength"),
            ({"l1_iterations": True}, "l1_iterations"),
            ({"l1_iterations": 9}, "l1_iterations"),
            ({"l1_iterations": 10.5}, "l1_iterations"),
            ({"l1_iterations": 10_001}, "l1_iterations"),
        )
        for options, message in invalid_cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    isar_mode.form_isar(self.grid, **options)

    def test_band_result_carries_exportable_sparse_diagnostics(self):
        bands, _elapsed = isar_mode.form_isar(
            self.grid,
            reconstruction="sparse",
            window="Rectangular",
            l1_iterations=10,
            retain_complex=True,
        )
        self.assertEqual(len(bands), 1)
        diagnostics = bands[0]["sparse_diagnostics"]
        self.assertEqual(
            diagnostics["sparse_converged"], bands[0]["sparse_converged"]
        )
        self.assertEqual(
            diagnostics["sparse_iterations"], bands[0]["sparse_iterations"]
        )
        self.assertEqual(len(bands[0]["source_selection_digest"]), 40)
        self.assertEqual(
            bands[0]["complex_image"].shape, bands[0]["magnitude"].shape
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
