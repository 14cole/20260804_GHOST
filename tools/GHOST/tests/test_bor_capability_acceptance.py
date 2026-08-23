#!/usr/bin/env python3
"""Physics and resource acceptance gates for missile-class BoR solves."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

from bor_solver import (  # noqa: E402
    BOR_LINEAR_BACKWARD_ERROR_MAX,
    BorPecSolver,
    solve_bor,
)


FREQUENCY_HZ = 1.0e9
ASPECTS_DEG = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]


def _slender_spheroid(elements):
    """Closed, smooth 7.5:1 missile-class generatrix (+z to -z)."""

    theta = np.linspace(0.0, math.pi, int(elements) + 1)
    return np.column_stack((
        0.04 * np.sin(theta),
        0.30 * np.cos(theta),
    ))


def _complex_channels(result):
    return np.concatenate((
        np.asarray(result["amp_vv"], dtype=complex),
        np.asarray(result["amp_hh"], dtype=complex),
    ))


class MissileClassPhysicsAcceptanceTests(unittest.TestCase):
    """Use invariants that are exact for Maxwell's equations, not snapshots."""

    @classmethod
    def setUpClass(cls):
        common = dict(
            thetas_deg=ASPECTS_DEG,
            formulation="cfie",
            n_modes=10,
            workers=1,
            assembly="tables",
            table_precision="double",
        )
        cls.coarse = solve_bor(
            _slender_spheroid(24), FREQUENCY_HZ, **common
        )
        fine_points = _slender_spheroid(32)
        cls.fine = solve_bor(fine_points, FREQUENCY_HZ, **common)
        cls.scale = 1.7
        cls.scaled = solve_bor(
            cls.scale * fine_points, FREQUENCY_HZ / cls.scale, **common
        )

    def test_complex_output_contract_is_feature_ready(self):
        result = self.fine
        self.assertEqual(result["theta_deg"], ASPECTS_DEG)
        self.assertTrue(result["mode_converged"])
        self.assertLessEqual(
            result["linear_backward_error"],
            BOR_LINEAR_BACKWARD_ERROR_MAX,
        )
        for sigma_key, amplitude_key in (
            ("sigma_vv", "amp_vv"),
            ("sigma_hh", "amp_hh"),
        ):
            sigma = np.asarray(result[sigma_key], dtype=float)
            amplitude = np.asarray(result[amplitude_key], dtype=complex)
            self.assertTrue(np.all(np.isfinite(sigma)))
            self.assertTrue(np.all(np.isfinite(amplitude)))
            np.testing.assert_allclose(
                sigma,
                4.0 * math.pi * np.abs(amplitude) ** 2,
                rtol=2.0e-14,
                atol=0.0,
            )

    def test_maxwell_scale_invariance_including_complex_phase(self):
        base = _complex_channels(self.fine)
        scaled = _complex_channels(self.scaled)
        np.testing.assert_allclose(
            scaled, self.scale * base, rtol=3.0e-12, atol=3.0e-13
        )
        for key in ("sigma_vv", "sigma_hh"):
            np.testing.assert_allclose(
                np.asarray(self.scaled[key]),
                self.scale ** 2 * np.asarray(self.fine[key]),
                rtol=6.0e-12,
                atol=1.0e-14,
            )

    def test_slender_body_mesh_convergence(self):
        coarse = _complex_channels(self.coarse)
        fine = _complex_channels(self.fine)
        peak_relative = (
            np.max(np.abs(fine - coarse)) / np.max(np.abs(fine))
        )
        l2_relative = np.linalg.norm(fine - coarse) / np.linalg.norm(fine)
        self.assertLess(peak_relative, 0.01)
        self.assertLess(l2_relative, 0.01)

    def test_fore_aft_and_axial_symmetry_are_coherent(self):
        for key in ("amp_vv", "amp_hh"):
            amplitude = np.asarray(self.fine[key], dtype=complex)
            np.testing.assert_allclose(
                amplitude, amplitude[::-1], rtol=2.0e-11, atol=2.0e-12
            )
        np.testing.assert_allclose(
            np.asarray(self.fine["amp_vv"])[[0, -1]],
            np.asarray(self.fine["amp_hh"])[[0, -1]],
            rtol=2.0e-11,
            atol=2.0e-12,
        )


class StreamingResourceAcceptanceTests(unittest.TestCase):
    def test_streaming_does_not_materialize_dense_point_basis_matrices(self):
        solver = BorPecSolver(_slender_spheroid(12), FREQUENCY_HZ)
        self.assertIsNone(solver._B_T)
        self.assertIsNone(solver._B_D)
        solver.enable_streaming(
            2,
            efie=True,
            mfie=False,
            tile_budget_gb=0.25,
            workers=1,
        )
        solver.prepare_operators(2, efie=True, workers=1)
        matrix = solver.assemble_mode(0, 2)
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertIsNone(solver._B_T)
        self.assertIsNone(solver._B_D)

    def test_table_path_materializes_dense_basis_only_when_needed(self):
        solver = BorPecSolver(_slender_spheroid(8), FREQUENCY_HZ)
        self.assertIsNone(solver._B_T)
        solver.prepare_operators(1, efie=True, workers=1)
        solver.assemble_mode(0, 1)
        self.assertIsNotNone(solver._B_T)
        self.assertIsNotNone(solver._B_D)


if __name__ == "__main__":
    unittest.main(verbosity=2)
