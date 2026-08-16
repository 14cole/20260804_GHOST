#!/usr/bin/env python3
"""BoR physics, workflow, and optimized-assembly release regressions."""

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import bor_dispatch  # noqa: E402
import run_local_bor  # noqa: E402
from bor_kernels import C0  # noqa: E402
from bor_solver import (  # noqa: E402
    solve_bor,
    solve_bor_coated_pec,
    solve_bor_dielectric,
)
from mie_sphere import (  # noqa: E402
    sigma_coated_pec_sphere,
    sigma_dielectric_sphere,
    sigma_impedance_sphere,
    sigma_pec_sphere,
)


FREQUENCY_HZ = 1.0e9


def _sphere(radius_m, elements):
    theta = np.linspace(0.0, math.pi, int(elements) + 1)
    return np.column_stack((
        float(radius_m) * np.sin(theta),
        float(radius_m) * np.cos(theta),
    ))


def _error_db(numerical, reference):
    return abs(10.0 * math.log10(float(numerical) / float(reference)))


def _pec_sphere_snapshot(radius_m=0.04, explicit_elements=20):
    points = _sphere(radius_m, 12)
    pairs = [
        {
            "x1": float(points[index, 0]),
            "y1": float(points[index, 1]),
            "x2": float(points[index + 1, 0]),
            "y2": float(points[index + 1, 1]),
        }
        for index in range(len(points) - 1)
    ]
    return {
        "title": "PEC sphere",
        "segments": [{
            "name": "sphere",
            "seg_type": 2,
            "properties": ["2", str(explicit_elements), "0", "0", "0"],
            "point_pairs": pairs,
        }],
        "ibcs": [],
        "dielectrics": [],
    }


class AnalyticSphereRegressionTests(unittest.TestCase):
    def _assert_spherical_channels(self, result, reference, tolerance_db):
        vv = np.asarray(result["sigma_vv"], dtype=float)
        hh = np.asarray(result["sigma_hh"], dtype=float)
        self.assertTrue(np.all(np.isfinite(vv)))
        self.assertTrue(np.all(np.isfinite(hh)))
        self.assertLess(_error_db(vv[0], reference), tolerance_db)
        self.assertLess(_error_db(hh[0], reference), tolerance_db)
        self.assertLess(_error_db(vv[0], hh[0]), 0.08)

    def test_pec_sphere_matches_mie(self):
        radius = 3.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        result = solve_bor(
            _sphere(radius, 45), FREQUENCY_HZ, [0.0],
            formulation="cfie", workers=2,
            assembly="streaming", stream_budget_gb=0.25,
        )
        self._assert_spherical_channels(
            result, sigma_pec_sphere(radius, FREQUENCY_HZ), 0.08
        )

    def test_passive_impedance_sphere_matches_mie(self):
        radius = 1.5 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        impedance = 50.0 + 10.0j
        result = solve_bor(
            _sphere(radius, 45), FREQUENCY_HZ, [0.0],
            formulation="efie", zs=impedance, workers=2,
        )
        self._assert_spherical_channels(
            result,
            sigma_impedance_sphere(radius, FREQUENCY_HZ, impedance),
            0.10,
        )

    def test_lossy_dielectric_sphere_matches_mie(self):
        radius = 1.5 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        eps_r = 3.0 - 0.1j
        mu_r = 1.0 - 0.02j
        result = solve_bor_dielectric(
            _sphere(radius, 45), FREQUENCY_HZ, [0.0], eps_r, mu_r,
            workers=2,
        )
        self._assert_spherical_channels(
            result,
            sigma_dielectric_sphere(
                radius, eps_r, mu_r, FREQUENCY_HZ
            ),
            0.15,
        )

    def test_lossy_coated_pec_sphere_matches_mie(self):
        outer_radius = 0.1
        core_radius = 0.07
        eps_r = 3.0 - 0.1j
        mu_r = 1.0 - 0.02j
        result = solve_bor_coated_pec(
            _sphere(outer_radius, 50),
            _sphere(core_radius, 40),
            FREQUENCY_HZ,
            [0.0],
            eps_r,
            mu_r,
            workers=2,
        )
        self._assert_spherical_channels(
            result,
            sigma_coated_pec_sphere(
                core_radius, outer_radius, eps_r, mu_r, FREQUENCY_HZ
            ),
            0.15,
        )


class BoRWorkflowRegressionTests(unittest.TestCase):
    def test_dispatch_builds_each_frequency_on_its_own_mesh(self):
        snapshot = _pec_sphere_snapshot(explicit_elements=-20)
        point_counts = []

        def fake_solve(points, freq_hz, thetas_deg, **_kwargs):
            point_counts.append((float(freq_hz), len(points)))
            amplitude = 1.0 / math.sqrt(4.0 * math.pi)
            count = len(thetas_deg)
            return {
                "sigma_vv": [1.0] * count,
                "sigma_hh": [1.0] * count,
                "amp_vv": [amplitude + 0.0j] * count,
                "amp_hh": [amplitude + 0.0j] * count,
                "modes_used": 3,
                "mode_cap": 8,
                "mode_converged": True,
                "mode_quiet_count": 2,
                "mode_last_relative_increment": 0.0,
                "n_unknowns": 2 * len(points),
                "linear_residual": 1.0e-12,
                "max_cond": 10.0,
                "condition_est_computed": True,
                "condition_est_method": "test",
            }

        with mock.patch.object(bor_dispatch, "solve_bor", fake_solve):
            result = bor_dispatch.solve_monostatic_rcs_bor(
                snapshot,
                [1.0, 4.0],
                [0.0, 90.0],
                "VV",
                geometry_units="meters",
                workers=1,
            )
        self.assertEqual([entry[0] for entry in point_counts], [1.0e9, 4.0e9])
        self.assertGreater(point_counts[1][1], point_counts[0][1])
        per_frequency = result["metadata"]["per_frequency"]
        self.assertLess(
            per_frequency[1]["mesh_wavelength_m"],
            per_frequency[0]["mesh_wavelength_m"],
        )

    def test_streaming_and_table_paths_are_equivalent(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        points = _sphere(radius, 18)
        common = dict(
            points=points,
            freq_hz=FREQUENCY_HZ,
            thetas_deg=[0.0, 37.0, 90.0],
            formulation="cfie",
            n_modes=8,
            workers=1,
            table_precision="double",
        )
        table = solve_bor(assembly="tables", **common)
        stream = solve_bor(
            assembly="streaming", stream_budget_gb=0.25, **common
        )
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
            )

    def test_survey_entry_point_is_unambiguously_marked(self):
        raw = {
            "samples": [{"rcs_linear": 1.0}],
            "metadata": {"quality_gate": {}},
        }
        with mock.patch.object(
            bor_dispatch, "solve_monostatic_rcs_bor", return_value=raw
        ):
            result = bor_dispatch.solve_monostatic_rcs_bor_survey(
                {}, [1.0], [0.0], "VV"
            )
        metadata = result["metadata"]
        self.assertTrue(metadata["survey_mode"])
        self.assertFalse(metadata["mesh_convergence_certified"])
        self.assertEqual(metadata["published_mesh"], "base")

    def test_resource_preview_uses_frequency_specific_meshes(self):
        snapshot = _pec_sphere_snapshot(explicit_elements=-20)
        low = bor_dispatch.estimate_bor_resources(
            snapshot, 1.0, [0.0, 90.0], geometry_units="meters",
            workers=2, mesh_certification=False,
        )
        high = bor_dispatch.estimate_bor_resources(
            snapshot, 4.0, [0.0, 90.0], geometry_units="meters",
            workers=2, mesh_certification=False,
        )
        self.assertGreater(high["mesh_elements"], low["mesh_elements"])
        self.assertGreater(
            high["estimated_peak_gb"], low["estimated_peak_gb"]
        )

    def test_vv_hh_outputs_are_paired_without_manifest_change(self):
        units = [
            {
                "geometry": "/tmp/body.geo",
                "geometry_stem": "body",
                "geometry_input_sha256": "abc",
                "polarization": polarization,
                "frequency_ghz": frequency,
            }
            for polarization in ("VV", "HH")
            for frequency in (1.0, 2.0)
        ]
        pairs = run_local_bor._paired_solve_units(units)
        self.assertEqual(len(units), 4)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(len(pair["channel_units"]) == 2 for pair in pairs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
