#!/usr/bin/env python3
"""Focused physical/numerical regressions for the 2-D RCS solver.

Run directly; pytest is not required:

    python3 tests/test_rcs_physics_regression.py
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import rcs_solver as rcs  # noqa: E402
from mie_reference import (  # noqa: E402
    sigma_dielectric_cylinder,
    sigma_pec_cylinder,
)


def _linear_element(y, index):
    return rcs.LinearElement(
        name="panel",
        seg_type=2,
        ibc_flag=0,
        pos_mat=0,
        neg_mat=0,
        node_ids=(2 * index, 2 * index + 1),
        p0=np.array([0.0, float(y)]),
        p1=np.array([1.0, float(y)]),
        center=np.array([0.5, float(y)]),
        tangent=np.array([1.0, 0.0]),
        normal=np.array([0.0, 1.0]),
        length=1.0,
        panel_index=index,
    )


def _circle_segment(radius, count, seg_type, ibc=0, material=0):
    theta = np.linspace(0.0, -2.0 * np.pi, count + 1)
    pairs = []
    for idx in range(count):
        pairs.append({
            "x1": float(radius * np.cos(theta[idx])),
            "y1": float(radius * np.sin(theta[idx])),
            "x2": float(radius * np.cos(theta[idx + 1])),
            "y2": float(radius * np.sin(theta[idx + 1])),
        })
    return {
        "name": "circle",
        "seg_type": seg_type,
        "properties": [
            str(seg_type), "1", str(ibc), str(material), "0",
        ],
        "point_pairs": pairs,
    }


class NearPairQuadratureTests(unittest.TestCase):
    def test_close_parallel_pair_converges(self):
        obs = _linear_element(0.0, 0)
        k0 = 2.0 * np.pi
        for gap in (0.03, 0.01, 0.003, 0.001):
            src = _linear_element(gap, 1)
            actual, _ = rcs._integrate_linear_pair_adaptive_sk(
                obs, src, k0, False, 16, 16,
                compute_single_layer=True,
                compute_double_layer=False,
                rtol=1.0e-9,
                max_depth=12,
            )
            reference, _ = rcs._integrate_linear_pair_adaptive_sk(
                obs, src, k0, False, 16, 16,
                compute_single_layer=True,
                compute_double_layer=False,
                rtol=1.0e-12,
                max_depth=16,
            )
            error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
            self.assertLess(error, 2.0e-8, msg=f"gap/length={gap:g}")

    def test_fixed_order_would_not_be_adequate(self):
        obs = _linear_element(0.0, 0)
        src = _linear_element(0.01, 1)
        fixed, _ = rcs._integrate_linear_pair_box_sk_vectorized(
            obs, src, 2.0 * np.pi, False,
            (0.0, 1.0), (0.0, 1.0), 16, 16, True, False,
        )
        reference, _ = rcs._integrate_linear_pair_adaptive_sk(
            obs, src, 2.0 * np.pi, False, 16, 16,
            compute_single_layer=True,
            compute_double_layer=False,
            rtol=1.0e-12,
            max_depth=16,
        )
        error = np.linalg.norm(fixed - reference) / np.linalg.norm(reference)
        self.assertGreater(error, 0.05)


class WeightedGalerkinTests(unittest.TestCase):
    def _two_element_mesh(self):
        panels = []
        for idx, (x0, x1) in enumerate(((0.0, 0.5), (0.5, 1.0))):
            p0 = np.array([x0, 0.0])
            p1 = np.array([x1, 0.0])
            panels.append(rcs.Panel(
                "line", 1, 1, 0, 0, p0, p1,
                0.5 * (p0 + p1), np.array([1.0, 0.0]),
                np.array([0.0, 1.0]), x1 - x0,
            ))
        return rcs._build_linear_mesh(panels)

    def test_weighted_mass_constant_limit(self):
        mesh = self._two_element_mesh()
        coeff = 2.3 - 0.7j
        plain = rcs._assemble_linear_mass_matrix(mesh)
        weighted = rcs._assemble_linear_weighted_mass_matrix(
            mesh, np.full(len(mesh.elements), coeff)
        )
        np.testing.assert_allclose(weighted, coeff * plain, rtol=0.0, atol=1e-15)

    def test_weighted_mass_keeps_element_coefficients_inside_integral(self):
        mesh = self._two_element_mesh()
        coeff = np.array([1.0 + 0.0j, 4.0 + 0.0j])
        actual = rcs._assemble_linear_weighted_mass_matrix(mesh, coeff)
        expected = np.zeros_like(actual)
        for eidx, elem in enumerate(mesh.elements):
            ids = np.asarray(elem.node_ids, dtype=int)
            expected[np.ix_(ids, ids)] += coeff[eidx] * rcs._linear_mass_block(elem)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_weighted_single_layer_constant_limit(self):
        mesh = self._two_element_mesh()
        coeff = 1.7 + 0.2j
        plain, k_plain = rcs._assemble_linear_operator_matrices(
            mesh, 3.0, obs_normal_deriv=True
        )
        weighted, k_weighted = rcs._assemble_linear_operator_matrices(
            mesh, 3.0, obs_normal_deriv=True,
            single_layer_observation_coefficients=np.full(
                len(mesh.elements), coeff
            ),
        )
        np.testing.assert_allclose(weighted, coeff * plain, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(k_weighted, k_plain, rtol=0.0, atol=0.0)


class FarFieldProjectionTests(unittest.TestCase):
    def _mesh(self):
        radius = 0.37
        count = 31
        snapshot = {
            "segments": [_circle_segment(radius, count, 2)],
            "ibcs": [],
            "dielectrics": [],
        }
        panels = rcs._build_panels(snapshot, 1.0, 1.0, max_panels=1000)
        return rcs._build_linear_mesh(panels)

    @staticmethod
    def _scalar_reference(mesh, density, k_air, angles, potential, order=8):
        obs = np.asarray(angles, dtype=float)
        dirs = np.column_stack((
            np.cos(np.deg2rad(obs)), np.sin(np.deg2rad(obs))
        ))
        qt, qw = rcs._get_quadrature(order)
        amp = np.zeros(obs.size, dtype=np.complex128)
        rho = np.asarray(density, dtype=np.complex128)
        if rho.ndim == 1:
            rho = rho[:, None]
        for elem in mesh.elements:
            ids = np.asarray(elem.node_ids, dtype=int)
            local = rho[ids, :]
            for t, w in zip(qt, qw):
                shape = rcs._linear_shape_values(float(t))[:, None]
                point = elem.p0 + float(t) * (elem.p1 - elem.p0)
                phase = np.exp(1j * k_air * (dirs @ point))
                rho_t = np.sum(shape * local, axis=0)
                if rho.shape[1] == 1:
                    rho_t = np.full(obs.size, rho_t[0], dtype=np.complex128)
                if potential == "DLP":
                    phase *= 1j * k_air * (dirs @ elem.normal)
                amp += float(w) * float(elem.length) * phase * rho_t
        return amp

    def test_vectorized_far_field_matches_scalar_single_and_many(self):
        mesh = self._mesh()
        rng = np.random.default_rng(7254)
        angles = np.linspace(-83.0, 76.0, 19)
        one = rng.normal(size=len(mesh.nodes)) + 1j * rng.normal(size=len(mesh.nodes))
        many = (
            rng.normal(size=(len(mesh.nodes), angles.size))
            + 1j * rng.normal(size=(len(mesh.nodes), angles.size))
        )
        for potential in ("SLP", "DLP"):
            for density in (one, many):
                actual = rcs._farfield_linear_density_many(
                    mesh, density, 4.2, angles, potential, order=8
                )
                expected = self._scalar_reference(
                    mesh, density, 4.2, angles, potential, order=8
                )
                np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


class CylinderPhysicsTests(unittest.TestCase):
    def _solve(self, snapshot, pol, freq_hz):
        result = rcs.solve_monostatic_rcs_2d(
            snapshot,
            frequencies_ghz=[freq_hz / 1.0e9],
            elevations_deg=[0.0],
            polarization=pol,
            geometry_units="meters",
            strict_quality_gate=False,
            max_panels=1000,
        )
        return float(result["samples"][0]["rcs_linear"])

    def test_pec_and_dielectric_cylinders_both_polarizations(self):
        radius = 0.1
        freq_hz = rcs.C0 / (2.0 * math.pi * radius)  # ka = 1
        pec = {
            "segments": [_circle_segment(radius, 96, 2)],
            "ibcs": [],
            "dielectrics": [],
        }
        dielectric = {
            "segments": [_circle_segment(radius, 96, 3, material=1)],
            "ibcs": [],
            "dielectrics": [["1", "4", "-0.2", "1", "0"]],
        }
        for pol in ("TM", "TE"):
            got_pec = self._solve(pec, pol, freq_hz)
            ref_pec = sigma_pec_cylinder(radius, freq_hz, pol)
            got_diel = self._solve(dielectric, pol, freq_hz)
            ref_diel = sigma_dielectric_cylinder(
                radius, 4.0 - 0.2j, 1.0 + 0.0j, freq_hz, pol
            )
            self.assertLess(abs(10.0 * math.log10(got_pec / ref_pec)), 0.01)
            self.assertLess(abs(10.0 * math.log10(got_diel / ref_diel)), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
