#!/usr/bin/env python3
"""Focused physical/numerical regressions for the 2-D RCS solver.

Run directly; pytest is not required:

    python3 tests/test_rcs_physics_regression.py
"""

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import rcs_solver as rcs  # noqa: E402
from mie_reference import (  # noqa: E402
    pec_cylinder_backscatter_amplitude,
    sigma_coated_pec_cylinder,
    sigma_dielectric_cylinder,
    sigma_impedance_cylinder,
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
    def test_batched_fixed_pairs_match_individual_blocks(self):
        elements = [_linear_element(0.0, 0)] + [
            _linear_element(gap, index)
            for index, gap in enumerate((0.8, 1.1, 1.6), start=1)
        ]
        obs_indices = np.zeros(3, dtype=np.int64)
        src_indices = np.arange(1, 4, dtype=np.int64)
        for obs_normal_deriv in (False, True):
            actual_s, actual_k = rcs._integrate_linear_pairs_box_sk_batched(
                elements,
                obs_indices,
                src_indices,
                3.7 - 0.08j,
                obs_normal_deriv,
                12,
            )
            for position, src_index in enumerate(src_indices):
                expected_s, expected_k = (
                    rcs._integrate_linear_pair_box_sk_vectorized(
                        elements[0],
                        elements[int(src_index)],
                        3.7 - 0.08j,
                        obs_normal_deriv,
                        (0.0, 1.0),
                        (0.0, 1.0),
                        12,
                        12,
                    )
                )
                np.testing.assert_allclose(
                    actual_s[position], expected_s, rtol=2e-14, atol=2e-14
                )
                np.testing.assert_allclose(
                    actual_k[position], expected_k, rtol=2e-14, atol=2e-14
                )

    def test_geometrically_touching_split_nodes_use_duffy_rule(self):
        """Interface-split DOFs must not hide a physical endpoint singularity."""

        angle = 0.4
        obs = _linear_element(0.0, 0)
        obs.p0 = np.array([0.0, 0.0])
        obs.p1 = np.array([1.0, 0.0])
        obs.center = 0.5 * (obs.p0 + obs.p1)
        obs.tangent = np.array([1.0, 0.0])
        obs.normal = np.array([0.0, 1.0])
        obs.length = 1.0
        src = _linear_element(0.0, 1)
        src.p0 = np.array([0.0, 0.0])
        src.p1 = np.array([math.cos(angle), math.sin(angle)])
        src.center = 0.5 * (src.p0 + src.p1)
        src.tangent = src.p1.copy()
        src.normal = np.array([-math.sin(angle), math.cos(angle)])
        src.length = 1.0
        # _linear_element assigned disjoint node IDs, as interface-aware
        # meshing does for different physical boundary signatures.
        self.assertFalse(set(obs.node_ids) & set(src.node_ids))

        actual_s, actual_k = rcs._sk_blocks_near_linear(
            obs, src, 2.0 * np.pi, False, 16, 16
        )
        reference_s, reference_k = (
            rcs._integrate_linear_touching_duffy_sk_vectorized(
                obs, src, 2.0 * np.pi, False,
                (0.0, 1.0), (0.0, 1.0), True, True, 17, True, True,
            )
        )
        np.testing.assert_allclose(actual_s, reference_s, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual_k, reference_k, rtol=0.0, atol=0.0)

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

    def test_adaptive_path_supports_legacy_positional_only_sum(self):
        """Exercise the close-pair path with an HPC-style legacy built-in."""

        obs = _linear_element(0.0, 0)
        src = _linear_element(0.01, 1)
        builtin_sum = sum

        def positional_only_sum(iterable, *args, **kwargs):
            if kwargs:
                raise TypeError("sum() takes no keyword arguments")
            return builtin_sum(iterable, *args)

        with mock.patch("builtins.sum", side_effect=positional_only_sum):
            s_block, k_block = rcs._integrate_linear_pair_adaptive_sk(
                obs, src, 2.0 * np.pi, False, 16, 16,
                compute_single_layer=True,
                compute_double_layer=True,
            )
        self.assertTrue(np.all(np.isfinite(s_block)))
        self.assertTrue(np.all(np.isfinite(k_block)))




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


    def test_multi_output_weighted_assembly_matches_individual_calls(self):
        mesh = self._two_element_mesh()
        masks = [
            np.ones(len(mesh.elements), dtype=bool),
            np.asarray([True, False], dtype=bool),
        ]
        coeff = np.asarray([1.2 - 0.1j, 0.4 + 0.3j])
        outputs = rcs._assemble_linear_operator_matrices_multi(
            mesh,
            3.0,
            True,
            masks,
            compute_double_layer_many=[True, False],
            single_layer_observation_coefficients_many=[None, coeff],
        )
        plain = rcs._assemble_linear_operator_matrices(
            mesh, 3.0, True, source_element_mask=masks[0]
        )
        weighted = rcs._assemble_linear_operator_matrices(
            mesh,
            3.0,
            True,
            source_element_mask=masks[1],
            compute_double_layer=False,
            single_layer_observation_coefficients=coeff,
        )
        np.testing.assert_allclose(outputs[0][0], plain[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(outputs[0][1], plain[1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(outputs[1][0], weighted[0], rtol=0.0, atol=0.0)
        self.assertFalse(outputs[1][1].flags.writeable)

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


class NumericalCertificationTests(unittest.TestCase):
    def test_blocked_condition_scaling_matches_dense_reference(self):
        rng = np.random.default_rng(29)
        matrix = rng.normal(size=(19, 19)) + 1j * rng.normal(size=(19, 19))
        matrix[3, :] *= 1.0e-7
        matrix[:, 11] *= 1.0e5

        magnitude = np.abs(matrix)
        expected_rows = np.max(magnitude, axis=1)
        expected_rows = np.where(expected_rows > 0.0, expected_rows, 1.0)
        row_equilibrated = magnitude / expected_rows[:, None]
        expected_cols = np.max(row_equilibrated, axis=0)
        expected_cols = np.where(expected_cols > 0.0, expected_cols, 1.0)
        expected_norm = float(np.max(np.sum(
            row_equilibrated / expected_cols[None, :], axis=0
        )))

        rows, cols, norm_value = rcs._equilibrated_scaling_and_norm_1(
            matrix,
            # Force seven blocks so this cannot silently pass through the
            # old full-matrix allocation path.
            max_block_bytes=8 * matrix.shape[0] * 3,
        )
        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_array_equal(cols, expected_cols)
        self.assertEqual(norm_value, expected_norm)

    def test_condition_estimate_reuses_lu_without_svd(self):
        rng = np.random.default_rng(17)
        matrix = (
            rng.normal(size=(18, 18))
            + 1j * rng.normal(size=(18, 18))
            + 12.0 * np.eye(18)
        )
        rhs = rng.normal(size=(18, 3)) + 1j * rng.normal(size=(18, 3))
        diagnostics = {}
        with mock.patch("numpy.linalg.cond", side_effect=AssertionError("SVD used")):
            solution = rcs._solve_dense_system(
                matrix, rhs, diagnostics, "test matrix"
            )
        np.testing.assert_allclose(
            matrix @ solution, rhs, rtol=2e-14, atol=2e-14
        )
        self.assertTrue(math.isfinite(diagnostics["condition_est"]))
        self.assertEqual(
            diagnostics["condition_method"],
            "equilibrated_1norm_lu_onenormest",
        )
        self.assertLessEqual(
            diagnostics["linear_backward_error"],
            rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX,
        )
        self.assertGreaterEqual(diagnostics["linear_refinement_steps"], 0)

    def test_opt_in_gpu_dense_solve_keeps_cpu_backward_error_gate(self):
        matrix = np.asarray(
            [[4.0 + 0.2j, 1.0], [1.0, 3.0 - 0.1j]],
            dtype=np.complex128,
        )
        rhs = np.asarray([1.0 + 0.5j, -0.2j], dtype=np.complex128)

        def fake_gpu(a_eval, rhs_eval):
            return np.linalg.solve(a_eval, rhs_eval), "test CUDA device"

        rcs._reset_dense_backend_telemetry()
        with mock.patch.dict(
            rcs.os.environ,
            {rcs.DENSE_GPU_BACKEND_ENV: "gpu"},
        ), mock.patch.object(
            rcs, "_solve_dense_gpu", side_effect=fake_gpu
        ) as gpu_solve:
            solution = rcs._solve_dense_system(
                matrix, rhs, None, "GPU regression"
            )
        gpu_solve.assert_called_once()
        np.testing.assert_allclose(
            matrix @ solution, rhs, rtol=2.0e-14, atol=2.0e-14
        )
        summary = rcs._dense_backend_summary()
        self.assertEqual(summary["linear_backend"], "gpu_cupy")
        self.assertEqual(summary["dense_gpu_solve_count"], 1)
        self.assertEqual(summary["dense_gpu_devices"], ["test CUDA device"])

    def test_condition_estimation_forces_audited_cpu_fallback(self):
        matrix = np.asarray(
            [[3.0, 0.5], [0.5, 2.0]], dtype=np.complex128
        )
        rhs = np.asarray([1.0, -1.0], dtype=np.complex128)
        diagnostics = {}
        rcs._reset_dense_backend_telemetry()
        with mock.patch.dict(
            rcs.os.environ,
            {rcs.DENSE_GPU_BACKEND_ENV: "gpu"},
        ), mock.patch.object(
            rcs, "_solve_dense_gpu",
            side_effect=AssertionError("GPU must not be called"),
        ):
            solution = rcs._solve_dense_system(
                matrix, rhs, diagnostics, "condition regression"
            )
        np.testing.assert_allclose(matrix @ solution, rhs)
        summary = rcs._dense_backend_summary()
        self.assertEqual(summary["linear_backend"], "cpu")
        self.assertEqual(summary["dense_cpu_solve_count"], 1)
        self.assertIn(
            "condition estimation",
            summary["dense_gpu_fallback_reasons"][0],
        )

    def test_condition_gate_is_explicit_and_threshold_override_is_auditable(self):
        metadata = {
            "residual_norm_max": 1.0e-12,
            "residual_nonfinite_count": 0,
            "condition_est_max": 2.0e7,
            "condition_est_computed": True,
            "warnings": [],
        }
        default_gate = rcs.evaluate_quality_gate(metadata)
        self.assertFalse(default_gate["passed"])
        self.assertIn("condition_est_max", default_gate["reason"])

        reviewed_gate = rcs.evaluate_quality_gate(
            metadata, thresholds={"condition_est_max": 5.0e7}
        )
        self.assertTrue(reviewed_gate["passed"])
        self.assertEqual(
            reviewed_gate["thresholds"]["condition_est_max"], 5.0e7
        )
        self.assertEqual(
            reviewed_gate["values"]["condition_est_max"], 2.0e7
        )

        no_condition = dict(metadata)
        no_condition.update({
            "condition_est_max": float("nan"),
            "condition_est_computed": False,
        })
        residual_only_gate = rcs.evaluate_quality_gate(no_condition)
        self.assertTrue(residual_only_gate["passed"])
        self.assertIn(
            "condition number was not requested",
            residual_only_gate["reason"],
        )

    def test_nonzero_cfie_is_rejected_instead_of_ignored(self):
        snapshot = {
            "segments": [_circle_segment(0.05, 24, 2)],
            "ibcs": [],
            "dielectrics": [],
        }
        with self.assertRaisesRegex(ValueError, "cfie_alpha is not implemented"):
            rcs.solve_monostatic_rcs_2d_single_polarization(
                snapshot,
                frequencies_ghz=[1.0],
                elevations_deg=[0.0],
                polarization="TM",
                geometry_units="meters",
                cfie_alpha=0.5,
                strict_quality_gate=False,
                max_panels=1000,
            )

    def test_every_nonzero_or_nonfinite_2d_cfie_value_fails_closed(self):
        """An unimplemented algorithm selector must never use EPS semantics."""

        snapshot = {
            "segments": [_circle_segment(0.05, 24, 2)],
            "ibcs": [],
            "dielectrics": [],
        }
        for value in (1.0e-15, -1.0e-15, float("nan"), float("inf")):
            with self.subTest(value=value), mock.patch.object(
                rcs,
                "validate_geometry_snapshot_for_solver",
                side_effect=AssertionError("geometry work started"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "cfie_alpha is not implemented"
                ):
                    rcs.solve_monostatic_rcs_2d_single_polarization(
                        snapshot,
                        frequencies_ghz=[1.0],
                        elevations_deg=[0.0],
                        polarization="TM",
                        geometry_units="meters",
                        cfie_alpha=value,
                        strict_quality_gate=False,
                        max_panels=1000,
                    )

    def test_bistatic_and_density_cfie_guards_run_before_geometry(self):
        snapshot = {
            "segments": [_circle_segment(0.05, 24, 2)],
            "ibcs": [],
            "dielectrics": [],
        }
        with mock.patch.object(
            rcs,
            "validate_geometry_snapshot_for_solver",
            side_effect=AssertionError("geometry work started"),
        ):
            with self.assertRaisesRegex(
                ValueError, "cfie_alpha is not implemented"
            ):
                rcs.solve_bistatic_rcs_2d_single_polarization(
                    snapshot,
                    frequencies_ghz=[1.0],
                    incidence_angles_deg=[0.0],
                    observation_angles_deg=[0.0],
                    polarization="TM",
                    geometry_units="meters",
                    cfie_alpha=float("nan"),
                    strict_quality_gate=False,
                    max_panels=1000,
                )
            with self.assertRaisesRegex(
                ValueError, "cfie_alpha is not implemented"
            ):
                rcs.compute_boundary_densities(
                    snapshot,
                    frequency_ghz=1.0,
                    elevation_deg=0.0,
                    polarization="TM",
                    geometry_units="meters",
                    cfie_alpha=1.0e-15,
                    max_panels=1000,
                )

    def test_sheet_memory_gate_runs_before_dense_assembly(self):
        snapshot = {
            "segments": [{
                "name": "sheet",
                "seg_type": 1,
                "properties": ["1", "20", "1", "0", "0"],
                "point_pairs": [{
                    "x1": -0.1, "y1": 0.0, "x2": 0.1, "y2": 0.0,
                }],
            }],
            "ibcs": [["1", "constant", "75", "0", "0", "0"]],
            "dielectrics": [],
        }
        with (
            mock.patch.object(rcs, "_solve_memory_limit_gb", return_value=0.0),
            mock.patch.object(
                rcs, "_solve_tm_sheet",
                side_effect=AssertionError("dense assembly started"),
            ),
        ):
            with self.assertRaisesRegex(MemoryError, "sheet"):
                rcs.solve_monostatic_rcs_2d_single_polarization(
                    snapshot,
                    frequencies_ghz=[1.0],
                    elevations_deg=[0.0],
                    polarization="TM",
                    geometry_units="meters",
                    strict_quality_gate=False,
                    max_panels=1000,
                )


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

    def test_rectangular_far_field_matches_repeated_single_density(self):
        mesh = self._mesh()
        rng = np.random.default_rng(1804)
        angles = np.linspace(-70.0, 85.0, 11)
        densities = (
            rng.normal(size=(len(mesh.nodes), 5))
            + 1j * rng.normal(size=(len(mesh.nodes), 5))
        )
        for potential in ("SLP", "DLP"):
            actual = rcs._farfield_linear_density_many(
                mesh,
                densities,
                4.2,
                angles,
                potential,
                order=8,
                projection="grid",
            )
            expected = np.vstack([
                rcs._farfield_linear_density_many(
                    mesh,
                    densities[:, column],
                    4.2,
                    angles,
                    potential,
                    order=8,
                )
                for column in range(densities.shape[1])
            ])
            np.testing.assert_allclose(
                actual, expected, rtol=2e-14, atol=2e-14
            )


class CylinderPhysicsTests(unittest.TestCase):
    def _solve(self, snapshot, pol, freq_hz):
        result = rcs.solve_monostatic_rcs_2d_single_polarization(
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

    def test_pec_circle_complex_field_survives_first_interior_resonance(self):
        """The indirect resonant nullspace must not contaminate radiation."""

        radius = 0.1
        # First zero of J_0.  This is an interior Dirichlet eigenvalue and is
        # shared by the TM SLP-EFIE and TE adjoint-MFIE formulations.
        resonance_ka = 2.404825557695773
        snapshot = {
            "segments": [_circle_segment(radius, 64, 2)],
            "ibcs": [],
            "dielectrics": [],
        }

        for pol in ("TM", "TE"):
            conditions = []
            resonant_error = None
            for delta_ka in (-0.02, 0.0, 0.02):
                ka = resonance_ka + delta_ka
                frequency_hz = rcs.C0 * ka / (2.0 * math.pi * radius)
                result = rcs.solve_monostatic_rcs_2d_single_polarization(
                    snapshot,
                    frequencies_ghz=[frequency_hz / 1.0e9],
                    elevations_deg=[0.0],
                    polarization=pol,
                    geometry_units="meters",
                    strict_quality_gate=False,
                    compute_condition_number=True,
                    max_panels=1000,
                )
                conditions.append(float(
                    result["metadata"]["condition_est_max"]
                ))
                if delta_ka == 0.0:
                    sample = result["samples"][0]
                    actual = complex(
                        sample["rcs_amp_real"], sample["rcs_amp_imag"]
                    )
                    reference = pec_cylinder_backscatter_amplitude(
                        radius, frequency_hz, pol
                    )
                    resonant_error = abs(actual - reference) / abs(reference)
                    self.assertLess(
                        float(sample["linear_residual"]), 1.0e-10, msg=pol
                    )

            # Ensure this regression truly exercises the conditioning spike,
            # rather than merely sampling an ordinary PEC-circle frequency.
            shoulder = max(conditions[0], conditions[2])
            self.assertGreater(conditions[1] / shoulder, 3.0, msg=pol)
            self.assertLess(resonant_error, 5.0e-3, msg=pol)

    def test_impedance_cylinder_both_polarizations(self):
        radius = 0.1
        freq_hz = rcs.C0 / (2.0 * math.pi * radius)  # ka = 1
        z_s = 75.0 - 20.0j
        impedance = {
            "segments": [_circle_segment(radius, 64, 2, ibc=1)],
            "ibcs": [["1", "constant", "75", "-20", "0", "0"]],
            "dielectrics": [],
        }
        for pol in ("TM", "TE"):
            got = self._solve(impedance, pol, freq_hz)
            reference = sigma_impedance_cylinder(
                radius, z_s, freq_hz, pol
            )
            error_db = abs(10.0 * math.log10(got / reference))
            self.assertLess(error_db, 0.01, msg=pol)

    def test_coated_pec_cylinder_both_polarizations(self):
        inner_radius = 0.07
        outer_radius = 0.1
        freq_hz = rcs.C0 / (2.0 * math.pi * outer_radius)  # k0*b = 1
        eps_r = 3.0 - 0.12j
        coated = {
            "segments": [
                _circle_segment(outer_radius, 64, 3, material=1),
                _circle_segment(inner_radius, 64, 4, material=1),
            ],
            "ibcs": [],
            "dielectrics": [["1", "3", "-0.12", "1", "0"]],
        }
        for pol in ("TM", "TE"):
            got = self._solve(coated, pol, freq_hz)
            reference = sigma_coated_pec_cylinder(
                inner_radius, outer_radius, eps_r, 1.0 + 0.0j,
                freq_hz, pol,
            )
            error_db = abs(10.0 * math.log10(got / reference))
            self.assertLess(error_db, 0.01, msg=pol)


class TaperedImpedanceTests(unittest.TestCase):
    _ANGLES = [-40.0, 0.0, 35.0]

    @staticmethod
    def _snapshot(kind, start, end, reverse=False):
        p0 = (-0.1, 0.0)
        p1 = (0.1, 0.0)
        z0 = complex(start)
        z1 = complex(end)
        if reverse:
            p0, p1 = p1, p0
            z0, z1 = z1, z0
        return {
            "segments": [{
                "name": "sheet",
                "seg_type": 1,
                "properties": ["1", "40", "1", "0", "0"],
                "point_pairs": [{
                    "x1": p0[0], "y1": p0[1],
                    "x2": p1[0], "y2": p1[1],
                }],
            }],
            "ibcs": [[
                "1", kind,
                str(z0.real), str(z0.imag),
                str(z1.real), str(z1.imag),
            ]],
            "dielectrics": [],
        }

    def _amplitudes(self, snapshot, pol):
        result = rcs.solve_monostatic_rcs_2d_single_polarization(
            snapshot,
            frequencies_ghz=[1.0],
            elevations_deg=self._ANGLES,
            polarization=pol,
            geometry_units="meters",
            strict_quality_gate=False,
            max_panels=1000,
        )
        return np.asarray([
            complex(sample["rcs_amp_real"], sample["rcs_amp_imag"])
            for sample in result["samples"]
        ])

    def test_reversing_sheet_and_taper_preserves_physics(self):
        for pol in ("TM", "TE"):
            forward = self._amplitudes(
                self._snapshot("linear", 20.0 - 5.0j, 150.0 + 20.0j), pol
            )
            reverse = self._amplitudes(
                self._snapshot(
                    "linear", 20.0 - 5.0j, 150.0 + 20.0j,
                    reverse=True,
                ),
                pol,
            )
            np.testing.assert_allclose(
                reverse, forward, rtol=2e-12, atol=2e-12,
                err_msg=pol,
            )

    def test_equal_endpoints_reduce_every_taper_to_constant(self):
        z_s = 60.0 + 10.0j
        for pol in ("TM", "TE"):
            reference = self._amplitudes(
                self._snapshot("constant", z_s, z_s), pol
            )
            for kind in ("linear", "cosine", "exp"):
                actual = self._amplitudes(
                    self._snapshot(kind, z_s, z_s), pol
                )
                np.testing.assert_allclose(
                    actual, reference, rtol=2e-12, atol=2e-12,
                    err_msg=f"{pol} {kind}",
                )

    def test_constant_impedance_is_not_classified_as_spatial_taper(self):
        materials = rcs.MaterialLibrary.from_entries(
            [["1", "constant", "60", "10", "60", "10"]],
            [],
            base_dir=".",
        )
        self.assertFalse(materials.is_tapered_impedance(1))


class SheetLimitAndMixedTests(unittest.TestCase):
    _ANGLES = [-35.0, 0.0, 40.0]

    @staticmethod
    def _line(seg_type, ibc=0, y=0.0):
        return {
            "name": f"line_{seg_type}_{y:g}",
            "seg_type": seg_type,
            "properties": [str(seg_type), "36", str(ibc), "0", "0"],
            "point_pairs": [{
                "x1": -0.1, "y1": y, "x2": 0.1, "y2": y,
            }],
        }

    def _amplitudes(self, snapshot, pol):
        result = rcs.solve_monostatic_rcs_2d_single_polarization(
            snapshot,
            frequencies_ghz=[0.75],
            elevations_deg=self._ANGLES,
            polarization=pol,
            geometry_units="meters",
            strict_quality_gate=False,
            max_panels=1000,
        )
        return np.asarray([
            complex(sample["rcs_amp_real"], sample["rcs_amp_imag"])
            for sample in result["samples"]
        ])

    def test_zero_impedance_sheet_matches_tm_open_pec(self):
        sheet = {
            "segments": [self._line(1, ibc=1)],
            "ibcs": [["1", "constant", "1e-8", "0", "0", "0"]],
            "dielectrics": [],
        }
        pec = {
            "segments": [self._line(2)],
            "ibcs": [],
            "dielectrics": [],
        }
        np.testing.assert_allclose(
            self._amplitudes(sheet, "TM"),
            self._amplitudes(pec, "TM"),
            rtol=2e-8, atol=2e-8,
        )

    def test_large_impedance_sheet_approaches_transparency(self):
        ordinary = {
            "segments": [self._line(1, ibc=1)],
            "ibcs": [["1", "constant", "75", "0", "0", "0"]],
            "dielectrics": [],
        }
        transparent = {
            "segments": [self._line(1, ibc=1)],
            "ibcs": [["1", "constant", "1e9", "0", "0", "0"]],
            "dielectrics": [],
        }
        for pol in ("TM", "TE"):
            reference = self._amplitudes(ordinary, pol)
            limit = self._amplitudes(transparent, pol)
            self.assertLess(
                float(np.max(np.abs(limit))),
                1.0e-5 * float(np.max(np.abs(reference))),
                msg=pol,
            )

    def test_mixed_pec_limit_matches_all_sheet_dispatch(self):
        circle_pec = _circle_segment(0.04, 32, 2)
        circle_sheet = _circle_segment(0.04, 32, 1, ibc=1)
        line = self._line(1, ibc=1, y=0.09)
        ibc = [["1", "constant", "1e-8", "0", "0", "0"]]
        mixed = {
            "segments": [circle_pec, line],
            "ibcs": ibc,
            "dielectrics": [],
        }
        all_sheet = {
            "segments": [circle_sheet, line],
            "ibcs": ibc,
            "dielectrics": [],
        }
        for pol in ("TM", "TE"):
            np.testing.assert_allclose(
                self._amplitudes(mixed, pol),
                self._amplitudes(all_sheet, pol),
                rtol=3e-8, atol=3e-8,
                err_msg=pol,
            )


class BistaticReciprocityTests(unittest.TestCase):
    def test_dielectric_triangle_reciprocity_both_polarizations(self):
        points = [(-0.12, -0.08), (-0.02, 0.14), (0.15, -0.06),
                  (-0.12, -0.08)]
        triangle = {
            "name": "triangle",
            "seg_type": 3,
            "properties": ["3", "50", "0", "1", "0"],
            "point_pairs": [
                {"x1": p0[0], "y1": p0[1], "x2": p1[0], "y2": p1[1]}
                for p0, p1 in zip(points[:-1], points[1:])
            ],
        }
        snapshot = {
            "segments": [triangle],
            "ibcs": [],
            "dielectrics": [["1", "3.1", "-0.08", "1", "0"]],
        }
        angles = [20.0, 110.0]
        for pol in ("TM", "TE"):
            result = rcs.solve_bistatic_rcs_2d_single_polarization(
                snapshot,
                frequencies_ghz=[1.0],
                incidence_angles_deg=angles,
                observation_angles_deg=angles,
                polarization=pol,
                geometry_units="meters",
                strict_quality_gate=False,
                max_panels=1000,
            )
            amplitudes = {
                (sample["theta_inc_deg"], sample["theta_scat_deg"]):
                complex(sample["rcs_amp_real"], sample["rcs_amp_imag"])
                for sample in result["samples"]
            }
            forward = amplitudes[(20.0, 110.0)]
            reciprocal = amplitudes[(110.0, 20.0)]
            relative_error = abs(forward - reciprocal) / max(
                abs(forward), abs(reciprocal), np.finfo(float).tiny
            )
            self.assertLess(relative_error, 1.0e-4, msg=pol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
