from __future__ import annotations

import unittest

import numpy as np

from isar_bpde import (
    ComponentDictionary,
    DenseComponentDictionary,
    PointScattererDictionary,
    assess_component_identifiability,
    solve_bpdn_components,
)


class TestIsarBpdeFoundation(unittest.TestCase):
    def test_point_dictionary_adjoint_and_known_phase(self):
        theta = np.deg2rad([-4.0, 0.0, 4.0])
        frequency = np.asarray([9.0e9, 9.5e9, 10.0e9])
        dictionary = PointScattererDictionary(
            "target", theta, frequency, np.asarray([[0.0, 0.0], [0.2, 0.4]])
        )
        coefficients = np.asarray([1.0 + 0.2j, -0.3 + 0.5j])
        samples = np.linspace(0.1, 0.9, 9) + 1j * np.linspace(-0.4, 0.2, 9)
        left = np.vdot(dictionary.forward(coefficients), samples)
        right = np.vdot(coefficients, dictionary.adjoint(samples))
        self.assertAlmostEqual(left.real, right.real, places=10)
        self.assertAlmostEqual(left.imag, right.imag, places=10)
        np.testing.assert_allclose(
            dictionary.forward([1.0, 0.0]), np.ones(9), atol=1.0e-13
        )

    def test_point_sampled_columns_compute_only_selected_columns(self):
        theta = np.deg2rad([-3.0, 3.0])
        frequency = np.asarray([8.0e9, 9.0e9, 10.0e9])
        positions = np.column_stack(
            (np.linspace(-0.3, 0.3, 7), np.linspace(0.1, 0.7, 7))
        )
        dictionary = PointScattererDictionary(
            "sampled",
            theta,
            frequency,
            positions,
            maximum_temporary_cells=18,
            maximum_cached_phase_cells=0,
        )
        sampled = dictionary.sampled_columns(maximum_columns=4)
        self.assertEqual(sampled.shape, (6, 3))

        reference = PointScattererDictionary(
            "reference", theta, frequency, positions
        )
        expected = []
        for index in (0, 3, 6):
            basis = np.zeros(7, dtype=np.complex128)
            basis[index] = 1.0
            expected.append(reference.forward(basis))
        np.testing.assert_allclose(sampled, np.column_stack(expected))
        status = dictionary.cache_status()
        self.assertEqual(status["phase_block_computations"], 0)
        self.assertEqual(status["sampled_column_computations"], 3)

    def test_point_phase_blocks_are_reused_with_bounded_cache(self):
        theta = np.deg2rad([-2.0, 2.0])
        frequency = np.asarray([8.5e9, 9.0e9, 9.5e9])
        positions = np.asarray(
            [[-0.2, 0.1], [-0.1, 0.2], [0.0, 0.3], [0.1, 0.4], [0.2, 0.5]]
        )
        dictionary = PointScattererDictionary(
            "cached",
            theta,
            frequency,
            positions,
            maximum_temporary_cells=12,
            maximum_cached_phase_cells=30,
            maximum_cached_phase_bytes=30 * 16,
        )
        coefficients = np.asarray(
            [1.0 + 0.1j, -0.2j, 0.3, -0.4 + 0.2j, 0.1]
        )
        samples = np.linspace(0.1, 0.6, 6) + 1j * np.linspace(-0.3, 0.2, 6)

        forward = dictionary.forward(coefficients)
        after_forward = dictionary.cache_status()
        self.assertEqual(after_forward["phase_block_computations"], 3)
        self.assertEqual(after_forward["phase_cache_misses"], 3)
        self.assertTrue(after_forward["full_phase_matrix_cached"])

        adjoint = dictionary.adjoint(samples)
        dictionary.forward(coefficients)
        after_reuse = dictionary.cache_status()
        self.assertEqual(after_reuse["phase_block_computations"], 3)
        self.assertEqual(after_reuse["phase_cache_hits"], 6)
        np.testing.assert_allclose(
            np.vdot(forward, samples),
            np.vdot(coefficients, adjoint),
            atol=1.0e-10,
        )

        byte_bounded = PointScattererDictionary(
            "byte-bounded",
            theta,
            frequency,
            positions,
            maximum_temporary_cells=12,
            maximum_cached_phase_cells=30,
            maximum_cached_phase_bytes=12 * 16,
        )
        byte_bounded.forward(coefficients)
        bounded_status = byte_bounded.cache_status()
        self.assertEqual(bounded_status["cache_budget_cells"], 12)
        self.assertLessEqual(
            bounded_status["cached_phase_cells"],
            bounded_status["cache_budget_cells"],
        )

        partial = PointScattererDictionary(
            "partial",
            theta,
            frequency,
            positions,
            maximum_temporary_cells=12,
            maximum_cached_phase_cells=12,
            maximum_cached_phase_bytes=12 * 16,
        )
        partial.forward(coefficients)
        partial.adjoint(samples)
        after_partial_pair = partial.cache_status()
        self.assertEqual(after_partial_pair["phase_cache_hits"], 1)
        self.assertEqual(after_partial_pair["phase_block_computations"], 5)
        partial.forward(coefficients)
        after_partial_reuse = partial.cache_status()
        self.assertEqual(after_partial_reuse["phase_cache_hits"], 2)
        self.assertEqual(after_partial_reuse["phase_block_computations"], 7)

        block_bounded = PointScattererDictionary(
            "block-bounded",
            theta,
            frequency,
            positions,
            maximum_temporary_cells=12,
            maximum_cached_phase_cells=30,
            maximum_cached_phase_bytes=30 * 16,
            maximum_cached_phase_blocks=1,
        )
        block_bounded.forward(coefficients)
        block_status = block_bounded.cache_status()
        self.assertEqual(block_status["cached_phase_blocks"], 1)
        self.assertFalse(block_status["full_phase_matrix_cacheable"])
        self.assertIn("metadata", block_status["cache_byte_scope"])

    def test_phase_cache_readiness_uses_whole_blocks_and_can_reset(self):
        with self.assertRaisesRegex(ValueError, "one complete point column"):
            PointScattererDictionary(
                "undersized-temporary",
                np.linspace(-0.1, 0.1, 10),
                np.linspace(8.0e9, 9.0e9, 5),
                np.asarray([[0.1, 0.2]]),
                maximum_temporary_cells=49,
            )
        dictionary = PointScattererDictionary(
            "indivisible",
            np.linspace(-0.1, 0.1, 10),
            np.linspace(8.0e9, 9.0e9, 5),
            np.asarray([[0.1, 0.2]]),
            maximum_temporary_cells=50,
            maximum_cached_phase_cells=49,
            maximum_cached_phase_bytes=49 * 16,
            maximum_uncached_iteration_cells=499,
        )
        readiness = dictionary.iterative_readiness(
            1, norm_iterations=2
        )
        self.assertEqual(readiness["phase_matrix_cells"], 50)
        self.assertEqual(readiness["cache_budget_cells"], 49)
        self.assertEqual(readiness["usable_canonical_cache_cells"], 0)
        self.assertEqual(
            readiness["projected_uncached_cells_lower_bound"], 500
        )
        self.assertTrue(readiness["uncached_oversized"])
        dictionary.forward([1.0])
        self.assertEqual(dictionary.cache_status()["cached_phase_cells"], 0)
        dictionary.clear_phase_cache(reset_statistics=True)
        reset = dictionary.cache_status()
        self.assertEqual(reset["phase_cache_misses"], 0)
        self.assertEqual(reset["phase_block_computations"], 0)
        with self.assertRaises(AttributeError):
            dictionary.maximum_cached_phase_cells = 100

    def test_uncached_workload_upper_bound_includes_every_solve_phase(self):
        dictionary = PointScattererDictionary(
            "accounted",
            np.asarray([-0.1, 0.1]),
            np.asarray([9.0e9]),
            np.asarray([[0.0, 0.0]]),
            maximum_temporary_cells=2,
            maximum_cached_phase_cells=0,
            maximum_cached_phase_bytes=0,
            maximum_uncached_iteration_cells=1_000_000,
        )
        readiness = dictionary.iterative_readiness(
            1, norm_iterations=2, history_stride=1
        )
        self.assertEqual(
            readiness["planned_column_normalization_applications"], 1
        )
        self.assertEqual(readiness["planned_norm_operator_applications"], 8)
        self.assertEqual(readiness["planned_solver_operator_applications"], 2)
        self.assertEqual(readiness["planned_history_operator_applications"], 1)
        self.assertEqual(
            readiness["planned_final_reconstruction_applications"], 1
        )
        self.assertEqual(
            readiness["planned_operator_applications_upper_bound"], 13
        )
        self.assertEqual(
            readiness[
                "projected_identifiability_sample_cells_upper_bound"
            ],
            2,
        )
        self.assertEqual(
            readiness["projected_uncached_cells_upper_bound"], 28
        )

        result = solve_bpdn_components(
            np.ones(2, dtype=np.complex128),
            [dictionary],
            sigma=1.0e6,
            max_iterations=1,
            norm_iterations=2,
            history_stride=1,
        )
        status = result.operator_readiness["accounted"]
        actual_uncached_cells = (
            int(status["solve_phase_block_computations"])
            * int(status["phase_matrix_cells"])
            + int(status["solve_sampled_column_computations"])
            * dictionary.measurement_count
        )
        self.assertEqual(actual_uncached_cells, 28)
        self.assertLessEqual(
            actual_uncached_cells,
            int(status["projected_uncached_cells_upper_bound"]),
        )

        guarded = PointScattererDictionary(
            "guarded-accounting",
            np.asarray([-0.1, 0.1]),
            np.asarray([9.0e9]),
            np.asarray([[0.0, 0.0]]),
            maximum_temporary_cells=2,
            maximum_cached_phase_cells=0,
            maximum_cached_phase_bytes=0,
            maximum_uncached_iteration_cells=27,
        )
        with self.assertRaisesRegex(
            ValueError, "allow_uncached_oversized_point_operator=True"
        ):
            solve_bpdn_components(
                np.ones(2, dtype=np.complex128),
                [guarded],
                sigma=1.0e6,
                max_iterations=1,
                norm_iterations=2,
                history_stride=1,
            )
        self.assertEqual(
            guarded.cache_status()["phase_block_computations"], 0
        )

    def test_uncached_oversized_point_operator_requires_explicit_opt_in(self):
        dictionary = PointScattererDictionary(
            "large-direct",
            np.deg2rad([-2.0, 2.0]),
            np.asarray([8.5e9, 9.0e9, 9.5e9]),
            np.asarray(
                [
                    [-0.2, 0.1],
                    [-0.1, 0.2],
                    [0.0, 0.3],
                    [0.1, 0.4],
                    [0.2, 0.5],
                ]
            ),
            maximum_temporary_cells=12,
            maximum_cached_phase_cells=0,
            maximum_cached_phase_bytes=0,
            maximum_uncached_iteration_cells=100,
        )
        measurements = np.ones(6, dtype=np.complex128)
        with self.assertRaisesRegex(
            ValueError,
            "allow_uncached_oversized_point_operator=True",
        ) as raised:
            solve_bpdn_components(
                measurements,
                [dictionary],
                sigma=1.0e6,
                max_iterations=1001,
                norm_iterations=2,
            )
        self.assertIn("cache", str(raised.exception).lower())
        self.assertIn("NUFFT", str(raised.exception))
        self.assertEqual(
            dictionary.cache_status()["phase_block_computations"], 0
        )

        result = solve_bpdn_components(
            measurements,
            [dictionary],
            sigma=1.0e6,
            max_iterations=1001,
            norm_iterations=2,
            allow_uncached_oversized_point_operator=True,
        )
        self.assertTrue(result.converged, result.stopping_reason)
        self.assertEqual(result.iterations, 1)
        readiness = result.operator_readiness["large-direct"]
        self.assertEqual(
            readiness["readiness"], "uncached_oversized_opt_in_accepted"
        )
        self.assertEqual(
            readiness["guard_readiness"],
            "uncached_oversized_opt_in_required",
        )
        self.assertTrue(readiness["uncached_oversized_opt_in_used"])
        self.assertGreater(readiness["solve_phase_block_computations"], 0)

    def test_combined_uncached_point_budget_is_guarded(self):
        common = dict(
            azimuth_radians=np.asarray([-0.1, 0.1]),
            frequency_hz=np.asarray([9.0e9]),
            positions_m=np.asarray([[0.0, 0.0]]),
            maximum_temporary_cells=2,
            maximum_cached_phase_cells=0,
            maximum_cached_phase_bytes=0,
            maximum_uncached_iteration_cells=500,
        )
        target = PointScattererDictionary(
            "target",
            visibility=np.asarray([[1.0], [0.0]]),
            **common,
        )
        support = PointScattererDictionary(
            "support",
            visibility=np.asarray([[0.0], [1.0]]),
            **common,
        )
        measurements = np.ones(2, dtype=np.complex128)
        with self.assertRaisesRegex(
            ValueError, "combined uncached cell evaluations"
        ) as raised:
            solve_bpdn_components(
                measurements,
                [target, support],
                sigma=1.0e6,
                max_iterations=100,
                norm_iterations=2,
                maximum_total_uncached_iteration_cells=700,
            )
        self.assertIn(
            "allow_uncached_oversized_point_operator=True",
            str(raised.exception),
        )

        result = solve_bpdn_components(
            measurements,
            [target, support],
            sigma=1.0e6,
            max_iterations=100,
            norm_iterations=2,
            maximum_total_uncached_iteration_cells=700,
            allow_uncached_oversized_point_operator=True,
        )
        self.assertTrue(result.converged, result.stopping_reason)
        for status in result.operator_readiness.values():
            self.assertTrue(status["aggregate_uncached_oversized"])
            self.assertTrue(status["uncached_oversized_opt_in_used"])
            self.assertEqual(
                status["readiness"],
                "uncached_oversized_opt_in_accepted",
            )

    def test_duplicate_component_basis_is_rejected_as_unidentifiable(self):
        column = np.asarray([[1.0], [1.0j], [-1.0]])
        target = DenseComponentDictionary("target", column)
        support = DenseComponentDictionary("support", column.copy())
        report = assess_component_identifiability([target, support])
        self.assertFalse(report.identifiable)
        with self.assertRaisesRegex(ValueError, "not identifiable"):
            solve_bpdn_components(column[:, 0], [target, support], sigma=0.0)

    def test_bpdn_separates_distinct_components_and_returns_residual(self):
        identity = np.eye(6, dtype=np.complex128)
        target = DenseComponentDictionary("target", identity[:, :3])
        support = DenseComponentDictionary("support", identity[:, 3:])
        target_coefficients = np.asarray([1.0 + 0.2j, 0.0, -0.5j])
        support_coefficients = np.asarray([0.0, 0.7 - 0.1j, 0.0])
        measurements = (
            target.forward(target_coefficients)
            + support.forward(support_coefficients)
        )
        result = solve_bpdn_components(
            measurements,
            [target, support],
            sigma=1.0e-6,
            tolerance=2.0e-5,
            max_iterations=5000,
        )
        self.assertTrue(result.converged, result.stopping_reason)
        self.assertLessEqual(result.residual_norm, 3.0e-5)
        np.testing.assert_allclose(
            result.reconstructed_components["target"],
            target.forward(target_coefficients),
            atol=3.0e-5,
        )
        np.testing.assert_allclose(
            result.reconstructed_components["support"],
            support.forward(support_coefficients),
            atol=3.0e-5,
        )

    def test_pdhg_steps_use_certified_bound_not_power_underestimate(self):
        # The deterministic power seed is nearly orthogonal to the dominant
        # six-column singular subspace. Even after the default 20 iterations,
        # its estimate is 2 while the true operator norm is sqrt(6). Using the
        # estimate for both PDHG steps violates tau*sigma*||K||^2 < 1.
        phases = np.asarray(
            [
                -1.00204285,
                -2.85213549,
                -0.54263917,
                -2.84683416,
                1.62349440,
                1.25506161,
            ]
        )
        matrix = np.zeros((2, 10), dtype=np.complex128)
        matrix[0, :6] = np.exp(1j * phases)
        matrix[1, 6:] = 1.0
        dictionary = DenseComponentDictionary("adversarial", matrix)
        result = solve_bpdn_components(
            np.asarray([1.0, 0.0], dtype=np.complex128),
            [dictionary],
            sigma=0.0,
            tolerance=1.0e-8,
            max_iterations=500,
            norm_iterations=20,
            history_stride=1,
        )
        true_norm = float(np.linalg.norm(matrix, ord=2))
        self.assertLess(result.operator_norm_estimate, 0.9 * true_norm)
        self.assertGreaterEqual(
            result.operator_norm,
            true_norm * (1.0 - 1.0e-12),
        )
        self.assertLess(
            (0.99 * true_norm / result.operator_norm) ** 2,
            1.0,
        )
        self.assertIn("bound", result.operator_norm_bound_kind)
        self.assertTrue(result.converged, result.stopping_reason)
        self.assertLess(result.residual_norm, 1.0e-8)

    def test_sigma_is_a_residual_bound_not_a_lambda_fraction(self):
        identity = DenseComponentDictionary("target", np.eye(4))
        measurements = np.asarray([2.0, 1.0, 0.25, 0.0], dtype=np.complex128)
        result = solve_bpdn_components(
            measurements,
            [identity],
            sigma=0.5,
            tolerance=2.0e-5,
            max_iterations=5000,
        )
        self.assertTrue(result.converged, result.stopping_reason)
        self.assertLessEqual(result.residual_norm, 0.5001)
        self.assertGreater(result.residual_norm, 0.48)
        self.assertLess(result.l1_norm, float(np.sum(np.abs(measurements))))

    def test_tiny_equality_problem_is_not_falsely_converged_at_zero(self):
        identity = DenseComponentDictionary("target", np.eye(2))
        first_iteration = solve_bpdn_components(
            np.asarray([1.0e-6, 0.0], dtype=np.complex128),
            [identity],
            sigma=0.0,
            tolerance=1.0e-5,
            max_iterations=1,
            norm_iterations=2,
            history_stride=1,
        )
        self.assertFalse(first_iteration.converged)
        self.assertEqual(first_iteration.coefficients["target"][0], 0.0)
        self.assertGreater(first_iteration.fixed_point_residual, 0.9)
        self.assertLess(first_iteration.feasibility_tolerance, 1.0e-10)

    def test_equality_solution_and_iterations_are_invariant_across_amplitude(self):
        identity = DenseComponentDictionary("target", np.eye(2))
        iteration_counts = []
        for amplitude in (1.0e-12, 1.0e-6, 1.0e-4, 1.0, 1.0e6):
            with self.subTest(amplitude=amplitude):
                measurements = np.asarray(
                    [amplitude, 0.0], dtype=np.complex128
                )
                result = solve_bpdn_components(
                    measurements,
                    [identity],
                    sigma=0.0,
                    tolerance=1.0e-6,
                    max_iterations=500,
                    norm_iterations=5,
                    history_stride=1,
                )
                self.assertTrue(result.converged, result.stopping_reason)
                self.assertGreater(result.iterations, 1)
                iteration_counts.append(result.iterations)
                np.testing.assert_allclose(
                    result.coefficients["target"],
                    measurements,
                    rtol=1.0e-6,
                    atol=amplitude * 1.0e-12,
                )
                self.assertLessEqual(
                    result.residual_norm,
                    result.feasibility_tolerance,
                )
                self.assertLessEqual(
                    result.fixed_point_residual, 1.0e-6
                )
                self.assertAlmostEqual(
                    result.internal_problem_scale, amplitude
                )
        self.assertEqual(len(set(iteration_counts)), 1)

    def test_nonzero_sigma_solution_is_invariant_across_data_amplitude(self):
        identity = DenseComponentDictionary("target", np.eye(4))
        base_measurements = np.asarray(
            [2.0, 1.0, 0.25, 0.0], dtype=np.complex128
        )
        normalized_coefficients = []
        normalized_residuals = []
        iteration_counts = []
        for amplitude in (1.0e-8, 1.0, 1.0e8):
            result = solve_bpdn_components(
                amplitude * base_measurements,
                [identity],
                sigma=amplitude * 0.5,
                tolerance=2.0e-5,
                max_iterations=5000,
            )
            self.assertTrue(result.converged, result.stopping_reason)
            normalized_coefficients.append(
                result.coefficients["target"] / amplitude
            )
            normalized_residuals.append(result.residual_norm / amplitude)
            iteration_counts.append(result.iterations)
        for coefficients in normalized_coefficients[1:]:
            np.testing.assert_allclose(
                coefficients,
                normalized_coefficients[0],
                atol=2.0e-6,
            )
        np.testing.assert_allclose(
            normalized_residuals,
            np.repeat(normalized_residuals[0], 3),
            atol=2.0e-6,
        )
        self.assertEqual(len(set(iteration_counts)), 1)

    def test_zero_is_valid_optimum_when_sigma_contains_measurement(self):
        identity = DenseComponentDictionary("target", np.eye(2))
        result = solve_bpdn_components(
            np.asarray([3.0e-12, 4.0e-12], dtype=np.complex128),
            [identity],
            sigma=5.0e-12,
            tolerance=1.0e-7,
            max_iterations=10,
            history_stride=1,
        )
        self.assertTrue(result.converged, result.stopping_reason)
        self.assertEqual(result.iterations, 1)
        np.testing.assert_array_equal(
            result.coefficients["target"], np.zeros(2)
        )
        self.assertEqual(result.fixed_point_residual, 0.0)

    def test_component_attribution_is_invariant_to_dictionary_column_scaling(self):
        target_column = np.asarray([[1.0], [0.0], [0.0]], dtype=np.complex128)
        support_column = np.asarray([[0.0], [1.0], [0.0]], dtype=np.complex128)
        target = DenseComponentDictionary("target", target_column)
        support = DenseComponentDictionary("support", support_column)
        target_coefficient = 2.0 + 0.4j
        support_coefficient = -0.7 + 1.1j
        measurements = (
            target.forward([target_coefficient])
            + support.forward([support_coefficient])
        )
        baseline = solve_bpdn_components(
            measurements,
            [target, support],
            sigma=1.0e-7,
            tolerance=1.0e-6,
            max_iterations=6000,
        )

        target_scale = 1.0e100 * np.exp(0.37j)
        support_scale = 1.0e-100 * np.exp(-0.22j)
        scaled_target = DenseComponentDictionary(
            "target", target_column * target_scale
        )
        scaled_support = DenseComponentDictionary(
            "support", support_column * support_scale
        )
        scaled = solve_bpdn_components(
            measurements,
            [scaled_target, scaled_support],
            sigma=1.0e-7,
            tolerance=1.0e-6,
            max_iterations=6000,
        )

        self.assertTrue(baseline.converged, baseline.stopping_reason)
        self.assertTrue(scaled.converged, scaled.stopping_reason)
        for name in ("target", "support"):
            np.testing.assert_allclose(
                scaled.reconstructed_components[name],
                baseline.reconstructed_components[name],
                atol=2.0e-6,
            )
        np.testing.assert_allclose(
            scaled.coefficients["target"] * target_scale,
            baseline.coefficients["target"],
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            scaled.coefficients["support"] * support_scale,
            baseline.coefficients["support"],
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        self.assertAlmostEqual(scaled.l1_norm, baseline.l1_norm, places=7)
        self.assertIn("||W*g_j||", scaled.l1_norm_definition)

    def test_identifiability_uses_masked_observed_operator(self):
        target = DenseComponentDictionary(
            "target", np.asarray([[1.0], [0.0]], dtype=np.complex128)
        )
        support = DenseComponentDictionary(
            "support", np.asarray([[1.0], [1.0]], dtype=np.complex128)
        )
        unweighted = assess_component_identifiability([target, support])
        self.assertTrue(unweighted.screen_passed)

        weights = np.asarray([1.0, 0.0])
        observed = assess_component_identifiability(
            [target, support], noise_whitening_weights=weights
        )
        self.assertFalse(observed.screen_passed)
        self.assertAlmostEqual(observed.maximum_cross_coherence, 1.0)
        self.assertIn("W*G", observed.screen_note)
        self.assertIn("not proof", observed.screen_note)
        with self.assertRaisesRegex(ValueError, "observed weighted samples"):
            solve_bpdn_components(
                np.asarray([1.0, 0.0], dtype=np.complex128),
                [target, support],
                sigma=0.0,
                noise_whitening_weights=weights,
            )

    def test_nonfinite_dictionary_measurements_weights_and_operator_outputs_fail(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            DenseComponentDictionary("bad", np.asarray([[np.nan], [1.0]]))

        identity = DenseComponentDictionary("target", np.eye(2))
        with self.assertRaisesRegex(ValueError, "measurements.*finite"):
            solve_bpdn_components(
                np.asarray([1.0, np.nan]),
                [identity],
                sigma=0.0,
                noise_whitening_weights=np.asarray([1.0, 0.0]),
            )
        with self.assertRaisesRegex(ValueError, "weights must be finite"):
            solve_bpdn_components(
                np.asarray([1.0, 0.0]),
                [identity],
                sigma=0.0,
                noise_whitening_weights=np.asarray([1.0, np.nan]),
            )
        with self.assertRaisesRegex(ValueError, "sigma must be finite"):
            solve_bpdn_components(
                np.asarray([1.0, 0.0]), [identity], sigma=np.nan
            )

        class _NonfiniteAdjointDictionary(ComponentDictionary):
            def __init__(self):
                super().__init__("nonfinite-adjoint", 1, 2)

            def forward(self, coefficients):
                return np.asarray([complex(coefficients[0]), 0.0j])

            def adjoint(self, measurements):
                return np.asarray([np.nan + 0.0j])

        with self.assertRaisesRegex(ValueError, "adjoint output.*finite"):
            solve_bpdn_components(
                np.asarray([1.0, 0.0]),
                [_NonfiniteAdjointDictionary()],
                sigma=0.0,
            )


if __name__ == "__main__":
    unittest.main()
