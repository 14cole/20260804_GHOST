"""Focused regressions for dataset transforms and Pioneer interchange."""

from __future__ import annotations

import os
import tempfile
import unittest
from itertools import permutations
from unittest import mock

import numpy as np

import grim_dataset
from grim_dataset import RcsGrid
from plot_modes.az_vs_range_mode import _range_display_values


class DatasetCorrectnessTests(unittest.TestCase):
    @staticmethod
    def _single_sample(*, phase=0.0, power=1.0) -> RcsGrid:
        return RcsGrid(
            [0.0],
            [0.0],
            [1.0],
            ["VV"],
            rcs_power=np.asarray([power], dtype=np.float64).reshape(1, 1, 1, 1),
            rcs_phase=np.asarray([phase], dtype=np.float64).reshape(1, 1, 1, 1),
            units={"frequency": "GHz"},
        )

    def test_dbke_std_uses_frequency_unit_exactly_once(self):
        expected = float(np.std(10.0 * np.log10([1.0, 2.0])))
        frequencies = {
            "Hz": 1.0e9,
            "kHz": 1.0e6,
            "MHz": 1.0e3,
            "GHz": 1.0,
        }
        for unit, frequency in frequencies.items():
            with self.subTest(unit=unit):
                power = np.asarray([1.0, 2.0]).reshape(1, 2, 1, 1)
                grid = RcsGrid(
                    [0.0],
                    [0.0, 1.0],
                    [frequency],
                    ["VV"],
                    rcs_power=power,
                    rcs_phase=np.zeros_like(power),
                    units={"frequency": unit},
                )
                reduced = grid.statistics_dataset(
                    "std", ["elevation"], domain="dbke"
                )
                displayed = reduced.linear_to_dbke(
                    reduced.rcs_power,
                    reduced.frequencies.reshape(1, 1, -1, 1),
                )
                self.assertAlmostEqual(float(displayed.item()), expected, places=10)

    def test_save_rejects_object_metadata_before_publishing_archive(self):
        grid = self._single_sample()
        grid.extra["placement"] = {"offset": [1.0, 2.0, 3.0]}
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "unsafe.grim")
            with self.assertRaisesRegex(ValueError, "object-typed.*placement"):
                grid.save(target)
            self.assertFalse(os.path.exists(target))

    def test_load_rejects_corrupt_units_instead_of_guessing_defaults(self):
        grid = self._single_sample()
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "corrupt.grim")
            with open(target, "wb") as stream:
                np.savez(
                    stream,
                    azimuths=grid.azimuths,
                    elevations=grid.elevations,
                    frequencies=grid.frequencies,
                    polarizations=grid.polarizations,
                    rcs_power=grid.rcs_power,
                    rcs_phase=grid.rcs_phase,
                    units="{not-json",
                )
            with self.assertRaisesRegex(ValueError, "corrupt units metadata"):
                RcsGrid.load(target)

    def test_join_fills_complementary_phase_independent_of_input_order(self):
        missing = self._single_sample(phase=np.nan)
        known = self._single_sample(phase=0.75)
        for grids in ((missing, known), (known, missing)):
            with self.subTest(order="missing-first" if grids[0] is missing else "known-first"):
                joined = RcsGrid.join_many(*grids)
                self.assertAlmostEqual(float(joined.rcs_phase.item()), 0.75)

    def test_join_accounts_for_peak_and_adopts_fresh_output_arrays(self):
        left = self._single_sample()
        right = RcsGrid(
            [1.0],
            [0.0],
            [1.0],
            ["VV"],
            rcs_power=np.ones((1, 1, 1, 1), dtype=np.float64),
            rcs_phase=np.zeros((1, 1, 1, 1), dtype=np.float64),
            units={"frequency": "GHz"},
        )
        retained_bytes = 2 * 2 * np.dtype(np.float64).itemsize
        with self.assertRaisesRegex(MemoryError, "peak"):
            RcsGrid.join_many(left, right, max_output_bytes=retained_bytes)

        # Inputs have already been constructed. A join result should not invoke
        # the copying cleaner again after its in-place sanitation step.
        with mock.patch.object(
            RcsGrid,
            "_clean_power",
            side_effect=AssertionError("join copied its fresh power array"),
        ):
            joined = RcsGrid.join_many(left, right, max_output_bytes=1_000_000)
        np.testing.assert_array_equal(joined.azimuths, [0.0, 1.0])
        np.testing.assert_array_equal(joined.rcs_power.ravel(), [1.0, 1.0])

    def test_join_bounded_blocks_preserve_multiaxis_mapping_and_precision(self):
        left_power = np.arange(1, 17, dtype=np.float32).reshape(2, 2, 2, 2)
        left = RcsGrid(
            [0.0, 2.0],
            [-1.0, 1.0],
            [1.0, 3.0],
            ["VV", "HH"],
            rcs_power=left_power,
            rcs_phase=np.zeros_like(left_power),
            units={"frequency": "GHz"},
        )
        right_power = (100.0 + np.arange(8, dtype=np.float64)).reshape(1, 2, 2, 2)
        right = RcsGrid(
            [1.0],
            [-1.0, 1.0],
            [2.0, 3.0],
            ["VV", "HH"],
            rcs_power=right_power,
            rcs_phase=np.full_like(right_power, 0.25),
            units={"frequency": "GHz"},
        )
        joined = RcsGrid.join_many(left, right)
        self.assertEqual(joined.rcs_power.dtype, np.float64)
        np.testing.assert_array_equal(joined.azimuths, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(joined.frequencies, [1.0, 2.0, 3.0])

        for source in (left, right):
            for ai, azimuth in enumerate(source.azimuths):
                oi = int(np.flatnonzero(joined.azimuths == azimuth)[0])
                for ei, elevation in enumerate(source.elevations):
                    oj = int(np.flatnonzero(joined.elevations == elevation)[0])
                    for fi, frequency in enumerate(source.frequencies):
                        ok = int(np.flatnonzero(joined.frequencies == frequency)[0])
                        np.testing.assert_allclose(
                            joined.rcs_power[oi, oj, ok],
                            source.rcs_power[ai, ei, fi],
                        )
                        np.testing.assert_allclose(
                            joined.rcs_phase[oi, oj, ok],
                            source.rcs_phase[ai, ei, fi],
                        )

    def test_join_preserves_complete_raw_solver_amplitude_for_coherent_delta(self):
        scale = np.sqrt(4.0 * np.pi)

        def solver_grid(frequency, raw_value):
            raw = np.asarray([raw_value], dtype=np.complex128).reshape(1, 1, 1, 1)
            normalized = raw * scale
            return RcsGrid(
                [0.0],
                [0.0],
                [frequency],
                ["VV"],
                rcs_power=np.abs(normalized).astype(np.float32) ** 2,
                rcs_phase=np.angle(normalized).astype(np.float32),
                units={
                    "frequency": "GHz",
                    "rcs_linear_quantity": "sigma_3d",
                },
                extra={
                    "rcs_amp_real": raw.real,
                    "rcs_amp_imag": raw.imag,
                    "phase_reference": "origin=(0,0), convention=exp(+jwt)",
                    "time_convention": "exp(+jwt)",
                    "polarization_basis": "earth V/H",
                    "amplitude_convention": "solver far-field amplitude",
                    "complex_field_domain": "far_field_scattering_amplitude",
                },
            )

        opn = RcsGrid.join_many(
            solver_grid(1.0, 1.000001 + 2.0e-8j),
            solver_grid(2.0, 1.000002 + 3.0e-8j),
        )
        frd = RcsGrid.join_many(
            solver_grid(1.0, 1.0 + 0.0j),
            solver_grid(2.0, 1.0 + 0.0j),
        )

        self.assertTrue(opn.extra["raw_complex_amplitude_preserved"])
        np.testing.assert_array_equal(
            opn.extra["rcs_amp_real"].ravel(), [1.000001, 1.000002]
        )
        expected = scale * np.asarray([1.0e-6 + 2.0e-8j, 2.0e-6 + 3.0e-8j])
        delta = opn.coherent_subtract(frd)
        np.testing.assert_allclose(delta.rcs.ravel(), expected, rtol=1.0e-10, atol=1.0e-15)

    def test_join_drops_raw_amplitude_when_any_input_lacks_it(self):
        with_raw = self._single_sample()
        with_raw.extra["rcs_amp_real"] = np.ones((1, 1, 1, 1), dtype=np.float64)
        with_raw.extra["rcs_amp_imag"] = np.zeros((1, 1, 1, 1), dtype=np.float64)
        without_raw = RcsGrid(
            [1.0],
            [0.0],
            [1.0],
            ["VV"],
            rcs_power=np.ones((1, 1, 1, 1), dtype=np.float64),
            rcs_phase=np.zeros((1, 1, 1, 1), dtype=np.float64),
            units={"frequency": "GHz"},
        )
        joined = RcsGrid.join_many(with_raw, without_raw)
        self.assertNotIn("rcs_amp_real", joined.extra)
        self.assertNotIn("rcs_amp_imag", joined.extra)

    def test_join_detects_raw_field_conflict_hidden_by_float32_power(self):
        def raw_grid(value):
            grid = RcsGrid(
                [0.0], [0.0], [1.0], ["VV"],
                rcs_power=np.asarray([1.0], dtype=np.float32).reshape(1, 1, 1, 1),
                rcs_phase=np.asarray([0.0], dtype=np.float32).reshape(1, 1, 1, 1),
                units={
                    "frequency": "GHz",
                    "rcs_linear_quantity": "sigma_3d",
                },
                extra={
                    "rcs_amp_real": np.asarray([value], dtype=np.float64).reshape(
                        1, 1, 1, 1
                    ),
                    "rcs_amp_imag": np.zeros((1, 1, 1, 1), dtype=np.float64),
                },
            )
            return grid

        with self.assertRaisesRegex(ValueError, "conflicting finite samples"):
            RcsGrid.join_many(raw_grid(1.0), raw_grid(1.0 + 1.0e-10))

    def test_join_preserves_matching_assembly_role_and_rejects_base_mismatch(self):
        provenance = '{"schema":"ghost.workflow.coherent-feature-addition.v1"}'

        def assembled_slice(azimuth, base_digest, response_digest="c" * 64):
            return RcsGrid(
                [azimuth], [0.0], [1.0], ["VV"],
                rcs_power=np.ones((1, 1, 1, 1)),
                rcs_phase=np.zeros((1, 1, 1, 1)),
                units={"frequency": "GHz"},
                extra={
                    "assembly_response_role": "body_plus_features",
                    "assembly_base_sha256": base_digest,
                    "assembly_base_response_sha256": response_digest,
                    "feature_provenance_json": provenance,
                },
            )

        joined = RcsGrid.join_many(
            assembled_slice(0.0, "a" * 64),
            assembled_slice(1.0, "a" * 64),
        )
        self.assertEqual(
            joined.extra["assembly_response_role"], "body_plus_features"
        )
        self.assertEqual(joined.extra["assembly_base_sha256"], "a" * 64)
        self.assertEqual(
            joined.extra["assembly_base_response_sha256"], "c" * 64
        )
        self.assertEqual(joined.extra["feature_provenance_json"], provenance)

        with self.assertRaisesRegex(ValueError, "Assembly base identities"):
            RcsGrid.join_many(
                assembled_slice(0.0, "a" * 64),
                assembled_slice(1.0, "b" * 64),
            )
        with self.assertRaisesRegex(
            ValueError, "Assembly base response identities"
        ):
            RcsGrid.join_many(
                assembled_slice(0.0, "a" * 64, "c" * 64),
                assembled_slice(1.0, "a" * 64, "d" * 64),
            )

    @staticmethod
    def _raw_solver_grid(raw_values, *, azimuths, frequencies, role=None):
        raw = np.asarray(raw_values, dtype=np.complex128).reshape(
            len(azimuths), 1, len(frequencies), 1
        )
        shape = raw.shape
        extra = {
            "rcs_amp_real": raw.real.copy(),
            "rcs_amp_imag": raw.imag.copy(),
            "phase_reference": "vehicle origin",
            "time_convention": "exp(+jwt)",
            "polarization_basis": "earth V/H",
            "amplitude_convention": "GHOST raw far-field amplitude",
            "complex_field_domain": "far_field_scattering_amplitude",
        }
        if role is not None:
            extra.update({
                "assembly_response_role": role,
                "assembly_base_sha256": "a" * 64,
                "assembly_base_response_sha256": "b" * 64,
                "feature_provenance_json": (
                    '[{"schema":"ghost.workflow.coherent-feature-addition.v1",'
                    '"source_monostatic_sha256":"' + "a" * 64 + '"}]'
                ),
            })
        # Deliberately erase the small raw differences from the display pair.
        # The regression proves every exact transform keeps the authoritative
        # float64 field rather than laundering it through float32 power/phase.
        return RcsGrid(
            azimuths,
            [0.0],
            frequencies,
            ["VV"],
            rcs_power=np.ones(shape, dtype=np.float32),
            rcs_phase=np.zeros(shape, dtype=np.float32),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_linear_quantity": "sigma_3d",
            },
            extra=extra,
        )

    def test_exact_phase_crop_and_azimuth_wrap_keep_float64_cancellation(self):
        epsilon = 2.0e-10 + 3.0e-11j
        open_grid = self._raw_solver_grid(
            [1.0 + epsilon, 1.0 + epsilon, 1.0 + epsilon, 1.0 + epsilon],
            azimuths=[0.0, 360.0],
            frequencies=[1.0, 2.0],
        )
        closed_grid = self._raw_solver_grid(
            [1.0, 1.0, 1.0, 1.0],
            azimuths=[0.0, 360.0],
            frequencies=[1.0, 2.0],
        )

        def transformed(grid):
            return (
                grid.wrap_phase("0_360")
                .axis_crop(frequencies=[2.0])
                .wrap_azimuth("0_360")
            )

        delta = transformed(open_grid).coherent_subtract(
            transformed(closed_grid)
        )
        expected = np.sqrt(4.0 * np.pi) * epsilon
        self.assertEqual(delta.rcs_power.dtype, np.float64)
        np.testing.assert_allclose(
            delta.rcs.item(), expected, rtol=1.0e-7, atol=1.0e-15
        )

    def test_azimuth_wrap_rejects_raw_conflict_hidden_by_float32_display(self):
        conflict = self._raw_solver_grid(
            [1.0, 1.0 + 1.0e-10],
            azimuths=[0.0, 360.0],
            frequencies=[1.0],
        )
        with self.assertRaisesRegex(
            ValueError, "conflicting authoritative raw seam samples"
        ):
            conflict.wrap_azimuth("0_360")

    def test_interpolation_uses_authoritative_raw_complex_field(self):
        source = self._raw_solver_grid(
            [1.0, 1.0 + 2.0e-10],
            azimuths=[0.0, 2.0],
            frequencies=[1.0],
        )
        interpolated = source.interpolate_axis("azimuth", [1.0])
        expected = np.sqrt(4.0 * np.pi) * (1.0 + 1.0e-10)
        np.testing.assert_allclose(
            interpolated.rcs.item(), expected, rtol=0.0, atol=1.0e-14
        )
        self.assertNotIn("rcs_amp_real", interpolated.extra)
        self.assertNotIn("rcs_amp_imag", interpolated.extra)

    def test_exact_transforms_drop_partial_raw_pair_instead_of_creating_nans(self):
        malformed = self._raw_solver_grid(
            [1.0, 1.0],
            azimuths=[0.0, 1.0],
            frequencies=[1.0],
        )
        malformed.extra["rcs_amp_imag"] = np.asarray(
            malformed.extra["rcs_amp_imag"], dtype=np.float64
        ).copy()
        malformed.extra["rcs_amp_imag"][1, 0, 0, 0] = np.nan

        # The malformed producer field is not authoritative even before a
        # transform; valid display samples continue to reconstruct normally.
        self.assertTrue(np.isfinite(malformed.rcs).all())
        for transformed in (
            malformed.wrap_phase("0_360"),
            malformed.axis_crop(azimuths=[1.0]),
        ):
            with self.subTest(shape=transformed.rcs_power.shape):
                self.assertNotIn("rcs_amp_real", transformed.extra)
                self.assertNotIn("rcs_amp_imag", transformed.extra)
                self.assertNotIn(
                    "raw_complex_amplitude_preserved", transformed.extra
                )
                self.assertTrue(np.isfinite(transformed.rcs).all())

    def test_exact_transforms_retain_assembly_role_and_provenance(self):
        source = self._raw_solver_grid(
            [1.0, 1.0],
            azimuths=[0.0, 10.0],
            frequencies=[1.0],
            role="features_only_delta",
        )
        cropped = source.axis_crop(azimuths=[10.0])
        self.assertEqual(
            cropped.extra["assembly_response_role"], "features_only_delta"
        )
        self.assertEqual(cropped.extra["assembly_base_sha256"], "a" * 64)
        self.assertEqual(
            cropped.extra["assembly_base_response_sha256"], "b" * 64
        )
        self.assertIn("feature_provenance_json", cropped.extra)
        np.testing.assert_array_equal(
            cropped.extra["rcs_amp_real"].ravel(), [1.0]
        )

        shifted = cropped.shift_azimuth(5.0)
        self.assertEqual(
            shifted.extra["assembly_response_role"], "features_only_delta"
        )
        self.assertEqual(shifted.extra["assembly_base_sha256"], "a" * 64)
        self.assertIn("feature_provenance_json", shifted.extra)
        self.assertNotIn("assembly_base_response_sha256", shifted.extra)
        self.assertEqual(
            shifted.extra["assembly_source_base_response_sha256"], "b" * 64
        )

    def test_align_intersect_preserves_raw_field_and_invalidates_relabels(self):
        source = self._raw_solver_grid(
            [1.0, 1.0 + 4.0e-10],
            azimuths=[0.0, 2.0],
            frequencies=[1.0],
            role="features_only_delta",
        )
        exact_target = self._raw_solver_grid(
            [1.0], azimuths=[2.0], frequencies=[1.0]
        )
        exact = source.align_to(exact_target, mode="intersect")
        np.testing.assert_array_equal(exact.extra["rcs_amp_real"], [[[[1.0 + 4.0e-10]]]])
        self.assertEqual(
            exact.extra["assembly_base_response_sha256"], "b" * 64
        )

        relabeled_target = self._raw_solver_grid(
            [1.0], azimuths=[2.0 + 5.0e-7], frequencies=[1.0]
        )
        relabeled = source.align_to(relabeled_target, mode="intersect")
        np.testing.assert_array_equal(relabeled.azimuths, [2.0 + 5.0e-7])
        np.testing.assert_array_equal(
            relabeled.extra["rcs_amp_real"], [[[[1.0 + 4.0e-10]]]]
        )
        self.assertEqual(
            relabeled.extra["assembly_response_role"], "features_only_delta"
        )
        self.assertNotIn("assembly_base_response_sha256", relabeled.extra)
        self.assertEqual(
            relabeled.extra["assembly_source_base_response_sha256"], "b" * 64
        )

    def test_nonexact_operations_cannot_impersonate_reusable_feature_delta(self):
        source = self._raw_solver_grid(
            [1.0, 1.0],
            azimuths=[0.0, 10.0],
            frequencies=[1.0],
            role="features_only_delta",
        )
        other = self._raw_solver_grid(
            [0.5, 0.5],
            azimuths=[0.0, 10.0],
            frequencies=[1.0],
        )
        results = (
            source.incoherent_add(other),
            source.statistics_dataset(
                "mean", axes=["azimuth"], domain="magnitude"
            ),
        )
        for result in results:
            with self.subTest(shape=result.rcs_power.shape):
                self.assertEqual(
                    result.extra["assembly_response_role"],
                    "incoherent_power_sum",
                )
                self.assertEqual(result.extra["combine_role"], "power")
                self.assertEqual(
                    result.extra["assembly_base_sha256"], "a" * 64
                )
                self.assertIn("feature_provenance_json", result.extra)
                self.assertNotIn(
                    "assembly_base_response_sha256", result.extra
                )

        coherent_mixed_role = source.coherent_add(other)
        self.assertEqual(
            coherent_mixed_role.extra["assembly_response_role"],
            "coherent_field_sum",
        )
        self.assertNotEqual(
            coherent_mixed_role.extra["assembly_response_role"],
            "features_only_delta",
        )

    def test_join_rejects_coordinate_aliases_instead_of_misplacing_samples(self):
        aliased = self._indexed_grid(
            [0.0, 5.0e-7, 1.0], [0.0], [1.0], ["VV"], 10.0
        )
        separate = self._indexed_grid(
            [2.0], [0.0], [1.0], ["VV"], 40.0
        )
        with self.assertRaisesRegex(
            ValueError, "input azimuth axis.*collapse.*tolerance"
        ):
            RcsGrid.join_many(aliased, separate)

    def test_elevation_pair_combine_rejects_azimuth_aliases(self):
        grid = self._indexed_grid(
            [0.0, 5.0e-7, 90.0], [-1.0, 1.0], [1.0], ["VV"], 1.0
        )
        with self.assertRaisesRegex(
            ValueError, "input azimuth axis contains coordinates closer"
        ):
            grid.combine_elevation_pair_to_azimuth_360()

    def test_join_tiles_polarization_and_reserves_ownership_bypass(self):
        polarizations = [f"P{i}" for i in range(5)]
        power = np.arange(1.0, 6.0).reshape(1, 1, 1, 5)
        grid = RcsGrid(
            [0.0], [0.0], [1.0], polarizations,
            rcs_power=power,
            rcs_phase=np.zeros_like(power),
            units={"frequency": "GHz"},
        )
        with mock.patch.object(grim_dataset, "_JOIN_MERGE_BLOCK_CELLS", 2):
            joined = RcsGrid.join_many(grid, max_output_bytes=10_000)
        np.testing.assert_array_equal(joined.rcs_power, power)

        with self.assertRaisesRegex(ValueError, "reserved for internal"):
            RcsGrid(
                [0.0], [0.0], [1.0], ["VV"],
                rcs_power=np.asarray([[-1.0]]).reshape(1, 1, 1, 1),
                _adopt_clean_arrays=True,
            )

    @staticmethod
    def _indexed_grid(azimuths, elevations, frequencies, polarizations, offset):
        shape = (
            len(azimuths),
            len(elevations),
            len(frequencies),
            len(polarizations),
        )
        power = offset + np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        return RcsGrid(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=power / 1000.0,
            units={"frequency": "GHz"},
        )

    def test_overlap_many_uses_common_ranges_across_every_grid(self):
        grids = [
            self._indexed_grid(
                [0.0, 1.0, 2.0, 3.0],
                [-1.0, 0.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
                ["VV", "HH"],
                1000.0,
            ),
            self._indexed_grid(
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0],
                [2.0, 3.0, 4.0, 5.0],
                ["HH", "VV"],
                2000.0,
            ),
            self._indexed_grid(
                [2.0, 3.0, 4.0, 5.0],
                [-2.0, 0.0, 1.0],
                [3.0, 4.0, 5.0, 6.0],
                ["VV", "HH"],
                3000.0,
            ),
        ]

        outputs = RcsGrid.overlap_many(*grids)
        for source, output in zip(grids, outputs):
            np.testing.assert_array_equal(output.azimuths, [2.0, 3.0])
            np.testing.assert_array_equal(output.elevations, [0.0, 1.0])
            np.testing.assert_array_equal(output.frequencies, [3.0, 4.0])
            np.testing.assert_array_equal(output.polarizations, ["HH", "VV"])

            indices = []
            for source_axis, output_axis in (
                (source.azimuths, output.azimuths),
                (source.elevations, output.elevations),
                (source.frequencies, output.frequencies),
                (source.polarizations, output.polarizations),
            ):
                indices.append(
                    [int(np.flatnonzero(source_axis == value)[0]) for value in output_axis]
                )
            selection = np.ix_(*indices)
            np.testing.assert_array_equal(output.rcs_power, source.rcs_power[selection])
            np.testing.assert_array_equal(output.rcs_phase, source.rcs_phase[selection])

    def test_overlap_many_intersects_finite_cells_across_every_grid(self):
        grids = [
            self._indexed_grid([0.0, 1.0], [0.0], [1.0, 2.0], ["HH", "VV"], offset)
            for offset in (100.0, 200.0, 300.0)
        ]
        grids[1].rcs_power[0, 0, 0, 0] = np.nan
        grids[1].rcs_phase[0, 0, 0, 0] = np.nan
        grids[2].rcs_power[:, :, 1, :] = np.nan
        grids[2].rcs_phase[:, :, 1, :] = np.nan

        outputs = RcsGrid.overlap_many(*grids)
        for output in outputs:
            np.testing.assert_array_equal(output.frequencies, [1.0])
            self.assertTrue(np.isnan(output.rcs_power[0, 0, 0, 0]))
            self.assertTrue(np.isnan(output.rcs_phase[0, 0, 0, 0]))
            self.assertTrue(np.isfinite(output.rcs_power[0, 0, 0, 1]))
            self.assertTrue(np.isfinite(output.rcs_power[1, 0, 0, 0]))

    def test_overlap_many_tolerance_and_labels_are_order_invariant(self):
        grids = (
            self._indexed_grid(
                [2.0, 0.0, 1.0], [0.0], [2.0, 1.0], ["VV", "HH"], 1000.0
            ),
            self._indexed_grid(
                [1.0000002, 2.0000002, 0.0000002],
                [0.0000002],
                [1.0000002, 2.0000002],
                ["HH", "VV"],
                2000.0,
            ),
            self._indexed_grid(
                [0.0000008, 1.0000008, 2.0000008],
                [0.0000008],
                [2.0000008, 1.0000008],
                ["VV", "HH"],
                3000.0,
            ),
        )
        baseline_outputs = RcsGrid.overlap_many(*grids, tol=1.0e-6)
        baseline_by_grid = {
            id(grid): output for grid, output in zip(grids, baseline_outputs)
        }

        for ordered_grids in permutations(grids):
            with self.subTest(order=tuple(id(grid) for grid in ordered_grids)):
                outputs = RcsGrid.overlap_many(*ordered_grids, tol=1.0e-6)
                for source, output in zip(ordered_grids, outputs):
                    baseline = baseline_by_grid[id(source)]
                    np.testing.assert_array_equal(output.azimuths, [0.0, 1.0, 2.0])
                    np.testing.assert_array_equal(output.elevations, [0.0])
                    np.testing.assert_array_equal(output.frequencies, [1.0, 2.0])
                    np.testing.assert_array_equal(output.polarizations, ["HH", "VV"])
                    np.testing.assert_array_equal(output.azimuths, baseline.azimuths)
                    np.testing.assert_array_equal(output.elevations, baseline.elevations)
                    np.testing.assert_array_equal(output.frequencies, baseline.frequencies)
                    np.testing.assert_array_equal(output.polarizations, baseline.polarizations)
                    np.testing.assert_array_equal(output.rcs_power, baseline.rcs_power)
                    np.testing.assert_array_equal(output.rcs_phase, baseline.rcs_phase)

    def test_common_axis_alignment_does_not_chain_or_reuse_samples(self):
        chained = ([0.0], [0.0000009], [0.0000018])
        for ordered_axes in permutations(chained):
            common, _indices = RcsGrid._common_axis_alignment(
                ordered_axes, tol=1.0e-6
            )
            self.assertEqual(common.size, 0)

        common, indices = RcsGrid._common_axis_alignment(
            ([0.0, 0.5], [0.25]), tol=1.0
        )
        np.testing.assert_array_equal(common, [0.0])
        self.assertEqual(indices, [[0], [0]])

    def test_azimuth_range_db_uses_amplitude_scaling(self):
        grid = self._single_sample()
        displayed = _range_display_values(
            grid, np.asarray([1.0, 0.1]), linear=False
        )
        np.testing.assert_allclose(displayed, [0.0, -20.0], atol=1.0e-12)

    def test_pio_converts_units_and_round_trips_elevation(self):
        field = np.asarray(
            [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
            dtype=np.complex128,
        )[:, np.newaxis, :, np.newaxis]
        source = RcsGrid(
            [0.0, np.pi / 2.0],
            [np.pi / 6.0],
            [1_000.0, 1_250.0],
            ["VH"],
            rcs=field,
            units={
                "azimuth": "rad",
                "elevation": "radians",
                "frequency": "MHz",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = source.save_pio(
                os.path.join(temp_dir, "roundtrip.pio"), precision="double"
            )
            restored = RcsGrid.load_pio(path)

        np.testing.assert_allclose(restored.azimuths, [0.0, 90.0])
        np.testing.assert_allclose(restored.elevations, [30.0])
        np.testing.assert_allclose(restored.frequencies, [1.0, 1.25])
        np.testing.assert_allclose(restored.rcs, source.rcs, rtol=0.0, atol=1.0e-14)
        self.assertEqual(restored.units["azimuth"], "deg")
        self.assertEqual(restored.units["elevation"], "deg")
        self.assertEqual(restored.units["frequency"], "GHz")

    def test_pio_rejects_unknown_units_on_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for key, value, message in (
                ("azimuth", "grad", "azimuth unit"),
                ("elevation", "grad", "elevation unit"),
                ("frequency", "THz", "frequency unit"),
            ):
                with self.subTest(save_unit=key):
                    units = {
                        "azimuth": "deg",
                        "elevation": "deg",
                        "frequency": "GHz",
                    }
                    units[key] = value
                    bad_source = RcsGrid(
                        [0.0],
                        [0.0],
                        [1.0],
                        ["VV"],
                        rcs=np.ones((1, 1, 1, 1), dtype=np.complex64),
                        units=units,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        bad_source.save_pio(
                            os.path.join(temp_dir, f"bad-save-{key}.pio")
                        )

            good = self._single_sample()
            path = good.save_pio(os.path.join(temp_dir, "bad-load.pio"))
            with open(path, "rb") as stream:
                payload = stream.read()
            for old, new, message in (
                (b"XUnits=deg", b"XUnits=BAD", "azimuth unit"),
                (b"YUnits=GHz", b"YUnits=BAD", "frequency unit"),
                (b"ElevationUnits=deg", b"ElevationUnits=BAD", "elevation unit"),
            ):
                with self.subTest(load_unit=old.decode("ascii")):
                    self.assertIn(old, payload)
                    bad_payload = payload.replace(old, new)
                    with open(path, "wb") as stream:
                        stream.write(bad_payload)
                    with self.assertRaisesRegex(ValueError, message):
                        RcsGrid.load_pio(path)

    def test_pio_rejects_sigma_2d_instead_of_relabeling_it_sigma_3d(self):
        source = self._single_sample()
        source.units.update(
            {"rcs_log_unit": "dBke", "rcs_linear_quantity": "sigma_2d"}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "invalid-2d.pio")
            with self.assertRaisesRegex(ValueError, "requires a sigma_3d"):
                source.save_pio(path)
            self.assertFalse(os.path.exists(path))

    def test_pio_failed_publication_preserves_existing_file(self):
        source = self._single_sample()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "existing.pio")
            with open(path, "wb") as stream:
                stream.write(b"original-pio")
            with mock.patch.object(
                grim_dataset.os, "replace", side_effect=OSError("disk failure")
            ):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    source.save_pio(path)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), b"original-pio")
            self.assertFalse(
                any(name.startswith(".pio-write-") for name in os.listdir(temp_dir))
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
