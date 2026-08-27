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
