"""Focused regressions for dataset transforms and Pioneer interchange."""

from __future__ import annotations

import os
import tempfile
import unittest
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
