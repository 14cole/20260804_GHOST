"""Regression coverage for spreadsheet-compatible RCS CSV export."""

import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from grim_cut_dataset_mixin import _load_dataset_csv, _write_dataset_csv
from grim_dataset import GRIM_GC_CONVENTION, RcsGrid


class TestCsvExport(unittest.TestCase):
    def setUp(self):
        descriptor, self.path = tempfile.mkstemp(suffix=".csv")
        os.close(descriptor)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _rows(self):
        with open(self.path, newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_power_only_magnitude_is_not_lost_when_phase_is_unknown(self):
        power = np.asarray([1.0, 4.0], dtype=np.float32).reshape(2, 1, 1, 1)
        dataset = RcsGrid(
            [0.0, 90.0],
            [0.0],
            [3.0],
            ["VV"],
            rcs_power=power,
            units={"frequency": "GHz"},
        )

        _write_dataset_csv(
            dataset, self.path, scale="both", include_phase=True
        )
        rows = self._rows()

        self.assertEqual(
            [float(row["magnitude_power_linear"]) for row in rows], [1.0, 4.0]
        )
        self.assertAlmostEqual(float(rows[0]["magnitude_dbsm"]), 0.0, places=6)
        self.assertAlmostEqual(
            float(rows[1]["magnitude_dbsm"]), 6.020600, places=6
        )
        self.assertEqual([row["phase_deg"] for row in rows], ["", ""])
        self.assertEqual(
            {row["frequency_unit"] for row in rows}, {"GHz"}
        )
        self.assertEqual(
            {row["rcs_log_unit"] for row in rows}, {"dBsm"}
        )
        self.assertEqual(
            {row["angular_coordinate_system"] for row in rows}, {"conic"}
        )
        self.assertEqual({row["angular_roll_deg"] for row in rows}, {"0"})
        self.assertEqual({row["angular_tilt_deg"] for row in rows}, {"0"})

        loaded = _load_dataset_csv(self.path)
        np.testing.assert_allclose(loaded.rcs_power, power)
        self.assertTrue(np.isnan(loaded.rcs_phase).all())
        self.assertEqual(loaded.units["frequency"], "GHz")
        self.assertEqual(loaded.angular_coordinate_system(), "conic")

    def test_great_circle_coordinate_tag_survives_csv_round_trip(self):
        dataset = RcsGrid(
            [-10.0, 10.0],
            [5.0],
            [3.0],
            ["VV"],
            rcs=np.ones((2, 1, 1, 1), dtype=np.complex64),
            units={
                "frequency": "GHz",
                "angular_coordinate_system": "great_circle",
                "great_circle_coordinate_convention": GRIM_GC_CONVENTION,
                "angular_roll_deg": 12.5,
                "angular_tilt_deg": -1.0,
            },
        )

        _write_dataset_csv(dataset, self.path, scale="linear", include_phase=True)
        loaded = _load_dataset_csv(self.path)

        self.assertEqual(loaded.angular_coordinate_system(), "great_circle")
        self.assertEqual(
            loaded.great_circle_coordinate_convention(), GRIM_GC_CONVENTION
        )
        self.assertEqual(loaded.angular_frame_orientation_deg(), (12.5, -1.0))
        self.assertEqual(
            {row["angular_coordinate_system"] for row in self._rows()},
            {"great_circle"},
        )
        self.assertEqual(
            {
                row["great_circle_coordinate_convention"]
                for row in self._rows()
            },
            {GRIM_GC_CONVENTION},
        )

    def test_statistics_output_exports_finite_magnitude(self):
        field = np.asarray(
            [1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64
        ).reshape(2, 1, 1, 1)
        source = RcsGrid(
            [0.0, 90.0],
            [0.0],
            [3.0],
            ["VV"],
            rcs=field,
            units={"frequency": "GHz"},
        )
        statistic = source.statistics_dataset(
            statistic="mean",
            axes=["azimuth"],
            domain="magnitude",
            broadcast_reduced=True,
        )
        self.assertTrue(np.isnan(statistic.rcs_phase).all())

        _write_dataset_csv(statistic, self.path, scale="linear")
        values = [
            float(row["magnitude_power_linear"]) for row in self._rows()
        ]
        self.assertEqual(values, [2.5, 2.5])

    def test_known_phase_round_trip_is_preserved(self):
        field = np.asarray(
            [1.0 + 0.0j, 0.0 + 2.0j], dtype=np.complex64
        ).reshape(2, 1, 1, 1)
        dataset = RcsGrid(
            [0.0, 90.0],
            [0.0],
            [3.0],
            ["HH"],
            rcs=field,
            units={"frequency": "GHz"},
        )

        _write_dataset_csv(
            dataset, self.path, scale="both", include_phase=True
        )
        loaded = _load_dataset_csv(self.path)

        np.testing.assert_allclose(
            loaded.rcs_power.ravel(), [1.0, 4.0], rtol=1.0e-6
        )
        np.testing.assert_allclose(
            np.degrees(loaded.rcs_phase.ravel()),
            [0.0, 90.0],
            atol=1.0e-5,
        )

    def test_missing_magnitude_stays_localized(self):
        power = np.asarray([1.0, np.nan], dtype=np.float32).reshape(
            2, 1, 1, 1
        )
        dataset = RcsGrid(
            [0.0, 90.0],
            [0.0],
            [3.0],
            ["VV"],
            rcs_power=power,
            units={"frequency": "GHz"},
        )

        _write_dataset_csv(dataset, self.path, scale="linear")
        rows = self._rows()
        self.assertEqual(rows[0]["magnitude_power_linear"], "1")
        self.assertEqual(rows[1]["magnitude_power_linear"], "")

        loaded = _load_dataset_csv(self.path)
        self.assertEqual(float(loaded.rcs_power.ravel()[0]), 1.0)
        self.assertTrue(np.isnan(loaded.rcs_power.ravel()[1]))

    def test_frequency_units_survive_dbke_round_trip(self):
        cases = {
            "GHz": 3.0,
            "MHz": 3.0e3,
            "kHz": 3.0e6,
            "Hz": 3.0e9,
        }
        for unit, frequency in cases.items():
            with self.subTest(unit=unit):
                dataset = RcsGrid(
                    [0.0],
                    [0.0],
                    [frequency],
                    ["VV"],
                    rcs_power=np.ones((1, 1, 1, 1), dtype=np.float32),
                    units={
                        "frequency": unit,
                        "rcs_log_unit": "dBke",
                    },
                )
                _write_dataset_csv(dataset, self.path, scale="dbke")
                loaded = _load_dataset_csv(self.path)
                self.assertEqual(loaded.units["frequency"], unit)
                self.assertEqual(loaded.units["rcs_log_unit"], "dBke")
                self.assertAlmostEqual(
                    float(loaded.frequencies[0]), frequency
                )
                self.assertAlmostEqual(
                    float(loaded.rcs_power.ravel()[0]), 1.0, places=5
                )

    def test_export_refuses_log_units_that_mislabel_physical_quantity(self):
        sigma_3d = RcsGrid(
            [0.0], [0.0], [3.0], ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            units={
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
        )
        with self.assertRaisesRegex(ValueError, "cannot be labeled dBke"):
            _write_dataset_csv(sigma_3d, self.path, scale="dbke")

        sigma_2d = RcsGrid(
            [0.0], [0.0], [3.0], ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            units={
                "frequency": "GHz",
                "rcs_log_unit": "dBke",
                "rcs_linear_quantity": "sigma_2d",
            },
        )
        with self.assertRaisesRegex(ValueError, "cannot be labeled dBsm"):
            _write_dataset_csv(sigma_2d, self.path, scale="dbsm")
        _write_dataset_csv(sigma_2d, self.path, scale="both")
        row = self._rows()[0]
        self.assertIn("magnitude_dbke", row)
        self.assertNotIn("magnitude_dbsm", row)

    def test_legacy_csv_without_unit_column_is_still_supported(self):
        with open(self.path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "azimuth",
                "elevation",
                "frequency",
                "polarization",
                "magnitude_linear",
            ])
            writer.writerow([0.0, 0.0, 3.0e9, "VV", 2.0])

        loaded = _load_dataset_csv(self.path)
        self.assertEqual(loaded.units["frequency"], "Hz")
        self.assertEqual(float(loaded.rcs_power.ravel()[0]), 2.0)

    def test_failed_csv_write_keeps_existing_target_intact(self):
        with open(self.path, "w", encoding="utf-8") as stream:
            stream.write("existing artifact\n")
        dataset = RcsGrid(
            [0.0], [0.0], [3.0], ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            units={"frequency": "GHz"},
        )

        def fail_after_partial(_dataset, stage_path, **_kwargs):
            with open(stage_path, "w", encoding="utf-8") as stream:
                stream.write("partial replacement\n")
            raise OSError("simulated CSV failure")

        with mock.patch(
            "grim_cut_dataset_mixin.write_flat_csv",
            side_effect=fail_after_partial,
        ):
            with self.assertRaisesRegex(OSError, "simulated CSV failure"):
                _write_dataset_csv(dataset, self.path)

        with open(self.path, encoding="utf-8") as stream:
            self.assertEqual(stream.read(), "existing artifact\n")
        self.assertFalse(
            any(
                name.startswith(".grim-csv-")
                for name in os.listdir(os.path.dirname(self.path))
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
