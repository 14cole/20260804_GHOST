"""Regression coverage for the shared versioned flat RCS CSV contract."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import grim_csv_schema as flat_schema
from grim_csv_schema import FLAT_CSV_SCHEMA, write_flat_csv
from grim_dataset import RcsGrid
from grim_headless import load_dataset, load_flat_csv


V1_HEADER = [
    "grim_csv_schema",
    "azimuth",
    "azimuth_unit",
    "elevation",
    "elevation_unit",
    "frequency",
    "frequency_unit",
    "polarization",
    "rcs_linear_quantity",
    "rcs_log_unit",
    "angular_coordinate_system",
    "great_circle_coordinate_convention",
    "angular_roll_deg",
    "angular_tilt_deg",
    "polarization_basis",
    "time_convention",
    "phase_reference",
    "magnitude_power_linear",
    "magnitude_dbsm",
    "phase_deg",
]


def _v1_row(**updates):
    values = {
        "grim_csv_schema": FLAT_CSV_SCHEMA,
        "azimuth": 0.0,
        "azimuth_unit": "deg",
        "elevation": 0.0,
        "elevation_unit": "deg",
        "frequency": 3.0,
        "frequency_unit": "GHz",
        "polarization": "VV",
        "rcs_linear_quantity": "sigma_3d",
        "rcs_log_unit": "dBsm",
        "angular_coordinate_system": "conic",
        "great_circle_coordinate_convention": "",
        "angular_roll_deg": 0.0,
        "angular_tilt_deg": 0.0,
        "polarization_basis": "grim_conic_spherical_vh_v1",
        "time_convention": "exp(+j omega t)",
        "phase_reference": "origin=(0, 0, 0)",
        "magnitude_power_linear": 4.0,
        "magnitude_dbsm": 6.020599913279624,
        "phase_deg": 90.0,
    }
    values.update(updates)
    return [values[name] for name in V1_HEADER]


class FlatCsvSchemaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _path(self, name="data.csv"):
        return os.path.join(self.temp.name, name)

    def _write(self, header, rows, name="data.csv", delimiter=","):
        path = self._path(name)
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_v1_preserves_angle_units_phase_and_physical_metadata(self):
        path = self._write(
            V1_HEADER,
            [
                _v1_row(
                    azimuth=np.pi / 2.0,
                    azimuth_unit="rad",
                    elevation=np.pi / 4.0,
                    elevation_unit="rad",
                    phase_reference="origin=(1, 2, 3), calibrated",
                )
            ],
        )

        grid = load_dataset(path)

        self.assertAlmostEqual(float(grid.azimuths[0]), np.pi / 2.0)
        self.assertAlmostEqual(float(grid.elevations[0]), np.pi / 4.0)
        self.assertEqual(grid.units["azimuth"], "rad")
        self.assertEqual(grid.units["elevation"], "rad")
        self.assertEqual(grid.units["frequency"], "GHz")
        self.assertEqual(grid.linear_quantity(), "sigma_3d")
        self.assertEqual(grid.default_log_unit(), "dBsm")
        self.assertEqual(
            grid.units["polarization_basis"], "grim_conic_spherical_vh_v1"
        )
        self.assertEqual(grid.units["time_convention"], "exp(+j omega t)")
        self.assertEqual(
            grid.extra["phase_reference"], "origin=(1, 2, 3), calibrated"
        )
        self.assertEqual(grid.extra["flat_csv_schema"], FLAT_CSV_SCHEMA)
        self.assertAlmostEqual(float(grid.rcs_power.item()), 4.0)
        self.assertAlmostEqual(float(np.degrees(grid.rcs_phase.item())), 90.0)

    def test_redundant_magnitude_columns_are_cross_checked(self):
        valid = self._write(V1_HEADER, [_v1_row()])
        self.assertAlmostEqual(float(load_flat_csv(valid).rcs_power.item()), 4.0)

        invalid = self._write(
            V1_HEADER,
            [_v1_row(magnitude_dbsm=3.0)],
            name="conflict.csv",
        )
        with self.assertRaisesRegex(ValueError, "redundant magnitude columns conflict"):
            load_flat_csv(invalid)

    def test_quantity_log_unit_and_column_must_agree(self):
        path = self._write(
            V1_HEADER,
            [_v1_row(rcs_linear_quantity="sigma_2d", rcs_log_unit="dBke")],
        )
        with self.assertRaisesRegex(ValueError, "magnitude_dbsm is incompatible"):
            load_flat_csv(path)

        log_mismatch = self._write(
            V1_HEADER,
            [_v1_row(rcs_linear_quantity="sigma_3d", rcs_log_unit="dBke")],
            name="log_mismatch.csv",
        )
        with self.assertRaisesRegex(ValueError, "sigma_3d requires rcs_log_unit=dBsm"):
            load_flat_csv(log_mismatch)

    def test_legacy_grim_frequency_inference_is_explicit(self):
        path = self._write(
            ["azimuth", "elevation", "frequency", "polarization", "magnitude_linear"],
            [[0.0, 0.0, 3.0e9, "VV", 2.0]],
        )
        grid = load_flat_csv(path)
        self.assertEqual(grid.units["frequency"], "Hz")
        self.assertIs(grid.extra["frequency_unit_inferred"], True)
        self.assertIn("legacy magnitude heuristic", grid.extra["frequency_unit_inference"])
        self.assertIn("WARNING", grid.history)
        self.assertEqual(float(grid.rcs_power.item()), 2.0)

    def test_legacy_cem_amplitude_table_is_supported_deliberately(self):
        path = self._write(
            [
                "azimuth_deg", "elevation_deg", "frequency_GHz",
                "polarization", "magnitude_linear", "phase_deg",
            ],
            [[0.0, 0.0, 3.0, "HH", 2.0, -45.0]],
        )
        grid = load_dataset(path)
        self.assertEqual(grid.units["azimuth"], "deg")
        self.assertEqual(grid.units["frequency"], "GHz")
        self.assertEqual(float(grid.rcs_power.item()), 4.0)
        self.assertAlmostEqual(float(np.degrees(grid.rcs_phase.item())), -45.0)
        self.assertEqual(grid.extra["flat_csv_schema"], "legacy_cem_amplitude")

    def test_dense_cartesian_allocation_is_refused_before_array_creation(self):
        rows = []
        for value in range(4):
            rows.append(_v1_row(
                azimuth=value,
                elevation=value,
                frequency=1.0 + value,
                polarization="P{}".format(value),
                magnitude_dbsm=6.020599913279624,
            ))
        path = self._write(V1_HEADER, rows)
        with mock.patch.dict(
            os.environ, {"GRIM_MAX_CSV_GRID_GB": "0.000001"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "dense grid of 256 cells"):
                load_flat_csv(path)

    def test_dense_preflight_accounts_for_tracker_and_constructor_scratch(self):
        cells = 2 * 3 * 4 * 5
        with mock.patch.object(
            flat_schema,
            "_allocation_budget_bytes",
            return_value=(10**9, "test budget"),
        ):
            actual_cells, dense_payload, estimated_peak = (
                flat_schema._preflight_dense_grid((2, 3, 4, 5), 1)
            )
        self.assertEqual(actual_cells, cells)
        self.assertEqual(dense_payload, cells * 16)
        # Two input float64 arrays + clean-power float64 + bool mask +
        # float64 gather + float64 ufunc output = 41 bytes per dense cell.
        self.assertEqual(estimated_peak, cells * 41)

    def test_dense_duplicate_tracker_preserves_first_line_diagnostics(self):
        identical = self._write(
            V1_HEADER,
            [_v1_row(), _v1_row()],
            name="identical_duplicate.csv",
        )
        loaded = load_flat_csv(identical)
        self.assertEqual(float(loaded.rcs_power.item()), 4.0)

        conflicting = self._write(
            V1_HEADER,
            [_v1_row(), _v1_row(phase_deg=45.0)],
            name="conflicting_duplicate.csv",
        )
        with self.assertRaisesRegex(
            ValueError,
            "line 3: conflicting duplicate CSV sample; first defined on line 2",
        ):
            load_flat_csv(conflicting)

    def test_v1_rejects_ambiguous_legacy_linear_column(self):
        header = [name for name in V1_HEADER if name != "magnitude_power_linear"]
        header.append("magnitude_linear")
        values = dict(zip(V1_HEADER, _v1_row()))
        row = [values.get(name, 2.0) for name in header]
        path = self._write(header, [row])
        with self.assertRaisesRegex(ValueError, "does not permit ambiguous magnitude_linear"):
            load_flat_csv(path)

    def test_shared_writer_round_trips_units_phase_reference_and_both_scales(self):
        grid = RcsGrid(
            [0.25], [-0.5], [3.0e9], ["VV"],
            rcs_power=np.asarray([4.0]).reshape(1, 1, 1, 1),
            rcs_phase=np.asarray([np.pi / 2.0]).reshape(1, 1, 1, 1),
            units={
                "azimuth": "rad", "elevation": "rad", "frequency": "Hz",
                "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm",
                "angular_coordinate_system": "conic",
                "polarization_basis": "test spherical basis",
                "time_convention": "exp(+j omega t)",
            },
            extra={"phase_reference": "origin=(1, 2, 3)"},
        )
        path = self._path("written.csv")
        write_flat_csv(grid, path, scale="both", include_phase=True)
        with open(path, "r", newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["grim_csv_schema"], FLAT_CSV_SCHEMA)
        self.assertEqual(row["magnitude_power_linear"], "4")
        self.assertAlmostEqual(float(row["magnitude_dbsm"]), 6.020599913279624)
        self.assertNotIn("magnitude_linear", row)

        loaded = load_flat_csv(path)
        self.assertEqual(loaded.units["azimuth"], "rad")
        self.assertEqual(loaded.units["frequency"], "Hz")
        self.assertEqual(float(loaded.rcs_power.item()), 4.0)
        self.assertEqual(loaded.extra["phase_reference"], "origin=(1, 2, 3)")

    def test_writer_preserves_conventions_from_either_metadata_container(self):
        grid = RcsGrid(
            [0.0],
            [0.0],
            [3.0],
            ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_linear_quantity": "sigma_3d",
                "rcs_log_unit": "dBsm",
                "phase_reference": "phase center declared in units",
            },
            extra={
                "time_convention": "exp(+j omega t)",
                "polarization_basis": "basis declared in extra",
            },
        )
        path = self._path("container_metadata.csv")
        write_flat_csv(grid, path)
        with open(path, "r", newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(
            row["phase_reference"], "phase center declared in units"
        )
        self.assertEqual(row["time_convention"], "exp(+j omega t)")
        self.assertEqual(
            row["polarization_basis"], "basis declared in extra"
        )

        restored = load_flat_csv(path)
        self.assertEqual(
            restored.extra["phase_reference"],
            "phase center declared in units",
        )
        self.assertEqual(
            restored.units["time_convention"], "exp(+j omega t)"
        )
        self.assertEqual(
            restored.units["polarization_basis"], "basis declared in extra"
        )

    def test_writer_rejects_conflicting_convention_containers(self):
        for key in (
            "phase_reference",
            "time_convention",
            "polarization_basis",
        ):
            with self.subTest(key=key):
                grid = RcsGrid(
                    [0.0],
                    [0.0],
                    [3.0],
                    ["VV"],
                    rcs_power=np.ones((1, 1, 1, 1)),
                    units={
                        "azimuth": "deg",
                        "elevation": "deg",
                        "frequency": "GHz",
                        "rcs_linear_quantity": "sigma_3d",
                        "rcs_log_unit": "dBsm",
                        key: "left declaration",
                    },
                    extra={key: "right declaration"},
                )
                path = self._path("conflicting_{}.csv".format(key))
                with self.assertRaisesRegex(
                    ValueError, "contradictory {} metadata".format(key)
                ):
                    write_flat_csv(grid, path)
                self.assertFalse(os.path.exists(path))

    def test_shared_writer_dbke_round_trip_uses_declared_frequency_unit(self):
        for unit, frequency in (("GHz", 3.0), ("Hz", 3.0e9)):
            with self.subTest(unit=unit):
                grid = RcsGrid(
                    [0.0], [0.0], [frequency], ["TE"],
                    rcs_power=np.ones((1, 1, 1, 1)),
                    units={
                        "azimuth": "deg", "elevation": "deg",
                        "frequency": unit, "rcs_linear_quantity": "sigma_2d",
                        "rcs_log_unit": "dBke",
                    },
                )
                path = self._path("dbke_{}.csv".format(unit))
                write_flat_csv(grid, path, scale="dbke")
                loaded = load_flat_csv(path)
                self.assertEqual(loaded.units["frequency"], unit)
                self.assertAlmostEqual(float(loaded.rcs_power.item()), 1.0)

    def test_shared_writer_failure_preserves_existing_destination(self):
        grid = RcsGrid(
            [0.0], [0.0], [3.0], ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm",
            },
        )
        # Public arrays remain editable for legacy callers; force a failure
        # after the staged CSV header has been emitted.
        grid.azimuths[0] = np.inf
        path = self._path("existing.csv")
        with open(path, "wb") as stream:
            stream.write(b"trusted prior artifact")

        with self.assertRaisesRegex(ValueError, "positive infinity"):
            write_flat_csv(grid, path)

        with open(path, "rb") as stream:
            self.assertEqual(stream.read(), b"trusted prior artifact")
        self.assertFalse(
            any(name.startswith(".grim-flat-csv-") for name in os.listdir(self.temp.name))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
