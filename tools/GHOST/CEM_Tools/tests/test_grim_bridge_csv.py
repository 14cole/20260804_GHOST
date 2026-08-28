"""Flat CSV interoperability regressions shared with the GRIM Plotting tab."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from cem_tools.grim_bridge import (
    _rcs_grid_class,
    export_dataset,
    grim_project_path,
    load_dataset,
)


class GrimBridgeCsvTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cem_export_uses_versioned_power_schema_and_round_trips_metadata(self):
        grid_class = _rcs_grid_class()
        grid = grid_class(
            [0.25],
            [-0.5],
            [3.0e9],
            ["VV"],
            rcs_power=np.asarray([4.0]).reshape(1, 1, 1, 1),
            rcs_phase=np.asarray([np.pi / 2.0]).reshape(1, 1, 1, 1),
            units={
                "azimuth": "rad",
                "elevation": "rad",
                "frequency": "Hz",
                "rcs_linear_quantity": "sigma_3d",
                "rcs_log_unit": "dBsm",
                "angular_coordinate_system": "conic",
                "polarization_basis": "test spherical basis",
                "time_convention": "exp(+j omega t)",
            },
            extra={"phase_reference": "origin=(1, 2, 3)"},
        )
        output = export_dataset(grid, self.root / "exchange", ".csv")[0]
        with output.open("r", newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))

        self.assertEqual(row["grim_csv_schema"], "grim.flat-rcs.v1")
        self.assertEqual(row["magnitude_power_linear"], "4")
        self.assertNotIn("magnitude_linear", row)
        self.assertEqual(row["azimuth_unit"], "rad")
        self.assertEqual(row["frequency_unit"], "Hz")
        self.assertEqual(row["phase_reference"], "origin=(1, 2, 3)")

        loaded = load_dataset(output)
        self.assertEqual(float(loaded.rcs_power.item()), 4.0)
        self.assertAlmostEqual(float(loaded.rcs_phase.item()), np.pi / 2.0)
        self.assertEqual(loaded.units["azimuth"], "rad")
        self.assertEqual(loaded.units["frequency"], "Hz")
        self.assertEqual(loaded.units["polarization_basis"], "test spherical basis")
        self.assertEqual(loaded.units["time_convention"], "exp(+j omega t)")
        self.assertEqual(loaded.extra["phase_reference"], "origin=(1, 2, 3)")

        project_path = str(grim_project_path())
        sys.path.insert(0, project_path)
        try:
            from grim_headless import load_dataset as load_plotting_dataset

            plotting_grid = load_plotting_dataset(str(output))
        finally:
            sys.path.remove(project_path)
        self.assertEqual(float(plotting_grid.rcs_power.item()), 4.0)
        self.assertEqual(plotting_grid.units["azimuth"], "rad")
        self.assertEqual(
            plotting_grid.extra["phase_reference"], "origin=(1, 2, 3)"
        )

    def test_old_cem_amplitude_table_keeps_legacy_square_semantics(self):
        path = self.root / "legacy.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "azimuth_deg", "elevation_deg", "frequency_GHz",
                "polarization", "magnitude_linear", "phase_deg",
            ])
            writer.writerow([0.0, 0.0, 3.0, "HH", 3.0, 30.0])

        loaded = load_dataset(path)

        self.assertEqual(float(loaded.rcs_power.item()), 9.0)
        self.assertAlmostEqual(float(np.degrees(loaded.rcs_phase.item())), 30.0)
        self.assertEqual(loaded.extra["flat_csv_schema"], "legacy_cem_amplitude")


if __name__ == "__main__":
    unittest.main(verbosity=2)
