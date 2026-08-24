"""Regression tests for the central dataset-format dispatch registry."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from unittest import mock

from grim_dataset import GRIM_GC_CONVENTION, RcsGrid
from grim_headless import (
    SUPPORTED_EXTENSIONS,
    is_supported_path,
    load_dataset,
    load_flat_csv,
    read_CST,
)


class DatasetFormatDispatchTest(unittest.TestCase):
    def test_ptm_and_cst_data_are_in_the_canonical_registry(self) -> None:
        self.assertIn(".ptm", SUPPORTED_EXTENSIONS)
        self.assertIn(".cst_data", SUPPORTED_EXTENSIONS)
        self.assertTrue(is_supported_path("CUT.PTM"))
        self.assertTrue(is_supported_path("CUT.CST_DATA"))
        self.assertFalse(is_supported_path("CUT.PTM.docx"))

    def test_ptm_dispatches_to_the_binary_reader(self) -> None:
        sentinel = object()
        with mock.patch.object(
            RcsGrid, "load_ptm", return_value=sentinel
        ) as loader:
            self.assertIs(load_dataset("example.ptm"), sentinel)
        loader.assert_called_once_with("example.ptm")

    def test_named_cst_entry_and_dispatch_share_one_reader(self) -> None:
        sentinel = object()
        with mock.patch.object(
            RcsGrid, "read_CST", return_value=sentinel
        ) as loader:
            self.assertIs(read_CST("example.cst_data"), sentinel)
            self.assertIs(load_dataset("example.cst_data"), sentinel)
        self.assertEqual(
            loader.call_args_list,
            [
                mock.call("example.cst_data"),
                mock.call("example.cst_data"),
            ],
        )

    def test_headless_flat_csv_preserves_angular_coordinate_system(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "great_circle.csv")
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "azimuth",
                        "elevation",
                        "frequency",
                        "frequency_unit",
                        "polarization",
                        "rcs_log_unit",
                        "angular_coordinate_system",
                        "great_circle_coordinate_convention",
                        "angular_roll_deg",
                        "angular_tilt_deg",
                        "magnitude_linear",
                        "phase_deg",
                    ]
                )
                writer.writerow(
                    [
                        0.0, 5.0, 10.0, "GHz", "VV", "dBsm",
                        "great_circle", GRIM_GC_CONVENTION,
                        12.5, -1.0, 1.0, 0.0,
                    ]
                )

            grid = load_flat_csv(path)

        self.assertEqual(grid.angular_coordinate_system(), "great_circle")
        self.assertEqual(
            grid.great_circle_coordinate_convention(), GRIM_GC_CONVENTION
        )
        self.assertEqual(grid.angular_frame_orientation_deg(), (12.5, -1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
