#!/usr/bin/env python3
"""Acceptance tests for the strict point-feature placement workflow."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import place_features  # noqa: E402
from frame import to_axis_frame  # noqa: E402


HEADER = (
    "placement_id,dataset_id,x,y,z,nx,ny,nz,"
    "roll_x,roll_y,roll_z\n"
)


class _FlatSurface:
    def distance(self, points):
        return np.zeros(len(np.atleast_2d(points)), dtype=float)

    def normal(self, points):
        normal = to_axis_frame([0.0, 0.0, 1.0])
        return np.tile(normal, (len(np.atleast_2d(points)), 1))


class StrictPointCsvTests(unittest.TestCase):
    def _write(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_exact_schema_and_unique_ids_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self._write(
                directory,
                "valid.csv",
                HEADER + "fastener_001,fastener,1,2,3,0,0,1,1,0,0\n",
            )
            rows = place_features._placement_rows(valid)
            self.assertEqual(rows[0]["placement_id"], "fastener_001")
            self.assertEqual(rows[0]["dataset_id"], "fastener")

            old_dialect = self._write(
                directory, "old.csv", "x,y,z\n1,2,3\n"
            )
            with self.assertRaisesRegex(ValueError, "header must be exactly"):
                place_features._placement_rows(old_dialect)

            duplicate = self._write(
                directory,
                "duplicate.csv",
                HEADER
                + "same,fastener,1,2,3,0,0,1,1,0,0\n"
                + "same,fastener,4,5,6,0,0,1,1,0,0\n",
            )
            with self.assertRaisesRegex(ValueError, "duplicate placement_id"):
                place_features._placement_rows(duplicate)

    def test_nonfinite_and_extra_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            nonfinite = self._write(
                directory,
                "nonfinite.csv",
                HEADER + "bad,fastener,nan,2,3,0,0,1,1,0,0\n",
            )
            with self.assertRaisesRegex(ValueError, "NaN/infinite"):
                place_features._placement_rows(nonfinite)

            extra = self._write(
                directory,
                "extra.csv",
                HEADER + "bad,fastener,1,2,3,0,0,1,1,0,0,unused\n",
            )
            with self.assertRaisesRegex(ValueError, "exactly 11 columns"):
                place_features._placement_rows(extra)


class PointDatasetMappingTests(unittest.TestCase):
    def test_one_csv_places_repeated_fasteners_and_one_antenna(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fastener = root / "fastener.grim"
            antenna = root / "antenna.grim"
            fastener.touch()
            antenna.touch()
            placements = root / "points.csv"
            placements.write_text(
                HEADER
                + "fastener_001,fastener,1,2,3,0,0,1,1,0,0\n"
                + "fastener_002,fastener,4,5,6,0,0,1,1,0,0\n"
                + "antenna_001,antenna,7,8,9,0,0,1,1,0,0\n",
                encoding="utf-8",
            )

            loaded = {}

            def prepare(path, **settings):
                self.assertEqual(settings, {
                    "declared_coherent_delta": True,
                    "delta_sign": 1.0,
                    "assume_missing_cross_pol_zero": False,
                })
                loaded[path] = loaded.get(path, 0) + 1
                return "prepared:" + Path(path).name

            with (
                mock.patch.object(
                    place_features, "POINT_FEATURE_LOCATIONS_CSV", placements
                ),
                mock.patch.object(
                    place_features,
                    "POINT_FEATURE_DATASETS",
                    {"fastener": fastener, "antenna": antenna},
                ),
                mock.patch.object(
                    place_features, "prepare_point_pattern", side_effect=prepare
                ),
            ):
                points, records = place_features._compact_points(
                    None, _FlatSurface(), 1.0, 1.0e-3, 1.0
                )

            self.assertEqual(len(points), 3)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(all(count == 1 for count in loaded.values()))
            self.assertEqual(
                [record["placement_id"] for record in records],
                ["fastener_001", "fastener_002", "antenna_001"],
            )
            self.assertEqual(
                [record["dataset_id"] for record in records],
                ["fastener", "fastener", "antenna"],
            )
            np.testing.assert_allclose(
                points[0]["location"], to_axis_frame([1.0, 2.0, 3.0])
            )
            np.testing.assert_allclose(
                points[0]["aperture_normal"],
                to_axis_frame([0.0, 0.0, 1.0]),
            )
            np.testing.assert_allclose(
                points[0]["roll_ref"], to_axis_frame([1.0, 0.0, 0.0])
            )
            self.assertTrue(all(
                record["input_subtraction_order"]
                == "OPN-FRD (featured-clean)"
                for record in records
            ))

    def test_unknown_dataset_and_parallel_roll_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fastener = root / "fastener.grim"
            fastener.touch()
            unknown = root / "unknown.csv"
            unknown.write_text(
                HEADER + "p1,antenna,1,2,3,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    place_features, "POINT_FEATURE_LOCATIONS_CSV", unknown
                ),
                mock.patch.object(
                    place_features,
                    "POINT_FEATURE_DATASETS",
                    {"fastener": fastener},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unknown dataset_id"):
                    place_features._compact_points(
                        None, _FlatSurface(), 1.0, 1.0e-3, 1.0
                    )

            parallel = root / "parallel.csv"
            parallel.write_text(
                HEADER + "p1,fastener,1,2,3,0,0,1,0,0,2\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    place_features, "POINT_FEATURE_LOCATIONS_CSV", parallel
                ),
                mock.patch.object(
                    place_features,
                    "POINT_FEATURE_DATASETS",
                    {"fastener": fastener},
                ),
                mock.patch.object(
                    place_features,
                    "prepare_point_pattern",
                    return_value="prepared",
                ),
            ):
                with self.assertRaisesRegex(ValueError, "parallel"):
                    place_features._compact_points(
                        None, _FlatSurface(), 1.0, 1.0e-3, 1.0
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
