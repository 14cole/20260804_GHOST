#!/usr/bin/env python3
"""Acceptance tests for the strict line-feature placement workflow."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import place_features  # noqa: E402
import feature_sum  # noqa: E402
from frame import to_axis_frame  # noqa: E402
from line_expand import SeamCoefficients, expand_perimeter  # noqa: E402


HEADER = (
    "line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,"
    "n1x,n1y,n1z,n2x,n2y,n2z\n"
)


class _FlatSurface:
    def distance(self, points):
        return np.zeros(len(np.atleast_2d(points)), dtype=float)

    def normal(self, points):
        normal = to_axis_frame([0.0, 0.0, 1.0])
        return np.tile(normal, (len(np.atleast_2d(points)), 1))


class StrictLineCsvTests(unittest.TestCase):
    def _write(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_exact_schema_and_ordered_topology_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self._write(
                directory, "valid.csv",
                HEADER
                + "door_1,seam,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "door_1,seam,2,1,0,0,1,1,0,0,0,1,0,0,1\n",
            )
            rows = place_features._line_rows(valid)
            self.assertEqual([row["segment_index"] for row in rows], [1, 2])

            old_dialect = self._write(
                directory, "old.txt", "0 0 0 1 0 0\n"
            )
            with self.assertRaisesRegex(ValueError, "header must be exactly"):
                place_features._line_rows(old_dialect)

            skipped = self._write(
                directory, "skipped.csv",
                HEADER + "door_1,seam,2,0,0,0,1,0,0,0,0,1,0,0,1\n",
            )
            with self.assertRaisesRegex(ValueError, "expected 1"):
                place_features._line_rows(skipped)

            split_group = self._write(
                directory, "split.csv",
                HEADER
                + "door_1,seam,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "door_2,seam,1,0,1,0,1,1,0,0,0,1,0,0,1\n"
                + "door_1,seam,1,1,0,0,2,0,0,0,0,1,0,0,1\n",
            )
            with self.assertRaisesRegex(ValueError, "must be contiguous"):
                place_features._line_rows(split_group)

    def test_nonfinite_extra_columns_and_dataset_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            nonfinite = self._write(
                directory, "nonfinite.csv",
                HEADER + "door_1,seam,1,nan,0,0,1,0,0,0,0,1,0,0,1\n",
            )
            with self.assertRaisesRegex(ValueError, "NaN/infinite"):
                place_features._line_rows(nonfinite)

            extra = self._write(
                directory, "extra.csv",
                HEADER + "door_1,seam,1,0,0,0,1,0,0,0,0,1,0,0,1,unused\n",
            )
            with self.assertRaisesRegex(ValueError, "exactly 15 columns"):
                place_features._line_rows(extra)

            changed = self._write(
                directory, "changed.csv",
                HEADER
                + "door_1,seam_a,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "door_1,seam_b,2,1,0,0,2,0,0,0,0,1,0,0,1\n",
            )
            with self.assertRaisesRegex(ValueError, "same dataset_id"):
                place_features._line_rows(changed)


class LineDatasetMappingTests(unittest.TestCase):
    def test_one_csv_places_repeated_lines_and_multiple_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seam = root / "seam.grim"
            gap = root / "gap.grim"
            seam.touch()
            gap.touch()
            csv_path = root / "lines.csv"
            csv_path.write_text(
                HEADER
                + "door_1,seam,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "door_1,seam,2,1,0,0,1,1,0,0,0,1,0,0,1\n"
                + "door_2,seam,1,2,0,0,3,0,0,0,0,1,0,0,1\n"
                + "panel_1,gap,1,4,0,0,5,0,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    place_features, "LINE_FEATURE_LOCATIONS_CSV", csv_path
                ),
                mock.patch.object(
                    place_features, "LINE_FEATURE_DATASETS",
                    {"seam": seam, "gap": gap},
                ),
            ):
                placements, records = place_features._line_placements(
                    None, _FlatSurface(), 1.0, 1.0e-3, 1.0
                )

            self.assertEqual(len(placements), 3)
            self.assertEqual(
                [record["line_id"] for record in records],
                ["door_1", "door_2", "panel_1"],
            )
            self.assertEqual(
                [record["dataset_id"] for record in records],
                ["seam", "seam", "gap"],
            )
            self.assertEqual([record["segment_count"] for record in records], [2, 1, 1])
            self.assertTrue(all(
                placement["delta_sign"] == 1.0 for placement in placements
            ))
            np.testing.assert_allclose(
                placements[0]["perimeter"][0],
                to_axis_frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            )
            np.testing.assert_allclose(
                placements[0]["segment_normals"][0],
                np.tile(to_axis_frame([0.0, 0.0, 1.0]), (2, 1)),
            )

    def test_disconnected_segments_unknown_dataset_and_bad_normal_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seam = root / "seam.grim"
            seam.touch()
            disconnected = root / "disconnected.csv"
            disconnected.write_text(
                HEADER
                + "door_1,seam,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "door_1,seam,2,2,0,0,3,0,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    place_features, "LINE_FEATURE_LOCATIONS_CSV", disconnected
                ),
                mock.patch.object(
                    place_features, "LINE_FEATURE_DATASETS", {"seam": seam}
                ),
            ):
                with self.assertRaisesRegex(ValueError, "not head-to-tail"):
                    place_features._line_placements(
                        None, _FlatSurface(), 1.0, 1.0e-3, 1.0
                    )

            unknown = root / "unknown.csv"
            unknown.write_text(
                HEADER + "door_1,gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    place_features, "LINE_FEATURE_LOCATIONS_CSV", unknown
                ),
                mock.patch.object(
                    place_features, "LINE_FEATURE_DATASETS", {"seam": seam}
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unknown dataset_id"):
                    place_features._line_placements(
                        None, _FlatSurface(), 1.0, 1.0e-3, 1.0
                    )

            bad_normal = root / "bad_normal.csv"
            bad_normal.write_text(
                HEADER + "door_1,seam,1,0,0,0,1,0,0,0,0,-1,0,0,-1\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    place_features, "LINE_FEATURE_LOCATIONS_CSV", bad_normal
                ),
                mock.patch.object(
                    place_features, "LINE_FEATURE_DATASETS", {"seam": seam}
                ),
            ):
                with self.assertRaisesRegex(ValueError, "outward skin normal"):
                    place_features._line_placements(
                        None, _FlatSurface(), 1.0, 1.0e-3, 1.0
                    )


class EndpointNormalExpansionTests(unittest.TestCase):
    def test_constant_endpoint_normals_match_callable_normals(self):
        coefficients = SeamCoefficients(
            1.0,
            np.asarray([0.0, 90.0, 180.0]),
            np.asarray([1.0 + 0.2j] * 3),
            np.asarray([0.7 - 0.1j] * 3),
        )
        segments = np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]]])
        normals = np.asarray([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        directions = np.asarray([[1.0, 0.0, 0.0]])
        callable_result = expand_perimeter(
            segments, coefficients,
            lambda points: np.tile([1.0, 0.0, 0.0], (len(points), 1)),
            directions, frequency_ghz=1.0,
        )
        endpoint_result = expand_perimeter(
            segments, coefficients, None, directions,
            frequency_ghz=1.0, segment_normals=normals,
        )
        for channel in ("F_vv", "F_hh", "F_vh"):
            np.testing.assert_allclose(
                endpoint_result[channel], callable_result[channel],
                rtol=2.0e-14, atol=2.0e-14,
            )

    def test_repeated_line_instances_prepare_one_coefficient(self):
        source = str(REPO / "same_line_delta.grim")
        placements = [
            {
                "delta": source, "perimeter": np.zeros((1, 2, 3)),
                "kind": "delta", "declared_coherent_delta": True,
                "delta_sign": 1.0,
            },
            {
                "delta": source, "perimeter": np.ones((1, 2, 3)),
                "kind": "delta", "declared_coherent_delta": True,
                "delta_sign": 1.0,
            },
        ]
        coefficient = object()
        with (
            mock.patch.object(feature_sum, "_load_grim", return_value={}) as loader,
            mock.patch.object(
                feature_sum, "load_seam_from_grim", return_value=coefficient
            ) as prepare,
        ):
            resolved = feature_sum._prepared_line_placements_at_frequency(
                placements, 2.0, {}
            )
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(prepare.call_count, 1)
        self.assertIs(resolved[0]["delta"], coefficient)
        self.assertIs(resolved[1]["delta"], coefficient)

    def test_line_expansion_writes_canonical_grim_complex_field(self):
        coefficients = SeamCoefficients(
            1.0,
            np.asarray([0.0, 90.0, 180.0]),
            np.asarray([1.0 + 0.2j] * 3),
            np.asarray([0.7 - 0.1j] * 3),
        )
        placement = {
            "delta": coefficients,
            "perimeter": np.asarray([
                [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]]
            ]),
            "segment_normals": np.asarray([
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            ]),
            "kind": "delta",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "line_feature.grim"
            feature_sum.export_radar_grim(
                str(output), bor_result=None, placements=[placement],
                frequencies_ghz=[1.0], azimuths_deg=[0.0],
                elevations_deg=[0.0],
            )
            with np.load(output, allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["polarizations"], ["VV", "HH", "VH"]
                )
                field = payload["rcs_amp_real"] + 1j * payload["rcs_amp_imag"]
                power = payload["rcs_power"]
                np.testing.assert_allclose(
                    power, 4.0 * np.pi * np.abs(field) ** 2,
                    rtol=1.0e-6, atol=1.0e-12,
                )
                self.assertGreater(float(np.max(np.abs(field))), 0.0)
                self.assertEqual(
                    str(payload["complex_field_domain"]),
                    "coherent_radar_frame_far_field_amplitude",
                )
                self.assertTrue(bool(payload["raw_complex_amplitude_preserved"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
