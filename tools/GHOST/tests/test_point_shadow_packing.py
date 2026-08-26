"""Production-scale point body-shadow packing and integration regressions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import feature_sum  # noqa: E402
from occluder import (  # noqa: E402
    Occluder,
    PackedVisibility,
    PackedVisibilityRow,
)


def _box_triangles() -> np.ndarray:
    corners = np.asarray([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    return np.asarray([[corners[index] for index in face]
                       for face in faces], dtype=float)


def _zero_point_field(
        _pattern, _location, _normal, directions, _frequency, **kwargs):
    count = len(np.atleast_2d(directions))
    return {
        "F_vv": np.zeros(count, dtype=complex),
        "F_hh": np.zeros(count, dtype=complex),
        "F_vh": np.zeros(count, dtype=complex),
    }


class PointShadowPackingTests(unittest.TestCase):
    def test_packed_visibility_matches_dense_reference_and_skips_back_faces(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        points = np.asarray([
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        directions = np.asarray([
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        normals = np.asarray([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        unit_directions = directions / np.linalg.norm(
            directions, axis=1
        )[:, None]
        unit_normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        facing = (unit_normals @ unit_directions.T) > 0.0
        reference = blocker.visible_many(points, directions).T & facing

        traced_point_counts = []
        real_visible = blocker.visible

        def counted_visible(query_points, direction, **kwargs):
            traced_point_counts.append(len(np.atleast_2d(query_points)))
            return real_visible(query_points, direction, **kwargs)

        with mock.patch.object(
                blocker, "visible", side_effect=counted_visible):
            packed = blocker.visible_many_packed(
                points, directions, facing_normals=normals
            )

        self.assertIsInstance(packed, PackedVisibility)
        self.assertEqual(packed.shape, reference.shape)
        self.assertEqual(
            packed.nbytes,
            len(points) * ((len(directions) + 7) // 8),
        )
        self.assertLess(packed.nbytes, reference.nbytes)
        self.assertEqual(sum(traced_point_counts), int(np.count_nonzero(facing)))
        np.testing.assert_array_equal(packed.to_dense(), reference)
        with self.assertRaises(ValueError):
            packed._packed[0, 0] = np.uint8(0xff)
        for point_index in range(len(points)):
            row = packed.row(point_index)
            self.assertIsInstance(row, PackedVisibilityRow)
            self.assertEqual(row.shape, (len(directions),))
            np.testing.assert_array_equal(
                [row[look_index] for look_index in range(len(directions))],
                reference[point_index],
            )

    def test_packed_progress_is_monotone_and_cancellation_is_cooperative(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        points = np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        directions = np.repeat([[1.0, 0.0, 0.0]], 5, axis=0)
        # Every pair is back-facing, so this also proves cancellation and
        # progress do not depend on entering the expensive ray tracer.
        normals = np.repeat([[-1.0, 0.0, 0.0]], len(points), axis=0)
        progress = []
        with mock.patch.object(
                blocker, "visible",
                side_effect=AssertionError("back-facing pair was traced")):
            packed = blocker.visible_many_packed(
                points,
                directions,
                facing_normals=normals,
                progress_callback=lambda done, total: progress.append(
                    (int(done), int(total))
                ),
            )
        self.assertFalse(np.any(packed.to_dense()))
        self.assertEqual(progress, [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)])

        cancellation_checks = 0
        cancelled_progress = []

        def cancel_on_third_direction():
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 3

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            blocker.visible_many_packed(
                points,
                directions,
                facing_normals=normals,
                cancel_check=cancel_on_third_direction,
                progress_callback=lambda done, total: cancelled_progress.append(
                    (int(done), int(total))
                ),
            )
        self.assertEqual(cancelled_progress, [(1, 5), (2, 5)])

    def test_sum_features_passes_packed_rows_to_each_point(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        directions = np.asarray([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        points = [
            {
                "pattern": object(),
                "location": np.asarray([2.0, 0.0, 0.0]),
                "aperture_normal": np.asarray([1.0, 0.0, 0.0]),
            },
            {
                "pattern": object(),
                "location": np.asarray([0.0, 2.0, 0.0]),
                "aperture_normal": np.asarray([0.0, 1.0, 0.0]),
            },
        ]
        observed_rows = []

        def capture_point(*args, **kwargs):
            observed_rows.append(kwargs.get("_visibility"))
            return _zero_point_field(*args, **kwargs)

        with (
            mock.patch.object(
                blocker, "visible_many",
                side_effect=AssertionError("dense visibility path used"),
            ),
            mock.patch.object(
                feature_sum,
                "point_scatterer_amplitude",
                side_effect=capture_point,
            ),
        ):
            result = feature_sum.sum_features(
                None,
                [],
                directions,
                1.0,
                points=points,
                occluder=blocker,
            )

        self.assertEqual(len(observed_rows), len(points))
        self.assertTrue(all(
            isinstance(row, PackedVisibilityRow) for row in observed_rows
        ))
        np.testing.assert_array_equal(result["amp_vv"], np.zeros(3))
        np.testing.assert_array_equal(result["amp_hh"], np.zeros(3))
        np.testing.assert_array_equal(result["amp_vh"], np.zeros(3))

    def test_export_reuses_one_packed_shadow_pass_with_monotone_progress(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        points = [{
            "pattern": object(),
            "location": np.asarray([2.0, 0.0, 0.0]),
            "aperture_normal": np.asarray([1.0, 0.0, 0.0]),
        }]
        updates = []
        observed_rows = []

        def capture_point(*args, **kwargs):
            observed_rows.append(kwargs.get("_visibility"))
            return _zero_point_field(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "point-shadow.grim"
            with (
                mock.patch.object(
                    blocker,
                    "visible_many_packed",
                    wraps=blocker.visible_many_packed,
                ) as packed_query,
                mock.patch.object(
                    blocker, "visible_many",
                    side_effect=AssertionError("dense visibility path used"),
                ),
                mock.patch.object(
                    feature_sum,
                    "point_scatterer_amplitude",
                    side_effect=capture_point,
                ),
            ):
                saved = feature_sum.export_radar_grim(
                    str(output),
                    bor_result=None,
                    placements=[],
                    frequencies_ghz=[1.0, 2.0],
                    azimuths_deg=[0.0, 90.0, 180.0, 270.0],
                    elevations_deg=[0.0],
                    points=points,
                    occluder=blocker,
                    progress_callback=lambda done, total, message: updates.append(
                        (int(done), int(total), str(message))
                    ),
                )

            self.assertEqual(Path(saved), output.resolve())
            packed_query.assert_called_once()

        self.assertEqual(len(observed_rows), 2)
        self.assertTrue(all(
            isinstance(row, PackedVisibilityRow) for row in observed_rows
        ))
        fractions = [done / total for done, total, _message in updates]
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertEqual(fractions, sorted(fractions))
        self.assertTrue(any(
            "point body-shadow" in message for _done, _total, message in updates
        ))


if __name__ == "__main__":
    unittest.main()
