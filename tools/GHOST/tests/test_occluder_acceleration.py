"""Correctness and operational tests for vehicle-mesh body shadowing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from occluder import Occluder  # noqa: E402


def _projected_reference(triangles, points, direction, bias, diag):
    """The original exact projected-triangle implementation as an oracle."""
    tris = np.asarray(triangles, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    seed = (np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9
            else np.array([0.0, 1.0, 0.0]))
    e1 = seed - (seed @ d) * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    au = np.column_stack([a @ e1, a @ e2])
    bu = np.column_stack([b @ e1, b @ e2])
    cu = np.column_stack([c @ e1, c @ e2])
    aw, bw, cw = a @ d, b @ d, c @ d
    v0, v1 = bu - au, cu - au
    determinant = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid_det = np.abs(determinant) > 1e-14 * diag ** 2
    safe_det = np.where(valid_det, determinant, 1.0)
    pu = np.column_stack([pts @ e1, pts @ e2])
    pw = pts @ d
    visible = np.ones(len(pts), dtype=bool)
    for index in range(len(pts)):
        v2 = pu[index] - au
        beta = (v2[:, 0] * v1[:, 1]
                - v1[:, 0] * v2[:, 1]) / safe_det
        gamma = (v0[:, 0] * v2[:, 1]
                 - v2[:, 0] * v0[:, 1]) / safe_det
        alpha = 1.0 - beta - gamma
        inside = (valid_det & (alpha >= -1e-9) & (beta >= -1e-9)
                  & (gamma >= -1e-9))
        depth = alpha * aw + beta * bw + gamma * cw
        if np.any(inside & (depth > pw[index] + bias)):
            visible[index] = False
    return visible


def _box_triangles():
    corners = np.asarray([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    return np.asarray([[corners[i] for i in face] for face in faces])


class OccluderAccelerationTests(unittest.TestCase):
    def test_bvh_matches_projected_reference_for_random_mesh_and_rays(self):
        rng = np.random.default_rng(48291)
        triangles = rng.normal(size=(257, 3, 3))
        # Keep random facets comfortably nondegenerate.
        triangles[:, 1] += np.array([0.7, 0.0, 0.0])
        triangles[:, 2] += np.array([0.0, 0.7, 0.0])
        points = rng.uniform(-2.0, 2.0, size=(61, 3))
        directions = rng.normal(size=(7, 3))
        blocker = Occluder(triangles, bias=2.5e-7)
        original_triangles = np.array(blocker.tris, copy=True)

        for direction in directions:
            expected = _projected_reference(
                original_triangles, points, direction, blocker.bias,
                blocker.diag)
            actual = blocker.visible(points, direction)
            np.testing.assert_array_equal(actual, expected)

        info = blocker.acceleration_info
        self.assertTrue(info["ready"])
        self.assertEqual(info["triangle_count"], len(triangles))
        self.assertEqual(info["kind"], "morton_bvh")

    def test_known_box_shadow_and_surface_bias(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        points = np.asarray([
            [0.0, 0.0, 0.0],       # inside: +x wall blocks radar
            [-2.0, 0.0, 0.0],      # body ahead toward +x
            [2.0, 0.0, 0.0],       # already radar-side of body
            [-2.0, 2.0, 0.0],      # ray misses body
            [1.0, 0.0, 0.0],       # own surface ignored by positive bias
        ])
        np.testing.assert_array_equal(
            blocker.visible(points, [1.0, 0.0, 0.0]),
            [False, False, True, True, True],
        )

    def test_bvh_does_not_leak_near_triangle_edges_or_vertices(self):
        triangle = np.asarray([[
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]])
        blocker = Occluder(triangle, bias=0.0)
        # Include the adversarial parallel-AABB case and small excursions at
        # an edge/vertex that the exact barycentric tolerance accepts.
        points = np.asarray([
            [0.0, -5e-10, 0.25],
            [0.0, 0.5, -5e-10],
            [0.0, -4e-10, -4e-10],
            [0.0, 0.25, 0.25],
        ])
        direction = np.asarray([1.0, 0.0, 0.0])
        expected = _projected_reference(
            triangle, points, direction, blocker.bias, blocker.diag
        )
        actual = blocker.visible(points, direction)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.any(actual))

    def test_default_bias_is_numerical_not_vehicle_scale(self):
        triangles = _box_triangles() * 2.5  # five-metre body width
        blocker = Occluder(triangles)
        self.assertLess(blocker.bias, 1e-4)

    def test_visible_many_reports_progress_and_supports_cancellation(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        progress = []
        result = blocker.visible_many(
            [[-2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            progress_callback=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(result.shape, (2, 1))
        np.testing.assert_array_equal(result[:, 0], [False, True])
        self.assertEqual(progress, [(1, 2), (2, 2)])

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            blocker.visible_many(
                [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]],
                cancel_check=lambda: True,
            )

    def test_invalid_queries_fail_closed(self):
        blocker = Occluder(_box_triangles())
        for points in ([[1.0, 2.0]], [[np.nan, 0.0, 0.0]]):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    blocker.visible(points, [1.0, 0.0, 0.0])
        for direction in ([0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]):
            with self.subTest(direction=direction):
                with self.assertRaises(ValueError):
                    blocker.visible([[0.0, 0.0, 0.0]], direction)

    def test_execution_snapshot_shares_only_immutable_acceleration(self):
        blocker = Occluder(_box_triangles(), bias=2.5e-7)
        expected = blocker.visible(
            [[-2.0, 0.0, 0.0]], [1.0, 0.0, 0.0]
        )
        snapshot = blocker.execution_snapshot()

        self.assertIs(snapshot.tris, blocker.tris)
        self.assertIs(snapshot._bvh_lo, blocker._bvh_lo)
        self.assertIs(snapshot._bvh_hi, blocker._bvh_hi)
        self.assertIsNot(snapshot._bvh_lock, blocker._bvh_lock)
        self.assertFalse(snapshot.tris.flags.writeable)
        with self.assertRaises(ValueError):
            snapshot.tris.setflags(write=True)
        for boxes in (blocker._bvh_lo, blocker._bvh_hi):
            self.assertFalse(boxes.flags.writeable)
            with self.assertRaises(ValueError):
                boxes.setflags(write=True)

        blocker._bias = 100.0
        self.assertAlmostEqual(snapshot.bias, 2.5e-7)
        np.testing.assert_array_equal(
            snapshot.visible(
                [[-2.0, 0.0, 0.0]], [1.0, 0.0, 0.0]
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
