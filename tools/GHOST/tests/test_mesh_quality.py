"""Tests for topology checks that protect placement normals and shadowing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mesh_quality import audit_triangle_topology  # noqa: E402


def _tetrahedron():
    # Consistent outward winding.
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    return np.asarray([[vertices[index] for index in face] for face in faces])


class MeshQualityTests(unittest.TestCase):
    def test_closed_consistent_tetrahedron_passes(self):
        report = audit_triangle_topology(_tetrahedron())
        self.assertTrue(report.watertight)
        self.assertTrue(report.edge_manifold)
        self.assertTrue(report.consistently_wound)
        self.assertEqual(report.boundary_edge_count, 0)
        self.assertEqual(report.duplicate_triangle_count, 0)
        self.assertEqual(report.global_orientation, "outward")
        self.assertEqual(report.outward_closed_component_count, 1)
        self.assertEqual(report.messages(shadow_requested=True), ())

    def test_global_reversal_is_identified_and_flip_resolves_effective_normals(self):
        report = audit_triangle_topology(_tetrahedron()[:, ::-1])
        self.assertTrue(report.watertight)
        self.assertTrue(report.consistently_wound)
        self.assertEqual(report.global_orientation, "inward")
        self.assertEqual(report.inward_closed_component_count, 1)
        self.assertTrue(any(
            "wound inward" in message for message in report.messages()
        ))
        self.assertEqual(report.messages(normals_flipped=True), ())

    def test_oppositely_oriented_closed_components_cannot_use_one_global_flip(self):
        outward = _tetrahedron()
        inward = _tetrahedron()[:, ::-1] + np.asarray([3.0, 0.0, 0.0])
        report = audit_triangle_topology(np.concatenate([outward, inward]))
        self.assertTrue(report.watertight)
        self.assertTrue(report.consistently_wound)
        self.assertEqual(report.connected_component_count, 2)
        self.assertEqual(report.global_orientation, "mixed")
        self.assertEqual(report.outward_closed_component_count, 1)
        self.assertEqual(report.inward_closed_component_count, 1)
        for flipped in (False, True):
            self.assertTrue(any(
                "mixed global orientation" in message
                for message in report.messages(normals_flipped=flipped)
            ))

    def test_open_surface_warns_more_specifically_for_shadowing(self):
        report = audit_triangle_topology(_tetrahedron()[:-1])
        self.assertFalse(report.watertight)
        self.assertEqual(report.boundary_edge_count, 3)
        self.assertIn("placement patch", report.messages()[0])
        self.assertIn("rays can leak", report.messages(shadow_requested=True)[0])

    def test_reversed_face_detects_mixed_winding(self):
        triangles = _tetrahedron()
        triangles[0] = triangles[0, ::-1]
        report = audit_triangle_topology(triangles)
        self.assertFalse(report.consistently_wound)
        self.assertEqual(report.inconsistent_winding_edge_count, 3)
        self.assertTrue(any("mixed winding" in item for item in report.messages()))

    def test_duplicate_face_is_nonmanifold_and_reported(self):
        triangles = np.concatenate([_tetrahedron(), _tetrahedron()[[0]]])
        report = audit_triangle_topology(triangles)
        self.assertEqual(report.duplicate_triangle_count, 1)
        self.assertEqual(report.nonmanifold_edge_count, 3)
        messages = " ".join(report.messages())
        self.assertIn("duplicate", messages)
        self.assertIn("non-manifold", messages)

    def test_welding_handles_repeated_float_vertices_without_mutating_geometry(self):
        triangles = _tetrahedron()
        noisy = np.array(triangles, copy=True)
        noisy[1, 0] += np.array([2e-11, -1e-11, 1e-11])
        before = np.array(noisy, copy=True)
        report = audit_triangle_topology(noisy, weld_tolerance_m=1e-9)
        self.assertTrue(report.watertight)
        np.testing.assert_array_equal(noisy, before)

    def test_excessive_weld_tolerance_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "collapses"):
            audit_triangle_topology(_tetrahedron(), weld_tolerance_m=2.0)

    def test_weld_tolerance_is_euclidean_across_quantization_boundaries(self):
        # Two copies of the same open triangle have corresponding vertices
        # 0.02 m apart. A rounding-cell scheme can split them when tolerance is
        # 1 m; an actual distance weld must identify three vertices/faces.
        first = np.asarray([[
            [0.49, 0.0, 0.0],
            [2.49, 0.0, 0.0],
            [0.49, 2.0, 0.0],
        ]])
        second = first + np.asarray([0.02, 0.0, 0.0])
        report = audit_triangle_topology(
            np.concatenate([first, second]), weld_tolerance_m=1.0
        )
        self.assertEqual(report.welded_vertex_count, 3)
        self.assertEqual(report.duplicate_triangle_count, 1)


if __name__ == "__main__":
    unittest.main()
