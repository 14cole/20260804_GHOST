#!/usr/bin/env python3
"""Manufactured placement controls for wedge and swept-wing bodies.

These are deliberately *not* full-wave reference cases.  The clean bodies use
an independent faceted physical-optics fixture, while the installed features
use two separate closed-form oracles: a Cartesian reciprocal dyadic for the
compact scatterer and a finite segmented-path Fourier integral for the line
feature.  Production receives files only through
``feature_workflow.prepare_feature_assembly`` and
``feature_workflow.execute_feature_assembly``.

The wedge isolates an arbitrarily sloped host face.  The swept-wing skin is a
non-symmetric, swept, tapered, dihedral panel whose hinge line follows shared
mesh edges across a change in facet normal.  No position, roll, amplitude,
phase, or range fitting is performed.  Passing these tests certifies the file,
frame, validation, translation-phase, polarization, and coherent-addition
paths within the reduced-order model; it does not certify body-feature mutual
coupling, multiple scattering, or a Maxwell solver.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "Backend"))
sys.path.insert(0, str(TESTS))

import feature_workflow  # noqa: E402
from line_expand import C0  # noqa: E402
import test_point_scatter_physics as point_oracle  # noqa: E402
import test_triaxial_ellipsoid_features as curved_oracle  # noqa: E402


FREQUENCY_GHZ = curved_oracle.FREQUENCY_GHZ
WAVELENGTH_M = C0 / (FREQUENCY_GHZ * 1.0e9)
WAVE_NUMBER = 2.0 * math.pi / WAVELENGTH_M
AZIMUTHS_DEG = curved_oracle.AZIMUTHS_DEG
ELEVATIONS_DEG = curved_oracle.ELEVATIONS_DEG


def _unit(vector):
    value = np.asarray(vector, dtype=float)
    return value / np.linalg.norm(value)


def _triangles_and_normals(vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    triangles = vertices[faces]
    raw = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas_twice = np.linalg.norm(raw, axis=1)
    if np.any(areas_twice <= 1.0e-14):
        raise AssertionError("fixture contains a degenerate triangle")
    return triangles, raw / areas_twice[:, None]


def _outward_closed_faces(vertices, faces, interior):
    """Orient triangles of a convex closed fixture away from ``interior``."""

    vertices = np.asarray(vertices, dtype=float)
    output = []
    for raw_face in faces:
        face = np.asarray(raw_face, dtype=int)
        triangle = vertices[face]
        normal = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        if float(normal @ (np.mean(triangle, axis=0) - interior)) < 0.0:
            face = face[[0, 2, 1]]
        output.append(face)
    return np.asarray(output, dtype=int)


def _clean_faceted_po(triangles_cad, normals_cad):
    """Independent centroid-PO manufactured field in the documented frame."""

    triangles_earth = (
        np.asarray(triangles_cad, dtype=float)
        @ point_oracle.CAD_TO_EARTH.T
    )
    normals_earth = (
        np.asarray(normals_cad, dtype=float)
        @ point_oracle.CAD_TO_EARTH.T
    )
    directions = curved_oracle._radar_directions()
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles_earth[:, 1] - triangles_earth[:, 0],
            triangles_earth[:, 2] - triangles_earth[:, 0],
        ),
        axis=1,
    )
    centroids = np.mean(triangles_earth, axis=1)
    scalar = np.zeros(len(directions), dtype=np.complex128)
    for index, direction in enumerate(directions):
        illumination = np.maximum(normals_earth @ direction, 0.0)
        scalar[index] = (
            -1j
            / WAVELENGTH_M
            * np.sum(
                areas
                * illumination
                * np.exp(2j * WAVE_NUMBER * (centroids @ direction))
            )
        )
    scalar = scalar.reshape(len(AZIMUTHS_DEG), len(ELEVATIONS_DEG))
    amplitude = np.zeros(
        (len(AZIMUTHS_DEG), len(ELEVATIONS_DEG), 1, 3),
        dtype=np.complex128,
    )
    amplitude[:, :, 0, 0] = scalar
    amplitude[:, :, 0, 1] = scalar
    return amplitude


def _ramp_case():
    """Closed, asymmetric truncated wedge with both features on its ramp."""

    half_width = 0.31
    y_rear, y_front = -0.30, 0.35
    rear_height, front_height = 0.055, 0.275
    vertices = np.asarray([
        [-half_width, y_rear, 0.0],
        [half_width, y_rear, 0.0],
        [half_width, y_front, 0.0],
        [-half_width, y_front, 0.0],
        [-half_width, y_rear, rear_height],
        [half_width, y_rear, rear_height],
        [half_width, y_front, front_height],
        [-half_width, y_front, front_height],
    ])
    raw_faces = (
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 5, 1), (0, 4, 5),
        (3, 2, 6), (3, 6, 7),
        (0, 3, 7), (0, 7, 4),
        (1, 5, 6), (1, 6, 2),
    )
    interior = np.mean(vertices, axis=0)
    faces = _outward_closed_faces(vertices, raw_faces, interior)
    triangles, normals = _triangles_and_normals(vertices, faces)

    slope = (front_height - rear_height) / (y_front - y_rear)
    top_normal = _unit([0.0, -slope, 1.0])

    def top_z(y):
        return rear_height + slope * (float(y) - y_rear)

    line_y = -0.075
    line_points = np.asarray([
        [-0.225, line_y, top_z(line_y)],
        [0.018, line_y, top_z(line_y)],
        [0.215, line_y, top_z(line_y)],
    ])
    line_segments = np.stack((line_points[:-1], line_points[1:]), axis=1)
    line_normals = np.tile(top_normal, (len(line_segments), 2, 1))
    point_location = np.asarray([0.137, 0.092, top_z(0.092)])
    point_roll = _unit(np.asarray([1.0, 0.0, 0.0]) + 0.29 * top_normal)
    return {
        "name": "wedge_ramp",
        "vertices": vertices,
        "faces": faces,
        "triangles": triangles,
        "face_normals": normals,
        "line_id": "ramp_door_seal_001",
        "line_dataset_id": "ramp_seal_delta",
        "line_segments": line_segments,
        "line_normals": line_normals,
        "point_id": "ramp_fastener_001",
        "point_dataset_id": "ramp_fastener_delta",
        "point_location": point_location,
        "point_normal": top_normal,
        "point_roll": point_roll,
        "line_coefficient": 0.041 - 0.014j,
        "translation_control": np.asarray([0.027, 0.0, 0.0]),
    }


def _swept_wing_case():
    """Open swept wing skin with a faceted hinge-line crease."""

    span_x = np.asarray([0.08, 0.46, 0.83])
    leading_y = np.asarray([0.43, 0.365, 0.245])
    chords = np.asarray([0.62, 0.49, 0.35])
    dihedral_z = np.asarray([0.035, 0.078, 0.145])
    chord_fraction = np.asarray([0.0, 0.64, 1.0])
    camber_scale = np.asarray([0.0, 0.025, 0.0])
    vertices = []
    for x, leading, chord, z in zip(
        span_x, leading_y, chords, dihedral_z
    ):
        for fraction, camber in zip(chord_fraction, camber_scale):
            vertices.append([
                x,
                leading - fraction * chord,
                z + camber,
            ])
    vertices = np.asarray(vertices, dtype=float)

    def vertex_index(span_index, chord_index):
        return 3 * span_index + chord_index

    faces = []
    face_lookup = {}
    for span_index in range(2):
        for chord_index in range(2):
            a = vertex_index(span_index, chord_index)
            b = vertex_index(span_index + 1, chord_index)
            c = vertex_index(span_index + 1, chord_index + 1)
            d = vertex_index(span_index, chord_index + 1)
            pair = [[a, b, c], [a, c, d]]
            oriented = []
            for face in pair:
                triangle = vertices[np.asarray(face)]
                normal = np.cross(
                    triangle[1] - triangle[0],
                    triangle[2] - triangle[0],
                )
                if normal[2] < 0.0:
                    face = [face[0], face[2], face[1]]
                oriented.append(face)
                faces.append(face)
            face_lookup[(span_index, chord_index)] = (
                len(faces) - 2,
                len(faces) - 1,
            )
    faces = np.asarray(faces, dtype=int)
    triangles, normals = _triangles_and_normals(vertices, faces)

    line_points = vertices[[
        vertex_index(0, 1),
        vertex_index(1, 1),
        vertex_index(2, 1),
    ]]
    line_segments = np.stack((line_points[:-1], line_points[1:]), axis=1)
    line_normals = []
    for span_index in range(2):
        # Use the aft skin triangle incident to the hinge edge.  This makes
        # ownership at the shared mesh edge explicit and lets the normal jump
        # at the spanwise panel break without averaging it away.
        face_index = face_lookup[(span_index, 1)][0]
        line_normals.append([normals[face_index], normals[face_index]])
    line_normals = np.asarray(line_normals)

    point_face_index = face_lookup[(1, 0)][1]
    point_triangle = triangles[point_face_index]
    point_location = (
        0.22 * point_triangle[0]
        + 0.33 * point_triangle[1]
        + 0.45 * point_triangle[2]
    )
    point_normal = normals[point_face_index]
    point_tangent = _unit(point_triangle[1] - point_triangle[0])
    point_roll = _unit(point_tangent + 0.31 * point_normal)
    return {
        "name": "swept_wing_panel",
        "vertices": vertices,
        "faces": faces,
        "triangles": triangles,
        "face_normals": normals,
        "line_id": "wing_flap_hinge_001",
        "line_dataset_id": "wing_hinge_delta",
        "line_segments": line_segments,
        "line_normals": line_normals,
        "point_id": "wing_access_fastener_001",
        "point_dataset_id": "wing_fastener_delta",
        "point_location": point_location,
        "point_normal": point_normal,
        "point_roll": point_roll,
        "line_coefficient": 0.036 + 0.019j,
        "translation_control": np.asarray([0.0, -0.021, 0.0]),
    }


def _normalized_complex_rms(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


class WedgeAndWingFeatureAssemblyRegression(unittest.TestCase):
    def _make_fixture(self, root, case):
        clean = _clean_faceted_po(case["triangles"], case["face_normals"])
        surface = root / f"{case['name']}.facet"
        curved_oracle._write_facet(surface, case["vertices"], case["faces"])
        base = curved_oracle._write_external_body(
            root / f"{case['name']}_clean.grim", clean
        )
        output = root / f"{case['name']}_featured.grim"
        line_dataset = root / f"{case['line_dataset_id']}.grim"
        point_dataset = root / f"{case['point_dataset_id']}.grim"
        line_csv = root / f"{case['name']}_lines.csv"
        point_csv = root / f"{case['name']}_points.csv"
        curved_oracle._write_isotropic_line_delta(
            line_dataset, case["line_coefficient"]
        )
        curved_oracle._write_anisotropic_point_delta(point_dataset)
        curved_oracle._write_line_csv(
            line_csv,
            case["line_id"],
            case["line_dataset_id"],
            case["line_segments"],
            case["line_normals"],
        )
        curved_oracle._write_point_csv(
            point_csv,
            case["point_id"],
            case["point_dataset_id"],
            case["point_location"],
            case["point_normal"],
            case["point_roll"],
        )
        request = feature_workflow.FeatureAssemblyRequest(
            base_grim=base,
            output_grim=output,
            coordinate_units="inches",
            surface_mesh=surface,
            surface_units="meters",
            line_locations_csv=line_csv,
            line_datasets={case["line_dataset_id"]: line_dataset},
            point_locations_csv=point_csv,
            point_datasets={case["point_dataset_id"]: point_dataset},
            enabled_line_ids=(case["line_id"],),
            enabled_point_placement_ids=(case["point_id"],),
            skin_tol_m=2.0e-8,
            skin_phase_tol_deg=0.1,
            normal_tol_deg=7.0,
            history=f"manufactured {case['name']} placement control",
        )
        return {
            **case,
            "clean": clean,
            "surface": surface,
            "base": base,
            "output": output,
            "line_dataset": line_dataset,
            "point_dataset": point_dataset,
            "line_csv": line_csv,
            "point_csv": point_csv,
            "request": request,
        }

    def _assert_public_workflow(self, fixture):
        plan = feature_workflow.prepare_feature_assembly(fixture["request"])
        self.assertIsNone(plan.body_profile)
        self.assertEqual(
            plan.dataset_requirements.line_instances,
            ((fixture["line_id"], fixture["line_dataset_id"], 2),),
        )
        self.assertEqual(
            plan.dataset_requirements.point_instances,
            ((fixture["point_id"], fixture["point_dataset_id"]),),
        )
        self.assertLess(plan.line_records[0]["max_skin_offset_m"], 2.0e-14)
        self.assertLess(plan.point_records[0]["skin_offset_m"], 2.0e-14)

        expected_line = curved_oracle._direct_isotropic_line(
            fixture["line_segments"],
            fixture["line_normals"],
            fixture["line_coefficient"],
        )
        expected_point = curved_oracle._direct_anisotropic_point(
            fixture["point_location"],
            fixture["point_normal"],
            fixture["point_roll"],
        )
        expected_feature = (expected_line + expected_point).reshape(
            len(AZIMUTHS_DEG), len(ELEVATIONS_DEG), 1, 3
        )
        expected_total = fixture["clean"] + expected_feature

        saved = feature_workflow.execute_feature_assembly(plan)
        with np.load(fixture["base"], allow_pickle=False) as stored_base:
            clean_after = (
                stored_base["rcs_amp_real"] + 1j * stored_base["rcs_amp_imag"]
            )
        with np.load(saved, allow_pickle=False) as stored:
            actual = stored["rcs_amp_real"] + 1j * stored["rcs_amp_imag"]
            provenance = json.loads(str(stored["feature_provenance_json"]))

        np.testing.assert_array_equal(clean_after, fixture["clean"])
        feature_nrms = _normalized_complex_rms(
            expected_feature, actual - clean_after
        )
        total_nrms = _normalized_complex_rms(expected_total, actual)
        self.assertLess(feature_nrms, 2.0e-10)
        self.assertLess(total_nrms, 2.0e-11)
        self.assertEqual(provenance[-1]["line_feature_count"], 1)
        self.assertEqual(provenance[-1]["compact_feature_count"], 1)
        self.assertFalse(
            plan.feature_provenance["model_scope"][
                "body_feature_mutual_coupling"
            ]
        )
        self.assertFalse(
            plan.feature_provenance["model_scope"]["multiple_scattering"]
        )

        # Two non-fitted controls must be materially distinguishable from the
        # truth: a translated line and a rotated anisotropic fastener frame.
        shifted_line = curved_oracle._direct_isotropic_line(
            fixture["line_segments"] + fixture["translation_control"],
            fixture["line_normals"],
            fixture["line_coefficient"],
        )
        tangent_x = point_oracle._local_frame(
            fixture["point_normal"], fixture["point_roll"]
        )[:, 0]
        tangent_y = _unit(np.cross(fixture["point_normal"], tangent_x))
        wrong_roll = curved_oracle._direct_anisotropic_point(
            fixture["point_location"],
            fixture["point_normal"],
            math.cos(0.71) * tangent_x + math.sin(0.71) * tangent_y,
        )
        self.assertGreater(
            _normalized_complex_rms(expected_line, shifted_line), 0.10
        )
        self.assertGreater(
            _normalized_complex_rms(expected_point, wrong_roll), 0.08
        )

    def test_sloped_wedge_and_swept_faceted_wing_match_independent_oracles(self):
        for factory in (_ramp_case, _swept_wing_case):
            case = factory()
            with self.subTest(body=case["name"]):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._make_fixture(Path(directory), case)
                    self._assert_public_workflow(fixture)

    def test_each_body_rejects_off_skin_and_inward_feature_definitions(self):
        for factory in (_ramp_case, _swept_wing_case):
            case = factory()
            with self.subTest(body=case["name"]):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._make_fixture(Path(directory), case)
                    curved_oracle._write_point_csv(
                        fixture["point_csv"],
                        fixture["point_id"],
                        fixture["point_dataset_id"],
                        fixture["point_location"]
                        + 5.0e-4 * fixture["point_normal"],
                        fixture["point_normal"],
                        fixture["point_roll"],
                    )
                    with self.assertRaisesRegex(ValueError, "off the skin"):
                        feature_workflow.prepare_feature_assembly(
                            fixture["request"]
                        )

                    curved_oracle._write_point_csv(
                        fixture["point_csv"],
                        fixture["point_id"],
                        fixture["point_dataset_id"],
                        fixture["point_location"],
                        fixture["point_normal"],
                        fixture["point_roll"],
                    )
                    curved_oracle._write_line_csv(
                        fixture["line_csv"],
                        fixture["line_id"],
                        fixture["line_dataset_id"],
                        fixture["line_segments"],
                        -fixture["line_normals"],
                    )
                    with self.assertRaisesRegex(
                        ValueError, "outward skin normal"
                    ):
                        feature_workflow.prepare_feature_assembly(
                            fixture["request"]
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
