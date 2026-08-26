#!/usr/bin/env python3
"""Curved non-BoR placement and response regressions.

The platform is a triangulated triaxial ellipsoid: all three semi-axes differ,
so it is neither a body of revolution nor reducible to the existing plate,
folded-panel, or box fixtures.  A line seal follows mesh edges across several
facets and an anisotropic compact feature uses an arbitrary face normal and
roll reference.  Both enter through :mod:`feature_workflow`'s public request,
prepare, and execute service.

The response oracle is separate from GHOST.  The line coupon is phase-adjusted
so its installed TM/TE coefficients are identical; its response is therefore
the closed-form Fourier integral of the segmented path.  The compact feature
uses a constant reciprocal local Jones matrix which is rotated directly into
the earth radar basis.  No position, amplitude, phase, or frame fitting is
performed.  This certifies reduced-order placement and coherent combination;
it does not claim body-feature mutual coupling or multiple scattering.
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

import components  # noqa: E402
import feature_sum  # noqa: E402
import feature_workflow  # noqa: E402
from frame import to_axis_frame  # noqa: E402
from grim_io import _save_grim_npz  # noqa: E402
from line_expand import C0, PSI_HH_DEG, PSI_VV_DEG  # noqa: E402
import test_point_scatter_physics as point_oracle  # noqa: E402


FREQUENCY_GHZ = 1.6
WAVELENGTH_M = C0 / (FREQUENCY_GHZ * 1.0e9)
WAVE_NUMBER = 2.0 * math.pi / WAVELENGTH_M
AZIMUTHS_DEG = np.asarray([0.0, 25.0, 50.0])
ELEVATIONS_DEG = np.asarray([35.0, 50.0, 65.0])
POLARIZATIONS = np.asarray(["VV", "HH", "VH"])
INCH_M = 0.0254
POINT_AZIMUTH_NODES = np.asarray([0.0, 90.0, 180.0, 270.0, 360.0])
POINT_AZIMUTH_PROFILE = np.asarray([0.0, 1.0, -0.7, 0.55, 0.0])
POINT_ELEVATION_NODES = np.asarray([-90.0, 0.0, 90.0])
POINT_BASE_JONES = np.asarray([
    [0.0053 + 0.0009j, 0.0008 - 0.0003j],
    [0.0008 - 0.0003j, 0.0031 - 0.0007j],
])
POINT_MODULATION_JONES = np.asarray([
    [0.0032 + 0.0008j, 0.0014 - 0.0005j],
    [0.0014 - 0.0005j, -0.0021 + 0.0006j],
])


def _unit(vector):
    value = np.asarray(vector, dtype=float)
    return value / np.linalg.norm(value)


def _radar_directions():
    return np.asarray([
        point_oracle._direction_and_radar_basis(azimuth, elevation)[0]
        for azimuth in AZIMUTHS_DEG
        for elevation in ELEVATIONS_DEG
    ])


def _ellipsoid_mesh(
    center=(0.08, -0.04, 0.12),
    axes=(0.42, 0.31, 0.23),
    latitude_bands=24,
    longitude_count=48,
):
    """Return outward-wound vertices/faces and latitude-ring indices."""

    center = np.asarray(center, dtype=float)
    axes = np.asarray(axes, dtype=float)
    vertices = [center + np.asarray([0.0, 0.0, axes[2]])]
    rings = []
    for latitude_index in range(1, latitude_bands):
        theta = math.pi * latitude_index / latitude_bands
        ring = []
        for longitude_index in range(longitude_count):
            phi = 2.0 * math.pi * longitude_index / longitude_count
            ring.append(len(vertices))
            vertices.append(center + np.asarray([
                axes[0] * math.sin(theta) * math.cos(phi),
                axes[1] * math.sin(theta) * math.sin(phi),
                axes[2] * math.cos(theta),
            ]))
        rings.append(ring)
    south = len(vertices)
    vertices.append(center - np.asarray([0.0, 0.0, axes[2]]))

    faces = []
    north = 0
    for longitude_index in range(longitude_count):
        following = (longitude_index + 1) % longitude_count
        faces.append((
            north,
            rings[0][longitude_index],
            rings[0][following],
        ))
    for ring_index in range(len(rings) - 1):
        upper = rings[ring_index]
        lower = rings[ring_index + 1]
        for longitude_index in range(longitude_count):
            following = (longitude_index + 1) % longitude_count
            faces.append((
                upper[longitude_index],
                lower[longitude_index],
                lower[following],
            ))
            faces.append((
                upper[longitude_index],
                lower[following],
                upper[following],
            ))
    for longitude_index in range(longitude_count):
        following = (longitude_index + 1) % longitude_count
        faces.append((
            south,
            rings[-1][following],
            rings[-1][longitude_index],
        ))

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    triangles = vertices[faces]
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_normals = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]
    # Winding is checked independently against the smooth ellipsoid gradient.
    centroids = np.mean(triangles, axis=1)
    smooth_outward = (centroids - center) / (axes * axes)
    if np.any(np.sum(face_normals * smooth_outward, axis=1) <= 0.0):
        raise AssertionError("ellipsoid fixture contains an inward-wound face")
    return center, axes, vertices, faces, triangles, face_normals, rings


def _write_facet(path, vertices, faces):
    rows = [f"{len(vertices)} {len(faces)}"]
    rows.extend(
        f"{index + 1} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}"
        for index, point in enumerate(np.asarray(vertices, dtype=float))
    )
    rows.extend(
        f"{index + 1} {face[0] + 1} {face[1] + 1} {face[2] + 1}"
        for index, face in enumerate(np.asarray(faces, dtype=int))
    )
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def _clean_faceted_po(triangles, face_normals):
    """Independent centroid PO fixture for the clean curved platform."""

    directions = _radar_directions()
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    centroids = np.mean(triangles, axis=1)
    scalar = np.zeros(len(directions), dtype=np.complex128)
    for index, direction in enumerate(directions):
        illumination = np.maximum(face_normals @ direction, 0.0)
        scalar[index] = (
            -1j
            / WAVELENGTH_M
            * np.sum(
                areas
                * illumination
                * np.exp(2j * WAVE_NUMBER * (centroids @ direction))
            )
        )
    amplitude = np.zeros(
        (len(AZIMUTHS_DEG), len(ELEVATIONS_DEG), 1, 3),
        dtype=np.complex128,
    )
    scalar = scalar.reshape(len(AZIMUTHS_DEG), len(ELEVATIONS_DEG))
    amplitude[:, :, 0, 0] = scalar
    amplitude[:, :, 0, 1] = scalar
    return amplitude


def _write_external_body(path, amplitude):
    amplitude = np.asarray(amplitude, dtype=np.complex128)
    payload = {
        "azimuths": AZIMUTHS_DEG,
        "elevations": ELEVATIONS_DEG,
        "frequencies": np.asarray([FREQUENCY_GHZ]),
        "polarizations": POLARIZATIONS,
        "combine_role": np.asarray("coherent"),
        "rcs_power": (4.0 * math.pi * np.abs(amplitude) ** 2).astype(np.float32),
        "rcs_phase": np.angle(amplitude).astype(np.float32),
        "rcs_domain": np.asarray("power_phase"),
        "power_domain": np.asarray("linear_rcs"),
        "source_path": np.asarray("independent triaxial-ellipsoid PO fixture"),
        "history": np.asarray("clean curved non-BoR external platform"),
        "units": np.asarray(json.dumps({
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        })),
        "phase_reference": np.asarray(components.COMPONENT_PHASE_REFERENCE),
        "amplitude_convention": np.asarray(
            components.COMPONENT_AMPLITUDE_CONVENTION
        ),
        "complex_field_domain": np.asarray(
            components.COMPONENT_COMPLEX_FIELD_DOMAIN
        ),
        "raw_complex_amplitude_preserved": np.asarray(True),
        "rcs_amp_real": amplitude.real.astype(np.float64),
        "rcs_amp_imag": amplitude.imag.astype(np.float64),
    }
    return Path(_save_grim_npz(payload, str(path)))


def _local_point_jones(azimuth_deg, elevation_deg):
    """Reciprocal piecewise-bilinear local pattern, independent of GHOST."""

    azimuth = float(azimuth_deg) % 360.0
    azimuth_weight = float(np.interp(
        azimuth, POINT_AZIMUTH_NODES, POINT_AZIMUTH_PROFILE
    ))
    elevation_weight = max(0.0, 1.0 - abs(float(elevation_deg)) / 90.0)
    return (
        POINT_BASE_JONES
        + azimuth_weight * elevation_weight * POINT_MODULATION_JONES
    )


def _write_anisotropic_point_delta(path):
    amplitude = np.empty(
        (
            len(POINT_AZIMUTH_NODES),
            len(POINT_ELEVATION_NODES),
            1,
            3,
        ),
        dtype=np.complex128,
    )
    for azimuth_index, azimuth in enumerate(POINT_AZIMUTH_NODES):
        for elevation_index, elevation in enumerate(POINT_ELEVATION_NODES):
            local_jones = _local_point_jones(azimuth, elevation)
            amplitude[azimuth_index, elevation_index, 0] = (
                local_jones[0, 0],
                local_jones[1, 1],
                local_jones[0, 1],
            )
    payload = {
        "azimuths": POINT_AZIMUTH_NODES,
        "elevations": POINT_ELEVATION_NODES,
        "frequencies": np.asarray([FREQUENCY_GHZ]),
        "polarizations": POLARIZATIONS,
        "rcs_power": (4.0 * math.pi * np.abs(amplitude) ** 2).astype(np.float32),
        "rcs_phase": np.angle(amplitude).astype(np.float32),
        "rcs_domain": np.asarray("delta"),
        "power_domain": np.asarray("linear_rcs"),
        "units": np.asarray(json.dumps({
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        })),
        "raw_complex_amplitude_preserved": np.asarray(True),
        "rcs_amp_real": amplitude.real.astype(np.float64),
        "rcs_amp_imag": amplitude.imag.astype(np.float64),
        **feature_sum.point_pattern_convention_metadata(),
    }
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _write_isotropic_line_delta(path, installed_coefficient):
    """Write raw TM/TE values that become equal after legacy phase mapping."""

    angles = np.asarray([0.0, 90.0, 180.0])
    raw_te = complex(installed_coefficient) * np.exp(
        -1j * math.radians(PSI_VV_DEG)
    )
    raw_tm = complex(installed_coefficient) * np.exp(
        -1j * math.radians(PSI_HH_DEG)
    )
    amplitude = np.empty((len(angles), 1, 1, 2), dtype=np.complex128)
    amplitude[..., 0] = raw_te
    amplitude[..., 1] = raw_tm
    payload = {
        "azimuths": angles,
        "elevations": np.asarray([0.0]),
        "frequencies": np.asarray([FREQUENCY_GHZ]),
        "polarizations": np.asarray(["VV", "HH"]),
        "rcs_power": (
            np.abs(amplitude) ** 2 / (4.0 * WAVE_NUMBER)
        ).astype(np.float32),
        "rcs_phase": np.angle(amplitude).astype(np.float32),
        "units": np.asarray(json.dumps({
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_linear_quantity": "sigma_2d",
        })),
        "raw_complex_amplitude_preserved": np.asarray(True),
        "rcs_amp_real": amplitude.real.astype(np.float64),
        "rcs_amp_imag": amplitude.imag.astype(np.float64),
    }
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _direct_anisotropic_point(location_cad, normal_cad, roll_cad):
    frame_cad = point_oracle._local_frame(normal_cad, roll_cad)
    frame_earth = point_oracle.CAD_TO_EARTH @ frame_cad
    location_earth = point_oracle.CAD_TO_EARTH @ np.asarray(
        location_cad, dtype=float
    )
    normal_earth = frame_earth[:, 2]
    output = np.zeros((len(_radar_directions()), 3), dtype=np.complex128)
    for index, (azimuth, elevation) in enumerate(
        (a, e) for a in AZIMUTHS_DEG for e in ELEVATIONS_DEG
    ):
        direction, radar_v, radar_h = (
            point_oracle._direction_and_radar_basis(azimuth, elevation)
        )
        if float(direction @ normal_earth) <= 0.0:
            continue
        local_direction = frame_earth.T @ direction
        local_azimuth = math.degrees(math.atan2(
            float(local_direction[1]), float(local_direction[0])
        ))
        local_elevation = math.degrees(math.asin(np.clip(
            float(local_direction[2]), -1.0, 1.0
        )))
        local_v, local_h = point_oracle._local_spherical_basis(
            local_azimuth, local_elevation
        )
        local_jones = _local_point_jones(local_azimuth, local_elevation)
        feature_v = frame_earth @ local_v
        feature_h = frame_earth @ local_h
        mapping = np.asarray([
            [feature_v @ radar_v, feature_v @ radar_h],
            [feature_h @ radar_v, feature_h @ radar_h],
        ])
        radar_jones = mapping.T @ local_jones @ mapping
        phase = np.exp(
            2j * WAVE_NUMBER * float(direction @ location_earth)
        )
        output[index] = (
            radar_jones[0, 0] * phase,
            radar_jones[1, 1] * phase,
            radar_jones[0, 1] * phase,
        )
    return output


def _direct_isotropic_line(
    segments_cad, endpoint_normals_cad, installed_coefficient
):
    directions = _radar_directions()
    segments_earth = np.asarray(segments_cad) @ point_oracle.CAD_TO_EARTH.T
    normals_earth = (
        np.asarray(endpoint_normals_cad) @ point_oracle.CAD_TO_EARTH.T
    )
    scalar = np.zeros(len(directions), dtype=np.complex128)
    parameters = (np.arange(33, dtype=float) + 0.5) / 33.0
    for direction_index, direction in enumerate(directions):
        for segment, endpoint_normals in zip(segments_earth, normals_earth):
            interpolated = (
                endpoint_normals[0, None, :] * (1.0 - parameters)[:, None]
                + endpoint_normals[1, None, :] * parameters[:, None]
            )
            interpolated /= np.linalg.norm(interpolated, axis=1)[:, None]
            # The test intentionally stays outside the 10-degree grazing
            # taper, making the independent closed-form path integral exact.
            if float(np.min(interpolated @ direction)) <= math.sin(
                math.radians(10.0)
            ):
                raise AssertionError("fixture entered the grazing taper")
            start, stop = segment
            displacement = stop - start
            length = float(np.linalg.norm(displacement))
            tangent = displacement / length
            x = WAVE_NUMBER * length * float(direction @ tangent)
            scalar[direction_index] += (
                complex(installed_coefficient)
                / (4.0 * math.pi)
                * np.exp(2j * WAVE_NUMBER * float(direction @ start))
                * length
                * np.exp(1j * x)
                * np.sinc(x / math.pi)
            )
    return np.column_stack((scalar, scalar, np.zeros_like(scalar)))


def _write_point_csv(path, placement_id, dataset_id, location, normal, roll):
    values = [
        placement_id,
        dataset_id,
        *(np.asarray(location, dtype=float) / INCH_M),
        *np.asarray(normal, dtype=float),
        *np.asarray(roll, dtype=float),
    ]
    row = ",".join(
        value if isinstance(value, str) else f"{float(value):.17g}"
        for value in values
    )
    Path(path).write_text(
        ",".join(feature_workflow.POINT_CSV_COLUMNS) + "\n" + row + "\n",
        encoding="utf-8",
    )


def _write_line_csv(path, line_id, dataset_id, segments, endpoint_normals):
    rows = [",".join(feature_workflow.LINE_CSV_COLUMNS)]
    for index, (segment, normals) in enumerate(
        zip(np.asarray(segments), np.asarray(endpoint_normals)), 1
    ):
        values = [
            line_id,
            dataset_id,
            index,
            *(segment[0] / INCH_M),
            *(segment[1] / INCH_M),
            *normals[0],
            *normals[1],
        ]
        rows.append(",".join(
            value if isinstance(value, str) else f"{float(value):.17g}"
            for value in values
        ))
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def _normalized_complex_rms(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


class FacetCreaseOwnershipRegression(unittest.TestCase):
    @staticmethod
    def _folded_surface():
        angle = math.radians(35.0)
        normal_a = np.asarray([0.0, 0.0, 1.0])
        normal_b = np.asarray([-math.sin(angle), 0.0, math.cos(angle)])
        triangles_cad = np.asarray([
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [
                [0.0, 0.0, 0.0],
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
            ],
        ])
        return (
            feature_workflow.TriangleSurface(to_axis_frame(triangles_cad)),
            normal_a,
            normal_b,
        )

    def test_crossing_and_edge_aligned_lines_use_the_supplied_incident_side(self):
        angle = math.radians(35.0)
        surface, normal_a, normal_b = self._folded_surface()
        joint = np.asarray([0.0, 0.4, 0.0])
        crossing_segments = np.asarray([
            [[-0.3, 0.4, 0.0], joint],
            [joint, [0.3 * math.cos(angle), 0.4, 0.3 * math.sin(angle)]],
        ])
        crossing_normals = np.asarray([
            [normal_a, normal_a],
            [normal_b, normal_b],
        ])
        edge_segment = np.asarray([[[0.0, 0.2, 0.0], [0.0, 0.8, 0.0]]])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "seal.grim"
            dataset.touch()
            cases = (
                ("crossing", crossing_segments, crossing_normals),
                ("edge_side_a", edge_segment, np.asarray([[normal_a, normal_a]])),
                ("edge_side_b", edge_segment, np.asarray([[normal_b, normal_b]])),
            )
            for label, segments, normals in cases:
                with self.subTest(label=label):
                    csv_path = root / f"{label}.csv"
                    _write_line_csv(csv_path, label, "seal", segments, normals)
                    placements, records = feature_workflow.prepare_line_placements(
                        None,
                        surface,
                        coordinate_scale=INCH_M,
                        skin_limit_m=1.0e-10,
                        wavelength_m=0.3,
                        normal_tolerance_deg=5.0,
                        locations_csv=csv_path,
                        datasets={"seal": dataset},
                    )
                    self.assertEqual(len(placements), 1)
                    self.assertLess(records[0]["max_skin_offset_m"], 5.0e-16)
                    self.assertLess(records[0]["max_normal_error_deg"], 1.0e-8)

            wrong = root / "wrong_normal.csv"
            wrong_normal = -_unit(normal_a + normal_b)
            _write_line_csv(
                wrong,
                "wrong",
                "seal",
                edge_segment,
                np.asarray([[wrong_normal, wrong_normal]]),
            )
            with self.assertRaisesRegex(ValueError, "outward skin normal"):
                feature_workflow.prepare_line_placements(
                    None,
                    surface,
                    coordinate_scale=INCH_M,
                    skin_limit_m=1.0e-10,
                    wavelength_m=0.3,
                    normal_tolerance_deg=180.0,
                    locations_csv=wrong,
                    datasets={"seal": dataset},
                )

    def test_point_on_shared_edge_accepts_either_incident_outward_normal(self):
        surface, normal_a, normal_b = self._folded_surface()
        location = np.asarray([0.0, 0.4, 0.0])
        roll = np.asarray([0.0, 1.0, 0.0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "fastener.grim"
            dataset.touch()
            csv_path = root / "point.csv"
            for label, normal in (("side_a", normal_a), ("side_b", normal_b)):
                with self.subTest(label=label):
                    _write_point_csv(
                        csv_path, label, "fastener", location, normal, roll
                    )
                    points, records = feature_workflow.prepare_point_placements(
                        None,
                        surface,
                        coordinate_scale=INCH_M,
                        skin_limit_m=1.0e-10,
                        wavelength_m=0.3,
                        normal_tolerance_deg=5.0,
                        locations_csv=csv_path,
                        datasets={"fastener": dataset},
                        pattern_loader=lambda *_args, **_kwargs: object(),
                    )
                    self.assertEqual(len(points), 1)
                    self.assertLess(records[0]["skin_offset_m"], 5.0e-16)

            _write_point_csv(
                csv_path,
                "inward",
                "fastener",
                location,
                -_unit(normal_a + normal_b),
                roll,
            )
            with self.assertRaisesRegex(ValueError, "outward skin normal"):
                feature_workflow.prepare_point_placements(
                    None,
                    surface,
                    coordinate_scale=INCH_M,
                    skin_limit_m=1.0e-10,
                    wavelength_m=0.3,
                    normal_tolerance_deg=180.0,
                    locations_csv=csv_path,
                    datasets={"fastener": dataset},
                    pattern_loader=lambda *_args, **_kwargs: object(),
                )


class TriaxialEllipsoidFeatureAssemblyRegression(unittest.TestCase):
    def _fixture(self, root):
        (
            center,
            axes,
            vertices,
            faces,
            triangles,
            face_normals,
            rings,
        ) = _ellipsoid_mesh()
        surface = root / "triaxial_ellipsoid.facet"
        _write_facet(surface, vertices, faces)

        # A compact feature sits strictly inside an oblique face.  The roll
        # reference deliberately has both tangent and normal components; the
        # production and oracle independently project it into the local plane.
        target = center + np.asarray([
            axes[0] * math.sin(math.radians(50.0)) * math.cos(math.radians(330.0)),
            axes[1] * math.sin(math.radians(50.0)) * math.sin(math.radians(330.0)),
            axes[2] * math.cos(math.radians(50.0)),
        ])
        target_face_index = int(np.argmin(np.linalg.norm(
            np.mean(triangles, axis=1) - target, axis=1
        )))
        target_triangle = triangles[target_face_index]
        point_location = (
            0.23 * target_triangle[0]
            + 0.31 * target_triangle[1]
            + 0.46 * target_triangle[2]
        )
        point_normal = face_normals[target_face_index]
        edge_tangent = _unit(target_triangle[1] - target_triangle[0])
        point_roll = _unit(edge_tangent + 0.37 * point_normal)

        # Four latitude-ring edges form one continuous seal across several
        # genuine facet boundaries.  Smooth ellipsoid gradients are a CAD
        # user's natural endpoint normals and fall between incident face
        # normals on this sufficiently fine mesh.
        ring = rings[3]
        ring_indices = [ring[index % len(ring)] for index in range(43, 48)]
        line_points = vertices[ring_indices]
        line_segments = np.stack((line_points[:-1], line_points[1:]), axis=1)
        line_endpoint_normals = []
        for segment in line_segments:
            endpoint = []
            for point in segment:
                endpoint.append(_unit((point - center) / (axes * axes)))
            line_endpoint_normals.append(endpoint)
        line_endpoint_normals = np.asarray(line_endpoint_normals)

        clean = _clean_faceted_po(triangles, face_normals)
        base = _write_external_body(root / "clean_ellipsoid.grim", clean)
        output = root / "ellipsoid_with_seal_and_fastener.grim"
        point_dataset = root / "anisotropic_fastener_delta.grim"
        line_dataset = root / "isotropic_seal_delta.grim"
        point_csv = root / "point_features.csv"
        line_csv = root / "line_features.csv"
        installed_line_coefficient = 0.043 - 0.017j
        _write_anisotropic_point_delta(point_dataset)
        _write_isotropic_line_delta(line_dataset, installed_line_coefficient)
        _write_point_csv(
            point_csv,
            "fastener_upper_001",
            "anisotropic_fastener",
            point_location,
            point_normal,
            point_roll,
        )
        _write_line_csv(
            line_csv,
            "door_seal_upper_001",
            "door_seal",
            line_segments,
            line_endpoint_normals,
        )
        request = feature_workflow.FeatureAssemblyRequest(
            base_grim=base,
            output_grim=output,
            coordinate_units="inches",
            surface_mesh=surface,
            surface_units="meters",
            point_locations_csv=point_csv,
            point_datasets={"anisotropic_fastener": point_dataset},
            line_locations_csv=line_csv,
            line_datasets={"door_seal": line_dataset},
            enabled_point_placement_ids=("fastener_upper_001",),
            enabled_line_ids=("door_seal_upper_001",),
            skin_tol_m=1.0e-8,
            skin_phase_tol_deg=0.1,
            normal_tol_deg=8.0,
        )
        return {
            "center": center,
            "axes": axes,
            "triangles": triangles,
            "face_normals": face_normals,
            "surface": surface,
            "base": base,
            "output": output,
            "point_dataset": point_dataset,
            "line_dataset": line_dataset,
            "point_csv": point_csv,
            "line_csv": line_csv,
            "point_location": point_location,
            "point_normal": point_normal,
            "point_roll": point_roll,
            "line_segments": line_segments,
            "line_endpoint_normals": line_endpoint_normals,
            "installed_line_coefficient": installed_line_coefficient,
            "clean": clean,
            "request": request,
        }

    def test_public_workflow_matches_independent_curved_body_feature_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            plan = feature_workflow.prepare_feature_assembly(fixture["request"])

            self.assertIsNone(plan.body_profile)
            self.assertIsNotNone(plan.surface)
            self.assertEqual(
                plan.dataset_requirements.point_instances,
                (("fastener_upper_001", "anisotropic_fastener"),),
            )
            self.assertEqual(
                plan.dataset_requirements.line_instances,
                (("door_seal_upper_001", "door_seal", 4),),
            )
            self.assertLess(plan.point_records[0]["skin_offset_m"], 2.0e-15)
            self.assertLess(plan.line_records[0]["max_skin_offset_m"], 2.0e-15)
            self.assertLess(plan.line_records[0]["max_normal_error_deg"], 8.0)
            np.testing.assert_allclose(
                plan.point_locations_cad_m["anisotropic_fastener"],
                fixture["point_location"][None, :],
                rtol=0.0,
                atol=3.0e-16,
            )
            np.testing.assert_allclose(
                plan.line_paths_cad_m["door_seal"]["door_seal_upper_001"],
                np.concatenate((
                    fixture["line_segments"][:, 0],
                    fixture["line_segments"][-1:, 1],
                )),
                rtol=0.0,
                atol=3.0e-16,
            )

            expected_point = _direct_anisotropic_point(
                fixture["point_location"],
                fixture["point_normal"],
                fixture["point_roll"],
            )
            expected_line = _direct_isotropic_line(
                fixture["line_segments"],
                fixture["line_endpoint_normals"],
                fixture["installed_line_coefficient"],
            )
            expected_feature = (expected_point + expected_line).reshape(
                len(AZIMUTHS_DEG), len(ELEVATIONS_DEG), 1, 3
            )
            expected_total = fixture["clean"] + expected_feature

            saved = feature_workflow.execute_feature_assembly(plan)
            self.assertEqual(Path(saved), fixture["output"].resolve())
            with np.load(fixture["base"], allow_pickle=False) as stored_base:
                clean_after = (
                    stored_base["rcs_amp_real"]
                    + 1j * stored_base["rcs_amp_imag"]
                )
            with np.load(saved, allow_pickle=False) as stored:
                actual = stored["rcs_amp_real"] + 1j * stored["rcs_amp_imag"]
                power = np.asarray(stored["rcs_power"], dtype=float)
                provenance = json.loads(str(stored["feature_provenance_json"]))

            np.testing.assert_array_equal(clean_after, fixture["clean"])
            feature_nrms = _normalized_complex_rms(
                expected_feature, actual - clean_after
            )
            total_nrms = _normalized_complex_rms(expected_total, actual)
            self.assertLess(feature_nrms, 2.0e-10)
            self.assertLess(total_nrms, 2.0e-11)
            np.testing.assert_allclose(
                power,
                4.0 * math.pi * np.abs(actual) ** 2,
                rtol=8.0 * np.finfo(np.float32).eps,
                atol=np.finfo(np.float32).tiny,
            )
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

            incoherent = 4.0 * math.pi * (
                np.abs(fixture["clean"]) ** 2
                + np.abs(expected_feature) ** 2
            )
            self.assertGreater(float(np.max(np.abs(power - incoherent))), 1.0e-5)

            wrong_location = _direct_anisotropic_point(
                fixture["point_location"] + 0.012 * fixture["point_normal"],
                fixture["point_normal"],
                fixture["point_roll"],
            )
            tangent_x = point_oracle._local_frame(
                fixture["point_normal"], fixture["point_roll"]
            )[:, 0]
            tangent_y = np.cross(fixture["point_normal"], tangent_x)
            wrong_roll = _direct_anisotropic_point(
                fixture["point_location"],
                fixture["point_normal"],
                math.cos(0.63) * tangent_x + math.sin(0.63) * tangent_y,
            )
            self.assertGreater(
                _normalized_complex_rms(expected_point, wrong_location), 0.25
            )
            self.assertGreater(
                _normalized_complex_rms(expected_point, wrong_roll), 0.10
            )

    def test_wrong_skin_location_normal_and_frame_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            original = (
                fixture["point_location"].copy(),
                fixture["point_normal"].copy(),
                fixture["point_roll"].copy(),
            )
            cases = (
                (
                    "off_skin",
                    original[0] + 5.0e-4 * original[1],
                    original[1],
                    original[2],
                    "off the skin",
                ),
                (
                    "inward",
                    original[0],
                    -original[1],
                    original[2],
                    "outward skin normal",
                ),
                (
                    "parallel_roll",
                    original[0],
                    original[1],
                    original[1],
                    "parallel",
                ),
            )
            for label, location, normal, roll, message in cases:
                with self.subTest(label=label):
                    _write_point_csv(
                        fixture["point_csv"],
                        "fastener_upper_001",
                        "anisotropic_fastener",
                        location,
                        normal,
                        roll,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        feature_workflow.prepare_feature_assembly(
                            fixture["request"]
                        )

            _write_point_csv(
                fixture["point_csv"],
                "fastener_upper_001",
                "anisotropic_fastener",
                *original,
            )
            wrong_line_normals = -fixture["line_endpoint_normals"]
            _write_line_csv(
                fixture["line_csv"],
                "door_seal_upper_001",
                "door_seal",
                fixture["line_segments"],
                wrong_line_normals,
            )
            with self.assertRaisesRegex(ValueError, "outward skin normal"):
                feature_workflow.prepare_feature_assembly(fixture["request"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
