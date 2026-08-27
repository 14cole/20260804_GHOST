"""Independent geometry/Jones oracles for Wedge-to-Conic conversion."""

from __future__ import annotations

import math
import unittest

import numpy as np

from grim_dataset import (
    CONIC_VH_BASIS_CONVENTION,
    RcsGrid,
    rotate_wedge_jones_to_conic,
    wedge_to_conic_geometry_deg,
)
from grim_python import wedge_to_conic as scripted_wedge_to_conic


def _rotation_y(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])


def _rotation_z(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _oracle_frames(phi_deg, tau_deg):
    """Frames derived directly from the two physical range attitudes."""

    phi = math.radians(float(phi_deg))
    tau = math.radians(float(tau_deg))
    body_to_world = _rotation_z(phi) @ _rotation_y(tau)
    direction = body_to_world.T @ np.asarray([1.0, 0.0, 0.0])
    wedge_v = body_to_world.T @ np.asarray([0.0, 0.0, 1.0])
    wedge_h = body_to_world.T @ np.asarray([0.0, 1.0, 0.0])
    longitude = math.atan2(direction[1], direction[0])
    latitude = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
    conic_v = np.asarray([
        -math.sin(latitude) * math.cos(longitude),
        -math.sin(latitude) * math.sin(longitude),
        math.cos(latitude),
    ])
    conic_h = np.asarray([
        -math.sin(longitude), math.cos(longitude), 0.0
    ])
    return direction, np.stack((wedge_v, wedge_h)), np.stack((conic_v, conic_h))


def _oracle_inverse(longitude_deg, latitude_deg):
    longitude = math.radians(float(longitude_deg))
    latitude = math.radians(float(latitude_deg))
    x = math.cos(latitude) * math.cos(longitude)
    y = math.cos(latitude) * math.sin(longitude)
    z = math.sin(latitude)
    sin_phi = -y
    cos_phi = math.copysign(
        math.sqrt(max(0.0, 1.0 - sin_phi * sin_phi)),
        x if abs(x) > 1.0e-15 else 1.0,
    )
    phi = math.atan2(sin_phi, cos_phi)
    branch = -1.0 if cos_phi < 0.0 else 1.0
    tau = math.atan2(branch * z, branch * x)
    return math.degrees(phi), math.degrees(tau)


class WedgePhysicsTests(unittest.TestCase):
    def test_direction_map_matches_attitude_matrix_oracle(self):
        for phi in (-170.0, -83.0, -21.0, 0.0, 47.0, 121.0, 179.0):
            for tau in (-32.0, -7.0, 0.0, 19.0, 41.0):
                direction, _old, _new = _oracle_frames(phi, tau)
                expected_lon = math.degrees(math.atan2(direction[1], direction[0]))
                expected_lat = math.degrees(math.asin(direction[2]))
                actual_lon, actual_lat = wedge_to_conic_geometry_deg(phi, tau)
                self.assertAlmostEqual(float(actual_lon), expected_lon, places=12)
                self.assertAlmostEqual(float(actual_lat), expected_lat, places=12)

    def test_jones_rotation_matches_object_fixed_dyadic_oracle(self):
        dyadic = np.asarray([
            [2.0 + 0.3j, 0.2 - 0.1j, -0.4 + 0.05j],
            [0.2 - 0.1j, 0.9 - 0.2j, 0.35 + 0.15j],
            [-0.4 + 0.05j, 0.35 + 0.15j, 1.4 + 0.4j],
        ])
        for phi, tau in ((37.0, 24.0), (-63.0, 18.0), (132.0, -27.0)):
            _direction, wedge_basis, conic_basis = _oracle_frames(phi, tau)
            measured = wedge_basis @ dyadic @ wedge_basis.T
            expected = conic_basis @ dyadic @ conic_basis.T
            actual = rotate_wedge_jones_to_conic(measured, phi, tau)
            np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)

    def _constant_jones_grid(self, *, elevations=(-30.0, -15.0, 0.0, 15.0, 30.0)):
        azimuths = np.arange(-180.0, 180.0, 30.0)
        elevations = np.asarray(elevations, dtype=float)
        matrix = np.asarray([
            [2.0 + 0.4j, 0.35 - 0.2j],
            [0.35 - 0.2j, 0.8 + 0.1j],
        ])
        field = np.empty((azimuths.size, elevations.size, 1, 3), dtype=np.complex128)
        field[..., 0] = matrix[0, 0]
        field[..., 1] = matrix[1, 1]
        field[..., 2] = matrix[0, 1]
        grid = RcsGrid(
            azimuths,
            elevations,
            [10.0],
            ["VV", "HH", "VH"],
            rcs=field,
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "wedge_turntable",
            },
        )
        return grid, matrix

    def test_end_to_end_normal_grid_uses_inverse_map_and_jones_rotation(self):
        source, measured = self._constant_jones_grid()
        converted = source.convert_wedge_to_conic()

        lon_index = int(np.flatnonzero(np.isclose(converted.azimuths, 30.0))[0])
        lat_index = int(np.flatnonzero(np.isclose(converted.elevations, 15.0))[0])
        phi, tau = _oracle_inverse(30.0, 15.0)
        _direction, wedge_basis, conic_basis = _oracle_frames(phi, tau)
        change = wedge_basis @ conic_basis.T
        expected = change.T @ measured @ change
        actual = converted.rcs[lon_index, lat_index, 0]
        np.testing.assert_allclose(
            actual,
            [expected[0, 0], expected[1, 1], expected[0, 1]],
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        self.assertEqual(converted.angular_coordinate_system(), "conic")
        self.assertEqual(
            converted.units["polarization_basis"], CONIC_VH_BASIS_CONVENTION
        )

        side = int(np.flatnonzero(np.isclose(converted.azimuths, 90.0))[0])
        self.assertTrue(np.all(np.isnan(converted.rcs_power[side, lat_index, 0])))

    def test_single_tilt_and_copol_only_fail_closed(self):
        single, _matrix = self._constant_jones_grid(elevations=(15.0,))
        with self.assertRaisesRegex(ValueError, "One fixed wedge tilt"):
            single.convert_wedge_to_conic()

        full, _matrix = self._constant_jones_grid()
        copol = RcsGrid(
            full.azimuths,
            full.elevations,
            full.frequencies,
            ["VV", "HH"],
            rcs=full.rcs[..., :2],
            units=dict(full.units),
        )
        with self.assertRaisesRegex(ValueError, "cannot rotate VV/HH alone"):
            copol.convert_wedge_to_conic()
        assumed = copol.convert_wedge_to_conic(
            assume_missing_cross_pol_zero=True
        )
        self.assertIn(
            "missing VH=HV=0",
            assumed.extra["wedge_to_conic_cross_pol_treatment"],
        )

    def test_headless_replay_uses_the_same_physical_converter(self):
        source, _matrix = self._constant_jones_grid()
        direct = source.convert_wedge_to_conic()
        replayed = scripted_wedge_to_conic(source, mode="regrid")
        np.testing.assert_array_equal(replayed.azimuths, direct.azimuths)
        np.testing.assert_array_equal(replayed.elevations, direct.elevations)
        np.testing.assert_allclose(
            replayed.rcs_power, direct.rcs_power, rtol=0.0, atol=0.0,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            replayed.rcs_phase, direct.rcs_phase, rtol=0.0, atol=0.0,
            equal_nan=True,
        )
        with self.assertRaisesRegex(ValueError, "not physically representable"):
            scripted_wedge_to_conic(source, mode="relabel")


if __name__ == "__main__":
    unittest.main()
