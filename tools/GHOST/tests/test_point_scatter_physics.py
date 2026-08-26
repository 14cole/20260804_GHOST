#!/usr/bin/env python3
"""Independent physics checks for compact-feature placement on non-BoR bodies.

The production implementation rotates a tabulated local Jones matrix into the
vehicle/radar frames.  These tests use a separate Cartesian reciprocal-dyadic
oracle: a compact Rayleigh-like scatterer is represented by one complex,
symmetric 3-D tensor, and its monostatic response is evaluated directly as
``e_receive.T @ alpha @ e_transmit``.  The oracle never calls GHOST's
polarization or attitude helpers.

The end-to-end case uses a faceted rectangular plate as an external platform,
two fastener families, inch-valued placement CSV coordinates, and the public
feature-assembly service.  It therefore exercises the non-BoR path from files
through skin/normal validation, CAD-to-vehicle conversion, coherent phase,
polarization rotation, and final GRIM normalization.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

import components  # noqa: E402
import feature_sum  # noqa: E402
import feature_workflow  # noqa: E402
from line_expand import C0  # noqa: E402


POLARIZATIONS = np.asarray(["VV", "HH", "VH"])
CAD_TO_EARTH = np.asarray([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])


def _direction_and_radar_basis(azimuth_deg, elevation_deg):
    """Independent earth-frame coming-from direction and radar V/H basis."""

    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    direction = np.asarray([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ])
    horizontal = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0])
    vertical = np.cross(horizontal, direction)
    return direction, vertical, horizontal


def _local_spherical_basis(azimuth_deg, elevation_deg):
    """Documented cavity theta/phi basis, implemented independently."""

    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    cosine = math.cos(elevation)
    if abs(cosine) < 1.0e-12:
        # At a pole azimuth is undefined.  The compact-pattern contract fixes a
        # deterministic x/y transverse basis, matching its phase-origin frame.
        return np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])
    vertical = np.asarray([
        math.sin(elevation) * math.cos(azimuth),
        math.sin(elevation) * math.sin(azimuth),
        -cosine,
    ])
    horizontal = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0])
    return vertical, horizontal


def _jones_from_tensor(tensor, vertical, horizontal):
    basis = (np.asarray(vertical, dtype=float), np.asarray(horizontal, dtype=float))
    matrix = np.empty((2, 2), dtype=np.complex128)
    for receive in range(2):
        for transmit in range(2):
            matrix[receive, transmit] = (
                basis[receive] @ np.asarray(tensor, dtype=np.complex128)
                @ basis[transmit]
            )
    return matrix


def _frequency_tensor(reference_tensor, frequency_ghz):
    """Rayleigh-like k^2 scaling around a 1 GHz reference tensor."""

    return np.asarray(reference_tensor, dtype=np.complex128) * float(frequency_ghz) ** 2


def _pattern_amplitude(reference_tensor, azimuths, elevations, frequencies):
    amplitude = np.empty(
        (len(azimuths), len(elevations), len(frequencies), 3),
        dtype=np.complex128,
    )
    for azimuth_index, azimuth in enumerate(azimuths):
        for elevation_index, elevation in enumerate(elevations):
            vertical, horizontal = _local_spherical_basis(azimuth, elevation)
            for frequency_index, frequency in enumerate(frequencies):
                jones = _jones_from_tensor(
                    _frequency_tensor(reference_tensor, frequency),
                    vertical,
                    horizontal,
                )
                amplitude[azimuth_index, elevation_index, frequency_index] = (
                    jones[0, 0], jones[1, 1], jones[0, 1]
                )
    return amplitude


def _pattern_dict(reference_tensor, frequencies=(0.8, 1.25)):
    # Every regression query is a five-degree node, keeping this a placement
    # test rather than an angular-interpolation-accuracy test.
    azimuths = np.arange(0.0, 361.0, 5.0)
    elevations = np.arange(-90.0, 91.0, 5.0)
    frequencies = np.asarray(frequencies, dtype=float)
    return {
        "azimuths": azimuths,
        "elevations": elevations,
        "frequencies": frequencies,
        "polarizations": POLARIZATIONS.copy(),
        "amp": _pattern_amplitude(
            reference_tensor, azimuths, elevations, frequencies
        ),
        **feature_sum.point_pattern_convention_metadata(),
    }


def _write_point_grim(path, reference_tensor, frequencies):
    pattern = _pattern_dict(reference_tensor, frequencies)
    amplitude = np.asarray(pattern.pop("amp"), dtype=np.complex128)
    payload = {
        **pattern,
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
    }
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _write_component_grim(path, azimuths, elevations, frequencies, amplitude):
    amplitude = np.asarray(amplitude, dtype=np.complex128)
    payload = {
        "azimuths": np.asarray(azimuths, dtype=float),
        "elevations": np.asarray(elevations, dtype=float),
        "frequencies": np.asarray(frequencies, dtype=float),
        "polarizations": POLARIZATIONS.copy(),
        "combine_role": np.asarray("coherent"),
        "rcs_power": (4.0 * math.pi * np.abs(amplitude) ** 2).astype(np.float32),
        "rcs_phase": np.angle(amplitude).astype(np.float32),
        "rcs_domain": np.asarray("power_phase"),
        "power_domain": np.asarray("linear_rcs"),
        "source_path": np.asarray("analytic rectangular-plate fixture"),
        "history": np.asarray("independent clean non-BoR plate field"),
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
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _plate_field(azimuths, elevations, frequencies, width_m, length_m):
    """Independent physical-optics clean rectangular-plate approximation."""

    result = np.zeros(
        (len(azimuths), len(elevations), len(frequencies), 3),
        dtype=np.complex128,
    )
    area = float(width_m) * float(length_m)
    for azimuth_index, azimuth in enumerate(azimuths):
        for elevation_index, elevation in enumerate(elevations):
            direction, _vertical, _horizontal = _direction_and_radar_basis(
                azimuth, elevation
            )
            # Express the earth look in the CAD plate axes.  np.sinc(x/pi) is
            # sin(x)/x, the exact transform of each uniform rectangular edge.
            direction_cad = CAD_TO_EARTH.T @ direction
            cosine = max(float(direction_cad[2]), 0.0)
            for frequency_index, frequency in enumerate(frequencies):
                wave_number = 2.0 * math.pi * float(frequency) * 1.0e9 / C0
                wavelength = C0 / (float(frequency) * 1.0e9)
                aperture = (
                    np.sinc(wave_number * width_m * direction_cad[0] / math.pi)
                    * np.sinc(wave_number * length_m * direction_cad[1] / math.pi)
                )
                field = -1j * area / wavelength * cosine * aperture
                result[azimuth_index, elevation_index, frequency_index, 0] = field
                result[azimuth_index, elevation_index, frequency_index, 1] = field
    return result


def _local_frame(normal, roll_reference):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    local_x = np.asarray(roll_reference, dtype=float)
    local_x -= float(local_x @ normal) * normal
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(normal, local_x)
    return np.column_stack((local_x, local_y, normal))


def _direct_point_field(
    reference_tensor,
    location_cad_m,
    normal_cad,
    roll_cad,
    azimuths,
    elevations,
    frequencies,
):
    """Direct earth-frame dyadic result, independent of feature_sum."""

    frame_earth = CAD_TO_EARTH @ _local_frame(normal_cad, roll_cad)
    location_earth = CAD_TO_EARTH @ np.asarray(location_cad_m, dtype=float)
    normal_earth = frame_earth[:, 2]
    result = np.zeros(
        (len(azimuths), len(elevations), len(frequencies), 3),
        dtype=np.complex128,
    )
    for azimuth_index, azimuth in enumerate(azimuths):
        for elevation_index, elevation in enumerate(elevations):
            direction, vertical, horizontal = _direction_and_radar_basis(
                azimuth, elevation
            )
            if float(direction @ normal_earth) <= 0.0:
                continue
            for frequency_index, frequency in enumerate(frequencies):
                tensor_earth = (
                    frame_earth
                    @ _frequency_tensor(reference_tensor, frequency)
                    @ frame_earth.T
                )
                jones = _jones_from_tensor(
                    tensor_earth, vertical, horizontal
                )
                wave_number = 2.0 * math.pi * float(frequency) * 1.0e9 / C0
                phase = np.exp(
                    2j * wave_number * float(direction @ location_earth)
                )
                result[azimuth_index, elevation_index, frequency_index] = (
                    jones[0, 0] * phase,
                    jones[1, 1] * phase,
                    jones[0, 1] * phase,
                )
    return result


class PointScatterDyadicTests(unittest.TestCase):
    def test_azimuth_dependent_pole_is_rejected(self):
        reference_tensor = np.diag(np.asarray([
            0.006 + 0.001j,
            0.002 - 0.0005j,
            0.001 + 0.0002j,
        ]))
        for elevation_index, label in ((0, "-90"), (-1, "+90")):
            with self.subTest(elevation=label):
                pattern = _pattern_dict(reference_tensor)
                pattern["amp"][7, elevation_index, 0, 0] += 1.0e-3 + 2.0e-4j
                with self.assertRaisesRegex(
                    ValueError, "fixed local x/y pole basis"
                ):
                    feature_sum.prepare_point_pattern(pattern)

    def test_arbitrary_frame_matches_independent_reciprocal_dyadic(self):
        reference_tensor = np.asarray([
            [0.006 + 0.001j, 0.0012 - 0.0003j, -0.0007 + 0.0002j],
            [0.0012 - 0.0003j, 0.0035 - 0.0008j, 0.0005 + 0.0004j],
            [-0.0007 + 0.0002j, 0.0005 + 0.0004j, 0.002 + 0.0006j],
        ])
        pattern = feature_sum.prepare_point_pattern(_pattern_dict(reference_tensor))
        normal = np.asarray([1.0, -2.0, 3.0])
        roll = np.asarray([2.0, 1.0, 0.5])
        frame = _local_frame(normal, roll)
        local_looks = (
            (15.0, 10.0),
            (80.0, 25.0),
            (150.0, 40.0),
            (210.0, 60.0),
            (285.0, 75.0),
            # Azimuth is geometrically undefined at normal incidence.  A
            # canonical point pattern uses its fixed local x/y pole basis.
            (340.0, 90.0),
        )
        directions = []
        for azimuth, elevation in local_looks:
            azimuth_rad = math.radians(azimuth)
            elevation_rad = math.radians(elevation)
            local_direction = np.asarray([
                math.cos(elevation_rad) * math.cos(azimuth_rad),
                math.cos(elevation_rad) * math.sin(azimuth_rad),
                math.sin(elevation_rad),
            ])
            directions.append(frame @ local_direction)
        directions = np.asarray(directions)
        location = np.asarray([0.037, -0.021, 0.014])
        frequency = 1.25

        actual = feature_sum.point_scatterer_amplitude(
            pattern,
            location,
            normal,
            directions,
            frequency,
            roll_ref=roll,
        )

        tensor_body = frame @ _frequency_tensor(reference_tensor, frequency) @ frame.T
        wave_number = 2.0 * math.pi * frequency * 1.0e9 / C0
        expected = {key: [] for key in ("F_vv", "F_hh", "F_vh")}
        for direction in directions:
            azimuth = math.atan2(float(direction[1]), float(direction[0]))
            transverse = math.hypot(float(direction[0]), float(direction[1]))
            if transverse < 1.0e-12:
                vertical = np.asarray([1.0, 0.0, 0.0])
                horizontal = np.asarray([0.0, 1.0, 0.0])
            else:
                horizontal = np.asarray(
                    [-math.sin(azimuth), math.cos(azimuth), 0.0]
                )
                vertical = np.asarray([
                    direction[2] * math.cos(azimuth),
                    direction[2] * math.sin(azimuth),
                    -transverse,
                ])
            jones = _jones_from_tensor(tensor_body, vertical, horizontal)
            phase = np.exp(2j * wave_number * float(direction @ location))
            expected["F_vv"].append(jones[0, 0] * phase)
            expected["F_hh"].append(jones[1, 1] * phase)
            expected["F_vh"].append(jones[0, 1] * phase)

        for channel in expected:
            np.testing.assert_allclose(
                actual[channel], expected[channel], rtol=2.0e-11, atol=2.0e-13
            )

    def test_round_fastener_is_roll_invariant_and_backside_is_dark(self):
        reference_tensor = np.diag(
            np.asarray([0.004 + 0.001j, 0.004 + 0.001j, 0.0015 - 0.0002j])
        )
        pattern = feature_sum.prepare_point_pattern(_pattern_dict(reference_tensor))
        normal = np.asarray([0.0, 0.0, 1.0])
        directions = np.asarray([
            [math.sqrt(0.5), 0.0, math.sqrt(0.5)],
            [0.0, math.sqrt(0.5), math.sqrt(0.5)],
            [math.sqrt(0.5), 0.0, -math.sqrt(0.5)],
        ])
        common = {
            "pattern": pattern,
            "location": [0.01, -0.02, 0.0],
            "aperture_normal": normal,
            "directions": directions,
            "frequency_ghz": 0.8,
        }
        zero_roll = feature_sum.point_scatterer_amplitude(
            roll_ref=[1.0, 0.0, 0.0], **common
        )
        arbitrary_roll = feature_sum.point_scatterer_amplitude(
            roll_ref=[math.cos(0.73), math.sin(0.73), 0.0], **common
        )
        for channel in ("F_vv", "F_hh", "F_vh"):
            np.testing.assert_allclose(
                arbitrary_roll[channel], zero_roll[channel],
                rtol=2.0e-11, atol=2.0e-13,
            )
            self.assertEqual(arbitrary_roll[channel][2], 0.0j)


class NonBorFastenerAssemblyRegression(unittest.TestCase):
    def test_clean_plate_plus_placed_fasteners_matches_direct_featured_truth(self):
        frequencies = np.asarray([0.8, 1.25])
        azimuths = np.asarray([0.0, 30.0, 75.0, 145.0, 220.0, 305.0])
        elevations = np.asarray([10.0, 25.0, 45.0, 70.0])
        width_m = 0.20
        length_m = 0.30
        clean_field = _plate_field(
            azimuths, elevations, frequencies, width_m, length_m
        )
        round_tensor = np.diag(np.asarray([
            0.004 + 0.0008j,
            0.004 + 0.0008j,
            0.0012 - 0.0003j,
        ]))
        slotted_tensor = np.asarray([
            [0.006 + 0.001j, 0.0007 - 0.0002j, 0.0003 + 0.0001j],
            [0.0007 - 0.0002j, 0.0025 - 0.0006j, -0.0004 + 0.0002j],
            [0.0003 + 0.0001j, -0.0004 + 0.0002j, 0.0018 + 0.0004j],
        ])
        round_location = np.asarray([1.25, -2.0, 0.0]) * 0.0254
        slot_location = np.asarray([-2.25, 3.0, 0.0]) * 0.0254
        slot_roll_deg = 35.0
        slot_roll = np.asarray([
            math.cos(math.radians(slot_roll_deg)),
            math.sin(math.radians(slot_roll_deg)),
            0.0,
        ])
        expected_feature = (
            _direct_point_field(
                round_tensor,
                round_location,
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                azimuths,
                elevations,
                frequencies,
            )
            + _direct_point_field(
                slotted_tensor,
                slot_location,
                [0.0, 0.0, 1.0],
                slot_roll,
                azimuths,
                elevations,
                frequencies,
            )
        )
        expected_total = clean_field + expected_feature

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean_rectangular_plate.grim"
            output = root / "plate_with_fasteners.grim"
            surface = root / "rectangular_plate.facet"
            locations = root / "fastener_locations.csv"
            round_pattern = root / "round_fastener_delta.grim"
            slot_pattern = root / "slotted_fastener_delta.grim"

            _write_component_grim(
                base, azimuths, elevations, frequencies, clean_field
            )
            _write_point_grim(round_pattern, round_tensor, frequencies)
            _write_point_grim(slot_pattern, slotted_tensor, frequencies)
            surface.write_text(
                "4 2\n"
                f"1 {-width_m / 2} {-length_m / 2} 0\n"
                f"2 {width_m / 2} {-length_m / 2} 0\n"
                f"3 {width_m / 2} {length_m / 2} 0\n"
                f"4 {-width_m / 2} {length_m / 2} 0\n"
                "1 1 2 3\n"
                "2 1 3 4\n",
                encoding="utf-8",
            )
            locations.write_text(
                "placement_id,dataset_id,x,y,z,nx,ny,nz,"
                "roll_x,roll_y,roll_z\n"
                "round_001,round_fastener,1.25,-2,0,0,0,1,1,0,0\n"
                "slot_001,slotted_fastener,-2.25,3,0,0,0,1,"
                f"{slot_roll[0]:.17g},{slot_roll[1]:.17g},0\n",
                encoding="utf-8",
            )

            request = feature_workflow.FeatureAssemblyRequest(
                base_grim=base,
                output_grim=output,
                coordinate_units="inches",
                surface_mesh=surface,
                surface_units="meters",
                point_locations_csv=locations,
                point_datasets={
                    "round_fastener": round_pattern,
                    "slotted_fastener": slot_pattern,
                },
                skin_tol_m=1.0e-6,
                skin_phase_tol_deg=1.0,
                normal_tol_deg=0.1,
            )
            plan = feature_workflow.prepare_feature_assembly(request)
            self.assertIsNone(plan.body_profile)
            self.assertIsNotNone(plan.surface)
            self.assertEqual(len(plan.point_placements), 2)
            self.assertEqual(plan.dataset_requirements.point_placement_count, 2)
            np.testing.assert_allclose(
                plan.point_locations_cad_m["round_fastener"],
                round_location[None, :],
                rtol=0.0,
                atol=2.0e-16,
            )
            saved = feature_workflow.execute_feature_assembly(plan)
            self.assertEqual(Path(saved), output.resolve())

            with np.load(base, allow_pickle=False) as clean_payload:
                stored_clean = (
                    clean_payload["rcs_amp_real"]
                    + 1j * clean_payload["rcs_amp_imag"]
                )
            with np.load(output, allow_pickle=False) as featured_payload:
                stored_total = (
                    featured_payload["rcs_amp_real"]
                    + 1j * featured_payload["rcs_amp_imag"]
                )
                stored_power = np.asarray(featured_payload["rcs_power"], float)
                self.assertEqual(str(featured_payload["combine_role"]), "coherent")
                provenance = json.loads(str(featured_payload["feature_provenance_json"]))
                self.assertEqual(provenance[-1]["compact_feature_count"], 2)

            np.testing.assert_allclose(
                stored_clean, clean_field, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                stored_total - stored_clean,
                expected_feature,
                rtol=3.0e-11,
                atol=3.0e-13,
            )
            np.testing.assert_allclose(
                stored_total,
                expected_total,
                rtol=3.0e-11,
                atol=3.0e-13,
            )
            np.testing.assert_allclose(
                stored_power,
                4.0 * math.pi * np.abs(stored_total) ** 2,
                rtol=8.0 * np.finfo(np.float32).eps,
                atol=np.finfo(np.float32).tiny,
            )
            self.assertGreater(float(np.max(np.abs(expected_feature[..., 2]))), 1.0e-5)
            # The coherent answer must contain material interference with the
            # clean plate, not merely the sum of separate RCS powers.
            incoherent = (
                4.0 * math.pi * np.abs(clean_field) ** 2
                + 4.0 * math.pi * np.abs(expected_feature) ** 2
            )
            self.assertGreater(
                float(np.max(np.abs(stored_power - incoherent))), 1.0e-5
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
