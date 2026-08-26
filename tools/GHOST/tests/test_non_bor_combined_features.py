#!/usr/bin/env python3
"""Combined non-BoR line and point reconstruction on a closed PEC box.

The clean box and explicit finite-width line use an independent analytic
thin-sheet/physical-optics oracle.  The two fasteners use the independent
Cartesian reciprocal-dyadic oracle from ``test_point_scatter_physics``.
Production receives only the clean external GRIM, one line centerline, and two
local point patterns through ``add_features_to_monostatic_grim``.

This certifies placement, frames, coherent phase, polarization, and artifact
combination within those reduced-order models.  It does not claim full-wave
body-feature mutual coupling or multiple scattering.
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
from grim_io import _save_grim_npz  # noqa: E402
from line_expand import C0, SeamCoefficients  # noqa: E402
import test_line_feature_non_bor_physics as line_oracle  # noqa: E402
import test_point_scatter_physics as point_oracle  # noqa: E402


FREQUENCY_GHZ = 2.0
WAVELENGTH_M = C0 / (FREQUENCY_GHZ * 1.0e9)
WAVE_NUMBER = 2.0 * math.pi / WAVELENGTH_M
AZIMUTHS_DEG = np.asarray([0.0])
ELEVATIONS_DEG = np.asarray([30.0, 50.0, 70.0])


def _directions():
    return np.asarray([
        point_oracle._direction_and_radar_basis(0.0, elevation)[0]
        for elevation in ELEVATIONS_DEG
    ])


def _closed_box_field(center, dimensions, directions):
    """Independent lit-face scalar PO field of all six box faces."""

    center = np.asarray(center, dtype=float)
    lx, ly, lz = (float(value) for value in dimensions)
    face_definitions = (
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], ly, lz, lx / 2.0),
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], ly, lz, lx / 2.0),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], lx, lz, ly / 2.0),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], lx, lz, ly / 2.0),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], lx, ly, lz / 2.0),
        ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], lx, ly, lz / 2.0),
    )
    scalar = np.zeros(len(directions), dtype=np.complex128)
    reflectivity = 1.0 + 0.0j
    for normal, axis_u, axis_v, length_u, length_v, offset in face_definitions:
        normal = np.asarray(normal, dtype=float)
        illumination = np.maximum(np.asarray(directions) @ normal, 0.0)
        face_center = center + float(offset) * normal
        scalar += (
            -1j
            * reflectivity
            / WAVELENGTH_M
            * illumination
            * line_oracle._rect_integral(
                face_center,
                axis_u,
                axis_v,
                length_u,
                length_v,
                directions,
            )
        )
    field = np.zeros((1, len(ELEVATIONS_DEG), 1, 3), dtype=np.complex128)
    field[0, :, 0, 0] = scalar
    field[0, :, 0, 1] = scalar
    return field


def _direct_point_field(reference_tensor, location, normal, roll_reference):
    """Independent global Cartesian-dyadic fastener field."""

    frame = point_oracle._local_frame(normal, roll_reference)
    tensor = (
        frame
        @ point_oracle._frequency_tensor(reference_tensor, FREQUENCY_GHZ)
        @ frame.T
    )
    location = np.asarray(location, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    result = np.zeros((len(ELEVATIONS_DEG), 3), dtype=np.complex128)
    for index, elevation in enumerate(ELEVATIONS_DEG):
        direction, vertical, horizontal = (
            point_oracle._direction_and_radar_basis(0.0, elevation)
        )
        if float(direction @ normal) <= 0.0:
            continue
        jones = point_oracle._jones_from_tensor(
            tensor, vertical, horizontal
        )
        phase = np.exp(
            2j * WAVE_NUMBER * float(direction @ location)
        )
        result[index] = (
            jones[0, 0] * phase,
            jones[1, 1] * phase,
            jones[0, 1] * phase,
        )
    return result


def _write_external_body(path, amplitude):
    amplitude = np.asarray(amplitude, dtype=np.complex128)
    payload = {
        "azimuths": AZIMUTHS_DEG,
        "elevations": ELEVATIONS_DEG,
        "frequencies": np.asarray([FREQUENCY_GHZ]),
        "polarizations": np.asarray(["VV", "HH", "VH"]),
        "combine_role": np.asarray("coherent"),
        "rcs_power": (
            4.0 * math.pi * np.abs(amplitude) ** 2
        ).astype(np.float32),
        "rcs_phase": np.angle(amplitude).astype(np.float32),
        "rcs_domain": np.asarray("power_phase"),
        "power_domain": np.asarray("linear_rcs"),
        "source_path": np.asarray("independent analytic closed-box fixture"),
        "history": np.asarray("clean non-BoR closed PEC box approximation"),
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


class CombinedClosedBoxFeatureTests(unittest.TestCase):
    def test_line_and_orthogonal_face_fasteners_match_explicit_truth(self):
        directions = _directions()
        box_center = np.asarray([0.07, -0.05, 0.14])
        box_dimensions = np.asarray([0.40, 0.32, 0.24])
        clean = _closed_box_field(box_center, box_dimensions, directions)

        line_normal = np.asarray([0.0, 0.0, 1.0])
        point_normal = np.asarray([1.0, 0.0, 0.0])
        self.assertEqual(float(line_normal @ point_normal), 0.0)

        line_center = box_center + np.asarray([
            0.055, -0.028, box_dimensions[2] / 2.0
        ])
        self.assertAlmostEqual(
            line_center[2], box_center[2] + box_dimensions[2] / 2.0
        )
        line_length = 0.235
        line_width = 0.003
        line_contrast = -0.63 + 0.24j
        line_start = line_center - np.asarray([0.0, line_length / 2.0, 0.0])
        line_stop = line_center + np.asarray([0.0, line_length / 2.0, 0.0])
        line_segments = np.asarray([[line_start, line_stop]])
        line_normals = line_oracle._constant_endpoint_normals(
            line_segments, line_normal
        )
        coefficient_angles = np.arange(0.0, 181.0, 5.0)
        per_length = (
            line_contrast
            * line_width
            / WAVELENGTH_M
            * np.sinc(
                WAVE_NUMBER
                * line_width
                * np.cos(np.radians(coefficient_angles))
                / math.pi
            )
        )
        line_coefficient = SeamCoefficients(
            FREQUENCY_GHZ,
            coefficient_angles,
            4.0 * math.pi * per_length,
            4.0 * math.pi * per_length,
            label="top-face finite-width door seal",
        )
        explicit_line_scalar = (
            line_contrast
            / WAVELENGTH_M
            * line_oracle._rect_integral(
                line_center,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                line_width,
                line_length,
                directions,
            )
        )
        explicit_line = np.column_stack((
            explicit_line_scalar,
            explicit_line_scalar,
            np.zeros_like(explicit_line_scalar),
        ))

        round_tensor = np.diag(np.asarray([
            0.0042 + 0.0008j,
            0.0042 + 0.0008j,
            0.0011 - 0.0003j,
        ]))
        slotted_tensor = np.asarray([
            [0.0061 + 0.0010j, 0.0008 - 0.0002j, 0.0003 + 0.0001j],
            [0.0008 - 0.0002j, 0.0027 - 0.0006j, -0.0004 + 0.0002j],
            [0.0003 + 0.0001j, -0.0004 + 0.0002j, 0.0019 + 0.0004j],
        ])
        point_roll = np.asarray([0.0, 0.0, 1.0])
        side_x = box_center[0] + box_dimensions[0] / 2.0
        round_location = np.asarray([
            side_x, box_center[1] - 0.061, box_center[2] + 0.037
        ])
        slot_location = np.asarray([
            side_x, box_center[1] + 0.073, box_center[2] - 0.046
        ])
        self.assertEqual(round_location[0], side_x)
        self.assertEqual(slot_location[0], side_x)
        for location in (round_location, slot_location):
            self.assertLessEqual(
                abs(location[1] - box_center[1]), box_dimensions[1] / 2.0
            )
            self.assertLessEqual(
                abs(location[2] - box_center[2]), box_dimensions[2] / 2.0
            )
        explicit_round = _direct_point_field(
            round_tensor,
            round_location,
            point_normal,
            point_roll,
        )
        explicit_slot = _direct_point_field(
            slotted_tensor,
            slot_location,
            point_normal,
            point_roll,
        )
        expected_feature = explicit_line + explicit_round + explicit_slot
        expected_total = clean.copy()
        expected_total[0, :, 0, :] += expected_feature

        round_pattern = feature_sum.prepare_point_pattern(
            point_oracle._pattern_dict(
                round_tensor, frequencies=(FREQUENCY_GHZ,)
            )
        )
        slot_pattern = feature_sum.prepare_point_pattern(
            point_oracle._pattern_dict(
                slotted_tensor, frequencies=(FREQUENCY_GHZ,)
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _write_external_body(root / "clean_box.grim", clean)
            output = root / "box_with_line_and_fasteners.grim"
            feature_sum.add_features_to_monostatic_grim(
                str(base),
                str(output),
                placements=[{
                    "delta": line_coefficient,
                    "perimeter": line_segments,
                    "segment_normals": line_normals,
                    "kind": "delta",
                }],
                points=[
                    {
                        "pattern": round_pattern,
                        "location": round_location,
                        "aperture_normal": point_normal,
                        "roll_ref": point_roll,
                    },
                    {
                        "pattern": slot_pattern,
                        "location": slot_location,
                        "aperture_normal": point_normal,
                        "roll_ref": point_roll,
                    },
                ],
                radar_grid={
                    "frequencies_ghz": [FREQUENCY_GHZ],
                    "azimuths_deg": AZIMUTHS_DEG.tolist(),
                    "elevations_deg": ELEVATIONS_DEG.tolist(),
                    "axis_az_deg": 0.0,
                    "axis_el_deg": 90.0,
                    "roll_deg": 0.0,
                },
                psi_tm_deg=0.0,
                psi_te_deg=0.0,
                history="placed top-face line and side-face fasteners",
            )

            with np.load(base, allow_pickle=False) as stored_base:
                clean_after = (
                    stored_base["rcs_amp_real"]
                    + 1j * stored_base["rcs_amp_imag"]
                )
                np.testing.assert_array_equal(clean_after, clean)
            with np.load(output, allow_pickle=False) as stored:
                actual = (
                    stored["rcs_amp_real"] + 1j * stored["rcs_amp_imag"]
                )
                power = np.asarray(stored["rcs_power"], dtype=float)
                provenance = json.loads(str(stored["feature_provenance_json"]))
                self.assertNotIn("body_model_metadata_json", stored.files)

        actual_feature = actual - clean
        feature_nrms = line_oracle._normalized_complex_rms(
            expected_feature, actual_feature[0, :, 0, :]
        )
        whole_nrms = line_oracle._normalized_complex_rms(
            expected_total, actual
        )
        coherence = line_oracle._complex_coherence(
            expected_feature, actual_feature[0, :, 0, :]
        )
        self.assertLess(feature_nrms, 2.0e-10)
        self.assertLess(whole_nrms, 2.0e-11)
        self.assertGreater(coherence, 1.0 - 2.0e-12)
        self.assertGreater(float(np.max(np.abs(expected_feature[:, 2]))), 1.0e-5)

        np.testing.assert_allclose(
            power,
            4.0 * math.pi * np.abs(actual) ** 2,
            rtol=8.0 * np.finfo(np.float32).eps,
            atol=np.finfo(np.float32).tiny,
        )
        incoherent_power = 4.0 * math.pi * (
            np.abs(clean) ** 2
            + np.abs(explicit_line[None, :, None, :]) ** 2
            + np.abs(explicit_round[None, :, None, :]) ** 2
            + np.abs(explicit_slot[None, :, None, :]) ** 2
        )
        self.assertGreater(
            float(np.max(np.abs(power - incoherent_power))), 1.0e-4
        )
        self.assertEqual(provenance[-1]["line_feature_count"], 1)
        self.assertEqual(provenance[-1]["compact_feature_count"], 2)
        self.assertEqual(
            provenance[-1]["line_phase_mapping_deg"],
            {"TM": 0.0, "TE": 0.0},
        )
        self.assertFalse(
            provenance[-1]["model_scope"]["body_feature_mutual_coupling"]
        )
        self.assertFalse(
            provenance[-1]["model_scope"]["multiple_scattering"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
