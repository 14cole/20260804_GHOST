#!/usr/bin/env python3
"""Physics regression for a placed 2-D gap on a clean BoR body.

The direct reference is a PEC cylinder whose generatrix contains an explicit
circumferential groove.  The reduced-order path starts from the clean cylinder,
forms a certified ``featured - clean`` 2-D coupon coefficient, expands that
coefficient around a ring with endpoint normals, and calls the public
``feature_sum.sum_features`` API.  No fitted amplitude, phase, or coordinate
correction is applied.

The fixed mesh is intentionally modest enough for the test suite.  Its exact
element counts are tied to the independent refinement evidence documented in
``geometry_tests/ring_gap_reconstruction/README.md``.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

from bor_solver import BOR_LINEAR_BACKWARD_ERROR_MAX, solve_bor  # noqa: E402
from feature_sum import directions_from_aspect_roll, sum_features  # noqa: E402
from line_expand import seam_coefficients_from_2d  # noqa: E402


FREQUENCY_GHZ = 1.0
FREQUENCY_HZ = FREQUENCY_GHZ * 1.0e9
RADIUS_M = 0.08
LENGTH_M = 0.30
GAP_WIDTH_M = 0.02
GAP_DEPTH_M = 0.01
ASPECTS_DEG = np.asarray([60.0, 75.0, 90.0, 105.0, 120.0])
COEFFICIENT_ANGLES_DEG = np.arange(0.0, 181.0, 15.0)
RING_SEGMENTS = 256

# These are the fixed meshes covered by the refinement evidence in the fixture
# README.  Matching vertices at +/- GAP_WIDTH_M/2 isolate the groove band.
CLEAN_VERTICES = [
    (0.0, LENGTH_M / 2.0),
    (RADIUS_M, LENGTH_M / 2.0),
    (RADIUS_M, GAP_WIDTH_M / 2.0),
    (RADIUS_M, -GAP_WIDTH_M / 2.0),
    (RADIUS_M, -LENGTH_M / 2.0),
    (0.0, -LENGTH_M / 2.0),
]
CLEAN_EDGE_COUNTS = [4, 10, 2, 10, 4]
GROOVED_VERTICES = [
    (0.0, LENGTH_M / 2.0),
    (RADIUS_M, LENGTH_M / 2.0),
    (RADIUS_M, GAP_WIDTH_M / 2.0),
    (RADIUS_M - GAP_DEPTH_M, GAP_WIDTH_M / 2.0),
    (RADIUS_M - GAP_DEPTH_M, -GAP_WIDTH_M / 2.0),
    (RADIUS_M, -GAP_WIDTH_M / 2.0),
    (RADIUS_M, -LENGTH_M / 2.0),
    (0.0, -LENGTH_M / 2.0),
]
GROOVED_EDGE_COUNTS = [4, 10, 3, 4, 3, 10, 4]


def _subdivide(vertices, edge_counts):
    """Subdivide each profile edge while retaining shared vertices once."""

    points = []
    for count, (start, stop) in zip(
        edge_counts, zip(vertices[:-1], vertices[1:])
    ):
        start = np.asarray(start, dtype=float)
        stop = np.asarray(stop, dtype=float)
        if not points:
            points.append(start)
        points.extend(
            start + (stop - start) * (index / count)
            for index in range(1, count + 1)
        )
    return np.asarray(points, dtype=float)


def _pec_coupon(vertices, name, panels_per_primitive=10):
    """Return a clockwise TYPE-2 PEC coupon with N panels per primitive."""

    return {
        "title": name,
        "segments": [{
            "name": name,
            "seg_type": 2,
            "properties": [
                "2", str(int(panels_per_primitive)), "0", "0", "0"
            ],
            "point_pairs": [
                {
                    "x1": float(start[0]),
                    "y1": float(start[1]),
                    "x2": float(stop[0]),
                    "y2": float(stop[1]),
                }
                for start, stop in zip(vertices[:-1], vertices[1:])
            ],
        }],
        "ibcs": [],
        "dielectrics": [],
    }


def _ring_with_endpoint_normals():
    """Build the ring in the solver frame with its outward radial normals.

    Decreasing azimuth is deliberate: tangent cross radial normal points along
    +solver-z, so the coupon's +x direction maps to the BoR axis.
    """

    angle = np.linspace(0.0, -2.0 * math.pi, RING_SEGMENTS + 1)
    points = np.column_stack((
        RADIUS_M * np.cos(angle),
        RADIUS_M * np.sin(angle),
        np.zeros_like(angle),
    ))
    radial = np.column_stack((
        np.cos(angle), np.sin(angle), np.zeros_like(angle)
    ))
    return (
        np.stack((points[:-1], points[1:]), axis=1),
        np.stack((radial[:-1], radial[1:]), axis=1),
    )


def _normalized_complex_rms(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


def _complex_coherence(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    return float(
        abs(np.vdot(reference, estimate))
        / (np.linalg.norm(reference) * np.linalg.norm(estimate))
    )


def _active_point_errors(reference, estimate, floor_db=-40.0):
    """Return direct magnitude p95 and phase RMS away from reference nulls."""

    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    active = (
        np.abs(reference)
        >= np.max(np.abs(reference)) * 10.0 ** (float(floor_db) / 20.0)
    )
    tiny = np.finfo(float).tiny
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(estimate[active]), tiny)
        / np.maximum(np.abs(reference[active]), tiny)
    )
    phase_deg = np.degrees(np.angle(
        estimate[active] * np.conjugate(reference[active])
    ))
    return (
        float(np.percentile(np.abs(magnitude_db), 95.0)),
        float(np.sqrt(np.mean(phase_deg ** 2))),
    )


class RingGapFeatureReconstructionTests(unittest.TestCase):
    """Compare direct grooved geometry with a placed certified 2-D delta."""

    @classmethod
    def setUpClass(cls):
        cls.clean_profile = _subdivide(CLEAN_VERTICES, CLEAN_EDGE_COUNTS)
        cls.grooved_profile = _subdivide(
            GROOVED_VERTICES, GROOVED_EDGE_COUNTS
        )

        bor_kwargs = {
            "formulation": "cfie",
            "n_modes": 8,
            "workers": 1,
            "assembly": "tables",
            "table_precision": "double",
        }
        cls.clean = solve_bor(
            cls.clean_profile,
            FREQUENCY_HZ,
            ASPECTS_DEG,
            **bor_kwargs,
        )
        cls.explicit = solve_bor(
            cls.grooved_profile,
            FREQUENCY_HZ,
            ASPECTS_DEG,
            **bor_kwargs,
        )

        coupon_half_width = 0.16
        coupon_depth = 0.10
        clean_coupon_vertices = [
            (-coupon_half_width, 0.0),
            (-GAP_WIDTH_M / 2.0, 0.0),
            (GAP_WIDTH_M / 2.0, 0.0),
            (coupon_half_width, 0.0),
            (coupon_half_width, -coupon_depth),
            (-coupon_half_width, -coupon_depth),
            (-coupon_half_width, 0.0),
        ]
        grooved_coupon_vertices = [
            (-coupon_half_width, 0.0),
            (-GAP_WIDTH_M / 2.0, 0.0),
            (-GAP_WIDTH_M / 2.0, -GAP_DEPTH_M),
            (GAP_WIDTH_M / 2.0, -GAP_DEPTH_M),
            (GAP_WIDTH_M / 2.0, 0.0),
            (coupon_half_width, 0.0),
            (coupon_half_width, -coupon_depth),
            (-coupon_half_width, -coupon_depth),
            (-coupon_half_width, 0.0),
        ]
        cls.coefficients = seam_coefficients_from_2d(
            _pec_coupon(grooved_coupon_vertices, "grooved_coupon"),
            _pec_coupon(clean_coupon_vertices, "clean_coupon"),
            FREQUENCY_GHZ,
            COEFFICIENT_ANGLES_DEG,
            geometry_units="meters",
            label="20 mm wide by 10 mm deep PEC groove",
        )

        ring, endpoint_normals = _ring_with_endpoint_normals()
        directions, _, _ = directions_from_aspect_roll(ASPECTS_DEG, [0.0])
        # Do not pass psi_tm_deg or psi_te_deg here.  The production API's
        # legacy production defaults are part of the reconstruction being tested.
        cls.placed = sum_features(
            cls.clean,
            [{
                "delta": cls.coefficients,
                "perimeter": ring,
                "segment_normals": endpoint_normals,
                "kind": "delta",
            }],
            directions,
            FREQUENCY_GHZ,
        )

        cls.truth_delta = {}
        cls.placed_delta = {}
        for channel in ("vv", "hh"):
            clean = np.asarray(cls.clean[f"amp_{channel}"], dtype=np.complex128)
            explicit = np.asarray(
                cls.explicit[f"amp_{channel}"], dtype=np.complex128
            )
            reconstructed = np.asarray(
                cls.placed[f"amp_{channel}"], dtype=np.complex128
            )
            cls.truth_delta[channel] = explicit - clean
            cls.placed_delta[channel] = reconstructed - clean

    def test_fixed_fixture_and_solver_quality_contracts(self):
        self.assertEqual(len(self.clean_profile) - 1, 30)
        self.assertEqual(len(self.grooved_profile) - 1, 38)
        np.testing.assert_array_equal(
            np.asarray(self.clean["theta_deg"], dtype=float), ASPECTS_DEG
        )
        np.testing.assert_array_equal(
            np.asarray(self.explicit["theta_deg"], dtype=float), ASPECTS_DEG
        )

        for label, result in (("clean", self.clean), ("explicit", self.explicit)):
            self.assertTrue(result["mode_converged"], msg=label)
            self.assertLessEqual(
                result["linear_backward_error"],
                BOR_LINEAR_BACKWARD_ERROR_MAX,
                msg=label,
            )
            for channel in ("vv", "hh"):
                amplitude = np.asarray(
                    result[f"amp_{channel}"], dtype=np.complex128
                )
                sigma = np.asarray(result[f"sigma_{channel}"], dtype=float)
                self.assertTrue(np.all(np.isfinite(amplitude)), msg=label)
                self.assertTrue(np.all(np.isfinite(sigma)), msg=label)
                np.testing.assert_allclose(
                    sigma,
                    4.0 * math.pi * np.abs(amplitude) ** 2,
                    rtol=2.0e-14,
                    atol=0.0,
                    err_msg=f"{label} {channel}",
                )

        self.assertEqual(
            self.coefficients.phi_deg.tolist(),
            COEFFICIENT_ANGLES_DEG.tolist(),
        )
        for values in (
            self.coefficients.dA_tm,
            self.coefficients.dA_te,
        ):
            self.assertTrue(np.all(np.isfinite(values)))

    def test_placed_output_is_finite_physical_and_axisymmetric(self):
        co_polarized_peak = max(
            float(np.max(np.abs(self.placed["amp_vv"]))),
            float(np.max(np.abs(self.placed["amp_hh"]))),
        )
        for channel in ("vv", "hh", "vh"):
            amplitude = np.asarray(
                self.placed[f"amp_{channel}"], dtype=np.complex128
            )
            sigma = np.asarray(self.placed[f"sigma_{channel}"], dtype=float)
            self.assertTrue(np.all(np.isfinite(amplitude)), msg=channel)
            self.assertTrue(np.all(np.isfinite(sigma)), msg=channel)
            np.testing.assert_allclose(
                sigma,
                4.0 * math.pi * np.abs(amplitude) ** 2,
                rtol=2.0e-14,
                atol=0.0,
                err_msg=channel,
            )

        # A complete circumferential ring on an axisymmetric body cannot
        # produce monostatic cross-polarization.  This also catches incorrect
        # ring ordering, endpoint normals, or local polarization rotation.
        vh_peak = float(np.max(np.abs(self.placed["amp_vh"])))
        self.assertLess(vh_peak, 1.0e-12 * co_polarized_peak)

    def test_clean_explicit_and_placed_fields_have_fore_aft_symmetry(self):
        # The geometry and centered circumferential ring are mirror symmetric.
        # Reversal pairs 60 with 120 degrees and 75 with 105 degrees while the
        # broadside sample maps to itself.  Compare complex fields so a hidden
        # phase-origin or placement-direction error cannot pass on RCS alone.
        for label, result in (
            ("clean", self.clean),
            ("explicit", self.explicit),
            ("placed", self.placed),
        ):
            for channel in ("vv", "hh"):
                amplitude = np.asarray(
                    result[f"amp_{channel}"], dtype=np.complex128
                )
                np.testing.assert_allclose(
                    amplitude,
                    amplitude[::-1],
                    rtol=2.0e-11,
                    atol=2.0e-12,
                    err_msg=f"{label} {channel}",
                )

    def test_unfitted_whole_field_matches_explicit_groove(self):
        reference = np.concatenate((
            np.asarray(self.explicit["amp_vv"], dtype=np.complex128),
            np.asarray(self.explicit["amp_hh"], dtype=np.complex128),
        ))
        estimate = np.concatenate((
            np.asarray(self.placed["amp_vv"], dtype=np.complex128),
            np.asarray(self.placed["amp_hh"], dtype=np.complex128),
        ))
        self.assertLess(_normalized_complex_rms(reference, estimate), 0.05)

    def test_unfitted_feature_delta_preserves_complex_field(self):
        reference = np.concatenate((
            self.truth_delta["vv"], self.truth_delta["hh"]
        ))
        estimate = np.concatenate((
            self.placed_delta["vv"], self.placed_delta["hh"]
        ))
        self.assertLess(_normalized_complex_rms(reference, estimate), 0.50)
        self.assertGreater(_complex_coherence(reference, estimate), 0.93)

        for channel in ("vv", "hh"):
            magnitude_p95_db, phase_rms_deg = _active_point_errors(
                self.truth_delta[channel], self.placed_delta[channel]
            )
            self.assertLess(
                magnitude_p95_db,
                2.5,
                msg=f"{channel} magnitude p95 = {magnitude_p95_db:.6g} dB",
            )
            self.assertLess(
                phase_rms_deg,
                40.0,
                msg=f"{channel} phase RMS = {phase_rms_deg:.6g} deg",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
