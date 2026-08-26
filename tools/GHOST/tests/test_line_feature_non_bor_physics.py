#!/usr/bin/env python3
"""Non-BoR physics regressions for line-feature placement.

GHOST does not contain a general three-dimensional full-wave solver.  These
tests therefore use an independent analytic thin-sheet reference rather than
silently treating the line-expansion equation as its own ground truth.  The
reference integrates a finite rectangular host surface and explicit finite-
width feature strips.  Production reconstructs those strips from their
centerlines with ``sum_features`` or ``add_features_to_monostatic_grim``.

The straight gap case is exactly separable in the scalar physical-optics
model.  The closed door and folded-panel cases deliberately compare the line
model against finite-width geometry, so their small residual measures the
documented narrow-feature approximation (strip width and corner overlap).
No amplitude, phase, range, or coordinate fit is performed.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import feature_sum  # noqa: E402
from grim_io import _save_grim_npz  # noqa: E402
from line_expand import (  # noqa: E402
    C0,
    SeamCoefficients,
)


FREQUENCY_GHZ = 2.0
WAVE_NUMBER = 2.0 * math.pi * FREQUENCY_GHZ * 1.0e9 / C0
WAVELENGTH_M = C0 / (FREQUENCY_GHZ * 1.0e9)


def _unit(vector):
    values = np.asarray(vector, dtype=float)
    return values / np.linalg.norm(values)


def _rotation_matrix():
    """Fixed proper rotation for the tilted, non-axisymmetric panel cases."""

    ax, ay, az = np.radians([11.0, -17.0, 23.0])
    rx = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(ax), -math.sin(ax)],
        [0.0, math.sin(ax), math.cos(ax)],
    ])
    ry = np.asarray([
        [math.cos(ay), 0.0, math.sin(ay)],
        [0.0, 1.0, 0.0],
        [-math.sin(ay), 0.0, math.cos(ay)],
    ])
    rz = np.asarray([
        [math.cos(az), -math.sin(az), 0.0],
        [math.sin(az), math.cos(az), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rz @ ry @ rx


def _rect_integral(center, axis_u, axis_v, length_u, length_v, directions):
    """Exact integral of exp(+2jk d.r) over a finite rectangle."""

    center = np.asarray(center, dtype=float)
    u = _unit(axis_u)
    v = _unit(axis_v)
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    lu = float(length_u)
    lv = float(length_v)
    phase = np.exp(2j * WAVE_NUMBER * (directions @ center))
    along_u = lu * np.sinc(
        WAVE_NUMBER * lu * (directions @ u) / math.pi
    )
    along_v = lv * np.sinc(
        WAVE_NUMBER * lv * (directions @ v) / math.pi
    )
    return phase * along_u * along_v


def _line_segments(points):
    points = np.asarray(points, dtype=float)
    return np.stack((points[:-1], points[1:]), axis=1)


def _constant_endpoint_normals(segments, normals):
    values = np.asarray(normals, dtype=float)
    if values.shape == (3,):
        values = np.tile(values, (len(segments), 1))
    return np.stack((values, values), axis=1)


def _normalized_complex_rms(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    return float(
        np.linalg.norm(estimate - reference) / np.linalg.norm(reference)
    )


def _complex_coherence(reference, estimate):
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    return float(
        abs(np.vdot(reference, estimate))
        / (np.linalg.norm(reference) * np.linalg.norm(estimate))
    )


def _radar_basis(direction):
    """Independent VV/HH basis for one coming-from direction."""

    d = _unit(direction)
    transverse = math.hypot(float(d[0]), float(d[1]))
    if transverse < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])
    azimuth = math.atan2(float(d[1]), float(d[0]))
    hh = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0])
    vv = np.asarray([
        d[2] * math.cos(azimuth),
        d[2] * math.sin(azimuth),
        -transverse,
    ])
    return vv, hh


def _transverse_line_basis(tangent, direction):
    """Independent TM/TE axes of a locally invariant line."""

    t = _unit(tangent)
    d = _unit(direction)
    tm = t - float(t @ d) * d
    tm = _unit(tm)
    te = _unit(np.cross(t, d))
    return tm, te


def _coefficient_sample(coefficients, phi_deg):
    """Independent piecewise-linear complex table interpolation."""

    q = float(phi_deg)
    angle = np.asarray(coefficients.phi_deg, dtype=float)

    def sample(values):
        values = np.asarray(values, dtype=np.complex128)
        return complex(
            np.interp(q, angle, values.real),
            np.interp(q, angle, values.imag),
        )

    return sample(coefficients.dA_tm), sample(coefficients.dA_te)


def _explicit_anisotropic_strip(
    center,
    tangent,
    normal,
    length,
    width,
    directions,
    coefficients,
):
    """Finite-width strip truth for a locally anisotropic seam response."""

    t = _unit(tangent)
    n = _unit(normal)
    b = _unit(np.cross(t, n))
    phase_area = _rect_integral(
        center, t, b, length, width, directions
    )
    output = {
        "F_vv": np.zeros(len(directions), dtype=np.complex128),
        "F_hh": np.zeros(len(directions), dtype=np.complex128),
        "F_vh": np.zeros(len(directions), dtype=np.complex128),
    }
    for index, direction in enumerate(np.asarray(directions, dtype=float)):
        d = _unit(direction)
        d_n = float(d @ n)
        if d_n <= 0.0:
            continue
        phi = math.degrees(math.atan2(d_n, float(d @ b)))
        a_tm, a_te = _coefficient_sample(coefficients, phi)
        e_tm, e_te = _transverse_line_basis(t, d)
        e_vv, e_hh = _radar_basis(d)
        scale = phase_area[index] / (4.0 * math.pi * float(width))
        for key, transmit, receive in (
            ("F_vv", e_vv, e_vv),
            ("F_hh", e_hh, e_hh),
            ("F_vh", e_vv, e_hh),
        ):
            dyad = (
                float(e_tm @ transmit) * float(e_tm @ receive) * a_tm
                + float(e_te @ transmit) * float(e_te @ receive) * a_te
            )
            output[key][index] = scale * dyad
    return output


def _local_look_grid(normal, axis_b, axis_t):
    """A deterministic common-lit grid with conical incidence."""

    values = []
    for off_normal_deg, azimuth_deg in (
        (0.0, 0.0),
        (12.0, 20.0),
        (18.0, 75.0),
        (25.0, 140.0),
        (32.0, 215.0),
        (38.0, 300.0),
        (44.0, 45.0),
    ):
        theta = math.radians(off_normal_deg)
        azimuth = math.radians(azimuth_deg)
        values.append(
            math.cos(theta) * _unit(normal)
            + math.sin(theta)
            * (
                math.cos(azimuth) * _unit(axis_b)
                + math.sin(azimuth) * _unit(axis_t)
            )
        )
    return np.asarray(values, dtype=float)


class StraightPanelGapTests(unittest.TestCase):
    """Exact scalar sheet truth for an off-center gap or lossy seal."""

    @classmethod
    def setUpClass(cls):
        rotation = _rotation_matrix()
        cls.b = rotation @ np.asarray([1.0, 0.0, 0.0])
        cls.t = rotation @ np.asarray([0.0, 1.0, 0.0])
        cls.n = rotation @ np.asarray([0.0, 0.0, 1.0])
        cls.panel_center = np.asarray([0.17, -0.09, 0.13])
        cls.gap_center = cls.panel_center + 0.081 * cls.b - 0.047 * cls.t
        cls.panel_width = 0.72
        cls.panel_length = 0.55
        cls.gap_length = 0.31
        cls.gap_width = 0.004
        cls.phi_deg = np.asarray([25.0, 45.0, 70.0, 90.0, 115.0, 145.0, 155.0])
        cls.directions = np.asarray([
            math.cos(math.radians(phi)) * cls.b
            + math.sin(math.radians(phi)) * cls.n
            for phi in cls.phi_deg
        ])
        start = cls.gap_center - 0.5 * cls.gap_length * cls.t
        stop = cls.gap_center + 0.5 * cls.gap_length * cls.t
        cls.segments = np.asarray([[start, stop]])
        cls.normals = _constant_endpoint_normals(cls.segments, cls.n)
        cls.clean = (
            (0.83 + 0.17j)
            / WAVELENGTH_M
            * _rect_integral(
                cls.panel_center,
                cls.b,
                cls.t,
                cls.panel_width,
                cls.panel_length,
                cls.directions,
            )
        )

    def _reconstruct(self, contrast):
        angle = np.arange(0.0, 180.0 + 1.0e-12, 5.0)
        across = np.cos(np.radians(angle))
        per_length = (
            contrast
            * self.gap_width
            / WAVELENGTH_M
            * np.sinc(WAVE_NUMBER * self.gap_width * across / math.pi)
        )
        coefficient = SeamCoefficients(
            FREQUENCY_GHZ,
            angle,
            4.0 * math.pi * per_length,
            4.0 * math.pi * per_length,
            label="analytic finite-width panel strip",
        )
        placed = feature_sum.sum_features(
            None,
            [{
                "delta": coefficient,
                "perimeter": self.segments,
                "segment_normals": self.normals,
                "kind": "delta",
            }],
            self.directions,
            FREQUENCY_GHZ,
            psi_tm_deg=0.0,
            psi_te_deg=0.0,
        )
        explicit_delta = (
            contrast
            / WAVELENGTH_M
            * _rect_integral(
                self.gap_center,
                self.b,
                self.t,
                self.gap_width,
                self.gap_length,
                self.directions,
            )
        )
        return placed, explicit_delta

    def test_open_gap_and_complex_seal_match_explicit_featured_panel(self):
        for label, contrast in (
            ("open gap", -1.0 + 0.0j),
            ("lossy seal", -0.42 + 0.27j),
        ):
            with self.subTest(feature=label):
                placed, truth_delta = self._reconstruct(contrast)
                for channel in ("vv", "hh"):
                    actual_delta = np.asarray(
                        placed[f"amp_{channel}"], dtype=np.complex128
                    )
                    np.testing.assert_allclose(
                        actual_delta,
                        truth_delta,
                        rtol=3.0e-13,
                        atol=3.0e-14,
                        err_msg=channel,
                    )
                    np.testing.assert_allclose(
                        self.clean + actual_delta,
                        self.clean + truth_delta,
                        rtol=3.0e-13,
                        atol=3.0e-14,
                        err_msg=f"whole featured panel {channel}",
                    )
                self.assertLess(
                    float(np.max(np.abs(placed["amp_vh"]))), 3.0e-14
                )


class ClosedDoorOutlineTests(unittest.TestCase):
    """Narrow-strip reconstruction of an explicit rectangular door seam."""

    @classmethod
    def setUpClass(cls):
        rotation = _rotation_matrix()
        cls.b = rotation @ np.asarray([1.0, 0.0, 0.0])
        cls.t = rotation @ np.asarray([0.0, 1.0, 0.0])
        cls.n = rotation @ np.asarray([0.0, 0.0, 1.0])
        cls.panel_center = np.asarray([-0.13, 0.08, 0.21])
        cls.door_center = cls.panel_center + 0.09 * cls.b - 0.055 * cls.t
        cls.door_width = 0.38
        cls.door_height = 0.24
        cls.directions = _local_look_grid(cls.n, cls.b, cls.t)
        cls.clean = (
            (0.91 - 0.08j)
            / WAVELENGTH_M
            * _rect_integral(
                cls.panel_center,
                cls.b,
                cls.t,
                0.86,
                0.62,
                cls.directions,
            )
        )
        w2 = cls.door_width / 2.0
        h2 = cls.door_height / 2.0
        cls.local_vertices = np.asarray([
            [-w2, -h2],
            [w2, -h2],
            [w2, h2],
            [-w2, h2],
            [-w2, -h2],
        ])
        points = np.asarray([
            cls.door_center + x * cls.b + y * cls.t
            for x, y in cls.local_vertices
        ])
        cls.segments = _line_segments(points)
        cls.normals = _constant_endpoint_normals(cls.segments, cls.n)

    def _explicit_union_delta(self, width, contrast):
        strip_sum = np.zeros(len(self.directions), dtype=np.complex128)
        for start, stop in self.segments:
            vector = stop - start
            length = float(np.linalg.norm(vector))
            tangent = vector / length
            across = np.cross(tangent, self.n)
            strip_sum += _rect_integral(
                0.5 * (start + stop),
                tangent,
                across,
                length,
                width,
                self.directions,
            )

        # Adjacent centerline strips overlap over one (width/2)^2 rectangle
        # inside each corner.  Subtract those four regions once to obtain the
        # explicit finite-width union rather than four independently added
        # line strips.
        w2 = self.door_width / 2.0
        h2 = self.door_height / 2.0
        overlap_sum = np.zeros_like(strip_sum)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            center = (
                self.door_center
                + sx * (w2 - width / 4.0) * self.b
                + sy * (h2 - width / 4.0) * self.t
            )
            overlap_sum += _rect_integral(
                center,
                self.b,
                self.t,
                width / 2.0,
                width / 2.0,
                self.directions,
            )
        return contrast / WAVELENGTH_M * (strip_sum - overlap_sum)

    def _placed_delta(self, width, contrast):
        constant = 4.0 * math.pi * contrast * width / WAVELENGTH_M
        coefficient = SeamCoefficients(
            FREQUENCY_GHZ,
            np.asarray([0.0, 90.0, 180.0]),
            np.asarray([constant, constant, constant]),
            np.asarray([constant, constant, constant]),
            label="thin closed door seam",
        )
        return feature_sum.sum_features(
            None,
            [{
                "delta": coefficient,
                "perimeter": self.segments,
                "segment_normals": self.normals,
                "kind": "delta",
            }],
            self.directions,
            FREQUENCY_GHZ,
            psi_tm_deg=0.0,
            psi_te_deg=0.0,
        )

    def test_closed_outline_matches_explicit_finite_width_door_gap(self):
        contrast = -0.68 + 0.19j
        errors = []
        for width in (0.0015, 0.00075):
            truth_delta = self._explicit_union_delta(width, contrast)
            placed = self._placed_delta(width, contrast)
            actual = np.asarray(placed["amp_vv"], dtype=np.complex128)
            np.testing.assert_allclose(
                placed["amp_hh"], actual, rtol=5.0e-13, atol=5.0e-14
            )
            self.assertLess(float(np.max(np.abs(placed["amp_vh"]))), 5.0e-14)
            error = _normalized_complex_rms(truth_delta, actual)
            errors.append(error)
            self.assertLess(error, 0.003)
            self.assertGreater(_complex_coherence(truth_delta, actual), 0.999995)
            self.assertLess(
                _normalized_complex_rms(
                    self.clean + truth_delta, self.clean + actual
                ),
                5.0e-4,
            )

        # The line model is the zero-width limit.  Halving width should reduce
        # the leading corner-overlap error, proving this is a controlled model
        # residual and not an unexplained fitted tolerance.
        self.assertLess(errors[1], 0.56 * errors[0])


class FoldedPanelSeamTests(unittest.TestCase):
    """A seam crossing a faceted, non-axisymmetric panel transition."""

    @classmethod
    def setUpClass(cls):
        cls.join = np.asarray([0.09, -0.04, 0.18])
        cls.n_a = _unit([-0.25, 0.0, 1.0])
        cls.n_b = _unit([0.35, 0.05, 1.0])
        cls.t_a = _unit([1.0, 0.30, 0.25])
        cls.t_b = _unit([1.0, -0.25, -0.3375])
        cls.length_a = 0.23
        cls.length_b = 0.27
        cls.points = np.asarray([
            cls.join - cls.length_a * cls.t_a,
            cls.join,
            cls.join + cls.length_b * cls.t_b,
        ])
        cls.segments = _line_segments(cls.points)
        cls.normals = _constant_endpoint_normals(
            cls.segments, np.asarray([cls.n_a, cls.n_b])
        )
        mean_normal = _unit(cls.n_a + cls.n_b)
        mean_b = _unit(np.cross(cls.t_a + cls.t_b, mean_normal))
        mean_t = _unit(np.cross(mean_normal, mean_b))
        candidates = _local_look_grid(mean_normal, mean_b, mean_t)
        cls.directions = np.asarray([
            direction
            for direction in candidates
            if direction @ cls.n_a > 0.45 and direction @ cls.n_b > 0.45
        ])
        cls.clean = (
            (0.79 + 0.06j)
            / WAVELENGTH_M
            * (
                _rect_integral(
                    0.5 * (cls.points[0] + cls.points[1]),
                    cls.t_a,
                    np.cross(cls.t_a, cls.n_a),
                    cls.length_a,
                    0.34,
                    cls.directions,
                )
                + _rect_integral(
                    0.5 * (cls.points[1] + cls.points[2]),
                    cls.t_b,
                    np.cross(cls.t_b, cls.n_b),
                    cls.length_b,
                    0.34,
                    cls.directions,
                )
            )
        )
        angle = np.linspace(0.0, 180.0, 721)
        radians = np.radians(angle)
        scale = 4.0 * math.pi * 0.001 / WAVELENGTH_M
        cls.coefficients = SeamCoefficients(
            FREQUENCY_GHZ,
            angle,
            scale
            * (0.76 - 0.18j)
            * (1.0 + 0.52 * np.cos(radians) + 0.08j * np.sin(radians)),
            scale
            * (0.31 + 0.42j)
            * (1.0 - 0.37 * np.cos(radians) - 0.11j * np.sin(radians)),
            label="anisotropic folded-panel seal",
        )

    def _explicit(self, width):
        # Scale the tabulated 1 mm coefficient to the requested physical width.
        coefficients = SeamCoefficients(
            FREQUENCY_GHZ,
            self.coefficients.phi_deg,
            self.coefficients.dA_tm * (width / 0.001),
            self.coefficients.dA_te * (width / 0.001),
            label=self.coefficients.label,
        )
        total = {
            key: np.zeros(len(self.directions), dtype=np.complex128)
            for key in ("F_vv", "F_hh", "F_vh")
        }
        for index, (start, stop) in enumerate(self.segments):
            reference = _explicit_anisotropic_strip(
                0.5 * (start + stop),
                stop - start,
                (self.n_a, self.n_b)[index],
                np.linalg.norm(stop - start),
                width,
                self.directions,
                coefficients,
            )
            for key in total:
                total[key] += reference[key]
        return coefficients, total

    def test_discontinuous_facet_normals_match_explicit_anisotropic_strips(self):
        width = 0.001
        coefficients, truth = self._explicit(width)
        placed = feature_sum.sum_features(
            None,
            [{
                "delta": coefficients,
                "perimeter": self.segments,
                "segment_normals": self.normals,
                "kind": "delta",
            }],
            self.directions,
            FREQUENCY_GHZ,
            psi_tm_deg=0.0,
            psi_te_deg=0.0,
        )
        reference = np.concatenate([truth[key] for key in ("F_vv", "F_hh", "F_vh")])
        actual = np.concatenate([
            np.asarray(placed[key.replace("F_", "amp_")], dtype=np.complex128)
            for key in ("F_vv", "F_hh", "F_vh")
        ])
        self.assertLess(_normalized_complex_rms(reference, actual), 7.0e-4)
        self.assertGreater(_complex_coherence(reference, actual), 0.999999)
        self.assertGreater(float(np.max(np.abs(placed["amp_vh"]))), 1.0e-5)

        explicit_featured = np.concatenate((
            self.clean + truth["F_vv"],
            self.clean + truth["F_hh"],
            truth["F_vh"],
        ))
        reconstructed = np.concatenate((
            self.clean + np.asarray(placed["amp_vv"]),
            self.clean + np.asarray(placed["amp_hh"]),
            np.asarray(placed["amp_vh"]),
        ))
        self.assertLess(
            _normalized_complex_rms(explicit_featured, reconstructed),
            2.0e-5,
        )

        # This sensitivity check makes the test capable of catching a future
        # implementation that incorrectly smooths a facet-normal discontinuity.
        averaged = _unit(self.n_a + self.n_b)
        wrong_normals = _constant_endpoint_normals(self.segments, averaged)
        wrong = feature_sum.sum_features(
            None,
            [{
                "delta": coefficients,
                "perimeter": self.segments,
                "segment_normals": wrong_normals,
                "kind": "delta",
            }],
            self.directions,
            FREQUENCY_GHZ,
            psi_tm_deg=0.0,
            psi_te_deg=0.0,
        )
        wrong_field = np.concatenate([
            np.asarray(wrong[key], dtype=np.complex128)
            for key in ("amp_vv", "amp_hh", "amp_vh")
        ])
        self.assertGreater(_normalized_complex_rms(actual, wrong_field), 0.02)


class PathOrientationContractTests(unittest.TestCase):
    """Line direction fixes the coupon's signed across-seam axis."""

    def test_reversing_path_requires_mirroring_asymmetric_coupon(self):
        angle = np.linspace(0.0, 180.0, 361)
        radians = np.radians(angle)
        original = SeamCoefficients(
            FREQUENCY_GHZ,
            angle,
            (0.8 + 0.2j) * (1.0 + 0.55 * np.cos(radians)),
            (0.3 - 0.4j) * (1.0 - 0.35 * np.cos(radians)),
            label="signed asymmetric seal",
        )
        mirrored = SeamCoefficients(
            FREQUENCY_GHZ,
            angle,
            original.dA_tm[::-1],
            original.dA_te[::-1],
            label="same seal with mirrored cut x axis",
        )
        center = np.asarray([0.08, -0.06, 0.12])
        tangent = _unit([0.15, 1.0, 0.0])
        normal = np.asarray([0.0, 0.0, 1.0])
        start = center - 0.14 * tangent
        stop = center + 0.14 * tangent
        forward_segments = np.asarray([[start, stop]])
        reverse_segments = np.asarray([[stop, start]])
        directions = _local_look_grid(
            normal, np.cross(tangent, normal), tangent
        )

        def solve(coefficients, segments):
            return feature_sum.sum_features(
                None,
                [{
                    "delta": coefficients,
                    "perimeter": segments,
                    "segment_normals": _constant_endpoint_normals(
                        segments, normal
                    ),
                    "kind": "delta",
                }],
                directions,
                FREQUENCY_GHZ,
                psi_tm_deg=0.0,
                psi_te_deg=0.0,
            )

        forward = solve(original, forward_segments)
        physically_equivalent_reverse = solve(mirrored, reverse_segments)
        unmirrored_reverse = solve(original, reverse_segments)
        forward_field = np.concatenate([
            np.asarray(forward[key], dtype=np.complex128)
            for key in ("amp_vv", "amp_hh", "amp_vh")
        ])
        equivalent_field = np.concatenate([
            np.asarray(physically_equivalent_reverse[key], dtype=np.complex128)
            for key in ("amp_vv", "amp_hh", "amp_vh")
        ])
        wrong_field = np.concatenate([
            np.asarray(unmirrored_reverse[key], dtype=np.complex128)
            for key in ("amp_vv", "amp_hh", "amp_vh")
        ])
        np.testing.assert_allclose(
            equivalent_field, forward_field, rtol=8.0e-13, atol=8.0e-14
        )
        self.assertGreater(_normalized_complex_rms(forward_field, wrong_field), 0.10)


class ExternalBodyArtifactTests(unittest.TestCase):
    """End-to-end line placement on a non-BoR monostatic GRIM body."""

    def test_external_clean_body_plus_gap_matches_explicit_featured_body(self):
        azimuths = [0.0]
        elevations = [30.0, 50.0, 70.0]
        directions = np.asarray([
            [math.cos(math.radians(value)), 0.0, math.sin(math.radians(value))]
            for value in elevations
        ])
        panel_center = np.asarray([0.06, -0.03, 0.11])
        gap_center = panel_center + np.asarray([0.09, -0.04, 0.0])
        gap_width = 0.003
        gap_length = 0.26
        contrast = -0.58 + 0.21j
        clean = (
            (0.88 + 0.04j)
            / WAVELENGTH_M
            * _rect_integral(
                panel_center,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                0.68,
                0.49,
                directions,
            )
        )
        explicit_delta = (
            contrast
            / WAVELENGTH_M
            * _rect_integral(
                gap_center,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                gap_width,
                gap_length,
                directions,
            )
        )

        angles = np.arange(0.0, 181.0, 5.0)
        per_length = (
            contrast
            * gap_width
            / WAVELENGTH_M
            * np.sinc(
                WAVE_NUMBER
                * gap_width
                * np.cos(np.radians(angles))
                / math.pi
            )
        )
        # This analytic coefficient already uses the reference 3-D field
        # convention, so the explicitly recorded inter-solver mapping is zero.
        coefficient = SeamCoefficients(
            FREQUENCY_GHZ,
            angles,
            4.0 * math.pi * per_length,
            4.0 * math.pi * per_length,
            label="external-body analytic gap",
        )
        start = gap_center - np.asarray([0.0, gap_length / 2.0, 0.0])
        stop = gap_center + np.asarray([0.0, gap_length / 2.0, 0.0])
        segments = np.asarray([[start, stop]])
        normals = _constant_endpoint_normals(segments, [0.0, 0.0, 1.0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "external_clean.grim"
            combined = root / "external_with_gap.grim"
            feature_sum.export_radar_grim(
                str(base),
                bor_result=None,
                placements=[],
                frequencies_ghz=[FREQUENCY_GHZ],
                azimuths_deg=azimuths,
                elevations_deg=elevations,
                axis_az_deg=0.0,
                axis_el_deg=90.0,
                roll_deg=0.0,
                history="analytic external flat-panel body",
            )
            with np.load(base, allow_pickle=False) as stored:
                payload = {
                    key: np.array(stored[key], copy=True) for key in stored.files
                }
            field = np.zeros((1, len(elevations), 1, 3), dtype=np.complex128)
            field[0, :, 0, 0] = clean
            field[0, :, 0, 1] = clean
            payload["rcs_amp_real"] = field.real
            payload["rcs_amp_imag"] = field.imag
            payload["rcs_phase"] = np.angle(field).astype(np.float32)
            payload["rcs_power"] = (
                4.0 * math.pi * np.abs(field) ** 2
            ).astype(np.float32)
            _save_grim_npz(payload, str(base))

            feature_sum.add_features_to_monostatic_grim(
                str(base),
                str(combined),
                placements=[{
                    "delta": coefficient,
                    "perimeter": segments,
                    "segment_normals": normals,
                    "kind": "delta",
                }],
                radar_grid={
                    "frequencies_ghz": [FREQUENCY_GHZ],
                    "azimuths_deg": azimuths,
                    "elevations_deg": elevations,
                    "axis_az_deg": 0.0,
                    "axis_el_deg": 90.0,
                    "roll_deg": 0.0,
                },
                psi_tm_deg=0.0,
                psi_te_deg=0.0,
                history="placed analytic gap on external body",
            )

            with np.load(base, allow_pickle=False) as clean_payload:
                clean_after = (
                    clean_payload["rcs_amp_real"]
                    + 1j * clean_payload["rcs_amp_imag"]
                )
                np.testing.assert_allclose(clean_after[0, :, 0, 0], clean)
            with np.load(combined, allow_pickle=False) as result:
                actual = result["rcs_amp_real"] + 1j * result["rcs_amp_imag"]
                expected = clean + explicit_delta
                np.testing.assert_allclose(
                    actual[0, :, 0, 0], expected, rtol=4.0e-13, atol=4.0e-14
                )
                np.testing.assert_allclose(
                    actual[0, :, 0, 1], expected, rtol=4.0e-13, atol=4.0e-14
                )
                self.assertLess(float(np.max(np.abs(actual[..., 2]))), 4.0e-14)
                np.testing.assert_allclose(
                    result["rcs_power"],
                    4.0 * math.pi * np.abs(actual) ** 2,
                    rtol=2.0e-6,
                    atol=1.0e-12,
                )
                provenance = json.loads(str(result["feature_provenance_json"]))
                self.assertEqual(provenance[-1]["line_feature_count"], 1)
                self.assertEqual(
                    provenance[-1]["line_phase_mapping_deg"],
                    {"TM": 0.0, "TE": 0.0},
                )
                self.assertFalse(
                    provenance[-1]["model_scope"][
                        "body_feature_mutual_coupling"
                    ]
                )
                self.assertNotIn("body_model_metadata_json", result.files)


if __name__ == "__main__":
    unittest.main(verbosity=2)
