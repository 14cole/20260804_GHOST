#!/usr/bin/env python3
"""Capability acceptance gate for the general-purpose 2-D RCS solver.

This suite complements the lower-level numerical regressions.  Its cases map
directly to the supported geometry contract:

* air/PEC and air/IBC (TYPE 2),
* air/dielectric (TYPE 3),
* dielectric/PEC-or-IBC (TYPE 4),
* dielectric/dielectric (TYPE 5),
* air/air impedance sheets (TYPE 1),
* disconnected bodies that share material properties, and
* air/PEC/dielectric high-degree junctions.

Run directly; pytest is not required:

    python tests/test_2d_capability_acceptance.py
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

import grim_io  # noqa: E402
import rcs_solver as rcs  # noqa: E402

try:
    from mie_reference import (  # noqa: E402
        sigma_coated_impedance_cylinder,
        sigma_dielectric_cylinder,
        sigma_impedance_cylinder,
        sigma_pec_cylinder,
        sigma_two_layer_dielectric_cylinder,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise
    HAVE_SCIPY = False
else:
    HAVE_SCIPY = True


def _segment(name, seg_type, points, *, panels=1, ibc=0, pos=0, neg=0):
    return {
        "name": str(name),
        "seg_type": int(seg_type),
        "properties": [
            str(seg_type), str(panels), str(ibc), str(pos), str(neg),
        ],
        "point_pairs": [
            {
                "x1": float(p0[0]), "y1": float(p0[1]),
                "x2": float(p1[0]), "y2": float(p1[1]),
            }
            for p0, p1 in zip(points[:-1], points[1:])
        ],
    }


def _circle(
    name,
    radius,
    count,
    seg_type,
    *,
    center=(0.0, 0.0),
    ibc=0,
    pos=0,
    neg=0,
):
    # Clockwise winding gives a top-level TYPE 2/3 contour an air-pointing
    # user normal.  For an inner TYPE 4 or TYPE 5 contour the same winding
    # points from the core toward the surrounding shell.
    theta = np.linspace(0.0, -2.0 * np.pi, int(count) + 1)
    cx, cy = center
    points = [
        (cx + radius * np.cos(value), cy + radius * np.sin(value))
        for value in theta
    ]
    return _segment(
        name, seg_type, points, panels=1, ibc=ibc, pos=pos, neg=neg
    )


def _solve(snapshot, polarization, frequency_hz, angles_deg=(0.0,)):
    return rcs.solve_monostatic_rcs_2d(
        snapshot,
        frequencies_ghz=[float(frequency_hz) / 1.0e9],
        elevations_deg=list(angles_deg),
        polarization=polarization,
        geometry_units="meters",
        strict_quality_gate=False,
        max_panels=4000,
    )


def _solve_bistatic(snapshot, polarization, frequency_hz, angles_deg):
    return rcs.solve_bistatic_rcs_2d(
        snapshot,
        frequencies_ghz=[float(frequency_hz) / 1.0e9],
        incidence_angles_deg=list(angles_deg),
        observation_angles_deg=list(angles_deg),
        polarization=polarization,
        geometry_units="meters",
        strict_quality_gate=False,
        max_panels=4000,
    )


def _amplitudes(result):
    return np.asarray([
        complex(row["rcs_amp_real"], row["rcs_amp_imag"])
        for row in result["samples"]
    ])


def _assert_reference_db(testcase, actual, reference, tolerance_db, label):
    testcase.assertGreater(actual, 0.0, msg=label)
    testcase.assertGreater(reference, 0.0, msg=label)
    error_db = abs(10.0 * math.log10(actual / reference))
    testcase.assertLess(error_db, tolerance_db, msg=f"{label}: {error_db:.6g} dB")


@unittest.skipUnless(HAVE_SCIPY, "SciPy is required for trusted 2-D physics references")
class BoundaryReferenceAcceptanceTests(unittest.TestCase):
    """Analytical circular references for every closed-boundary material type."""

    def test_type2_air_pec_and_air_ibc(self):
        radius = 0.08
        frequency_hz = rcs.C0 / (2.0 * math.pi * radius)
        z_s = 55.0 - 12.0j
        pec = {
            "segments": [_circle("pec", radius, 80, 2)],
            "ibcs": [],
            "dielectrics": [],
        }
        ibc = {
            "segments": [_circle("ibc", radius, 80, 2, ibc=1)],
            "ibcs": [["1", "constant", "55", "-12", "0", "0"]],
            "dielectrics": [],
        }
        for pol in ("TM", "TE"):
            got_pec = float(_solve(pec, pol, frequency_hz)["samples"][0]["rcs_linear"])
            got_ibc = float(_solve(ibc, pol, frequency_hz)["samples"][0]["rcs_linear"])
            _assert_reference_db(
                self, got_pec, sigma_pec_cylinder(radius, frequency_hz, pol),
                0.02, f"TYPE 2 PEC {pol}",
            )
            _assert_reference_db(
                self, got_ibc,
                sigma_impedance_cylinder(radius, z_s, frequency_hz, pol),
                0.02, f"TYPE 2 IBC {pol}",
            )

    def test_type3_air_dielectric(self):
        radius = 0.08
        frequency_hz = rcs.C0 / (2.0 * math.pi * radius)
        eps_r = 3.4 - 0.08j
        snapshot = {
            "segments": [_circle("dielectric", radius, 80, 3, pos=1)],
            "ibcs": [],
            "dielectrics": [["1", "3.4", "-0.08", "1", "0"]],
        }
        for pol in ("TM", "TE"):
            got = float(_solve(snapshot, pol, frequency_hz)["samples"][0]["rcs_linear"])
            reference = sigma_dielectric_cylinder(
                radius, eps_r, 1.0 + 0.0j, frequency_hz, pol
            )
            _assert_reference_db(self, got, reference, 0.02, f"TYPE 3 {pol}")

    def test_type3_plus_type5_two_dielectric_layers(self):
        core_radius = 0.045
        outer_radius = 0.08
        frequency_hz = rcs.C0 / (2.0 * math.pi * outer_radius)
        eps_shell = 2.3 - 0.04j
        eps_core = 4.1 - 0.10j
        snapshot = {
            "segments": [
                _circle("air_shell", outer_radius, 88, 3, pos=1),
                _circle(
                    "shell_core", core_radius, 88, 5, pos=1, neg=2
                ),
            ],
            "ibcs": [],
            "dielectrics": [
                ["1", "2.3", "-0.04", "1", "0"],
                ["2", "4.1", "-0.10", "1", "0"],
            ],
        }
        for pol in ("TM", "TE"):
            got = float(_solve(snapshot, pol, frequency_hz)["samples"][0]["rcs_linear"])
            reference = sigma_two_layer_dielectric_cylinder(
                core_radius,
                outer_radius,
                eps_shell,
                1.0 + 0.0j,
                eps_core,
                1.0 + 0.0j,
                frequency_hz,
                pol,
            )
            _assert_reference_db(self, got, reference, 0.05, f"TYPE 5 {pol}")

    def test_type4_dielectric_impedance_boundary(self):
        inner_radius = 0.052
        outer_radius = 0.08
        frequency_hz = rcs.C0 / (2.0 * math.pi * outer_radius)
        eps_r = 2.7 - 0.06j
        z_s = 42.0 - 9.0j
        snapshot = {
            "segments": [
                _circle("air_coating", outer_radius, 88, 3, pos=1),
                _circle(
                    "coating_ibc", inner_radius, 88, 4, ibc=1, pos=1
                ),
            ],
            "ibcs": [["1", "constant", "42", "-9", "0", "0"]],
            "dielectrics": [["1", "2.7", "-0.06", "1", "0"]],
        }
        for pol in ("TM", "TE"):
            got = float(_solve(snapshot, pol, frequency_hz)["samples"][0]["rcs_linear"])
            reference = sigma_coated_impedance_cylinder(
                inner_radius,
                outer_radius,
                eps_r,
                1.0 + 0.0j,
                z_s,
                frequency_hz,
                pol,
            )
            _assert_reference_db(self, got, reference, 0.05, f"TYPE 4 IBC {pol}")


@unittest.skipUnless(HAVE_SCIPY, "SciPy is required for trusted 2-D production solves")
class SheetAndArbitraryGeometryAcceptanceTests(unittest.TestCase):
    def test_type1_air_air_sheet_transparency_limit(self):
        line = _segment("sheet", 1, [(-0.1, 0.0), (0.1, 0.0)], panels=48, ibc=1)
        ordinary = {
            "segments": [line],
            "ibcs": [["1", "constant", "75", "0", "0", "0"]],
            "dielectrics": [],
        }
        transparent = {
            "segments": [line],
            "ibcs": [["1", "constant", "1e10", "0", "0", "0"]],
            "dielectrics": [],
        }
        for pol in ("TM", "TE"):
            reference = _amplitudes(_solve(ordinary, pol, 0.8e9, (-35.0, 0.0, 40.0)))
            limit = _amplitudes(_solve(transparent, pol, 0.8e9, (-35.0, 0.0, 40.0)))
            self.assertLess(
                float(np.max(np.abs(limit))),
                1.0e-5 * float(np.max(np.abs(reference))),
                msg=pol,
            )

    def test_concave_piecewise_linear_body_mesh_converges(self):
        points = [
            (-0.08, -0.06), (-0.08, 0.06), (0.0, 0.015),
            (0.08, 0.06), (0.08, -0.06), (-0.08, -0.06),
        ]
        angles = (-30.0, 0.0, 45.0)
        for pol in ("TM", "TE"):
            coarse = {
                "segments": [_segment("concave", 2, points, panels=8)],
                "ibcs": [], "dielectrics": [],
            }
            fine = {
                "segments": [_segment("concave", 2, points, panels=16)],
                "ibcs": [], "dielectrics": [],
            }
            amp_coarse = _amplitudes(_solve(coarse, pol, 0.6e9, angles))
            amp_fine = _amplitudes(_solve(fine, pol, 0.6e9, angles))
            self.assertTrue(np.all(np.isfinite(amp_fine)), msg=pol)
            relative = np.linalg.norm(amp_fine - amp_coarse) / max(
                np.linalg.norm(amp_fine), np.finfo(float).tiny
            )
            self.assertLess(relative, 0.08, msg=f"{pol}: relative={relative:g}")


@unittest.skipUnless(HAVE_SCIPY, "SciPy is required for trusted 2-D production solves")
class TopologyAcceptanceTests(unittest.TestCase):
    def _disconnected_dielectrics(self, distinct_flags):
        second_flag = 2 if distinct_flags else 1
        dielectrics = [["1", "3.0", "-0.03", "1", "0"]]
        if distinct_flags:
            dielectrics.append(["2", "3.0", "-0.03", "1", "0"])
        return {
            "segments": [
                _circle(
                    "body_a", 0.028, 56, 3, center=(-0.075, 0.01), pos=1
                ),
                _circle(
                    "body_b", 0.022, 56, 3, center=(0.072, -0.015),
                    pos=second_flag,
                ),
            ],
            "ibcs": [],
            "dielectrics": dielectrics,
        }

    def test_disconnected_identical_material_complex_field_is_flag_invariant(self):
        """Material bookkeeping must not change coherent field or power."""

        angles = (-25.0, 20.0, 65.0)
        for pol in ("TM", "TE"):
            shared_result = _solve(
                self._disconnected_dielectrics(False), pol, 0.9e9, angles
            )
            distinct_result = _solve(
                self._disconnected_dielectrics(True), pol, 0.9e9, angles
            )
            shared = _amplitudes(shared_result)
            distinct = _amplitudes(distinct_result)
            shared_power = np.asarray([
                row["rcs_linear"] for row in shared_result["samples"]
            ])
            distinct_power = np.asarray([
                row["rcs_linear"] for row in distinct_result["samples"]
            ])
            # First distinguish a physical-power error from a coherent phase
            # convention error.  Both are acceptance failures, but only the
            # latter can hide behind an otherwise correct RCS plot in GRIM.
            np.testing.assert_allclose(
                shared_power,
                distinct_power,
                rtol=5.0e-4,
                atol=1.0e-12,
                err_msg=f"{pol}: material flags changed monostatic power",
            )
            magnitude_relative = np.linalg.norm(
                np.abs(shared) - np.abs(distinct)
            ) / max(
                np.linalg.norm(np.abs(distinct)), np.finfo(float).tiny
            )
            self.assertLess(
                magnitude_relative,
                0.02,
                msg=(
                    f"{pol}: identical media changed magnitude by "
                    f"{magnitude_relative:.3%} when only flags were split"
                ),
            )
            phase_delta = np.angle(shared * np.conjugate(distinct))
            max_phase_deg = float(np.max(np.abs(np.degrees(phase_delta))))
            self.assertLess(
                max_phase_deg,
                1.0,
                msg=(
                    f"{pol}: identical media changed coherent phase by up to "
                    f"{max_phase_deg:.6g} deg when only flags were split"
                ),
            )

    @staticmethod
    def _partial_coating(panels):
        return {
            "segments": [
                _segment("bare_pec", 2, [
                    (0.05, 0.1), (0.1, 0.1), (0.1, -0.1),
                    (-0.1, -0.1), (-0.1, 0.1), (-0.05, 0.1),
                ], panels=panels),
                _segment("coating_air", 3, [
                    (-0.05, 0.1), (-0.05, 0.12),
                    (0.05, 0.12), (0.05, 0.1),
                ], panels=panels, pos=1),
                _segment(
                    "coating_pec", 4,
                    [(-0.05, 0.1), (0.05, 0.1)],
                    panels=panels, pos=1,
                ),
            ],
            "ibcs": [],
            "dielectrics": [["1", "2.5", "-0.02", "1", "0"]],
        }

    def test_air_pec_dielectric_junction_treatment_is_attested(self):
        """Metadata must describe the active junction treatment truthfully."""

        for pol in ("TM", "TE"):
            result = _solve(self._partial_coating(8), pol, 0.6e9, (0.0,))
            metadata = result["metadata"]
            self.assertEqual(metadata["preflight"]["high_degree_nodes"], 2)
            self.assertGreater(metadata["junction_constraint_candidates"], 0)
            self.assertEqual(
                metadata["junction_treatment"],
                "implicit_multi_region_indirect",
            )
            self.assertFalse(metadata["junction_constraints_applied"])
            self.assertEqual(metadata["junction_constraints_applied_count"], 0)
            self.assertFalse(metadata["constraint_residual_applicable"])
            self.assertIn("multi-region indirect SLP", metadata["formulation"])
            self.assertTrue(math.isfinite(metadata["residual_norm_max"]))
            self.assertLess(metadata["residual_norm_max"], 1.0e-6)

    def test_air_pec_dielectric_junction_mesh_converges(self):
        angles = (-20.0, 0.0, 35.0)
        for pol in ("TM", "TE"):
            coarse = _amplitudes(_solve(
                self._partial_coating(6), pol, 0.6e9, angles
            ))
            fine = _amplitudes(_solve(
                self._partial_coating(12), pol, 0.6e9, angles
            ))
            relative = np.linalg.norm(fine - coarse) / max(
                np.linalg.norm(fine), np.finfo(float).tiny
            )
            self.assertLess(relative, 0.08, msg=f"{pol}: relative={relative:g}")

    def test_air_pec_dielectric_junction_is_reciprocal(self):
        angles = (20.0, 110.0)
        for pol in ("TM", "TE"):
            result = _solve_bistatic(
                self._partial_coating(8), pol, 0.6e9, angles
            )
            amplitudes = {
                (sample["theta_inc_deg"], sample["theta_scat_deg"]): complex(
                    sample["rcs_amp_real"], sample["rcs_amp_imag"]
                )
                for sample in result["samples"]
            }
            forward = amplitudes[(20.0, 110.0)]
            reciprocal = amplitudes[(110.0, 20.0)]
            relative = abs(forward - reciprocal) / max(
                abs(forward), abs(reciprocal), np.finfo(float).tiny
            )
            self.assertLess(relative, 1.0e-3, msg=f"{pol}: relative={relative:g}")

    def test_air_pec_dielectric_junction_is_segment_order_invariant(self):
        angles = (-20.0, 0.0, 35.0)
        original = self._partial_coating(8)
        reordered = dict(original)
        reordered["segments"] = list(reversed(original["segments"]))
        for pol in ("TM", "TE"):
            expected = _amplitudes(_solve(original, pol, 0.6e9, angles))
            actual = _amplitudes(_solve(reordered, pol, 0.6e9, angles))
            relative = np.linalg.norm(actual - expected) / max(
                np.linalg.norm(expected), np.finfo(float).tiny
            )
            self.assertLess(relative, 1.0e-8, msg=f"{pol}: relative={relative:g}")


class GrimInterchangeAcceptanceTests(unittest.TestCase):
    """Producer-side assertions for the `.grim` contract consumed by GRIM."""

    def test_monostatic_2d_export_preserves_physical_complex_field(self):
        samples = []
        expected = {}
        for frequency_ghz in (1.0, 1.5):
            k0 = 2.0 * math.pi * frequency_ghz * 1.0e9 / rcs.C0
            for angle in (-20.0, 30.0):
                amplitude = complex(1.0 + angle / 100.0, -0.2 * frequency_ghz)
                sigma = abs(amplitude) ** 2 / (4.0 * k0)
                expected[(angle, frequency_ghz)] = (amplitude, sigma)
                samples.append({
                    "frequency_ghz": frequency_ghz,
                    "theta_inc_deg": angle,
                    "theta_scat_deg": angle,
                    "rcs_linear": sigma,
                    "rcs_db": 10.0 * math.log10(sigma),
                    "rcs_amp_real": amplitude.real,
                    "rcs_amp_imag": amplitude.imag,
                    "rcs_amp_phase_deg": math.degrees(np.angle(amplitude)),
                    "linear_residual": 1.0e-12,
                })

        result = {
            "solver": "2d_bie_mom_rcs",
            "scattering_mode": "monostatic",
            "polarization": "TM",
            "samples": samples,
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            written = grim_io.export_result_to_grim(
                result, str(Path(tmp) / "acceptance")
            )
            self.assertEqual(len(written), 1)
            with np.load(written[0], allow_pickle=False) as data:
                units = json.loads(str(data["units"].item()))
                self.assertEqual(units["rcs_log_unit"], "dBke")
                self.assertEqual(units["rcs_linear_quantity"], "sigma_2d")
                np.testing.assert_array_equal(data["azimuths"], [-20.0, 30.0])
                np.testing.assert_array_equal(data["elevations"], [0.0])
                np.testing.assert_array_equal(data["frequencies"], [1.0, 1.5])
                np.testing.assert_array_equal(data["polarizations"], ["HH"])
                self.assertEqual(data["rcs_amp_real"].dtype, np.float64)
                self.assertEqual(data["rcs_amp_imag"].dtype, np.float64)
                for ai, angle in enumerate((-20.0, 30.0)):
                    for fi, frequency_ghz in enumerate((1.0, 1.5)):
                        amplitude, sigma = expected[(angle, frequency_ghz)]
                        self.assertAlmostEqual(
                            float(data["rcs_amp_real"][ai, 0, fi, 0]),
                            amplitude.real,
                        )
                        self.assertAlmostEqual(
                            float(data["rcs_amp_imag"][ai, 0, fi, 0]),
                            amplitude.imag,
                        )
                        self.assertAlmostEqual(
                            float(data["rcs_power"][ai, 0, fi, 0]),
                            sigma,
                            delta=2.0e-7 * max(1.0, sigma),
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
