#!/usr/bin/env python3
"""Contract tests for mandatory VV/HH 2-D solves and one-file GRIM export."""

import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import grim_io  # noqa: E402
import rcs_solver as rcs  # noqa: E402


def _samples(internal_polarization, *, bistatic=False):
    amplitudes = {
        "TE": (2.0 + 1.0j, -1.0 + 0.5j),
        "TM": (0.25 - 0.75j, 1.5 + 0.25j),
    }[internal_polarization]
    rows = []
    k0 = 2.0 * math.pi * 1.0e9 / grim_io.C0
    for angle, amplitude in zip((0.0, 30.0), amplitudes):
        sigma = float(abs(amplitude) ** 2 / (4.0 * k0))
        rows.append({
            "frequency_ghz": 1.0,
            "theta_inc_deg": 10.0 if bistatic else angle,
            "theta_scat_deg": angle,
            "rcs_linear": sigma,
            "rcs_db": 10.0 * math.log10(sigma),
            "rcs_amp_real": float(amplitude.real),
            "rcs_amp_imag": float(amplitude.imag),
            "rcs_amp_phase_deg": math.degrees(np.angle(amplitude)),
            "linear_residual": 1.0e-12,
        })
    return rows


def _single_result(internal_polarization, *, bistatic=False, certified=False):
    export = "VV" if internal_polarization == "TE" else "HH"
    metadata = {
        "source_path": "fixture.geo",
        "segment_count": 1,
        "panel_count": 20,
        "panel_count_min": 20,
        "panel_count_max": 20,
        "formulation": f"fixture {internal_polarization}",
        "solver_method": "dense_lu",
        "solver_method_requested": "direct",
        "residual_norm_max": 1.0e-12,
        "constraint_residual_norm_max": 0.0,
        "condition_est_max": 2.0,
        "condition_est_computed": True,
        "warnings": [],
        "preflight": {"passed": True},
        "quality_gate": {"passed": True, "reason": "fixture"},
    }
    if certified:
        metadata.update({
            "mesh_convergence": {
                "schema": "ghost.solver.mesh-convergence.v1",
                "passed": True,
                "reason": "fixture",
                "published_mesh": "fine",
                "base_quality_gate": {"passed": True},
                "fine_quality_gate": {"passed": True},
            },
            "mesh_convergence_certified": True,
            "certified_entry_point": True,
            "published_mesh": "fine",
        })
    return {
        "solver": "2d_bie_mom_rcs",
        "scattering_mode": "bistatic" if bistatic else "monostatic",
        "amplitude_convention": rcs.RCS_AMPLITUDE_CONVENTION,
        "polarization": internal_polarization,
        "polarization_export": internal_polarization,
        "samples": _samples(internal_polarization, bistatic=bistatic),
        "metadata": metadata,
        "_export": export,
    }


class CoPolarizedSolverContractTests(unittest.TestCase):
    def test_canonical_entry_points_expose_no_dead_solver_controls(self):
        canonical = (
            rcs.solve_monostatic_rcs_2d,
            rcs.solve_monostatic_rcs_2d_certified,
            rcs.solve_monostatic_rcs_2d_survey,
            rcs.solve_bistatic_rcs_2d,
            rcs.solve_bistatic_rcs_2d_certified,
            rcs.solve_bistatic_rcs_2d_survey,
        )
        for solve in canonical:
            parameters = inspect.signature(solve).parameters
            for removed in ("polarization", "solver_method", "cfie_alpha"):
                self.assertNotIn(removed, parameters)

    def test_monostatic_co_solve_is_exact_union_of_separate_results(self):
        expected = {
            "VV": _single_result("TE"),
            "HH": _single_result("TM"),
        }
        calls = []

        def fake_single(**kwargs):
            calls.append(kwargs)
            return expected["VV" if kwargs["polarization"] == "TE" else "HH"]

        with mock.patch.object(
            rcs, "solve_monostatic_rcs_2d_single_polarization",
            side_effect=fake_single,
        ):
            result = rcs.solve_monostatic_rcs_2d(
                geometry_snapshot={"segments": []},
                frequencies_ghz=[1.0],
                elevations_deg=[0.0, 30.0],
            )

        self.assertEqual([call["polarization"] for call in calls], ["TE", "TM"])
        self.assertIs(
            calls[0]["_shared_discretization_cache"],
            calls[1]["_shared_discretization_cache"],
        )
        self.assertEqual(result["polarizations"], ["VV", "HH"])
        self.assertEqual(result["polarization_mapping"], {"VV": "TE", "HH": "TM"})
        self.assertNotIn("polarization", result)
        self.assertEqual(len(result["samples"]), 4)
        for export, internal in (("VV", "TE"), ("HH", "TM")):
            got = result["co_solved_samples"][export]
            want = expected[export]["samples"]
            self.assertEqual(len(got), len(want))
            for got_row, want_row in zip(got, want):
                self.assertEqual(got_row["polarization"], export)
                self.assertEqual(got_row["polarization_internal"], internal)
                for key, value in want_row.items():
                    self.assertEqual(got_row[key], value)

    @unittest.skipUnless(
        rcs._SCIPY_SPECIAL is not None,
        "SciPy is required for the real 2-D kernel regression",
    )
    def test_real_pec_kernel_matches_explicit_te_tm_solves(self):
        count = 12
        radius = 0.05
        theta = np.linspace(0.0, -2.0 * np.pi, count + 1)
        snapshot = {
            "title": "closed PEC co-polarized regression",
            "segments": [{
                "name": "pec_circle",
                "seg_type": 2,
                "properties": ["2", "1", "0", "0", "0"],
                "point_pairs": [
                    {
                        "x1": float(radius * np.cos(theta[index])),
                        "y1": float(radius * np.sin(theta[index])),
                        "x2": float(radius * np.cos(theta[index + 1])),
                        "y2": float(radius * np.sin(theta[index + 1])),
                    }
                    for index in range(count)
                ],
            }],
            "ibcs": [],
            "dielectrics": [],
        }
        common = {
            "geometry_snapshot": snapshot,
            "frequencies_ghz": [0.3],
            "elevations_deg": [0.0, 45.0],
            "geometry_units": "meters",
            "strict_quality_gate": False,
            "compute_condition_number": False,
            "max_panels": 200,
        }

        combined = rcs.solve_monostatic_rcs_2d(**common)
        explicit = {
            "VV": rcs.solve_monostatic_rcs_2d_single_polarization(
                polarization="TE", **common
            ),
            "HH": rcs.solve_monostatic_rcs_2d_single_polarization(
                polarization="TM", **common
            ),
        }

        for channel in ("VV", "HH"):
            actual_rows = combined["co_solved_samples"][channel]
            expected_rows = explicit[channel]["samples"]
            self.assertEqual(len(actual_rows), len(expected_rows))
            actual_amplitude = np.asarray([
                complex(row["rcs_amp_real"], row["rcs_amp_imag"])
                for row in actual_rows
            ])
            expected_amplitude = np.asarray([
                complex(row["rcs_amp_real"], row["rcs_amp_imag"])
                for row in expected_rows
            ])
            np.testing.assert_allclose(
                actual_amplitude,
                expected_amplitude,
                rtol=2.0e-13,
                atol=2.0e-14,
            )
            np.testing.assert_allclose(
                [row["rcs_linear"] for row in actual_rows],
                [row["rcs_linear"] for row in expected_rows],
                rtol=2.0e-13,
                atol=2.0e-14,
            )

    def test_bistatic_co_solve_exposes_both_channels(self):
        def fake_single(**kwargs):
            return _single_result(
                kwargs["polarization"], bistatic=True
            )

        with mock.patch.object(
            rcs, "solve_bistatic_rcs_2d_single_polarization",
            side_effect=fake_single,
        ):
            result = rcs.solve_bistatic_rcs_2d(
                geometry_snapshot={"segments": []},
                frequencies_ghz=[1.0],
                incidence_angles_deg=[10.0],
                observation_angles_deg=[0.0, 30.0],
            )
        self.assertEqual(set(result["co_solved_samples"]), {"VV", "HH"})
        self.assertEqual(
            {row["polarization"] for row in result["samples"]},
            {"VV", "HH"},
        )

    def test_certified_result_requires_both_channel_certificates(self):
        def fake_certified(**kwargs):
            return _single_result(kwargs["polarization"], certified=True)

        with mock.patch.object(
            rcs, "solve_monostatic_rcs_2d_certified_single_polarization",
            side_effect=fake_certified,
        ):
            result = rcs.solve_monostatic_rcs_2d_certified(
                geometry_snapshot={"segments": []},
                frequencies_ghz=[1.0],
                elevations_deg=[0.0, 30.0],
            )
        mesh = result["metadata"]["mesh_convergence"]
        self.assertTrue(result["metadata"]["mesh_convergence_certified"])
        self.assertTrue(mesh["passed"])
        self.assertEqual(set(mesh["channels"]), {"VV", "HH"})
        self.assertTrue(mesh["base_quality_gate"]["passed"])
        self.assertTrue(mesh["fine_quality_gate"]["passed"])


class CoPolarizedGrimTests(unittest.TestCase):
    def test_incomplete_declared_dual_result_is_rejected(self):
        result = rcs._merge_co_polarized_2d_results({
            "VV": _single_result("TE"),
            "HH": _single_result("TM"),
        })
        result["co_solved_samples"].pop("HH")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "single-channel fallback"):
                grim_io.export_result_to_grim(
                    result, str(Path(tmp) / "incomplete")
                )

    def test_one_grim_contains_vv_and_hh_raw_complex_fields(self):
        result = rcs._merge_co_polarized_2d_results({
            "VV": _single_result("TE"),
            "HH": _single_result("TM"),
        })
        with tempfile.TemporaryDirectory() as tmp:
            written = grim_io.export_result_to_grim(
                result, str(Path(tmp) / "dual")
            )
            self.assertEqual(len(written), 1)
            with np.load(written[0], allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["polarizations"], ["VV", "HH"]
                )
                self.assertEqual(payload["rcs_power"].shape, (2, 1, 1, 2))
                self.assertEqual(payload["rcs_amp_real"].dtype, np.float64)
                vv = np.asarray([
                    complex(row["rcs_amp_real"], row["rcs_amp_imag"])
                    for row in result["co_solved_samples"]["VV"]
                ])
                hh = np.asarray([
                    complex(row["rcs_amp_real"], row["rcs_amp_imag"])
                    for row in result["co_solved_samples"]["HH"]
                ])
                np.testing.assert_allclose(
                    payload["rcs_amp_real"][:, 0, 0, 0]
                    + 1j * payload["rcs_amp_imag"][:, 0, 0, 0],
                    vv,
                )
                np.testing.assert_allclose(
                    payload["rcs_amp_real"][:, 0, 0, 1]
                    + 1j * payload["rcs_amp_imag"][:, 0, 0, 1],
                    hh,
                )
                metadata = json.loads(str(payload["solver_metadata_json"]))
                self.assertEqual(metadata["polarizations"], ["VV", "HH"])
                self.assertEqual(
                    metadata["polarization_mapping"],
                    {"HH": "TM", "VV": "TE"},
                )

    def test_bistatic_collection_keeps_both_channels_per_incidence(self):
        result = rcs._merge_co_polarized_2d_results({
            "VV": _single_result("TE", bistatic=True),
            "HH": _single_result("TM", bistatic=True),
        })
        with tempfile.TemporaryDirectory() as tmp:
            written = grim_io.export_result_to_grim(
                result, str(Path(tmp) / "bistatic_dual")
            )
            self.assertEqual(len(written), 1)
            self.assertIn("inc_10", Path(written[0]).stem)
            with np.load(written[0], allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["polarizations"], ["VV", "HH"]
                )
                self.assertEqual(payload["rcs_power"].shape, (2, 1, 1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
