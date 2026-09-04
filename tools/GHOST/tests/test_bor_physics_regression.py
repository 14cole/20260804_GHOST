#!/usr/bin/env python3
"""BoR physics, workflow, and optimized-assembly release regressions."""

import json
import inspect
import math
import copy
import shlex
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import bor_dispatch  # noqa: E402
import bor_streaming  # noqa: E402
import feature_sum  # noqa: E402
import grim_io  # noqa: E402
import hpc_common  # noqa: E402
import occluder  # noqa: E402
import run_hpc_bor_monostatic  # noqa: E402
import run_local_bor  # noqa: E402
import surface_mesh  # noqa: E402
import validate_feature_reconstruction  # noqa: E402
from bor_kernels import (  # noqa: E402
    C0,
    ETA0,
    N_XI_SAFETY_CAP,
    n_xi_for_pairs,
)
from bor_solver import (  # noqa: E402
    BOR_LINEAR_BACKWARD_ERROR_MAX,
    BOR_LINEAR_RESIDUAL_MAX,
    BorCrossOperators,
    BorPecSolver,
    _MultiRegionBor,
    _cell_points,
    _segment_distance,
    _mode_sweep,
    _solve_multiregion,
    solve_bor,
    solve_bor_coated_pec,
    solve_bor_coating_patch,
    solve_bor_dielectric,
    solve_bor_partial_coating,
)
from mie_sphere import (  # noqa: E402
    sigma_coated_pec_sphere,
    sigma_dielectric_sphere,
    sigma_impedance_sphere,
    sigma_pec_sphere,
)
from line_expand import SeamCoefficients, expand_perimeter  # noqa: E402


FREQUENCY_HZ = 1.0e9
SPHERE_ASPECTS_DEG = [0.0, 31.0, 73.0, 90.0, 137.0, 180.0]


def _sphere(radius_m, elements):
    theta = np.linspace(0.0, math.pi, int(elements) + 1)
    return np.column_stack((
        float(radius_m) * np.sin(theta),
        float(radius_m) * np.cos(theta),
    ))


def _hemisphere(radius_m, elements, upper=True):
    start, stop = (0.0, 0.5 * math.pi) if upper else (
        0.5 * math.pi, math.pi
    )
    theta = np.linspace(start, stop, int(elements) + 1)
    return np.column_stack((
        float(radius_m) * np.sin(theta),
        float(radius_m) * np.cos(theta),
    ))


def _bulged_upper_interface(radius_m, elements, bulge=0.12):
    theta = np.linspace(0.0, 0.5 * math.pi, int(elements) + 1)
    local_radius = float(radius_m) * (
        1.0 + float(bulge) * np.sin(2.0 * theta)
    )
    return np.column_stack((
        local_radius * np.sin(theta),
        local_radius * np.cos(theta),
    ))


def _error_db(numerical, reference):
    return abs(10.0 * math.log10(float(numerical) / float(reference)))


def _pec_sphere_snapshot(radius_m=0.04, explicit_elements=20):
    points = _sphere(radius_m, 12)
    pairs = [
        {
            "x1": float(points[index, 0]),
            "y1": float(points[index, 1]),
            "x2": float(points[index + 1, 0]),
            "y2": float(points[index + 1, 1]),
        }
        for index in range(len(points) - 1)
    ]
    return {
        "title": "PEC sphere",
        "segments": [{
            "name": "sphere",
            "seg_type": 2,
            "properties": ["2", str(explicit_elements), "0", "0", "0"],
            "point_pairs": pairs,
        }],
        "ibcs": [],
        "dielectrics": [],
    }


def _partial_coating_snapshot(radius_m=0.04, explicit_elements=6):
    def segment(name, seg_type, points, positive_material=0):
        return {
            "name": name,
            "seg_type": seg_type,
            "properties": [
                str(seg_type), str(explicit_elements), "0",
                str(positive_material), "0",
            ],
            "point_pairs": [
                {
                    "x1": float(points[index, 0]),
                    "y1": float(points[index, 1]),
                    "x2": float(points[index + 1, 0]),
                    "y2": float(points[index + 1, 1]),
                }
                for index in range(len(points) - 1)
            ],
        }

    return {
        "title": "Partial coated sphere",
        "segments": [
            segment(
                "coating interface", 3,
                _bulged_upper_interface(radius_m, 6), 1,
            ),
            segment(
                "covered core", 4,
                _hemisphere(radius_m, 6, upper=True), 1,
            ),
            segment(
                "bare core", 2,
                _hemisphere(radius_m, 6, upper=False), 0,
            ),
        ],
        "ibcs": [],
        "dielectrics": [["1", "2.5", "-0.08", "1.0", "-0.01"]],
    }


def _constant_compact_pattern(frequency_ghz=1.0):
    metadata = feature_sum.point_pattern_convention_metadata()
    amplitude = np.empty((2, 3, 1, 3), dtype=np.complex128)
    amplitude[..., 0] = 1.0 + 0.2j
    amplitude[..., 1] = 0.7 - 0.1j
    amplitude[..., 2] = 0.15 + 0.05j
    return {
        "azimuths": np.asarray([0.0, 360.0]),
        "elevations": np.asarray([-90.0, 0.0, 90.0]),
        "frequencies": np.asarray([float(frequency_ghz)]),
        "polarizations": np.asarray(["VV", "HH", "VH"]),
        "amp": amplitude,
        **metadata,
    }


class AnalyticSphereRegressionTests(unittest.TestCase):
    def _assert_spherical_channels(self, result, reference, tolerance_db):
        vv = np.asarray(result["sigma_vv"], dtype=float)
        hh = np.asarray(result["sigma_hh"], dtype=float)
        amp_vv = np.asarray(result["amp_vv"], dtype=complex)
        amp_hh = np.asarray(result["amp_hh"], dtype=complex)
        self.assertTrue(result["signed_mode_symmetry_used"])
        self.assertTrue(np.all(np.isfinite(vv)))
        self.assertTrue(np.all(np.isfinite(hh)))
        self.assertEqual(result["theta_deg"], SPHERE_ASPECTS_DEG)
        for value in vv:
            self.assertLess(_error_db(value, reference), tolerance_db)
        for value in hh:
            self.assertLess(_error_db(value, reference), tolerance_db)
        for value_vv, value_hh in zip(vv, hh):
            self.assertLess(_error_db(value_vv, value_hh), 0.08)
        np.testing.assert_allclose(
            vv, 4.0 * math.pi * np.abs(amp_vv) ** 2,
            rtol=2.0e-14, atol=0.0,
        )
        np.testing.assert_allclose(
            hh, 4.0 * math.pi * np.abs(amp_hh) ** 2,
            rtol=2.0e-14, atol=0.0,
        )
        # At an axial look the polarization basis is arbitrary.  A sphere is
        # also fore/aft symmetric, so both channels and both poles must share
        # one coherent complex amplitude, not merely the same RCS power.
        np.testing.assert_allclose(
            amp_vv[[0, -1]], amp_hh[[0, -1]],
            rtol=2.0e-12, atol=2.0e-13,
        )
        np.testing.assert_allclose(
            amp_vv[0], amp_vv[-1], rtol=2.0e-12, atol=2.0e-13,
        )

    def test_pec_sphere_matches_mie(self):
        radius = 3.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        result = solve_bor(
            _sphere(radius, 45), FREQUENCY_HZ, SPHERE_ASPECTS_DEG,
            formulation="cfie", workers=2,
            assembly="streaming", stream_budget_gb=0.25,
        )
        self._assert_spherical_channels(
            result, sigma_pec_sphere(radius, FREQUENCY_HZ), 0.08
        )

    def test_passive_impedance_sphere_matches_mie(self):
        radius = 1.5 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        impedance = 50.0 + 10.0j
        result = solve_bor(
            _sphere(radius, 45), FREQUENCY_HZ, SPHERE_ASPECTS_DEG,
            formulation="efie", zs=impedance, workers=2,
        )
        self._assert_spherical_channels(
            result,
            sigma_impedance_sphere(radius, FREQUENCY_HZ, impedance),
            0.10,
        )

    def test_lossy_dielectric_sphere_matches_mie(self):
        radius = 1.5 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        eps_r = 3.0 - 0.1j
        mu_r = 1.0 - 0.02j
        result = solve_bor_dielectric(
            _sphere(radius, 45), FREQUENCY_HZ, SPHERE_ASPECTS_DEG, eps_r, mu_r,
            workers=2,
        )
        self._assert_spherical_channels(
            result,
            sigma_dielectric_sphere(
                radius, eps_r, mu_r, FREQUENCY_HZ
            ),
            0.15,
        )

    def test_lossy_coated_pec_sphere_matches_mie(self):
        outer_radius = 0.1
        core_radius = 0.07
        eps_r = 3.0 - 0.1j
        mu_r = 1.0 - 0.02j
        result = solve_bor_coated_pec(
            _sphere(outer_radius, 60),
            _sphere(core_radius, 48),
            FREQUENCY_HZ,
            SPHERE_ASPECTS_DEG,
            eps_r,
            mu_r,
            workers=2,
        )
        self._assert_spherical_channels(
            result,
            sigma_coated_pec_sphere(
                core_radius, outer_radius, eps_r, mu_r, FREQUENCY_HZ
            ),
            0.15,
        )


class BoRWorkflowRegressionTests(unittest.TestCase):

    def test_certification_refines_realized_presegmented_mesh(self):
        snapshot = _pec_sphere_snapshot(
            radius_m=0.04, explicit_elements=0
        )
        wavelength = C0 / 1.0e6
        base_chains = bor_dispatch._chains_from_snapshot(snapshot, 1.0)

        fine = copy.deepcopy(snapshot)
        fine["segments"][0]["properties"][1] = "-30"
        fine["_bor_certification_refinement_factor"] = 1.5
        fine["_bor_certification_base_segment_n"] = ["0"]
        fine_chains = bor_dispatch._chains_from_snapshot(fine, 1.0)

        base_count = bor_dispatch._run_element_count(
            base_chains, wavelength
        )
        fine_count = bor_dispatch._run_element_count(
            fine_chains, wavelength
        )
        self.assertEqual(base_count, 12)
        self.assertEqual(fine_count, 24)
        self.assertGreater(fine_count, base_count)

    def test_near_depth_changes_graded_quadrature(self):
        shallow = _cell_points("diag", depth=1)
        deep = _cell_points("diag", depth=5)
        self.assertGreater(len(deep[0]), len(shallow[0]))
        self.assertIsNot(shallow[0], deep[0])
        with self.assertRaisesRegex(ValueError, "near_depth"):
            BorPecSolver(_sphere(0.04, 4), FREQUENCY_HZ, near_depth=-1)

    def test_axis_transform_enforces_exact_m1_pole_regularity(self):
        solver = BorPecSolver(_sphere(0.04, 8), FREQUENCY_HZ)
        for mode in (-1, 1):
            transform = solver.basis_transform(mode)
            for end, element in (
                (0, 0),
                (solver.Nn - 1, solver.gen.n_elems - 1),
            ):
                columns = np.flatnonzero(np.abs(transform[end]) > 0.0)
                self.assertEqual(columns.size, 1)
                column = int(columns[0])
                radial_sign = (
                    1.0 if solver.gen.trho[element] >= 0.0 else -1.0
                )
                self.assertEqual(
                    transform[solver.Nn + end, column],
                    1j * mode * radial_sign,
                )

    def test_production_entry_points_have_no_polarization_selector(self):
        for entry_point in (
            bor_dispatch.solve_monostatic_rcs_bor,
            bor_dispatch.solve_monostatic_rcs_bor_certified,
            bor_dispatch.solve_monostatic_rcs_bor_survey,
        ):
            self.assertNotIn(
                "polarization", inspect.signature(entry_point).parameters
            )

    def test_body_solver_certificate_round_trip_requires_both_channels(self):
        quality = {"passed": True}
        mesh = {
            "schema": "ghost.solver.mesh-convergence.v1",
            "passed": True,
            "published_mesh": "fine",
            "co_solved_polarizations": ["VV", "HH"],
            "polarizations": {
                "VV": {"passed": True},
                "HH": {"passed": True},
            },
            "base_quality_gate": quality,
            "fine_quality_gate": quality,
        }
        diagnostics = {1.0: {
            "solver": "bor_mom_rcs",
            "scattering_mode": "monostatic",
            "polarizations": ["VV", "HH"],
            "polarization_mapping": {"VV": "VV", "HH": "HH"},
            "certification_frequency_scope": "single_frequency_unit",
            "certification_frequency_scope_ghz": [1.0],
            "metadata": {
                "mesh_convergence_certified": True,
                "certified_entry_point": True,
                "quality_gate": quality,
                "mesh_convergence": mesh,
            },
        }}
        amplitude = 1.0 / math.sqrt(4.0 * math.pi)
        bodies = {1.0: {
            "theta_deg": np.asarray([0.0, 90.0, 180.0]),
            "amp_vv": np.asarray([amplitude] * 3, complex),
            "amp_hh": np.asarray([amplitude] * 3, complex),
        }}
        profile = np.asarray([[0.0, 1.0], [0.2, 0.0], [0.0, -1.0]])
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "certified.grim"
            feature_sum.save_monostatic_grim(
                bodies,
                profile,
                str(valid),
                azimuths_deg=[0.0, 90.0],
                elevations_deg=[0.0],
                solver_diagnostics=diagnostics,
            )
            certificate = feature_sum.require_body_mesh_certification(
                str(valid)
            )
            self.assertTrue(certificate["passed"])

            malformed = copy.deepcopy(diagnostics)
            malformed[1.0]["polarizations"] = ["VV"]
            with self.assertRaisesRegex(ValueError, "dual-channel"):
                feature_sum.save_monostatic_grim(
                    bodies,
                    profile,
                    str(Path(directory) / "missing_hh.grim"),
                    azimuths_deg=[0.0, 90.0],
                    elevations_deg=[0.0],
                    solver_diagnostics=malformed,
                )

            wrong_scope = copy.deepcopy(diagnostics)
            wrong_scope[1.0]["certification_frequency_scope_ghz"] = [2.0]
            with self.assertRaisesRegex(ValueError, "frequency-certification"):
                feature_sum.save_monostatic_grim(
                    bodies,
                    profile,
                    str(Path(directory) / "wrong_scope.grim"),
                    azimuths_deg=[0.0, 90.0],
                    elevations_deg=[0.0],
                    solver_diagnostics=wrong_scope,
                )

            with self.assertRaisesRegex(ValueError, "cannot be injected"):
                feature_sum.save_monostatic_grim(
                    bodies,
                    profile,
                    str(Path(directory) / "injected.grim"),
                    azimuths_deg=[0.0, 90.0],
                    elevations_deg=[0.0],
                    artifact_metadata={"solver_metadata_json": "{}"},
                )

            failed_hh = copy.deepcopy(diagnostics)
            failed_hh[1.0]["metadata"]["mesh_convergence"][
                "polarizations"
            ]["HH"]["passed"] = False
            failed = Path(directory) / "failed_hh.grim"
            feature_sum.save_monostatic_grim(
                bodies,
                profile,
                str(failed),
                azimuths_deg=[0.0, 90.0],
                elevations_deg=[0.0],
                solver_diagnostics=failed_hh,
            )
            with self.assertRaisesRegex(ValueError, "lacks passed VV/HH"):
                feature_sum.require_body_mesh_certification(str(failed))

            nonbody = Path(directory) / "not_a_body.grim"
            with nonbody.open("wb") as stream:
                np.savez(
                    stream,
                    frequencies=np.asarray([], dtype=float),
                    solver_metadata_json=np.asarray("{}"),
                )
            with self.assertRaisesRegex(
                ValueError, "certified GHOST BoR body"
            ):
                feature_sum.require_body_mesh_certification(str(nonbody))

    def test_bor_unit_audits_are_advisory_but_integrity_checked(self):
        metadata = {
            "mesh_convergence_certified": False,
            "certified_entry_point": False,
            "quality_gate": {"passed": True},
        }

        def unit(polarization):
            return {
                "stem": "body",
                "freq_ghz": 1.0,
                "polarizations": [polarization],
                "path": Path(f"{polarization}_1.000GHz_body.grim"),
                "solver_audit": {
                    "schema": grim_io.SOLVER_METADATA_SCHEMA,
                    "solver": "bor_mom_rcs",
                    "scattering_mode": "monostatic",
                    "polarization": polarization,
                    "polarization_export": polarization,
                    "polarizations": [],
                    "polarization_mapping": {},
                    "rcs_log_unit": "dBsm",
                    "rcs_linear_quantity": "sigma_3d",
                    "metadata": copy.deepcopy(metadata),
                },
            }

        records = [unit("VV"), unit("HH")]
        collected = hpc_common.bor_solver_diagnostics_from_units(
            records, stem="body"
        )
        self.assertEqual(set(collected), {1.0})
        self.assertEqual(collected[1.0]["polarizations"], ["VV", "HH"])

        disagreeing = copy.deepcopy(records)
        disagreeing[1]["solver_audit"]["metadata"]["quality_gate"][
            "passed"
        ] = False
        with self.assertRaisesRegex(ValueError, "diagnostics disagree"):
            hpc_common.bor_solver_diagnostics_from_units(
                disagreeing, stem="body"
            )

        certification_different = copy.deepcopy(records)
        certification_different[1]["solver_audit"]["metadata"].update({
            "mesh_convergence_certified": True,
            "certified_entry_point": True,
            "published_mesh": "fine",
            "mesh_convergence": {
                "schema": "ghost.solver.mesh-convergence.v1",
                "passed": True,
                "published_mesh": "fine",
            },
        })
        self.assertIsNone(
            hpc_common.bor_solver_diagnostics_from_units(
                certification_different, stem="body"
            )
        )

        partially_audited = copy.deepcopy(records)
        partially_audited[1]["solver_audit"] = None
        self.assertIsNone(
            hpc_common.bor_solver_diagnostics_from_units(
                partially_audited, stem="body"
            )
        )

        with self.assertRaisesRegex(ValueError, "exactly one VV and one HH"):
            hpc_common.bor_solver_diagnostics_from_units(
                records[:1], stem="body"
            )

        missing_schema = copy.deepcopy(records)
        missing_schema[1]["solver_audit"].pop("schema")
        with self.assertRaisesRegex(ValueError, "not a monostatic BoR"):
            hpc_common.bor_solver_diagnostics_from_units(
                missing_schema, stem="body"
            )

        wrong_attestation = copy.deepcopy(records)
        wrong_attestation[1]["solver_audit"]["metadata"][
            "output_attestation"
        ] = {"frequency_ghz": 2.0, "polarization": "HH"}
        with self.assertRaisesRegex(ValueError, "does not match its field"):
            hpc_common.bor_solver_diagnostics_from_units(
                wrong_attestation, stem="body"
            )

        with tempfile.TemporaryDirectory() as directory:
            mismatched = Path(directory) / "VV_1.000GHz_body.grim"
            with mismatched.open("wb") as stream:
                np.savez(
                    stream,
                    raw_complex_amplitude_preserved=np.asarray(True),
                    rcs_amp_real=np.zeros((1, 1, 1, 1)),
                    rcs_amp_imag=np.zeros((1, 1, 1, 1)),
                    polarizations=np.asarray(["HH"]),
                    azimuths=np.asarray([0.0]),
                    frequencies=np.asarray([1.0]),
                )
            with self.assertRaisesRegex(ValueError, "does not match stored"):
                hpc_common.read_unit_grims(directory)

    def test_declared_gui_compact_subtraction_reconstructs_physical_field(self):
        source = _constant_compact_pattern()
        source["azimuths"] = np.arange(
            0.0, 360.0, 0.1, dtype=np.float32
        )
        amplitude = np.repeat(
            source["amp"][:1], len(source["azimuths"]), axis=0
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui_coherent_subtraction.grim"
            payload = {
                "azimuths": source["azimuths"],
                "elevations": source["elevations"],
                "frequencies": source["frequencies"],
                "polarizations": source["polarizations"],
                "rcs_power": (
                    4.0 * math.pi * np.abs(amplitude) ** 2
                ).astype(np.float32),
                "rcs_phase": np.angle(amplitude).astype(np.float32),
                "rcs_domain": np.asarray("power-phase"),
                "power_domain": np.asarray("linear_rcs"),
                "units": np.asarray(json.dumps({
                    "azimuth": "deg", "elevation": "deg",
                    "frequency": "GHz",
                    "rcs_linear_quantity": "sigma_3d",
                })),
            }
            with path.open("wb") as stream:
                np.savez(stream, **payload)

            with self.assertRaisesRegex(ValueError, "rcs_domain"):
                feature_sum.prepare_point_pattern(str(path))
            prepared = feature_sum.prepare_point_pattern(
                str(path), declared_coherent_delta=True
            )
            self.assertEqual(len(prepared.azimuths), 3601)
            self.assertEqual(prepared.azimuths[-1], 360.0)
            np.testing.assert_allclose(
                prepared.amplitude[:-1], amplitude,
                rtol=2.0e-6, atol=2.0e-7
            )
            np.testing.assert_array_equal(
                prepared.amplitude[-1], prepared.amplitude[0]
            )

            # The POINT_FEATURE_DATASETS entry itself declares the role. A GUI is
            # allowed to omit the redundant domain tag entirely.
            payload.pop("rcs_domain")
            with path.open("wb") as stream:
                np.savez(stream, **payload)
            untagged = feature_sum.prepare_point_pattern(
                str(path), declared_coherent_delta=True,
                delta_sign=-1.0,
            )
            np.testing.assert_allclose(
                untagged.amplitude[:-1], -amplitude,
                rtol=2.0e-6, atol=2.0e-7,
            )
            payload["rcs_domain"] = np.asarray("power-phase")

            two_pol = dict(payload)
            two_pol["polarizations"] = np.asarray(["VV", "HH"])
            two_pol["rcs_power"] = payload["rcs_power"][..., :2]
            two_pol["rcs_phase"] = payload["rcs_phase"][..., :2]
            with path.open("wb") as stream:
                np.savez(stream, **two_pol)
            with self.assertRaisesRegex(ValueError, r"missing \['VH'\]"):
                feature_sum.prepare_point_pattern(
                    str(path), declared_coherent_delta=True
                )
            diagonal = feature_sum.prepare_point_pattern(
                str(path), declared_coherent_delta=True,
                assume_missing_cross_pol_zero=True,
            )
            np.testing.assert_array_equal(diagonal.amplitude[..., 2], 0.0)

            # GUI-derived grids can carry stale source semantics. The explicit
            # feature declaration supersedes those strings.
            payload["phase_reference"] = np.asarray("wrong origin")
            with path.open("wb") as stream:
                np.savez(stream, **payload)
            stale_metadata = feature_sum.prepare_point_pattern(
                str(path), declared_coherent_delta=True
            )
            self.assertEqual(len(stale_metadata.azimuths), 3601)

            payload.pop("phase_reference")
            payload["azimuths"] = source["azimuths"][:-1]
            for key in ("rcs_power", "rcs_phase"):
                payload[key] = payload[key][:-1]
            with path.open("wb") as stream:
                np.savez(stream, **payload)
            with self.assertRaisesRegex(ValueError, "Partial data"):
                feature_sum.prepare_point_pattern(
                    str(path), declared_coherent_delta=True
                )

    def test_declared_gui_line_delta_needs_no_semantic_metadata(self):
        frequency = 2.0
        angles = np.asarray([60.0, 90.0, 120.0])
        amplitudes = np.empty((3, 1, 1, 2), dtype=np.complex128)
        amplitudes[..., 0] = np.asarray([
            1.0 + 0.2j, 1.2 - 0.1j, 0.9 + 0.3j
        ])[:, None, None]
        amplitudes[..., 1] = np.asarray([
            0.5 - 0.4j, 0.7 + 0.2j, 0.6 - 0.1j
        ])[:, None, None]
        wave_number = 2.0 * math.pi * frequency * 1.0e9 / C0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui_2d_subtraction.grim"
            payload = {
                "azimuths": angles,
                "elevations": np.asarray([0.0]),
                "frequencies": np.asarray([frequency]),
                "polarizations": np.asarray(["VV", "HH"]),
                "rcs_power": (
                    np.abs(amplitudes) ** 2 / (4.0 * wave_number)
                ).astype(np.float32),
                "rcs_phase": np.angle(amplitudes).astype(np.float32),
                "units": np.asarray(json.dumps({
                    "azimuth": "deg", "elevation": "deg",
                    "frequency": "GHz",
                    "rcs_linear_quantity": "sigma_2d",
                })),
            }
            with path.open("wb") as stream:
                np.savez(stream, **payload)

            with self.assertRaisesRegex(ValueError, "rcs_domain"):
                feature_sum.load_seam_from_grim(str(path), frequency)
            positive = feature_sum.load_seam_from_grim(
                str(path), frequency, declared_coherent_delta=True
            )
            negative = feature_sum.load_seam_from_grim(
                str(path), frequency, declared_coherent_delta=True,
                delta_sign=-1.0,
            )
            np.testing.assert_allclose(
                positive.dA_te, amplitudes[:, 0, 0, 0],
                rtol=2.0e-6, atol=2.0e-7,
            )
            np.testing.assert_allclose(
                positive.dA_tm, amplitudes[:, 0, 0, 1],
                rtol=2.0e-6, atol=2.0e-7,
            )
            np.testing.assert_allclose(negative.dA_te, -positive.dA_te)
            np.testing.assert_allclose(negative.dA_tm, -positive.dA_tm)

            payload["phase_reference"] = np.asarray("wrong origin")
            with path.open("wb") as stream:
                np.savez(stream, **payload)
            stale_metadata = feature_sum.load_seam_from_grim(
                str(path), frequency, declared_coherent_delta=True
            )
            np.testing.assert_allclose(
                stale_metadata.dA_te, positive.dA_te
            )

            placement = {
                "delta": str(path), "perimeter": np.zeros((1, 2, 3)),
                "kind": "delta", "declared_coherent_delta": True,
            }
            cache = {}
            with mock.patch.object(
                feature_sum, "_load_grim", wraps=feature_sum._load_grim
            ) as loader:
                feature_sum._prepared_line_placements_at_frequency(
                    [placement], frequency, cache
                )
                feature_sum._prepared_line_placements_at_frequency(
                    [placement], frequency, cache
                )
            self.assertEqual(loader.call_count, 1)

    def test_indexed_facet_surface_preserves_skin_and_winding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "platform.facet"
            path.write_text(
                "4 1\n"
                "10 0 0 0\n"
                "20 1 0 0\n"
                "30 1 1 0\n"
                "40 0 1 0\n"
                "7 10 20 30 40\n",
                encoding="ascii",
            )
            # NumPy releases on older HPC images reject tuple reduction axes
            # in np.ptp. Surface loading/construction must not depend on it.
            with mock.patch.object(
                surface_mesh.np, "ptp",
                side_effect=TypeError("tuple axis unsupported"),
            ):
                triangles = surface_mesh.read_surface_mesh(str(path))
                self.assertEqual(triangles.shape, (2, 3, 3))
                surface = surface_mesh.TriangleSurface(triangles)
            distance, closest, normal, _index = surface.nearest(
                [[0.25, 0.75, 0.2], [1.2, 0.5, 0.0]]
            )
            np.testing.assert_allclose(distance, [0.2, 0.2], atol=2.0e-15)
            np.testing.assert_allclose(
                closest, [[0.25, 0.75, 0.0], [1.0, 0.5, 0.0]], atol=2.0e-15
            )
            np.testing.assert_allclose(normal, [[0, 0, 1], [0, 0, 1]])

    def test_external_monostatic_grim_accepts_coherent_feature_addition(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "external.grim"
            combined = Path(directory) / "external_with_cavity.grim"
            shadowed = Path(directory) / "external_shadowed_cavity.grim"
            gui_base = Path(directory) / "gui_platform.grim"
            gui_combined = Path(directory) / "gui_platform_with_cavity.grim"
            vv_base = Path(directory) / "gui_platform_vv_only.grim"
            vv_combined = Path(directory) / "gui_platform_vv_with_cavity.grim"
            grid = {
                "frequencies_ghz": [1.0],
                "azimuths_deg": [0.0],
                "elevations_deg": [0.0],
                "axis_az_deg": 0.0,
                "axis_el_deg": 0.0,
                "roll_deg": 0.0,
            }
            feature_sum.export_radar_grim(
                str(base), bor_result=None, placements=[], **grid
            )
            point = {
                "pattern": feature_sum.prepare_point_pattern(
                    _constant_compact_pattern()
                ),
                "location": [0.0, 0.0, 0.0],
                "aperture_normal": [0.0, 0.0, 1.0],
                "roll_ref": [1.0, 0.0, 0.0],
            }
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(combined), points=[point], radar_grid=grid
            )
            blocker = occluder.Occluder(np.asarray([[
                [-1.0, -1.0, 0.5],
                [1.0, -1.0, 0.5],
                [0.0, 1.0, 0.5],
            ]]), bias=1.0e-9)
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(shadowed), points=[point], radar_grid=grid,
                occluder=blocker,
            )
            with np.load(base, allow_pickle=False) as original, np.load(
                combined, allow_pickle=False
            ) as updated, np.load(shadowed, allow_pickle=False) as blocked:
                original_field = original["rcs_amp_real"] + 1j * original["rcs_amp_imag"]
                updated_field = updated["rcs_amp_real"] + 1j * updated["rcs_amp_imag"]
                blocked_field = blocked["rcs_amp_real"] + 1j * blocked["rcs_amp_imag"]
            self.assertTrue(np.all(original_field == 0.0))
            self.assertGreater(float(np.max(np.abs(updated_field))), 0.0)
            np.testing.assert_array_equal(blocked_field, original_field)

            with np.load(base, allow_pickle=False) as stored:
                gui_payload = {
                    key: np.array(stored[key], copy=True)
                    for key in stored.files
                }
            for key in (
                "combine_role", "combine_role_note", "phase_reference",
                "amplitude_convention", "complex_field_domain",
                "raw_complex_amplitude_preserved", "rcs_amp_real",
                "rcs_amp_imag",
            ):
                gui_payload.pop(key, None)
            gui_payload["rcs_domain"] = np.asarray("power-phase")
            gui_units = json.loads(str(gui_payload["units"]))
            gui_units.pop("rcs_log_unit", None)
            gui_payload["units"] = np.asarray(json.dumps(gui_units))
            order = [2, 0, 1]
            gui_payload["polarizations"] = np.asarray(["HV", "VV", "HH"])
            for key in ("rcs_power", "rcs_phase"):
                gui_payload[key] = np.asarray(gui_payload[key])[..., order]
            with gui_base.open("wb") as stream:
                np.savez(stream, **gui_payload)

            with self.assertRaisesRegex(ValueError, "missing combine_role"):
                feature_sum.add_features_to_monostatic_grim(
                    str(gui_base), str(gui_combined), points=[point],
                    radar_grid=grid,
                )
            feature_sum.add_features_to_monostatic_grim(
                str(gui_base), str(gui_combined), points=[point],
                radar_grid=grid, declared_coherent_base=True,
            )
            with np.load(gui_combined, allow_pickle=False) as updated:
                self.assertEqual(str(updated["combine_role"]), "coherent")
                self.assertTrue(bool(updated["raw_complex_amplitude_preserved"]))
                self.assertIn("rcs_amp_real", updated.files)
                self.assertIn("rcs_amp_imag", updated.files)
                np.testing.assert_array_equal(
                    updated["polarizations"], ["VV", "HH", "VH"]
                )

            vv_payload = dict(gui_payload)
            vv_payload["polarizations"] = np.asarray(["V"])
            for key in ("rcs_power", "rcs_phase"):
                vv_payload[key] = np.asarray(gui_payload[key])[..., [1]]
            with vv_base.open("wb") as stream:
                np.savez(stream, **vv_payload)
            with self.assertRaisesRegex(
                ValueError, "require VV, HH, and VH/HV"
            ):
                feature_sum.add_features_to_monostatic_grim(
                    str(vv_base), str(vv_combined), points=[point],
                    radar_grid=grid, declared_coherent_base=True,
                )
            self.assertFalse(vv_combined.exists())

            gui_payload["combine_role"] = np.asarray("power")
            with gui_base.open("wb") as stream:
                np.savez(stream, **gui_payload)
            with self.assertRaisesRegex(ValueError, "power-only"):
                feature_sum.add_features_to_monostatic_grim(
                    str(gui_base), str(gui_combined), points=[point],
                    radar_grid=grid, declared_coherent_base=True,
                )

    def test_feature_output_drops_stale_solver_certification_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "certified_platform.grim"
            combined = Path(directory) / "platform_with_feature.grim"
            grid = {
                "frequencies_ghz": [1.0],
                "azimuths_deg": [0.0],
                "elevations_deg": [0.0],
                "axis_az_deg": 0.0,
                "axis_el_deg": 0.0,
                "roll_deg": 0.0,
            }
            feature_sum.export_radar_grim(
                str(base), bor_result=None, placements=[], **grid
            )
            with np.load(base, allow_pickle=False) as stored:
                payload = {
                    key: np.array(stored[key], copy=True)
                    for key in stored.files
                }
            payload.update({
                "solver_metadata_json": np.asarray(json.dumps({
                    "schema": grim_io.SOLVER_METADATA_SCHEMA,
                    "metadata": {"mesh_convergence_certified": True},
                })),
                "production_mesh_certification_json": np.asarray(
                    '{"passed":true}'
                ),
                "source_body_mesh_certification_json": np.asarray(
                    '{"passed":true}'
                ),
            })
            with base.open("wb") as stream:
                np.savez_compressed(stream, **payload)

            point = {
                "pattern": feature_sum.prepare_point_pattern(
                    _constant_compact_pattern()
                ),
                "location": [0.0, 0.0, 0.0],
                "aperture_normal": [0.0, 0.0, 1.0],
                "roll_ref": [1.0, 0.0, 0.0],
            }
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(combined), points=[point], radar_grid=grid
            )
            with np.load(combined, allow_pickle=False) as updated:
                for key in (
                    "solver_metadata_json",
                    "production_mesh_certification_json",
                    "source_body_mesh_certification_json",
                ):
                    self.assertNotIn(key, updated.files)
                field = (
                    np.asarray(updated["rcs_amp_real"], dtype=float)
                    + 1j * np.asarray(updated["rcs_amp_imag"], dtype=float)
                )
                self.assertGreater(float(np.max(np.abs(field))), 0.0)
                self.assertIn("feature_provenance_json", updated.files)

    def test_feature_reconstruction_comparison_does_not_fit_away_phase_error(self):
        amplitude = 1.0 / math.sqrt(4.0 * math.pi)
        bodies = {1.0: {
            "theta_deg": np.asarray([0.0, 90.0]),
            "amp_vv": np.asarray([amplitude, 2.0 * amplitude], complex),
            "amp_hh": np.asarray([1.5 * amplitude, 0.8 * amplitude], complex),
        }}
        profile = np.asarray([[0.0, 1.0], [0.2, 0.0], [0.0, -1.0]])
        with tempfile.TemporaryDirectory() as directory:
            truth = Path(directory) / "truth.grim"
            shifted = Path(directory) / "shifted.grim"
            feature_sum.save_monostatic_grim(
                bodies, profile, str(truth),
                azimuths_deg=[0.0, 90.0], elevations_deg=[0.0],
            )
            exact = validate_feature_reconstruction.compare_grims(truth, truth)
            self.assertTrue(exact["passed"])
            self.assertEqual(exact["normalized_complex_rms"], 0.0)
            with np.load(truth, allow_pickle=False) as source:
                payload = {
                    key: np.array(source[key], copy=True) for key in source.files
                }
            phase = np.exp(1j * math.radians(40.0))
            field = (
                payload["rcs_amp_real"] + 1j * payload["rcs_amp_imag"]
            ) * phase
            payload["rcs_amp_real"] = field.real
            payload["rcs_amp_imag"] = field.imag
            payload["rcs_phase"] = np.angle(field).astype(np.float32)
            payload["rcs_power"] = (
                4.0 * math.pi * np.abs(field) ** 2
            ).astype(np.float32)
            grim_io._save_grim_npz(payload, str(shifted))
            result = validate_feature_reconstruction.compare_grims(
                truth, shifted
            )
            self.assertFalse(result["passed"])
            self.assertAlmostEqual(
                result["best_fit_global_phase_diagnostic_deg"], 40.0, places=10
            )
            self.assertGreater(result["phase_error_rms_deg"], 39.9)

    def test_compact_feature_translation_is_exact_two_way_phase(self):
        frequency = 1.0
        direction = np.asarray([[0.6, 0.0, 0.8]], dtype=float)
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        location = np.asarray([0.037, -0.011, 0.023])
        common = dict(
            pattern=_constant_compact_pattern(frequency),
            aperture_normal=[0.0, 0.0, 1.0],
            directions=direction,
            frequency_ghz=frequency,
            roll_ref=[1.0, 0.0, 0.0],
        )
        origin = feature_sum.point_scatterer_amplitude(
            location=[0.0, 0.0, 0.0], **common
        )
        translated = feature_sum.point_scatterer_amplitude(
            location=location, **common
        )
        wave_number = 2.0 * math.pi * frequency * 1.0e9 / C0
        expected = np.exp(2j * wave_number * float(direction[0] @ location))
        for channel in ("F_vv", "F_hh", "F_vh"):
            self.assertGreater(abs(origin[channel][0]), 1.0e-12)
            np.testing.assert_allclose(
                translated[channel], origin[channel] * expected,
                rtol=2.0e-14, atol=2.0e-14,
            )

    def test_line_expansion_is_invariant_to_segment_splitting(self):
        coefficients = SeamCoefficients(
            1.0,
            np.asarray([0.0, 90.0, 180.0]),
            np.asarray([1.0 + 0.2j] * 3),
            np.asarray([0.7 - 0.1j] * 3),
        )
        whole = np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]]])
        split = np.asarray([
            [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]],
            [[0.0, 0.01, 0.0], [0.0, 0.02, 0.0]],
        ])
        normal = lambda points: np.tile(  # noqa: E731
            np.asarray([1.0, 0.0, 0.0]), (len(points), 1)
        )
        common = dict(
            coefficients=coefficients,
            normal_fn=normal,
            directions=np.asarray([[1.0, 0.0, 0.0]]),
            frequency_ghz=1.0,
            max_piece_wavelengths=0.05,
        )
        first = expand_perimeter(whole, **common)
        second = expand_perimeter(split, **common)
        for channel in ("F_vv", "F_hh", "F_vh"):
            np.testing.assert_allclose(
                first[channel], second[channel], rtol=2.0e-14, atol=2.0e-14
            )

    def test_one_monostatic_artifact_is_reusable_for_feature_addition(self):
        amplitude = 1.0 / math.sqrt(4.0 * math.pi)
        bodies = {
            1.0: {
                "theta_deg": np.asarray([0.0, 90.0]),
                "amp_vv": np.asarray([amplitude, 2.0 * amplitude], complex),
                "amp_hh": np.asarray([3.0 * amplitude, 4.0 * amplitude], complex),
            }
        }
        profile = np.asarray([[0.0, 1.0], [0.2, 0.0], [0.0, -1.0]])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "body.grim"
            combined = Path(directory) / "body_features.grim"
            feature_sum.save_monostatic_grim(
                bodies, profile, str(base),
                azimuths_deg=[0.0, 90.0], elevations_deg=[0.0],
                roll_deg=12.0,
            )
            loaded = feature_sum.load_body_grim(str(base))
            np.testing.assert_array_equal(
                loaded[1.0]["theta_deg"], bodies[1.0]["theta_deg"]
            )
            grid = feature_sum.load_body_requested_radar_grid(str(base))
            self.assertEqual(grid["roll_deg"], 12.0)
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(combined), feature_provenance={"test": True}
            )
            with np.load(base, allow_pickle=False) as original, np.load(
                combined, allow_pickle=False
            ) as updated:
                np.testing.assert_array_equal(
                    updated["rcs_amp_real"], original["rcs_amp_real"]
                )
                np.testing.assert_array_equal(
                    updated["rcs_amp_imag"], original["rcs_amp_imag"]
                )
                provenance = json.loads(str(updated["feature_provenance_json"]))
            self.assertEqual(provenance[-1]["line_feature_count"], 0)
            self.assertEqual(provenance[-1]["compact_feature_count"], 0)
            self.assertEqual(
                provenance[-1]["line_phase_mapping_deg"],
                {"TM": feature_sum.PSI_HH_DEG, "TE": feature_sum.PSI_VV_DEG},
            )
            self.assertFalse(
                provenance[-1]["model_scope"]["body_feature_mutual_coupling"]
            )

            pattern = feature_sum.prepare_point_pattern(
                _constant_compact_pattern()
            )
            point_a = {
                "pattern": pattern,
                "location": [0.01, 0.0, 0.02],
                "aperture_normal": [0.0, 0.0, 1.0],
                "roll_ref": [1.0, 0.0, 0.0],
            }
            point_b = {
                "pattern": pattern,
                "location": [0.0, 0.015, 0.01],
                "aperture_normal": [0.0, 0.0, 1.0],
                "roll_ref": [1.0, 0.0, 0.0],
            }
            direct = Path(directory) / "direct.grim"
            point_a_total = Path(directory) / "point_a_total.grim"
            point_b_total = Path(directory) / "point_b_total.grim"
            sequential = Path(directory) / "sequential.grim"
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(direct), points=[point_a, point_b]
            )
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(point_a_total), points=[point_a]
            )
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(point_b_total), points=[point_b]
            )

            direct_delta_path = Path(
                feature_sum.feature_only_output_path(str(direct))
            )
            point_a_delta_path = Path(
                feature_sum.feature_only_output_path(str(point_a_total))
            )
            point_b_delta_path = Path(
                feature_sum.feature_only_output_path(str(point_b_total))
            )

            def coherent_field(payload):
                return (
                    payload["rcs_amp_real"]
                    + 1j * payload["rcs_amp_imag"]
                )

            with np.load(base, allow_pickle=False) as clean, np.load(
                direct, allow_pickle=False
            ) as direct_total, np.load(
                direct_delta_path, allow_pickle=False
            ) as direct_delta, np.load(
                point_a_total, allow_pickle=False
            ) as total_a, np.load(
                point_a_delta_path, allow_pickle=False
            ) as delta_a, np.load(
                point_b_total, allow_pickle=False
            ) as total_b, np.load(
                point_b_delta_path, allow_pickle=False
            ) as delta_b:
                clean_field = coherent_field(clean)
                direct_field = coherent_field(direct_total)
                direct_delta_field = coherent_field(direct_delta)
                delta_a_field = coherent_field(delta_a)
                delta_b_field = coherent_field(delta_b)
                np.testing.assert_allclose(
                    coherent_field(total_a),
                    clean_field + delta_a_field,
                    rtol=2.0e-14,
                    atol=2.0e-14,
                )
                np.testing.assert_allclose(
                    coherent_field(total_b),
                    clean_field + delta_b_field,
                    rtol=2.0e-14,
                    atol=2.0e-14,
                )
                self.assertEqual(
                    str(direct_total["assembly_response_role"]),
                    "body_plus_features",
                )
                self.assertEqual(
                    str(direct_delta["assembly_response_role"]),
                    "features_only_delta",
                )
            np.testing.assert_allclose(
                direct_delta_field,
                delta_a_field + delta_b_field,
                rtol=2.0e-14,
                atol=2.0e-14,
            )
            np.testing.assert_allclose(
                direct_field,
                clean_field + delta_a_field + delta_b_field,
                rtol=2.0e-14,
                atol=2.0e-14,
            )

            # A clean base can be reused for independent trade-study builds,
            # but a body-plus-features result must not become the next base.
            # That sequential workflow obscures cross-batch coupling and can
            # make downstream tree combinations count the body twice.
            with self.assertRaisesRegex(
                ValueError, "cannot add another batch to a feature-bearing base"
            ):
                feature_sum.add_features_to_monostatic_grim(
                    str(point_a_total), str(sequential), points=[point_b]
                )
            self.assertFalse(sequential.exists())
            self.assertFalse(
                Path(feature_sum.feature_only_output_path(str(sequential))).exists()
            )

    def test_dispatch_builds_each_frequency_on_its_own_mesh(self):
        snapshot = _pec_sphere_snapshot(explicit_elements=-20)
        point_counts = []

        def fake_solve(points, freq_hz, thetas_deg, **_kwargs):
            point_counts.append((float(freq_hz), len(points)))
            amplitude = 1.0 / math.sqrt(4.0 * math.pi)
            count = len(thetas_deg)
            return {
                "sigma_vv": [1.0] * count,
                "sigma_hh": [1.0] * count,
                "amp_vv": [amplitude + 0.0j] * count,
                "amp_hh": [amplitude + 0.0j] * count,
                "modes_used": 3,
                "mode_cap": 8,
                "mode_converged": True,
                "mode_quiet_count": 2,
                "mode_last_relative_increment": 0.0,
                "n_unknowns": 2 * len(points),
                "linear_residual": 1.0e-12,
                "linear_backward_error": 1.0e-16,
                "max_cond": 10.0,
                "condition_est_computed": True,
                "condition_est_method": "test",
            }

        with mock.patch.object(bor_dispatch, "solve_bor", fake_solve):
            result = bor_dispatch.solve_monostatic_rcs_bor(
                snapshot,
                [1.0, 4.0],
                [0.0, 90.0],
                geometry_units="meters",
                workers=1,
            )
        self.assertEqual([entry[0] for entry in point_counts], [1.0e9, 4.0e9])
        self.assertGreater(point_counts[1][1], point_counts[0][1])
        self.assertEqual(result["polarizations"], ["VV", "HH"])
        self.assertEqual(
            result["polarization_mapping"], {"VV": "VV", "HH": "HH"}
        )
        diagnostics = feature_sum.bor_solver_diagnostics_by_frequency(result)
        self.assertEqual(set(diagnostics), {1.0, 4.0})
        self.assertTrue(all(
            record["certification_frequency_scope"]
            == "joint_requested_frequency_grid"
            and record["certification_frequency_scope_ghz"] == [1.0, 4.0]
            for record in diagnostics.values()
        ))
        bodies = feature_sum.bodies_from_bor_solver_result(result)
        self.assertEqual(set(bodies), {1.0, 4.0})
        np.testing.assert_array_equal(
            bodies[1.0]["theta_deg"], [0.0, 90.0]
        )
        with tempfile.TemporaryDirectory() as directory:
            written = grim_io.export_result_to_grim(
                result, str(Path(directory) / "dual_bor")
            )
            with np.load(written[0], allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["polarizations"], ["VV", "HH"]
                )
                self.assertEqual(
                    str(payload["polarization_alias_primary"]), "VV,HH"
                )
        per_frequency = result["metadata"]["per_frequency"]
        self.assertLess(
            per_frequency[1]["mesh_wavelength_m"],
            per_frequency[0]["mesh_wavelength_m"],
        )

    def test_streaming_and_table_paths_are_equivalent(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        points = _sphere(radius, 18)
        common = dict(
            points=points,
            freq_hz=FREQUENCY_HZ,
            thetas_deg=[0.0, 37.0, 90.0],
            formulation="cfie",
            n_modes=8,
            workers=1,
            table_precision="double",
        )
        table = solve_bor(assembly="tables", **common)
        stream = solve_bor(
            assembly="streaming", stream_budget_gb=0.25, **common
        )
        self.assertIn(
            stream["stream_sampling_backend"], {"native_c", "numpy"}
        )
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
            )

    def test_dielectric_streaming_and_table_paths_are_equivalent(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        common = dict(
            points=_sphere(radius, 10),
            freq_hz=FREQUENCY_HZ,
            thetas_deg=[0.0, 37.0, 90.0],
            eps_r=3.0 - 0.1j,
            mu_r=1.0 - 0.02j,
            n_modes=8,
            workers=1,
            table_precision="double",
        )
        table = solve_bor_dielectric(assembly="tables", **common)
        stream = solve_bor_dielectric(
            assembly="streaming", stream_budget_gb=0.25, **common
        )
        self.assertEqual(stream["assembly"], "streaming")
        self.assertEqual(
            set(stream["stream_sampling_backends"]),
            {"exterior", "interior"},
        )
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
            )

    def test_rectangular_cross_streaming_matches_all_table_modes(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        medium = (3.0 - 0.1j, 1.0 - 0.02j)
        table_cross = BorCrossOperators(
            BorPecSolver(
                _sphere(1.3 * radius, 9), FREQUENCY_HZ, medium=medium
            ),
            BorPecSolver(
                _sphere(0.8 * radius, 7), FREQUENCY_HZ, medium=medium
            ),
        )
        stream_cross = BorCrossOperators(
            BorPecSolver(
                _sphere(1.3 * radius, 9), FREQUENCY_HZ, medium=medium
            ),
            BorPecSolver(
                _sphere(0.8 * radius, 7), FREQUENCY_HZ, medium=medium
            ),
        )
        table_cross.prepare(3)
        stream_cross.enable_streaming(
            3, tile_budget_gb=0.02, workers=1
        )
        stream_cross.prepare(3)
        self.assertIsNone(stream_cross.sp._B_T)
        self.assertIsNone(stream_cross.sq._B_T)
        for mode in range(-3, 4):
            np.testing.assert_allclose(
                stream_cross.assemble_T(mode, 3),
                table_cross.assemble_T(mode, 3),
                rtol=2.0e-12, atol=2.0e-13,
            )
            np.testing.assert_allclose(
                stream_cross.assemble_P(mode, 3),
                table_cross.assemble_P(mode, 3),
                rtol=2.0e-12, atol=2.0e-13,
            )

    def test_coated_streaming_and_table_paths_are_equivalent(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        common = dict(
            points_outer=_sphere(1.2 * radius, 12),
            points_core=_sphere(0.8 * radius, 10),
            freq_hz=FREQUENCY_HZ,
            thetas_deg=[0.0, 37.0, 90.0],
            eps_r=3.0 - 0.1j,
            mu_r=1.0 - 0.02j,
            n_modes=8,
            workers=1,
            table_precision="double",
        )
        table = solve_bor_coated_pec(assembly="tables", **common)
        stream = solve_bor_coated_pec(
            assembly="streaming", stream_budget_gb=0.25, **common
        )
        self.assertEqual(stream["assembly"], "streaming")
        self.assertEqual(
            set(stream["stream_sampling_backends"]),
            {
                "exterior_outer", "coating_outer", "coating_core",
                "cross_outer_core", "cross_core_outer",
            },
        )
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
            )

    def test_partial_junction_streaming_matches_tables(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        common = dict(
            points_interface=_bulged_upper_interface(radius, 5),
            points_covered=_hemisphere(radius, 5, upper=True),
            bare_pieces=[_hemisphere(radius, 5, upper=False)],
            freq_hz=FREQUENCY_HZ,
            thetas_deg=[0.0, 37.0, 90.0],
            eps_r=2.5 - 0.08j,
            mu_r=1.0 - 0.01j,
            n_modes=8,
            workers=1,
            table_precision="double",
        )
        table = solve_bor_partial_coating(assembly="tables", **common)
        stream = solve_bor_partial_coating(
            assembly="streaming", stream_budget_gb=0.25, **common
        )
        self.assertEqual(stream["assembly"], "streaming")
        self.assertGreater(stream["stream_auxiliary_peak_gb"], 0.0)
        self.assertEqual(stream["n_junctions"], 1)
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
            )

    def test_layered_patch_junction_streaming_matches_tables(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        common = dict(
            points_patch=_bulged_upper_interface(radius, 4),
            points_mid_covered=_hemisphere(radius, 4, upper=True),
            points_mid_bare=[_hemisphere(radius, 4, upper=False)],
            points_core=_sphere(0.72 * radius, 8),
            freq_hz=FREQUENCY_HZ,
            thetas_deg=[0.0, 37.0, 90.0],
            eps_inner=2.2 - 0.06j,
            mu_inner=1.0 - 0.01j,
            eps_patch=3.0 - 0.10j,
            mu_patch=1.0 - 0.02j,
            n_modes=8,
            workers=1,
            table_precision="double",
        )
        table = solve_bor_coating_patch(assembly="tables", **common)
        stream = solve_bor_coating_patch(
            assembly="streaming", stream_budget_gb=0.25, **common
        )
        self.assertEqual(stream["assembly"], "streaming")
        self.assertGreater(stream["stream_auxiliary_peak_gb"], 0.0)
        self.assertGreaterEqual(stream["n_junctions"], 1)
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
            )

    def test_banded_junction_streaming_matches_tables(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        outer_radius = 1.12 * radius

        def system():
            surfaces = [
                (_hemisphere(outer_radius, 4, upper=True), False),
                (_hemisphere(outer_radius, 4, upper=False), False),
                (np.asarray([[outer_radius, 0.0], [radius, 0.0]]), False),
                (_hemisphere(radius, 4, upper=True), True),
                (_hemisphere(radius, 4, upper=False), True),
            ]
            regions = [
                {
                    "medium": None,
                    "bounds": [(0, +1), (1, +1)],
                    "exterior": True,
                },
                {
                    "medium": (2.4 - 0.06j, 1.0 - 0.01j),
                    "bounds": [(0, -1), (2, +1), (3, +1)],
                },
                {
                    "medium": (3.1 - 0.10j, 1.0 - 0.02j),
                    "bounds": [(1, -1), (2, -1), (4, +1)],
                },
            ]
            return _MultiRegionBor(
                surfaces, regions, FREQUENCY_HZ, near_factor=2.0, near_order=12
            )

        common = dict(
            freq_hz=FREQUENCY_HZ,
            thetas_deg=[0.0, 37.0, 90.0],
            n_modes=8,
            mode_tol=1.0e-6,
            workers=1,
            progress=None,
            check_abort=None,
            formulation="pmchwt-banded",
            extra={},
            table_precision="double",
        )
        table = _solve_multiregion(system(), assembly="tables", **common)
        stream_system = system()
        stream = _solve_multiregion(
            stream_system,
            assembly="streaming",
            stream_budget_gb=0.25,
            **common,
        )
        self.assertEqual(stream["assembly"], "streaming")
        self.assertGreater(stream["stream_auxiliary_peak_gb"], 0.0)
        self.assertEqual(stream["n_junctions"], 2)
        self.assertTrue(all(
            solver._B_T is None and solver._B_D is None
            for solver in stream_system.solv.values()
        ))
        self.assertTrue(all(
            cross._G is None and cross._B is None
            for cross in stream_system.X.values()
        ))

        def retained_bytes(value):
            if isinstance(value, np.ndarray):
                return value.nbytes
            if isinstance(value, dict):
                return sum(retained_bytes(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return sum(retained_bytes(item) for item in value)
            return 0

        actual_auxiliary_bytes = retained_bytes(stream_system._Q_cache)
        actual_auxiliary_bytes += sum(
            retained_bytes(solver._near_contractions)
            for solver in stream_system.solv.values()
        )
        actual_auxiliary_bytes += sum(
            retained_bytes(cross._cache)
            for cross in stream_system.X.values()
        )
        self.assertGreaterEqual(
            stream["stream_auxiliary_peak_gb"],
            actual_auxiliary_bytes / 1.0e9,
        )
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
            )

    def test_streaming_source_column_chunks_match_full_width_sampling(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        solver = BorPecSolver(_sphere(radius, 10), FREQUENCY_HZ)
        full = bor_streaming.StreamingFarBlocks(
            solver, 2, efie=True, mfie=True, tile_budget_gb=1.0
        )
        chunked = bor_streaming.StreamingFarBlocks(
            solver, 2, efie=True, mfie=True, tile_budget_gb=1.0e-3
        )
        self.assertEqual(full._cols, solver.P)
        self.assertLess(chunked._cols, solver.P)
        np.testing.assert_allclose(chunked.Z, full.Z, rtol=2.0e-14, atol=2.0e-14)
        np.testing.assert_allclose(chunked.K, full.K, rtol=2.0e-14, atol=2.0e-14)

    def test_large_smooth_body_routes_sharp_far_pairs_to_direct_path(self):
        wavelength = C0 / FREQUENCY_HZ
        radius = 20.0 * wavelength
        solver = BorPecSolver(_sphere(radius, 1257), FREQUENCY_HZ)

        self.assertTrue(any(
            abs(e - f) > solver.near_span
            for e, sources in enumerate(solver._near_sources_by_element)
            for f in sources
        ))
        required = n_xi_for_pairs(
            solver.k,
            float(np.max(solver.gen.nodes[:, 0])),
            0,
            solver._far_gap(),
        )
        self.assertIsNotNone(solver._far_gap_pair)
        self.assertNotIn(
            solver._far_gap_pair[1],
            solver._near_sources_by_element[solver._far_gap_pair[0]],
        )
        self.assertLessEqual(required, N_XI_SAFETY_CAP)

    def test_far_gap_uses_closest_remaining_fft_pair(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        solver = BorPecSolver(_sphere(radius, 16), FREQUENCY_HZ)

        brute_gap = math.inf
        brute_pair = None
        for e in range(solver.gen.n_elems):
            for f in range(e + 1, solver.gen.n_elems):
                if f in solver._near_sources_by_element[e]:
                    continue
                gap = _segment_distance(
                    solver.gen.nodes[solver.gen.elem_n0[e]],
                    solver.gen.nodes[solver.gen.elem_n1[e]],
                    solver.gen.nodes[solver.gen.elem_n0[f]],
                    solver.gen.nodes[solver.gen.elem_n1[f]],
                )
                if gap < brute_gap:
                    brute_gap = gap
                    brute_pair = (e, f)

        self.assertIsNotNone(brute_pair)
        self.assertAlmostEqual(solver._far_gap(), brute_gap, places=14)
        self.assertNotIn(
            solver._far_gap_pair[1],
            solver._near_sources_by_element[solver._far_gap_pair[0]],
        )
        required = n_xi_for_pairs(
            solver.k,
            float(np.max(solver.gen.nodes[:, 0])),
            0,
            solver._far_gap(),
        )
        # This one-wavelength-circumference sphere needs 256 samples.  Using
        # the near-routing threshold instead of the actual remaining gap
        # incorrectly pins this ordinary case at the 8192-sample safety cap.
        self.assertEqual(required, 256)

    def test_batched_rhs_and_farfield_match_scalar_paths(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        solver = BorPecSolver(_sphere(radius, 16), FREQUENCY_HZ)
        thetas = np.asarray([0.0, 23.0, 71.0, 90.0])
        mode = 2
        alpha = 0.5
        scalar_rhs = np.stack([
            alpha * solver.rhs_mode(mode, theta, pol)
            + (1.0 - alpha) * ETA0
            * solver.rhs_mfie_mode(mode, theta, pol)
            for theta in thetas for pol in ("VV", "HH")
        ], axis=1)
        batch_rhs = solver.rhs_vv_hh_batch(
            mode,
            thetas,
            efie_scale=alpha,
            mfie_scale=(1.0 - alpha) * ETA0,
            angle_chunk=2,
        )
        np.testing.assert_allclose(
            batch_rhs, scalar_rhs, rtol=5.0e-14, atol=5.0e-14
        )

        rng = np.random.default_rng(17)
        solutions = (
            rng.normal(size=(2 * solver.Nn, 2 * len(thetas)))
            + 1j * rng.normal(size=(2 * solver.Nn, 2 * len(thetas)))
        )
        zs_pt = np.full(solver.P, 35.0 + 8.0j, dtype=np.complex128)
        scalar_far = np.empty((2, len(thetas)), dtype=np.complex128)
        for it, theta in enumerate(thetas):
            vv = solver.farfield_mode(
                mode, solutions[:, 2 * it], theta, zs_pt=zs_pt
            )
            hh = solver.farfield_mode(
                mode, solutions[:, 2 * it + 1], theta, zs_pt=zs_pt
            )
            scalar_far[0, it] = vv[0]
            scalar_far[1, it] = hh[1]
        batch_far = solver.farfield_vv_hh_batch(
            mode, solutions, thetas, zs_pt=zs_pt, angle_chunk=2
        )
        np.testing.assert_allclose(
            batch_far, scalar_far, rtol=5.0e-14, atol=5.0e-14
        )

    def test_survey_entry_point_is_unambiguously_marked(self):
        raw = {
            "samples": [{"rcs_linear": 1.0}],
            "metadata": {"quality_gate": {}},
        }
        with mock.patch.object(
            bor_dispatch, "solve_monostatic_rcs_bor", return_value=raw
        ):
            result = bor_dispatch.solve_monostatic_rcs_bor_survey(
                {}, [1.0], [0.0]
            )
        metadata = result["metadata"]
        self.assertTrue(metadata["survey_mode"])
        self.assertFalse(metadata["mesh_convergence_certified"])
        self.assertEqual(metadata["published_mesh"], "base")

    def test_resource_preview_uses_frequency_specific_meshes(self):
        snapshot = _pec_sphere_snapshot(explicit_elements=-20)
        low = bor_dispatch.estimate_bor_resources(
            snapshot, 1.0, [0.0, 90.0], geometry_units="meters",
            workers=2, mesh_certification=False,
        )
        high = bor_dispatch.estimate_bor_resources(
            snapshot, 4.0, [0.0, 90.0], geometry_units="meters",
            workers=2, mesh_certification=False,
        )
        self.assertGreater(high["mesh_elements"], low["mesh_elements"])
        self.assertGreater(
            high["estimated_peak_gb"], low["estimated_peak_gb"]
        )

    def test_streaming_resource_preview_uses_runtime_budget_planner(self):
        estimate = bor_dispatch.estimate_bor_resources(
            _pec_sphere_snapshot(explicit_elements=83),
            1.0,
            [0.0, 90.0],
            geometry_units="meters",
            n_modes=100,
            workers=64,
            table_precision="double",
            assembly="streaming",
            stream_budget_gb=8.0,
            mesh_certification=False,
        )
        self.assertEqual(estimate["assembly_estimate"], "streaming")
        self.assertEqual(estimate["stream_mode_block_estimate"], 28)
        self.assertEqual(estimate["active_mode_workers"], 28)
        self.assertLessEqual(estimate["held_assembly_gb"], 8.0)
        self.assertAlmostEqual(
            estimate["held_assembly_gb"],
            bor_streaming.estimate_streaming_block_gb(
                estimate["mesh_elements"], 100, 28, "cfie", False, False
            ),
        )

    def test_junction_preview_counts_projection_and_near_operator_storage(self):
        layout = [(12, False), (10, False), (8, True)]
        estimate = bor_dispatch._estimate_junction_auxiliary_gb(layout, 2)
        unknowns = 4 * 13 + 4 * 11 + 2 * 9
        expected_projection = (
            1.10 * 4 * unknowns ** 2 * np.dtype(np.complex128).itemsize
            / 1.0e9
        )
        self.assertAlmostEqual(
            estimate["projection_gb"], expected_projection, places=15
        )
        self.assertGreater(estimate["near_gb"], 0.0)
        self.assertGreaterEqual(estimate["peak_gb"], estimate["retained_gb"])

    def test_partial_preview_accepts_streaming_and_reports_junction_storage(self):
        estimate = bor_dispatch.estimate_bor_resources(
            _partial_coating_snapshot(),
            1.0,
            [0.0, 90.0],
            geometry_units="meters",
            n_modes=8,
            workers=2,
            table_precision="double",
            assembly="streaming",
            stream_budget_gb=0.25,
            mesh_certification=False,
        )
        self.assertEqual(estimate["geometry_kind"], "partial")
        self.assertEqual(estimate["assembly_estimate"], "streaming")
        self.assertGreater(estimate["junction_projection_gb"], 0.0)
        self.assertGreater(estimate["near_junction_operator_gb"], 0.0)

    def test_dielectric_resource_preview_counts_retained_operators(self):
        snapshot = _pec_sphere_snapshot(explicit_elements=12)
        snapshot["segments"][0]["seg_type"] = 3
        snapshot["segments"][0]["properties"] = ["3", "12", "0", "1", "0"]
        snapshot["dielectrics"] = [["1", "3.0", "-0.05", "1", "0"]]
        estimate = bor_dispatch.estimate_bor_resources(
            snapshot,
            1.0,
            [0.0, 90.0],
            geometry_units="meters",
            workers=2,
            mesh_certification=False,
        )
        self.assertEqual(estimate["geometry_kind"], "dielectric")
        self.assertEqual(estimate["surface_count"], 1)
        self.assertEqual(
            estimate["n_unknowns_estimate"],
            4 * (estimate["mesh_elements"] + 1),
        )
        self.assertGreater(estimate["persistent_assembly_gb"], 0.0)
        self.assertGreater(
            estimate["held_assembly_gb"],
            estimate["persistent_assembly_gb"],
        )

    def test_dielectric_streaming_preview_honors_combined_budget(self):
        snapshot = _pec_sphere_snapshot(explicit_elements=24)
        snapshot["segments"][0]["seg_type"] = 3
        snapshot["segments"][0]["properties"] = ["3", "24", "0", "1", "0"]
        snapshot["dielectrics"] = [["1", "3.0", "-0.05", "1", "0"]]
        estimate = bor_dispatch.estimate_bor_resources(
            snapshot,
            1.0,
            [0.0, 90.0],
            geometry_units="meters",
            n_modes=100,
            workers=64,
            table_precision="double",
            assembly="streaming",
            stream_budget_gb=0.25,
            mesh_certification=False,
        )
        block = estimate["stream_mode_block_estimate"]
        self.assertEqual(estimate["assembly_estimate"], "streaming")
        self.assertLessEqual(estimate["held_assembly_gb"], 0.25)
        self.assertAlmostEqual(
            estimate["held_assembly_gb"],
            2.0 * bor_streaming.estimate_streaming_block_gb(
                estimate["mesh_elements"], 100, block,
                "efie", True, False,
            ),
        )

    def test_coated_streaming_preview_counts_self_and_cross_blocks(self):
        outer = _pec_sphere_snapshot(
            radius_m=0.04, explicit_elements=24
        )["segments"][0]
        outer["seg_type"] = 3
        outer["properties"] = ["3", "24", "0", "1", "0"]
        core = _pec_sphere_snapshot(
            radius_m=0.03, explicit_elements=18
        )["segments"][0]
        core["name"] = "core"
        core["seg_type"] = 4
        core["properties"] = ["4", "18", "0", "1", "0"]
        snapshot = {
            "segments": [outer, core],
            "ibcs": [],
            "dielectrics": [["1", "3.0", "-0.05", "1", "0"]],
        }
        estimate = bor_dispatch.estimate_bor_resources(
            snapshot,
            1.0,
            [0.0, 90.0],
            geometry_units="meters",
            n_modes=100,
            workers=64,
            table_precision="double",
            assembly="streaming",
            stream_budget_gb=0.25,
            mesh_certification=False,
        )
        block = estimate["stream_mode_block_estimate"]
        # The snapshot helper contains twelve primitive arcs; its explicit
        # density applies to each primitive.
        outer_elements = 12 * 24
        core_elements = 12 * 18
        requirements = (
            (outer_elements, outer_elements, True, False),
            (outer_elements, outer_elements, True, False),
            (core_elements, core_elements, False, False),
            (outer_elements, core_elements, True, False),
            (core_elements, outer_elements, True, False),
        )
        expected = sum(
            bor_streaming.estimate_rectangular_streaming_block_gb(
                nt, ns, 100, block, rotated, single
            )
            for nt, ns, rotated, single in requirements
        )
        self.assertEqual(estimate["geometry_kind"], "coated")
        self.assertEqual(estimate["assembly_estimate"], "streaming")
        self.assertAlmostEqual(estimate["held_assembly_gb"], expected)
        self.assertLessEqual(estimate["held_assembly_gb"], 0.25)

    def test_total_element_gate_applies_across_bor_surfaces(self):
        outer = _pec_sphere_snapshot(
            radius_m=0.04, explicit_elements=6
        )["segments"][0]
        outer["seg_type"] = 3
        outer["properties"] = ["3", "6", "0", "1", "0"]
        core = _pec_sphere_snapshot(
            radius_m=0.03, explicit_elements=6
        )["segments"][0]
        core["name"] = "core"
        core["seg_type"] = 4
        core["properties"] = ["4", "6", "0", "1", "0"]
        snapshot = {
            "segments": [outer, core],
            "ibcs": [],
            "dielectrics": [["1", "3.0", "-0.05", "1", "0"]],
        }
        with self.assertRaisesRegex(ValueError, "total elements"):
            bor_dispatch.estimate_bor_resources(
                snapshot,
                1.0,
                [90.0],
                geometry_units="meters",
                max_elements=100,
                mesh_certification=False,
            )

    def test_adaptive_tail_waits_for_physical_modal_bandwidth(self):
        def contribution(mode, _solution, _theta, _polarization):
            return 1.0 if abs(mode) == 3 else 0.0

        field, modes_used, stats = _mode_sweep(
            1,
            [90.0],
            ("VV",),
            6,
            1.0e-6,
            lambda _mode: (np.ones((1, 1), dtype=np.complex128), None),
            lambda _mode, _theta, _polarization: np.ones(
                1, dtype=np.complex128
            ),
            contribution,
            min_mode_before_tail=3,
        )
        self.assertEqual(modes_used, 5)
        self.assertEqual(stats["mode_tail_start"], 3)
        self.assertTrue(stats["mode_converged"])
        np.testing.assert_allclose(field, [[2.0 + 0.0j]])

    def test_modal_tail_convergence_is_samplewise(self):
        def contribution(mode, _solution, _theta, polarization):
            if mode == 0 and polarization == "VV":
                return 1.0
            if 1 <= abs(mode) <= 3 and polarization == "HH":
                return 2.0e-7
            return 0.0

        field, modes_used, stats = _mode_sweep(
            1,
            [90.0],
            ("VV", "HH"),
            6,
            1.0e-6,
            lambda _mode: (np.ones((1, 1), dtype=np.complex128), None),
            lambda _mode, _theta, _polarization: np.ones(
                1, dtype=np.complex128
            ),
            contribution,
            min_mode_before_tail=0,
        )
        self.assertEqual(modes_used, 5)
        self.assertTrue(stats["mode_converged"])
        np.testing.assert_allclose(field, [[1.0], [1.2e-6]])

    def test_signed_mode_symmetry_skips_redundant_negative_solves(self):
        calls = []

        def contribution(mode, _solution, _theta, _polarization):
            calls.append(mode)
            return 1.0 if abs(mode) <= 2 else 0.0

        common = dict(
            n_dofs=1,
            thetas=[90.0],
            pols=("VV",),
            m_max=5,
            mode_tol=1.0e-6,
            assemble=lambda _mode: (
                np.ones((1, 1), dtype=np.complex128), None
            ),
            rhs=lambda _mode, _theta, _polarization: np.ones(
                1, dtype=np.complex128
            ),
            farfield=contribution,
            min_mode_before_tail=0,
        )
        full, full_modes, _ = _mode_sweep(**common)
        full_calls = list(calls)
        calls.clear()
        reduced, reduced_modes, stats = _mode_sweep(
            signed_mode_symmetry=True, **common
        )

        np.testing.assert_allclose(reduced, full)
        self.assertEqual(reduced_modes, full_modes)
        self.assertTrue(stats["signed_mode_symmetry_used"])
        self.assertTrue(any(mode < 0 for mode in full_calls))
        self.assertFalse(any(mode < 0 for mode in calls))

    def test_signed_modal_pairs_obey_axisymmetric_cfie_symmetry(self):
        """Verify +m/-m operators independently, before their fields sum.

        For the supported isotropic axisymmetric formulations, changing the
        sign of m is a diagonal similarity transform that flips the phi basis.
        Monostatic co-polarized +m and -m far-field contributions are equal.
        This catches a signed FFT bin, Gs parity, excitation, or projection
        regression that a final power-only sphere comparison could conceal.
        """

        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        solver = BorPecSolver(_sphere(radius, 10), FREQUENCY_HZ)
        mode_cap = 2
        alpha = 0.5
        solver.prepare_operators(
            mode_cap, efie=True, mfie=True, workers=1
        )
        transform = np.concatenate((
            np.ones(solver.Nn), -np.ones(solver.Nn)
        ))

        for mode in (1, 2):
            systems = {}
            contributions = {"VV": {}, "HH": {}}
            for signed_mode in (mode, -mode):
                matrix = (
                    alpha * solver.assemble_mode(signed_mode, mode_cap)
                    + (1.0 - alpha) * ETA0
                    * solver.assemble_mfie_mode(signed_mode, mode_cap)
                )
                mask = solver.basis_mask(signed_mode)
                systems[signed_mode] = matrix[np.ix_(mask, mask)]
                for polarization, component in (("VV", 0), ("HH", 1)):
                    excitation = (
                        alpha * solver.rhs_mode(
                            signed_mode, 47.0, polarization
                        )
                        + (1.0 - alpha) * ETA0
                        * solver.rhs_mfie_mode(
                            signed_mode, 47.0, polarization
                        )
                    )
                    solution = np.zeros(
                        2 * solver.Nn, dtype=np.complex128
                    )
                    solution[mask] = np.linalg.solve(
                        systems[signed_mode], excitation[mask]
                    )
                    contributions[polarization][signed_mode] = (
                        solver.farfield_mode(
                            signed_mode, solution, 47.0
                        )[component]
                    )

            active_transform = transform[solver.basis_mask(mode)]
            expected_negative = (
                active_transform[:, None]
                * systems[mode]
                * active_transform[None, :]
            )
            np.testing.assert_allclose(
                systems[-mode], expected_negative,
                rtol=2.0e-11, atol=2.0e-11,
            )
            for polarization in ("VV", "HH"):
                np.testing.assert_allclose(
                    contributions[polarization][-mode],
                    contributions[polarization][mode],
                    rtol=2.0e-11, atol=2.0e-12,
                    err_msg=f"m={mode}, {polarization}",
                )

    def test_hpc_bor_discovers_only_configured_bor_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bor_dir = root / "BOR"
            other_dir = root / "FRD"
            bor_dir.mkdir()
            other_dir.mkdir()
            (bor_dir / "body.geo").write_text("Properties: 2 1 0 0 0\n")
            (other_dir / "coupon.geo").write_text("Properties: 2 1 0 0 0\n")
            with mock.patch.object(
                run_hpc_bor_monostatic,
                "GEOMETRY_DIRS",
                [str(bor_dir)],
            ):
                found = run_hpc_bor_monostatic._discover_geometries()
            self.assertEqual([path.name for path in found], ["body.geo"])

    def test_vv_hh_outputs_are_paired_without_manifest_change(self):
        units = [
            {
                "geometry": "/tmp/body.geo",
                "geometry_stem": "body",
                "geometry_input_sha256": "abc",
                "polarization": polarization,
                "frequency_ghz": frequency,
            }
            for polarization in ("VV", "HH")
            for frequency in (1.0, 2.0)
        ]
        pairs = run_local_bor._paired_solve_units(units)
        self.assertEqual(len(units), 4)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(len(pair["channel_units"]) == 2 for pair in pairs))

    def test_bor_frequency_outputs_are_immediately_user_visible(self):
        unit = {
            "geometry_stem": "cylinder",
            "polarization": "VV",
            "frequency_ghz": 3.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            expected = (
                run_dir / "results" / "by_frequency" /
                "VV_3.000GHz_cylinder.grim"
            )
            self.assertEqual(
                run_hpc_bor_monostatic._unit_output_path(run_dir, unit),
                expected,
            )
            self.assertEqual(
                run_local_bor._unit_output_path(
                    run_dir / "results" / "by_frequency", unit
                ),
                expected,
            )

    def test_hpc_bor_publication_is_elected_once_across_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)
            manifest = {
                "schema": "ghost.hpc.bor-run.v1",
                "unit_output_dir": "results/by_frequency",
                "units": [
                    {
                        "geometry_stem": "body",
                        "frequency_ghz": 1.0,
                        "polarization": polarization,
                    }
                    for polarization in ("VV", "HH")
                ],
                "n_units": 2,
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            publication = results_dir / "body.grim"
            publish_started = threading.Event()
            allow_publish = threading.Event()
            published = threading.Event()
            calls = []
            calls_lock = threading.Lock()

            def fake_status(_run_dir):
                done = publication.is_file() and published.is_set()
                return {
                    "unit_complete": True,
                    "complete": done,
                    "derived_expected": (publication,),
                    "derived_done": (publication,) if done else (),
                    "publication_error": "",
                }

            def fake_publish(_run_dir, require_complete=True):
                self.assertTrue(require_complete)
                with calls_lock:
                    calls.append(threading.get_ident())
                publish_started.set()
                self.assertTrue(allow_publish.wait(2.0))
                publication.write_bytes(b"published")
                published.set()
                return 1, 0

            outcomes = []
            errors = []

            def run_publisher():
                try:
                    outcomes.append(
                        run_hpc_bor_monostatic._publish_monostatic_coordinated(
                            run_dir, 60.0, poll_seconds=0.005
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            environment = {
                "SLURM_JOB_ID": "",
                "SLURM_ARRAY_JOB_ID": "",
                "SLURM_ARRAY_TASK_ID": "",
                "SLURM_CLUSTER_NAME": "",
                "SLURM_RESTART_COUNT": "0",
            }
            with (
                mock.patch.dict("os.environ", environment, clear=False),
                mock.patch.object(hpc_common, "run_status", side_effect=fake_status),
                mock.patch.object(
                    run_hpc_bor_monostatic,
                    "publish_monostatic",
                    side_effect=fake_publish,
                ),
            ):
                first = threading.Thread(target=run_publisher)
                second = threading.Thread(target=run_publisher)
                first.start()
                self.assertTrue(publish_started.wait(2.0))
                second.start()
                time.sleep(0.05)
                self.assertEqual(len(calls), 1)
                allow_publish.set()
                first.join(timeout=2.0)
                second.join(timeout=2.0)

            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(calls), 1)
            self.assertCountEqual(outcomes, [(1, 0), (0, 1)])

    def test_scaled_stable_lu_is_not_rejected_by_rhs_residual_alone(self):
        rng = np.random.default_rng(1)
        size = 8
        left, _ = np.linalg.qr(rng.normal(size=(size, size)))
        right, _ = np.linalg.qr(rng.normal(size=(size, size)))
        singular_values = np.geomspace(1.0, 1.0e-9, size)
        matrix = ((left * singular_values) @ right.T).astype(np.complex128)
        excitations = rng.normal(size=(size, 4)).astype(np.complex128)

        column = iter(range(excitations.shape[1]))

        def rhs(_mode, _theta, _polarization):
            return excitations[:, next(column)]

        field, _modes, stats = _mode_sweep(
            size,
            [0.0, 90.0],
            ("VV", "HH"),
            0,
            1.0,
            lambda _mode: (matrix, None),
            rhs,
            lambda _mode, solution, _theta, _polarization: solution[0],
            monitor_cond=True,
        )
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertLessEqual(
            stats["linear_backward_error"],
            BOR_LINEAR_BACKWARD_ERROR_MAX,
        )
        # This matrix is deliberately scaled so the old diagnostic can cross
        # its cutoff while the standard backward error remains near epsilon.
        direct = np.linalg.solve(matrix, excitations)
        relative = np.max(
            np.linalg.norm(matrix @ direct - excitations, axis=0)
            / np.linalg.norm(excitations, axis=0)
        )
        self.assertGreater(relative, BOR_LINEAR_RESIDUAL_MAX)

    def test_bor_slurm_pins_the_submitting_backend_first(self):
        text = run_hpc_bor_monostatic._build_slurm(
            Path("/tmp/run/driver_configured.py"),
            Path("/tmp/run"),
            0,
        )
        backend = str(Path(run_hpc_bor_monostatic.__file__).resolve().parent)
        self.assertIn(
            f"export PYTHONPATH={shlex.quote(backend)}:${{PYTHONPATH:-}}",
            text,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
