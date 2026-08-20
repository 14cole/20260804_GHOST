#!/usr/bin/env python3
"""BoR physics, workflow, and optimized-assembly release regressions."""

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

import bor_dispatch  # noqa: E402
import feature_sum  # noqa: E402
import grim_io  # noqa: E402
import occluder  # noqa: E402
import run_hpc_bor_monostatic  # noqa: E402
import run_local_bor  # noqa: E402
import surface_mesh  # noqa: E402
import validate_feature_reconstruction  # noqa: E402
from bor_kernels import C0  # noqa: E402
from bor_solver import (  # noqa: E402
    BOR_LINEAR_BACKWARD_ERROR_MAX,
    BOR_LINEAR_RESIDUAL_MAX,
    _mode_sweep,
    solve_bor,
    solve_bor_coated_pec,
    solve_bor_dielectric,
)
from mie_sphere import (  # noqa: E402
    sigma_coated_pec_sphere,
    sigma_dielectric_sphere,
    sigma_impedance_sphere,
    sigma_pec_sphere,
)
from line_expand import SeamCoefficients, expand_perimeter  # noqa: E402


FREQUENCY_HZ = 1.0e9


def _sphere(radius_m, elements):
    theta = np.linspace(0.0, math.pi, int(elements) + 1)
    return np.column_stack((
        float(radius_m) * np.sin(theta),
        float(radius_m) * np.cos(theta),
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
        self.assertTrue(np.all(np.isfinite(vv)))
        self.assertTrue(np.all(np.isfinite(hh)))
        self.assertLess(_error_db(vv[0], reference), tolerance_db)
        self.assertLess(_error_db(hh[0], reference), tolerance_db)
        self.assertLess(_error_db(vv[0], hh[0]), 0.08)

    def test_pec_sphere_matches_mie(self):
        radius = 3.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        result = solve_bor(
            _sphere(radius, 45), FREQUENCY_HZ, [0.0],
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
            _sphere(radius, 45), FREQUENCY_HZ, [0.0],
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
            _sphere(radius, 45), FREQUENCY_HZ, [0.0], eps_r, mu_r,
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
            _sphere(outer_radius, 50),
            _sphere(core_radius, 40),
            FREQUENCY_HZ,
            [0.0],
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
                    "frequency": "GHz", "rcs_log_unit": "dBsm",
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

            # A retained convention is authoritative and cannot contradict
            # the explicit COMPACT_FEATURES declaration.
            payload["phase_reference"] = np.asarray("wrong origin")
            with path.open("wb") as stream:
                np.savez(stream, **payload)
            with self.assertRaisesRegex(ValueError, "phase_reference"):
                feature_sum.prepare_point_pattern(
                    str(path), declared_coherent_delta=True
                )

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
            step_one = Path(directory) / "step_one.grim"
            sequential = Path(directory) / "sequential.grim"
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(direct), points=[point_a, point_b]
            )
            feature_sum.add_features_to_monostatic_grim(
                str(base), str(step_one), points=[point_a]
            )
            feature_sum.add_features_to_monostatic_grim(
                str(step_one), str(sequential), points=[point_b]
            )
            with np.load(direct, allow_pickle=False) as one_pass, np.load(
                sequential, allow_pickle=False
            ) as two_pass:
                direct_field = (
                    one_pass["rcs_amp_real"] + 1j * one_pass["rcs_amp_imag"]
                )
                sequential_field = (
                    two_pass["rcs_amp_real"] + 1j * two_pass["rcs_amp_imag"]
                )
            np.testing.assert_allclose(
                direct_field, sequential_field, rtol=2.0e-14, atol=2.0e-14
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
                "VV",
                geometry_units="meters",
                workers=1,
            )
        self.assertEqual([entry[0] for entry in point_counts], [1.0e9, 4.0e9])
        self.assertGreater(point_counts[1][1], point_counts[0][1])
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
        for key in ("sigma_vv", "sigma_hh", "amp_vv", "amp_hh"):
            np.testing.assert_allclose(
                np.asarray(stream[key]), np.asarray(table[key]),
                rtol=2.0e-10, atol=2.0e-12,
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
                {}, [1.0], [0.0], "VV"
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

    def test_azel_writer_accepts_per_channel_metadata(self):
        grid = {
            "azimuths_deg": np.asarray([0.0]),
            "elevations_deg": np.asarray([0.0]),
            "frequencies_ghz": np.asarray([1.0]),
            "axis_az_deg": 0.0,
            "axis_el_deg": 0.0,
            "amp": {
                channel: np.ones((1, 1, 1), dtype=np.complex128)
                for channel in ("VV", "HH", "VH")
            },
        }
        metadata = {
            channel: {"output_attestation": {"channel": channel}}
            for channel in ("VV", "HH", "VH")
        }
        with tempfile.TemporaryDirectory() as directory:
            written = grim_io.save_bor_az_el_grim(
                grid,
                str(Path(directory) / "azel.grim"),
                channel_metadata=metadata,
            )
            self.assertEqual(len(written), 3)
            with np.load(written[0], allow_pickle=False) as payload:
                stored = json.loads(str(payload["solver_metadata_json"]))
            self.assertIn("output_attestation", stored["metadata"])

    def test_bor_slurm_pins_the_submitting_backend_first(self):
        text = run_hpc_bor_monostatic._build_slurm(
            Path("/tmp/run/driver_configured.py"),
            Path("/tmp/run"),
            0,
        )
        backend = str(Path(run_hpc_bor_monostatic.__file__).resolve().parent)
        self.assertIn(
            f"export PYTHONPATH={backend}:${{PYTHONPATH:-}}",
            text,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
