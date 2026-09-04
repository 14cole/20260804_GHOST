"""Regressions for the Assembly audit, with independent numerical oracles."""
import json
import math
import tempfile
import tracemalloc
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/"GRIM_Revised_2"), str(REPO/"tools/GHOST/Backend")]
import feature_sum as fs
import feature_workflow as fw
import line_expand as le
from test_point_scatter_physics import _pattern_dict, _local_spherical_basis, _write_point_grim


class PointPoleTests(unittest.TestCase):
    def test_anisotropic_pole_limit_is_continuous_for_each_meridian(self):
        tensor = np.asarray([[1, .3+.2j, 0], [.3+.2j, 4, 0], [0, 0, 0]])
        pattern = _pattern_dict(tensor, [1.0])
        for az in (0, 45, 90, 180, 270, 355):
            el = 89.99
            direction = fs._direction(az, el)
            actual = fs.point_scatterer_amplitude(pattern, np.zeros(3), [0, 0, 1], direction, 1.0, roll_ref=[1, 0, 0])
            v, h = _local_spherical_basis(az, el)
            expected = {"F_vv": v@tensor@v, "F_hh": h@tensor@h, "F_vh": v@tensor@h}
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=3e-5, atol=2e-5)

    def test_declared_file_and_dictionary_keep_samples_despite_annotations(self):
        pattern = _pattern_dict(np.eye(3), [1.0])
        pattern["time_convention"] = "exp(-jwt)"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"delta.grim"
            _write_point_grim(path, np.eye(3), [1.0])
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["time_convention"] = "exp(-jwt)"
            with path.open("wb") as stream:
                np.savez(stream, **payload)
            for source in (pattern, str(path)):
                with self.subTest(source=type(source).__name__):
                    actual = fs.prepare_point_pattern(source, declared_coherent_delta=True)
                    expected = fs.prepare_point_pattern(_pattern_dict(np.eye(3), [1.0]))
                    np.testing.assert_array_equal(actual.amplitude, expected.amplitude)
            self.assertEqual(pattern["time_convention"], "exp(-jwt)")

    def test_origin_cache_keeps_translation_and_visibility_per_instance(self):
        pattern = fs.prepare_point_pattern(_pattern_dict(np.diag([1, 3, 0]), [1.0]))
        looks = np.asarray([fs._direction(a, e) for a, e in ((0, 25), (45, 80), (180, 35))])
        cache = {}
        for position in ([0, 0, 0], [.12, -.2, .03]):
            expected = fs.point_scatterer_amplitude(pattern, position, [0, 0, 1], looks, 1., roll_ref=[1, 0, 0])
            actual = fs.point_scatterer_amplitude(pattern, position, [0, 0, 1], looks, 1., roll_ref=[1, 0, 0], _oriented_pattern_cache=cache)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-13)
        self.assertEqual(len(cache), 1)


class BoundedLineTests(unittest.TestCase):
    def test_frames_reuse_across_frequencies_and_separate_subdivision(self):
        placement = self.placement()
        cache = {}
        original = le.prepare_perimeter_frame
        with patch.object(le, "prepare_perimeter_frame", wraps=original) as prepare:
            for frequency in (1., 2.):
                coefficient = le.SeamCoefficients(frequency, np.array([0., 90., 180.]), np.ones(3, complex), np.ones(3, complex))
                expected = le.expand_perimeter(placement["perimeter"], coefficient, None, [[0, 0, 1]], segment_normals=placement["segment_normals"], max_piece_length_m=.002)
                actual = le.expand_perimeter(placement["perimeter"], coefficient, None, [[0, 0, 1]], segment_normals=placement["segment_normals"], max_piece_length_m=.002, _frame_cache=cache)
                for key in actual:
                    np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(prepare.call_count, 3)
            le.expand_perimeter(placement["perimeter"], coefficient, None, [[0, 0, 1]], segment_normals=placement["segment_normals"], max_piece_length_m=.004, _frame_cache=cache)
            self.assertEqual(prepare.call_count, 4)

    @staticmethod
    def placement():
        return {"perimeter": np.asarray([[[0, 0, 0], [2, 0, 0]]], float),
                "segment_normals": np.asarray([[[0, 0, 1], [0, 0, 1]]], float),
                "max_piece_length_m": .002}

    def test_tiled_metrics_match_dense_geometry_oracle(self):
        rng = np.random.default_rng(93)
        directions = rng.normal(size=(73, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        placement = self.placement()
        frame = le.prepare_perimeter_frame(placement["perimeter"], .002, segment_normals=placement["segment_normals"])
        normals, tangents = frame[-2:]
        pn, pb = directions@normals.T, directions@np.cross(tangents, normals).T
        lit = pn > 0
        phi = np.degrees(np.arctan2(pn[lit], pb[lit]))
        result = fw._line_applicability_metrics(placement, directions, requested_frequencies_ghz=[1., 2.])
        row = result["required_cut_angle_ranges_deg"][0]
        self.assertEqual(row["lit_query_count"], np.count_nonzero(lit))
        self.assertAlmostEqual(row["minimum_deg"], phi.min(), places=12)
        self.assertAlmostEqual(row["maximum_deg"], phi.max(), places=12)
        self.assertEqual(result["illuminated_requested_look_count"], np.count_nonzero(np.any(lit, axis=1)))

    def test_peak_scratch_does_not_scale_as_looks_times_pieces(self):
        directions = np.tile([[0., 0., 1.]], (3000, 1))
        tracemalloc.start()
        fw._line_applicability_metrics(self.placement(), directions, requested_frequencies_ghz=[10.])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 6*1024**2)  # old path was >100 MB
        with self.assertRaises(InterruptedError):
            fw._line_applicability_metrics(self.placement(), directions, requested_frequencies_ghz=[10.], cancel_check=lambda: True)


class CurrentAmplitudeTests(unittest.TestCase):
    def test_native_old_or_missing_amplitude_identity_is_advisory(self):
        for version in (None, 1, "1", 3):
            with self.subTest(version=version):
                metadata = {"solver_metadata_json": json.dumps({"amplitude_version": version})}
                fs.require_current_2d_amplitude(metadata, "test")
                self.assertIn("amplitude_version", str(metadata["metadata_advisories_json"]))
        fs.require_current_2d_amplitude({"amplitude_version": "2"}, "test")

    def test_gui_subtraction_preserves_agreed_identity_and_accepts_mixed_versions(self):
        from grim_dataset import RcsGrid
        def grid(version):
            return RcsGrid([0.], [0.], [1.], ["VV", "HH"], rcs=np.ones((1, 1, 1, 2), complex),
                units={"rcs_linear_quantity": "sigma_2d", "rcs_log_unit": "dBke"},
                extra={"solver_metadata_json": json.dumps({"amplitude_version": version})})
        mixed = grid(2).coherent_subtract(grid(1))
        np.testing.assert_array_equal(mixed.rcs_power, 0.0)
        self.assertNotIn("amplitude_version", mixed.extra)
        self.assertIn("amplitude_version", mixed.extra["coherent_metadata_assumption_json"])
        result = grid(2).coherent_subtract(grid(2))
        self.assertEqual(str(result.extra["amplitude_version"]), "2")
        self.assertNotIn("solver_metadata_json", result.extra)


class WorkflowUpdatesTests(unittest.TestCase):
    def test_embedded_bor_shadow_and_frequency_subset_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, output, point = root/"body.grim", root/"total.grim", root/"point.grim"
            profile = np.asarray([[0, 1], [.1, 1], [.1, -1], [0, -1]], float)
            aspects = np.linspace(0, 180, 181)
            bodies = {frequency: {"theta_deg": aspects, "amp_vv": np.full(181, .1+.2j), "amp_hh": np.full(181, .1+.2j)} for frequency in (1., 2.)}
            fs.save_monostatic_grim(bodies, profile, str(base), azimuths_deg=[0., 90., 180., 270.], elevations_deg=[0., 45.])
            _write_point_grim(point, np.diag([.001, .002, 0]), [1., 2.])
            csv = root/"point.csv"
            csv.write_text("placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z\np,f,0,0,0.1,0,0,1,1,0,0\n")
            request = fw.FeatureAssemblyRequest(base, output, point_locations_csv=csv, point_datasets={"f": point}, coordinate_units="meters", shadow=True, study_frequencies_ghz=(2.,), require_feature_manifests=False)
            plan = fw.prepare_feature_assembly(request)
            self.assertIsNotNone(plan.occluder)
            fw.execute_feature_assembly(plan)
            result = fs._load_grim(str(output))
            self.assertEqual(result["_amp"].shape, (4, 2, 1, 3))
            self.assertTrue(np.all(np.isfinite(result["_amp"])))
            self.assertEqual(result["body_model_amp_vv_real"].shape, (181, 1))
            declared = json.loads(str(result["requested_radar_grid_json"]))
            self.assertEqual(declared["frequencies_ghz"], [2.])
            self.assertIn("generated_shadow_surface", str(result["feature_provenance_json"]))

    def test_host_stack_and_principal_curvature_are_checked(self):
        definition = {"host": {"material": "PEC", "stack_id": "coupon-A"}, "applicability": {"minimum_principal_radius_m": .3}}
        good = dict(material="pec", stack_id="coupon-A", minimum_radius_m=.5)
        self.assertTrue(fw.validate_installed_host(definition, **good)["principal_curvature_checked"])
        for override in ({"material": "dielectric"}, {"stack_id": "coupon-B"}, {"minimum_radius_m": .2}, {"minimum_radius_m": None}, {"material": ""}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                fw.validate_installed_host(definition, **{**good, **override})

    def test_bor_shadow_mesh_is_closed_outward_and_meets_sag_bound(self):
        profile = np.asarray([[0, 1], [1, 1], [1, -1], [0, -1]], float)
        triangles, report = fw.bor_shadow_triangles(profile, max_sag_m=.0001, normal_tolerance_deg=2.)
        self.assertLessEqual(report["maximum_radial_sag_m"], .0001)
        surface = fw.TriangleSurface(triangles)
        self.assertEqual(surface.topology_report.boundary_edge_count, 0)
        self.assertEqual(surface.topology_report.outward_closed_component_count, 1)

    def test_body_only_exact_subset_preserves_nonzero_complex_field(self):
        with tempfile.TemporaryDirectory() as directory:
            base, out = Path(directory)/"body.grim", Path(directory)/"study.grim"
            fs.export_radar_grim(str(base), bor_result=None, placements=[], frequencies_ghz=[1., 2.], azimuths_deg=[0., 30., 60.], elevations_deg=[-10., 0., 10.])
            with np.load(base, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            amp = np.arange(54).reshape(3, 3, 2, 3)*(.001+.002j)
            payload.update(rcs_amp_real=amp.real, rcs_amp_imag=amp.imag, rcs_power=(4*math.pi*np.abs(amp)**2).astype(np.float32), rcs_phase=np.angle(amp).astype(np.float32))
            with base.open("wb") as stream:
                np.savez(stream, **payload)
            request = fw.FeatureAssemblyRequest(base, out, study_frequencies_ghz=(2.,), study_azimuths_deg=(0., 60.), study_elevations_deg=(0.,), require_feature_manifests=True, allow_legacy_base_metadata=False)
            plan = fw.prepare_feature_assembly(request)
            fw.execute_feature_assembly(plan)
            actual = fs._load_grim(str(out))
            np.testing.assert_array_equal(actual["_amp"], amp[np.ix_([0, 2], [1], [1], [0, 1, 2])])
            delta = fs._load_grim(fs.feature_only_output_path(str(out)))
            np.testing.assert_array_equal(delta["_amp"], 0.)
            with self.assertRaisesRegex(ValueError, "stored body-grid"):
                fw.prepare_feature_assembly(fw.FeatureAssemblyRequest(base, Path(directory)/"bad.grim", study_frequencies_ghz=(1.5,)))

    def test_capacity_rejection_precedes_full_body_load(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/"body.grim"
            fs.export_radar_grim(str(base), bor_result=None, placements=[], frequencies_ghz=[1.], azimuths_deg=[0.], elevations_deg=[0.])
            with patch.object(fw, "preflight_feature_assembly_capacity", side_effect=MemoryError("capacity")), patch.object(fw, "_load_grim") as loader:
                with self.assertRaisesRegex(MemoryError, "capacity"):
                    fw.prepare_feature_assembly(fw.FeatureAssemblyRequest(base, Path(directory)/"out.grim"))
                loader.assert_not_called()

    def test_straight_constant_line_uses_exact_analytic_integral(self):
        placement = BoundedLineTests.placement()
        angles = np.linspace(0, 180, 181)
        coeff = le.SeamCoefficients(10., angles, np.ones(181, complex), np.ones(181, complex))
        directions = np.asarray([fs._direction(a, 35) for a in range(0, 180, 10)])
        kwargs = dict(segment_normals=placement["segment_normals"], psi_tm_deg=0., psi_te_deg=0.)
        coarse = le.expand_perimeter(placement["perimeter"], coeff, None, directions, **kwargs)
        fine = le.expand_perimeter(placement["perimeter"], coeff, None, directions, max_piece_length_m=.0015, **kwargs)
        for key in coarse:
            np.testing.assert_allclose(coarse[key], fine[key], rtol=2e-11, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
