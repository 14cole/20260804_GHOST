"""User workflows accept optional annotations without converting the fields."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/"GRIM_Revised_2"), str(REPO/"tools/GHOST/Backend")]

from grim_dataset import RcsGrid
import feature_sum as fs
import feature_workflow as fw
from grim_io import _save_grim_npz
from test_point_scatter_physics import _write_point_grim
from test_ptm import _independent_fixture


ANNOTATIONS = ("phase_reference", "amplitude_convention", "complex_field_domain",
               "time_convention", "polarization_basis", "solver_metadata_json",
               "amplitude_version", "feature_library_manifest_json",
               "production_mesh_certification_json")


def rewrite(path, *, remove=(), additions=None):
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files if key not in remove}
    payload.update(additions or {})
    with open(path, "wb") as stream:
        np.savez(stream, **payload)


class MetadataInteropTests(unittest.TestCase):
    def test_2d_subtraction_and_line_loading_accept_external_annotations(self):
        cem_root = Path(__file__).resolve().parents[1]/"CEM_Tools"
        if str(cem_root) not in sys.path:
            sys.path.insert(0, str(cem_root))
        from cem_tools.grim_native import subtract_payloads
        clean = np.broadcast_to(np.array([1+2j, 2-1j]), (3, 1, 1, 2)).copy()
        featured = clean + [.3+.4j, -.2+.1j]
        scale = 1/(4*(2*np.pi*2e9/fs.C0))
        def payload(amplitude):
            return dict(azimuths=np.array([0., 90., 180.]), elevations=np.array([0.]),
                frequencies=np.array([2.]), polarizations=np.array(["VV", "HH"]),
                rcs_amp_real=amplitude.real, rcs_amp_imag=amplitude.imag,
                rcs_power=(scale*np.abs(amplitude)**2).astype(np.float32),
                rcs_phase=np.angle(amplitude).astype(np.float32),
                rcs_domain="power_phase", power_domain="linear_rcs",
                units=json.dumps({"azimuth":"deg", "elevation":"deg", "frequency":"GHz",
                                  "rcs_linear_quantity":"sigma_2d", "rcs_log_unit":"dBke"}))
        c, f = payload(clean), payload(featured)
        f.update(phase_reference="external local origin", amplitude_version=1, time_convention="unknown")
        before_keys = set(f)
        delta = subtract_payloads(f, c, featured_label="featured", clean_label="clean")
        np.testing.assert_array_equal(delta["rcs_amp_real"]+1j*delta["rcs_amp_imag"], featured-clean)
        self.assertEqual(set(f), before_keys)
        self.assertEqual(f["phase_reference"], "external local origin")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("clean", c), ("featured", f)):
                with (root/(name+".grim")).open("wb") as stream:
                    np.savez(stream, **value)
            path = root/"delta.grim"
            fs.make_delta_grim(str(root/"clean.grim"), str(root/"featured.grim"), str(path))
            loaded = fs._load_grim(str(path))
            np.testing.assert_array_equal(loaded["_amp"], featured-clean)
            self.assertNotIn("amplitude_version", loaded)
            rewrite(path, remove=ANNOTATIONS)
            coefficient = fs.load_seam_from_grim(str(path), 2., declared_coherent_delta=True)
            np.testing.assert_array_equal(coefficient.dA_te, (featured-clean)[:, 0, 0, 0])
            np.testing.assert_array_equal(coefficient.dA_tm, (featured-clean)[:, 0, 0, 1])

    def test_external_formats_subtract_without_ghost_annotations(self):
        header = "Frequency,Theta,Phi,RCSPhiScat_PhiInc,PhasePhi_Phi,RCSThetaScat_ThetaInc,PhaseTheta_Theta,RCSPhiScat_ThetaInc,PhasePhi_Theta,RCSThetaScat_PhiInc,PhaseTheta_Phi"
        units = "Hz,deg,deg,dBsm,deg,dBsm,deg,dBsm,deg,dBsm,deg"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentri = root/"sentri.csv"
            sentri.write_text(header+"\n"+units+"\n1000000000,80,20,0,90,-20,-45,-6.020599913,180,6.020599913,0\n")
            sentri_grid = RcsGrid.read_SENTRi(str(sentri))
            ptm = root/"external.ptm"
            _independent_fixture(str(ptm))
            ptm_grid = RcsGrid.load_ptm(str(ptm))
            pio_source = RcsGrid([0., 20.], [0.], [1., 2.], ["VV"],
                rcs=np.array([1+2j, 3-4j, -5+6j, 7+8j]).reshape(2, 1, 2, 1))
            pio = root/"external.pio"
            pio_source.save_pio(str(pio), precision="double")
            pio_grid = RcsGrid.load_pio(str(pio))
            for name, grid in (("SENTRi", sentri_grid), ("PIO", pio_grid), ("PTM", ptm_grid)):
                with self.subTest(format=name):
                    for key in ANNOTATIONS:
                        grid.extra.pop(key, None)
                        grid.units.pop(key, None)
                    before = grid.rcs.copy()
                    result = grid.coherent_subtract(grid)
                    np.testing.assert_array_equal(result.rcs_power, 0.)
                    np.testing.assert_array_equal(grid.rcs, before)
                    path = root/(name+"_delta.grim")
                    result.save(str(path))
                    restored = RcsGrid.load(str(path))
                    np.testing.assert_array_equal(restored.rcs_power, 0.)
            # Native SENTRi theta needs an actual coordinate conversion for
            # placement. Once converted it is usable as a body with no stamp.
            converted = sentri_grid.convert_sentri_elevation_to_grim()
            base, out = root/"sentri_body.grim", root/"sentri_study.grim"
            converted.save(str(base))
            rewrite(base, remove=ANNOTATIONS+("assembly_angular_coordinate_contract",))
            plan = fw.prepare_feature_assembly(fw.FeatureAssemblyRequest(base, out))
            fw.execute_feature_assembly(plan)
            actual, expected = fs._load_grim(str(out)), fs._load_grim(str(base))
            for channel in expected["polarizations"]:
                i = list(actual["polarizations"]).index(channel)
                j = list(expected["polarizations"]).index(channel)
                np.testing.assert_array_equal(actual["_amp"][..., i], expected["_amp"][..., j])

    def test_bor_body_and_point_build_without_certificates_or_convention_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, point, output = root/"body.grim", root/"point.grim", root/"total.grim"
            aspects = np.linspace(0, 180, 181)
            profile = np.array([[0, 1], [.1, 1], [.1, -1], [0, -1]])
            fs.save_monostatic_grim({1.: {"theta_deg": aspects, "amp_vv": np.full(181, .1+.2j), "amp_hh": np.full(181, .3-.1j)}}, profile, str(base), azimuths_deg=[0., 90., 180., 270.], elevations_deg=[0., 45.])
            _write_point_grim(point, np.diag([.001, .002, 0]), [1.])
            rewrite(base, remove=ANNOTATIONS+("requested_radar_grid_json", "body_model_metadata_json"))
            rewrite(point, remove=ANNOTATIONS)
            points = root/"points.csv"
            points.write_text("placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z\np,f,0,0,0.1,0,0,1,1,0,0\n")
            request = fw.FeatureAssemblyRequest(base, output, point_locations_csv=points, point_datasets={"f": point}, coordinate_units="meters")
            baseline = fw.prepare_feature_assembly(request)
            fw.execute_feature_assembly(baseline)
            expected = fs._load_grim(str(output))["_amp"].copy()
            # Unusable certificates/sidecars and contradictory conventions
            # remain inspectable annotations, and do not alter the result.
            point.with_suffix(".feature.json").write_text('{"schema":"foreign-format"}')
            rewrite(base, additions={"phase_reference": ["stale", "annotation"], "solver_metadata_json": "not JSON", "production_mesh_certification_json": "not JSON"})
            rewrite(point, additions={"time_convention": "exp(-jwt)", "phase_reference": "foreign origin"})
            plan = fw.prepare_feature_assembly(request)
            self.assertEqual(plan.feature_provenance["metadata_policy"], "advisory")
            self.assertTrue(any("Metadata advisory" in warning for warning in plan.validation_warnings))
            fw.execute_feature_assembly(plan)
            np.testing.assert_array_equal(fs._load_grim(str(output))["_amp"], expected)
            # An Assembly result is also a usable body-only study input.
            reused = root/"reused.grim"
            reuse_plan = fw.prepare_feature_assembly(fw.FeatureAssemblyRequest(output, reused))
            fw.execute_feature_assembly(reuse_plan)
            np.testing.assert_array_equal(fs._load_grim(str(reused))["_amp"], expected)

    def test_stale_raw_flag_and_solver_annotation_do_not_block_read_or_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root/"body.grim"
            fs.export_radar_grim(str(base), bor_result=None, placements=[], frequencies_ghz=[1.], azimuths_deg=[0.], elevations_deg=[0.])
            rewrite(base, remove=("rcs_amp_real", "rcs_amp_imag"), additions={"raw_complex_amplitude_preserved": True, "solver_metadata_json": "invalid"})
            grid = RcsGrid.load(str(base))
            np.testing.assert_array_equal(grid.coherent_subtract(grid).rcs_power, 0.)
            payload = fs._load_grim(str(base))
            saved = _save_grim_npz(payload, str(root/"saved.grim"))
            self.assertEqual(str(fs._load_grim(saved)["solver_metadata_json"]), "invalid")
            # Actual non-finite field data still fail.
            rewrite(base, additions={"rcs_phase": np.full((1, 1, 1, 3), np.inf)})
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                fs._load_grim(str(base))

    def test_bad_annotation_cannot_mask_inconsistent_numerical_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/"body.grim"
            fs.export_radar_grim(str(base), bor_result=None, placements=[], frequencies_ghz=[1.], azimuths_deg=[0.], elevations_deg=[0.])
            rewrite(base, additions={"raw_complex_amplitude_preserved": [True, False]})
            # The malformed flag is advisory when the actual arrays agree.
            RcsGrid.load(str(base))
            rewrite(base, additions={"rcs_amp_real": np.ones((1, 1, 1, 3))})
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                RcsGrid.load(str(base))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                fs._load_grim(str(base))


if __name__ == "__main__":
    unittest.main()
