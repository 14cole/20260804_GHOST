"""Inspector and Assembly must agree on fields, aliases and independent HV."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ghost_backend.assembly.fields as fs
import ghost_backend.assembly.workflow as fw
from ghost_backend.assembly.inspector import ContributionInspector, _stored_complex_sample
from test_point_scatter_physics import _write_point_grim


def save(path, payload):
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def response_payload(field, *, raw=True, fortran=False, quantity="sigma_3d"):
    shape = field.shape
    frequencies = np.arange(1, shape[2] + 1, dtype=float)
    scale = 4*np.pi if quantity == "sigma_3d" else 1/(4*(2*np.pi*frequencies*1e9/fs.C0))[None, None, :, None]
    payload = dict(azimuths=np.arange(shape[0], dtype=float),
        elevations=np.arange(shape[1], dtype=float), frequencies=frequencies,
        polarizations=np.array(["VV", "HH", "VH", "HV"][:shape[3]]),
        units=json.dumps(dict(frequency="GHz", rcs_linear_quantity=quantity)),
        rcs_power=(scale*abs(field)**2).astype(np.float32),
        rcs_phase=np.angle(field).astype(np.float32))
    if raw:
        payload.update(rcs_amp_real=field.real, rcs_amp_imag=field.imag)
    if fortran:
        for key in ("rcs_power", "rcs_phase", "rcs_amp_real", "rcs_amp_imag"):
            if key in payload:
                payload[key] = np.asfortranarray(payload[key])
    return payload


class InspectorSampleTests(unittest.TestCase):
    def test_bounded_c_and_fortran_reads_share_full_grid_normalization(self):
        rng = np.random.default_rng(125)
        field = rng.normal(size=(129, 129, 2, 4)) + 1j*rng.normal(size=(129, 129, 2, 4))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "body.grim"
            for fortran, raw, quantity in ((False, True, "sigma_3d"),
                    (True, True, "sigma_3d"), (True, False, "sigma_3d"),
                    (False, False, "sigma_2d")):
                with self.subTest(fortran=fortran, raw=raw, quantity=quantity):
                    save(path, response_payload(field, raw=raw, fortran=fortran, quantity=quantity))
                    expected = fs._load_grim(path)["_amp"][128, 127, 1]
                    read_sizes = []
                    original = zipfile.ZipExtFile.read
                    def bounded_read(stream, size=-1):
                        read_sizes.append(size)
                        return original(stream, size)
                    with mock.patch.object(zipfile.ZipExtFile, "read", bounded_read):
                        sample = fs._load_grim_sample(path, 2., 128., 127.)["_amp"][0, 0, 0]
                    np.testing.assert_array_equal(sample, expected)
                    self.assertTrue(all(0 <= size <= 262144 for size in read_sizes), read_sizes)

    def test_numeric_errors_and_incomplete_complex_arrays_are_not_hidden(self):
        field = np.full((2, 2, 1, 3), 1+.5j)
        baseline = response_payload(field)
        invalid = [dict(rcs_phase=np.full(field.shape, np.nan)),
                   dict(rcs_power=np.full(field.shape, -1.)),
                   dict(rcs_power=baseline["rcs_power"]*2),
                   dict(rcs_phase=baseline["rcs_phase"]+.5),
                   dict(rcs_amp_imag=None), dict(azimuths=np.array([0., 0.]))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "body.grim"
            for changes in invalid:
                payload = dict(baseline, **changes)
                payload = {key: value for key, value in payload.items() if value is not None}
                save(path, payload)
                for reader in (lambda: fs._load_grim(path), lambda: fs._load_grim_sample(path, 1., 0., 0.)):
                    with self.subTest(changes=list(changes), reader=reader):
                        with self.assertRaises(ValueError):
                            reader()
            save(path, baseline)
            with self.assertRaisesRegex(ValueError, "exact stored"):
                _stored_complex_sample(path, 1., .5, 0.)
            with self.assertRaises(InterruptedError):
                fs._load_grim_sample(path, 1., 0., 0., lambda: True)
            # Cancellation remains active during decompression, not only on entry.
            with self.assertRaises(InterruptedError):
                fs._load_grim_sample(path, 1., 1., 1., mock.Mock(side_effect=[False]*4 + [True]))

    def test_missing_or_duplicate_physical_channels_are_rejected(self):
        field = np.full((1, 1, 1, 3), 1+.5j)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "body.grim"
            for labels in (["VV", "HH", "V"], ["VV", "HH", "XX"]):
                save(path, dict(response_payload(field), polarizations=labels))
                with self.assertRaises(ValueError):
                    _stored_complex_sample(path, 1., 0., 0.)


class InspectorAssemblyTests(unittest.TestCase):
    def test_inspected_total_matches_built_raw_reconstructed_and_four_channel_assemblies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, point, output = root/"body.grim", root/"point.grim", root/"total.grim"
            fs.save_monostatic_grim({1.: {"theta_deg": np.linspace(0, 180, 181),
                "amp_vv": np.full(181, .1+.2j), "amp_hh": np.full(181, .3-.1j)}},
                np.array([[0, 1], [.1, 1], [.1, -1], [0, -1]]), str(base),
                azimuths_deg=[0., 90., 180., 270.], elevations_deg=[0., 45.])
            with np.load(base, allow_pickle=False) as archive:
                original = {key: archive[key] for key in archive.files}
            _write_point_grim(point, np.diag([.001, .002, 0]), [1.])
            locations = root/"points.csv"
            locations.write_text("placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z\n"
                                 "p,f,0,0,0.1,0,0,1,1,0,0\n")
            for mode in ("raw", "power_phase", "HV_alias", "lowercase", "four_channel", "four_power_phase",
                         "minimal_units", "mixed_case_units"):
                with self.subTest(mode=mode):
                    payload = dict(original)
                    if mode.startswith("four"):
                        original_field = original["rcs_amp_real"] + 1j*original["rcs_amp_imag"]
                        field = np.concatenate((original_field, np.full(original_field.shape[:-1]+(1,), .03+.015j)), axis=-1)
                        # Test independent cross-pols AND nonstandard ordering.
                        field = field[..., [3, 1, 0, 2]]
                        payload.update(polarizations=["hv", " hh ", "V", " vh "],
                            rcs_amp_real=field.real, rcs_amp_imag=field.imag,
                            rcs_power=(4*np.pi*abs(field)**2).astype(np.float32),
                            rcs_phase=np.angle(field).astype(np.float32))
                    elif mode == "HV_alias":
                        payload["polarizations"] = ["VV", "HH", "HV"]
                    elif mode == "lowercase":
                        payload["polarizations"] = ["vv", "hh", "vh"]
                    elif mode == "minimal_units":
                        payload["units"] = json.dumps(dict(rcs_linear_quantity="sigma_3d"))
                    elif mode == "mixed_case_units":
                        payload["units"] = json.dumps(dict(rcs_linear_quantity="SIGMA_3D",
                                                           frequency="ghz", azimuth="DEG", elevation="DEG"))
                    if "power_phase" in mode:
                        payload.pop("rcs_amp_real")
                        payload.pop("rcs_amp_imag")
                    save(base, payload)
                    plan = fw.prepare_feature_assembly(fw.FeatureAssemblyRequest(base, output,
                        point_locations_csv=locations, point_datasets={"f": point}, coordinate_units="meters"))
                    fw.execute_feature_assembly(plan)
                    expected = fs._load_grim(output)["_amp"][0, 1, 0]
                    inspector = ContributionInspector()
                    actual = inspector.evaluate(plan, 1., 0., 45.)
                    np.testing.assert_allclose(actual["body"] + actual["fields"].sum(axis=0), expected,
                                               rtol=1e-12, atol=1e-14)
                    self.assertFalse(actual["body"].flags.writeable)
                    self.assertIs(inspector.evaluate(plan, 1., 0., 45.), actual)
                    if mode.startswith("four"):
                        self.assertEqual(actual["polarizations"], ["VV", "HH", "VH", "HV"])
                        self.assertGreater(abs(actual["body"][2]-actual["body"][3]), .01)
                        np.testing.assert_allclose(actual["fields"][:, 2], actual["fields"][:, 3], atol=1e-16)


if __name__ == "__main__":
    unittest.main()
