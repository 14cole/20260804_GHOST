
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CEM_Tools"))
sys.path.insert(0, str(ROOT / "Backend"))

from cem_tools.grim_native import (
    MESH_CERTIFICATION_KEY,
    load_grim,
    require_production_mesh_certification,
)
from cem_tools.errors import CemToolError
from cem_tools.operations import (
    concatenate_frequencies,
    concatenate_polarizations,
    convert_files,
    rename_files,
    subtract_datasets,
)
from feature_sum import (
    C0,
    PHYSICAL_2D_AMPLITUDE_CONVENTION,
    PHYSICAL_2D_FIELD_DOMAIN,
    PHYSICAL_2D_PHASE_REFERENCE,
    load_seam_from_grim,
    make_delta_grim,
)


def write_source(path: 'Path', frequency: 'float', polarization: 'str', amplitude: 'np.ndarray') -> 'None':
    k = 2.0 * math.pi * frequency * 1e9 / C0
    shape = (len(amplitude), 1, 1, 1)
    amp = np.asarray(amplitude, complex).reshape(shape)
    units = {
        "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
        "rcs_log_unit": "dBke", "rcs_linear_quantity": "sigma_2d",
    }
    with path.open("wb") as stream:
        np.savez(
            stream,
            azimuths=np.asarray([-30.0, 0.0, 30.0]),
            elevations=np.asarray([0.0]),
            frequencies=np.asarray([frequency]),
            polarizations=np.asarray([polarization]),
            rcs_power=(np.abs(amp) ** 2 / (4.0 * k)).astype(np.float32),
            rcs_phase=np.angle(amp).astype(np.float32),
            rcs_amp_real=amp.real.astype(np.float64),
            rcs_amp_imag=amp.imag.astype(np.float64),
            rcs_domain=np.asarray("power_phase"),
            power_domain=np.asarray("linear_rcs"),
            units=np.asarray(json.dumps(units)),
            phase_reference=np.asarray(PHYSICAL_2D_PHASE_REFERENCE),
            amplitude_convention=np.asarray(PHYSICAL_2D_AMPLITUDE_CONVENTION),
            complex_field_domain=np.asarray(PHYSICAL_2D_FIELD_DOMAIN),
            raw_complex_amplitude_preserved=np.asarray(True),
        )


def certify_source(path: 'Path') -> 'None':
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    frequency = float(np.asarray(payload["frequencies"]).ravel()[0])
    polarization = str(np.asarray(payload["polarizations"]).ravel()[0])
    quality = {"passed": True}
    payload["solver_metadata_json"] = np.asarray(json.dumps({
        "schema": "ghost.solver.audit.v1",
        "metadata": {
            "workflow_unit": {
                "unit_sha256": path.stem,
                "frequency_ghz": frequency,
                "polarization": polarization,
                "published_mesh": "fine",
                "mesh_convergence_policy": {"fine_factor": 1.5},
            },
            "mesh_convergence": {
                "passed": True,
                "published_mesh": "fine",
                "base_quality_gate": quality,
                "fine_quality_gate": quality,
            },
        },
    }))
    with path.open("wb") as stream:
        np.savez(stream, **payload)


class OperationsTest(unittest.TestCase):
    def setUp(self) -> 'None':
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> 'None':
        self.temporary.cleanup()

    def library(self) -> 'tuple[Path, Path]':
        opn = self.root / "OPN"
        frd = self.root / "FRD"
        opn.mkdir()
        frd.mkdir()
        for frequency in (3.0, 6.0):
            for polarization, offset in (("TM", 0.2j), ("TE", 0.4 + 0j)):
                clean = np.asarray([1 + 1j, 2 - 0.5j, -0.2 + 0.8j])
                featured = clean + offset + frequency * 0.01
                common = f"{frequency:.3f}GHz_SEAL-00-01_0.010gap"
                write_source(frd / f"{polarization}_{common}_FRD.grim",
                             frequency, polarization, clean)
                write_source(opn / f"{polarization}_{common}_OPN.grim",
                             frequency, polarization, featured)
        return opn, frd

    def test_concat_and_coherent_subtract_are_solver_compatible(self) -> 'None':
        opn, frd = self.library()
        pol_out = self.root / "pol"
        pol_result = concatenate_polarizations(opn, pol_out)
        self.assertEqual(len(pol_result.written), 2)
        for path in pol_result.written:
            self.assertEqual(set(load_grim(path)["polarizations"]), {"TM", "TE"})
        both_out = self.root / "both"
        both_result = concatenate_frequencies(pol_out, both_out)
        self.assertEqual(len(both_result.written), 1)
        both = load_grim(both_result.written[0])
        self.assertEqual(set(both["polarizations"]), {"TM", "TE"})
        self.assertEqual(list(both["frequencies"]), [3.0, 6.0])

        freq_out = self.root / "freq"
        freq_result = concatenate_frequencies(opn, freq_out)
        self.assertEqual(len(freq_result.written), 2)
        for path in freq_result.written:
            self.assertEqual(list(load_grim(path)["frequencies"]), [3.0, 6.0])

        delta_out = self.root / "delta"
        result = subtract_datasets(opn, frd, delta_out)
        self.assertEqual(len(result.written), 1)
        payload = load_grim(result.written[0])
        self.assertEqual(list(payload["frequencies"]), [3.0, 6.0])
        self.assertEqual(list(payload["polarizations"]), ["VV", "HH"])
        amplitude = payload["rcs_amp_real"] + 1j * payload["rcs_amp_imag"]
        for jf, frequency in enumerate((3.0, 6.0)):
            for jp, polarization in enumerate(payload["polarizations"]):
                expected_delta = frequency * 0.01 + (
                    0.2j if polarization == "HH" else 0.4
                )
                np.testing.assert_allclose(
                    amplitude[:, 0, jf, jp], expected_delta,
                    rtol=0.0, atol=1e-15,
                )
        for frequency in (3.0, 6.0):
            coefficients = load_seam_from_grim(str(result.written[0]), frequency)
            self.assertEqual(coefficients.frequency_ghz, frequency)

        # The standalone tools and the solver-native API must publish the same
        # physical delta, not merely files that each loader happens to accept.
        joined_opn = self.root / "joined_opn"
        joined_frd = self.root / "joined_frd"
        opn_joined = concatenate_frequencies(
            pol_out, joined_opn
        ).written[0]
        frd_pol = self.root / "frd_pol"
        concatenate_polarizations(frd, frd_pol)
        frd_joined = concatenate_frequencies(
            frd_pol, joined_frd
        ).written[0]
        native_path = self.root / "native_delta.grim"
        make_delta_grim(
            str(frd_joined), str(opn_joined), str(native_path)
        )
        native = load_grim(native_path)
        np.testing.assert_array_equal(
            payload["polarizations"], native["polarizations"]
        )
        np.testing.assert_allclose(
            payload["rcs_amp_real"], native["rcs_amp_real"], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            payload["rcs_amp_imag"], native["rcs_amp_imag"], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            payload["rcs_power"], native["rcs_power"], rtol=0.0, atol=0.0
        )

    def test_subtraction_requires_both_channels_and_accepts_zero_fields(self):
        opn = self.root / "OPN"
        frd = self.root / "FRD"
        opn.mkdir()
        frd.mkdir()
        zero = np.zeros(3, dtype=complex)
        for folder, role in ((opn, "OPN"), (frd, "FRD")):
            write_source(
                folder / f"TM_3.000GHz_ZERO-00-00_0.000gap_{role}.grim",
                3.0, "TM", zero,
            )
        with self.assertRaisesRegex(CemToolError, "require exactly both"):
            subtract_datasets(opn, frd, self.root / "incomplete")

        write_source(
            opn / "TE_3.000GHz_ZERO-00-00_0.000gap_OPN.grim",
            3.0, "TE", zero,
        )
        write_source(
            frd / "TE_3.000GHz_ZERO-00-00_0.000gap_FRD.grim",
            3.0, "TE", zero,
        )
        delta = subtract_datasets(opn, frd, self.root / "zero_delta").written[0]
        payload = load_grim(delta)
        np.testing.assert_array_equal(payload["rcs_power"], 0.0)
        np.testing.assert_array_equal(payload["rcs_amp_real"], 0.0)
        np.testing.assert_array_equal(payload["rcs_amp_imag"], 0.0)

    def test_subtraction_aligns_equivalent_polarization_aliases(self):
        opn = self.root / "OPN"
        frd = self.root / "FRD"
        opn.mkdir()
        frd.mkdir()
        clean = np.asarray([1.0 + 0.2j, 0.5 - 0.1j, -0.2 + 0.3j])
        for filename_pol, stored_pol, offset in (
            ("VV", "VV", 0.4 + 0.1j),
            ("HH", "HH", -0.2 + 0.3j),
        ):
            write_source(
                opn / f"{filename_pol}_3.000GHz_ALIAS-00-00_0.010gap_OPN.grim",
                3.0, stored_pol, clean + offset,
            )
        for filename_pol, stored_pol in (("TE", "TE"), ("TM", "TM")):
            write_source(
                frd / f"{filename_pol}_3.000GHz_ALIAS-00-00_0.010gap_FRD.grim",
                3.0, stored_pol, clean,
            )
        delta = load_grim(
            subtract_datasets(opn, frd, self.root / "alias_delta").written[0]
        )
        self.assertEqual(list(delta["polarizations"]), ["VV", "HH"])
        amplitude = delta["rcs_amp_real"] + 1j * delta["rcs_amp_imag"]
        np.testing.assert_allclose(amplitude[..., 0], 0.4 + 0.1j)
        np.testing.assert_allclose(amplitude[..., 1], -0.2 + 0.3j)

    def test_mesh_certification_propagates_through_join_and_subtract(self):
        opn, frd = self.library()
        for folder in (opn, frd):
            for path in folder.glob("*.grim"):
                certify_source(path)

        pol_out = self.root / "cert_pol"
        for path in concatenate_polarizations(opn, pol_out).written:
            payload = load_grim(path)
            self.assertIn(MESH_CERTIFICATION_KEY, payload)
            require_production_mesh_certification(payload, str(path))

        freq_out = self.root / "cert_freq"
        joined = concatenate_frequencies(pol_out, freq_out).written[0]
        require_production_mesh_certification(
            load_grim(joined), str(joined)
        )

        delta = subtract_datasets(
            opn, frd, self.root / "cert_delta"
        ).written[0]
        payload = load_grim(delta)
        require_production_mesh_certification(payload, str(delta))
        decoded = json.loads(str(payload[MESH_CERTIFICATION_KEY]))
        self.assertEqual(decoded["source_count"], 8)

    def test_uncertified_source_is_rejected_by_production_preflight(self):
        opn, _frd = self.library()
        path = next(opn.glob("*.grim"))
        with self.assertRaisesRegex(
            CemToolError, "no production mesh certification"
        ):
            require_production_mesh_certification(
                load_grim(path), str(path)
            )

    def test_rename_copy_and_in_place(self) -> 'None':
        source = self.root / "source"
        source.mkdir()
        (source / "VV_SEAL.dat").write_text("one")
        (source / "HH_OTHER.dat").write_text("two")
        copied = rename_files(source, self.root / "copy", "SEAL", "SEAM")
        self.assertEqual([path.name for path in copied.written], ["VV_SEAM.dat"])
        self.assertTrue((source / "VV_SEAL.dat").exists())
        moved = rename_files(source, None, "SEAL", "SEAM", in_place=True)
        self.assertEqual([path.name for path in moved.written], ["VV_SEAM.dat"])
        self.assertFalse((source / "VV_SEAL.dat").exists())

    def test_one_frd_baseline_is_reused_for_multiple_opn_cases(self) -> 'None':
        opn = self.root / "OPN"
        frd = self.root / "FRD"
        opn.mkdir()
        frd.mkdir()
        clean = np.asarray([1 + 1j, 2 - 0.5j, -0.2 + 0.8j])
        cases = (
            ("0.010het_0.020crv", 0.1 + 0.2j),
            ("0.015het_0.025crv", 0.3 + 0.1j),
            ("0.020het_0.020crv", -0.1 + 0.4j),
        )
        for polarization in ("TM", "TE"):
            write_source(
                frd / (
                    f"{polarization}_3.000GHz_"
                    "SEAL-00-00_0.050bmag_FRD.grim"
                ),
                3.0, polarization, clean,
            )
            for suffix, difference in cases:
                write_source(
                    opn / (
                        f"{polarization}_3.000GHz_SEAL-00-00_0.050bmag_"
                        f"{suffix}_OPN.grim"
                    ),
                    3.0, polarization, clean + difference,
                )
        result = subtract_datasets(opn, frd, self.root / "Deltas")
        self.assertEqual(len(result.written), 3)
        self.assertFalse(result.warnings)
        self.assertEqual(
            {path.name for path in result.written},
            {
                f"SEAL-00-00_0.050bmag_{suffix}.grim"
                for suffix, _difference in cases
            },
        )
        expected = dict(cases)
        for path in result.written:
            payload = load_grim(path)
            suffix = path.stem.split("_", 2)[2]
            amplitude = payload["rcs_amp_real"] + 1j * payload["rcs_amp_imag"]
            np.testing.assert_allclose(
                amplitude,
                expected[suffix],
                rtol=0.0,
                atol=1e-15,
            )

    def test_only_rename_allows_in_place_output(self) -> 'None':
        opn, _ = self.library()
        with self.assertRaisesRegex(CemToolError, "only Rename Files"):
            concatenate_polarizations(opn, opn)
        with self.assertRaisesRegex(CemToolError, "only Rename Files"):
            convert_files(opn, opn, ".csv")

    def test_grim_csv_grim_conversion_keeps_grid_values(self) -> 'None':
        source = self.root / "source"
        source.mkdir()
        original = source / "TM_3.000GHz_CASE_0.010gap_OPN.grim"
        write_source(original, 3.0, "TM", np.asarray([1 + 1j, 2 - 0.5j, 0.2j]))
        lossless_result = convert_files(source, self.root / "lossless", ".grim")
        lossless = load_grim(lossless_result.written[0])
        expected = load_grim(original)
        np.testing.assert_array_equal(lossless["rcs_amp_real"], expected["rcs_amp_real"])
        np.testing.assert_array_equal(lossless["rcs_amp_imag"], expected["rcs_amp_imag"])
        self.assertEqual(
            str(lossless["amplitude_convention"]),
            str(expected["amplitude_convention"]),
        )
        csv_result = convert_files(source, self.root / "csv", ".csv")
        self.assertEqual(
            csv_result.written[0].name,
            "TM_3.000GHz_CASE_0.010gap_OPN.csv",
        )
        grim_result = convert_files(self.root / "csv", self.root / "grim", ".grim")
        round_trip = load_grim(grim_result.written[0])
        np.testing.assert_allclose(round_trip["rcs_power"], expected["rcs_power"])
        np.testing.assert_allclose(round_trip["rcs_phase"], expected["rcs_phase"])


if __name__ == "__main__":
    unittest.main()
