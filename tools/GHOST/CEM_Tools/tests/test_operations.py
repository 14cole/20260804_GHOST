
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
    _mesh_certification_sources,
    load_grim,
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


def assert_source_mesh_certification(payload, label):
    """Explicitly audit optional source certification metadata in a test."""

    sources = _mesh_certification_sources(payload, label)
    if not sources:
        raise AssertionError(f"{label}: expected source-field certification")
    return sources


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
    """Attach the original single-polarization workflow-unit certificate."""

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


def _test_output_attestation(
    frequency: 'float',
    *,
    polarizations: 'list[str] | None' = None,
    polarization: 'str | None' = None,
) -> 'dict':
    attestation = {
        "schema": "ghost.workflow.embedded-attestation.v1",
        "run_id": "test-run",
        "solver_source_sha256": "1" * 64,
        "runtime_environment_sha256": "2" * 64,
        "geometry_input_sha256": "3" * 64,
        "run_solve_spec_sha256": "4" * 64,
        "unit_solve_spec_sha256": "5" * 64,
        "solver_config_sha256": "6" * 64,
        "angular_grid_kind": "azimuths_deg",
        "angular_grid_sha256": "7" * 64,
        "frequency_ghz": float(frequency),
    }
    if polarizations is not None:
        attestation["polarizations"] = list(polarizations)
    if polarization is not None:
        attestation["polarization"] = str(polarization)
    return attestation


def certify_embedded_single_source(path: 'Path') -> 'None':
    """Attach the pre-dual-output embedded single-channel contract."""

    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    frequency = float(np.asarray(payload["frequencies"]).ravel()[0])
    polarization = str(np.asarray(payload["polarizations"]).ravel()[0])
    quality = {"passed": True}
    payload["solver_metadata_json"] = np.asarray(json.dumps({
        "schema": "ghost.solver_metadata.v1",
        "polarization": polarization,
        "metadata": {
            "quality_gate": quality,
            "mesh_convergence": {
                "passed": True,
                "published_mesh": "fine",
                "base_quality_gate": quality,
                "fine_quality_gate": quality,
            },
            "output_attestation": _test_output_attestation(
                frequency, polarization=polarization
            ),
        },
    }))
    with path.open("wb") as stream:
        np.savez(stream, **payload)


def write_dual_source(
    path: 'Path',
    frequency: 'float',
    vv_amplitude: 'np.ndarray',
    hh_amplitude: 'np.ndarray',
) -> 'None':
    k = 2.0 * math.pi * frequency * 1e9 / C0
    amplitude = np.stack(
        [np.asarray(vv_amplitude, complex), np.asarray(hh_amplitude, complex)],
        axis=-1,
    ).reshape(len(vv_amplitude), 1, 1, 2)
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
            polarizations=np.asarray(["VV", "HH"]),
            rcs_power=(np.abs(amplitude) ** 2 / (4.0 * k)).astype(np.float32),
            rcs_phase=np.angle(amplitude).astype(np.float32),
            rcs_amp_real=amplitude.real.astype(np.float64),
            rcs_amp_imag=amplitude.imag.astype(np.float64),
            rcs_domain=np.asarray("power_phase"),
            power_domain=np.asarray("linear_rcs"),
            units=np.asarray(json.dumps(units)),
            phase_reference=np.asarray(PHYSICAL_2D_PHASE_REFERENCE),
            amplitude_convention=np.asarray(PHYSICAL_2D_AMPLITUDE_CONVENTION),
            complex_field_domain=np.asarray(PHYSICAL_2D_FIELD_DOMAIN),
            raw_complex_amplitude_preserved=np.asarray(True),
        )


def certify_dual_source(
    path: 'Path', *, failed_channel: 'str | None' = None,
    failed_gate: 'str' = "quality",
) -> 'None':
    """Attach the aggregate VV/HH production certificate and attestation."""

    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    frequency = float(np.asarray(payload["frequencies"]).ravel()[0])

    channel_metadata = {}
    mesh_channels = {}
    quality_channels = {}
    base_channels = {}
    fine_channels = {}
    for polarization in ("VV", "HH"):
        quality_passed = not (
            failed_channel == polarization and failed_gate == "quality"
        )
        mesh_passed = not (
            failed_channel == polarization and failed_gate == "mesh"
        )
        quality = {"passed": quality_passed}
        base_quality = {"passed": True}
        fine_quality = {"passed": True}
        mesh = {
            "schema": "ghost.solver.mesh-convergence.v1",
            "passed": mesh_passed,
            "published_mesh": "fine",
            "base_quality_gate": base_quality,
            "fine_quality_gate": fine_quality,
        }
        channel_metadata[polarization] = {
            "quality_gate": quality,
            "mesh_convergence": mesh,
            "mesh_convergence_certified": True,
            "certified_entry_point": True,
            "published_mesh": "fine",
        }
        mesh_channels[polarization] = mesh
        quality_channels[polarization] = quality
        base_channels[polarization] = base_quality
        fine_channels[polarization] = fine_quality

    aggregate_quality = {
        # Keep the aggregate true so the consumer must inspect each channel.
        "passed": True,
        "channels": quality_channels,
    }
    aggregate_mesh = {
        "schema": "ghost.solver.mesh-convergence.co-polarized.v1",
        "passed": True,
        "published_mesh": "fine",
        "channels": mesh_channels,
        "polarizations": mesh_channels,
        "base_quality_gate": {
            "passed": True,
            "channels": base_channels,
        },
        "fine_quality_gate": {
            "passed": True,
            "channels": fine_channels,
        },
    }
    payload["solver_metadata_json"] = np.asarray(json.dumps({
        "schema": "ghost.solver_metadata.v1",
        "polarizations": ["VV", "HH"],
        "polarization_mapping": {"VV": "TE", "HH": "TM"},
        "metadata": {
            "polarizations": ["VV", "HH"],
            "polarization_mapping": {"VV": "TE", "HH": "TM"},
            "mesh_convergence": aggregate_mesh,
            "mesh_convergence_certified": True,
            "quality_gate": aggregate_quality,
            "channel_metadata": channel_metadata,
            "certified_entry_point": True,
            "published_mesh": "fine",
            "output_attestation": _test_output_attestation(
                frequency, polarizations=["VV", "HH"]
            ),
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

    def test_certification_is_advisory_and_not_inherited(self):
        """Transforms accept certified inputs without certifying their output."""

        opn, frd = self.library()
        for folder in (opn, frd):
            for path in folder.glob("*.grim"):
                certify_source(path)
                assert_source_mesh_certification(load_grim(path), str(path))

        pol_out = self.root / "cert_pol"
        for path in concatenate_polarizations(opn, pol_out).written:
            payload = load_grim(path)
            self.assertNotIn(MESH_CERTIFICATION_KEY, payload)
            self.assertNotIn("solver_metadata_json", payload)

        freq_out = self.root / "cert_freq"
        joined = concatenate_frequencies(pol_out, freq_out).written[0]
        joined_payload = load_grim(joined)
        self.assertNotIn(MESH_CERTIFICATION_KEY, joined_payload)
        self.assertNotIn("solver_metadata_json", joined_payload)

        delta = subtract_datasets(
            opn, frd, self.root / "cert_delta"
        ).written[0]
        payload = load_grim(delta)
        self.assertNotIn(MESH_CERTIFICATION_KEY, payload)
        self.assertNotIn("solver_metadata_json", payload)

    def test_dual_channel_certification_is_advisory(self):
        opn = self.root / "OPN"
        frd = self.root / "FRD"
        opn.mkdir()
        frd.mkdir()
        clean_vv = np.asarray([1.0 + 0.2j, 0.5 - 0.1j, -0.2 + 0.3j])
        clean_hh = np.asarray([0.2 + 0.8j, -0.4 + 0.1j, 0.7 - 0.2j])
        featured_path = opn / "3.000GHz_DUAL-00-00_0.010gap_OPN.grim"
        clean_path = frd / "3.000GHz_DUAL-00-00_0.010gap_FRD.grim"
        write_dual_source(
            featured_path, 3.0, clean_vv + (0.4 + 0.1j),
            clean_hh + (-0.2 + 0.3j),
        )
        write_dual_source(clean_path, 3.0, clean_vv, clean_hh)
        certify_dual_source(featured_path)
        certify_dual_source(clean_path)

        for path in (featured_path, clean_path):
            assert_source_mesh_certification(load_grim(path), str(path))
        delta_path = subtract_datasets(
            opn, frd, self.root / "dual_delta"
        ).written[0]
        delta = load_grim(delta_path)
        self.assertNotIn(MESH_CERTIFICATION_KEY, delta)
        self.assertNotIn("solver_metadata_json", delta)
        np.testing.assert_array_equal(delta["polarizations"], ["VV", "HH"])
        amplitude = delta["rcs_amp_real"] + 1j * delta["rcs_amp_imag"]
        np.testing.assert_allclose(amplitude[..., 0], 0.4 + 0.1j)
        np.testing.assert_allclose(amplitude[..., 1], -0.2 + 0.3j)

    def test_failed_channel_certification_does_not_block_subtraction(self):
        clean_vv = np.asarray([1.0 + 0.2j, 0.5 - 0.1j, -0.2 + 0.3j])
        clean_hh = np.asarray([0.2 + 0.8j, -0.4 + 0.1j, 0.7 - 0.2j])
        for failed_gate, expected in (
            ("quality", "HH.*quality gate"),
            ("mesh", "HH.*mesh-convergence"),
        ):
            with self.subTest(failed_gate=failed_gate):
                case_root = self.root / failed_gate
                opn = case_root / "OPN"
                frd = case_root / "FRD"
                opn.mkdir(parents=True)
                frd.mkdir(parents=True)
                featured_path = (
                    opn / "3.000GHz_DUAL-00-00_0.010gap_OPN.grim"
                )
                clean_path = (
                    frd / "3.000GHz_DUAL-00-00_0.010gap_FRD.grim"
                )
                write_dual_source(
                    featured_path, 3.0, clean_vv + 0.4, clean_hh + 0.2j
                )
                write_dual_source(clean_path, 3.0, clean_vv, clean_hh)
                certify_dual_source(
                    featured_path,
                    failed_channel="HH",
                    failed_gate=failed_gate,
                )
                certify_dual_source(clean_path)
                with self.assertRaisesRegex(CemToolError, expected):
                    _mesh_certification_sources(
                        load_grim(featured_path), str(featured_path)
                    )
                delta = load_grim(
                    subtract_datasets(
                        opn, frd, case_root / "delta"
                    ).written[0]
                )
                self.assertNotIn(MESH_CERTIFICATION_KEY, delta)
                amplitude = (
                    delta["rcs_amp_real"] + 1j * delta["rcs_amp_imag"]
                )
                np.testing.assert_allclose(amplitude[..., 0], 0.4)
                np.testing.assert_allclose(amplitude[..., 1], 0.2j)

    def test_mixed_certification_is_advisory(self):
        opn, frd = self.library()
        certify_source(next(opn.glob("*.grim")))
        delta = load_grim(
            subtract_datasets(opn, frd, self.root / "delta").written[0]
        )
        self.assertNotIn(MESH_CERTIFICATION_KEY, delta)

    def test_malformed_certification_is_advisory(self):
        opn, frd = self.library()
        source_path = next(opn.glob("*.grim"))
        with np.load(source_path, allow_pickle=False) as source:
            payload = {
                key: np.array(source[key], copy=True)
                for key in source.files
            }
        payload[MESH_CERTIFICATION_KEY] = np.asarray("{not-json")
        with source_path.open("wb") as stream:
            np.savez(stream, **payload)
        delta = load_grim(
            subtract_datasets(opn, frd, self.root / "delta").written[0]
        )
        self.assertNotIn(MESH_CERTIFICATION_KEY, delta)

    def test_embedded_single_channel_contract_remains_valid(self):
        path = self.root / "TE_3.000GHz_LEGACY-00-00_FRD.grim"
        write_source(
            path, 3.0, "TE",
            np.asarray([1.0 + 0.2j, 0.5 - 0.1j, -0.2 + 0.3j]),
        )
        certify_embedded_single_source(path)
        payload = load_grim(path)
        assert_source_mesh_certification(payload, str(path))

    def test_uncertified_source_has_no_certificate_evidence(self):
        opn, _frd = self.library()
        path = next(opn.glob("*.grim"))
        self.assertIsNone(
            _mesh_certification_sources(load_grim(path), str(path))
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
