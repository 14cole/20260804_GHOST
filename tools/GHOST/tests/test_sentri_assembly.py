"""SENTRi import/save/Assembly field and radar-coordinate interoperability."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))
sys.path.insert(0, str(ROOT.parent.parent / "GRIM_Revised_2"))

import feature_sum
from components import (
    COMPONENT_AMPLITUDE_CONVENTION,
    COMPONENT_COMPLEX_FIELD_DOMAIN,
    COMPONENT_PHASE_REFERENCE,
)
from grim_dataset import RcsGrid
from test_sentri_io import COMPACT_HEADER, DESCRIPTIVE_HEADER
from test_point_scatter_physics import _pattern_dict


class SentriAssemblyTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def imported(self, *, descriptive=False, theta_values=(0, 90, 180)):
        # Reciprocal cross-pol with distinct nonzero phases in every channel.
        # theta/phi poles and waterline exercise ordering and the signed basis.
        rows = []
        for theta in theta_values:
            for phi in (0, 90):
                phase = 12 + phi / 3 + theta / 4
                frequency = 1e9 if descriptive else 1000
                rows.append(
                    f"{frequency},{theta},{phi},0,{phase + 10},"
                    f"6.020599913,{phase + 20},-20,{phase + 30},-20,{phase + 30}"
                )
        source = self.root / "sentri.csv"
        header = DESCRIPTIVE_HEADER if descriptive else COMPACT_HEADER
        source.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return RcsGrid.read_SENTRi(source)

    def save_payload(self, grid):
        path = self.root / "body.grim"
        grid.save(path)
        return path, feature_sum._load_grim(str(path))

    def validate(self, payload):
        return feature_sum._validate_declared_coherent_base(
            payload, "SENTRi body", allow_legacy_metadata=False
        )

    def test_import_save_and_strict_assembly_preserve_complex_fields(self):
        for descriptive in (False, True):
            with self.subTest(descriptive=descriptive):
                native = self.imported(descriptive=descriptive)
                grid = native.convert_sentri_elevation_to_grim()
                _path, payload = self.save_payload(grid)
                validated = self.validate(payload)
                self.assertEqual(str(validated["phase_reference"]), COMPONENT_PHASE_REFERENCE)
                self.assertEqual(str(validated["amplitude_convention"]), COMPONENT_AMPLITUDE_CONVENTION)
                self.assertEqual(str(validated["complex_field_domain"]), COMPONENT_COMPLEX_FIELD_DOMAIN)
                self.assertEqual(validated[feature_sum._LEGACY_BASE_ASSUMPTIONS_KEY], ())
                # Independent oracle from the SENTRi CSV: F=sqrt(sigma/4pi)e^jphase.
                # Co- and cross-pol phases retain their signs, with no pi offsets.
                for ia, phi in enumerate(grid.azimuths):
                    for ie, elevation in enumerate(grid.elevations):
                        theta = 90 - elevation
                        phase = 12 + phi / 3 + theta / 4
                        expected = np.sqrt(np.array([4, .01, .01, 1]) / (4 * np.pi))
                        expected = expected * np.exp(1j * np.deg2rad(
                            phase + np.array([20, 30, 30, 10])
                        ))
                        np.testing.assert_allclose(validated["_amp"][ia, ie, 0], expected, rtol=2e-6)
                contract = feature_sum.validate_assembly_base_grid_metadata(
                    payload,
                    {"azimuths_deg": grid.azimuths, "elevations_deg": grid.elevations,
                     "frequencies_ghz": grid.frequencies, "axis_az_deg": 0., "axis_el_deg": 0.},
                    "SENTRi body", allow_legacy_metadata=False,
                )
                self.assertEqual(contract["legacy_missing_metadata"], [])

    def test_existing_sentri_grim_resolves_missing_fields_without_rewriting_source(self):
        grid = self.imported().convert_sentri_elevation_to_grim()
        for key in ("phase_reference", "amplitude_convention", "complex_field_domain",
                    "time_convention", "sentri_far_field_reference"):
            grid.extra.pop(key)
        path, payload = self.save_payload(grid)
        before = hashlib.sha256(path.read_bytes()).digest()
        original = payload["_amp"].copy()
        validated = self.validate(payload)
        np.testing.assert_array_equal(validated["_amp"], original)
        self.assertEqual(validated[feature_sum._LEGACY_BASE_ASSUMPTIONS_KEY], ())
        self.assertNotIn("phase_reference", payload)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)
        # Exercise publication as well: an empty feature batch adds zero and
        # must retain the imported complex body field and record its convention.
        output = self.root / "assembled.grim"
        feature_sum.add_features_to_monostatic_grim(
            str(path), str(output), declared_coherent_base=True,
            allow_legacy_base_metadata=False,
            radar_grid={"azimuths_deg": grid.azimuths, "elevations_deg": grid.elevations,
                        "frequencies_ghz": grid.frequencies, "axis_az_deg": 0., "axis_el_deg": 0.},
        )
        result = feature_sum._load_grim(str(output))
        np.testing.assert_array_equal(result["polarizations"], ["VV", "HH", "VH", "HV"])
        np.testing.assert_allclose(result["_amp"], original[..., [0, 3, 2, 1]], rtol=2e-6)
        provenance = json.loads(str(result["feature_provenance_json"]))
        self.assertIn("outgoing exp(-jkr)/r removed",
                      provenance[-1]["base_coherent_metadata_contract"]["source_field_convention"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)

    def test_sentri_resolution_does_not_override_conflicting_fields(self):
        _path, payload = self.save_payload(self.imported().convert_sentri_elevation_to_grim())
        for key in ("phase_reference", "amplitude_convention", "complex_field_domain", "time_convention"):
            with self.subTest(key=key):
                conflicting = dict(payload, **{key: np.asarray("different convention")})
                with self.assertRaisesRegex(ValueError, "contradict"):
                    self.validate(conflicting)

    def test_feature_addition_preserves_distinct_hv_and_vh_body_samples(self):
        grid = self.imported(theta_values=(45,)).convert_sentri_elevation_to_grim()
        # Distinct source cross-pols must survive; do not silently average them.
        grid.rcs_phase[..., 1] += .4
        path, payload = self.save_payload(grid)
        original = payload["_amp"][..., [0, 3, 2, 1]]
        pattern = feature_sum.prepare_point_pattern(_pattern_dict(np.asarray([
            [.006 + .001j, .0012 - .0003j, -.0007 + .0002j],
            [.0012 - .0003j, .0035 - .0008j, .0005 + .0004j],
            [-.0007 + .0002j, .0005 + .0004j, .002 + .0006j],
        ]), frequencies=(1.,)))
        output = self.root / "with_point.grim"
        feature_sum.add_features_to_monostatic_grim(
            str(path), str(output), declared_coherent_base=True,
            allow_legacy_base_metadata=False,
            points=[{"pattern": pattern, "location": [.03, -.02, .01],
                     "aperture_normal": [0., 0., 1.], "roll_ref": [1., 0., 0.]}],
            radar_grid={"azimuths_deg": grid.azimuths, "elevations_deg": grid.elevations,
                        "frequencies_ghz": grid.frequencies, "axis_az_deg": 0., "axis_el_deg": 90.},
        )
        total = feature_sum._load_grim(str(output))["_amp"]
        delta = feature_sum._load_grim(feature_sum.feature_only_output_path(str(output)))["_amp"]
        self.assertGreater(np.min(np.abs(delta[..., 2])), 1e-5)
        np.testing.assert_allclose(total, original + delta[..., [0, 1, 2, 2]], atol=1e-14)
        np.testing.assert_allclose(total[..., 2] - total[..., 3],
                                   original[..., 2] - original[..., 3], atol=1e-14)
        # Provenance must also account for changes to either measured cross-pol.
        validated = self.validate(payload)
        before = feature_sum.assembly_response_physics_sha256(validated)
        changed = dict(validated, _amp=validated["_amp"].copy())
        changed["_amp"][..., 1] += .001j
        self.assertNotEqual(before, feature_sum.assembly_response_physics_sha256(changed))

    def test_source_label_alone_cannot_supply_missing_field_conventions(self):
        grid = self.imported().convert_sentri_elevation_to_grim()
        for key in ("phase_reference", "amplitude_convention", "complex_field_domain"):
            grid.extra.pop(key)
        grid.extra.pop("sentri_phase_mapping")
        _path, payload = self.save_payload(grid)
        with self.assertRaisesRegex(ValueError, "strict coherent-base validation is missing"):
            self.validate(payload)

    def test_native_theta_still_requires_coordinate_conversion(self):
        grid = self.imported(theta_values=(90,))
        _path, payload = self.save_payload(grid)
        self.validate(payload)
        with self.assertRaisesRegex(ValueError, "unconverted SENTRi"):
            feature_sum.validate_assembly_base_grid_metadata(
                payload,
                {"azimuths_deg": grid.azimuths, "elevations_deg": grid.elevations,
                 "frequencies_ghz": grid.frequencies, "axis_az_deg": 0., "axis_el_deg": 0.},
                "SENTRi body", allow_legacy_metadata=False,
            )


if __name__ == "__main__":
    unittest.main()
