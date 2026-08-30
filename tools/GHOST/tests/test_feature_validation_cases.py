#!/usr/bin/env python3
"""Tests for four-artifact clean/featured reconstruction validation."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

from components import (  # noqa: E402
    COMPONENT_AMPLITUDE_CONVENTION,
    COMPONENT_COMPLEX_FIELD_DOMAIN,
    COMPONENT_PHASE_REFERENCE,
)
from grim_io import _save_grim_npz  # noqa: E402
import validate_feature_reconstruction as validation  # noqa: E402


def _write_field(path, amplitude, *, feature_response_sha256=None):
    field = np.asarray(amplitude, dtype=np.complex128)
    if field.shape != (3, 2, 1, 3):
        raise AssertionError(f"unexpected test field shape {field.shape}")
    payload = {
        "azimuths": np.asarray([0.0, 120.0, 240.0]),
        "elevations": np.asarray([-20.0, 20.0]),
        "frequencies": np.asarray([9.5]),
        "polarizations": np.asarray(["VV", "HH", "VH"]),
        "polarization_alias_primary": "VV",
        "polarization_aliases_json": json.dumps(["VV", "HH", "VH"]),
        "combine_role": "coherent",
        "rcs_power": (4.0 * math.pi * np.abs(field) ** 2).astype(np.float32),
        "rcs_phase": np.angle(field).astype(np.float32),
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "source_path": "independent 3-D validation fixture",
        "history": "non-BoR clean/featured validation test",
        "units": json.dumps({
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        }),
        "phase_reference": COMPONENT_PHASE_REFERENCE,
        "amplitude_convention": COMPONENT_AMPLITUDE_CONVENTION,
        "complex_field_domain": COMPONENT_COMPLEX_FIELD_DOMAIN,
        "raw_complex_amplitude_preserved": True,
        "rcs_amp_real": field.real,
        "rcs_amp_imag": field.imag,
    }
    if feature_response_sha256 is not None:
        payload["feature_provenance_json"] = json.dumps([{
            "schema": "ghost.workflow.coherent-feature-addition.v1",
            "details": {
                "placements": [{
                    "dataset_id": "test-feature",
                    "dataset_content_sha256": feature_response_sha256,
                }],
            },
        }])
    return Path(_save_grim_npz(payload, str(path)))


class FeatureCaseComparisonTests(unittest.TestCase):
    def test_weak_cross_pol_channel_cannot_hide_behind_strong_copol(self):
        reference = np.ones((20, 3), dtype=np.complex128)
        reference[:, 2] = 1.0e-6 + 0.0j
        estimate = reference.copy()
        estimate[:, 2] *= -1.0

        result = validation._comparison_metrics(
            reference,
            estimate,
            ["VV", "HH", "VH"],
        )

        self.assertLess(result["normalized_complex_rms"], 1.0e-4)
        self.assertFalse(result["per_channel"]["VH"]["passed"])
        self.assertFalse(result["gates"]["every_polarization_channel"])
        self.assertFalse(result["passed"])

    def test_exact_clean_featured_and_delta_fields_pass(self):
        base = np.full((3, 2, 1, 3), 2.0 + 0.4j, dtype=np.complex128)
        delta = np.full((3, 2, 1, 3), 0.15 - 0.08j, dtype=np.complex128)
        response_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_truth = _write_field(root / "clean_truth.grim", base)
            clean_prediction = _write_field(root / "clean_prediction.grim", base)
            featured_truth = _write_field(root / "featured_truth.grim", base + delta)
            featured_prediction = _write_field(
                root / "featured_prediction.grim",
                base + delta,
                feature_response_sha256=response_sha256,
            )

            result = validation.compare_feature_case(
                clean_truth=clean_truth,
                clean_prediction=clean_prediction,
                featured_truth=featured_truth,
                featured_prediction=featured_prediction,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["feature_response_content_sha256"], [response_sha256]
        )
        self.assertEqual(
            set(result["artifact_sha256"]),
            set(validation.CASE_REQUIRED_PATHS),
        )
        for section in (
            "clean_baseline",
            "featured_total",
            "isolated_feature_delta",
        ):
            self.assertEqual(result[section]["normalized_complex_rms"], 0.0)
            self.assertTrue(result[section]["passed"])

    def test_isolated_delta_catches_error_hidden_by_large_clean_body(self):
        base = np.full((3, 2, 1, 3), 100.0 + 15.0j, dtype=np.complex128)
        delta = np.full((3, 2, 1, 3), 0.1 - 0.04j, dtype=np.complex128)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "clean_truth": _write_field(root / "clean_truth.grim", base),
                "clean_prediction": _write_field(root / "clean_prediction.grim", base),
                "featured_truth": _write_field(root / "featured_truth.grim", base + delta),
                # A subtraction-order or phase-convention error reverses the
                # installed-minus-clean feature while barely moving total RCS.
                "featured_prediction": _write_field(
                    root / "featured_prediction.grim", base - delta
                ),
            }
            result = validation.compare_feature_case(**paths)

        self.assertTrue(result["clean_baseline"]["passed"])
        self.assertTrue(result["featured_total"]["passed"])
        self.assertLess(
            result["featured_total"]["normalized_complex_rms"], 0.01
        )
        self.assertFalse(result["isolated_feature_delta"]["passed"])
        self.assertAlmostEqual(
            result["isolated_feature_delta"]["normalized_complex_rms"],
            2.0,
            places=12,
        )
        self.assertGreater(
            result["isolated_feature_delta"]["phase_error_rms_deg"], 179.0
        )
        self.assertFalse(result["passed"])


class FeatureManifestTests(unittest.TestCase):
    def test_paths_are_relative_to_manifest_and_gates_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "validation" / "cases.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({
                "schema": validation.CASE_MANIFEST_SCHEMA,
                "gates": {"max_normalized_rms": 0.12},
                "cases": [{
                    "id": "flat-plate-fastener",
                    "name": "flat plate with one fastener",
                    "body": "finite rectangular PEC plate",
                    "feature": "off-center installed-minus-clean fastener",
                    "clean_truth": "results/clean_truth.grim",
                    "clean_prediction": "results/clean_import.grim",
                    "featured_truth": "results/featured_truth.grim",
                    "featured_prediction": "results/reconstructed.grim",
                    "gates": {"min_coherence": 0.99},
                }],
            }), encoding="utf-8")

            cases = validation.load_case_manifest(manifest)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["case_id"], "flat-plate-fastener")
        self.assertEqual(cases[0]["name"], "flat plate with one fastener")
        self.assertEqual(cases[0]["gates"]["max_normalized_rms"], 0.12)
        self.assertEqual(cases[0]["gates"]["min_coherence"], 0.99)
        self.assertEqual(
            cases[0]["paths"]["clean_truth"],
            str((manifest.parent / "results" / "clean_truth.grim").resolve()),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
