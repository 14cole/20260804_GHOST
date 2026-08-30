#!/usr/bin/env python3
"""Tests for the definition-only external full-wave case handoff."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


GHOST_ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = GHOST_ROOT / "geometry_tests" / "non_bor_feature_validation"
sys.path.insert(0, str(CASE_ROOT))
sys.path.insert(0, str(GHOST_ROOT / "Backend"))

from components import (  # noqa: E402
    COMPONENT_AMPLITUDE_CONVENTION,
    COMPONENT_COMPLEX_FIELD_DOMAIN,
    COMPONENT_PHASE_REFERENCE,
)
from grim_io import _save_grim_npz  # noqa: E402
import prepare_external_cases as harness  # noqa: E402


PLAN_PATH = CASE_ROOT / "external_case_plan.json"


def _write_structural_grim(
    path, contract, *, frequencies=None, include_feature_provenance=True
):
    grid = contract["acceptance_grid"]
    azimuths = np.asarray(grid["azimuths_deg"], dtype=float)
    elevations = np.asarray(grid["elevations_deg"], dtype=float)
    if frequencies is None:
        frequencies = grid["frequencies_GHz"]
    frequencies = np.asarray(frequencies, dtype=float)
    polarizations = np.asarray(contract["polarizations"], dtype=str)
    shape = (
        len(azimuths), len(elevations), len(frequencies), len(polarizations)
    )
    amplitude = np.full(shape, 0.2 + 0.1j, dtype=np.complex128)
    payload = {
        "azimuths": azimuths,
        "elevations": elevations,
        "frequencies": frequencies,
        "polarizations": polarizations,
        "polarization_alias_primary": "VV",
        "polarization_aliases_json": json.dumps(["VV", "HH", "VH"]),
        "combine_role": "coherent",
        "rcs_power": (4.0 * math.pi * np.abs(amplitude) ** 2).astype(np.float32),
        "rcs_phase": np.angle(amplitude).astype(np.float32),
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "source_path": "temporary structural preflight fixture",
        "history": "metadata/grid test only; not a Maxwell reference result",
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
        "rcs_amp_real": amplitude.real,
        "rcs_amp_imag": amplitude.imag,
    }
    if Path(path).stem == "featured_prediction" and include_feature_provenance:
        payload["feature_provenance_json"] = json.dumps([{
            "schema": "ghost.workflow.coherent-feature-addition.v1",
            "details": {
                "placements": [{
                    "dataset_content_sha256": "a" * 64,
                }],
            },
        }])
    return Path(_save_grim_npz(payload, str(path)))


class ExternalCasePlanTests(unittest.TestCase):
    def test_plan_covers_four_bodies_and_fourteen_concrete_configurations(self):
        plan = harness.load_external_plan(PLAN_PATH)

        self.assertEqual(
            set(plan["bodies"]),
            {
                "wedge-ramp",
                "swept-wing-panel",
                "rounded-enclosure",
                "vehicle-door-section",
            },
        )
        self.assertEqual(len(plan["cases"]), 14)
        self.assertEqual(
            plan["recommended_execution_order"],
            [
                "rounded-enclosure",
                "wedge-ramp",
                "swept-wing-panel",
                "vehicle-door-section",
            ],
        )
        body_sequence = [case["body_id"] for case in plan["cases"]]
        self.assertEqual(body_sequence[:6], ["rounded-enclosure"] * 6)
        self.assertEqual(body_sequence[6:8], ["wedge-ramp", "swept-wing-panel"])
        self.assertEqual(body_sequence[8:], ["vehicle-door-section"] * 6)
        self.assertEqual(
            plan["solve_contract"]["acceptance_grid"]["frequencies_GHz"],
            [8.0, 10.0, 12.0],
        )
        self.assertEqual(
            plan["solve_contract"]["acceptance_grid"]["azimuths_deg"],
            [float(value) for value in range(0, 360, 5)],
        )
        self.assertEqual(
            plan["artifact_filenames"],
            {
                "clean_truth": "clean_truth.grim",
                "clean_prediction": "clean_prediction.grim",
                "featured_truth": "featured_truth.grim",
                "featured_prediction": "featured_prediction.grim",
            },
        )
        for body_id in ("wedge-ramp", "swept-wing-panel"):
            source = CASE_ROOT / plan["bodies"][body_id]["canonical_source_spec"]
            self.assertTrue(source.is_file(), source)

    def test_prepare_writes_specs_and_existing_schema_manifest_but_no_results(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = harness.prepare_run(PLAN_PATH, directory)
            root = Path(directory)
            manifest = json.loads(
                (root / "feature_cases.json").read_text(encoding="utf-8")
            )

            self.assertEqual(prepared["case_count"], 14)
            self.assertEqual(manifest["schema"], "ghost.validation.feature-cases.v1")
            self.assertEqual(len(manifest["cases"]), 14)
            for entry in manifest["cases"]:
                self.assertTrue(entry["id"])
                for role in harness.CASE_REQUIRED_PATHS:
                    self.assertEqual(Path(entry[role]).name, f"{role}.grim")
                case_id = Path(entry["clean_truth"]).parent.name
                spec_path = root / "cases" / case_id / "case_spec.json"
                self.assertTrue(spec_path.is_file())
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                self.assertEqual(spec["schema"], harness.CASE_SPEC_SCHEMA)
                self.assertEqual(spec["case_id"], case_id)
            self.assertEqual(list(root.rglob("*.grim")), [])

    def test_preflight_reports_every_missing_artifact_without_creating_one(self):
        with tempfile.TemporaryDirectory() as directory:
            harness.prepare_run(PLAN_PATH, directory)
            report = harness.preflight_run(PLAN_PATH, directory)

            self.assertFalse(report["passed"])
            self.assertEqual(report["case_count"], 14)
            failed = [
                artifact
                for case in report["cases"]
                for artifact in case["artifacts"].values()
                if not artifact["passed"]
            ]
            self.assertEqual(len(failed), 14 * 4)
            self.assertTrue(all(not artifact["exists"] for artifact in failed))
            self.assertTrue(all(
                artifact["errors"] == ["required artifact is missing"]
                for artifact in failed
            ))
            self.assertEqual(list(Path(directory).rglob("*.grim")), [])

    def test_preflight_accepts_exact_contract_and_rejects_off_grid_export(self):
        plan = harness.load_external_plan(PLAN_PATH)
        case = plan["cases"][0]
        with tempfile.TemporaryDirectory() as directory:
            harness.prepare_run(PLAN_PATH, directory)
            case_dir = Path(directory) / "cases" / case["id"]
            for role, filename in plan["artifact_filenames"].items():
                _write_structural_grim(case_dir / filename, plan["solve_contract"])

            report = harness.preflight_run(PLAN_PATH, directory)
            first = report["cases"][0]
            self.assertTrue(first["passed"])
            self.assertTrue(all(
                artifact["passed"] for artifact in first["artifacts"].values()
            ))

            wrong_path = case_dir / plan["artifact_filenames"]["featured_truth"]
            _write_structural_grim(
                wrong_path,
                plan["solve_contract"],
                frequencies=[8.0, 10.0],
            )
            report = harness.preflight_run(PLAN_PATH, directory)
            artifact = report["cases"][0]["artifacts"]["featured_truth"]
            self.assertFalse(artifact["passed"])
            self.assertTrue(any("frequencies" in error for error in artifact["errors"]))

            _write_structural_grim(
                wrong_path,
                plan["solve_contract"],
            )
            prediction = (
                case_dir / plan["artifact_filenames"]["featured_prediction"]
            )
            _write_structural_grim(
                prediction,
                plan["solve_contract"],
                include_feature_provenance=False,
            )
            report = harness.preflight_run(PLAN_PATH, directory)
            artifact = report["cases"][0]["artifacts"]["featured_prediction"]
            self.assertFalse(artifact["passed"])
            self.assertTrue(any(
                "dataset_content_sha256" in error for error in artifact["errors"]
            ))

    def test_duplicate_case_id_is_rejected(self):
        payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        payload["cases"].append(dict(payload["cases"][0]))
        for body in payload["bodies"].values():
            body.pop("canonical_source_spec", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                harness.load_external_plan(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
