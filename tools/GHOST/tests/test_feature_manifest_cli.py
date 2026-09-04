#!/usr/bin/env python3
"""Focused tests for the supported feature-manifest and headless workflows."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


GHOST = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GHOST / "Backend"))

import create_feature_manifest  # noqa: E402
import feature_workflow  # noqa: E402
import place_features  # noqa: E402


def _write_response(path: Path, marker: float = 1.0) -> None:
    with path.open("wb") as stream:
        np.savez(stream, response_marker=np.asarray([marker], dtype=float))


def _write_validation_report(
    response: Path, *, case_id: str = "full-wave-case-007"
) -> Path:
    paths = {}
    digests = {}
    for role in feature_workflow.FEATURE_VALIDATION_ARTIFACT_ROLES:
        artifact = response.parent / f"{case_id}-{role}.grim"
        artifact.write_bytes(f"{case_id}:{role}\n".encode("ascii"))
        paths[role] = str(artifact.resolve())
        digests[role] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report = response.parent / f"{case_id}-validation-report.json"
    gate_limits = {
        "active_floor_db": -40.0,
        **feature_workflow.FEATURE_VALIDATION_RELEASE_CEILINGS,
    }
    passed_section = {
        "passed": True,
        "gate_limits": gate_limits,
        "gates": {
            "normalized_complex_rms": True,
            "magnitude_error_p95_db": True,
            "phase_error_rms_deg": True,
            "complex_coherence": True,
            "every_polarization_channel": True,
        },
    }
    report.write_text(json.dumps({
        "schema": "ghost.validation.feature-case-report.v1",
        "passed": True,
        "comparisons": [{
            "case_id": case_id,
            "passed": True,
            "paths": paths,
            "artifact_sha256": digests,
            "feature_response_content_sha256": [
                feature_workflow.feature_response_content_sha256(response)
            ],
            "clean_baseline": passed_section,
            "featured_total": passed_section,
            "isolated_feature_delta": passed_section,
        }],
    }), encoding="utf-8")
    return report


def _create_args(response: Path, kind: str) -> list[str]:
    report = _write_validation_report(response)
    args = [
        "create", str(response),
        "--dataset-id", "door_seam" if kind == "line" else "fastener",
        "--feature-kind", kind,
        "--host-material", "primer + topcoat stack v3",
        "--frequency-min-ghz", "1.0",
        "--frequency-max-ghz", "12.0",
        "--footprint-radius-m", "0.02",
        "--validation-status", "validated",
        "--validation-case-id", "full-wave-case-007",
        "--validation-report", str(report),
        "--attest-reviewed-evidence",
    ]
    if kind == "line":
        args.extend([
            "--minimum-along-line-normal-turn-radius-m", "0.5",
            "--maximum-conical-incidence-deg", "25.0",
            "--maximum-path-vertex-turn-deg", "35.0",
            "--phase-calibration-case-id", "full-wave-case-007",
        ])
    return args


class FeatureManifestCliTests(unittest.TestCase):
    def test_validated_manifest_refuses_unbound_or_changed_full_wave_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = root / "fastener.grim"
            _write_response(response)
            args = _create_args(response, "point")
            report = Path(args[args.index("--validation-report") + 1])
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["comparisons"][0][
                "feature_response_content_sha256"
            ] = ["9" * 64]
            report.write_text(json.dumps(payload), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main(args)
            self.assertEqual(raised.exception.code, 2)

            args = _create_args(response, "point")
            report = Path(args[args.index("--validation-report") + 1])
            payload = json.loads(report.read_text(encoding="utf-8"))
            for section in (
                "clean_baseline", "featured_total", "isolated_feature_delta"
            ):
                payload["comparisons"][0][section]["gate_limits"][
                    "active_floor_db"
                ] = 0.0
            report.write_text(json.dumps(payload), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main(args)
            self.assertEqual(raised.exception.code, 2)

            args = _create_args(response, "point")
            report = Path(args[args.index("--validation-report") + 1])
            payload = json.loads(report.read_text(encoding="utf-8"))
            for section in (
                "clean_baseline", "featured_total", "isolated_feature_delta"
            ):
                payload["comparisons"][0][section]["gate_limits"][
                    "max_normalized_rms"
                ] = 10.0
            report.write_text(json.dumps(payload), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main(args)
            self.assertEqual(raised.exception.code, 2)

            args = _create_args(response, "point")
            report = Path(args[args.index("--validation-report") + 1])
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["comparisons"][0][
                "feature_response_content_sha256"
            ].append("8" * 64)
            report.write_text(json.dumps(payload), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main(args)
            self.assertEqual(raised.exception.code, 2)

            args = _create_args(response, "point")
            report = Path(args[args.index("--validation-report") + 1])
            payload = json.loads(report.read_text(encoding="utf-8"))
            changed = Path(payload["comparisons"][0]["paths"]["featured_truth"])
            changed.write_bytes(b"changed after validation")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main(args)
            self.assertEqual(raised.exception.code, 2)

    def test_create_refuses_raw_opn_frd_response_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in (
                "FASTENER-00-01_0.010gap_OPN.grim",
                "FASTENER-00-01_0.010gap_fRd.GrIm",
            ):
                response = root / filename
                _write_response(response)
                with self.subTest(filename=filename):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            create_feature_manifest.main(
                                _create_args(response, "point")
                            )
                    self.assertEqual(raised.exception.code, 2)
                    self.assertFalse(
                        Path(str(response) + ".feature.json").exists()
                    )

    def test_create_and_check_line_manifest_with_solver_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "door_seam.grim"
            _write_response(response)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    create_feature_manifest.main(_create_args(response, "line")),
                    0,
                )
            sidecar = Path(str(response) + ".feature.json")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["response_content_sha256"],
                feature_workflow.feature_response_content_sha256(response),
            )
            self.assertEqual(
                manifest["applicability"]["maximum_path_vertex_turn_deg"],
                35.0,
            )
            self.assertEqual(
                manifest["line_phase_calibration"]["grazing_taper_deg"],
                10.0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(create_feature_manifest.main([
                    "check", str(response),
                    "--dataset-id", "door_seam",
                    "--feature-kind", "line",
                ]), 0)

    def test_create_requires_explicit_team_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "fastener.grim"
            _write_response(response)
            args = _create_args(response, "point")
            args.remove("--attest-reviewed-evidence")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(Path(str(response) + ".feature.json").exists())

    def test_check_rejects_response_changed_after_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "fastener.grim"
            _write_response(response)
            with contextlib.redirect_stdout(io.StringIO()):
                create_feature_manifest.main(_create_args(response, "point"))
            _write_response(response, marker=2.0)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main([
                        "check", str(response),
                        "--dataset-id", "fastener",
                        "--feature-kind", "point",
                    ])
            self.assertEqual(raised.exception.code, 2)

    def test_check_rejects_line_manifest_without_fixed_taper_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "door_seam.grim"
            _write_response(response)
            with contextlib.redirect_stdout(io.StringIO()):
                create_feature_manifest.main(_create_args(response, "line"))
            sidecar = Path(str(response) + ".feature.json")
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            del manifest["line_phase_calibration"]["grazing_taper_deg"]
            sidecar.write_text(json.dumps(manifest), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main([
                        "check", str(response),
                        "--dataset-id", "door_seam",
                        "--feature-kind", "line",
                    ])
            self.assertEqual(raised.exception.code, 2)


class SurfaceBindingCliTests(unittest.TestCase):
    def _create_args(self, base: Path, surface: Path) -> list[str]:
        return [
            "create-surface-binding",
            str(base),
            str(surface),
            "--surface-units", "in",
            "--geometry-id", "vehicle-door-mesh-r7",
            "--attestation-case-id", "solver-registration-042",
            "--attest-reviewed-registration",
        ]

    def test_create_and_check_canonical_exact_surface_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean_vehicle.grim"
            surface = root / "vehicle.stl"
            _write_response(base)
            surface.write_bytes(b"solid exact-surface\nendsolid exact-surface\n")

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    create_feature_manifest.main(self._create_args(base, surface)),
                    0,
                )
            sidecar = Path(str(surface) + ".assembly.json")
            binding = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                binding["schema"], "ghost.assembly-surface-binding.v1"
            )
            self.assertEqual(binding["surface_units"], "inches")
            self.assertEqual(
                binding["frame_convention"],
                "CAD:+y=nose;+x=right;+z=up",
            )
            self.assertEqual(binding["geometry_id"], "vehicle-door-mesh-r7")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(create_feature_manifest.main([
                    "check-surface-binding",
                    str(base),
                    str(surface),
                    "--surface-units", "inches",
                    "--geometry-id", "vehicle-door-mesh-r7",
                    "--attestation-case-id", "solver-registration-042",
                ]), 0)

    def test_surface_binding_rejects_changed_geometry_and_wrong_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean_vehicle.grim"
            surface = root / "vehicle.facet"
            _write_response(base)
            surface.write_bytes(b"4 2\noriginal mesh\n")
            with contextlib.redirect_stdout(io.StringIO()):
                create_feature_manifest.main(self._create_args(base, surface))

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main([
                        "check-surface-binding", str(base), str(surface),
                        "--surface-units", "meters",
                    ])
            self.assertEqual(raised.exception.code, 2)

            surface.write_bytes(b"4 2\nchanged mesh\n")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_feature_manifest.main([
                        "check-surface-binding", str(base), str(surface),
                        "--surface-units", "inches",
                    ])
            self.assertEqual(raised.exception.code, 2)

    def test_surface_binding_loader_accepts_utf8_bom_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean_vehicle.grim"
            surface = root / "vehicle.stl"
            _write_response(base)
            surface.write_bytes(b"solid exact-surface\nendsolid exact-surface\n")
            with contextlib.redirect_stdout(io.StringIO()):
                create_feature_manifest.main(self._create_args(base, surface))

            sidecar = feature_workflow.surface_binding_path(surface)
            sidecar.write_bytes(b"\xef\xbb\xbf" + sidecar.read_bytes())
            binding, loaded_path, sidecar_digest = (
                feature_workflow.load_surface_binding(
                    base,
                    surface,
                    base_grim_sha256=feature_workflow.sha256_file(str(base)),
                    surface_sha256=feature_workflow.sha256_file(str(surface)),
                    surface_units="inches",
                )
            )
            self.assertEqual(loaded_path, sidecar)
            self.assertEqual(binding["geometry_id"], "vehicle-door-mesh-r7")
            self.assertEqual(
                sidecar_digest,
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )


class HeadlessPlacementPolicyTests(unittest.TestCase):
    def _run_main(self, *, warnings=(), acknowledged_sha=None, profile="production"):
        plan = SimpleNamespace(
            validation_warnings=tuple(warnings),
            occluder=None,
            prepared_plan_sha256="a" * 64,
        )
        prepare = mock.Mock(return_value=plan)
        execute = mock.Mock(return_value="combined.grim")
        patches = dict(
            LINE_FEATURE_LOCATIONS_CSV="line.csv",
            POINT_FEATURE_LOCATIONS_CSV=None,
            LINE_FEATURE_DATASETS={"door_seam": "door.grim"},
            VALIDATION_PROFILE=profile,
            ACKNOWLEDGED_PLAN_SHA256=acknowledged_sha,
            prepare_feature_assembly=prepare,
            execute_feature_assembly=execute,
        )
        with mock.patch.multiple(place_features, **patches):
            with contextlib.redirect_stdout(io.StringIO()):
                result = place_features.main()
        return result, prepare, execute

    def test_default_profile_is_advisory_without_host_ids(self):
        self.assertEqual(place_features.VALIDATION_PROFILE, "advisory")
        _result, prepare, execute = self._run_main(profile="advisory", warnings=("missing metadata",))
        request = prepare.call_args.args[0]
        self.assertTrue(request.allow_legacy_base_metadata)
        self.assertFalse(request.require_feature_manifests)
        self.assertFalse(request.require_body_mesh_certification)
        self.assertFalse(hasattr(request, "expected_host_material"))
        self.assertFalse(hasattr(request, "expected_host_materials"))
        execute.assert_called_once()
        self.assertIsNone(execute.call_args.kwargs["acknowledged_plan_sha256"])

    def test_legacy_profile_is_explicit(self):
        _result, prepare, _execute = self._run_main(profile="legacy")
        request = prepare.call_args.args[0]
        self.assertTrue(request.allow_legacy_base_metadata)
        self.assertFalse(request.require_feature_manifests)
        self.assertFalse(request.require_body_mesh_certification)

    def test_external_profile_keeps_strict_feature_contracts_with_body_waiver(self):
        _result, prepare, _execute = self._run_main(profile="external")
        request = prepare.call_args.args[0]
        self.assertFalse(request.allow_legacy_base_metadata)
        self.assertTrue(request.require_feature_manifests)
        self.assertFalse(request.require_body_mesh_certification)

    def test_warning_plan_is_not_executed_without_exact_acknowledgement(self):
        with self.assertRaisesRegex(SystemExit, "not published"):
            self._run_main(warnings=("review this applicability warning",))

    def test_reviewed_warning_plan_executes_after_acknowledgement(self):
        _result, _prepare, execute = self._run_main(
            warnings=("reviewed warning",), acknowledged_sha="a" * 64
        )
        execute.assert_called_once()

    def test_changed_plan_invalidates_prior_warning_acknowledgement(self):
        with self.assertRaisesRegex(SystemExit, "exact digest"):
            self._run_main(
                warnings=("reviewed warning",), acknowledged_sha="b" * 64
            )


if __name__ == "__main__":
    unittest.main()
