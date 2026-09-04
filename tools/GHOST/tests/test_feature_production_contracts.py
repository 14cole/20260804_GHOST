#!/usr/bin/env python3
"""Production safety contracts for reusable Assembly feature libraries."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))
sys.path.append(str(REPO.parent.parent / "GRIM_Revised_2"))

import feature_sum  # noqa: E402
import feature_workflow  # noqa: E402
from grim_dataset import RcsGrid  # noqa: E402


def _write_response_placeholder(path):
    with Path(path).open("wb") as stream:
        np.savez(stream, response_marker=np.asarray([1.0, 2.0, 3.0]))
    return feature_workflow.feature_response_content_sha256(path)


def manifest(
    dataset_id,
    kind,
    *,
    conical=90.0,
    path_turn=180.0,
    footprint=0.1,
    response_content_sha256="a" * 64,
):
    applicability = {
        "frequency_ghz": {"min": 1.0, "max": 10.0},
        "footprint_radius_m": footprint,
    }
    if kind == "line":
        applicability["minimum_along_line_normal_turn_radius_m"] = 0.0
        applicability["maximum_conical_incidence_deg"] = conical
        applicability["maximum_path_vertex_turn_deg"] = path_turn
    result = {
        "schema": feature_workflow.FEATURE_LIBRARY_MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "feature_kind": kind,
        "subtraction_order": "featured_minus_clean",
        "phase_origin": feature_workflow._FEATURE_PHASE_ORIGINS[kind],
        "frame_convention": feature_workflow._FEATURE_FRAME_CONVENTIONS[kind],
        "time_convention": "exp(+jwt)",
        "response_content_sha256": response_content_sha256,
        "host": {"material": "PEC outer skin"},
        "applicability": applicability,
        "validation": {
            "status": "validated",
            "case_ids": ["fixture-001"],
            "evidence": [{
                "schema": feature_workflow.FEATURE_VALIDATION_EVIDENCE_SCHEMA,
                "case_id": "fixture-001",
                "passed": True,
                "report_sha256": "b" * 64,
                "comparison_sha256": "c" * 64,
                "feature_response_content_sha256": response_content_sha256,
                "artifact_sha256": {
                    role: character * 64
                    for role, character in zip(
                        feature_workflow.FEATURE_VALIDATION_ARTIFACT_ROLES,
                        "def0",
                    )
                },
                "gate_limits": {
                    "active_floor_db": -40.0,
                    **feature_workflow.FEATURE_VALIDATION_RELEASE_CEILINGS,
                },
            }],
        },
    }
    if kind == "line":
        result["line_phase_calibration"] = {
            "schema": feature_workflow.LINE_PHASE_CALIBRATION_SCHEMA,
            "tm_deg": feature_workflow.PSI_HH_DEG,
            "te_deg": feature_workflow.PSI_VV_DEG,
            "grazing_taper_deg": feature_workflow.GRAZING_TAPER_DEG,
            "case_ids": ["fixture-001"],
        }
    return result


def _point_pattern_stub():
    return SimpleNamespace(
        azimuths=np.asarray([0.0, 120.0, 240.0, 360.0]),
        frequencies=np.asarray([1.0, 10.0]),
        elevations=np.asarray([-90.0, 90.0]),
        amplitude=np.zeros((4, 2, 2, 3), dtype=np.complex128),
        channel_indices={"VV": 0, "HH": 1, "VH": 2},
    )


class FeatureManifestTests(unittest.TestCase):
    def test_current_validated_manifest_requires_response_bound_evidence(self):
        missing = manifest("fastener", "point")
        missing["validation"].pop("evidence")
        with self.assertRaisesRegex(ValueError, "validation.evidence"):
            feature_workflow.validate_feature_library_manifest(
                missing, dataset_id="fastener", feature_kind="point"
            )

        wrong_response = manifest("fastener", "point")
        wrong_response["validation"]["evidence"][0][
            "feature_response_content_sha256"
        ] = "9" * 64
        with self.assertRaisesRegex(ValueError, "not this manifest response"):
            feature_workflow.validate_feature_library_manifest(
                wrong_response, dataset_id="fastener", feature_kind="point"
            )

        peak_only = manifest("fastener", "point")
        peak_only["validation"]["evidence"][0]["gate_limits"][
            "active_floor_db"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "excludes more weak-field samples"):
            feature_workflow.validate_feature_library_manifest(
                peak_only, dataset_id="fastener", feature_kind="point"
            )

    def test_v2_manifest_remains_legacy_readable_without_evidence(self):
        previous = manifest("fastener", "point")
        previous["schema"] = (
            feature_workflow.PREVIOUS_FEATURE_LIBRARY_MANIFEST_SCHEMA
        )
        previous["validation"].pop("evidence")
        normalized = feature_workflow.validate_feature_library_manifest(
            previous, dataset_id="fastener", feature_kind="point"
        )
        self.assertEqual(
            normalized["schema"],
            feature_workflow.PREVIOUS_FEATURE_LIBRARY_MANIFEST_SCHEMA,
        )

    def test_manifest_requires_canonical_identity_and_conventions(self):
        validated = feature_workflow.validate_feature_library_manifest(
            manifest("gap", "line"), dataset_id="gap", feature_kind="line"
        )
        self.assertEqual(validated["subtraction_order"], "featured_minus_clean")
        self.assertEqual(
            validated["applicability"]["maximum_conical_incidence_deg"], 90.0
        )
        self.assertEqual(
            validated["applicability"]["maximum_path_vertex_turn_deg"], 180.0
        )
        self.assertEqual(
            validated["line_phase_calibration"]["grazing_taper_deg"],
            feature_workflow.GRAZING_TAPER_DEG,
        )

        bad = manifest("gap", "line")
        bad["subtraction_order"] = "clean_minus_featured"
        with self.assertRaisesRegex(ValueError, "featured_minus_clean"):
            feature_workflow.validate_feature_library_manifest(
                bad, dataset_id="gap", feature_kind="line"
            )

    def test_v1_line_manifest_is_legacy_readable_but_production_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "legacy-gap.grim"
            response_digest = _write_response_placeholder(dataset)
            legacy = manifest(
                "gap",
                "line",
                response_content_sha256=response_digest,
            )
            legacy["schema"] = (
                feature_workflow.LEGACY_FEATURE_LIBRARY_MANIFEST_SCHEMA
            )
            legacy["applicability"].pop("maximum_path_vertex_turn_deg")
            legacy["line_phase_calibration"]["schema"] = (
                feature_workflow.LEGACY_LINE_PHASE_CALIBRATION_SCHEMA
            )
            legacy["line_phase_calibration"].pop("grazing_taper_deg")

            normalized = feature_workflow.validate_feature_library_manifest(
                legacy, dataset_id="gap", feature_kind="line"
            )
            self.assertEqual(
                normalized["schema"],
                feature_workflow.LEGACY_FEATURE_LIBRARY_MANIFEST_SCHEMA,
            )
            self.assertEqual(
                normalized["applicability"]["maximum_path_vertex_turn_deg"],
                180.0,
            )
            self.assertEqual(
                normalized["line_phase_calibration"]["grazing_taper_deg"],
                feature_workflow.GRAZING_TAPER_DEG,
            )

            dataset.with_suffix(".feature.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            tangent = np.asarray([0.0, 1.0, 0.0])
            normal = np.asarray([0.0, 0.0, 1.0])
            placement = {
                "perimeter": np.asarray([[np.zeros(3), tangent]]),
                "segment_normals": np.asarray([[normal, normal]]),
            }
            record = {
                "schema": feature_workflow.LINE_PLACEMENT_SCHEMA,
                "kind": "line_2d_delta",
                "dataset": str(dataset),
                "dataset_sha256": feature_workflow.sha256_file(str(dataset)),
                "dataset_id": "gap",
                "line_id": "gap-1",
            }
            kwargs = dict(
                line_placements=[placement],
                line_records=[record],
                point_placements=[],
                point_records=[],
                radar_grid={
                    "frequencies_ghz": [1.0],
                    "azimuths_deg": [0.0],
                    "elevations_deg": [0.0],
                    "axis_az_deg": 0.0,
                    "axis_el_deg": 0.0,
                    "roll_deg": 0.0,
                },
            )
            with (
                mock.patch.object(
                    feature_workflow,
                    "_load_grim",
                    return_value={"frequencies": np.asarray([1.0, 10.0])},
                ),
                mock.patch.object(
                    feature_workflow,
                    "load_seam_from_grim",
                    return_value=feature_sum.SeamCoefficients(
                        1.0,
                        np.asarray([0.0, 90.0, 180.0]),
                        np.ones(3, dtype=np.complex128),
                        np.ones(3, dtype=np.complex128),
                    ),
                ),
            ):
                _contracts, warnings, _hashes, _absent = (
                    feature_workflow._apply_feature_library_contracts(
                        **kwargs, require_manifests=False
                    )
                )
                self.assertTrue(
                    any("annotations are advisory" in value for value in warnings)
                )
                with self.assertRaisesRegex(ValueError, "Production requires"):
                    feature_workflow._apply_feature_library_contracts(
                        **kwargs, require_manifests=True
                    )

    def test_embedded_and_sidecar_manifest_must_not_disagree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "gap.grim"
            response_digest = _write_response_placeholder(dataset)
            with dataset.open("wb") as stream:
                np.savez(
                    stream,
                    response_marker=np.asarray([1.0, 2.0, 3.0]),
                    feature_library_manifest_json=np.asarray(json.dumps(
                        manifest(
                            "gap",
                            "line",
                            response_content_sha256=response_digest,
                        )
                    )),
                )
            sidecar = dataset.with_suffix(".feature.json")
            conflicting = manifest(
                "gap", "line", response_content_sha256=response_digest
            )
            conflicting["host"]["material"] = "different stack"
            sidecar.write_text(json.dumps(conflicting), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifests disagree"):
                feature_workflow.load_feature_library_manifest(
                    dataset, dataset_id="gap", feature_kind="line"
                )

    def test_manifest_footprints_warn_or_gate_cluster_coupling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "fastener.grim"
            response_digest = _write_response_placeholder(dataset)
            dataset.with_suffix(".feature.json").write_text(
                json.dumps(manifest(
                    "fastener",
                    "point",
                    footprint=0.1,
                    response_content_sha256=response_digest,
                )),
                encoding="utf-8",
            )
            digest = feature_workflow.sha256_file(str(dataset))
            placements = [
                {
                    "pattern": _point_pattern_stub(),
                    "location": np.asarray([0.0, 0.0, 0.0]),
                    "aperture_normal": np.asarray([0.0, 0.0, 1.0]),
                    "roll_ref": np.asarray([1.0, 0.0, 0.0]),
                },
                {
                    "pattern": _point_pattern_stub(),
                    "location": np.asarray([0.15, 0.0, 0.0]),
                    "aperture_normal": np.asarray([0.0, 0.0, 1.0]),
                    "roll_ref": np.asarray([1.0, 0.0, 0.0]),
                },
            ]
            records = [
                {
                    "schema": feature_workflow.POINT_PLACEMENT_SCHEMA,
                    "kind": "compact_3d_delta",
                    "dataset": str(dataset),
                    "dataset_sha256": digest,
                    "dataset_id": "fastener",
                    "placement_id": identity,
                }
                for identity in ("p1", "p2")
            ]
            kwargs = dict(
                host_material="PEC outer skin",
                line_placements=[],
                line_records=[],
                point_placements=placements,
                point_records=records,
                radar_grid={
                    "frequencies_ghz": [1.0],
                    "azimuths_deg": [0.0],
                    "elevations_deg": [0.0],
                    "axis_az_deg": 0.0,
                    "axis_el_deg": 0.0,
                    "roll_deg": 0.0,
                },
            )
            _contracts, warnings, _hashes, _absent = (
                feature_workflow._apply_feature_library_contracts(
                    **kwargs,
                    require_manifests=False,
                )
            )
            self.assertTrue(any("annotations are advisory" in value for value in warnings))
            self.assertEqual(_contracts["point:fastener"]["source_manifest"]["applicability"]["footprint_radius_m"], .1)
            with self.assertRaisesRegex(ValueError, "footprints overlap"):
                feature_workflow._apply_feature_library_contracts(
                    **kwargs,
                    require_manifests=True,
                )

    def test_current_manifests_require_host_and_production_matches_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "fastener.grim"
            response_digest = _write_response_placeholder(dataset)
            definition = manifest(
                "fastener", "point", response_content_sha256=response_digest
            )
            placement = {
                "pattern": _point_pattern_stub(),
                "location": np.zeros(3),
                "aperture_normal": np.asarray([0.0, 0.0, 1.0]),
                "roll_ref": np.asarray([1.0, 0.0, 0.0]),
            }
            record = {
                "schema": feature_workflow.POINT_PLACEMENT_SCHEMA,
                "kind": "compact_3d_delta",
                "dataset": str(dataset),
                "dataset_sha256": feature_workflow.sha256_file(str(dataset)),
                "dataset_id": "fastener", "placement_id": "p1",
            }
            for host in (None, {}, {"material": "PEC outer skin"}):
                definition.pop("host", None)
                if host is not None:
                    definition["host"] = host
                if not host:
                    with self.assertRaisesRegex(ValueError, "host.material"):
                        feature_workflow.validate_feature_library_manifest(definition, dataset_id="fastener", feature_kind="point")
                    continue
                dataset.with_suffix(".feature.json").write_text(
                    json.dumps(definition), encoding="utf-8"
                )
                for strict in (False, True):
                    with self.subTest(host=host, strict=strict):
                        contracts, warnings, _hashes, _absent = (
                            feature_workflow._apply_feature_library_contracts(
                                line_placements=[], line_records=[],
                                point_placements=[placement], point_records=[dict(record)],
                                radar_grid={
                                    "frequencies_ghz": [1.0], "azimuths_deg": [0.0],
                                    "elevations_deg": [0.0], "axis_az_deg": 0.0,
                                    "axis_el_deg": 0.0, "roll_deg": 0.0,
                                },
                                require_manifests=strict,
                                host_material="PEC outer skin" if strict else "",
                            )
                        )
                        self.assertIn("point:fastener", contracts)
                        self.assertEqual(any("annotations are advisory" in w for w in warnings), not strict)

    def test_line_manifest_gates_uncertified_conical_incidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "gap.grim"
            response_digest = _write_response_placeholder(dataset)
            dataset.with_suffix(".feature.json").write_text(
                json.dumps(manifest(
                    "gap",
                    "line",
                    conical=30.0,
                    response_content_sha256=response_digest,
                )),
                encoding="utf-8",
            )
            tangent = np.asarray([2.0 ** -0.5, 0.0, 2.0 ** -0.5])
            normal = np.asarray([-2.0 ** -0.5, 0.0, 2.0 ** -0.5])
            placement = {
                "perimeter": np.asarray([[np.zeros(3), tangent]]),
                "segment_normals": np.asarray([[normal, normal]]),
            }
            record = {
                "schema": feature_workflow.LINE_PLACEMENT_SCHEMA,
                "kind": "line_2d_delta",
                "dataset": str(dataset),
                "dataset_sha256": feature_workflow.sha256_file(str(dataset)),
                "dataset_id": "gap",
                "line_id": "gap-1",
            }
            with (
                mock.patch.object(
                    feature_workflow,
                    "_load_grim",
                    return_value={"frequencies": np.asarray([1.0, 10.0])},
                ),
                mock.patch.object(
                    feature_workflow,
                    "load_seam_from_grim",
                    return_value=feature_sum.SeamCoefficients(
                        1.0,
                        np.asarray([0.0, 90.0, 180.0]),
                        np.ones(3, dtype=np.complex128),
                        np.ones(3, dtype=np.complex128),
                    ),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "conical incidence"):
                    feature_workflow._apply_feature_library_contracts(
                        host_material="PEC outer skin",
                        line_placements=[placement],
                        line_records=[record],
                        point_placements=[],
                        point_records=[],
                        radar_grid={
                            "frequencies_ghz": [1.0],
                            "azimuths_deg": [0.0],
                            "elevations_deg": [0.0],
                            "axis_az_deg": 0.0,
                            "axis_el_deg": 0.0,
                            "roll_deg": 0.0,
                        },
                        require_manifests=True,
                    )

    def test_manifest_is_bound_to_exact_response_content(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "fastener.grim"
            digest = _write_response_placeholder(dataset)
            dataset.with_suffix(".feature.json").write_text(
                json.dumps(manifest(
                    "fastener",
                    "point",
                    response_content_sha256=digest,
                )),
                encoding="utf-8",
            )
            loaded, _sources = feature_workflow.load_feature_library_manifest(
                dataset, dataset_id="fastener", feature_kind="point"
            )
            self.assertEqual(loaded["response_content_sha256"], digest)

            with dataset.open("wb") as stream:
                np.savez(stream, response_marker=np.asarray([9.0, 8.0, 7.0]))
            with self.assertRaisesRegex(ValueError, "bound to response content"):
                feature_workflow.load_feature_library_manifest(
                    dataset, dataset_id="fastener", feature_kind="point"
                )

    def test_malformed_embedded_manifest_never_fails_open(self):
        invalid_values = (
            np.asarray(b"\xff"),
            np.asarray("{"),
            np.asarray({"not": "loadable without pickle"}, dtype=object),
        )
        for with_sidecar in (False, True):
            for invalid in invalid_values:
                with self.subTest(
                    with_sidecar=with_sidecar, dtype=str(invalid.dtype)
                ), tempfile.TemporaryDirectory() as directory:
                    dataset = Path(directory) / "fastener.grim"
                    with dataset.open("wb") as stream:
                        np.savez(
                            stream,
                            response_marker=np.asarray([1.0, 2.0, 3.0]),
                            feature_library_manifest_json=invalid,
                        )
                    if with_sidecar:
                        digest = feature_workflow.feature_response_content_sha256(
                            dataset
                        )
                        dataset.with_suffix(".feature.json").write_text(
                            json.dumps(manifest(
                                "fastener",
                                "point",
                                response_content_sha256=digest,
                            )),
                            encoding="utf-8",
                        )
                    with self.assertRaisesRegex(
                        ValueError, "embedded feature-library manifest"
                    ):
                        feature_workflow.load_feature_library_manifest(
                            dataset,
                            dataset_id="fastener",
                            feature_kind="point",
                        )

    def test_physics_identity_ignores_history_and_archive_repacking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.grim"
            second = root / "second.grim"
            grid = {
                "frequencies_ghz": [1.0],
                "azimuths_deg": [0.0],
                "elevations_deg": [0.0],
                "axis_az_deg": 0.0,
                "axis_el_deg": 0.0,
                "roll_deg": 0.0,
            }
            feature_sum.export_radar_grim(
                str(first), bor_result=None, placements=[], history="first", **grid
            )
            with np.load(first, allow_pickle=False) as stored:
                payload = {
                    key: np.array(stored[key], copy=True)
                    for key in reversed(stored.files)
                }
            payload["history"] = np.asarray("different benign provenance")
            payload["source_path"] = np.asarray("elsewhere/source.grim")
            with second.open("wb") as stream:
                np.savez_compressed(stream, **payload)

            self.assertNotEqual(
                feature_workflow.feature_response_content_sha256(first),
                feature_workflow.feature_response_content_sha256(second),
            )
            self.assertEqual(
                feature_workflow.feature_response_physics_sha256(first),
                feature_workflow.feature_response_physics_sha256(second),
            )

    def test_grim_round_trip_preserves_embedded_feature_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.grim"
            output = Path(directory) / "roundtrip.grim"
            grid = {
                "frequencies_ghz": [1.0],
                "azimuths_deg": [0.0],
                "elevations_deg": [0.0],
                "axis_az_deg": 0.0,
                "axis_el_deg": 0.0,
                "roll_deg": 0.0,
            }
            feature_sum.export_radar_grim(
                str(source), bor_result=None, placements=[], **grid
            )
            payload = feature_sum._load_grim(str(source))
            marker = '{"schema":"test.manifest"}'
            payload[feature_workflow.FEATURE_LIBRARY_MANIFEST_KEY] = np.asarray(
                marker
            )
            import grim_io
            grim_io._save_grim_npz(payload, str(output))
            with np.load(output, allow_pickle=False) as stored:
                self.assertEqual(
                    str(stored[feature_workflow.FEATURE_LIBRARY_MANIFEST_KEY]),
                    marker,
                )

    def test_output_cannot_alias_any_selected_geometry_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            base.touch()
            for field_name, label in (
                ("surface_mesh", "surface mesh"),
                ("point_locations_csv", "point-placement CSV"),
                ("line_locations_csv", "line-placement CSV"),
            ):
                with self.subTest(field=field_name):
                    selected = root / f"selected-{field_name}.grim"
                    selected.touch()
                    request = feature_workflow.FeatureAssemblyRequest(
                        base_grim=base,
                        output_grim=selected,
                        **{field_name: selected},
                    )
                    with self.assertRaisesRegex(ValueError, label):
                        feature_workflow._reject_output_aliases(
                            request,
                            base=base.resolve(),
                            output=selected.resolve(),
                        )


class BaseContractTests(unittest.TestCase):
    def _base(self, root):
        path = root / "base.grim"
        grid = {
            "frequencies_ghz": [1.0],
            "azimuths_deg": [0.0],
            "elevations_deg": [0.0],
            "axis_az_deg": 0.0,
            "axis_el_deg": 0.0,
            "roll_deg": 0.0,
        }
        feature_sum.export_radar_grim(
            str(path), bor_result=None, placements=[], **grid
        )
        return path, grid, feature_sum._load_grim(str(path))

    def test_sentri_theta_mapping_fails_closed_until_explicitly_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, grid, payload = self._base(Path(directory))
            payload.pop("assembly_angular_coordinate_contract", None)
            payload["source_format"] = np.asarray("SENTRi descriptive Hz RCS table")
            payload["sentri_coordinate_mapping"] = np.asarray(
                "elevation=theta; azimuth=wrapped phi"
            )
            with self.assertRaisesRegex(ValueError, "unconverted SENTRi"):
                feature_sum.validate_assembly_base_grid_metadata(
                    payload, grid, str(path)
                )
            payload["assembly_angular_coordinate_contract"] = np.asarray(
                feature_sum.ASSEMBLY_RADAR_ANGULAR_CONTRACT
            )
            report = feature_sum.validate_assembly_base_grid_metadata(
                payload, grid, str(path)
            )
            self.assertEqual(
                report["angular_contract"],
                feature_sum.ASSEMBLY_RADAR_ANGULAR_CONTRACT,
            )

    def test_sentri_phase_wrap_cannot_launder_native_theta_metadata(self):
        native = RcsGrid(
            [0.0],
            [90.0],
            [1.0],
            ["VV"],
            rcs=np.ones((1, 1, 1, 1), dtype=np.complex128),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "conic",
                "elevation_coordinate_convention": "sentri_theta_top_zero",
            },
            extra={
                "source_format": "SENTRi descriptive Hz RCS table",
                "sentri_coordinate_mapping": (
                    "elevation=theta; azimuth=wrapped phi"
                ),
                "sentri_elevation_convention": "sentri_theta_top_zero",
            },
        )
        wrapped = native.wrap_phase("0_360")
        self.assertEqual(
            wrapped.extra["sentri_elevation_convention"],
            "sentri_theta_top_zero",
        )
        self.assertIn("elevation=theta", wrapped.extra["sentri_coordinate_mapping"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(wrapped.save(Path(directory) / "native-theta.grim"))
            payload = feature_sum._load_grim(str(path))

        # Prove the modeled units tag is independently fail-closed even if a
        # legacy producer loses every passthrough SENTRi marker.
        for key in tuple(payload):
            if key == "source_format" or key.startswith("sentri_"):
                payload.pop(key, None)
        with self.assertRaisesRegex(ValueError, "unconverted SENTRi polar theta"):
            feature_sum.validate_assembly_base_grid_metadata(
                payload,
                {
                    "frequencies_ghz": [1.0],
                    "azimuths_deg": [0.0],
                    "elevations_deg": [90.0],
                    "axis_az_deg": 0.0,
                    "axis_el_deg": 0.0,
                    "roll_deg": 0.0,
                },
                str(path),
            )

    def test_saved_sentri_conversion_is_a_canonical_assembly_base(self):
        native = RcsGrid(
            [0.0],
            [0.0, 90.0, 180.0],
            [1.0],
            ["VV"],
            rcs=np.ones((1, 3, 1, 1), dtype=np.complex128),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "conic",
                "elevation_coordinate_convention": "sentri_theta_top_zero",
            },
            extra={
                "source_format": "SENTRi descriptive Hz RCS table",
                "sentri_coordinate_mapping": (
                    "elevation=theta; azimuth=wrapped phi"
                ),
                "sentri_elevation_convention": "sentri_theta_top_zero",
            },
        )
        converted = native.convert_sentri_elevation_to_grim()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(converted.save(Path(directory) / "converted.grim"))
            payload = feature_sum._load_grim(str(path))
            report = feature_sum.validate_assembly_base_grid_metadata(
                payload,
                {
                    "frequencies_ghz": converted.frequencies.tolist(),
                    "azimuths_deg": converted.azimuths.tolist(),
                    "elevations_deg": converted.elevations.tolist(),
                    "axis_az_deg": 0.0,
                    "axis_el_deg": 0.0,
                    "roll_deg": 0.0,
                },
                str(path),
                allow_legacy_metadata=False,
            )

        self.assertEqual(report["status"], "canonical")
        self.assertEqual(
            report["angular_contract"],
            feature_sum.ASSEMBLY_RADAR_ANGULAR_CONTRACT,
        )

    def test_declared_base_rejects_contradictions_and_can_gate_missing_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            _path, _grid, payload = self._base(Path(directory))
            payload["phase_reference"] = np.asarray("opposite time sign")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                feature_sum._validate_declared_coherent_base(payload, "base", allow_legacy_metadata=False)
            accepted = feature_sum._validate_declared_coherent_base(payload, "base")
            np.testing.assert_array_equal(accepted["_amp"], payload["_amp"])
            self.assertIn("opposite time sign", str(accepted["metadata_advisories_json"]))

            payload.pop("phase_reference")
            payload.pop("amplitude_convention")
            with self.assertRaisesRegex(ValueError, "strict coherent-base"):
                feature_sum._validate_declared_coherent_base(
                    payload, "base", allow_legacy_metadata=False
                )
            accepted = feature_sum._validate_declared_coherent_base(
                payload, "base", allow_legacy_metadata=True
            )
            self.assertIn(
                "phase_reference",
                accepted[feature_sum._LEGACY_BASE_ASSUMPTIONS_KEY],
            )

    def test_reusing_assembled_base_rejects_same_id_or_component_signature(self):
        prior = [{
            "details": {
                "placements": [{
                    "schema": feature_workflow.POINT_PLACEMENT_SCHEMA,
                    "placement_id": "fastener-1",
                    "component_signature": "a" * 64,
                }]
            }
        }]
        with self.assertRaisesRegex(ValueError, "already present"):
            feature_sum._reject_reused_feature_components(
                prior,
                {"placements": [{
                    "schema": feature_workflow.POINT_PLACEMENT_SCHEMA,
                    "placement_id": "new-name",
                    "component_signature": "a" * 64,
                }]},
                "assembled.grim",
            )


if __name__ == "__main__":
    unittest.main()
