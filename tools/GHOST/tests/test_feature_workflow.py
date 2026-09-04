#!/usr/bin/env python3
"""Focused acceptance tests for the reusable feature-assembly service."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import feature_workflow  # noqa: E402
import feature_sum  # noqa: E402
import grim_io  # noqa: E402


POINT_HEADER = (
    "placement_id,dataset_id,x,y,z,nx,ny,nz,"
    "roll_x,roll_y,roll_z\n"
)
LINE_HEADER = (
    "line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,"
    "n1x,n1y,n1z,n2x,n2y,n2z\n"
)


def _write_empty_base(path, grid=None):
    """Write a real coherent GRIM instead of a metadata-free placeholder."""

    selected = dict(grid or {
        "frequencies_ghz": [1.0],
        "azimuths_deg": [0.0],
        "elevations_deg": [0.0],
        "axis_az_deg": 0.0,
        "axis_el_deg": 0.0,
        "roll_deg": 0.0,
    })
    feature_sum.export_radar_grim(
        str(path), bor_result=None, placements=[], **selected
    )


class DeclaredDeltaResponseRoleTests(unittest.TestCase):
    @staticmethod
    def _write_domain(path: Path, domain=None):
        payload = {}
        if domain is not None:
            payload["rcs_domain"] = np.asarray(domain)
        with path.open("wb") as stream:
            np.savez(stream, **payload)

    def test_canonical_opn_frd_roles_fail_before_point_or_line_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, expected_role in (
                ("SEAL-00-01_0.010gap_OPN.grim", "_OPN"),
                ("SEAL-00-01_0.010gap_FRD.grim", "_FRD"),
                ("SEAL-00-01_0.010gap_oPn.GrIm", "_OPN"),
            ):
                response = root / filename
                response.write_bytes(b"must not reach a response loader")
                with self.subTest(filename=filename, kind="line"):
                    with self.assertRaisesRegex(
                        ValueError, rf"filename role {expected_role}"
                    ):
                        feature_sum.load_seam_from_grim(
                            str(response),
                            1.0,
                            declared_coherent_delta=True,
                        )
                with self.subTest(filename=filename, kind="point"):
                    with self.assertRaisesRegex(
                        ValueError, rf"filename role {expected_role}"
                    ):
                        feature_sum.prepare_point_pattern(
                            str(response), declared_coherent_delta=True
                        )

    def test_role_free_canonical_and_gui_deltas_are_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "SEAL-00-01_0.010gap.grim"
            for domain, expected_status in (
                ("delta", "canonical_delta"),
                ("power_phase", "declared_gui_derived_delta"),
                ("power-phase", "declared_gui_derived_delta"),
                ("complex_amplitude", "declared_gui_derived_delta"),
                (None, "legacy_declared_delta_missing_domain"),
            ):
                self._write_domain(response, domain)
                with self.subTest(domain=domain):
                    record = (
                        feature_workflow.validate_declared_feature_delta_response(
                            response
                        )
                    )
                    self.assertEqual(record["filename_role"], "role_free")
                    self.assertEqual(record["embedded_rcs_domain"], domain)
                    self.assertEqual(record["status"], expected_status)

    def test_unknown_explicit_domain_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "role_free_delta.grim"
            self._write_domain(response, "whole_object_coefficient")
            with self.assertRaisesRegex(
                ValueError, "contradicts the embedded rcs_domain"
            ):
                feature_workflow.validate_declared_feature_delta_response(
                    response
                )


class DatasetDiscoveryTests(unittest.TestCase):
    def test_discovers_first_use_order_from_both_strict_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,fastener,0,0,0,0,0,1,1,0,0\n"
                + "p2,antenna,0,0,0,0,0,1,1,0,0\n"
                + "p3,fastener,0,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "gap_1,gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "seam_1,seam,1,0,1,0,1,1,0,0,0,1,0,0,1\n"
                + "seam_2,seam,1,0,2,0,1,2,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )

            required = feature_workflow.discover_feature_dataset_ids(
                point_locations_csv="points.csv",
                line_locations_csv="lines.csv",
                base_dir=root,
            )

            self.assertEqual(
                required.point_dataset_ids, ("fastener", "antenna")
            )
            self.assertEqual(required.line_dataset_ids, ("gap", "seam"))
            self.assertEqual(required.point_placement_count, 3)
            self.assertEqual(required.line_path_count, 3)
            self.assertEqual(required.line_segment_count, 3)
            self.assertEqual(
                required.point_instances,
                (("p1", "fastener"), ("p2", "antenna"), ("p3", "fastener")),
            )
            self.assertEqual(
                required.line_instances,
                (
                    ("gap_1", "gap", 1),
                    ("seam_1", "seam", 1),
                    ("seam_2", "seam", 1),
                ),
            )

    def test_discovery_enforces_the_exact_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "points.csv").write_text(
                "x,y,z\n0,0,0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "header must be exactly"):
                feature_workflow.discover_feature_dataset_ids(
                    point_locations_csv="points.csv", base_dir=root
                )

    def test_point_csv_reports_independent_bad_rows_together(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "points.csv"
            source.write_text(
                POINT_HEADER
                + "p1,fastener,bad,0,0,0,0,1,1,0,0\n"
                + "p2,,0,0,nan,0,0,1,1,0,0\n"
                + "p1,fastener,0,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as caught:
                feature_workflow.read_point_placement_csv(source)

            message = str(caught.exception)
            self.assertIn("4 validation error(s)", message)
            self.assertIn(
                "line 2: coordinates and vectors must be numeric; "
                "invalid column(s) x",
                message,
            )
            self.assertIn("line 3: placement_id and dataset_id are required", message)
            self.assertIn("line 3: NaN/infinite value in column(s) z", message)
            self.assertIn("line 4: duplicate placement_id 'p1'", message)

    def test_line_csv_aggregates_errors_without_index_cascade(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lines.csv"
            source.write_text(
                LINE_HEADER
                + "l1,seam,1,bad,0,0,1,0,0,0,0,1,0,0,1\n"
                + "l1,seam,not-an-index,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "l1,other,3,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "l2,seam,2,0,0,0,1,0,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as caught:
                feature_workflow.read_line_placement_csv(source)

            message = str(caught.exception)
            self.assertIn("4 validation error(s)", message)
            self.assertIn(
                "line 2: endpoints and normals must be numeric; "
                "invalid column(s) x1",
                message,
            )
            self.assertIn(
                "line 3: segment_index must be a canonical positive integer",
                message,
            )
            self.assertIn(
                "every segment of line_id 'l1' must use the same dataset_id",
                message,
            )
            self.assertIn(
                "line 5: line_id 'l2' requires one consecutive segment_index",
                message,
            )
            self.assertNotIn("expected 2", message)
            self.assertNotIn("expected 3", message)

    def test_csv_error_report_is_bounded_and_counts_omitted_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "points.csv"
            source.write_text(
                POINT_HEADER
                + "".join(
                    f"p{index},fastener,bad,0,0,0,0,1,1,0,0\n"
                    for index in range(30)
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as caught:
                feature_workflow.read_point_placement_csv(source)

            message = str(caught.exception)
            self.assertIn("30 validation error(s)", message)
            self.assertIn("5 additional error(s) omitted", message)
            self.assertNotIn("line 31:", message)


class RequestPlanTests(unittest.TestCase):
    def test_feature_plan_preserves_original_positional_field_layout(self):
        request = feature_workflow.FeatureAssemblyRequest(
            base_grim="body.grim", output_grim="assembled.grim"
        )
        preview = feature_workflow.FeaturePreviewGeometry(
            surface_triangles_cad_m=None,
            body_profile_rho_z_m=None,
            point_locations_cad_m={},
            line_paths_cad_m={},
        )
        plan = feature_workflow.FeatureAssemblyPlan(
            request,
            Path("body.grim"),
            Path("assembled.grim"),
            {},
            None,
            None,
            None,
            lambda points: np.zeros_like(points),
            None,
            [],
            [],
            [],
            [],
            feature_workflow.FeatureDatasetRequirements(),
            preview,
            0.0,
            1.0,
            {},
            {"body.grim": "a" * 64},
            ("legacy warning",),
            ("missing.sidecar",),
            "b" * 64,
            False,
            "legacy-positional-plan-seal",
        )

        self.assertEqual(plan.prepared_plan_sha256, "legacy-positional-plan-seal")
        self.assertIsNone(plan.prepared_features_only_output_sha256)
        self.assertTrue(plan.prepared_features_only_output_absent)

    def test_physical_units_are_unset_and_required_only_for_selected_inputs(self):
        request = feature_workflow.FeatureAssemblyRequest(
            base_grim="body.grim",
            output_grim="assembled.grim",
            point_locations_csv="points.csv",
        )
        self.assertIsNone(request.coordinate_units)
        self.assertIsNone(request.surface_units)
        with self.assertRaisesRegex(
            ValueError, "coordinate_units must be selected explicitly"
        ):
            feature_workflow.prepare_feature_assembly(request)

        with self.assertRaisesRegex(
            ValueError, "surface_units must be selected explicitly"
        ):
            feature_workflow.prepare_feature_input_preview(
                surface_mesh="body.stl"
            )
        with self.assertRaisesRegex(
            ValueError, "coordinate_units must be selected explicitly"
        ):
            feature_workflow.prepare_feature_input_preview(
                point_locations_csv="points.csv"
            )

        # A base-only preview needs no arbitrary placement or mesh unit.
        with mock.patch.object(
            feature_workflow,
            "load_body_requested_radar_grid",
            return_value=None,
        ):
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "body.grim"
                base.touch()
                preview = feature_workflow.prepare_feature_input_preview(
                    base_grim=base
                )
        self.assertEqual(preview.body_source, "none")

    def test_4500_mm_surface_preview_is_interpreted_as_4_5_m(self):
        with tempfile.TemporaryDirectory() as directory:
            surface = Path(directory) / "vehicle.stl"
            surface.write_bytes(b"mesh bytes")
            raw_mm = np.asarray(
                [
                    [[0.0, 0.0, 0.0], [4500.0, 0.0, 0.0], [0.0, 1000.0, 0.0]],
                    [[4500.0, 1000.0, 500.0], [4500.0, 0.0, 500.0], [0.0, 1000.0, 500.0]],
                ]
            )
            with mock.patch.object(
                feature_workflow,
                "read_surface_mesh",
                return_value=raw_mm,
            ):
                preview = feature_workflow.prepare_feature_input_preview(
                    surface_mesh=surface,
                    surface_units="millimeters",
                )

        vertices = preview.surface_triangles_cad_m.reshape(-1, 3)
        np.testing.assert_allclose(
            np.ptp(vertices, axis=0),
            [4.5, 1.0, 0.5],
            rtol=0.0,
            atol=1.0e-15,
        )

    def _prepare_minimal_line_plan(self, root):
        base = root / "body.grim"
        _write_empty_base(base)
        line_csv = root / "lines.csv"
        line_csv.write_text(
            LINE_HEADER
            + "gap_1,gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n",
            encoding="utf-8",
        )
        dataset = root / "gap.grim"
        dataset.write_bytes(b"prepared line response")
        line_placements = [{
            "delta": str(dataset.resolve()),
            "kind": "delta",
            "declared_coherent_delta": True,
            "delta_sign": 1.0,
            "perimeter": np.asarray([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
            "segment_normals": np.asarray([[0.0, 0.0, 1.0]]),
        }]
        line_records = [{
            "schema": feature_workflow.LINE_PLACEMENT_SCHEMA,
            "line_id": "gap_1",
            "dataset_id": "gap",
            "dataset": str(dataset.resolve()),
            "dataset_sha256": feature_workflow.sha256_file(str(dataset)),
            "segment_count": 1,
            "input_subtraction_order": "OPN-FRD (featured-clean)",
        }]
        request = feature_workflow.FeatureAssemblyRequest(
            base_grim=base,
            output_grim=root / "assembled.grim",
            coordinate_units="meters",
            line_locations_csv=line_csv,
            line_datasets={"gap": dataset},
            base_dir=root,
        )
        grid = {
            "frequencies_ghz": [1.0],
            "azimuths_deg": [0.0],
            "elevations_deg": [0.0],
            "axis_az_deg": 0.0,
            "axis_el_deg": 0.0,
            "roll_deg": 0.0,
        }
        profile = np.asarray([[1.0, 1.0], [1.0, -1.0]])
        with (
            mock.patch.object(
                feature_workflow,
                "load_body_requested_radar_grid",
                return_value=grid,
            ),
            mock.patch.object(
                feature_workflow,
                "load_body_profile_grim",
                return_value=profile,
            ),
            mock.patch.object(
                feature_workflow,
                "prepare_line_placements",
                return_value=(line_placements, line_records),
            ),
            mock.patch.object(
                feature_workflow,
                "prepare_point_placements",
                return_value=([], []),
            ),
            mock.patch.object(
                feature_workflow,
                "_apply_feature_library_contracts",
                return_value=({}, [], {}, set()),
            ),
        ):
            return feature_workflow.prepare_feature_assembly(request)

    @staticmethod
    def _external_plane_case(root):
        base = root / "body.grim"
        surface = root / "body.facet"
        coordinates = root / "lines.csv"
        response = root / "gap.grim"
        grid = {
            "frequencies_ghz": [1.0],
            "azimuths_deg": [0.0],
            "elevations_deg": [0.0],
            "axis_az_deg": 0.0,
            "axis_el_deg": 0.0,
            "roll_deg": 0.0,
        }
        _write_empty_base(base, grid)
        surface.write_bytes(b"external plane mesh snapshot")
        coordinates.write_text(
            LINE_HEADER
            + "gap_1,gap,1,-0.25,0,0,0.25,0,0,0,0,1,0,0,1\n",
            encoding="utf-8",
        )
        response.write_bytes(b"line response snapshot")
        request = feature_workflow.FeatureAssemblyRequest(
            base_grim=base,
            output_grim=root / "assembled.grim",
            coordinate_units="meters",
            surface_mesh=surface,
            surface_units="meters",
            line_locations_csv=coordinates,
            line_datasets={"gap": response},
            base_dir=root,
        )
        payload = feature_sum._load_grim(str(base))
        triangles = np.asarray([[[-1.0, -1.0, 0.0],
                                 [1.0, -1.0, 0.0],
                                 [0.0, 1.0, 0.0]]])
        return request, payload, triangles, surface, coordinates

    def test_selected_mesh_with_shadow_off_requires_release_warning_waiver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, payload, triangles, _surface, _coordinates = (
                self._external_plane_case(root)
            )
            self.assertFalse(request.shadow)
            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=None,
                ),
                mock.patch.object(
                    feature_workflow, "_load_grim", return_value=payload
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_surface_mesh",
                    return_value=triangles,
                ),
                mock.patch.object(
                    feature_workflow,
                    "_apply_feature_library_contracts",
                    return_value=({}, [], {}, set()),
                ),
            ):
                plan = feature_workflow.prepare_feature_assembly(request)

        warning = next(
            value
            for value in plan.validation_warnings
            if "shadowing is OFF" in value
        )
        self.assertIn("Hidden point and line features", warning)
        self.assertIn("full modeled amplitude", warning)
        self.assertIn("one-time release-warning waiver", warning)

    def test_embedded_body_without_external_mesh_has_no_shadow_off_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._prepare_minimal_line_plan(Path(directory))

        self.assertFalse(plan.request.shadow)
        self.assertIsNone(plan.request.surface_mesh)
        self.assertFalse(any(
            "shadowing is OFF" in warning
            for warning in plan.validation_warnings
        ))

    def test_requested_body_certificate_is_authoritative_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, payload, triangles, _surface, _coordinates = (
                self._external_plane_case(root)
            )
            request = feature_workflow.replace(
                request, require_body_mesh_certification=True
            )
            certificate = {
                "schema": "ghost.workflow.body-mesh-certification.v1",
                "passed": True,
                "published_mesh": "fine",
                "frequencies_ghz": [1.0],
                "per_frequency": {},
            }
            with (
                mock.patch.object(
                    feature_workflow,
                    "audit_body_mesh_certification",
                    return_value=certificate,
                ) as audit,
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=None,
                ),
                mock.patch.object(
                    feature_workflow, "_load_grim", return_value=payload
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_surface_mesh",
                    return_value=triangles,
                ),
                mock.patch.object(
                    feature_workflow,
                    "_apply_feature_library_contracts",
                    return_value=({}, [], {}, set()),
                ),
            ):
                plan = feature_workflow.prepare_feature_assembly(request)

        audit.assert_called_once_with(
            str(request.base_grim.resolve()), loaded_grim=payload
        )
        self.assertEqual(
            plan.feature_provenance["body_mesh_certification_policy"],
            "required_validated",
        )
        self.assertEqual(
            plan.feature_provenance["body_mesh_certification"], certificate
        )

    def test_requested_body_certificate_failure_stops_before_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, _payload, _triangles, _surface, _coordinates = (
                self._external_plane_case(root)
            )
            request = feature_workflow.replace(
                request, require_body_mesh_certification=True
            )
            with mock.patch.object(
                feature_workflow,
                "audit_body_mesh_certification",
                side_effect=ValueError("not a passed fine-mesh body result"),
            ):
                with self.assertRaisesRegex(ValueError, "fine-mesh"):
                    feature_workflow.prepare_feature_assembly(request)

    def test_external_body_certificate_error_identifies_optional_profile_and_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "external.grim"
            _write_empty_base(base)
            with self.assertRaises(ValueError) as caught:
                feature_sum.require_body_mesh_certification(str(base))
        message = str(caught.exception)
        self.assertIn("General body profile", message)
        self.assertIn("Diagnostic detail:", message)
        self.assertIn("azimuth_meaning", message)
        self.assertNotIn("not a valid BoR body artifact", message)

    def test_enabled_snapshot_filters_preview_after_strict_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,fastener,1,0,0,0,0,1,1,0,0\n"
                + "p2,antenna,2,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "gap_1,gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "seal_1,seal,1,0,1,0,1,1,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )

            preview = feature_workflow.prepare_feature_input_preview(
                point_locations_csv="points.csv",
                line_locations_csv="lines.csv",
                enabled_point_placement_ids=("p2",),
                enabled_line_ids=("seal_1",),
                base_dir=root,
                coordinate_units="meters",
            )

            self.assertEqual(set(preview.point_locations_cad_m), {"antenna"})
            self.assertEqual(
                preview.point_placement_ids, {"antenna": ("p2",)}
            )
            self.assertEqual(set(preview.line_paths_cad_m), {"seal"})
            self.assertEqual(
                preview.dataset_requirements.point_instances,
                (("p1", "fastener"), ("p2", "antenna")),
            )
            self.assertEqual(
                preview.dataset_requirements.line_instances,
                (("gap_1", "gap", 1), ("seal_1", "seal", 1)),
            )

            with self.assertRaisesRegex(ValueError, "not present in the parsed"):
                feature_workflow.prepare_feature_input_preview(
                    point_locations_csv="points.csv",
                    enabled_point_placement_ids=("stale-id",),
                    base_dir=root,
                    coordinate_units="meters",
                )
            with self.assertRaisesRegex(TypeError, "sequence of complete IDs"):
                feature_workflow.prepare_feature_input_preview(
                    point_locations_csv="points.csv",
                    enabled_point_placement_ids="p1",
                    base_dir=root,
                    coordinate_units="meters",
                )
            clean_preview = feature_workflow.prepare_feature_input_preview(
                point_locations_csv="points.csv",
                enabled_point_placement_ids=(),
                base_dir=root,
                coordinate_units="meters",
            )
            self.assertEqual(clean_preview.point_locations_cad_m, {})
            self.assertEqual(
                clean_preview.dataset_requirements.point_instances,
                (("p1", "fastener"), ("p2", "antenna")),
            )

            # Disabled rows still pass through the strict parser before any
            # selection is applied; selection is not a format-error bypass.
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,fastener,not-a-number,0,0,0,0,1,1,0,0\n"
                + "p2,antenna,2,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be numeric"):
                feature_workflow.prepare_feature_input_preview(
                    point_locations_csv="points.csv",
                    enabled_point_placement_ids=("p2",),
                    base_dir=root,
                    coordinate_units="meters",
                )

    def test_all_disabled_request_fails_before_response_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_empty_base(root / "body.grim")
            (root / "points.csv").write_text(
                POINT_HEADER + "p1,fastener,0,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            request = feature_workflow.FeatureAssemblyRequest(
                base_grim="body.grim",
                output_grim="assembled.grim",
                coordinate_units="meters",
                point_locations_csv="points.csv",
                point_datasets={},
                enabled_point_placement_ids=(),
                base_dir=root,
            )
            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value={"frequencies_ghz": [1.0]},
                ),
                mock.patch.object(
                    feature_workflow,
                    "load_body_profile_grim",
                    return_value=np.asarray([[1.0, 1.0], [1.0, -1.0]]),
                ),
                mock.patch.object(
                    feature_workflow, "_resolved_dataset_paths"
                ) as response_loader,
            ):
                with self.assertRaisesRegex(ValueError, "No enabled spatial features"):
                    feature_workflow.prepare_feature_assembly(request)
            response_loader.assert_not_called()

    def test_disabled_dataset_mapping_is_not_loaded_or_hashed(self):
        class FlatSurface:
            def distance(self, points):
                return np.zeros(len(np.atleast_2d(points)), dtype=float)

            def normal(self, points):
                outward = feature_workflow.to_axis_frame([0.0, 0.0, 1.0])
                return np.tile(outward, (len(np.atleast_2d(points)), 1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "points.csv").write_text(
                POINT_HEADER
                + "keep,active,0,0,0,0,0,1,1,0,0\n"
                + "omit,disabled,0,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            (root / "active.grim").write_bytes(b"active delta")

            points, records = feature_workflow.prepare_point_placements(
                None,
                FlatSurface(),
                coordinate_scale=1.0,
                skin_limit_m=1.0,
                wavelength_m=1.0,
                normal_tolerance_deg=15.0,
                locations_csv="points.csv",
                datasets={
                    "active": "active.grim",
                    "disabled": "does-not-exist.grim",
                },
                enabled_point_placement_ids=("keep",),
                base_dir=root,
                pattern_loader=lambda *_args, **_kwargs: object(),
            )

            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]["placement_id"], "keep")
            self.assertEqual([record["placement_id"] for record in records], ["keep"])

    def test_disabled_line_mapping_is_not_loaded_or_hashed(self):
        class FlatSurface:
            def distance(self, points):
                return np.zeros(len(np.atleast_2d(points)), dtype=float)

            def normal(self, points):
                outward = feature_workflow.to_axis_frame([0.0, 0.0, 1.0])
                return np.tile(outward, (len(np.atleast_2d(points)), 1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "keep,active,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "omit,disabled,1,0,1,0,1,1,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            (root / "active.grim").write_bytes(b"active line delta")

            placements, records = feature_workflow.prepare_line_placements(
                None,
                FlatSurface(),
                coordinate_scale=1.0,
                skin_limit_m=1.0,
                wavelength_m=1.0,
                normal_tolerance_deg=15.0,
                locations_csv="lines.csv",
                datasets={
                    "active": "active.grim",
                    "disabled": "does-not-exist.grim",
                },
                enabled_line_ids=("keep",),
                base_dir=root,
            )

            self.assertEqual(len(placements), 1)
            self.assertEqual(placements[0]["line_id"], "keep")
            self.assertEqual([record["line_id"] for record in records], ["keep"])

    def test_disabled_physically_invalid_rows_do_not_block_trade_study(self):
        class FlatSurface:
            def distance(self, points):
                return np.zeros(len(np.atleast_2d(points)), dtype=float)

            def normal(self, points):
                outward = feature_workflow.to_axis_frame([0.0, 0.0, 1.0])
                return np.tile(outward, (len(np.atleast_2d(points)), 1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_dataset = root / "point.grim"
            line_dataset = root / "line.grim"
            point_dataset.write_bytes(b"point delta")
            line_dataset.write_bytes(b"line delta")
            (root / "points.csv").write_text(
                POINT_HEADER
                + "keep,point,0,0,0,0,0,1,1,0,0\n"
                + "omit,point,0,0,0,0,0,-1,1,0,0\n",
                encoding="utf-8",
            )
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "keep,line,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "omit,line,1,0,1,0,1,1,0,0,0,-1,0,0,-1\n",
                encoding="utf-8",
            )

            points, point_records = feature_workflow.prepare_point_placements(
                None,
                FlatSurface(),
                coordinate_scale=1.0,
                skin_limit_m=1.0,
                wavelength_m=1.0,
                normal_tolerance_deg=15.0,
                locations_csv="points.csv",
                datasets={"point": point_dataset},
                enabled_point_placement_ids=("keep",),
                base_dir=root,
                pattern_loader=lambda *_args, **_kwargs: object(),
            )
            lines, line_records = feature_workflow.prepare_line_placements(
                None,
                FlatSurface(),
                coordinate_scale=1.0,
                skin_limit_m=1.0,
                wavelength_m=1.0,
                normal_tolerance_deg=15.0,
                locations_csv="lines.csv",
                datasets={"line": line_dataset},
                enabled_line_ids=("keep",),
                base_dir=root,
            )

            self.assertEqual(len(points), 1)
            self.assertEqual(len(lines), 1)
            self.assertEqual(
                [record["placement_id"] for record in point_records], ["keep"]
            )
            self.assertEqual(
                [record["line_id"] for record in line_records], ["keep"]
            )

    def test_input_preview_needs_no_output_or_response_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "body.grim").touch()
            (root / "body.stl").touch()
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,fastener,1,2,3,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "gap_1,panel_gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n"
                + "gap_1,panel_gap,2,1,0,0,1,1,0,0,2,0,0,0,3\n",
                encoding="utf-8",
            )
            triangles = np.asarray([[[0, 0, 0], [2, 0, 0], [0, 2, 0]]], dtype=float)
            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=None,
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_surface_mesh",
                    return_value=triangles,
                ),
            ):
                preview = feature_workflow.prepare_feature_input_preview(
                    base_grim="body.grim",
                    surface_mesh="body.stl",
                    coordinate_units="millimeters",
                    surface_units="millimeters",
                    point_locations_csv="points.csv",
                    line_locations_csv="lines.csv",
                    base_dir=root,
                )

            self.assertEqual(preview.preview_stage, "input")
            self.assertEqual(preview.body_source, "surface_mesh")
            self.assertEqual(
                preview.dataset_requirements.point_dataset_ids, ("fastener",)
            )
            self.assertEqual(
                preview.dataset_requirements.line_dataset_ids, ("panel_gap",)
            )
            np.testing.assert_allclose(
                preview.surface_triangles_cad_m, triangles * 1.0e-3
            )
            np.testing.assert_allclose(
                preview.point_locations_cad_m["fastener"],
                [[1.0e-3, 2.0e-3, 3.0e-3]],
            )
            self.assertEqual(
                preview.point_placement_ids, {"fastener": ("p1",)}
            )
            np.testing.assert_allclose(
                preview.line_paths_cad_m["panel_gap"]["gap_1"],
                [[0, 0, 0], [1.0e-3, 0, 0], [1.0e-3, 1.0e-3, 0]],
            )
            # Orientation vectors remain raw, unitless CAD-frame values. In
            # particular, both copies of the shared line vertex survive even
            # when their supplied endpoint normals differ.
            np.testing.assert_allclose(
                preview.point_normals_cad["fastener"], [[0.0, 0.0, 1.0]]
            )
            np.testing.assert_allclose(
                preview.point_roll_references_cad["fastener"],
                [[1.0, 0.0, 0.0]],
            )
            line_normals = preview.line_endpoint_normals_cad[
                "panel_gap"
            ]["gap_1"]
            self.assertEqual(line_normals.shape, (2, 2, 3))
            np.testing.assert_allclose(line_normals[0, 1], [0.0, 0.0, 1.0])
            np.testing.assert_allclose(line_normals[1, 0], [0.0, 2.0, 0.0])
            np.testing.assert_allclose(line_normals[1, 1], [0.0, 0.0, 3.0])

    def test_input_preview_uses_embedded_bor_profile_without_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "body.grim").touch()
            profile = np.asarray([[0.0, 1.0], [0.5, 0.0], [0.0, -1.0]])
            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value={"frequencies_ghz": [1.0]},
                ),
                mock.patch.object(
                    feature_workflow,
                    "load_body_profile_grim",
                    return_value=profile,
                ),
            ):
                preview = feature_workflow.prepare_feature_input_preview(
                    base_grim="body.grim", base_dir=root
                )

            self.assertEqual(preview.body_source, "embedded_bor_profile")
            self.assertIsNone(preview.surface_triangles_cad_m)
            np.testing.assert_allclose(preview.body_profile_rho_z_m, profile)

    def test_input_preview_captures_invalid_vectors_without_validating_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,fastener,0,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "gap_1,gap,1,0,0,0,1,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )

            preview = feature_workflow.prepare_feature_input_preview(
                point_locations_csv="points.csv",
                line_locations_csv="lines.csv",
                base_dir=root,
                coordinate_units="meters",
            )

            np.testing.assert_array_equal(
                preview.point_normals_cad["fastener"], [[0.0, 0.0, 0.0]]
            )
            np.testing.assert_array_equal(
                preview.point_roll_references_cad["fastener"],
                [[0.0, 0.0, 0.0]],
            )
            np.testing.assert_array_equal(
                preview.line_endpoint_normals_cad["gap"]["gap_1"],
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            )

    def test_output_must_not_overwrite_clean_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "body.grim").touch()
            for output in ("body.grim", "body"):
                with self.subTest(output=output):
                    request = feature_workflow.FeatureAssemblyRequest(
                        base_grim="body.grim",
                        output_grim=output,
                        line_locations_csv="lines.csv",
                        line_datasets={"gap": "gap.grim"},
                        base_dir=root,
                    )
                    with self.assertRaisesRegex(ValueError, "must differ"):
                        feature_workflow.prepare_feature_assembly(request)

    def test_feature_only_sibling_must_not_overwrite_clean_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "assembled_features_only.grim"
            base.write_bytes(b"clean body")
            request = feature_workflow.FeatureAssemblyRequest(
                base_grim=base,
                output_grim=root / "assembled.grim",
                point_locations_csv=root / "points.csv",
            )
            with self.assertRaisesRegex(ValueError, "feature-only sibling"):
                feature_workflow.prepare_feature_assembly(request)

    def test_output_must_not_overwrite_a_mapped_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "body.grim").touch()
            (root / "response.grim").touch()
            cases = (
                {
                    "point_locations_csv": "points.csv",
                    "point_datasets": {"fastener": "response.grim"},
                },
                {
                    "line_locations_csv": "lines.csv",
                    "line_datasets": {"gap": "response.grim"},
                },
            )
            for settings in cases:
                with self.subTest(settings=settings):
                    request = feature_workflow.FeatureAssemblyRequest(
                        base_grim="body.grim",
                        output_grim="response",
                        base_dir=root,
                        **settings,
                    )
                    with self.assertRaisesRegex(
                        ValueError, "mapped response input"
                    ):
                        feature_workflow.prepare_feature_assembly(request)

    def test_execute_rechecks_output_alias_created_after_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "body.grim"
            output = root / "assembled.grim"
            base.write_bytes(b"clean body")
            request = feature_workflow.FeatureAssemblyRequest(
                base_grim=base,
                output_grim=output,
                point_locations_csv=root / "points.csv",
            )
            preview = feature_workflow.FeaturePreviewGeometry(
                surface_triangles_cad_m=None,
                body_profile_rho_z_m=None,
                point_locations_cad_m={},
                line_paths_cad_m={},
            )
            plan = feature_workflow.FeatureAssemblyPlan(
                request=request,
                base_path=base.resolve(),
                output_path=output.resolve(),
                radar_grid={},
                body_profile=None,
                surface_path=None,
                surface=None,
                surface_normal_fn=lambda points: np.zeros_like(points),
                occluder=None,
                line_placements=[],
                point_placements=[],
                line_records=[],
                point_records=[],
                dataset_requirements=feature_workflow.FeatureDatasetRequirements(),
                preview_geometry=preview,
                skin_limit_m=0.0,
                highest_frequency_wavelength_m=1.0,
                feature_provenance={},
            )
            try:
                os.link(base, output)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")

            with mock.patch.object(
                feature_workflow, "add_features_to_monostatic_grim"
            ) as add_features:
                with self.assertRaisesRegex(ValueError, "must differ from base"):
                    feature_workflow.execute_feature_assembly(plan)
            add_features.assert_not_called()

    def test_changed_base_after_prepare_cannot_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._prepare_minimal_line_plan(root)
            plan.base_path.write_bytes(b"changed clean body")

            with self.assertRaisesRegex(
                RuntimeError, "source changed before execution"
            ):
                feature_workflow.execute_feature_assembly(plan)

            self.assertFalse(plan.output_path.exists())

    def test_changed_active_line_response_after_prepare_cannot_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._prepare_minimal_line_plan(root)
            line_source = Path(plan.line_records[0]["dataset"])
            line_source.write_bytes(b"changed line response")

            with self.assertRaisesRegex(
                RuntimeError, "source changed before execution"
            ):
                feature_workflow.execute_feature_assembly(plan)

            self.assertFalse(plan.output_path.exists())

    def test_changed_prepared_geometry_requires_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._prepare_minimal_line_plan(Path(directory))
            plan.line_placements[0]["perimeter"][0, 0] = 123.0
            with self.assertRaisesRegex(
                RuntimeError, "plan changed after validation"
            ):
                feature_workflow.execute_feature_assembly(plan)
            self.assertFalse(plan.output_path.exists())

    def test_capacity_rejection_precedes_execution_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._prepare_minimal_line_plan(Path(directory))
            with (
                mock.patch.object(
                    feature_workflow,
                    "preflight_feature_assembly_capacity",
                    side_effect=MemoryError("capacity rejected"),
                ),
                mock.patch.object(
                    feature_workflow, "_execution_plan_snapshot"
                ) as snapshot,
            ):
                with self.assertRaisesRegex(MemoryError, "capacity rejected"):
                    feature_workflow.execute_feature_assembly(plan)
            snapshot.assert_not_called()
            self.assertFalse(plan.output_path.exists())

    def test_large_workload_warning_requires_exact_sealed_plan_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._prepare_minimal_line_plan(Path(directory))
            workload = feature_workflow.estimate_assembly_workload(
                look_count=1,
                frequency_count=1,
                point_count=250_000_000,
                line_path_count=0,
                line_segment_count=0,
                line_piece_count=0,
                quantities_validated=True,
            )
            warning = feature_workflow.workload_review_warning(workload)
            self.assertIsNotNone(warning)
            object.__setattr__(plan, "validation_warnings", (warning,))
            object.__setattr__(
                plan,
                "prepared_plan_sha256",
                feature_workflow.feature_assembly_plan_sha256(plan),
            )

            with mock.patch.object(
                feature_workflow, "preflight_feature_assembly_capacity"
            ) as capacity:
                with self.assertRaisesRegex(
                    RuntimeError, "workload review was not acknowledged"
                ):
                    feature_workflow.execute_feature_assembly(plan)
            capacity.assert_not_called()

            with (
                mock.patch.object(
                    feature_workflow,
                    "preflight_feature_assembly_capacity",
                    return_value=SimpleNamespace(),
                ),
                mock.patch.object(
                    feature_workflow,
                    "add_features_to_monostatic_grim",
                    return_value=str(plan.output_path),
                ) as publish,
            ):
                saved = feature_workflow.execute_feature_assembly(
                    plan,
                    acknowledged_plan_sha256=plan.prepared_plan_sha256,
                )
            self.assertEqual(saved, str(plan.output_path))
            publish.assert_called_once()

    def test_prepared_large_workload_warning_is_part_of_sealed_plan(self):
        workload = feature_workflow.estimate_assembly_workload(
            look_count=1,
            frequency_count=1,
            point_count=250_000_000,
            line_path_count=0,
            line_segment_count=0,
            line_piece_count=0,
            quantities_validated=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                feature_workflow,
                "_prepared_assembly_workload",
                return_value=workload,
            ):
                plan = self._prepare_minimal_line_plan(Path(directory))

        warning = next(
            value
            for value in plan.validation_warnings
            if value.startswith("Assembly workload review required")
        )
        self.assertIn("250,000,000 point look-frequency", warning)
        self.assertEqual(
            plan.feature_provenance["assembly_workload_preflight"]
            ["point_field_cell_count"],
            250_000_000,
        )
        self.assertEqual(
            plan.prepared_plan_sha256,
            feature_workflow.feature_assembly_plan_sha256(plan),
        )

    def test_progress_callback_cannot_mutate_private_execution_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._prepare_minimal_line_plan(Path(directory))
            reviewed = np.array(
                plan.line_placements[0]["perimeter"], copy=True
            )

            def mutate_public_plan(_done, _total, _message):
                plan.line_placements[0]["perimeter"][:] = -999.0

            def fake_build(_base, output, *, placements, progress_callback, **_kwargs):
                progress_callback(1, 2, "building")
                np.testing.assert_array_equal(
                    placements[0]["perimeter"], reviewed
                )
                self.assertFalse(placements[0]["perimeter"].flags.writeable)
                return output

            with mock.patch.object(
                feature_workflow,
                "add_features_to_monostatic_grim",
                side_effect=fake_build,
            ):
                saved = feature_workflow.execute_feature_assembly(
                    plan, progress_callback=mutate_public_plan
                )
            self.assertEqual(saved, str(plan.output_path))

    def test_line_csv_mutated_after_parse_fails_prepare_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, payload, triangles, _surface, coordinates = (
                self._external_plane_case(root)
            )
            original_reader = feature_workflow.read_line_placement_csv

            def read_then_mutate(path, **kwargs):
                parsed = original_reader(path, **kwargs)
                coordinates.write_text(
                    coordinates.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                return parsed

            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=None,
                ),
                mock.patch.object(
                    feature_workflow, "_load_grim", return_value=payload
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_surface_mesh",
                    return_value=triangles,
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_line_placement_csv",
                    side_effect=read_then_mutate,
                ),
                mock.patch.object(
                    feature_workflow,
                    "_apply_feature_library_contracts",
                    return_value=({}, [], {}, set()),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "source changed during preparation"
                ):
                    feature_workflow.prepare_feature_assembly(request)

            self.assertFalse(Path(request.output_grim).exists())

    def test_surface_mutated_while_reading_fails_prepare_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, payload, triangles, surface, _coordinates = (
                self._external_plane_case(root)
            )

            def read_then_mutate(_path):
                surface.write_bytes(surface.read_bytes() + b" changed")
                return triangles

            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=None,
                ),
                mock.patch.object(
                    feature_workflow, "_load_grim", return_value=payload
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_surface_mesh",
                    side_effect=read_then_mutate,
                ),
                mock.patch.object(
                    feature_workflow,
                    "_apply_feature_library_contracts",
                    return_value=({}, [], {}, set()),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "source changed during preparation"
                ):
                    feature_workflow.prepare_feature_assembly(request)

            self.assertFalse(Path(request.output_grim).exists())

    def test_changed_spatial_csv_after_prepare_cannot_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, payload, triangles, _surface, coordinates = (
                self._external_plane_case(root)
            )
            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=None,
                ),
                mock.patch.object(
                    feature_workflow, "_load_grim", return_value=payload
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_surface_mesh",
                    return_value=triangles,
                ),
                mock.patch.object(
                    feature_workflow,
                    "_apply_feature_library_contracts",
                    return_value=({}, [], {}, set()),
                ),
            ):
                plan = feature_workflow.prepare_feature_assembly(request)

            coordinates.write_text(
                coordinates.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError, "source changed before execution"
            ):
                feature_workflow.execute_feature_assembly(plan)

            self.assertFalse(plan.output_path.exists())

    def test_plan_carries_full_cad_preview_geometry_in_meters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_empty_base(
                root / "body.grim",
                {
                    "frequencies_ghz": [1.0],
                    "azimuths_deg": [0.0, 90.0],
                    "elevations_deg": [0.0],
                    "axis_az_deg": 0.0,
                    "axis_el_deg": 0.0,
                    "roll_deg": 0.0,
                },
            )
            (root / "body.facet").touch()
            (root / "gap.grim").write_bytes(b"line delta")
            (root / "antenna.grim").write_bytes(b"point delta")
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "gap_1,gap,1,0,0,0,1,0,0,0,0,2,0,0,3\n",
                encoding="utf-8",
            )
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,antenna,0.25,0.25,0,0,0,2,2,0,1\n",
                encoding="utf-8",
            )
            surface_in = np.asarray([[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]])
            request = feature_workflow.FeatureAssemblyRequest(
                base_grim="body.grim",
                output_grim="assembled",
                coordinate_units="inches",
                surface_mesh="body.facet",
                surface_units="inches",
                line_locations_csv="lines.csv",
                line_datasets={"gap": "gap.grim"},
                point_locations_csv="points.csv",
                point_datasets={"antenna": "antenna.grim"},
                base_dir=root,
            )
            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=None,
                ),
                mock.patch.object(
                    feature_workflow,
                    "read_surface_mesh",
                    return_value=surface_in,
                ),
                mock.patch.object(
                    feature_workflow,
                    "prepare_point_pattern",
                    return_value="prepared point pattern",
                ),
                mock.patch.object(
                    feature_workflow,
                    "_apply_feature_library_contracts",
                    return_value=({}, [], {}, set()),
                ),
            ):
                plan = feature_workflow.prepare_feature_assembly(request)

            input_sources = plan.feature_provenance[
                "prepared_input_sources"
            ]
            for role, filename in (
                ("base_grim", "body.grim"),
                ("surface_mesh", "body.facet"),
                ("line_locations_csv", "lines.csv"),
                ("point_locations_csv", "points.csv"),
            ):
                source = (root / filename).resolve()
                expected_hash = feature_workflow.sha256_file(str(source))
                self.assertEqual(
                    input_sources[role],
                    {"path": str(source), "sha256": expected_hash},
                )
                self.assertEqual(
                    plan.prepared_source_sha256[str(source)], expected_hash
                )
            for filename in ("gap.grim", "antenna.grim"):
                source = (root / filename).resolve()
                self.assertEqual(
                    plan.prepared_source_sha256[str(source)],
                    feature_workflow.sha256_file(str(source)),
                )

            preview = plan.preview_geometry
            topology = plan.feature_provenance["surface_mesh_topology"]
            self.assertEqual(topology["schema"], "ghost.assembly-mesh-topology.v1")
            self.assertEqual(topology["boundary_edge_count"], 3)
            self.assertFalse(topology["watertight"])
            self.assertTrue(any(
                "open boundary edge" in warning
                for warning in plan.validation_warnings
            ))
            expected_surface_cad_m = surface_in * 0.0254
            np.testing.assert_allclose(
                preview.surface_triangles_cad_m, expected_surface_cad_m
            )
            self.assertIs(plan.surface_triangles_cad_m, preview.surface_triangles_cad_m)
            self.assertIsNone(preview.body_profile_rho_z_m)
            np.testing.assert_allclose(
                preview.point_locations_cad_m["antenna"],
                [[0.25 * 0.0254, 0.25 * 0.0254, 0.0]],
            )
            self.assertEqual(
                preview.point_placement_ids, {"antenna": ("p1",)}
            )
            np.testing.assert_allclose(
                preview.line_paths_cad_m["gap"]["gap_1"],
                [[0.0, 0.0, 0.0], [0.0254, 0.0, 0.0]],
            )
            np.testing.assert_allclose(
                preview.point_normals_cad["antenna"], [[0.0, 0.0, 2.0]]
            )
            np.testing.assert_allclose(
                preview.point_roll_references_cad["antenna"],
                [[2.0, 0.0, 1.0]],
            )
            np.testing.assert_allclose(
                preview.line_endpoint_normals_cad["gap"]["gap_1"],
                [[[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]],
            )

            # Physics keeps using the rotated BoR axis frame; preview remains
            # in the CAD frame the user supplied.
            np.testing.assert_allclose(
                plan.surface.triangles,
                feature_workflow.to_axis_frame(expected_surface_cad_m),
            )
            np.testing.assert_allclose(
                plan.point_placements[0]["location"],
                feature_workflow.to_axis_frame(
                    [0.25 * 0.0254, 0.25 * 0.0254, 0.0]
                ),
            )
            np.testing.assert_allclose(
                plan.line_placements[0]["perimeter"],
                feature_workflow.to_axis_frame(
                    [[[0.0, 0.0, 0.0], [0.0254, 0.0, 0.0]]]
                ),
            )

    def test_prepare_and_execute_reuse_the_physics_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "body.grim"
            grid = {
                "frequencies_ghz": [1.0, 2.0],
                "azimuths_deg": [0.0, 90.0],
                "elevations_deg": [0.0],
                "axis_az_deg": 0.0,
                "axis_el_deg": 0.0,
                "roll_deg": 0.0,
            }
            _write_empty_base(base, grid)
            line_csv = root / "lines.csv"
            line_csv.write_text(
                LINE_HEADER
                + "gap_1,gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            dataset = root / "gap.grim"
            dataset.write_bytes(b"delta")
            profile = np.asarray([[1.0, 1.0], [1.0, -1.0]])
            line_placements = [{
                "delta": str(dataset),
                "kind": "delta",
                "declared_coherent_delta": True,
                "delta_sign": 1.0,
                "perimeter": np.asarray([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]),
                "segment_normals": np.asarray([[0.0, 0.0, 1.0]]),
            }]
            line_records = [{
                "schema": feature_workflow.LINE_PLACEMENT_SCHEMA,
                "line_id": "gap_1",
                "dataset_id": "gap",
                "segment_count": 1,
                "input_subtraction_order": "OPN-FRD (featured-clean)",
            }]

            request = feature_workflow.FeatureAssemblyRequest(
                base_grim="body.grim",
                output_grim="assembled.grim",
                coordinate_units="meters",
                line_locations_csv="lines.csv",
                line_datasets={"gap": "gap.grim"},
                base_dir=root,
                history="service test",
            )
            with (
                mock.patch.object(
                    feature_workflow,
                    "load_body_requested_radar_grid",
                    return_value=grid,
                ),
                mock.patch.object(
                    feature_workflow,
                    "load_body_profile_grim",
                    return_value=profile,
                ),
                mock.patch.object(
                    feature_workflow,
                    "prepare_line_placements",
                    return_value=(line_placements, line_records),
                ) as prepare_lines,
                mock.patch.object(
                    feature_workflow,
                    "prepare_point_placements",
                    return_value=([], []),
                ),
            ):
                plan = feature_workflow.prepare_feature_assembly(request)

            self.assertEqual(plan.base_path, base.resolve())
            self.assertEqual(plan.output_path, (root / "assembled.grim").resolve())
            self.assertEqual(
                plan.features_only_output_path,
                (root / "assembled_features_only.grim").resolve(),
            )
            self.assertTrue(plan.prepared_features_only_output_absent)
            self.assertIsNone(plan.prepared_features_only_output_sha256)
            self.assertEqual(plan.dataset_requirements.line_dataset_ids, ("gap",))
            self.assertEqual(plan.dataset_requirements.point_placement_count, 0)
            self.assertEqual(plan.dataset_requirements.line_path_count, 1)
            self.assertEqual(plan.dataset_requirements.line_segment_count, 1)
            self.assertEqual(plan.point_placements, [])
            self.assertEqual(
                plan.feature_provenance["placements"], line_records
            )
            self.assertEqual(
                plan.feature_provenance["request_schema"],
                feature_workflow.FEATURE_ASSEMBLY_REQUEST_SCHEMA,
            )
            self.assertEqual(
                plan.feature_provenance["line_phase_mapping_deg"],
                {
                    "TM": feature_workflow.PSI_HH_DEG,
                    "TE": feature_workflow.PSI_VV_DEG,
                },
            )
            self.assertFalse(
                plan.feature_provenance["model_scope"][
                    "body_feature_mutual_coupling"
                ]
            )
            self.assertEqual(
                line_placements[0]["delta_sign"], 1.0
            )
            prepare_lines.assert_called_once()

            expected_output = str((root / "assembled.grim").resolve())
            with mock.patch.object(
                feature_workflow,
                "add_features_to_monostatic_grim",
                return_value=expected_output,
            ) as add_features:
                saved = feature_workflow.execute_feature_assembly(plan)

            self.assertEqual(saved, expected_output)
            args, kwargs = add_features.call_args
            self.assertEqual(args, (str(base.resolve()), expected_output))
            # Execution receives a sealed deep snapshot. A callback or caller
            # mutating the public plan after validation must not alter the
            # arrays used to build the output.
            self.assertIsNot(kwargs["placements"], plan.line_placements)
            self.assertTrue(kwargs["expect_features_only_output_absent"])
            self.assertIsNone(
                kwargs["expected_features_only_output_sha256"]
            )
            np.testing.assert_allclose(
                kwargs["placements"][0]["perimeter"],
                plan.line_placements[0]["perimeter"],
            )
            np.testing.assert_allclose(
                kwargs["placements"][0]["segment_normals"],
                plan.line_placements[0]["segment_normals"],
            )
            self.assertEqual(kwargs["points"], plan.point_placements)
            self.assertTrue(kwargs["declared_coherent_base"])
            self.assertIsNot(
                kwargs["feature_provenance"], plan.feature_provenance
            )
            self.assertEqual(
                kwargs["feature_provenance"], plan.feature_provenance
            )
            self.assertEqual(kwargs["psi_tm_deg"], feature_workflow.PSI_HH_DEG)
            self.assertEqual(kwargs["psi_te_deg"], feature_workflow.PSI_VV_DEG)
            self.assertEqual(kwargs["history"], "service test")
            self.assertEqual(
                kwargs["expected_source_sha256"],
                plan.prepared_source_sha256,
            )

    def test_mapping_must_resolve_every_csv_dataset_id(self):
        class FlatSurface:
            def distance(self, points):
                return np.zeros(len(np.atleast_2d(points)))

            def normal(self, points):
                return np.tile([1.0, 0.0, 0.0], (len(np.atleast_2d(points)), 1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,antenna,0,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            (root / "fastener.grim").write_bytes(b"delta")
            with self.assertRaisesRegex(ValueError, "unknown dataset_id"):
                feature_workflow.prepare_point_placements(
                    None,
                    FlatSurface(),
                    coordinate_scale=1.0,
                    skin_limit_m=1.0,
                    wavelength_m=1.0,
                    normal_tolerance_deg=15.0,
                    locations_csv="points.csv",
                    datasets={"fastener": "fastener.grim"},
                    base_dir=root,
                    pattern_loader=lambda *_args, **_kwargs: object(),
                )


class AtomicSourceSnapshotTests(unittest.TestCase):
    @staticmethod
    def _grid():
        return {
            "frequencies_ghz": [1.0],
            "azimuths_deg": [0.0],
            "elevations_deg": [0.0],
            "axis_az_deg": 0.0,
            "axis_el_deg": 0.0,
            "roll_deg": 0.0,
        }

    def test_unchanged_prepared_source_snapshot_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            output = root / "assembled.grim"
            grid = self._grid()
            feature_sum.export_radar_grim(
                str(base), bor_result=None, placements=[], **grid
            )
            expected = {
                str(base.resolve()): feature_workflow.sha256_file(str(base))
            }

            saved = feature_sum.add_features_to_monostatic_grim(
                str(base),
                str(output),
                radar_grid=grid,
                expected_source_sha256=expected,
            )

            self.assertEqual(saved, str(output.resolve()))
            self.assertTrue(output.is_file())

    def test_source_change_during_execution_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            output = root / "assembled.grim"
            sentinel = b"previous valid assembly"
            output.write_bytes(sentinel)
            grid = self._grid()
            feature_sum.export_radar_grim(
                str(base), bor_result=None, placements=[], **grid
            )
            expected = {
                str(base.resolve()): feature_workflow.sha256_file(str(base))
            }
            original_save = grim_io._save_grim_npz

            def save_then_change_source(payload, path):
                saved = original_save(payload, path)
                base.write_bytes(base.read_bytes() + b"changed")
                return saved

            with mock.patch.object(
                grim_io,
                "_save_grim_npz",
                side_effect=save_then_change_source,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "base response changed during assembly"
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=grid,
                        expected_source_sha256=expected,
                    )

            self.assertEqual(output.read_bytes(), sentinel)
            self.assertFalse(any(root.glob(".assembled.grim.*.grim")))


class PlacementSafetyTests(unittest.TestCase):
    class _FlatSurface:
        def distance(self, points):
            return np.zeros(len(np.atleast_2d(points)), dtype=float)

        def normal(self, points):
            outward = feature_workflow.to_axis_frame([0.0, 0.0, 1.0])
            return np.tile(outward, (len(np.atleast_2d(points)), 1))

    def test_high_tolerance_cannot_admit_non_outward_normals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_dataset = root / "point.grim"
            line_dataset = root / "line.grim"
            point_dataset.touch()
            line_dataset.touch()
            for label, normal in (("inward", "0,0,-1"), ("orthogonal", "1,0,0")):
                point_csv = root / f"point_{label}.csv"
                point_csv.write_text(
                    POINT_HEADER
                    + f"p1,point,0.2,0.2,0,{normal},0,1,0\n",
                    encoding="utf-8",
                )
                with self.subTest(kind="point", normal=label):
                    with self.assertRaisesRegex(ValueError, "outward skin normal"):
                        feature_workflow.prepare_point_placements(
                            None,
                            self._FlatSurface(),
                            coordinate_scale=1.0,
                            skin_limit_m=1.0,
                            wavelength_m=1.0,
                            normal_tolerance_deg=180.0,
                            locations_csv=point_csv,
                            datasets={"point": point_dataset},
                            pattern_loader=lambda *_args, **_kwargs: object(),
                        )

                line_csv = root / f"line_{label}.csv"
                line_csv.write_text(
                    LINE_HEADER
                    + f"l1,line,1,0.1,0.1,0,0.6,0.1,0,{normal},{normal}\n",
                    encoding="utf-8",
                )
                with self.subTest(kind="line", normal=label):
                    with self.assertRaisesRegex(ValueError, "outward skin normal"):
                        feature_workflow.prepare_line_placements(
                            None,
                            self._FlatSurface(),
                            coordinate_scale=1.0,
                            skin_limit_m=1.0,
                            wavelength_m=1.0,
                            normal_tolerance_deg=180.0,
                            locations_csv=line_csv,
                            datasets={"line": line_dataset},
                        )

    def test_triangle_surface_reuses_one_nearest_query_per_point_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point_dataset = root / "point.grim"
            line_dataset = root / "line.grim"
            point_dataset.touch()
            line_dataset.touch()
            point_csv = root / "points.csv"
            point_csv.write_text(
                POINT_HEADER + "p1,point,0.2,0.2,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            line_csv = root / "lines.csv"
            line_csv.write_text(
                LINE_HEADER
                + "l1,line,1,0.1,0.1,0,0.6,0.1,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            triangles_cad = np.asarray([[[
                0.0, 0.0, 0.0
            ], [
                1.0, 0.0, 0.0
            ], [
                0.0, 1.0, 0.0
            ]]])
            surface = feature_workflow.TriangleSurface(
                feature_workflow.to_axis_frame(triangles_cad)
            )
            surface.nearest = mock.Mock(wraps=surface.nearest)

            feature_workflow.prepare_point_placements(
                None,
                surface,
                coordinate_scale=1.0,
                skin_limit_m=1.0e-6,
                wavelength_m=1.0,
                normal_tolerance_deg=15.0,
                locations_csv=point_csv,
                datasets={"point": point_dataset},
                pattern_loader=lambda *_args, **_kwargs: object(),
            )
            self.assertEqual(surface.nearest.call_count, 1)

            surface.nearest.reset_mock()
            feature_workflow.prepare_line_placements(
                None,
                surface,
                coordinate_scale=1.0,
                skin_limit_m=1.0e-6,
                wavelength_m=1.0,
                normal_tolerance_deg=15.0,
                locations_csv=line_csv,
                datasets={"line": line_dataset},
            )
            self.assertEqual(surface.nearest.call_count, 1)


class ApplicabilityGeometryTests(unittest.TestCase):
    def test_point_support_requires_every_exact_frequency(self):
        pattern = SimpleNamespace(
            frequencies=np.asarray([1.0, 3.0]),
            elevations=np.asarray([-90.0, 90.0]),
        )
        placement = {
            "pattern": pattern,
            "aperture_normal": np.asarray([0.0, 0.0, 1.0]),
            "roll_ref": np.asarray([1.0, 0.0, 0.0]),
        }
        with self.assertRaisesRegex(ValueError, "no exact 2 GHz"):
            feature_workflow._validate_point_requested_support(
                placement,
                np.asarray([[0.0, 0.0, 1.0]]),
                np.asarray([1.0, 2.0]),
                dataset_id="fastener",
            )

    def test_point_support_checks_installed_lit_elevation(self):
        pattern = SimpleNamespace(
            frequencies=np.asarray([1.0]),
            elevations=np.asarray([40.0, 90.0]),
        )
        placement = {
            "pattern": pattern,
            "aperture_normal": np.asarray([0.0, 0.0, 1.0]),
            "roll_ref": np.asarray([1.0, 0.0, 0.0]),
        }
        look = np.asarray([[np.cos(np.deg2rad(30.0)), 0.0,
                            np.sin(np.deg2rad(30.0))]])
        with self.assertRaisesRegex(ValueError, "elevation support"):
            feature_workflow._validate_point_requested_support(
                placement, look, np.asarray([1.0]), dataset_id="fastener"
            )

    def test_zero_illumination_is_recorded_and_warned_for_points_and_lines(self):
        pattern = SimpleNamespace(
            frequencies=np.asarray([1.0]),
            azimuths=np.asarray([0.0]),
            elevations=np.asarray([-90.0, 90.0]),
            channel_indices={"VV": 0, "HH": 1, "VH": 2},
            amplitude=np.zeros((1, 2, 1, 3), dtype=np.complex128),
        )
        point_placement = {
            "pattern": pattern,
            "location": np.asarray([0.0, 0.0, 0.0]),
            "aperture_normal": np.asarray([1.0, 0.0, 0.0]),
            "roll_ref": np.asarray([0.0, 1.0, 0.0]),
        }
        point_record = {
            "placement_id": "hidden_fastener",
            "dataset_id": "fastener",
            "dataset": "fastener_delta.grim",
            "dataset_sha256": "1" * 64,
        }
        line_placement = {
            "perimeter": np.asarray([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            "segment_normals": np.asarray([[[1.0, 0.0, 0.0],
                                               [1.0, 0.0, 0.0]]]),
            "max_piece_length_m": 0.1,
        }
        line_record = {
            "line_id": "hidden_seal",
            "dataset_id": "seal",
            "dataset": "seal_delta.grim",
            "dataset_sha256": "2" * 64,
        }
        coefficient = SimpleNamespace(
            frequency_ghz=1.0,
            phi_deg=np.asarray([-180.0, 180.0]),
            dA_tm=np.zeros(2, dtype=np.complex128),
            dA_te=np.zeros(2, dtype=np.complex128),
        )
        radar_grid = {
            "frequencies_ghz": np.asarray([1.0]),
            "azimuths_deg": np.asarray([0.0]),
            "elevations_deg": np.asarray([0.0]),
            "axis_az_deg": 0.0,
            "axis_el_deg": 0.0,
            "roll_deg": 0.0,
        }
        with mock.patch.object(
            feature_workflow,
            "load_feature_library_manifest",
            return_value=(None, []),
        ), mock.patch.object(
            feature_workflow,
            "feature_response_content_sha256",
            return_value="3" * 64,
        ), mock.patch.object(
            feature_workflow,
            "_load_grim",
            return_value={"frequencies": np.asarray([1.0])},
        ), mock.patch.object(
            feature_workflow,
            "load_seam_from_grim",
            return_value=coefficient,
        ):
            _contracts, warnings, _hashes, _absent = (
                feature_workflow._apply_feature_library_contracts(
                    line_placements=[line_placement],
                    line_records=[line_record],
                    point_placements=[point_placement],
                    point_records=[point_record],
                    radar_grid=radar_grid,
                    require_manifests=False,
                )
            )

        self.assertEqual(point_record["requested_look_count"], 1)
        self.assertEqual(point_record["illuminated_requested_look_count"], 0)
        self.assertEqual(line_record["requested_look_count"], 1)
        self.assertEqual(line_record["illuminated_requested_look_count"], 0)
        self.assertEqual(
            line_record["required_cut_angle_ranges_deg"][0][
                "illuminated_requested_look_count"
            ],
            0,
        )
        warning_text = "\n".join(warnings)
        self.assertIn("Point 'hidden_fastener' has zero illuminated", warning_text)
        self.assertIn("Line 'hidden_seal' has zero illuminated", warning_text)
        self.assertIn("one-time release-warning waiver", warning_text)

    def test_line_metrics_use_solver_frame_and_exact_normal_turn_radius(self):
        placement = {
            "perimeter": np.asarray([[[0.0, 0.0, 0.0],
                                       [2.0, 0.0, 0.0]]]),
            "segment_normals": np.asarray([[[0.0, 0.0, 1.0],
                                              [0.0, 1.0, 0.0]]]),
            "max_piece_length_m": 0.1,
        }
        look = np.asarray([[0.0, 0.0, 1.0]])
        with mock.patch.object(
            feature_workflow,
            "prepare_perimeter_frame",
            wraps=feature_workflow.prepare_perimeter_frame,
        ) as prepare:
            metrics = feature_workflow._line_applicability_metrics(
                placement, look, requested_frequencies_ghz=[1.0, 10.0]
            )
        self.assertEqual(prepare.call_count, 1)
        self.assertAlmostEqual(
            metrics["estimated_min_along_line_normal_turn_radius_m"],
            1.0,
            places=12,
        )
        self.assertTrue(metrics["along_line_normal_turn_detected"])
        self.assertEqual(len(metrics["required_cut_angle_ranges_deg"]), 2)
        for support in metrics["required_cut_angle_ranges_deg"]:
            self.assertGreater(support["lit_query_count"], 0)
            self.assertGreaterEqual(support["minimum_deg"], 0.0)
            self.assertLessEqual(support["maximum_deg"], 180.0)

    def test_line_vertex_normal_jump_and_path_turn_are_explicit(self):
        placement = {
            "perimeter": np.asarray([
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ]),
            "segment_normals": np.asarray([
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]),
            "max_piece_length_m": 0.1,
        }
        metrics = feature_workflow._line_applicability_metrics(
            placement,
            np.asarray([[0.0, 0.0, 1.0]]),
            requested_frequencies_ghz=[1.0],
        )
        self.assertEqual(
            metrics["estimated_min_along_line_normal_turn_radius_m"], 0.0
        )
        self.assertAlmostEqual(
            metrics["maximum_shared_vertex_normal_jump_deg"], 90.0
        )
        self.assertAlmostEqual(metrics["maximum_path_vertex_turn_deg"], 90.0)

    def test_one_line_cannot_fold_back_inside_its_own_footprint(self):
        segments = np.asarray([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
            [[1.0, 0.1, 0.0], [0.0, 0.1, 0.0]],
        ])
        overlap = feature_workflow._line_self_footprint_overlap(
            segments, 0.06
        )
        self.assertIsNotNone(overlap)
        self.assertEqual(overlap[:2], (0, 2))
        self.assertAlmostEqual(overlap[2], 0.1)

    def test_near_retrace_is_nonlocal_but_ordinary_corner_is_not(self):
        nearly_retraced = np.asarray([
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [1.0, 0.001, 0.0]],
        ])
        overlap = feature_workflow._line_self_footprint_overlap(
            nearly_retraced, 0.1
        )
        self.assertIsNotNone(overlap)
        self.assertEqual(overlap[:2], (0, 1))

        right_angle = np.asarray([
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
        ])
        self.assertIsNone(feature_workflow._line_self_footprint_overlap(
            right_angle, 0.1
        ))

    def test_bor_binding_checks_facet_interiors_not_only_vertices(self):
        profile = np.asarray([[1.0, -1.0], [1.0, 1.0]])
        # Every vertex lies exactly on the unit cylinder, but the long chord
        # and triangle interior cut deeply through it. Vertex-only validation
        # would accept this unrelated/coarse shadow mesh.
        triangle = np.asarray([[
            [1.0, 0.0, -1.0],
            [-0.5, np.sqrt(0.75), -1.0],
            [1.0, 0.0, 1.0],
        ]])
        surface = feature_workflow.TriangleSurface(triangle)
        with self.assertRaisesRegex(ValueError, "sampled facet point"):
            feature_workflow._validate_bor_surface_agreement(
                profile,
                surface,
                skin_limit_m=1.0e-3,
                shadow_requested=False,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
