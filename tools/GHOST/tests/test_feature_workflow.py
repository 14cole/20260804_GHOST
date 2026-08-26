#!/usr/bin/env python3
"""Focused acceptance tests for the reusable feature-assembly service."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
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


class RequestPlanTests(unittest.TestCase):
    def _prepare_minimal_line_plan(self, root):
        base = root / "body.grim"
        base.write_bytes(b"prepared clean body")
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
        ):
            return feature_workflow.prepare_feature_assembly(request)

    @staticmethod
    def _external_plane_case(root):
        base = root / "body.grim"
        surface = root / "body.facet"
        coordinates = root / "lines.csv"
        response = root / "gap.grim"
        base.write_bytes(b"external coherent body")
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
        payload = {
            "frequencies": np.asarray([1.0]),
            "azimuths": np.asarray([0.0]),
            "elevations": np.asarray([0.0]),
        }
        triangles = np.asarray([[[-1.0, -1.0, 0.0],
                                 [1.0, -1.0, 0.0],
                                 [0.0, 1.0, 0.0]]])
        return request, payload, triangles, surface, coordinates

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
                )
            with self.assertRaisesRegex(TypeError, "sequence of complete IDs"):
                feature_workflow.prepare_feature_input_preview(
                    point_locations_csv="points.csv",
                    enabled_point_placement_ids="p1",
                    base_dir=root,
                )
            clean_preview = feature_workflow.prepare_feature_input_preview(
                point_locations_csv="points.csv",
                enabled_point_placement_ids=(),
                base_dir=root,
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
                )

    def test_all_disabled_request_fails_before_response_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "body.grim").touch()
            (root / "points.csv").write_text(
                POINT_HEADER + "p1,fastener,0,0,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            request = feature_workflow.FeatureAssemblyRequest(
                base_grim="body.grim",
                output_grim="assembled.grim",
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
            (root / "body.grim").touch()
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
            payload = {
                "frequencies": np.asarray([1.0]),
                "azimuths": np.asarray([0.0, 90.0]),
                "elevations": np.asarray([0.0]),
            }
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
                    return_value=surface_in,
                ),
                mock.patch.object(
                    feature_workflow,
                    "prepare_point_pattern",
                    return_value="prepared point pattern",
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
            base.touch()
            line_csv = root / "lines.csv"
            line_csv.write_text(
                LINE_HEADER
                + "gap_1,gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            dataset = root / "gap.grim"
            dataset.write_bytes(b"delta")
            grid = {
                "frequencies_ghz": [1.0, 2.0],
                "azimuths_deg": [0.0, 90.0],
                "elevations_deg": [0.0],
                "axis_az_deg": 0.0,
                "axis_el_deg": 0.0,
                "roll_deg": 0.0,
            }
            profile = np.asarray([[1.0, 1.0], [1.0, -1.0]])
            line_placements = [{
                "delta": str(dataset),
                "kind": "delta",
                "declared_coherent_delta": True,
                "delta_sign": 1.0,
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
            self.assertIs(kwargs["placements"], plan.line_placements)
            self.assertIs(kwargs["points"], plan.point_placements)
            self.assertTrue(kwargs["declared_coherent_base"])
            self.assertIs(kwargs["feature_provenance"], plan.feature_provenance)
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
                    RuntimeError, "source changed during execution"
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
