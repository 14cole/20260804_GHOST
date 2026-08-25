#!/usr/bin/env python3
"""Focused acceptance tests for the reusable feature-assembly service."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

import feature_workflow  # noqa: E402


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
                + "gap_1,panel_gap,2,1,0,0,1,1,0,0,0,1,0,0,1\n",
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

    def test_output_must_not_overwrite_clean_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "body.grim").touch()
            request = feature_workflow.FeatureAssemblyRequest(
                base_grim="body.grim",
                output_grim="body.grim",
                line_locations_csv="lines.csv",
                line_datasets={"gap": "gap.grim"},
                base_dir=root,
            )
            with self.assertRaisesRegex(ValueError, "must differ"):
                feature_workflow.prepare_feature_assembly(request)

    def test_plan_carries_full_cad_preview_geometry_in_meters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "body.grim").touch()
            (root / "body.facet").touch()
            (root / "gap.grim").write_bytes(b"line delta")
            (root / "antenna.grim").write_bytes(b"point delta")
            (root / "lines.csv").write_text(
                LINE_HEADER
                + "gap_1,gap,1,0,0,0,1,0,0,0,0,1,0,0,1\n",
                encoding="utf-8",
            )
            (root / "points.csv").write_text(
                POINT_HEADER
                + "p1,antenna,0.25,0.25,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            surface_in = np.asarray([[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]])
            request = feature_workflow.FeatureAssemblyRequest(
                base_grim="body.grim",
                output_grim="assembled.grim",
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
                "dataset_id": "gap",
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
            self.assertEqual(plan.point_placements, [])
            self.assertEqual(
                plan.feature_provenance["placements"], line_records
            )
            self.assertEqual(
                plan.feature_provenance["request_schema"],
                feature_workflow.FEATURE_ASSEMBLY_REQUEST_SCHEMA,
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
            self.assertEqual(kwargs["history"], "service test")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
