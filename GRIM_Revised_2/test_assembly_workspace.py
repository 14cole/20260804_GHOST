from __future__ import annotations

import os
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Must be selected before a QApplication is created on headless CI runners.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from assembly_workspace import (  # noqa: E402
    BODY_RENDER_MODES,
    DISPLAY_UNIT_SPECS,
    AssemblySceneCanvas,
    AssemblySceneModel,
    AssemblyWorkspace,
    FEATURE_PREVIEW_ROOT_KEY,
    FeatureBuildResult,
    GUI_AVAILABLE,
    decimate_triangles_for_display,
    display_unit_spec,
    feature_preview_group_id,
    format_length_tick,
    normalize_body_render_mode,
    revolve_bor_profile_cad,
    triangle_detail_cap,
)


class AssemblyGeometryTests(unittest.TestCase):
    def test_display_unit_helpers_convert_ticks_without_geometry_conversion(self):
        self.assertEqual(tuple(DISPLAY_UNIT_SPECS), ("Meters", "Inches", "Feet"))
        self.assertEqual(display_unit_spec("meters"), ("m", 1.0))
        self.assertAlmostEqual(float(format_length_tick(0.0254, "Inches")), 1.0)
        self.assertAlmostEqual(float(format_length_tick(0.3048, "Feet")), 1.0)
        with self.assertRaisesRegex(ValueError, "Meters"):
            display_unit_spec("yards")

    def test_named_render_and_detail_choices_are_bounded(self):
        self.assertEqual(BODY_RENDER_MODES, ("Solid", "Solid + edges", "Wireframe"))
        self.assertEqual(normalize_body_render_mode("wireFRAME"), "Wireframe")
        self.assertEqual(triangle_detail_cap("Fast"), 4_000)
        self.assertEqual(triangle_detail_cap("Balanced"), 12_000)
        self.assertEqual(triangle_detail_cap("High"), 30_000)
        with self.assertRaisesRegex(ValueError, "Fast"):
            triangle_detail_cap("Full")

    def test_bor_profile_revolves_about_cad_nose_axis(self):
        profile = np.asarray(
            [
                [0.0, 2.0],
                [0.5, 1.0],
                [0.5, -1.0],
                [0.0, -2.0],
            ]
        )
        triangles = revolve_bor_profile_cad(
            profile, circumferential_samples=12
        )
        vertices = triangles.reshape(-1, 3)

        # Axial profile z becomes CAD y; the radial plane is CAD x/z.
        self.assertAlmostEqual(float(np.min(vertices[:, 1])), -2.0)
        self.assertAlmostEqual(float(np.max(vertices[:, 1])), 2.0)
        self.assertAlmostEqual(float(np.min(vertices[:, 0])), -0.5)
        self.assertAlmostEqual(float(np.max(vertices[:, 0])), 0.5)
        self.assertAlmostEqual(float(np.min(vertices[:, 2])), -0.5)
        self.assertAlmostEqual(float(np.max(vertices[:, 2])), 0.5)

        area_vectors = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        self.assertTrue(np.all(np.linalg.norm(area_vectors, axis=1) > 0.0))

    def test_bor_profile_rejects_negative_radius(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            revolve_bor_profile_cad([[0.0, 1.0], [-0.1, 0.0], [0.0, -1.0]])

    def test_display_decimation_is_deterministic_and_preserves_extrema(self):
        triangles = []
        for index in range(101):
            x = float(index - 50)
            y = float((index % 11) - 5)
            z = float((index % 17) - 8)
            triangles.append(
                [[x, y, z], [x + 0.1, y, z], [x, y + 0.1, z + 0.1]]
            )
        source = np.asarray(triangles, dtype=float)
        first = decimate_triangles_for_display(source, 13)
        second = decimate_triangles_for_display(source, 13)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (13, 3, 3))

        source_vertices = source.reshape(-1, 3)
        proxy_vertices = first.reshape(-1, 3)
        for axis in range(3):
            self.assertEqual(
                float(np.min(proxy_vertices[:, axis])),
                float(np.min(source_vertices[:, axis])),
            )
            self.assertEqual(
                float(np.max(proxy_vertices[:, axis])),
                float(np.max(source_vertices[:, axis])),
            )

    def test_feature_preview_group_ids_are_stable_and_unambiguous(self):
        self.assertEqual(
            feature_preview_group_id("body"),
            "feature-assembly/body",
        )
        self.assertEqual(
            feature_preview_group_id("points", "fastener / M4"),
            "feature-assembly/points/fastener%20%2F%20M4",
        )
        self.assertEqual(
            feature_preview_group_id("lines", "gap%forward"),
            "feature-assembly/lines/gap%25forward",
        )
        with self.assertRaisesRegex(ValueError, "nonempty"):
            feature_preview_group_id("points", "")
        with self.assertRaisesRegex(ValueError, "body"):
            feature_preview_group_id("body", "unexpected")


class AssemblySceneModelTests(unittest.TestCase):
    def setUp(self):
        self.model = AssemblySceneModel()

    def test_stable_group_api_and_visibility_bounds(self):
        events = []
        self.model.add_listener(lambda event, group_id: events.append((event, group_id)))
        body = self.model.add_bor_profile(
            "body:bor",
            [[0.0, 1.0], [0.25, 0.0], [0.0, -1.0]],
            circumferential_samples=8,
        )
        points = self.model.add_points(
            "points:fasteners", [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
        )
        self.model.add_lines(
            "lines:gaps",
            np.asarray(
                [
                    [[0.0, -0.5, 0.25], [0.0, 0.5, 0.25]],
                    [[0.1, -0.5, 0.2], [0.1, 0.5, 0.2]],
                ]
            ),
        )

        self.assertEqual(
            self.model.group_ids,
            ("body:bor", "points:fasteners", "lines:gaps"),
        )
        self.assertEqual(body.kind, "surface")
        self.assertTrue(body.display_only)
        self.assertEqual(points.source_count, 2)
        self.assertEqual(float(self.model.bounds()[1, 0]), 5.0)

        self.model.set_group_visible("points:fasteners", False)
        self.assertFalse(self.model.group("points:fasteners").visible)
        self.assertLess(float(self.model.bounds()[1, 0]), 1.0)
        self.assertIn(("visibility", "points:fasteners"), events)

        with self.assertRaisesRegex(KeyError, "unknown"):
            self.model.set_group_visible("points:missing", False)

    def test_body_proxy_does_not_claim_to_be_the_physics_mesh(self):
        source = np.asarray(
            [
                [[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 0.0, 1.0]]
                for i in range(100)
            ]
        )
        group = self.model.add_body_triangles(
            "body:stl", source, max_triangles=9
        )
        self.assertEqual(group.source_count, 100)
        self.assertEqual(group.display_count, 9)
        self.assertEqual(group.geometry.shape, (9, 3, 3))
        self.assertTrue(group.display_only)
        # Bounds come from the complete source, not the display proxy contract.
        np.testing.assert_array_equal(
            group.bounds_m,
            [[0.0, 0.0, 0.0], [99.0, 1.0, 1.0]],
        )

    def test_surface_detail_and_style_preserve_source_contract_and_visibility(self):
        source = np.asarray(
            [
                [[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 0.0, 1.0]]
                for i in range(200)
            ]
        )
        group = self.model.add_body_triangles(
            "body:detail", source, max_triangles=13, visible=False
        )
        original_bounds = np.array(group.bounds_m, copy=True)
        proxy_13 = group.geometry

        self.model.set_surface_detail("body:detail", 7)
        self.assertEqual(group.display_count, 7)
        self.assertEqual(group.source_count, 200)
        self.assertFalse(group.visible)
        np.testing.assert_array_equal(group.bounds_m, original_bounds)

        self.model.set_surface_detail("body:detail", 13)
        self.assertIs(group.geometry, proxy_13)
        geometry_before_style = group.geometry
        self.model.set_surface_rendering("body:detail", "Wireframe", 0.4)
        self.assertIs(group.geometry, geometry_before_style)
        self.assertEqual(group.style["render_mode"], "Wireframe")
        self.assertEqual(group.style["alpha"], 0.4)
        self.assertFalse(group.visible)

    def test_large_surface_retains_only_bounded_master_but_full_counts_and_bounds(self):
        count = 30_101
        source = np.zeros((count, 3, 3), dtype=float)
        source[:, :, 0] = np.arange(count, dtype=float)[:, None]
        source[:, 1, 1] = 1.0
        source[:, 2, 2] = 1.0

        group = self.model.add_body_triangles(
            "body:large", source, max_triangles=12_000
        )
        proxy_12k = group.geometry

        self.assertEqual(group.source_count, count)
        self.assertEqual(len(group.master_geometry), 30_000)
        self.assertEqual(group.display_count, 12_000)
        self.assertFalse(group.master_geometry.flags.writeable)
        self.assertFalse(np.shares_memory(group.master_geometry, source))
        np.testing.assert_array_equal(
            group.bounds_m,
            [[0.0, 0.0, 0.0], [float(count - 1), 1.0, 1.0]],
        )

        self.model.set_surface_detail("body:large", 4_000)
        self.assertEqual(group.display_count, 4_000)
        proxy_4k = group.geometry
        self.model.set_surface_detail("body:large", 30_000)
        self.assertEqual(group.display_count, 30_000)
        self.assertIs(group.geometry, group.master_geometry)
        self.model.set_surface_detail("body:large", 12_000)
        self.assertIs(group.geometry, proxy_12k)
        self.model.set_surface_detail("body:large", 4_000)
        self.assertIs(group.geometry, proxy_4k)

    def test_clear_and_replace_keep_string_identity(self):
        self.model.add_points("points:a", [[0.0, 0.0, 0.0]])
        self.model.add_points("points:a", [[1.0, 2.0, 3.0]])
        self.assertEqual(self.model.group_ids, ("points:a",))
        np.testing.assert_array_equal(
            self.model.group("points:a").geometry, [[1.0, 2.0, 3.0]]
        )
        self.model.clear()
        self.assertEqual(self.model.group_ids, ())


@unittest.skipUnless(GUI_AVAILABLE, "PySide6/Matplotlib GUI dependencies unavailable")
class AssemblyGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_canvas_artist_visibility_and_fit(self):
        canvas = AssemblySceneCanvas()
        self.assertEqual(canvas.preview_state, "empty")
        self.assertIn("Nothing to preview", canvas.feedback_text)
        np.testing.assert_allclose(
            canvas.figure.get_facecolor()[:3],
            np.asarray([11.0, 18.0, 34.0]) / 255.0,
        )
        self.assertEqual(canvas.axes.xaxis.label.get_color(), "#dbeafe")
        canvas.add_points("points:a", [[1.0, 2.0, 3.0]])
        canvas.add_lines("lines:a", [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        original_points = np.array(canvas.model.group("points:a").geometry, copy=True)
        original_limits = np.asarray(
            [canvas.axes.get_xlim(), canvas.axes.get_ylim(), canvas.axes.get_zlim()]
        )
        canvas.set_display_units("Inches")
        self.assertEqual(canvas.axes.xaxis.label.get_text(), "X right (in)")
        self.assertEqual(canvas.axes.xaxis.get_major_formatter()(0.0254), "1")
        np.testing.assert_array_equal(
            canvas.model.group("points:a").geometry, original_points
        )
        np.testing.assert_allclose(
            [canvas.axes.get_xlim(), canvas.axes.get_ylim(), canvas.axes.get_zlim()],
            original_limits,
        )
        canvas.set_group_visible("points:a", False)
        self.assertFalse(canvas.model.group("points:a").visible)
        self.assertFalse(canvas._artists["points:a"].get_visible())
        canvas.set_group_visible("lines:a", False)
        self.assertEqual(canvas.preview_state, "hidden")
        self.assertIn("Show box", canvas.feedback_text)
        canvas.set_group_visible("lines:a", True)
        self.assertEqual(canvas.preview_state, "ready")
        canvas.fit_visible()
        canvas.draw()
        canvas._detach_model_listener()
        canvas.model.add_points("points:detached", [[0.0, 0.0, 0.0]])
        self.assertNotIn("points:detached", canvas._artists)

    def test_drag_lod_restores_selected_surface_proxy(self):
        source = np.zeros((4_100, 3, 3), dtype=float)
        source[:, :, 0] = np.arange(4_100, dtype=float)[:, None]
        source[:, 1, 1] = 1.0
        source[:, 2, 2] = 1.0
        model = AssemblySceneModel()
        group = model.add_body_triangles(
            "body:interactive", source, max_triangles=30_000
        )
        canvas = AssemblySceneCanvas(model=model)

        canvas._begin_interaction_lod(SimpleNamespace(inaxes=canvas.axes))
        self.assertEqual(group.display_count, 4_000)
        self.assertTrue(canvas._lod_artist.get_visible())
        self.assertEqual(group.detail_cap, 30_000)

        canvas._end_interaction_lod(None)
        self.assertEqual(group.display_count, 4_100)
        self.assertFalse(canvas._lod_artist.get_visible())
        self.assertEqual(group.detail_cap, 30_000)

    def test_prepopulated_scene_is_drawn_and_fitted_on_canvas_creation(self):
        model = AssemblySceneModel()
        model.add_bor_profile(
            "body:bor",
            [[0.0, 1.0], [0.25, 0.0], [0.0, -1.0]],
            circumferential_samples=8,
        )

        canvas = AssemblySceneCanvas(model=model)

        self.assertEqual(canvas.preview_state, "ready")
        self.assertIn("body:bor", canvas._artists)
        self.assertTrue(canvas._artists["body:bor"].get_visible())

    def test_workspace_service_hook_and_deferred_tree_visibility(self):
        from assembly_tree import AssemblyTreePanel, _TYPE_ROOT

        panel = AssemblyTreePanel()
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)
        workspace.set_group_visible("points:later", False, defer_unknown=True)
        group = workspace.add_points("points:later", [[0.0, 0.0, 0.0]])
        self.assertFalse(group.visible)

        item = panel.tree._make_node("Fasteners", _TYPE_ROOT, edit=False)
        workspace.bind_tree_item_groups(item, "points:later")
        panel.tree.set_item_preview_visible(item, True)
        self.assertTrue(workspace.scene_model.group("points:later").visible)
        panel.tree.set_item_preview_visible(item, False)
        self.assertFalse(workspace.scene_model.group("points:later").visible)

        emitted = []
        workspace.feature_built.connect(
            lambda name, payload, history: emitted.append((name, payload, history))
        )
        marker = object()
        workspace.set_feature_service(
            lambda request: FeatureBuildResult("assembled", marker, str(request))
        )
        result = workspace.request_feature_build("request-1")
        self.assertIsNotNone(result)
        self.assertEqual(emitted, [("assembled", marker, "request-1")])

    def test_workspace_separates_placement_from_dataset_combination(self):
        from PySide6.QtWidgets import QLabel
        from assembly_tree import AssemblyTreePanel

        panel = AssemblyTreePanel()
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)

        self.assertEqual(workspace.left_tabs.count(), 2)
        self.assertEqual(workspace.left_tabs.tabText(0), "Place Features")
        self.assertEqual(
            workspace.left_tabs.tabText(1), "Combine Datasets / Visibility"
        )
        self.assertIs(panel.parentWidget(), workspace.combine_visibility_tab)
        self.assertTrue(workspace.cmb_display_units.isEnabled())
        self.assertFalse(workspace.cmb_body_render.isEnabled())
        self.assertFalse(workspace.sld_body_opacity.isEnabled())
        self.assertFalse(workspace.cmb_triangle_detail.isEnabled())
        self.assertFalse(workspace.chk_interaction_lod.isEnabled())

        controls = QLabel("feature controls")
        workspace.set_feature_controls(controls)
        self.assertIs(
            workspace.left_tabs.currentWidget(), workspace.place_features_tab
        )
        self.assertFalse(workspace.feature_controls_host.isHidden())
        self.assertIs(controls.parentWidget(), workspace.feature_controls_host)

    @staticmethod
    def _feature_plan():
        return SimpleNamespace(
            surface_triangles_cad_m=np.asarray(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                ]
            ),
            body_profile_rho_z_m=None,
            point_locations_cad_m={
                "antenna": np.asarray([[0.5, 0.5, 0.2]]),
                "fastener / M4": np.asarray(
                    [[0.1, 0.1, 0.0], [0.9, 0.1, 0.0]]
                ),
            },
            line_paths_cad_m={
                "gap main": {
                    "path-b": np.asarray([[0.0, 0.8, 0.0], [1.0, 0.8, 0.0]]),
                    "path-a": np.asarray([[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]]),
                },
                "panel": {
                    "edge": np.asarray([[0.2, 0.0, 0.0], [0.2, 1.0, 0.0]])
                },
            },
        )

    @staticmethod
    def _bor_feature_plan():
        return SimpleNamespace(
            surface_triangles_cad_m=None,
            body_profile_rho_z_m=np.asarray(
                [[0.0, 1.0], [0.3, 0.5], [0.3, -0.5], [0.0, -1.0]]
            ),
            point_locations_cad_m={
                "antenna": np.asarray([[0.0, 0.25, 0.3]])
            },
            line_paths_cad_m={
                "panel gap": {
                    "gap-1": np.asarray(
                        [[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]]
                    )
                }
            },
        )

    def test_feature_plan_builds_typed_hierarchy_and_branch_visibility(self):
        from assembly_tree import AssemblyTreePanel

        panel = AssemblyTreePanel()
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)
        root = workspace.load_feature_preview(self._feature_plan())

        expected_ids = (
            feature_preview_group_id("body"),
            feature_preview_group_id("points", "antenna"),
            feature_preview_group_id("points", "fastener / M4"),
            feature_preview_group_id("lines", "gap main"),
            feature_preview_group_id("lines", "panel"),
        )
        self.assertEqual(workspace.group_ids, expected_ids)
        self.assertEqual(panel.tree.item_purpose(root), "preview")
        self.assertEqual(panel.tree.item_preview_key(root), FEATURE_PREVIEW_ROOT_KEY)
        self.assertEqual(root.text(1), "")
        self.assertEqual(root.text(0), "Feature Assembly")
        self.assertEqual(root.childCount(), 3)

        body_item, point_branch, line_branch = (
            root.child(0), root.child(1), root.child(2)
        )
        self.assertIn("2 triangles", body_item.text(0))
        self.assertEqual(point_branch.text(0), "Point Features (3)")
        self.assertEqual(line_branch.text(0), "Line Features (3)")
        self.assertEqual(point_branch.childCount(), 2)
        self.assertEqual(line_branch.childCount(), 2)
        self.assertIn("1 point", point_branch.child(0).text(0))
        self.assertIn("2 points", point_branch.child(1).text(0))
        self.assertIn("2 lines", line_branch.child(0).text(0))

        panel.tree.set_item_preview_visible(point_branch, False)
        for group_id in expected_ids[1:3]:
            self.assertFalse(workspace.scene_model.group(group_id).visible)
        for group_id in (expected_ids[0], *expected_ids[3:]):
            self.assertTrue(workspace.scene_model.group(group_id).visible)

    def test_bor_body_points_and_lines_are_previewed_together(self):
        from assembly_tree import AssemblyTreePanel

        workspace = AssemblyWorkspace(assembly_tree_panel=AssemblyTreePanel())

        workspace.load_feature_preview(self._bor_feature_plan())

        expected = {
            feature_preview_group_id("body"),
            feature_preview_group_id("points", "antenna"),
            feature_preview_group_id("lines", "panel gap"),
        }
        self.assertEqual(set(workspace.group_ids), expected)
        self.assertTrue(
            all(workspace.scene_model.group(key).visible for key in expected)
        )
        self.assertEqual(workspace.scene_canvas.preview_state, "ready")
        self.assertIn("BoR body", workspace.lbl_status.text())
        self.assertIn("1 point placement", workspace.lbl_status.text())
        self.assertIn("1 line path", workspace.lbl_status.text())
        self.assertIn("never the assembled RCS", workspace.lbl_status.text())
        self.assertTrue(workspace.cmb_body_render.isEnabled())
        self.assertEqual(
            workspace.cmb_triangle_detail.currentData(), "Balanced"
        )
        self.assertIn(" body triangles shown", workspace.lbl_body_detail.text())

        body_id = feature_preview_group_id("body")
        body = workspace.scene_model.group(body_id)
        original_geometry = body.geometry
        original_visibility = body.visible
        workspace.cmb_display_units.setCurrentIndex(1)
        self.assertEqual(workspace.scene_canvas.display_units, "Inches")
        self.assertIs(body.geometry, original_geometry)
        workspace.cmb_body_render.setCurrentText("Wireframe")
        workspace.sld_body_opacity.setValue(40)
        workspace._apply_body_rendering()
        self.assertEqual(body.style["render_mode"], "Wireframe")
        self.assertEqual(body.style["alpha"], 0.4)
        self.assertEqual(body.visible, original_visibility)
        workspace.scene_canvas.draw()
        artist = workspace.scene_canvas._artists[body_id]
        facecolors = np.asarray(artist.get_facecolor())
        edgecolors = np.asarray(artist.get_edgecolor())
        self.assertTrue(len(facecolors) == 0 or np.all(facecolors[:, 3] == 0.0))
        self.assertGreater(len(edgecolors), 0)
        self.assertTrue(np.allclose(edgecolors[:, 3], 0.4))
        self.assertIn("inches", workspace.lbl_body_detail.text())
        self.assertIn("Original geometry is unchanged", workspace.lbl_body_detail.text())

    def test_input_preview_and_stale_state_are_unmistakable_but_nonmutating(self):
        from assembly_tree import AssemblyTreePanel

        workspace = AssemblyWorkspace(assembly_tree_panel=AssemblyTreePanel())
        plan = self._bor_feature_plan()
        plan.preview_stage = "input"

        workspace.load_feature_preview(plan)
        original_ids = workspace.group_ids
        original_geometry = {
            key: workspace.scene_model.group(key).geometry
            for key in original_ids
        }

        self.assertEqual(workspace.scene_canvas.preview_stage, "input")
        self.assertIn("NOT PHYSICS-VALIDATED", workspace.scene_canvas._stage_artist.get_text())
        self.assertIn("not physics-validated", workspace.lbl_status.text())

        workspace.mark_preview_stale("Point CSV changed.")

        self.assertEqual(workspace.scene_canvas.preview_stage, "stale")
        self.assertIn("STALE PREVIEW", workspace.scene_canvas._stage_artist.get_text())
        self.assertIn("Point CSV changed", workspace.lbl_status.text())
        self.assertEqual(workspace.group_ids, original_ids)
        for key, geometry in original_geometry.items():
            self.assertIs(workspace.scene_model.group(key).geometry, geometry)

    def test_invalid_prepared_preview_gives_visible_error_and_cleans_scene(self):
        from assembly_tree import AssemblyTreePanel

        workspace = AssemblyWorkspace(assembly_tree_panel=AssemblyTreePanel())
        invalid = self._feature_plan()
        invalid.point_locations_cad_m["antenna"] = np.asarray(
            [[np.nan, 0.0, 0.0]]
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            workspace.load_feature_preview(invalid)

        self.assertEqual(workspace.group_ids, ())
        self.assertEqual(workspace.scene_canvas.preview_state, "error")
        self.assertIn("Preview unavailable", workspace.scene_canvas.feedback_text)
        self.assertIn("no assembly result was changed", workspace.lbl_status.text())

    def test_preview_is_excluded_from_response_build_and_serialization(self):
        from assembly_tree import (
            AssemblyTree,
            _TYPE_ROOT,
            _item_to_dict,
            build_assembly_grid,
        )
        from grim_dataset import RcsGrid

        tree = AssemblyTree()
        response_root = tree._make_node("Response", _TYPE_ROOT, edit=False)
        grid = RcsGrid(
            [0.0], [0.0], [1.0], ["VV"],
            rcs=np.ones((1, 1, 1, 1), dtype=np.complex128),
        )
        response_root.addChild(tree._make_leaf("body-response", grid))

        preview = tree.add_preview_root("Preview", stable_key="test-preview")
        tree.invisibleRootItem().removeChild(preview)
        response_root.addChild(preview)  # simulate malformed/mixed legacy state

        with self.assertRaisesRegex(ValueError, "preview-only"):
            build_assembly_grid(preview)
        combined, _history = build_assembly_grid(response_root)
        self.assertIsNotNone(combined)
        np.testing.assert_allclose(combined.rcs_power, grid.rcs_power)
        np.testing.assert_allclose(combined.rcs_phase, grid.rcs_phase)
        self.assertIsNone(_item_to_dict(preview))
        serialized = _item_to_dict(response_root)
        self.assertIsNotNone(serialized)
        self.assertEqual(
            [child["name"] for child in serialized["children"]],
            ["body-response"],
        )

    def test_workspace_clear_removes_preview_but_preserves_response_tree(self):
        from assembly_tree import AssemblyTreePanel, _TYPE_ROOT

        panel = AssemblyTreePanel()
        response_root = panel.tree._make_node("Response", _TYPE_ROOT, edit=False)
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)
        workspace.load_feature_preview(self._feature_plan())

        workspace.clear()

        self.assertIsNone(panel.tree.preview_item_for_key(FEATURE_PREVIEW_ROOT_KEY))
        self.assertEqual(workspace.group_ids, ())
        self.assertEqual(panel.tree.topLevelItemCount(), 1)
        self.assertIs(panel.tree.topLevelItem(0), response_root)

    def test_remove_preview_child_or_group_cleans_bound_scene_groups(self):
        from assembly_tree import AssemblyTreePanel

        panel = AssemblyTreePanel()
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)
        workspace.load_feature_preview(self._feature_plan())
        antenna_id = feature_preview_group_id("points", "antenna")
        fastener_id = feature_preview_group_id("points", "fastener / M4")
        line_ids = {
            feature_preview_group_id("lines", "gap main"),
            feature_preview_group_id("lines", "panel"),
        }

        self.assertTrue(panel.tree.remove_preview_key(antenna_id))
        self.assertNotIn(antenna_id, workspace.group_ids)
        self.assertIn(fastener_id, workspace.group_ids)
        self.assertNotIn(antenna_id, workspace._pending_visibility)

        self.assertTrue(
            panel.tree.remove_preview_key(f"{FEATURE_PREVIEW_ROOT_KEY}/lines")
        )
        self.assertTrue(line_ids.isdisjoint(workspace.group_ids))
        self.assertIn(feature_preview_group_id("body"), workspace.group_ids)
        self.assertIn(fastener_id, workspace.group_ids)
        self.assertTrue(line_ids.isdisjoint(workspace._pending_visibility))

    def test_asy_load_clears_runtime_preview_artists(self):
        from assembly_tree import AssemblyTreePanel, QFileDialog

        panel = AssemblyTreePanel()
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)
        workspace.load_feature_preview(self._feature_plan())

        payload = {
            "version": 3,
            "tree": [
                {"name": "Loaded Response", "type": "root", "children": []}
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "loaded.asy")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            with patch.object(
                QFileDialog,
                "getOpenFileName",
                return_value=(path, "Assembly Files (*.asy)"),
            ):
                panel._load()

        self.assertEqual(workspace.group_ids, ())
        self.assertIsNone(panel.tree.preview_item_for_key(FEATURE_PREVIEW_ROOT_KEY))
        self.assertEqual(panel.tree.topLevelItemCount(), 1)
        self.assertEqual(panel.tree.topLevelItem(0).text(0), "Loaded Response")

    def test_malformed_asy_does_not_replace_live_tree_or_preview(self):
        from assembly_tree import AssemblyTreePanel, QFileDialog, _TYPE_ROOT

        panel = AssemblyTreePanel()
        response_root = panel.tree._make_node("Live Response", _TYPE_ROOT, edit=False)
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)
        preview_root = workspace.load_feature_preview(self._feature_plan())
        original_group_ids = workspace.group_ids
        payload = {
            "version": 3,
            "tree": [
                {"name": "Invalid", "type": "preview_root", "children": []}
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "malformed.asy")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            with patch.object(
                QFileDialog,
                "getOpenFileName",
                return_value=(path, "Assembly Files (*.asy)"),
            ):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    panel._load()

        self.assertEqual(workspace.group_ids, original_group_ids)
        self.assertIs(
            panel.tree.preview_item_for_key(FEATURE_PREVIEW_ROOT_KEY),
            preview_root,
        )
        self.assertEqual(panel.tree.topLevelItemCount(), 2)
        self.assertIs(panel.tree.topLevelItem(0), response_root)

    def test_toolbar_delete_refuses_service_owned_preview(self):
        from assembly_tree import AssemblyTreePanel

        panel = AssemblyTreePanel()
        workspace = AssemblyWorkspace(assembly_tree_panel=panel)
        root = workspace.load_feature_preview(self._feature_plan())
        panel.tree.setCurrentItem(root)
        with patch.object(panel, "_notify") as notify:
            panel._delete_selected()

        self.assertIs(panel.tree.preview_item_for_key(FEATURE_PREVIEW_ROOT_KEY), root)
        self.assertTrue(workspace.group_ids)
        notify.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
