from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

# Select a headless Qt platform before this module conditionally imports Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from feature_assembly_panel import (  # noqa: E402
    GUI_AVAILABLE,
    LINE_PLACEMENT_COLUMNS,
    POINT_PLACEMENT_COLUMNS,
    FeatureAssemblyFormModel,
    FeatureAssemblyPanel,
    FeatureAssemblyValues,
    FeatureBuildDispatch,
    FeatureWorkflowAdapter,
    placement_csv_template_text,
    write_placement_csv_template,
)


class _CapturedRequest:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _FakeWorkflow:
    FeatureAssemblyRequest = _CapturedRequest

    def __init__(self):
        self.calls = []
        self.requirements = SimpleNamespace(
            point_dataset_ids=("fastener",),
            line_dataset_ids=("panel_gap",),
        )

    def discover_feature_dataset_ids(self, **kwargs):
        self.calls.append(("discover", dict(kwargs)))
        return self.requirements

    def prepare_feature_assembly(self, request):
        self.calls.append(("prepare", request))
        return SimpleNamespace(request=request, preview_geometry="preview")

    def prepare_feature_input_preview(self, **kwargs):
        self.calls.append(("input_preview", dict(kwargs)))
        return SimpleNamespace(preview_stage="inputs", **kwargs)

    def execute_feature_assembly(self, plan):
        self.calls.append(("execute", plan))
        return plan.request.kwargs["output_grim"]


def _ready_point_model() -> FeatureAssemblyFormModel:
    values = FeatureAssemblyValues(
        base_grim="clean_body.grim",
        output_grim="assembled.grim",
        coordinate_units="millimeters",
        surface_mesh="body.stl",
        surface_units="meters",
        flip_surface_normals=True,
        shadow=True,
        shadow_bias_m=2.5e-5,
        point_locations_csv="points.csv",
        skin_tol_m=8.0e-4,
        skin_phase_tol_deg=12.0,
        normal_tol_deg=9.0,
    )
    model = FeatureAssemblyFormModel(values)
    model.update_dataset_requirements(
        {"point_dataset_ids": ("fastener",), "line_dataset_ids": ()}
    )
    model.set_point_dataset("fastener", "fastener_opn_minus_frd.grim")
    return model


class FeatureAssemblyModelTests(unittest.TestCase):
    def test_displayed_templates_are_the_strict_shared_placement_contracts(self):
        self.assertEqual(
            POINT_PLACEMENT_COLUMNS,
            (
                "placement_id", "dataset_id", "x", "y", "z", "nx", "ny",
                "nz", "roll_x", "roll_y", "roll_z",
            ),
        )
        self.assertEqual(
            LINE_PLACEMENT_COLUMNS,
            (
                "line_id", "dataset_id", "segment_index", "x1", "y1", "z1",
                "x2", "y2", "z2", "n1x", "n1y", "n1z", "n2x", "n2y",
                "n2z",
            ),
        )
        self.assertEqual(
            placement_csv_template_text("point"),
            ",".join(POINT_PLACEMENT_COLUMNS) + "\n",
        )
        self.assertEqual(
            placement_csv_template_text("line"),
            ",".join(LINE_PLACEMENT_COLUMNS) + "\n",
        )

        with tempfile.TemporaryDirectory() as directory:
            target = write_placement_csv_template(
                "point", Path(directory) / "placements"
            )
            self.assertEqual(target.suffix, ".csv")
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                placement_csv_template_text("point"),
            )

    def test_adapter_rejects_backend_csv_contract_drift(self):
        workflow = _FakeWorkflow()
        workflow.POINT_CSV_COLUMNS = ("different",)

        with self.assertRaisesRegex(RuntimeError, "no longer matches"):
            FeatureWorkflowAdapter.from_module(workflow)

    def test_input_preview_does_not_require_output_or_response_mapping(self):
        workflow = _FakeWorkflow()
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                base_grim="body.grim",
                surface_mesh="body.stl",
                coordinate_units="millimeters",
                surface_units="meters",
                point_locations_csv="points.csv",
                line_locations_csv="lines.csv",
            )
        )

        preview = model.prepare_input_preview(workflow)

        self.assertEqual(preview.preview_stage, "inputs")
        self.assertEqual(
            workflow.calls,
            [
                (
                    "input_preview",
                    {
                        "base_grim": "body.grim",
                        "surface_mesh": "body.stl",
                        "coordinate_units": "millimeters",
                        "surface_units": "meters",
                        "point_locations_csv": "points.csv",
                        "line_locations_csv": "lines.csv",
                        "base_dir": None,
                    },
                )
            ],
        )

    def test_invalidating_one_csv_discards_only_its_mapping_rows(self):
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                point_datasets={"fastener": "fastener.grim"},
                line_datasets={"gap": "gap.grim"},
            )
        )
        model.update_dataset_requirements(
            {"point_dataset_ids": ("fastener",), "line_dataset_ids": ("gap",)}
        )

        model.invalidate_dataset_requirements("point")

        self.assertEqual(model.point_dataset_ids, ())
        self.assertEqual(model.values.point_datasets, {})
        self.assertEqual(model.line_dataset_ids, ("gap",))
        self.assertEqual(model.values.line_datasets, {"gap": "gap.grim"})

    def test_changed_csv_path_cannot_reuse_previous_discovery(self):
        model = _ready_point_model()
        model.values.point_locations_csv = "replacement_points.csv"

        with self.assertRaisesRegex(ValueError, "changed after its last"):
            model.validate()

    def test_request_construction_uses_exact_mapping_and_controls(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        request = model.build_request(workflow)

        self.assertIsInstance(request, _CapturedRequest)
        self.assertEqual(request.kwargs["base_grim"], "clean_body.grim")
        self.assertEqual(request.kwargs["output_grim"], "assembled.grim")
        self.assertEqual(request.kwargs["coordinate_units"], "millimeters")
        self.assertEqual(request.kwargs["surface_mesh"], "body.stl")
        self.assertEqual(request.kwargs["surface_units"], "meters")
        self.assertTrue(request.kwargs["flip_surface_normals"])
        self.assertTrue(request.kwargs["shadow"])
        self.assertEqual(request.kwargs["shadow_bias_m"], 2.5e-5)
        self.assertEqual(request.kwargs["point_locations_csv"], "points.csv")
        self.assertEqual(
            request.kwargs["point_datasets"],
            {"fastener": "fastener_opn_minus_frd.grim"},
        )
        self.assertIsNone(request.kwargs["line_locations_csv"])
        self.assertEqual(request.kwargs["line_datasets"], {})
        self.assertEqual(request.kwargs["skin_tol_m"], 8.0e-4)
        self.assertEqual(request.kwargs["skin_phase_tol_deg"], 12.0)
        self.assertEqual(request.kwargs["normal_tol_deg"], 9.0)

    def test_discovery_updates_ids_and_preserves_surviving_paths(self):
        workflow = _FakeWorkflow()
        model = FeatureAssemblyFormModel(
            FeatureAssemblyValues(
                point_locations_csv="points.csv",
                line_locations_csv="lines.csv",
            )
        )

        model.discover_dataset_ids(workflow)
        model.set_point_dataset("fastener", "fastener.grim")
        model.set_line_dataset("panel_gap", "gap.grim")

        workflow.requirements = SimpleNamespace(
            point_dataset_ids=("fastener", "antenna"),
            line_dataset_ids=("seam",),
        )
        model.discover_dataset_ids(workflow)

        self.assertEqual(model.point_dataset_ids, ("fastener", "antenna"))
        self.assertEqual(model.line_dataset_ids, ("seam",))
        self.assertEqual(
            model.values.point_datasets,
            {"fastener": "fastener.grim", "antenna": ""},
        )
        self.assertEqual(model.values.line_datasets, {"seam": ""})
        discover_call = workflow.calls[-1]
        self.assertEqual(discover_call[0], "discover")
        self.assertEqual(
            discover_call[1],
            {
                "point_locations_csv": "points.csv",
                "line_locations_csv": "lines.csv",
                "base_dir": None,
            },
        )

    def test_preview_and_build_dispatch_through_injected_service(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()

        preview_plan = model.prepare_preview(workflow)
        dispatch = model.assemble(workflow)

        self.assertEqual(preview_plan.preview_geometry, "preview")
        self.assertIsInstance(dispatch, FeatureBuildDispatch)
        self.assertEqual(dispatch.output_path, "assembled.grim")
        self.assertEqual(
            [name for name, _value in workflow.calls],
            ["prepare", "prepare", "execute"],
        )
        self.assertIs(workflow.calls[-1][1], dispatch.plan)

    def test_missing_mapping_is_rejected_before_backend_prepare(self):
        workflow = _FakeWorkflow()
        model = _ready_point_model()
        model.set_point_dataset("fastener", "")

        with self.assertRaisesRegex(ValueError, "point:fastener"):
            model.prepare_preview(workflow)

        self.assertEqual(workflow.calls, [])

    def test_adapter_accepts_small_grim_service_contract(self):
        class Service:
            def make_request(self, **kwargs):
                return kwargs

            def discover_dataset_ids(self, point_csv=None, line_csv=None):
                return (point_csv, line_csv)

            def prepare(self, request):
                return request

            def execute(self, plan):
                return "out.grim"

        adapter = FeatureWorkflowAdapter.from_service(Service())
        self.assertEqual(
            adapter.discover(
                point_locations_csv="p.csv",
                line_locations_csv="l.csv",
                base_dir="ignored",
            ),
            ("p.csv", "l.csv"),
        )


@unittest.skipUnless(GUI_AVAILABLE, "PySide6 GUI dependency unavailable")
class FeatureAssemblyPanelQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_public_busy_and_close_contract_starts_idle(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        self.assertFalse(panel.is_busy())
        self.assertTrue(panel.can_close())
        self.assertIn("locally and on HPC", panel.point_help_label.text())
        self.assertIn(",".join(POINT_PLACEMENT_COLUMNS), panel.point_schema_label.text())
        self.assertIn(",".join(LINE_PLACEMENT_COLUMNS), panel.line_schema_label.text())
        self.assertEqual(panel.input_preview_button.text(), "Preview Inputs in 3-D")
        self.assertIn("Ready", panel.status_label.text())
        panel.close()

    def test_input_change_marks_a_current_preview_stale(self):
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        messages = []
        panel.preview_stale.connect(messages.append)
        panel._preview_is_current = True

        panel.set_surface_mesh("body.stl")

        self.assertFalse(panel._preview_is_current)
        self.assertEqual(len(messages), 1)
        self.assertIn("out of date", messages[0])
        self.assertIn("out of date", panel.status_label.text())
        panel.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
