from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

# Select a headless Qt platform before this module conditionally imports Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from feature_assembly_panel import (  # noqa: E402
    GUI_AVAILABLE,
    FeatureAssemblyFormModel,
    FeatureAssemblyPanel,
    FeatureAssemblyValues,
    FeatureBuildDispatch,
    FeatureWorkflowAdapter,
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
        panel.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
