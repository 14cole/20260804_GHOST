"""Focused shell regressions for the unified GRIM application."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QToolButton, QVBoxLayout, QWidget

import grim_cut_gui
import ghost_integration
from assembly_tree import (
    AssemblyTreePanel,
    _TYPE_BRANCH,
    _TYPE_ROOT,
    _attach,
)
from grim_dataset import RcsGrid


class _FakeGhostIntegration(QWidget):
    files_exported = Signal(list, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.backend_path = None
        self.running = False
        self.focus_called = False
        self.setLayout(QVBoxLayout())

    def solve_is_running(self) -> bool:
        return self.running

    def focus_solver(self) -> None:
        self.focus_called = True


class _RecordingWindow(grim_cut_gui.GrimCutWindow):
    def __init__(self) -> None:
        self.loaded_path_batches: list[list[str]] = []
        super().__init__()

    def _handle_files_dropped(self, paths) -> None:
        self.loaded_path_batches.append([os.fspath(path) for path in paths])


class _FakeFeatureWorkflow:
    FeatureAssemblyRequest = staticmethod(lambda **values: values)

    @staticmethod
    def discover_feature_dataset_ids(**_values):
        return {"point_dataset_ids": (), "line_dataset_ids": ()}

    @staticmethod
    def prepare_feature_assembly(request):
        return request

    @staticmethod
    def execute_feature_assembly(_plan):
        return "assembled.grim"


def _grid(amplitude: float = 1.0) -> RcsGrid:
    field = np.asarray([[[[complex(amplitude)]]]], dtype=np.complex128)
    return RcsGrid(
        [0.0],
        [0.0],
        [10.0],
        ["VV"],
        rcs=field,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        },
    )


class UnifiedGuiShellTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.feature_service = _FakeFeatureWorkflow()
        self.ghost_patch = mock.patch.object(
            grim_cut_gui, "GhostIntegrationWidget", _FakeGhostIntegration
        )
        self.feature_patch = mock.patch.object(
            grim_cut_gui,
            "load_ghost_module",
            return_value=self.feature_service,
        )
        self.ghost_patch.start()
        self.feature_patch.start()
        self.window = _RecordingWindow()

    def tearDown(self) -> None:
        self.window.ghost_integration.running = False
        self.window.deleteLater()
        self.app.processEvents()
        self.feature_patch.stop()
        self.ghost_patch.stop()

    def test_tabs_have_one_canonical_assembly_workspace(self) -> None:
        labels = [
            self.window.main_tabs.tabText(index)
            for index in range(self.window.main_tabs.count())
        ]
        self.assertEqual(labels, ["Plotting", "ISAR", "Assembly", "GHOST"])
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.assembly_workspace), 2
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.ghost_integration), 3
        )

        panels = self.window.findChildren(AssemblyTreePanel)
        self.assertEqual(panels, [self.window.assembly_workspace.assembly_tree_panel])
        assembly_buttons = [
            button
            for button in self.window.findChildren(QToolButton)
            if button.text() == "Assembly Tree"
        ]
        self.assertEqual(assembly_buttons, [])
        for context in self.window._plot_contexts.values():
            self.assertFalse(hasattr(context, "assembly_tree_panel"))
            self.assertFalse(hasattr(context, "btn_assembly_tree"))
        self.assertIs(
            self.window.feature_assembly_panel.service(), self.feature_service
        )

    def test_bundled_ghost_backend_is_the_primary_builtin_candidate(self) -> None:
        expected = (
            Path(ghost_integration.__file__).resolve().parents[1]
            / "tools"
            / "GHOST"
            / "Backend"
        ).resolve()
        with mock.patch.dict(
            os.environ, {ghost_integration.GHOST_BACKEND_ENV: ""}, clear=False
        ):
            candidates = list(ghost_integration.ghost_backend_candidates())
            discovered = ghost_integration.discover_ghost_backend()

        self.assertEqual(candidates[0], expected)
        self.assertEqual(discovered, expected)

    def test_workspace_and_ghost_outputs_enter_existing_dataset_paths(self) -> None:
        self.window.assembly_workspace.files_to_load.emit(["assembly.grim"])
        self.window.ghost_integration.files_exported.emit(
            ["ghost_vv_hh.grim"], "2d"
        )
        self.assertEqual(
            self.window.loaded_path_batches,
            [["assembly.grim"], ["ghost_vv_hh.grim"]],
        )

        start_rows = self.window.table.rowCount()
        self.window.assembly_workspace.platform_built.emit(
            "platform", _grid(1.0), "platform history"
        )
        self.window.assembly_workspace.feature_built.emit(
            "featured", _grid(2.0), "feature history"
        )
        self.assertEqual(self.window.table.rowCount(), start_rows + 2)
        self.assertEqual(
            self.window.table.item(start_rows, 0).text(), "platform"
        )
        self.assertEqual(
            self.window.table.item(start_rows + 1, 0).text(), "featured"
        )

    def test_feature_panel_preview_and_output_use_workspace_paths(self) -> None:
        plan = SimpleNamespace(
            surface_triangles_cad_m=np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
            ),
            body_profile_rho_z_m=None,
            point_locations_cad_m={"antenna": np.asarray([[0.2, 0.2, 0.0]])},
            line_paths_cad_m={
                "gap": {"g1": np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])}
            },
        )

        self.window.feature_assembly_panel.preview_ready.emit(plan)
        self.assertIn(
            "feature-assembly/points/antenna",
            self.window.assembly_workspace.group_ids,
        )
        self.assertIn(
            "feature-assembly/lines/gap",
            self.window.assembly_workspace.group_ids,
        )

        self.window.feature_assembly_panel.feature_built.emit("assembled.grim")
        self.assertEqual(self.window.loaded_path_batches, [["assembled.grim"]])

    def test_running_ghost_solve_blocks_close_and_focuses_solver(self) -> None:
        self.window.ghost_integration.running = True
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.ghost_integration
        )
        self.assertTrue(self.window.ghost_integration.focus_called)

    def test_running_feature_job_blocks_close_on_assembly_tab(self) -> None:
        event = QCloseEvent()
        with (
            mock.patch.object(
                self.window.feature_assembly_panel,
                "job_is_running",
                return_value=True,
            ),
            mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.assembly_workspace
        )

    def test_branch_drop_uses_canonical_workspace_tree(self) -> None:
        tree = self.window.assembly_workspace.assembly_tree_panel.tree
        root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
        branch = tree._make_node("Payload", _TYPE_BRANCH, parent=root, edit=False)
        leaf = tree._make_leaf("part", _grid(1.5))
        _attach(tree, leaf, branch)
        tree._branch_drag_item = branch

        start_rows = self.window.table.rowCount()
        self.window._on_assembly_branch_dropped("Payload", [])

        self.assertEqual(self.window.table.rowCount(), start_rows + 1)
        self.assertEqual(self.window.table.item(start_rows, 0).text(), "Payload")


if __name__ == "__main__":
    unittest.main()
