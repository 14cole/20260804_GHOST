#!/usr/bin/env python3
"""Smoke tests for the GHOST GUI application shell."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ghost_gui import GhostMainWindow, GhostWorkspace, main  # noqa: E402

try:  # noqa: E402
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:  # noqa: E402
    from PySide2.QtGui import QCloseEvent  # type: ignore
    from PySide2.QtWidgets import QApplication, QMessageBox  # type: ignore


class TestGuiEntrypoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dependency_check_does_not_start_event_loop(self):
        self.assertEqual(main(["--check"]), 0)

    def test_main_window_hosts_geometry_and_solver_tabs(self):
        window = GhostMainWindow()
        try:
            self.assertEqual(window.windowTitle(), "GHOST 2-D RCS Solver")
            self.assertEqual(window.tabs.count(), 2)
            self.assertEqual(window.tabs.tabText(0), "Geometry")
            self.assertEqual(window.tabs.tabText(1), "Solver")
            self.assertIs(window.solver_tab.geometry_tab, window.geometry_tab)
        finally:
            window.close()

    def test_workspace_is_embeddable_and_forwards_exports(self):
        workspace = GhostWorkspace()
        received = []
        workspace.files_exported.connect(
            lambda paths, kind: received.append((list(paths), kind))
        )
        try:
            self.assertEqual(workspace.count(), 2)
            self.assertIs(
                workspace.solver_tab.geometry_tab, workspace.geometry_tab
            )
            workspace.solver_tab.files_exported.emit(
                ["example.grim"], "2d"
            )
            self.assertEqual(received, [(["example.grim"], "2d")])
            self.assertFalse(workspace.solve_is_running())
        finally:
            workspace.close()

    def test_workspace_forwards_typed_freddy_artifact_to_geometry(self):
        workspace = GhostWorkspace()
        try:
            with mock.patch.object(
                workspace.geometry_tab,
                "attach_material_artifact",
                return_value=True,
            ) as attach:
                self.assertTrue(
                    workspace.attach_material_artifact(
                        "ibc", "C:/exports/nominal.csv"
                    )
                )
            attach.assert_called_once_with(
                "ibc", "C:/exports/nominal.csv"
            )
            self.assertIs(workspace.currentWidget(), workspace.geometry_tab)
        finally:
            workspace.close()

    def test_geometry_dirty_state_marks_tab_and_can_block_standalone_close(self):
        window = GhostMainWindow()
        try:
            self.assertFalse(window.geometry_tab.is_dirty())
            window.geometry_tab._ibc_add_row()
            self.assertTrue(window.geometry_tab.is_dirty())
            self.assertEqual(window.tabs.tabText(0), "Geometry*")

            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            event = QCloseEvent()
            with mock.patch.object(
                QMessageBox, "warning", return_value=buttons.Cancel
            ):
                window.closeEvent(event)
            self.assertFalse(event.isAccepted())
            self.assertIs(window.tabs.currentWidget(), window.geometry_tab)
        finally:
            window.geometry_tab._set_dirty(False)
            window.close()

    def test_geometry_status_and_embedded_plot_follow_dark_host_colors(self):
        from matplotlib.colors import to_hex

        workspace = GhostWorkspace()
        try:
            self.assertNotIn("#333", workspace.geometry_tab.lbl_status.styleSheet())
            workspace.geometry_tab.apply_plot_theme(
                background="#0b1222", text="#dbeafe", grid="#475569"
            )
            self.assertEqual(
                to_hex(workspace.geometry_tab.canvas.fig.get_facecolor()),
                "#0b1222",
            )
            self.assertEqual(
                to_hex(workspace.geometry_tab.canvas.ax.get_facecolor()),
                "#0b1222",
            )
            self.assertEqual(
                workspace.geometry_tab.canvas.ax.xaxis.label.get_color(),
                "#dbeafe",
            )
            workspace.solver_tab.apply_plot_theme(
                background="#0b1222", text="#dbeafe", grid="#475569"
            )
            self.assertEqual(
                to_hex(workspace.solver_tab.canvas.fig.get_facecolor()),
                "#0b1222",
            )
            self.assertEqual(
                to_hex(workspace.solver_tab.canvas.ax.get_facecolor()),
                "#0b1222",
            )
            self.assertEqual(
                workspace.solver_tab.canvas.ax.xaxis.label.get_color(),
                "#dbeafe",
            )
        finally:
            workspace.close()


if __name__ == "__main__":
    unittest.main()
