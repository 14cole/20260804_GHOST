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

    def test_geometry_edit_marks_last_tab_result_stale_and_blocks_export(self):
        workspace = GhostWorkspace()
        try:
            solver = workspace.solver_tab
            solver.last_result = {"samples": []}
            solver.last_solve_context = {"uses_geometry_tab": True}
            solver._last_result_stale = False
            solver._sync_export_state()
            self.assertTrue(solver.btn_export.isEnabled())

            workspace.geometry_tab.dirty_changed.emit(True)

            self.assertTrue(solver._last_result_stale)
            self.assertFalse(solver.btn_export.isEnabled())
            self.assertIn("Stale", solver.btn_export.text())
        finally:
            workspace.close()

    def test_edit_during_solve_stales_result_when_geometry_was_already_dirty(self):
        workspace = GhostWorkspace()
        try:
            geometry = workspace.geometry_tab
            solver = workspace.solver_tab

            # The solve snapshot is allowed to include unsaved in-memory
            # edits.  Its dirty state therefore predates the solve and will
            # not transition again when the user makes another edit.
            geometry._ibc_add_row()
            self.assertTrue(geometry.is_dirty())
            solver._pending_solve_context = {
                "uses_geometry_tab": True,
                "geometry_stale": False,
            }

            geometry._diel_add_row()

            self.assertTrue(
                solver._pending_solve_context["geometry_stale"]
            )

            # Completion must carry the invalidation into the published-result
            # guard even though dirty_changed had no second True transition.
            solver.chk_export_after_solve.setChecked(False)
            with (
                mock.patch.object(solver, "_populate_results_table"),
                mock.patch.object(solver, "_plot_results"),
            ):
                solver._on_solver_finished(
                    {"samples": [], "metadata": {}}, ""
                )
            self.assertTrue(solver._last_result_stale)
            self.assertFalse(solver.btn_export.isEnabled())
            self.assertIn("during the solve", solver.lbl_status.text())
        finally:
            workspace.geometry_tab._set_dirty(False)
            workspace.close()

    def test_loading_clean_geometry_stales_in_flight_tab_snapshot(self):
        workspace = GhostWorkspace()
        try:
            geometry = workspace.geometry_tab
            solver = workspace.solver_tab
            solver._pending_solve_context = {
                "uses_geometry_tab": True,
                "geometry_stale": False,
            }
            fixture = ROOT / "geometries" / "body.geo"

            with (
                mock.patch(
                    "geometry_tab.QFileDialog.getOpenFileName",
                    return_value=(str(fixture), "Geometry Files (*.geo)"),
                ),
                mock.patch("geometry_tab.QMessageBox.information"),
            ):
                self.assertTrue(geometry.load_geo())

            self.assertFalse(geometry.is_dirty())
            self.assertTrue(
                solver._pending_solve_context["geometry_stale"]
            )
        finally:
            workspace.close()

    def test_old_thread_cleanup_cannot_clear_newer_solve_handles(self):
        workspace = GhostWorkspace()
        try:
            solver = workspace.solver_tab
            newer_thread = object()
            newer_worker = object()
            newer_abort = object()
            solver._solve_thread = newer_thread
            solver._solve_worker = newer_worker
            solver._abort_event = newer_abort
            solver._active_solve_run_id = 2

            solver._on_solver_thread_finished(1)

            self.assertIs(solver._solve_thread, newer_thread)
            self.assertIs(solver._solve_worker, newer_worker)
            self.assertIs(solver._abort_event, newer_abort)
            self.assertEqual(solver._active_solve_run_id, 2)

            solver._on_solver_thread_finished(2)
            self.assertIsNone(solver._solve_thread)
            self.assertIsNone(solver._solve_worker)
            self.assertIsNone(solver._abort_event)
            self.assertIsNone(solver._active_solve_run_id)
        finally:
            workspace.close()

    def test_automatic_export_rechecks_stale_after_confirmation(self):
        workspace = GhostWorkspace()
        try:
            geometry = workspace.geometry_tab
            solver = workspace.solver_tab
            solver._is_solving = True
            solver._pending_solve_context = {
                "uses_geometry_tab": True,
                "geometry_stale": False,
            }
            solver.chk_export_after_solve.setChecked(True)
            result = {"samples": [], "metadata": {}}

            def confirm_then_edit(_paths):
                geometry._diel_add_row()
                return True

            with (
                mock.patch.object(solver, "_populate_results_table"),
                mock.patch.object(solver, "_plot_results"),
                mock.patch.object(
                    solver, "_resolve_output_path", return_value="result.grim"
                ),
                mock.patch(
                    "solver_tab._planned_export_paths",
                    return_value=["result.grim"],
                ),
                mock.patch.object(
                    solver,
                    "_confirm_export_replacements",
                    side_effect=confirm_then_edit,
                ),
                mock.patch.object(
                    solver, "_export_result_files", return_value=["result.grim"]
                ) as export,
            ):
                solver._on_solver_finished(result, "")

            self.assertTrue(solver._last_result_stale)
            self.assertFalse(export.called)
            self.assertFalse(solver._is_solving)
            self.assertIn("Automatic export skipped", solver.lbl_status.text())
        finally:
            workspace.geometry_tab._set_dirty(False)
            workspace.close()

    def test_manual_export_rechecks_result_identity_after_confirmation(self):
        workspace = GhostWorkspace()
        try:
            solver = workspace.solver_tab
            original = {"samples": [], "metadata": {}}
            replacement = {"samples": [], "metadata": {}}
            original_context = {"uses_geometry_tab": True}
            replacement_context = {"uses_geometry_tab": True}
            solver.last_result = original
            solver.last_solve_context = original_context
            solver._last_result_stale = False

            def confirm_then_replace(_paths):
                solver.last_result = replacement
                solver.last_solve_context = replacement_context
                return True

            with (
                mock.patch.object(
                    solver, "_resolve_output_path", return_value="result.grim"
                ),
                mock.patch(
                    "solver_tab._planned_export_paths",
                    return_value=["result.grim"],
                ),
                mock.patch.object(
                    solver,
                    "_confirm_export_replacements",
                    side_effect=confirm_then_replace,
                ),
                mock.patch.object(
                    solver, "_export_result_files", return_value=["result.grim"]
                ) as export,
                mock.patch("solver_tab.QMessageBox.warning") as warning,
            ):
                solver._export_last_result()

            self.assertFalse(export.called)
            self.assertIs(solver.last_result, replacement)
            self.assertIn("No files were written", solver.lbl_status.text())
            warning.assert_called_once()
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
