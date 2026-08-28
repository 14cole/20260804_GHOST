from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ibc.compute import LayerConfig, MaterialTable, MixComponent

try:
    from ibc.ui import DARK_THEME, LIGHT_THEME, ImpedanceGui
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication, QMessageBox

    UI_IMPORTABLE = True
except Exception:
    ImpedanceGui = None  # type: ignore[assignment]
    UI_IMPORTABLE = False


@unittest.skipUnless(UI_IMPORTABLE, "GUI dependencies are unavailable")
class MaterialMixUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_workspace_accepts_optional_parent(self) -> None:
        signature = inspect.signature(ImpedanceGui.__init__)  # type: ignore[union-attr]
        parent = signature.parameters["parent"]
        self.assertIsNone(parent.default)

    def test_themes_use_grim_blue_slate_contract(self) -> None:
        expected_dark = {
            "window_bg": "#0f172a",
            "panel_bg": "#0b1222",
            "head_bg": "#172554",
            "text": "#dbeafe",
            "field_bg": "#0b1222",
            "button_active_bg": "#1d4ed8",
            "selection_bg": "#2563eb",
            "accent": "#3b82f6",
            "preview_border": "#1e3a8a",
            "plot_bg": "#0b1222",
            "plot_axes_bg": "#0b1222",
            "plot_text": "#dbeafe",
            "plot_grid": "#475569",
        }
        for role, color in expected_dark.items():
            self.assertEqual(DARK_THEME[role], color, role)

        legacy_colors = {
            "#661111",
            "#273c1d",
            "#16210f",
            "#243a1c",
            "#101a0a",
            "#2f4a24",
            "#a01e1e",
        }
        for theme in (LIGHT_THEME, DARK_THEME):
            used = {
                str(color).lower()
                for value in theme.values()
                for color in (value if isinstance(value, list) else [value])
            }
            self.assertTrue(used.isdisjoint(legacy_colors), used & legacy_colors)

    def test_dark_theme_applies_to_widgets_and_plot_canvas(self) -> None:
        from matplotlib.colors import to_hex

        workspace = ImpedanceGui()
        try:
            self.assertEqual(workspace._colors, DARK_THEME)
            qss = workspace.styleSheet().lower()
            for color in ("#0f172a", "#0b1222", "#2563eb", "#3b82f6"):
                self.assertIn(color, qss)
            self.assertNotIn("#661111", qss)
            self.assertNotIn("#273c1d", qss)
            self.assertEqual(
                to_hex(workspace.fig.get_facecolor()), DARK_THEME["plot_bg"]
            )
            self.assertEqual(
                to_hex(workspace.ax_heatmap.get_facecolor()),
                DARK_THEME["plot_axes_bg"],
            )
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_host_theme_override_is_presentation_only_and_reversible(self) -> None:
        from matplotlib.colors import to_hex

        workspace = ImpedanceGui()
        try:
            host_theme = dict(DARK_THEME)
            host_theme.update(
                {
                    "window_bg": "#102030",
                    "panel_bg": "#203040",
                    "plot_bg": "#203040",
                    "plot_axes_bg": "#304050",
                    "accent": "#40c0ff",
                }
            )
            was_dirty = workspace.is_dirty()
            dark_value = workspace.dark_mode_var.get()

            workspace.apply_host_theme(host_theme)

            self.assertEqual(workspace._colors, host_theme)
            self.assertEqual(workspace.dark_mode_var.get(), dark_value)
            self.assertEqual(workspace.is_dirty(), was_dirty)
            self.assertFalse(workspace.dark_mode_action.isVisible())
            self.assertFalse(workspace.view_menu.menuAction().isVisible())
            self.assertIn("#102030", workspace.styleSheet().lower())
            self.assertEqual(
                to_hex(workspace.fig.get_facecolor()), "#203040"
            )
            self.assertEqual(
                to_hex(workspace.ax_heatmap.get_facecolor()), "#304050"
            )

            workspace.clear_host_theme()

            self.assertIsNone(workspace._host_theme_override)
            self.assertEqual(workspace._colors, DARK_THEME)
            self.assertTrue(workspace.dark_mode_action.isVisible())
            self.assertTrue(workspace.view_menu.menuAction().isVisible())
            self.assertEqual(workspace.is_dirty(), was_dirty)
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_workspace_exposes_background_job_close_contract(self) -> None:
        class WorkspaceState:
            _task_running = False
            job_is_running = ImpedanceGui.job_is_running  # type: ignore[union-attr]
            can_close = ImpedanceGui.can_close  # type: ignore[union-attr]

        state = WorkspaceState()
        self.assertFalse(state.job_is_running())
        self.assertTrue(state.can_close())

        state._task_running = True
        self.assertTrue(state.job_is_running())
        self.assertFalse(state.can_close())

    def test_cwd_material_csv_is_not_silently_added_to_stack(self) -> None:
        original_cwd = Path.cwd()
        workspace = None
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path("material.csv").write_text(
                    "Frequency_GHz,Eps_real,Eps_imag,Mu_real,Mu_imag\n",
                    encoding="utf-8",
                )
                workspace = ImpedanceGui()
                self.assertEqual(workspace.layers, [])
                self.assertFalse(workspace.is_dirty())
            finally:
                os.chdir(original_cwd)
                if workspace is not None:
                    workspace.deleteLater()
                    self.app.processEvents()

    def test_unsaved_project_close_can_cancel_or_discard(self) -> None:
        workspace = ImpedanceGui()
        try:
            workspace.f_start_var.set("2.0")
            self.assertTrue(workspace.is_dirty())
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)

            cancelled = QCloseEvent()
            with mock.patch.object(
                QMessageBox, "warning", return_value=buttons.Cancel
            ):
                workspace.closeEvent(cancelled)
            self.assertFalse(cancelled.isAccepted())

            discarded = QCloseEvent()
            with mock.patch.object(
                QMessageBox, "warning", return_value=buttons.Discard
            ):
                workspace.closeEvent(discarded)
            self.assertTrue(discarded.isAccepted())
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_failed_project_save_keeps_path_and_dirty_state(self) -> None:
        workspace = ImpedanceGui()
        try:
            workspace.f_stop_var.set("12.0")
            self.assertTrue(workspace.is_dirty())
            with tempfile.TemporaryDirectory() as folder, mock.patch(
                "ibc.ui.filedialog.asksaveasfilename",
                return_value=str(Path(folder) / "coating.json"),
            ), mock.patch(
                "ibc.ui.save_project_file", side_effect=OSError("disk full")
            ), mock.patch("ibc.ui.messagebox.showerror"):
                self.assertFalse(workspace._save_project())

            self.assertIsNone(workspace.project_path)
            self.assertTrue(workspace.is_dirty())
        finally:
            workspace._mark_project_clean()
            workspace.deleteLater()
            self.app.processEvents()

    def test_existing_outputs_require_one_explicit_replacement_confirmation(self) -> None:
        workspace = ImpedanceGui()
        try:
            with tempfile.TemporaryDirectory() as folder:
                first = Path(folder) / "nominal.csv"
                second = Path(folder) / "uncertainty.csv"
                first.write_text("old nominal", encoding="utf-8")
                second.write_text("old uncertainty", encoding="utf-8")
                buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
                with mock.patch.object(
                    QMessageBox, "question", return_value=buttons.No
                ) as question:
                    allowed = workspace._confirm_output_replacements(
                        [first, second, first], operation="Impedance"
                    )
                self.assertFalse(allowed)
                question.assert_called_once()
                self.assertIn("2 output file(s)", question.call_args.args[2])
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_ibc_batch_preflight_shows_count_and_canonical_names(self) -> None:
        workspace = ImpedanceGui()
        try:
            with tempfile.TemporaryDirectory() as folder:
                workspace.layers = [
                    LayerConfig(
                        thickness_in=0.020,
                        anisotropic=False,
                        file_0deg="coating.csv",
                        file_90deg="",
                        polarization_deg=0.0,
                    )
                ]
                workspace.ibc_batch_output_dir_var.set(folder)
                workspace.ibc_batch_prefix_var.set("skin")
                workspace._refresh_layers()

                preview = workspace.ibc_batch_preview_label.text()
                self.assertIn("16 nominal PEC-backed IBC file(s)", preview)
                self.assertIn("171 frequency points each", preview)
                self.assertIn("2,736 total rows", preview)
                self.assertIn("skin_15mil.csv", preview)
                self.assertIn("skin_30mil.csv", preview)
                self.assertTrue(workspace.ibc_batch_export_btn.isEnabled())
                self.assertIn("IBC Batch", workspace._mode_labels)
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_ibc_batch_existing_files_use_one_overwrite_prompt(self) -> None:
        workspace = ImpedanceGui()
        try:
            with tempfile.TemporaryDirectory() as folder:
                workspace.ibc_batch_output_dir_var.set(folder)
                plan = workspace._plan_ibc_batch()
                plan[0].path.write_text("old", encoding="utf-8")
                plan[-1].path.write_text("old", encoding="utf-8")
                buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
                with mock.patch.object(
                    QMessageBox, "question", return_value=buttons.Yes
                ) as question:
                    allowed = workspace._confirm_output_replacements(
                        [item.path for item in plan], operation="IBC Batch"
                    )
                self.assertTrue(allowed)
                question.assert_called_once()
                self.assertIn("2 output file(s)", question.call_args.args[2])
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_multifile_ibc_batch_clears_prior_host_artifact_at_start(self) -> None:
        workspace = ImpedanceGui()
        cleared: list[bool] = []
        workspace.nominal_artifact_cleared.connect(lambda: cleared.append(True))
        try:
            with tempfile.TemporaryDirectory() as folder:
                workspace.layers = [
                    LayerConfig(
                        thickness_in=0.020,
                        anisotropic=False,
                        file_0deg="coating.csv",
                        file_90deg="",
                        polarization_deg=0.0,
                    )
                ]
                workspace.ibc_batch_output_dir_var.set(folder)
                workspace._refresh_layers()
                with mock.patch.object(
                    workspace,
                    "_confirm_output_replacements",
                    return_value=True,
                ), mock.patch.object(workspace, "_run_background_task") as run:
                    workspace._export_ibc_batch()

                self.assertEqual(cleared, [True])
                run.assert_called_once()
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_project_portability_warning_is_shown_in_gui(self) -> None:
        workspace = ImpedanceGui()
        try:
            with tempfile.TemporaryDirectory() as folder:
                project = Path(folder) / "legacy.json"
                project.write_text("{}", encoding="utf-8")

                def fake_load(_path, *, warning_handler=None):
                    self.assertIsNotNone(warning_handler)
                    warning_handler("Legacy project needs path review.")
                    return {}

                with mock.patch(
                    "ibc.ui.filedialog.askopenfilename",
                    return_value=str(project),
                ), mock.patch(
                    "ibc.ui.load_project_file", side_effect=fake_load
                ), mock.patch.object(
                    workspace, "_apply_project_state"
                ), mock.patch(
                    "ibc.ui.messagebox.showwarning"
                ) as showwarning, mock.patch(
                    "ibc.ui.messagebox.showinfo"
                ):
                    workspace._load_project()

                showwarning.assert_called_once()
                self.assertIn(
                    "Legacy project needs path review.",
                    showwarning.call_args.args[1],
                )
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_standalone_close_is_blocked_while_background_job_runs(self) -> None:
        workspace = ImpedanceGui()
        try:
            workspace._task_running = True
            blocked = QCloseEvent()
            with mock.patch.object(QMessageBox, "warning") as warning:
                workspace.closeEvent(blocked)
            self.assertFalse(blocked.isAccepted())
            warning.assert_called_once()

            workspace._task_running = False
            allowed = QCloseEvent()
            workspace.closeEvent(allowed)
            self.assertTrue(allowed.isAccepted())
        finally:
            workspace._task_running = False
            workspace.deleteLater()
            self.app.processEvents()

    def test_forward_display_honors_selected_frequency_grid(self) -> None:
        first = MaterialTable(
            [1.0, 2.0, 3.0],
            [2.0 - 0.1j] * 3,
            [1.0 + 0j] * 3,
        )
        second = MaterialTable(
            [1.0, 2.0, 3.0],
            [6.0 - 0.3j] * 3,
            [1.0 + 0j] * 3,
        )
        grid = [1.5, 2.5]
        display = ImpedanceGui._build_mix_display(  # type: ignore[union-attr]
            None,
            [MixComponent(first, 1.0), MixComponent(second, 1.0)],
            "linear",
            0.125,
            grid,
        )
        self.assertEqual(display["freqs"], grid)

    def test_performance_gap_uses_worst_grid_point(self) -> None:
        at_most = {"direction": "at_most", "target": -10.0}
        at_least = {"direction": "at_least", "target": 90.0}
        self.assertAlmostEqual(
            ImpedanceGui._mix_performance_gap([-15.0, -9.0, -20.0], at_most),  # type: ignore[union-attr]
            1.0,
        )
        self.assertAlmostEqual(
            ImpedanceGui._mix_performance_gap([95.0, 88.0, 92.0], at_least),  # type: ignore[union-attr]
            2.0,
        )

    def test_air_layer_fails_pec_absorber_target(self) -> None:
        air = MaterialTable(
            [1.0, 2.0],
            [1.0 + 0j, 1.0 + 0j],
            [1.0 + 0j, 1.0 + 0j],
        )
        config = {
            "label": "PEC-backed absorption (%)",
            "metric_key": "metal_absorption_db",
            "direction": "at_least",
            "unit": "%",
            "target": 90.0,
            "angles": [0.0, 45.0],
            "wave_pol": "te",
        }
        result = ImpedanceGui._evaluate_mix_performance(  # type: ignore[union-attr]
            ImpedanceGui, air, 0.125, config  # type: ignore[arg-type]
        )
        self.assertGreater(result["gap"], 89.999)
        self.assertEqual(len(result["grid"]), 2)
        self.assertEqual(len(result["grid"][0]), 2)


if __name__ == "__main__":
    unittest.main()
