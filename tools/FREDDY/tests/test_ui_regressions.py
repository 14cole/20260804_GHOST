from __future__ import annotations

import inspect
import unittest
from unittest import mock

from ibc.compute import MaterialTable, MixComponent

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
