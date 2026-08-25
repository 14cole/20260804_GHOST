"""Focused regressions for embedding FREDDY without package collisions."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

import freddy_integration
from grim_cut_gui import BLUE_PALETTE


class _FocusProbe(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def setFocus(self, _reason) -> None:  # noqa: N802 - Qt-compatible probe
        self.called = True


class _FakeImpedanceGui(QMainWindow):
    nominal_artifact_exported = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.running = False
        self.focus_probe = _FocusProbe()
        self.setCentralWidget(self.focus_probe)

    def job_is_running(self) -> bool:
        return self.running


class FreddyIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        prefix = freddy_integration.FREDDY_PACKAGE_NAMESPACE
        self.saved_private_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == prefix or name.startswith(prefix + ".")
        }
        for name in self.saved_private_modules:
            sys.modules.pop(name, None)

    def tearDown(self) -> None:
        prefix = freddy_integration.FREDDY_PACKAGE_NAMESPACE
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)
        sys.modules.update(self.saved_private_modules)

    def test_authoritative_ui_loads_privately_without_replacing_generic_ibc(self) -> None:
        generic_ibc = ModuleType("ibc")
        previous = sys.modules.get("ibc")
        sys.modules["ibc"] = generic_ibc
        try:
            module = freddy_integration.load_freddy_ui_module()
            self.assertEqual(
                module.__name__,
                freddy_integration.FREDDY_PACKAGE_NAMESPACE + ".ui",
            )
            self.assertEqual(
                module.ImpedanceGui.__module__,
                freddy_integration.FREDDY_PACKAGE_NAMESPACE + ".ui",
            )
            self.assertIs(sys.modules["ibc"], generic_ibc)
        finally:
            if previous is None:
                sys.modules.pop("ibc", None)
            else:
                sys.modules["ibc"] = previous

    def test_bundled_dark_theme_matches_grim_application_palette(self) -> None:
        theme = freddy_integration.load_freddy_ui_module().DARK_THEME
        expected = {
            "window_bg": BLUE_PALETTE["win_bg"],
            "panel_bg": BLUE_PALETTE["panel_bg"],
            "head_bg": BLUE_PALETTE["head_bg"],
            "text": BLUE_PALETTE["text"],
            "button_active_bg": BLUE_PALETTE["hover"],
            "selection_bg": BLUE_PALETTE["checked_bg"],
            "accent": BLUE_PALETTE["checked_border"],
            "preview_border": BLUE_PALETTE["border"],
            "plot_grid": BLUE_PALETTE["grid"],
        }
        for role, color in expected.items():
            self.assertEqual(theme[role], color, role)

    def test_widget_embeds_workspace_and_delegates_busy_state_and_focus(self) -> None:
        with mock.patch.object(
            freddy_integration,
            "_load_impedance_gui_class",
            return_value=_FakeImpedanceGui,
        ):
            widget = freddy_integration.FreddyIntegrationWidget()

        try:
            workspace = widget.workspace
            self.assertIsInstance(workspace, _FakeImpedanceGui)
            self.assertFalse(workspace.isWindow())
            self.assertFalse(widget.job_is_running())
            self.assertFalse(widget.attach_to_ghost_button.isEnabled())

            workspace.running = True
            self.assertTrue(widget.job_is_running())
            widget.focus_workspace()
            self.assertTrue(workspace.focus_probe.called)
        finally:
            widget.deleteLater()
            self.app.processEvents()

    def test_widget_forwards_only_existing_typed_nominal_artifacts(self) -> None:
        with mock.patch.object(
            freddy_integration,
            "_load_impedance_gui_class",
            return_value=_FakeImpedanceGui,
        ):
            widget = freddy_integration.FreddyIntegrationWidget()

        received = []
        widget.attach_to_ghost_requested.connect(
            lambda kind, path: received.append((kind, path))
        )
        try:
            with tempfile.TemporaryDirectory() as folder:
                artifact = Path(folder) / "nominal.csv"
                artifact.write_text(
                    "frequency_hz,resistance_ohm,reactance_ohm\n"
                    "1000000000,50,0\n",
                    encoding="utf-8",
                )
                widget.workspace.nominal_artifact_exported.emit(
                    "analysis", str(artifact)
                )
                self.assertFalse(widget.attach_to_ghost_button.isEnabled())

                widget.workspace.nominal_artifact_exported.emit(
                    "ibc", str(artifact)
                )
                self.assertTrue(widget.attach_to_ghost_button.isEnabled())
                widget.attach_to_ghost_button.click()

                self.assertEqual(
                    received, [("ibc", str(artifact.resolve()))]
                )
        finally:
            widget.deleteLater()
            self.app.processEvents()

    def test_bundled_authoritative_workspace_is_an_embedded_child(self) -> None:
        widget = freddy_integration.FreddyIntegrationWidget()
        try:
            self.assertIsNotNone(widget.root_path)
            self.assertEqual(widget.load_error, "")
            self.assertIsNotNone(widget.workspace)
            self.assertFalse(widget.workspace.isWindow())
            self.assertEqual(
                widget.workspace.__class__.__module__,
                freddy_integration.FREDDY_PACKAGE_NAMESPACE + ".ui",
            )
        finally:
            widget.deleteLater()
            self.app.processEvents()

    def test_unavailable_freddy_has_safe_inactive_fallback(self) -> None:
        with mock.patch.object(
            freddy_integration,
            "_load_impedance_gui_class",
            side_effect=ImportError("test unavailable"),
        ):
            widget = freddy_integration.FreddyIntegrationWidget()

        try:
            self.assertIsNone(widget.workspace)
            self.assertIn("test unavailable", widget.load_error)
            self.assertFalse(widget.job_is_running())
            widget.focus_workspace()
        finally:
            widget.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
