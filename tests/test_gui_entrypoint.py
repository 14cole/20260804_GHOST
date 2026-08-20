#!/usr/bin/env python3
"""Smoke tests for the GHOST GUI application shell."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ghost_gui import GhostMainWindow, main  # noqa: E402

try:  # noqa: E402
    from PySide6.QtWidgets import QApplication
except ImportError:  # noqa: E402
    from PySide2.QtWidgets import QApplication  # type: ignore


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


if __name__ == "__main__":
    unittest.main()
