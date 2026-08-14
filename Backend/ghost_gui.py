#!/usr/bin/env python3
"""Desktop entry point for the GHOST geometry editor and RCS solver."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QMessageBox,
        QTabWidget,
    )
except ImportError:
    from PySide2.QtWidgets import (  # type: ignore
        QApplication,
        QMainWindow,
        QMessageBox,
        QTabWidget,
    )

from geometry_tab import GeometryTab
from solver_tab import SolverTab


class GhostMainWindow(QMainWindow):
    """Host the existing geometry editor and solver as one application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GHOST 2-D RCS Solver")
        self.resize(1500, 900)
        self.setMinimumSize(1000, 650)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.geometry_tab = GeometryTab(self)
        self.solver_tab = SolverTab(self.geometry_tab, self)
        self.tabs.addTab(self.geometry_tab, "Geometry")
        self.tabs.addTab(self.solver_tab, "Solver")
        self.tabs.setTabToolTip(
            0, "Load, edit, visualize, validate, and save 2-D geometry."
        )
        self.tabs.setTabToolTip(
            1, "Solve the current Geometry tab or an explicitly selected .geo file."
        )
        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage(
            "Load or build a geometry, validate it, then open the Solver tab."
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        thread = getattr(self.solver_tab, "_solve_thread", None)
        if thread is not None and thread.isRunning():
            QMessageBox.warning(
                self,
                "Solver Still Running",
                "A solve is still running. Click Cancel in the Solver tab, "
                "wait for cancellation to finish, and then close GHOST.",
            )
            self.tabs.setCurrentWidget(self.solver_tab)
            event.ignore()
            return
        super().closeEvent(event)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the GHOST 2-D geometry and RCS solver GUI."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the GUI and solver modules import, then exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if args.check:
        print("GHOST GUI dependencies: OK")
        return 0

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("GHOST 2-D RCS Solver")
    app.setOrganizationName("GHOST")

    window = GhostMainWindow()
    window.show()
    if not owns_app:
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
