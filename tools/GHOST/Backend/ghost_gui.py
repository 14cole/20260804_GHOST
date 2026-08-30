#!/usr/bin/env python3
"""Desktop entry point for the GHOST geometry editor and RCS solver."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

try:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QMessageBox,
        QTabWidget,
    )
except ImportError:
    from PySide2.QtCore import Signal  # type: ignore
    from PySide2.QtWidgets import (  # type: ignore
        QApplication,
        QMainWindow,
        QMessageBox,
        QTabWidget,
    )

from geometry_tab import GeometryTab
from solver_tab import SolverTab


class GhostWorkspace(QTabWidget):
    """Reusable GHOST geometry/solver workspace for standalone or GRIM hosts."""

    files_exported = Signal(list, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)

        self.geometry_tab = GeometryTab(self)
        self.solver_tab = SolverTab(self.geometry_tab, self)
        self.addTab(self.geometry_tab, "Geometry")
        self.addTab(self.solver_tab, "Solver")
        self.setTabToolTip(
            0, "Load, edit, visualize, validate, and save 2-D geometry."
        )
        self.setTabToolTip(
            1, "Solve the current Geometry tab or an explicitly selected .geo file."
        )
        self.geometry_tab.dirty_changed.connect(self._sync_geometry_tab_title)
        self.solver_tab.files_exported.connect(self.files_exported.emit)

    def _sync_geometry_tab_title(self, dirty: bool) -> None:
        index = self.indexOf(self.geometry_tab)
        if index >= 0:
            self.setTabText(index, "Geometry*" if dirty else "Geometry")

    def solve_is_running(self) -> bool:
        # Retain the historical name for unified-host compatibility.  Both a
        # field solve and a boundary-density diagnostic own a QThread whose
        # destruction must be blocked during application close.
        return bool(self.solver_tab.job_is_running())

    def attach_material_artifact(
        self, artifact_kind: str, csv_path: str
    ) -> bool:
        """Attach a typed nominal FREDDY CSV to the current Geometry tab."""

        attached = self.geometry_tab.attach_material_artifact(
            artifact_kind, csv_path
        )
        if attached:
            self.setCurrentWidget(self.geometry_tab)
        return bool(attached)

    def request_close(self, parent=None) -> bool:
        """Resolve unsaved native geometry before a host closes."""

        if self.solve_is_running():
            return False
        return bool(self.geometry_tab.request_close(parent or self))


class GhostMainWindow(QMainWindow):
    """Host the existing geometry editor and solver as one application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GHOST 2-D RCS Solver")
        self.resize(1500, 900)
        self.setMinimumSize(1000, 650)

        self.workspace = GhostWorkspace(self)
        # Backward-compatible attributes for scripts/tests that used the thin
        # standalone window directly.
        self.tabs = self.workspace
        self.geometry_tab = self.workspace.geometry_tab
        self.solver_tab = self.workspace.solver_tab
        self.setCentralWidget(self.workspace)

        self.statusBar().showMessage(
            "Load or build a geometry, validate it, then open the Solver tab."
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self.workspace.solve_is_running():
            QMessageBox.warning(
                self,
                "Solver Task Still Running",
                "A solver task is still running. Click Cancel in the Solver tab, "
                "wait for cancellation to finish, and then close GHOST.",
            )
            self.tabs.setCurrentWidget(self.solver_tab)
            event.ignore()
            return
        if not self.workspace.request_close(self):
            self.tabs.setCurrentWidget(self.geometry_tab)
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
