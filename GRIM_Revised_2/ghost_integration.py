"""Optional application-shell integration for the GHOST solver workspace.

GRIM remains usable as a viewer when the solver backend is not installed.  In
the combined source-tree layout used by this project, the bridge also discovers
a sibling ``rcs-acceptance/Backend`` directory.  A packaged deployment can put
the same modules on Python's import path and needs no environment setting.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


GHOST_BACKEND_ENV = "GHOST_BACKEND_PATH"


def ghost_backend_candidates(explicit: str | os.PathLike[str] | None = None):
    """Return deterministic candidate directories for flat GHOST modules."""

    seen: set[Path] = set()
    raw = []
    if explicit:
        raw.append(Path(explicit))
    configured = os.environ.get(GHOST_BACKEND_ENV, "").strip()
    if configured:
        raw.append(Path(configured))
    here = Path(__file__).resolve()
    # Development/worktree layout:
    #   <shared>/grim-acceptance/GRIM_Revised_2/ghost_integration.py
    #   <shared>/rcs-acceptance/Backend/ghost_gui.py
    raw.append(here.parents[2] / "rcs-acceptance" / "Backend")
    for value in raw:
        candidate = value.expanduser().resolve()
        if candidate.name.lower() != "backend" and (
            candidate / "Backend"
        ).is_dir():
            candidate = (candidate / "Backend").resolve()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def discover_ghost_backend(
    explicit: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Locate a directory containing the reusable GHOST GUI workspace."""

    for candidate in ghost_backend_candidates(explicit):
        if all(
            (candidate / filename).is_file()
            for filename in ("ghost_gui.py", "geometry_tab.py", "solver_tab.py")
        ):
            return candidate
    return None


def load_ghost_module(
    module_name: str,
    backend_path: str | os.PathLike[str] | None = None,
):
    """Import one authoritative GHOST backend module.

    This is the shared bridge for the embedded solver workspace and the
    Assembly feature controller.  Keeping path discovery here prevents the
    GRIM widgets from duplicating GHOST's current flat-module packaging
    workaround.  The imported module still owns all solver and feature
    physics.
    """

    name = str(module_name).strip()
    if not name or "." in name or not name.isidentifier():
        raise ValueError("GHOST module_name must be one top-level Python identifier")
    backend = discover_ghost_backend(backend_path)
    if backend is not None:
        backend_text = str(backend)
        if backend_text not in sys.path:
            # GHOST's current modules use flat imports, including lazy imports
            # during solve/export, so this compatibility path must remain for
            # the lifetime of the application.  A future namespaced package can
            # remove this bridge without changing the widget contract.
            sys.path.insert(0, backend_text)
    return importlib.import_module(name)


def _load_workspace_class(backend: Path | None):
    """Import ``GhostWorkspace`` while preserving flat-backend compatibility."""

    module = load_ghost_module("ghost_gui", backend)
    workspace = getattr(module, "GhostWorkspace", None)
    if workspace is None:
        raise ImportError(
            "The located GHOST backend predates the reusable GhostWorkspace."
        )
    return workspace


class GhostIntegrationWidget(QWidget):
    """Host GHOST inside GRIM, or show an actionable unavailable message."""

    files_exported = Signal(list, str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        backend_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend_path = discover_ghost_backend(backend_path)
        self.workspace = None
        self.load_error = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            workspace_class = _load_workspace_class(self.backend_path)
            self.workspace = workspace_class(self)
        except Exception as exc:
            self.load_error = str(exc)
            message = QLabel(
                "GHOST solver workspace is unavailable.\n\n"
                "Install the GHOST backend modules, place the rcs-acceptance "
                "checkout beside this GRIM checkout, or set "
                f"{GHOST_BACKEND_ENV} to its Backend folder.\n\n"
                f"Details: {self.load_error}"
            )
            message.setWordWrap(True)
            layout.addWidget(message)
        else:
            layout.addWidget(self.workspace)
            self.workspace.files_exported.connect(self.files_exported.emit)

    def solve_is_running(self) -> bool:
        return bool(
            self.workspace is not None
            and self.workspace.solve_is_running()
        )

    def focus_solver(self) -> None:
        if self.workspace is not None:
            self.workspace.setCurrentWidget(self.workspace.solver_tab)
