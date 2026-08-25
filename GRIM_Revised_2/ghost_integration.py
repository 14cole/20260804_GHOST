"""Optional application-shell integration for the GHOST solver workspace.

GRIM remains usable as a viewer when the solver backend is not installed.  The
single-checkout distribution bundles the authoritative backend at
``tools/GHOST/Backend``.  An explicit environment override remains available
for development, but backend discovery never falls through to an unrelated
sibling checkout or whatever flat modules happen to be on ``sys.path``.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget

from grim_diagnostics import GHOST_SENTINELS


GHOST_BACKEND_ENV = "GHOST_BACKEND_PATH"

_GHOST_WORKSPACE_FILES = GHOST_SENTINELS


def ghost_backend_candidates(explicit: str | os.PathLike[str] | None = None):
    """Return the one authoritative candidate for flat GHOST modules.

    A caller argument or ``GHOST_BACKEND_PATH`` is an explicit override, not
    a hint.  If it is stale or incomplete, discovery must report that problem
    rather than silently selecting another checkout.
    """

    if explicit:
        raw = Path(explicit)
    else:
        configured = os.environ.get(GHOST_BACKEND_ENV, "").strip()
        if configured:
            raw = Path(configured)
        else:
            here = Path(__file__).resolve()
            # Single-checkout distribution:
            #   <repo>/GRIM_Revised_2/ghost_integration.py
            #   <repo>/tools/GHOST/Backend/ghost_gui.py
            raw = here.parents[1] / "tools" / "GHOST" / "Backend"

    candidate = raw.expanduser().resolve()
    if candidate.name.lower() != "backend" and (candidate / "Backend").is_dir():
        candidate = (candidate / "Backend").resolve()
    yield candidate


def discover_ghost_backend(
    explicit: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Locate a directory containing the reusable GHOST GUI workspace."""

    for candidate in ghost_backend_candidates(explicit):
        if all(
            (candidate / filename).is_file()
            for filename in _GHOST_WORKSPACE_FILES
        ):
            return candidate
    return None


def _module_origin(module) -> Path | None:
    """Return a resolved source path for a loaded module, when it has one."""

    value = getattr(module, "__file__", None)
    if not value:
        spec = getattr(module, "__spec__", None)
        value = getattr(spec, "origin", None)
    if not value or value in {"built-in", "frozen"}:
        return None
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _backend_module_names(backend: Path) -> set[str]:
    """Names owned by the selected flat-module backend."""

    return {
        path.stem
        for path in backend.glob("*.py")
        if path.stem != "__init__"
    }


def _assert_loaded_module_origins(backend: Path) -> None:
    """Reject a process containing GHOST modules from another checkout."""

    conflicts = []
    for name in sorted(_backend_module_names(backend)):
        module = sys.modules.get(name)
        if module is None:
            continue
        origin = _module_origin(module)
        if origin is None or origin.parent != backend:
            shown = str(origin) if origin is not None else "unknown origin"
            conflicts.append(f"{name} ({shown})")
    if conflicts:
        raise ImportError(
            "GHOST cannot mix backend modules from different checkouts. "
            f"Selected backend: {backend}. Already loaded elsewhere: "
            + ", ".join(conflicts)
            + ". Restart GRIM after correcting GHOST_BACKEND_PATH."
        )


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
    if backend is None:
        attempted = next(ghost_backend_candidates(backend_path), None)
        expected = ", ".join(_GHOST_WORKSPACE_FILES)
        raise ImportError(
            "No complete GHOST backend was found. "
            f"Expected {expected} in {attempted}. Keep tools/GHOST with this "
            f"checkout or set {GHOST_BACKEND_ENV} explicitly to one complete "
            "Backend folder."
        )

    expected_module = backend / f"{name}.py"
    if not expected_module.is_file():
        raise ImportError(
            f"The selected GHOST backend does not contain {name}.py: {backend}"
        )

    _assert_loaded_module_origins(backend)
    backend_text = str(backend)
    # GHOST currently uses flat imports, including lazy imports during
    # solve/export.  Put the selected backend first even if it was already
    # present later in sys.path; origin validation prevents a mixed process.
    sys.path[:] = [entry for entry in sys.path if entry != backend_text]
    sys.path.insert(0, backend_text)

    module = importlib.import_module(name)
    origin = _module_origin(module)
    if origin is None or origin != expected_module.resolve():
        shown = str(origin) if origin is not None else "unknown origin"
        raise ImportError(
            f"GHOST loaded {name} from {shown}, not the selected backend "
            f"{expected_module.resolve()}. Restart GRIM after removing stale "
            "flat modules or correcting GHOST_BACKEND_PATH."
        )
    _assert_loaded_module_origins(backend)
    return module


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
                "Keep tools/GHOST with this GRIM checkout, or set "
                f"{GHOST_BACKEND_ENV} explicitly to one complete Backend "
                "folder.\n\n"
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

    def attach_material_artifact(
        self,
        kind: str,
        path: str | os.PathLike[str],
    ) -> bool:
        """Attach a FREDDY material/IBC artifact to the active GHOST geometry."""

        if self.workspace is None:
            QMessageBox.warning(
                self,
                "GHOST Unavailable",
                "The material file was exported, but it could not be attached "
                "because the GHOST workspace is unavailable. Open the GHOST "
                "tab for backend details, then attach the file after GHOST loads.",
            )
            return False
        attach = getattr(self.workspace, "attach_material_artifact", None)
        if not callable(attach):
            QMessageBox.warning(
                self,
                "GHOST Attachment Unavailable",
                "The material file was exported, but this GHOST backend does "
                "not support automatic attachment. Update tools/GHOST with "
                "the rest of this GRIM checkout, then retry.",
            )
            return False
        try:
            return bool(attach(str(kind), os.fspath(path)))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "GHOST Attachment Failed",
                "The material file was exported, but GHOST could not attach "
                f"it to the current geometry.\n\nDetails: {exc}",
            )
            return False
