"""Optional application-shell integration for the FREDDY IBC workspace.

The bundled FREDDY source is intentionally kept authoritative and isolated.
Its ``ibc`` package is loaded below a GRIM-private package name so embedding
FREDDY cannot replace (or be replaced by) an unrelated top-level ``ibc``
package in the user's Python environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


FREDDY_ROOT_ENV = "FREDDY_ROOT_PATH"
FREDDY_PACKAGE_NAMESPACE = "_grim_embedded_freddy_ibc"


def freddy_root_candidates(
    explicit: str | os.PathLike[str] | None = None,
):
    """Yield deterministic candidate directories containing FREDDY."""

    seen: set[Path] = set()
    raw: list[Path] = []
    if explicit:
        raw.append(Path(explicit))
    configured = os.environ.get(FREDDY_ROOT_ENV, "").strip()
    if configured:
        raw.append(Path(configured))

    # Single-checkout distribution:
    #   <repo>/GRIM_Revised_2/freddy_integration.py
    #   <repo>/tools/FREDDY/ibc/ui.py
    raw.append(Path(__file__).resolve().parents[1] / "tools" / "FREDDY")

    for value in raw:
        candidate = value.expanduser().resolve()
        # Accept either the FREDDY root or its ibc package directory as an
        # explicit/environment override, while exposing one canonical root.
        if candidate.name.lower() == "ibc" and candidate.is_dir():
            candidate = candidate.parent
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def discover_freddy_root(
    explicit: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Locate a complete FREDDY package without importing it."""

    required = (
        Path("ibc") / "__init__.py",
        Path("ibc") / "compute.py",
        Path("ibc") / "io.py",
        Path("ibc") / "ui.py",
    )
    for candidate in freddy_root_candidates(explicit):
        if all((candidate / relative).is_file() for relative in required):
            return candidate
    return None


def _package_origin(module: ModuleType) -> Path | None:
    paths = getattr(module, "__path__", None)
    if not paths:
        return None
    try:
        return Path(next(iter(paths))).resolve()
    except (StopIteration, OSError, TypeError):
        return None


def load_freddy_package(
    root_path: str | os.PathLike[str] | None = None,
) -> ModuleType:
    """Load FREDDY's ``ibc`` package under a private package namespace."""

    root = discover_freddy_root(root_path)
    if root is None:
        raise ImportError(
            "Could not locate FREDDY (expected an ibc package below its root)."
        )
    package_dir = (root / "ibc").resolve()

    existing = sys.modules.get(FREDDY_PACKAGE_NAMESPACE)
    if existing is not None:
        if _package_origin(existing) == package_dir:
            return existing
        raise ImportError(
            f"{FREDDY_PACKAGE_NAMESPACE} is already loaded from another path."
        )

    init_path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        FREDDY_PACKAGE_NAMESPACE,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create a package loader for {init_path}.")

    module = importlib.util.module_from_spec(spec)
    before = set(sys.modules)
    sys.modules[FREDDY_PACKAGE_NAMESPACE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Do not leave a half-imported private package behind after a failed
        # optional integration attempt.
        for name in set(sys.modules) - before:
            if name == FREDDY_PACKAGE_NAMESPACE or name.startswith(
                FREDDY_PACKAGE_NAMESPACE + "."
            ):
                sys.modules.pop(name, None)
        sys.modules.pop(FREDDY_PACKAGE_NAMESPACE, None)
        raise
    return module


def load_freddy_ui_module(
    root_path: str | os.PathLike[str] | None = None,
) -> ModuleType:
    """Import FREDDY's authoritative UI through its private package."""

    load_freddy_package(root_path)
    return importlib.import_module(FREDDY_PACKAGE_NAMESPACE + ".ui")


def _load_impedance_gui_class(root: Path | None):
    module = load_freddy_ui_module(root)
    gui_class = getattr(module, "ImpedanceGui", None)
    if gui_class is None:
        raise ImportError("The located FREDDY package has no ImpedanceGui.")
    if not bool(getattr(module, "QT_AVAILABLE", True)):
        raise ImportError("FREDDY requires PySide6 for its GUI workspace.")
    return gui_class


class FreddyIntegrationWidget(QWidget):
    """Host the authoritative FREDDY workspace inside GRIM."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        root_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.root_path = discover_freddy_root(root_path)
        self.workspace = None
        self.load_error = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            gui_class = _load_impedance_gui_class(self.root_path)
            # ImpedanceGui is also FREDDY's standalone QMainWindow. Its parent
            # contract makes the same authoritative implementation a child
            # widget here, without a second UI or physics path.
            workspace = gui_class(self)
            workspace.setWindowFlags(Qt.Widget)
            self.workspace = workspace
        except Exception as exc:
            self.load_error = str(exc)
            message = QLabel(
                "FREDDY material/IBC workspace is unavailable.\n\n"
                "Keep tools/FREDDY with this GRIM checkout, or set "
                f"{FREDDY_ROOT_ENV} to the FREDDY root folder.\n\n"
                f"Details: {self.load_error}"
            )
            message.setWordWrap(True)
            layout.addWidget(message)
        else:
            layout.addWidget(self.workspace)

    def job_is_running(self) -> bool:
        """Delegate background-job state to the authoritative workspace."""

        if self.workspace is None:
            return False
        checker = getattr(self.workspace, "job_is_running", None)
        return bool(checker()) if callable(checker) else False

    def focus_workspace(self) -> None:
        """Give keyboard focus to FREDDY after selecting its GRIM tab."""

        if self.workspace is None:
            return
        target = self.workspace.centralWidget() or self.workspace
        target.setFocus(Qt.OtherFocusReason)
