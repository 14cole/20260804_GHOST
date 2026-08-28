"""Optional application-shell integration for the FREDDY IBC workspace.

The bundled FREDDY source is intentionally kept authoritative and isolated.
Its ``ibc`` package is loaded below a GRIM-private package name so embedding
FREDDY cannot replace (or be replaced by) an unrelated top-level ``ibc``
package in the user's Python environment.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from grim_diagnostics import FREDDY_SENTINELS


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

    required = FREDDY_SENTINELS
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


def freddy_theme_from_application_palette(
    palette: Mapping[str, object],
) -> dict[str, object]:
    """Translate GRIM application roles into FREDDY's richer theme contract."""

    required = {
        "win_bg",
        "panel_bg",
        "head_bg",
        "text",
        "muted",
        "border",
        "hover",
        "checked_bg",
        "checked_border",
        "grid",
    }
    missing = sorted(required.difference(palette))
    if missing:
        raise ValueError(
            "Application palette is missing FREDDY roles: "
            + ", ".join(missing)
        )

    def color(role: str) -> str:
        return str(palette[role])

    layer_colors = tuple(palette.get("layer_colors", ()))
    if not layer_colors:
        layer_colors = (
            color("checked_bg"),
            color("checked_border"),
            color("border"),
            color("hover"),
        )
    return {
        "window_bg": color("win_bg"),
        "panel_bg": color("panel_bg"),
        "head_bg": color("head_bg"),
        "text": color("text"),
        "muted_text": color("muted"),
        "field_bg": color("panel_bg"),
        "field_fg": color("text"),
        "field_disabled_bg": color("head_bg"),
        "field_disabled_fg": color("muted"),
        "button_bg": color("panel_bg"),
        "button_active_bg": color("hover"),
        "selection_bg": color("checked_bg"),
        "selection_fg": "#ffffff",
        "accent": color("checked_border"),
        "preview_bg": color("panel_bg"),
        "preview_border": color("border"),
        "preview_outline": color("grid"),
        "preview_text": color("text"),
        "preview_empty": color("muted"),
        "preview_layer_text": "#ffffff",
        "preview_layer_border": color("panel_bg"),
        "layer_colors": [str(value) for value in layer_colors],
        "plot_bg": color("panel_bg"),
        "plot_axes_bg": color("panel_bg"),
        "plot_text": color("text"),
        "plot_spine": color("border"),
        "plot_grid": color("grid"),
        "plot_line_freq": str(
            palette.get("plot_line_freq", palette["checked_border"])
        ),
        "plot_line_angle": str(
            palette.get("plot_line_angle", palette["hover"])
        ),
        "plot_worst": str(
            palette.get("plot_worst", "#fbbf24")
        ),
        "plot_crosshair": color("text"),
    }


class FreddyIntegrationWidget(QWidget):
    """Host the authoritative FREDDY workspace inside GRIM."""

    # Deliberately bypasses GRIM's RCS loader. The combined shell connects this
    # typed artifact request directly to the embedded GHOST workspace.
    attach_to_ghost_requested = Signal(str, str)

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
        self._attachable_kind = ""
        self._attachable_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.attach_to_ghost_button = QPushButton(
            "Export and attach to current GHOST geometry"
        )
        self.attach_to_ghost_button.setEnabled(False)
        self.attach_to_ghost_button.setToolTip(
            "First compute a nominal PEC-backed IBC or export a nominal mixed "
            "material CSV. This action copies that exact solver artifact beside "
            "the active saved .geo file and adds the appropriate GHOST row."
        )
        self.attach_to_ghost_button.clicked.connect(
            self._request_attach_to_ghost
        )
        layout.addWidget(self.attach_to_ghost_button)
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
            artifact_signal = getattr(
                self.workspace, "nominal_artifact_exported", None
            )
            connector = getattr(artifact_signal, "connect", None)
            if callable(connector):
                connector(self._remember_nominal_artifact)
            clear_signal = getattr(
                self.workspace, "nominal_artifact_cleared", None
            )
            clear_connector = getattr(clear_signal, "connect", None)
            if callable(clear_connector):
                clear_connector(self._clear_attachable_artifact)
            layout.addWidget(self.workspace)

    def _remember_nominal_artifact(self, kind: str, path: str) -> None:
        """Remember only FREDDY's two solver-facing artifact types."""

        artifact_kind = str(kind).strip().lower()
        if artifact_kind not in {"ibc", "material"}:
            return
        artifact_path = Path(path).expanduser().resolve()
        if artifact_path.suffix.lower() != ".csv" or not artifact_path.is_file():
            return
        self._attachable_kind = artifact_kind
        self._attachable_path = artifact_path
        friendly_kind = "IBC" if artifact_kind == "ibc" else "material"
        self.attach_to_ghost_button.setEnabled(True)
        self.attach_to_ghost_button.setToolTip(
            f"Copy nominal {friendly_kind} CSV '{artifact_path.name}' beside "
            "the active saved GHOST .geo file and add its reference."
        )

    def _clear_attachable_artifact(self) -> None:
        self._attachable_kind = ""
        self._attachable_path = None
        self.attach_to_ghost_button.setEnabled(False)
        self.attach_to_ghost_button.setToolTip(
            "First compute a nominal PEC-backed IBC or export a nominal mixed "
            "material CSV."
        )

    def _request_attach_to_ghost(self) -> None:
        """Forward an explicit typed handoff request to embedded GHOST."""

        if self.job_is_running():
            QMessageBox.warning(
                self,
                "FREDDY Task Still Running",
                "Wait for the current FREDDY task to finish before attaching "
                "an artifact to GHOST.",
            )
            return
        path = self._attachable_path
        if (
            self._attachable_kind not in {"ibc", "material"}
            or path is None
            or path.suffix.lower() != ".csv"
            or not path.is_file()
        ):
            self._clear_attachable_artifact()
            QMessageBox.warning(
                self,
                "No Nominal FREDDY Export",
                "Compute a nominal PEC-backed IBC or export a nominal mixed "
                "material CSV first. Analysis CSVs cannot be attached to GHOST.",
            )
            return
        self.attach_to_ghost_requested.emit(
            self._attachable_kind, str(path)
        )

    def job_is_running(self) -> bool:
        """Delegate background-job state to the authoritative workspace."""

        if self.workspace is None:
            return False
        checker = getattr(self.workspace, "job_is_running", None)
        return bool(checker()) if callable(checker) else False

    def apply_application_palette(
        self,
        palette: Mapping[str, object],
    ) -> bool:
        """Apply the host palette to FREDDY chrome, previews, and plots."""

        if self.workspace is None:
            return False
        apply_host_theme = getattr(self.workspace, "apply_host_theme", None)
        if not callable(apply_host_theme):
            return False
        apply_host_theme(freddy_theme_from_application_palette(palette))
        return True

    def focus_workspace(self) -> None:
        """Give keyboard focus to FREDDY after selecting its GRIM tab."""

        if self.workspace is None:
            return
        target = self.workspace.centralWidget() or self.workspace
        target.setFocus(Qt.OtherFocusReason)
