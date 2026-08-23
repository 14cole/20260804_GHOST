from __future__ import annotations

"""Assembly workspace and dependency-free 3-D scene model.

The scene is a *display* model only.  In particular, a triangle surface may be
decimated before it is handed to Matplotlib.  The caller must retain and pass
the original surface to the feature-placement service for skin checks and
shadowing; nothing in this module performs or approximates those calculations.

All scene coordinates are metres in the documented vehicle CAD frame::

    +x = right, +y = nose, +z = up

The optional Qt/Matplotlib classes are defined when those normal GRIM runtime
dependencies are importable.  The NumPy geometry helpers and scene model stay
usable in headless tests even when GUI packages are absent.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence
from urllib.parse import quote

import numpy as np


_GUI_IMPORT_ERROR: Exception | None = None
try:  # Keep the pure scene model importable in headless/minimal environments.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment-specific
    _GUI_IMPORT_ERROR = exc


GUI_AVAILABLE = _GUI_IMPORT_ERROR is None

FEATURE_PREVIEW_ROOT_KEY = "feature-assembly"


def feature_preview_group_id(kind: str, dataset_id: str | None = None) -> str:
    """Return the stable scene ID used for one prepared feature-plan layer."""

    category = str(kind).strip().lower()
    if category == "body":
        if dataset_id is not None:
            raise ValueError("body preview group does not take a dataset_id")
        return f"{FEATURE_PREVIEW_ROOT_KEY}/body"
    if category not in {"points", "lines"}:
        raise ValueError("feature preview kind must be body, points, or lines")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"{category} preview requires a nonempty string dataset_id")
    return f"{FEATURE_PREVIEW_ROOT_KEY}/{category}/{quote(dataset_id, safe='')}"


def _group_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("scene group id must be a string")
    if not value or value != value.strip():
        raise ValueError(
            "scene group id must be nonempty and have no leading/trailing whitespace"
        )
    return value


def _finite_points(values: Any, *, label: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{label} must have shape (n, 3) with n > 0")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return np.array(points, dtype=float, copy=True)


def _bounds_from_points(points: np.ndarray) -> np.ndarray:
    return np.asarray([np.min(points, axis=0), np.max(points, axis=0)], dtype=float)


def _finite_triangles(values: Any, *, label: str = "triangles") -> np.ndarray:
    triangles = np.asarray(values, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError(f"{label} must have shape (n, 3, 3) with n > 0")
    if not np.all(np.isfinite(triangles)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return triangles


def decimate_triangles_for_display(
    triangles: Any,
    max_triangles: int | None,
) -> np.ndarray:
    """Return a deterministic display proxy, never a physics mesh.

    The six triangles owning the Cartesian extrema are retained whenever the
    cap permits it.  Remaining slots are spread evenly through source order.
    That makes repeated loads stable and normally preserves the exact scene
    bounds without bringing in a mesh-processing dependency.
    """

    source = _finite_triangles(triangles)
    if max_triangles is None:
        return np.array(source, copy=True)
    cap = int(max_triangles)
    if cap < 1:
        raise ValueError("max_triangles must be at least 1 or None")
    count = len(source)
    if count <= cap:
        return np.array(source, copy=True)

    vertices = source.reshape(-1, 3)
    extrema: set[int] = set()
    for axis in range(3):
        extrema.add(int(np.argmin(vertices[:, axis])) // 3)
        extrema.add(int(np.argmax(vertices[:, axis])) // 3)
    selected = sorted(extrema)
    if len(selected) >= cap:
        return np.array(source[np.asarray(selected[:cap], dtype=int)], copy=True)

    candidate_mask = np.ones(count, dtype=bool)
    candidate_mask[np.asarray(selected, dtype=int)] = False
    candidates = np.flatnonzero(candidate_mask)
    needed = cap - len(selected)
    # needed <= len(candidates), so a linspace with step >= 1 yields unique
    # monotonically increasing sample locations.
    locations = np.rint(
        np.linspace(0, len(candidates) - 1, needed, endpoint=True)
    ).astype(int)
    selected.extend(int(value) for value in candidates[locations])
    selected = sorted(selected)
    return np.array(source[np.asarray(selected, dtype=int)], copy=True)


def revolve_bor_profile_cad(
    profile_rho_z_m: Any,
    *,
    circumferential_samples: int = 48,
) -> np.ndarray:
    """Revolve a ``(rho, axial-z)`` BoR profile about CAD ``+y``.

    The returned triangles use ``x = rho*cos(phi)``, ``y = axial-z``, and
    ``z = rho*sin(phi)``.  Degenerate triangles at axis endpoints are removed.
    This is a visualization mesh and is not a replacement for the solver's
    generatrix or a shadow/skin surface.
    """

    profile = np.asarray(profile_rho_z_m, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2:
        raise ValueError("BoR profile must have shape (n, 2) with n >= 2")
    if not np.all(np.isfinite(profile)):
        raise ValueError("BoR profile must contain only finite rho/z coordinates")
    extent = max(1.0, float(np.max(np.abs(profile))))
    if np.any(profile[:, 0] < -64.0 * np.finfo(float).eps * extent):
        raise ValueError("BoR profile radius rho must be nonnegative")
    rho = np.maximum(profile[:, 0], 0.0)
    axial = profile[:, 1]

    samples = int(circumferential_samples)
    if samples < 3 or samples != circumferential_samples:
        raise ValueError("circumferential_samples must be an integer >= 3")
    phi = 2.0 * np.pi * np.arange(samples, dtype=float) / float(samples)
    rings = np.empty((len(profile), samples, 3), dtype=float)
    rings[:, :, 0] = rho[:, None] * np.cos(phi)[None, :]
    rings[:, :, 1] = axial[:, None]
    rings[:, :, 2] = rho[:, None] * np.sin(phi)[None, :]

    first = rings[:-1]
    second = rings[1:]
    first_next = np.roll(first, -1, axis=1)
    second_next = np.roll(second, -1, axis=1)
    triangles_a = np.stack((first, second, second_next), axis=2)
    triangles_b = np.stack((first, second_next, first_next), axis=2)
    surface = np.concatenate(
        (triangles_a.reshape(-1, 3, 3), triangles_b.reshape(-1, 3, 3)),
        axis=0,
    )
    cross = np.cross(surface[:, 1] - surface[:, 0], surface[:, 2] - surface[:, 0])
    scale = max(
        1.0,
        float(np.max(np.ptp(surface.reshape(-1, 3), axis=0))),
    )
    keep = np.linalg.norm(cross, axis=1) > 1.0e-14 * scale * scale
    surface = surface[keep]
    if len(surface) == 0:
        raise ValueError("BoR profile produces no nondegenerate display triangles")
    return surface


def _line_paths(values: Any) -> tuple[np.ndarray, ...]:
    """Normalize one path, fixed-shape paths, or a sequence of paths."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        array = np.asarray([], dtype=float)

    if array.ndim == 2 and array.shape[1:] == (3,):
        candidates: Iterable[Any] = (array,)
    elif array.ndim == 3 and array.shape[2] == 3:
        candidates = tuple(array[index] for index in range(len(array)))
    else:
        try:
            candidates = tuple(values)
        except TypeError as exc:
            raise ValueError(
                "lines must be one (n,3) path or a sequence of (n,3) paths"
            ) from exc

    paths: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        path = np.asarray(candidate, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            raise ValueError(f"line path {index} must have shape (n, 3), n >= 2")
        if not np.all(np.isfinite(path)):
            raise ValueError(f"line path {index} contains a nonfinite coordinate")
        paths.append(np.array(path, dtype=float, copy=True))
    if not paths:
        raise ValueError("lines must contain at least one path")
    return tuple(paths)


@dataclass
class AssemblySceneGroup:
    """One addressable render group in a common CAD/metre scene."""

    group_id: str
    kind: str
    geometry: Any
    bounds_m: np.ndarray
    visible: bool = True
    label: str = ""
    style: dict[str, Any] = field(default_factory=dict)
    source_count: int = 0
    display_count: int = 0
    display_only: bool = True


class AssemblySceneModel:
    """Pure NumPy scene state with stable string-addressed groups."""

    def __init__(self) -> None:
        self._groups: OrderedDict[str, AssemblySceneGroup] = OrderedDict()
        self._listeners: list[Callable[[str, str | None], None]] = []

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(self._groups)

    def group(self, group_id: str) -> AssemblySceneGroup:
        key = _group_id(group_id)
        try:
            return self._groups[key]
        except KeyError as exc:
            raise KeyError(f"unknown assembly scene group {key!r}") from exc

    def add_listener(self, callback: Callable[[str, str | None], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, str | None], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, event: str, group_id: str | None) -> None:
        for callback in tuple(self._listeners):
            callback(event, group_id)

    def _store(self, group: AssemblySceneGroup) -> AssemblySceneGroup:
        event = "replaced" if group.group_id in self._groups else "added"
        self._groups[group.group_id] = group
        self._notify(event, group.group_id)
        return group

    def clear(self) -> None:
        if not self._groups:
            return
        self._groups.clear()
        self._notify("cleared", None)

    def remove_group(self, group_id: str) -> None:
        key = _group_id(group_id)
        if key not in self._groups:
            raise KeyError(f"unknown assembly scene group {key!r}")
        del self._groups[key]
        self._notify("removed", key)

    def add_body_triangles(
        self,
        group_id: str,
        triangles_m: Any,
        *,
        max_triangles: int | None = 30_000,
        visible: bool = True,
        label: str = "Body surface",
        color: str = "#78909c",
        alpha: float = 0.32,
        edgecolor: str = "none",
    ) -> AssemblySceneGroup:
        """Add a decimated render proxy while retaining full-source bounds."""

        key = _group_id(group_id)
        source = _finite_triangles(triangles_m, label="body triangles")
        full_bounds = _bounds_from_points(source.reshape(-1, 3))
        proxy = decimate_triangles_for_display(source, max_triangles)
        return self._store(
            AssemblySceneGroup(
                key,
                "surface",
                proxy,
                full_bounds,
                bool(visible),
                str(label),
                {"color": color, "alpha": float(alpha), "edgecolor": edgecolor},
                source_count=len(source),
                display_count=len(proxy),
                display_only=True,
            )
        )

    def add_bor_profile(
        self,
        group_id: str,
        profile_rho_z_m: Any,
        *,
        circumferential_samples: int = 48,
        max_triangles: int | None = 30_000,
        visible: bool = True,
        label: str = "BoR body",
        color: str = "#78909c",
        alpha: float = 0.32,
        edgecolor: str = "none",
    ) -> AssemblySceneGroup:
        key = _group_id(group_id)
        profile = np.asarray(profile_rho_z_m, dtype=float)
        surface = revolve_bor_profile_cad(
            profile, circumferential_samples=circumferential_samples
        )
        radial = float(np.max(np.maximum(profile[:, 0], 0.0)))
        full_bounds = np.asarray(
            [
                [-radial, float(np.min(profile[:, 1])), -radial],
                [radial, float(np.max(profile[:, 1])), radial],
            ],
            dtype=float,
        )
        proxy = decimate_triangles_for_display(surface, max_triangles)
        return self._store(
            AssemblySceneGroup(
                key,
                "surface",
                proxy,
                full_bounds,
                bool(visible),
                str(label),
                {"color": color, "alpha": float(alpha), "edgecolor": edgecolor},
                source_count=len(surface),
                display_count=len(proxy),
                display_only=True,
            )
        )

    def add_points(
        self,
        group_id: str,
        points_m: Any,
        *,
        visible: bool = True,
        label: str = "Point features",
        color: str = "#ef5350",
        size: float = 28.0,
        depthshade: bool = True,
    ) -> AssemblySceneGroup:
        key = _group_id(group_id)
        points = _finite_points(points_m, label="points")
        return self._store(
            AssemblySceneGroup(
                key,
                "points",
                points,
                _bounds_from_points(points),
                bool(visible),
                str(label),
                {
                    "color": color,
                    "size": float(size),
                    "depthshade": bool(depthshade),
                },
                source_count=len(points),
                display_count=len(points),
                display_only=True,
            )
        )

    def add_lines(
        self,
        group_id: str,
        paths_m: Any,
        *,
        visible: bool = True,
        label: str = "Line features",
        color: str = "#ffca28",
        linewidth: float = 2.0,
        alpha: float = 1.0,
    ) -> AssemblySceneGroup:
        key = _group_id(group_id)
        paths = _line_paths(paths_m)
        points = np.concatenate(paths, axis=0)
        return self._store(
            AssemblySceneGroup(
                key,
                "lines",
                paths,
                _bounds_from_points(points),
                bool(visible),
                str(label),
                {
                    "color": color,
                    "linewidth": float(linewidth),
                    "alpha": float(alpha),
                },
                source_count=sum(max(0, len(path) - 1) for path in paths),
                display_count=sum(max(0, len(path) - 1) for path in paths),
                display_only=True,
            )
        )

    def set_group_visible(self, group_id: str, visible: bool) -> None:
        group = self.group(group_id)
        state = bool(visible)
        if group.visible == state:
            return
        group.visible = state
        self._notify("visibility", group.group_id)

    def bounds(self, *, visible_only: bool = True) -> np.ndarray | None:
        selected = [
            group.bounds_m
            for group in self._groups.values()
            if not visible_only or group.visible
        ]
        if not selected:
            return None
        stacked = np.stack(selected, axis=0)
        return np.asarray(
            [np.min(stacked[:, 0, :], axis=0), np.max(stacked[:, 1, :], axis=0)],
            dtype=float,
        )


@dataclass(frozen=True)
class FeatureBuildResult:
    """Neutral result envelope returned by a future placement service."""

    name: str
    payload: object
    history: str = ""


class FeatureBuildService(Protocol):
    """Extension contract; the implementation owns all placement physics."""

    def build_features(self, request: object) -> FeatureBuildResult:
        ...


if GUI_AVAILABLE:

    # Runtime-only association between a tree item and one or more stable scene
    # group IDs. AssemblyTree's .asy serializer intentionally ignores it: the
    # controller reconstructs preview geometry and bindings from source files.
    _TREE_SCENE_GROUPS_ROLE = Qt.UserRole + 80

    class AssemblySceneCanvas(FigureCanvas):
        """Qt Matplotlib canvas rendering an :class:`AssemblySceneModel`."""

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            model: AssemblySceneModel | None = None,
        ) -> None:
            self.figure = Figure(figsize=(8.0, 6.0), dpi=100, tight_layout=True)
            self.axes = self.figure.add_subplot(111, projection="3d")
            super().__init__(self.figure)
            self.setParent(parent)
            self.setMinimumSize(360, 280)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.model = model if model is not None else AssemblySceneModel()
            self.model.add_listener(self._on_model_change)
            self._model_listener_attached = True
            self.destroyed.connect(self._detach_model_listener)
            self._artists: dict[str, Any] = {}
            self._style_axes()
            self.fit_visible()

        @property
        def group_ids(self) -> tuple[str, ...]:
            return self.model.group_ids

        def _style_axes(self) -> None:
            background = "#0b1222"
            foreground = "#dbeafe"
            grid_color = "#475569"
            axis_color = "#64748b"
            self.figure.patch.set_facecolor(background)
            self.axes.set_facecolor(background)
            self.axes.set_title("Assembly geometry", color=foreground)
            self.axes.set_xlabel("X right (m)", color=foreground)
            self.axes.set_ylabel("Y nose (m)", color=foreground)
            self.axes.set_zlabel("Z up (m)", color=foreground)
            self.axes.tick_params(colors=foreground)
            pane_rgba = (11.0 / 255.0, 18.0 / 255.0, 34.0 / 255.0, 0.72)
            for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
                axis.pane.set_facecolor(pane_rgba)
                axis.pane.set_edgecolor(grid_color)
                axis.line.set_color(axis_color)
                # Axes3D keeps its grid styling in the per-axis descriptor;
                # pyplot's 2-D grid kwargs do not consistently reach it.
                axis._axinfo["grid"].update(
                    color=grid_color,
                    linewidth=0.6,
                    linestyle="-",
                )
            self.axes.view_init(elev=24.0, azim=-58.0)
            self.axes.grid(True)

        def _remove_artist(self, group_id: str) -> None:
            artist = self._artists.pop(group_id, None)
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, AttributeError):
                    pass

        def _add_artist(self, group: AssemblySceneGroup) -> None:
            self._remove_artist(group.group_id)
            style = group.style
            if group.kind == "surface":
                artist = Poly3DCollection(
                    group.geometry,
                    facecolors=style.get("color", "#78909c"),
                    edgecolors=style.get("edgecolor", "none"),
                    alpha=style.get("alpha", 0.32),
                    label=group.label,
                )
                self.axes.add_collection3d(artist)
            elif group.kind == "points":
                points = group.geometry
                artist = self.axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=style.get("color", "#ef5350"),
                    s=style.get("size", 28.0),
                    depthshade=style.get("depthshade", True),
                    label=group.label,
                )
            elif group.kind == "lines":
                artist = Line3DCollection(
                    group.geometry,
                    colors=style.get("color", "#ffca28"),
                    linewidths=style.get("linewidth", 2.0),
                    alpha=style.get("alpha", 1.0),
                    label=group.label,
                )
                self.axes.add_collection3d(artist)
            else:  # Defensive: models should never store an unknown kind.
                raise ValueError(f"unsupported assembly scene kind {group.kind!r}")
            artist.set_visible(group.visible)
            self._artists[group.group_id] = artist

        def _on_model_change(self, event: str, group_id: str | None) -> None:
            if event == "cleared":
                for key in tuple(self._artists):
                    self._remove_artist(key)
            elif event == "removed" and group_id is not None:
                self._remove_artist(group_id)
            elif event == "visibility" and group_id is not None:
                artist = self._artists.get(group_id)
                if artist is not None:
                    artist.set_visible(self.model.group(group_id).visible)
            elif group_id is not None:
                self._add_artist(self.model.group(group_id))
            self.draw_idle()

        def _detach_model_listener(self, *_args) -> None:
            """Release the model's strong callback when Qt destroys the canvas."""

            if not self._model_listener_attached:
                return
            self.model.remove_listener(self._on_model_change)
            self._model_listener_attached = False

        def clear(self) -> None:
            self.model.clear()

        def add_body_triangles(self, group_id: str, triangles_m: Any, **kwargs):
            return self.model.add_body_triangles(group_id, triangles_m, **kwargs)

        def add_bor_profile(self, group_id: str, profile_rho_z_m: Any, **kwargs):
            return self.model.add_bor_profile(group_id, profile_rho_z_m, **kwargs)

        def add_points(self, group_id: str, points_m: Any, **kwargs):
            return self.model.add_points(group_id, points_m, **kwargs)

        def add_lines(self, group_id: str, paths_m: Any, **kwargs):
            return self.model.add_lines(group_id, paths_m, **kwargs)

        def set_group_visible(self, group_id: str, visible: bool) -> None:
            self.model.set_group_visible(group_id, visible)

        def fit_visible(self, *, padding_fraction: float = 0.06) -> None:
            """Fit visible bounds with equal data scale on x/y/z."""

            padding = float(padding_fraction)
            if not np.isfinite(padding) or padding < 0.0:
                raise ValueError("padding_fraction must be finite and nonnegative")
            bounds = self.model.bounds(visible_only=True)
            if bounds is None:
                bounds = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
            center = 0.5 * (bounds[0] + bounds[1])
            span = bounds[1] - bounds[0]
            reference = max(float(np.max(span)), float(np.max(np.abs(center))), 1.0)
            minimum_span = 1.0e-6 * reference
            span = np.maximum(span, minimum_span)
            span = span * (1.0 + 2.0 * padding)
            lower = center - 0.5 * span
            upper = center + 0.5 * span
            self.axes.set_xlim(float(lower[0]), float(upper[0]))
            self.axes.set_ylim(float(lower[1]), float(upper[1]))
            self.axes.set_zlim(float(lower[2]), float(upper[2]))
            self.axes.set_box_aspect(tuple(float(value) for value in span))
            self.draw_idle()


    class AssemblyWorkspace(QWidget):
        """Standalone Assembly tab shell around one authoritative tree.

        Feature physics is intentionally injected through ``set_feature_service``
        or handled externally through ``feature_build_requested``.  This class
        never imports or reimplements point/line placement mathematics.
        """

        files_to_load = Signal(list)
        platform_built = Signal(str, object, str)
        feature_build_requested = Signal(object)
        feature_built = Signal(str, object, str)
        feature_build_failed = Signal(str)

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            assembly_tree_panel: QWidget | None = None,
            scene_model: AssemblySceneModel | None = None,
        ) -> None:
            super().__init__(parent)
            if assembly_tree_panel is None:
                from assembly_tree import AssemblyTreePanel

                assembly_tree_panel = AssemblyTreePanel(self)
            self.assembly_tree_panel = assembly_tree_panel
            self.scene_canvas = AssemblySceneCanvas(self, model=scene_model)
            self.scene_model = self.scene_canvas.model
            self._feature_service: FeatureBuildService | Callable[[object], Any] | None = None
            self._pending_visibility: dict[str, bool] = {}
            self._tree_visibility_signal = None
            self._tree_clearing_signal = None
            self._tree_preview_removing_signal = None
            self._feature_preview_group_ids: set[str] = set()

            outer = QVBoxLayout(self)
            outer.setContentsMargins(10, 10, 10, 10)
            outer.setSpacing(6)

            toolbar = QHBoxLayout()
            title = QLabel("Assembly")
            title.setStyleSheet("font-weight: 600;")
            self.btn_fit_visible = QPushButton("Fit visible")
            self.lbl_status = QLabel(
                "Attach a body and feature placement controls through the Assembly service."
            )
            self.lbl_status.setWordWrap(True)
            toolbar.addWidget(title)
            toolbar.addStretch(1)
            toolbar.addWidget(self.btn_fit_visible)
            outer.addLayout(toolbar)

            splitter = QSplitter(Qt.Horizontal)
            left_host = QWidget(self)
            left_layout = QVBoxLayout(left_host)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(6)
            left_layout.addWidget(self.assembly_tree_panel, 1)
            self.feature_controls_host = QWidget(left_host)
            self.feature_controls_layout = QVBoxLayout(self.feature_controls_host)
            self.feature_controls_layout.setContentsMargins(0, 0, 0, 0)
            self.feature_controls_host.setVisible(False)
            left_layout.addWidget(self.feature_controls_host, 0)
            splitter.addWidget(left_host)

            viewer = QWidget(self)
            viewer_layout = QVBoxLayout(viewer)
            viewer_layout.setContentsMargins(0, 0, 0, 0)
            viewer_layout.addWidget(self.scene_canvas, 1)
            viewer_layout.addWidget(self.lbl_status)
            splitter.addWidget(viewer)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([360, 1000])
            outer.addWidget(splitter, 1)
            self.splitter = splitter
            self.left_host = left_host

            self.btn_fit_visible.clicked.connect(self.scene_canvas.fit_visible)
            self._connect_tree_panel_signals()

        @property
        def group_ids(self) -> tuple[str, ...]:
            return self.scene_model.group_ids

        def _connect_tree_panel_signals(self) -> None:
            panel = self.assembly_tree_panel
            signal = getattr(panel, "files_to_load", None)
            if signal is not None:
                signal.connect(self.files_to_load.emit)
            signal = getattr(panel, "platform_built", None)
            if signal is not None:
                signal.connect(self.platform_built.emit)
            self.connect_tree_visibility()
            self.connect_tree_lifecycle()

        def connect_tree_lifecycle(self) -> bool:
            """Clear service-owned artists before the tree is fully replaced."""

            tree = getattr(self.assembly_tree_panel, "tree", None)
            if self._tree_clearing_signal is None:
                signal = getattr(tree, "tree_clearing", None)
                if signal is not None:
                    signal.connect(self.clear_feature_preview)
                    self._tree_clearing_signal = signal
            if self._tree_preview_removing_signal is None:
                signal = getattr(tree, "preview_removing", None)
                if signal is not None:
                    signal.connect(self._on_tree_preview_removing)
                    self._tree_preview_removing_signal = signal
            return (
                self._tree_clearing_signal is not None
                or self._tree_preview_removing_signal is not None
            )

        @staticmethod
        def _item_bound_group_ids(item: Any) -> set[str]:
            """Collect explicit runtime scene bindings from one tree subtree."""

            identifiers: set[str] = set()
            stack = [item]
            while stack:
                current = stack.pop()
                getter = getattr(current, "data", None)
                data = getter(0, _TREE_SCENE_GROUPS_ROLE) if callable(getter) else None
                values = (data,) if isinstance(data, str) else data
                if isinstance(values, (list, tuple, set)):
                    identifiers.update(
                        value for value in values if isinstance(value, str)
                    )
                child_count = getattr(current, "childCount", None)
                child = getattr(current, "child", None)
                if callable(child_count) and callable(child):
                    stack.extend(
                        child(index) for index in range(int(child_count()))
                    )
            return identifiers

        def _on_tree_preview_removing(self, item: Any) -> None:
            """Remove scene state owned by a typed preview subtree."""

            identifiers = self._item_bound_group_ids(item)

            # Remove stale aggregate bindings from surviving ancestors before
            # AssemblyTree refreshes their visibility after the detach.
            parent_getter = getattr(item, "parent", None)
            parent = parent_getter() if callable(parent_getter) else None
            while parent is not None:
                data = parent.data(0, _TREE_SCENE_GROUPS_ROLE)
                values = (data,) if isinstance(data, str) else data
                if isinstance(values, (list, tuple, set)):
                    filtered = tuple(
                        value for value in values if value not in identifiers
                    )
                    parent.setData(
                        0,
                        _TREE_SCENE_GROUPS_ROLE,
                        filtered if filtered else None,
                    )
                parent = parent.parent()

            # Clear bindings in the removed subtree so its final visibility
            # notification cannot recreate a deferred state for deleted IDs.
            stack = [item]
            while stack:
                current = stack.pop()
                current.setData(0, _TREE_SCENE_GROUPS_ROLE, None)
                stack.extend(
                    current.child(index) for index in range(current.childCount())
                )

            for identifier in identifiers:
                if identifier in self.scene_model.group_ids:
                    self.scene_model.remove_group(identifier)
                self._pending_visibility.pop(identifier, None)
                self._feature_preview_group_ids.discard(identifier)

        def connect_tree_visibility(self) -> bool:
            """Connect a future tree ``visibility_changed(id, bool)`` signal.

            The lookup is deliberately dynamic so this workspace works both
            before and after the AssemblyTree visibility patch lands.
            """

            if self._tree_visibility_signal is not None:
                return True
            candidates = (
                self.assembly_tree_panel,
                getattr(self.assembly_tree_panel, "tree", None),
            )
            for candidate in candidates:
                signal = getattr(candidate, "visibility_changed", None)
                if signal is not None:
                    signal.connect(self._on_tree_visibility_changed)
                    self._tree_visibility_signal = signal
                    return True
            return False

        def _on_tree_visibility_changed(self, group_id: Any, visible: Any) -> None:
            identifiers: tuple[Any, ...]
            if isinstance(group_id, str):
                identifiers = (group_id,)
            elif isinstance(group_id, (list, tuple, set)):
                identifiers = tuple(group_id)
            else:
                # The current AssemblyTree signal carries QTreeWidgetItem.
                # Resolve only explicit bindings; item text/dataset names are
                # not guaranteed unique and therefore are not stable IDs.
                data = None
                getter = getattr(group_id, "data", None)
                if callable(getter):
                    data = getter(0, _TREE_SCENE_GROUPS_ROLE)
                if isinstance(data, str):
                    identifiers = (data,)
                elif isinstance(data, (list, tuple, set)):
                    identifiers = tuple(data)
                else:
                    identifiers = ()
            for identifier in identifiers:
                if not isinstance(identifier, str):
                    continue
                self.set_group_visible(identifier, bool(visible), defer_unknown=True)

        def bind_tree_item_groups(
            self,
            item: Any,
            group_ids: str | Sequence[str],
        ) -> None:
            """Bind a tree item's preview checkbox to stable scene group IDs.

            The binding has no effect on ``build_assembly_grid`` membership or
            coherent/incoherent modes. It is a runtime preview association only.
            """

            identifiers = (group_ids,) if isinstance(group_ids, str) else tuple(group_ids)
            normalized = tuple(_group_id(value) for value in identifiers)
            if not normalized:
                raise ValueError("bind_tree_item_groups requires at least one group id")
            setter = getattr(item, "setData", None)
            if not callable(setter):
                raise TypeError("tree item must provide setData(column, role, value)")
            setter(0, _TREE_SCENE_GROUPS_ROLE, normalized)
            tree = getattr(self.assembly_tree_panel, "tree", None)
            visible = True
            item_visible = getattr(tree, "item_preview_visible", None)
            if callable(item_visible):
                visible = bool(item_visible(item))
            for identifier in normalized:
                self.set_group_visible(identifier, visible, defer_unknown=True)

        def _visibility_for_new_group(self, group_id: str, fallback: bool) -> bool:
            key = _group_id(group_id)
            return self._pending_visibility.pop(key, bool(fallback))

        def clear(self) -> None:
            self.clear_feature_preview()
            self._pending_visibility.clear()
            self.scene_canvas.clear()

        def clear_feature_preview(self) -> None:
            """Remove only the service-owned feature scene and typed tree root."""

            tree = getattr(self.assembly_tree_panel, "tree", None)
            remover = getattr(tree, "remove_preview_root", None)
            if callable(remover):
                # The tree's preview_removing signal lets us clear the bound
                # artists before the runtime-only root is detached.
                remover(FEATURE_PREVIEW_ROOT_KEY)
            for group_id in tuple(self._feature_preview_group_ids):
                if group_id in self.scene_model.group_ids:
                    self.scene_model.remove_group(group_id)
                self._pending_visibility.pop(group_id, None)
            self._feature_preview_group_ids.clear()

        @staticmethod
        def _feature_plan_geometry(plan: object):
            """Read only the prepared CAD-metre preview contract from a plan."""

            required = (
                "surface_triangles_cad_m",
                "body_profile_rho_z_m",
                "point_locations_cad_m",
                "line_paths_cad_m",
            )
            missing = [name for name in required if not hasattr(plan, name)]
            if missing:
                raise TypeError(
                    "feature preview plan is missing prepared field(s): "
                    + ", ".join(missing)
                )
            return tuple(getattr(plan, name) for name in required)

        def load_feature_preview(self, plan: object):
            """Render one already-prepared feature plan in CAD metres.

            This method never reads a CSV, mesh file, or response dataset. The
            feature workflow owns parsing, units, skin checks, normals, and all
            electromagnetic operations; GRIM consumes only its validated
            ``FeaturePreviewGeometry`` arrays.

            Point Features and Line Features containers are retained at zero
            count so the hierarchy is predictable while a user configures one
            feature type at a time.
            """

            surface, profile, point_groups, line_groups = self._feature_plan_geometry(plan)
            if not isinstance(point_groups, dict) or not isinstance(line_groups, dict):
                raise TypeError(
                    "prepared point_locations_cad_m and line_paths_cad_m must be dicts"
                )
            tree = getattr(self.assembly_tree_panel, "tree", None)
            if tree is None or not all(
                callable(getattr(tree, name, None))
                for name in (
                    "add_preview_root",
                    "add_preview_group",
                    "add_preview_item",
                    "remove_preview_root",
                )
            ):
                raise RuntimeError("AssemblyTree does not provide typed preview APIs")

            self.clear_feature_preview()
            root = tree.add_preview_root(
                "Feature Assembly", stable_key=FEATURE_PREVIEW_ROOT_KEY
            )
            scene_ids: list[str] = []
            try:
                body_id = feature_preview_group_id("body")
                if surface is not None:
                    body_group = self.add_body_triangles(body_id, surface)
                    scene_ids.append(body_id)
                    body_label = f"Body ({body_group.source_count:,} triangles)"
                    body_item = tree.add_preview_item(
                        root, body_label, stable_key=body_id
                    )
                    self.bind_tree_item_groups(body_item, body_id)
                elif profile is not None:
                    body_group = self.add_bor_profile(body_id, profile)
                    scene_ids.append(body_id)
                    profile_count = int(np.asarray(profile).shape[0])
                    body_label = f"Body (BoR, {profile_count:,} profile points)"
                    body_item = tree.add_preview_item(
                        root, body_label, stable_key=body_id
                    )
                    self.bind_tree_item_groups(body_item, body_id)
                else:
                    tree.add_preview_item(
                        root,
                        "Body (no preview geometry)",
                        stable_key=f"{FEATURE_PREVIEW_ROOT_KEY}/body-empty",
                    )

                point_counts = {
                    dataset_id: len(np.asarray(locations))
                    for dataset_id, locations in point_groups.items()
                }
                point_total = sum(point_counts.values())
                point_root = tree.add_preview_group(
                    root,
                    f"Point Features ({point_total:,})",
                    stable_key=f"{FEATURE_PREVIEW_ROOT_KEY}/points",
                )
                point_ids: list[str] = []
                for dataset_id in sorted(point_groups):
                    if not isinstance(dataset_id, str) or not dataset_id:
                        raise ValueError(
                            "prepared point preview dataset IDs must be nonempty strings"
                        )
                    count = point_counts[dataset_id]
                    group_id = feature_preview_group_id("points", dataset_id)
                    self.add_points(group_id, point_groups[dataset_id])
                    scene_ids.append(group_id)
                    item = tree.add_preview_item(
                        point_root,
                        f"{dataset_id} ({count:,} points)",
                        stable_key=group_id,
                    )
                    self.bind_tree_item_groups(item, group_id)
                    point_ids.append(group_id)
                if point_ids:
                    self.bind_tree_item_groups(point_root, point_ids)

                line_counts = {
                    dataset_id: len(paths)
                    for dataset_id, paths in line_groups.items()
                }
                line_total = sum(line_counts.values())
                line_root = tree.add_preview_group(
                    root,
                    f"Line Features ({line_total:,})",
                    stable_key=f"{FEATURE_PREVIEW_ROOT_KEY}/lines",
                )
                line_ids: list[str] = []
                for dataset_id in sorted(line_groups):
                    if not isinstance(dataset_id, str) or not dataset_id:
                        raise ValueError(
                            "prepared line preview dataset IDs must be nonempty strings"
                        )
                    paths_by_id = line_groups[dataset_id]
                    if not isinstance(paths_by_id, dict):
                        raise TypeError(
                            "each prepared line dataset preview must map line_id to a CAD path"
                        )
                    paths = tuple(paths_by_id[key] for key in sorted(paths_by_id))
                    count = line_counts[dataset_id]
                    group_id = feature_preview_group_id("lines", dataset_id)
                    self.add_lines(group_id, paths)
                    scene_ids.append(group_id)
                    item = tree.add_preview_item(
                        line_root,
                        f"{dataset_id} ({count:,} lines)",
                        stable_key=group_id,
                    )
                    self.bind_tree_item_groups(item, group_id)
                    line_ids.append(group_id)
                if line_ids:
                    self.bind_tree_item_groups(line_root, line_ids)

                if scene_ids:
                    self.bind_tree_item_groups(root, scene_ids)
                self._feature_preview_group_ids = set(scene_ids)
                tree.expandItem(root)
                tree.expandItem(point_root)
                tree.expandItem(line_root)
                self.fit_visible()
                self.lbl_status.setText(
                    f"Feature preview loaded: {point_total:,} points, "
                    f"{line_total:,} lines (CAD metres)."
                )
                return root
            except Exception:
                # Never leave a half-populated tree or stale artists after a
                # malformed third-party plan reaches this display boundary.
                self._feature_preview_group_ids.update(scene_ids)
                self.clear_feature_preview()
                raise

        def add_body_triangles(self, group_id: str, triangles_m: Any, **kwargs):
            kwargs["visible"] = self._visibility_for_new_group(
                group_id, kwargs.get("visible", True)
            )
            return self.scene_canvas.add_body_triangles(group_id, triangles_m, **kwargs)

        def add_bor_profile(self, group_id: str, profile_rho_z_m: Any, **kwargs):
            kwargs["visible"] = self._visibility_for_new_group(
                group_id, kwargs.get("visible", True)
            )
            return self.scene_canvas.add_bor_profile(group_id, profile_rho_z_m, **kwargs)

        def add_points(self, group_id: str, points_m: Any, **kwargs):
            kwargs["visible"] = self._visibility_for_new_group(
                group_id, kwargs.get("visible", True)
            )
            return self.scene_canvas.add_points(group_id, points_m, **kwargs)

        def add_lines(self, group_id: str, paths_m: Any, **kwargs):
            kwargs["visible"] = self._visibility_for_new_group(
                group_id, kwargs.get("visible", True)
            )
            return self.scene_canvas.add_lines(group_id, paths_m, **kwargs)

        def set_group_visible(
            self,
            group_id: str,
            visible: bool,
            *,
            defer_unknown: bool = False,
        ) -> None:
            key = _group_id(group_id)
            if key in self.scene_model.group_ids:
                self.scene_canvas.set_group_visible(key, bool(visible))
            elif defer_unknown:
                self._pending_visibility[key] = bool(visible)
            else:
                raise KeyError(f"unknown assembly scene group {key!r}")

        def fit_visible(self) -> None:
            self.scene_canvas.fit_visible()

        def set_feature_service(
            self,
            service: FeatureBuildService | Callable[[object], Any] | None,
        ) -> None:
            """Install an injected service; ``None`` returns to signal-only mode."""

            self._feature_service = service

        def set_feature_controls(self, widget: QWidget | None) -> None:
            """Install or clear a controller-owned point/line setup widget."""

            while self.feature_controls_layout.count():
                entry = self.feature_controls_layout.takeAt(0)
                old_widget = entry.widget()
                if old_widget is not None:
                    old_widget.setParent(None)
            if widget is None:
                self.feature_controls_host.setVisible(False)
                return
            self.feature_controls_layout.addWidget(widget)
            self.feature_controls_host.setVisible(True)

        @staticmethod
        def _normalize_feature_result(value: Any) -> FeatureBuildResult:
            if isinstance(value, FeatureBuildResult):
                return value
            if isinstance(value, dict):
                if "name" not in value or "payload" not in value:
                    raise ValueError("feature result dict requires name and payload")
                return FeatureBuildResult(
                    str(value["name"]), value["payload"], str(value.get("history", ""))
                )
            if isinstance(value, tuple) and len(value) == 3:
                return FeatureBuildResult(str(value[0]), value[1], str(value[2]))
            raise TypeError(
                "feature service must return FeatureBuildResult, a name/payload/history "
                "tuple, or a dict with those fields"
            )

        def request_feature_build(self, request: object) -> FeatureBuildResult | None:
            """Request placement without owning any electromagnetic operations."""

            self.feature_build_requested.emit(request)
            service = self._feature_service
            if service is None:
                self.lbl_status.setText(
                    "Feature build request sent to the application controller."
                )
                return None
            try:
                builder = getattr(service, "build_features", None)
                raw = builder(request) if callable(builder) else service(request)
                result = self._normalize_feature_result(raw)
            except Exception as exc:
                message = f"Feature build failed: {exc}"
                self.lbl_status.setText(message)
                self.feature_build_failed.emit(message)
                return None
            self.publish_feature_build(result.name, result.payload, result.history)
            return result

        def publish_feature_build(
            self,
            name: str,
            payload: object,
            history: str = "",
        ) -> None:
            """Completion hook for an asynchronous external feature service."""

            label = str(name).strip() or "Assembly with features"
            self.lbl_status.setText(f"Feature build completed: {label}")
            self.feature_built.emit(label, payload, str(history))


else:

    class _GuiUnavailable:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError(
                "Assembly Qt/Matplotlib widgets are unavailable; install the GRIM "
                f"runtime dependencies. Original import error: {_GUI_IMPORT_ERROR}"
            )


    class AssemblySceneCanvas(_GuiUnavailable):
        pass


    class AssemblyWorkspace(_GuiUnavailable):
        pass


__all__ = [
    "AssemblySceneCanvas",
    "AssemblySceneGroup",
    "AssemblySceneModel",
    "AssemblyWorkspace",
    "FEATURE_PREVIEW_ROOT_KEY",
    "FeatureBuildResult",
    "FeatureBuildService",
    "GUI_AVAILABLE",
    "decimate_triangles_for_display",
    "feature_preview_group_id",
    "revolve_bor_profile_cad",
]
