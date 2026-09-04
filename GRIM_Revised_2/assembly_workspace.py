from __future__ import annotations

"""Assembly workspace and dependency-free 3-D scene model.

The scene is a *display* model only.  In particular, a triangle surface may be
decimated before it is handed to Matplotlib.  The caller must retain and pass
the original surface to the feature-placement service for skin checks and
shadowing; nothing in this module performs or approximates those calculations.

All scene coordinates are meters in the documented vehicle CAD frame::

    +x = right, +y = nose, +z = up

The optional Qt/Matplotlib classes are defined when those normal GRIM runtime
dependencies are importable.  The NumPy geometry helpers and scene model stay
usable in headless tests even when GUI packages are absent.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence
from urllib.parse import quote

import numpy as np


_GUI_IMPORT_ERROR: Exception | None = None
try:  # Keep the pure scene model importable in headless/minimal environments.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSlider,
        QSplitter,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment-specific
    _GUI_IMPORT_ERROR = exc


GUI_AVAILABLE = _GUI_IMPORT_ERROR is None

FEATURE_PREVIEW_ROOT_KEY = "feature-assembly"
FEATURE_SELECTION_GROUP_KEY = f"{FEATURE_PREVIEW_ROOT_KEY}/selection"

DISPLAY_UNIT_SPECS = {
    "Meters": ("m", 1.0),
    "Inches": ("in", 1.0 / 0.0254),
    "Feet": ("ft", 1.0 / 0.3048),
}

TRIANGLE_DETAIL_CAPS = {
    "Fast": 4_000,
    "Balanced": 12_000,
    "High": 30_000,
}
MAX_DISPLAY_TRIANGLES = max(TRIANGLE_DETAIL_CAPS.values())

BODY_RENDER_MODES = ("Solid", "Solid + edges", "Wireframe")

# Camera names include the visible CAD plane so the view remains physically
# unambiguous even when a vehicle's informal "front" naming differs by team.
CAMERA_PRESETS = {
    "Isometric": (24.0, -58.0),
    "Nose (X–Z)": (0.0, 90.0),
    "Right side (Y–Z)": (0.0, 0.0),
    "Top (X–Y)": (90.0, -90.0),
}

NORMAL_VECTOR_COLOR = "#f472b6"
ROLL_VECTOR_COLOR = "#c084fc"
LINE_TANGENT_VECTOR_COLOR = "#67e8f9"
LINE_BINORMAL_VECTOR_COLOR = "#818cf8"
DEFAULT_ORIENTATION_VECTOR_LENGTH_M = 0.0254
ORIENTATION_VECTOR_FRACTION = 0.07
MAX_LINE_FRAME_ARROWS = 250


def display_unit_spec(name: str) -> tuple[str, float]:
    """Return the axis suffix and meters-to-display scale for one UI choice."""

    key = str(name).strip().casefold()
    if key in {"meter", "metre", "metres"}:
        key = "meters"
    for label, spec in DISPLAY_UNIT_SPECS.items():
        if label.casefold() == key:
            return spec
    raise ValueError(
        "display units must be Meters, Inches, or Feet"
    )


def format_length_tick(value_m: float, units: str) -> str:
    """Format a meter-valued axis coordinate without changing scene data."""

    _suffix, scale = display_unit_spec(units)
    value = float(value_m) * scale
    if not np.isfinite(value):
        return ""
    if abs(value) < 5.0e-13:
        value = 0.0
    return f"{value:.6g}"


def triangle_detail_cap(name: str) -> int | None:
    """Resolve a named visualization-only triangle detail level."""

    key = str(name).strip().casefold()
    for label, cap in TRIANGLE_DETAIL_CAPS.items():
        if label.casefold() == key:
            return cap
    raise ValueError(
        "triangle detail must be Fast, Balanced, or High"
    )


def normalize_body_render_mode(name: str) -> str:
    """Return the canonical body rendering label used by the GUI/model."""

    key = str(name).strip().casefold()
    for label in BODY_RENDER_MODES:
        if label.casefold() == key:
            return label
    raise ValueError(
        "body rendering must be Solid, Solid + edges, or Wireframe"
    )


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


def _finite_points(
    values: Any,
    *,
    label: str,
    copy: bool = True,
) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{label} must have shape (n, 3) with n > 0")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} must contain only finite coordinates")
    if copy:
        return np.array(points, dtype=float, copy=True)
    return points


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
    if needed == 1:
        locations = np.asarray([len(candidates) // 2], dtype=int)
    else:
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


def _line_paths(values: Any, *, copy: bool = True) -> tuple[np.ndarray, ...]:
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
        paths.append(
            np.array(path, dtype=float, copy=True) if copy else path
        )
    if not paths:
        raise ValueError("lines must contain at least one path")
    return tuple(paths)


def orientation_vector_length_m(bounds_m: Any) -> float:
    """Choose a visible, scene-relative arrow length in meters.

    The value is display-only.  A one-inch fallback keeps an isolated point
    frame visible when there is no body or path extent from which to derive a
    scale.  No coordinate or orientation value is changed.
    """

    bounds = np.asarray(bounds_m, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("orientation bounds must be finite with shape (2, 3)")
    span = bounds[1] - bounds[0]
    if np.any(span < 0.0):
        raise ValueError("orientation bounds must be ordered lower than upper")
    extent = float(np.max(span))
    if extent <= 1.0e-12:
        return DEFAULT_ORIENTATION_VECTOR_LENGTH_M
    return ORIENTATION_VECTOR_FRACTION * extent


def _feature_preview_nonvector_bounds(
    body_bounds_m: Any | None,
    point_groups: dict[str, Any],
    line_groups: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Bound body/locations/paths before orientation arrows are added."""

    bounds: list[np.ndarray] = []
    if body_bounds_m is not None:
        body_bounds = np.asarray(body_bounds_m, dtype=float)
        if body_bounds.shape != (2, 3) or not np.all(np.isfinite(body_bounds)):
            raise ValueError("body preview bounds must be finite with shape (2, 3)")
        bounds.append(body_bounds)
    for dataset_id, values in point_groups.items():
        # This pass only reads the prepared arrays to establish a common
        # orientation-arrow scale.  The scene-model add pass below owns the
        # defensive copies it retains, so copying the full point set here would
        # double peak preview memory for no validation benefit.
        points = _finite_points(
            values,
            label=f"point dataset {dataset_id!r}",
            copy=False,
        )
        bounds.append(_bounds_from_points(points))
    for dataset_id, paths_by_id in line_groups.items():
        if not isinstance(paths_by_id, dict):
            raise TypeError(
                "each prepared line dataset preview must map line_id to a CAD path"
            )
        for line_id, values in paths_by_id.items():
            # Avoid both path copies and a second, concatenated copy of every
            # vertex.  Per-path extrema combine exactly into the same overall
            # bounds and retain all shape/finite-value validation.
            paths = _line_paths(values, copy=False)
            for path in paths:
                bounds.append(_bounds_from_points(path))
    if not bounds:
        return np.zeros((2, 3), dtype=float)
    stacked = np.stack(bounds, axis=0)
    return np.asarray(
        [np.min(stacked[:, 0, :], axis=0), np.max(stacked[:, 1, :], axis=0)],
        dtype=float,
    )


def _vector_rows(values: Any, *, count: int, label: str) -> np.ndarray:
    vectors = np.asarray(values, dtype=float)
    if vectors.shape != (count, 3):
        raise ValueError(f"{label} must have shape ({count}, 3)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{label} must contain only finite vectors")
    return np.array(vectors, dtype=float, copy=True)


def _point_orientation_overlays(
    points: np.ndarray,
    normals: Any | None,
    roll_references: Any | None,
) -> dict[str, np.ndarray]:
    """Normalize drawable point-frame arrows without validating placement.

    Zero normals and zero/parallel roll references are omitted from the input
    preview instead of becoming an early physical-validation gate.  The
    authoritative GHOST validation reports those errors when the user selects
    Validate Placements.
    """

    overlays = {
        "normal_origins": np.empty((0, 3), dtype=float),
        "normal_directions": np.empty((0, 3), dtype=float),
        "roll_origins": np.empty((0, 3), dtype=float),
        "roll_directions": np.empty((0, 3), dtype=float),
    }
    if normals is None:
        if roll_references is not None:
            raise ValueError("point roll references require point normals")
        return overlays

    normal_rows = _vector_rows(normals, count=len(points), label="point normals")
    normal_magnitudes = np.linalg.norm(normal_rows, axis=1)
    normal_valid = normal_magnitudes > 1.0e-12
    normal_units = np.zeros_like(normal_rows)
    normal_units[normal_valid] = (
        normal_rows[normal_valid] / normal_magnitudes[normal_valid, None]
    )
    overlays["normal_origins"] = np.array(points[normal_valid], copy=True)
    overlays["normal_directions"] = np.array(
        normal_units[normal_valid], copy=True
    )

    if roll_references is None:
        return overlays
    roll_rows = _vector_rows(
        roll_references, count=len(points), label="point roll references"
    )
    # The solver defines local +x as this projected direction.  Drawing the
    # projection, rather than the raw reference, makes the visible frame match
    # the actual point-scatterer rotation convention.
    projected = roll_rows - (
        np.sum(roll_rows * normal_units, axis=1)[:, None] * normal_units
    )
    roll_magnitudes = np.linalg.norm(projected, axis=1)
    roll_valid = normal_valid & (roll_magnitudes > 1.0e-12)
    projected[roll_valid] /= roll_magnitudes[roll_valid, None]
    overlays["roll_origins"] = np.array(points[roll_valid], copy=True)
    overlays["roll_directions"] = np.array(projected[roll_valid], copy=True)
    return overlays


def _line_endpoint_orientation_overlays(
    paths: tuple[np.ndarray, ...],
    endpoint_normals: Any | None,
) -> dict[str, np.ndarray]:
    """Return every drawable segment-end normal, preserving duplicates."""

    empty = {
        "normal_origins": np.empty((0, 3), dtype=float),
        "normal_directions": np.empty((0, 3), dtype=float),
    }
    if endpoint_normals is None:
        return empty

    if len(paths) == 1:
        array = np.asarray(endpoint_normals, dtype=float)
        if array.ndim == 3:
            candidates: tuple[Any, ...] = (array,)
        else:
            try:
                candidates = tuple(endpoint_normals)
            except TypeError as exc:
                raise ValueError(
                    "line endpoint normals must match the preview paths"
                ) from exc
    else:
        try:
            candidates = tuple(endpoint_normals)
        except TypeError as exc:
            raise ValueError(
                "line endpoint normals must match the preview paths"
            ) from exc
    if len(candidates) != len(paths):
        raise ValueError("line endpoint normals must match the preview paths")

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for index, (path, values) in enumerate(zip(paths, candidates)):
        vectors = np.asarray(values, dtype=float)
        expected = (len(path) - 1, 2, 3)
        if vectors.shape != expected:
            raise ValueError(
                f"line endpoint normals {index} must have shape {expected}"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError(
                f"line endpoint normals {index} contain a nonfinite vector"
            )
        segment_origins = np.stack((path[:-1], path[1:]), axis=1)
        flat_vectors = vectors.reshape(-1, 3)
        magnitudes = np.linalg.norm(flat_vectors, axis=1)
        valid = magnitudes > 1.0e-12
        # Flattening the (segment, endpoint) axes deliberately retains both
        # copies of a shared vertex. Their supplied normals may differ.
        origins.append(segment_origins.reshape(-1, 3)[valid])
        directions.append(flat_vectors[valid] / magnitudes[valid, None])
    if not origins:
        return empty
    return {
        "normal_origins": np.concatenate(origins, axis=0),
        "normal_directions": np.concatenate(directions, axis=0),
    }


def _line_frame_orientation_overlays(
    paths: tuple[np.ndarray, ...],
    endpoint_normals: Any | None,
    *,
    max_frames: int = MAX_LINE_FRAME_ARROWS,
) -> dict[str, Any]:
    """Build display-only signed ``+t``/``+b`` frames along line paths.

    ``+t`` follows increasing CSV ``segment_index``. With usable endpoint
    normals it is projected into the local skin plane exactly as the line
    expansion's frame construction does; ``+b = +t × +n``. A deterministic
    cap prevents a finely segmented vehicle seam from flooding Matplotlib with
    thousands of quiver arrows. Geometry and physics remain full-resolution.
    """

    cap = int(max_frames)
    if cap < 1:
        raise ValueError("max_frames must be a positive integer")
    normal_candidates: tuple[Any, ...] | None = None
    if endpoint_normals is not None:
        if len(paths) == 1:
            array = np.asarray(endpoint_normals, dtype=float)
            if array.ndim == 3:
                normal_candidates = (array,)
            else:
                try:
                    normal_candidates = tuple(endpoint_normals)
                except TypeError as exc:
                    raise ValueError(
                        "line endpoint normals must match the preview paths"
                    ) from exc
        else:
            try:
                normal_candidates = tuple(endpoint_normals)
            except TypeError as exc:
                raise ValueError(
                    "line endpoint normals must match the preview paths"
                ) from exc
        if len(normal_candidates) != len(paths):
            raise ValueError("line endpoint normals must match the preview paths")

    tangent_origins: list[np.ndarray] = []
    tangent_directions: list[np.ndarray] = []
    binormal_origins: list[np.ndarray] = []
    binormal_directions: list[np.ndarray] = []
    for path_index, path in enumerate(paths):
        starts = path[:-1]
        ends = path[1:]
        origins = 0.5 * (starts + ends)
        chords = ends - starts
        chord_lengths = np.linalg.norm(chords, axis=1)
        chord_valid = chord_lengths > 1.0e-12
        raw_tangents = np.zeros_like(chords)
        raw_tangents[chord_valid] = (
            chords[chord_valid] / chord_lengths[chord_valid, None]
        )
        tangents = np.array(raw_tangents, copy=True)
        binormal_valid = np.zeros(len(chords), dtype=bool)
        binormals = np.zeros_like(chords)
        if normal_candidates is not None:
            normals = np.asarray(normal_candidates[path_index], dtype=float)
            expected = (len(path) - 1, 2, 3)
            if normals.shape != expected:
                raise ValueError(
                    f"line endpoint normals {path_index} must have shape {expected}"
                )
            if not np.all(np.isfinite(normals)):
                raise ValueError(
                    f"line endpoint normals {path_index} contain a nonfinite vector"
                )
            midpoint_normals = normals[:, 0, :] + normals[:, 1, :]
            normal_lengths = np.linalg.norm(midpoint_normals, axis=1)
            normal_valid = normal_lengths > 1.0e-12
            unit_normals = np.zeros_like(midpoint_normals)
            unit_normals[normal_valid] = (
                midpoint_normals[normal_valid]
                / normal_lengths[normal_valid, None]
            )
            projected = raw_tangents - (
                np.sum(raw_tangents * unit_normals, axis=1)[:, None]
                * unit_normals
            )
            projected_lengths = np.linalg.norm(projected, axis=1)
            frame_valid = chord_valid & normal_valid & (projected_lengths > 1.0e-12)
            tangents[frame_valid] = (
                projected[frame_valid] / projected_lengths[frame_valid, None]
            )
            binormals[frame_valid] = np.cross(
                tangents[frame_valid], unit_normals[frame_valid]
            )
            binormal_lengths = np.linalg.norm(binormals, axis=1)
            binormal_valid = frame_valid & (binormal_lengths > 1.0e-12)
            binormals[binormal_valid] /= binormal_lengths[binormal_valid, None]
        tangent_origins.append(origins[chord_valid])
        tangent_directions.append(tangents[chord_valid])
        if np.any(binormal_valid):
            binormal_origins.append(origins[binormal_valid])
            binormal_directions.append(binormals[binormal_valid])

    def combine(values: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(values, axis=0)
            if values
            else np.empty((0, 3), dtype=float)
        )

    all_tangent_origins = combine(tangent_origins)
    all_tangent_directions = combine(tangent_directions)
    all_binormal_origins = combine(binormal_origins)
    all_binormal_directions = combine(binormal_directions)
    source_count = len(all_tangent_origins)

    def sample_pair(
        origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(origins) <= cap:
            return origins, directions
        indices = np.rint(
            np.linspace(0, len(origins) - 1, cap, endpoint=True)
        ).astype(int)
        return origins[indices], directions[indices]

    all_tangent_origins, all_tangent_directions = sample_pair(
        all_tangent_origins, all_tangent_directions
    )
    all_binormal_origins, all_binormal_directions = sample_pair(
        all_binormal_origins, all_binormal_directions
    )
    return {
        "tangent_origins": all_tangent_origins,
        "tangent_directions": all_tangent_directions,
        "binormal_origins": all_binormal_origins,
        "binormal_directions": all_binormal_directions,
        "frame_source_count": source_count,
        "frame_display_count": len(all_tangent_origins),
    }


@dataclass
class AssemblySceneGroup:
    """One addressable render group in a common CAD/meter scene."""

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
    master_geometry: Any | None = field(default=None, repr=False)
    display_cache: dict[int | None, np.ndarray] = field(
        default_factory=dict, repr=False
    )
    detail_cap: int | None = None


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

    @staticmethod
    def _normalize_surface_cap(max_triangles: int | None) -> int | None:
        if max_triangles is None:
            return MAX_DISPLAY_TRIANGLES
        cap = int(max_triangles)
        if cap < 1:
            raise ValueError("max_triangles must be at least 1 or None")
        return cap

    @staticmethod
    def _cached_surface_proxy(
        group: AssemblySceneGroup,
        max_triangles: int | None,
    ) -> np.ndarray:
        source = group.master_geometry
        if source is None:
            raise ValueError("surface group does not retain a display master")
        cap = AssemblySceneModel._normalize_surface_cap(max_triangles)
        cached = group.display_cache.get(cap)
        if cached is not None:
            return cached
        if len(source) <= cap:
            proxy = source
        else:
            proxy = decimate_triangles_for_display(source, cap)
            proxy.setflags(write=False)
        group.display_cache[cap] = proxy
        return proxy

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
        alpha: float = 0.75,
        edgecolor: str = "none",
        render_mode: str = "Solid",
    ) -> AssemblySceneGroup:
        """Add a bounded display master while retaining full counts/bounds."""

        key = _group_id(group_id)
        source = _finite_triangles(triangles_m, label="body triangles")
        source_count = len(source)
        full_bounds = _bounds_from_points(source.reshape(-1, 3))
        # Cache only a deterministic, bounded master. Retaining the entire
        # prepared STL here can double peak memory; the backend remains the
        # authoritative owner of full physics geometry.
        master = decimate_triangles_for_display(
            source, MAX_DISPLAY_TRIANGLES
        )
        master.setflags(write=False)
        cap = self._normalize_surface_cap(max_triangles)
        group = AssemblySceneGroup(
            key,
            "surface",
            master,
            full_bounds,
            bool(visible),
            str(label),
            {
                "color": color,
                "alpha": float(alpha),
                "edgecolor": edgecolor,
                "render_mode": normalize_body_render_mode(render_mode),
            },
            source_count=source_count,
            display_count=len(master),
            display_only=True,
            master_geometry=master,
            display_cache={MAX_DISPLAY_TRIANGLES: master},
            detail_cap=cap,
        )
        proxy = self._cached_surface_proxy(group, cap)
        group.geometry = proxy
        group.display_count = len(proxy)
        return self._store(group)

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
        alpha: float = 0.75,
        edgecolor: str = "none",
        render_mode: str = "Solid",
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
        source_count = len(surface)
        master = decimate_triangles_for_display(
            surface, MAX_DISPLAY_TRIANGLES
        )
        master.setflags(write=False)
        cap = self._normalize_surface_cap(max_triangles)
        group = AssemblySceneGroup(
            key,
            "surface",
            master,
            full_bounds,
            bool(visible),
            str(label),
            {
                "color": color,
                "alpha": float(alpha),
                "edgecolor": edgecolor,
                "render_mode": normalize_body_render_mode(render_mode),
            },
            source_count=source_count,
            display_count=len(master),
            display_only=True,
            master_geometry=master,
            display_cache={MAX_DISPLAY_TRIANGLES: master},
            detail_cap=cap,
        )
        proxy = self._cached_surface_proxy(group, cap)
        group.geometry = proxy
        group.display_count = len(proxy)
        return self._store(group)

    def set_surface_detail(
        self,
        group_id: str,
        max_triangles: int | None,
        *,
        remember: bool = True,
    ) -> None:
        """Rebuild one proxy from its deterministic 30k display master.

        ``remember=False`` is reserved for temporary interaction LOD. Neither
        path changes source geometry, bounds, visibility, or physics state.
        """

        group = self.group(group_id)
        if group.kind != "surface":
            raise ValueError("triangle detail applies only to surface groups")
        cap = self._normalize_surface_cap(max_triangles)
        proxy = self._cached_surface_proxy(group, cap)
        if remember:
            group.detail_cap = cap
        if group.geometry is proxy:
            return
        group.geometry = proxy
        group.display_count = len(proxy)
        self._notify("replaced", group.group_id)

    def set_surface_rendering(
        self,
        group_id: str,
        mode: str,
        opacity: float,
    ) -> None:
        """Update surface appearance without touching its geometry or visibility."""

        group = self.group(group_id)
        if group.kind != "surface":
            raise ValueError("body rendering applies only to surface groups")
        alpha = float(opacity)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("body opacity must be finite and between 0 and 1")
        normalized_mode = normalize_body_render_mode(mode)
        if (
            group.style.get("render_mode") == normalized_mode
            and group.style.get("alpha") == alpha
        ):
            return
        group.style["render_mode"] = normalized_mode
        group.style["alpha"] = alpha
        self._notify("replaced", group.group_id)

    def add_points(
        self,
        group_id: str,
        points_m: Any,
        *,
        visible: bool = True,
        label: str = "Point features",
        color: str = "#38bdf8",
        size: float = 28.0,
        depthshade: bool = True,
        normals: Any | None = None,
        roll_references: Any | None = None,
        orientation_length_m: float = DEFAULT_ORIENTATION_VECTOR_LENGTH_M,
    ) -> AssemblySceneGroup:
        key = _group_id(group_id)
        points = _finite_points(points_m, label="points")
        length = float(orientation_length_m)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("orientation_length_m must be finite and positive")
        overlays = _point_orientation_overlays(
            points, normals, roll_references
        )
        bounds_points = [points]
        for prefix in ("normal", "roll"):
            origins = overlays[f"{prefix}_origins"]
            directions = overlays[f"{prefix}_directions"]
            if len(origins):
                bounds_points.extend((origins, origins + length * directions))
        return self._store(
            AssemblySceneGroup(
                key,
                "points",
                points,
                _bounds_from_points(np.concatenate(bounds_points, axis=0)),
                bool(visible),
                str(label),
                {
                    "color": color,
                    "size": float(size),
                    "depthshade": bool(depthshade),
                    "orientation_length_m": length,
                    "normal_color": NORMAL_VECTOR_COLOR,
                    "roll_color": ROLL_VECTOR_COLOR,
                    **overlays,
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
        color: str = "#f59e0b",
        linewidth: float = 2.0,
        alpha: float = 1.0,
        endpoint_normals: Any | None = None,
        orientation_length_m: float = DEFAULT_ORIENTATION_VECTOR_LENGTH_M,
    ) -> AssemblySceneGroup:
        key = _group_id(group_id)
        paths = _line_paths(paths_m)
        points = np.concatenate(paths, axis=0)
        length = float(orientation_length_m)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("orientation_length_m must be finite and positive")
        overlays = _line_endpoint_orientation_overlays(paths, endpoint_normals)
        frames = _line_frame_orientation_overlays(paths, endpoint_normals)
        bounds_points = [points]
        origins = overlays["normal_origins"]
        directions = overlays["normal_directions"]
        if len(origins):
            bounds_points.extend((origins, origins + length * directions))
        for prefix in ("tangent", "binormal"):
            frame_origins = frames[f"{prefix}_origins"]
            frame_directions = frames[f"{prefix}_directions"]
            if len(frame_origins):
                bounds_points.extend(
                    (frame_origins, frame_origins + length * frame_directions)
                )
        return self._store(
            AssemblySceneGroup(
                key,
                "lines",
                paths,
                _bounds_from_points(np.concatenate(bounds_points, axis=0)),
                bool(visible),
                str(label),
                {
                    "color": color,
                    "linewidth": float(linewidth),
                    "alpha": float(alpha),
                    "orientation_length_m": length,
                    "normal_color": NORMAL_VECTOR_COLOR,
                    "tangent_color": LINE_TANGENT_VECTOR_COLOR,
                    "binormal_color": LINE_BINORMAL_VECTOR_COLOR,
                    **overlays,
                    **frames,
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

        _FEEDBACK_DEFAULTS = {
            "empty": (
                "Nothing to preview yet\n"
                "Choose a body and/or add point or line placements"
            ),
            "loading": "Preparing 3-D placement preview\u2026",
            "error": "The 3-D preview could not be prepared",
            "hidden": (
                "All preview geometry is hidden\n"
                "Check a Show box in the Assembly tree"
            ),
            "ready": "",
        }

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            model: AssemblySceneModel | None = None,
        ) -> None:
            # Fixed margins avoid Matplotlib's expensive tight-layout pass on
            # every interactive 3-D redraw.
            self.figure = Figure(figsize=(8.0, 6.0), dpi=100)
            self.figure.subplots_adjust(
                left=0.04, right=0.94, bottom=0.07, top=0.91
            )
            self.axes = self.figure.add_subplot(111, projection="3d")
            self.axes.set_proj_type("ortho")
            super().__init__(self.figure)
            self.setParent(parent)
            self.setMinimumSize(360, 280)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.model = model if model is not None else AssemblySceneModel()
            self._display_units = "Meters"
            self._orientation_scale = 1.0
            self._orientation_vectors_visible = True
            self._orthographic_projection = True
            self._interaction_lod_enabled = True
            self._interaction_detail_caps: dict[str, int | None] = {}
            self._scene_update_depth = 0
            self._scene_draw_pending = False
            self._scene_feedback_pending = False
            self._theme: dict[str, str] = {
                "background": "#0b1222",
                "foreground": "#dbeafe",
                "grid": "#475569",
                "border": "#1e3a8a",
                "head_bg": "#172554",
                "muted": "#94a3b8",
                "accent": "#3b82f6",
            }
            self._is_dark_theme = True
            self.model.add_listener(self._on_model_change)
            self._model_listener_attached = True
            self.destroyed.connect(self._detach_model_listener)
            self._artists: dict[str, Any] = {}
            self._style_axes()
            self._feedback_artist = self.axes.text2D(
                0.5,
                0.5,
                "",
                transform=self.axes.transAxes,
                ha="center",
                va="center",
                color=self._theme["foreground"],
                fontsize=11,
                linespacing=1.5,
                bbox={
                    "boxstyle": "round,pad=0.8",
                    "facecolor": self._theme["head_bg"],
                    "edgecolor": self._theme["accent"],
                    "alpha": 0.94,
                },
                zorder=20,
            )
            self._stage_artist = self.axes.text2D(
                0.02,
                0.98,
                "",
                transform=self.axes.transAxes,
                ha="left",
                va="top",
                color=self._theme["foreground"],
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": self._theme["head_bg"],
                    "edgecolor": self._theme["accent"],
                    "alpha": 0.92,
                },
                zorder=21,
            )
            self._lod_artist = self.axes.text2D(
                0.02,
                0.02,
                "FAST ROTATION PROXY \u2022 DISPLAY ONLY",
                transform=self.axes.transAxes,
                ha="left",
                va="bottom",
                color="#fde68a",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": self._theme["head_bg"],
                    "edgecolor": "#f59e0b",
                    "alpha": 0.92,
                },
                visible=False,
                zorder=21,
            )
            self._preview_state = "empty"
            self._preview_stage = "none"
            self.set_preview_stage("none")
            for group_id in self.model.group_ids:
                self._add_artist(self.model.group(group_id))
            self.refresh_scene_feedback()
            self.set_display_units("Meters")
            self.mpl_connect("button_press_event", self._begin_interaction_lod)
            self.mpl_connect("button_release_event", self._end_interaction_lod)
            self.fit_visible()

        @property
        def group_ids(self) -> tuple[str, ...]:
            return self.model.group_ids

        def begin_scene_updates(self) -> None:
            """Defer canvas feedback scans and redraws for a scene transaction."""

            self._scene_update_depth += 1

        def end_scene_updates(self) -> None:
            """Flush one feedback update/redraw for a scene transaction."""

            if self._scene_update_depth <= 0:
                raise RuntimeError(
                    "end_scene_updates called without begin_scene_updates"
                )
            self._scene_update_depth -= 1
            if self._scene_update_depth:
                return

            feedback_pending = self._scene_feedback_pending
            draw_pending = self._scene_draw_pending
            self._scene_feedback_pending = False
            self._scene_draw_pending = False
            if feedback_pending:
                # set_feedback() requests the one final draw after deriving the
                # state from the fully populated/cleared model.
                self.refresh_scene_feedback()
            elif draw_pending:
                self.draw_idle()

        def _request_draw(self) -> None:
            if self._scene_update_depth:
                self._scene_draw_pending = True
            else:
                self.draw_idle()

        def _request_scene_feedback(self) -> None:
            if self._scene_update_depth:
                self._scene_feedback_pending = True
                self._scene_draw_pending = True
            else:
                # refresh_scene_feedback() also requests the artist redraw.
                self.refresh_scene_feedback()

        @property
        def orthographic_projection(self) -> bool:
            return self._orthographic_projection

        def set_projection_mode(self, orthographic: bool) -> None:
            """Select a display-only orthographic or perspective camera."""

            use_ortho = bool(orthographic)
            if use_ortho == self._orthographic_projection:
                return
            self.axes.set_proj_type("ortho" if use_ortho else "persp")
            self._orthographic_projection = use_ortho
            self._request_draw()

        def set_camera_preset(self, name: str) -> None:
            """Apply one CAD-axis camera preset without changing scene limits."""

            key = str(name).strip()
            try:
                elevation, azimuth = CAMERA_PRESETS[key]
            except KeyError as exc:
                raise ValueError(
                    "camera preset must be one of: " + ", ".join(CAMERA_PRESETS)
                ) from exc
            self.axes.view_init(elev=elevation, azim=azimuth)
            self._request_draw()

        def _style_axes(self, *, reset_view: bool = True) -> None:
            background = self._theme["background"]
            foreground = self._theme["foreground"]
            grid_color = self._theme["grid"]
            axis_color = self._theme["muted"]
            self.figure.patch.set_facecolor(background)
            self.axes.set_facecolor(background)
            if reset_view:
                self.axes.set_title(
                    "3-D placement preview (display only)", color=foreground
                )
                self.axes.set_xlabel("X right (m)", color=foreground)
                self.axes.set_ylabel("Y nose (m)", color=foreground)
                self.axes.set_zlabel("Z up (m)", color=foreground)
            else:
                self.axes.title.set_color(foreground)
                self.axes.xaxis.label.set_color(foreground)
                self.axes.yaxis.label.set_color(foreground)
                self.axes.zaxis.label.set_color(foreground)
            self.axes.tick_params(colors=foreground)
            pane_rgba = to_rgba(background, alpha=0.72)
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
            if reset_view:
                self.axes.view_init(elev=24.0, azim=-58.0)
            self.axes.grid(True)

        def apply_theme(self, palette: Mapping[str, object]) -> None:
            """Restyle the display-only canvas without moving its camera/data."""

            required = {
                "panel_bg", "text", "grid", "border", "head_bg",
                "muted", "checked_border",
            }
            missing = sorted(required.difference(palette))
            if missing:
                raise ValueError(
                    "Assembly theme is missing roles: " + ", ".join(missing)
                )
            self._theme = {
                "background": str(palette["panel_bg"]),
                "foreground": str(palette["text"]),
                "grid": str(palette["grid"]),
                "border": str(palette["border"]),
                "head_bg": str(palette["head_bg"]),
                "muted": str(palette["muted"]),
                "accent": str(palette["checked_border"]),
            }
            self._is_dark_theme = bool(palette.get("is_dark", True))
            self._style_axes(reset_view=False)
            for artist in (
                self._feedback_artist,
                self._stage_artist,
                self._lod_artist,
            ):
                artist.get_bbox_patch().set_facecolor(self._theme["head_bg"])
            self._lod_artist.set_color(
                "#fde68a" if self._is_dark_theme else "#92400e"
            )
            self._lod_artist.get_bbox_patch().set_edgecolor(
                "#f59e0b" if self._is_dark_theme else "#b45309"
            )
            current_feedback = self.feedback_text
            self.set_preview_stage(self._preview_stage)
            self.set_feedback(self._preview_state, current_feedback)
            self._request_draw()

        def _remove_artist(self, group_id: str) -> None:
            stored = self._artists.pop(group_id, None)
            if stored is None:
                return
            artists = stored if isinstance(stored, tuple) else (stored,)
            for artist in artists:
                try:
                    artist.remove()
                except (ValueError, AttributeError):
                    pass

        @staticmethod
        def _set_artist_visible(stored: Any, visible: bool) -> None:
            artists = stored if isinstance(stored, tuple) else (stored,)
            for artist in artists:
                artist.set_visible(bool(visible))

        def _orientation_quiver(
            self,
            origins: np.ndarray,
            directions: np.ndarray,
            *,
            length_m: float,
            color: str,
            label: str,
        ) -> Any | None:
            """Draw one collection for a complete orientation-vector type."""

            if len(origins) == 0 or not self._orientation_vectors_visible:
                return None
            return self.axes.quiver(
                origins[:, 0],
                origins[:, 1],
                origins[:, 2],
                directions[:, 0],
                directions[:, 1],
                directions[:, 2],
                length=float(length_m) * self._orientation_scale,
                normalize=False,
                color=color,
                linewidth=1.15,
                arrow_length_ratio=0.28,
                pivot="tail",
                label=label,
            )

        def _add_artist(self, group: AssemblySceneGroup) -> None:
            self._remove_artist(group.group_id)
            style = group.style
            artists: list[Any] = []
            if group.kind == "surface":
                mode = normalize_body_render_mode(
                    style.get("render_mode", "Solid")
                )
                color = style.get("color", "#78909c")
                if mode == "Wireframe":
                    facecolor = (0.0, 0.0, 0.0, 0.0)
                    edgecolor = color
                    linewidth = 0.45
                elif mode == "Solid + edges":
                    facecolor = color
                    edgecolor = style.get("edgecolor", "#bfdbfe")
                    if edgecolor == "none":
                        edgecolor = "#bfdbfe"
                    linewidth = 0.22
                else:
                    facecolor = color
                    edgecolor = "none"
                    linewidth = 0.0
                alpha = float(style.get("alpha", 0.75))
                if mode == "Wireframe":
                    artist = Poly3DCollection(
                        group.geometry,
                        # Matplotlib 3.10 cannot depth-sort a 3-D polygon
                        # collection with a truly empty facecolor array. A
                        # transparent face is visually identical and keeps
                        # the collection drawable across supported versions.
                        facecolors=to_rgba(color, 0.0),
                        edgecolors=to_rgba(edgecolor, alpha),
                        linewidths=linewidth,
                        label=group.label,
                    )
                else:
                    artist = Poly3DCollection(
                        group.geometry,
                        facecolors=facecolor,
                        edgecolors=edgecolor,
                        linewidths=linewidth,
                        alpha=alpha,
                        label=group.label,
                    )
                self.axes.add_collection3d(artist)
                artists.append(artist)
            elif group.kind == "points":
                points = group.geometry
                artist = self.axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=style.get("color", "#38bdf8"),
                    s=style.get("size", 28.0),
                    depthshade=style.get("depthshade", True),
                    label=group.label,
                )
                artists.append(artist)
                length = float(
                    style.get(
                        "orientation_length_m",
                        DEFAULT_ORIENTATION_VECTOR_LENGTH_M,
                    )
                )
                normal_artist = self._orientation_quiver(
                    style.get("normal_origins", np.empty((0, 3))),
                    style.get("normal_directions", np.empty((0, 3))),
                    length_m=length,
                    color=style.get("normal_color", NORMAL_VECTOR_COLOR),
                    label=f"{group.label} normals",
                )
                if normal_artist is not None:
                    artists.append(normal_artist)
                roll_artist = self._orientation_quiver(
                    style.get("roll_origins", np.empty((0, 3))),
                    style.get("roll_directions", np.empty((0, 3))),
                    length_m=length,
                    color=style.get("roll_color", ROLL_VECTOR_COLOR),
                    label=f"{group.label} projected roll +x",
                )
                if roll_artist is not None:
                    artists.append(roll_artist)
            elif group.kind == "lines":
                artist = Line3DCollection(
                    group.geometry,
                    colors=style.get("color", "#f59e0b"),
                    linewidths=style.get("linewidth", 2.0),
                    alpha=style.get("alpha", 1.0),
                    label=group.label,
                )
                # Variable-length line paths form a ragged segment sequence.
                # Matplotlib's automatic limits path coerces that sequence to
                # a rectangular ndarray; GRIM fits from the scene-model bounds
                # instead, so disable that redundant and fragile conversion.
                self.axes.add_collection3d(artist, autolim=False)
                artists.append(artist)
                normal_artist = self._orientation_quiver(
                    style.get("normal_origins", np.empty((0, 3))),
                    style.get("normal_directions", np.empty((0, 3))),
                    length_m=float(
                        style.get(
                            "orientation_length_m",
                            DEFAULT_ORIENTATION_VECTOR_LENGTH_M,
                        )
                    ),
                    color=style.get("normal_color", NORMAL_VECTOR_COLOR),
                    label=f"{group.label} endpoint normals",
                )
                if normal_artist is not None:
                    artists.append(normal_artist)
                tangent_artist = self._orientation_quiver(
                    style.get("tangent_origins", np.empty((0, 3))),
                    style.get("tangent_directions", np.empty((0, 3))),
                    length_m=float(
                        style.get(
                            "orientation_length_m",
                            DEFAULT_ORIENTATION_VECTOR_LENGTH_M,
                        )
                    ),
                    color=style.get(
                        "tangent_color", LINE_TANGENT_VECTOR_COLOR
                    ),
                    label=f"{group.label} path +t",
                )
                if tangent_artist is not None:
                    artists.append(tangent_artist)
                binormal_artist = self._orientation_quiver(
                    style.get("binormal_origins", np.empty((0, 3))),
                    style.get("binormal_directions", np.empty((0, 3))),
                    length_m=float(
                        style.get(
                            "orientation_length_m",
                            DEFAULT_ORIENTATION_VECTOR_LENGTH_M,
                        )
                    ),
                    color=style.get(
                        "binormal_color", LINE_BINORMAL_VECTOR_COLOR
                    ),
                    label=f"{group.label} signed +b",
                )
                if binormal_artist is not None:
                    artists.append(binormal_artist)
            else:  # Defensive: models should never store an unknown kind.
                raise ValueError(f"unsupported assembly scene kind {group.kind!r}")
            stored: Any = artists[0] if len(artists) == 1 else tuple(artists)
            self._set_artist_visible(stored, group.visible)
            self._artists[group.group_id] = stored

        @property
        def orientation_scale(self) -> float:
            return self._orientation_scale

        @property
        def orientation_vectors_visible(self) -> bool:
            return self._orientation_vectors_visible

        def _refresh_orientation_artists(self) -> None:
            self.begin_scene_updates()
            try:
                for group_id in self.model.group_ids:
                    group = self.model.group(group_id)
                    if group.kind in {"points", "lines"}:
                        self._add_artist(group)
                self._scene_draw_pending = True
            finally:
                self.end_scene_updates()

        def set_orientation_scale(self, scale: float) -> None:
            """Set display-only frame arrow scale without changing geometry."""

            value = float(scale)
            if not np.isfinite(value) or not 0.1 <= value <= 4.0:
                raise ValueError("orientation scale must be between 0.1 and 4.0")
            if value == self._orientation_scale:
                return
            self._orientation_scale = value
            self._refresh_orientation_artists()

        def set_orientation_vectors_visible(self, visible: bool) -> None:
            """Show or hide all frame arrows while retaining points and paths."""

            state = bool(visible)
            if state == self._orientation_vectors_visible:
                return
            self._orientation_vectors_visible = state
            self._refresh_orientation_artists()

        @property
        def display_units(self) -> str:
            return self._display_units

        def set_display_units(self, units: str) -> None:
            """Change axis labels/ticks while keeping all limits/data in meters."""

            suffix, _scale = display_unit_spec(units)
            for label in DISPLAY_UNIT_SPECS:
                if label.casefold() == str(units).strip().casefold():
                    self._display_units = label
                    break
            for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
                axis.set_major_formatter(
                    FuncFormatter(
                        lambda value, _position: format_length_tick(
                            value, self._display_units
                        )
                    )
                )
            self.axes.set_xlabel(f"X right ({suffix})")
            self.axes.set_ylabel(f"Y nose ({suffix})")
            self.axes.set_zlabel(f"Z up ({suffix})")
            self.setToolTip(
                f"Drag to rotate and use the mouse wheel to zoom. Axis ticks are "
                f"shown in {self._display_units.lower()}; underlying CAD and "
                "physics geometry remains in meters."
            )
            self._request_draw()

        def set_interaction_lod_enabled(self, enabled: bool) -> None:
            self._interaction_lod_enabled = bool(enabled)
            if not self._interaction_lod_enabled:
                self._end_interaction_lod(None)

        def _begin_interaction_lod(self, event: Any) -> None:
            """Temporarily switch large surfaces to the cached Fast proxy."""

            if (
                not self._interaction_lod_enabled
                or getattr(event, "inaxes", None) is not self.axes
                or self._interaction_detail_caps
            ):
                return
            fast_cap = int(TRIANGLE_DETAIL_CAPS["Fast"])
            for group_id in self.model.group_ids:
                group = self.model.group(group_id)
                if (
                    group.kind != "surface"
                    or not group.visible
                    or group.display_count <= fast_cap
                ):
                    continue
                self._interaction_detail_caps[group_id] = group.detail_cap
                self.model.set_surface_detail(
                    group_id, fast_cap, remember=False
                )
            self._lod_artist.set_visible(bool(self._interaction_detail_caps))
            self._request_draw()

        def _end_interaction_lod(self, _event: Any) -> None:
            """Restore the selected detail after a rotate/zoom drag."""

            saved = self._interaction_detail_caps
            self._interaction_detail_caps = {}
            for group_id, cap in saved.items():
                if group_id in self.model.group_ids:
                    self.model.set_surface_detail(
                        group_id, cap, remember=False
                    )
            self._lod_artist.set_visible(False)
            self._request_draw()

        @property
        def preview_state(self) -> str:
            """Current user-facing preview state."""

            return self._preview_state

        @property
        def feedback_text(self) -> str:
            return str(self._feedback_artist.get_text())

        @property
        def preview_stage(self) -> str:
            """Preparation stage shown independently from scene visibility."""

            return self._preview_stage

        def set_preview_stage(self, stage: str) -> None:
            """Label input-only, validated, or stale geometry on the canvas."""

            normalized = str(stage).strip().lower()
            labels = {
                "none": (
                    "",
                    self._theme["foreground"],
                    self._theme["accent"],
                ),
                "input": (
                    "INPUT PREVIEW \u2022 NOT PHYSICS-VALIDATED",
                    "#fde68a" if self._is_dark_theme else "#92400e",
                    "#f59e0b" if self._is_dark_theme else "#b45309",
                ),
                "validated": (
                    "PLACEMENTS VALIDATED",
                    self._theme["foreground"],
                    self._theme["accent"],
                ),
                "stale": (
                    "STALE PREVIEW \u2022 INPUTS CHANGED",
                    "#fed7aa" if self._is_dark_theme else "#9a3412",
                    "#f97316" if self._is_dark_theme else "#c2410c",
                ),
            }
            if normalized not in labels:
                raise ValueError(
                    "preview stage must be none, input, validated, or stale"
                )
            text, foreground, border = labels[normalized]
            self._preview_stage = normalized
            self._stage_artist.set_text(text)
            self._stage_artist.set_color(foreground)
            self._stage_artist.get_bbox_patch().set_edgecolor(border)
            self._stage_artist.set_visible(bool(text))
            self._request_draw()

        def set_feedback(self, state: str, message: str | None = None) -> None:
            """Show explicit canvas feedback without changing scene geometry.

            This is presentation state only. It is deliberately stored on the
            canvas rather than :class:`AssemblySceneModel`, so loading/error
            messages can never affect feature membership or assembly physics.
            """

            # An explicit loading/error/empty message supersedes any generic
            # model-change feedback queued earlier in the same scene transaction.
            self._scene_feedback_pending = False
            normalized = str(state).strip().lower()
            if normalized not in self._FEEDBACK_DEFAULTS:
                raise ValueError(
                    "preview feedback state must be empty, loading, error, hidden, "
                    "or ready"
                )
            text = self._FEEDBACK_DEFAULTS[normalized] if message is None else str(message)
            self._preview_state = normalized
            self._feedback_artist.set_text(text)
            self._feedback_artist.set_visible(normalized != "ready")
            if normalized == "error":
                self._feedback_artist.set_color(
                    "#fecaca" if self._is_dark_theme else "#991b1b"
                )
                self._feedback_artist.get_bbox_patch().set_edgecolor(
                    "#f87171" if self._is_dark_theme else "#dc2626"
                )
            else:
                self._feedback_artist.set_color(self._theme["foreground"])
                self._feedback_artist.get_bbox_patch().set_edgecolor(
                    self._theme["accent"]
                )
            self._request_draw()

        def refresh_scene_feedback(self) -> None:
            """Reflect whether the current display has visible geometry."""

            # An explicit refresh satisfies any queued model-change request.
            self._scene_feedback_pending = False
            if not self.model.group_ids:
                self.set_feedback("empty")
            elif any(
                self.model.group(group_id).visible
                for group_id in self.model.group_ids
            ):
                self.set_feedback("ready")
            else:
                self.set_feedback("hidden")

        def _on_model_change(self, event: str, group_id: str | None) -> None:
            if event == "cleared":
                for key in tuple(self._artists):
                    self._remove_artist(key)
                self._interaction_detail_caps.clear()
                self._lod_artist.set_visible(False)
            elif event == "removed" and group_id is not None:
                self._remove_artist(group_id)
                self._interaction_detail_caps.pop(group_id, None)
                if not self._interaction_detail_caps:
                    self._lod_artist.set_visible(False)
            elif event == "visibility" and group_id is not None:
                artist = self._artists.get(group_id)
                if artist is not None:
                    self._set_artist_visible(
                        artist, self.model.group(group_id).visible
                    )
            elif group_id is not None:
                self._add_artist(self.model.group(group_id))
            # Direct scene API users receive the same positive feedback as
            # feature-plan users. Workspace loading subsequently adds a more
            # detailed count summary beneath the canvas.
            self._request_scene_feedback()

        def _detach_model_listener(self, *_args) -> None:
            """Release the model's strong callback when Qt destroys the canvas."""

            if not self._model_listener_attached:
                return
            self.model.remove_listener(self._on_model_change)
            self._model_listener_attached = False

        def clear(self) -> None:
            self.model.clear()
            if not self.model.group_ids:
                self.set_feedback("empty")

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

            bounds = self.model.bounds(visible_only=True)
            if bounds is None:
                bounds = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
            self.fit_bounds(bounds, padding_fraction=padding_fraction)

        def fit_bounds(
            self,
            bounds_m: Any,
            *,
            padding_fraction: float = 0.06,
        ) -> None:
            """Fit explicit CAD-meter bounds without altering scene geometry."""

            padding = float(padding_fraction)
            if not np.isfinite(padding) or padding < 0.0:
                raise ValueError("padding_fraction must be finite and nonnegative")
            bounds = np.asarray(bounds_m, dtype=float)
            if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
                raise ValueError("bounds_m must be a finite 2-by-3 array")
            if np.any(bounds[1] < bounds[0]):
                raise ValueError("bounds_m upper values must not be below lower values")
            center = 0.5 * (bounds[0] + bounds[1])
            span = bounds[1] - bounds[0]
            reference = max(float(np.max(span)), float(np.max(np.abs(center))), 1.0)
            minimum_span = 1.0e-6 * reference
            # Keep flat STL facets and isolated point placements visibly
            # rotatable without distorting the data scale. The plot-box aspect
            # follows these same limits, so one meter remains one meter on all
            # three axes.
            largest_span = max(float(np.max(span)), minimum_span)
            minimum_span = max(minimum_span, 0.03 * largest_span)
            span = np.maximum(span, minimum_span)
            span = span * (1.0 + 2.0 * padding)
            lower = center - 0.5 * span
            upper = center + 0.5 * span
            self.axes.set_xlim(float(lower[0]), float(upper[0]))
            self.axes.set_ylim(float(lower[1]), float(upper[1]))
            self.axes.set_zlim(float(lower[2]), float(upper[2]))
            self.axes.set_box_aspect(tuple(float(value) for value in span))
            self._request_draw()


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
            self._feature_instance_geometry: dict[tuple[str, str], np.ndarray] = {}
            self._triangle_detail_name = "Balanced"
            self._body_render_mode = "Solid"
            self._body_opacity = 0.75

            outer = QVBoxLayout(self)
            outer.setContentsMargins(10, 10, 10, 10)
            outer.setSpacing(6)

            toolbar = QHBoxLayout()
            self.btn_fit_visible = QPushButton("Fit")
            self.btn_fit_visible.setToolTip(
                "Refit the 3-D camera to the body, points, and lines whose Show "
                "boxes are checked. This changes the view only."
            )
            self.btn_preview_layers = QPushButton("Layers")
            self.btn_preview_layers.setToolTip(
                "Open advanced whole-dataset combination and preview visibility "
                "without interrupting the feature workflow."
            )
            self.lbl_legend = QLabel(self._legend_html("#94a3b8"))
            self.lbl_status = QLabel(
                "Preview is empty. Choose an optional STL/facet or BoR body, then "
                "add point or line placements. The tree Show boxes control only the "
                "3-D display, including orientation arrows; they do not change "
                "the assembled RCS."
            )
            self.lbl_status.setWordWrap(True)
            toolbar.addStretch(1)
            self.btn_legend = QPushButton("Legend")
            self.btn_legend.setCheckable(True)
            self.btn_legend.setToolTip(
                "Show or hide the semantic color key for display-only geometry."
            )
            toolbar.addWidget(self.btn_legend)
            self.btn_display_options = QPushButton("View")
            self.btn_display_options.setCheckable(True)
            self.btn_display_options.setToolTip(
                "Adjust display units, body appearance, detail, and orientation arrows."
            )
            toolbar.addWidget(self.btn_display_options)
            toolbar.addWidget(self.btn_preview_layers)
            toolbar.addWidget(self.btn_fit_visible)
            outer.addLayout(toolbar)

            self.legend_bar = QWidget(self)
            legend_layout = QHBoxLayout(self.legend_bar)
            legend_layout.setContentsMargins(8, 4, 8, 4)
            legend_layout.addStretch(1)
            legend_layout.addWidget(self.lbl_legend)
            legend_layout.addStretch(1)
            self.legend_bar.setVisible(False)
            self.btn_legend.toggled.connect(self.legend_bar.setVisible)
            outer.addWidget(self.legend_bar)

            self.display_options = QWidget(self)
            display_layout = QVBoxLayout(self.display_options)
            display_layout.setContentsMargins(0, 0, 0, 0)
            self.display_options.setVisible(False)
            self.btn_display_options.toggled.connect(self.display_options.setVisible)
            outer.addWidget(self.display_options)
            display_heading = QLabel(
                "3-D display only \u2014 does not affect RCS"
            )
            display_heading.setStyleSheet("font-weight: 600;")
            display_layout.addWidget(display_heading)

            display_row = QHBoxLayout()
            display_row.setSpacing(6)
            display_row.addWidget(QLabel("Camera"))
            self.cmb_camera_preset = QComboBox(self)
            self.cmb_camera_preset.addItems(tuple(CAMERA_PRESETS))
            self.cmb_camera_preset.setToolTip(
                "Choose a reproducible CAD-axis view. Parentheses name the plane "
                "visible in the viewport."
            )
            display_row.addWidget(self.cmb_camera_preset)
            self.chk_orthographic = QCheckBox("Orthographic QA", self)
            self.chk_orthographic.setChecked(True)
            self.chk_orthographic.setToolTip(
                "Remove perspective foreshortening for placement inspection. "
                "This changes the camera only."
            )
            display_row.addWidget(self.chk_orthographic)
            display_row.addWidget(QLabel("Display units"))
            self.cmb_display_units = QComboBox(self)
            for unit_name, (suffix, _scale) in DISPLAY_UNIT_SPECS.items():
                self.cmb_display_units.addItem(
                    f"{unit_name} ({suffix})", unit_name
                )
            self.cmb_display_units.setToolTip(
                "Changes 3-D axis labels and tick values only. CAD and physics "
                "coordinates always remain meters."
            )
            display_row.addWidget(self.cmb_display_units)

            display_row.addWidget(QLabel("Body view"))
            self.cmb_body_render = QComboBox(self)
            self.cmb_body_render.addItems(BODY_RENDER_MODES)
            self.cmb_body_render.setToolTip(
                "Choose a solid, edged, or wireframe display. Rendering never "
                "changes the body used for validation or shadowing."
            )
            display_row.addWidget(self.cmb_body_render)

            display_row.addWidget(QLabel("Body opacity"))
            self.sld_body_opacity = QSlider(Qt.Horizontal, self)
            self.sld_body_opacity.setRange(5, 100)
            self.sld_body_opacity.setValue(round(100.0 * self._body_opacity))
            self.sld_body_opacity.setFixedWidth(110)
            self.sld_body_opacity.setToolTip(
                "Visualization-only body opacity (minimum 5%). To hide the body, "
                "uncheck its Show box in the Assembly tree."
            )
            self.lbl_body_opacity = QLabel("75%")
            display_row.addWidget(self.sld_body_opacity)
            display_row.addWidget(self.lbl_body_opacity)
            display_row.addStretch(1)
            display_layout.addLayout(display_row)

            detail_row = QHBoxLayout()
            detail_row.setSpacing(6)
            detail_row.addWidget(QLabel("Preview facet detail"))
            self.cmb_triangle_detail = QComboBox(self)
            for detail_name, cap in TRIANGLE_DETAIL_CAPS.items():
                self.cmb_triangle_detail.addItem(
                    f"{detail_name} ({cap:,} max)", detail_name
                )
            self.cmb_triangle_detail.setCurrentIndex(
                tuple(TRIANGLE_DETAIL_CAPS).index(self._triangle_detail_name)
            )
            self.cmb_triangle_detail.setToolTip(
                "Display-only facet caps: Fast 4k, Balanced 12k, or High 30k. "
                "The unbounded full mesh is intentionally not rendered because "
                "large STL files can freeze Matplotlib. Backend validation and "
                "shadowing are never rebuilt from this view."
            )
            detail_row.addWidget(self.cmb_triangle_detail)

            self.chk_interaction_lod = QCheckBox(
                "Faster rotation (display only)", self
            )
            self.chk_interaction_lod.setChecked(True)
            self.chk_interaction_lod.setToolTip(
                "Temporarily uses at most 4,000 cached body facets while dragging, "
                "then restores the selected detail on release."
            )
            detail_row.addWidget(self.chk_interaction_lod)

            self.chk_orientation_frames = QCheckBox(
                "Show orientation frames", self
            )
            self.chk_orientation_frames.setChecked(True)
            self.chk_orientation_frames.setToolTip(
                "Show display-only point +z/+x and line +n/+t/+b arrows. "
                "For lines, +t follows increasing segment_index and +b = +t × +n."
            )
            detail_row.addWidget(self.chk_orientation_frames)
            detail_row.addWidget(QLabel("Arrow size"))
            self.sld_orientation_scale = QSlider(Qt.Horizontal, self)
            self.sld_orientation_scale.setRange(25, 250)
            self.sld_orientation_scale.setValue(100)
            self.sld_orientation_scale.setTracking(False)
            self.sld_orientation_scale.setFixedWidth(90)
            self.sld_orientation_scale.setToolTip(
                "Scale orientation arrows from 25% to 250%. This never changes "
                "feature coordinates, frames, validation, or RCS."
            )
            self.lbl_orientation_scale = QLabel("100%")
            detail_row.addWidget(self.sld_orientation_scale)
            detail_row.addWidget(self.lbl_orientation_scale)
            detail_row.addStretch(1)
            display_layout.addLayout(detail_row)

            self.lbl_body_detail = QLabel(
                "Display units: meters. No body preview loaded. Original geometry "
                "is unchanged for validation, shadowing, and assembly."
            )
            self.lbl_body_detail.setWordWrap(True)
            display_layout.addWidget(self.lbl_body_detail)

            splitter = QSplitter(Qt.Horizontal)
            left_host = QWidget(self)
            left_layout = QVBoxLayout(left_host)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(6)

            self.feature_controls_host = QWidget(left_host)
            self.feature_controls_layout = QVBoxLayout(self.feature_controls_host)
            self.feature_controls_layout.setContentsMargins(0, 0, 0, 0)
            self.feature_controls_host.setVisible(False)
            left_layout.addWidget(self.feature_controls_host, 1)

            # Feature Assembly owns Body / Point Features / Line Features / Review
            # tabs. Keep advanced whole-response arithmetic available from the
            # toolbar without competing with that primary workflow.
            self.preview_layers_dialog = QDialog(self)
            self.preview_layers_dialog.setWindowTitle(
                "Datasets and Preview Layers"
            )
            self.preview_layers_dialog.setModal(False)
            self.preview_layers_dialog.resize(720, 760)
            combine_layout = QVBoxLayout(self.preview_layers_dialog)
            combine_layout.setContentsMargins(8, 8, 8, 8)
            combine_layout.setSpacing(6)
            combine_help = QLabel(
                "Advanced: combine complete GRIM responses or change which layers "
                "are visible in the 3-D preview. This does not replace the Body, "
                "Point Features, Line Features, and Review workflow.",
                self.preview_layers_dialog,
            )
            combine_help.setWordWrap(True)
            combine_layout.addWidget(combine_help)
            combine_layout.addWidget(self.assembly_tree_panel, 1)
            self.combine_visibility_tab = self.preview_layers_dialog
            self.left_tabs = None
            self.place_features_tab = None
            splitter.addWidget(left_host)

            viewer = QWidget(self)
            viewer_layout = QVBoxLayout(viewer)
            viewer_layout.setContentsMargins(0, 0, 0, 0)
            viewer_layout.addWidget(self.scene_canvas, 1)
            viewer_layout.addWidget(self.lbl_status)
            splitter.addWidget(viewer)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            left_host.setMinimumWidth(400)
            splitter.setSizes([500, 1000])
            outer.addWidget(splitter, 1)
            self.splitter = splitter
            self.left_host = left_host

            self._opacity_timer = QTimer(self)
            self._opacity_timer.setSingleShot(True)
            self._opacity_timer.setInterval(120)
            self._opacity_timer.timeout.connect(self._apply_body_rendering)
            self.btn_fit_visible.clicked.connect(self.scene_canvas.fit_visible)
            self.btn_preview_layers.clicked.connect(self._show_preview_layers)
            self.cmb_camera_preset.currentTextChanged.connect(
                self.scene_canvas.set_camera_preset
            )
            self.chk_orthographic.toggled.connect(
                self.scene_canvas.set_projection_mode
            )
            self.cmb_display_units.currentIndexChanged.connect(
                self._apply_display_units
            )
            self.cmb_body_render.currentTextChanged.connect(
                self._apply_body_rendering
            )
            self.sld_body_opacity.valueChanged.connect(
                self._queue_body_opacity
            )
            self.sld_body_opacity.sliderReleased.connect(
                self._apply_body_rendering
            )
            self.cmb_triangle_detail.currentIndexChanged.connect(
                self._apply_triangle_detail
            )
            self.chk_interaction_lod.toggled.connect(
                self.scene_canvas.set_interaction_lod_enabled
            )
            self.chk_orientation_frames.toggled.connect(
                self.scene_canvas.set_orientation_vectors_visible
            )
            self.sld_orientation_scale.valueChanged.connect(
                self._apply_orientation_scale
            )
            self._connect_tree_panel_signals()
            self._update_body_detail_label()

        @property
        def group_ids(self) -> tuple[str, ...]:
            return self.scene_model.group_ids

        def apply_application_palette(
            self,
            palette: Mapping[str, object],
        ) -> None:
            """Apply host presentation colors to the 3-D preview canvas."""

            self.scene_canvas.apply_theme(palette)
            self.lbl_legend.setText(
                self._legend_html(str(palette.get("muted", "#94a3b8")))
            )

        @staticmethod
        def _legend_html(body_color: str) -> str:
            """Build the semantic preview legend with readable body text."""

            return (
                f'<span style="color:{body_color}">\u25a0 Body</span>&nbsp;&nbsp;'
                '<span style="color:#38bdf8">\u25cf Points</span>&nbsp;&nbsp;'
                '<span style="color:#f59e0b">\u2501 Lines</span>&nbsp;&nbsp;'
                '<span style="color:#f472b6">\u2197 Normals</span>&nbsp;&nbsp;'
                '<span style="color:#c084fc">\u2197 Point +x</span>&nbsp;&nbsp;'
                '<span style="color:#67e8f9">\u2197 Line +t</span>&nbsp;&nbsp;'
                '<span style="color:#818cf8">\u2197 Line +b</span>'
            )

        def _surface_group_ids(self) -> tuple[str, ...]:
            return tuple(
                group_id
                for group_id in self.scene_model.group_ids
                if self.scene_model.group(group_id).kind == "surface"
            )

        def _apply_display_units(self, index: int) -> None:
            units = self.cmb_display_units.itemData(int(index))
            self.scene_canvas.set_display_units(str(units))
            self._update_body_detail_label()

        def _apply_orientation_scale(self, value: int) -> None:
            percent = int(value)
            self.lbl_orientation_scale.setText(f"{percent}%")
            self.scene_canvas.set_orientation_scale(percent / 100.0)

        def _queue_body_opacity(self, value: int) -> None:
            self.lbl_body_opacity.setText(f"{int(value)}%")
            self._opacity_timer.start()

        def _apply_body_rendering(self, *_args: Any) -> None:
            self._opacity_timer.stop()
            self._body_render_mode = normalize_body_render_mode(
                self.cmb_body_render.currentText()
            )
            self._body_opacity = self.sld_body_opacity.value() / 100.0
            self.lbl_body_opacity.setText(
                f"{self.sld_body_opacity.value()}%"
            )
            for group_id in self._surface_group_ids():
                self.scene_model.set_surface_rendering(
                    group_id,
                    self._body_render_mode,
                    self._body_opacity,
                )
            self._update_body_detail_label()

        def _apply_triangle_detail(self, index: int) -> None:
            name = str(self.cmb_triangle_detail.itemData(int(index)))
            cap = triangle_detail_cap(name)
            for label in TRIANGLE_DETAIL_CAPS:
                if label.casefold() == str(name).strip().casefold():
                    self._triangle_detail_name = label
                    break
            for group_id in self._surface_group_ids():
                self.scene_model.set_surface_detail(group_id, cap)
            self._update_body_detail_label()

        def _update_body_detail_label(self) -> None:
            surfaces = [
                self.scene_model.group(group_id)
                for group_id in self._surface_group_ids()
            ]
            has_body = bool(surfaces)
            has_features = any(
                self.scene_model.group(group_id).kind in {"points", "lines"}
                for group_id in self.scene_model.group_ids
            )
            for control in (
                self.cmb_body_render,
                self.sld_body_opacity,
                self.cmb_triangle_detail,
                self.chk_interaction_lod,
            ):
                control.setEnabled(has_body)
            self.chk_orientation_frames.setEnabled(has_features)
            self.sld_orientation_scale.setEnabled(has_features)
            self.lbl_orientation_scale.setEnabled(has_features)
            unit_text = self.scene_canvas.display_units.lower()
            if not surfaces:
                self.lbl_body_detail.setText(
                    f"Display units: {unit_text}. No body preview loaded. Original "
                    "geometry is unchanged for validation, shadowing, and assembly."
                )
                return
            displayed = sum(group.display_count for group in surfaces)
            source = sum(group.source_count for group in surfaces)
            self.lbl_body_detail.setText(
                f"Display: {unit_text}; {self._body_render_mode} at "
                f"{round(100.0 * self._body_opacity)}% opacity; {displayed:,} of "
                f"{source:,} body triangles shown ({self._triangle_detail_name}). "
                "Original geometry is unchanged for validation, shadowing, and "
                "assembly."
            )

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

            self.scene_canvas.begin_scene_updates()
            try:
                for identifier in identifiers:
                    if identifier in self.scene_model.group_ids:
                        self.scene_model.remove_group(identifier)
                    self._pending_visibility.pop(identifier, None)
                    self._feature_preview_group_ids.discard(identifier)
            finally:
                self.scene_canvas.end_scene_updates()
            self._update_body_detail_label()

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
            self.lbl_status.setText(
                "Preview cleared. Choose an optional STL/facet or BoR body, then "
                "add point or line placements."
            )

        def clear_feature_preview(self) -> None:
            """Remove only the service-owned feature scene and typed tree root."""

            self.scene_canvas.begin_scene_updates()
            try:
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
                self._feature_instance_geometry.clear()
                self.scene_canvas.set_preview_stage("none")
            finally:
                self.scene_canvas.end_scene_updates()
            self._update_body_detail_label()

        @staticmethod
        def _feature_plan_geometry(plan: object):
            """Read only the prepared CAD-meter preview contract from a plan."""

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
            base_geometry = tuple(getattr(plan, name) for name in required)
            orientation = (
                getattr(plan, "point_normals_cad", {}),
                getattr(plan, "point_roll_references_cad", {}),
                getattr(plan, "line_endpoint_normals_cad", {}),
            )
            if not all(isinstance(value, dict) for value in orientation):
                raise TypeError(
                    "prepared point/line orientation previews must be dicts"
                )
            point_groups = base_geometry[2]
            line_groups = base_geometry[3]
            if not isinstance(point_groups, dict) or not isinstance(
                line_groups, dict
            ):
                raise TypeError(
                    "prepared point_locations_cad_m and line_paths_cad_m must "
                    "be dicts"
                )
            point_normals, point_rolls, line_normals = orientation
            point_placement_ids = getattr(plan, "point_placement_ids", {})
            if not isinstance(point_placement_ids, dict):
                raise TypeError("prepared point_placement_ids must be a dict")
            # Empty orientation maps mean a legacy four-field preview object;
            # it remains displayable with markers/paths only. Once any new
            # orientation data is supplied, require an exact keyed contract so
            # vectors can never be silently paired with the wrong placement.
            if any(bool(value) for value in orientation):
                point_keys = set(point_groups)
                if set(point_normals) != point_keys or set(point_rolls) != point_keys:
                    raise ValueError(
                        "point orientation dataset IDs must exactly match point "
                        "location dataset IDs"
                    )
                line_keys = set(line_groups)
                if set(line_normals) != line_keys:
                    raise ValueError(
                        "line-normal dataset IDs must exactly match line-path "
                        "dataset IDs"
                    )
                for dataset_id in line_keys:
                    paths_by_id = line_groups[dataset_id]
                    normals_by_id = line_normals[dataset_id]
                    if not isinstance(paths_by_id, dict) or not isinstance(
                        normals_by_id, dict
                    ):
                        raise TypeError(
                            "line paths and endpoint normals must map line_id "
                            "to prepared arrays"
                        )
                    if set(normals_by_id) != set(paths_by_id):
                        raise ValueError(
                            f"line-normal IDs for dataset {dataset_id!r} must "
                            "exactly match its line-path IDs"
                        )
            if point_placement_ids:
                if set(point_placement_ids) != set(point_groups):
                    raise ValueError(
                        "point placement-ID dataset keys must exactly match point "
                        "location dataset IDs"
                    )
                all_point_ids: list[str] = []
                for dataset_id, locations in point_groups.items():
                    identifiers = tuple(
                        str(value) for value in point_placement_ids[dataset_id]
                    )
                    if len(identifiers) != len(np.asarray(locations)):
                        raise ValueError(
                            f"point placement-ID count for dataset {dataset_id!r} "
                            "does not match its location count"
                        )
                    if any(not value for value in identifiers):
                        raise ValueError("point placement IDs must be nonempty strings")
                    all_point_ids.extend(identifiers)
                if len(all_point_ids) != len(set(all_point_ids)):
                    raise ValueError("point placement IDs must be globally unique")
            return (*base_geometry, *orientation, point_placement_ids)

        def _show_preview_error(self, exc: Exception) -> None:
            """Publish one consistent, non-destructive preview failure state."""

            message = f"Preview unavailable: {exc}"
            self.scene_canvas.set_feedback("error", message)
            self.scene_canvas.set_preview_stage("none")
            self.lbl_status.setText(
                f"{message}. Correct the reported body or placement input; "
                "no assembly result was changed."
            )

        def load_feature_preview(self, plan: object):
            """Render one already-prepared feature plan in CAD meters.

            This method never reads a CSV, mesh file, or response dataset. The
            feature workflow owns parsing, units, skin checks, normals, and all
            electromagnetic operations; GRIM consumes only its validated
            ``FeaturePreviewGeometry`` arrays.

            Point Features and Line Features containers are retained at zero
            count so the hierarchy is predictable while a user configures one
            feature type at a time.
            """

            try:
                (
                    surface,
                    profile,
                    point_groups,
                    line_groups,
                    point_normals,
                    point_rolls,
                    line_normals,
                    point_placement_ids,
                ) = (
                    self._feature_plan_geometry(plan)
                )
                preview_stage = str(
                    getattr(plan, "preview_stage", "validated")
                ).lower()
                if preview_stage not in {"input", "validated"}:
                    raise ValueError(
                        "prepared preview_stage must be 'input' or 'validated'"
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
                    raise RuntimeError(
                        "AssemblyTree does not provide typed preview APIs"
                    )
            except Exception as exc:
                self.clear_feature_preview()
                self._show_preview_error(exc)
                raise

            scene_ids: list[str] = []
            self.scene_canvas.begin_scene_updates()
            try:
                self.clear_feature_preview()
                self.scene_canvas.set_feedback("loading")
                self.lbl_status.setText(
                    "Preparing the body and feature placements for the 3-D preview\u2026"
                )
                root = tree.add_preview_root(
                    "Feature Assembly", stable_key=FEATURE_PREVIEW_ROOT_KEY
                )
                body_description = "no body geometry"
                body_id = feature_preview_group_id("body")
                body_group: AssemblySceneGroup | None = None
                if surface is not None:
                    body_group = self.add_body_triangles(body_id, surface)
                    scene_ids.append(body_id)
                    body_label = (
                        f"Body (STL/facet, {body_group.source_count:,} triangles)"
                    )
                    body_description = (
                        f"STL/facet body ({body_group.source_count:,} source triangles)"
                    )
                    body_item = tree.add_preview_item(
                        root, body_label, stable_key=body_id
                    )
                    self.bind_tree_item_groups(body_item, body_id)
                elif profile is not None:
                    body_group = self.add_bor_profile(body_id, profile)
                    scene_ids.append(body_id)
                    profile_count = int(np.asarray(profile).shape[0])
                    body_label = f"Body (BoR, {profile_count:,} profile points)"
                    body_description = f"BoR body ({profile_count:,} profile points)"
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

                vector_length_m = orientation_vector_length_m(
                    _feature_preview_nonvector_bounds(
                        None if body_group is None else body_group.bounds_m,
                        point_groups,
                        line_groups,
                    )
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
                    self.add_points(
                        group_id,
                        point_groups[dataset_id],
                        normals=(
                            point_normals[dataset_id]
                            if point_normals
                            else None
                        ),
                        roll_references=(
                            point_rolls[dataset_id] if point_rolls else None
                        ),
                        orientation_length_m=vector_length_m,
                    )
                    if point_placement_ids:
                        locations = np.asarray(
                            point_groups[dataset_id], dtype=float
                        ).reshape(-1, 3)
                        for placement_id, location in zip(
                            point_placement_ids[dataset_id], locations
                        ):
                            self._feature_instance_geometry[
                                ("point", str(placement_id))
                            ] = np.array(location, dtype=float, copy=True)
                    scene_ids.append(group_id)
                    point_word = "point" if count == 1 else "points"
                    item = tree.add_preview_item(
                        point_root,
                        f"{dataset_id} ({count:,} {point_word})",
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
                    ordered_line_ids = tuple(sorted(paths_by_id))
                    paths = tuple(paths_by_id[key] for key in ordered_line_ids)
                    endpoint_normals = (
                        tuple(
                            line_normals[dataset_id][key]
                            for key in ordered_line_ids
                        )
                        if line_normals
                        else None
                    )
                    count = line_counts[dataset_id]
                    group_id = feature_preview_group_id("lines", dataset_id)
                    self.add_lines(
                        group_id,
                        paths,
                        endpoint_normals=endpoint_normals,
                        orientation_length_m=vector_length_m,
                    )
                    for line_id in ordered_line_ids:
                        self._feature_instance_geometry[
                            ("line", str(line_id))
                        ] = np.array(
                            paths_by_id[line_id], dtype=float, copy=True
                        )
                    scene_ids.append(group_id)
                    line_word = "line" if count == 1 else "lines"
                    item = tree.add_preview_item(
                        line_root,
                        f"{dataset_id} ({count:,} {line_word})",
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
                self.scene_canvas.refresh_scene_feedback()
                self.scene_canvas.set_preview_stage(
                    preview_stage if scene_ids else "none"
                )
                if scene_ids:
                    point_summary = (
                        "1 point placement"
                        if point_total == 1
                        else f"{point_total:,} point placements"
                    )
                    line_summary = (
                        "1 line path"
                        if line_total == 1
                        else f"{line_total:,} line paths"
                    )
                    if preview_stage == "input":
                        self.lbl_status.setText(
                            f"Input preview (not physics-validated) \u2014 "
                            f"{body_description}; {point_summary}; {line_summary}. "
                            "Check the locations, then "
                            "validate before building. Axes use the selected display "
                            "units; backend CAD geometry remains meters. Tree Show "
                            "boxes change points/paths and their orientation arrows "
                            "together, only in this display. Zero or parallel input "
                            "vectors are omitted here and reported by Validate."
                        )
                    else:
                        self.lbl_status.setText(
                            f"Validated preview ready \u2014 {body_description}; "
                            f"{point_summary}; {line_summary}. Axes use the selected "
                            "display units; backend CAD geometry remains meters. "
                            "Magenta arrows show normalized outward normals; lavender "
                            "point arrows show projected roll/local +x. Cyan line "
                            "+t follows increasing segment_index; blue +b is the "
                            "signed local across-line direction (+t × +n). "
                            "Drag to rotate and "
                            "scroll to zoom. Tree Show boxes change only this preview, "
                            "never the assembled RCS."
                        )
                else:
                    self.lbl_status.setText(
                        "Nothing is ready to preview yet. Choose an optional STL/facet "
                        "or BoR body and add at least one point or line placement."
                    )
                return root
            except Exception as exc:
                # Never leave a half-populated tree or stale artists after a
                # malformed third-party plan reaches this display boundary.
                self._feature_preview_group_ids.update(scene_ids)
                self.clear_feature_preview()
                self._show_preview_error(exc)
                raise
            finally:
                self.scene_canvas.end_scene_updates()

        def mark_preview_stale(self, reason: str = "") -> None:
            """Keep the last geometry visible while clearly marking it outdated.

            Controllers call this as soon as a body, response, or placement
            selection changes. It never mutates prepared geometry or assembly
            membership; the next :meth:`load_feature_preview` replaces it.
            """

            if not self._feature_preview_group_ids:
                return
            detail = str(reason).strip()
            self.scene_canvas.set_preview_stage("stale")
            self.lbl_status.setText(
                "Preview is stale because the inputs changed. Refresh or validate "
                "the preview before Build. The previous geometry remains visible "
                "for reference only."
                + (f" {detail}" if detail else "")
            )

        def add_body_triangles(self, group_id: str, triangles_m: Any, **kwargs):
            kwargs["visible"] = self._visibility_for_new_group(
                group_id, kwargs.get("visible", True)
            )
            kwargs["max_triangles"] = triangle_detail_cap(
                self._triangle_detail_name
            )
            kwargs["render_mode"] = self._body_render_mode
            kwargs["alpha"] = self._body_opacity
            group = self.scene_canvas.add_body_triangles(
                group_id, triangles_m, **kwargs
            )
            self._update_body_detail_label()
            return group

        def add_bor_profile(self, group_id: str, profile_rho_z_m: Any, **kwargs):
            kwargs["visible"] = self._visibility_for_new_group(
                group_id, kwargs.get("visible", True)
            )
            kwargs["max_triangles"] = triangle_detail_cap(
                self._triangle_detail_name
            )
            kwargs["render_mode"] = self._body_render_mode
            kwargs["alpha"] = self._body_opacity
            group = self.scene_canvas.add_bor_profile(
                group_id, profile_rho_z_m, **kwargs
            )
            self._update_body_detail_label()
            return group

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

        def focus_feature_instance(self, kind: str, instance_id: str) -> bool:
            """Highlight and frame one validated QA instance in the 3-D view."""

            category = str(kind).strip().lower()
            identifier = str(instance_id).strip()
            geometry = self._feature_instance_geometry.get((category, identifier))
            if geometry is None or category not in {"point", "line"}:
                return False
            self.scene_canvas.begin_scene_updates()
            try:
                if FEATURE_SELECTION_GROUP_KEY in self.scene_model.group_ids:
                    self.scene_model.remove_group(FEATURE_SELECTION_GROUP_KEY)
                if category == "point":
                    selected = self.scene_canvas.add_points(
                        FEATURE_SELECTION_GROUP_KEY,
                        np.asarray(geometry, dtype=float).reshape(1, 3),
                        label=f"Selected point {identifier}",
                        color="#fde047",
                        size=150.0,
                        depthshade=False,
                    )
                else:
                    selected = self.scene_canvas.add_lines(
                        FEATURE_SELECTION_GROUP_KEY,
                        np.asarray(geometry, dtype=float),
                        label=f"Selected line {identifier}",
                        color="#fde047",
                        linewidth=5.0,
                    )
                self._feature_preview_group_ids.add(FEATURE_SELECTION_GROUP_KEY)
            finally:
                self.scene_canvas.end_scene_updates()
            focus_bounds = np.array(selected.bounds_m, dtype=float, copy=True)
            scene_bounds = self.scene_model.bounds(visible_only=True)
            if scene_bounds is not None:
                scene_extent = max(
                    float(np.max(scene_bounds[1] - scene_bounds[0])),
                    DEFAULT_ORIENTATION_VECTOR_LENGTH_M,
                )
                minimum_focus_span = 0.08 * scene_extent
                center = 0.5 * (focus_bounds[0] + focus_bounds[1])
                focus_span = np.maximum(
                    focus_bounds[1] - focus_bounds[0], minimum_focus_span
                )
                focus_bounds[0] = center - 0.5 * focus_span
                focus_bounds[1] = center + 0.5 * focus_span
            self.scene_canvas.fit_bounds(focus_bounds, padding_fraction=0.35)
            self.lbl_status.setText(
                f"Focused validated {category} feature {identifier!r}. "
                "The yellow overlay and camera framing are display only."
            )
            return True

        def _show_preview_layers(self, _checked: bool = False) -> None:
            """Show the secondary dataset/layer editor without changing steps."""

            self.preview_layers_dialog.show()
            self.preview_layers_dialog.raise_()
            self.preview_layers_dialog.activateWindow()

        def set_feature_service(
            self,
            service: FeatureBuildService | Callable[[object], Any] | None,
        ) -> None:
            """Install an injected service; ``None`` returns to signal-only mode."""

            self._feature_service = service

        def set_feature_controls(self, widget: QWidget | None) -> None:
            """Install or clear the controller-owned feature workflow."""

            while self.feature_controls_layout.count():
                entry = self.feature_controls_layout.takeAt(0)
                old_widget = entry.widget()
                if old_widget is not None:
                    old_widget.setParent(None)
            if widget is None:
                self.feature_controls_host.setVisible(False)
                self.left_tabs = None
                self.place_features_tab = None
                return
            self.feature_controls_layout.addWidget(widget)
            self.feature_controls_host.setVisible(True)
            self.left_tabs = getattr(widget, "workflow_tabs", None)
            self.place_features_tab = getattr(widget, "body_step_page", None)
            if self.left_tabs is not None and self.place_features_tab is not None:
                self.left_tabs.setCurrentWidget(self.place_features_tab)

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
    "BODY_RENDER_MODES",
    "CAMERA_PRESETS",
    "DISPLAY_UNIT_SPECS",
    "TRIANGLE_DETAIL_CAPS",
    "AssemblySceneCanvas",
    "AssemblySceneGroup",
    "AssemblySceneModel",
    "AssemblyWorkspace",
    "FEATURE_PREVIEW_ROOT_KEY",
    "FEATURE_SELECTION_GROUP_KEY",
    "FeatureBuildResult",
    "FeatureBuildService",
    "GUI_AVAILABLE",
    "decimate_triangles_for_display",
    "display_unit_spec",
    "feature_preview_group_id",
    "format_length_tick",
    "normalize_body_render_mode",
    "orientation_vector_length_m",
    "revolve_bor_profile_cad",
    "triangle_detail_cap",
]
