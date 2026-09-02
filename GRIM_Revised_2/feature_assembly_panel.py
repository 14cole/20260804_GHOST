"""Compact, non-blocking controls for coherent feature assembly.

The physics and placement validation deliberately do not live here.
``FeatureWorkflowAdapter`` accepts the authoritative GHOST
``feature_workflow`` module (or a compatible injected service), while
``FeatureAssemblyFormModel`` keeps request construction testable without Qt or
GHOST on the import path.  The fixed CSV headers are mirrored here only so the
GUI can explain the contract and write blank templates before a backend is
connected; GHOST remains the authoritative parser.

Preview visibility is intentionally absent from the request model.  Hiding a
point or line group in the Assembly 3-D view must never remove that response
from the coherent physical assembly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable
import zipfile


UNIT_CHOICES = (
    ("inches (in)", "inches"),
    ("millimeters (mm)", "millimeters"),
    ("meters (m)", "meters"),
    ("feet (ft)", "feet"),
)
UNIT_SCALE_M = {
    "inches": 0.0254,
    "millimeters": 1.0e-3,
    "meters": 1.0,
    "feet": 0.3048,
}
UNIT_ABBREVIATIONS = {
    "inches": "in",
    "millimeters": "mm",
    "meters": "m",
    "feet": "ft",
}

# Keep the always-visible trade-study status readable for large fastener sets.
# The complete disabled-ID list remains available through the explicit copy
# action next to the summary.
FEATURE_SELECTION_DISPLAY_ID_LIMIT = 8

FEATURE_RECIPE_SCHEMA = "grim.feature-assembly-recipe"
FEATURE_RECIPE_VERSION = 4
FEATURE_RECIPE_SUFFIX = ".assembly.json"
# Hash normal placement/library inputs while keeping recipe saves responsive for
# very large vehicle meshes and clean-body response files. Large inputs still
# retain size and nanosecond modification-time identity in the manifest.
FEATURE_RECIPE_HASH_LIMIT_BYTES = 16 * 1024 * 1024

VALIDATION_PROFILES = (
    ("Production — certified GHOST body (recommended)", "production", False, True, True),
    ("External/HPC body — reviewed", "external", False, True, False),
    ("Legacy compatibility", "legacy", True, False, False),
)
DEFAULT_SKIN_TOL_MM = 1.0
DEFAULT_SKIN_PHASE_TOL_DEG = 15.0
DEFAULT_NORMAL_TOL_DEG = 15.0

# Conservative operator-review thresholds mirrored from GHOST's Qt-free
# ``assembly_workload`` contract. They are operation counts, not elapsed-time
# claims. The bundled backend records the same counts in the sealed plan and
# turns a threshold crossing into an acknowledged validation warning.
ASSEMBLY_REVIEW_RADAR_GRID_CELLS = 10_000_000
ASSEMBLY_REVIEW_POINT_FIELD_CELLS = 250_000_000
ASSEMBLY_REVIEW_LINE_FIELD_CELLS = 500_000_000
ASSEMBLY_REVIEW_SHADOW_RAYS = 5_000_000
ASSEMBLY_REVIEW_LARGE_MESH_TRIANGLES = 1_000_000
ASSEMBLY_REVIEW_LARGE_MESH_SHADOW_RAYS = 100_000
WORKLOAD_REVIEW_WARNING_PREFIX = "Assembly workload review required"
_PREVALIDATION_LINE_PIECES_PER_SEGMENT = 64


@dataclass(frozen=True)
class AssemblyWorkEstimate:
    """Auditable operation counts derived from visible Assembly quantities."""

    available: bool
    quantities_validated: bool = False
    look_count: int = 0
    frequency_count: int = 0
    point_count: int = 0
    line_path_count: int = 0
    line_segment_count: int = 0
    line_piece_count: int = 0
    line_piece_count_exact: bool = False
    mesh_triangle_count: int = 0
    mesh_triangle_count_exact: bool = False
    shadow_enabled: bool = False
    radar_grid_cell_count: int = 0
    point_field_cell_count: int = 0
    line_field_cell_count: int = 0
    shadow_ray_upper_bound: int = 0
    packed_visibility_bytes_upper_bound: int = 0
    review_reasons: tuple[str, ...] = ()


def estimate_assembly_workload(
    *,
    look_count: int,
    frequency_count: int,
    point_count: int,
    line_path_count: int,
    line_segment_count: int,
    line_piece_count: int,
    mesh_triangle_count: int = 0,
    shadow_enabled: bool = False,
    quantities_validated: bool = False,
    line_piece_count_exact: bool = False,
    mesh_triangle_count_exact: bool = False,
) -> AssemblyWorkEstimate:
    """Return exact/upper-bound operation counts without inventing an ETA."""

    values = {
        "look_count": look_count,
        "frequency_count": frequency_count,
        "point_count": point_count,
        "line_path_count": line_path_count,
        "line_segment_count": line_segment_count,
        "line_piece_count": line_piece_count,
        "mesh_triangle_count": mesh_triangle_count,
    }
    normalized: dict[str, int] = {}
    for name, value in values.items():
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 0
        normalized[name] = max(0, number)
    looks = normalized["look_count"]
    frequencies = normalized["frequency_count"]
    if looks <= 0 or frequencies <= 0:
        return AssemblyWorkEstimate(available=False)

    points = normalized["point_count"]
    pieces = normalized["line_piece_count"]
    triangles = normalized["mesh_triangle_count"]
    grid_cells = looks * frequencies
    point_cells = grid_cells * points
    line_cells = grid_cells * pieces
    shadow_rays = looks * (points + pieces) if shadow_enabled else 0
    packed_bytes = (
        (points + pieces) * ((looks + 7) // 8) if shadow_enabled else 0
    )
    reasons: list[str] = []
    if grid_cells >= ASSEMBLY_REVIEW_RADAR_GRID_CELLS:
        reasons.append(
            f"{grid_cells:,} radar look-frequency cells "
            f"(review threshold {ASSEMBLY_REVIEW_RADAR_GRID_CELLS:,})"
        )
    if point_cells >= ASSEMBLY_REVIEW_POINT_FIELD_CELLS:
        reasons.append(
            f"{point_cells:,} point look-frequency evaluations "
            f"(review threshold {ASSEMBLY_REVIEW_POINT_FIELD_CELLS:,})"
        )
    if line_cells >= ASSEMBLY_REVIEW_LINE_FIELD_CELLS:
        reasons.append(
            f"{line_cells:,} line-piece look-frequency evaluations "
            f"(review threshold {ASSEMBLY_REVIEW_LINE_FIELD_CELLS:,})"
        )
    if shadow_rays >= ASSEMBLY_REVIEW_SHADOW_RAYS:
        reasons.append(
            f"up to {shadow_rays:,} body-shadow candidate rays "
            f"(review threshold {ASSEMBLY_REVIEW_SHADOW_RAYS:,})"
        )
    if (
        shadow_enabled
        and triangles >= ASSEMBLY_REVIEW_LARGE_MESH_TRIANGLES
        and shadow_rays >= ASSEMBLY_REVIEW_LARGE_MESH_SHADOW_RAYS
    ):
        reasons.append(
            f"{triangles:,}-triangle body mesh with up to "
            f"{shadow_rays:,} shadow candidates"
        )
    return AssemblyWorkEstimate(
        available=True,
        quantities_validated=bool(quantities_validated),
        look_count=looks,
        frequency_count=frequencies,
        point_count=points,
        line_path_count=normalized["line_path_count"],
        line_segment_count=normalized["line_segment_count"],
        line_piece_count=pieces,
        line_piece_count_exact=bool(line_piece_count_exact),
        mesh_triangle_count=triangles,
        mesh_triangle_count_exact=bool(mesh_triangle_count_exact),
        shadow_enabled=bool(shadow_enabled),
        radar_grid_cell_count=int(grid_cells),
        point_field_cell_count=int(point_cells),
        line_field_cell_count=int(line_cells),
        shadow_ray_upper_bound=int(shadow_rays),
        packed_visibility_bytes_upper_bound=int(packed_bytes),
        review_reasons=tuple(reasons),
    )


def assembly_build_confirmation_required(
    estimate: AssemblyWorkEstimate,
) -> bool:
    """Require review only for an authoritative threshold-crossing plan."""

    return bool(
        isinstance(estimate, AssemblyWorkEstimate)
        and estimate.available
        and estimate.quantities_validated
        and bool(estimate.review_reasons)
    )


def _format_binary_bytes(value: int) -> str:
    count = max(0, int(value))
    if count < 1024:
        return f"{count:,} B"
    if count < 1024 ** 2:
        return f"{count / 1024.0:.2f} KiB"
    if count < 1024 ** 3:
        return f"{count / (1024.0 ** 2):.2f} MiB"
    return f"{count / (1024.0 ** 3):.2f} GiB"


def format_assembly_work_estimate(estimate: AssemblyWorkEstimate) -> str:
    """Render auditable counts and explicitly decline a runtime prediction."""

    if not estimate.available:
        return (
            "Workload preflight (operation counts; no elapsed-time estimate): "
            "choose a valid clean-body "
            "GRIM and refresh the placement CSVs to expose the radar workload."
        )
    stage = "Validated quantities" if estimate.quantities_validated else (
        "Pre-validation quantities"
    )
    piece_prefix = "" if estimate.line_piece_count_exact else "about "
    parts = [
        f"{estimate.look_count:,} looks x {estimate.frequency_count:,} frequencies",
        f"{estimate.point_count:,} point(s)",
        f"{estimate.line_path_count:,} line path(s)",
        f"{piece_prefix}{estimate.line_piece_count:,} solver line piece(s)",
        f"{estimate.point_field_cell_count:,} point look-frequency evaluations",
        f"{estimate.line_field_cell_count:,} line-piece look-frequency evaluations",
    ]
    if estimate.mesh_triangle_count:
        triangle_prefix = "" if estimate.mesh_triangle_count_exact else "up to "
        parts.append(
            f"{triangle_prefix}{estimate.mesh_triangle_count:,} mesh triangle(s)"
        )
    if estimate.shadow_enabled:
        parts.append(
            f"up to {estimate.shadow_ray_upper_bound:,} body-shadow candidate ray(s)"
        )
        parts.append(
            "up to "
            + _format_binary_bytes(estimate.packed_visibility_bytes_upper_bound)
            + " packed visibility"
        )
    else:
        parts.append("body shadowing off")
    refinement = (
        " Work quantities come from the validated plan."
        if estimate.quantities_validated
        else " Line subdivision and mesh cost are refined after Validate."
    )
    if estimate.review_reasons:
        review = (
            " Operator review required: " + "; ".join(estimate.review_reasons) + "."
            if estimate.quantities_validated
            else " Validate to confirm whether the conservative review gate applies."
        )
    else:
        review = " No count-based review threshold is crossed."
    shadow_note = (
        " Shadow candidates are computed once and reused across frequencies; "
        "front-facing culling can reduce actual BVH traces. Ray cost depends "
        "strongly on mesh/ray geometry and hardware."
        if estimate.shadow_enabled
        else ""
    )
    return (
        "Workload preflight (operation counts; no elapsed-time estimate). "
        + stage
        + ": "
        + "; ".join(parts)
        + "."
        + review
        + shadow_note
        + refinement
    )


# Display/template mirrors of GHOST's versioned point- and line-placement v1
# contracts.  These are intentionally strict: the GUI, local scripts, and HPC
# workflow all accept the same files without column inference or conversion.
POINT_PLACEMENT_COLUMNS = (
    "placement_id",
    "dataset_id",
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "roll_x",
    "roll_y",
    "roll_z",
)
LINE_PLACEMENT_COLUMNS = (
    "line_id",
    "dataset_id",
    "segment_index",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "n1x",
    "n1y",
    "n1z",
    "n2x",
    "n2y",
    "n2z",
)
POINT_PLACEMENT_EXAMPLE = (
    "fastener_001,fastener,1.2,8.4,0.5,0,0,1,1,0,0"
)
LINE_PLACEMENT_EXAMPLE = (
    "gap_001,panel_gap,1,-2,6,0,-2,10,0,0,0,1,0,0,1"
)


def placement_csv_template_text(kind: str) -> str:
    """Return the exact blank v1 placement template for ``kind``."""

    normalized = str(kind).strip().lower()
    if normalized == "point":
        columns = POINT_PLACEMENT_COLUMNS
    elif normalized == "line":
        columns = LINE_PLACEMENT_COLUMNS
    else:
        raise ValueError("Placement template kind must be 'point' or 'line'.")
    return ",".join(columns) + "\n"


def write_placement_csv_template(kind: str, path: str | Path) -> Path:
    """Write a blank strict placement CSV and return its final path."""

    raw = _clean_path(path)
    if not raw:
        raise ValueError("Choose where to save the placement CSV template.")
    target = Path(raw)
    if not target.suffix:
        target = target.with_suffix(".csv")
    target.write_text(placement_csv_template_text(kind), encoding="utf-8")
    return target


@runtime_checkable
class _FeatureWorkflowModule(Protocol):
    FeatureAssemblyRequest: Callable[..., Any]

    def discover_feature_dataset_ids(self, **kwargs: Any) -> Any: ...

    def prepare_feature_assembly(self, request: Any) -> Any: ...

    def execute_feature_assembly(self, plan: Any) -> Any: ...


@dataclass(frozen=True)
class FeatureWorkflowAdapter:
    """Neutral adapter around the authoritative feature-workflow API.

    Pass ``FeatureWorkflowAdapter.from_module(feature_workflow)`` to the panel.
    Keeping the core callables explicit avoids importing GHOST from GRIM and
    makes dependency injection straightforward in tests and packaged builds.
    The binding callables are optional for older/limited services; Production
    external-body readiness reports their absence instead of guessing.
    """

    request_factory: Callable[..., Any]
    discover: Callable[..., Any]
    prepare: Callable[[Any], Any]
    execute: Callable[[Any], Any]
    preview_inputs: Callable[..., Any] | None = None
    check_surface_binding: Callable[..., Any] | None = None
    write_surface_binding: Callable[..., Any] | None = None
    surface_binding_path: Callable[[Any], Path] | None = None

    @classmethod
    def from_module(cls, module: _FeatureWorkflowModule) -> "FeatureWorkflowAdapter":
        missing = [
            name
            for name in (
                "FeatureAssemblyRequest",
                "discover_feature_dataset_ids",
                "prepare_feature_assembly",
                "execute_feature_assembly",
            )
            if not callable(getattr(module, name, None))
        ]
        if missing:
            raise TypeError(
                "Feature workflow service is missing callable(s): "
                + ", ".join(missing)
            )
        mirrored_contracts = (
            ("POINT_CSV_COLUMNS", POINT_PLACEMENT_COLUMNS),
            ("LINE_CSV_COLUMNS", LINE_PLACEMENT_COLUMNS),
        )
        for name, expected in mirrored_contracts:
            backend_columns = getattr(module, name, None)
            if backend_columns is not None and tuple(backend_columns) != expected:
                raise RuntimeError(
                    f"GHOST {name} no longer matches the placement format "
                    "shown by GRIM. Update the GUI template before assembly."
                )
        return cls(
            request_factory=module.FeatureAssemblyRequest,
            discover=module.discover_feature_dataset_ids,
            prepare=module.prepare_feature_assembly,
            execute=module.execute_feature_assembly,
            preview_inputs=getattr(module, "prepare_feature_input_preview", None),
            check_surface_binding=getattr(module, "check_surface_binding", None),
            write_surface_binding=getattr(module, "write_surface_binding", None),
            surface_binding_path=getattr(module, "surface_binding_path", None),
        )

    @classmethod
    def from_service(cls, service: Any) -> "FeatureWorkflowAdapter":
        """Adapt GRIM's small integration service contract.

        The integration-facing names are ``make_request``,
        ``discover_dataset_ids(point_csv=None, line_csv=None)``, ``prepare``,
        and ``execute``.  ``from_module`` remains available when the GHOST
        module itself is injected directly.
        """

        missing = [
            name
            for name in ("make_request", "discover_dataset_ids", "prepare", "execute")
            if not callable(getattr(service, name, None))
        ]
        if missing:
            raise TypeError(
                "Feature assembly service is missing callable(s): "
                + ", ".join(missing)
            )

        def discover(**kwargs: Any) -> Any:
            return service.discover_dataset_ids(
                point_csv=kwargs.get("point_locations_csv"),
                line_csv=kwargs.get("line_locations_csv"),
            )

        return cls(
            request_factory=service.make_request,
            discover=discover,
            prepare=service.prepare,
            execute=service.execute,
            preview_inputs=getattr(service, "prepare_input_preview", None),
            check_surface_binding=getattr(service, "check_surface_binding", None),
            write_surface_binding=getattr(service, "write_surface_binding", None),
            surface_binding_path=getattr(service, "surface_binding_path", None),
        )


def coerce_feature_workflow(service: Any) -> FeatureWorkflowAdapter:
    """Return an adapter for an adapter or feature-workflow-like module."""

    if isinstance(service, FeatureWorkflowAdapter):
        return service
    if service is None:
        raise RuntimeError(
            "Feature assembly is unavailable because no GHOST feature service "
            "has been connected."
        )
    if callable(getattr(service, "make_request", None)):
        return FeatureWorkflowAdapter.from_service(service)
    return FeatureWorkflowAdapter.from_module(service)


@dataclass
class FeatureAssemblyValues:
    """User-editable values, independent of any GUI toolkit."""

    base_grim: str = ""
    output_grim: str = ""
    # Placement and mesh formats do not reliably encode length units.  Empty
    # defaults force a deliberate choice instead of silently assuming inches.
    coordinate_units: str = ""
    surface_mesh: str = ""
    surface_units: str = ""
    flip_surface_normals: bool = False
    shadow: bool = False
    shadow_bias_m: float | None = None
    point_locations_csv: str = ""
    line_locations_csv: str = ""
    skin_tol_m: float = 1.0e-3
    skin_phase_tol_deg: float = 15.0
    normal_tol_deg: float = 15.0
    allow_legacy_base_metadata: bool = False
    require_feature_manifests: bool = True
    require_body_mesh_certification: bool = True
    expected_host_material: str = ""
    base_dir: str | None = None
    point_datasets: dict[str, str] = field(default_factory=dict)
    line_datasets: dict[str, str] = field(default_factory=dict)
    point_host_materials: dict[str, str] = field(default_factory=dict)
    line_host_materials: dict[str, str] = field(default_factory=dict)
    # Spatial feature-definition state. These stable CSV IDs are independent
    # from both preview visibility and whole-response dataset arithmetic.
    excluded_point_placement_ids: set[str] = field(default_factory=set)
    excluded_line_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LoadedFeatureAssemblyRecipe:
    """One validated recipe plus non-fatal source-integrity observations."""

    path: Path
    name: str
    variant: str
    values: FeatureAssemblyValues
    source_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaseGrimPreflight:
    """Cheap ZIP-key classification used only for honest GUI readiness."""

    valid: bool
    embedded_bor: bool
    requires_surface_mesh: bool
    summary: str
    keys: frozenset[str] = frozenset()
    azimuth_count: int = 0
    elevation_count: int = 0
    frequency_count: int = 0


@dataclass(frozen=True)
class SurfaceBindingReadiness:
    """Cheap UI state for an explicitly checked external-body binding."""

    code: str
    message: str
    ready: bool
    required: bool
    external_body: bool
    sidecar_path: Path | None = None
    identity_key: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class FeatureBuildDispatch:
    """Result returned by the combined prepare/execute operation."""

    plan: Any
    output_path: str
    reused_validated_plan: bool = False
    features_only_output_path: str = ""
    features_only_output_published: bool = False


@dataclass(frozen=True)
class _FileFingerprint:
    """Cheap GUI cache identity; backend content hashes remain authoritative."""

    resolved_path: str
    exists: bool
    size: int | None = None
    mtime_ns: int | None = None
    ctime_ns: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class _DatasetDiscovery:
    """Parser result paired with the exact CSV bytes that were parsed."""

    requirements: Any
    point_fingerprint: _FileFingerprint | None = None
    line_fingerprint: _FileFingerprint | None = None


@dataclass(frozen=True)
class _VerifiedInputPreview:
    """Input preview paired with the exact placement CSVs that produced it."""

    preview: Any
    discovery: _DatasetDiscovery | None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.preview, name)


@dataclass(frozen=True)
class _PreparedPlanCache:
    """A physically validated plan that is safe to reuse while inputs match."""

    plan: Any
    semantic_signature: tuple[Any, ...]
    source_fingerprints: tuple[tuple[str, _FileFingerprint], ...]
    service_key: tuple[Any, ...]


@dataclass(frozen=True)
class LoadedDatasetEntry:
    """Small, file-oriented dataset reference accepted by the Assembly UI.

    Feature assembly is intentionally a file-backed workflow: the GHOST
    backend consumes response paths rather than live ``RcsGrid`` objects.
    ``dirty`` therefore prevents an inherited source path from being offered
    as though it represented the current in-memory data.
    """

    dataset_id: str
    name: str
    path: str = ""
    dirty: bool = False
    _usable_path: str = field(init=False, repr=False)
    _unavailable_reason: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        name = str(self.name).strip()
        path = str(self.path or "").strip()
        dirty = bool(self.dirty)
        if not dataset_id:
            raise ValueError(
                "An Assembly loaded-dataset entry requires a stable dataset_id."
            )
        if not name:
            raise ValueError(
                "An Assembly loaded-dataset entry requires a display name."
            )
        if dirty or not path:
            usable_path = ""
            reason = "save unsaved derived dataset first"
        else:
            candidate = Path(path)
            if candidate.suffix.casefold() != ".grim":
                usable_path = ""
                reason = "not a .grim file"
            elif not candidate.is_file():
                usable_path = ""
                reason = "saved file is missing"
            else:
                usable_path = str(candidate)
                reason = ""
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "dirty", dirty)
        object.__setattr__(self, "_usable_path", usable_path)
        object.__setattr__(self, "_unavailable_reason", reason)

    @property
    def usable_path(self) -> str:
        """Return the backend-safe path, or an empty string when unavailable."""

        return self._usable_path

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason


def _entry_value(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _coerce_loaded_dataset_entry(value: Any) -> LoadedDatasetEntry:
    """Accept shell-facing mappings, tuples, or objects with familiar names."""

    if isinstance(value, LoadedDatasetEntry):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            dataset_id, name, path = value
            return LoadedDatasetEntry(
                str(dataset_id or ""), str(name or ""), _clean_path(path)
            )
        if len(value) == 4:
            dataset_id, name, path, dirty = value
            return LoadedDatasetEntry(
                str(dataset_id or ""),
                str(name or ""),
                _clean_path(path),
                bool(dirty),
            )
        raise TypeError(
            "Assembly loaded-dataset tuples must contain "
            "(dataset_id, name, path[, dirty])."
        )

    dataset_id = _entry_value(value, "dataset_id", "stable_id", "id", "key")
    name = _entry_value(value, "name", "display_name", "label")
    path = _entry_value(
        value,
        "path",
        "source",
        "source_path",
        "file_path",
        "output_path",
        default="",
    )
    dirty = _entry_value(
        value,
        "dirty",
        "is_dirty",
        "unsaved",
        "is_unsaved",
        default=False,
    )
    if dataset_id is None and name is None and path is None:
        raise TypeError(
            "Assembly loaded-dataset entries must be mappings, "
            "(dataset_id, name, path[, dirty]) tuples, or objects with "
            "matching attributes."
        )
    return LoadedDatasetEntry(
        str(dataset_id or ""),
        str(name or ""),
        _clean_path(path),
        bool(dirty),
    )


def _coerce_loaded_dataset_catalog(
    entries: Iterable[Any],
) -> tuple[LoadedDatasetEntry, ...]:
    catalog = tuple(_coerce_loaded_dataset_entry(entry) for entry in entries)
    identifiers = [entry.dataset_id for entry in catalog]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Assembly loaded dataset_id values must be unique.")
    return catalog


def _clean_path(value: Any) -> str:
    return str(value or "").strip()


def _normalize_host_material(value: Any) -> str:
    """Collapse insignificant host-ID whitespace while preserving display case."""

    return " ".join(str(value or "").split())


def _resolved_user_path(value: Any, *, base_dir: Any = None) -> Path:
    """Resolve a user path with the same base-directory rule as GHOST."""

    path = Path(_clean_path(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path.cwd() if not _clean_path(base_dir) else Path(base_dir).expanduser()
    return (root.resolve() / path).resolve()


def _normalized_grim_output_path(value: Any, *, base_dir: Any = None) -> Path:
    """Return the destination the writer will actually use."""

    resolved = _resolved_user_path(value, base_dir=base_dir)
    if str(resolved).casefold().endswith(".grim"):
        return resolved
    return Path(str(resolved) + ".grim")


def _features_only_grim_output_path(
    value: Any, *, base_dir: Any = None
) -> Path:
    """Return the published feature-delta sibling for one Assembly output."""

    output = _normalized_grim_output_path(value, base_dir=base_dir)
    return output.with_name(output.stem + "_features_only" + output.suffix)


def _path_key(path: Path) -> str:
    """Comparable canonical path key (case-insensitive on Windows)."""

    return os.path.normcase(str(path.resolve()))


def _publication_snapshot(path: Path) -> tuple[Any, ...]:
    """Return observable file identity used only as publication evidence."""

    try:
        result = os.stat(path, follow_symlinks=False)
    except OSError:
        return (False, None, None, None, None, None)
    return (
        True,
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _published_during_execution(
    path: Path,
    before: tuple[Any, ...],
) -> bool:
    """Conservatively require a new or observably replaced regular file."""

    after = _publication_snapshot(path)
    return bool(after[0] and after != before and path.is_file())


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether two paths name the same target, including hard links."""

    if _path_key(first) == _path_key(second):
        return True
    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError):
        return False


_BASE_GRIM_REQUIRED_KEYS = frozenset(
    {
        "azimuths",
        "elevations",
        "frequencies",
        "polarizations",
        "rcs_power",
        "rcs_phase",
    }
)


def _npy_member_vector_count(
    archive: zipfile.ZipFile,
    member_name: str,
) -> int:
    """Read only one NPY header and return its 1-D length."""

    with archive.open(member_name, "r") as stream:
        if stream.read(6) != b"\x93NUMPY":
            return 0
        version = stream.read(2)
        if len(version) != 2:
            return 0
        major = version[0]
        length_bytes = stream.read(2 if major == 1 else 4)
        if len(length_bytes) not in (2, 4):
            return 0
        header_length = int.from_bytes(length_bytes, "little")
        header = stream.read(header_length)
    try:
        metadata = ast.literal_eval(header.decode("latin1").strip())
        shape = metadata["shape"]
    except (KeyError, SyntaxError, UnicodeDecodeError, ValueError):
        return 0
    if not isinstance(shape, tuple) or len(shape) != 1:
        return 0
    try:
        return max(0, int(shape[0]))
    except (TypeError, ValueError):
        return 0


@lru_cache(maxsize=64)
def _preflight_base_grim_zip(
    resolved_path: str,
    size: int,
    mtime_ns: int,
) -> BaseGrimPreflight:
    """Inspect only the immutable ZIP directory identified by path/stat."""

    del size, mtime_ns  # Values deliberately participate in the cache key.
    path = Path(resolved_path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = {
                Path(name).stem: name
                for name in archive.namelist()
                if name.casefold().endswith(".npy")
                and not name.endswith(("/", "\\"))
            }
            keys = frozenset(members)
            axis_counts = {
                key: _npy_member_vector_count(archive, members[key])
                for key in ("azimuths", "elevations", "frequencies")
                if key in members
            }
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        return BaseGrimPreflight(
            False,
            False,
            False,
            f"Invalid GRIM container: {exc}",
        )
    missing = sorted(_BASE_GRIM_REQUIRED_KEYS - keys)
    if missing:
        return BaseGrimPreflight(
            False,
            False,
            False,
            "Malformed GRIM response; missing key(s): " + ", ".join(missing),
            keys,
        )
    has_real = "rcs_amp_real" in keys
    has_imag = "rcs_amp_imag" in keys
    if has_real != has_imag:
        return BaseGrimPreflight(
            False,
            False,
            False,
            "Malformed GRIM response; complex amplitude must contain both "
            "rcs_amp_real and rcs_amp_imag.",
            keys,
        )
    has_rho = "body_profile_rho_m" in keys
    has_z = "body_profile_z_m" in keys
    if has_rho != has_z:
        return BaseGrimPreflight(
            False,
            False,
            False,
            "Malformed embedded body profile; both rho and z arrays are required.",
            keys,
        )
    has_requested_grid = "requested_radar_grid_json" in keys
    if has_requested_grid and not (has_rho and has_z):
        return BaseGrimPreflight(
            False,
            False,
            False,
            "Malformed embedded BoR metadata; requested radar grid has no body profile.",
            keys,
        )
    embedded_bor = bool(has_rho and has_z and has_requested_grid)
    if embedded_bor:
        summary = (
            "Embedded BoR geometry detected; a separate mesh is optional unless "
            "geometric shadowing is enabled."
        )
    elif has_rho and has_z:
        summary = (
            "Legacy body profile lacks the requested radar-grid record; provide a "
            "matching STL/facet mesh or regenerate the body response."
        )
    else:
        summary = (
            "External 3-D body response detected; choose its matching STL/facet "
            "surface before validation."
        )
    return BaseGrimPreflight(
        True,
        embedded_bor,
        not embedded_bor,
        summary,
        keys,
        azimuth_count=int(axis_counts.get("azimuths", 0)),
        elevation_count=int(axis_counts.get("elevations", 0)),
        frequency_count=int(axis_counts.get("frequencies", 0)),
    )


def preflight_base_grim(
    value: Any,
    *,
    base_dir: Any = None,
) -> BaseGrimPreflight:
    """Classify one selected base without loading its potentially large arrays."""

    if not _clean_path(value):
        return BaseGrimPreflight(
            False, False, False, "Choose a clean-body .grim response."
        )
    try:
        path = _resolved_user_path(value, base_dir=base_dir)
        if not path.is_file():
            return BaseGrimPreflight(
                False, False, False, f"Clean-body file was not found: {path}"
            )
        if path.suffix.casefold() != ".grim":
            return BaseGrimPreflight(
                False, False, False, "Clean-body response must use the .grim extension."
            )
        stat = path.stat()
        return _preflight_base_grim_zip(
            str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
        )
    except OSError as exc:
        return BaseGrimPreflight(
            False, False, False, f"Clean-body preflight failed: {exc}"
        )


@lru_cache(maxsize=64)
def _surface_mesh_triangle_hint_cached(
    resolved_path: str,
    size: int,
    mtime_ns: int,
) -> tuple[int, bool]:
    """Read only a mesh header; return (triangle hint, exact)."""

    del mtime_ns
    path = Path(resolved_path)
    try:
        if path.suffix.casefold() == ".facet":
            with path.open("r", encoding="utf-8-sig") as stream:
                for raw in stream:
                    text = raw.split("#", 1)[0].strip()
                    if not text:
                        continue
                    tokens = text.split()
                    if len(tokens) != 2:
                        return 0, False
                    facets = int(tokens[1])
                    # A facet is a triangle or quad; two triangles per declared
                    # facet is a safe pre-validation upper hint.
                    return max(0, 2 * facets), False
            return 0, False
        if path.suffix.casefold() == ".stl" and size >= 84:
            with path.open("rb") as stream:
                header = stream.read(84)
            if len(header) != 84:
                return 0, False
            triangles = int(struct.unpack("<I", header[80:84])[0])
            if 84 + 50 * triangles == size:
                return max(0, triangles), True
    except (OSError, UnicodeError, ValueError, struct.error):
        return 0, False
    return 0, False


def surface_mesh_triangle_hint(
    value: Any,
    *,
    base_dir: Any = None,
) -> tuple[int, bool]:
    """Return a cheap triangle-count hint without loading mesh coordinates."""

    if not _clean_path(value):
        return 0, False
    try:
        path = _resolved_user_path(value, base_dir=base_dir)
        if not path.is_file():
            return 0, False
        stat = path.stat()
        return _surface_mesh_triangle_hint_cached(
            str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
        )
    except OSError:
        return 0, False


def _axis_size(value: Any) -> int:
    try:
        size = getattr(value, "size")
    except (AttributeError, TypeError):
        size = None
    if size is not None:
        try:
            return max(0, int(size))
        except (TypeError, ValueError):
            pass
    try:
        return max(0, len(value))
    except (TypeError, ValueError):
        return 0


def estimate_validated_assembly_plan_workload(plan: Any) -> AssemblyWorkEstimate:
    """Derive exact stage quantities from one authoritative validated plan."""

    grid = getattr(plan, "radar_grid", {}) or {}
    azimuth_count = _axis_size(grid.get("azimuths_deg", ()))
    elevation_count = _axis_size(grid.get("elevations_deg", ()))
    frequency_count = _axis_size(grid.get("frequencies_ghz", ()))
    points = tuple(getattr(plan, "point_placements", ()) or ())
    lines = tuple(getattr(plan, "line_placements", ()) or ())
    line_piece_count = 0
    line_segment_count = 0
    pieces_exact = True
    for placement in lines:
        segment_count = 0
        try:
            perimeter = placement["perimeter"]
            segment_count = len(perimeter)
            line_segment_count += int(segment_count)
            shadow_points = placement.get("shadow_points")
            if shadow_points is not None:
                line_piece_count += int(len(shadow_points))
                continue
            maximum_piece_length = float(placement["max_piece_length_m"])
            if not math.isfinite(maximum_piece_length) or maximum_piece_length <= 0.0:
                raise ValueError
            for segment in perimeter:
                start = segment[0]
                end = segment[1]
                length = math.sqrt(sum(
                    (float(end[index]) - float(start[index])) ** 2
                    for index in range(3)
                ))
                line_piece_count += max(1, int(math.ceil(
                    length / maximum_piece_length
                )))
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            pieces_exact = False
            line_piece_count += max(
                1, int(segment_count)
            ) * _PREVALIDATION_LINE_PIECES_PER_SEGMENT

    request = getattr(plan, "request", None)
    shadow_enabled = bool(getattr(request, "shadow", False))
    surface = getattr(plan, "surface", None)
    triangles = getattr(surface, "triangles", None)
    try:
        triangle_count = len(triangles) if triangles is not None else 0
    except TypeError:
        triangle_count = 0
    return estimate_assembly_workload(
        look_count=azimuth_count * elevation_count,
        frequency_count=frequency_count,
        point_count=len(points),
        line_path_count=len(lines),
        line_segment_count=line_segment_count,
        line_piece_count=line_piece_count,
        mesh_triangle_count=triangle_count,
        shadow_enabled=shadow_enabled,
        quantities_validated=True,
        line_piece_count_exact=pieces_exact,
        mesh_triangle_count_exact=bool(triangles is not None),
    )


def _fingerprint_file(
    value: Any,
    *,
    base_dir: Any = None,
    include_hash: bool = True,
) -> _FileFingerprint:
    """Fingerprint one input and reject a file that changes while hashing."""

    resolved = _resolved_user_path(value, base_dir=base_dir)
    key = _path_key(resolved)
    if not resolved.is_file():
        return _FileFingerprint(resolved_path=key, exists=False)

    if not include_hash:
        stat = resolved.stat()
        return _FileFingerprint(
            resolved_path=key,
            exists=True,
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
        )

    for _attempt in range(2):
        before = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = resolved.stat()
        if (
            before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
            and before.st_ctime_ns == after.st_ctime_ns
        ):
            return _FileFingerprint(
                resolved_path=key,
                exists=True,
                size=int(after.st_size),
                mtime_ns=int(after.st_mtime_ns),
                ctime_ns=int(after.st_ctime_ns),
                sha256=digest.hexdigest(),
            )
    raise RuntimeError(f"Input changed while it was being read: {resolved}")


def _surface_binding_sidecar_path(
    surface_mesh: Any,
    *,
    base_dir: Any = None,
) -> Path | None:
    """Return the backend's canonical ``<surface>.assembly.json`` path."""

    if not _clean_path(surface_mesh):
        return None
    resolved = _resolved_user_path(surface_mesh, base_dir=base_dir)
    return Path(str(resolved) + ".assembly.json")


def _surface_preview_identity_key(
    surface_mesh: Any,
    surface_units: Any,
    *,
    base_dir: Any = None,
) -> tuple[Any, ...] | None:
    """Stat-only identity for one already interpreted surface preview."""

    path = _clean_path(surface_mesh)
    units = str(surface_units or "").strip()
    if not path or units not in UNIT_SCALE_M:
        return None
    fingerprint = _fingerprint_file(
        path,
        base_dir=base_dir,
        include_hash=False,
    )
    return (
        units,
        fingerprint.resolved_path,
        fingerprint.exists,
        fingerprint.size,
        fingerprint.mtime_ns,
        fingerprint.ctime_ns,
    )


def _surface_dimensions_summary(
    surface_triangles_cad_m: Any,
    *,
    surface_units: Any,
) -> str:
    """Describe interpreted mesh spans in meters and selected source units."""

    units = str(surface_units or "").strip()
    if surface_triangles_cad_m is None or units not in UNIT_SCALE_M:
        return ""
    try:
        vertices = surface_triangles_cad_m.reshape((-1, 3))
        if int(vertices.shape[0]) == 0:
            return ""
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        spans_m = [
            max(0.0, float(maximum[i]) - float(minimum[i]))
            for i in range(3)
        ]
    except (AttributeError, IndexError, TypeError, ValueError):
        return ""
    if not all(math.isfinite(value) for value in spans_m):
        return ""
    scale = UNIT_SCALE_M[units]
    spans_source = [value / scale for value in spans_m]

    def format_triplet(values: Iterable[float]) -> str:
        return " x ".join(f"{float(value):.6g}" for value in values)

    return (
        "Interpreted physical size: "
        + format_triplet(spans_m)
        + " m (x/y/z). Source-coordinate spans: "
        + format_triplet(spans_source)
        + f" {UNIT_ABBREVIATIONS[units]} ({units} selected)."
    )


def _surface_binding_identity_key(
    base_grim: Any,
    surface_mesh: Any,
    surface_units: Any,
    *,
    base_dir: Any = None,
) -> tuple[Any, ...] | None:
    """Stat-only cache key; this deliberately never hashes large body files."""

    sidecar = _surface_binding_sidecar_path(surface_mesh, base_dir=base_dir)
    if sidecar is None:
        return None
    fingerprints = (
        _fingerprint_file(base_grim, base_dir=base_dir, include_hash=False),
        _fingerprint_file(surface_mesh, base_dir=base_dir, include_hash=False),
        _fingerprint_file(sidecar, include_hash=False),
    )
    return (
        str(surface_units).strip(),
        *(
            (
                value.resolved_path,
                value.exists,
                value.size,
                value.mtime_ns,
                value.ctime_ns,
            )
            for value in fingerprints
        ),
    )


def assess_surface_binding_readiness(
    *,
    base_grim: Any,
    surface_mesh: Any,
    surface_units: Any,
    production_profile: bool,
    base_dir: Any = None,
    checked_key: tuple[Any, ...] | None = None,
    checked_binding: Mapping[str, Any] | None = None,
    error_key: tuple[Any, ...] | None = None,
    check_error: str = "",
    tools_available: bool = True,
) -> SurfaceBindingReadiness:
    """Describe binding readiness without content-hashing from a GUI refresh.

    Exact base/surface hashes are intentionally delegated to the explicit
    backend Check/Bind actions. A matching stat key means that explicit check
    still describes the same selected paths, units, and sidecar bytes.
    """

    required = bool(production_profile)
    preflight = preflight_base_grim(base_grim, base_dir=base_dir)
    if not preflight.valid:
        return SurfaceBindingReadiness(
            "waiting",
            "Select a valid clean-body GRIM before checking registration.",
            ready=not required,
            required=False,
            external_body=False,
        )
    if preflight.embedded_bor:
        return SurfaceBindingReadiness(
            "not_required",
            "✓ Embedded BoR geometry is self-bound; no external surface "
            "binding is required.",
            ready=True,
            required=False,
            external_body=False,
        )

    surface_path = (
        _resolved_user_path(surface_mesh, base_dir=base_dir)
        if _clean_path(surface_mesh)
        else None
    )
    sidecar = _surface_binding_sidecar_path(surface_mesh, base_dir=base_dir)
    if (
        surface_path is None
        or not surface_path.is_file()
        or surface_path.suffix.casefold() not in {".stl", ".facet"}
    ):
        return SurfaceBindingReadiness(
            "waiting",
            "Choose the matching STL/facet mesh before checking solve-to-CAD "
            "registration.",
            ready=not required,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
        )
    selected_units = str(surface_units or "").strip()
    if selected_units not in UNIT_SCALE_M:
        message = (
            "Choose the physical units of the selected surface mesh before "
            "checking solve-to-CAD registration."
            if not selected_units
            else f"Unsupported surface mesh units: {selected_units!r}."
        )
        return SurfaceBindingReadiness(
            "waiting",
            message,
            ready=not required,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
        )
    if not tools_available:
        return SurfaceBindingReadiness(
            "unavailable",
            "✗ The connected GHOST backend cannot check external-body bindings.",
            ready=not required,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
        )
    if sidecar is None or not sidecar.is_file():
        qualifier = "Production requires" if required else "Production will require"
        return SurfaceBindingReadiness(
            "missing",
            f"✗ Binding missing — {qualifier} {surface_path.name}.assembly.json.",
            ready=not required,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
        )
    try:
        identity = _surface_binding_identity_key(
            base_grim,
            surface_mesh,
            surface_units,
            base_dir=base_dir,
        )
    except OSError as exc:
        return SurfaceBindingReadiness(
            "invalid",
            f"✗ Binding status could not be read: {exc}",
            ready=not required,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
        )
    if error_key == identity and str(check_error).strip():
        return SurfaceBindingReadiness(
            "invalid",
            "✗ Binding is stale or invalid: " + str(check_error).strip(),
            ready=not required,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
            identity_key=identity,
        )
    if checked_key == identity and isinstance(checked_binding, Mapping):
        geometry = str(checked_binding.get("geometry_id", "")).strip()
        case_id = str(checked_binding.get("attestation_case_id", "")).strip()
        return SurfaceBindingReadiness(
            "valid",
            f"✓ Current reviewed binding — geometry {geometry}; registration {case_id}.",
            ready=True,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
            identity_key=identity,
        )
    if checked_key is not None or error_key is not None:
        message = (
            "⚠ Binding check is stale — the body, mesh, units, or sidecar changed. "
            "Click Check binding again."
        )
        code = "stale"
    else:
        message = (
            "○ Binding found but not checked for the current body, mesh, and units. "
            "Click Check binding before Production validation."
        )
        code = "unchecked"
    return SurfaceBindingReadiness(
        code,
        message,
        ready=not required,
        required=required,
        external_body=True,
        sidecar_path=sidecar,
        identity_key=identity,
    )


def _recipe_target_path(value: str | Path) -> Path:
    raw = _clean_path(value)
    if not raw:
        raise ValueError("Choose where to save the Assembly recipe.")
    target = Path(raw).expanduser()
    if target.suffix.casefold() != ".json":
        target = Path(str(target) + FEATURE_RECIPE_SUFFIX)
    return target.resolve()


def _recipe_relative_path(
    value: Any,
    *,
    source_base_dir: Any,
    recipe_dir: Path,
) -> str:
    """Store one effective path relative to the recipe when possible."""

    if not _clean_path(value):
        return ""
    resolved = _resolved_user_path(value, base_dir=source_base_dir)
    try:
        relative = os.path.relpath(str(resolved), str(recipe_dir))
    except ValueError:  # Different Windows drives cannot form a relative path.
        return str(resolved)
    return Path(relative).as_posix()


def _recipe_absolute_path(value: Any, *, recipe_dir: Path) -> str:
    raw = _clean_path(value)
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = recipe_dir / path
    return str(path.resolve())


def _recipe_source_items(
    values: FeatureAssemblyValues,
) -> tuple[tuple[str, str | None, str], ...]:
    items: list[tuple[str, str | None, str]] = [
        ("base_grim", None, _clean_path(values.base_grim)),
        ("surface_mesh", None, _clean_path(values.surface_mesh)),
        ("point_locations_csv", None, _clean_path(values.point_locations_csv)),
        ("line_locations_csv", None, _clean_path(values.line_locations_csv)),
    ]
    items.extend(
        ("point_dataset", str(dataset_id), _clean_path(path))
        for dataset_id, path in sorted(values.point_datasets.items())
    )
    items.extend(
        ("line_dataset", str(dataset_id), _clean_path(path))
        for dataset_id, path in sorted(values.line_datasets.items())
    )
    return tuple(item for item in items if item[2])


def feature_assembly_recipe_payload(
    values: FeatureAssemblyValues,
    *,
    recipe_path: str | Path,
    name: str,
    variant: str,
) -> dict[str, Any]:
    """Return a portable, versioned recipe with lightweight source identity."""

    if not isinstance(values, FeatureAssemblyValues):
        raise TypeError("values must be FeatureAssemblyValues")
    target = _recipe_target_path(recipe_path)
    recipe_dir = target.parent
    clean_name = str(name).strip()
    clean_variant = str(variant).strip()
    if not clean_name:
        raise ValueError("Enter an Assembly recipe name.")
    if not clean_variant:
        raise ValueError("Enter a variant name, such as Baseline or Option A.")

    def relative(path: Any) -> str:
        return _recipe_relative_path(
            path,
            source_base_dir=values.base_dir,
            recipe_dir=recipe_dir,
        )

    serialized_values: dict[str, Any] = {
        "base_grim": relative(values.base_grim),
        "output_grim": relative(values.output_grim),
        "coordinate_units": str(values.coordinate_units),
        "surface_mesh": relative(values.surface_mesh),
        "surface_units": str(values.surface_units),
        "flip_surface_normals": bool(values.flip_surface_normals),
        "shadow": bool(values.shadow),
        "shadow_bias_m": (
            None
            if values.shadow_bias_m is None
            else float(values.shadow_bias_m)
        ),
        "point_locations_csv": relative(values.point_locations_csv),
        "line_locations_csv": relative(values.line_locations_csv),
        "skin_tol_m": float(values.skin_tol_m),
        "skin_phase_tol_deg": float(values.skin_phase_tol_deg),
        "normal_tol_deg": float(values.normal_tol_deg),
        "allow_legacy_base_metadata": bool(values.allow_legacy_base_metadata),
        "require_feature_manifests": bool(values.require_feature_manifests),
        "require_body_mesh_certification": bool(
            values.require_body_mesh_certification
        ),
        "expected_host_material": str(values.expected_host_material).strip(),
        # Every effective path above is rebased to this recipe directory.
        "base_dir": ".",
        "point_datasets": {
            str(dataset_id): relative(path)
            for dataset_id, path in sorted(values.point_datasets.items())
        },
        "line_datasets": {
            str(dataset_id): relative(path)
            for dataset_id, path in sorted(values.line_datasets.items())
        },
        "point_host_materials": {
            str(dataset_id): str(
                values.point_host_materials.get(dataset_id, "")
            ).strip()
            for dataset_id in sorted(values.point_datasets)
        },
        "line_host_materials": {
            str(dataset_id): str(
                values.line_host_materials.get(dataset_id, "")
            ).strip()
            for dataset_id in sorted(values.line_datasets)
        },
        "excluded_point_placement_ids": sorted(
            str(value) for value in values.excluded_point_placement_ids
        ),
        "excluded_line_ids": sorted(
            str(value) for value in values.excluded_line_ids
        ),
    }

    manifest: list[dict[str, Any]] = []
    for role, dataset_id, path in _recipe_source_items(values):
        resolved = _resolved_user_path(path, base_dir=values.base_dir)
        try:
            size = int(resolved.stat().st_size) if resolved.is_file() else None
            include_hash = bool(
                size is not None and size <= FEATURE_RECIPE_HASH_LIMIT_BYTES
            )
            fingerprint = _fingerprint_file(
                path,
                base_dir=values.base_dir,
                include_hash=include_hash,
            )
        except OSError:
            fingerprint = _FileFingerprint(
                resolved_path=_path_key(resolved), exists=False
            )
        record: dict[str, Any] = {
            "role": role,
            "path": relative(path),
            "exists": fingerprint.exists,
            "size": fingerprint.size,
            "mtime_ns": fingerprint.mtime_ns,
        }
        if dataset_id is not None:
            record["dataset_id"] = dataset_id
        if fingerprint.sha256 is not None:
            record["sha256"] = fingerprint.sha256
        manifest.append(record)

    return {
        "schema": FEATURE_RECIPE_SCHEMA,
        "version": FEATURE_RECIPE_VERSION,
        "name": clean_name,
        "variant": clean_variant,
        "path_policy": "relative-to-recipe",
        "values": serialized_values,
        "source_manifest": manifest,
    }


def write_feature_assembly_recipe(
    values: FeatureAssemblyValues,
    path: str | Path,
    *,
    name: str,
    variant: str,
) -> Path:
    """Atomically save one portable Assembly recipe."""

    target = _recipe_target_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"Assembly recipe folder does not exist: {target.parent}"
        )
    payload = feature_assembly_recipe_payload(
        values,
        recipe_path=target,
        name=name,
        variant=variant,
    )
    serialized = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return target


def _recipe_string_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Assembly recipe {label} must be an object.")
    result: dict[str, str] = {}
    for key, path in value.items():
        dataset_id = str(key).strip()
        if not dataset_id or not isinstance(path, str):
            raise ValueError(
                f"Assembly recipe {label} must map nonempty IDs to paths."
            )
        result[dataset_id] = path
    return result


def _recipe_id_set(value: Any, label: str) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"Assembly recipe {label} must be a list.")
    result = {str(item).strip() for item in value}
    if "" in result or len(result) != len(value):
        raise ValueError(
            f"Assembly recipe {label} contains a blank or duplicate ID."
        )
    return result


def read_feature_assembly_recipe(
    path: str | Path,
) -> LoadedFeatureAssemblyRecipe:
    """Load one recipe and report missing or changed referenced inputs."""

    source = Path(_clean_path(path)).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Assembly recipe is not valid JSON ({exc.msg} at line {exc.lineno})."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Assembly recipe root must be a JSON object.")
    if payload.get("schema") != FEATURE_RECIPE_SCHEMA:
        raise ValueError(
            f"Not a {FEATURE_RECIPE_SCHEMA!r} Assembly recipe."
        )
    version = payload.get("version")
    if version not in {1, 2, 3, FEATURE_RECIPE_VERSION}:
        raise ValueError(
            f"Unsupported Assembly recipe version {version!r}; this GRIM build "
            f"supports versions 1 through {FEATURE_RECIPE_VERSION}."
        )
    name = payload.get("name")
    variant = payload.get("variant")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Assembly recipe name must be a nonempty string.")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("Assembly recipe variant must be a nonempty string.")
    raw_values = payload.get("values")
    if not isinstance(raw_values, Mapping):
        raise ValueError("Assembly recipe values must be a JSON object.")

    required = {
        "base_grim",
        "output_grim",
        "coordinate_units",
        "surface_mesh",
        "surface_units",
        "flip_surface_normals",
        "shadow",
        "shadow_bias_m",
        "point_locations_csv",
        "line_locations_csv",
        "skin_tol_m",
        "skin_phase_tol_deg",
        "normal_tol_deg",
        "allow_legacy_base_metadata",
        "require_feature_manifests",
        "base_dir",
        "point_datasets",
        "line_datasets",
        "excluded_point_placement_ids",
        "excluded_line_ids",
    }
    if version >= 2:
        required.add("expected_host_material")
    if version >= 3:
        required.update({"point_host_materials", "line_host_materials"})
    if version >= 4:
        required.add("require_body_mesh_certification")
    missing = sorted(required - set(raw_values))
    if missing:
        raise ValueError(
            "Assembly recipe is missing value(s): " + ", ".join(missing)
        )
    for key in (
        "base_grim",
        "output_grim",
        "surface_mesh",
        "point_locations_csv",
        "line_locations_csv",
    ):
        if not isinstance(raw_values[key], str):
            raise ValueError(f"Assembly recipe {key} must be a path string.")
    if any(
        not isinstance(raw_values[key], bool)
        for key in (
            "flip_surface_normals",
            "shadow",
            "allow_legacy_base_metadata",
            "require_feature_manifests",
            *(
                ("require_body_mesh_certification",)
                if version >= 4
                else ()
            ),
        )
    ):
        raise ValueError("Assembly recipe boolean settings must be true or false.")
    expected_host_material = raw_values.get("expected_host_material", "")
    if not isinstance(expected_host_material, str):
        raise ValueError(
            "Assembly recipe expected_host_material must be a text value."
        )

    coordinate_units = str(raw_values["coordinate_units"])
    surface_units = str(raw_values["surface_units"])
    supported_units = {value for _label, value in UNIT_CHOICES}
    if coordinate_units and coordinate_units not in supported_units:
        raise ValueError("Assembly recipe contains unsupported coordinate units.")
    if surface_units and surface_units not in supported_units:
        raise ValueError("Assembly recipe contains unsupported surface units.")
    skin_tol = _require_finite_nonnegative(
        raw_values["skin_tol_m"], "Recipe skin distance tolerance"
    )
    phase_tol = _require_finite_nonnegative(
        raw_values["skin_phase_tol_deg"], "Recipe skin phase tolerance"
    )
    normal_tol = _require_finite_nonnegative(
        raw_values["normal_tol_deg"], "Recipe normal tolerance"
    )
    if skin_tol > 0.1:
        raise ValueError(
            "Assembly recipe skin distance tolerance must not exceed 100 mm."
        )
    if not 0.0 < phase_tol <= 90.0:
        raise ValueError(
            "Assembly recipe skin phase tolerance must be above 0 and at most "
            "90 degrees."
        )
    if normal_tol >= 90.0:
        raise ValueError("Assembly recipe normal tolerance must be below 90 degrees.")
    shadow_bias_raw = raw_values["shadow_bias_m"]
    shadow_bias = (
        None
        if shadow_bias_raw is None
        else _require_finite_nonnegative(shadow_bias_raw, "Recipe shadow bias")
    )

    point_paths = _recipe_string_mapping(
        raw_values["point_datasets"], "point_datasets"
    )
    line_paths = _recipe_string_mapping(
        raw_values["line_datasets"], "line_datasets"
    )
    point_host_materials = _recipe_string_mapping(
        raw_values.get("point_host_materials", {}), "point_host_materials"
    )
    line_host_materials = _recipe_string_mapping(
        raw_values.get("line_host_materials", {}), "line_host_materials"
    )
    unknown_point_hosts = sorted(set(point_host_materials) - set(point_paths))
    unknown_line_hosts = sorted(set(line_host_materials) - set(line_paths))
    if unknown_point_hosts or unknown_line_hosts:
        raise ValueError(
            "Assembly recipe host-material rows reference unknown response IDs: "
            f"point={unknown_point_hosts}, line={unknown_line_hosts}."
        )
    recipe_dir = source.parent
    values = FeatureAssemblyValues(
        base_grim=_recipe_absolute_path(raw_values["base_grim"], recipe_dir=recipe_dir),
        output_grim=_recipe_absolute_path(raw_values["output_grim"], recipe_dir=recipe_dir),
        coordinate_units=coordinate_units,
        surface_mesh=_recipe_absolute_path(raw_values["surface_mesh"], recipe_dir=recipe_dir),
        surface_units=surface_units,
        flip_surface_normals=raw_values["flip_surface_normals"],
        shadow=raw_values["shadow"],
        shadow_bias_m=shadow_bias,
        point_locations_csv=_recipe_absolute_path(
            raw_values["point_locations_csv"], recipe_dir=recipe_dir
        ),
        line_locations_csv=_recipe_absolute_path(
            raw_values["line_locations_csv"], recipe_dir=recipe_dir
        ),
        skin_tol_m=skin_tol,
        skin_phase_tol_deg=phase_tol,
        normal_tol_deg=normal_tol,
        allow_legacy_base_metadata=raw_values["allow_legacy_base_metadata"],
        require_feature_manifests=raw_values["require_feature_manifests"],
        require_body_mesh_certification=bool(
            raw_values.get("require_body_mesh_certification", False)
        ),
        expected_host_material=expected_host_material.strip(),
        base_dir=None,
        point_datasets={
            dataset_id: _recipe_absolute_path(value, recipe_dir=recipe_dir)
            for dataset_id, value in point_paths.items()
        },
        line_datasets={
            dataset_id: _recipe_absolute_path(value, recipe_dir=recipe_dir)
            for dataset_id, value in line_paths.items()
        },
        point_host_materials=point_host_materials,
        line_host_materials=line_host_materials,
        excluded_point_placement_ids=_recipe_id_set(
            raw_values["excluded_point_placement_ids"],
            "excluded_point_placement_ids",
        ),
        excluded_line_ids=_recipe_id_set(
            raw_values["excluded_line_ids"], "excluded_line_ids"
        ),
    )

    current_sources = {
        (role, dataset_id): path_value
        for role, dataset_id, path_value in _recipe_source_items(values)
    }
    warnings: list[str] = []
    if version == 1:
        warnings.append(
            "Recipe v1 predates host material/coating identity. Enter that ID "
            "before using the Production validation profile."
        )
    elif version == 2:
        warnings.append(
            "Recipe v2 has only one global host material/coating ID. Review "
            "per-response host IDs before Production validation of mixed stacks."
        )
    if version < 4:
        warnings.append(
            "This older recipe does not claim a certified GHOST body mesh. "
            "It was loaded with the explicit External/HPC body waiver; select "
            "Production only after validating a locally certified body result."
        )
    raw_manifest = payload.get("source_manifest", [])
    if not isinstance(raw_manifest, list):
        raise ValueError("Assembly recipe source_manifest must be a list.")
    seen_manifest_keys: set[tuple[str, str | None]] = set()
    for index, record in enumerate(raw_manifest):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Assembly recipe source_manifest entry {index} must be an object."
            )
        role = str(record.get("role", "")).strip()
        dataset_raw = record.get("dataset_id")
        dataset_id = None if dataset_raw is None else str(dataset_raw).strip()
        key = (role, dataset_id)
        if not role or key in seen_manifest_keys:
            raise ValueError(
                "Assembly recipe source_manifest contains a blank or duplicate role."
            )
        seen_manifest_keys.add(key)
        current_path = current_sources.get(key)
        if not current_path:
            warnings.append(f"{role}: referenced source is no longer configured")
            continue
        display = role if dataset_id is None else f"{role} {dataset_id!r}"
        saved_exists = record.get("exists")
        if not isinstance(saved_exists, bool):
            raise ValueError(
                f"Assembly recipe source_manifest {display} has invalid exists state."
            )
        try:
            current = _fingerprint_file(
                current_path,
                include_hash=isinstance(record.get("sha256"), str),
            )
        except OSError as exc:
            warnings.append(f"{display}: could not verify source ({exc})")
            continue
        if not current.exists:
            warnings.append(f"{display}: file is missing")
            continue
        if not saved_exists:
            warnings.append(f"{display}: file was missing when this recipe was saved")
            continue
        saved_hash = record.get("sha256")
        if isinstance(saved_hash, str):
            if current.sha256 != saved_hash:
                warnings.append(f"{display}: file content changed since recipe save")
            continue
        saved_size = record.get("size")
        if isinstance(saved_size, int) and current.size != saved_size:
            warnings.append(f"{display}: file size changed since recipe save")
        elif (
            isinstance(record.get("mtime_ns"), int)
            and current.mtime_ns != record["mtime_ns"]
        ):
            warnings.append(
                f"{display}: timestamp changed; large-file content was not hashed"
            )

    return LoadedFeatureAssemblyRecipe(
        path=source,
        name=name.strip(),
        variant=variant.strip(),
        values=values,
        source_warnings=tuple(warnings),
    )


def _callable_key(value: Callable[..., Any]) -> tuple[int, int]:
    """Identify a bound function without depending on transient method objects."""

    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", value)
    return (id(owner), id(function))


def _callable_accepts_runtime_hooks(value: Callable[..., Any]) -> bool:
    """Whether ``value`` accepts Assembly progress/cancellation keywords."""

    try:
        parameters = inspect.signature(value).parameters
    except (TypeError, ValueError):
        return False
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    return {"cancel_check", "progress_callback"}.issubset(parameters)


def _callable_accepts_keyword(value: Callable[..., Any], name: str) -> bool:
    """Whether a service accepts one optional execution keyword."""

    try:
        parameters = inspect.signature(value).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _require_finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def _requirements_ids(requirements: Any, attribute: str) -> tuple[str, ...]:
    if isinstance(requirements, Mapping):
        values = requirements.get(attribute, ())
    else:
        values = getattr(requirements, attribute, ())
    ordered = tuple(dict.fromkeys(str(value).strip() for value in values))
    if any(not value for value in ordered):
        raise ValueError("Placement CSV returned an empty dataset_id.")
    return ordered


def _requirements_count(requirements: Any, attribute: str) -> int:
    if isinstance(requirements, Mapping):
        value = requirements.get(attribute, 0)
    else:
        value = getattr(requirements, attribute, 0)
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def _requirements_point_instances(
    requirements: Any,
) -> tuple[tuple[str, str], ...]:
    raw = (
        requirements.get("point_instances", ())
        if isinstance(requirements, Mapping)
        else getattr(requirements, "point_instances", ())
    )
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in raw or ():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                "Placement parser returned an invalid point instance descriptor."
            )
        placement_id, dataset_id = (str(part).strip() for part in value)
        if not placement_id or not dataset_id or placement_id in seen:
            raise ValueError(
                "Placement parser returned an empty or duplicate point placement_id."
            )
        seen.add(placement_id)
        result.append((placement_id, dataset_id))
    return tuple(result)


def _requirements_line_instances(
    requirements: Any,
) -> tuple[tuple[str, str, int], ...]:
    raw = (
        requirements.get("line_instances", ())
        if isinstance(requirements, Mapping)
        else getattr(requirements, "line_instances", ())
    )
    result: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for value in raw or ():
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError(
                "Placement parser returned an invalid line instance descriptor."
            )
        line_id = str(value[0]).strip()
        dataset_id = str(value[1]).strip()
        try:
            segment_count = int(value[2])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Placement parser returned a non-integer line segment count."
            ) from exc
        if (
            not line_id
            or not dataset_id
            or line_id in seen
            or segment_count <= 0
        ):
            raise ValueError(
                "Placement parser returned an empty, duplicate, or invalid line_id."
            )
        seen.add(line_id)
        result.append((line_id, dataset_id, segment_count))
    return tuple(result)


class FeatureAssemblyFormModel:
    """Headless state, validation, discovery, and service dispatch."""

    def __init__(self, values: FeatureAssemblyValues | None = None) -> None:
        self.values = values if values is not None else FeatureAssemblyValues()
        self._point_dataset_ids: tuple[str, ...] = ()
        self._line_dataset_ids: tuple[str, ...] = ()
        self._point_requirements_csv = ""
        self._line_requirements_csv = ""
        self._point_requirements_fingerprint: _FileFingerprint | None = None
        self._line_requirements_fingerprint: _FileFingerprint | None = None
        self._point_placement_count = 0
        self._line_path_count = 0
        self._line_segment_count = 0
        self._point_instances: tuple[tuple[str, str], ...] = ()
        self._line_instances: tuple[tuple[str, str, int], ...] = ()
        self._prepared_plan_cache: _PreparedPlanCache | None = None

    @property
    def point_dataset_ids(self) -> tuple[str, ...]:
        return self._point_dataset_ids

    @property
    def line_dataset_ids(self) -> tuple[str, ...]:
        return self._line_dataset_ids

    @property
    def point_placement_count(self) -> int:
        return self._point_placement_count

    @property
    def line_path_count(self) -> int:
        return self._line_path_count

    @property
    def line_segment_count(self) -> int:
        return self._line_segment_count

    @property
    def point_instances(self) -> tuple[tuple[str, str], ...]:
        return self._point_instances

    @property
    def line_instances(self) -> tuple[tuple[str, str, int], ...]:
        return self._line_instances

    @property
    def prepared_plan(self) -> Any | None:
        """Return the cached reviewed plan for read-only GUI summaries."""

        cache = self._prepared_plan_cache
        return None if cache is None else cache.plan

    @property
    def enabled_line_segment_count(self) -> int:
        enabled = self.enabled_line_ids
        if enabled is None:
            return int(self._line_segment_count)
        selected = set(enabled)
        return sum(
            int(segment_count)
            for line_id, _dataset_id, segment_count in self._line_instances
            if line_id in selected
        )

    @property
    def enabled_point_placement_ids(self) -> tuple[str, ...] | None:
        if not self._point_instances:
            return None
        excluded = self.values.excluded_point_placement_ids
        return tuple(
            placement_id
            for placement_id, _dataset_id in self._point_instances
            if placement_id not in excluded
        )

    @property
    def enabled_line_ids(self) -> tuple[str, ...] | None:
        if not self._line_instances:
            return None
        excluded = self.values.excluded_line_ids
        return tuple(
            line_id
            for line_id, _dataset_id, _segments in self._line_instances
            if line_id not in excluded
        )

    def active_point_dataset_ids(self) -> tuple[str, ...]:
        enabled = self.enabled_point_placement_ids
        if enabled is None:
            return self._point_dataset_ids
        selected = set(enabled)
        return tuple(dict.fromkeys(
            dataset_id
            for placement_id, dataset_id in self._point_instances
            if placement_id in selected
        ))

    def active_line_dataset_ids(self) -> tuple[str, ...]:
        enabled = self.enabled_line_ids
        if enabled is None:
            return self._line_dataset_ids
        selected = set(enabled)
        return tuple(dict.fromkeys(
            dataset_id
            for line_id, dataset_id, _segments in self._line_instances
            if line_id in selected
        ))

    def set_feature_instance_enabled(
        self, kind: str, instance_id: str, enabled: bool
    ) -> None:
        normalized = str(kind).strip().lower()
        key = str(instance_id).strip()
        if normalized == "point":
            known = {value[0] for value in self._point_instances}
            excluded = self.values.excluded_point_placement_ids
        elif normalized == "line":
            known = {value[0] for value in self._line_instances}
            excluded = self.values.excluded_line_ids
        else:
            raise ValueError("Feature instance kind must be point or line.")
        if key not in known:
            raise KeyError(f"Unknown {normalized} feature instance {key!r}.")
        if enabled:
            excluded.discard(key)
        else:
            excluded.add(key)
        self.invalidate_prepared_plan()

    def set_excluded_feature_instances(
        self,
        *,
        point_ids: Iterable[str],
        line_ids: Iterable[str],
    ) -> None:
        points = {str(value).strip() for value in point_ids}
        lines = {str(value).strip() for value in line_ids}
        known_points = {value[0] for value in self._point_instances}
        known_lines = {value[0] for value in self._line_instances}
        unknown_points = sorted(points - known_points)
        unknown_lines = sorted(lines - known_lines)
        if "" in points or "" in lines or unknown_points or unknown_lines:
            raise ValueError(
                "Feature selection contains blank or stale IDs: "
                f"point={unknown_points}, line={unknown_lines}. Refresh the CSVs."
            )
        self.values.excluded_point_placement_ids = points
        self.values.excluded_line_ids = lines
        self.invalidate_prepared_plan()

    def feature_selection_summary(
        self, *, max_disabled_ids_per_kind: int | None = None
    ) -> str:
        """Describe exact membership, optionally shortening displayed ID lists.

        The default is deliberately lossless for logs, the clipboard action,
        and headless callers. The GUI passes a small limit only to its
        always-visible label so a large fastener trade study does not become a
        wall of text.
        """

        if max_disabled_ids_per_kind is not None:
            max_disabled_ids_per_kind = int(max_disabled_ids_per_kind)
            if max_disabled_ids_per_kind < 0:
                raise ValueError("Disabled-ID summary limit must be non-negative.")

        def disabled_text(kind: str, identifiers: Iterable[str]) -> str:
            ordered = sorted(str(value) for value in identifiers)
            if (
                max_disabled_ids_per_kind is None
                or len(ordered) <= max_disabled_ids_per_kind
            ):
                return f"{kind}=[" + ", ".join(ordered) + "]"
            visible = ordered[:max_disabled_ids_per_kind]
            omitted = len(ordered) - len(visible)
            prefix = ", ".join(visible)
            if prefix:
                prefix += ", "
            return (
                f"{kind}=[{prefix}… +{omitted} more]"
                " (use Copy full selection)"
            )

        point_total = len(self._point_instances)
        line_total = len(self._line_instances)
        point_enabled = self.enabled_point_placement_ids
        line_enabled = self.enabled_line_ids
        summary = (
            f"Enabled spatial features: {len(point_enabled or ())}/{point_total} "
            f"point placement(s), {len(line_enabled or ())}/{line_total} line "
            "path(s). Disabled features remain in the parsed configuration but "
            "are omitted from preview, validation, response loading, and build."
        )
        disabled_parts = []
        if self.values.excluded_point_placement_ids:
            disabled_parts.append(disabled_text(
                "point", self.values.excluded_point_placement_ids
            ))
        if self.values.excluded_line_ids:
            disabled_parts.append(disabled_text(
                "line", self.values.excluded_line_ids
            ))
        if disabled_parts:
            return summary + " Disabled IDs: " + "; ".join(disabled_parts) + "."
        if point_total or line_total:
            return summary + " All parsed spatial features are enabled."
        return summary + " Refresh a placement CSV to populate the hierarchy."

    def clear_feature_selection(self, kind: str | None = None) -> None:
        normalized = None if kind is None else str(kind).strip().lower()
        if normalized not in (None, "point", "line"):
            raise ValueError("Feature selection kind must be point, line, or None.")
        if normalized in (None, "point"):
            self.values.excluded_point_placement_ids.clear()
        if normalized in (None, "line"):
            self.values.excluded_line_ids.clear()
        self.invalidate_prepared_plan()

    def feature_selection_source_changed(self, kind: str, path: Any) -> bool:
        """Return whether ``path`` differs from the CSV that defined the IDs.

        Exclusions are a live trade-study choice tied to one parsed CSV. They
        survive an in-place rescan of that file, but must not silently migrate
        to another file merely because it reuses the same placement IDs.
        """

        normalized = str(kind).strip().lower()
        if normalized not in {"point", "line"}:
            raise ValueError("Feature selection kind must be point or line.")
        recorded = (
            self._point_requirements_fingerprint
            if normalized == "point"
            else self._line_requirements_fingerprint
        )
        if recorded is None:
            return False
        cleaned = _clean_path(path)
        if not cleaned:
            return True
        candidate = _path_key(
            _resolved_user_path(cleaned, base_dir=self.values.base_dir)
        )
        return candidate != recorded.resolved_path

    def _selected_csv_fingerprint(self, kind: str) -> _FileFingerprint | None:
        path = (
            self.values.point_locations_csv
            if kind == "point"
            else self.values.line_locations_csv
        )
        if not _clean_path(path):
            return None
        return _fingerprint_file(path, base_dir=self.values.base_dir)

    def requirements_are_current(self, kind: str) -> bool:
        """Return whether discovered IDs still describe the selected CSV bytes."""

        normalized = str(kind).strip().lower()
        if normalized not in {"point", "line"}:
            raise ValueError("Dataset requirement kind must be point or line.")
        ids = self._point_dataset_ids if normalized == "point" else self._line_dataset_ids
        recorded = (
            self._point_requirements_fingerprint
            if normalized == "point"
            else self._line_requirements_fingerprint
        )
        if not ids or recorded is None:
            return False
        try:
            return self._selected_csv_fingerprint(normalized) == recorded
        except OSError:
            return False

    def requirements_look_current(self, kind: str) -> bool:
        """Cheap UI hint using path/stat identity; validation still hashes bytes."""

        normalized = str(kind).strip().lower()
        if normalized not in {"point", "line"}:
            raise ValueError("Dataset requirement kind must be point or line.")
        ids = self._point_dataset_ids if normalized == "point" else self._line_dataset_ids
        recorded = (
            self._point_requirements_fingerprint
            if normalized == "point"
            else self._line_requirements_fingerprint
        )
        if not ids or recorded is None:
            return False
        path = (
            self.values.point_locations_csv
            if normalized == "point"
            else self.values.line_locations_csv
        )
        try:
            current = _fingerprint_file(
                path,
                base_dir=self.values.base_dir,
                include_hash=False,
            )
        except OSError:
            return False
        return (
            current.resolved_path == recorded.resolved_path
            and current.exists == recorded.exists
            and current.size == recorded.size
            and current.mtime_ns == recorded.mtime_ns
            and current.ctime_ns == recorded.ctime_ns
        )

    def invalidate_prepared_plan(self) -> None:
        """Release cached validated geometry after any semantic input edit."""

        self._prepared_plan_cache = None

    def update_dataset_requirements(self, requirements: Any) -> None:
        """Apply discovered IDs while preserving paths for surviving IDs."""

        discovery = requirements if isinstance(requirements, _DatasetDiscovery) else None
        payload = discovery.requirements if discovery is not None else requirements
        point_ids = _requirements_ids(payload, "point_dataset_ids")
        line_ids = _requirements_ids(payload, "line_dataset_ids")
        self._point_dataset_ids = point_ids
        self._line_dataset_ids = line_ids
        self._point_placement_count = _requirements_count(
            payload, "point_placement_count"
        )
        self._line_path_count = _requirements_count(payload, "line_path_count")
        self._line_segment_count = _requirements_count(
            payload, "line_segment_count"
        )
        self._point_instances = _requirements_point_instances(payload)
        self._line_instances = _requirements_line_instances(payload)
        unknown_point_datasets = sorted(
            {dataset_id for _placement_id, dataset_id in self._point_instances}
            - set(point_ids)
        )
        unknown_line_datasets = sorted(
            {
                dataset_id
                for _line_id, dataset_id, _segments in self._line_instances
            }
            - set(line_ids)
        )
        if unknown_point_datasets or unknown_line_datasets:
            raise ValueError(
                "Placement parser returned instance descriptors for unknown "
                f"dataset IDs: point={unknown_point_datasets}, "
                f"line={unknown_line_datasets}."
            )
        # Stable exclusions survive a re-scan only while the same explicit
        # CSV IDs survive. Newly parsed instances default enabled.
        self.values.excluded_point_placement_ids.intersection_update(
            placement_id for placement_id, _dataset_id in self._point_instances
        )
        self.values.excluded_line_ids.intersection_update(
            line_id for line_id, _dataset_id, _segments in self._line_instances
        )
        self._point_requirements_csv = _clean_path(
            self.values.point_locations_csv
        )
        self._line_requirements_csv = _clean_path(
            self.values.line_locations_csv
        )
        self._point_requirements_fingerprint = (
            discovery.point_fingerprint
            if discovery is not None
            else self._selected_csv_fingerprint("point")
        )
        self._line_requirements_fingerprint = (
            discovery.line_fingerprint
            if discovery is not None
            else self._selected_csv_fingerprint("line")
        )
        self.values.point_datasets = {
            dataset_id: _clean_path(self.values.point_datasets.get(dataset_id))
            for dataset_id in point_ids
        }
        self.values.line_datasets = {
            dataset_id: _clean_path(self.values.line_datasets.get(dataset_id))
            for dataset_id in line_ids
        }
        self.values.point_host_materials = {
            dataset_id: str(
                self.values.point_host_materials.get(dataset_id, "")
            ).strip()
            for dataset_id in point_ids
        }
        self.values.line_host_materials = {
            dataset_id: str(
                self.values.line_host_materials.get(dataset_id, "")
            ).strip()
            for dataset_id in line_ids
        }
        self.invalidate_prepared_plan()

    def invalidate_dataset_requirements(self, kind: str | None = None) -> None:
        """Discard IDs that no longer describe the selected/on-disk CSV."""

        normalized = None if kind is None else str(kind).strip().lower()
        if normalized not in (None, "point", "line"):
            raise ValueError("Dataset requirement kind must be point, line, or None.")
        if normalized in (None, "point"):
            self._point_dataset_ids = ()
            self._point_placement_count = 0
            self._point_instances = ()
            self._point_requirements_csv = ""
            self._point_requirements_fingerprint = None
            self.values.point_datasets = {}
            self.values.point_host_materials = {}
        if normalized in (None, "line"):
            self._line_dataset_ids = ()
            self._line_path_count = 0
            self._line_segment_count = 0
            self._line_instances = ()
            self._line_requirements_csv = ""
            self._line_requirements_fingerprint = None
            self.values.line_datasets = {}
            self.values.line_host_materials = {}
        self.invalidate_prepared_plan()

    def query_dataset_ids(self, service: Any) -> Any:
        """Validate CSVs without applying IDs; stale prior IDs are invalidated."""

        adapter = coerce_feature_workflow(service)
        point_csv = _clean_path(self.values.point_locations_csv)
        line_csv = _clean_path(self.values.line_locations_csv)
        if not point_csv and not line_csv:
            raise ValueError("Select a point or line placement CSV first.")
        point_before = self._selected_csv_fingerprint("point")
        line_before = self._selected_csv_fingerprint("line")
        requirements = adapter.discover(
            point_locations_csv=point_csv or None,
            line_locations_csv=line_csv or None,
            base_dir=self.values.base_dir,
        )
        point_after = self._selected_csv_fingerprint("point")
        line_after = self._selected_csv_fingerprint("line")
        if point_before != point_after or line_before != line_after:
            if point_before != point_after:
                self.invalidate_dataset_requirements("point")
            if line_before != line_after:
                self.invalidate_dataset_requirements("line")
            raise RuntimeError(
                "A placement CSV changed while it was being read. Save it, then refresh."
            )
        return _DatasetDiscovery(
            requirements=requirements,
            point_fingerprint=point_after,
            line_fingerprint=line_after,
        )

    def discover_dataset_ids(self, service: Any) -> Any:
        """Ask the authoritative parser to validate CSVs and apply their IDs."""

        discovery = self.query_dataset_ids(service)
        self.update_dataset_requirements(discovery)
        return discovery.requirements

    def set_point_dataset(self, dataset_id: str, path: str) -> None:
        self._set_dataset("point", dataset_id, path)

    def set_line_dataset(self, dataset_id: str, path: str) -> None:
        self._set_dataset("line", dataset_id, path)

    def set_point_host_material(self, dataset_id: str, material: str) -> None:
        self._set_host_material("point", dataset_id, material)

    def set_line_host_material(self, dataset_id: str, material: str) -> None:
        self._set_host_material("line", dataset_id, material)

    def _set_dataset(self, kind: str, dataset_id: str, path: str) -> None:
        key = str(dataset_id).strip()
        ids = self._point_dataset_ids if kind == "point" else self._line_dataset_ids
        if key not in ids:
            raise KeyError(f"Unknown {kind} dataset_id {key!r}.")
        mapping = (
            self.values.point_datasets
            if kind == "point"
            else self.values.line_datasets
        )
        mapping[key] = _clean_path(path)
        self.invalidate_prepared_plan()

    def _set_host_material(
        self, kind: str, dataset_id: str, material: str
    ) -> None:
        key = str(dataset_id).strip()
        ids = self._point_dataset_ids if kind == "point" else self._line_dataset_ids
        if key not in ids:
            raise KeyError(f"Unknown {kind} dataset_id {key!r}.")
        mapping = (
            self.values.point_host_materials
            if kind == "point"
            else self.values.line_host_materials
        )
        mapping[key] = str(material).strip()
        self.invalidate_prepared_plan()

    def missing_dataset_mappings(self) -> tuple[str, ...]:
        missing = [
            f"point:{dataset_id}"
            for dataset_id in self.active_point_dataset_ids()
            if not _clean_path(self.values.point_datasets.get(dataset_id))
        ]
        missing.extend(
            f"line:{dataset_id}"
            for dataset_id in self.active_line_dataset_ids()
            if not _clean_path(self.values.line_datasets.get(dataset_id))
        )
        return tuple(missing)

    def effective_host_materials(self) -> dict[str, str]:
        """Resolve per-response host IDs, using the global value as a default."""

        default = _normalize_host_material(self.values.expected_host_material)
        effective: dict[str, str] = {}
        for kind, dataset_ids, overrides in (
            (
                "point",
                self.active_point_dataset_ids(),
                self.values.point_host_materials,
            ),
            (
                "line",
                self.active_line_dataset_ids(),
                self.values.line_host_materials,
            ),
        ):
            for dataset_id in dataset_ids:
                override = _normalize_host_material(overrides.get(dataset_id, ""))
                # Preserve one stable spelling when a per-response value is
                # equivalent to the global default after the backend's
                # whitespace/case normalization.
                if (
                    override
                    and default
                    and override.casefold() == default.casefold()
                ):
                    override = default
                effective[f"{kind}:{dataset_id}"] = override or default
        return effective

    def missing_host_material_mappings(self) -> tuple[str, ...]:
        effective = self.effective_host_materials()
        missing: list[str] = []
        missing.extend(
            f"point:{dataset_id}"
            for dataset_id in self.active_point_dataset_ids()
            if not effective.get(f"point:{dataset_id}")
        )
        missing.extend(
            f"line:{dataset_id}"
            for dataset_id in self.active_line_dataset_ids()
            if not effective.get(f"line:{dataset_id}")
        )
        return tuple(missing)

    def _validate_output_target(self) -> None:
        values = self.values
        output = _normalized_grim_output_path(
            values.output_grim, base_dir=values.base_dir
        )
        output_targets = (
            ("assembled response", output),
            (
                "feature-only sibling",
                _features_only_grim_output_path(
                    values.output_grim, base_dir=values.base_dir
                ),
            ),
        )
        protected: list[tuple[str, str]] = [("clean-body response", values.base_grim)]
        protected.extend(
            (
                ("surface mesh", values.surface_mesh),
                ("point placement CSV", values.point_locations_csv),
                ("line placement CSV", values.line_locations_csv),
            )
        )
        protected.extend(
            (f"point response {dataset_id!r}", path)
            for dataset_id, path in values.point_datasets.items()
        )
        protected.extend(
            (f"line response {dataset_id!r}", path)
            for dataset_id, path in values.line_datasets.items()
        )
        for label, path in protected:
            if not _clean_path(path):
                continue
            source = _resolved_user_path(path, base_dir=values.base_dir)
            for output_label, target in output_targets:
                if _paths_alias(target, source):
                    raise ValueError(
                        f"The {output_label} must not overwrite the {label}. "
                        "Choose a new file name."
                    )

    def validate(self) -> None:
        values = self.values
        if not _clean_path(values.base_grim):
            raise ValueError("Select the clean-body/base GRIM file.")
        if not _clean_path(values.output_grim):
            raise ValueError("Choose an output GRIM file.")

        point_csv = _clean_path(values.point_locations_csv)
        line_csv = _clean_path(values.line_locations_csv)
        if not point_csv and not line_csv:
            raise ValueError("Select a point or line placement CSV.")
        if point_csv and not self.requirements_are_current("point"):
            raise ValueError(
                "The point CSV changed after its last successful scan. "
                "Re-scan it before continuing."
            )
        if line_csv and not self.requirements_are_current("line"):
            raise ValueError(
                "The line CSV changed after its last successful scan. "
                "Re-scan it before continuing."
            )
        if point_csv and not self._point_dataset_ids:
            raise ValueError(
                "Point dataset IDs have not been discovered. Re-scan the "
                "point CSV before continuing."
            )
        if line_csv and not self._line_dataset_ids:
            raise ValueError(
                "Line dataset IDs have not been discovered. Re-scan the line "
                "CSV before continuing."
            )
        if (
            (self._point_instances or self._line_instances)
            and not (self.enabled_point_placement_ids or self.enabled_line_ids)
        ):
            raise ValueError(
                "No enabled spatial features remain. Enable at least one point "
                "placement or line path before validating or building."
            )
        missing = self.missing_dataset_mappings()
        if missing:
            raise ValueError(
                "Choose an OPN-FRD GRIM response for: " + ", ".join(missing)
            )
        missing_hosts = self.missing_host_material_mappings()
        if (
            values.require_feature_manifests
            and not values.allow_legacy_base_metadata
            and missing_hosts
        ):
            raise ValueError(
                "Production validation requires a host material/coating ID for: "
                + ", ".join(missing_hosts)
                + ". Enter a per-response ID or the global default."
            )
        if values.shadow and not _clean_path(values.surface_mesh):
            raise ValueError(
                "Geometric shadowing requires an STL or facet surface mesh."
            )
        supported_units = {value for _, value in UNIT_CHOICES}
        if point_csv or line_csv:
            if not str(values.coordinate_units).strip():
                raise ValueError(
                    "Choose the coordinate units used by the selected placement CSV(s)."
                )
            if values.coordinate_units not in supported_units:
                raise ValueError(
                    f"Unsupported coordinate units: {values.coordinate_units!r}."
                )
        if _clean_path(values.surface_mesh):
            if not str(values.surface_units).strip():
                raise ValueError(
                    "Choose the physical units of the selected surface mesh."
                )
            if values.surface_units not in supported_units:
                raise ValueError(
                    f"Unsupported surface units: {values.surface_units!r}."
                )
        skin = _require_finite_nonnegative(
            values.skin_tol_m, "Skin distance tolerance"
        )
        phase = _require_finite_nonnegative(
            values.skin_phase_tol_deg, "Skin phase tolerance"
        )
        if skin > 0.1:
            raise ValueError("Skin distance tolerance must not exceed 100 mm.")
        if not 0.0 < phase <= 90.0:
            raise ValueError(
                "Skin phase tolerance must be above 0 and at most 90 degrees."
            )
        normal = _require_finite_nonnegative(
            values.normal_tol_deg, "Normal tolerance"
        )
        if normal >= 90.0:
            raise ValueError(
                "Normal tolerance must be less than 90 degrees so inward-facing "
                "feature frames cannot pass validation."
            )
        if values.shadow_bias_m is not None:
            _require_finite_nonnegative(values.shadow_bias_m, "Shadow bias")
        if values.require_body_mesh_certification and (
            values.allow_legacy_base_metadata
            or not values.require_feature_manifests
        ):
            raise ValueError(
                "Certified-body Production validation also requires strict "
                "base metadata and certified feature manifests. Choose the "
                "Production profile again or use the explicit External/HPC "
                "profile."
            )
        self._validate_output_target()

    def build_request(self, service: Any) -> Any:
        """Create the backend request only after local completeness checks."""

        adapter = coerce_feature_workflow(service)
        self.validate()
        values = self.values
        return adapter.request_factory(
            base_grim=_clean_path(values.base_grim),
            output_grim=_clean_path(values.output_grim),
            coordinate_units=values.coordinate_units,
            surface_mesh=_clean_path(values.surface_mesh) or None,
            surface_units=values.surface_units,
            flip_surface_normals=bool(values.flip_surface_normals),
            shadow=bool(values.shadow),
            shadow_bias_m=(
                None
                if values.shadow_bias_m is None
                else float(values.shadow_bias_m)
            ),
            point_locations_csv=_clean_path(values.point_locations_csv) or None,
            point_datasets={
                key: _clean_path(values.point_datasets[key])
                for key in self.active_point_dataset_ids()
            },
            enabled_point_placement_ids=self.enabled_point_placement_ids,
            line_locations_csv=_clean_path(values.line_locations_csv) or None,
            line_datasets={
                key: _clean_path(values.line_datasets[key])
                for key in self.active_line_dataset_ids()
            },
            enabled_line_ids=self.enabled_line_ids,
            skin_tol_m=float(values.skin_tol_m),
            skin_phase_tol_deg=float(values.skin_phase_tol_deg),
            normal_tol_deg=float(values.normal_tol_deg),
            allow_legacy_base_metadata=bool(
                values.allow_legacy_base_metadata
            ),
            require_feature_manifests=bool(
                values.require_feature_manifests
            ),
            require_body_mesh_certification=bool(
                values.require_body_mesh_certification
            ),
            expected_host_material=(
                str(values.expected_host_material).strip() or None
            ),
            expected_host_materials=self.effective_host_materials(),
            base_dir=values.base_dir,
        )

    def _semantic_signature(self) -> tuple[Any, ...]:
        values = self.values

        def path_signature(path: Any, *, output: bool = False) -> str:
            if not _clean_path(path):
                return ""
            resolved = (
                _normalized_grim_output_path(path, base_dir=values.base_dir)
                if output
                else _resolved_user_path(path, base_dir=values.base_dir)
            )
            return _path_key(resolved)

        return (
            path_signature(values.base_grim),
            path_signature(values.output_grim, output=True),
            values.coordinate_units,
            path_signature(values.surface_mesh),
            values.surface_units,
            bool(values.flip_surface_normals),
            bool(values.shadow),
            values.shadow_bias_m,
            path_signature(values.point_locations_csv),
            self.enabled_point_placement_ids,
            tuple(
                (dataset_id, path_signature(values.point_datasets.get(dataset_id)))
                for dataset_id in self.active_point_dataset_ids()
            ),
            path_signature(values.line_locations_csv),
            self.enabled_line_ids,
            tuple(
                (dataset_id, path_signature(values.line_datasets.get(dataset_id)))
                for dataset_id in self.active_line_dataset_ids()
            ),
            float(values.skin_tol_m),
            float(values.skin_phase_tol_deg),
            float(values.normal_tol_deg),
            bool(values.allow_legacy_base_metadata),
            bool(values.require_feature_manifests),
            bool(values.require_body_mesh_certification),
            str(values.expected_host_material).strip(),
            tuple(
                (dataset_id, str(values.point_host_materials.get(dataset_id, "")).strip())
                for dataset_id in self.active_point_dataset_ids()
            ),
            tuple(
                (dataset_id, str(values.line_host_materials.get(dataset_id, "")).strip())
                for dataset_id in self.active_line_dataset_ids()
            ),
            (
                _path_key(Path(values.base_dir).expanduser().resolve())
                if _clean_path(values.base_dir)
                else ""
            ),
        )

    def _source_fingerprints(
        self,
    ) -> tuple[tuple[str, _FileFingerprint], ...]:
        values = self.values
        # The prepared backend plan already hashes exact source bytes and
        # verifies them again immediately before atomic publication. Re-reading
        # multi-GB GRIM/STL inputs in the GUI cache added no correctness and
        # could dominate Validate -> Build latency; stat identity is sufficient
        # to decide whether to reuse the plan optimistically.
        sources: list[tuple[str, str, bool]] = [
            ("base", _clean_path(values.base_grim), False),
            ("surface", _clean_path(values.surface_mesh), False),
            ("point CSV", _clean_path(values.point_locations_csv), False),
            ("line CSV", _clean_path(values.line_locations_csv), False),
        ]
        sources.extend(
            (
                f"point:{dataset_id}",
                _clean_path(values.point_datasets.get(dataset_id)),
                False,
            )
            for dataset_id in self.active_point_dataset_ids()
        )
        sources.extend(
            (
                f"line:{dataset_id}",
                _clean_path(values.line_datasets.get(dataset_id)),
                False,
            )
            for dataset_id in self.active_line_dataset_ids()
        )
        return tuple(
            (
                label,
                _fingerprint_file(
                    path,
                    base_dir=values.base_dir,
                    include_hash=include_hash,
                ),
            )
            for label, path, include_hash in sources
            if path
        )

    @staticmethod
    def _service_key(adapter: FeatureWorkflowAdapter) -> tuple[Any, ...]:
        return (
            _callable_key(adapter.request_factory),
            _callable_key(adapter.prepare),
            _callable_key(adapter.execute),
        )

    def _prepare_and_cache(
        self,
        adapter: FeatureWorkflowAdapter,
        request: Any,
        *,
        before: tuple[tuple[str, _FileFingerprint], ...] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Any:
        # ``assemble`` may already have captured this exact full-content
        # snapshot while deciding whether a validated plan can be reused.  Pass
        # it through so a cache miss does not immediately reread every large
        # GRIM/STL response before the authoritative prepare operation.
        # A user-requested re-validation is a new review attempt: the old plan
        # must not remain publishable if this attempt fails or is cancelled.
        self.invalidate_prepared_plan()
        semantic_before = self._semantic_signature()
        if before is None:
            before = self._source_fingerprints()
        if cancel_check is not None and cancel_check():
            self.invalidate_prepared_plan()
            raise InterruptedError(
                "Placement validation cancelled; no reviewed plan was retained."
            )
        try:
            if _callable_accepts_runtime_hooks(adapter.prepare):
                plan = adapter.prepare(
                    request,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                )
            else:
                plan = adapter.prepare(request)
        except InterruptedError:
            self.invalidate_prepared_plan()
            raise
        if cancel_check is not None and cancel_check():
            self.invalidate_prepared_plan()
            raise InterruptedError(
                "Placement validation cancelled; no reviewed plan was retained."
            )
        after = self._source_fingerprints()
        semantic_after = self._semantic_signature()
        if semantic_before != semantic_after:
            self.invalidate_prepared_plan()
            raise RuntimeError(
                "The spatial feature configuration changed during validation. "
                "Review the enabled features and validate again."
            )
        if before != after:
            before_by_label = dict(before)
            after_by_label = dict(after)
            if before_by_label.get("point CSV") != after_by_label.get("point CSV"):
                self.invalidate_dataset_requirements("point")
            if before_by_label.get("line CSV") != after_by_label.get("line CSV"):
                self.invalidate_dataset_requirements("line")
            raise RuntimeError(
                "An Assembly input changed during validation. Save the input, "
                "refresh the placement CSVs, and validate again."
            )
        # Close the final cooperative-cancel window after potentially lengthy
        # source re-fingerprinting and immediately before making the reviewed
        # plan publishable from the cache.
        if cancel_check is not None and cancel_check():
            self.invalidate_prepared_plan()
            raise InterruptedError(
                "Placement validation cancelled; no reviewed plan was retained."
            )
        self._prepared_plan_cache = _PreparedPlanCache(
            plan=plan,
            semantic_signature=semantic_before,
            source_fingerprints=after,
            service_key=self._service_key(adapter),
        )
        return plan

    def prepare_preview(
        self,
        service: Any,
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Any:
        adapter = coerce_feature_workflow(service)
        request = self.build_request(adapter)
        return self._prepare_and_cache(
            adapter,
            request,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )

    def validated_plan_is_current(
        self,
        service: Any,
        *,
        verify_sources: bool = False,
    ) -> bool:
        """Return whether the cached authoritative validation matches live inputs."""

        adapter = coerce_feature_workflow(service)
        cached = self._prepared_plan_cache
        if cached is None:
            return False
        if cached.semantic_signature != self._semantic_signature():
            return False
        if cached.service_key != self._service_key(adapter):
            return False
        if verify_sources and cached.source_fingerprints != self._source_fingerprints():
            return False
        return True

    def prepare_input_preview(self, service: Any) -> Any:
        """Preview selected geometry/locations before response mapping.

        This is a deliberately non-physical staging preview.  The optional
        backend callable owns CSV parsing and geometry loading; the GUI only
        passes paths and units through unchanged.
        """

        adapter = coerce_feature_workflow(service)
        if not callable(adapter.preview_inputs):
            raise RuntimeError(
                "This GHOST backend does not support staged input preview. "
                "Use Validate placements after mapping responses."
            )
        values = self.values
        base_grim = _clean_path(values.base_grim)
        surface_mesh = _clean_path(values.surface_mesh)
        point_csv = _clean_path(values.point_locations_csv)
        line_csv = _clean_path(values.line_locations_csv)
        if not any((base_grim, surface_mesh, point_csv, line_csv)):
            raise ValueError(
                "Choose a clean-body GRIM, body mesh, or placement CSV to preview."
            )
        supported_units = {value for _, value in UNIT_CHOICES}
        if point_csv or line_csv:
            if not str(values.coordinate_units).strip():
                raise ValueError(
                    "Choose the coordinate units used by the selected placement CSV(s)."
                )
            if values.coordinate_units not in supported_units:
                raise ValueError(
                    f"Unsupported coordinate units: {values.coordinate_units!r}."
                )
        if _clean_path(values.surface_mesh):
            if not str(values.surface_units).strip():
                raise ValueError(
                    "Choose the physical units of the selected surface mesh."
                )
            if values.surface_units not in supported_units:
                raise ValueError(
                    f"Unsupported surface units: {values.surface_units!r}."
                )
        source_values = (
            ("base", base_grim, False),
            ("surface", surface_mesh, False),
            ("point CSV", point_csv, True),
            ("line CSV", line_csv, True),
        )
        enabled_point_ids = self.enabled_point_placement_ids
        enabled_line_ids = self.enabled_line_ids
        def snapshot() -> tuple[tuple[str, _FileFingerprint], ...]:
            return tuple(
                (
                    label,
                    _fingerprint_file(
                        path,
                        base_dir=values.base_dir,
                        include_hash=include_hash,
                    ),
                )
                for label, path, include_hash in source_values
                if path
            )

        before = snapshot()
        preview = adapter.preview_inputs(
            base_grim=base_grim or None,
            surface_mesh=surface_mesh or None,
            coordinate_units=values.coordinate_units,
            surface_units=values.surface_units,
            point_locations_csv=point_csv or None,
            line_locations_csv=line_csv or None,
            enabled_point_placement_ids=enabled_point_ids,
            enabled_line_ids=enabled_line_ids,
            base_dir=values.base_dir,
        )
        after = snapshot()
        if (
            enabled_point_ids != self.enabled_point_placement_ids
            or enabled_line_ids != self.enabled_line_ids
        ):
            raise RuntimeError(
                "The spatial feature configuration changed while the input "
                "preview was loading. Preview again to use the current selection."
            )
        if before != after:
            before_by_label = dict(before)
            after_by_label = dict(after)
            if before_by_label.get("point CSV") != after_by_label.get("point CSV"):
                self.invalidate_dataset_requirements("point")
            if before_by_label.get("line CSV") != after_by_label.get("line CSV"):
                self.invalidate_dataset_requirements("line")
            raise RuntimeError(
                "An Assembly input changed while the input preview was loading. "
                "Save it, then preview again."
            )
        requirements = getattr(preview, "dataset_requirements", None)
        after_by_label = dict(after)
        discovery = (
            None
            if requirements is None
            else _DatasetDiscovery(
                requirements=requirements,
                point_fingerprint=after_by_label.get("point CSV"),
                line_fingerprint=after_by_label.get("line CSV"),
            )
        )
        return _VerifiedInputPreview(preview=preview, discovery=discovery)

    def assemble(
        self,
        service: Any,
        *,
        acknowledged_plan_sha256: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> FeatureBuildDispatch:
        adapter = coerce_feature_workflow(service)
        request = self.build_request(adapter)
        signature = self._semantic_signature()
        service_key = self._service_key(adapter)
        cache = self._prepared_plan_cache
        reused = False
        if (
            cache is not None
            and cache.semantic_signature == signature
            and cache.service_key == service_key
        ):
            fingerprints = self._source_fingerprints()
            if cache.source_fingerprints == fingerprints:
                plan = cache.plan
                reused = True
            else:
                plan = self._prepare_and_cache(
                    adapter,
                    request,
                    before=fingerprints,
                )
        else:
            plan = self._prepare_and_cache(adapter, request)
        # Recheck filesystem aliases immediately before publication. A link may
        # have appeared after the initial request validation while a cached plan
        # was being accepted.
        self._validate_output_target()
        if cancel_check is not None and cancel_check():
            raise InterruptedError(
                "Feature assembly cancelled; existing output kept."
            )
        plan_features_path = getattr(plan, "features_only_output_path", None)
        features_path = (
            Path(plan_features_path)
            if plan_features_path is not None
            else _features_only_grim_output_path(
                self.values.output_grim,
                base_dir=self.values.base_dir,
            )
        )
        features_before = _publication_snapshot(features_path)
        execute_kwargs: dict[str, Any] = {}
        if _callable_accepts_runtime_hooks(adapter.execute):
            execute_kwargs.update(
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
        if (
            acknowledged_plan_sha256 is not None
            and _callable_accepts_keyword(
                adapter.execute, "acknowledged_plan_sha256"
            )
        ):
            execute_kwargs["acknowledged_plan_sha256"] = (
                acknowledged_plan_sha256
            )
        if execute_kwargs:
            output = adapter.execute(plan, **execute_kwargs)
        else:
            # Compatible injected/test services may predate runtime hooks. The
            # authoritative bundled backend accepts them; legacy services still
            # execute, but cannot be interrupted mid-call.
            output = adapter.execute(plan)
        return FeatureBuildDispatch(
            plan=plan,
            output_path=str(output),
            reused_validated_plan=reused,
            features_only_output_path=str(features_path),
            features_only_output_published=_published_during_execution(
                features_path, features_before
            ),
        )

    def assemble_validated(
        self,
        service: Any,
        *,
        acknowledged_plan_sha256: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> FeatureBuildDispatch:
        """Publish only the exact, unchanged plan produced by ``prepare_preview``."""

        adapter = coerce_feature_workflow(service)
        # Preserve all local completeness/alias checks, but never silently
        # prepare a replacement plan after the operator's review gate.
        self.build_request(adapter)
        if not self.validated_plan_is_current(adapter, verify_sources=True):
            self.invalidate_prepared_plan()
            raise RuntimeError(
                "Assembly inputs changed or have not been validated. Run Validate "
                "placements, review the current QA result, then assemble again."
            )
        cache = self._prepared_plan_cache
        if cache is None:  # Defensive: covered by validated_plan_is_current.
            raise RuntimeError("No current validated Assembly plan is available.")
        self._validate_output_target()
        if cancel_check is not None and cancel_check():
            raise InterruptedError(
                "Feature assembly cancelled; existing output kept."
            )
        plan_features_path = getattr(
            cache.plan, "features_only_output_path", None
        )
        features_path = (
            Path(plan_features_path)
            if plan_features_path is not None
            else _features_only_grim_output_path(
                self.values.output_grim,
                base_dir=self.values.base_dir,
            )
        )
        features_before = _publication_snapshot(features_path)
        execute_kwargs: dict[str, Any] = {}
        if _callable_accepts_runtime_hooks(adapter.execute):
            execute_kwargs.update(
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
        if (
            acknowledged_plan_sha256 is not None
            and _callable_accepts_keyword(
                adapter.execute, "acknowledged_plan_sha256"
            )
        ):
            execute_kwargs["acknowledged_plan_sha256"] = (
                acknowledged_plan_sha256
            )
        if execute_kwargs:
            output = adapter.execute(cache.plan, **execute_kwargs)
        else:
            output = adapter.execute(cache.plan)
        return FeatureBuildDispatch(
            plan=cache.plan,
            output_path=str(output),
            reused_validated_plan=True,
            features_only_output_path=str(features_path),
            features_only_output_published=_published_during_execution(
                features_path, features_before
            ),
        )


_GUI_IMPORT_ERROR: Exception | None = None
try:  # Keep the model importable on headless/minimal installations.
    from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMenu,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QToolButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment-specific
    _GUI_IMPORT_ERROR = exc


GUI_AVAILABLE = _GUI_IMPORT_ERROR is None


if GUI_AVAILABLE:

    class _OperationWorker(QObject):
        succeeded = Signal(object)
        failed = Signal(str)
        cancelled = Signal(str)
        progress = Signal(int, str)

        def __init__(
            self,
            operation: Callable[..., Any],
            *,
            cooperative: bool = False,
        ) -> None:
            super().__init__()
            self._operation = operation
            self._cooperative = bool(cooperative)
            self._cancel_event = threading.Event()
            self._last_percent = -1

        def request_cancel(self) -> None:
            """Thread-safe direct call from the GUI thread."""

            self._cancel_event.set()

        def is_cancelled(self) -> bool:
            return self._cancel_event.is_set()

        def report_progress(self, done: int, total: int, message: str) -> None:
            count = max(1, int(total))
            percent = max(0, min(100, int(round(100.0 * int(done) / count))))
            # Avoid flooding the Qt event queue on dense direction grids.
            if percent != self._last_percent or percent in (0, 100):
                self._last_percent = percent
                self.progress.emit(percent, str(message))

        @Slot()
        def run(self) -> None:
            try:
                result = (
                    self._operation(self.is_cancelled, self.report_progress)
                    if self._cooperative
                    else self._operation()
                )
            except InterruptedError as exc:
                self.cancelled.emit(
                    str(exc) or "Feature assembly cancelled; existing output kept."
                )
            except Exception as exc:  # The UI reports authoritative validation errors.
                self.failed.emit(str(exc) or type(exc).__name__)
            else:
                self.succeeded.emit(result)


    class _DisclosureSection(QWidget):
        """Small local disclosure; avoids importing the shell and a Qt cycle."""

        def __init__(
            self,
            title: str,
            parent: QWidget | None = None,
            *,
            expanded: bool = False,
        ) -> None:
            super().__init__(parent)
            self._title = str(title)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self.header = QToolButton(self)
            self.header.setObjectName("sectionHeader")
            self.header.setCheckable(True)
            self.header.setChecked(bool(expanded))
            self.header.setCursor(Qt.CursorShape.PointingHandCursor)
            self.header.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self.body = QWidget(self)
            self.body.setObjectName("sectionBody")
            self.body_layout = QVBoxLayout(self.body)
            self.body_layout.setContentsMargins(8, 8, 8, 8)
            self.body_layout.setSpacing(6)
            layout.addWidget(self.header)
            layout.addWidget(self.body)
            self.header.toggled.connect(self._sync)
            self._sync(bool(expanded))

        def _sync(self, expanded: bool) -> None:
            self.header.setText(("▾  " if expanded else "▸  ") + self._title)
            self.body.setVisible(bool(expanded))

        def addWidget(self, widget: QWidget, stretch: int = 0) -> None:
            self.body_layout.addWidget(widget, stretch)

        def addLayout(self, layout: Any, stretch: int = 0) -> None:
            self.body_layout.addLayout(layout, stretch)


    class _LoadedDatasetButton(QPushButton):
        """Menu button that exposes only backend-usable loaded artifacts."""

        path_selected = Signal(str)
        notice = Signal(str)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__("Use loaded…", parent)
            self.setAutoDefault(False)
            self._catalog: tuple[LoadedDatasetEntry, ...] = ()
            self._menu = QMenu(self)
            self.setMenu(self._menu)
            self._menu.aboutToShow.connect(self._announce_constraints)
            self.set_catalog(())

        def catalog_menu(self) -> QMenu:
            """Return the owned menu for focused UI tests and shell tooling."""

            return self._menu

        def set_catalog(
            self, entries: tuple[LoadedDatasetEntry, ...]
        ) -> None:
            self._catalog = tuple(entries)
            self._menu.clear()
            usable = [entry for entry in self._catalog if entry.usable_path]
            unavailable = [entry for entry in self._catalog if not entry.usable_path]

            if usable:
                for entry in usable:
                    file_name = Path(entry.usable_path).name
                    action = self._menu.addAction(f"{entry.name} — {file_name}")
                    action.setData(entry.dataset_id)
                    action.setToolTip(entry.usable_path)
                    action.setStatusTip(entry.usable_path)
                    action.triggered.connect(
                        lambda _checked=False, path=entry.usable_path: (
                            self.path_selected.emit(path)
                        )
                    )
            else:
                action = self._menu.addAction("No saved .grim datasets available")
                action.setEnabled(False)

            if unavailable:
                self._menu.addSeparator()
                heading = self._menu.addAction(
                    "Save unsaved derived datasets first"
                )
                heading.setEnabled(False)
                for entry in unavailable:
                    reason = entry.unavailable_reason
                    action = self._menu.addAction(f"{entry.name} — {reason}")
                    action.setData(entry.dataset_id)
                    action.setToolTip(_clean_path(entry.path) or reason)
                    action.setEnabled(False)

            usable_count = len(usable)
            unavailable_count = len(unavailable)
            if usable_count:
                tooltip = (
                    f"Choose one of {usable_count} loaded, saved .grim "
                    "dataset(s)."
                )
                if unavailable_count:
                    tooltip += (
                        f" {unavailable_count} unsaved or unavailable "
                        "dataset(s) are disabled; save unsaved derived "
                        "datasets first."
                    )
            else:
                tooltip = (
                    "No usable saved .grim dataset is loaded. Save unsaved "
                    "derived datasets first, or use Browse…."
                )
            self.setToolTip(tooltip)

        def _announce_constraints(self) -> None:
            usable_count = sum(bool(entry.usable_path) for entry in self._catalog)
            unavailable_count = len(self._catalog) - usable_count
            if unavailable_count:
                self.notice.emit(
                    "Assembly requires an existing .grim file. Save unsaved "
                    "derived datasets first; unavailable entries are disabled."
                )
            elif not usable_count:
                self.notice.emit(
                    "No saved .grim dataset is currently loaded. Save the "
                    "required dataset first, or use Browse…."
                )


    class _PathPicker(QWidget):
        editing_finished = Signal()
        catalog_notice = Signal(str)

        def __init__(
            self,
            *,
            caption: str,
            file_filter: str,
            save: bool = False,
            allow_loaded_dataset: bool = False,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.caption = caption
            self.file_filter = file_filter
            self.save = bool(save)
            self.edit = QLineEdit(self)
            self.loaded_button: _LoadedDatasetButton | None = None
            self.button = QPushButton("Browse…", self)
            self.button.setAutoDefault(False)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.edit, 1)
            if allow_loaded_dataset:
                self.loaded_button = _LoadedDatasetButton(self)
                self.loaded_button.path_selected.connect(self._use_loaded_path)
                self.loaded_button.notice.connect(self.catalog_notice.emit)
                layout.addWidget(self.loaded_button)
            layout.addWidget(self.button)
            self.button.clicked.connect(self._browse)
            self.edit.editingFinished.connect(self.editing_finished.emit)

        def path(self) -> str:
            return self.edit.text().strip()

        def set_path(self, path: str) -> None:
            self.edit.setText(_clean_path(path))

        def set_loaded_dataset_catalog(
            self, entries: tuple[LoadedDatasetEntry, ...]
        ) -> None:
            if self.loaded_button is not None:
                self.loaded_button.set_catalog(entries)

        @Slot(str)
        def _use_loaded_path(self, path: str) -> None:
            self.set_path(path)
            self.editing_finished.emit()

        def _browse(self) -> None:
            start = self.path()
            if self.save:
                path, _ = QFileDialog.getSaveFileName(
                    self, self.caption, start, self.file_filter
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, self.caption, start, self.file_filter
                )
            if not path:
                return
            if self.save and Path(path).suffix == "":
                path += ".grim"
            self.set_path(path)
            self.editing_finished.emit()


    class _SurfaceBindingDialog(QDialog):
        """Small attestation form for one exact external body/mesh pair."""

        def __init__(
            self,
            parent: QWidget,
            *,
            geometry_id: str = "",
            attestation_case_id: str = "",
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Bind clean-body solve to surface mesh")
            self.setModal(True)
            layout = QVBoxLayout(self)
            explanation = QLabel(
                "Create the canonical reviewed registration record for the exact "
                "clean-body GRIM, selected mesh bytes, and selected mesh units.",
                self,
            )
            explanation.setWordWrap(True)
            layout.addWidget(explanation)
            form = QFormLayout()
            self.geometry_id_edit = QLineEdit(self)
            self.geometry_id_edit.setPlaceholderText("Example: vehicle-door-mesh-r7")
            self.geometry_id_edit.setText(str(geometry_id))
            self.case_id_edit = QLineEdit(self)
            self.case_id_edit.setPlaceholderText("Example: solver-registration-042")
            self.case_id_edit.setText(str(attestation_case_id))
            form.addRow("Team geometry revision ID:", self.geometry_id_edit)
            form.addRow("Reviewed registration / case ID:", self.case_id_edit)
            layout.addLayout(form)
            attestation_text = QLabel(
                "Required review: a responsible team member confirmed that this "
                "exact mesh, selected units, CAD axes, and origin match the exact "
                "clean-body solve.",
                self,
            )
            attestation_text.setWordWrap(True)
            layout.addWidget(attestation_text)
            self.attestation = QCheckBox(
                "I attest that the required registration review is complete.", self
            )
            layout.addWidget(self.attestation)
            limitation = QLabel(
                "This records the review and exact file identities; it does not "
                "independently prove electromagnetic or solve-to-CAD correctness.",
                self,
            )
            limitation.setObjectName("featureHint")
            limitation.setWordWrap(True)
            layout.addWidget(limitation)
            self.buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel,
                parent=self,
            )
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
                "Create reviewed binding"
            )
            self.buttons.accepted.connect(self.accept)
            self.buttons.rejected.connect(self.reject)
            layout.addWidget(self.buttons)
            self.geometry_id_edit.textChanged.connect(self._update_accept_enabled)
            self.case_id_edit.textChanged.connect(self._update_accept_enabled)
            self.attestation.toggled.connect(self._update_accept_enabled)
            self._update_accept_enabled()

        def _update_accept_enabled(self, *_args: Any) -> None:
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
                bool(
                    self.geometry_id_edit.text().strip()
                    and self.case_id_edit.text().strip()
                    and self.attestation.isChecked()
                )
            )

        def binding_values(self) -> tuple[str, str]:
            return (
                self.geometry_id_edit.text().strip(),
                self.case_id_edit.text().strip(),
            )


    class _DatasetMappingEditor(QWidget):
        mapping_changed = Signal()
        catalog_notice = Signal(str)

        def __init__(self, empty_text: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._empty_text = empty_text
            self._ids: tuple[str, ...] = ()
            self._required_ids: tuple[str, ...] = ()
            self._catalog: tuple[LoadedDatasetEntry, ...] = ()
            self._loaded_buttons: dict[str, _LoadedDatasetButton] = {}
            self.table = QTableWidget(0, 5, self)
            self.table.setHorizontalHeaderLabels(
                [
                    "dataset_id",
                    "OPN − FRD response (.grim)",
                    "Host material / coating ID",
                    "Loaded",
                    "",
                ]
            )
            self.table.setToolTip(
                "Every dataset_id used by the placement CSV must map to the "
                "matching coherent OPN-FRD .grim response."
            )
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setAlternatingRowColors(True)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self.empty_label = QLabel(empty_text, self)
            self.empty_label.setWordWrap(True)
            self.completeness_label = QLabel(self)
            self.completeness_label.setWordWrap(True)
            self.completeness_label.setObjectName("featureMappingStatus")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.empty_label)
            layout.addWidget(self.table)
            layout.addWidget(self.completeness_label)
            self.table.cellChanged.connect(self._table_changed)
            self.set_dataset_ids(())

        @property
        def dataset_ids(self) -> tuple[str, ...]:
            return self._ids

        def mapping(self) -> dict[str, str]:
            result: dict[str, str] = {}
            for row, dataset_id in enumerate(self._ids):
                item = self.table.item(row, 1)
                result[dataset_id] = "" if item is None else item.text().strip()
            return result

        def host_materials(self) -> dict[str, str]:
            result: dict[str, str] = {}
            for row, dataset_id in enumerate(self._ids):
                item = self.table.item(row, 2)
                result[dataset_id] = "" if item is None else item.text().strip()
            return result

        def missing_ids(self) -> tuple[str, ...]:
            current = self.mapping()
            return tuple(
                dataset_id
                for dataset_id in self._required_ids
                if not _clean_path(current.get(dataset_id))
            )

        def _update_completeness(self) -> None:
            missing = self.missing_ids()
            if not self._ids:
                self.completeness_label.setText("")
                self.completeness_label.setVisible(False)
            elif missing:
                self.completeness_label.setText(
                    f"○ {len(missing)} response file(s) still required: "
                    + ", ".join(missing)
                )
                self.completeness_label.setVisible(True)
            else:
                self.completeness_label.setText(
                    f"✓ All {len(self._required_ids)} enabled response(s) mapped."
                )
                self.completeness_label.setVisible(True)

        def _table_changed(self, *_args: Any) -> None:
            self._update_completeness()
            self.mapping_changed.emit()

        def set_dataset_ids(
            self,
            dataset_ids: tuple[str, ...] | list[str],
            mapping: Mapping[str, str] | None = None,
            host_materials: Mapping[str, str] | None = None,
        ) -> None:
            existing = self.mapping() if self._ids else {}
            existing_hosts = self.host_materials() if self._ids else {}
            if mapping is not None:
                existing.update({str(key): _clean_path(value) for key, value in mapping.items()})
            if host_materials is not None:
                existing_hosts.update(
                    {
                        str(key): str(value).strip()
                        for key, value in host_materials.items()
                    }
                )
            self._ids = tuple(str(value) for value in dataset_ids)
            self._required_ids = self._ids
            self._loaded_buttons = {}
            self.table.blockSignals(True)
            self.table.setRowCount(len(self._ids))
            for row, dataset_id in enumerate(self._ids):
                id_item = QTableWidgetItem(dataset_id)
                id_item.setFlags(
                    id_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self.table.setItem(row, 0, id_item)
                self.table.setItem(
                    row, 1, QTableWidgetItem(existing.get(dataset_id, ""))
                )
                host_item = QTableWidgetItem(existing_hosts.get(dataset_id, ""))
                host_item.setToolTip(
                    "Optional per-response override. Leave blank to use the "
                    "global default host material/coating ID."
                )
                self.table.setItem(row, 2, host_item)
                loaded_button = _LoadedDatasetButton(self.table)
                loaded_button.set_catalog(self._catalog)
                loaded_button.path_selected.connect(
                    lambda path, key=dataset_id: self.set_path(key, path)
                )
                loaded_button.notice.connect(self.catalog_notice.emit)
                self._loaded_buttons[dataset_id] = loaded_button
                self.table.setCellWidget(row, 3, loaded_button)
                button = QPushButton("Browse…", self.table)
                button.clicked.connect(
                    lambda _checked=False, key=dataset_id: self._browse(key)
                )
                self.table.setCellWidget(row, 4, button)
            self.table.blockSignals(False)
            has_rows = bool(self._ids)
            self.empty_label.setVisible(not has_rows)
            self.table.setVisible(has_rows)
            self.table.setMinimumHeight(112 if has_rows else 0)
            self._update_completeness()

        def set_required_dataset_ids(self, dataset_ids: Iterable[str]) -> None:
            required = tuple(dict.fromkeys(str(value) for value in dataset_ids))
            unknown = sorted(set(required) - set(self._ids))
            if unknown:
                raise ValueError(
                    f"Enabled spatial features reference unknown dataset IDs {unknown}."
                )
            self._required_ids = required
            self._update_completeness()

        def set_loaded_dataset_catalog(
            self, entries: tuple[LoadedDatasetEntry, ...]
        ) -> None:
            self._catalog = tuple(entries)
            for button in self._loaded_buttons.values():
                button.set_catalog(self._catalog)

        def loaded_dataset_button(
            self, dataset_id: str
        ) -> _LoadedDatasetButton:
            """Return the row's chooser without exposing table-column details."""

            try:
                return self._loaded_buttons[str(dataset_id)]
            except KeyError as exc:
                raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc

        def set_path(self, dataset_id: str, path: str) -> None:
            try:
                row = self._ids.index(str(dataset_id))
            except ValueError as exc:
                raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc
            self.table.item(row, 1).setText(_clean_path(path))

        def set_host_material(self, dataset_id: str, material: str) -> None:
            try:
                row = self._ids.index(str(dataset_id))
            except ValueError as exc:
                raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc
            self.table.item(row, 2).setText(str(material).strip())

        def _browse(self, dataset_id: str) -> None:
            current = self.mapping().get(dataset_id, "")
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Choose OPN-FRD response for {dataset_id}",
                current,
                "GRIM response (*.grim);;All files (*)",
            )
            if path:
                self.set_path(dataset_id, path)


    class _SpatialFeatureTree(QTreeWidget):
        """Checkable spatial definition tree, separate from response math."""

        selection_changed = Signal()
        _ROLE_KIND = Qt.ItemDataRole.UserRole
        _ROLE_INSTANCE_ID = Qt.ItemDataRole.UserRole + 1
        _USE_COLUMN = 2

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setColumnCount(3)
            self.setHeaderLabels(["Body / spatial features", "Response", "Use"])
            self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.header().setSectionResizeMode(
                1, QHeaderView.ResizeMode.ResizeToContents
            )
            self.header().setSectionResizeMode(
                2, QHeaderView.ResizeMode.ResizeToContents
            )
            self.headerItem().setToolTip(
                self._USE_COLUMN,
                "Include this spatial feature in preview, physical validation, "
                "response loading, and assembly.",
            )
            self.setAlternatingRowColors(True)
            self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.setMinimumHeight(190)
            self._syncing = False
            self._filter_text = ""
            self.itemChanged.connect(self._on_item_changed)

        @staticmethod
        def _search_text(item: QTreeWidgetItem) -> str:
            return " ".join(item.text(column) for column in (0, 1)).casefold()

        def _set_default_expansion(self) -> None:
            """Expand structural roots while leaving response groups compact."""

            for top_index in range(self.topLevelItemCount()):
                body = self.topLevelItem(top_index)
                body.setExpanded(True)
                for kind_index in range(body.childCount()):
                    kind_root = body.child(kind_index)
                    kind_root.setExpanded(True)
                    for dataset_index in range(kind_root.childCount()):
                        kind_root.child(dataset_index).setExpanded(False)

        def set_filter_text(self, text: str) -> None:
            """Filter by instance, dataset ID, or mapped response text.

            Filtering is display-only. Hidden leaves remain part of recursive
            ``Use`` operations and :meth:`excluded_ids`, so searching can never
            silently change assembly membership.
            """

            query = str(text or "").strip().casefold()
            self._filter_text = query

            def visit(item: QTreeWidgetItem, ancestor_matches: bool = False) -> bool:
                own_match = bool(query) and query in self._search_text(item)
                reveal_subtree = ancestor_matches or own_match
                child_visible = False
                for index in range(item.childCount()):
                    child_visible = (
                        visit(item.child(index), reveal_subtree) or child_visible
                    )
                visible = not query or reveal_subtree or child_visible
                item.setHidden(not visible)
                if query and visible and item.childCount():
                    item.setExpanded(True)
                return visible

            for top_index in range(self.topLevelItemCount()):
                visit(self.topLevelItem(top_index))
            if not query:
                self._set_default_expansion()

        @staticmethod
        def _checkable_flags(item: QTreeWidgetItem) -> None:
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )

        def _set_checked(self, item: QTreeWidgetItem, enabled: bool) -> None:
            item.setCheckState(
                self._USE_COLUMN,
                Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked,
            )

        def _set_subtree(self, item: QTreeWidgetItem, enabled: bool) -> None:
            if item.data(0, self._ROLE_KIND) != "body":
                self._set_checked(item, enabled)
            for index in range(item.childCount()):
                self._set_subtree(item.child(index), enabled)

        def _sync_ancestors(self, item: QTreeWidgetItem | None) -> None:
            current = item
            while current is not None:
                # The body is the required host response, not a switchable
                # feature group. Keep its Use cell labelled ``Required`` while
                # its point/line children summarize their own selections.
                if current.data(0, self._ROLE_KIND) == "body":
                    break
                states = [
                    current.child(index).checkState(self._USE_COLUMN)
                    for index in range(current.childCount())
                ]
                if states:
                    if all(state == Qt.CheckState.Checked for state in states):
                        state = Qt.CheckState.Checked
                    elif all(state == Qt.CheckState.Unchecked for state in states):
                        state = Qt.CheckState.Unchecked
                    else:
                        state = Qt.CheckState.PartiallyChecked
                    current.setCheckState(self._USE_COLUMN, state)
                current = current.parent()

        @Slot(QTreeWidgetItem, int)
        def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
            if self._syncing or column != self._USE_COLUMN:
                return
            if item.data(0, self._ROLE_KIND) == "body":
                return
            state = item.checkState(self._USE_COLUMN)
            self._syncing = True
            try:
                if state in (Qt.CheckState.Checked, Qt.CheckState.Unchecked):
                    enabled = state == Qt.CheckState.Checked
                    for index in range(item.childCount()):
                        self._set_subtree(item.child(index), enabled)
                self._sync_ancestors(item.parent())
            finally:
                self._syncing = False
            self.selection_changed.emit()

        def set_configuration(self, model: FeatureAssemblyFormModel) -> None:
            """Rebuild from parsed descriptors while honoring model exclusions."""
            self._syncing = True
            try:
                self.clear()
                body_name = Path(_clean_path(model.values.base_grim)).name
                body = QTreeWidgetItem(
                    ["Body", body_name or "clean-body response not selected", "Required"]
                )
                body.setData(0, self._ROLE_KIND, "body")
                body.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                body.setToolTip(
                    0,
                    "The clean-body response is always required. Feature checkboxes "
                    "control installed-minus-clean deltas added to it.\n"
                    f"Response: {_clean_path(model.values.base_grim) or 'not selected'}\n"
                    f"Surface: {_clean_path(model.values.surface_mesh) or 'embedded/not selected'}\n"
                    f"Surface units: {model.values.surface_units}; "
                    f"flip normals: {bool(model.values.flip_surface_normals)}",
                )
                self.addTopLevelItem(body)

                self._add_kind(
                    model,
                    parent=body,
                    kind="point",
                    label="Point features",
                    descriptors=model.point_instances,
                    mappings=model.values.point_datasets,
                    excluded=model.values.excluded_point_placement_ids,
                )
                self._add_kind(
                    model,
                    parent=body,
                    kind="line",
                    label="Line features",
                    descriptors=model.line_instances,
                    mappings=model.values.line_datasets,
                    excluded=model.values.excluded_line_ids,
                )
                self.set_filter_text(self._filter_text)
            finally:
                self._syncing = False

        def _add_kind(
            self,
            model: FeatureAssemblyFormModel,
            *,
            parent: QTreeWidgetItem,
            kind: str,
            label: str,
            descriptors: tuple,
            mappings: Mapping[str, str],
            excluded: set[str],
        ) -> None:
            root = QTreeWidgetItem([f"{label} ({len(descriptors)})", "", ""])
            root.setData(0, self._ROLE_KIND, f"{kind}_root")
            self._checkable_flags(root)
            parent.addChild(root)
            by_dataset: dict[str, list[tuple]] = {}
            for descriptor in descriptors:
                by_dataset.setdefault(str(descriptor[1]), []).append(descriptor)
            for dataset_id, instances in by_dataset.items():
                mapped_path = _clean_path(mappings.get(dataset_id))
                mapped = Path(mapped_path).name or "not mapped"
                group = QTreeWidgetItem(
                    [f"dataset_id: {dataset_id} ({len(instances)})", mapped, ""]
                )
                group.setData(0, self._ROLE_KIND, f"{kind}_dataset")
                group.setToolTip(
                    1,
                    mapped_path or "No OPN-FRD response is mapped for this dataset ID.",
                )
                self._checkable_flags(group)
                root.addChild(group)
                for descriptor in instances:
                    instance_id = str(descriptor[0])
                    suffix = (
                        ""
                        if kind == "point"
                        else f" ({int(descriptor[2])} segment(s))"
                    )
                    leaf = QTreeWidgetItem([instance_id + suffix, "", ""])
                    leaf.setData(0, self._ROLE_KIND, kind)
                    leaf.setData(0, self._ROLE_INSTANCE_ID, instance_id)
                    self._checkable_flags(leaf)
                    self._set_checked(leaf, instance_id not in excluded)
                    group.addChild(leaf)
                self._sync_ancestors(group)
            self._sync_ancestors(root)

        def excluded_ids(self) -> tuple[set[str], set[str]]:
            point_ids: set[str] = set()
            line_ids: set[str] = set()
            iterator = self.invisibleRootItem()
            pending = [iterator.child(index) for index in range(iterator.childCount())]
            while pending:
                item = pending.pop()
                pending.extend(
                    item.child(index) for index in range(item.childCount())
                )
                kind = item.data(0, self._ROLE_KIND)
                instance_id = item.data(0, self._ROLE_INSTANCE_ID)
                if (
                    kind in {"point", "line"}
                    and instance_id
                    and item.checkState(self._USE_COLUMN) == Qt.CheckState.Unchecked
                ):
                    (point_ids if kind == "point" else line_ids).add(
                        str(instance_id)
                    )
            return point_ids, line_ids

        def select_instance(self, kind: str, instance_id: str) -> bool:
            """Reveal and select one QA-linked spatial feature leaf."""

            normalized = str(kind).strip().lower()
            target = str(instance_id).strip()
            if normalized not in {"point", "line"} or not target:
                return False
            pending = [
                self.topLevelItem(index)
                for index in range(self.topLevelItemCount())
            ]
            while pending:
                item = pending.pop()
                pending.extend(
                    item.child(index) for index in range(item.childCount())
                )
                if (
                    item.data(0, self._ROLE_KIND) == normalized
                    and str(item.data(0, self._ROLE_INSTANCE_ID) or "") == target
                ):
                    self.setCurrentItem(item)
                    current = item.parent()
                    while current is not None:
                        current.setExpanded(True)
                        current = current.parent()
                    self.scrollToItem(
                        item, QAbstractItemView.ScrollHint.PositionAtCenter
                    )
                    return True
            return False


    class FeatureAssemblyPanel(QWidget):
        """New-user-facing feature assembly form with background execution."""

        preview_ready = Signal(object)
        preview_stale = Signal(str)
        feature_built = Signal(str)
        build_failed = Signal(str)
        status_changed = Signal(str)

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            service: Any = None,
        ) -> None:
            super().__init__(parent)
            self.model = FeatureAssemblyFormModel()
            self._service: Any = service
            self._thread: QThread | None = None
            self._worker: _OperationWorker | None = None
            self._active_kind = ""
            self._discovery_paths: tuple[str, str] | None = None
            self._preview_is_current = False
            self._validated_plan_current = False
            self._validation_warning_count = 0
            self._loaded_dataset_catalog: tuple[LoadedDatasetEntry, ...] = ()
            self._recipe_path: Path | None = None
            self._recipe_dirty = False
            self._recipe_source_warnings: tuple[str, ...] = ()
            self._loading_recipe = False
            self._surface_binding_checked_key: tuple[Any, ...] | None = None
            self._surface_binding_checked: Mapping[str, Any] | None = None
            self._surface_binding_error_key: tuple[Any, ...] | None = None
            self._surface_binding_error = ""
            self._surface_dimensions_key: tuple[Any, ...] | None = None
            self._surface_dimensions_text = ""
            self._current_work_estimate = AssemblyWorkEstimate(available=False)
            self._build_ui()

        def _build_ui(self) -> None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(6)

            intro = QLabel(
                "Build one coherent body + features response in three steps.",
                self,
            )
            intro.setWordWrap(True)
            intro.setObjectName("featurePanelIntro")
            outer.addWidget(intro)

            self.workflow_steps_label = QLabel(
                "Choose the body, map feature responses, then validate and run.",
                self,
            )
            self.workflow_steps_label.setWordWrap(True)
            self.workflow_steps_label.setObjectName("featureWorkflowSteps")
            self.workflow_steps_label.setVisible(False)
            self.next_step_label = QLabel(self)
            self.next_step_label.setObjectName("featureNextStep")
            self.next_step_label.setWordWrap(True)
            outer.addWidget(self.next_step_label)

            recipe_group = QGroupBox("Reusable assembly recipe", self)
            recipe_group.setObjectName("featureRecipeBar")
            recipe_layout = QVBoxLayout(recipe_group)
            recipe_layout.setContentsMargins(8, 6, 8, 6)
            recipe_layout.setSpacing(5)
            recipe_fields = QHBoxLayout()
            recipe_fields.addWidget(QLabel("Assembly:"))
            self.recipe_name_edit = QLineEdit(recipe_group)
            self.recipe_name_edit.setPlaceholderText("Vehicle feature assembly")
            self.recipe_name_edit.setText("Vehicle feature assembly")
            self.recipe_name_edit.setToolTip(
                "A human-readable name stored in the portable recipe."
            )
            recipe_fields.addWidget(self.recipe_name_edit, 2)
            recipe_fields.addWidget(QLabel("Variant:"))
            self.recipe_variant_edit = QLineEdit(recipe_group)
            self.recipe_variant_edit.setPlaceholderText("Baseline / Option A")
            self.recipe_variant_edit.setText("Baseline")
            self.recipe_variant_edit.setToolTip(
                "Name this exact feature membership for repeatable trade studies."
            )
            recipe_fields.addWidget(self.recipe_variant_edit, 2)
            recipe_layout.addLayout(recipe_fields)
            recipe_actions = QHBoxLayout()
            self.recipe_status_label = QLabel(recipe_group)
            self.recipe_status_label.setObjectName("featureRecipeStatus")
            self.recipe_status_label.setWordWrap(True)
            recipe_actions.addWidget(self.recipe_status_label, 1)
            self.load_recipe_button = QPushButton("Load…", recipe_group)
            self.load_recipe_button.setToolTip(
                "Restore body, placements, response mappings, tolerances, and exact "
                "enabled/disabled feature membership from a versioned recipe."
            )
            recipe_actions.addWidget(self.load_recipe_button)
            self.save_recipe_button = QPushButton("Save", recipe_group)
            self.save_recipe_button.setToolTip(
                "Save changes to the current recipe. Source identities are recorded "
                "so moved, missing, or modified inputs can be reported on load."
            )
            recipe_actions.addWidget(self.save_recipe_button)
            self.save_recipe_as_button = QPushButton("Save as…", recipe_group)
            self.save_recipe_as_button.setToolTip(
                "Save this named variant as a separate portable .assembly.json file."
            )
            recipe_actions.addWidget(self.save_recipe_as_button)
            recipe_layout.addLayout(recipe_actions)
            self.recipe_section = _DisclosureSection(
                "Reusable recipe (optional)", self, expanded=False
            )
            self.recipe_section.addWidget(recipe_group)
            outer.addWidget(self.recipe_section)

            self.workflow_tabs = QTabWidget(self)
            self.workflow_tabs.setObjectName("featureWorkflowTabs")
            self.body_step_page = QWidget(self.workflow_tabs)
            self.map_step_page = QWidget(self.workflow_tabs)
            self.review_step_page = QWidget(self.workflow_tabs)

            def _step_scroll(page: QWidget, object_name: str):
                page_layout = QVBoxLayout(page)
                page_layout.setContentsMargins(0, 0, 0, 0)
                scroll = QScrollArea(page)
                scroll.setObjectName(object_name)
                scroll.setFrameShape(QFrame.Shape.NoFrame)
                scroll.setAutoFillBackground(False)
                scroll.viewport().setAutoFillBackground(False)
                scroll.setWidgetResizable(True)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                content = QWidget(scroll)
                content.setObjectName("featureAssemblyContent")
                content.setAutoFillBackground(False)
                content_layout = QVBoxLayout(content)
                content_layout.setContentsMargins(0, 0, 0, 0)
                content_layout.setSpacing(7)
                scroll.setWidget(content)
                page_layout.addWidget(scroll, 1)
                return content, content_layout, page_layout

            body_content, body_content_layout, self.body_page_layout = _step_scroll(
                self.body_step_page, "featureBodyScroll"
            )
            map_content, map_content_layout, self.map_page_layout = _step_scroll(
                self.map_step_page, "featureMapScroll"
            )
            review_content, review_content_layout, self.review_page_layout = _step_scroll(
                self.review_step_page, "featureReviewScroll"
            )
            self.form_content = self.workflow_tabs
            self.workflow_tabs.addTab(self.body_step_page, "Body (1)")
            self.workflow_tabs.addTab(self.map_step_page, "Map Features (2)")
            self.workflow_tabs.addTab(self.review_step_page, "Review (3)")
            outer.addWidget(self.workflow_tabs, 1)

            body_group = QGroupBox("Body response and geometry", body_content)
            body_group.setObjectName("featureStepCard")
            body_form = QFormLayout(body_group)
            body_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.base_picker = _PathPicker(
                caption="Choose clean-body/base GRIM",
                file_filter="GRIM response (*.grim);;All files (*)",
                allow_loaded_dataset=True,
            )
            self.surface_picker = _PathPicker(
                caption="Choose body surface mesh",
                file_filter="Surface mesh (*.stl *.facet);;All files (*)",
            )
            self.output_picker = _PathPicker(
                caption="Save assembled GRIM",
                file_filter="GRIM response (*.grim);;All files (*)",
                save=True,
            )
            self.coordinate_units = QComboBox(body_group)
            self.surface_units = QComboBox(body_group)
            self.coordinate_units.addItem("Choose coordinate units...", "")
            self.surface_units.addItem("Choose mesh units...", "")
            for label, value in UNIT_CHOICES:
                self.coordinate_units.addItem(label, value)
                self.surface_units.addItem(label, value)
            self.flip_normals = QCheckBox("Flip mesh normals", body_group)
            self.shadow = QCheckBox(
                "Apply geometric body shadowing (requires mesh)", body_group
            )
            mesh_options = QWidget(body_group)
            mesh_layout = QVBoxLayout(mesh_options)
            mesh_layout.setContentsMargins(0, 0, 0, 0)
            mesh_layout.addWidget(self.flip_normals)
            mesh_layout.addWidget(self.shadow)
            self.base_picker.setToolTip(
                "Clean-body response to which the point and/or line feature "
                "responses will be coherently added."
            )
            self.surface_picker.setToolTip(
                "Choose the matching STL/facet surface for a 3-D body. Leave "
                "blank when the base GRIM contains an embedded BoR profile."
            )
            self.coordinate_units.setToolTip(
                "Units used by every x/y/z coordinate in both placement CSVs."
            )
            self.surface_units.setToolTip(
                "Units of the selected STL/facet surface, independent of the CSV units."
            )
            body_form.addRow("Clean-body response:", self.base_picker)
            body_form.addRow("Surface mesh (optional):", self.surface_picker)
            body_form.addRow("Surface mesh units:", self.surface_units)
            self.surface_dimensions_label = QLabel(body_group)
            self.surface_dimensions_label.setObjectName("featureSummary")
            self.surface_dimensions_label.setWordWrap(True)
            self.surface_dimensions_label.setText(
                "No external mesh selected; physical mesh dimensions are not "
                "available yet."
            )
            body_form.addRow("Interpreted mesh size:", self.surface_dimensions_label)
            binding_box = QWidget(body_group)
            binding_layout = QVBoxLayout(binding_box)
            binding_layout.setContentsMargins(0, 0, 0, 0)
            binding_layout.setSpacing(4)
            self.surface_binding_status = QLabel(binding_box)
            self.surface_binding_status.setObjectName("featureSurfaceBindingStatus")
            self.surface_binding_status.setWordWrap(True)
            binding_layout.addWidget(self.surface_binding_status)
            binding_actions = QHBoxLayout()
            binding_actions.setContentsMargins(0, 0, 0, 0)
            self.check_surface_binding_button = QPushButton(
                "Check binding", binding_box
            )
            self.check_surface_binding_button.setToolTip(
                "Explicitly hash and verify the exact clean-body response, mesh, "
                "units, frame declaration, and reviewed IDs."
            )
            self.bind_surface_button = QPushButton(
                "Bind / refresh…", binding_box
            )
            self.bind_surface_button.setToolTip(
                "Create the canonical <surface>.assembly.json after a responsible "
                "team member reviews solve-to-CAD registration."
            )
            binding_actions.addWidget(self.check_surface_binding_button)
            binding_actions.addWidget(self.bind_surface_button)
            binding_actions.addStretch(1)
            binding_layout.addLayout(binding_actions)
            body_form.addRow("Solve ↔ mesh registration:", binding_box)
            body_form.addRow("Mesh options:", mesh_options)
            self.body_preview_help = QLabel(
                "Body preview: selected mesh, or the base file's embedded BoR. "
                "A 3-D base without embedded geometry needs its matching mesh.",
                body_group,
            )
            self.body_preview_help.setWordWrap(True)
            self.body_preview_help.setObjectName("featureHint")
            body_form.addRow("", self.body_preview_help)
            body_content_layout.addWidget(body_group)
            body_content_layout.addStretch(1)

            feature_group = QGroupBox("Placements and feature responses", map_content)
            feature_group.setObjectName("featureStepCard")
            feature_layout = QVBoxLayout(feature_group)
            units_row = QHBoxLayout()
            units_label = QLabel("All CSV coordinates:", feature_group)
            units_label.setBuddy(self.coordinate_units)
            units_row.addWidget(units_label)
            units_row.addWidget(self.coordinate_units, 1)
            feature_layout.addLayout(units_row)
            self.shared_units_label = QLabel(
                "One point CSV can contain every point family; one line CSV can "
                "contain every ordered line chain. Both selected CSVs share this unit.",
                feature_group,
            )
            self.shared_units_label.setObjectName("featureHint")
            self.shared_units_label.setWordWrap(True)
            feature_layout.addWidget(self.shared_units_label)
            self.feature_summary_label = QLabel(feature_group)
            self.feature_summary_label.setObjectName("featureSummary")
            self.feature_summary_label.setWordWrap(True)
            feature_layout.addWidget(self.feature_summary_label)
            self.feature_tabs = QTabWidget(feature_group)

            point_page = QWidget(self.feature_tabs)
            point_layout = QVBoxLayout(point_page)
            self.point_csv_picker = _PathPicker(
                caption="Choose point placement CSV",
                file_filter="CSV placement file (*.csv);;All files (*)",
            )
            self.point_csv_picker.setToolTip(
                "Strict GHOST point-placement CSV. This is the same file used "
                "by local scripts and the HPC workflow."
            )
            point_layout.addWidget(QLabel("Point location/orientation CSV:"))
            point_layout.addWidget(self.point_csv_picker)
            self.point_csv_summary = QLabel("No point CSV selected.", point_page)
            self.point_csv_summary.setObjectName("featureCsvSummary")
            self.point_csv_summary.setWordWrap(True)
            point_layout.addWidget(self.point_csv_summary)
            self.point_help_label = QLabel(
                "This is the same strict GHOST CSV used locally and on HPC. The "
                "header is followed directly by data rows—no units row or comments. "
                "Normal is local +z; projected roll is local +x / azimuth zero. "
                "placement_id values must be unique.",
                point_page,
            )
            self.point_help_label.setWordWrap(True)
            self.point_help_label.setVisible(False)
            point_layout.addWidget(self.point_help_label)
            point_format_row = QHBoxLayout()
            self.point_format_button = QPushButton(
                "CSV guide", point_page
            )
            self.point_format_button.setCheckable(True)
            point_format_row.addWidget(self.point_format_button)
            self.point_schema_label = QLabel(
                "Exact header (column order is fixed):\n"
                + ",".join(POINT_PLACEMENT_COLUMNS)
                + "\nExample row:\n"
                + POINT_PLACEMENT_EXAMPLE,
                point_page,
            )
            self.point_schema_label.setWordWrap(True)
            self.point_schema_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.point_schema_label.setObjectName("featureCsvSchema")
            self.point_template_button = QPushButton(
                "Save template…", point_page
            )
            self.point_template_button.setToolTip(
                "Write the exact required point header to a new .csv file."
            )
            point_format_row.addWidget(self.point_template_button)
            self.point_clear_button = QPushButton("Remove", point_page)
            self.point_clear_button.setToolTip(
                "Remove the point CSV and its response mappings from this build."
            )
            point_format_row.addWidget(self.point_clear_button)
            point_format_row.addStretch(1)
            point_layout.addLayout(point_format_row)
            self.point_schema_label.setVisible(False)
            point_layout.addWidget(self.point_schema_label)
            point_response_help = QLabel(
                "Response contract: each dataset_id maps to a coherent OPN − FRD "
                "delta (installed/featured minus clean skin), with VV, HH, and "
                "reciprocal cross-polar response.",
                point_page,
            )
            point_response_help.setWordWrap(True)
            point_response_help.setObjectName("featureContract")
            point_layout.addWidget(point_response_help)
            self.point_mapping = _DatasetMappingEditor(
                "Choose a point CSV; one response row will appear per dataset_id.",
                point_page,
            )
            point_layout.addWidget(self.point_mapping)
            self.feature_tabs.addTab(point_page, "Point features")

            line_page = QWidget(self.feature_tabs)
            line_layout = QVBoxLayout(line_page)
            self.line_csv_picker = _PathPicker(
                caption="Choose line placement CSV",
                file_filter="CSV placement file (*.csv);;All files (*)",
            )
            self.line_csv_picker.setToolTip(
                "Strict GHOST ordered-segment line-placement CSV. This is the "
                "same file used by local scripts and the HPC workflow."
            )
            line_layout.addWidget(QLabel("Line path/orientation CSV:"))
            line_layout.addWidget(self.line_csv_picker)
            self.line_csv_summary = QLabel("No line CSV selected.", line_page)
            self.line_csv_summary.setObjectName("featureCsvSummary")
            self.line_csv_summary.setWordWrap(True)
            line_layout.addWidget(self.line_csv_summary)
            self.line_help_label = QLabel(
                "This is the same strict GHOST CSV used locally and on HPC. The "
                "header is followed directly by data rows—no units row or comments. "
                "Rows for each line_id stay together, segment_index starts at 1, "
                "segments meet head-to-tail, and endpoint normals point outward.",
                line_page,
            )
            self.line_help_label.setWordWrap(True)
            self.line_help_label.setVisible(False)
            line_layout.addWidget(self.line_help_label)
            line_format_row = QHBoxLayout()
            self.line_format_button = QPushButton(
                "CSV guide", line_page
            )
            self.line_format_button.setCheckable(True)
            line_format_row.addWidget(self.line_format_button)
            self.line_schema_label = QLabel(
                "Exact header (column order is fixed):\n"
                + ",".join(LINE_PLACEMENT_COLUMNS)
                + "\nExample row:\n"
                + LINE_PLACEMENT_EXAMPLE,
                line_page,
            )
            self.line_schema_label.setWordWrap(True)
            self.line_schema_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.line_schema_label.setObjectName("featureCsvSchema")
            self.line_template_button = QPushButton(
                "Save template…", line_page
            )
            self.line_template_button.setToolTip(
                "Write the exact required line header to a new .csv file."
            )
            line_format_row.addWidget(self.line_template_button)
            self.line_clear_button = QPushButton("Remove", line_page)
            self.line_clear_button.setToolTip(
                "Remove the line CSV and its response mappings from this build."
            )
            line_format_row.addWidget(self.line_clear_button)
            line_format_row.addStretch(1)
            line_layout.addLayout(line_format_row)
            self.line_schema_label.setVisible(False)
            line_layout.addWidget(self.line_schema_label)
            line_response_help = QLabel(
                "Response contract: each dataset_id maps to a coherent OPN − FRD "
                "delta (installed/featured minus clean skin) containing TE and TM.",
                line_page,
            )
            line_response_help.setWordWrap(True)
            line_response_help.setObjectName("featureContract")
            line_layout.addWidget(line_response_help)
            self.line_mapping = _DatasetMappingEditor(
                "Choose a line CSV; one response row will appear per dataset_id.",
                line_page,
            )
            line_layout.addWidget(self.line_mapping)
            self.feature_tabs.addTab(line_page, "Line features")
            feature_layout.addWidget(self.feature_tabs)
            scan_row = QHBoxLayout()
            self.scan_button = QPushButton("Refresh selected CSVs", feature_group)
            self.scan_button.setToolTip(
                "Parse the selected CSVs with the authoritative GHOST parser "
                "and list every response dataset that must be supplied."
            )
            scan_row.addWidget(self.scan_button)
            scan_hint = QLabel(
                "CSV files are read automatically after Browse.", feature_group
            )
            scan_hint.setObjectName("featureHint")
            scan_hint.setWordWrap(True)
            scan_row.addWidget(scan_hint, 1)
            feature_layout.addLayout(scan_row)
            hierarchy_help = QLabel(
                "Spatial configuration — separate from whole-response dataset "
                "arithmetic. Uncheck a dataset family or individual placement "
                "to omit it from preview, physical validation, response loading, "
                "and assembly. The CSV itself is never rewritten.",
                feature_group,
            )
            hierarchy_help.setWordWrap(True)
            hierarchy_help.setObjectName("featureHint")
            feature_layout.addWidget(hierarchy_help)
            filter_row = QHBoxLayout()
            filter_label = QLabel("Find feature:", feature_group)
            self.spatial_feature_filter = QLineEdit(feature_group)
            self.spatial_feature_filter.setPlaceholderText(
                "Instance ID, dataset ID, or response file"
            )
            self.spatial_feature_filter.setClearButtonEnabled(True)
            self.spatial_feature_filter.setToolTip(
                "Filters the displayed hierarchy only. A parent Use checkbox still "
                "applies recursively to its complete subtree, including hidden items."
            )
            filter_row.addWidget(filter_label)
            filter_row.addWidget(self.spatial_feature_filter, 1)
            feature_layout.addLayout(filter_row)
            self.spatial_feature_tree = _SpatialFeatureTree(feature_group)
            feature_layout.addWidget(self.spatial_feature_tree)
            self.spatial_feature_filter.textChanged.connect(
                self.spatial_feature_tree.set_filter_text
            )
            self.spatial_selection_summary = QLabel(feature_group)
            self.spatial_selection_summary.setWordWrap(True)
            self.spatial_selection_summary.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.spatial_selection_summary.setToolTip(
                "Large disabled-ID lists are shortened here. Use Copy full selection "
                "to record exact trade-study membership."
            )
            self.spatial_selection_summary.setObjectName("featureSummary")
            summary_row = QHBoxLayout()
            summary_row.addWidget(self.spatial_selection_summary, 1)
            self.copy_spatial_selection_button = QPushButton(
                "Copy full selection", feature_group
            )
            self.copy_spatial_selection_button.setToolTip(
                "Copy the complete unshortened enabled/disabled membership summary."
            )
            self.copy_spatial_selection_button.clicked.connect(
                self._copy_full_spatial_selection_summary
            )
            summary_row.addWidget(self.copy_spatial_selection_button)
            feature_layout.addLayout(summary_row)
            map_content_layout.addWidget(feature_group)
            map_content_layout.addStretch(1)

            advanced = QWidget(review_content)
            advanced_form = QFormLayout(advanced)
            advanced_form.setContentsMargins(8, 8, 8, 8)
            advanced_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.skin_tol = QDoubleSpinBox(advanced)
            self.skin_tol.setDecimals(6)
            self.skin_tol.setRange(0.0, 100.0)
            self.skin_tol.setSingleStep(0.01)
            self.skin_tol.setValue(DEFAULT_SKIN_TOL_MM)
            self.skin_tol.setSuffix(" mm")
            self.skin_tol.setToolTip(
                "Maximum accepted distance from a feature to the host skin. This "
                "control is displayed in millimeters; recipes store meters."
            )
            self.phase_tol = QDoubleSpinBox(advanced)
            self.phase_tol.setDecimals(1)
            self.phase_tol.setRange(0.1, 90.0)
            self.phase_tol.setSingleStep(1.0)
            self.phase_tol.setValue(DEFAULT_SKIN_PHASE_TOL_DEG)
            self.phase_tol.setSuffix("°")
            self.phase_tol.setToolTip(
                "Maximum two-way phase error used to derive a frequency-aware "
                "skin-distance limit. Values above 90° are intentionally blocked."
            )
            self.normal_tol = QDoubleSpinBox(advanced)
            self.normal_tol.setDecimals(1)
            self.normal_tol.setRange(0.1, 89.9)
            self.normal_tol.setSingleStep(1.0)
            self.normal_tol.setValue(DEFAULT_NORMAL_TOL_DEG)
            self.normal_tol.setSuffix("°")
            self.shadow_bias = QLineEdit(advanced)
            self.shadow_bias.setPlaceholderText("Auto (recommended)")
            self.validation_profile = QComboBox(advanced)
            for (
                label,
                key,
                allow_legacy,
                require_manifests,
                require_body_certification,
            ) in VALIDATION_PROFILES:
                self.validation_profile.addItem(
                    label,
                    (
                        key,
                        allow_legacy,
                        require_manifests,
                        require_body_certification,
                    ),
                )
            self.validation_profile.setCurrentIndex(0)
            self.validation_profile.setToolTip(
                "Production requires a certified, fine-mesh dual-polarization "
                "GHOST body and certified feature responses. External/HPC keeps "
                "strict metadata and feature manifests but explicitly waives the "
                "local body certificate. Legacy compatibility reports missing "
                "applicability evidence as review warnings."
            )
            self.expected_host_material = QLineEdit(advanced)
            self.expected_host_material.setPlaceholderText(
                "Default for blank response rows, e.g. PEC or paint-stack-v3"
            )
            self.expected_host_material.setToolTip(
                "Convenience default for response rows left blank. Enter a per-row "
                "override beside each mapped response when the vehicle has mixed "
                "host materials or coating stacks."
            )
            self.reset_qa_defaults_button = QPushButton(
                "Reset placement-check defaults", advanced
            )
            self.reset_qa_defaults_button.setToolTip(
                "Restore 1 mm skin distance, 15° phase, and 15° normal limits."
            )
            advanced_form.addRow("Maximum skin distance:", self.skin_tol)
            advanced_form.addRow("Maximum two-way phase error:", self.phase_tol)
            advanced_form.addRow("Maximum normal mismatch:", self.normal_tol)
            advanced_form.addRow("Shadow ray bias (m):", self.shadow_bias)
            advanced_form.addRow("Validation profile:", self.validation_profile)
            advanced_form.addRow(
                "Default host material / coating ID:", self.expected_host_material
            )
            advanced_form.addRow("", self.reset_qa_defaults_button)
            self.advanced_section = _DisclosureSection(
                "Advanced placement checks · defaults active",
                review_content,
                expanded=False,
            )
            self.advanced_section.addWidget(advanced)
            self.advanced_section.header.setToolTip(
                "The displayed defaults remain active while this section is collapsed."
            )
            review_content_layout.addWidget(self.advanced_section)

            self.preview_help_label = QLabel(
                "Preview Geometry is visual QA only. Validate Placements additionally "
                "checks skin distance, outward normals, frame validity, and response "
                "mappings. Magenta arrows are normals; lavender arrows are point-roll "
                "references. Cyan line +t follows increasing segment_index; blue +b "
                "is the signed across-line axis (+t × +n). Preview Layers → Show "
                "changes only the display. Spatial "
                "Feature Configuration → Use controls which parsed instances enter "
                "preview, validation, response loading, and build.",
                review_content,
            )
            self.preview_help_label.setWordWrap(True)
            self.preview_guide = _DisclosureSection(
                "How to read the 3-D preview", review_content, expanded=False
            )
            self.preview_guide.addWidget(self.preview_help_label)
            review_content_layout.addWidget(self.preview_guide)

            review_group = QGroupBox("Readiness and output", review_content)
            review_group.setObjectName("featureStepCard")
            review_form = QFormLayout(review_group)
            review_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            review_form.addRow("Output response:", self.output_picker)
            self.readiness_checklist = QTreeWidget(review_group)
            self.readiness_checklist.setObjectName("featureReadinessChecklist")
            self.readiness_checklist.setHeaderLabels(["Requirement", "Status"])
            self.readiness_checklist.setRootIsDecorated(True)
            self.readiness_checklist.setAlternatingRowColors(True)
            self.readiness_checklist.setSelectionMode(
                QAbstractItemView.SelectionMode.NoSelection
            )
            self.readiness_checklist.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.readiness_checklist.header().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
            self.readiness_checklist.header().setSectionResizeMode(
                1, QHeaderView.ResizeMode.ResizeToContents
            )
            self.readiness_checklist.setMinimumHeight(245)
            self.readiness_checklist.setToolTip(
                "Updates immediately when an Assembly input, mapping, option, or "
                "validation result changes. Every required row must be ready before "
                "the final run is enabled."
            )
            review_form.addRow("Run checklist:", self.readiness_checklist)
            self.readiness_label = QLabel(review_group)
            self.readiness_label.setObjectName("featureReadiness")
            self.readiness_label.setWordWrap(True)
            self.readiness_label.setVisible(False)
            self.build_summary_label = QLabel(review_group)
            self.build_summary_label.setObjectName("featureBuildSummary")
            self.build_summary_label.setWordWrap(True)
            review_form.addRow("This build:", self.build_summary_label)
            self.work_estimate_label = QLabel(review_group)
            self.work_estimate_label.setObjectName("featureWorkEstimate")
            self.work_estimate_label.setWordWrap(True)
            self.work_estimate_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.work_estimate_label.setToolTip(
                "A broad static workload model, not a runtime benchmark. It "
                "uses radar looks/frequencies, enabled features, solver line "
                "pieces, mesh triangles, and optional body-shadow rays."
            )
            review_form.addRow("Rough workload:", self.work_estimate_label)
            self.model_scope_label = QLabel(
                "Model boundary: Assembly coherently superposes reviewed local "
                "feature deltas. It does not solve body–feature mutual coupling, "
                "feature–feature multiple scattering, diffraction, or creeping "
                "waves. Production validation certifies the inputs and declared "
                "applicability envelope—not full-vehicle Maxwell accuracy.",
                review_group,
            )
            self.model_scope_label.setWordWrap(True)
            self.model_scope_label.setObjectName("featureHint")
            self.model_scope_section = _DisclosureSection(
                "Physics scope", review_group, expanded=False
            )
            self.model_scope_section.addWidget(self.model_scope_label)
            review_form.addRow("", self.model_scope_section)
            self.validation_qa_label = QLabel(
                "Run Validate placements to see a row for every enabled point "
                "and line path.",
                review_group,
            )
            self.validation_qa_label.setWordWrap(True)
            self.validation_qa_label.setObjectName("featureValidationSummary")
            review_form.addRow("Placement QA:", self.validation_qa_label)
            self.validation_warning_label = QLabel(review_group)
            self.validation_warning_label.setWordWrap(True)
            self.validation_warning_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.validation_warning_label.setObjectName("featureValidationWarning")
            self.validation_warning_label.setVisible(False)
            review_form.addRow("QA warnings:", self.validation_warning_label)
            self.validation_warning_ack = QCheckBox(
                "I reviewed and accept these warnings for this output.",
                review_group,
            )
            self.validation_warning_ack.setToolTip(
                "This acknowledgment applies only to the current successful "
                "validation. Any input change or re-validation clears it."
            )
            self.validation_warning_ack.setVisible(False)
            review_form.addRow("Warning waiver:", self.validation_warning_ack)
            self.validation_qa_table = QTableWidget(0, 6, review_group)
            self.validation_qa_table.setHorizontalHeaderLabels(
                ["Type", "Instance", "Response ID", "Skin offset", "Normal error", "Result"]
            )
            self.validation_qa_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents
            )
            self.validation_qa_table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.Stretch
            )
            self.validation_qa_table.horizontalHeader().setSectionResizeMode(
                2, QHeaderView.ResizeMode.Stretch
            )
            for column in (3, 4, 5):
                self.validation_qa_table.horizontalHeader().setSectionResizeMode(
                    column, QHeaderView.ResizeMode.ResizeToContents
                )
            self.validation_qa_table.verticalHeader().setVisible(False)
            self.validation_qa_table.setEditTriggers(
                QAbstractItemView.EditTrigger.NoEditTriggers
            )
            self.validation_qa_table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.validation_qa_table.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection
            )
            self.validation_qa_table.setAlternatingRowColors(True)
            self.validation_qa_table.setMinimumHeight(135)
            self.validation_qa_table.setToolTip(
                "Authoritative placement records. WARN means the placement "
                "passed physical checks but is not illuminated by any requested "
                "look and therefore contributes zero. Click a row to reveal the "
                "same instance in Spatial Feature Configuration."
            )
            review_form.addRow("", self.validation_qa_table)
            review_content_layout.addWidget(review_group)

            self.status_label = QLabel(
                "No Assembly operation is running.",
                self,
            )
            self.status_label.setObjectName("featureAssemblyStatus")
            self.status_label.setWordWrap(True)
            self.status_label.setFrameShape(QFrame.Shape.StyledPanel)
            self.status_label.setMargin(6)
            self.review_page_layout.addWidget(self.status_label)

            operation_row = QHBoxLayout()
            self.operation_progress = QProgressBar(self)
            self.operation_progress.setRange(0, 100)
            self.operation_progress.setValue(0)
            self.operation_progress.setTextVisible(True)
            self.operation_progress.setFormat("Preparing…")
            self.operation_progress.setToolTip(
                "Live progress for the current Assembly operation. Cancellation "
                "is cooperative and never publishes a partial response."
            )
            self.operation_progress.setVisible(False)
            operation_row.addWidget(self.operation_progress, 1)
            self.cancel_operation_button = QPushButton("Cancel operation", self)
            self.cancel_operation_button.setToolTip(
                "Cooperatively stop validation or assembly after the current safe "
                "physics step. Cancellation never publishes a partial output or "
                "retains a partially validated plan."
            )
            self.cancel_operation_button.setVisible(False)
            self.cancel_operation_button.setEnabled(False)
            operation_row.addWidget(self.cancel_operation_button)
            self.review_page_layout.addLayout(operation_row)

            action_row = QHBoxLayout()
            self.input_preview_button = QPushButton("Preview geometry", self)
            self.input_preview_button.setToolTip(
                "Show available body geometry and enabled CSV locations without "
                "requiring response mappings or an output path. This is visual QA only."
            )
            self.preview_button = QPushButton("Validate placements", self)
            self.preview_button.setToolTip(
                "Validate body skin, normals, and response mapping completeness, then "
                "show the prepared body and features in the 3-D Assembly view."
            )
            self.build_button = QPushButton("Assemble validated && save", self)
            self.build_button.setToolTip(
                "Publish the exact current validation: coherently add every enabled "
                "mapped feature and atomically save the selected output .grim file."
            )
            self.build_button.setDefault(True)
            action_row.addWidget(self.input_preview_button)
            action_row.addWidget(self.preview_button)
            action_row.addWidget(self.build_button)
            self.review_page_layout.addLayout(action_row)
            review_content_layout.addStretch(1)
            self._busy_form_widgets = (
                body_group,
                feature_group,
                self.advanced_section,
                review_group,
            )

            self.status_changed.connect(self.status_label.setText)
            self.recipe_name_edit.textEdited.connect(self._recipe_metadata_changed)
            self.recipe_variant_edit.textEdited.connect(self._recipe_metadata_changed)
            self.load_recipe_button.clicked.connect(self._load_recipe_dialog)
            self.save_recipe_button.clicked.connect(self._save_recipe)
            self.save_recipe_as_button.clicked.connect(self._save_recipe_as)
            self.base_picker.editing_finished.connect(self._base_path_changed)
            self.surface_picker.editing_finished.connect(self._surface_path_changed)
            self.output_picker.editing_finished.connect(self._output_path_changed)
            self.point_csv_picker.editing_finished.connect(
                lambda: self._placement_csv_changed("point")
            )
            self.line_csv_picker.editing_finished.connect(
                lambda: self._placement_csv_changed("line")
            )
            self.coordinate_units.currentIndexChanged.connect(
                self._mark_preview_stale
            )
            self.surface_units.currentIndexChanged.connect(self._mark_preview_stale)
            self.check_surface_binding_button.clicked.connect(
                self.check_selected_surface_binding
            )
            self.bind_surface_button.clicked.connect(
                self.bind_selected_surface
            )
            self.flip_normals.toggled.connect(self._mark_preview_stale)
            self.shadow.toggled.connect(self._mark_preview_stale)
            self.skin_tol.valueChanged.connect(self._mark_preview_stale)
            self.phase_tol.valueChanged.connect(self._mark_preview_stale)
            self.normal_tol.valueChanged.connect(self._mark_preview_stale)
            self.shadow_bias.editingFinished.connect(self._mark_preview_stale)
            self.validation_profile.currentIndexChanged.connect(
                self._validation_profile_changed
            )
            self.expected_host_material.textEdited.connect(
                self._mark_preview_stale
            )
            self.reset_qa_defaults_button.clicked.connect(
                self._reset_qa_defaults
            )
            self.validation_warning_ack.toggled.connect(
                self._update_workflow_readiness
            )
            self.point_mapping.mapping_changed.connect(self._mapping_changed)
            self.line_mapping.mapping_changed.connect(self._mapping_changed)
            self.spatial_feature_tree.selection_changed.connect(
                self._spatial_selection_changed
            )
            self.validation_qa_table.cellClicked.connect(self._qa_row_clicked)
            self.base_picker.catalog_notice.connect(
                self._loaded_dataset_notice
            )
            self.point_mapping.catalog_notice.connect(
                self._loaded_dataset_notice
            )
            self.line_mapping.catalog_notice.connect(
                self._loaded_dataset_notice
            )
            self.point_format_button.toggled.connect(
                lambda checked: self._toggle_schema_help("point", checked)
            )
            self.line_format_button.toggled.connect(
                lambda checked: self._toggle_schema_help("line", checked)
            )
            self.point_template_button.clicked.connect(
                lambda _checked=False: self._save_template("point")
            )
            self.line_template_button.clicked.connect(
                lambda _checked=False: self._save_template("line")
            )
            self.point_clear_button.clicked.connect(
                lambda _checked=False: self._clear_placement_csv("point")
            )
            self.line_clear_button.clicked.connect(
                lambda _checked=False: self._clear_placement_csv("line")
            )
            self.scan_button.clicked.connect(self.refresh_dataset_ids)
            self.input_preview_button.clicked.connect(self.preview_inputs)
            self.preview_button.clicked.connect(self.validate_and_preview)
            self.build_button.clicked.connect(self.assemble_and_save)
            self.cancel_operation_button.clicked.connect(
                self.request_cancel
            )
            self._refresh_spatial_feature_tree()
            self._update_recipe_status()
            self._update_workflow_readiness()

        def _validation_profile_flags(self) -> tuple[bool, bool, bool]:
            data = self.validation_profile.currentData()
            if not isinstance(data, (tuple, list)) or len(data) != 4:
                return False, True, True
            return bool(data[1]), bool(data[2]), bool(data[3])

        def _set_validation_profile_from_values(
            self, values: FeatureAssemblyValues
        ) -> None:
            target = (
                bool(values.allow_legacy_base_metadata),
                bool(values.require_feature_manifests),
                bool(values.require_body_mesh_certification),
            )
            index = 0
            for candidate in range(self.validation_profile.count()):
                data = self.validation_profile.itemData(candidate)
                if (
                    isinstance(data, (tuple, list))
                    and len(data) == 4
                    and (
                        bool(data[1]),
                        bool(data[2]),
                        bool(data[3]),
                    ) == target
                ):
                    index = candidate
                    break
            self.validation_profile.setCurrentIndex(index)

        @Slot()
        def _validation_profile_changed(self, *_args: Any) -> None:
            (
                allow_legacy,
                require_manifests,
                require_body_certification,
            ) = self._validation_profile_flags()
            self.model.values.allow_legacy_base_metadata = allow_legacy
            self.model.values.require_feature_manifests = require_manifests
            self.model.values.require_body_mesh_certification = (
                require_body_certification
            )
            self._mark_preview_stale()

        @Slot()
        def _reset_qa_defaults(self) -> None:
            self.skin_tol.setValue(DEFAULT_SKIN_TOL_MM)
            self.phase_tol.setValue(DEFAULT_SKIN_PHASE_TOL_DEG)
            self.normal_tol.setValue(DEFAULT_NORMAL_TOL_DEG)
            self.shadow_bias.clear()
            self.status_changed.emit(
                "Restored the conservative placement-check defaults. Validate "
                "again before assembly."
            )

        def set_service(self, service: Any) -> None:
            coerce_feature_workflow(service)  # Fail early with an actionable API error.
            self._service = service
            self._update_workflow_readiness()

        def service(self) -> Any:
            return self._service

        def _update_recipe_status(self) -> None:
            name = self.recipe_name_edit.text().strip() or "Unnamed assembly"
            variant = self.recipe_variant_edit.text().strip() or "Unnamed variant"
            state = "modified — save to keep changes" if self._recipe_dirty else "saved"
            if self._recipe_path is None:
                text = f"{name} · {variant} · not saved yet"
            else:
                text = (
                    f"{name} · {variant} · {state} · "
                    f"{self._recipe_path.name}"
                )
            if self._recipe_source_warnings:
                count = len(self._recipe_source_warnings)
                text += f" · ⚠ {count} source warning(s)"
                self.recipe_status_label.setToolTip(
                    "\n".join(self._recipe_source_warnings)
                )
            else:
                self.recipe_status_label.setToolTip(
                    "Recipes preserve all effective paths, units, mappings, "
                    "tolerances, and feature membership."
                )
            self.recipe_status_label.setText(text)
            self.save_recipe_button.setEnabled(
                not self.job_is_running() and self._recipe_path is not None
            )

        @Slot(str)
        def _recipe_metadata_changed(self, _text: str) -> None:
            self._set_recipe_dirty()

        def _set_recipe_dirty(self) -> None:
            if self._loading_recipe:
                return
            self._recipe_dirty = True
            # Once edited, the saved source warning snapshot no longer exactly
            # describes the live configuration. The next save records a new one.
            self._recipe_source_warnings = ()
            self._update_recipe_status()

        def _recipe_default_path(self) -> str:
            anchor = self.output_picker.path() or self.base_picker.path()
            parent = Path(anchor).expanduser().parent if anchor else Path.cwd()
            raw_name = self.recipe_variant_edit.text().strip() or "baseline"
            safe_name = "_".join(raw_name.split())
            safe_name = "".join(
                character
                for character in safe_name
                if character.isalnum() or character in {"-", "_"}
            ) or "baseline"
            return str(parent / f"{safe_name}{FEATURE_RECIPE_SUFFIX}")

        def save_recipe_path(self, path: str | Path) -> Path:
            """Save the live form to ``path``; exposed for integration tests."""

            if self.job_is_running():
                raise RuntimeError(
                    "Wait for the current feature operation before saving its recipe."
                )
            self._pull_values()
            saved = write_feature_assembly_recipe(
                self.model.values,
                path,
                name=self.recipe_name_edit.text(),
                variant=self.recipe_variant_edit.text(),
            )
            self._recipe_path = saved
            self._recipe_dirty = False
            self._recipe_source_warnings = ()
            self._update_recipe_status()
            self.status_changed.emit(
                f"Saved Assembly recipe {saved.name}. This named variant can be "
                "reloaded locally or after copying its referenced files."
            )
            return saved

        @Slot()
        def _save_recipe(self) -> None:
            if self._recipe_path is None:
                self._save_recipe_as()
                return
            try:
                self.save_recipe_path(self._recipe_path)
            except Exception as exc:
                self._show_error(str(exc))

        @Slot()
        def _save_recipe_as(self) -> None:
            if self.job_is_running():
                self.status_changed.emit(
                    "Wait for the current feature operation before saving a recipe."
                )
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Assembly recipe",
                self._recipe_default_path(),
                "GRIM Assembly recipe (*.assembly.json);;JSON file (*.json);;All files (*)",
            )
            if not path:
                return
            try:
                self.save_recipe_path(path)
            except Exception as exc:
                self._show_error(str(exc))

        @Slot()
        def _load_recipe_dialog(self) -> None:
            if self.job_is_running():
                self.status_changed.emit(
                    "Wait for the current feature operation before loading a recipe."
                )
                return
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Assembly recipe",
                str(self._recipe_path or Path.cwd()),
                "GRIM Assembly recipe (*.assembly.json *.json);;All files (*)",
            )
            if not path:
                return
            if not self._confirm_dirty_recipe("load another recipe"):
                return
            try:
                self.load_recipe_path(path)
            except Exception as exc:
                self._show_error(str(exc))

        def _confirm_dirty_recipe(self, action: str) -> bool:
            """Offer Save/Discard/Cancel before losing edited recipe state."""

            if not self._recipe_dirty:
                return True
            answer = QMessageBox.warning(
                self,
                "Unsaved Assembly recipe",
                "This Assembly recipe has unsaved changes. Save them before you "
                f"{action}?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Discard:
                self._recipe_dirty = False
                self._recipe_source_warnings = ()
                self._update_recipe_status()
                return True
            if answer != QMessageBox.StandardButton.Save:
                return False
            if self._recipe_path is not None:
                try:
                    self.save_recipe_path(self._recipe_path)
                except Exception as exc:
                    self._show_error(str(exc))
                    return False
            else:
                self._save_recipe_as()
            return not self._recipe_dirty

        def request_close(self, parent: QWidget | None = None) -> bool:
            """Return True only when active work and unsaved recipes are resolved."""

            if self.is_busy():
                if self._active_kind in {"preview", "build"}:
                    self.request_cancel()
                else:
                    self.status_changed.emit(
                        "Feature validation is still running; wait before closing."
                    )
                return False
            return self._confirm_dirty_recipe("close GRIM")

        def load_recipe_path(
            self,
            path: str | Path,
            *,
            refresh: bool = True,
        ) -> LoadedFeatureAssemblyRecipe:
            """Restore one recipe and optionally parse its placement CSVs."""

            if self.job_is_running():
                raise RuntimeError(
                    "Wait for the current feature operation before loading a recipe."
                )
            loaded = read_feature_assembly_recipe(path)
            previous_preview = self._preview_is_current
            self._loading_recipe = True
            try:
                self.model = FeatureAssemblyFormModel(loaded.values)
                values = self.model.values
                self.base_picker.set_path(values.base_grim)
                self.surface_picker.set_path(values.surface_mesh)
                self.output_picker.set_path(values.output_grim)
                self.point_csv_picker.set_path(values.point_locations_csv)
                self.line_csv_picker.set_path(values.line_locations_csv)
                coordinate_index = self.coordinate_units.findData(
                    values.coordinate_units
                )
                surface_index = self.surface_units.findData(values.surface_units)
                if coordinate_index < 0 or surface_index < 0:
                    raise ValueError("Recipe units are unavailable in this GRIM build.")
                self.coordinate_units.setCurrentIndex(coordinate_index)
                self.surface_units.setCurrentIndex(surface_index)
                self.flip_normals.setChecked(values.flip_surface_normals)
                self.shadow.setChecked(values.shadow)
                self.skin_tol.setValue(values.skin_tol_m * 1.0e3)
                self.phase_tol.setValue(values.skin_phase_tol_deg)
                self.normal_tol.setValue(values.normal_tol_deg)
                self._set_validation_profile_from_values(values)
                self.expected_host_material.setText(values.expected_host_material)
                self.shadow_bias.setText(
                    "" if values.shadow_bias_m is None else f"{values.shadow_bias_m:.12g}"
                )
                # Display saved mappings immediately, while readiness still
                # requires the authoritative CSV re-scan before validation.
                self.point_mapping.set_dataset_ids(
                    tuple(values.point_datasets),
                    values.point_datasets,
                    values.point_host_materials,
                )
                self.line_mapping.set_dataset_ids(
                    tuple(values.line_datasets),
                    values.line_datasets,
                    values.line_host_materials,
                )
                self.spatial_feature_filter.clear()
                self.recipe_name_edit.setText(loaded.name)
                self.recipe_variant_edit.setText(loaded.variant)
                self._recipe_path = loaded.path
                self._recipe_dirty = False
                self._recipe_source_warnings = loaded.source_warnings
                self._preview_is_current = False
                self._validated_plan_current = False
                self._validation_warning_count = 0
                self._refresh_spatial_feature_tree()
                self._clear_validation_qa(
                    "Recipe loaded. Run Validate placements to refresh per-instance QA."
                )
                self._update_recipe_status()
                self._update_workflow_readiness()
            finally:
                self._loading_recipe = False

            if previous_preview:
                message = (
                    "Assembly recipe loaded — the previous 3-D preview is out of "
                    "date until this recipe is previewed or validated."
                )
                self.preview_stale.emit(message)

            warning_text = ""
            if loaded.source_warnings:
                warning_text = (
                    f" {len(loaded.source_warnings)} referenced source warning(s) "
                    "are listed on the recipe status tooltip."
                )
            self.status_changed.emit(
                f"Loaded Assembly recipe {loaded.name} · {loaded.variant}."
                + warning_text
            )

            placement_paths = tuple(
                value
                for value in (
                    loaded.values.point_locations_csv,
                    loaded.values.line_locations_csv,
                )
                if value
            )
            can_refresh = bool(
                refresh
                and placement_paths
                and all(Path(value).is_file() for value in placement_paths)
            )
            if can_refresh:
                try:
                    coerce_feature_workflow(self._service)
                except (RuntimeError, TypeError):
                    can_refresh = False
            if can_refresh:
                self.refresh_dataset_ids()
            return loaded

        def set_loaded_dataset_catalog(self, entries: Iterable[Any]) -> None:
            """Offer saved, file-backed GRIM rows without replacing Browse.

            The combined shell may pass its existing stable dataset catalog.
            Entries can be :class:`LoadedDatasetEntry` instances, mappings,
            ``(dataset_id, name, path[, dirty])`` tuples, or objects exposing
            equivalent attributes. Dirty/in-memory/missing entries remain
            visible as disabled explanations so users know to save first.
            """

            catalog = _coerce_loaded_dataset_catalog(entries)
            self._loaded_dataset_catalog = catalog
            self.base_picker.set_loaded_dataset_catalog(catalog)
            self.point_mapping.set_loaded_dataset_catalog(catalog)
            self.line_mapping.set_loaded_dataset_catalog(catalog)

        def loaded_dataset_catalog(self) -> tuple[LoadedDatasetEntry, ...]:
            """Return the last normalized catalog snapshot."""

            return self._loaded_dataset_catalog

        def set_base_grim(self, path: str) -> None:
            self.base_picker.set_path(path)
            self._base_path_changed()

        def set_surface_mesh(self, path: str) -> None:
            self.surface_picker.set_path(path)
            self._surface_path_changed()

        def set_point_csv(self, path: str, *, discover: bool = True) -> None:
            self.point_csv_picker.set_path(path)
            if discover:
                self._placement_csv_changed("point")
            else:
                if self.model.feature_selection_source_changed("point", path):
                    self.model.clear_feature_selection("point")
                self.model.values.point_locations_csv = _clean_path(path)
                self.model.invalidate_dataset_requirements("point")
                self.point_mapping.set_dataset_ids(())
                self._refresh_spatial_feature_tree()
                self._mark_preview_stale()

        def set_line_csv(self, path: str, *, discover: bool = True) -> None:
            self.line_csv_picker.set_path(path)
            if discover:
                self._placement_csv_changed("line")
            else:
                if self.model.feature_selection_source_changed("line", path):
                    self.model.clear_feature_selection("line")
                self.model.values.line_locations_csv = _clean_path(path)
                self.model.invalidate_dataset_requirements("line")
                self.line_mapping.set_dataset_ids(())
                self._refresh_spatial_feature_tree()
                self._mark_preview_stale()

        def set_output_grim(self, path: str) -> None:
            self.output_picker.set_path(path)
            self._output_path_changed()

        @Slot(str)
        def _loaded_dataset_notice(self, message: str) -> None:
            self.status_changed.emit(message)

        def _surface_binding_inputs(
            self,
        ) -> tuple[FeatureWorkflowAdapter, Path, Path, str]:
            """Return validated absolute inputs for an explicit binding action."""

            self._pull_values()
            values = self.model.values
            preflight = preflight_base_grim(
                values.base_grim, base_dir=values.base_dir
            )
            if not preflight.valid:
                raise ValueError(preflight.summary)
            if preflight.embedded_bor:
                raise ValueError(
                    "This clean-body GRIM embeds its BoR geometry; an external "
                    "solve-to-mesh binding is not required."
                )
            base = _resolved_user_path(values.base_grim, base_dir=values.base_dir)
            surface = _resolved_user_path(
                values.surface_mesh, base_dir=values.base_dir
            )
            if not surface.is_file() or surface.suffix.casefold() not in {
                ".stl", ".facet"
            }:
                raise ValueError(
                    "Choose the matching STL or .facet body surface first."
                )
            if values.surface_units not in UNIT_SCALE_M:
                raise ValueError(
                    "Choose the physical units of the selected surface mesh "
                    "before checking or creating its binding."
                )
            adapter = coerce_feature_workflow(self._service)
            return adapter, base, surface, str(values.surface_units)

        def _prompt_surface_binding_details(
            self,
            sidecar: Path,
        ) -> tuple[str, str] | None:
            """Collect reviewed IDs and a deliberate attestation in one dialog."""

            geometry_id = ""
            case_id = ""
            if isinstance(self._surface_binding_checked, Mapping):
                geometry_id = str(
                    self._surface_binding_checked.get("geometry_id", "")
                ).strip()
                case_id = str(
                    self._surface_binding_checked.get("attestation_case_id", "")
                ).strip()
            elif sidecar.is_file():
                # Prefill human IDs only. This is convenience, never validation;
                # the backend hashes and validates again after the dialog.
                try:
                    if sidecar.stat().st_size <= 1024 * 1024:
                        raw = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                        if isinstance(raw, Mapping):
                            geometry_id = str(raw.get("geometry_id", "")).strip()
                            case_id = str(
                                raw.get("attestation_case_id", "")
                            ).strip()
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
            dialog = _SurfaceBindingDialog(
                self,
                geometry_id=geometry_id,
                attestation_case_id=case_id,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            return dialog.binding_values()

        @Slot()
        def check_selected_surface_binding(self) -> None:
            """Explicitly verify the exact current external body registration."""

            if self.job_is_running():
                self.status_changed.emit("An Assembly operation is already running.")
                return
            try:
                adapter, base, surface, units = self._surface_binding_inputs()
                if not callable(adapter.check_surface_binding):
                    raise RuntimeError(
                        "The connected GHOST backend cannot check external-body "
                        "surface bindings."
                    )
            except Exception as exc:
                self._show_error(str(exc))
                return

            def operation() -> Mapping[str, Any]:
                binding, sidecar = adapter.check_surface_binding(
                    base,
                    surface,
                    surface_units=units,
                )
                return {
                    "binding": dict(binding),
                    "sidecar": str(sidecar),
                    "base": str(base),
                    "surface": str(surface),
                    "surface_units": units,
                    "identity_key": _surface_binding_identity_key(
                        base, surface, units
                    ),
                }

            self._start_operation("binding_check", operation)

        @Slot()
        def bind_selected_surface(self) -> None:
            """Create or refresh one explicitly reviewed exact-file binding."""

            if self.job_is_running():
                self.status_changed.emit("An Assembly operation is already running.")
                return
            try:
                adapter, base, surface, units = self._surface_binding_inputs()
                if not callable(adapter.write_surface_binding):
                    raise RuntimeError(
                        "The connected GHOST backend cannot create external-body "
                        "surface bindings."
                    )
                sidecar = _surface_binding_sidecar_path(surface)
                if sidecar is None:  # Defensive; surface is validated above.
                    raise RuntimeError("Could not resolve the canonical binding path.")
                details = self._prompt_surface_binding_details(sidecar)
                if details is None:
                    self.status_changed.emit("Surface binding was not changed.")
                    return
                geometry_id, case_id = details
                overwrite = sidecar.exists()
                if overwrite:
                    answer = QMessageBox.warning(
                        self,
                        "Replace reviewed surface binding?",
                        f"{sidecar.name} already exists. Replace it with a new "
                        "binding for the exact current body, mesh, units, and "
                        "reviewed IDs?",
                        QMessageBox.StandardButton.Yes
                        | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        self.status_changed.emit(
                            "Surface binding refresh cancelled; existing sidecar kept."
                        )
                        return
            except Exception as exc:
                self._show_error(str(exc))
                return

            def operation() -> Mapping[str, Any]:
                binding, written = adapter.write_surface_binding(
                    base,
                    surface,
                    surface_units=units,
                    geometry_id=geometry_id,
                    attestation_case_id=case_id,
                    attest_reviewed_registration=True,
                    overwrite=overwrite,
                )
                return {
                    "binding": dict(binding),
                    "sidecar": str(written),
                    "base": str(base),
                    "surface": str(surface),
                    "surface_units": units,
                    "identity_key": _surface_binding_identity_key(
                        base, surface, units
                    ),
                }

            self._start_operation("binding_write", operation)

        def _refresh_work_estimate(
            self,
            base_preflight: BaseGrimPreflight,
            *,
            validation_current: bool,
        ) -> AssemblyWorkEstimate:
            """Refresh the displayed estimate from current or reviewed inputs."""

            estimate = AssemblyWorkEstimate(available=False)
            plan = self.model.prepared_plan if validation_current else None
            if plan is not None:
                estimate = estimate_validated_assembly_plan_workload(plan)
            if not estimate.available:
                enabled_points = self.model.enabled_point_placement_ids
                enabled_lines = self.model.enabled_line_ids
                point_count = (
                    len(enabled_points)
                    if enabled_points is not None
                    else self.model.point_placement_count
                )
                line_count = (
                    len(enabled_lines)
                    if enabled_lines is not None
                    else self.model.line_path_count
                )
                segment_count = self.model.enabled_line_segment_count
                line_piece_count = max(
                    line_count,
                    segment_count * _PREVALIDATION_LINE_PIECES_PER_SEGMENT,
                )
                triangle_count, triangle_exact = surface_mesh_triangle_hint(
                    self.model.values.surface_mesh,
                    base_dir=self.model.values.base_dir,
                )
                estimate = estimate_assembly_workload(
                    look_count=(
                        int(base_preflight.azimuth_count)
                        * int(base_preflight.elevation_count)
                    ),
                    frequency_count=int(base_preflight.frequency_count),
                    point_count=point_count,
                    line_path_count=line_count,
                    line_segment_count=segment_count,
                    line_piece_count=line_piece_count,
                    mesh_triangle_count=triangle_count,
                    shadow_enabled=bool(self.model.values.shadow),
                    quantities_validated=False,
                    line_piece_count_exact=False,
                    mesh_triangle_count_exact=triangle_exact,
                )
            self._current_work_estimate = estimate
            self.work_estimate_label.setText(
                format_assembly_work_estimate(estimate)
            )
            return estimate

        def _set_readiness_checklist(self, groups) -> None:
            """Render the live, grouped run gate shown on the Review step."""

            tree = self.readiness_checklist
            tree.setUpdatesEnabled(False)
            try:
                tree.clear()
                for group_label, requirements in groups:
                    required_rows = [row for row in requirements if row[2]]
                    group_ready = bool(required_rows) and all(
                        bool(row[1]) for row in required_rows
                    )
                    group = QTreeWidgetItem(
                        [
                            str(group_label),
                            "✓ Ready" if group_ready else "○ Action needed",
                        ]
                    )
                    group.setData(0, Qt.ItemDataRole.UserRole, bool(group_ready))
                    tree.addTopLevelItem(group)
                    for label, ready, required in requirements:
                        if not required:
                            status = "— Not required"
                        else:
                            status = "✓ Ready" if ready else "○ Needed"
                        child = QTreeWidgetItem([str(label), status])
                        child.setData(0, Qt.ItemDataRole.UserRole, bool(ready))
                        child.setData(1, Qt.ItemDataRole.UserRole, bool(required))
                        group.addChild(child)
                    group.setExpanded(True)
            finally:
                tree.setUpdatesEnabled(True)

        def _update_workflow_readiness(self) -> None:
            """Keep the compact step summary and actions honest and actionable."""

            values = self.model.values
            values.base_grim = self.base_picker.path()
            values.output_grim = self.output_picker.path()
            values.surface_mesh = self.surface_picker.path()
            values.point_locations_csv = self.point_csv_picker.path()
            values.line_locations_csv = self.line_csv_picker.path()
            values.point_datasets = self.point_mapping.mapping()
            values.line_datasets = self.line_mapping.mapping()
            values.point_host_materials = self.point_mapping.host_materials()
            values.line_host_materials = self.line_mapping.host_materials()
            values.coordinate_units = str(self.coordinate_units.currentData())
            values.surface_units = str(self.surface_units.currentData())
            values.flip_surface_normals = self.flip_normals.isChecked()
            values.shadow = self.shadow.isChecked()
            values.skin_tol_m = self.skin_tol.value() * 1.0e-3
            values.skin_phase_tol_deg = self.phase_tol.value()
            values.normal_tol_deg = self.normal_tol.value()
            (
                values.allow_legacy_base_metadata,
                values.require_feature_manifests,
                values.require_body_mesh_certification,
            ) = self._validation_profile_flags()
            values.expected_host_material = (
                self.expected_host_material.text().strip()
            )

            point_selected = bool(values.point_locations_csv)
            line_selected = bool(values.line_locations_csv)
            placement_units_ready = bool(
                not (point_selected or line_selected)
                or values.coordinate_units in UNIT_SCALE_M
            )
            try:
                point_current = (
                    point_selected and self.model.requirements_look_current("point")
                )
                line_current = (
                    line_selected and self.model.requirements_look_current("line")
                )
            except Exception:
                point_current = False
                line_current = False

            point_ids = len(self.model.point_dataset_ids)
            line_ids = len(self.model.line_dataset_ids)
            point_count = self.model.point_placement_count
            line_count = self.model.line_path_count
            segment_count = self.model.line_segment_count
            missing_mappings = self.model.missing_dataset_mappings()
            point_missing = tuple(
                value.split(":", 1)[1]
                for value in missing_mappings
                if value.startswith("point:")
            )
            line_missing = tuple(
                value.split(":", 1)[1]
                for value in missing_mappings
                if value.startswith("line:")
            )
            active_point_ids = self.model.active_point_dataset_ids()
            active_line_ids = self.model.active_line_dataset_ids()
            enabled_point_count = len(
                self.model.enabled_point_placement_ids or ()
            )
            enabled_line_count = len(self.model.enabled_line_ids or ())
            try:
                adapter = coerce_feature_workflow(self._service)
                service_ready = True
            except (RuntimeError, TypeError):
                adapter = None
                service_ready = False

            def existing_file(path: str) -> bool:
                if not _clean_path(path):
                    return False
                try:
                    return _resolved_user_path(
                        path, base_dir=values.base_dir
                    ).is_file()
                except OSError:
                    return False

            def existing_grim_file(path: str) -> bool:
                if not _clean_path(path):
                    return False
                try:
                    resolved = _resolved_user_path(path, base_dir=values.base_dir)
                    return (
                        resolved.is_file()
                        and resolved.suffix.casefold() == ".grim"
                    )
                except OSError:
                    return False

            def existing_surface_file(path: str) -> bool:
                if not _clean_path(path):
                    return False
                try:
                    resolved = _resolved_user_path(path, base_dir=values.base_dir)
                    return (
                        resolved.is_file()
                        and resolved.suffix.casefold() in {".stl", ".facet"}
                    )
                except OSError:
                    return False

            point_unusable = tuple(
                dataset_id
                for dataset_id in active_point_ids
                if _clean_path(values.point_datasets.get(dataset_id))
                and not existing_grim_file(values.point_datasets.get(dataset_id, ""))
            )
            line_unusable = tuple(
                dataset_id
                for dataset_id in active_line_ids
                if _clean_path(values.line_datasets.get(dataset_id))
                and not existing_grim_file(values.line_datasets.get(dataset_id, ""))
            )

            if not point_selected:
                point_text = "No point CSV selected."
                point_tab = "Point features"
            elif not point_current:
                point_text = "Point CSV needs refresh."
                point_tab = "Point features · refresh"
            else:
                count_label = (
                    f"{point_count} placement(s)"
                    if point_count
                    else f"{point_ids} response type(s)"
                )
                mapping_label = (
                    f"{len(point_missing)} response(s) missing"
                    if point_missing
                    else (
                        f"{len(point_unusable)} response file(s) not found"
                        if point_unusable
                        else "response files ready"
                    )
                )
                point_text = f"Point CSV ready — {count_label}; {mapping_label}."
                point_tab = f"Point features · {point_count or point_ids}"

            if not line_selected:
                line_text = "No line CSV selected."
                line_tab = "Line features"
            elif not line_current:
                line_text = "Line CSV needs refresh."
                line_tab = "Line features · refresh"
            else:
                count_label = (
                    f"{line_count} path(s), {segment_count} segment(s)"
                    if line_count or segment_count
                    else f"{line_ids} response type(s)"
                )
                mapping_label = (
                    f"{len(line_missing)} response(s) missing"
                    if line_missing
                    else (
                        f"{len(line_unusable)} response file(s) not found"
                        if line_unusable
                        else "response files ready"
                    )
                )
                line_text = f"Line CSV ready — {count_label}; {mapping_label}."
                line_tab = f"Line features · {line_count or line_ids}"

            self.point_csv_summary.setText(point_text)
            self.line_csv_summary.setText(line_text)
            self.feature_tabs.setTabText(0, point_tab)
            self.feature_tabs.setTabText(1, line_tab)
            unit_text = self.coordinate_units.currentText()
            self.shared_units_label.setText(
                f"Shared units: {unit_text}. One point CSV may contain every point "
                "family; one line CSV may contain every ordered line chain."
            )

            selected_parts = []
            if point_selected:
                selected_parts.append(
                    f"{enabled_point_count}/{point_count or '?'} point placement(s) enabled"
                )
            if line_selected:
                selected_parts.append(
                    f"{enabled_line_count}/{line_count or '?'} line path(s) enabled / "
                    f"{segment_count or '?'} parsed segment(s)"
                )
            self.feature_summary_label.setText(
                "Selected: " + ("; ".join(selected_parts) if selected_parts else "none yet")
            )
            self.feature_summary_label.setVisible(point_selected and line_selected)
            strict_feature_library = bool(values.require_feature_manifests)
            production_profile = bool(
                strict_feature_library and not values.allow_legacy_base_metadata
            )
            certified_body_profile = bool(
                values.require_body_mesh_certification
            )
            if certified_body_profile:
                qa_mode = (
                    "Production — certified fine-mesh body, strict metadata, "
                    "certified response manifests"
                )
            elif production_profile:
                qa_mode = (
                    "External/HPC — strict metadata and response manifests; "
                    "local body certificate explicitly waived"
                )
            else:
                qa_mode = (
                    "legacy response compatibility (warnings shown after validation)"
                )
            host_id = values.expected_host_material.strip()
            try:
                effective_hosts = self.model.effective_host_materials()
                missing_host_materials = self.model.missing_host_material_mappings()
                host_mapping_error = ""
            except ValueError as exc:
                effective_hosts = {}
                missing_host_materials = tuple(
                    f"point:{value}" for value in active_point_ids
                ) + tuple(f"line:{value}" for value in active_line_ids)
                host_mapping_error = str(exc)
            explicit_host_count = sum(
                bool(str(value).strip())
                for value in (
                    *values.point_host_materials.values(),
                    *values.line_host_materials.values(),
                )
            )
            host_text = (
                f"default {host_id}"
                if host_id
                else (
                    f"{explicit_host_count} per-response override(s)"
                    if explicit_host_count
                    else "not set"
                )
            )
            if not production_profile and missing_host_materials:
                host_text += " (Legacy warning expected)"
            self.build_summary_label.setText(
                (
                    "; ".join(selected_parts)
                    + f"; QA: {qa_mode}; host: {host_text}"
                )
                if selected_parts
                else "Choose a point or line placement CSV."
            )
            self.advanced_section.header.setText(
                "Advanced placement checks · "
                + (
                    "Production certified-body validation"
                    if certified_body_profile
                    else (
                        "External/HPC reviewed-body validation"
                        if production_profile
                        else "legacy-library compatibility"
                    )
                )
            )

            has_body = bool(values.base_grim)
            base_preflight = preflight_base_grim(
                values.base_grim, base_dir=values.base_dir
            )
            body_ready = base_preflight.valid
            self.body_preview_help.setText(base_preflight.summary)
            has_placements = point_selected or line_selected
            has_enabled_features = bool(
                self.model.enabled_point_placement_ids
                or self.model.enabled_line_ids
                or (
                    not self.model.point_instances
                    and not self.model.line_instances
                    and has_placements
                )
            )
            placement_files_ready = (
                (not point_selected or existing_file(values.point_locations_csv))
                and (not line_selected or existing_file(values.line_locations_csv))
            )
            scans_current = (
                placement_files_ready
                and (not point_selected or point_current)
                and (not line_selected or line_current)
            )
            mappings_complete = not point_missing and not line_missing
            response_files_ready = (
                mappings_complete and not point_unusable and not line_unusable
            )
            surface_selected = bool(values.surface_mesh)
            surface_units_ready = bool(
                not surface_selected or values.surface_units in UNIT_SCALE_M
            )
            surface_file_ready = existing_surface_file(values.surface_mesh)
            surface_required = bool(
                body_ready
                and (base_preflight.requires_surface_mesh or self.shadow.isChecked())
            )
            surface_ready = (
                surface_file_ready
                if surface_required
                else not surface_selected or surface_file_ready
            )
            self._update_surface_dimensions_display(base_preflight)
            binding_status = assess_surface_binding_readiness(
                base_grim=values.base_grim,
                surface_mesh=values.surface_mesh,
                surface_units=values.surface_units,
                production_profile=production_profile,
                base_dir=values.base_dir,
                checked_key=self._surface_binding_checked_key,
                checked_binding=self._surface_binding_checked,
                error_key=self._surface_binding_error_key,
                check_error=self._surface_binding_error,
                tools_available=bool(
                    adapter is not None
                    and callable(adapter.check_surface_binding)
                ),
            )
            self.surface_binding_status.setText(binding_status.message)
            self.surface_binding_status.setProperty(
                "bindingState", binding_status.code
            )
            binding_action_ready = bool(
                not self.job_is_running()
                and binding_status.external_body
                and surface_file_ready
                and surface_units_ready
            )
            self.check_surface_binding_button.setEnabled(
                binding_action_ready
                and adapter is not None
                and callable(adapter.check_surface_binding)
                and binding_status.sidecar_path is not None
                and binding_status.sidecar_path.is_file()
            )
            self.bind_surface_button.setEnabled(
                binding_action_ready
                and adapter is not None
                and callable(adapter.write_surface_binding)
            )
            self.bind_surface_button.setText(
                "Refresh binding…"
                if binding_status.sidecar_path is not None
                and binding_status.sidecar_path.is_file()
                else "Bind body to mesh…"
            )
            host_material_ready = bool(
                not host_mapping_error
                and (not production_profile or not missing_host_materials)
            )
            has_output = bool(values.output_grim)
            output_ready = has_output
            if has_output:
                try:
                    self.model._validate_output_target()
                except (OSError, ValueError):
                    output_ready = False
            bias_text = self.shadow_bias.text().strip()
            settings_ready = True
            if bias_text:
                try:
                    bias_value = float(bias_text)
                    settings_ready = math.isfinite(bias_value) and bias_value >= 0.0
                except ValueError:
                    settings_ready = False
            self.build_summary_label.setVisible(
                has_placements and (point_current or line_current)
            )
            full_ready = all(
                (
                    service_ready,
                    body_ready,
                    has_placements,
                    has_enabled_features,
                    scans_current,
                    response_files_ready,
                    output_ready,
                    surface_ready,
                    placement_units_ready,
                    surface_units_ready,
                    binding_status.ready,
                    settings_ready,
                    host_material_ready,
                )
            )
            validation_current = bool(
                self._validated_plan_current
                and service_ready
                and adapter is not None
                and self.model.validated_plan_is_current(adapter)
            )
            if not validation_current:
                self._validated_plan_current = False
            self._refresh_work_estimate(
                base_preflight,
                validation_current=validation_current,
            )

            checks = [
                (service_ready, "GHOST backend"),
                (body_ready, "valid body GRIM"),
                (has_placements, "placements"),
                (has_enabled_features, "features enabled"),
                (scans_current and has_placements, "CSV read"),
                (
                    scans_current and response_files_ready and has_placements,
                    "response files",
                ),
            ]
            if surface_selected or surface_required:
                checks.append((surface_ready, "surface mesh"))
            if surface_selected:
                checks.append((surface_units_ready, "mesh units"))
            if has_placements:
                checks.append((placement_units_ready, "placement units"))
            if binding_status.external_body and production_profile:
                checks.append((binding_status.ready, "reviewed body binding"))
            if certified_body_profile:
                checks.append((validation_current, "body mesh certificate"))
            checks.append((host_material_ready, "host material IDs"))
            if not settings_ready:
                checks.append((False, "advanced settings"))
            checks.append((output_ready, "output"))
            self.readiness_label.setText(
                "   ".join(("✓" if ok else "○") + " " + label for ok, label in checks)
            )
            warnings_reviewed = bool(
                not self._validation_warning_count
                or self.validation_warning_ack.isChecked()
            )
            self._set_readiness_checklist(
                (
                    (
                        "Body (1)",
                        (
                            ("GHOST feature backend", service_ready, True),
                            ("Clean-body response", body_ready, True),
                            (
                                "Surface mesh",
                                surface_ready,
                                bool(surface_selected or surface_required),
                            ),
                            (
                                "Surface mesh units",
                                surface_units_ready,
                                bool(surface_selected),
                            ),
                            (
                                "Reviewed solve ↔ mesh binding",
                                binding_status.ready,
                                bool(binding_status.required),
                            ),
                        ),
                    ),
                    (
                        "Map Features (2)",
                        (
                            ("Placement CSV selected", has_placements, True),
                            ("Placement coordinate units", placement_units_ready, True),
                            ("Placement CSV read", scans_current, True),
                            ("At least one feature enabled", has_enabled_features, True),
                            ("Every dataset_id mapped", mappings_complete, True),
                            ("Mapped response files available", response_files_ready, True),
                            ("Host material / coating IDs", host_material_ready, True),
                        ),
                    ),
                    (
                        "Review (3)",
                        (
                            ("Advanced settings valid", settings_ready, True),
                            ("Output response selected", output_ready, True),
                            ("Placements validated", validation_current, True),
                            (
                                "Body mesh certificate",
                                validation_current,
                                bool(certified_body_profile),
                            ),
                            (
                                "Validation warnings reviewed",
                                warnings_reviewed,
                                bool(self._validation_warning_count),
                            ),
                        ),
                    ),
                )
            )

            if not service_ready:
                next_step = (
                    "GHOST feature backend unavailable; repair the integration "
                    "to continue."
                )
            elif not has_body:
                next_step = "Next: choose the clean-body .grim response."
            elif not body_ready:
                next_step = "Next: " + base_preflight.summary
            elif not surface_ready:
                next_step = (
                    "Next: choose the matching .stl or .facet surface mesh required "
                    "for this external 3-D body or shadowing."
                )
            elif surface_selected and not surface_units_ready:
                next_step = (
                    "Next: choose the physical units stored in the selected "
                    "surface mesh."
                )
            elif has_placements and not placement_units_ready:
                next_step = (
                    "Next: choose the coordinate units used by the selected "
                    "placement CSV(s)."
                )
            elif binding_status.required and not binding_status.ready:
                next_step = "Next: " + binding_status.message.lstrip("✗⚠○ ")
            elif not host_material_ready:
                next_step = "Next: " + (
                    host_mapping_error
                    if host_mapping_error
                    else (
                        "enter a per-response host material/coating ID for "
                        + ", ".join(missing_host_materials)
                        + ", or enter one global default."
                    )
                )
            elif not has_placements:
                next_step = "Next: choose a point or line placement CSV."
            elif not scans_current:
                next_step = "Next: refresh the selected CSV and correct any format error."
            elif not has_enabled_features:
                next_step = (
                    "Next: enable at least one item in Spatial Feature Configuration."
                )
            elif not mappings_complete:
                next_step = "Next: map every dataset_id to its OPN − FRD response."
            elif not response_files_ready:
                next_step = "Next: map each response to an existing .grim file."
            elif not settings_ready:
                next_step = "Next: enter a finite, non-negative shadow ray bias or leave it blank."
            elif not has_output:
                next_step = "Next: choose the assembled output file."
            elif not output_ready:
                next_step = "Next: choose an output that does not alias an Assembly input."
            elif not validation_current:
                next_step = (
                    "Ready to review: run Validate placements. Assembly is locked "
                    "until that current validation succeeds."
                )
            elif self._validation_warning_count and not self.validation_warning_ack.isChecked():
                next_step = (
                    "Validation passed with warnings. Review every warning and check "
                    "the one-time waiver before assembly."
                )
            else:
                next_step = "Validated and reviewed — ready to assemble and save."
            self.next_step_label.setText(next_step)

            busy = self.job_is_running()
            input_preview_supported = bool(
                service_ready
                and adapter is not None
                and callable(adapter.preview_inputs)
            )
            preview_possible = bool(
                (not has_body or body_ready)
                and placement_units_ready
                and surface_units_ready
                and any(
                    (
                        body_ready,
                        surface_file_ready,
                        point_selected and existing_file(values.point_locations_csv),
                        line_selected and existing_file(values.line_locations_csv),
                    )
                )
            )
            self.scan_button.setEnabled(not busy and service_ready and has_placements)
            self.input_preview_button.setEnabled(
                not busy
                and input_preview_supported
                and preview_possible
            )
            self.preview_button.setEnabled(not busy and full_ready)
            self.build_button.setEnabled(
                not busy and full_ready and validation_current and warnings_reviewed
            )
            self.point_clear_button.setEnabled(not busy and point_selected)
            self.line_clear_button.setEnabled(not busy and line_selected)

        def job_is_running(self) -> bool:
            return bool(self._thread is not None and self._thread.isRunning())

        def is_busy(self) -> bool:
            """Return whether discovery, validation, or assembly is active."""

            return self.job_is_running()

        def busy_operation(self) -> str:
            return str(self._active_kind)

        def can_close(self) -> bool:
            """Closing is safe after any active worker reaches a safe boundary."""

            return not self.is_busy()

        @Slot()
        def request_cancel(self) -> None:
            """Request cooperative cancellation of validation or assembly."""

            if self._active_kind not in {"preview", "build"} or self._worker is None:
                return
            self._worker.request_cancel()
            self.cancel_operation_button.setEnabled(False)
            if self._active_kind == "preview":
                self.operation_progress.setFormat("Cancelling validation safely…")
                self.status_changed.emit(
                    "Validation cancellation requested. Finishing the current safe "
                    "check; no reviewed plan will be retained."
                )
            else:
                self.operation_progress.setFormat("Cancelling assembly safely…")
                self.status_changed.emit(
                    "Assembly cancellation requested. Finishing the current safe "
                    "numerical step; no partial output will be published."
                )

        def closeEvent(self, event: Any) -> None:
            if not self.request_close(self):
                event.ignore()
                return
            super().closeEvent(event)

        def _base_path_changed(self) -> None:
            self._mark_preview_stale()
            base = self.base_picker.path()
            if base and not self.output_picker.path():
                source = Path(base)
                suggestion = source.with_name(source.stem + "_features.grim")
                self.output_picker.set_path(str(suggestion))
            self.model.values.base_grim = base
            self._refresh_spatial_feature_tree()
            self._update_workflow_readiness()

        def _input_setting_changed(self, *_args: Any) -> None:
            self._mark_preview_stale()

        def _surface_path_changed(self) -> None:
            """Default a newly selected mesh to physically safer shadowing."""

            selected = self.surface_picker.path()
            previous = _clean_path(self.model.values.surface_mesh)
            newly_selected = bool(selected and selected != previous)
            if newly_selected and not self._loading_recipe:
                self.shadow.blockSignals(True)
                try:
                    self.shadow.setChecked(True)
                finally:
                    self.shadow.blockSignals(False)
            self.model.values.surface_mesh = selected
            self._mark_preview_stale()

        def _output_path_changed(self) -> None:
            self._mark_preview_stale()

        @Slot()
        def _mark_preview_stale(self, *_args: Any) -> None:
            if self._loading_recipe:
                return
            had_current_review = bool(
                self._preview_is_current or self._validated_plan_current
            )
            self.model.invalidate_prepared_plan()
            self._preview_is_current = False
            self._validated_plan_current = False
            self._clear_validation_qa(
                "Inputs changed. Run Validate placements to refresh per-instance QA."
            )
            self._set_recipe_dirty()
            self._update_workflow_readiness()
            if not had_current_review:
                return
            message = (
                "Inputs changed — the 3-D preview is out of date. Preview "
                "inputs again, or validate placements for an authoritative preview."
            )
            self.status_changed.emit(message)
            self.preview_stale.emit(message)

        def _placement_csv_changed(self, kind: str) -> None:
            selected_path = (
                self.point_csv_picker.path()
                if kind == "point"
                else self.line_csv_picker.path()
            )
            if self.model.feature_selection_source_changed(kind, selected_path):
                self.model.clear_feature_selection(kind)
            if kind == "point":
                self.model.values.point_locations_csv = selected_path
            else:
                self.model.values.line_locations_csv = selected_path

            if self.model.requirements_look_current(kind):
                self._update_workflow_readiness()
                return

            self._mark_preview_stale()
            self.model.invalidate_dataset_requirements(kind)
            if kind == "point":
                self.point_mapping.set_dataset_ids(())
            else:
                self.line_mapping.set_dataset_ids(())
            self._refresh_spatial_feature_tree()
            picker = self.point_csv_picker if kind == "point" else self.line_csv_picker
            if picker.path():
                self.refresh_dataset_ids()
            else:
                self.status_changed.emit(
                    f"{kind.capitalize()} placements removed from this build."
                )
                self._update_workflow_readiness()

        def _clear_placement_csv(self, kind: str) -> None:
            picker = self.point_csv_picker if kind == "point" else self.line_csv_picker
            picker.set_path("")
            self.model.clear_feature_selection(kind)
            self._placement_csv_changed(kind)

        def _mapping_changed(self) -> None:
            self.model.values.point_datasets = self.point_mapping.mapping()
            self.model.values.line_datasets = self.line_mapping.mapping()
            self.model.values.point_host_materials = (
                self.point_mapping.host_materials()
            )
            self.model.values.line_host_materials = (
                self.line_mapping.host_materials()
            )
            self._refresh_spatial_feature_tree()
            self._mark_preview_stale()
            missing = [
                f"point:{dataset_id}" for dataset_id in self.point_mapping.missing_ids()
            ]
            missing.extend(
                f"line:{dataset_id}" for dataset_id in self.line_mapping.missing_ids()
            )
            if missing:
                self.status_changed.emit(
                    "Response mapping incomplete — choose an OPN-FRD .grim for: "
                    + ", ".join(missing)
                )
            elif self.point_mapping.dataset_ids or self.line_mapping.dataset_ids:
                self.status_changed.emit(
                    "All discovered dataset IDs are mapped. Next, validate "
                    "placements and inspect them in the 3-D Assembly view."
                )
            self._update_workflow_readiness()

        def _toggle_schema_help(self, kind: str, checked: bool) -> None:
            if kind == "point":
                button = self.point_format_button
                label = self.point_schema_label
                help_label = self.point_help_label
            else:
                button = self.line_format_button
                label = self.line_schema_label
                help_label = self.line_help_label
            label.setVisible(bool(checked))
            help_label.setVisible(bool(checked))
            button.setText("Hide guide" if checked else "CSV guide")

        def _remember_surface_dimensions(self, preview: Any) -> None:
            """Cache physical mesh spans only for the exact previewed selection."""

            geometry = getattr(preview, "preview_geometry", preview)
            triangles = getattr(geometry, "surface_triangles_cad_m", None)
            text = _surface_dimensions_summary(
                triangles,
                surface_units=self.model.values.surface_units,
            )
            try:
                key = _surface_preview_identity_key(
                    self.model.values.surface_mesh,
                    self.model.values.surface_units,
                    base_dir=self.model.values.base_dir,
                )
            except OSError:
                key = None
            self._surface_dimensions_key = key if text else None
            self._surface_dimensions_text = text

        def _update_surface_dimensions_display(
            self,
            base_preflight: BaseGrimPreflight | None = None,
        ) -> None:
            values = self.model.values
            if not _clean_path(values.surface_mesh):
                preflight = base_preflight or preflight_base_grim(
                    values.base_grim, base_dir=values.base_dir
                )
                if preflight.embedded_bor:
                    text = (
                        "Embedded BoR geometry is authoritative and already "
                        "stored in meters in the clean-body response."
                    )
                elif preflight.valid and preflight.requires_surface_mesh:
                    text = (
                        "No external mesh selected; choose the matching mesh "
                        "before physical body dimensions can be interpreted."
                    )
                else:
                    text = (
                        "No external mesh selected; physical mesh dimensions "
                        "are not available yet."
                    )
                self.surface_dimensions_label.setText(text)
                return
            if values.surface_units not in UNIT_SCALE_M:
                self.surface_dimensions_label.setText(
                    "Not interpreted: choose the physical units stored in this "
                    "mesh before previewing, binding, or building."
                )
                return
            if (
                self._surface_dimensions_key is None
                or not self._surface_dimensions_text
            ):
                self.surface_dimensions_label.setText(
                    "Not interpreted yet: click Preview geometry to confirm the "
                    "selected units and physical x/y/z dimensions in meters."
                )
                return
            try:
                current_key = _surface_preview_identity_key(
                    values.surface_mesh,
                    values.surface_units,
                    base_dir=values.base_dir,
                )
            except OSError:
                current_key = None
            if (
                current_key is not None
                and current_key == self._surface_dimensions_key
                and self._surface_dimensions_text
            ):
                self.surface_dimensions_label.setText(
                    self._surface_dimensions_text
                )
                return
            self.surface_dimensions_label.setText(
                "Not interpreted yet: click Preview geometry to confirm the "
                "selected units and physical x/y/z dimensions in meters."
            )

        def _save_template(self, kind: str) -> None:
            default_name = (
                "point_features_template.csv"
                if kind == "point"
                else "line_features_template.csv"
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Save blank {kind} placement CSV template",
                default_name,
                "CSV placement file (*.csv);;All files (*)",
            )
            if not path:
                return
            try:
                saved = write_placement_csv_template(kind, path)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self.status_changed.emit(
                f"Saved blank {kind} template: {saved}. Add placement rows, "
                "then choose that CSV above to validate it."
            )

        def _pull_values(self) -> None:
            values = self.model.values
            values.base_grim = self.base_picker.path()
            values.output_grim = self.output_picker.path()
            values.coordinate_units = str(self.coordinate_units.currentData())
            values.surface_mesh = self.surface_picker.path()
            values.surface_units = str(self.surface_units.currentData())
            values.flip_surface_normals = self.flip_normals.isChecked()
            values.shadow = self.shadow.isChecked()
            bias = self.shadow_bias.text().strip()
            try:
                values.shadow_bias_m = None if not bias else float(bias)
            except ValueError as exc:
                raise ValueError("Shadow bias must be a number in meters or blank.") from exc
            values.point_locations_csv = self.point_csv_picker.path()
            values.line_locations_csv = self.line_csv_picker.path()
            values.point_datasets = self.point_mapping.mapping()
            values.line_datasets = self.line_mapping.mapping()
            values.point_host_materials = self.point_mapping.host_materials()
            values.line_host_materials = self.line_mapping.host_materials()
            values.skin_tol_m = self.skin_tol.value() * 1.0e-3
            values.skin_phase_tol_deg = self.phase_tol.value()
            values.normal_tol_deg = self.normal_tol.value()
            (
                values.allow_legacy_base_metadata,
                values.require_feature_manifests,
                values.require_body_mesh_certification,
            ) = self._validation_profile_flags()
            values.expected_host_material = (
                self.expected_host_material.text().strip()
            )

        def _show_error(self, text: str) -> None:
            message = str(text).strip() or "Feature assembly failed."
            self.status_changed.emit(message)
            self.build_failed.emit(message)
            self._update_workflow_readiness()

        @Slot()
        def refresh_dataset_ids(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                if not (
                    self.model.values.point_locations_csv
                    or self.model.values.line_locations_csv
                ):
                    self.model.update_dataset_requirements(
                        {"point_dataset_ids": (), "line_dataset_ids": ()}
                    )
                    self._apply_requirements_to_tables()
                    self.status_changed.emit("Select a placement CSV to discover IDs.")
                    return
                adapter = coerce_feature_workflow(self._service)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._discovery_paths = (
                self.model.values.point_locations_csv,
                self.model.values.line_locations_csv,
            )
            self._start_operation(
                "discover",
                lambda: self.model.query_dataset_ids(adapter),
            )

        def _apply_requirements_to_tables(self) -> None:
            self.point_mapping.set_dataset_ids(
                self.model.point_dataset_ids,
                self.model.values.point_datasets,
                self.model.values.point_host_materials,
            )
            self.line_mapping.set_dataset_ids(
                self.model.line_dataset_ids,
                self.model.values.line_datasets,
                self.model.values.line_host_materials,
            )
            self._refresh_spatial_feature_tree()

        def _refresh_spatial_feature_tree(self) -> None:
            self.spatial_feature_tree.set_configuration(self.model)
            self.point_mapping.set_required_dataset_ids(
                self.model.active_point_dataset_ids()
            )
            self.line_mapping.set_required_dataset_ids(
                self.model.active_line_dataset_ids()
            )
            self._update_spatial_selection_summary()

        def _clear_validation_qa(self, message: str) -> None:
            self._validated_plan_current = False
            self._validation_warning_count = 0
            self.validation_qa_table.setRowCount(0)
            self.validation_qa_label.setText(str(message))
            self.validation_warning_label.clear()
            self.validation_warning_label.setVisible(False)
            self.validation_warning_ack.blockSignals(True)
            self.validation_warning_ack.setChecked(False)
            self.validation_warning_ack.blockSignals(False)
            self.validation_warning_ack.setVisible(False)

        def _show_validation_qa(self, plan: Any) -> None:
            """Present backend-produced pass records without redoing physics."""

            validation_warnings = tuple(
                str(value).strip()
                for value in (getattr(plan, "validation_warnings", ()) or ())
                if str(value).strip()
            )
            self._validation_warning_count = len(validation_warnings)
            self.validation_warning_ack.blockSignals(True)
            self.validation_warning_ack.setChecked(False)
            self.validation_warning_ack.blockSignals(False)
            self.validation_warning_ack.setVisible(bool(validation_warnings))
            if validation_warnings:
                self.validation_warning_label.setText(
                    "⚠ VALIDATION PASSED WITH RELEASE WARNINGS:\n• "
                    + "\n• ".join(validation_warnings)
                    + "\nResolve them where possible. Otherwise review each warning "
                    "and use the one-time waiver below before assembly."
                )
                self.validation_warning_label.setVisible(True)
            else:
                self.validation_warning_label.clear()
                self.validation_warning_label.setVisible(False)

            raw_rows: list[tuple[str, Mapping[str, Any]]] = []
            for kind, attribute in (
                ("line", "line_records"),
                ("point", "point_records"),
            ):
                records = getattr(plan, attribute, ()) or ()
                for record in records:
                    if isinstance(record, Mapping):
                        raw_rows.append((kind, record))
            self.validation_qa_table.setRowCount(len(raw_rows))
            skin_limit = getattr(plan, "skin_limit_m", None)
            try:
                skin_limit_value = float(skin_limit)
            except (TypeError, ValueError):
                skin_limit_value = float("nan")
            normal_limit = float(self.model.values.normal_tol_deg)
            offsets: list[float] = []
            normal_errors: list[float] = []
            not_illuminated_count = 0
            for row, (kind, record) in enumerate(raw_rows):
                identifier_key = "line_id" if kind == "line" else "placement_id"
                identifier = str(record.get(identifier_key, "")).strip()
                dataset_id = str(record.get("dataset_id", "")).strip()
                offset_raw = record.get(
                    "max_skin_offset_m" if kind == "line" else "skin_offset_m"
                )
                try:
                    offset = float(offset_raw)
                except (TypeError, ValueError):
                    offset = float("nan")
                if math.isfinite(offset):
                    offsets.append(offset)
                    ratio = (
                        0.0
                        if skin_limit_value == 0.0 and offset == 0.0
                        else (
                            100.0 * offset / skin_limit_value
                            if math.isfinite(skin_limit_value)
                            and skin_limit_value > 0.0
                            else float("nan")
                        )
                    )
                    offset_text = f"{offset * 1e3:.4g} mm"
                    if math.isfinite(ratio):
                        offset_text += f" ({ratio:.1f}%)"
                else:
                    offset_text = "checked"
                normal_raw = record.get("max_normal_error_deg")
                try:
                    normal_error = float(normal_raw)
                except (TypeError, ValueError):
                    normal_error = float("nan")
                if math.isfinite(normal_error):
                    normal_errors.append(normal_error)
                    normal_text = f"{normal_error:.3g}° / {normal_limit:.3g}°"
                else:
                    normal_text = "outward ✓"
                illumination_raw = record.get(
                    "illuminated_requested_look_count"
                )
                requested_raw = record.get("requested_look_count")
                try:
                    illuminated_looks = int(illumination_raw)
                except (TypeError, ValueError):
                    illuminated_looks = -1
                try:
                    requested_looks = int(requested_raw)
                except (TypeError, ValueError):
                    requested_looks = -1
                not_illuminated = (
                    "illuminated_requested_look_count" in record
                    and illuminated_looks == 0
                )
                if not_illuminated:
                    not_illuminated_count += 1
                result_text = (
                    "WARN: not illuminated" if not_illuminated else "PASS"
                )
                values = (
                    "Line" if kind == "line" else "Point",
                    identifier,
                    dataset_id,
                    offset_text,
                    normal_text,
                    result_text,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, (kind, identifier))
                    if column == 5:
                        if not_illuminated:
                            requested_text = (
                                str(requested_looks)
                                if requested_looks >= 0
                                else "the"
                            )
                            item.setToolTip(
                                "Not illuminated: 0 of " + requested_text
                                + " requested looks illuminate this enabled "
                                "feature, so it contributes zero on this radar "
                                "grid. Physical placement checks passed; review "
                                "the normal/aperture or accept the one-time "
                                "release-warning waiver if this is intentional."
                            )
                        else:
                            item.setToolTip(
                                "Passed the authoritative skin, outward-normal, "
                                "frame, response-mapping, illumination, and "
                                "source-integrity checks."
                            )
                    self.validation_qa_table.setItem(row, column, item)

            if not raw_rows:
                self.validation_qa_label.setText(
                    "Validation completed, but this service returned no per-instance "
                    "QA records."
                )
                return
            point_count = sum(kind == "point" for kind, _record in raw_rows)
            line_count = len(raw_rows) - point_count
            summary = (
                f"✓ {len(raw_rows)} enabled placement(s) passed physical checks: "
                f"{point_count} point, {line_count} line."
            )
            if not_illuminated_count:
                summary += (
                    f" ⚠ {not_illuminated_count} WARN/not illuminated and "
                    "contributes zero on this radar grid."
                )
            if validation_warnings:
                summary += (
                    f" ⚠ {len(validation_warnings)} production QA warning(s) "
                    "require review."
                )
            if offsets:
                summary += f" Worst skin offset {max(offsets) * 1e3:.4g} mm."
            if normal_errors:
                summary += f" Worst recorded normal error {max(normal_errors):.3g}°."
            summary += " Click a row to find that instance above."
            self.validation_qa_label.setText(summary)

        @Slot(int, int)
        def _qa_row_clicked(self, row: int, _column: int) -> None:
            item = self.validation_qa_table.item(int(row), 0)
            payload = None if item is None else item.data(Qt.ItemDataRole.UserRole)
            if (
                isinstance(payload, (tuple, list))
                and len(payload) == 2
                and self.spatial_feature_tree.select_instance(
                    str(payload[0]), str(payload[1])
                )
            ):
                self.status_changed.emit(
                    f"Selected validated {payload[0]} feature {payload[1]!r} "
                    "in Spatial Feature Configuration."
                )

        def _update_spatial_selection_summary(self) -> None:
            self.spatial_selection_summary.setText(
                self.model.feature_selection_summary(
                    max_disabled_ids_per_kind=FEATURE_SELECTION_DISPLAY_ID_LIMIT
                )
            )
            self.copy_spatial_selection_button.setEnabled(
                bool(self.model.point_instances or self.model.line_instances)
            )

        @Slot()
        def _copy_full_spatial_selection_summary(self) -> None:
            summary = self.model.feature_selection_summary()
            QApplication.clipboard().setText(summary)
            self.status_changed.emit(
                "Copied the full spatial feature selection summary to the clipboard."
            )

        @Slot()
        def _spatial_selection_changed(self) -> None:
            point_ids, line_ids = self.spatial_feature_tree.excluded_ids()
            try:
                self.model.set_excluded_feature_instances(
                    point_ids=point_ids,
                    line_ids=line_ids,
                )
            except Exception as exc:
                self._show_error(str(exc))
                self._refresh_spatial_feature_tree()
                return
            self._update_spatial_selection_summary()
            self.point_mapping.set_required_dataset_ids(
                self.model.active_point_dataset_ids()
            )
            self.line_mapping.set_required_dataset_ids(
                self.model.active_line_dataset_ids()
            )
            self._mark_preview_stale()
            if not (
                self.model.enabled_point_placement_ids
                or self.model.enabled_line_ids
            ):
                self.status_changed.emit(
                    "No spatial features are enabled. Enable at least one point "
                    "placement or line path before validation or assembly."
                )

        @Slot()
        def preview_inputs(self) -> None:
            """Show geometry/locations without requiring response mappings."""

            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
                if not callable(adapter.preview_inputs):
                    raise RuntimeError(
                        "This GHOST backend does not support staged input preview. "
                        "Use Validate placements after mapping responses."
                    )
                if not any(
                    (
                        self.model.values.base_grim,
                        self.model.values.surface_mesh,
                        self.model.values.point_locations_csv,
                        self.model.values.line_locations_csv,
                    )
                ):
                    raise ValueError(
                        "Choose a clean-body GRIM, body mesh, or placement CSV "
                        "to preview."
                    )
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._start_operation(
                "input_preview", lambda: self.model.prepare_input_preview(adapter)
            )

        @Slot()
        def validate_and_preview(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._clear_validation_qa(
                "Validation is running. Assembly remains locked until it succeeds."
            )
            self.model.invalidate_prepared_plan()
            self._update_workflow_readiness()
            self._start_operation(
                "preview",
                lambda cancel_check, progress_callback: self.model.prepare_preview(
                    adapter,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                ),
                cooperative=True,
            )

        def _validated_build_work_estimate(self) -> AssemblyWorkEstimate:
            plan = self.model.prepared_plan
            if plan is None:
                return AssemblyWorkEstimate(available=False)
            return estimate_validated_assembly_plan_workload(plan)

        @Slot()
        def assemble_and_save(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
                if not self._validated_plan_current:
                    raise ValueError(
                        "Run Validate placements and review its current QA result "
                        "before assembling."
                    )
                if not self.model.validated_plan_is_current(
                    adapter, verify_sources=True
                ):
                    self._validated_plan_current = False
                    raise ValueError(
                        "An Assembly input changed after validation. Validate and "
                        "review the current configuration again."
                    )
                if (
                    self._validation_warning_count
                    and not self.validation_warning_ack.isChecked()
                ):
                    raise ValueError(
                        "Review the validation warnings and check the one-time "
                        "warning waiver before assembling."
                    )
            except Exception as exc:
                self._show_error(str(exc))
                return
            output = _normalized_grim_output_path(
                self.model.values.output_grim,
                base_dir=self.model.values.base_dir,
            )
            features_only_output = _features_only_grim_output_path(
                self.model.values.output_grim,
                base_dir=self.model.values.base_dir,
            )
            existing_outputs = [
                path for path in (output, features_only_output) if path.exists()
            ]
            if existing_outputs:
                listed = "\n".join(f"• {path.name}" for path in existing_outputs)
                answer = QMessageBox.question(
                    self,
                    "Replace Assembly output files?",
                    "Assembly publishes the body-plus-features response and a "
                    "feature-only delta sibling. The following existing file(s) "
                    "will be replaced together:\n\n"
                    + listed
                    + "\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self.status_changed.emit(
                        "Assembly cancelled; existing output files kept."
                    )
                    return
            build_estimate = self._validated_build_work_estimate()
            prepared_plan = self.model.prepared_plan
            plan_warnings = tuple(
                str(value)
                for value in (
                    getattr(prepared_plan, "validation_warnings", ()) or ()
                )
            )
            sealed_workload_warning = any(
                value.startswith(WORKLOAD_REVIEW_WARNING_PREFIX)
                for value in plan_warnings
            )
            if (
                assembly_build_confirmation_required(build_estimate)
                and not sealed_workload_warning
            ):
                answer = QMessageBox.warning(
                    self,
                    "Review large Assembly workload",
                    format_assembly_work_estimate(build_estimate)
                    + "\n\nThese are operation counts, not an elapsed-time "
                    "prediction. Continue with this reviewed plan? You can "
                    "cancel cooperatively without publishing a partial output.",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self.status_changed.emit(
                        "Large Assembly workload cancelled before computation; "
                        "existing output files kept."
                    )
                    return
            acknowledged_plan_sha256 = (
                str(getattr(prepared_plan, "prepared_plan_sha256", "")).strip()
                if self._validation_warning_count
                and self.validation_warning_ack.isChecked()
                else None
            )
            self._start_operation(
                "build",
                lambda cancel_check, progress_callback: self.model.assemble_validated(
                    adapter,
                    acknowledged_plan_sha256=acknowledged_plan_sha256,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                ),
                cooperative=True,
            )

        def _set_busy(self, busy: bool) -> None:
            # Keep step navigation, progress, and cooperative cancellation live
            # while preventing edits that would invalidate the running plan.
            for widget in self._busy_form_widgets:
                widget.setEnabled(not busy)
            self.load_recipe_button.setEnabled(not busy)
            self.save_recipe_as_button.setEnabled(not busy)
            if busy:
                self.scan_button.setEnabled(False)
                self.input_preview_button.setEnabled(False)
                self.preview_button.setEnabled(False)
                self.build_button.setEnabled(False)
                self.save_recipe_button.setEnabled(False)
                self.operation_progress.setVisible(True)
                if self._active_kind in {"preview", "build"}:
                    self.operation_progress.setRange(0, 100)
                    self.operation_progress.setValue(0)
                    if self._active_kind == "preview":
                        self.operation_progress.setFormat(
                            "0% · Checking Assembly inputs"
                        )
                        self.cancel_operation_button.setText("Cancel validation")
                    else:
                        self.operation_progress.setFormat("0% · Preparing assembly")
                        self.cancel_operation_button.setText("Cancel assembly")
                    self.cancel_operation_button.setVisible(True)
                    self.cancel_operation_button.setEnabled(True)
                else:
                    self.operation_progress.setRange(0, 0)
                    self.operation_progress.setFormat("Working…")
                    self.cancel_operation_button.setVisible(False)
                    self.cancel_operation_button.setEnabled(False)
            else:
                self.operation_progress.setVisible(False)
                self.operation_progress.setRange(0, 100)
                self.cancel_operation_button.setVisible(False)
                self.cancel_operation_button.setEnabled(False)
                self._update_recipe_status()
                self._update_workflow_readiness()

        def _start_operation(
            self,
            kind: str,
            operation: Callable[..., Any],
            *,
            cooperative: bool = False,
        ) -> None:
            if self.job_is_running():
                raise RuntimeError("A feature operation is already running.")
            thread = QThread(self)
            worker = _OperationWorker(operation, cooperative=cooperative)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.succeeded.connect(self._operation_succeeded)
            worker.failed.connect(self._operation_failed)
            worker.cancelled.connect(self._operation_cancelled)
            worker.progress.connect(self._operation_progress)
            worker.succeeded.connect(thread.quit)
            worker.failed.connect(thread.quit)
            worker.cancelled.connect(thread.quit)
            worker.succeeded.connect(worker.deleteLater)
            worker.failed.connect(worker.deleteLater)
            worker.cancelled.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._operation_thread_finished)
            self._thread = thread
            self._worker = worker
            self._active_kind = kind
            self._set_busy(True)
            status = {
                "discover": "Reading placement CSV schemas…",
                "binding_check": "Checking exact body-to-mesh binding…",
                "binding_write": "Writing and verifying reviewed body binding…",
                "input_preview": (
                    "Loading body geometry and placement locations for visual preview…"
                ),
                "preview": "Validating placements and preparing preview…",
                "build": "Assembling coherent feature responses…",
            }[kind]
            self.status_changed.emit(status)
            thread.start()

        @Slot(int, str)
        def _operation_progress(self, percent: int, message: str) -> None:
            if self._active_kind not in {"preview", "build"}:
                return
            value = max(0, min(100, int(percent)))
            self.operation_progress.setRange(0, 100)
            self.operation_progress.setValue(value)
            self.operation_progress.setFormat(f"{value}% · {message}")

        @Slot(str)
        def _operation_cancelled(self, text: str) -> None:
            if self._active_kind == "preview":
                self.model.invalidate_prepared_plan()
                self._preview_is_current = False
                stale_message = (
                    "Validation cancelled — the Assembly preview is not a current "
                    "authoritative review."
                )
                self._clear_validation_qa(
                    "Validation cancelled. Assembly remains locked; run Validate "
                    "placements again when ready."
                )
                self.status_changed.emit(
                    text
                    or "Placement validation cancelled; no reviewed plan was retained."
                )
                self.preview_stale.emit(stale_message)
                return
            self.status_changed.emit(text or "Assembly cancelled; existing output kept.")

        @Slot(object)
        def _operation_succeeded(self, result: Any) -> None:
            kind = self._active_kind
            if kind in {"binding_check", "binding_write"}:
                try:
                    values = self.model.values
                    current_base = _resolved_user_path(
                        self.base_picker.path(), base_dir=values.base_dir
                    )
                    current_surface = _resolved_user_path(
                        self.surface_picker.path(), base_dir=values.base_dir
                    )
                    current_units = str(self.surface_units.currentData())
                    current_key = _surface_binding_identity_key(
                        current_base, current_surface, current_units
                    )
                    result_key = result.get("identity_key")
                    same_selection = (
                        _path_key(current_base) == _path_key(Path(result["base"]))
                        and _path_key(current_surface)
                        == _path_key(Path(result["surface"]))
                        and current_units == str(result["surface_units"])
                    )
                    if not same_selection or current_key != result_key:
                        raise RuntimeError(
                            "Body binding inputs changed while the operation was "
                            "finishing. Check the current selection again."
                        )
                    binding = result["binding"]
                    if not isinstance(binding, Mapping):
                        raise RuntimeError(
                            "The backend returned an invalid binding result."
                        )
                except Exception as exc:
                    self._surface_binding_error_key = None
                    self._surface_binding_error = str(exc)
                    self.status_changed.emit(str(exc))
                    self._update_workflow_readiness()
                    return
                self._surface_binding_checked_key = current_key
                self._surface_binding_checked = dict(binding)
                self._surface_binding_error_key = None
                self._surface_binding_error = ""
                if kind == "binding_write":
                    self.model.invalidate_prepared_plan()
                    self._preview_is_current = False
                    self._clear_validation_qa(
                        "Body binding refreshed. Validate placements again so the "
                        "reviewed plan includes the new exact sidecar."
                    )
                    self.preview_stale.emit(
                        "The body binding changed; the prior Assembly review is stale."
                    )
                    verb = "Created and verified"
                else:
                    verb = "Verified"
                self.status_changed.emit(
                    f"✓ {verb} body binding: geometry "
                    f"{binding.get('geometry_id')!r}; registration "
                    f"{binding.get('attestation_case_id')!r}."
                )
                self._update_workflow_readiness()
            elif kind == "discover":
                current_paths = (
                    self.point_csv_picker.path(),
                    self.line_csv_picker.path(),
                )
                if current_paths != self._discovery_paths:
                    self.model.invalidate_dataset_requirements()
                    self._apply_requirements_to_tables()
                    self.status_changed.emit(
                        "CSV paths changed during discovery; re-scan them."
                    )
                    return
                recipe_state_before = (
                    dict(self.model.values.point_datasets),
                    dict(self.model.values.line_datasets),
                    dict(self.model.values.point_host_materials),
                    dict(self.model.values.line_host_materials),
                    set(self.model.values.excluded_point_placement_ids),
                    set(self.model.values.excluded_line_ids),
                )
                self.model.update_dataset_requirements(result)
                recipe_state_after = (
                    dict(self.model.values.point_datasets),
                    dict(self.model.values.line_datasets),
                    dict(self.model.values.point_host_materials),
                    dict(self.model.values.line_host_materials),
                    set(self.model.values.excluded_point_placement_ids),
                    set(self.model.values.excluded_line_ids),
                )
                if recipe_state_after != recipe_state_before:
                    self._set_recipe_dirty()
                self._apply_requirements_to_tables()
                point_count = len(self.model.point_dataset_ids)
                line_count = len(self.model.line_dataset_ids)
                missing = self.model.missing_dataset_mappings()
                if missing:
                    self.status_changed.emit(
                        f"✓ CSV schema valid: found {point_count} point and "
                        f"{line_count} line dataset ID(s). Next, choose an "
                        "OPN-FRD .grim response for: " + ", ".join(missing)
                    )
                else:
                    self.status_changed.emit(
                        f"✓ CSV schema valid: found {point_count} point and "
                        f"{line_count} line dataset ID(s); every response is mapped. "
                        "Next, validate placements and preview in 3-D."
                    )
                self._update_workflow_readiness()
            elif kind == "input_preview":
                preview_result = (
                    result.preview
                    if isinstance(result, _VerifiedInputPreview)
                    else result
                )
                requirements = (
                    result.discovery
                    if isinstance(result, _VerifiedInputPreview)
                    else getattr(preview_result, "dataset_requirements", None)
                )
                if requirements is not None:
                    self.model.update_dataset_requirements(requirements)
                    self._apply_requirements_to_tables()
                point_groups = getattr(preview_result, "point_locations_cad_m", {})
                line_groups = getattr(preview_result, "line_paths_cad_m", {})
                try:
                    point_total = sum(len(group) for group in point_groups.values())
                    line_total = sum(len(group) for group in line_groups.values())
                    count_text = (
                        f" ({point_total} point placement(s), "
                        f"{line_total} line path(s))"
                    )
                except (AttributeError, TypeError):
                    count_text = ""
                self._preview_is_current = True
                self._clear_validation_qa(
                    "Geometry preview only — physical placement QA has not run."
                )
                self.status_changed.emit(
                    "Geometry preview prepared"
                    + count_text
                    + ". Visual QA only: physical placement and response checks "
                    "have not run. Preview Layers → Show only displays or hides "
                    "artists; Spatial Feature Configuration → Use controls "
                    "preview, validation, response loading, and build membership."
                )
                self._remember_surface_dimensions(preview_result)
                self.preview_ready.emit(preview_result)
                self._update_workflow_readiness()
            elif kind == "preview":
                self._preview_is_current = True
                self._show_validation_qa(result)
                self._validated_plan_current = True
                warning_count = len(
                    getattr(result, "validation_warnings", ()) or ()
                )
                warning_text = (
                    f" ⚠ Review {warning_count} production QA warning(s) below."
                    if warning_count
                    else ""
                )
                skin_limit = getattr(result, "skin_limit_m", None)
                skin_text = (
                    f" Effective skin limit: {float(skin_limit) * 1e3:.3f} mm."
                    if skin_limit is not None
                    else ""
                )
                self.status_changed.emit(
                    "Enabled placements validated and every enabled dataset ID is mapped. "
                    "Showing the body and feature groups in the 3-D Assembly "
                    "view."
                    + skin_text
                    + " Build will reuse this validation while inputs remain unchanged."
                    + warning_text
                )
                self._remember_surface_dimensions(result)
                self.preview_ready.emit(result)
                self._update_workflow_readiness()
            elif kind == "build":
                dispatch = result
                self._preview_is_current = True
                self._show_validation_qa(dispatch.plan)
                self._validated_plan_current = True
                self._remember_surface_dimensions(dispatch.plan)
                self.preview_ready.emit(dispatch.plan)
                self.feature_built.emit(str(dispatch.output_path))
                features_only_path = str(
                    dispatch.features_only_output_path or ""
                ).strip()
                features_only_saved = bool(
                    dispatch.features_only_output_published
                    and features_only_path
                    and _path_key(Path(features_only_path))
                    != _path_key(Path(dispatch.output_path))
                    and Path(features_only_path).is_file()
                )
                if features_only_saved:
                    # Preserve the existing one-path signal contract while
                    # routing both published artifacts through GRIM's normal
                    # dataset loader for immediate before/after plotting.
                    self.feature_built.emit(features_only_path)
                features_status = (
                    f" Saved reusable feature-only delta: {features_only_path}."
                    if features_only_saved
                    else " No feature-only delta was published by this workflow service."
                )
                reuse_text = (
                    " Reused the unchanged validated preview."
                    if dispatch.reused_validated_plan
                    else ""
                )
                warning_count = len(
                    getattr(dispatch.plan, "validation_warnings", ()) or ()
                )
                warning_text = (
                    f" ⚠ Output saved with {warning_count} recorded production "
                    "QA warning(s); review them before release."
                    if warning_count
                    else ""
                )
                self.status_changed.emit(
                    f"Saved body-plus-features response: {dispatch.output_path}."
                    + features_status
                    + reuse_text
                    + warning_text
                    + " The result is ready in GRIM for plotting or further "
                    "dataset assembly."
                )
                self._update_workflow_readiness()

        @Slot(str)
        def _operation_failed(self, text: str) -> None:
            if self._active_kind in {"binding_check", "binding_write"}:
                try:
                    self._surface_binding_error_key = (
                        _surface_binding_identity_key(
                            self.base_picker.path(),
                            self.surface_picker.path(),
                            str(self.surface_units.currentData()),
                            base_dir=self.model.values.base_dir,
                        )
                    )
                except OSError:
                    self._surface_binding_error_key = None
                self._surface_binding_error = str(text)
            elif self._active_kind == "discover":
                changed = False
                for kind, picker in (
                    ("point", self.point_csv_picker),
                    ("line", self.line_csv_picker),
                ):
                    if picker.path() and not self.model.requirements_look_current(kind):
                        self.model.invalidate_dataset_requirements(kind)
                        changed = True
                if changed:
                    self._apply_requirements_to_tables()
            elif self._active_kind in {"input_preview", "preview", "build"}:
                # A worker invalidates only the CSV rows whose bytes changed
                # during its operation. Mirror that safe model state into the
                # mapping tables before readiness is recomputed.
                self._apply_requirements_to_tables()
                if self._active_kind in {"preview", "build"}:
                    self._validated_plan_current = False
                    self.validation_qa_label.setText(
                        "Validation stopped at the reported error. Correct that "
                        "instance, then validate again."
                    )
            self._show_error(text)

        @Slot()
        def _operation_thread_finished(self) -> None:
            self._thread = None
            self._worker = None
            self._active_kind = ""
            self._set_busy(False)


else:

    class FeatureAssemblyPanel:  # pragma: no cover - exercised only without Qt
        """Placeholder that preserves an actionable import-time API."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "FeatureAssemblyPanel requires PySide6. "
                f"Original import error: {_GUI_IMPORT_ERROR}"
            )


__all__ = [
    "GUI_AVAILABLE",
    "ASSEMBLY_REVIEW_LARGE_MESH_SHADOW_RAYS",
    "ASSEMBLY_REVIEW_LARGE_MESH_TRIANGLES",
    "ASSEMBLY_REVIEW_LINE_FIELD_CELLS",
    "ASSEMBLY_REVIEW_POINT_FIELD_CELLS",
    "ASSEMBLY_REVIEW_RADAR_GRID_CELLS",
    "ASSEMBLY_REVIEW_SHADOW_RAYS",
    "WORKLOAD_REVIEW_WARNING_PREFIX",
    "FEATURE_RECIPE_SCHEMA",
    "FEATURE_RECIPE_SUFFIX",
    "FEATURE_RECIPE_VERSION",
    "LINE_PLACEMENT_COLUMNS",
    "LINE_PLACEMENT_EXAMPLE",
    "POINT_PLACEMENT_COLUMNS",
    "POINT_PLACEMENT_EXAMPLE",
    "UNIT_CHOICES",
    "VALIDATION_PROFILES",
    "BaseGrimPreflight",
    "AssemblyWorkEstimate",
    "SurfaceBindingReadiness",
    "FeatureAssemblyFormModel",
    "FeatureAssemblyPanel",
    "FeatureAssemblyValues",
    "FeatureBuildDispatch",
    "FeatureWorkflowAdapter",
    "LoadedDatasetEntry",
    "LoadedFeatureAssemblyRecipe",
    "coerce_feature_workflow",
    "assess_surface_binding_readiness",
    "assembly_build_confirmation_required",
    "estimate_assembly_workload",
    "estimate_validated_assembly_plan_workload",
    "feature_assembly_recipe_payload",
    "format_assembly_work_estimate",
    "placement_csv_template_text",
    "preflight_base_grim",
    "read_feature_assembly_recipe",
    "surface_mesh_triangle_hint",
    "write_feature_assembly_recipe",
    "write_placement_csv_template",
]
