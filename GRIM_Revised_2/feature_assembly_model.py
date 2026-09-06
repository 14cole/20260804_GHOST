"""Qt-free Assembly form state, preflight, and backend adaptation.

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
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable
import zipfile

# Compatibility imports retain the established module entrypoints.
from feature_assembly_recipe import (
    _recipe_absolute_path,
    _recipe_id_set,
    _recipe_relative_path,
    _recipe_source_items,
    _recipe_string_mapping,
    _recipe_target_path,
    feature_assembly_recipe_payload,
    read_feature_assembly_recipe,
    write_feature_assembly_recipe,
)


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
FEATURE_RECIPE_VERSION = 5
FEATURE_RECIPE_SUFFIX = ".assembly.json"
# Hash normal placement/library inputs while keeping recipe saves responsive for
# very large vehicle meshes and clean-body response files. Large inputs still
# retain size and nanosecond modification-time identity in the manifest.
FEATURE_RECIPE_HASH_LIMIT_BYTES = 16 * 1024 * 1024

VALIDATION_PROFILES = (
    ("Require certified GHOST BoR body", "production", False, True, True),
    ("General body — metadata advisory (default)", "advisory", True, False, False),
    ("Strict library metadata (optional)", "external", False, True, False),
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


def parse_study_samples(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        values = tuple(float(part) for part in (value.replace(",", " ").split() if isinstance(value, str) else value))
    except (ValueError, TypeError) as exc:
        raise ValueError("Study samples must be comma-separated finite numbers.") from exc
    if not values or not all(math.isfinite(v) for v in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("Study samples must be finite, unique and increasing.")
    return values


from feature_assembly_values import (
    FeatureAssemblyValues,
    LoadedFeatureAssemblyRecipe,
)




@lru_cache(maxsize=64)
def _response_summary_cached(path, size, modified):
    import numpy as np
    fields = {}
    with zipfile.ZipFile(path) as archive:
        for key in ("azimuths", "elevations", "frequencies", "polarizations", "units", "phase_reference", "amplitude_convention", "amplitude_version", "complex_field_domain", "time_convention", "feature_library_manifest_json"):
            try:
                member = archive.getinfo(key+".npy")
            except KeyError:
                continue
            if member.file_size <= 512*1024:
                fields[key] = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
    parts = []
    for key, label in (("frequencies", "GHz"), ("azimuths", "azimuth°"), ("elevations", "elevation°")):
        values = np.asarray(fields.get(key, []), float).ravel()
        if len(values):
            parts.append(f"{label}: {values[0]:g}…{values[-1]:g} ({len(values):,})")
    channels = np.asarray(fields.get("polarizations", [])).ravel()
    parts.append("Channels: "+", ".join(map(str, channels)))
    parts.append("Phase origin / frame: "+str(fields.get("phase_reference", "missing; validation required")))
    parts.append("Amplitude: "+str(fields.get("amplitude_convention", "missing; validation required")))
    if "amplitude_version" in fields:
        parts.append("Physical amplitude version: "+str(fields["amplitude_version"]))
    if "units" in fields:
        units = json.loads(str(fields["units"]))
        parts.append("Quantity: "+str(units.get("rcs_linear_quantity", "unspecified"))+" / "+str(units.get("rcs_log_unit", "unspecified")))
    return "\n".join(parts)


def response_summary(path, *, base_dir=None):
    if not str(path).strip():
        return "Choose a response to inspect its stored grid, channels and field frame."
    try:
        source = _resolved_user_path(path, base_dir=base_dir)
        stat = source.stat()
        summary = _response_summary_cached(str(source), stat.st_size, stat.st_mtime_ns)
        manifests = [candidate for candidate in (Path(str(source)+".feature.json"), source.with_suffix(".feature.json")) if candidate.is_file()]
        if len(manifests) > 1:
            return summary+"\nConflicting manifest sidecars: keep exactly one."
        manifest = None
        if manifests and manifests[0].stat().st_size <= 512*1024:
            manifest = json.loads(manifests[0].read_text(encoding="utf-8-sig"))
        elif not manifests:
            with zipfile.ZipFile(source) as archive:
                key = "feature_library_manifest_json.npy"
                if key in archive.namelist() and archive.getinfo(key).file_size <= 512*1024:
                    import numpy as np
                    manifest = json.loads(str(np.load(io.BytesIO(archive.read(key)), allow_pickle=False)))
        if manifest is not None:
            host = manifest.get("host", {})
            envelope = manifest.get("applicability", {})
            summary += f"\nLibrary host: {host.get('material', 'missing')}; stack: {host.get('stack_id', 'unspecified')}"
            summary += f"\nFootprint radius: {envelope.get('footprint_radius_m', 'missing')} m; validation: {manifest.get('validation', {}).get('status', 'missing')} (source declaration; checked on Validate)"
        return summary
    except Exception as exc:
        return "Response summary unavailable: "+str(exc)


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
            "Select a valid clean-body GRIM before checking binding integrity.",
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
            "Choose the matching STL/facet mesh before checking body-binding "
            "integrity.",
            ready=not required,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
        )
    selected_units = str(surface_units or "").strip()
    if selected_units not in UNIT_SCALE_M:
        message = (
            "Choose the physical units of the selected surface mesh before "
            "checking body-binding integrity."
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
            f"✓ Integrity-checked reviewed binding — geometry {geometry}; "
            f"attested registration case {case_id}.",
            ready=True,
            required=required,
            external_body=True,
            sidecar_path=sidecar,
            identity_key=identity,
        )
    if checked_key is not None or error_key is not None:
        message = (
            "⚠ Binding check is stale — the body, mesh, units, or sidecar changed. "
            "Click Check binding integrity again."
        )
        code = "stale"
    else:
        message = (
            "○ Binding found but not checked for the current body, mesh, and units. "
            "Click Check binding integrity before Production validation."
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
        if normalized in (None, "line"):
            self._line_dataset_ids = ()
            self._line_path_count = 0
            self._line_segment_count = 0
            self._line_instances = ()
            self._line_requirements_csv = ""
            self._line_requirements_fingerprint = None
            self.values.line_datasets = {}
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
        missing = self.missing_dataset_mappings()
        if missing:
            raise ValueError(
                "Choose an OPN-FRD GRIM response for: " + ", ".join(missing)
            )
        if values.shadow and not _clean_path(values.surface_mesh) and not preflight_base_grim(values.base_grim, base_dir=values.base_dir).embedded_bor:
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
            **{key: getattr(values, key) for key in ("study_frequencies_ghz", "study_azimuths_deg", "study_elevations_deg")},
            host_material=values.host_material,
            host_stack_id=values.host_stack_id,
            host_minimum_radius_m=values.host_minimum_radius_m,
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
            values.host_material, values.host_stack_id, values.host_minimum_radius_m,
            values.study_frequencies_ghz, values.study_azimuths_deg, values.study_elevations_deg,
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
