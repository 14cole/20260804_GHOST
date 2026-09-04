#!/usr/bin/env python3
"""Qt-free orchestration for coherent point and line feature placement.

The numerical implementation remains in :mod:`feature_sum`,
:mod:`line_expand`, :mod:`surface_mesh`, and :mod:`occluder`.  This module
turns the strict placement CSV contracts and their dataset mappings into a
reusable request/plan API suitable for scripts, tests, and desktop clients.

Selecting a dataset in either mapping is an explicit declaration that it is
the canonical coherent OPN-FRD (installed/featured minus clean-skin) delta.
The workflow never guesses or reverses the subtraction order.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
import uuid
import zipfile

import numpy as np

from assembly_workload import (
    WORKLOAD_REVIEW_WARNING_PREFIX,
    estimate_assembly_workload,
    warnings_require_workload_acknowledgement,
    workload_review_warning,
)
from feature_sum import (
    ASSEMBLY_RADAR_ANGULAR_CONTRACT,
    POINT_PATTERN_FRAME_CONVENTION,
    PreparedPointPattern,
    _LEGACY_BASE_ASSUMPTIONS_KEY,
    _attitude,
    _canonical_3d_channel_indices,
    _decoded_feature_provenance,
    _direction,
    _load_grim,
    _validate_declared_coherent_base,
    add_features_to_monostatic_grim,
    feature_only_output_path,
    exact_assembly_subset,
    load_seam_from_grim,
    load_body_profile_grim,
    load_body_requested_radar_grid,
    preflight_feature_assembly_capacity,
    prepare_point_pattern,
    require_body_mesh_certification as audit_body_mesh_certification,
    surface_of_revolution_distance,
    validate_declared_coherent_delta_domain,
    validate_assembly_base_grid_metadata,
    validate_radar_grid,
)
from frame import (
    CAD2AXIS,
    AXIS_AZ_DEG,
    AXIS_EL_DEG,
    ROLL_DEG,
    scale_for,
    to_axis_frame,
)
from line_expand import (
    C0,
    GRAZING_TAPER_DEG,
    PSI_HH_DEG,
    PSI_VV_DEG,
    constant_normal_piece_length,
    perimeter_surface_deviation,
    prepare_perimeter_frame,
    surface_of_revolution_normal,
)
from grim_naming import require_role_free_declared_delta
from occluder import Occluder
from surface_mesh import TriangleSurface, read_surface_mesh
from workflow_provenance import sha256_file


POINT_CSV_COLUMNS = (
    "placement_id", "dataset_id",
    "x", "y", "z",
    "nx", "ny", "nz",
    "roll_x", "roll_y", "roll_z",
)
POINT_PLACEMENT_SCHEMA = "ghost.point-placement.v1"

LINE_CSV_COLUMNS = (
    "line_id", "dataset_id", "segment_index",
    "x1", "y1", "z1", "x2", "y2", "z2",
    "n1x", "n1y", "n1z", "n2x", "n2y", "n2z",
)
LINE_PLACEMENT_SCHEMA = "ghost.line-placement.v1"

FEATURE_ASSEMBLY_REQUEST_SCHEMA = "ghost.feature-assembly-request.v1"
FEATURE_LIBRARY_MANIFEST_SCHEMA = "ghost.feature-library-manifest.v3"
PREVIOUS_FEATURE_LIBRARY_MANIFEST_SCHEMA = "ghost.feature-library-manifest.v2"
LEGACY_FEATURE_LIBRARY_MANIFEST_SCHEMA = "ghost.feature-library-manifest.v1"
FEATURE_LIBRARY_MANIFEST_KEY = "feature_library_manifest_json"
FEATURE_VALIDATION_EVIDENCE_SCHEMA = (
    "ghost.validation.feature-case-evidence.v1"
)
FEATURE_VALIDATION_ARTIFACT_ROLES = (
    "clean_truth",
    "clean_prediction",
    "featured_truth",
    "featured_prediction",
)
# A project may tighten these release ceilings, but a manifest must not turn a
# numerically weak comparison into "validated" by silently loosening them.
FEATURE_VALIDATION_RELEASE_CEILINGS = {
    "max_normalized_rms": 0.25,
    "max_magnitude_p95_db": 3.5,
    "max_phase_rms_deg": 25.0,
    "min_coherence": 0.95,
}
# Raising the active-field floor discards more of each pattern from the
# pointwise magnitude/phase gates.  Production evidence may include weaker
# samples (a more-negative floor), but must not validate only the pattern peak.
FEATURE_VALIDATION_MAX_ACTIVE_FLOOR_DB = -40.0
DECLARED_FEATURE_DELTA_RESPONSE_SCHEMA = (
    "ghost.declared-feature-delta-response.v1"
)
LINE_PHASE_CALIBRATION_SCHEMA = "ghost.line-phase-calibration.v2"
LEGACY_LINE_PHASE_CALIBRATION_SCHEMA = "ghost.line-phase-calibration.v1"
SURFACE_BINDING_SCHEMA = "ghost.assembly-surface-binding.v1"
SURFACE_FRAME_CONVENTION = "CAD:+y=nose;+x=right;+z=up"
_OUTWARD_ALIGNMENT_EPS = 1.0e-12

PathValue = str | os.PathLike[str]


@dataclass(frozen=True)
class FeatureAssemblyRequest:
    """All user selections needed to validate and assemble placed features.

    Relative paths are resolved against ``base_dir``.  When it is ``None``,
    they are resolved against the process working directory.  The point and
    line dataset mappings use the exact ``dataset_id`` strings found in their
    respective CSV files.
    """

    base_grim: PathValue
    output_grim: PathValue
    # Physical units are deliberately unset until the caller chooses them.
    # Placement files do not carry a trustworthy length unit, so silently
    # interpreting an omitted value as inches can scale an otherwise
    # self-consistent vehicle by exactly 25.4x without tripping skin/normal QA.
    coordinate_units: Optional[str] = None

    surface_mesh: Optional[PathValue] = None
    # Required only when ``surface_mesh`` is supplied.  Embedded BoR geometry
    # is already stored in meters inside the clean-body response.
    surface_units: Optional[str] = None
    flip_surface_normals: bool = False
    shadow: bool = False
    shadow_bias_m: Optional[float] = None

    point_locations_csv: Optional[PathValue] = None
    point_datasets: Mapping[str, PathValue] = field(default_factory=dict)
    line_locations_csv: Optional[PathValue] = None
    line_datasets: Mapping[str, PathValue] = field(default_factory=dict)

    skin_tol_m: float = 1.0e-3
    skin_phase_tol_deg: float = 15.0
    normal_tol_deg: float = 15.0

    base_dir: Optional[PathValue] = None
    history: str = "feature_workflow.py coherent platform line/compact placement"
    # Appended for positional backward compatibility with v1 script callers.
    # None means every strictly parsed instance; an empty tuple means none.
    enabled_point_placement_ids: Optional[tuple[str, ...]] = None
    enabled_line_ids: Optional[tuple[str, ...]] = None
    # Legacy GUI grids can omit descriptive coherent/angle tags.  Present
    # contradictions are always rejected; this switch controls only missing
    # metadata and its use is recorded in output provenance.
    allow_legacy_base_metadata: bool = True
    # Feature-library manifests are optional for existing team libraries.  In
    # strict production deployments this makes their applicability evidence a
    # hard requirement instead of a recorded validation warning.
    require_feature_manifests: bool = False
    # Production builds from a locally solved GHOST body can require the
    # embedded, dual-polarization fine-mesh certificate. External/HPC body
    # responses need an explicit caller-selected waiver because GRIM cannot
    # reconstruct solver convergence evidence that is absent from the file.
    # Appended for positional compatibility with earlier request layouts.
    require_body_mesh_certification: bool = False
    # Declarations about the installation region, not guessed from mesh shape.
    # Use a material/stack identity shared with the characterized feature library.
    host_material: str = ""
    host_stack_id: str = ""
    host_minimum_radius_m: Optional[float] = None
    study_frequencies_ghz: Optional[tuple[float, ...]] = None
    study_azimuths_deg: Optional[tuple[float, ...]] = None
    study_elevations_deg: Optional[tuple[float, ...]] = None


@dataclass(frozen=True)
class FeatureDatasetRequirements:
    """Dataset IDs discovered from already schema-validated placement CSVs."""

    point_dataset_ids: tuple[str, ...] = ()
    line_dataset_ids: tuple[str, ...] = ()
    point_placement_count: int = 0
    line_path_count: int = 0
    line_segment_count: int = 0
    # Stable spatial-instance descriptors used by GRIM's feature-definition
    # tree. These are parsed placement identities, never response-grid leaves.
    point_instances: tuple[tuple[str, str], ...] = ()
    line_instances: tuple[tuple[str, str, int], ...] = ()


@dataclass(frozen=True)
class FeaturePreviewGeometry:
    """Validated full-resolution geometry in the user-visible CAD frame.

    Coordinates are meters.  The body profile remains the frame-free BoR
    ``rho,z`` generatrix.  GRIM may decimate a copy for display, but the full
    surface stored here is the same geometry from which the physics surface
    was constructed.
    """

    surface_triangles_cad_m: Optional[np.ndarray]
    body_profile_rho_z_m: Optional[np.ndarray]
    point_locations_cad_m: dict[str, np.ndarray]
    line_paths_cad_m: dict[str, dict[str, np.ndarray]]
    # Orientation vectors are unitless and remain in the same user-visible
    # CAD frame as the placement coordinates.  GRIM projects each point roll
    # reference onto the plane normal to local +z when it draws the frame,
    # matching the point-scatterer convention without changing placement
    # validation or solver inputs.
    point_normals_cad: dict[str, np.ndarray] = field(default_factory=dict)
    point_roll_references_cad: dict[str, np.ndarray] = field(default_factory=dict)
    line_endpoint_normals_cad: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    # Line paths retain line_id as dictionary keys. Point arrays are grouped by
    # response dataset, so carry their IDs explicitly for exact GUI QA focus.
    point_placement_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureInputPreview:
    """Strictly parsed placement geometry for interactive 3-D setup.

    This preview deliberately stops before response loading, skin/normal
    certification, and coherent assembly. It lets a user verify file choice,
    units, CAD orientation, IDs, and location paths while completing the form;
    :func:`prepare_feature_assembly` remains the authoritative physical gate.
    """

    preview_geometry: FeaturePreviewGeometry
    dataset_requirements: FeatureDatasetRequirements
    body_source: str
    preview_stage: str = "input"

    @property
    def surface_triangles_cad_m(self) -> Optional[np.ndarray]:
        return self.preview_geometry.surface_triangles_cad_m

    @property
    def body_profile_rho_z_m(self) -> Optional[np.ndarray]:
        return self.preview_geometry.body_profile_rho_z_m

    @property
    def point_locations_cad_m(self) -> dict[str, np.ndarray]:
        return self.preview_geometry.point_locations_cad_m

    @property
    def line_paths_cad_m(self) -> dict[str, dict[str, np.ndarray]]:
        return self.preview_geometry.line_paths_cad_m

    @property
    def point_normals_cad(self) -> dict[str, np.ndarray]:
        return self.preview_geometry.point_normals_cad

    @property
    def point_roll_references_cad(self) -> dict[str, np.ndarray]:
        return self.preview_geometry.point_roll_references_cad

    @property
    def line_endpoint_normals_cad(self) -> dict[str, dict[str, np.ndarray]]:
        return self.preview_geometry.line_endpoint_normals_cad

    @property
    def point_placement_ids(self) -> dict[str, tuple[str, ...]]:
        return self.preview_geometry.point_placement_ids


@dataclass(frozen=True)
class FeatureAssemblyPlan:
    """Prepared, physically validated inputs ready for coherent execution."""

    request: FeatureAssemblyRequest
    base_path: Path
    output_path: Path
    radar_grid: dict[str, Any]
    body_profile: Optional[np.ndarray]
    surface_path: Optional[Path]
    surface: Optional[TriangleSurface]
    surface_normal_fn: Callable[[np.ndarray], np.ndarray]
    occluder: Optional[Occluder]
    line_placements: list[dict[str, Any]]
    point_placements: list[dict[str, Any]]
    line_records: list[dict[str, Any]]
    point_records: list[dict[str, Any]]
    dataset_requirements: FeatureDatasetRequirements
    preview_geometry: FeaturePreviewGeometry
    skin_limit_m: float
    highest_frequency_wavelength_m: float
    feature_provenance: dict[str, Any]
    # Exact bytes validated during preparation.  The default preserves
    # compatibility with callers that constructed plans before this guard was
    # introduced; plans returned by prepare_feature_assembly always populate it.
    prepared_source_sha256: dict[str, str] = field(default_factory=dict)
    # Appended after the existing prepared-source field so positional service
    # callers from the v1 plan layout retain their historical binding.
    validation_warnings: tuple[str, ...] = ()
    # Legal feature-manifest sidecar names that did not exist during
    # preparation. Creation of one before publication invalidates the plan just
    # like changing an existing sidecar, because it could contradict the
    # claims that were actually validated.
    prepared_absent_paths: tuple[str, ...] = ()
    # Destination state observed by the authoritative validation.  Execution
    # checks it while holding the interprocess publication lock so a second
    # window cannot replace a newer output that the user never reviewed.
    prepared_output_sha256: Optional[str] = None
    prepared_output_absent: bool = True
    # Canonical hash of every mutable plan input consumed by execution. This
    # prevents a cached/headless caller from editing validated placement arrays
    # or provenance between Validate and Build.
    prepared_plan_sha256: str = ""
    # Keep all v2 additions after prepared_plan_sha256.  It was the final field
    # in the original positional plan layout, and older injected services may
    # still bind that field positionally.
    prepared_features_only_output_sha256: Optional[str] = None
    prepared_features_only_output_absent: bool = True

    @property
    def surface_triangles_cad_m(self) -> Optional[np.ndarray]:
        return self.preview_geometry.surface_triangles_cad_m

    @property
    def body_profile_rho_z_m(self) -> Optional[np.ndarray]:
        return self.preview_geometry.body_profile_rho_z_m

    @property
    def point_locations_cad_m(self) -> dict[str, np.ndarray]:
        return self.preview_geometry.point_locations_cad_m

    @property
    def line_paths_cad_m(self) -> dict[str, dict[str, np.ndarray]]:
        return self.preview_geometry.line_paths_cad_m

    @property
    def point_normals_cad(self) -> dict[str, np.ndarray]:
        return self.preview_geometry.point_normals_cad

    @property
    def point_roll_references_cad(self) -> dict[str, np.ndarray]:
        return self.preview_geometry.point_roll_references_cad

    @property
    def line_endpoint_normals_cad(self) -> dict[str, dict[str, np.ndarray]]:
        return self.preview_geometry.line_endpoint_normals_cad

    @property
    def point_placement_ids(self) -> dict[str, tuple[str, ...]]:
        return self.preview_geometry.point_placement_ids

    @property
    def preview_stage(self) -> str:
        """Identify this scene as physically prepared, not an input-only view."""

        return "validated"

    @property
    def features_only_output_path(self) -> Path:
        return Path(feature_only_output_path(str(self.output_path)))


def feature_assembly_plan_sha256(plan: FeatureAssemblyPlan) -> str:
    """Return the canonical execution-input digest for a prepared plan."""

    digest = hashlib.sha256()
    digest.update(b"ghost-feature-assembly-plan-v1\0")

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): normalize(child)
                for key, child in sorted(
                    value.items(), key=lambda item: str(item[0])
                )
            }
        if isinstance(value, (list, tuple)):
            return [normalize(child) for child in value]
        if isinstance(value, (Path, os.PathLike)):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float):
            if math.isnan(value):
                raise ValueError("Feature Assembly plan contains NaN metadata.")
            if math.isinf(value):
                return "+Infinity" if value > 0.0 else "-Infinity"
            return value
        if value is None or isinstance(value, (str, int, bool)):
            return value
        return str(value)

    request_state = {
        name: normalize(value)
        for name, value in vars(plan.request).items()
    }
    state = {
        "base_path": str(plan.base_path),
        "output_path": str(plan.output_path),
        "request": request_state,
        "line_records": normalize(plan.line_records),
        "point_records": normalize(plan.point_records),
        "feature_provenance": normalize(plan.feature_provenance),
        "prepared_source_sha256": normalize(plan.prepared_source_sha256),
        "prepared_absent_paths": normalize(plan.prepared_absent_paths),
        "prepared_output_sha256": plan.prepared_output_sha256,
        "prepared_output_absent": bool(plan.prepared_output_absent),
        "prepared_features_only_output_sha256": (
            plan.prepared_features_only_output_sha256
        ),
        "prepared_features_only_output_absent": bool(
            plan.prepared_features_only_output_absent
        ),
        "validation_warnings": normalize(plan.validation_warnings),
    }
    digest.update(json.dumps(
        state, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))

    def update_array(label: str, value: Any) -> None:
        raw = np.asarray(value)
        array = np.ascontiguousarray(
            raw, dtype="<c16" if np.iscomplexobj(raw) else "<f8"
        )
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        # The canonical array is contiguous; feed its buffer directly so a
        # multi-million-triangle surface does not require another full byte copy
        # merely to seal the reviewed plan.
        digest.update(memoryview(array).cast("B"))

    for key in (
        "frequencies_ghz",
        "azimuths_deg",
        "elevations_deg",
        "axis_az_deg",
        "axis_el_deg",
        "roll_deg",
    ):
        if key in plan.radar_grid:
            update_array(f"radar_grid:{key}", plan.radar_grid[key])
    if plan.occluder is not None:
        update_array("occluder:triangles", plan.occluder.tris)
        update_array("occluder:bias", [plan.occluder.bias])
    for index, placement in enumerate(plan.line_placements):
        update_array(f"line:{index}:perimeter", placement["perimeter"])
        update_array(
            f"line:{index}:segment_normals", placement["segment_normals"]
        )
        if placement.get("shadow_points") is not None:
            update_array(
                f"line:{index}:shadow_points", placement["shadow_points"]
            )
        digest.update(json.dumps(normalize({
            key: value
            for key, value in placement.items()
            if key not in {"perimeter", "segment_normals", "shadow_points"}
        }), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    unique_patterns: dict[int, int] = {}
    for index, placement in enumerate(plan.point_placements):
        for key in (
            "location", "aperture_normal", "roll_ref", "shadow_location"
        ):
            if key not in placement:
                continue
            update_array(f"point:{index}:{key}", placement[key])
        pattern = placement.get("pattern")
        if pattern is not None:
            pattern_identity = id(pattern)
            pattern_index = unique_patterns.get(pattern_identity)
            if pattern_index is None:
                pattern_index = len(unique_patterns)
                unique_patterns[pattern_identity] = pattern_index
                for key in (
                    "azimuths", "elevations", "frequencies", "amplitude"
                ):
                    if hasattr(pattern, key):
                        update_array(
                            f"pattern:{pattern_index}:{key}", getattr(pattern, key)
                        )
                if hasattr(pattern, "channel_indices"):
                    digest.update(json.dumps(
                        normalize(dict(pattern.channel_indices)),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"))
            digest.update(
                f"point:{index}:pattern:{pattern_index}\0".encode("ascii")
            )
    return digest.hexdigest()


def resolve_path(value: PathValue, *, base_dir: Optional[PathValue] = None) -> Path:
    """Resolve one user path without imposing a repository-specific root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
    return (root.resolve() / path).resolve()


def _canonical_grim_output_path(
    value: PathValue, *, base_dir: Optional[PathValue] = None
) -> Path:
    """Resolve the path exactly as the GRIM writer will publish it."""

    resolved = resolve_path(value, base_dir=base_dir)
    if str(resolved).lower().endswith(".grim"):
        return resolved
    return Path(str(resolved) + ".grim")


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether two resolved paths name the same filesystem target."""

    if first == second:
        return True
    try:
        return first.samefile(second)
    except (FileNotFoundError, OSError):
        return False


def _reject_output_aliases(
    request: FeatureAssemblyRequest, *, base: Path, output: Path
) -> None:
    """Protect every selected input from accidental output overwrite."""

    outputs = (
        ("output_grim", output),
        ("feature-only sibling", Path(feature_only_output_path(str(output)))),
    )
    for output_label, candidate in outputs:
        if _paths_alias(candidate, base):
            raise ValueError(
                f"{output_label} must differ from base_grim so the clean-body "
                "response is not overwritten."
            )
    for kind, datasets in (
        ("point", request.point_datasets),
        ("line", request.line_datasets),
    ):
        for dataset_id, value in datasets.items():
            response = resolve_path(value, base_dir=request.base_dir)
            for output_label, candidate in outputs:
                if _paths_alias(candidate, response):
                    raise ValueError(
                        f"{output_label} must differ from every mapped response "
                        f"input; it aliases {kind} dataset_id "
                        f"{str(dataset_id)!r}: {response}"
                    )

    for role, value in (
        ("surface mesh", request.surface_mesh),
        ("point-placement CSV", request.point_locations_csv),
        ("line-placement CSV", request.line_locations_csv),
    ):
        if value is None:
            continue
        source = resolve_path(value, base_dir=request.base_dir)
        for output_label, candidate in outputs:
            if _paths_alias(candidate, source):
                raise ValueError(
                    f"{output_label} must differ from every selected Assembly "
                    f"input; it aliases the {role}: {source}"
                )


def _csv_rows(path: Path, *, label: str) -> list[tuple[list[str], int]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return [
                ([cell.strip() for cell in row], line_number)
                for line_number, row in enumerate(csv.reader(stream), 1)
                if row and any(cell.strip() for cell in row)
            ]
    except OSError as exc:
        raise OSError(f"{path}: cannot read {label} CSV: {exc}") from exc


_CSV_ERROR_DISPLAY_LIMIT = 25


def _raise_csv_row_errors(
    source: Path,
    *,
    label: str,
    errors: Sequence[tuple[int, str]],
) -> None:
    """Raise one bounded, actionable report for independent CSV row errors."""

    if not errors:
        return
    count = len(errors)
    shown = list(errors[:_CSV_ERROR_DISPLAY_LIMIT])
    lines = [
        f"{source}: {label} CSV has {count} validation error(s):",
        *(f"  - line {number}: {message}" for number, message in shown),
    ]
    omitted = count - len(shown)
    if omitted:
        lines.append(
            f"  - ... {omitted} additional error(s) omitted; fix the listed "
            "rows, then validate again to reveal any remainder."
        )
    else:
        lines.append("Fix the listed rows, then validate the CSV again.")
    raise ValueError("\n".join(lines))


def _compact_csv_values(values: Sequence[Any], *, limit: int = 12) -> str:
    """Render a bounded sequence inside a CSV validation message."""

    selected = list(values[:limit])
    if len(values) <= limit:
        return repr(selected)
    return f"{selected!r} ... ({len(values)} values total)"


def read_point_placement_csv(
    path: PathValue,
    *,
    base_dir: Optional[PathValue] = None,
) -> list[dict[str, Any]]:
    """Read the one strict point-placement CSV schema."""

    source = resolve_path(path, base_dir=base_dir)
    rows = _csv_rows(source, label="point-placement")
    if not rows:
        raise ValueError(f"{source}: placement CSV is empty.")
    header, header_line = rows.pop(0)
    if tuple(header) != POINT_CSV_COLUMNS:
        raise ValueError(
            f"{source}:{header_line}: header must be exactly "
            f"{','.join(POINT_CSV_COLUMNS)}."
        )
    if not rows:
        raise ValueError(f"{source}: placement CSV has a header but no placements.")

    parsed: list[dict[str, Any]] = []
    errors: list[tuple[int, str]] = []
    seen: set[str] = set()
    numeric_columns = POINT_CSV_COLUMNS[2:]
    for row, number in rows:
        if len(row) != len(POINT_CSV_COLUMNS):
            errors.append((
                number,
                f"expected exactly {len(POINT_CSV_COLUMNS)} columns; "
                f"found {len(row)}.",
            ))
            continue
        placement_id, dataset_id = row[:2]
        row_errors: list[str] = []
        if not placement_id or not dataset_id:
            row_errors.append("placement_id and dataset_id are required")
        if placement_id:
            if placement_id in seen:
                row_errors.append(
                    f"duplicate placement_id {placement_id!r}"
                )
            else:
                seen.add(placement_id)

        numeric: list[float] = []
        nonnumeric: list[str] = []
        for column, value in zip(numeric_columns, row[2:]):
            try:
                numeric.append(float(value))
            except ValueError:
                nonnumeric.append(column)
        if nonnumeric:
            row_errors.append(
                "coordinates and vectors must be numeric; invalid column(s) "
                + ", ".join(nonnumeric)
            )
        elif not np.all(np.isfinite(numeric)):
            nonfinite = [
                column
                for column, value in zip(numeric_columns, numeric)
                if not math.isfinite(value)
            ]
            row_errors.append(
                "NaN/infinite value in column(s) " + ", ".join(nonfinite)
            )
        if row_errors:
            errors.extend((number, message) for message in row_errors)
            continue
        values: dict[str, Any] = {
            "placement_id": placement_id,
            "dataset_id": dataset_id,
        }
        values.update(dict(zip(numeric_columns, numeric)))
        values["_csv_line"] = number
        parsed.append(values)
    _raise_csv_row_errors(source, label="point-placement", errors=errors)
    return parsed


def read_line_placement_csv(
    path: PathValue,
    *,
    base_dir: Optional[PathValue] = None,
) -> list[dict[str, Any]]:
    """Read the one strict ordered-segment line-placement CSV schema."""

    source = resolve_path(path, base_dir=base_dir)
    rows = _csv_rows(source, label="line-placement")
    if not rows:
        raise ValueError(f"{source}: line-placement CSV is empty.")
    header, header_line = rows.pop(0)
    if tuple(header) != LINE_CSV_COLUMNS:
        raise ValueError(
            f"{source}:{header_line}: header must be exactly "
            f"{','.join(LINE_CSV_COLUMNS)}."
        )
    if not rows:
        raise ValueError(f"{source}: line-placement CSV has a header but no segments.")

    parsed: list[dict[str, Any]] = []
    errors: list[tuple[int, str]] = []
    semantic_rows: list[dict[str, Any]] = []
    sequence_tainted_line_ids: set[str] = set()
    identity_tainted_line_ids: set[str] = set()
    numeric_columns = LINE_CSV_COLUMNS[3:]
    for row, number in rows:
        if len(row) != len(LINE_CSV_COLUMNS):
            errors.append((
                number,
                f"expected exactly {len(LINE_CSV_COLUMNS)} columns; "
                f"found {len(row)}.",
            ))
            if row and row[0]:
                sequence_tainted_line_ids.add(row[0])
                identity_tainted_line_ids.add(row[0])
            continue
        line_id, dataset_id, raw_index = row[:3]
        row_errors: list[str] = []
        if not line_id or not dataset_id:
            row_errors.append("line_id and dataset_id are required")

        segment_index: Optional[int]
        try:
            segment_index = int(raw_index)
        except ValueError:
            segment_index = None
        if (
            segment_index is None
            or str(segment_index) != raw_index
            or segment_index < 1
        ):
            row_errors.append(
                "segment_index must be a canonical positive integer"
            )
            if line_id:
                sequence_tainted_line_ids.add(line_id)

        numeric: list[float] = []
        nonnumeric: list[str] = []
        for column, value in zip(numeric_columns, row[3:]):
            try:
                numeric.append(float(value))
            except ValueError:
                nonnumeric.append(column)
        if nonnumeric:
            row_errors.append(
                "endpoints and normals must be numeric; invalid column(s) "
                + ", ".join(nonnumeric)
            )
        elif not np.all(np.isfinite(numeric)):
            nonfinite = [
                column
                for column, value in zip(numeric_columns, numeric)
                if not math.isfinite(value)
            ]
            row_errors.append(
                "NaN/infinite value in column(s) " + ", ".join(nonfinite)
            )

        semantic_rows.append({
            "line_id": line_id,
            "dataset_id": dataset_id,
            "segment_index": segment_index,
            "_csv_line": number,
        })
        if row_errors:
            errors.extend((number, message) for message in row_errors)
            continue
        values: dict[str, Any] = dict(semantic_rows[-1])
        values.update(dict(zip(numeric_columns, numeric)))
        parsed.append(values)

    # Sequence checks are reported once per path, not once per downstream row.
    # A malformed identity/index row taints only that line_id so it cannot
    # generate a cascade of misleading "expected N" messages.
    completed_line_ids: set[str] = set()
    start = 0
    while start < len(semantic_rows):
        line_id = str(semantic_rows[start]["line_id"])
        end = start + 1
        while (
            end < len(semantic_rows)
            and str(semantic_rows[end]["line_id"]) == line_id
        ):
            end += 1
        group = semantic_rows[start:end]
        if line_id:
            first_line = int(group[0]["_csv_line"])
            if (
                line_id not in identity_tainted_line_ids
                and line_id in completed_line_ids
            ):
                errors.append((
                    first_line,
                    f"rows for line_id {line_id!r} must be contiguous",
                ))
            completed_line_ids.add(line_id)
            dataset_ids = tuple(dict.fromkeys(
                str(item["dataset_id"])
                for item in group
                if str(item["dataset_id"])
            ))
            if len(dataset_ids) > 1:
                errors.append((
                    first_line,
                    f"every segment of line_id {line_id!r} must use the same "
                    "dataset_id; found " + _compact_csv_values(dataset_ids),
                ))
            if line_id not in sequence_tainted_line_ids:
                indices = [int(item["segment_index"]) for item in group]
                expected = list(range(1, len(group) + 1))
                if indices != expected:
                    errors.append((
                        first_line,
                        f"line_id {line_id!r} requires one consecutive "
                        "segment_index sequence in CSV row order; expected "
                        + _compact_csv_values(expected)
                        + ", found "
                        + _compact_csv_values(indices),
                    ))
        start = end

    _raise_csv_row_errors(source, label="line-placement", errors=errors)
    return parsed


def _ordered_dataset_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(row["dataset_id"]) for row in rows))


def _line_instance_descriptors(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, int], ...]:
    """Return one stable descriptor for each already validated line path."""
    descriptors: list[tuple[str, str, int]] = []
    start = 0
    while start < len(rows):
        line_id = str(rows[start]["line_id"])
        dataset_id = str(rows[start]["dataset_id"])
        end = start + 1
        while end < len(rows) and str(rows[end]["line_id"]) == line_id:
            end += 1
        descriptors.append((line_id, dataset_id, end - start))
        start = end
    return tuple(descriptors)


def _filter_enabled_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_key: str,
    enabled_ids: Optional[Sequence[str]],
    label: str,
) -> list[Mapping[str, Any]]:
    """Filter after strict parsing and reject stale selections fail-closed."""
    parsed = list(rows)
    if enabled_ids is None:
        return parsed
    if isinstance(enabled_ids, (str, bytes)):
        raise TypeError(
            f"enabled {label} IDs must be a sequence of complete IDs, not text."
        )
    requested = tuple(str(value).strip() for value in enabled_ids)
    if any(not value for value in requested):
        raise ValueError(f"enabled {label} IDs must be nonempty strings.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"enabled {label} IDs must be unique.")
    available = tuple(dict.fromkeys(str(row[id_key]) for row in parsed))
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            f"enabled {label} ID(s) {unknown} are not present in the parsed "
            f"placement CSV; available IDs are {list(available)}. Refresh the "
            "feature configuration before validating or building."
        )
    selected = set(requested)
    return [row for row in parsed if str(row[id_key]) in selected]


def discover_feature_dataset_ids(
    *,
    point_locations_csv: Optional[PathValue] = None,
    line_locations_csv: Optional[PathValue] = None,
    base_dir: Optional[PathValue] = None,
) -> FeatureDatasetRequirements:
    """Validate selected CSVs and return dataset IDs in first-use order."""

    point_rows = (
        read_point_placement_csv(point_locations_csv, base_dir=base_dir)
        if point_locations_csv is not None else []
    )
    line_rows = (
        read_line_placement_csv(line_locations_csv, base_dir=base_dir)
        if line_locations_csv is not None else []
    )
    return FeatureDatasetRequirements(
        point_dataset_ids=_ordered_dataset_ids(point_rows),
        line_dataset_ids=_ordered_dataset_ids(line_rows),
        point_placement_count=len(point_rows),
        line_path_count=len({str(row["line_id"]) for row in line_rows}),
        line_segment_count=len(line_rows),
        point_instances=tuple(
            (str(row["placement_id"]), str(row["dataset_id"]))
            for row in point_rows
        ),
        line_instances=_line_instance_descriptors(line_rows),
    )


def _input_line_preview_paths(
    rows: Sequence[Mapping[str, Any]],
    *,
    coordinate_scale: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Build CAD-meter polylines from already schema-validated line rows."""

    result: dict[str, dict[str, np.ndarray]] = {}
    start = 0
    while start < len(rows):
        line_id = str(rows[start]["line_id"])
        end = start + 1
        while end < len(rows) and str(rows[end]["line_id"]) == line_id:
            end += 1
        group = rows[start:end]
        dataset_id = str(group[0]["dataset_id"])
        segments = np.asarray(
            [
                [
                    [row["x1"], row["y1"], row["z1"]],
                    [row["x2"], row["y2"], row["z2"]],
                ]
                for row in group
            ],
            dtype=float,
        ) * float(coordinate_scale)
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        if np.any(lengths <= 0.0):
            index = int(np.flatnonzero(lengths <= 0.0)[0])
            raise ValueError(
                f"line-placement row {group[index]['_csv_line']} has a "
                "zero-length segment."
            )
        extent = float(np.max(np.ptp(segments.reshape(-1, 3), axis=0)))
        continuity_tolerance = max(
            1.0e-12,
            1.0e-6 * max(extent, float(np.max(lengths))),
        )
        for index in range(len(segments) - 1):
            gap = float(np.linalg.norm(
                segments[index, 1] - segments[index + 1, 0]
            ))
            if gap > continuity_tolerance:
                raise ValueError(
                    f"line-placement rows {group[index]['_csv_line']} and "
                    f"{group[index + 1]['_csv_line']} of line_id {line_id!r} "
                    f"are not head-to-tail (gap {gap:.3e} m)."
                )
        path = np.concatenate(
            [segments[:, 0, :], segments[-1:, 1, :]], axis=0
        )
        result.setdefault(dataset_id, {})[line_id] = np.array(
            path, dtype=float, copy=True
        )
        start = end
    return result


def _input_line_preview_normals(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Collect supplied line endpoint normals without certifying them.

    Input preview is intentionally visual QA only.  The authoritative line
    placement path still performs every nonzero, interpolation, and surface
    agreement check in :func:`prepare_line_placements`.
    """

    result: dict[str, dict[str, np.ndarray]] = {}
    start = 0
    while start < len(rows):
        line_id = str(rows[start]["line_id"])
        end = start + 1
        while end < len(rows) and str(rows[end]["line_id"]) == line_id:
            end += 1
        group = rows[start:end]
        dataset_id = str(group[0]["dataset_id"])
        normals = np.asarray(
            [
                [
                    [row["n1x"], row["n1y"], row["n1z"]],
                    [row["n2x"], row["n2y"], row["n2z"]],
                ]
                for row in group
            ],
            dtype=float,
        )
        result.setdefault(dataset_id, {})[line_id] = np.array(
            normals, dtype=float, copy=True
        )
        start = end
    return result


def prepare_feature_input_preview(
    *,
    base_grim: Optional[PathValue] = None,
    surface_mesh: Optional[PathValue] = None,
    coordinate_units: Optional[str] = None,
    surface_units: Optional[str] = None,
    point_locations_csv: Optional[PathValue] = None,
    line_locations_csv: Optional[PathValue] = None,
    enabled_point_placement_ids: Optional[Sequence[str]] = None,
    enabled_line_ids: Optional[Sequence[str]] = None,
    base_dir: Optional[PathValue] = None,
) -> FeatureInputPreview:
    """Prepare an input-only CAD preview without response-dataset mappings.

    The same strict placement parsers and unit conversions used by local/HPC
    feature assembly are used here. No electromagnetic result is produced and
    this preview is not evidence that skin distance, outward normals, response
    metadata, or coherent compatibility have passed.
    """

    if not any((base_grim, surface_mesh, point_locations_csv, line_locations_csv)):
        raise ValueError(
            "Select a base GRIM, STL/facet mesh, or placement CSV to preview."
        )

    surface_scale: Optional[float] = None
    if surface_mesh is not None:
        surface_scale = _required_unit_scale(
            surface_units,
            label="surface_units",
            used_for="surface_mesh",
        )
    coordinate_scale = 1.0
    if point_locations_csv is not None or line_locations_csv is not None:
        coordinate_scale = _required_unit_scale(
            coordinate_units,
            label="coordinate_units",
            used_for="a point or line placement CSV",
        )

    profile: Optional[np.ndarray] = None
    body_source = "none"
    if base_grim is not None:
        base = resolve_path(base_grim, base_dir=base_dir)
        if not base.is_file():
            raise FileNotFoundError(f"Base monostatic GRIM not found: {base}")
        embedded_grid = load_body_requested_radar_grid(str(base))
        if embedded_grid is not None:
            profile = np.array(
                load_body_profile_grim(str(base)), dtype=float, copy=True
            )
            body_source = "embedded_bor_profile"

    surface_triangles: Optional[np.ndarray] = None
    if surface_mesh is not None:
        surface_path = resolve_path(surface_mesh, base_dir=base_dir)
        if not surface_path.is_file():
            raise FileNotFoundError(f"Surface mesh not found: {surface_path}")
        assert surface_scale is not None
        surface_triangles = (
            np.asarray(read_surface_mesh(str(surface_path)), dtype=float)
            * surface_scale
        )
        body_source = "surface_mesh"

    point_rows = (
        read_point_placement_csv(point_locations_csv, base_dir=base_dir)
        if point_locations_csv is not None else []
    )
    line_rows = (
        read_line_placement_csv(line_locations_csv, base_dir=base_dir)
        if line_locations_csv is not None else []
    )
    all_point_rows = list(point_rows)
    all_line_rows = list(line_rows)
    point_rows = _filter_enabled_rows(
        point_rows,
        id_key="placement_id",
        enabled_ids=enabled_point_placement_ids,
        label="point placement",
    )
    line_rows = _filter_enabled_rows(
        line_rows,
        id_key="line_id",
        enabled_ids=enabled_line_ids,
        label="line path",
    )
    # Input preview is also the clean-body comparison view.  An all-disabled
    # feature mask therefore renders the body with no feature artists while
    # retaining the complete parsed descriptor catalog for re-enabling items.
    # Authoritative validation/build still rejects an empty feature set.
    point_groups: dict[str, list[np.ndarray]] = {}
    point_id_groups: dict[str, list[str]] = {}
    point_normal_groups: dict[str, list[np.ndarray]] = {}
    point_roll_groups: dict[str, list[np.ndarray]] = {}
    for row in point_rows:
        dataset_id = str(row["dataset_id"])
        point_id_groups.setdefault(dataset_id, []).append(
            str(row["placement_id"])
        )
        point_groups.setdefault(dataset_id, []).append(
            np.asarray([row["x"], row["y"], row["z"]], dtype=float)
            * coordinate_scale
        )
        point_normal_groups.setdefault(dataset_id, []).append(
            np.asarray([row["nx"], row["ny"], row["nz"]], dtype=float)
        )
        point_roll_groups.setdefault(dataset_id, []).append(
            np.asarray(
                [row["roll_x"], row["roll_y"], row["roll_z"]], dtype=float
            )
        )
    point_locations = {
        dataset_id: np.asarray(locations, dtype=float).reshape(-1, 3)
        for dataset_id, locations in point_groups.items()
    }
    line_paths = _input_line_preview_paths(
        line_rows, coordinate_scale=coordinate_scale
    )
    line_normals = _input_line_preview_normals(line_rows)
    requirements = FeatureDatasetRequirements(
        point_dataset_ids=_ordered_dataset_ids(all_point_rows),
        line_dataset_ids=_ordered_dataset_ids(all_line_rows),
        point_placement_count=len(all_point_rows),
        line_path_count=len({str(row["line_id"]) for row in all_line_rows}),
        line_segment_count=len(all_line_rows),
        point_instances=tuple(
            (str(row["placement_id"]), str(row["dataset_id"]))
            for row in all_point_rows
        ),
        line_instances=_line_instance_descriptors(all_line_rows),
    )
    geometry = FeaturePreviewGeometry(
        surface_triangles_cad_m=(
            None
            if surface_triangles is None
            else np.array(surface_triangles, dtype=float, copy=True)
        ),
        body_profile_rho_z_m=profile,
        point_locations_cad_m=point_locations,
        line_paths_cad_m=line_paths,
        point_normals_cad={
            dataset_id: np.asarray(vectors, dtype=float).reshape(-1, 3)
            for dataset_id, vectors in point_normal_groups.items()
        },
        point_roll_references_cad={
            dataset_id: np.asarray(vectors, dtype=float).reshape(-1, 3)
            for dataset_id, vectors in point_roll_groups.items()
        },
        line_endpoint_normals_cad=line_normals,
        point_placement_ids={
            dataset_id: tuple(placement_ids)
            for dataset_id, placement_ids in point_id_groups.items()
        },
    )
    return FeatureInputPreview(
        preview_geometry=geometry,
        dataset_requirements=requirements,
        body_source=body_source,
    )


def unit_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if (
        vector.shape != (3,)
        or not np.all(np.isfinite(vector))
        or magnitude <= 1.0e-12
    ):
        raise ValueError(f"{label} must be one finite nonzero 3-vector.")
    return vector / magnitude


def validate_normal_tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 180.0:
        raise ValueError(
            "normal_tol_deg must be finite and between 0 and 180."
        )
    return tolerance


def compute_skin_limit(
    frequencies_ghz: Sequence[float],
    *,
    skin_tol_m: float,
    skin_phase_tol_deg: float,
) -> tuple[float, float]:
    """Return the enforced distance and highest-frequency wavelength."""

    frequencies = np.asarray(frequencies_ghz, dtype=float).ravel()
    if (
        frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
    ):
        raise ValueError("frequencies_ghz must contain positive finite values.")
    distance_limit = float(skin_tol_m)
    phase_limit = float(skin_phase_tol_deg)
    if (
        not math.isfinite(distance_limit)
        or not math.isfinite(phase_limit)
        or distance_limit < 0.0
        or phase_limit < 0.0
    ):
        raise ValueError("Skin tolerances must be finite and nonnegative.")
    wavelength = C0 / (float(np.max(frequencies)) * 1.0e9)
    limit = min(distance_limit, phase_limit * wavelength / 720.0)
    return limit, wavelength


def _sample_perimeter(
    perimeter: np.ndarray, samples_per_segment: int = 33
) -> np.ndarray:
    """Return evenly sampled segment points for legacy placement callers.

    The feature workflow now performs its production surface query directly so
    it can obtain distance and normal in one pass, but ``place_features.py``
    still imports this compatibility helper for the local/HPC script API.
    """

    segments = np.asarray(perimeter, dtype=float)
    parameter = np.linspace(0.0, 1.0, max(2, int(samples_per_segment)))
    return (
        segments[:, 0, None, :] * (1.0 - parameter)[None, :, None]
        + segments[:, 1, None, :] * parameter[None, :, None]
    ).reshape(-1, 3)


def _surface_distances_points_and_normals(
    surface: TriangleSurface,
    points: np.ndarray,
    *,
    normal_hints: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query a production surface once for distance, registration, and normal.

    Lightweight service tests historically supplied surface-like doubles with
    only ``distance`` and ``normal`` methods, so retain that compatibility while
    ensuring a real :class:`TriangleSurface` performs one nearest-facet search.
    """

    query = np.atleast_2d(np.asarray(points, dtype=float))
    if isinstance(surface, TriangleSurface):
        distances, nearest_points, normals, _facet_indices = surface.nearest(
            query, normal_hints=normal_hints
        )
    else:  # Compatibility for injected surface-like test/service objects.
        distances = surface.distance(query)
        normals = surface.normal(query)
        # Lightweight injected surfaces predate registration support.  Their
        # callers do not enable production mesh shadowing, so retaining the
        # query point is the only non-invented compatibility behavior.
        nearest_points = query
    return (
        np.asarray(distances, dtype=float),
        np.asarray(nearest_points, dtype=float),
        np.asarray(normals, dtype=float),
    )


def _surface_distances_and_normals(
    surface: TriangleSurface,
    points: np.ndarray,
    *,
    normal_hints: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    distances, _nearest_points, normals = _surface_distances_points_and_normals(
        surface, points, normal_hints=normal_hints
    )
    return distances, normals


def _validate_bor_surface_agreement(
    profile: np.ndarray,
    surface: TriangleSurface,
    *,
    skin_limit_m: float,
    shadow_requested: bool,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Bind a selected mesh geometrically to the embedded analytic BoR skin."""

    triangles = np.asarray(surface.triangles, dtype=float)
    extent = max(1.0, float(surface.extent))
    agreement_limit = max(
        1.0e-9 * extent,
        float(skin_limit_m),
    )
    maximum_surface_error = 0.0
    # Vertices alone are insufficient on curved bodies: a coarse inscribed
    # triangle can have every vertex on the analytic BoR while its interior is
    # many wavelengths inside. Check vertices, edge midpoints, and centroids.
    for start in range(0, len(triangles), 1024):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Feature placement validation cancelled.")
        chunk = triangles[start:start + 1024]
        samples = np.concatenate((
            chunk.reshape(-1, 3),
            (0.5 * (chunk[:, 0] + chunk[:, 1])),
            (0.5 * (chunk[:, 1] + chunk[:, 2])),
            (0.5 * (chunk[:, 2] + chunk[:, 0])),
            np.mean(chunk, axis=1),
        ), axis=0)
        distances = surface_of_revolution_distance(
            profile, samples
        )
        maximum_surface_error = max(
            maximum_surface_error, float(np.max(distances))
        )
    if maximum_surface_error > agreement_limit + 1.0e-12:
        raise ValueError(
            "Selected surface mesh does not match the embedded BoR body skin: "
            f"a sampled facet point is {maximum_surface_error:.6g} m from the revolved "
            f"profile (allowed {agreement_limit:.6g} m). Check units, CAD "
            "frame/origin, and select the mesh derived from this body."
        )

    analytic_normal = surface_of_revolution_normal(profile)
    minimum_alignment = 1.0
    centroids = np.asarray(surface.centroids, dtype=float)
    for start in range(0, len(centroids), 4096):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Feature placement validation cancelled.")
        derived = analytic_normal(centroids[start:start + 4096])
        alignment = np.sum(
            derived * surface.face_normals[start:start + 4096], axis=1
        )
        minimum_alignment = min(
            minimum_alignment, float(np.min(alignment))
        )
    if minimum_alignment <= _OUTWARD_ALIGNMENT_EPS:
        raise ValueError(
            "Selected BoR surface mesh has inward or incompatible face "
            f"orientation (minimum analytic-normal alignment "
            f"{minimum_alignment:.6g}). Toggle Flip normals only when the "
            "entire mesh is consistently inward; mixed winding must be repaired."
        )

    maximum_coverage_error = None
    if shadow_requested:
        gen = np.asarray(profile, dtype=float)
        profile_samples = np.concatenate((
            gen,
            0.5 * (gen[:-1] + gen[1:]),
        ))
        phi = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        rho = profile_samples[:, 0, None]
        z = profile_samples[:, 1, None]
        coverage_points = np.stack((
            np.broadcast_to(rho * np.cos(phi)[None, :], (len(profile_samples), len(phi))),
            np.broadcast_to(rho * np.sin(phi)[None, :], (len(profile_samples), len(phi))),
            np.broadcast_to(z, (len(profile_samples), len(phi))),
        ), axis=-1).reshape(-1, 3)
        coverage_distances = []
        for start in range(0, len(coverage_points), 512):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Feature placement validation cancelled.")
            distances, _points, _normals, _indices = surface.nearest(
                coverage_points[start:start + 512]
            )
            coverage_distances.append(np.asarray(distances, dtype=float))
        maximum_coverage_error = float(np.max(np.concatenate(coverage_distances)))
        if maximum_coverage_error > agreement_limit + 1.0e-12:
            raise ValueError(
                "Selected shadow mesh does not cover the embedded BoR skin "
                f"within {agreement_limit:.6g} m (maximum sampled gap "
                f"{maximum_coverage_error:.6g} m). Use a closed, sufficiently "
                "fine revolution mesh derived from this exact profile."
            )
    return {
        "schema": SURFACE_BINDING_SCHEMA,
        "status": "analytic_embedded_bor_geometry_match",
        "maximum_mesh_sample_to_profile_m": float(maximum_surface_error),
        "maximum_profile_to_mesh_sample_m": maximum_coverage_error,
        "agreement_limit_m": float(agreement_limit),
        "minimum_outward_normal_alignment": float(minimum_alignment),
        "shadow_coverage_checked": bool(shadow_requested),
    }


def _resolved_dataset_paths(
    datasets: Mapping[str, PathValue],
    *,
    kind: str,
    base_dir: Optional[PathValue],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for dataset_id, value in datasets.items():
        if not isinstance(dataset_id, str):
            raise ValueError(f"{kind}_datasets keys must be strings.")
        if not dataset_id or dataset_id != dataset_id.strip():
            raise ValueError(
                f"{kind}_datasets keys must be nonempty strings without "
                "leading/trailing whitespace."
            )
        dataset = resolve_path(value, base_dir=base_dir)
        if not dataset.is_file():
            raise FileNotFoundError(
                f"{kind.capitalize()} dataset {dataset_id!r} does not exist: "
                f"{dataset}"
            )
        paths[dataset_id] = dataset
        hashes[dataset_id] = sha256_file(str(dataset))
    return paths, hashes


def validate_declared_feature_delta_response(
    dataset: PathValue,
) -> dict[str, Any]:
    """Enforce and describe the response role behind an Assembly mapping.

    The canonical filename grammar is authoritative when it explicitly says
    OPN or FRD.  Role-free GUI Coherent-minus results remain accepted under the
    existing mapping attestation even though their storage-domain tag is often
    ``power_phase`` rather than ``delta``.  The classification is retained in
    placement provenance so that compatibility is visible rather than silent.

    A non-archive compatibility double is deferred to the real response loader;
    this keeps injected service tests/backends working without weakening real
    GRIM metadata checks.
    """

    path = resolve_path(dataset)
    variation = require_role_free_declared_delta(str(path))
    metadata_access = "readable"
    metadata: dict[str, Any] = {}
    embedded_domain: Optional[str] = None
    try:
        stored_context = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        stored_context = None
        metadata_access = "deferred_to_response_loader"
    if stored_context is not None:
        if not hasattr(stored_context, "files"):
            metadata_access = "deferred_to_response_loader"
        else:
            try:
                with stored_context as stored:
                    if "rcs_domain" not in stored.files:
                        pass
                    else:
                        raw_domain = np.asarray(stored["rcs_domain"])
                        if raw_domain.size != 1:
                            raise ValueError(
                                "rcs_domain metadata must be scalar."
                            )
                        embedded_domain = str(
                            raw_domain.reshape(-1)[0]
                        ).strip()
                        metadata["rcs_domain"] = embedded_domain
            except (
                OSError,
                EOFError,
                TypeError,
                ValueError,
                zipfile.BadZipFile,
            ) as exc:
                raise ValueError(
                    f"{path}: advertised rcs_domain metadata is unreadable or "
                    "malformed; it cannot be overridden by a declared delta role."
                ) from exc
    status = (
        validate_declared_coherent_delta_domain(metadata, str(path))
        if metadata_access == "readable"
        else "response_loader_deferred"
    )
    return {
        "schema": DECLARED_FEATURE_DELTA_RESPONSE_SCHEMA,
        "filename_role": "role_free",
        "filename_variation": variation,
        "embedded_rcs_domain": embedded_domain,
        "metadata_access": metadata_access,
        "status": status,
    }


def _require_known_dataset_ids(
    rows: Sequence[Mapping[str, Any]],
    dataset_paths: Mapping[str, Path],
    *,
    coordinates: Path,
) -> None:
    unknown = sorted(
        {str(row["dataset_id"]) for row in rows} - set(dataset_paths)
    )
    if unknown:
        raise ValueError(
            f"{coordinates}: unknown dataset_id value(s) {unknown}; configured "
            f"IDs are {sorted(dataset_paths)}."
        )


def prepare_line_placements(
    profile: Optional[np.ndarray],
    surface: Optional[TriangleSurface],
    *,
    coordinate_scale: float,
    skin_limit_m: float,
    wavelength_m: float,
    normal_tolerance_deg: float,
    locations_csv: Optional[PathValue],
    datasets: Mapping[str, PathValue],
    enabled_line_ids: Optional[Sequence[str]] = None,
    base_dir: Optional[PathValue] = None,
    preview_paths_cad_m: Optional[dict[str, dict[str, np.ndarray]]] = None,
    preview_endpoint_normals_cad: Optional[
        dict[str, dict[str, np.ndarray]]
    ] = None,
    prepare_shadow_origins: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and prepare line-expanded feature placements."""

    if locations_csv is None:
        if enabled_line_ids:
            raise ValueError(
                "enabled line path IDs require line_locations_csv."
            )
        if datasets:
            raise ValueError(
                "line_datasets is configured but line_locations_csv is None."
            )
        return [], []
    normal_tolerance = validate_normal_tolerance(normal_tolerance_deg)
    coordinates = resolve_path(locations_csv, base_dir=base_dir)
    rows = read_line_placement_csv(coordinates)
    rows = _filter_enabled_rows(
        rows,
        id_key="line_id",
        enabled_ids=enabled_line_ids,
        label="line path",
    )
    if not rows:
        return [], []
    if not datasets:
        raise ValueError(
            "enabled line paths require at least one mapped line response."
        )
    coordinates_sha256 = sha256_file(str(coordinates))
    active_dataset_ids = set(_ordered_dataset_ids(rows))
    dataset_paths, dataset_hashes = _resolved_dataset_paths(
        {
            dataset_id: value
            for dataset_id, value in datasets.items()
            if dataset_id in active_dataset_ids
        },
        kind="line",
        base_dir=base_dir,
    )
    _require_known_dataset_ids(rows, dataset_paths, coordinates=coordinates)
    dataset_role_validations = {
        dataset_id: validate_declared_feature_delta_response(dataset)
        for dataset_id, dataset in dataset_paths.items()
    }

    if surface is None:
        if profile is None:
            raise ValueError(
                "Line placement requires a BoR body profile or triangle surface."
            )
        normal_fn = surface_of_revolution_normal(profile)
    else:
        normal_fn = surface.normal

    scale = float(coordinate_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale must be positive and finite.")
    limit = float(skin_limit_m)
    wavelength = float(wavelength_m)
    if (
        not math.isfinite(limit)
        or limit < 0.0
        or not math.isfinite(wavelength)
        or wavelength <= 0.0
    ):
        raise ValueError("Skin limit and wavelength are invalid.")

    placements: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    start = 0
    while start < len(rows):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Feature placement validation cancelled.")
        line_id = str(rows[start]["line_id"])
        end = start + 1
        while end < len(rows) and rows[end]["line_id"] == line_id:
            end += 1
        group = rows[start:end]
        dataset_id = str(group[0]["dataset_id"])
        dataset = dataset_paths[dataset_id]
        perimeter_cad = np.asarray([
            [
                [row["x1"], row["y1"], row["z1"]],
                [row["x2"], row["y2"], row["z2"]],
            ]
            for row in group
        ], dtype=float) * scale
        normal_cad = np.asarray([
            [
                [row["n1x"], row["n1y"], row["n1z"]],
                [row["n2x"], row["n2y"], row["n2z"]],
            ]
            for row in group
        ], dtype=float)
        perimeter = to_axis_frame(perimeter_cad)
        segment_normals = to_axis_frame(normal_cad)
        lengths = np.linalg.norm(perimeter[:, 1] - perimeter[:, 0], axis=1)
        if np.any(lengths <= 0.0):
            index = int(np.flatnonzero(lengths <= 0.0)[0])
            raise ValueError(
                f"{coordinates}:line {group[index]['_csv_line']} has a "
                "zero-length segment."
            )
        extent = float(np.max(np.ptp(perimeter.reshape(-1, 3), axis=0)))
        continuity_tolerance = max(
            1.0e-12,
            1.0e-6 * max(extent, float(np.max(lengths))),
        )
        for index in range(len(perimeter) - 1):
            gap = float(np.linalg.norm(
                perimeter[index, 1] - perimeter[index + 1, 0]
            ))
            if gap > continuity_tolerance:
                raise ValueError(
                    f"{coordinates}: lines {group[index]['_csv_line']} and "
                    f"{group[index + 1]['_csv_line']} of line_id {line_id!r} "
                    f"are not head-to-tail (gap {gap:.3e} m)."
                )
        path_tangents = (
            perimeter[:, 1] - perimeter[:, 0]
        ) / lengths[:, None]
        tangent_pairs = [
            (index, index + 1) for index in range(len(perimeter) - 1)
        ]
        if (
            len(perimeter) > 1
            and float(np.linalg.norm(perimeter[-1, 1] - perimeter[0, 0]))
            <= continuity_tolerance
        ):
            tangent_pairs.append((len(perimeter) - 1, 0))
        for left_index, right_index in tangent_pairs:
            tangent_dot = float(
                path_tangents[left_index] @ path_tangents[right_index]
            )
            if tangent_dot <= -1.0 + 1.0e-10:
                raise ValueError(
                    f"{coordinates}: line_id {line_id!r} immediately "
                    f"backtracks between segments {left_index + 1} and "
                    f"{right_index + 1}. This double-counts the same physical "
                    "line; remove the overlapping return segment or split the "
                    "intended geometry into independently validated features."
                )

        normal_magnitudes = np.linalg.norm(segment_normals, axis=2)
        if np.any(normal_magnitudes <= 1.0e-12):
            segment_index, endpoint_index = np.argwhere(
                normal_magnitudes <= 1.0e-12
            )[0]
            raise ValueError(
                f"{coordinates}:line {group[int(segment_index)]['_csv_line']} "
                f"endpoint {int(endpoint_index) + 1} has a zero-length normal."
            )
        segment_normals = segment_normals / normal_magnitudes[:, :, None]

        # Use at least the exact highest-frequency solver subdivision density
        # (0.05 lambda), while retaining the historical 32 intervals on short
        # segments.  A fixed 33 samples per *input* segment could certify a
        # metre-long chord from only a few points even though the solver later
        # evaluated hundreds of pieces and crossed a narrow facet/gap between
        # them.
        maximum_solver_piece_m = 0.05 * wavelength
        validation_piece_counts = np.maximum(
            32,
            np.ceil(lengths / maximum_solver_piece_m).astype(np.int64),
        )
        distance_points_list = []
        normal_points_list = []
        supplied_list = []
        normal_parameters = []
        normal_segment_indices = []
        for segment_index, sample_count in enumerate(validation_piece_counts):
            count = int(sample_count)
            closed = np.linspace(0.0, 1.0, count + 1)
            opened = (np.arange(count, dtype=float) + 0.5) / count
            start_point, end_point = perimeter[segment_index]
            start_normal, end_normal = segment_normals[segment_index]
            distance_points_list.append(
                start_point[None, :] * (1.0 - closed[:, None])
                + end_point[None, :] * closed[:, None]
            )
            normal_points_list.append(
                start_point[None, :] * (1.0 - opened[:, None])
                + end_point[None, :] * opened[:, None]
            )
            supplied_list.append(
                start_normal[None, :] * (1.0 - opened[:, None])
                + end_normal[None, :] * opened[:, None]
            )
            normal_parameters.extend(float(value) for value in opened)
            normal_segment_indices.extend([segment_index] * count)
        distance_points = np.concatenate(distance_points_list, axis=0)
        normal_points = np.concatenate(normal_points_list, axis=0)
        supplied = np.concatenate(supplied_list, axis=0)
        normal_parameters = np.asarray(normal_parameters, dtype=float)
        normal_segment_indices = np.asarray(normal_segment_indices, dtype=int)
        supplied_magnitudes = np.linalg.norm(supplied, axis=1)
        if np.any(supplied_magnitudes <= 1.0e-12):
            raise ValueError(
                f"{coordinates}: line_id {line_id!r} endpoint-normal "
                "interpolation becomes singular; subdivide the line and "
                "supply the outward normal at the added vertex."
            )
        supplied /= supplied_magnitudes[:, None]
        solver_midpoints = None
        solver_normals = None
        if prepare_shadow_origins:
            # Freeze one highest-frequency piece grid for every requested
            # frequency. Lower frequencies are over-resolved rather than
            # changing visibility samples, so registered origins map exactly
            # to the numerical pieces.
            (
                _solver_starts,
                _solver_path_tangents,
                _solver_piece_lengths,
                solver_midpoints,
                solver_normals,
                _solver_frame_tangents,
            ) = prepare_perimeter_frame(
                perimeter,
                maximum_solver_piece_m,
                segment_normals=segment_normals,
            )
        shadow_points = None
        if surface is None:
            surface_distances = surface_of_revolution_distance(
                profile, distance_points
            )
            offset = float(np.max(surface_distances))
            derived = np.asarray(normal_fn(normal_points), dtype=float)
        else:
            # One batched exact surface query preserves the single-query
            # performance contract.  Closed samples own the distance gate;
            # open samples own the normal gate.  Hints resolve the remaining
            # deliberate case of a line lying exactly on a shared mesh edge.
            query_groups = [distance_points, normal_points]
            hint_groups = []
            distance_hints_list = []
            for segment_index, sample_count in enumerate(
                validation_piece_counts
            ):
                closed = np.linspace(0.0, 1.0, int(sample_count) + 1)
                start_normal, end_normal = segment_normals[segment_index]
                distance_hints_list.append(
                    start_normal[None, :] * (1.0 - closed[:, None])
                    + end_normal[None, :] * closed[:, None]
                )
            distance_hints = np.concatenate(distance_hints_list, axis=0)
            distance_hint_magnitudes = np.linalg.norm(distance_hints, axis=1)
            if np.any(distance_hint_magnitudes <= 1.0e-12):
                raise ValueError(
                    f"{coordinates}: line_id {line_id!r} endpoint-normal "
                    "interpolation becomes singular; subdivide the line and "
                    "supply the outward normal at the added vertex."
                )
            distance_hints /= distance_hint_magnitudes[:, None]
            hint_groups.extend((distance_hints, supplied))
            reuse_normal_registrations = bool(
                prepare_shadow_origins
                and len(solver_midpoints) == len(normal_points)
                and np.array_equal(solver_midpoints, normal_points)
            )
            if prepare_shadow_origins and not reuse_normal_registrations:
                query_groups.append(solver_midpoints)
                hint_groups.append(solver_normals)
            query_points = np.concatenate(query_groups)
            query_hints = np.concatenate(hint_groups)
            (
                surface_distances,
                nearest_points,
                query_normals,
            ) = _surface_distances_points_and_normals(
                surface, query_points, normal_hints=query_hints
            )
            if (
                surface_distances.shape != (len(query_points),)
                or not np.all(np.isfinite(surface_distances))
                or np.any(surface_distances < 0.0)
            ):
                raise ValueError("surface distance query returned invalid values.")
            offset = float(np.max(surface_distances[:len(distance_points)]))
            normal_end = len(distance_points) + len(normal_points)
            derived = query_normals[len(distance_points):normal_end]
            if prepare_shadow_origins:
                registered = (
                    nearest_points[len(distance_points):normal_end]
                    if reuse_normal_registrations
                    else nearest_points[normal_end:]
                )
                shadow_points = np.array(
                    registered, dtype=float, copy=True
                )
                shadow_points.setflags(write=False)
        if offset > limit:
            raise ValueError(
                f"{coordinates}: line_id {line_id!r} is {offset * 1e3:.3f} mm "
                f"off the skin ({720.0 * offset / wavelength:.1f} deg two-way "
                f"phase); allowed {limit * 1e3:.3f} mm."
            )

        if derived.shape != normal_points.shape or not np.all(np.isfinite(derived)):
            raise ValueError("surface normal query returned invalid vectors.")
        derived_magnitudes = np.linalg.norm(derived, axis=1)
        if np.any(derived_magnitudes <= 1.0e-12):
            raise ValueError("surface normal query returned a zero-length vector.")
        derived /= derived_magnitudes[:, None]
        alignments = np.clip(
            np.sum(supplied * derived, axis=1), -1.0, 1.0
        )
        differences = np.degrees(np.arccos(alignments))
        if np.any(alignments <= _OUTWARD_ALIGNMENT_EPS):
            flat_index = int(np.argmin(alignments))
            segment_index = int(normal_segment_indices[flat_index])
            raise ValueError(
                f"{coordinates}:line {group[segment_index]['_csv_line']} "
                "supplied normal interpolation is not an outward skin normal "
                f"({differences[flat_index]:.2f} deg at segment fraction "
                f"{normal_parameters[flat_index]:.5g}); outward normals must "
                "have a positive dot product with the body normal."
            )
        if np.any(differences > normal_tolerance):
            flat_index = int(np.argmax(differences))
            segment_index = int(normal_segment_indices[flat_index])
            raise ValueError(
                f"{coordinates}:line {group[segment_index]['_csv_line']} "
                "supplied normal interpolation differs from the outward skin "
                f"normal by {differences[flat_index]:.2f} deg at segment "
                f"fraction {normal_parameters[flat_index]:.5g}."
            )

        if not prepare_shadow_origins:
            analytic_length = constant_normal_piece_length(perimeter, segment_normals)
            if analytic_length is not None:
                maximum_solver_piece_m = analytic_length
        placements.append({
            "delta": str(dataset),
            "perimeter": perimeter,
            "segment_normals": segment_normals,
            "line_id": line_id,
            "kind": "delta",
            "declared_coherent_delta": True,
            "delta_sign": 1.0,
            "max_piece_length_m": float(maximum_solver_piece_m),
            **({} if shadow_points is None else {"shadow_points": shadow_points}),
        })
        records.append({
            "schema": LINE_PLACEMENT_SCHEMA,
            "kind": "line_2d_delta",
            "dataset": str(dataset),
            "dataset_sha256": dataset_hashes[dataset_id],
            "line_id": line_id,
            "dataset_id": dataset_id,
            "segment_count": len(group),
            "coordinates": str(coordinates),
            "first_csv_line": group[0]["_csv_line"],
            "last_csv_line": group[-1]["_csv_line"],
            "coordinates_sha256": coordinates_sha256,
            "max_skin_offset_m": float(offset),
            "max_normal_error_deg": float(np.max(differences)),
            "input_subtraction_order": "OPN-FRD (featured-clean)",
            "response_role_validation": dict(
                dataset_role_validations[dataset_id]
            ),
            "normal_source": "csv_endpoint_interpolation",
            "skin_validation_sample_count": int(len(distance_points)),
            "normal_validation_sample_count": int(len(normal_points)),
            "maximum_solver_piece_length_m": float(maximum_solver_piece_m),
        })
        if preview_paths_cad_m is not None:
            line_path = np.concatenate(
                [perimeter_cad[:, 0, :], perimeter_cad[-1:, 1, :]],
                axis=0,
            )
            preview_paths_cad_m.setdefault(dataset_id, {})[line_id] = np.array(
                line_path, dtype=float, copy=True
            )
        if preview_endpoint_normals_cad is not None:
            preview_endpoint_normals_cad.setdefault(dataset_id, {})[
                line_id
            ] = np.array(normal_cad, dtype=float, copy=True)
        start = end
    return placements, records


def prepare_point_placements(
    profile: Optional[np.ndarray],
    surface: Optional[TriangleSurface],
    *,
    coordinate_scale: float,
    skin_limit_m: float,
    wavelength_m: float,
    normal_tolerance_deg: float,
    locations_csv: Optional[PathValue],
    datasets: Mapping[str, PathValue],
    enabled_point_placement_ids: Optional[Sequence[str]] = None,
    base_dir: Optional[PathValue] = None,
    pattern_loader: Optional[Callable[..., Any]] = None,
    preview_locations_cad_m: Optional[dict[str, list[np.ndarray]]] = None,
    preview_normals_cad: Optional[dict[str, list[np.ndarray]]] = None,
    preview_roll_references_cad: Optional[
        dict[str, list[np.ndarray]]
    ] = None,
    prepare_shadow_origins: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and prepare compact 3-D point-feature placements."""

    if locations_csv is None:
        if enabled_point_placement_ids:
            raise ValueError(
                "enabled point placement IDs require point_locations_csv."
            )
        if datasets:
            raise ValueError(
                "point_datasets is configured but point_locations_csv is None."
            )
        return [], []
    normal_tolerance = validate_normal_tolerance(normal_tolerance_deg)
    coordinates = resolve_path(locations_csv, base_dir=base_dir)
    rows = read_point_placement_csv(coordinates)
    rows = _filter_enabled_rows(
        rows,
        id_key="placement_id",
        enabled_ids=enabled_point_placement_ids,
        label="point placement",
    )
    if not rows:
        return [], []
    if not datasets:
        raise ValueError(
            "enabled point placements require at least one mapped point response."
        )
    coordinates_sha256 = sha256_file(str(coordinates))
    active_dataset_ids = set(_ordered_dataset_ids(rows))
    dataset_paths, dataset_hashes = _resolved_dataset_paths(
        {
            dataset_id: value
            for dataset_id, value in datasets.items()
            if dataset_id in active_dataset_ids
        },
        kind="point",
        base_dir=base_dir,
    )
    _require_known_dataset_ids(rows, dataset_paths, coordinates=coordinates)
    dataset_role_validations = {
        dataset_id: validate_declared_feature_delta_response(dataset)
        for dataset_id, dataset in dataset_paths.items()
    }

    if surface is None:
        if profile is None:
            raise ValueError(
                "Point placement requires a BoR body profile or triangle surface."
            )
        normal_fn = surface_of_revolution_normal(profile)
    else:
        normal_fn = surface.normal

    scale = float(coordinate_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale must be positive and finite.")
    limit = float(skin_limit_m)
    wavelength = float(wavelength_m)
    if (
        not math.isfinite(limit)
        or limit < 0.0
        or not math.isfinite(wavelength)
        or wavelength <= 0.0
    ):
        raise ValueError("Skin limit and wavelength are invalid.")

    load_pattern = prepare_point_pattern if pattern_loader is None else pattern_loader
    points: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    patterns: dict[str, Any] = {}
    for row_index, row in enumerate(rows, 1):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Feature placement validation cancelled.")
        csv_line = int(row["_csv_line"])
        dataset_id = str(row["dataset_id"])
        dataset = dataset_paths[dataset_id]
        if dataset_id not in patterns:
            patterns[dataset_id] = load_pattern(
                str(dataset),
                declared_coherent_delta=True,
                delta_sign=1.0,
                assume_missing_cross_pol_zero=False,
            )
        pattern = patterns[dataset_id]
        location_cad_m = (
            np.array([row["x"], row["y"], row["z"]], dtype=float) * scale
        )
        location = to_axis_frame(location_cad_m)
        normal = unit_vector(
            to_axis_frame([row["nx"], row["ny"], row["nz"]]),
            "supplied normal",
        )
        if surface is None:
            offset = float(
                surface_of_revolution_distance(profile, location[None, :])[0]
            )
            derived_value = normal_fn(location[None, :])[0]
            shadow_location = location
        else:
            (
                surface_distances,
                nearest_points,
                surface_normals,
            ) = _surface_distances_points_and_normals(
                surface, location[None, :], normal_hints=normal[None, :]
            )
            if (
                surface_distances.shape != (1,)
                or not np.all(np.isfinite(surface_distances))
                or surface_distances[0] < 0.0
            ):
                raise ValueError("surface distance query returned invalid values.")
            offset = float(surface_distances[0])
            if surface_normals.shape != (1, 3):
                raise ValueError("surface normal query returned invalid vectors.")
            derived_value = surface_normals[0]
            shadow_location = np.array(
                nearest_points[0], dtype=float, copy=True
            )
            shadow_location.setflags(write=False)
        if offset > limit:
            raise ValueError(
                f"{coordinates}:line {csv_line} is {offset * 1e3:.3f} mm off "
                f"the skin ({720.0 * offset / wavelength:.1f} deg two-way phase)."
            )
        derived = unit_vector(derived_value, "derived normal")
        alignment = float(np.clip(float(normal @ derived), -1.0, 1.0))
        difference = math.degrees(math.acos(alignment))
        if alignment <= _OUTWARD_ALIGNMENT_EPS:
            raise ValueError(
                f"{coordinates}:line {csv_line} supplied normal is not an "
                f"outward skin normal ({difference:.2f} deg); outward normals "
                "must have a positive dot product with the body normal."
            )
        if difference > normal_tolerance:
            raise ValueError(
                f"{coordinates}:line {csv_line} supplied normal differs "
                f"from the outward skin normal by {difference:.2f} deg."
            )
        roll = unit_vector(to_axis_frame([
            row["roll_x"], row["roll_y"], row["roll_z"]
        ]), "roll reference")
        if np.linalg.norm(roll - float(roll @ normal) * normal) <= 1.0e-9:
            raise ValueError(
                f"{coordinates}:line {csv_line} roll reference is parallel "
                "to the supplied normal."
            )
        points.append({
            "pattern": pattern,
            "location": location,
            "aperture_normal": normal,
            "roll_ref": roll,
            "placement_id": str(row["placement_id"]),
            **(
                {"shadow_location": shadow_location}
                if prepare_shadow_origins else {}
            ),
        })
        records.append({
            "schema": POINT_PLACEMENT_SCHEMA,
            "kind": "compact_3d_delta",
            "dataset": str(dataset),
            "dataset_sha256": dataset_hashes[dataset_id],
            "placement_id": row["placement_id"],
            "dataset_id": dataset_id,
            "coordinates": str(coordinates),
            "row": row_index,
            "csv_line": csv_line,
            "coordinates_sha256": coordinates_sha256,
            "skin_offset_m": offset,
            "max_normal_error_deg": float(difference),
            "input_subtraction_order": "OPN-FRD (featured-clean)",
            "response_role_validation": dict(
                dataset_role_validations[dataset_id]
            ),
            "assumed_missing_cross_pol_zero": False,
            "roll_reference": "csv",
        })
        if preview_locations_cad_m is not None:
            preview_locations_cad_m.setdefault(dataset_id, []).append(
                np.array(location_cad_m, dtype=float, copy=True)
            )
        if preview_normals_cad is not None:
            preview_normals_cad.setdefault(dataset_id, []).append(
                np.asarray([row["nx"], row["ny"], row["nz"]], dtype=float)
            )
        if preview_roll_references_cad is not None:
            preview_roll_references_cad.setdefault(dataset_id, []).append(
                np.asarray(
                    [row["roll_x"], row["roll_y"], row["roll_z"]],
                    dtype=float,
                )
            )
    return points, records


def _library_unit_scale(units: str, *, label: str) -> float:
    """Use the canonical frame conversion without allowing a library exit."""

    try:
        return float(scale_for(units))
    except SystemExit as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc


def _required_unit_scale(
    units: Optional[str],
    *,
    label: str,
    used_for: str,
) -> float:
    """Return a unit scale only after a deliberate physical-unit choice."""

    if units is None or not str(units).strip():
        raise ValueError(
            f"{label} must be selected explicitly when {used_for} is configured."
        )
    return _library_unit_scale(str(units), label=label)


def surface_binding_path(surface_path: PathValue) -> Path:
    """Canonical reviewed binding sidecar for an external placement mesh."""

    return Path(str(Path(surface_path)) + ".assembly.json")


def validate_surface_binding(
    binding: Mapping[str, Any],
    *,
    base_grim_sha256: str,
    surface_sha256: str,
    surface_units: str,
) -> dict[str, Any]:
    """Validate an operator-reviewed external body/surface identity binding."""

    if not isinstance(binding, Mapping):
        raise ValueError("Assembly surface binding must be a JSON object.")
    required = {
        "schema", "base_grim_sha256", "surface_sha256", "surface_units",
        "frame_convention", "geometry_id", "attestation_case_id",
    }
    missing = sorted(required - set(binding))
    if missing:
        raise ValueError(f"Assembly surface binding is missing {missing}.")
    if str(binding["schema"]).strip() != SURFACE_BINDING_SCHEMA:
        raise ValueError(
            "Assembly surface binding schema must be "
            f"{SURFACE_BINDING_SCHEMA!r}."
        )

    def normalized_digest(value: Any, label: str) -> str:
        digest = str(value).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Assembly surface binding {label} is not SHA-256.")
        return digest

    bound_base = normalized_digest(
        binding["base_grim_sha256"], "base_grim_sha256"
    )
    bound_surface = normalized_digest(
        binding["surface_sha256"], "surface_sha256"
    )
    if bound_base != str(base_grim_sha256).strip().lower():
        raise ValueError(
            "Assembly surface binding names a different clean-body GRIM. "
            "Recreate/review the binding for the selected base response."
        )
    if bound_surface != str(surface_sha256).strip().lower():
        raise ValueError(
            "Assembly surface binding names different surface bytes. Recreate/"
            "review the binding after any mesh change."
        )
    bound_units = str(binding["surface_units"]).strip()
    if not math.isclose(
        _library_unit_scale(bound_units, label="binding surface_units"),
        _library_unit_scale(surface_units, label="surface_units"),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Assembly surface binding units do not match the selected surface "
            f"units ({bound_units!r} versus {surface_units!r})."
        )
    if str(binding["frame_convention"]).strip() != SURFACE_FRAME_CONVENTION:
        raise ValueError(
            "Assembly surface binding frame_convention must be exactly "
            f"{SURFACE_FRAME_CONVENTION!r}."
        )
    geometry_id = str(binding["geometry_id"]).strip()
    case_id = str(binding["attestation_case_id"]).strip()
    if not geometry_id or not case_id:
        raise ValueError(
            "Assembly surface binding geometry_id and attestation_case_id must "
            "be nonempty."
        )
    normalized = dict(binding)
    normalized.update({
        "schema": SURFACE_BINDING_SCHEMA,
        "base_grim_sha256": bound_base,
        "surface_sha256": bound_surface,
        "surface_units": bound_units,
        "frame_convention": SURFACE_FRAME_CONVENTION,
        "geometry_id": geometry_id,
        "attestation_case_id": case_id,
    })
    return normalized


def load_surface_binding(
    base_path: Path,
    surface_path: Path,
    *,
    base_grim_sha256: str,
    surface_sha256: str,
    surface_units: str,
) -> tuple[Optional[dict[str, Any]], Path, Optional[str]]:
    """Load the canonical external surface sidecar, failing on contradictions."""

    sidecar = surface_binding_path(surface_path)
    if not sidecar.is_file():
        return None, sidecar, None
    try:
        raw_bytes = sidecar.read_bytes()
        raw_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{sidecar}: surface binding is not UTF-8.") from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sidecar}: malformed surface binding JSON.") from exc
    normalized = validate_surface_binding(
        raw,
        base_grim_sha256=base_grim_sha256,
        surface_sha256=surface_sha256,
        surface_units=surface_units,
    )
    return normalized, sidecar, hashlib.sha256(raw_bytes).hexdigest()


def _surface_binding_inputs(
    base_grim: PathValue,
    surface_mesh: PathValue,
) -> tuple[Path, Path]:
    """Resolve the two exact files accepted by the reviewed binding tools."""

    base = resolve_path(base_grim)
    surface = resolve_path(surface_mesh)
    if not base.is_file():
        raise FileNotFoundError(f"External clean-body GRIM not found: {base}")
    if base.suffix.casefold() != ".grim":
        raise ValueError("External clean-body response must use the .grim extension.")
    if not surface.is_file():
        raise FileNotFoundError(f"Assembly surface not found: {surface}")
    if surface.suffix.casefold() not in {".stl", ".facet"}:
        raise ValueError(
            "Assembly surface must be an STL or indexed ASCII .facet file."
        )
    return base, surface


def _stable_binding_source_digest(path: Path) -> tuple[str, tuple[int, int, int]]:
    """Hash one binding source once and reject an ordinary concurrent write."""

    before = path.stat()
    digest = sha256_file(str(path))
    after = path.stat()
    before_key = (
        int(before.st_size), int(before.st_mtime_ns), int(before.st_ctime_ns)
    )
    after_key = (
        int(after.st_size), int(after.st_mtime_ns), int(after.st_ctime_ns)
    )
    if before_key != after_key:
        raise ValueError(
            f"{path} changed while it was being hashed. Wait for writes to "
            "finish, then retry the binding operation."
        )
    return digest, after_key


def _binding_stat_identity(path: Path) -> Optional[tuple[int, int, int]]:
    try:
        value = path.stat()
    except FileNotFoundError:
        return None
    return int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns)


def check_surface_binding(
    base_grim: PathValue,
    surface_mesh: PathValue,
    *,
    surface_units: str,
) -> tuple[dict[str, Any], Path]:
    """Check one canonical external-body binding against exact current bytes.

    This is intentionally an explicit operation for desktop clients: hashing a
    vehicle response and fine CAD mesh can be expensive, so callers should not
    invoke it from paint/readiness refresh loops.
    """

    base, surface = _surface_binding_inputs(base_grim, surface_mesh)
    base_digest, _base_stat = _stable_binding_source_digest(base)
    surface_digest, _surface_stat = _stable_binding_source_digest(surface)
    binding, sidecar, _binding_digest = load_surface_binding(
        base,
        surface,
        base_grim_sha256=base_digest,
        surface_sha256=surface_digest,
        surface_units=surface_units,
    )
    if binding is None:
        raise FileNotFoundError(
            f"Canonical Assembly surface binding not found: {sidecar}"
        )
    return binding, sidecar


def write_surface_binding(
    base_grim: PathValue,
    surface_mesh: PathValue,
    *,
    surface_units: str,
    geometry_id: str,
    attestation_case_id: str,
    attest_reviewed_registration: bool,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Atomically create the canonical reviewed external-body binding.

    The attestation records a responsible team's registration review; it does
    not infer or independently prove that the solve and CAD coordinate frames
    agree. The exact base and mesh bytes plus the selected mesh units are bound.
    """

    if not bool(attest_reviewed_registration):
        raise ValueError(
            "Creating a surface binding requires explicit reviewed-registration "
            "attestation."
        )
    geometry = str(geometry_id).strip()
    case_id = str(attestation_case_id).strip()
    if not geometry:
        raise ValueError("geometry_id must not be blank.")
    if not case_id:
        raise ValueError("attestation_case_id must not be blank.")
    base, surface = _surface_binding_inputs(base_grim, surface_mesh)
    sidecar = surface_binding_path(surface)
    replaced_identity = _binding_stat_identity(sidecar)
    if replaced_identity is not None and not overwrite:
        raise FileExistsError(
            f"Assembly surface binding already exists: {sidecar}. Confirm "
            "replacement explicitly to refresh it."
        )
    base_digest, base_stat = _stable_binding_source_digest(base)
    surface_digest, surface_stat = _stable_binding_source_digest(surface)
    binding = validate_surface_binding(
        {
            "schema": SURFACE_BINDING_SCHEMA,
            "base_grim_sha256": base_digest,
            "surface_sha256": surface_digest,
            "surface_units": str(surface_units).strip(),
            "frame_convention": SURFACE_FRAME_CONVENTION,
            "geometry_id": geometry,
            "attestation_case_id": case_id,
        },
        base_grim_sha256=base_digest,
        surface_sha256=surface_digest,
        surface_units=surface_units,
    )
    temporary = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        if not sidecar.parent.is_dir():
            raise ValueError(
                f"Surface binding parent directory does not exist: {sidecar.parent}"
            )
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(binding, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        current_base = base.stat()
        current_surface = surface.stat()
        if (
            (
                int(current_base.st_size),
                int(current_base.st_mtime_ns),
                int(current_base.st_ctime_ns),
            )
            != base_stat
            or (
                int(current_surface.st_size),
                int(current_surface.st_mtime_ns),
                int(current_surface.st_ctime_ns),
            )
            != surface_stat
        ):
            raise ValueError(
                "The clean-body GRIM or surface mesh changed while its binding "
                "was being created. Nothing was published; retry after writes finish."
            )
        current_sidecar_identity = _binding_stat_identity(sidecar)
        if current_sidecar_identity != replaced_identity:
            raise FileExistsError(
                f"Assembly surface binding changed during creation: {sidecar}. "
                "Nothing was replaced; review the newer sidecar and retry."
            )
        if current_sidecar_identity is not None and not overwrite:
            raise FileExistsError(
                f"Assembly surface binding appeared during creation: {sidecar}. "
                "Review it before choosing replacement."
            )
        os.replace(temporary, sidecar)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    checked, checked_path, _sidecar_digest = load_surface_binding(
        base,
        surface,
        base_grim_sha256=base_digest,
        surface_sha256=surface_digest,
        surface_units=surface_units,
    )
    if checked is None:  # Defensive: os.replace just published this path.
        raise RuntimeError(
            f"Assembly could not discover the surface binding just written: {sidecar}"
        )
    return checked, checked_path


_FEATURE_FRAME_CONVENTIONS = {
    "line": "line_local:+t=head_to_tail;+n=outward;+b=cross(t,n)",
    "point": POINT_PATTERN_FRAME_CONVENTION,
}
_FEATURE_PHASE_ORIGINS = {
    "line": "placement_line_on_host_outer_skin",
    "point": "placement_point_at_pattern_phase_center",
}


def _finite_manifest_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _manifest_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256 digest.")
    return digest


def _normalize_feature_validation_evidence(
    value: Any,
    *,
    response_content_sha256: str,
) -> list[dict[str, Any]]:
    """Validate machine-generated full-wave evidence bound to this response."""

    if not isinstance(value, list) or not value:
        raise ValueError(
            "A current validated feature manifest requires a nonempty "
            "validation.evidence array generated from passing full-wave cases."
        )
    normalized = []
    seen_case_ids = set()
    for index, raw in enumerate(value):
        label = f"validation.evidence[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be an object.")
        if str(raw.get("schema", "")).strip() != FEATURE_VALIDATION_EVIDENCE_SCHEMA:
            raise ValueError(
                f"{label}.schema must be {FEATURE_VALIDATION_EVIDENCE_SCHEMA!r}."
            )
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id or case_id in seen_case_ids:
            raise ValueError(
                f"{label}.case_id must be nonempty and unique within the manifest."
            )
        seen_case_ids.add(case_id)
        if raw.get("passed") is not True:
            raise ValueError(f"{label} must record passed=true.")
        report_sha256 = _manifest_sha256(
            raw.get("report_sha256"), f"{label}.report_sha256"
        )
        comparison_sha256 = _manifest_sha256(
            raw.get("comparison_sha256"), f"{label}.comparison_sha256"
        )
        evidence_response = _manifest_sha256(
            raw.get("feature_response_content_sha256"),
            f"{label}.feature_response_content_sha256",
        )
        if evidence_response != response_content_sha256:
            raise ValueError(
                f"{label} certifies feature response {evidence_response}, not "
                f"this manifest response {response_content_sha256}."
            )
        artifacts = raw.get("artifact_sha256")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(
            FEATURE_VALIDATION_ARTIFACT_ROLES
        ):
            raise ValueError(
                f"{label}.artifact_sha256 must contain exactly "
                f"{list(FEATURE_VALIDATION_ARTIFACT_ROLES)}."
            )
        normalized_artifacts = {
            role: _manifest_sha256(
                artifacts[role], f"{label}.artifact_sha256.{role}"
            )
            for role in FEATURE_VALIDATION_ARTIFACT_ROLES
        }
        raw_limits = raw.get("gate_limits")
        required_limits = {
            "active_floor_db",
            *FEATURE_VALIDATION_RELEASE_CEILINGS,
        }
        if not isinstance(raw_limits, Mapping) or set(raw_limits) != required_limits:
            raise ValueError(
                f"{label}.gate_limits must contain exactly "
                f"{sorted(required_limits)}."
            )
        gate_limits = {
            key: _finite_manifest_number(
                raw_limits[key], f"{label}.gate_limits.{key}"
            )
            for key in required_limits
        }
        if gate_limits["active_floor_db"] > FEATURE_VALIDATION_MAX_ACTIVE_FLOOR_DB:
            raise ValueError(
                f"{label}.gate_limits.active_floor_db="
                f"{gate_limits['active_floor_db']:g} excludes more weak-field "
                "samples than the Production maximum "
                f"{FEATURE_VALIDATION_MAX_ACTIVE_FLOOR_DB:g} dB."
            )
        for key in (
            "max_normalized_rms",
            "max_magnitude_p95_db",
            "max_phase_rms_deg",
        ):
            if not 0.0 <= gate_limits[key] <= FEATURE_VALIDATION_RELEASE_CEILINGS[key]:
                raise ValueError(
                    f"{label}.gate_limits.{key}={gate_limits[key]:g} is looser "
                    f"than the Production ceiling "
                    f"{FEATURE_VALIDATION_RELEASE_CEILINGS[key]:g}."
                )
        if not FEATURE_VALIDATION_RELEASE_CEILINGS[
            "min_coherence"
        ] <= gate_limits["min_coherence"] <= 1.0:
            raise ValueError(
                f"{label}.gate_limits.min_coherence={gate_limits['min_coherence']:g} "
                "is looser than the Production floor "
                f"{FEATURE_VALIDATION_RELEASE_CEILINGS['min_coherence']:g}."
            )
        normalized.append({
            "schema": FEATURE_VALIDATION_EVIDENCE_SCHEMA,
            "case_id": case_id,
            "passed": True,
            "report_sha256": report_sha256,
            "comparison_sha256": comparison_sha256,
            "feature_response_content_sha256": evidence_response,
            "artifact_sha256": normalized_artifacts,
            "gate_limits": {
                key: gate_limits[key] for key in sorted(gate_limits)
            },
        })
    return sorted(normalized, key=lambda item: item["case_id"])


def validate_feature_library_manifest(
    manifest: Mapping[str, Any], *, dataset_id: str, feature_kind: str
) -> dict[str, Any]:
    """Validate and normalize one reusable point/line response manifest.

    The manifest makes the local-model assumptions machine readable instead
    of leaving phase origin, subtraction sign, and validity
    envelope in a filename or tribal knowledge.  It is intentionally strict;
    legacy datasets remain usable only through the separately recorded
    missing-manifest compatibility path.
    """

    if feature_kind not in {"point", "line"}:
        raise ValueError("feature_kind must be 'point' or 'line'.")
    if not isinstance(manifest, Mapping):
        raise ValueError("Feature-library manifest must be a JSON object.")
    required = {
        "schema", "dataset_id", "feature_kind", "subtraction_order",
        "phase_origin", "frame_convention", "time_convention",
        "applicability", "validation", "response_content_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Feature-library manifest is missing {missing}.")
    unknown_identity = str(manifest["dataset_id"])
    if unknown_identity != dataset_id:
        raise ValueError(
            f"Feature-library manifest dataset_id={unknown_identity!r}; "
            f"expected {dataset_id!r}."
        )
    manifest_schema = str(manifest["schema"]).strip()
    if manifest_schema not in {
        FEATURE_LIBRARY_MANIFEST_SCHEMA,
        PREVIOUS_FEATURE_LIBRARY_MANIFEST_SCHEMA,
        LEGACY_FEATURE_LIBRARY_MANIFEST_SCHEMA,
    }:
        raise ValueError(
            f"Feature-library manifest schema={manifest['schema']!r}; require "
            f"{FEATURE_LIBRARY_MANIFEST_SCHEMA!r} (or Legacy-only "
            f"{PREVIOUS_FEATURE_LIBRARY_MANIFEST_SCHEMA!r}/"
            f"{LEGACY_FEATURE_LIBRARY_MANIFEST_SCHEMA!r})."
        )
    v1_manifest = manifest_schema == LEGACY_FEATURE_LIBRARY_MANIFEST_SCHEMA
    response_content_sha256 = str(
        manifest["response_content_sha256"]
    ).strip().lower()
    if len(response_content_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in response_content_sha256
    ):
        raise ValueError(
            "Feature-library manifest response_content_sha256 must be a "
            "64-character lowercase hexadecimal SHA-256 digest."
        )
    if str(manifest["feature_kind"]).strip().lower() != feature_kind:
        raise ValueError(
            f"Feature-library manifest feature_kind={manifest['feature_kind']!r}; "
            f"expected {feature_kind!r}."
        )
    if str(manifest["subtraction_order"]).strip().lower() != (
        "featured_minus_clean"
    ):
        raise ValueError(
            "Feature-library manifest subtraction_order must be exactly "
            "'featured_minus_clean'."
        )
    if str(manifest["time_convention"]).strip().replace(" ", "") != "exp(+jwt)":
        raise ValueError(
            "Feature-library manifest time_convention must be 'exp(+jwt)'."
        )
    expected_origin = _FEATURE_PHASE_ORIGINS[feature_kind]
    if str(manifest["phase_origin"]).strip() != expected_origin:
        raise ValueError(
            f"{feature_kind} manifest phase_origin must be "
            f"{expected_origin!r}."
        )
    expected_frame = _FEATURE_FRAME_CONVENTIONS[feature_kind]
    if str(manifest["frame_convention"]).strip() != expected_frame:
        raise ValueError(
            f"{feature_kind} manifest frame_convention must be "
            f"{expected_frame!r}."
        )

    host = manifest.get("host", {})
    if not isinstance(host, Mapping):
        raise ValueError("Feature-library manifest host must be an object.")
    material = str(host.get("material", "")).strip()
    if manifest_schema == FEATURE_LIBRARY_MANIFEST_SCHEMA and not material:
        raise ValueError("Current feature-library manifest requires host.material.")
    principal_radius = manifest.get("applicability", {}).get("minimum_principal_radius_m") if isinstance(manifest.get("applicability"), Mapping) else None
    if principal_radius is not None and (not math.isfinite(float(principal_radius)) or float(principal_radius) < 0):
        raise ValueError("minimum_principal_radius_m must be finite and nonnegative.")

    applicability = manifest["applicability"]
    if not isinstance(applicability, Mapping):
        raise ValueError(
            "Feature-library manifest applicability must be an object."
        )
    frequency = applicability.get("frequency_ghz")
    if not isinstance(frequency, Mapping):
        raise ValueError(
            "Feature-library manifest applicability.frequency_ghz must be an "
            "object with min and max."
        )
    frequency_min = _finite_manifest_number(
        frequency.get("min"), "applicability.frequency_ghz.min"
    )
    frequency_max = _finite_manifest_number(
        frequency.get("max"), "applicability.frequency_ghz.max"
    )
    if frequency_min <= 0.0 or frequency_max < frequency_min:
        raise ValueError(
            "Feature-library manifest frequency range must be positive and "
            "ordered."
        )
    footprint_radius = _finite_manifest_number(
        applicability.get("footprint_radius_m"),
        "applicability.footprint_radius_m",
    )
    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius_m must be positive.")

    conical_max: Optional[float] = None
    curvature_min: Optional[float] = None
    path_turn_max: Optional[float] = None
    if feature_kind == "line":
        curvature_min = _finite_manifest_number(
            applicability.get("minimum_along_line_normal_turn_radius_m"),
            "applicability.minimum_along_line_normal_turn_radius_m",
        )
        if curvature_min < 0.0:
            raise ValueError(
                "minimum_along_line_normal_turn_radius_m must be nonnegative."
            )
        path_turn_value = applicability.get("maximum_path_vertex_turn_deg")
        path_turn_max = (
            180.0
            if v1_manifest and path_turn_value is None
            else _finite_manifest_number(
                path_turn_value,
                "applicability.maximum_path_vertex_turn_deg",
            )
        )
        if not 0.0 <= path_turn_max <= 180.0:
            raise ValueError(
                "maximum_path_vertex_turn_deg must lie in [0, 180]."
            )
        conical_max = _finite_manifest_number(
            applicability.get("maximum_conical_incidence_deg"),
            "applicability.maximum_conical_incidence_deg",
        )
        if not 0.0 <= conical_max <= 90.0:
            raise ValueError(
                "maximum_conical_incidence_deg must lie in [0, 90]."
            )
        calibration = manifest.get("line_phase_calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError(
                "A line manifest requires line_phase_calibration with the "
                "exact solver mapping and independent calibration case IDs."
            )
        calibration_schema = str(calibration.get("schema", "")).strip()
        allowed_calibration_schemas = (
            {
                LINE_PHASE_CALIBRATION_SCHEMA,
                LEGACY_LINE_PHASE_CALIBRATION_SCHEMA,
            }
            if v1_manifest else {LINE_PHASE_CALIBRATION_SCHEMA}
        )
        if calibration_schema not in allowed_calibration_schemas:
            raise ValueError(
                "line_phase_calibration.schema must be "
                f"{LINE_PHASE_CALIBRATION_SCHEMA!r} for a current manifest."
            )
        calibrated_tm = _finite_manifest_number(
            calibration.get("tm_deg"), "line_phase_calibration.tm_deg"
        )
        calibrated_te = _finite_manifest_number(
            calibration.get("te_deg"), "line_phase_calibration.te_deg"
        )
        taper_value = calibration.get("grazing_taper_deg")
        calibrated_taper = (
            float(GRAZING_TAPER_DEG)
            if calibration_schema == LEGACY_LINE_PHASE_CALIBRATION_SCHEMA
            and taper_value is None
            else _finite_manifest_number(
                taper_value,
                "line_phase_calibration.grazing_taper_deg",
            )
        )
        if not math.isclose(calibrated_tm, PSI_HH_DEG, abs_tol=1.0e-12) or not (
            math.isclose(calibrated_te, PSI_VV_DEG, abs_tol=1.0e-12)
        ):
            raise ValueError(
                "line_phase_calibration does not match the phase mapping "
                f"executed by Assembly (TM={PSI_HH_DEG:g} deg, "
                f"TE={PSI_VV_DEG:g} deg)."
            )
        if not math.isclose(
            calibrated_taper, GRAZING_TAPER_DEG, abs_tol=1.0e-12
        ):
            raise ValueError(
                "line_phase_calibration.grazing_taper_deg does not match the "
                f"Assembly line model ({GRAZING_TAPER_DEG:g} deg)."
            )
        calibration_cases = calibration.get("case_ids")
        if not isinstance(calibration_cases, list) or not calibration_cases or any(
            not isinstance(value, str) or not value.strip()
            for value in calibration_cases
        ):
            raise ValueError(
                "line_phase_calibration.case_ids must contain at least one "
                "independent phase-calibration case ID."
            )

    validation = manifest["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("Feature-library manifest validation must be an object.")
    status = str(validation.get("status", "")).strip().lower()
    if status not in {"validated", "provisional", "uncertified"}:
        raise ValueError(
            "Feature-library manifest validation.status must be validated, "
            "provisional, or uncertified."
        )
    case_ids = validation.get("case_ids")
    if not isinstance(case_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in case_ids
    ):
        raise ValueError(
            "Feature-library manifest validation.case_ids must be a list of "
            "nonempty strings."
        )
    if status == "validated" and not case_ids:
        raise ValueError(
            "A validated feature-library manifest requires at least one "
            "validation.case_ids entry."
        )
    cleaned_case_ids = [value.strip() for value in case_ids]
    if len(set(cleaned_case_ids)) != len(cleaned_case_ids):
        raise ValueError(
            "Feature-library manifest validation.case_ids must be unique."
        )
    evidence: list[dict[str, Any]] = []
    if manifest_schema == FEATURE_LIBRARY_MANIFEST_SCHEMA:
        if status == "validated":
            evidence = _normalize_feature_validation_evidence(
                validation.get("evidence"),
                response_content_sha256=response_content_sha256,
            )
            evidence_case_ids = {item["case_id"] for item in evidence}
            if evidence_case_ids != set(cleaned_case_ids):
                raise ValueError(
                    "validation.case_ids must exactly match the passing "
                    "machine-generated validation.evidence case IDs."
                )
            if feature_kind == "line" and not set(
                value.strip() for value in calibration_cases
            ).issubset(evidence_case_ids):
                raise ValueError(
                    "line_phase_calibration.case_ids must refer to passing "
                    "full-wave evidence cases for this exact response."
                )
        elif validation.get("evidence") not in (None, []):
            raise ValueError(
                "A provisional or uncertified manifest must not carry passing "
                "validation.evidence."
            )

    normalized = json.loads(json.dumps(dict(manifest), sort_keys=True))
    normalized["response_content_sha256"] = response_content_sha256
    normalized["dataset_id"] = dataset_id
    normalized["feature_kind"] = feature_kind
    normalized["host"] = dict(host)
    normalized["host"]["material"] = material
    normalized["applicability"] = dict(applicability)
    normalized["applicability"]["frequency_ghz"] = {
        "min": frequency_min,
        "max": frequency_max,
    }
    normalized["applicability"]["footprint_radius_m"] = footprint_radius
    if curvature_min is not None:
        normalized["applicability"][
            "minimum_along_line_normal_turn_radius_m"
        ] = curvature_min
    if path_turn_max is not None:
        normalized["applicability"][
            "maximum_path_vertex_turn_deg"
        ] = path_turn_max
    if conical_max is not None:
        normalized["applicability"][
            "maximum_conical_incidence_deg"
        ] = conical_max
        normalized["line_phase_calibration"] = {
            "schema": LINE_PHASE_CALIBRATION_SCHEMA,
            "tm_deg": float(PSI_HH_DEG),
            "te_deg": float(PSI_VV_DEG),
            "grazing_taper_deg": float(GRAZING_TAPER_DEG),
            "case_ids": [value.strip() for value in calibration_cases],
        }
    normalized["validation"] = dict(validation)
    normalized["validation"]["status"] = status
    normalized["validation"]["case_ids"] = cleaned_case_ids
    if manifest_schema == FEATURE_LIBRARY_MANIFEST_SCHEMA:
        normalized["validation"]["evidence"] = evidence
    return normalized


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to a JSON object.")
    return value


def feature_response_content_sha256(dataset: PathValue) -> str:
    """Hash the exact serialized response payload, excluding its manifest.

    GRIM files are NPZ archives. Hashing each uncompressed member makes the
    identity independent of ZIP compression/order while binding every response
    array and metadata field. The manifest member is excluded so the same
    digest works for an adjacent sidecar or an embedded declaration without a
    self-referential file hash.
    """

    path = resolve_path(dataset)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [
                info for info in archive.infolist()
                if Path(info.filename).name
                != f"{FEATURE_LIBRARY_MANIFEST_KEY}.npy"
            ]
            names = [info.filename for info in members]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"{path}: response archive contains duplicate member names."
                )
            if not members:
                raise ValueError(f"{path}: response archive has no payload members.")
            digest = hashlib.sha256()
            digest.update(b"ghost-feature-response-content-v1\0")
            for info in sorted(members, key=lambda value: value.filename):
                name = info.filename.encode("utf-8")
                digest.update(len(name).to_bytes(8, "little"))
                digest.update(name)
                digest.update(int(info.file_size).to_bytes(8, "little"))
                with archive.open(info, "r") as stream:
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
            return digest.hexdigest()
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"{path}: feature response must be a readable GRIM/NPZ archive."
        ) from exc


def feature_response_physics_sha256(dataset: PathValue) -> str:
    """Hash canonical response axes/complex field and physical conventions.

    This identity deliberately ignores packaging and provenance such as ZIP
    order/compression, ``history``, and ``source_path``.  It is used only for
    duplicate physical-component detection; the manifest's separate content
    digest still binds every serialized payload member for integrity.
    """

    path = resolve_path(dataset)
    payload = _load_grim(str(path))
    digest = hashlib.sha256()
    digest.update(b"ghost-feature-response-physics-v1\0")

    def update_array(name: str, value: Any) -> None:
        array = np.asarray(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        if array.dtype.kind in "fc":
            canonical = np.ascontiguousarray(
                array,
                dtype="<c16" if array.dtype.kind == "c" else "<f8",
            )
            digest.update(canonical.dtype.str.encode("ascii") + b"\0")
            digest.update(canonical.tobytes())
        elif array.dtype.kind in "iub":
            canonical = np.ascontiguousarray(array, dtype="<i8")
            digest.update(b"<i8\0" + canonical.tobytes())
        else:
            for item in array.astype(str).reshape(-1):
                encoded = str(item).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little") + encoded)

    for key in ("azimuths", "elevations", "frequencies"):
        update_array(key, payload[key])
    aliases = {}
    for index, raw in enumerate(np.asarray(payload["polarizations"]).ravel()):
        label = str(raw).strip().upper()
        canonical = (
            "V" if label in {"VV", "V", "VERTICAL", "TE"}
            else "H" if label in {"HH", "H", "HORIZONTAL", "TM"}
            else "X" if label in {"VH", "HV"}
            else label
        )
        if canonical in aliases:
            raise ValueError(
                f"{path}: duplicate physical polarization alias {canonical}."
            )
        aliases[canonical] = index
    order_labels = [
        label for label in ("V", "H", "X") if label in aliases
    ] + sorted(set(aliases) - {"V", "H", "X"})
    order = [aliases[label] for label in order_labels]
    update_array("canonical_polarizations", np.asarray(order_labels))
    update_array(
        "complex_amplitude",
        np.asarray(payload["_amp"], dtype=np.complex128)[..., order],
    )
    return digest.hexdigest()


def load_feature_library_manifest(
    dataset: PathValue, *, dataset_id: str, feature_kind: str
) -> tuple[Optional[dict[str, Any]], list[dict[str, str]]]:
    """Load an embedded or adjacent feature-library manifest.

    Supported adjacent names are ``name.grim.feature.json`` and
    ``name.feature.json``.  When both an embedded and adjacent declaration are
    present they must be byte-semantically identical after JSON decoding.
    """

    path = resolve_path(dataset)
    candidates: list[
        tuple[str, dict[str, Any], Optional[Path], Optional[str]]
    ] = []
    try:
        stored_context = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError):
        # Response loaders perform the authoritative GRIM parse later. This
        # compatibility branch permits injected test/service loaders that are
        # not archives. Once an NPZ advertises the embedded key below, every
        # access/decode/JSON error is a hard failure.
        stored_context = None
    if stored_context is not None:
        with stored_context as stored:
            if FEATURE_LIBRARY_MANIFEST_KEY in stored.files:
                try:
                    raw = np.asarray(stored[FEATURE_LIBRARY_MANIFEST_KEY])
                    if raw.size != 1:
                        raise ValueError(
                            f"{FEATURE_LIBRARY_MANIFEST_KEY} must be scalar."
                        )
                    embedded = _json_object(
                        raw.reshape(()).item(),
                        label=f"{path}:{FEATURE_LIBRARY_MANIFEST_KEY}",
                    )
                except Exception as exc:
                    raise ValueError(
                        f"{path}: embedded feature-library manifest is "
                        "unreadable or malformed."
                    ) from exc
                candidates.append(("embedded", embedded, None, None))

    sidecars = list(dict.fromkeys((
        Path(str(path) + ".feature.json"),
        path.with_suffix(".feature.json"),
    )))
    existing_sidecars = [candidate for candidate in sidecars if candidate.is_file()]
    if len(existing_sidecars) > 1:
        raise ValueError(
            f"{path}: multiple feature-library sidecars exist: "
            f"{[str(value) for value in existing_sidecars]}. Keep one."
        )
    if existing_sidecars:
        sidecar = existing_sidecars[0]
        try:
            sidecar_bytes = sidecar.read_bytes()
            raw_manifest = json.loads(sidecar_bytes.decode("utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{sidecar}: feature-library sidecar is not valid UTF-8 JSON."
            ) from exc
        candidates.append((
            "sidecar",
            _json_object(raw_manifest, label=str(sidecar)),
            sidecar,
            hashlib.sha256(sidecar_bytes).hexdigest(),
        ))
    absent_sources = [
        {
            "source": "sidecar_absent",
            "path": str(candidate.resolve()),
            "absent": "true",
        }
        for candidate in sidecars
        if candidate not in existing_sidecars
    ]
    if not candidates:
        return None, absent_sources

    normalized = [
        validate_feature_library_manifest(
            value, dataset_id=dataset_id, feature_kind=feature_kind
        )
        for _source, value, _path, _digest in candidates
    ]
    reference = json.dumps(normalized[0], sort_keys=True, separators=(",", ":"))
    if any(
        json.dumps(value, sort_keys=True, separators=(",", ":")) != reference
        for value in normalized[1:]
    ):
        raise ValueError(
            f"{path}: embedded and adjacent feature-library manifests disagree."
        )
    actual_content_sha256 = feature_response_content_sha256(path)
    if normalized[0]["response_content_sha256"] != actual_content_sha256:
        raise ValueError(
            f"{path}: feature-library manifest is bound to response content "
            f"{normalized[0]['response_content_sha256']}, but this response is "
            f"{actual_content_sha256}. Regenerate the manifest for these exact "
            "GRIM bytes."
        )
    sources = list(absent_sources)
    for source, _value, source_path, source_digest in candidates:
        record = {"source": source}
        if source_path is not None:
            record.update({
                "path": str(source_path.resolve()),
                "sha256": str(source_digest),
            })
        sources.append(record)
    return normalized[0], sources


def assembly_sampling_warnings(radar_grid, lines, points):
    """Conservative phase-step diagnostics; not a body-model convergence claim."""
    extent = max([float(np.linalg.norm(p["location"])) for p in points]
                 + [float(np.max(np.linalg.norm(p["perimeter"], axis=-1))) for p in lines] + [0.])
    if extent == 0:
        return []
    frequency = np.asarray(radar_grid["frequencies_ghz"], float)
    wavelength = C0/(float(np.max(frequency))*1e9)
    warnings = []
    for key in ("azimuths_deg", "elevations_deg"):
        values = np.asarray(radar_grid[key], float)
        if len(values) < 2:
            continue
        step = np.deg2rad(np.max(np.diff(values)))
        phase_degrees = 720*extent*min(float(step), 2.)/wavelength
        if phase_degrees > 180:
            warnings.append(f"Sampling: {key} spacing permits up to {phase_degrees:.0f}° translated-feature phase change between stored looks (conservative bound). Narrow lobes/nulls may be missed; use a finer body/library grid and check convergence. This diagnostic does not bound body scattering or intrinsic feature angular variation.")
    if len(frequency) > 1:
        phase_degrees = 720*extent*float(np.max(np.diff(frequency)))*1e9/C0
        if phase_degrees > 180:
            warnings.append(f"Sampling: frequency spacing permits up to {phase_degrees:.0f}° translated-feature phase change. Refine stored frequency samples before interpreting broadband structure; this diagnostic does not certify intrinsic spectral variation.")
    return warnings


def _vehicle_radar_directions(radar_grid: Mapping[str, Any]) -> np.ndarray:
    azimuths, elevations = validate_radar_grid(
        radar_grid["azimuths_deg"], radar_grid["elevations_deg"]
    )
    R, _axis = _attitude(
        float(radar_grid["axis_az_deg"]),
        float(radar_grid["axis_el_deg"]),
        float(radar_grid.get("roll_deg", 0.0)),
    )
    az, el = np.meshgrid(np.deg2rad(azimuths), np.deg2rad(elevations), indexing="ij")
    earth = np.stack((np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)), axis=-1).reshape(-1, 3)
    return earth @ R


def _line_applicability_metrics(
    placement: Mapping[str, Any],
    radar_directions: np.ndarray,
    *,
    requested_frequencies_ghz: Sequence[float],
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Measure the exact installed line frame over every requested solve.

    The returned cut-angle ranges are calculated from the same piece grid and
    local ``(t,n,b)`` frame consumed by :func:`expand_perimeter`; validation can
    therefore prove that each coefficient table is sampleable before Build.
    """

    perimeter = np.asarray(placement["perimeter"], dtype=float)
    normals = np.asarray(placement["segment_normals"], dtype=float)
    frequencies = np.asarray(requested_frequencies_ghz, dtype=float).reshape(-1)
    if (
        frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
    ):
        raise ValueError("requested line frequencies must be positive and finite.")
    fixed_piece_length = placement.get("max_piece_length_m")
    def measure_frame(
        maximum_piece_length: float,
    ) -> tuple[float, float | None, float | None, int, int]:
        (
            _starts,
            _path_tangents,
            _piece_lengths,
            _midpoints,
            sampled_normals,
            frame_tangents,
        ) = prepare_perimeter_frame(
            perimeter,
            maximum_piece_length,
            segment_normals=normals,
        )
        # Bound every direction/piece product to 8192 pairs. In particular,
        # admission/validation must not allocate an entire look x piece grid.
        binormals = np.cross(frame_tangents, sampled_normals)
        lit_maximum_conical, low, high = 0.0, math.inf, -math.inf
        lit_count = lit_look_count = 0
        for look_start in range(0, len(radar_directions), 32):
            looks = radar_directions[look_start:look_start+32]
            lit_any = np.zeros(len(looks), dtype=bool)
            for piece_start in range(0, len(sampled_normals), 256):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("Line applicability validation cancelled.")
                sl = slice(piece_start, piece_start+256)
                normal_projection = looks @ sampled_normals[sl].T
                lit = normal_projection > 0.0
                lit_any |= np.any(lit, axis=1)
                if not np.any(lit):
                    continue
                conical_projection = np.abs(looks @ frame_tangents[sl].T)
                lit_maximum_conical = max(lit_maximum_conical, float(np.degrees(
                    np.arcsin(np.clip(np.max(conical_projection[lit]), 0.0, 1.0)))))
                binormal_projection = looks @ binormals[sl].T
                installed_phi = np.degrees(np.arctan2(normal_projection[lit], binormal_projection[lit]))
                low = min(low, float(np.min(installed_phi)))
                high = max(high, float(np.max(installed_phi)))
                lit_count += int(installed_phi.size)
            lit_look_count += int(np.count_nonzero(lit_any))
        cut_min = low if lit_count else None
        cut_max = high if lit_count else None
        return (
            lit_maximum_conical,
            cut_min,
            cut_max,
            lit_count,
            lit_look_count,
        )

    maximum_conical = 0.0
    cut_ranges = []
    if fixed_piece_length is not None:
        # A fixed installed piece grid is independent of frequency.  Measure it
        # once and report the identical geometric support for every requested
        # solve frequency.
        (
            maximum_conical,
            cut_min,
            cut_max,
            lit_count,
            lit_look_count,
        ) = measure_frame(
            float(fixed_piece_length)
        )
        measurements = [(
            cut_min,
            cut_max,
            lit_count,
            lit_look_count,
        )] * len(frequencies)
    else:
        measurements = []
        for frequency in frequencies:
            (
                measured_conical,
                cut_min,
                cut_max,
                lit_count,
                lit_look_count,
            ) = measure_frame(
                0.05 * C0 / (float(frequency) * 1.0e9)
            )
            maximum_conical = max(maximum_conical, measured_conical)
            measurements.append((
                cut_min,
                cut_max,
                lit_count,
                lit_look_count,
            ))

    for frequency, (
        cut_min,
        cut_max,
        lit_count,
        lit_look_count,
    ) in zip(frequencies, measurements):
        cut_ranges.append({
            "frequency_ghz": float(frequency),
            "minimum_deg": cut_min,
            "maximum_deg": cut_max,
            "lit_query_count": lit_count,
            "illuminated_requested_look_count": lit_look_count,
        })

    chords = perimeter[:, 1] - perimeter[:, 0]
    lengths = np.linalg.norm(chords, axis=1)
    path_tangents = chords / lengths[:, None]
    endpoint_norms = np.linalg.norm(normals, axis=2)
    normalized_normals = normals / endpoint_norms[:, :, None]
    endpoint_dot = np.clip(
        np.sum(
            normalized_normals[:, 0] * normalized_normals[:, 1], axis=1
        ),
        -1.0,
        1.0,
    )
    normal_turn = np.arccos(endpoint_dot)
    radii = np.full(len(lengths), math.inf, dtype=float)
    curved = normal_turn > 1.0e-12
    # Endpoint normals are linearly interpolated and then normalized by the
    # solver.  Its maximum angular rate occurs at the segment midpoint and is
    # 2*tan(theta/2)/L, not theta/L.
    radii[curved] = lengths[curved] / (
        2.0 * np.tan(0.5 * normal_turn[curved])
    )
    vertex_normal_jumps = []
    path_turns = []
    for index in range(len(perimeter) - 1):
        vertex_normal_jumps.append(math.degrees(math.acos(float(np.clip(
            normalized_normals[index, 1]
            @ normalized_normals[index + 1, 0], -1.0, 1.0
        )))))
        path_turns.append(math.degrees(math.acos(float(np.clip(
            path_tangents[index] @ path_tangents[index + 1], -1.0, 1.0
        )))))
    extent = max(1.0, float(np.max(np.abs(perimeter))))
    closed = (
        len(perimeter) > 1
        and float(np.linalg.norm(perimeter[-1, 1] - perimeter[0, 0]))
        <= 1.0e-9 * extent
    )
    if closed:
        vertex_normal_jumps.append(math.degrees(math.acos(float(np.clip(
            normalized_normals[-1, 1] @ normalized_normals[0, 0],
            -1.0, 1.0,
        )))))
        path_turns.append(math.degrees(math.acos(float(np.clip(
            path_tangents[-1] @ path_tangents[0], -1.0, 1.0
        )))))
    maximum_normal_jump = max(vertex_normal_jumps, default=0.0)
    minimum_radius = float(np.min(radii))
    if maximum_normal_jump > 1.0e-9:
        # A normal discontinuity at a zero-length shared vertex has zero
        # normal-turn radius.  A library may accept that only by explicitly
        # declaring a zero minimum radius from matching evidence.
        minimum_radius = 0.0
    return {
        "requested_look_count": int(len(radar_directions)),
        # Piece density can vary with frequency.  This is the largest number
        # of distinct requested directions that illuminate at least one piece
        # on any one solve frequency; a zero is unambiguous at every frequency.
        "illuminated_requested_look_count": int(max(
            (measurement[3] for measurement in measurements),
            default=0,
        )),
        "maximum_requested_conical_incidence_deg": maximum_conical,
        "estimated_min_along_line_normal_turn_radius_m": minimum_radius,
        "along_line_normal_turn_detected": bool(math.isfinite(minimum_radius)),
        "maximum_shared_vertex_normal_jump_deg": float(maximum_normal_jump),
        "maximum_path_vertex_turn_deg": float(max(path_turns, default=0.0)),
        "required_cut_angle_ranges_deg": cut_ranges,
    }


def _component_signature(
    feature_kind: str, dataset_sha256: str, *arrays: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(b"ghost-feature-component-v1\0")
    digest.update(feature_kind.encode("ascii") + b"\0")
    digest.update(str(dataset_sha256).lower().encode("ascii") + b"\0")
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_point_roll(
    aperture_normal: np.ndarray, roll_reference: np.ndarray
) -> np.ndarray:
    """Return the solver-effective local point-pattern ``+x`` direction."""

    normal = unit_vector(aperture_normal, "point aperture normal")
    roll = np.asarray(roll_reference, dtype=float)
    projected = roll - float(roll @ normal) * normal
    return unit_vector(projected, "point roll reference projected onto skin")


def _update_physics_digest_array(
    digest: Any, label: str, value: Any
) -> None:
    raw = np.asarray(value)
    array = np.ascontiguousarray(
        raw,
        dtype="<c16" if np.iscomplexobj(raw) else "<f8",
    )
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _prepared_point_response_physics_sha256(pattern: Any) -> str:
    """Hash the canonical point Jones pattern actually consumed by physics."""

    digest = hashlib.sha256()
    digest.update(b"ghost-prepared-point-response-v1\0")
    for key in ("azimuths", "elevations", "frequencies"):
        _update_physics_digest_array(digest, key, getattr(pattern, key))
    indices = dict(pattern.channel_indices)
    order = [indices[key] for key in ("VV", "HH", "VH")]
    _update_physics_digest_array(
        digest,
        "canonical_vv_hh_vh_amplitude",
        np.asarray(pattern.amplitude, dtype=np.complex128)[..., order],
    )
    return digest.hexdigest()


def _prepared_line_response_physics_sha256(coefficients: Sequence[Any]) -> str:
    """Hash canonical TM/TE seam samples at every requested frequency."""

    digest = hashlib.sha256()
    digest.update(b"ghost-prepared-line-response-v1\0")
    for index, coefficient in enumerate(coefficients):
        _update_physics_digest_array(
            digest, f"{index}:frequency_ghz", [coefficient.frequency_ghz]
        )
        _update_physics_digest_array(
            digest, f"{index}:cut_angle_deg", coefficient.phi_deg
        )
        _update_physics_digest_array(
            digest, f"{index}:tm", coefficient.dA_tm
        )
        _update_physics_digest_array(
            digest, f"{index}:te", coefficient.dA_te
        )
    return digest.hexdigest()


def _validate_point_requested_support(
    placement: Mapping[str, Any],
    radar_directions: np.ndarray,
    requested_frequencies: np.ndarray,
    *,
    dataset_id: str,
) -> dict[str, int]:
    """Preflight exact point frequency and lit-elevation support."""

    pattern = placement["pattern"]
    try:
        frequencies = np.asarray(pattern.frequencies, dtype=float)
        elevations = np.asarray(pattern.elevations, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"point dataset {dataset_id!r} is not a prepared point pattern."
        ) from exc
    for requested in requested_frequencies:
        if not np.any(np.isclose(
            frequencies, float(requested), rtol=0.0, atol=1.0e-6
        )):
            raise ValueError(
                f"point dataset {dataset_id!r} has no exact {requested:g} GHz "
                f"response (available {frequencies.tolist()})."
            )

    normal = unit_vector(
        np.asarray(placement["aperture_normal"], dtype=float),
        "point aperture normal",
    )
    local_x = _canonical_point_roll(
        normal, np.asarray(placement["roll_ref"], dtype=float)
    )
    local_y = np.cross(normal, local_x)
    rotation = np.column_stack((local_x, local_y, normal))
    local_directions = radar_directions @ rotation
    queried_elevation = np.degrees(np.arcsin(np.clip(
        local_directions[:, 2], -1.0, 1.0
    )))
    lit = radar_directions @ normal > 0.0
    outside = lit & (
        (queried_elevation < elevations[0] - 1.0e-9)
        | (queried_elevation > elevations[-1] + 1.0e-9)
    )
    if np.any(outside):
        queried = queried_elevation[outside]
        raise ValueError(
            f"point dataset {dataset_id!r} elevation support is "
            f"[{elevations[0]:g}, {elevations[-1]:g}] deg, but this placement "
            f"needs {float(np.min(queried)):g}..{float(np.max(queried)):g} deg "
            "over lit requested looks."
        )
    return {
        "requested_look_count": int(len(radar_directions)),
        "illuminated_requested_look_count": int(np.count_nonzero(lit)),
    }


def _point_segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    start, end = np.asarray(segment, dtype=float)
    chord = end - start
    fraction = float(np.dot(point - start, chord) / np.dot(chord, chord))
    nearest = start + float(np.clip(fraction, 0.0, 1.0)) * chord
    return float(np.linalg.norm(point - nearest))


def _segment_segment_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Exact closest distance between two finite non-degenerate 3-D segments."""

    p1, q1 = np.asarray(left, dtype=float)
    p2, q2 = np.asarray(right, dtype=float)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    b = float(d1 @ d2)
    c = float(d1 @ r)
    f = float(d2 @ r)
    denominator = a * e - b * b
    if denominator > 1.0e-30 * a * e:
        s = float(np.clip((b * f - c * e) / denominator, 0.0, 1.0))
    else:
        s = 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = float(np.clip(-c / a, 0.0, 1.0))
    elif t > 1.0:
        t = 1.0
        s = float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _line_self_footprint_overlap(
    segments: np.ndarray,
    footprint_radius_m: float,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[tuple[int, int, float]]:
    """Return the first nonlocal within-line footprint overlap, if any.

    Adjacent segments necessarily meet at one endpoint, so their footprint
    tubes overlap locally.  They are rejected only when a return segment stays
    inside the prior tube beyond two footprint diameters from that joint.  This
    catches near-retraces without rejecting an ordinary 90-degree corner.
    """

    values = np.asarray(segments, dtype=float)
    if len(values) < 2:
        return None
    radius = float(footprint_radius_m)
    lower = values.min(axis=1) - radius
    upper = values.max(axis=1) + radius
    order = sorted(range(len(values)), key=lambda index: (lower[index, 0], index))
    extent = max(1.0, float(np.max(np.abs(values))))
    closed = bool(
        np.linalg.norm(values[-1, 1] - values[0, 0]) <= 1.0e-9 * extent
    )
    active: list[int] = []

    def adjacent_retrace_clearance(
        left_index: int, right_index: int
    ) -> Optional[float]:
        wrap = bool(
            closed
            and {left_index, right_index} == {0, len(values) - 1}
        )
        if wrap:
            before = len(values) - 1
            after = 0
        else:
            before = min(left_index, right_index)
            after = max(left_index, right_index)
            if after != before + 1:
                return None
        joint = values[before, 1]
        if np.linalg.norm(joint - values[after, 0]) > 1.0e-8 * extent:
            return None
        away_before = values[before, 0] - joint
        away_after = values[after, 1] - joint
        before_length = float(np.linalg.norm(away_before))
        after_length = float(np.linalg.norm(away_after))
        common_length = min(before_length, after_length)
        # Inside two radii the overlap is the unavoidable endpoint cap. Probe
        # up to two diameters away; overlap that persists there is nonlocal.
        if common_length <= 2.0 * radius + 1.0e-12:
            return None
        probe = min(common_length, 4.0 * radius)
        before_point = joint + probe * away_before / before_length
        after_point = joint + probe * away_after / after_length
        clearance = float(np.linalg.norm(before_point - after_point))
        if clearance + 1.0e-12 < 2.0 * radius:
            return clearance
        return None

    comparisons = 0
    for index in order:
        active = [
            prior for prior in active
            if upper[prior, 0] + 1.0e-12 >= lower[index, 0]
        ]
        for prior in active:
            comparisons += 1
            if comparisons % 256 == 0 and (
                cancel_check is not None and cancel_check()
            ):
                raise InterruptedError("Feature placement validation cancelled.")
            adjacent = abs(index - prior) <= 1 or (
                closed and {index, prior} == {0, len(values) - 1}
            )
            if adjacent:
                clearance = adjacent_retrace_clearance(prior, index)
                if clearance is not None:
                    return min(prior, index), max(prior, index), clearance
                continue
            if np.any(upper[prior, 1:] + 1.0e-12 < lower[index, 1:]) or np.any(
                upper[index, 1:] + 1.0e-12 < lower[prior, 1:]
            ):
                continue
            clearance = _segment_segment_distance(values[prior], values[index])
            if clearance + 1.0e-12 < 2.0 * radius:
                return min(prior, index), max(prior, index), clearance
        active.append(index)
    return None


def _component_clearance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if left["kind"] == "point" and right["kind"] == "point":
        return float(np.linalg.norm(left["location"] - right["location"]))
    if left["kind"] == "point":
        return min(
            _point_segment_distance(left["location"], segment)
            for segment in right["segments"]
        )
    if right["kind"] == "point":
        return _component_clearance(right, left)
    return min(
        _segment_segment_distance(left_segment, right_segment)
        for left_segment in left["segments"]
        for right_segment in right["segments"]
    )


def _footprint_candidate_pairs(
    components: Sequence[Mapping[str, Any]],
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """Yield broad-phase-overlapping footprint pairs with an x-axis sweep.

    The previous all-pairs loop made 10,000 vehicle fasteners perform roughly
    50 million Python comparisons before a build. Expanded AABBs reject remote
    components without changing the exact point/segment clearance decision for
    candidates that may overlap.
    """

    bounded = []
    for index, component in enumerate(components):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Feature placement validation cancelled.")
        if component["kind"] == "point":
            vertices = np.asarray(component["location"], dtype=float).reshape(1, 3)
        else:
            vertices = np.asarray(component["segments"], dtype=float).reshape(-1, 3)
        radius = float(component["radius_m"])
        bounded.append((
            index,
            vertices.min(axis=0) - radius,
            vertices.max(axis=0) + radius,
        ))
    bounded.sort(key=lambda value: (float(value[1][0]), value[0]))
    active: list[tuple[int, np.ndarray, np.ndarray]] = []
    comparisons = 0
    for index, lower, upper in bounded:
        active = [
            entry for entry in active
            if float(entry[2][0]) + 1.0e-12 >= float(lower[0])
        ]
        for prior_index, prior_lower, prior_upper in active:
            comparisons += 1
            if comparisons % 256 == 0 and (
                cancel_check is not None and cancel_check()
            ):
                raise InterruptedError("Feature placement validation cancelled.")
            if np.any(prior_upper[1:] + 1.0e-12 < lower[1:]) or np.any(
                upper[1:] + 1.0e-12 < prior_lower[1:]
            ):
                continue
            yield components[prior_index], components[index]
        active.append((index, lower, upper))


def validate_installed_host(manifest, *, material, stack_id="", minimum_radius_m=None, required=True, label="feature"):
    """Match installation declarations to library evidence without guessing.

    A triangle's zero local curvature is not evidence that its parent surface
    is flat. Principal-radius bounds must describe the whole feature footprint.
    """
    def canonical(value):
        return " ".join(str(value).split()).casefold()
    host = manifest.get("host", {})
    warnings = []
    for key, installed in (("material", material), ("stack_id", stack_id)):
        expected = host.get(key, "")
        if expected and installed and canonical(expected) != canonical(installed):
            raise ValueError(f"{label}: installed host {key}={installed!r} differs from characterized {expected!r}.")
        if expected and not installed:
            message = f"{label}: declare installed host {key} to match characterized {expected!r}."
            if required:
                raise ValueError(message)
            warnings.append(message)
    bound = manifest.get("applicability", {}).get("minimum_principal_radius_m")
    if minimum_radius_m is not None:
        minimum_radius_m = float(minimum_radius_m)
        if not math.isfinite(minimum_radius_m) or minimum_radius_m < 0:
            raise ValueError("Installed host minimum radius must be finite and nonnegative.")
    if bound is None:
        warnings.append(f"{label}: library has no principal-curvature envelope over the feature footprint. Along-line normal checks do not certify transverse or point-footprint curvature.")
    elif minimum_radius_m is None:
        message = f"{label}: declare the minimum principal radius over every installed feature footprint; library requires at least {float(bound):g} m."
        if required:
            raise ValueError(message)
        warnings.append(message)
    elif minimum_radius_m < float(bound):
        raise ValueError(f"{label}: installed minimum principal radius {minimum_radius_m:g} m is below library limit {float(bound):g} m.")
    return {"material": material, "stack_id": stack_id,
            "minimum_principal_radius_m": minimum_radius_m,
            "principal_curvature_checked": bound is not None and minimum_radius_m is not None,
            "declaration_source": "user; match to library evidence", "warnings": warnings}


def _apply_feature_library_contracts(
    *,
    line_placements: Sequence[Mapping[str, Any]],
    line_records: Sequence[dict[str, Any]],
    point_placements: Sequence[Mapping[str, Any]],
    point_records: Sequence[dict[str, Any]],
    radar_grid: Mapping[str, Any],
    require_manifests: bool,
    cancel_check: Optional[Callable[[], bool]] = None,
    host_material: str = "",
    host_stack_id: str = "",
    host_minimum_radius_m: Optional[float] = None,
) -> tuple[dict[str, Any], list[str], dict[str, str], set[str]]:
    """Bind manifests, applicability gates, and component identities to a plan."""

    frequencies = np.asarray(radar_grid["frequencies_ghz"], dtype=float)
    directions = _vehicle_radar_directions(radar_grid)
    contracts: dict[str, Any] = {}
    warnings: list[str] = []
    source_hashes: dict[str, str] = {}
    absent_source_paths: set[str] = set()
    seen_components: dict[str, tuple[str, str]] = {}
    footprint_components: list[dict[str, Any]] = []
    groups = (
        ("line", line_placements, line_records),
        ("point", point_placements, point_records),
    )
    for feature_kind, placements, records in groups:
        if len(placements) != len(records):
            raise ValueError(
                f"Prepared {feature_kind} placement/record counts disagree."
            )
        manifests: dict[str, Optional[dict[str, Any]]] = {}
        response_identities: dict[str, str] = {}
        response_content_identities: dict[str, str] = {}
        response_frequency_bounds: dict[str, tuple[float, float]] = {}
        line_coefficients: dict[str, tuple[Any, ...]] = {}
        for placement, record in zip(placements, records):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Feature placement validation cancelled.")
            dataset_id = str(record["dataset_id"])
            dataset_value = record.get("dataset")
            dataset_digest = record.get("dataset_sha256")
            required_geometry = (
                {"perimeter", "segment_normals"}
                if feature_kind == "line"
                else {"location", "aperture_normal", "roll_ref"}
            )
            if (
                dataset_value is None
                or dataset_digest is None
                or not required_geometry.issubset(placement)
            ):
                message = (
                    f"Injected/custom {feature_kind} dataset {dataset_id!r} "
                    "predates the production manifest/component-identity "
                    "contract."
                )
                if require_manifests:
                    raise ValueError(message)
                warnings.append(message)
                contracts[f"{feature_kind}:{dataset_id}"] = {
                    "status": "legacy_injected_placement",
                }
                continue
            dataset = str(dataset_value)
            contract_key = f"{feature_kind}:{dataset_id}"
            if dataset_id not in manifests:
                try:
                    manifest, sources = load_feature_library_manifest(
                        dataset, dataset_id=dataset_id, feature_kind=feature_kind
                    )
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    if require_manifests:
                        raise
                    manifest, sources = None, []
                    warnings.append(f"Metadata advisory for {feature_kind} {dataset_id!r}: {exc}; response samples remain usable.")
                advisory_manifest = manifest if not require_manifests else None
                if not require_manifests:
                    manifest = None
                for source in sources:
                    if source.get("absent") == "true" and "path" in source:
                        absent_source_paths.add(str(source["path"]))
                    elif "path" in source and "sha256" in source:
                        source_hashes[str(source["path"])] = str(
                            source["sha256"]
                        )
                manifests[dataset_id] = manifest
                if manifest is not None:
                    response_content_identities[dataset_id] = str(
                        manifest["response_content_sha256"]
                    )
                else:
                    try:
                        response_content_identities[dataset_id] = (
                            feature_response_content_sha256(dataset)
                        )
                    except ValueError:
                        # Injected compatibility tests/services may not provide
                        # an actual NPZ. Production response loaders reject such
                        # a file later; retain the prepared raw identity here.
                        response_content_identities[dataset_id] = str(
                            dataset_digest
                        )
                if manifest is None:
                    description = "has an advisory" if advisory_manifest is not None else "has no"
                    message = (
                        f"{feature_kind} dataset {dataset_id!r} {description} "
                        "feature-library manifest; phase/frame are accepted "
                        "through the selected dataset role; host/curvature/"
                        "footprint annotations are advisory."
                    )
                    if require_manifests:
                        raise ValueError(message)
                    warnings.append(message)
                    contracts[contract_key] = {
                        "status": "metadata_advisory",
                        "dataset": dataset,
                        "source_manifest": advisory_manifest,
                    }
                else:
                    if manifest["schema"] != FEATURE_LIBRARY_MANIFEST_SCHEMA:
                        message = (
                            f"{feature_kind} dataset {dataset_id!r} uses "
                            f"Legacy manifest schema {manifest['schema']!r}; "
                            f"Production requires {FEATURE_LIBRARY_MANIFEST_SCHEMA!r} "
                            "so passing full-wave cases, all four artifacts, "
                            "and the exact exercised response are bound. "
                            "Migrate it with the supported manifest tool."
                        )
                        if require_manifests:
                            raise ValueError(message)
                        warnings.append(message)
                    contracts[contract_key] = {
                        "status": manifest["validation"]["status"],
                        "dataset": dataset,
                        "manifest": manifest,
                        "sources": sources,
                    }
                    if manifest["validation"]["status"] != "validated":
                        message = (
                            f"{feature_kind} dataset {dataset_id!r} manifest "
                            f"is {manifest['validation']['status']}, not validated."
                        )
                        if require_manifests:
                            raise ValueError(message)
                        warnings.append(message)

                if feature_kind == "line":
                    response_payload = _load_grim(dataset)
                    response_frequencies = np.asarray(
                        response_payload["frequencies"], dtype=float
                    )
                    prepared_coefficients = []
                    for requested_frequency in frequencies:
                        prepared_coefficients.append(load_seam_from_grim(
                            dataset,
                            float(requested_frequency),
                            declared_coherent_delta=True,
                            delta_sign=1.0,
                            _grim_payload=response_payload,
                        ))
                    response_identities[dataset_id] = (
                        _prepared_line_response_physics_sha256(
                            prepared_coefficients
                        )
                    )
                    line_coefficients[dataset_id] = tuple(prepared_coefficients)
                    response_frequency_bounds[dataset_id] = (
                        float(np.min(response_frequencies)),
                        float(np.max(response_frequencies)),
                    )
                else:
                    response_identities[dataset_id] = (
                        _prepared_point_response_physics_sha256(
                            placement["pattern"]
                        )
                    )
                    point_frequencies = np.asarray(
                        placement["pattern"].frequencies, dtype=float
                    )
                    response_frequency_bounds[dataset_id] = (
                        float(np.min(point_frequencies)),
                        float(np.max(point_frequencies)),
                    )
            manifest = manifests[dataset_id]
            if feature_kind == "point":
                point_support = _validate_point_requested_support(
                    placement, directions, frequencies, dataset_id=dataset_id
                )
                record.update(point_support)
                if point_support["illuminated_requested_look_count"] == 0:
                    warnings.append(
                        f"Point {str(record['placement_id'])!r} has zero "
                        "illuminated requested looks, so its enabled response "
                        "contributes zero on this radar grid. Review its "
                        "outward normal/orientation and requested aperture. If "
                        "intentional, review and accept the existing one-time "
                        "release-warning waiver before Build."
                    )
            if manifest is not None:
                host_result = validate_installed_host(
                    manifest, material=host_material, stack_id=host_stack_id,
                    minimum_radius_m=host_minimum_radius_m,
                    required=require_manifests, label=f"{feature_kind} dataset {dataset_id!r}",
                )
                record["host_applicability"] = host_result
                for message in host_result["warnings"]:
                    if message not in warnings:
                        warnings.append(message)
                applicability = manifest["applicability"]
                frequency_range = applicability["frequency_ghz"]
                if (
                    float(np.min(frequencies)) < frequency_range["min"] - 1e-12
                    or float(np.max(frequencies)) > frequency_range["max"] + 1e-12
                ):
                    raise ValueError(
                        f"{feature_kind} dataset {dataset_id!r} is certified "
                        f"only for {frequency_range['min']:g}-"
                        f"{frequency_range['max']:g} GHz; the Assembly grid is "
                        f"{float(np.min(frequencies)):g}-"
                        f"{float(np.max(frequencies)):g} GHz."
                    )
                response_min, response_max = response_frequency_bounds[dataset_id]
                if (
                    frequency_range["min"] < response_min - 1.0e-12
                    or frequency_range["max"] > response_max + 1.0e-12
                ):
                    raise ValueError(
                        f"{feature_kind} dataset {dataset_id!r} manifest declares "
                        f"{frequency_range['min']:g}-{frequency_range['max']:g} "
                        "GHz applicability, outside the bound response data "
                        f"range {response_min:g}-{response_max:g} GHz."
                    )

            if feature_kind == "line":
                metrics = _line_applicability_metrics(
                    placement,
                    directions,
                    requested_frequencies_ghz=frequencies,
                    cancel_check=cancel_check,
                )
                installed_radius = float(metrics[
                    "estimated_min_along_line_normal_turn_radius_m"
                ])
                public_metrics = dict(metrics)
                if not math.isfinite(installed_radius):
                    # Strict JSON has no Infinity token.  ``null`` plus the
                    # explicit boolean is an unambiguous straight/constant-
                    # normal result for external provenance consumers.
                    public_metrics[
                        "estimated_min_along_line_normal_turn_radius_m"
                    ] = None
                record.update(public_metrics)
                if metrics["illuminated_requested_look_count"] == 0:
                    warnings.append(
                        f"Line {str(record['line_id'])!r} has zero illuminated "
                        "requested looks, so its enabled response contributes "
                        "zero on this radar grid. Review its endpoint normals "
                        "and requested aperture. If intentional, review and "
                        "accept the existing one-time release-warning waiver "
                        "before Build."
                    )
                for coefficient, cut_range in zip(
                    line_coefficients[dataset_id],
                    metrics["required_cut_angle_ranges_deg"],
                ):
                    cut_min = cut_range["minimum_deg"]
                    cut_max = cut_range["maximum_deg"]
                    if cut_min is None:
                        continue
                    support_min = float(coefficient.phi_deg[0])
                    support_max = float(coefficient.phi_deg[-1])
                    if (
                        float(cut_min) < support_min - 1.0e-9
                        or float(cut_max) > support_max + 1.0e-9
                    ):
                        raise ValueError(
                            f"line dataset {dataset_id!r} at "
                            f"{float(coefficient.frequency_ghz):g} GHz covers "
                            f"cut angles [{support_min:g}, {support_max:g}] deg, "
                            f"but installed line {record['line_id']!r} needs "
                            f"[{float(cut_min):.6g}, {float(cut_max):.6g}] deg "
                            "over lit requested looks. Extend the coupon sweep "
                            "or change the requested/installed envelope."
                        )
                if manifest is not None:
                    applicability = manifest["applicability"]
                    conical_limit = applicability[
                        "maximum_conical_incidence_deg"
                    ]
                    if (
                        metrics["maximum_requested_conical_incidence_deg"]
                        > conical_limit + 1.0e-9
                    ):
                        raise ValueError(
                            f"line dataset {dataset_id!r} is certified through "
                            f"{conical_limit:g} deg conical incidence, but line "
                            f"{record['line_id']!r} reaches "
                            f"{metrics['maximum_requested_conical_incidence_deg']:.3g} "
                            "deg over illuminated requested looks. The 2-D line "
                            "coefficient lookup does not model arbitrary d.t."
                        )
                    curvature_limit = applicability[
                        "minimum_along_line_normal_turn_radius_m"
                    ]
                    if (
                        installed_radius + 1.0e-12 < curvature_limit
                    ):
                        raise ValueError(
                            f"line dataset {dataset_id!r} requires an along-line "
                            f"normal-turn radius >= {curvature_limit:g} m, but line "
                            f"{record['line_id']!r} is approximately "
                            f"{installed_radius:.3g} "
                            "m. This does not certify transverse/principal host "
                            "curvature."
                        )
                    path_turn_limit = applicability[
                        "maximum_path_vertex_turn_deg"
                    ]
                    installed_path_turn = float(
                        metrics["maximum_path_vertex_turn_deg"]
                    )
                    if installed_path_turn > path_turn_limit + 1.0e-9:
                        raise ValueError(
                            f"line dataset {dataset_id!r} permits at most "
                            f"{path_turn_limit:g} deg path turn at a shared "
                            f"vertex, but line {record['line_id']!r} reaches "
                            f"{installed_path_turn:.6g} deg. Split/validate the "
                            "corner as its own interaction or use matching "
                            "corner evidence."
                        )
                    footprint_radius = float(
                        applicability["footprint_radius_m"]
                    )
                    self_overlap = _line_self_footprint_overlap(
                        np.asarray(placement["perimeter"], dtype=float),
                        footprint_radius,
                        cancel_check=cancel_check,
                    )
                    if self_overlap is not None:
                        left_index, right_index, clearance = self_overlap
                        message = (
                            f"Line {record['line_id']!r} folds back within its "
                            "own applicability footprint: segments "
                            f"{left_index + 1} and {right_index + 1} are "
                            f"nonlocally {clearance:.6g} m apart, below "
                            f"{2.0 * footprint_radius:.6g} m. Independent "
                            "straight-seam superposition omits this self/corner "
                            "coupling."
                        )
                        if require_manifests:
                            raise ValueError(message)
                        warnings.append(message)
                signature = _component_signature(
                    feature_kind,
                    response_identities[dataset_id],
                    placement["perimeter"],
                    placement["segment_normals"],
                )
                instance_id = str(record["line_id"])
            else:
                effective_roll = _canonical_point_roll(
                    np.asarray(placement["aperture_normal"], dtype=float),
                    np.asarray(placement["roll_ref"], dtype=float),
                )
                signature = _component_signature(
                    feature_kind,
                    response_identities[dataset_id],
                    placement["location"],
                    placement["aperture_normal"],
                    effective_roll,
                )
                instance_id = str(record["placement_id"])
            prior = seen_components.get(signature)
            if prior is not None:
                raise ValueError(
                    f"Duplicate physical feature component: {feature_kind} "
                    f"{instance_id!r} repeats {prior[0]} {prior[1]!r} with the "
                    "same response, location/path, and orientation."
                )
            seen_components[signature] = (feature_kind, instance_id)
            record["component_signature"] = signature
            record["dataset_content_sha256"] = response_content_identities[
                dataset_id
            ]
            record["dataset_physics_sha256"] = response_identities[dataset_id]
            if manifest is not None:
                record["feature_library_manifest_schema"] = manifest["schema"]
                record["feature_library_validation_status"] = manifest[
                    "validation"
                ]["status"]
                record["feature_library_footprint_radius_m"] = manifest[
                    "applicability"
                ]["footprint_radius_m"]
                footprint = {
                    "kind": feature_kind,
                    "instance_id": instance_id,
                    "radius_m": float(manifest["applicability"][
                        "footprint_radius_m"
                    ]),
                }
                if feature_kind == "point":
                    footprint["location"] = np.asarray(
                        placement["location"], dtype=float
                    )
                else:
                    footprint["segments"] = np.asarray(
                        placement["perimeter"], dtype=float
                    )
                footprint_components.append(footprint)

    point_ids = {str(record["placement_id"]) for record in point_records}
    line_ids = {str(record["line_id"]) for record in line_records}
    collisions = sorted(point_ids & line_ids)
    if collisions:
        raise ValueError(
            "Point placement_id and line_id values share one Assembly identity "
            f"namespace; rename duplicate ID(s) {collisions}."
        )
    overlap_warning_limit = 100
    overlap_count = 0
    for left, right in _footprint_candidate_pairs(
        footprint_components, cancel_check=cancel_check
    ):
        clearance = _component_clearance(left, right)
        required_clearance = left["radius_m"] + right["radius_m"]
        if clearance + 1.0e-12 >= required_clearance:
            continue
        overlap_count += 1
        message = (
            f"Feature applicability footprints overlap: {left['kind']} "
            f"{left['instance_id']!r} and {right['kind']} "
            f"{right['instance_id']!r} are {clearance:.6g} m apart, below "
            f"their combined {required_clearance:.6g} m footprint. "
            "Independent superposition omits cluster coupling."
        )
        if require_manifests:
            raise ValueError(message)
        if overlap_count <= overlap_warning_limit:
            warnings.append(message)
        elif overlap_count == overlap_warning_limit + 1:
            warnings.append(
                "More than 100 feature applicability-footprint overlaps were "
                "found; additional pairs are not rendered. Use Production "
                "validation and resolve clustered-feature coupling before release."
            )
            break
    return contracts, warnings, source_hashes, absent_source_paths


def _prepared_assembly_workload(
    radar_grid: Mapping[str, Any],
    lines: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    *,
    triangle_count: int,
    shadow_enabled: bool,
):
    """Count the exact field grid and a conservative shadow-ray upper bound."""

    look_count = (
        len(radar_grid.get("azimuths_deg", ()))
        * len(radar_grid.get("elevations_deg", ()))
    )
    frequency_count = len(radar_grid.get("frequencies_ghz", ()))
    line_piece_count = 0
    line_segment_count = 0
    pieces_exact = True
    for placement in lines:
        segment_count = 1
        segments_counted = False
        try:
            perimeter = np.asarray(placement["perimeter"], dtype=float)
            if perimeter.ndim == 2 and perimeter.shape == (2, 3):
                perimeter = perimeter.reshape(1, 2, 3)
            if perimeter.ndim != 3 or perimeter.shape[1:] != (2, 3):
                raise ValueError
            segment_count = len(perimeter)
            line_segment_count += segment_count
            segments_counted = True
            shadow_points = placement.get("shadow_points")
            if shadow_points is not None:
                line_piece_count += len(shadow_points)
                continue
            maximum_piece_length = float(placement["max_piece_length_m"])
            if not math.isfinite(maximum_piece_length) or maximum_piece_length <= 0.0:
                raise ValueError
            lengths = np.linalg.norm(perimeter[:, 1] - perimeter[:, 0], axis=1)
            if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
                raise ValueError
            line_piece_count += sum(
                max(1, int(math.ceil(float(length) / maximum_piece_length)))
                for length in lengths
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            # This path is for old injected/test placements. Authoritative
            # prepared placements always carry the fixed subdivision length.
            pieces_exact = False
            if not segments_counted:
                line_segment_count += segment_count
            line_piece_count += max(1, segment_count)

    return estimate_assembly_workload(
        look_count=look_count,
        frequency_count=frequency_count,
        point_count=len(points),
        line_path_count=len(lines),
        line_segment_count=line_segment_count,
        line_piece_count=line_piece_count,
        mesh_triangle_count=triangle_count,
        shadow_enabled=shadow_enabled,
        quantities_validated=True,
        line_piece_count_exact=pieces_exact,
        mesh_triangle_count_exact=True,
    )


def bor_shadow_triangles(profile, *, max_sag_m, normal_tolerance_deg):
    """Revolve the authoritative profile with an explicit radial sag bound."""
    profile = np.asarray(profile, float)
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2 or not np.all(np.isfinite(profile)) or np.any(profile[:, 0] < 0):
        raise ValueError("BoR shadow profile must contain finite nonnegative rho,z vertices.")
    radius = float(np.max(profile[:, 0]))
    if radius <= 0 or max_sag_m <= 0:
        raise ValueError("BoR shadow tessellation needs a positive radius and sag tolerance.")
    step = min(2*math.acos(max(-1., 1.-min(float(max_sag_m)/radius, 1.))), math.radians(float(normal_tolerance_deg)))
    count = max(64, int(math.ceil(2*math.pi/max(step, 1e-12))))
    if 2*(len(profile)-1)*count > 1_000_000:
        raise MemoryError("Auto BoR shadow surface exceeds one million facets at the requested tolerance. Supply a reviewed surface mesh or use a smaller study with justified geometry tolerances.")
    phi = np.arange(count)*(2*math.pi/count)
    ring = np.stack((profile[:, 0, None]*np.cos(phi), profile[:, 0, None]*np.sin(phi), np.broadcast_to(profile[:, 1, None], (len(profile), count))), axis=-1)
    a, b = ring[:-1], ring[1:]
    an, bn = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
    triangles = np.stack((np.stack((a, b, bn), axis=-2), np.stack((a, bn, an), axis=-2)), axis=2).reshape(-1, 3, 3)
    valid = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) > 0
    return triangles[valid], {"azimuth_sector_count": count, "maximum_radial_sag_m": radius*(1-math.cos(math.pi/count)), "source": "authoritative embedded BoR profile"}


def prepare_feature_assembly(
    request: FeatureAssemblyRequest,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> FeatureAssemblyPlan:
    """Resolve, validate, and prepare one feature-assembly request."""

    if not isinstance(request, FeatureAssemblyRequest):
        raise TypeError("request must be a FeatureAssemblyRequest.")
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")
    if progress_callback is not None:
        progress_callback(0, 100, "Checking Assembly inputs")
    base = resolve_path(request.base_grim, base_dir=request.base_dir)
    output = _canonical_grim_output_path(
        request.output_grim, base_dir=request.base_dir
    )
    features_only_output = Path(feature_only_output_path(str(output)))
    _reject_output_aliases(request, base=base, output=output)
    active_features = bool(
        (request.point_locations_csv is not None and request.enabled_point_placement_ids != ())
        or (request.line_locations_csv is not None and request.enabled_line_ids != ())
    )
    coordinate_scale = _required_unit_scale(
        request.coordinate_units,
        label="coordinate_units",
        used_for="a point or line placement CSV",
    ) if request.point_locations_csv is not None or request.line_locations_csv is not None else 1.0
    surface_scale: Optional[float] = None
    if request.surface_mesh is not None:
        surface_scale = _required_unit_scale(
            request.surface_units,
            label="surface_units",
            used_for="surface_mesh",
        )
    if not base.is_file():
        raise FileNotFoundError(f"Base monostatic GRIM not found: {base}")
    if output.exists() and not output.is_file():
        raise ValueError(f"Assembly output exists but is not a file: {output}")
    if features_only_output.exists() and not features_only_output.is_file():
        raise ValueError(
            "Feature-only Assembly output exists but is not a file: "
            f"{features_only_output}"
        )
    prepared_output_absent = not output.is_file()
    prepared_output_sha256 = (
        None if prepared_output_absent else sha256_file(str(output))
    )
    prepared_features_only_output_absent = not features_only_output.is_file()
    prepared_features_only_output_sha256 = (
        None
        if prepared_features_only_output_absent
        else sha256_file(str(features_only_output))
    )
    base_sha256 = sha256_file(str(base))
    prepared_source_sha256 = {str(base): base_sha256}
    prepared_input_sources: dict[str, dict[str, str]] = {
        "base_grim": {"path": str(base), "sha256": base_sha256}
    }
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")
    if progress_callback is not None:
        progress_callback(12, 100, "Reading clean-body response")

    def snapshot_input_source(
        role: str,
        value: Optional[PathValue],
        *,
        label: str,
    ) -> Optional[Path]:
        if value is None:
            return None
        source = resolve_path(value, base_dir=request.base_dir)
        if not source.is_file():
            raise FileNotFoundError(f"{label} not found: {source}")
        digest = sha256_file(str(source))
        previous = prepared_source_sha256.get(str(source))
        if previous is not None and previous != digest:
            raise RuntimeError(
                f"Feature-assembly source changed while input files were "
                f"being snapshotted: {source}. Revalidate the assembly."
            )
        prepared_source_sha256[str(source)] = digest
        prepared_input_sources[role] = {
            "path": str(source),
            "sha256": digest,
        }
        return source

    # Snapshot every spatial-definition file before the first parser/mesh
    # reader sees it.  The end-of-prepare check below then proves that all
    # prepared geometry came from one immutable set of input bytes.
    surface_path = snapshot_input_source(
        "surface_mesh", request.surface_mesh, label="Surface mesh"
    )
    line_coordinates_path = snapshot_input_source(
        "line_locations_csv",
        request.line_locations_csv,
        label="Line-placement CSV",
    )
    point_coordinates_path = snapshot_input_source(
        "point_locations_csv",
        request.point_locations_csv,
        label="Point-placement CSV",
    )
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")

    # NPZ access is lazy: read only the small axes before admitting the full
    # coherent response and its metadata. The complete feature/mesh estimate
    # is repeated at execution after the prepared geometry is known.
    with np.load(str(base), allow_pickle=False) as archive:
        capacity_grid = {
            "frequencies_ghz": archive["frequencies"],
            "azimuths_deg": archive["azimuths"],
            "elevations_deg": archive["elevations"],
        }
    preflight_feature_assembly_capacity(str(base), str(output), radar_grid=capacity_grid)
    base_payload = _load_grim(str(base))
    body_mesh_certification = None
    if request.require_body_mesh_certification:
        if progress_callback is not None:
            progress_callback(10, 100, "Checking body mesh certification")
        body_mesh_certification = audit_body_mesh_certification(
            str(base), loaded_grim=base_payload
        )
    existing_feature_records = _decoded_feature_provenance(
        base_payload, base.name
    )
    if request.require_feature_manifests and existing_feature_records:
        raise ValueError(
            f"{base.name}: Production Assembly requires one complete batch "
            "starting from the clean-body response. This base already carries "
            "feature provenance, so cross-build applicability/coupling cannot "
            "be re-evaluated safely. Select the clean body and enable every "
            "point/line feature in one Assembly plan."
        )
    validated_base_payload = _validate_declared_coherent_base(
        base_payload,
        str(base),
        allow_legacy_metadata=request.allow_legacy_base_metadata,
    )
    _canonical_3d_channel_indices(
        validated_base_payload["polarizations"],
        str(base),
        require_all=True,
    )
    coherent_base_missing_metadata = tuple(
        validated_base_payload.get(_LEGACY_BASE_ASSUMPTIONS_KEY, ())
    )

    embedded_grid = load_body_requested_radar_grid(str(base))
    pre_validation_warnings: list[str] = []
    pre_absent_paths: set[str] = set()
    surface_geometry_contract: dict[str, Any] = {
        "schema": "ghost.assembly-surface-geometry-binding.v1",
        "status": "not_applicable_embedded_bor",
    }
    profile: Optional[np.ndarray] = None
    if embedded_grid is not None:
        profile = load_body_profile_grim(str(base))
        grid = dict(embedded_grid)
        base_grid_contract = {
            "schema": "ghost.assembly-base-grid-contract.v1",
            "status": "embedded_body_model",
            "angular_contract": ASSEMBLY_RADAR_ANGULAR_CONTRACT,
            "legacy_missing_metadata": [],
            "legacy_compatibility_enabled": bool(
                request.allow_legacy_base_metadata
            ),
        }
    else:
        grid = {
            "frequencies_ghz": np.asarray(
                base_payload["frequencies"], dtype=float
            ),
            "azimuths_deg": np.asarray(base_payload["azimuths"], dtype=float),
            "elevations_deg": np.asarray(
                base_payload["elevations"], dtype=float
            ),
            "axis_az_deg": float(AXIS_AZ_DEG),
            "axis_el_deg": float(AXIS_EL_DEG),
            "roll_deg": float(ROLL_DEG),
        }
        base_grid_contract = validate_assembly_base_grid_metadata(
            base_payload,
            grid,
            str(base),
            allow_legacy_metadata=request.allow_legacy_base_metadata,
        )

    for request_key, grid_key in (("study_frequencies_ghz", "frequencies_ghz"),
                                  ("study_azimuths_deg", "azimuths_deg"),
                                  ("study_elevations_deg", "elevations_deg")):
        selected = getattr(request, request_key)
        if selected is not None:
            grid[grid_key] = selected
    # Validation uses the exact requested solve scope. The original clean-body
    # file/hash still anchors registration and source provenance.
    _subset_payload, grid = exact_assembly_subset({key: base_payload[key] for key in ("frequencies", "azimuths", "elevations")}, grid)
    del _subset_payload

    skin_limit, wavelength = compute_skin_limit(grid["frequencies_ghz"], skin_tol_m=request.skin_tol_m, skin_phase_tol_deg=request.skin_phase_tol_deg)
    normal_tolerance = validate_normal_tolerance(request.normal_tol_deg)
    auto_shadow_report = None
    surface: Optional[TriangleSurface] = None
    surface_triangles_cad_m: Optional[np.ndarray] = None
    mesh_topology_report = None
    if surface_path is not None:
        assert surface_scale is not None
        surface_triangles_cad_m = (
            np.asarray(read_surface_mesh(str(surface_path)), dtype=float)
            * surface_scale
        )
        surface_triangles_cad_m.setflags(write=False)
        triangles = to_axis_frame(surface_triangles_cad_m)
        surface = TriangleSurface(
            triangles,
            flip_normals=bool(request.flip_surface_normals),
        )
        # STL/facet files do not carry trustworthy topology. Reconstruct it
        # once so normal and shadow risks are visible in the same validation
        # report as the feature instances. This is diagnostic rather than an
        # automatic repair: an intentional open placement patch remains usable.
        mesh_topology_report = surface.topology_report
    elif embedded_grid is not None and active_features and request.shadow:
        triangles, auto_shadow_report = bor_shadow_triangles(profile, max_sag_m=skin_limit/4, normal_tolerance_deg=max(normal_tolerance/2, 1e-6))
        surface = TriangleSurface(triangles)
        surface_triangles_cad_m = triangles @ CAD2AXIS
        surface_triangles_cad_m.setflags(write=False)
        mesh_topology_report = surface.topology_report
    elif embedded_grid is None and active_features:
        raise ValueError(
            "A non-BoR base requires surface_mesh=.facet or .stl for skin "
            "validation and outward normals."
        )
    if request.shadow and active_features and surface is None:
        raise ValueError("shadow=True requires surface_mesh.")
    if surface is not None and active_features and not request.shadow:
        pre_validation_warnings.append(
            "Geometric body shadowing is OFF while a body mesh is selected. "
            "Hidden point and line features are not occlusion-tested and can "
            "contribute at full modeled amplitude whenever they are front-face "
            "illuminated. Enable body shadowing for vehicle placement work, or "
            "review and accept the existing one-time release-warning waiver if "
            "this no-shadow trade study is intentional."
        )
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")
    if progress_callback is not None:
        progress_callback(35, 100, "Checking body surface and topology")

    if embedded_grid is not None:
        if surface is None:
            surface_geometry_contract = {
                "schema": SURFACE_BINDING_SCHEMA,
                "status": "embedded_bor_profile_is_authoritative_surface",
            }
        else:
            surface_geometry_contract = _validate_bor_surface_agreement(
                profile,
                surface,
                skin_limit_m=skin_limit,
                shadow_requested=bool(request.shadow),
                cancel_check=cancel_check,
            )
            surface_geometry_contract["surface_mesh"] = str(surface_path)
            if auto_shadow_report is not None:
                surface_geometry_contract["surface_mesh"] = None
                surface_geometry_contract["generated_shadow_surface"] = auto_shadow_report
    point_preview_lists: dict[str, list[np.ndarray]] = {}
    point_preview_normals: dict[str, list[np.ndarray]] = {}
    point_preview_roll_references: dict[str, list[np.ndarray]] = {}
    line_preview_paths: dict[str, dict[str, np.ndarray]] = {}
    line_preview_endpoint_normals: dict[str, dict[str, np.ndarray]] = {}
    lines, line_records = prepare_line_placements(
        profile,
        surface,
        coordinate_scale=coordinate_scale,
        skin_limit_m=skin_limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=normal_tolerance,
        locations_csv=line_coordinates_path,
        datasets=request.line_datasets,
        enabled_line_ids=request.enabled_line_ids,
        base_dir=request.base_dir,
        preview_paths_cad_m=line_preview_paths,
        preview_endpoint_normals_cad=line_preview_endpoint_normals,
        prepare_shadow_origins=bool(request.shadow),
        cancel_check=cancel_check,
    )
    if progress_callback is not None:
        progress_callback(55, 100, "Checking line paths")
    points, point_records = prepare_point_placements(
        profile,
        surface,
        coordinate_scale=coordinate_scale,
        skin_limit_m=skin_limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=normal_tolerance,
        locations_csv=point_coordinates_path,
        datasets=request.point_datasets,
        enabled_point_placement_ids=request.enabled_point_placement_ids,
        base_dir=request.base_dir,
        preview_locations_cad_m=point_preview_lists,
        preview_normals_cad=point_preview_normals,
        preview_roll_references_cad=point_preview_roll_references,
        prepare_shadow_origins=bool(request.shadow),
        cancel_check=cancel_check,
    )
    if progress_callback is not None:
        progress_callback(72, 100, "Checking point placements")

    point_preview_ids: dict[str, list[str]] = {}
    for record in point_records:
        point_preview_ids.setdefault(str(record["dataset_id"]), []).append(
            str(record["placement_id"])
        )

    if embedded_grid is None and (lines or points):
        surface_digest = prepared_source_sha256[str(surface_path)]
        try:
            binding, binding_path, binding_digest = load_surface_binding(
                base,
                surface_path,
                base_grim_sha256=base_sha256,
                surface_sha256=surface_digest,
                surface_units=request.surface_units,
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            if request.require_feature_manifests:
                raise
            binding, binding_path, binding_digest = None, surface_binding_path(surface_path), None
            pre_validation_warnings.append(f"Surface metadata advisory: {exc}; registration is assumed from the selected body and mesh.")
        if binding is None:
            message = (
                f"{base.name}: external body fields do not embed geometry, so "
                f"Production requires the reviewed surface binding {binding_path}. "
                "Create it with the supported binding tool after confirming "
                "that this exact mesh, units, CAD frame, and origin correspond "
                "to this exact clean-body response."
            )
            surface_geometry_contract = {
                "schema": SURFACE_BINDING_SCHEMA,
                "status": "unbound_missing_reviewed_sidecar",
                "surface_mesh": str(surface_path),
                "expected_sidecar": str(binding_path),
            }
            if request.require_feature_manifests:
                raise ValueError(message)
            pre_validation_warnings.append("Surface metadata advisory: selected body and mesh are assumed to share units, frame, and origin; no registration certificate is required.")
        else:
            if request.require_feature_manifests:
                prepared_source_sha256[str(binding_path)] = str(binding_digest)
                prepared_input_sources["surface_binding"] = {
                    "path": str(binding_path),
                    "sha256": str(binding_digest),
                }
            surface_geometry_contract = {
                **binding,
                "status": "reviewed_exact_file_binding",
                "surface_mesh": str(surface_path),
                "sidecar": str(binding_path),
            }

    (
        feature_library_contracts,
        contract_warnings,
        manifest_source_hashes,
        manifest_absent_paths,
    ) = (
        _apply_feature_library_contracts(
            line_placements=lines,
            line_records=line_records,
            point_placements=points,
            point_records=point_records,
            radar_grid=grid,
            require_manifests=bool(request.require_feature_manifests),
            cancel_check=cancel_check,
            host_material=request.host_material,
            host_stack_id=request.host_stack_id,
            host_minimum_radius_m=request.host_minimum_radius_m,
        )
    )
    validation_warnings = pre_validation_warnings + list(contract_warnings)
    validation_warnings.extend(assembly_sampling_warnings(grid, lines, points))
    manifest_absent_paths.update(pre_absent_paths)
    if progress_callback is not None:
        progress_callback(88, 100, "Checking feature-library applicability")
    if coherent_base_missing_metadata:
        validation_warnings.append(
            "Metadata advisory: clean-body convention fields are unspecified: "
            f"{list(coherent_base_missing_metadata)}; the selected body role "
            "supplies the Assembly frame. No field conversion was applied."
        )
    grid_missing_metadata = list(
        base_grid_contract.get("legacy_missing_metadata", ())
    )
    if grid_missing_metadata:
        validation_warnings.append(
            "Clean-body angular/grid metadata is missing "
            f"{grid_missing_metadata}; the selected body role supplies those "
            "semantics as a recorded assumption."
        )
    if mesh_topology_report is not None:
        strict_mesh_errors = []
        if mesh_topology_report.duplicate_triangle_count:
            strict_mesh_errors.append("duplicate facets")
        if mesh_topology_report.nonmanifold_edge_count:
            strict_mesh_errors.append("non-manifold edges")
        if mesh_topology_report.inconsistent_winding_edge_count:
            strict_mesh_errors.append("mixed face winding")
        normals_flipped = bool(request.flip_surface_normals)
        outward_components = int(
            mesh_topology_report.outward_closed_component_count
        )
        inward_components = int(
            mesh_topology_report.inward_closed_component_count
        )
        if outward_components and inward_components:
            strict_mesh_errors.append("mixed closed-component orientation")
        elif inward_components and not normals_flipped:
            strict_mesh_errors.append(
                "globally inward closed mesh without Flip normals"
            )
        elif outward_components and normals_flipped:
            strict_mesh_errors.append(
                "outward closed mesh made inward by Flip normals"
            )
        if request.shadow and mesh_topology_report.boundary_edge_count:
            strict_mesh_errors.append("open boundaries with body shadow enabled")
        if request.require_feature_manifests and strict_mesh_errors:
            raise ValueError(
                "Production Assembly surface mesh failed topology gates: "
                + ", ".join(strict_mesh_errors)
                + ". Repair the body mesh or use the explicitly waived Legacy "
                "profile for an intentional open placement patch."
            )
        validation_warnings.extend(
            mesh_topology_report.messages(
                shadow_requested=bool(request.shadow),
                normals_flipped=normals_flipped,
            )
        )
    for source, digest in manifest_source_hashes.items():
        previous = prepared_source_sha256.get(source)
        if previous is not None and previous != digest:
            raise ValueError(
                f"Prepared manifest source has conflicting hashes: {source}."
            )
        prepared_source_sha256[source] = digest

    normal_fn = (
        surface.normal
        if surface is not None
        else (surface_of_revolution_normal(profile) if profile is not None else None)
    )
    occluder = None
    maximum_shadow_registration_offset = 0.0
    if request.shadow and active_features:
        occluder = Occluder(surface.triangles, bias=request.shadow_bias_m)
        maximum_shadow_registration_offset = max(
            [float(record.get("max_skin_offset_m", 0.0)) for record in line_records]
            + [float(record.get("skin_offset_m", 0.0)) for record in point_records]
            + [0.0]
        )
        if (
            occluder.median_edge > 0.0
            and occluder.bias > 0.05 * occluder.median_edge
        ):
            validation_warnings.append(
                "Body-shadow bias exceeds 5% of the median mesh edge. It may "
                "skip a real nearby blocker; refine/register the mesh and use "
                "the smallest supported bias."
            )

    assembly_workload = _prepared_assembly_workload(
        grid,
        lines,
        points,
        triangle_count=(0 if surface is None else len(surface.triangles)),
        shadow_enabled=bool(request.shadow),
    )
    workload_warning = workload_review_warning(assembly_workload)
    if workload_warning is not None:
        validation_warnings.append(workload_warning)

    requirements = FeatureDatasetRequirements(
        point_dataset_ids=tuple(dict.fromkeys(
            str(record["dataset_id"]) for record in point_records
        )),
        line_dataset_ids=tuple(dict.fromkeys(
            str(record["dataset_id"]) for record in line_records
        )),
        point_placement_count=len(point_records),
        line_path_count=len(line_records),
        line_segment_count=sum(
            int(record["segment_count"]) for record in line_records
        ),
        point_instances=tuple(
            (str(record["placement_id"]), str(record["dataset_id"]))
            for record in point_records
        ),
        line_instances=tuple(
            (
                str(record["line_id"]),
                str(record["dataset_id"]),
                int(record["segment_count"]),
            )
            for record in line_records
        ),
    )
    preview = FeaturePreviewGeometry(
        surface_triangles_cad_m=(
            None
            if surface_triangles_cad_m is None
            else np.asarray(surface_triangles_cad_m, dtype=float)
        ),
        body_profile_rho_z_m=(
            None
            if profile is None
            else np.array(profile, dtype=float, copy=True)
        ),
        point_locations_cad_m={
            dataset_id: np.asarray(locations, dtype=float).reshape(-1, 3)
            for dataset_id, locations in point_preview_lists.items()
        },
        line_paths_cad_m={
            dataset_id: {
                line_id: np.array(path, dtype=float, copy=True)
                for line_id, path in paths.items()
            }
            for dataset_id, paths in line_preview_paths.items()
        },
        point_normals_cad={
            dataset_id: np.asarray(vectors, dtype=float).reshape(-1, 3)
            for dataset_id, vectors in point_preview_normals.items()
        },
        point_roll_references_cad={
            dataset_id: np.asarray(vectors, dtype=float).reshape(-1, 3)
            for dataset_id, vectors in point_preview_roll_references.items()
        },
        line_endpoint_normals_cad={
            dataset_id: {
                line_id: np.array(vectors, dtype=float, copy=True)
                for line_id, vectors in groups.items()
            }
            for dataset_id, groups in line_preview_endpoint_normals.items()
        },
        point_placement_ids={
            dataset_id: tuple(placement_ids)
            for dataset_id, placement_ids in point_preview_ids.items()
        },
    )
    provenance = {
        "request_schema": FEATURE_ASSEMBLY_REQUEST_SCHEMA,
        "study_grid": {key: np.asarray(grid[key]).tolist() for key in ("frequencies_ghz", "azimuths_deg", "elevations_deg")},
        "installed_host": {
            "material": request.host_material, "stack_id": request.host_stack_id,
            "minimum_principal_radius_m": request.host_minimum_radius_m,
            "source": "user-declared installation region; not inferred from body RCS",
        },
        "coordinate_units": request.coordinate_units,
        "surface_mesh": None if surface_path is None else str(surface_path),
        "surface_units": (
            None if surface_path is None else request.surface_units
        ),
        "surface_normals_flipped": bool(request.flip_surface_normals),
        "surface_mesh_topology": (
            None
            if mesh_topology_report is None
            else mesh_topology_report.as_dict()
        ),
        "shadow": bool(request.shadow),
        "shadow_bias_m": (
            None if occluder is None else float(occluder.bias)
        ),
        "shadow_visibility_origin": (
            None
            if occluder is None
            else "nearest validated point on selected body surface; authored "
                 "CSV coordinates remain the coherent phase locations"
        ),
        "maximum_shadow_registration_offset_m": (
            None
            if occluder is None
            else float(maximum_shadow_registration_offset)
        ),
        "surface_geometry_binding": surface_geometry_contract,
        "enabled_selection": {
            "point_placement_ids": [
                str(record["placement_id"]) for record in point_records
            ],
            "line_ids": [str(record["line_id"]) for record in line_records],
        },
        "line_phase_mapping_deg": {
            "TM": float(PSI_HH_DEG),
            "TE": float(PSI_VV_DEG),
        },
        "line_grazing_taper_deg": float(GRAZING_TAPER_DEG),
        "assembly_workload_preflight": assembly_workload.as_dict(),
        "base_grid_contract": base_grid_contract,
        "legacy_base_metadata_allowed": bool(
            request.allow_legacy_base_metadata
        ),
        "metadata_policy": "advisory" if request.allow_legacy_base_metadata and not request.require_feature_manifests else "strict_opt_in",
        "base_metadata_advisories": json.loads(str(validated_base_payload.get("metadata_advisories_json", "[]"))),
        "feature_field_assumptions": "Selected point/line datasets are installed-minus-clean local fields in the documented placement frames. No conversion is inferred from optional solver annotations.",
        "feature_library_manifest_policy": (
            "required_validated"
            if request.require_feature_manifests
            else "advisory"
        ),
        "body_mesh_certification_policy": (
            "required_validated"
            if request.require_body_mesh_certification
            else "external_or_survey_waiver"
        ),
        "body_mesh_certification": body_mesh_certification,
        "feature_library_contracts": feature_library_contracts,
        "validation_warnings": list(validation_warnings),
        "model_scope": {
            "body_feature_mutual_coupling": False,
            "multiple_scattering": False,
            "line_mapping_validation_scope": "circumferential PEC groove",
            "point_pattern_semantics": (
                "installed-feature-minus-clean-skin local complex Jones field"
            ),
        },
        "prepared_input_sources": {
            role: dict(source)
            for role, source in prepared_input_sources.items()
        },
        "placements": line_records + point_records,
    }
    for record in line_records + point_records:
        dataset = record.get("dataset")
        dataset_sha256 = record.get("dataset_sha256")
        if dataset is None or dataset_sha256 is None:
            # Retain compatibility with injected/custom placement preparers
            # whose records predate source-integrity metadata.
            continue
        source = str(resolve_path(dataset, base_dir=request.base_dir))
        digest = str(dataset_sha256)
        previous = prepared_source_sha256.get(source)
        if previous is not None and previous != digest:
            raise ValueError(
                f"Prepared response source has conflicting hashes: {source}."
            )
        prepared_source_sha256[source] = digest

    # Establish one coherent snapshot at the end of preparation.  This catches
    # a source modified while placement validation itself was still running.
    for source, expected in prepared_source_sha256.items():
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Feature placement validation cancelled.")
        try:
            actual = sha256_file(source)
        except OSError as exc:
            raise RuntimeError(
                "Feature-assembly source became unavailable during "
                f"preparation: {source}. Revalidate the assembly."
            ) from exc
        if actual != expected:
            raise RuntimeError(
                "Feature-assembly source changed during preparation: "
                f"{source}. Revalidate the assembly."
            )
    for absent_path in manifest_absent_paths:
        if Path(absent_path).exists():
            raise RuntimeError(
                "Feature-library sidecar state changed during preparation: "
                f"{absent_path} now exists. Revalidate the assembly."
            )
    if prepared_output_absent:
        if output.exists():
            raise RuntimeError(
                f"Assembly output was created during validation: {output}. "
                "Review the current destination and validate again."
            )
    elif (
        not output.is_file()
        or sha256_file(str(output)) != prepared_output_sha256
    ):
        raise RuntimeError(
            f"Assembly output changed during validation: {output}. Review "
            "the newer destination and validate again."
        )
    if prepared_features_only_output_absent:
        if features_only_output.exists():
            raise RuntimeError(
                "Feature-only Assembly output was created during validation: "
                f"{features_only_output}. Review the current destination and "
                "validate again."
            )
    elif (
        not features_only_output.is_file()
        or sha256_file(str(features_only_output))
        != prepared_features_only_output_sha256
    ):
        raise RuntimeError(
            "Feature-only Assembly output changed during validation: "
            f"{features_only_output}. Review the newer destination and "
            "validate again."
        )
    if progress_callback is not None:
        progress_callback(100, 100, "Placement validation complete")
    plan = FeatureAssemblyPlan(
        request=request,
        base_path=base,
        output_path=output,
        radar_grid=grid,
        body_profile=profile,
        surface_path=surface_path,
        surface=surface,
        surface_normal_fn=normal_fn,
        occluder=occluder,
        line_placements=lines,
        point_placements=points,
        line_records=line_records,
        point_records=point_records,
        dataset_requirements=requirements,
        preview_geometry=preview,
        skin_limit_m=float(skin_limit),
        highest_frequency_wavelength_m=float(wavelength),
        feature_provenance=provenance,
        validation_warnings=tuple(validation_warnings),
        prepared_source_sha256=prepared_source_sha256,
        prepared_absent_paths=tuple(sorted(manifest_absent_paths)),
        prepared_output_sha256=prepared_output_sha256,
        prepared_output_absent=bool(prepared_output_absent),
        prepared_features_only_output_sha256=(
            prepared_features_only_output_sha256
        ),
        prepared_features_only_output_absent=bool(
            prepared_features_only_output_absent
        ),
    )
    object.__setattr__(
        plan, "prepared_plan_sha256", feature_assembly_plan_sha256(plan)
    )
    return plan


def _execution_plan_snapshot(plan: FeatureAssemblyPlan) -> FeatureAssemblyPlan:
    """Copy every mutable execution input away from caller-owned containers.

    The snapshot is hashed *after* copying. A concurrent mutation during the
    copy therefore either changes the hash and fails closed or occurs after its
    value has already been isolated. Progress callbacks never receive this
    private snapshot, so they cannot alter validated physics mid-build.
    """

    def frozen_array(value: Any) -> np.ndarray:
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    pattern_copies: dict[int, Any] = {}

    def copied_pattern(pattern: Any) -> Any:
        if not isinstance(pattern, PreparedPointPattern):
            return copy.deepcopy(pattern)
        identity = id(pattern)
        existing = pattern_copies.get(identity)
        if existing is not None:
            return existing
        frozen = PreparedPointPattern(
            frozen_array(pattern.azimuths),
            frozen_array(pattern.elevations),
            frozen_array(pattern.frequencies),
            frozen_array(pattern.amplitude),
            dict(pattern.channel_indices),
        )
        pattern_copies[identity] = frozen
        return frozen

    def placement_copy(placement: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in placement.items():
            if isinstance(value, np.ndarray):
                result[key] = frozen_array(value)
            elif key == "pattern":
                # Deduplicate by response object so ten thousand fasteners copy
                # one reusable Jones cube, not ten thousand. The execution copy
                # is isolated from a callback that re-enables write flags on the
                # reviewed plan's public arrays.
                result[key] = copied_pattern(value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    radar_grid = {
        key: (frozen_array(value) if isinstance(value, np.ndarray) else copy.deepcopy(value))
        for key, value in plan.radar_grid.items()
    }
    request = replace(
        plan.request,
        point_datasets=dict(plan.request.point_datasets),
        line_datasets=dict(plan.request.line_datasets),
    )
    occluder = (
        None
        if plan.occluder is None
        else plan.occluder.execution_snapshot()
    )
    return replace(
        plan,
        request=request,
        radar_grid=radar_grid,
        line_placements=[placement_copy(value) for value in plan.line_placements],
        point_placements=[placement_copy(value) for value in plan.point_placements],
        line_records=copy.deepcopy(plan.line_records),
        point_records=copy.deepcopy(plan.point_records),
        feature_provenance=copy.deepcopy(plan.feature_provenance),
        occluder=occluder,
        prepared_source_sha256=dict(plan.prepared_source_sha256),
        prepared_absent_paths=tuple(plan.prepared_absent_paths),
        validation_warnings=tuple(plan.validation_warnings),
    )


def execute_feature_assembly(
    plan: FeatureAssemblyPlan, *,
    acknowledged_plan_sha256: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Coherently execute a prepared plan using the authoritative physics API."""

    if not isinstance(plan, FeatureAssemblyPlan):
        raise TypeError("plan must be a FeatureAssemblyPlan.")
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature assembly cancelled; existing output kept.")
    # Recheck immediately before publication.  A symlink or hard link may have
    # been created after preparation, and the backend must remain safe for
    # headless callers that do not pass through the GRIM GUI model.
    _reject_output_aliases(
        plan.request,
        base=plan.base_path,
        output=plan.output_path,
    )
    if not plan.prepared_plan_sha256:
        raise RuntimeError(
            "Feature Assembly plan predates the sealed validation contract; "
            "validate the placement configuration again before building."
        )
    if warnings_require_workload_acknowledgement(plan.validation_warnings):
        acknowledgement = str(acknowledged_plan_sha256 or "").strip().lower()
        expected = str(plan.prepared_plan_sha256).strip().lower()
        if acknowledgement != expected:
            raise RuntimeError(
                "Assembly workload review was not acknowledged. Review the "
                "operation counts in validation_warnings, then pass this exact "
                "sealed plan digest as acknowledged_plan_sha256: "
                f"{expected}. Output was not published."
            )
    capacity_estimate = preflight_feature_assembly_capacity(
        str(plan.base_path),
        str(plan.output_path),
        radar_grid=plan.radar_grid,
        placements=plan.line_placements,
        points=plan.point_placements,
        occluder=plan.occluder,
    )
    execution_plan = _execution_plan_snapshot(plan)
    actual_plan_sha256 = feature_assembly_plan_sha256(execution_plan)
    if actual_plan_sha256 != plan.prepared_plan_sha256:
        raise RuntimeError(
            "Feature Assembly plan changed after validation. Output was not "
            "published; validate the current placement configuration again."
        )
    return add_features_to_monostatic_grim(
        str(execution_plan.base_path),
        str(execution_plan.output_path),
        placements=execution_plan.line_placements,
        points=execution_plan.point_placements,
        radar_grid=execution_plan.radar_grid,
        surface_normal_fn=execution_plan.surface_normal_fn,
        occluder=execution_plan.occluder,
        psi_tm_deg=PSI_HH_DEG,
        psi_te_deg=PSI_VV_DEG,
        declared_coherent_base=True,
        allow_legacy_base_metadata=bool(
            execution_plan.request.allow_legacy_base_metadata
        ),
        feature_provenance=execution_plan.feature_provenance,
        history=str(execution_plan.request.history),
        expected_source_sha256=execution_plan.prepared_source_sha256,
        expected_absent_paths=execution_plan.prepared_absent_paths,
        expected_output_sha256=execution_plan.prepared_output_sha256,
        expect_output_absent=execution_plan.prepared_output_absent,
        expected_features_only_output_sha256=(
            execution_plan.prepared_features_only_output_sha256
        ),
        expect_features_only_output_absent=(
            execution_plan.prepared_features_only_output_absent
        ),
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        _capacity_estimate=capacity_estimate,
    )


def run_feature_assembly(
    request: FeatureAssemblyRequest, *,
    acknowledged_plan_sha256: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Prepare and execute one coherent feature assembly.

    Preparation occupies the first quarter of the public progress range and
    numerical execution/publication occupies the remainder.  This keeps a
    single GUI/headless callback monotone instead of resetting from 100 back
    to zero between the two phases.
    """

    def mapped_progress(start: int, span: int):
        if progress_callback is None:
            return None

        def report(done: int, total: int, message: str) -> None:
            fraction = max(0.0, min(1.0, float(done) / max(1, int(total))))
            progress_callback(
                int(round(start + span * fraction)), 100, str(message)
            )

        return report

    plan = prepare_feature_assembly(
        request,
        cancel_check=cancel_check,
        progress_callback=mapped_progress(0, 25),
    )
    execute_kwargs = {
        "cancel_check": cancel_check,
        "progress_callback": mapped_progress(25, 75),
    }
    if acknowledged_plan_sha256 is not None:
        execute_kwargs["acknowledged_plan_sha256"] = (
            acknowledged_plan_sha256
        )
    return execute_feature_assembly(plan, **execute_kwargs)


__all__ = [
    "DECLARED_FEATURE_DELTA_RESPONSE_SCHEMA",
    "FEATURE_ASSEMBLY_REQUEST_SCHEMA",
    "FEATURE_LIBRARY_MANIFEST_KEY",
    "FEATURE_LIBRARY_MANIFEST_SCHEMA",
    "FEATURE_VALIDATION_ARTIFACT_ROLES",
    "FEATURE_VALIDATION_EVIDENCE_SCHEMA",
    "FEATURE_VALIDATION_MAX_ACTIVE_FLOOR_DB",
    "FEATURE_VALIDATION_RELEASE_CEILINGS",
    "LEGACY_FEATURE_LIBRARY_MANIFEST_SCHEMA",
    "LEGACY_LINE_PHASE_CALIBRATION_SCHEMA",
    "LINE_CSV_COLUMNS",
    "LINE_PHASE_CALIBRATION_SCHEMA",
    "LINE_PLACEMENT_SCHEMA",
    "POINT_CSV_COLUMNS",
    "POINT_PLACEMENT_SCHEMA",
    "PREVIOUS_FEATURE_LIBRARY_MANIFEST_SCHEMA",
    "SURFACE_BINDING_SCHEMA",
    "SURFACE_FRAME_CONVENTION",
    "FeatureAssemblyPlan",
    "FeatureAssemblyRequest",
    "FeatureDatasetRequirements",
    "FeatureInputPreview",
    "FeaturePreviewGeometry",
    "compute_skin_limit",
    "check_surface_binding",
    "discover_feature_dataset_ids",
    "execute_feature_assembly",
    "feature_assembly_plan_sha256",
    "feature_only_output_path",
    "feature_response_content_sha256",
    "feature_response_physics_sha256",
    "load_feature_library_manifest",
    "load_surface_binding",
    "prepare_feature_input_preview",
    "prepare_feature_assembly",
    "prepare_line_placements",
    "prepare_point_placements",
    "read_line_placement_csv",
    "read_point_placement_csv",
    "resolve_path",
    "run_feature_assembly",
    "surface_binding_path",
    "unit_vector",
    "validate_declared_feature_delta_response",
    "validate_feature_library_manifest",
    "validate_normal_tolerance",
    "validate_surface_binding",
    "write_surface_binding",
]
