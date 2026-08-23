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
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from feature_sum import (
    _load_grim,
    add_features_to_monostatic_grim,
    load_body_profile_grim,
    load_body_requested_radar_grid,
    prepare_point_pattern,
    surface_of_revolution_distance,
)
from frame import (
    AXIS_AZ_DEG,
    AXIS_EL_DEG,
    ROLL_DEG,
    scale_for,
    to_axis_frame,
)
from line_expand import (
    C0,
    perimeter_surface_deviation,
    surface_of_revolution_normal,
)
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
    coordinate_units: str = "inches"

    surface_mesh: Optional[PathValue] = None
    surface_units: str = "inches"
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


@dataclass(frozen=True)
class FeatureDatasetRequirements:
    """Dataset IDs discovered from already schema-validated placement CSVs."""

    point_dataset_ids: tuple[str, ...] = ()
    line_dataset_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeaturePreviewGeometry:
    """Validated full-resolution geometry in the user-visible CAD frame.

    Coordinates are metres.  The body profile remains the frame-free BoR
    ``rho,z`` generatrix.  GRIM may decimate a copy for display, but the full
    surface stored here is the same geometry from which the physics surface
    was constructed.
    """

    surface_triangles_cad_m: Optional[np.ndarray]
    body_profile_rho_z_m: Optional[np.ndarray]
    point_locations_cad_m: dict[str, np.ndarray]
    line_paths_cad_m: dict[str, dict[str, np.ndarray]]


@dataclass
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


def resolve_path(value: PathValue, *, base_dir: Optional[PathValue] = None) -> Path:
    """Resolve one user path without imposing a repository-specific root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
    return (root.resolve() / path).resolve()


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
    seen: set[str] = set()
    numeric_columns = POINT_CSV_COLUMNS[2:]
    for row, number in rows:
        if len(row) != len(POINT_CSV_COLUMNS):
            raise ValueError(
                f"{source}:{number}: expected exactly "
                f"{len(POINT_CSV_COLUMNS)} columns."
            )
        placement_id, dataset_id = row[:2]
        if not placement_id or not dataset_id:
            raise ValueError(
                f"{source}:{number}: placement_id and dataset_id are required."
            )
        if placement_id in seen:
            raise ValueError(
                f"{source}:{number}: duplicate placement_id {placement_id!r}."
            )
        seen.add(placement_id)
        try:
            numeric = [float(value) for value in row[2:]]
        except ValueError as exc:
            raise ValueError(
                f"{source}:{number}: coordinates and vectors must be numeric."
            ) from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{source}:{number}: NaN/infinite value.")
        values: dict[str, Any] = {
            "placement_id": placement_id,
            "dataset_id": dataset_id,
        }
        values.update(dict(zip(numeric_columns, numeric)))
        values["_csv_line"] = number
        parsed.append(values)
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
    completed_line_ids: set[str] = set()
    current_line_id: Optional[str] = None
    current_dataset_id: Optional[str] = None
    expected_index = 1
    numeric_columns = LINE_CSV_COLUMNS[3:]
    for row, number in rows:
        if len(row) != len(LINE_CSV_COLUMNS):
            raise ValueError(
                f"{source}:{number}: expected exactly "
                f"{len(LINE_CSV_COLUMNS)} columns."
            )
        line_id, dataset_id, raw_index = row[:3]
        if not line_id or not dataset_id:
            raise ValueError(
                f"{source}:{number}: line_id and dataset_id are required."
            )
        if line_id != current_line_id:
            if current_line_id is not None:
                completed_line_ids.add(current_line_id)
            if line_id in completed_line_ids:
                raise ValueError(
                    f"{source}:{number}: rows for line_id {line_id!r} "
                    "must be contiguous."
                )
            current_line_id = line_id
            current_dataset_id = dataset_id
            expected_index = 1
        elif dataset_id != current_dataset_id:
            raise ValueError(
                f"{source}:{number}: every segment of line_id {line_id!r} "
                "must use the same dataset_id."
            )
        try:
            segment_index = int(raw_index)
        except ValueError as exc:
            raise ValueError(
                f"{source}:{number}: segment_index must be an integer."
            ) from exc
        if str(segment_index) != raw_index or segment_index != expected_index:
            raise ValueError(
                f"{source}:{number}: line_id {line_id!r} requires consecutive "
                f"segment_index values starting at 1; expected {expected_index}."
            )
        try:
            numeric = [float(value) for value in row[3:]]
        except ValueError as exc:
            raise ValueError(
                f"{source}:{number}: endpoints and normals must be numeric."
            ) from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{source}:{number}: NaN/infinite value.")
        values: dict[str, Any] = {
            "line_id": line_id,
            "dataset_id": dataset_id,
            "segment_index": segment_index,
            "_csv_line": number,
        }
        values.update(dict(zip(numeric_columns, numeric)))
        parsed.append(values)
        expected_index += 1
    return parsed


def _ordered_dataset_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(row["dataset_id"]) for row in rows))


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


def _sample_perimeter(perimeter: np.ndarray, samples_per_segment: int = 33) -> np.ndarray:
    segments = np.asarray(perimeter, dtype=float)
    parameter = np.linspace(0.0, 1.0, max(2, int(samples_per_segment)))
    return (
        segments[:, 0, None, :] * (1.0 - parameter)[None, :, None]
        + segments[:, 1, None, :] * parameter[None, :, None]
    ).reshape(-1, 3)


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
    base_dir: Optional[PathValue] = None,
    preview_paths_cad_m: Optional[dict[str, dict[str, np.ndarray]]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and prepare line-expanded feature placements."""

    if locations_csv is None:
        if datasets:
            raise ValueError(
                "line_datasets is configured but line_locations_csv is None."
            )
        return [], []
    if not datasets:
        raise ValueError(
            "line_locations_csv is configured but line_datasets is empty."
        )

    normal_tolerance = validate_normal_tolerance(normal_tolerance_deg)
    coordinates = resolve_path(locations_csv, base_dir=base_dir)
    rows = read_line_placement_csv(coordinates)
    coordinates_sha256 = sha256_file(str(coordinates))
    dataset_paths, dataset_hashes = _resolved_dataset_paths(
        datasets, kind="line", base_dir=base_dir
    )
    _require_known_dataset_ids(rows, dataset_paths, coordinates=coordinates)

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

        if surface is None:
            offset = perimeter_surface_deviation(
                perimeter, profile, samples_per_segment=33
            )
        else:
            offset = float(np.max(surface.distance(_sample_perimeter(perimeter))))
        if offset > limit:
            raise ValueError(
                f"{coordinates}: line_id {line_id!r} is {offset * 1e3:.3f} mm "
                f"off the skin ({720.0 * offset / wavelength:.1f} deg two-way "
                f"phase); allowed {limit * 1e3:.3f} mm."
            )

        normal_parameter = np.linspace(0.0, 1.0, 33)
        normal_points = (
            perimeter[:, 0, None, :]
            * (1.0 - normal_parameter)[None, :, None]
            + perimeter[:, 1, None, :]
            * normal_parameter[None, :, None]
        ).reshape(-1, 3)
        supplied = (
            segment_normals[:, 0, None, :]
            * (1.0 - normal_parameter)[None, :, None]
            + segment_normals[:, 1, None, :]
            * normal_parameter[None, :, None]
        ).reshape(-1, 3)
        supplied_magnitudes = np.linalg.norm(supplied, axis=1)
        if np.any(supplied_magnitudes <= 1.0e-12):
            raise ValueError(
                f"{coordinates}: line_id {line_id!r} endpoint-normal "
                "interpolation becomes singular; subdivide the line and "
                "supply the outward normal at the added vertex."
            )
        supplied /= supplied_magnitudes[:, None]
        derived = np.asarray(normal_fn(normal_points), dtype=float)
        if derived.shape != normal_points.shape or not np.all(np.isfinite(derived)):
            raise ValueError("surface normal query returned invalid vectors.")
        derived_magnitudes = np.linalg.norm(derived, axis=1)
        if np.any(derived_magnitudes <= 1.0e-12):
            raise ValueError("surface normal query returned a zero-length vector.")
        derived /= derived_magnitudes[:, None]
        differences = np.degrees(np.arccos(np.clip(
            np.sum(supplied * derived, axis=1), -1.0, 1.0
        )))
        if np.any(differences > normal_tolerance):
            flat_index = int(np.argmax(differences))
            segment_index, sample_index = divmod(
                flat_index, len(normal_parameter)
            )
            raise ValueError(
                f"{coordinates}:line {group[segment_index]['_csv_line']} "
                "supplied normal interpolation differs from the outward skin "
                f"normal by {differences[flat_index]:.2f} deg at segment "
                f"fraction {normal_parameter[sample_index]:.5g}."
            )

        placements.append({
            "delta": str(dataset),
            "perimeter": perimeter,
            "segment_normals": segment_normals,
            "kind": "delta",
            "declared_coherent_delta": True,
            "delta_sign": 1.0,
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
            "normal_source": "csv_endpoint_interpolation",
        })
        if preview_paths_cad_m is not None:
            line_path = np.concatenate(
                [perimeter_cad[:, 0, :], perimeter_cad[-1:, 1, :]],
                axis=0,
            )
            preview_paths_cad_m.setdefault(dataset_id, {})[line_id] = np.array(
                line_path, dtype=float, copy=True
            )
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
    base_dir: Optional[PathValue] = None,
    pattern_loader: Optional[Callable[..., Any]] = None,
    preview_locations_cad_m: Optional[dict[str, list[np.ndarray]]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and prepare compact 3-D point-feature placements."""

    if locations_csv is None:
        if datasets:
            raise ValueError(
                "point_datasets is configured but point_locations_csv is None."
            )
        return [], []
    if not datasets:
        raise ValueError(
            "point_locations_csv is configured but point_datasets is empty."
        )

    normal_tolerance = validate_normal_tolerance(normal_tolerance_deg)
    coordinates = resolve_path(locations_csv, base_dir=base_dir)
    rows = read_point_placement_csv(coordinates)
    coordinates_sha256 = sha256_file(str(coordinates))
    dataset_paths, dataset_hashes = _resolved_dataset_paths(
        datasets, kind="point", base_dir=base_dir
    )
    _require_known_dataset_ids(rows, dataset_paths, coordinates=coordinates)

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
        offset = float(
            surface_of_revolution_distance(profile, location[None, :])[0]
            if surface is None
            else surface.distance(location[None, :])[0]
        )
        if offset > limit:
            raise ValueError(
                f"{coordinates}:line {csv_line} is {offset * 1e3:.3f} mm off "
                f"the skin ({720.0 * offset / wavelength:.1f} deg two-way phase)."
            )
        derived = unit_vector(
            normal_fn(location[None, :])[0], "derived normal"
        )
        normal = unit_vector(
            to_axis_frame([row["nx"], row["ny"], row["nz"]]),
            "supplied normal",
        )
        difference = math.degrees(math.acos(np.clip(
            float(normal @ derived), -1.0, 1.0
        )))
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
            "input_subtraction_order": "OPN-FRD (featured-clean)",
            "assumed_missing_cross_pol_zero": False,
            "roll_reference": "csv",
        })
        if preview_locations_cad_m is not None:
            preview_locations_cad_m.setdefault(dataset_id, []).append(
                np.array(location_cad_m, dtype=float, copy=True)
            )
    return points, records


def _library_unit_scale(units: str, *, label: str) -> float:
    """Use the canonical frame conversion without allowing a library exit."""

    try:
        return float(scale_for(units))
    except SystemExit as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc


def prepare_feature_assembly(request: FeatureAssemblyRequest) -> FeatureAssemblyPlan:
    """Resolve, validate, and prepare one feature-assembly request."""

    if not isinstance(request, FeatureAssemblyRequest):
        raise TypeError("request must be a FeatureAssemblyRequest.")
    base = resolve_path(request.base_grim, base_dir=request.base_dir)
    output = resolve_path(request.output_grim, base_dir=request.base_dir)
    if output == base:
        raise ValueError(
            "output_grim must differ from base_grim so the clean-body "
            "response is not overwritten."
        )
    if request.line_locations_csv is None and request.point_locations_csv is None:
        raise ValueError(
            "Configure line_locations_csv or point_locations_csv."
        )
    if not base.is_file():
        raise FileNotFoundError(f"Base monostatic GRIM not found: {base}")

    embedded_grid = load_body_requested_radar_grid(str(base))
    profile: Optional[np.ndarray] = None
    if embedded_grid is not None:
        profile = load_body_profile_grim(str(base))
        grid = dict(embedded_grid)
    else:
        payload = _load_grim(str(base))
        grid = {
            "frequencies_ghz": np.asarray(payload["frequencies"], dtype=float),
            "azimuths_deg": np.asarray(payload["azimuths"], dtype=float),
            "elevations_deg": np.asarray(payload["elevations"], dtype=float),
            "axis_az_deg": float(AXIS_AZ_DEG),
            "axis_el_deg": float(AXIS_EL_DEG),
            "roll_deg": float(ROLL_DEG),
        }

    surface: Optional[TriangleSurface] = None
    surface_path: Optional[Path] = None
    surface_triangles_cad_m: Optional[np.ndarray] = None
    if request.surface_mesh is not None:
        surface_path = resolve_path(request.surface_mesh, base_dir=request.base_dir)
        if not surface_path.is_file():
            raise FileNotFoundError(f"Surface mesh not found: {surface_path}")
        surface_scale = _library_unit_scale(
            request.surface_units, label="surface_units"
        )
        surface_triangles_cad_m = (
            np.asarray(read_surface_mesh(str(surface_path)), dtype=float)
            * surface_scale
        )
        triangles = to_axis_frame(surface_triangles_cad_m)
        surface = TriangleSurface(
            triangles,
            flip_normals=bool(request.flip_surface_normals),
        )
    elif embedded_grid is None:
        raise ValueError(
            "A non-BoR base requires surface_mesh=.facet or .stl for skin "
            "validation and outward normals."
        )
    if request.shadow and surface is None:
        raise ValueError("shadow=True requires surface_mesh.")

    coordinate_scale = _library_unit_scale(
        request.coordinate_units, label="coordinate_units"
    )
    skin_limit, wavelength = compute_skin_limit(
        grid["frequencies_ghz"],
        skin_tol_m=request.skin_tol_m,
        skin_phase_tol_deg=request.skin_phase_tol_deg,
    )
    normal_tolerance = validate_normal_tolerance(request.normal_tol_deg)
    point_preview_lists: dict[str, list[np.ndarray]] = {}
    line_preview_paths: dict[str, dict[str, np.ndarray]] = {}
    lines, line_records = prepare_line_placements(
        profile,
        surface,
        coordinate_scale=coordinate_scale,
        skin_limit_m=skin_limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=normal_tolerance,
        locations_csv=request.line_locations_csv,
        datasets=request.line_datasets,
        base_dir=request.base_dir,
        preview_paths_cad_m=line_preview_paths,
    )
    points, point_records = prepare_point_placements(
        profile,
        surface,
        coordinate_scale=coordinate_scale,
        skin_limit_m=skin_limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=normal_tolerance,
        locations_csv=request.point_locations_csv,
        datasets=request.point_datasets,
        base_dir=request.base_dir,
        preview_locations_cad_m=point_preview_lists,
    )

    normal_fn = (
        surface.normal
        if surface is not None
        else surface_of_revolution_normal(profile)
    )
    occluder = None
    if request.shadow:
        occluder = Occluder(surface.triangles, bias=request.shadow_bias_m)

    requirements = FeatureDatasetRequirements(
        point_dataset_ids=tuple(dict.fromkeys(
            str(record["dataset_id"]) for record in point_records
        )),
        line_dataset_ids=tuple(dict.fromkeys(
            str(record["dataset_id"]) for record in line_records
        )),
    )
    preview = FeaturePreviewGeometry(
        surface_triangles_cad_m=(
            None
            if surface_triangles_cad_m is None
            else np.array(surface_triangles_cad_m, dtype=float, copy=True)
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
    )
    provenance = {
        "request_schema": FEATURE_ASSEMBLY_REQUEST_SCHEMA,
        "coordinate_units": request.coordinate_units,
        "surface_mesh": None if surface_path is None else str(surface_path),
        "surface_units": (
            None if surface_path is None else request.surface_units
        ),
        "surface_normals_flipped": bool(request.flip_surface_normals),
        "shadow": bool(request.shadow),
        "shadow_bias_m": (
            None if occluder is None else float(occluder.bias)
        ),
        "placements": line_records + point_records,
    }
    return FeatureAssemblyPlan(
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
    )


def execute_feature_assembly(plan: FeatureAssemblyPlan) -> str:
    """Coherently execute a prepared plan using the authoritative physics API."""

    if not isinstance(plan, FeatureAssemblyPlan):
        raise TypeError("plan must be a FeatureAssemblyPlan.")
    return add_features_to_monostatic_grim(
        str(plan.base_path),
        str(plan.output_path),
        placements=plan.line_placements,
        points=plan.point_placements,
        radar_grid=plan.radar_grid,
        surface_normal_fn=plan.surface_normal_fn,
        occluder=plan.occluder,
        declared_coherent_base=True,
        feature_provenance=plan.feature_provenance,
        history=str(plan.request.history),
    )


def run_feature_assembly(request: FeatureAssemblyRequest) -> str:
    """Prepare and execute one coherent feature assembly."""

    return execute_feature_assembly(prepare_feature_assembly(request))


__all__ = [
    "FEATURE_ASSEMBLY_REQUEST_SCHEMA",
    "LINE_CSV_COLUMNS",
    "LINE_PLACEMENT_SCHEMA",
    "POINT_CSV_COLUMNS",
    "POINT_PLACEMENT_SCHEMA",
    "FeatureAssemblyPlan",
    "FeatureAssemblyRequest",
    "FeatureDatasetRequirements",
    "FeaturePreviewGeometry",
    "compute_skin_limit",
    "discover_feature_dataset_ids",
    "execute_feature_assembly",
    "prepare_feature_assembly",
    "prepare_line_placements",
    "prepare_point_placements",
    "read_line_placement_csv",
    "read_point_placement_csv",
    "resolve_path",
    "run_feature_assembly",
    "unit_vector",
    "validate_normal_tolerance",
]
