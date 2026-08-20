#!/usr/bin/env python3
"""Coherently place point and line features on a platform result.

Edit only the USER SETTINGS block, then run:

    python place_features.py

The base may be a self-contained GHOST BoR monostatic GRIM or an attested
external monostatic GRIM paired with an indexed ASCII ``.facet``/STL platform
surface. GUI power/phase subtraction results are accepted directly. Point and
line datasets use the single canonical OPN-FRD (featured-clean) differential
response. Metadata that merely repeats that declaration is optional.
"""

import csv
import math
from pathlib import Path

import numpy as np

# =============================================================================
# USER SETTINGS
# =============================================================================

# Selecting this file explicitly attests that it is the coherent monostatic
# platform field in the GHOST global-origin radar VV/HH/VH convention. A GRIM
# GUI file may omit those tags; an explicitly power-only file is still refused.
BASE_MONOSTATIC_GRIM = "rcs_runs_bor/run_x/results/body.grim"
OUTPUT_MONOSTATIC_GRIM = "rcs_runs_bor/run_x/results/body_with_features.grim"
COORDINATE_UNITS = "inches"

# Required for a non-BoR base. It may also be supplied for a BoR result when
# mesh-based skin checks and shadowing are desired. The mesh and coordinate
# files must use the same CAD frame/origin (+y nose, +x right, +z up).
SURFACE_MESH = None               # e.g. "platform.facet" or "platform.stl"
SURFACE_UNITS = "inches"
FLIP_SURFACE_NORMALS = False      # True only when mesh winding points inward

# Geometric-optics blockage. False keeps the local outward-facing test but
# does not hide an otherwise facing feature behind another part of the body.
SHADOW = False
SHADOW_BIAS_M = None              # normally leave None for mesh-scaled default

# One strict line-placement table is used for every line-expanded feature.
# The CSV header must be exactly:
#
# line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,n1x,n1y,n1z,n2x,n2y,n2z
#
# Rows for each line_id must be contiguous and numbered from 1. Segments must
# chain head-to-tail in the CAD frame (+y nose, +x right, +z up). Endpoint
# normals are explicit so curved-skin frames are interpolated without guessing.
# The tangent and outward normal fully orient the local 2-D response, so no
# separate roll vector is needed.
LINE_FEATURE_LOCATIONS_CSV = None       # e.g. "line_features.csv"
LINE_FEATURE_DATASETS = {
    # "panel_gap": "panel_gap_opn_minus_frd.grim",
    # "door_seam": "door_seam_opn_minus_frd.grim",
}

# One strict point-placement table is used for every compact feature type.
# The CSV header must be exactly:
#
# placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z
#
# Coordinates and vectors use the CAD frame (+y nose, +x right, +z up).
# ``dataset_id`` selects one entry below. The normal and roll reference are
# always explicit: this avoids file-shape inference and makes the full 3-D
# orientation of an asymmetric antenna unambiguous. For an axisymmetric
# fastener, choose any stable roll vector not parallel to its normal.
POINT_FEATURE_LOCATIONS_CSV = None       # e.g. "point_features.csv"
POINT_FEATURE_DATASETS = {
    # "fastener": "fastener_opn_minus_frd.grim",
    # "antenna": "antenna_opn_minus_frd.grim",
}

# Listing a dataset above explicitly attests that it is the coherent
# installed-feature-minus-clean-skin delta, with its origin at the aperture
# phase center, exp(+jwt), and the documented cavity-frame VV/HH/VH basis.
# This accepts a GRIM GUI subtraction that dropped those convention tags and
# whose generic container remains tagged power_phase. Incorrect placement
# declarations produce incorrect coherent phase; do not list standalone fields.
# Every point dataset must contain VV, HH, and reciprocal VH/HV. Missing point
# cross-polarization is not guessed to be zero. Every line dataset must contain
# the 2-D TE and TM complex responses. OPN-FRD is the only accepted delta order.
# Convert an FRD-OPN file once when building the dataset; placement never infers
# or repairs its sign.

# Coordinate-to-skin validation.  The tighter of the distance and two-way
# phase limits is enforced at the highest frequency in the body file.
SKIN_TOL_M = 1.0e-3
SKIN_PHASE_TOL_DEG = 15.0
NORMAL_TOL_DEG = 15.0

# =============================================================================

from feature_sum import (  # noqa: E402
    _load_grim,
    add_features_to_monostatic_grim,
    load_body_profile_grim,
    load_body_requested_radar_grid,
    prepare_point_pattern,
    surface_of_revolution_distance,
)
from frame import (  # noqa: E402
    AXIS_AZ_DEG,
    AXIS_EL_DEG,
    ROLL_DEG,
    scale_for,
    to_axis_frame,
)
from line_expand import (  # noqa: E402
    C0,
    perimeter_surface_deviation,
    surface_of_revolution_normal,
)
from occluder import Occluder  # noqa: E402
from surface_mesh import TriangleSurface, read_surface_mesh  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _unit(value, label):
    vector = np.asarray(value, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or magnitude <= 1e-12:
        raise ValueError(f"{label} must be one finite nonzero 3-vector.")
    return vector / magnitude


def _skin_limit(frequencies_ghz):
    highest = max(float(value) for value in frequencies_ghz)
    wavelength = C0 / (highest * 1.0e9)
    limit = min(float(SKIN_TOL_M), float(SKIN_PHASE_TOL_DEG) * wavelength / 720.0)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("Skin tolerances must be finite and nonnegative.")
    return limit, wavelength


def _normal_tolerance():
    tolerance = float(NORMAL_TOL_DEG)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 180.0:
        raise ValueError("NORMAL_TOL_DEG must be finite and between 0 and 180.")
    return tolerance


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


def _placement_rows(path):
    """Read the one strict point-placement CSV schema."""
    with _path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = [([cell.strip() for cell in row], line_number)
                for line_number, row in enumerate(csv.reader(stream), 1)
                if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path}: placement CSV is empty.")
    header, header_line = rows.pop(0)
    if tuple(header) != POINT_CSV_COLUMNS:
        raise ValueError(
            f"{path}:{header_line}: header must be exactly "
            f"{','.join(POINT_CSV_COLUMNS)}."
        )
    if not rows:
        raise ValueError(f"{path}: placement CSV has a header but no placements.")
    parsed = []
    seen = set()
    numeric_columns = POINT_CSV_COLUMNS[2:]
    for row, number in rows:
        if len(row) != len(POINT_CSV_COLUMNS):
            raise ValueError(
                f"{path}:{number}: expected exactly "
                f"{len(POINT_CSV_COLUMNS)} columns."
            )
        placement_id, dataset_id = row[:2]
        if not placement_id or not dataset_id:
            raise ValueError(
                f"{path}:{number}: placement_id and dataset_id are required."
            )
        if placement_id in seen:
            raise ValueError(
                f"{path}:{number}: duplicate placement_id {placement_id!r}."
            )
        seen.add(placement_id)
        try:
            numeric = [float(value) for value in row[2:]]
        except ValueError as exc:
            raise ValueError(
                f"{path}:{number}: coordinates and vectors must be numeric."
            ) from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{path}:{number}: NaN/infinite value.")
        values = {"placement_id": placement_id, "dataset_id": dataset_id}
        values.update(dict(zip(numeric_columns, numeric)))
        values["_csv_line"] = number
        parsed.append(values)
    return parsed


def _line_rows(path):
    """Read the one strict ordered-segment line-placement CSV schema."""
    with _path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = [([cell.strip() for cell in row], line_number)
                for line_number, row in enumerate(csv.reader(stream), 1)
                if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path}: line-placement CSV is empty.")
    header, header_line = rows.pop(0)
    if tuple(header) != LINE_CSV_COLUMNS:
        raise ValueError(
            f"{path}:{header_line}: header must be exactly "
            f"{','.join(LINE_CSV_COLUMNS)}."
        )
    if not rows:
        raise ValueError(f"{path}: line-placement CSV has a header but no segments.")

    parsed = []
    completed_line_ids = set()
    current_line_id = None
    current_dataset_id = None
    expected_index = 1
    numeric_columns = LINE_CSV_COLUMNS[3:]
    for row, number in rows:
        if len(row) != len(LINE_CSV_COLUMNS):
            raise ValueError(
                f"{path}:{number}: expected exactly {len(LINE_CSV_COLUMNS)} columns."
            )
        line_id, dataset_id, raw_index = row[:3]
        if not line_id or not dataset_id:
            raise ValueError(f"{path}:{number}: line_id and dataset_id are required.")
        if line_id != current_line_id:
            if current_line_id is not None:
                completed_line_ids.add(current_line_id)
            if line_id in completed_line_ids:
                raise ValueError(
                    f"{path}:{number}: rows for line_id {line_id!r} must be contiguous."
                )
            current_line_id = line_id
            current_dataset_id = dataset_id
            expected_index = 1
        elif dataset_id != current_dataset_id:
            raise ValueError(
                f"{path}:{number}: every segment of line_id {line_id!r} must use "
                "the same dataset_id."
            )
        try:
            segment_index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: segment_index must be an integer.") from exc
        if str(segment_index) != raw_index or segment_index != expected_index:
            raise ValueError(
                f"{path}:{number}: line_id {line_id!r} requires consecutive "
                f"segment_index values starting at 1; expected {expected_index}."
            )
        try:
            numeric = [float(value) for value in row[3:]]
        except ValueError as exc:
            raise ValueError(
                f"{path}:{number}: endpoints and normals must be numeric."
            ) from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{path}:{number}: NaN/infinite value.")
        values = {
            "line_id": line_id, "dataset_id": dataset_id,
            "segment_index": segment_index, "_csv_line": number,
        }
        values.update(dict(zip(numeric_columns, numeric)))
        parsed.append(values)
        expected_index += 1
    return parsed


def _sample_perimeter(perimeter, samples_per_segment=33):
    segments = np.asarray(perimeter, dtype=float)
    parameter = np.linspace(0.0, 1.0, max(2, int(samples_per_segment)))
    return (
        segments[:, 0, None, :] * (1.0 - parameter)[None, :, None]
        + segments[:, 1, None, :] * parameter[None, :, None]
    ).reshape(-1, 3)


def _line_placements(profile, surface, scale, limit, wavelength):
    if LINE_FEATURE_LOCATIONS_CSV is None:
        if LINE_FEATURE_DATASETS:
            raise ValueError(
                "LINE_FEATURE_DATASETS is configured but "
                "LINE_FEATURE_LOCATIONS_CSV is None."
            )
        return [], []
    if not LINE_FEATURE_DATASETS:
        raise ValueError(
            "LINE_FEATURE_LOCATIONS_CSV is configured but "
            "LINE_FEATURE_DATASETS is empty."
        )

    normal_tolerance = _normal_tolerance()
    coordinates = _path(LINE_FEATURE_LOCATIONS_CSV)
    rows = _line_rows(coordinates)
    from workflow_provenance import sha256_file
    coordinates_sha256 = sha256_file(str(coordinates))
    dataset_paths = {}
    dataset_sha256 = {}
    for dataset_id, value in LINE_FEATURE_DATASETS.items():
        if not isinstance(dataset_id, str):
            raise ValueError("LINE_FEATURE_DATASETS keys must be strings.")
        identifier = dataset_id
        if not identifier or identifier != identifier.strip():
            raise ValueError(
                "LINE_FEATURE_DATASETS keys must be nonempty strings without "
                "leading/trailing whitespace."
            )
        dataset = _path(value)
        if not dataset.is_file():
            raise FileNotFoundError(
                f"Line dataset {identifier!r} does not exist: {dataset}"
            )
        dataset_paths[identifier] = dataset
        dataset_sha256[identifier] = sha256_file(str(dataset))
    unknown = sorted({row["dataset_id"] for row in rows} - set(dataset_paths))
    if unknown:
        raise ValueError(
            f"{coordinates}: unknown dataset_id value(s) {unknown}; configured "
            f"IDs are {sorted(dataset_paths)}."
        )

    normal_fn = surface_of_revolution_normal(profile) if surface is None else surface.normal
    placements = []
    records = []
    start = 0
    while start < len(rows):
        line_id = rows[start]["line_id"]
        end = start + 1
        while end < len(rows) and rows[end]["line_id"] == line_id:
            end += 1
        group = rows[start:end]
        dataset_id = group[0]["dataset_id"]
        dataset = dataset_paths[dataset_id]
        perimeter_cad = np.asarray([
            [[row["x1"], row["y1"], row["z1"]],
             [row["x2"], row["y2"], row["z2"]]]
            for row in group
        ], dtype=float) * scale
        normal_cad = np.asarray([
            [[row["n1x"], row["n1y"], row["n1z"]],
             [row["n2x"], row["n2y"], row["n2z"]]]
            for row in group
        ], dtype=float)
        perimeter = to_axis_frame(perimeter_cad)
        segment_normals = to_axis_frame(normal_cad)
        lengths = np.linalg.norm(perimeter[:, 1] - perimeter[:, 0], axis=1)
        if np.any(lengths <= 0.0):
            index = int(np.flatnonzero(lengths <= 0.0)[0])
            raise ValueError(
                f"{coordinates}:line {group[index]['_csv_line']} has a zero-length segment."
            )
        extent = float(np.max(np.ptp(perimeter.reshape(-1, 3), axis=0)))
        continuity_tolerance = max(1.0e-12, 1.0e-6 * max(
            extent, float(np.max(lengths))
        ))
        for index in range(len(perimeter) - 1):
            gap = float(np.linalg.norm(perimeter[index, 1] - perimeter[index + 1, 0]))
            if gap > continuity_tolerance:
                raise ValueError(
                    f"{coordinates}: lines {group[index]['_csv_line']} and "
                    f"{group[index + 1]['_csv_line']} of line_id {line_id!r} are "
                    f"not head-to-tail (gap {gap:.3e} m)."
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
                f"{coordinates}: line_id {line_id!r} is {offset*1e3:.3f} mm "
                f"off the skin ({720.0*offset/wavelength:.1f} deg two-way phase); "
                f"allowed {limit*1e3:.3f} mm."
            )
        normal_parameter = np.linspace(0.0, 1.0, 33)
        normal_points = (
            perimeter[:, 0, None, :] * (1.0 - normal_parameter)[None, :, None]
            + perimeter[:, 1, None, :] * normal_parameter[None, :, None]
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
            segment_index, sample_index = divmod(flat_index, len(normal_parameter))
            raise ValueError(
                f"{coordinates}:line {group[segment_index]['_csv_line']} supplied "
                f"normal interpolation differs from the outward skin normal by "
                f"{differences[flat_index]:.2f} deg at segment fraction "
                f"{normal_parameter[sample_index]:.5g}."
            )
        placements.append({
            "delta": str(dataset), "perimeter": perimeter,
            "segment_normals": segment_normals, "kind": "delta",
            "declared_coherent_delta": True, "delta_sign": 1.0,
        })
        records.append({
            "schema": LINE_PLACEMENT_SCHEMA,
            "kind": "line_2d_delta", "dataset": str(dataset),
            "dataset_sha256": dataset_sha256[dataset_id],
            "line_id": line_id, "dataset_id": dataset_id,
            "segment_count": len(group), "coordinates": str(coordinates),
            "first_csv_line": group[0]["_csv_line"],
            "last_csv_line": group[-1]["_csv_line"],
            "coordinates_sha256": coordinates_sha256,
            "max_skin_offset_m": float(offset),
            "max_normal_error_deg": float(np.max(differences)),
            "input_subtraction_order": "OPN-FRD (featured-clean)",
            "normal_source": "csv_endpoint_interpolation",
        })
        start = end
    return placements, records


def _compact_points(profile, surface, scale, limit, wavelength):
    if POINT_FEATURE_LOCATIONS_CSV is None:
        if POINT_FEATURE_DATASETS:
            raise ValueError(
                "POINT_FEATURE_DATASETS is configured but "
                "POINT_FEATURE_LOCATIONS_CSV is None."
            )
        return [], []
    if not POINT_FEATURE_DATASETS:
        raise ValueError(
            "POINT_FEATURE_LOCATIONS_CSV is configured but "
            "POINT_FEATURE_DATASETS is empty."
        )
    normal_tolerance = _normal_tolerance()
    normal_fn = (
        surface_of_revolution_normal(profile)
        if surface is None else surface.normal
    )
    coordinates = _path(POINT_FEATURE_LOCATIONS_CSV)
    rows = _placement_rows(coordinates)
    from workflow_provenance import sha256_file
    coordinates_sha256 = sha256_file(str(coordinates))
    dataset_paths = {}
    dataset_sha256 = {}
    for dataset_id, value in POINT_FEATURE_DATASETS.items():
        if not isinstance(dataset_id, str):
            raise ValueError("POINT_FEATURE_DATASETS keys must be strings.")
        identifier = dataset_id
        if not identifier or identifier != identifier.strip():
            raise ValueError(
                "POINT_FEATURE_DATASETS keys must be nonempty strings "
                "without leading/trailing whitespace."
            )
        dataset = _path(value)
        if not dataset.is_file():
            raise FileNotFoundError(
                f"Point dataset {identifier!r} does not exist: {dataset}"
            )
        dataset_paths[identifier] = dataset
        dataset_sha256[identifier] = sha256_file(str(dataset))
    unknown = sorted({row["dataset_id"] for row in rows} - set(dataset_paths))
    if unknown:
        raise ValueError(
            f"{coordinates}: unknown dataset_id value(s) {unknown}; configured "
            f"IDs are {sorted(dataset_paths)}."
        )

    points = []
    records = []
    patterns = {}
    for row_index, row in enumerate(rows, 1):
        csv_line = row["_csv_line"]
        dataset_id = row["dataset_id"]
        dataset = dataset_paths[dataset_id]
        if dataset_id not in patterns:
            patterns[dataset_id] = prepare_point_pattern(
                str(dataset),
                declared_coherent_delta=True,
                delta_sign=1.0,
                assume_missing_cross_pol_zero=False,
            )
        pattern = patterns[dataset_id]
        location = to_axis_frame(
            np.array([row["x"], row["y"], row["z"]]) * scale
        )
        offset = float(
            surface_of_revolution_distance(profile, location[None, :])[0]
            if surface is None else surface.distance(location[None, :])[0]
        )
        if offset > limit:
            raise ValueError(
                f"{coordinates}:line {csv_line} is {offset*1e3:.3f} mm off "
                f"the skin ({720.0*offset/wavelength:.1f} deg two-way phase)."
            )
        derived = _unit(normal_fn(location[None, :])[0], "derived normal")
        normal = _unit(
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
        roll = _unit(to_axis_frame([
            row["roll_x"], row["roll_y"], row["roll_z"]
        ]), "roll reference")
        if np.linalg.norm(roll - float(roll @ normal) * normal) <= 1e-9:
            raise ValueError(
                f"{coordinates}:line {csv_line} roll reference is parallel "
                "to the supplied normal."
            )
        points.append({
            "pattern": pattern, "location": location,
            "aperture_normal": normal, "roll_ref": roll,
        })
        records.append({
            "schema": POINT_PLACEMENT_SCHEMA,
            "kind": "compact_3d_delta", "dataset": str(dataset),
            "dataset_sha256": dataset_sha256[dataset_id],
            "placement_id": row["placement_id"],
            "dataset_id": dataset_id,
            "coordinates": str(coordinates), "row": row_index,
            "csv_line": csv_line,
            "coordinates_sha256": coordinates_sha256,
            "skin_offset_m": offset,
            "input_subtraction_order": "OPN-FRD (featured-clean)",
            "assumed_missing_cross_pol_zero": False,
            "roll_reference": "csv",
        })
    return points, records


def main():
    base = _path(BASE_MONOSTATIC_GRIM)
    output = _path(OUTPUT_MONOSTATIC_GRIM)
    if LINE_FEATURE_LOCATIONS_CSV is None and POINT_FEATURE_LOCATIONS_CSV is None:
        raise SystemExit(
            "Configure LINE_FEATURE_LOCATIONS_CSV or POINT_FEATURE_LOCATIONS_CSV."
        )
    if not base.is_file():
        raise SystemExit(f"Base monostatic GRIM not found: {base}")
    embedded_grid = load_body_requested_radar_grid(str(base))
    profile = None
    if embedded_grid is not None:
        profile = load_body_profile_grim(str(base))
        grid = embedded_grid
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

    surface = None
    surface_path = None
    if SURFACE_MESH:
        surface_path = _path(SURFACE_MESH)
        if not surface_path.is_file():
            raise SystemExit(f"Surface mesh not found: {surface_path}")
        triangles = to_axis_frame(
            read_surface_mesh(str(surface_path)) * scale_for(SURFACE_UNITS)
        )
        surface = TriangleSurface(
            triangles, flip_normals=FLIP_SURFACE_NORMALS
        )
    elif embedded_grid is None:
        raise SystemExit(
            "A non-BoR base requires SURFACE_MESH=.facet or .stl for skin "
            "validation and outward normals."
        )
    if SHADOW and surface is None:
        raise SystemExit("SHADOW=True requires SURFACE_MESH.")

    scale = scale_for(COORDINATE_UNITS)
    limit, wavelength = _skin_limit(grid["frequencies_ghz"])
    lines, line_records = _line_placements(
        profile, surface, scale, limit, wavelength
    )
    points, point_records = _compact_points(
        profile, surface, scale, limit, wavelength
    )
    normal_fn = (
        surface.normal if surface is not None
        else surface_of_revolution_normal(profile)
    )
    occluder = None
    if SHADOW:
        occluder = Occluder(surface.triangles, bias=SHADOW_BIAS_M)
        print(
            f"Geometric shadowing enabled: {len(surface.triangles)} triangles, "
            f"ray bias {occluder.bias*1e3:.4g} mm"
        )
    saved = add_features_to_monostatic_grim(
        str(base), str(output), placements=lines, points=points,
        radar_grid=grid,
        surface_normal_fn=normal_fn,
        occluder=occluder,
        declared_coherent_base=True,
        feature_provenance={
            "coordinate_units": COORDINATE_UNITS,
            "surface_mesh": None if surface_path is None else str(surface_path),
            "surface_units": None if surface_path is None else SURFACE_UNITS,
            "surface_normals_flipped": bool(FLIP_SURFACE_NORMALS),
            "shadow": bool(SHADOW),
            "shadow_bias_m": None if occluder is None else float(occluder.bias),
            "placements": line_records + point_records,
        },
        history="place_features.py coherent platform line/compact placement",
    )
    print(f"Wrote one combined monostatic dataset: {saved}")


if __name__ == "__main__":
    main()
