#!/usr/bin/env python3
"""Coherently place doors, seams, and compact 3-D features on a BoR result.

Edit only the USER SETTINGS block, then run:

    python add_bor_features.py

The input and output use the same monostatic GRIM schema.  Line datasets must
be coherent 2-D ``featured - clean`` delta GRIMs.  Compact datasets must be
calibrated installed-feature-minus-clean-skin 3-D patterns; using a standalone
cavity field would double-count the unbroken body skin and omit installation
coupling.
"""

import csv
import math
from pathlib import Path

import numpy as np

# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_MONOSTATIC_GRIM = "rcs_runs_bor/run_x/results/body.grim"
OUTPUT_MONOSTATIC_GRIM = "rcs_runs_bor/run_x/results/body_with_features.grim"
COORDINATE_UNITS = "inches"

# Perimeter text rows: x1 y1 z1 x2 y2 z2 in the CAD frame
# (+y nose, +x right, +z up). The dataset must be a 2-D coherent delta.
LINE_FEATURES = [
    # {"dataset": "door_delta.grim", "coordinates": "door_perimeter.txt"},
]

# Placement CSV rows: x,y,z and optional nx,ny,nz and rx,ry,rz. If the normal
# is omitted it is derived from the BoR skin. r* sets pattern clocking.
COMPACT_FEATURES = [
    # {"dataset": "cavity_delta.grim", "coordinates": "cavities.csv"},
]

# Coordinate-to-skin validation.  The tighter of the distance and two-way
# phase limits is enforced at the highest frequency in the body file.
SKIN_TOL_M = 1.0e-3
SKIN_PHASE_TOL_DEG = 15.0
NORMAL_TOL_DEG = 15.0
DEFAULT_ROLL_REF_CAD = (0.0, 0.0, 1.0)

# =============================================================================

from feature_sum import (  # noqa: E402
    add_features_to_monostatic_grim,
    load_body_profile_grim,
    load_body_requested_radar_grid,
    prepare_point_pattern,
    surface_of_revolution_distance,
)
from frame import scale_for, to_axis_frame  # noqa: E402
from line_expand import (  # noqa: E402
    C0,
    perimeter_surface_deviation,
    read_perimeter_txt,
    surface_of_revolution_normal,
)


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


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


def _placement_rows(path):
    """Read headered or headerless x,y,z[,nx,ny,nz[,rx,ry,rz]] CSV."""
    with _path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(stream)
            if row and any(cell.strip() for cell in row)
            and not row[0].lstrip().startswith("#")
        ]
    if not rows:
        raise ValueError(f"{path}: placement CSV is empty.")
    try:
        [float(value) for value in rows[0]]
        keys = ("x", "y", "z", "nx", "ny", "nz", "rx", "ry", "rz")[:len(rows[0])]
    except ValueError:
        keys = tuple(value.strip().lower() for value in rows.pop(0))
    if len(keys) not in (3, 6, 9) or tuple(keys[:3]) != ("x", "y", "z"):
        raise ValueError(
            f"{path}: columns must be x,y,z with optional nx,ny,nz and rx,ry,rz."
        )
    expected = set(keys)
    for group in ({"nx", "ny", "nz"}, {"rx", "ry", "rz"}):
        if expected & group and not group <= expected:
            raise ValueError(f"{path}: optional vector columns must be complete.")
    parsed = []
    for number, row in enumerate(rows, 2):
        if len(row) != len(keys):
            raise ValueError(f"{path}:{number}: expected {len(keys)} columns.")
        values = dict(zip(keys, (float(value) for value in row)))
        if not np.all(np.isfinite(list(values.values()))):
            raise ValueError(f"{path}:{number}: NaN/infinite coordinate.")
        parsed.append(values)
    return parsed


def _line_placements(profile, scale, limit, wavelength):
    placements = []
    records = []
    for specification in LINE_FEATURES:
        dataset = _path(specification["dataset"])
        coordinates = _path(specification["coordinates"])
        perimeter = to_axis_frame(read_perimeter_txt(str(coordinates), scale=scale))
        offset = perimeter_surface_deviation(perimeter, profile, samples_per_segment=33)
        if offset > limit:
            raise ValueError(
                f"{coordinates}: perimeter is {offset*1e3:.3f} mm off the skin "
                f"({720.0*offset/wavelength:.1f} deg two-way phase); allowed "
                f"{limit*1e3:.3f} mm."
            )
        placements.append({"delta": str(dataset), "perimeter": perimeter, "kind": "delta"})
        records.append({
            "kind": "line_delta", "dataset": str(dataset),
            "coordinates": str(coordinates), "max_skin_offset_m": float(offset),
        })
    return placements, records


def _compact_points(profile, scale, limit, wavelength):
    normal_fn = surface_of_revolution_normal(profile)
    points = []
    records = []
    for specification in COMPACT_FEATURES:
        dataset = _path(specification["dataset"])
        coordinates = _path(specification["coordinates"])
        pattern = prepare_point_pattern(str(dataset))
        for row_index, row in enumerate(_placement_rows(coordinates), 1):
            location = to_axis_frame(np.array([row["x"], row["y"], row["z"]]) * scale)
            offset = float(surface_of_revolution_distance(profile, location[None, :])[0])
            if offset > limit:
                raise ValueError(
                    f"{coordinates}:row {row_index} is {offset*1e3:.3f} mm off "
                    f"the skin ({720.0*offset/wavelength:.1f} deg two-way phase)."
                )
            derived = _unit(normal_fn(location[None, :])[0], "derived normal")
            if "nx" in row:
                normal = _unit(
                    to_axis_frame([row["nx"], row["ny"], row["nz"]]),
                    "supplied normal",
                )
                difference = math.degrees(math.acos(np.clip(float(normal @ derived), -1.0, 1.0)))
                if difference > float(NORMAL_TOL_DEG):
                    raise ValueError(
                        f"{coordinates}:row {row_index} supplied normal differs "
                        f"from the outward skin normal by {difference:.2f} deg."
                    )
            else:
                normal = derived
            roll_cad = (
                [row["rx"], row["ry"], row["rz"]]
                if "rx" in row else DEFAULT_ROLL_REF_CAD
            )
            roll = _unit(to_axis_frame(roll_cad), "roll reference")
            if np.linalg.norm(roll - float(roll @ normal) * normal) <= 1e-9:
                raise ValueError(f"{coordinates}:row {row_index} roll reference is parallel to normal.")
            points.append({
                "pattern": pattern, "location": location,
                "aperture_normal": normal, "roll_ref": roll,
            })
            records.append({
                "kind": "compact_3d_delta", "dataset": str(dataset),
                "coordinates": str(coordinates), "row": row_index,
                "skin_offset_m": offset,
            })
    return points, records


def main():
    base = _path(BASE_MONOSTATIC_GRIM)
    output = _path(OUTPUT_MONOSTATIC_GRIM)
    if not LINE_FEATURES and not COMPACT_FEATURES:
        raise SystemExit("Configure at least one LINE_FEATURES or COMPACT_FEATURES entry.")
    profile = load_body_profile_grim(str(base))
    grid = load_body_requested_radar_grid(str(base))
    if grid is None:
        raise SystemExit("Base file is not a current self-contained BoR monostatic GRIM.")
    scale = scale_for(COORDINATE_UNITS)
    limit, wavelength = _skin_limit(grid["frequencies_ghz"])
    lines, line_records = _line_placements(profile, scale, limit, wavelength)
    points, point_records = _compact_points(profile, scale, limit, wavelength)
    saved = add_features_to_monostatic_grim(
        str(base), str(output), placements=lines, points=points,
        feature_provenance={
            "coordinate_units": COORDINATE_UNITS,
            "placements": line_records + point_records,
        },
        history="add_bor_features.py coherent line/compact placement",
    )
    print(f"Wrote one combined monostatic dataset: {saved}")


if __name__ == "__main__":
    main()
