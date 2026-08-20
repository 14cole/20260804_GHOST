#!/usr/bin/env python3
"""Coherently place doors, seams, and compact features on a platform result.

Edit only the USER SETTINGS block, then run:

    python place_features.py

The base may be a self-contained GHOST BoR monostatic GRIM or an attested
external monostatic GRIM paired with an indexed ASCII ``.facet``/STL platform
surface. GUI power/phase subtraction results are accepted directly. The
feature entries declare their physical role and subtraction order; metadata
that merely repeats that declaration is optional.
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

# Perimeter text rows: x1 y1 z1 x2 y2 z2 in the CAD frame
# (+y nose, +x right, +z up). The dataset must be a 2-D coherent delta.
LINE_FEATURES = [
    # {"dataset": "door_delta.grim", "coordinates": "door_perimeter.txt",
    #  "subtraction_order": "featured-clean"},
]

# Placement CSV rows: x,y,z and optional nx,ny,nz and rx,ry,rz. If the normal
# is omitted it is derived from the platform skin. r* sets pattern clocking.
# Listing a dataset here explicitly attests that it is the coherent
# installed-feature-minus-clean-skin delta, with its origin at the aperture
# phase center, exp(+jwt), and the documented cavity-frame VV/HH/VH basis.
# This accepts a GRIM GUI subtraction that dropped those convention tags and
# whose generic container remains tagged power_phase. Incorrect placement
# declarations produce incorrect coherent phase; do not list standalone fields.
# Descriptive metadata in a GUI-derived grid is ignored in favor of this entry.
COMPACT_FEATURES = [
    # {"dataset": "cavity_delta.grim", "coordinates": "cavities.csv",
    #  "subtraction_order": "featured-clean"},
    # A locally diagonal/axisymmetric pattern exported with only VV and HH may
    # add "assume_missing_cross_pol_zero": True. This is a physical model
    # choice, not a file-format repair; do not use it for a general feature.
]

# Existing GHOST/GRIM OPN-FRD subtraction is featured-clean. If a GUI file was
# deliberately formed as FRD-OPN instead, set that feature's subtraction_order
# to "clean-featured"; the complex field (not just its displayed phase) is
# negated before placement.

# Coordinate-to-skin validation.  The tighter of the distance and two-way
# phase limits is enforced at the highest frequency in the body file.
SKIN_TOL_M = 1.0e-3
SKIN_PHASE_TOL_DEG = 15.0
NORMAL_TOL_DEG = 15.0
DEFAULT_ROLL_REF_CAD = (0.0, 0.0, 1.0)

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
    read_perimeter_txt,
    surface_of_revolution_normal,
)
from occluder import Occluder  # noqa: E402
from surface_mesh import TriangleSurface, read_surface_mesh  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _subtraction_sign(specification):
    """Convert a user-facing subtraction order to canonical featured-clean."""
    raw = str(
        specification.get("subtraction_order", "featured-clean")
    ).strip().lower().replace("_", "-").replace(" ", "")
    positive = {"featured-clean", "opn-frd", "feature-reference"}
    negative = {"clean-featured", "frd-opn", "reference-feature"}
    if raw in positive:
        return 1.0, "featured-clean"
    if raw in negative:
        return -1.0, "clean-featured"
    raise ValueError(
        "subtraction_order must be 'featured-clean'/'OPN-FRD' or "
        "'clean-featured'/'FRD-OPN'."
    )


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


def _sample_perimeter(perimeter, samples_per_segment=33):
    segments = np.asarray(perimeter, dtype=float)
    parameter = np.linspace(0.0, 1.0, max(2, int(samples_per_segment)))
    return (
        segments[:, 0, None, :] * (1.0 - parameter)[None, :, None]
        + segments[:, 1, None, :] * parameter[None, :, None]
    ).reshape(-1, 3)


def _line_placements(profile, surface, scale, limit, wavelength):
    placements = []
    records = []
    for specification in LINE_FEATURES:
        dataset = _path(specification["dataset"])
        coordinates = _path(specification["coordinates"])
        delta_sign, subtraction_order = _subtraction_sign(specification)
        perimeter = to_axis_frame(read_perimeter_txt(str(coordinates), scale=scale))
        if surface is None:
            offset = perimeter_surface_deviation(
                perimeter, profile, samples_per_segment=33
            )
        else:
            offset = float(np.max(surface.distance(_sample_perimeter(perimeter))))
        if offset > limit:
            raise ValueError(
                f"{coordinates}: perimeter is {offset*1e3:.3f} mm off the skin "
                f"({720.0*offset/wavelength:.1f} deg two-way phase); allowed "
                f"{limit*1e3:.3f} mm."
            )
        placements.append({
            "delta": str(dataset), "perimeter": perimeter, "kind": "delta",
            "declared_coherent_delta": True, "delta_sign": delta_sign,
        })
        records.append({
            "kind": "line_delta", "dataset": str(dataset),
            "coordinates": str(coordinates), "max_skin_offset_m": float(offset),
            "input_subtraction_order": subtraction_order,
        })
    return placements, records


def _compact_points(profile, surface, scale, limit, wavelength):
    normal_fn = (
        surface_of_revolution_normal(profile)
        if surface is None else surface.normal
    )
    points = []
    records = []
    for specification in COMPACT_FEATURES:
        dataset = _path(specification["dataset"])
        coordinates = _path(specification["coordinates"])
        delta_sign, subtraction_order = _subtraction_sign(specification)
        pattern = prepare_point_pattern(
            str(dataset), declared_coherent_delta=True,
            delta_sign=delta_sign,
            assume_missing_cross_pol_zero=bool(
                specification.get("assume_missing_cross_pol_zero", False)
            ),
        )
        for row_index, row in enumerate(_placement_rows(coordinates), 1):
            location = to_axis_frame(np.array([row["x"], row["y"], row["z"]]) * scale)
            offset = float(
                surface_of_revolution_distance(profile, location[None, :])[0]
                if surface is None else surface.distance(location[None, :])[0]
            )
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
            explicit_roll = "rx" in row
            roll_cad = (
                [row["rx"], row["ry"], row["rz"]]
                if explicit_roll else DEFAULT_ROLL_REF_CAD
            )
            roll = _unit(to_axis_frame(roll_cad), "roll reference")
            if np.linalg.norm(roll - float(roll @ normal) * normal) <= 1e-9:
                if explicit_roll:
                    raise ValueError(
                        f"{coordinates}:row {row_index} explicit roll "
                        "reference is parallel to the surface normal."
                    )
                # Let point_scatterer_amplitude select its stable transverse
                # fallback. This only defines clocking when the CSV omitted it.
                roll = None
            points.append({
                "pattern": pattern, "location": location,
                "aperture_normal": normal, "roll_ref": roll,
            })
            records.append({
                "kind": "compact_3d_delta", "dataset": str(dataset),
                "coordinates": str(coordinates), "row": row_index,
                "skin_offset_m": offset,
                "input_subtraction_order": subtraction_order,
                "assumed_missing_cross_pol_zero": bool(
                    specification.get("assume_missing_cross_pol_zero", False)
                ),
                "roll_reference": (
                    "automatic_transverse" if roll is None
                    else "csv" if explicit_roll else "default_cad"
                ),
            })
    return points, records


def main():
    base = _path(BASE_MONOSTATIC_GRIM)
    output = _path(OUTPUT_MONOSTATIC_GRIM)
    if not LINE_FEATURES and not COMPACT_FEATURES:
        raise SystemExit("Configure at least one LINE_FEATURES or COMPACT_FEATURES entry.")
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
