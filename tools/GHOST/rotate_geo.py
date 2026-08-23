#!/usr/bin/env python3
"""Rotate a GHOST .geo file using hard-coded settings.

Edit only the USER SETTINGS block, then run this file with no arguments:

    python3 rotate_geo.py

Positive angles rotate counterclockwise in the geometry x-y plane.  For a BoR
geometry the transformed coordinates must use x = rho >= 0 and y = z, with the
rotation axis at x = 0.  Rotation does not turn a full 2-D outline into a BoR
half-profile; the input must already describe one physical generatrix.
"""

import math
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Backend"))

from geometry_io import build_geometry_text, parse_geometry  # noqa: E402


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_GEO = HERE / "geometries" / "body.geo"
OUTPUT_GEO = HERE / "geometries" / "body_rotated.geo"

# Positive = counterclockwise in the x-y plane.
ROTATION_DEG = 0.0

# Point about which the geometry rotates.  Use the known center or reference
# point of the source geometry; this is not automatically inferred.
ROTATION_CENTER_X = 0.0
ROTATION_CENTER_Y = 0.0

# Optional rigid translation after rotation.  This is useful for moving the
# intended rotation axis onto x = 0 after orienting the profile.
TRANSLATE_X = 0.0
TRANSLATE_Y = 0.0

# BoR safety checks.  Disable only when using this as a general 2-D rotation
# utility.  A closed BoR body needs at least two distinct points on x = 0 and
# may not contain any point at x < 0.
VALIDATE_BOR_HALF_PROFILE = True
AXIS_TOLERANCE = 1.0e-10

# A valid BoR exterior must be drawn from its +z axis endpoint toward its -z
# axis endpoint.  If an already connected profile is backwards, setting this
# true reverses every segment's primitive order and endpoint direction.
# CAUTION: this also reverses the physical direction of any impedance taper.
REVERSE_EACH_SEGMENT = False

# =============================================================================


def _rotate_point(x, y, angle_deg, center, translation):
    """Rotate one point counterclockwise, then translate it."""

    angle = math.radians(float(angle_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    cx, cy = (float(center[0]), float(center[1]))
    tx, ty = (float(translation[0]), float(translation[1]))
    dx, dy = float(x) - cx, float(y) - cy
    return (
        cx + cosine * dx - sine * dy + tx,
        cy + sine * dx + cosine * dy + ty,
    )


def _snap_zero(value, tolerance):
    value = float(value)
    return 0.0 if abs(value) <= float(tolerance) else value


def _reverse_segment(segment):
    """Reverse a segment without changing its geometric curve."""

    primitives = [
        (
            segment.x[index],
            segment.y[index],
            segment.x[index + 1],
            segment.y[index + 1],
        )
        for index in range(0, len(segment.x), 2)
    ]
    segment.x = []
    segment.y = []
    for x1, y1, x2, y2 in reversed(primitives):
        segment.x.extend([x2, x1])
        segment.y.extend([y2, y1])


def _distinct_points(segments, tolerance):
    scale = max(float(tolerance), 1.0e-15)
    points = {}
    for segment in segments:
        for x, y in zip(segment.x, segment.y):
            key = (int(round(float(x) / scale)), int(round(float(y) / scale)))
            points.setdefault(key, (float(x), float(y)))
    return list(points.values())


def validate_bor_half_profile(segments, tolerance=AXIS_TOLERANCE):
    """Apply the fundamental BoR coordinate checks to transformed segments."""

    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("AXIS_TOLERANCE must be positive and finite.")
    points = _distinct_points(segments, tolerance)
    if not points:
        raise ValueError("The geometry contains no coordinate points.")

    min_x = min(point[0] for point in points)
    if min_x < -tolerance:
        raise ValueError(
            "The rotated geometry is not a BoR half-profile: its minimum "
            f"x coordinate is {min_x:.12g}. BoR uses x = rho >= 0 and the "
            "rotation axis x = 0. A full circle centered on the axis is not "
            "valid input; retain only one semicircle."
        )

    axis_points = [point for point in points if abs(point[0]) <= tolerance]
    if len(axis_points) < 2:
        raise ValueError(
            "The rotated geometry has fewer than two distinct points on "
            "x = 0. A closed BoR generatrix must begin and end on the "
            "rotation axis; adjust ROTATION_CENTER_* and TRANSLATE_X."
        )

    axis_y = [point[1] for point in axis_points]
    return {
        "point_count": len(points),
        "min_x": min_x,
        "max_x": max(point[0] for point in points),
        "min_y": min(point[1] for point in points),
        "max_y": max(point[1] for point in points),
        "axis_point_count": len(axis_points),
        "axis_y_min": min(axis_y),
        "axis_y_max": max(axis_y),
    }


def rotate_geometry_file(
    input_geo,
    output_geo,
    angle_deg,
    *,
    center=(0.0, 0.0),
    translation=(0.0, 0.0),
    validate_bor=True,
    axis_tolerance=AXIS_TOLERANCE,
    reverse_each_segment=False,
):
    """Rotate one .geo while preserving segment and material definitions."""

    source = Path(input_geo).expanduser().resolve()
    destination = Path(output_geo).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input geometry does not exist: {source}")
    if source == destination:
        raise ValueError("OUTPUT_GEO must differ from INPUT_GEO.")
    values = [angle_deg, *center, *translation]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Rotation angle, center, and translation must be finite.")

    title, segments, ibcs, dielectrics = parse_geometry(
        source.read_text(encoding="utf-8")
    )
    for segment in segments:
        transformed_x = []
        transformed_y = []
        for x, y in zip(segment.x, segment.y):
            new_x, new_y = _rotate_point(
                x, y, angle_deg, center, translation
            )
            transformed_x.append(_snap_zero(new_x, axis_tolerance))
            transformed_y.append(_snap_zero(new_y, axis_tolerance))
        segment.x = transformed_x
        segment.y = transformed_y
        if reverse_each_segment:
            _reverse_segment(segment)

    report = (
        validate_bor_half_profile(segments, axis_tolerance)
        if validate_bor else None
    )
    text = build_geometry_text(
        f"{title} | rotated {float(angle_deg):g} deg",
        segments,
        ibcs,
        dielectrics,
    )
    # Parse the serialized form before replacing/creating the destination.
    parse_geometry(text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(destination))
    return destination, report


def main():
    destination, report = rotate_geometry_file(
        INPUT_GEO,
        OUTPUT_GEO,
        ROTATION_DEG,
        center=(ROTATION_CENTER_X, ROTATION_CENTER_Y),
        translation=(TRANSLATE_X, TRANSLATE_Y),
        validate_bor=VALIDATE_BOR_HALF_PROFILE,
        axis_tolerance=AXIS_TOLERANCE,
        reverse_each_segment=REVERSE_EACH_SEGMENT,
    )
    print(f"Wrote: {destination}")
    if report is not None:
        print(
            "BoR half-profile check: "
            f"x=[{report['min_x']:.12g}, {report['max_x']:.12g}], "
            f"y=[{report['min_y']:.12g}, {report['max_y']:.12g}], "
            f"axis points={report['axis_point_count']}"
        )
        print(
            "Before solving, confirm the connected exterior is traversed "
            "from its +z axis endpoint to its -z axis endpoint."
        )


if __name__ == "__main__":
    main()
