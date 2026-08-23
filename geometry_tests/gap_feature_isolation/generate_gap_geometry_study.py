#!/usr/bin/env python3
"""Generate matched FRD/OPN rounded coupons for a PEC-groove study.

The default sweep covers 1--15 GHz and three target minimum clearances so the
complex OPN-FRD response can be checked for remote-boundary convergence:

    python geometry_tests/gap_feature_isolation/generate_gap_geometry_study.py

Override the electrical spacings explicitly when extending the study:

    python geometry_tests/gap_feature_isolation/generate_gap_geometry_study.py \
        --clearance-lambda 3.75 4.5 5.25

The flat illuminated face transitions into a faceted cubic-Bezier backing.
That D-backed contour has rounded, outward shoulders and no parallel flat back.
The envelope is enlarged until every backing chord is at least the requested
distance from the groove walls and floor. Its geometric facets are refined
automatically to a maximum chord length of lambda/20 by default.
"""

import argparse
import math
from pathlib import Path


C0_M_PER_S = 299_792_458.0
INCHES_PER_METER = 39.37007874015748

DEFAULT_FREQUENCIES_GHZ = tuple(float(value) for value in range(1, 16))
DEFAULT_CLEARANCES_LAMBDA = (3.5, 4.25, 5.0)
GAP_WIDTH_IN = 0.5
GAP_DEPTH_IN = 1.0
PANELS_PER_WAVELENGTH = 100
MIN_BACKING_SEGMENTS = 48
MAX_BACKING_SEGMENTS = 8192
DEFAULT_MAX_BACKING_CHORD_LAMBDA = 0.05
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "generated"

# Cubic-Bezier shape controls. The binary clearance solve below, rather than
# either of these aesthetic controls, enforces the requested physical spacing.
SHOULDER_BULGE_CLEARANCE_FRACTION = 0.25
LOWER_CONTROL_X_FRACTION = 0.55


def _number(value):
    value = 0.0 if abs(float(value)) < 5.0e-13 else float(value)
    return f"{value:.12g}"


def _bezier(p0, p1, p2, p3, parameter):
    one_minus = 1.0 - parameter
    weights = (
        one_minus ** 3,
        3.0 * one_minus ** 2 * parameter,
        3.0 * one_minus * parameter ** 2,
        parameter ** 3,
    )
    return tuple(
        sum(
            weight * point[axis]
            for weight, point in zip(weights, (p0, p1, p2, p3))
        )
        for axis in (0, 1)
    )


def _backing_points(clearance_in, envelope_scale, segment_count):
    """Return the right-shoulder -> bottom -> left-shoulder backing."""
    half_gap = 0.5 * GAP_WIDTH_IN
    top_half_width = half_gap + envelope_scale * clearance_in
    bottom_depth = GAP_DEPTH_IN + envelope_scale * clearance_in
    shoulder_bulge = SHOULDER_BULGE_CLEARANCE_FRACTION * clearance_in

    p0 = (top_half_width, 0.0)
    p1 = (top_half_width + shoulder_bulge, 0.0)
    p2 = (LOWER_CONTROL_X_FRACTION * top_half_width, -bottom_depth)
    p3 = (0.0, -bottom_depth)

    half_segments = segment_count // 2
    right = [
        _bezier(p0, p1, p2, p3, index / half_segments)
        for index in range(half_segments + 1)
    ]
    # Mirror the right half from bottom to the left shoulder. Exclude the
    # center-bottom point so the two halves do not create a zero-length edge.
    left = [(-x, y) for x, y in reversed(right[:-1])]
    return right + left


def _point_segment_distance(point, start, end):
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    wx = point[0] - start[0]
    wy = point[1] - start[1]
    length_squared = vx * vx + vy * vy
    if length_squared == 0.0:
        return math.hypot(wx, wy)
    parameter = max(0.0, min(1.0, (wx * vx + wy * vy) / length_squared))
    return math.hypot(
        point[0] - (start[0] + parameter * vx),
        point[1] - (start[1] + parameter * vy),
    )


def _orientation(a, b, c):
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def _segments_cross(a, b, c, d):
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return (
        ((ab_c > 0.0 and ab_d < 0.0) or (ab_c < 0.0 and ab_d > 0.0))
        and ((cd_a > 0.0 and cd_b < 0.0) or (cd_a < 0.0 and cd_b > 0.0))
    )


def _segment_distance(a, b, c, d):
    if _segments_cross(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _minimum_feature_clearance(backing):
    half_gap = 0.5 * GAP_WIDTH_IN
    feature_edges = (
        ((-half_gap, 0.0), (-half_gap, -GAP_DEPTH_IN)),
        ((-half_gap, -GAP_DEPTH_IN), (half_gap, -GAP_DEPTH_IN)),
        ((half_gap, -GAP_DEPTH_IN), (half_gap, 0.0)),
    )
    return min(
        _segment_distance(a, b, c, d)
        for a, b in zip(backing[:-1], backing[1:])
        for c, d in feature_edges
    )


def _clearance_backing(clearance_in, segment_count):
    """Size the rounded backing to guarantee the requested minimum spacing."""
    target = clearance_in * (1.0 + 1.0e-8)
    lower = 1.0
    upper = 1.0
    while _minimum_feature_clearance(
        _backing_points(clearance_in, upper, segment_count)
    ) < target:
        upper *= 1.25
        if upper > 10.0:
            raise RuntimeError("failed to construct a clearance-safe backing")

    for _ in range(60):
        middle = 0.5 * (lower + upper)
        backing = _backing_points(clearance_in, middle, segment_count)
        if _minimum_feature_clearance(backing) >= target:
            upper = middle
        else:
            lower = middle

    backing = _backing_points(clearance_in, upper, segment_count)
    return backing, _minimum_feature_clearance(backing), upper


def _maximum_chord_length(backing):
    return max(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(backing[:-1], backing[1:])
    )


def _resolved_backing(
    clearance_in,
    wavelength_in,
    max_chord_lambda,
    fixed_segment_count=None,
):
    """Build a clearance-safe contour with wavelength-limited facets."""
    if fixed_segment_count is not None:
        backing, actual_clearance, envelope_scale = _clearance_backing(
            clearance_in, fixed_segment_count
        )
        return (
            backing,
            actual_clearance,
            envelope_scale,
            fixed_segment_count,
            _maximum_chord_length(backing),
        )

    target_chord = max_chord_lambda * wavelength_in
    segment_count = MIN_BACKING_SEGMENTS
    for _ in range(20):
        backing, actual_clearance, envelope_scale = _clearance_backing(
            clearance_in, segment_count
        )
        maximum_chord = _maximum_chord_length(backing)
        if maximum_chord <= target_chord * (1.0 + 1.0e-12):
            return (
                backing,
                actual_clearance,
                envelope_scale,
                segment_count,
                maximum_chord,
            )

        estimated = int(math.ceil(
            segment_count * maximum_chord / target_chord * 1.01
        ))
        segment_count = max(segment_count + 2, estimated)
        if segment_count % 2:
            segment_count += 1
        if segment_count > MAX_BACKING_SEGMENTS:
            raise RuntimeError(
                "automatic backing refinement exceeded "
                f"{MAX_BACKING_SEGMENTS} segments"
            )

    raise RuntimeError("automatic backing refinement did not converge")


def _geo_text(
    role,
    frequency_ghz,
    clearance_lambda,
    wavelength_in,
    backing,
    actual_clearance_in,
    envelope_scale,
    backing_segments,
    maximum_chord_in,
):
    half_gap = 0.5 * GAP_WIDTH_IN
    top_half_width = backing[0][0]

    if role == "FRD":
        # Identical primitive endpoints at +/- half_gap make the common top
        # skin discretize consistently with the OPN member.
        points = [
            (-top_half_width, 0.0),
            (-half_gap, 0.0),
            (half_gap, 0.0),
        ] + backing
    elif role == "OPN":
        points = [
            (-top_half_width, 0.0),
            (-half_gap, 0.0),
            (-half_gap, -GAP_DEPTH_IN),
            (half_gap, -GAP_DEPTH_IN),
            (half_gap, 0.0),
        ] + backing
    else:
        raise ValueError("role must be FRD or OPN")

    actual_clearance_lambda = actual_clearance_in / wavelength_in
    maximum_chord_lambda = maximum_chord_in / wavelength_in
    lines = [
        (
            f"Title: gap_{frequency_ghz:.3f}GHz_{role} units=inches "
            f"backing=rounded_bezier "
            f"target_clearance={clearance_lambda:g}lambda "
            f"actual_min_clearance={actual_clearance_lambda:.8g}lambda "
            f"backing_segments={backing_segments} "
            f"max_backing_chord={maximum_chord_lambda:.8g}lambda "
            f"envelope_scale={envelope_scale:.8g}"
        ),
        f"Segment: gap_coupon_{role.lower()} 2",
        f"properties: 2 {-PANELS_PER_WAVELENGTH} 0 0 0",
    ]
    lines.extend(
        " ".join(_number(value) for point in (p0, p1) for value in point)
        for p0, p1 in zip(points[:-1], points[1:])
    )
    lines.extend(["", "IBCS_Resistances:", "Dielectrics:", ""])
    return "\n".join(lines)


def _positive_unique(values, label):
    result = []
    for raw_value in values:
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} values must be finite and positive")
        if value not in result:
            result.append(value)
    return result


def _clearance_tag(clearance_lambda):
    return f"C{clearance_lambda:05.3f}lambda".replace(".", "p")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequencies-ghz",
        nargs="+",
        type=float,
        default=DEFAULT_FREQUENCIES_GHZ,
        help="design frequencies in GHz (default: 1 through 15)",
    )
    parser.add_argument(
        "--clearance-lambda",
        nargs="+",
        type=float,
        default=DEFAULT_CLEARANCES_LAMBDA,
        help="target minimum backing clearances in wavelengths",
    )
    parser.add_argument(
        "--backing-segments",
        type=int,
        default=None,
        help=(
            "fixed even number of backing chords; overrides automatic "
            "wavelength-based refinement"
        ),
    )
    parser.add_argument(
        "--max-backing-chord-lambda",
        type=float,
        default=DEFAULT_MAX_BACKING_CHORD_LAMBDA,
        help="maximum backing chord in wavelengths (default: 0.05)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="output directory (default: the study's generated folder)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    frequencies_ghz = _positive_unique(args.frequencies_ghz, "frequency")
    clearances_lambda = _positive_unique(
        args.clearance_lambda, "clearance-lambda"
    )
    if GAP_WIDTH_IN <= 0.0 or GAP_DEPTH_IN <= 0.0:
        raise ValueError("GAP_WIDTH_IN and GAP_DEPTH_IN must be positive")
    if args.backing_segments is not None and (
        args.backing_segments < 8 or args.backing_segments % 2
    ):
        raise ValueError("backing-segments must be an even integer >= 8")
    if (
        not math.isfinite(args.max_backing_chord_lambda)
        or args.max_backing_chord_lambda <= 0.0
    ):
        raise ValueError("max-backing-chord-lambda must be finite and positive")
    if PANELS_PER_WAVELENGTH < 1:
        raise ValueError("PANELS_PER_WAVELENGTH must be positive")

    output_root = args.output_root.resolve()
    written = []
    for frequency_ghz in frequencies_ghz:
        wavelength_in = (
            C0_M_PER_S / (frequency_ghz * 1.0e9) * INCHES_PER_METER
        )
        frequency_dir = output_root / f"{frequency_ghz:06.3f}GHz"
        for clearance_lambda in clearances_lambda:
            clearance_in = clearance_lambda * wavelength_in
            (
                backing,
                actual_clearance_in,
                envelope_scale,
                backing_segments,
                maximum_chord_in,
            ) = _resolved_backing(
                clearance_in,
                wavelength_in,
                args.max_backing_chord_lambda,
                fixed_segment_count=args.backing_segments,
            )
            tag = _clearance_tag(clearance_lambda)
            for role in ("FRD", "OPN"):
                output_dir = frequency_dir / role / tag
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / (
                    f"gap_{tag}_{frequency_ghz:06.3f}GHz_{role}.geo"
                )
                path.write_text(
                    _geo_text(
                        role,
                        frequency_ghz,
                        clearance_lambda,
                        wavelength_in,
                        backing,
                        actual_clearance_in,
                        envelope_scale,
                        backing_segments,
                        maximum_chord_in,
                    ),
                    encoding="ascii",
                )
                written.append(path)
            print(
                f"{frequency_ghz:6.3f} GHz, {clearance_lambda:g} lambda: "
                f"lambda={wavelength_in:.6g} in, "
                f"minimum clearance={actual_clearance_in:.6g} in, "
                f"backing segments={backing_segments}, "
                f"max chord={maximum_chord_in / wavelength_in:.6g} lambda, "
                f"envelope scale={envelope_scale:.6g}"
            )

    print(f"Wrote {len(written)} files under {output_root}")


if __name__ == "__main__":
    main()
