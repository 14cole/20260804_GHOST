#!/usr/bin/env python3
"""Generate matched FRD/OPN PEC groove coupons for a frequency sweep.

Defaults reproduce the requested 1-15 GHz study in inches.  For every design
frequency, the 0.5-inch-wide by 1-inch-deep groove has four free-space
wavelengths of clean PEC between each aperture edge and the coupon side, and
four wavelengths between the groove floor and the coupon bottom.

Run from anywhere:

    python geometry_tests/gap_feature_isolation/generate_gap_geometry_study.py
"""

from pathlib import Path


C0_M_PER_S = 299_792_458.0
INCHES_PER_METER = 39.37007874015748

FREQUENCIES_GHZ = range(1, 16)
GAP_WIDTH_IN = 0.5
GAP_DEPTH_IN = 1.0
CLEARANCE_WAVELENGTHS = 4.0
PANELS_PER_WAVELENGTH = 100
OUTPUT_ROOT = Path(__file__).resolve().parent / "generated"


def _number(value):
    value = 0.0 if abs(float(value)) < 5.0e-13 else float(value)
    return f"{value:.12g}"


def _geo_text(role, frequency_ghz, wavelength_in):
    half_gap = 0.5 * GAP_WIDTH_IN
    clearance = CLEARANCE_WAVELENGTHS * wavelength_in
    side = half_gap + clearance
    bottom = -(GAP_DEPTH_IN + clearance)

    if role == "FRD":
        # The three top primitives deliberately match the OPN aperture
        # breakpoints so common-skin discretization cancels cleanly.
        points = [
            (-side, 0.0),
            (-half_gap, 0.0),
            (half_gap, 0.0),
            (side, 0.0),
            (side, bottom),
            (-side, bottom),
            (-side, 0.0),
        ]
    elif role == "OPN":
        points = [
            (-side, 0.0),
            (-half_gap, 0.0),
            (-half_gap, -GAP_DEPTH_IN),
            (half_gap, -GAP_DEPTH_IN),
            (half_gap, 0.0),
            (side, 0.0),
            (side, bottom),
            (-side, bottom),
            (-side, 0.0),
        ]
    else:
        raise ValueError("role must be FRD or OPN")

    lines = [
        (
            f"Title: gap_{frequency_ghz:.3f}GHz_{role} "
            f"units=inches clearance={CLEARANCE_WAVELENGTHS:g}lambda"
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


def main():
    if GAP_WIDTH_IN <= 0.0 or GAP_DEPTH_IN <= 0.0:
        raise ValueError("GAP_WIDTH_IN and GAP_DEPTH_IN must be positive")
    if CLEARANCE_WAVELENGTHS <= 0.0 or PANELS_PER_WAVELENGTH < 1:
        raise ValueError("clearance and panels per wavelength must be positive")

    written = []
    for raw_frequency in FREQUENCIES_GHZ:
        frequency_ghz = float(raw_frequency)
        if frequency_ghz <= 0.0:
            raise ValueError("all frequencies must be positive")
        wavelength_in = (
            C0_M_PER_S / (frequency_ghz * 1.0e9) * INCHES_PER_METER
        )
        frequency_dir = OUTPUT_ROOT / f"{frequency_ghz:06.3f}GHz"
        for role in ("FRD", "OPN"):
            output_dir = frequency_dir / role
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"gap_{frequency_ghz:06.3f}GHz_{role}.geo"
            path.write_text(
                _geo_text(role, frequency_ghz, wavelength_in),
                encoding="ascii",
            )
            written.append(path)
        print(
            f"{frequency_ghz:6.3f} GHz: lambda={wavelength_in:.6g} in, "
            f"clearance={CLEARANCE_WAVELENGTHS * wavelength_in:.6g} in"
        )

    print(f"Wrote {len(written)} files under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
