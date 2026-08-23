#!/usr/bin/env python3
"""
Measure what the far-pair quadrature override costs in accuracy and buys in time.

`GHOST_FAR_QUAD_ORDER` (equivalently `rcs_solver.set_far_quadrature_order`)
lowers the Gauss order used for element pairs that are separated by at least
`far_ratio` element lengths.  Those integrands are smooth, so the default order
8 -- inherited from the near-field rule, where it is needed -- is heavily
over-resolved there, and the Hankel evaluations that dominate a large assembly
scale as the square of the order.

This is a real change to computed values, not a faster evaluation of the same
rule, so it is off by default.  Run this on YOUR geometries and frequencies
before turning it on, and treat the reported dB shift as the error budget you
are accepting.  The near-field and singular quadrature are untouched, and the
mesh-convergence certificate still certifies the discretization -- not this
choice.

Usage:
    python tests/measure_far_quadrature.py [geometry.geo ...]
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

ORDERS = [8, 6, 5, 4, 3]
DEFAULT_CASES = [
    (REPO / "geometries" / "body.geo", 6.0, "TM"),
    (REPO / "geometries" / "body.geo", 18.0, "TM"),
    (REPO / "geometries" / "body.geo", 40.0, "TM"),
]
ANGLES = list(np.linspace(0.0, 180.0, 19))


def solve(geo, freq, pol, order):
    import rcs_solver
    from geometry_io import parse_geometry, build_geometry_snapshot

    rcs_solver.set_far_quadrature_order(0 if order == 8 else order)
    title, segments, ibcs, dielectrics = parse_geometry(Path(geo).read_text())
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snapshot["source_path"] = str(geo)
    started = time.time()
    result = rcs_solver.solve_monostatic_rcs_2d_single_polarization(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(freq)],
        elevations_deg=ANGLES,
        polarization=pol,
        geometry_units="meters",
        material_base_dir=str(Path(geo).parent),
    )
    rcs_solver.set_far_quadrature_order(0)
    return result, time.time() - started


def main():
    cases = DEFAULT_CASES
    if len(sys.argv) > 1:
        cases = [(Path(a), 10.0, "TM") for a in sys.argv[1:]]

    print("Far-pair quadrature order sweep (order 8 is the shipped default)\n")
    print(f"{'case':<34} {'order':>5} {'panels':>7} {'solve s':>8} {'speedup':>8} "
          f"{'max dB':>9} {'rms dB':>9}")
    for geo, freq, pol in cases:
        if not Path(geo).is_file():
            print(f"  [skip] {geo}")
            continue
        label = f"{Path(geo).name} {freq:g}GHz {pol}"
        reference = None
        base_time = None
        for order in ORDERS:
            try:
                result, elapsed = solve(geo, freq, pol, order)
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<34} {order:5d} raised {type(exc).__name__}: {exc}")
                continue
            db = np.asarray([float(s["rcs_db"]) for s in result["samples"]])
            panels = int(result.get("metadata", {}).get("panel_count", 0) or 0)
            if reference is None:
                reference = db
                base_time = elapsed
                print(f"{label:<34} {order:5d} {panels:7d} {elapsed:8.2f} "
                      f"{1.0:7.2f}x {0.0:9.2e} {0.0:9.2e}")
                continue
            diff = db - reference
            print(f"{label:<34} {order:5d} {panels:7d} {elapsed:8.2f} "
                  f"{base_time / max(elapsed, 1e-9):7.2f}x "
                  f"{np.max(np.abs(diff)):9.2e} "
                  f"{math.sqrt(float(np.mean(diff ** 2))):9.2e}")
        print()

    print("Read the dB columns as the error you would be accepting, not as a")
    print("bound: they are measured on these geometries at these frequencies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
