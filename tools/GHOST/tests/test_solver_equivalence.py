#!/usr/bin/env python3
"""
End-to-end equivalence check for the optimized solver.

`test_assembly_equivalence.py` compares individual operators; this compares
what the solver actually publishes -- complex backscatter amplitudes and RCS
in dB -- against a pristine copy of the pre-optimization module, across the
formulations the dispatcher can select (Robin/MFIE, sheet, dielectric,
multi-region as the shipped geometries exercise them).

Usage:
    python tests/test_solver_equivalence.py /path/to/rcs_solver_reference.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

# The optimizations reassociate floating-point sums (tiling changes the order
# in which element-pair contributions land in the global matrix), so exact
# equality is not the bar.  This is far below the solver's own quality gate
# and orders of magnitude below any physically meaningful RCS difference.
AMP_TOL = 1.0e-9
DB_TOL = 1.0e-7


def load_reference(path):
    spec = importlib.util.spec_from_file_location("rcs_solver_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rcs_solver_reference"] = module
    spec.loader.exec_module(module)
    return module


CASES = [
    ("geometries/body.geo", [3.0, 9.0], "TM", "meters"),
    ("geometries/body.geo", [3.0, 9.0], "TE", "meters"),
    ("geometries/body.geo", [18.0], "TM", "meters"),
    ("tests/fixtures/geometries/coupon_frd.geo", [2.0], "TM", "meters"),
    ("tests/fixtures/geometries/coupon_frd.geo", [4.0], "TE", "meters"),
    ("tests/fixtures/geometries/coupon_opn_017.geo", [3.0], "TM", "meters"),
]

ANGLES = list(np.linspace(0.0, 180.0, 19))


def solve(module, geo, freqs, pol, units):
    from geometry_io import parse_geometry, build_geometry_snapshot

    title, segments, ibcs, dielectrics = parse_geometry(Path(geo).read_text())
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snapshot["source_path"] = str(geo)
    started = time.time()
    result = module.solve_monostatic_rcs_2d_single_polarization(
        geometry_snapshot=snapshot,
        frequencies_ghz=list(freqs),
        elevations_deg=ANGLES,
        polarization=pol,
        geometry_units=units,
        material_base_dir=str(Path(geo).parent),
    )
    return result, time.time() - started


def amplitudes(result):
    return np.asarray([
        complex(s["rcs_amp_real"], s["rcs_amp_imag"]) for s in result["samples"]
    ])


def decibels(result):
    return np.asarray([float(s["rcs_db"]) for s in result["samples"]])


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not ref_path or not Path(ref_path).is_file():
        raise SystemExit("usage: test_solver_equivalence.py <reference rcs_solver.py>")

    import rcs_solver as new

    ref = load_reference(ref_path)
    print(f"reference: {ref_path}\n")
    print(f"{'case':<52} {'panels':>7} {'ref s':>8} {'new s':>8} "
          f"{'x':>6} {'amp err':>10} {'dB err':>9}")

    failures = []
    total_ref = 0.0
    total_new = 0.0
    for rel_geo, freqs, pol, units in CASES:
        geo = REPO / rel_geo
        if not geo.is_file():
            raise FileNotFoundError(
                f"required equivalence fixture is missing: {rel_geo}"
            )
        label = f"{Path(rel_geo).name} {freqs} {pol}"
        try:
            ref_result, ref_time = solve(ref, geo, freqs, pol, units)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<52} reference raised {type(exc).__name__}: {exc}")
            continue
        new_result, new_time = solve(new, geo, freqs, pol, units)
        total_ref += ref_time
        total_new += new_time

        ref_amp = amplitudes(ref_result)
        new_amp = amplitudes(new_result)
        if ref_amp.shape != new_amp.shape:
            failures.append((label, "sample count differs"))
            continue
        scale = float(np.max(np.abs(ref_amp))) or 1.0
        amp_err = float(np.max(np.abs(new_amp - ref_amp))) / scale
        db_err = float(np.max(np.abs(decibels(new_result) - decibels(ref_result))))
        panels = int(ref_result.get("metadata", {}).get("panel_count", 0) or 0)
        flag = " FAIL" if (amp_err > AMP_TOL or db_err > DB_TOL) else ""
        print(f"{label:<52} {panels:7d} {ref_time:8.2f} {new_time:8.2f} "
              f"{ref_time / max(new_time, 1e-9):5.2f}x {amp_err:10.2e} "
              f"{db_err:9.2e}{flag}", flush=True)
        if flag:
            failures.append((label, f"amp {amp_err:.2e} dB {db_err:.2e}"))

        ref_formulation = ref_result.get("metadata", {}).get("formulation")
        new_formulation = new_result.get("metadata", {}).get("formulation")
        if ref_formulation != new_formulation:
            failures.append((label, f"formulation {new_formulation!r} != {ref_formulation!r}"))

    print(f"\ntotal: reference {total_ref:.1f} s, optimized {total_new:.1f} s, "
          f"{total_ref / max(total_new, 1e-9):.2f}x")
    print(f"{len(failures)} failure(s) "
          f"(tolerances: amp {AMP_TOL:g} relative, dB {DB_TOL:g})")
    for label, why in failures:
        print(f"  {label}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
