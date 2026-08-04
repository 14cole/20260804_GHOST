#!/usr/bin/env python3
"""
STEP 0 -- OPTIONAL DETAILED BODY-SHADOWING DIAGNOSTIC
=====================================================

WHAT IT DOES
  Every component already hides itself when it faces away (d . normal <= 0).
  That is the right test only for a CONVEX body.  On a real vehicle a component
  can face the radar and still be blocked -- behind a boattail step, inside a
  bay, hidden by another part -- so step 3a can load the clean vehicle mesh and
  ask, per look direction, whether the component is actually visible.

  Geometric blockage only, at the same level as everything else here: it
  catches a component going dark behind structure, not diffraction or creeping
  waves leaking into the shadow.  The CAD is used ONLY to decide what is
  hidden; the body's own RCS still comes from the BoR solve.

WHY THIS IS A CALIBRATION AND NOT A RESULT
  On a CONVEX body the honest expected answer is that the occluder changes
  NOTHING: everything facing the radar is visible.  Any loss you see is SHADOW
  ACNE -- a component drawn on a curved surface (and the polyline tracing it) is
  INSCRIBED in that surface, so the faceted mesh sits a hair in front of it and
  the body appears to block its own features.

  The cure is BIAS: lift the point off the surface along the look before
  testing.  It has to exceed that sag, which scales with facet size.  The
  library default scales with the mesh (0.2 x median facet edge, capped at 1%
  of the body), so it should come out clean -- but the sweep below shows what
  too small a bias costs, and you should see that on YOUR mesh before you
  trust a shadowed number.

  Step 3a performs a conservative automatic bias check using every Coords file
  and its production look grid. This standalone step is no longer required for
  a normal run; use it when you want a wider manual sweep for a difficult mesh.

  Real blockage needs a NON-convex body.  If your hull is convex and the sweep
  is clean, the correct conclusion is that SHADOW buys you nothing here.

INPUTS
  <any>.stl                    the clean vehicle mesh, drawn in UNITS, CAD frame
  <any>.txt                    one component perimeter to test with
  DATASETS_DIR/*.grim          a delta to put on it (any one; the bias question
                               does not depend on which)
  BODY_GRIM                    one solved body from step 2a/2b

OUTPUTS
  bias_calibration.csv   per bias: the largest dB the occluder removed anywhere.
                         On a convex body the right answer is 0.00 dB.
  shadowing_roll.csv     a full roll sweep with and without the occluder at
                         BIAS_M, and the difference

KNOBS (below)
  FREQ_GHZ, ASPECT_DEG, ROLLS_DEG, BIAS_M, BIAS_SWEEP_M, UNITS,
  DATASETS_DIR, BODY_GRIM

  The occluder is built EXACTLY as step 3a builds it -- same CAD-to-solver
  rotation, same scale -- or this would be calibrating something else.

    python3 run.py
"""

import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
FREQ_GHZ = 6.0
ASPECT_DEG = 90.0                          # broadside, so roll does the work
ROLLS_DEG = np.arange(0.0, 359.1, 30.0)    # spin the vehicle past the component
BIAS_M = None                              # None = the mesh-scaled default
BIAS_SWEEP_M = [2e-5, 5e-5, 1e-4, 3e-4, None, 4e-3]   # metres; None = default
UNITS = "meters"                           # units the .stl and .txt are drawn in
DATASETS_DIR = os.path.join("..", "1c_build_deltas", "Deltas")
BODY_GRIM = os.path.join("..", "2b_solve_body_hpc", "results", "body.grim")
# ────────────────────────────────────────────────────────────────────────────

from frame import scale_for, to_axis_frame                            # noqa: E402
from feature_sum import (sum_features, directions_from_aspect_roll,  # noqa: E402
                         load_body_profile_grim)
from line_expand import read_perimeter_txt                            # noqa: E402
from occluder import Occluder, read_stl                               # noqa: E402

SCALE = scale_for(UNITS)


def _here(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def one(pattern, what):
    hits = sorted(glob.glob(os.path.join(HERE, pattern)))
    if len(hits) != 1:
        raise SystemExit(f"put exactly one {what} in {HERE} -- found "
                         f"{[os.path.basename(h) for h in hits]}.")
    return hits[0]


def main():
    stl, track = one("*.stl", "vehicle .stl"), one("*.txt", "perimeter .txt")
    ds = _here(DATASETS_DIR)
    deltas = sorted(glob.glob(os.path.join(ds, "*.grim")))
    if not deltas:
        raise SystemExit(f"no *.grim in {ds} -- run 1c_build_deltas first.")
    body_grim = _here(BODY_GRIM)
    if not os.path.exists(body_grim):
        raise SystemExit(f"no {body_grim} -- run step 2 first.")
    try:
        gen = load_body_profile_grim(body_grim)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tris = to_axis_frame(read_stl(stl))          # same rotation step 3a uses
    base = Occluder(tris, scale=SCALE)
    edge = max(float(np.linalg.norm(base.tris[:, i] - base.tris[:, (i + 1) % 3],
                                    axis=1).max()) for i in range(3))
    print(f"STEP 0   shadowing calibration on {os.path.basename(stl)} "
          f"({len(base.tris)} triangles)")
    print(f"         component: {os.path.basename(track)}  with "
          f"{os.path.basename(deltas[0])}")
    print(f"         facets: median edge {base.median_edge*1e3:.2f} mm, longest "
          f"{edge*1e3:.2f} mm;  default bias {base.bias*1e3:.3f} mm")

    per = to_axis_frame(read_perimeter_txt(track, scale=SCALE))
    place = [{"delta": deltas[0], "perimeter": per}]
    dirs, _asp, rol = directions_from_aspect_roll([ASPECT_DEG], ROLLS_DEG)
    kw = dict(generatrix=gen, mode="coherent")
    free = sum_features(None, place, dirs, FREQ_GHZ, **kw)["dbsm_vv"]

    # ---- the calibration: on a CONVEX body the occluder must change nothing --
    #  reported three ways, because grazing looks are the hardest: near the
    #  terminator the faceted surface genuinely self-occludes, and that is also
    #  where the component contributes least
    peak = float(np.max(free))
    near, lit = free > peak - 20.0, free > -199.0
    print(f"\n         component peak {peak:+.1f} dBsm;  spurious loss the "
          f"occluder removes (must be 0.00):")
    print("         bias (mm)   at the peak look   within 20 dB of peak   "
          "any lit look")
    rows = [("bias_mm", "loss_at_peak_dB", "loss_within_20dB_dB",
             "loss_any_lit_dB")]
    for b in BIAS_SWEEP_M:
        s = sum_features(None, place, dirs, FREQ_GHZ,
                         occluder=Occluder(tris, scale=SCALE, bias=b),
                         **kw)["dbsm_vv"]
        dl = free - s
        at_pk = float(dl[int(np.argmax(free))])
        in20 = float(np.max(dl[near])) if np.any(near) else 0.0
        any_lit = float(np.max(dl[lit])) if np.any(lit) else 0.0
        used = (b if b is not None else base.bias) * 1e3
        rows.append((f"{used:.5f}", f"{at_pk:.5f}", f"{in20:.5f}",
                     f"{any_lit:.5f}"))
        lbl = f"{base.bias*1e3:.3g}*" if b is None else f"{b*1e3:.3g}"
        tag = "   <- SHADOW ACNE" if in20 >= 0.01 else ""
        print(f"         {lbl:>9}   {at_pk:16.2f}   {in20:20.2f}   "
              f"{any_lit:12.2f}{tag}")
    print("         (* = the mesh-scaled default)")
    with open(os.path.join(HERE, "bias_calibration.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("         wrote bias_calibration.csv")

    # ---- and what the occluder actually does over a roll sweep --------------
    occ = Occluder(tris, scale=SCALE, bias=BIAS_M)
    print(f"\n         full roll sweep at bias = {occ.bias*1e3:g} mm"
          f"{' (the default)' if BIAS_M is None else ''}:")
    shad = sum_features(None, place, dirs, FREQ_GHZ, occluder=occ,
                        **kw)["dbsm_vv"]
    with open(os.path.join(HERE, "shadowing_roll.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(("roll_deg", "no_occluder_dBsm", "with_occluder_dBsm",
                    "difference_dB"))
        for r, a, b in zip(rol, free, shad):
            w.writerow((f"{r:g}", f"{a:.4f}", f"{b:.4f}", f"{b - a:.4f}"))
    print("\n         roll    no occluder   with occluder   change")
    for r, a, b in zip(rol, free, shad):
        note = "  <- newly hidden" if b <= -199.0 < a else ""
        print(f"         {r:5.0f}   {a:+11.1f}   {b:+13.1f}   {b - a:+6.2f}"
              f"{note}")
    print("         wrote shadowing_roll.csv")

    worst = float(np.max(free[lit] - shad[lit])) if np.any(lit) else 0.0
    print()
    if worst < 0.01:
        print(f"         VERDICT  the occluder removes at most {worst:.2f} dB "
              f"anywhere it is lit.\n                  On a convex hull that is "
              f"the right answer, and it means SHADOW\n                  buys "
              f"you nothing on this body -- leave it off unless your real\n"
              f"                  vehicle has a bay, a step, or one part behind "
              f"another.")
    else:
        print(f"         VERDICT  the occluder removes up to {worst:.2f} dB.  If "
              f"this hull is convex\n                  that is SHADOW ACNE, not "
              f"blockage -- raise the bias until the\n                  'within "
              f"20 dB of peak' column above reads 0.00, and use that\n"
              f"                  value as BIAS in your real run.")
    print("\nNEXT     3a_doors, with SHADOW set from what you just measured")


if __name__ == "__main__":
    main()
