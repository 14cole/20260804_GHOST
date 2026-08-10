#!/usr/bin/env python3
"""
Door / feature trade study -- compare delta designs at ONE perimeter.

Hold the perimeter (and body) fixed, sweep the feature delta.  For each design
this computes the ISOLATED feature response (rank the designs) and, if a body is
supplied, how far the feature pokes ABOVE the body (does it matter).

    from trade_study import door_trade_study
    rows = door_trade_study(
        delta_paths=["designA.grim", "designB.grim", "designC.grim"],
        perimeter="door.txt",              # one perimeter, shared by every design
        generatrix=gen,                    # body (rho, z) polyline, for the surface normal
        frequencies_ghz=[6.0, 10.0],
        aspects_deg=range(0, 181, 5), rolls_deg=[0.0],
        body=body,                         # optional: a result or {freq: result}
        out_dir="trade_out")               # optional: writes a body-frame .grim per design

Run ``python3 trade_study.py`` for a self-contained demo (fabricated deltas).
"""

import csv as _csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

C0 = 299_792_458.0


def door_trade_study(delta_paths: 'Sequence[str]',
                     perimeter,
                     generatrix,
                     frequencies_ghz: 'Sequence[float]',
                     aspects_deg: 'Sequence[float]',
                     rolls_deg: 'Sequence[float]' = (0.0,),
                     body=None,
                     mode: 'str' = "coherent",
                     occluder=None,
                     labels: 'Optional[Sequence[str]]' = None,
                     out_dir: 'Optional[str]' = None,
                     csv_path: 'Optional[str]' = None,
                     print_table: 'bool' = True) -> 'List[Dict[str, Any]]':
    """Compare feature delta designs at one perimeter.  Returns one summary row
    per (design, frequency): isolated peak dBsm (VV/HH), the look it peaks at,
    peak cross-pol, and -- if ``body`` is given -- the max the feature lifts the
    total over the bare body (dB).  ``body`` may be a single BoR result or a
    dict {freq_ghz: result}."""
    from feature_sum import (sum_features, directions_from_aspect_roll,
                             export_signature_grim, _pick_body)
    from line_expand import read_perimeter_txt, dbsm, surface_of_revolution_normal

    gen = np.asarray(generatrix, dtype=float)
    nfn = surface_of_revolution_normal(gen)
    per = read_perimeter_txt(perimeter) if isinstance(perimeter, str) else np.asarray(perimeter, float)
    dirs, asp, roll = directions_from_aspect_roll(aspects_deg, rolls_deg)
    if labels is None:
        labels = [os.path.splitext(os.path.basename(str(p)))[0] for p in delta_paths]

    rows: 'List[Dict[str, Any]]' = []
    for path, label in zip(delta_paths, labels):
        door = {"delta": str(path), "perimeter": per}
        for f in frequencies_ghz:
            iso = sum_features(None, [door], dirs, float(f), normal_fn=nfn, mode="coherent",
                               occluder=occluder)
            svv, shh, svh = iso["dbsm_vv"], iso["dbsm_hh"], iso["dbsm_vh"]
            iv = int(np.argmax(svv))
            row: 'Dict[str, Any]' = dict(
                design=label, freq_ghz=float(f),
                peak_vv_dbsm=float(svv[iv]), peak_hh_dbsm=float(np.max(shh)),
                peak_aspect_deg=float(asp[iv]), peak_roll_deg=float(roll[iv]),
                xpol_vh_dbsm=float(np.max(svh)))
            if body is not None:
                bf = _pick_body(body, float(f))
                full = sum_features(bf, [door], dirs, float(f), normal_fn=nfn, mode=mode,
                                    occluder=occluder)
                bod = sum_features(bf, [], dirs, float(f), normal_fn=nfn, mode=mode)
                lift = dbsm(full["sigma_vv"]) - dbsm(bod["sigma_vv"])
                # only where the feature actually contributes (within 40 dB of its peak)
                sig = svv >= (svv[iv] - 40.0)
                il = int(np.argmax(np.where(sig, lift, -np.inf)))
                row["max_lift_over_body_db"] = float(lift[il])
                row["lift_aspect_deg"] = float(asp[il])
            rows.append(row)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                export_signature_grim(os.path.join(out_dir, f"{label}_{f:g}GHz"),
                                      bor_result=None, placements=[door], generatrix=gen,
                                      frequencies_ghz=[float(f)], aspects_deg=aspects_deg,
                                      rolls_deg=rolls_deg, mode="coherent", occluder=occluder,
                                      history=f"trade_study isolated feature {label}")
    if print_table:
        _print_table(rows, body is not None)
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {csv_path}")
    return rows


def _print_table(rows: 'List[Dict[str, Any]]', has_body: 'bool') -> 'None':
    print("\n" + "=" * (92 if has_body else 74))
    hdr = (f"  {'design':16s} {'freq':>5} {'peakVV':>8} {'peakHH':>8} "
           f"{'@aspect':>8} {'@roll':>7} {'xpol VH':>8}")
    if has_body:
        hdr += f" {'lift/body':>10} {'@aspect':>8}"
    print(hdr)
    print("  " + "-" * (90 if has_body else 72))
    for r in rows:
        line = (f"  {r['design']:16s} {r['freq_ghz']:5.1f} {r['peak_vv_dbsm']:8.1f} "
                f"{r['peak_hh_dbsm']:8.1f} {r['peak_aspect_deg']:8.0f} "
                f"{r['peak_roll_deg']:7.0f} {r['xpol_vh_dbsm']:8.1f}")
        if has_body:
            line += f" {r.get('max_lift_over_body_db', float('nan')):10.1f} {r.get('lift_aspect_deg', 0):8.0f}"
        print(line)
    print("=" * (92 if has_body else 74))


# -- self-contained demo (fabricated delta designs; no solver) -----------------

def _fab_delta(path: 'str', freqs, scale: 'float') -> 'str':
    """Write a synthetic delta .grim (rcs_domain='delta') for the demo."""
    from feature_sum import (
        DELTA_FIELD_DOMAIN,
        DELTA_PHASE_SUFFIX,
        PHYSICAL_2D_AMPLITUDE_CONVENTION,
        PHYSICAL_2D_PHASE_REFERENCE,
    )
    phi = np.arange(0.0, 180.1, 5.0)
    bump = scale * np.exp(-((phi - 90.0) / 30.0) ** 2)          # a lobe near normal
    amp = np.zeros((len(phi), 1, len(freqs), 2), complex)
    for fi in range(len(freqs)):
        amp[:, 0, fi, 0] = 0.8 * bump * (1 + 0.3j)              # VV (=TE) channel
        amp[:, 0, fi, 1] = bump * (1 - 0.2j)                    # HH (=TM) channel
    units = json.dumps({"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                        "rcs_log_unit": "dBke", "rcs_linear_quantity": "sigma_2d"})
    payload = dict(
        azimuths=phi, elevations=np.array([0.0]), frequencies=np.asarray(freqs, float),
        polarizations=np.asarray(["VV", "HH"], dtype=str),
        polarization_alias_primary="VV,HH", polarization_aliases_json=json.dumps(["TE", "TM"]),
        # sigma_2d = |amp|^2/(4k): a delta is a 2-D quantity like any other 2-D cut
        rcs_power=(np.abs(amp) ** 2
                   / (4.0 * 2.0 * math.pi * np.asarray(freqs, float) * 1e9 / C0
                      )[None, None, :, None]).astype(np.float32),
        rcs_phase=np.angle(amp).astype(np.float32),
        rcs_domain="delta", power_domain="linear_rcs", source_path="", history="fabricated",
        units=units,
        phase_reference=PHYSICAL_2D_PHASE_REFERENCE + DELTA_PHASE_SUFFIX,
        amplitude_convention=PHYSICAL_2D_AMPLITUDE_CONVENTION,
        raw_complex_amplitude_preserved=True,
        rcs_amp_real=amp.real.astype(np.float64), rcs_amp_imag=amp.imag.astype(np.float64),
        complex_field_domain=DELTA_FIELD_DOMAIN)
    from grim_io import _save_grim_npz
    return _save_grim_npz(payload, path)


def main() -> 'None':
    import shutil
    out = "_trade_demo"
    os.makedirs(out, exist_ok=True)
    freqs = [6.0]
    # three "designs": weaker / baseline / stronger seam
    deltas = [_fab_delta(os.path.join(out, f"{n}.grim"), freqs, s)
              for n, s in (("weak", 0.5), ("baseline", 1.0), ("strong", 2.0))]

    # a body: a plain cylinder generatrix + a synthetic body result at 6 GHz
    lam = C0 / (6.0e9); a, L = 0.06, 0.30
    z = np.linspace(L / 2, -L / 2, 40)
    side = np.column_stack([np.full_like(z, a), z])
    gen = np.vstack([[[0, L / 2]], side, [[0, -L / 2]]])
    th = np.arange(0.0, 180.1, 5.0)
    body = {"theta_deg": list(th),
            "amp_vv": (0.05 * np.abs(np.cos(np.radians(th - 90))) + 0.001).tolist(),
            "amp_hh": (0.04 * np.abs(np.cos(np.radians(th - 90))) + 0.001).tolist()}

    # one door on the side, phi=0, near z=+60mm
    door = os.path.join(out, "door.txt")
    aa = a
    pts = []
    for ph, zz in [(-0.4, 0.04), (0.4, 0.04), (0.4, 0.08), (-0.4, 0.08), (-0.4, 0.04)]:
        pts.append((aa * math.cos(ph), aa * math.sin(ph), zz))
    with open(door, "w") as fh:
        for i in range(len(pts) - 1):
            fh.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (*pts[i], *pts[i + 1]))

    print("=" * 74)
    print("trade_study.py demo -- 3 fabricated door deltas at one perimeter")
    print("=" * 74)
    door_trade_study(deltas, door, gen, freqs,
                     aspects_deg=np.arange(0.0, 180.1, 5.0), rolls_deg=[0.0],
                     body=body, out_dir=out, csv_path=os.path.join(out, "trade.csv"))
    print(f"\n  per-design grims + trade.csv in {os.path.abspath(out)}/")
    # keep outputs for inspection; delete _trade_demo/ to clean up
    del shutil


if __name__ == "__main__":
    main()
