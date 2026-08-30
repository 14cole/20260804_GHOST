#!/usr/bin/env python3
"""
grim_naming.py -- the production filename grammar, and the two operations that
depend on it: JOIN the solver's per-(pol, frequency) files into one file per
variation, and PAIR featured with clean so deltas can be built unattended.

THE GRAMMAR
-----------
A variation is identified by a STUDY ID plus its PARAMETERS:

    SEAL-00-01_0.010gap_OPN.grim        featured  (OPN = feature present)
    SEAL-00-01_0.010gap_FRD.grim        clean     (FRD = faired / smooth)
    SEAL-00-01_0.010gap.grim            their delta (no role marker)

    <study-id>_<value><key>[_<value><key>...][_OPN|_FRD].grim

  * the STUDY ID is the first '_'-separated chunk -- seamtype-study-version, e.g.
    SEAL-00-01.  It is CATEGORICAL, not an axis: two studies are different seams
    or different versions of one, so you SELECT a study, you never interpolate
    across it.  It must not itself contain '_'.
  * every later chunk is a PARAMETER token <value><key> (0.010gap, 0.002bmag),
    sorted by key, one decimal width per key across a library.
  * the role marker is LAST: OPN featured, FRD clean, absent = the delta.

Straight off the 2-D solver (HPC or local) the same variation arrives split by
polarization and frequency, with those leading the name:

    HH_2.000GHz_SEAL-00-01_0.010gap_OPN.grim
    VV_2.000GHz_SEAL-00-01_0.010gap_OPN.grim
    HH_4.000GHz_SEAL-00-01_0.010gap_OPN.grim   ...

    <POL>_<FREQ>GHz_<variation>.grim

``join_grims`` folds all of those into ONE
``SEAL-00-01_0.010gap_OPN.grim`` carrying both axes.  That is not cosmetic:
feature_sum._amp_tables (so make_delta_grim and the seam/wing loaders) requires
every grim handed to it to share ONE (angle, frequency) grid -- it merges
polarizations, not frequencies -- so the split files cannot be subtracted as they
come.

NOTE the polarization in the FILENAME is the solver's (TM/TE or HH/VV) while the
label inside the file is the primary alias (TM -> HH, TE -> VV).  Both are
canonicalised here, so either naming joins correctly and the original primaries
are preserved in ``polarization_alias_primary``.
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROLE_FEATURED = "OPN"
ROLE_CLEAN = "FRD"
ROLES = (ROLE_FEATURED, ROLE_CLEAN)

_POL_CANON = {"VV": "VV", "TE": "VV", "V": "VV", "VERTICAL": "VV",
              "HH": "HH", "TM": "HH", "H": "HH", "HORIZONTAL": "HH",
              "VH": "VH", "HV": "VH", "CROSS": "VH"}
_POL_ORDER = ("VV", "HH", "VH")

_SOLVER_RE = re.compile(
    r"^(?P<pol>VV|HH|VH|HV|TM|TE)_(?P<freq>[0-9]*\.?[0-9]+)GHz_(?P<base>.+)$")
_PARAM_RE = re.compile(r"^(-?[0-9]+(?:\.[0-9]+)?)([A-Za-z][A-Za-z0-9]*)$")


def canon_pol(label: 'str') -> 'str':
    """Solver or file polarization label -> VV / HH / VH."""
    key = str(label).strip().upper()
    if key not in _POL_CANON:
        raise ValueError(f"unknown polarization {label!r}; expected one of "
                         f"{sorted(set(_POL_CANON))}.")
    return _POL_CANON[key]


# -----------------------------------------------------------------------------
# names
# -----------------------------------------------------------------------------

def parse_solver_name(path: 'str') -> 'Dict[str, Any]':
    """``HH_2.000GHz_SEAL-00-01_0.010gap_OPN.grim`` ->
    {pol, pol_canon, freq_ghz, variation, base, role}."""
    stem = _stem(path)
    m = _SOLVER_RE.match(stem)
    if not m:
        raise ValueError(f"{os.path.basename(str(path))}: not "
                         f"<POL>_<FREQ>GHz_<variation>.grim")
    variation = m.group("base")
    base, role = parse_variation(variation)
    return {"pol": m.group("pol"), "pol_canon": canon_pol(m.group("pol")),
            "freq_ghz": float(m.group("freq")), "variation": variation,
            "base": base, "role": role, "path": str(path)}


def parse_variation(name: 'str') -> 'Tuple[str, Optional[str]]':
    """``SEAL-00-01_0.010gap_OPN`` -> ("SEAL-00-01_0.010gap", "OPN").
    A name with no role marker is a delta and returns (base, None)."""
    stem = _stem(name)
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].upper() in ROLES:
        return "_".join(parts[:-1]), parts[-1].upper()
    return stem, None


def require_role_free_declared_delta(path: 'str') -> 'str':
    """Reject a canonical raw OPN/FRD source used as an asserted delta.

    ``declared_coherent_delta`` is an attestation for a role-free derived file;
    it must never override the filename grammar's explicit source role.  The
    returned value is the role-free variation/base name for provenance callers.
    Matching is intentionally case-insensitive through :func:`parse_variation`.
    """

    base, role = parse_variation(path)
    if role is None:
        return base
    source_kind = "featured/installed" if role == ROLE_FEATURED else "clean/faired"
    raise ValueError(
        f"{os.path.basename(str(path))}: canonical filename role _{role} "
        f"declares a raw {source_kind} source response, not a coherent "
        "featured-minus-clean delta. The declared_coherent_delta attestation "
        "cannot override an explicit OPN/FRD role. Coherently subtract OPN - "
        "FRD, then save the derived response under a role-free filename "
        "without an _OPN or _FRD suffix."
    )


def parse_base(name: 'str') -> 'Tuple[str, Dict[str, float], Dict[str, int]]':
    """``SEAL-00-01_0.010gap`` -> ("SEAL-00-01", {"gap": 0.010}, {"gap": 3}).

    The first chunk is the study id (categorical).  Every later chunk must be a
    <value><key> parameter token, so a typo is an error here rather than a
    silently-lost parameter."""
    base, _role = parse_variation(name)
    parts = base.split("_")
    study = parts[0]
    if _PARAM_RE.match(study):
        raise ValueError(f"{base}: first chunk {study!r} looks like a parameter "
                         f"token, not a study id (expected e.g. SEAL-00-01).")
    vals: 'Dict[str, float]' = {}
    decs: 'Dict[str, int]' = {}
    for tok in parts[1:]:
        m = _PARAM_RE.match(tok)
        if not m:
            raise ValueError(f"{base}: chunk {tok!r} is not <value><key> "
                             f"(e.g. 0.010gap).")
        num, key = m.group(1), m.group(2)
        if key in vals:
            raise ValueError(f"{base}: key {key!r} appears twice.")
        vals[key] = float(num)
        decs[key] = len(num.split(".")[1]) if "." in num else 0
    if not vals:
        raise ValueError(f"{base}: no parameter tokens (expected e.g. "
                         f"{study}_0.010gap).")
    return study, vals, decs


def format_base(study: 'str', params: 'Dict[str, float]',
                decimals: 'Dict[str, int]') -> 'str':
    """Canonical base name: study id first, then parameter tokens sorted by key."""
    missing = [k for k in params if k not in decimals]
    if missing:
        raise ValueError(f"no decimal width declared for {missing!r}.")
    toks = [f"{params[k]:.{decimals[k]}f}{k}" for k in sorted(params)]
    return "_".join([str(study)] + toks)


def variation_name(base: 'str', role: 'Optional[str]' = None) -> 'str':
    """base (+ optional OPN/FRD) -> a .grim filename."""
    if role is None:
        return f"{base}.grim"
    if str(role).upper() not in ROLES:
        raise ValueError(f"role must be one of {ROLES} or None, got {role!r}.")
    return f"{base}_{str(role).upper()}.grim"


def _stem(path: 'str') -> 'str':
    b = os.path.basename(str(path))
    return b[: -len(".grim")] if b.lower().endswith(".grim") else b


# -----------------------------------------------------------------------------
# join: many (pol, frequency) files -> one file per variation
# -----------------------------------------------------------------------------

def group_solver_files(paths: 'Sequence[str]'
                       ) -> 'Tuple[Dict[str, List[Dict[str, Any]]], List[Tuple[str, str]]]':
    """Group ``<POL>_<FREQ>GHz_<variation>.grim`` paths by variation.
    Returns (groups, unparsed) -- unparsed files are REPORTED, never skipped
    silently."""
    groups: 'Dict[str, List[Dict[str, Any]]]' = {}
    unparsed: 'List[Tuple[str, str]]' = []
    for p in sorted(paths):
        try:
            rec = parse_solver_name(p)
        except ValueError as exc:
            unparsed.append((str(p), str(exc)))
            continue
        groups.setdefault(rec["variation"], []).append(rec)
    return groups, unparsed


def join_grims(paths: 'Sequence[str]', out_path: 'str', *, history: 'str' = "") -> 'str':
    """Fold single-(pol, frequency) grims of ONE variation into one .grim with a
    full polarization and frequency axis.

    Every input must share the (azimuth, elevation) sweep, the units and the
    domain tags, and the (polarization x frequency) grid must be COMPLETE -- a
    ragged grid cannot be one array, so it raises rather than padding.
    """
    from feature_sum import _load_grim                        # noqa: PLC0415
    if not paths:
        raise ValueError("nothing to join.")
    cells: 'Dict[Tuple[str, float], Dict[str, Any]]' = {}
    primaries: 'Dict[str, str]' = {}
    template: 'Optional[Dict[str, Any]]' = None
    ang = el = None
    units_ref = dom_ref = None
    coherent_ref: 'Optional[Tuple[str, str, str, str]]' = None
    for p in sorted(paths):
        g = _load_grim(str(p))
        pols = [str(x) for x in np.asarray(g["polarizations"]).ravel()]
        if len(pols) != 1:
            raise ValueError(f"{os.path.basename(str(p))}: expected one "
                             f"polarization per solver file, got {pols}.")
        pc = canon_pol(pols[0])
        a = np.asarray(g["azimuths"], float)
        e = np.asarray(g["elevations"], float)
        u, d = str(g.get("units", "")), str(g.get("rcs_domain", ""))
        coherent = tuple(str(g.get(key, "")).strip() for key in (
            "phase_reference", "amplitude_convention",
            "complex_field_domain", "power_domain"))
        if any(not value for value in coherent):
            raise ValueError(
                f"{os.path.basename(str(p))}: missing a coherent-field "
                "convention (phase_reference, amplitude_convention, "
                "complex_field_domain, or power_domain).")
        if template is None:
            template, ang, el, units_ref, dom_ref = g, a, e, u, d
            coherent_ref = coherent
        else:
            if not (np.array_equal(a, ang) and np.array_equal(e, el)):
                raise ValueError(f"{os.path.basename(str(p))}: different "
                                 f"angle sweep from the first file.")
            if u != units_ref or d != dom_ref:
                raise ValueError(f"{os.path.basename(str(p))}: different units or "
                                 f"domain from the first file -- refusing to join "
                                 f"physically different quantities.")
            if coherent != coherent_ref:
                raise ValueError(
                    f"{os.path.basename(str(p))}: coherent-field convention "
                    "differs from the first file; refusing to join fields with "
                    "different phase origins or amplitude normalizations.")
        primaries.setdefault(pc, str(g.get("polarization_alias_primary", pols[0])))
        for kf, f in enumerate(np.asarray(g["frequencies"], float)):
            key = (pc, float(f))
            if key in cells:
                raise ValueError(f"two files give {pc} at {f:g} GHz "
                                 f"({os.path.basename(str(cells[key]['path']))} and "
                                 f"{os.path.basename(str(p))}).")
            cells[key] = {"amp": g["_amp"][:, :, kf, 0], "path": p}

    pol_list = ([p for p in _POL_ORDER if p in primaries]
                + sorted(set(primaries) - set(_POL_ORDER)))
    freqs = sorted({f for _p, f in cells})
    missing = [(p, f) for p in pol_list for f in freqs if (p, f) not in cells]
    if missing:
        raise ValueError(f"incomplete (polarization x frequency) grid, missing "
                         f"{[(p, f'{f:g}GHz') for p, f in missing][:6]} -- wait for "
                         f"those units to finish, or drop the odd frequency out.")

    amp = np.zeros((len(ang), len(el), len(freqs), len(pol_list)), dtype=complex)
    for jp, p in enumerate(pol_list):
        for kf, f in enumerate(freqs):
            amp[:, :, kf, jp] = cells[(p, f)]["amp"]

    out = str(out_path) if str(out_path).lower().endswith(".grim") else str(out_path) + ".grim"
    payload = {k: template[k] for k in template
               if k not in ("_amp", "rcs_power", "rcs_phase", "rcs_amp_real",
                            "rcs_amp_imag", "frequencies", "polarizations",
                            "polarization_alias_primary",
                            "polarization_aliases_json", "history",
                            # This describes one source sweep and would be
                            # false metadata after a multi-file join.
                            "solver_metadata_json")}
    # rcs_power keeps the template's convention: rebuild it from the per-cell
    # power so no assumption about sigma_2d vs sigma_3d is baked in here
    power = np.zeros(amp.shape, dtype=np.float32)
    for jp, p in enumerate(pol_list):
        for kf, f in enumerate(freqs):
            g = _load_grim(str(cells[(p, f)]["path"]))
            k0 = int(np.argmin(np.abs(np.asarray(g["frequencies"], float) - f)))
            power[:, :, kf, jp] = np.asarray(g["rcs_power"], float)[:, :, k0, 0]
    payload.update(
        frequencies=np.asarray(freqs, float),
        polarizations=np.asarray(pol_list, dtype=str),
        polarization_alias_primary=",".join(primaries[p] for p in pol_list),
        polarization_aliases_json=json.dumps([primaries[p] for p in pol_list]),
        rcs_power=power,
        rcs_phase=np.angle(amp).astype(np.float32),
        rcs_amp_real=amp.real.astype(np.float64),
        rcs_amp_imag=amp.imag.astype(np.float64),
        raw_complex_amplitude_preserved=True,
        history=(str(template.get("history", "")) + " | "
                 + (history or f"join_grims: {len(paths)} files -> "
                               f"{len(pol_list)} pol x {len(freqs)} freq")))
    from grim_io import _save_grim_npz
    return _save_grim_npz(payload, out)


# -----------------------------------------------------------------------------
# pair: featured (OPN) with clean (FRD)
# -----------------------------------------------------------------------------

def pair_variants(paths: 'Sequence[str]'
                  ) -> 'Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]':
    """Match each OPN to the most-specific compatible FRD.

    Returns (pairs, unmatched).  Each pair is
    {base, study, params, decimals, featured, clean, delta_name}; unmatched
    entries carry the reason, because a lone OPN or FRD is the most common
    reason an automated library comes out short.

    Compatibility means that the study IDs match and every FRD parameter
    key/value is present in the OPN. One baseline FRD may therefore serve many
    OPN cases that carry additional feature-only variables. An exact or more
    specific baseline wins over a broader one. Equal-specificity ambiguity is
    refused instead of being resolved by filename order.
    """
    byrole: 'Dict[str, Dict[str, Dict[str, Any]]]' = {
        ROLE_FEATURED: {}, ROLE_CLEAN: {}
    }
    others: 'List[Dict[str, Any]]' = []
    for p in sorted(paths):
        base, role = parse_variation(p)
        if role is None:
            others.append({"path": str(p), "base": base,
                           "reason": "no OPN/FRD marker (already a delta?)"})
            continue
        if base in byrole[role]:
            raise ValueError(f"two {role} files for {base}: "
                             f"{os.path.basename(byrole[role][base]['path'])} and "
                             f"{os.path.basename(str(p))}.")
        study, params, decimals = parse_base(base)
        byrole[role][base] = {
            "path": str(p), "base": base, "study": study,
            "params": params, "decimals": decimals,
        }

    pairs: 'List[Dict[str, Any]]' = []
    used_clean: 'set[str]' = set()
    for base, featured in sorted(byrole[ROLE_FEATURED].items()):
        compatible = []
        for clean in byrole[ROLE_CLEAN].values():
            if clean["study"] != featured["study"]:
                continue
            if all(
                key in featured["params"]
                and math.isclose(
                    float(featured["params"][key]), float(value),
                    rel_tol=0.0, abs_tol=1e-15,
                )
                for key, value in clean["params"].items()
            ):
                compatible.append(clean)
        if not compatible:
            others.append({
                "path": featured["path"], "base": base,
                "reason": "no compatible FRD whose parameters are a subset "
                          "of this OPN",
            })
            continue
        specificity = max(len(clean["params"]) for clean in compatible)
        best = [
            clean for clean in compatible
            if len(clean["params"]) == specificity
        ]
        if len(best) != 1:
            candidates = [
                clean["base"] + "_FRD"
                for clean in sorted(best, key=lambda item: item["base"])
            ]
            raise ValueError(
                f"{base}_OPN has ambiguous equally-specific FRD baselines: "
                f"{candidates}"
            )
        clean = best[0]
        used_clean.add(clean["base"])
        pairs.append({
            "base": base,
            "study": featured["study"],
            "params": featured["params"],
            "decimals": featured["decimals"],
            "featured": featured["path"],
            "clean": clean["path"],
            "clean_base": clean["base"],
            "delta_name": variation_name(base),
        })
    for base, clean in sorted(byrole[ROLE_CLEAN].items()):
        if base not in used_clean:
            others.append({
                "path": clean["path"], "base": base,
                "reason": "FRD is not used by any compatible OPN",
            })
    return pairs, others
