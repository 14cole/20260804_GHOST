#!/usr/bin/env python3
"""
Bridge to the GRIM_Revised_2 viewer/dataset tool (``grim_dataset.RcsGrid``).

Both projects read and write ``.grim`` (npz) files and they ARE compatible --
RcsGrid.load reads every file this repo writes, and phase agrees to float32.
There is exactly one thing to know before mixing them:

    rcs_power is the PHYSICAL quantity in both tools, but this repo's
    rcs_amp_real/imag is the SOLVER'S FIELD amplitude, which is NOT
    sqrt(rcs_power).  There are exactly TWO conventions, and they follow the
    dimensionality of the solve:

      data type   power stored             log unit   sqrt(power)/|amp|
      ------------------------------------------------------------------
      2-D         sigma_2d = |A|^2/(4k)    dBke       1/(2 sqrt(k))  <- per freq
      3-D         sigma    = 4 pi |F|^2    dBsm       sqrt(4 pi) = 3.5449

    (2-D "RCS" is a scattering WIDTH in meters, so the length comes from 1/k --
    which is why that factor moves with frequency; 3-D is a cross-section in m^2
    and the 4 pi is the isotropic reference in the definition of sigma.)

    A delta is a 2-D data type -- not a third one.  ``rcs_domain='delta'`` is an
    ORTHOGONAL axis saying the samples are a difference (featured - clean); the
    pipeline routes on it, and it does not change the units.

So anything RcsGrid does in the POWER domain (dBsm/dBke, crop, mirror, join,
statistics, plotting) is correct on these files as-is.  But ``RcsGrid.rcs`` --
which it rebuilds as sqrt(power) * exp(j*phase) -- is a scaled field: fine to add
or subtract WITHIN one export type (the constant cancels), wrong if you mix a 3-D
export with a 2-D delta.  ``field_amplitude()`` below removes the factor and
gives back this repo's amplitude.

Also note RcsGrid derives power from its own complex samples, so a grid it BUILT
satisfies power == |rcs|^2; the table above is about grids it LOADED from here.

What this module is for:
  * ``to_grid`` / ``from_grid``     -- hand a file to that tool and get it back
                                      without losing the complex amplitude
  * ``field_amplitude``            -- RcsGrid samples -> this repo's amplitude
  * ``load_pattern_any``           -- read the formats RcsGrid can import
                                      (.out, .ss, PIO, theta/phi CSV or TXT) into
                                      the dict ``point_scatterer_amplitude``
                                      wants, so an external 3-D MoM result does
                                      not have to be hand-built into a .grim
  * ``describe``                   -- print what a .grim is, in both tools' terms

The other repo is found via $GRIM_REVISED_PATH, else a sibling directory named
GRIM_Revised_2.  Everything here degrades to a clear error if it is absent; the
rest of this repo never imports it.
"""

import math
import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

C0 = 299_792_458.0

_SIBLINGS = ("GRIM_Revised_2", "GRIM_Revised", "grim_revised_2")


_SEARCH_LEVELS = 4          # this file lives Backend/ deep inside the workflow
                            # folder, which itself may sit inside a project
                            # folder -- so "beside this repo" is several
                            # ancestors up, not exactly one.


def _grim_revised_dir() -> 'str':
    env = os.environ.get("GRIM_REVISED_PATH", "").strip()
    cands = [env] if env else []
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(_SEARCH_LEVELS):
        parent = os.path.dirname(here)
        if parent == here:                      # hit the filesystem root
            break
        here = parent
        cands.extend(os.path.join(here, s) for s in _SIBLINGS)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "grim_dataset.py")):
            return c
    raise ImportError(
        "grim_dataset.py not found.  Set GRIM_REVISED_PATH to the GRIM_Revised_2 "
        f"folder, or place it beside this repo (looked in: {cands}).")


def rcsgrid_class():
    """The RcsGrid class from the viewer tool (imported on demand)."""
    d = _grim_revised_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    from grim_dataset import RcsGrid            # noqa: PLC0415
    return RcsGrid


# -----------------------------------------------------------------------------
# the one conversion that matters: power <-> field amplitude
# -----------------------------------------------------------------------------

def amp_scale(grim: 'Dict[str, Any] | str',
              frequencies_ghz: 'Optional[Sequence[float]]' = None) -> 'np.ndarray':
    """``sqrt(rcs_power) / |rcs_amp|`` for a .grim of this repo, as an array
    broadcastable over the frequency axis (it is frequency-dependent for 2-D
    cuts).  Divide RcsGrid's complex samples by this to get the field amplitude.

    There are only TWO conventions, read off the file's own tag -- not measured:

      rcs_linear_quantity == 'sigma_2d'   sigma_2d = |amp|^2/(4k)  -> 1/(2 sqrt(k))
      rcs_linear_quantity == 'sigma_3d'   sigma = 4 pi |amp|^2     -> sqrt(4 pi)

    'delta' is NOT a third convention: a delta is a 2-D quantity, so it is
    sigma_2d like any other 2-D cut.  ``rcs_domain='delta'`` lives on a separate
    axis -- it says the samples are a DIFFERENCE (featured - clean) rather than a
    whole object, which is what the pipeline routes on, and says nothing about
    units.

    LEGACY: deltas written before that was true stored bare |dA|^2 and are marked
    power_domain='delta_amp_sq'; those get 1.  Rebuild them (step 03/04, seconds)
    and the special case goes away -- their dBke display is otherwise 10log10(4k)
    high (+24 dB at 3 GHz, +27 at 6) with a 3 dB/octave tilt.
    """
    from feature_sum import _load_grim, convention_scale   # noqa: PLC0415
    g = _load_grim(str(grim)) if isinstance(grim, str) else grim
    return convention_scale(g, frequencies_ghz)   # one source of truth


def field_amplitude(grid, grim_or_path) -> 'np.ndarray':
    """RcsGrid samples -> this repo's field amplitude [az, el, f, pol].

    ``grid`` is an RcsGrid loaded from ``grim_or_path``; the file is needed for
    its tags (the scale factor depends on which export it is).  When the file
    still carries rcs_amp_real/imag that array is returned directly -- exact,
    and it side-steps float32 power/phase.
    """
    from feature_sum import _load_grim            # noqa: PLC0415
    g = _load_grim(str(grim_or_path)) if isinstance(grim_or_path, str) else grim_or_path
    if "_amp" in g:
        return g["_amp"]
    s = amp_scale(g)
    return np.asarray(grid.rcs) / (s if s.size == 1 else s[None, None, :, None])


# -----------------------------------------------------------------------------
# hand a file over and get it back
# -----------------------------------------------------------------------------

def to_grid(path: 'str'):
    """Load one of this repo's .grim files as an RcsGrid.

    Safe to crop / mirror / join / plot with that tool.  Note the amplitude
    caveat in the module docstring before using ``grid.rcs`` as a field, and see
    ``from_grid`` for writing the result back.
    """
    return rcsgrid_class().load(str(path))


def from_grid(grid, out_path: 'str', *, amp: 'Optional[np.ndarray]' = None,
              history: 'str' = "") -> 'str':
    """Write an RcsGrid back as a .grim this repo can read.

    ``grid.save`` already preserves rcs_amp_real/imag for a grid that was loaded
    and not reshaped (the viewer tool carries unmodelled keys through), so this
    is only needed when the grid was DERIVED -- cropped, joined, interpolated --
    and therefore dropped the stale amplitude, or when the amplitude came from
    somewhere else.  Pass ``amp`` (complex, shaped like the grid) to supply it.
    """
    out = str(out_path)
    if not out.endswith(".grim"):
        out += ".grim"
    grid.save(out)
    if amp is None:
        return out
    a = np.asarray(amp, complex)
    exp = (len(grid.azimuths), len(grid.elevations), len(grid.frequencies),
           len(grid.polarizations))
    if a.shape != exp:
        raise ValueError(f"amp shape {a.shape} != grid shape {exp}.")
    with np.load(out, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["rcs_amp_real"] = a.real.astype(np.float64)
    d["rcs_amp_imag"] = a.imag.astype(np.float64)
    d["raw_complex_amplitude_preserved"] = True
    if history:
        d["history"] = str(d.get("history", "")) + f" | {history}"
    with open(out, "wb") as fh:
        np.savez(fh, **d)
    return out


# -----------------------------------------------------------------------------
# read the viewer tool's import formats as a point-scatterer pattern
# -----------------------------------------------------------------------------

_LOADERS = {".grim": "load", ".out": "load_out", ".ss": "load_ss",
            ".pio": "load_pio", ".csv": "load_theta_phi_csv",
            ".txt": "load_theta_phi_txt"}


def load_pattern_any(path: 'str', *, pol_map: 'Optional[Dict[str, str]]' = None,
                     convention_metadata: 'Optional[Dict[str, str]]' = None
                     ) -> 'Dict[str, Any]':
    """Read ANY format RcsGrid can import into the dict that
    ``feature_sum.point_scatterer_amplitude`` accepts as its ``pattern``.

    Formats: .grim, .out, .ss, .pio, theta/phi .csv, theta/phi .txt -- so a
    cavity solved by an external 3-D MoM can be placed on the body without first
    being rewritten into this repo's schema by hand.

    The pattern must still follow the placement CONVENTION documented on
    point_scatterer_amplitude: az/el are the CAVITY-frame spherical angles with
    +z the aperture normal, VV = theta-pol and HH = phi-pol about that normal,
    the phase origin is the cavity location, and the samples are the DIFFERENCE
    (featured - clean) of two runs on the same background.  A .grim's convention
    metadata is preserved.  Other formats cannot encode all of those facts, so
    pass the exact explicit ``convention_metadata`` returned by
    ``feature_sum.point_pattern_convention_metadata()`` only after verifying
    the external solver/export setup.  Untagged patterns are refused by the
    placement code rather than silently assuming an origin or time sign.

    An external file gives power (+ phase where it has it), so the complex
    samples come out as sqrt(power) * exp(j*phase) -- the 3-D field amplitude up
    to sqrt(4 pi), which is what a sigma-valued pattern means.  Files with no
    phase load as NaN, and a pattern without phase cannot be placed coherently:
    that raises rather than guessing zero.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"{path}: no RcsGrid loader for {ext!r} "
                         f"(have {sorted(_LOADERS)}).")
    grid_class = rcsgrid_class()
    if ext in {".csv", ".txt"}:
        fallback_name = (
            "load_theta_phi_csv" if ext == ".csv" else "load_theta_phi_txt"
        )
        if grid_class.has_SENTRi_signature(str(path)):
            grid = grid_class.read_SENTRi(str(path))
        else:
            grid = getattr(grid_class, fallback_name)(str(path))
    else:
        grid = getattr(grid_class, _LOADERS[ext])(str(path))
    amp = np.asarray(grid.rcs)
    if not np.all(np.isfinite(amp)):
        n = int(np.sum(~np.isfinite(amp)))
        raise ValueError(
            f"{path}: {n} sample(s) have no phase (NaN).  A point scatterer is "
            f"placed with a phase term, so a magnitude-only pattern cannot be "
            f"used -- export phase from the 3-D solver, or model the feature "
            f"with an envelope/power-added mode instead.")
    # an external file stores sigma, so F = sqrt(sigma/4pi) * exp(j phase); one of
    # our own .grim files already carries F exactly, so use it verbatim
    amp = amp / math.sqrt(4.0 * math.pi)
    if ext == ".grim":
        try:
            amp = field_amplitude(grid, str(path))
        except Exception:                     # noqa: BLE001  (not one of ours)
            pass
    pols = [str(p) for p in np.asarray(grid.polarizations).ravel()]
    if pol_map:
        pols = [pol_map.get(p, p) for p in pols]
    result = {"azimuths": np.asarray(grid.azimuths, float),
              "elevations": np.asarray(grid.elevations, float),
              "frequencies": np.asarray(grid.frequencies, float),
              "polarizations": np.asarray(pols, dtype=str),
              "amp": amp}
    metadata_keys = (
        "rcs_domain", "phase_reference", "amplitude_convention",
        "complex_field_domain", "pattern_frame_convention",
    )
    if ext == ".grim":
        with np.load(str(path), allow_pickle=False) as source:
            for key in metadata_keys:
                if key in source:
                    value = np.asarray(source[key])
                    if value.size == 1:
                        result[key] = str(value.reshape(-1)[0])
    if convention_metadata:
        for key in metadata_keys:
            if key in convention_metadata:
                result[key] = str(convention_metadata[key])
    return result


# -----------------------------------------------------------------------------
def describe(path: 'str') -> 'str':
    """One-screen summary of a .grim in both tools' terms: axes, tags, and the
    power/amplitude relationship that applies to it."""
    from feature_sum import _load_grim            # noqa: PLC0415
    g = _load_grim(str(path))
    s = amp_scale(g)
    fr = np.asarray(g["frequencies"], float)
    pw = np.asarray(g["rcs_power"], float)
    lines = [f"{os.path.basename(str(path))}",
             f"  axes      az {len(np.atleast_1d(g['azimuths']))} x el "
             f"{len(np.atleast_1d(g['elevations']))} x f {len(fr)} x pol "
             f"{[str(p) for p in np.asarray(g['polarizations']).ravel()]}",
             f"  tags      rcs_domain={str(g.get('rcs_domain',''))!r} "
             f"power_domain={str(g.get('power_domain',''))!r}",
             f"  units     {str(g.get('units',''))}",
             f"  power     {np.nanmin(pw):.4g} .. {np.nanmax(pw):.4g}",
             f"  sqrt(power)/|amp| = "
             + (f"{float(s[0]):.5f}" if s.size == 1
                else ", ".join(f"{f:g}GHz: {v:.5f}" for f, v in zip(fr, s))),
             "  -> RcsGrid power-domain work is valid as-is; divide RcsGrid.rcs "
             "by that factor",
             "     for this repo's field amplitude (grim_compat.field_amplitude)."]
    text = "\n".join(lines)
    print(text)
    return text
