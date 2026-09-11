"""Load and convert datasets through the GRIM viewer API."""
from ghost_backend.paths import backend_root

import math
import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

C0 = 299_792_458.0

_SIBLINGS = ("GRIM_Revised_2", "GRIM_Revised", "grim_revised_2")


_SEARCH_LEVELS = 4


def _grim_revised_dir() -> 'str':
    env = os.environ.get("GRIM_REVISED_PATH", "").strip()
    cands = [env] if env else []
    here = str(backend_root())
    for _ in range(_SEARCH_LEVELS):
        parent = os.path.dirname(here)
        if parent == here:
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


def amp_scale(grim: 'Dict[str, Any] | str',
              frequencies_ghz: 'Optional[Sequence[float]]' = None) -> 'np.ndarray':
    """Return the frequency-broadcastable factor from field amplitude to RCS magnitude.

    For sigma_2d the factor is 1/(2*sqrt(k)); for sigma_3d it is sqrt(4*pi). A declared
    power_domain of delta_amp_sq uses factor 1. Divide the grid complex samples by this
    factor to recover field amplitude.
    """
    from ghost_backend.assembly.fields import _load_grim, convention_scale
    g = _load_grim(str(grim)) if isinstance(grim, str) else grim
    return convention_scale(g, frequencies_ghz)


def field_amplitude(grid, grim_or_path) -> 'np.ndarray':
    """RcsGrid samples -> this repo's field amplitude [az, el, f, pol].

    ``grid`` is an RcsGrid loaded from ``grim_or_path``; the file is needed for
    its tags (the scale factor depends on which export it is).  When the file
    still carries rcs_amp_real/imag that array is returned directly -- exact,
    and it side-steps float32 power/phase.
    """
    from ghost_backend.assembly.fields import _load_grim
    g = _load_grim(str(grim_or_path)) if isinstance(grim_or_path, str) else grim_or_path
    if "_amp" in g:
        return g["_amp"]
    s = amp_scale(g)
    return np.asarray(grid.rcs) / (s if s.size == 1 else s[None, None, :, None])


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


    amp = amp / math.sqrt(4.0 * math.pi)
    if ext == ".grim":
        try:
            amp = field_amplitude(grid, str(path))
        except Exception as exc:
            declared_quantity = str(
                (getattr(grid, "units", {}) or {}).get("rcs_linear_quantity", "")
            ).strip()
            if declared_quantity:
                raise ValueError(
                    f"{path} declares rcs_linear_quantity={declared_quantity!r}, "
                    "but its field-amplitude convention cannot be determined; "
                    "refusing to guess sigma_3d normalization"
                ) from exc


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


def describe(path: 'str') -> 'str':
    """One-screen summary of a .grim in both tools' terms: axes, tags, and the
    power/amplitude relationship that applies to it."""
    from ghost_backend.assembly.fields import _load_grim
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
