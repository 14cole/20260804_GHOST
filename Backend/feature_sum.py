#!/usr/bin/env python3
"""
Vehicle signature = BoR body  +  line-expanded surface features.

Pipeline the user drives:

  1.  Solve a 2D CROSS-SECTION of the joint twice with rcs_solver: the CLEAN
      background and the FEATURED one (same mesh, same units, same pols),
      export each to .grim.
  2.  ``make_delta_grim(clean, featured, "seam.grim")`` -- coherently subtracts
      the complex amplitudes into a reusable, VEHICLE-INDEPENDENT delta .grim.
  3.  For each place the joint appears on a vehicle, supply its perimeter file
      (segmented ``x1 y1 z1 x2 y2 z2`` in the vehicle frame) and hand a list of
      placements to ``sum_features`` together with the vehicle's BoR result.
      Any number of deltas at any number of locations combine in one pass.

Why a coupon delta can be reused under the line-model assumptions: both
rcs_solver (2D) and the BoR solver use the same origin and
exp(+2jk d.r') two-way translation factor. Thus its *modeled* placement phase
is consistent without a hand-applied range correction. Physical reuse still
requires the same feature cross-section, clean local material/coating stack,
frequency, units, phase origin, polarization convention, and sufficiently
similar local curvature; body-feature coupling remains omitted. The remaining
inter-solver constants are local-TE and local-TM phase factors
(line_expand.PSI_VV_DEG / PSI_HH_DEG), measured jointly by the ring gate and
applied before their polarization projections are added.

Conventions the delta MUST honour (all satisfied automatically if the coupon is
drawn per the guide below):

  * The 2D coupon's OUTER FACE is at y = 0 with the seam centred on x = 0, so
    the delta's phase centre is the seam line on the outer skin -- the same
    point the perimeter coordinates trace.
  * The coupon's cut angle is the solver's elevation angle, 90 deg = normal
    incidence on the outer face.  (Drawn outer-face-up, this is automatic.)
  * TM (== HH / phi-pol) = E along the seam; TE (== VV / theta-pol) = E across.
  * The CLEAN coupon must equal the body the BoR solves (bare PEC ground plane,
    or a MAGRAM-coated one) so the subtraction cancels everything but the joint.

The canonical PEC-groove benchmark supports broadside +/-
VALIDITY_HALF_ANGLE_DEG of the LOCAL edge under its measured error limits.
A smooth closed perimeter often concentrates its asymptotic contribution near
``d.t_hat == 0``, but that does not itself guarantee near-normal cut incidence
or bound corner/end/junction contributions.
"""

import json
import hashlib
import math
import os
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from geometry_io import material_sidecar_paths
from line_expand import (C0, PSI_HH_DEG, PSI_VV_DEG, SeamCoefficients,
                         _pol_unit_vectors, combine, dbsm, expand_perimeter,
                         read_perimeter_txt, surface_of_revolution_normal)

PathOrList = Union[str, Sequence[str]]

PHYSICAL_3D_AMPLITUDE_CONVENTION = (
    "F physical far-field amplitude; sigma_3d=4*pi*|F|^2"
)
PHYSICAL_2D_PHASE_REFERENCE = (
    "origin=(0,0), convention=exp(+jwt); stored complex field is the "
    "2D layer-potential bare-integral amplitude B. The coefficient "
    "in u_s~exp(-j(kr-pi/4))/sqrt(8*pi*k*r)*A is A=j*B."
)
PHYSICAL_2D_AMPLITUDE_CONVENTION = (
    "A_physical_asymptotic = +j * B_stored"
)
PHYSICAL_2D_FIELD_DOMAIN = (
    "2d_layer_potential_bare_integral_amplitude_B"
)
BOR_BODY_PHASE_REFERENCE = (
    "drawing origin (0,0,0), exp(+jwt), monostatic exp(+2jk d.r)"
)
BOR_BODY_FIELD_DOMAIN = (
    "bor_far_field_amplitude_F, sigma = 4 pi |F|^2"
)
RADAR_COMPONENT_PHASE_REFERENCE = (
    "origin=(0,0,0) vehicle frame, convention=exp(+jwt), "
    "radar earth-frame V/H monostatic amplitude"
)
RADAR_COMPONENT_FIELD_DOMAIN = (
    "coherent_radar_frame_far_field_amplitude"
)
DELTA_FIELD_DOMAIN = "featured_minus_clean_far_field_amplitude_delta"
DELTA_PHASE_SUFFIX = (
    "; coherent subtraction=featured-clean; placement phase center is the "
    "seam line on the coupon outer face y=0"
)

# A compact pattern is reusable only if its phase origin and field convention
# are explicitly tied to the placement inputs accepted by
# point_scatterer_amplitude.  These exact strings are intentionally strict:
# coherent placement cannot repair a hidden origin, time-sign, or 4*pi error.
POINT_PATTERN_PHASE_REFERENCE = (
    "origin=(0,0,0) at aperture phase center in cavity frame, "
    "convention=exp(+jwt)"
)
POINT_PATTERN_AMPLITUDE_CONVENTION = (
    "F physical featured-minus-clean far-field amplitude; "
    "sigma_3d=4*pi*|F|^2"
)
POINT_PATTERN_FIELD_DOMAIN = (
    "featured_minus_clean_cavity_frame_far_field_amplitude_F"
)
POINT_PATTERN_FRAME_CONVENTION = (
    "cavity spherical: +z=aperture outward; az=atan2(y,x); el=asin(z); "
    "VV=theta; HH=phi; VH=HV"
)


def point_pattern_convention_metadata() -> 'Dict[str, str]':
    """Exact metadata required for a compact 3-D differential pattern."""
    return {
        "rcs_domain": "delta",
        "phase_reference": POINT_PATTERN_PHASE_REFERENCE,
        "amplitude_convention": POINT_PATTERN_AMPLITUDE_CONVENTION,
        "complex_field_domain": POINT_PATTERN_FIELD_DOMAIN,
        "pattern_frame_convention": POINT_PATTERN_FRAME_CONVENTION,
    }


def geometry_input_fingerprint(path: 'str', geometry_units: 'str') -> 'str':
    """SHA-256 of all inputs that can change one geometry solve.

    A .geo may refer to explicit CSV (or legacy ``mat.<flag>``) files beside it, and
    the same coordinates mean different physical sizes under different unit
    settings.  Body caches must bind to all three: geometry bytes, every
    sidecar material table, and the declared units.
    """
    geo = os.path.abspath(str(path))
    if not os.path.isfile(geo):
        raise FileNotFoundError(geo)
    files = [geo] + material_sidecar_paths(geo)
    h = hashlib.sha256()
    h.update(b"rcs-solver-input-v2\0")
    h.update(str(geometry_units).strip().lower().encode("utf-8") + b"\0")
    for filename in files:
        h.update(os.path.basename(filename).encode("utf-8") + b"\0")
        with open(filename, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# .grim I/O (mirrors grim_io.py's NPZ layout)
# -----------------------------------------------------------------------------

def convention_scale(grim: 'Dict[str, Any]',
                     frequencies_ghz=None) -> 'np.ndarray':
    """``sqrt(rcs_power) / |amp|`` for one of this repo's .grim files, broadcastable
    over the frequency axis.  There are only TWO data types, read off the file's
    own tag -- never measured:

      rcs_linear_quantity == 'sigma_2d'   sigma_2d = |A|^2/(4k)  -> 1/(2 sqrt(k))
      rcs_linear_quantity == 'sigma_3d'   sigma = 4 pi |F|^2     -> sqrt(4 pi)

    LEGACY: deltas written before the 2-D convention was applied to their power
    are marked power_domain='delta_amp_sq' and stored bare |dA|^2 -> 1.
    """
    if str(grim.get("power_domain", "")) == "delta_amp_sq":
        if str(grim.get("rcs_domain", "")) != "delta":
            raise ValueError(
                "power_domain='delta_amp_sq' is valid only for an explicitly "
                "tagged legacy delta.")
        return np.ones(1)
    try:
        units = json.loads(str(grim.get("units", "")))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "GRIM units metadata must be valid JSON with an explicit "
            "rcs_linear_quantity.") from exc
    if not isinstance(units, dict):
        raise ValueError("GRIM units metadata must decode to an object.")
    quantity = str(units.get("rcs_linear_quantity", "")).strip().lower()
    if quantity == "sigma_2d":
        fr = np.asarray(frequencies_ghz if frequencies_ghz is not None
                        else grim["frequencies"], dtype=float)
        if fr.ndim != 1 or fr.size == 0 or not np.all(np.isfinite(fr)) \
                or np.any(fr <= 0.0):
            raise ValueError(
                "sigma_2d GRIM frequencies must be a nonempty positive "
                "finite one-dimensional array.")
        k = 2.0 * math.pi * fr * 1e9 / C0
        return 1.0 / (2.0 * np.sqrt(k))
    if quantity == "sigma_3d":
        return np.full(1, math.sqrt(4.0 * math.pi))
    raise ValueError(
        "GRIM units.rcs_linear_quantity must be exactly 'sigma_2d' or "
        "'sigma_3d'; an unknown normalization cannot be combined coherently.")


def _load_grim(path: 'str') -> 'Dict[str, Any]':
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}

    axes = {}
    for key in ("azimuths", "elevations", "frequencies"):
        if key not in d:
            raise ValueError(f"{path}: missing GRIM axis {key!r}.")
        values = np.asarray(d[key], dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{path}: {key} must be a nonempty 1-D array.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {key} contains NaN or infinity.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{path}: {key} must be strictly increasing.")
        axes[key] = values
    if np.any(axes["frequencies"] <= 0.0):
        raise ValueError(f"{path}: frequencies must be positive.")
    if "polarizations" not in d:
        raise ValueError(f"{path}: missing GRIM polarizations.")
    pols = np.asarray(d["polarizations"]).astype(str)
    if pols.ndim != 1 or pols.size == 0:
        raise ValueError(f"{path}: polarizations must be a nonempty 1-D array.")
    if any(not p.strip() for p in pols) or len(set(pols.tolist())) != len(pols):
        raise ValueError(
            f"{path}: polarization labels must be nonempty and unique.")

    shape = (
        len(axes["azimuths"]),
        len(axes["elevations"]),
        len(axes["frequencies"]),
        len(pols),
    )

    def _field(key: 'str', *, nonnegative: 'bool' = False) -> 'np.ndarray':
        if key not in d:
            raise ValueError(f"{path}: missing GRIM field {key!r}.")
        values = np.asarray(d[key], dtype=float)
        if values.shape != shape:
            raise ValueError(
                f"{path}: {key} shape {values.shape} does not match axes "
                f"{shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {key} contains NaN or infinity.")
        if nonnegative and np.any(values < 0.0):
            raise ValueError(f"{path}: {key} contains negative values.")
        return values

    power = _field("rcs_power", nonnegative=True)
    phase = _field("rcs_phase")
    has_real = "rcs_amp_real" in d
    has_imag = "rcs_amp_imag" in d
    if has_real != has_imag:
        raise ValueError(
            f"{path}: complex amplitude must provide both rcs_amp_real and "
            "rcs_amp_imag.")
    if bool(d.get("raw_complex_amplitude_preserved", False)) and not has_real:
        raise ValueError(
            f"{path}: raw_complex_amplitude_preserved is true but the raw "
            "amplitude arrays are absent.")

    scale = np.asarray(
        convention_scale(d, axes["frequencies"]), dtype=float)
    scale_grid = (
        scale[None, None, :, None] if scale.size > 1 else float(scale.ravel()[0])
    )
    if has_real:
        real = _field("rcs_amp_real")
        imag = _field("rcs_amp_imag")
        amp = real + 1j * imag
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_real = real * scale_grid
            scaled_imag = imag * scale_grid
            expected_power = (
                scaled_real * scaled_real + scaled_imag * scaled_imag
            )
        if not np.all(np.isfinite(expected_power)):
            raise ValueError(
                f"{path}: complex amplitude is too large to form finite "
                "physical power under the declared normalization."
            )
        tolerance = (
            16.0 * np.finfo(np.float32).eps
            * np.maximum(expected_power, power)
            + np.finfo(np.float32).tiny
        )
        if np.any(np.abs(power - expected_power) > tolerance):
            raise ValueError(
                f"{path}: rcs_power is inconsistent with its complex "
                "amplitude and declared 2-D/3-D normalization.")
        live = (
            np.abs(real) > np.finfo(np.float32).tiny
        ) | (
            np.abs(imag) > np.finfo(np.float32).tiny
        )
        if np.any(live):
            phase_error = np.abs(np.angle(np.exp(
                1j * (phase[live] - np.angle(amp[live]))
            )))
            if float(np.max(phase_error)) > 2.0e-5:
                raise ValueError(
                    f"{path}: rcs_phase is inconsistent with its complex "
                    "amplitude.")
        d["_amp"] = amp
    else:
        # A file that went through the viewer's dataset model: RcsGrid stores
        # power + phase, so any DERIVED grid (a crop, a join, a coherent subtract
        # -- e.g. a delta you built in the GUI) has no rcs_amp_* arrays to write.
        # Nothing was stripped: power and phase carry the same complex number up
        # to the file's own convention factor, so rebuild it rather than refuse.
        # (Two float32 round trips, so expect ~1e-6 relative, not exact.)
        amp = np.sqrt(power) * np.exp(1j * phase)
        d["_amp"] = amp / scale_grid
        d["_amp_from_power_phase"] = True
    d["_pol_primary"] = str(d.get("polarization_alias_primary", ""))
    return d


def _canon_pol(label: 'str') -> 'str':
    """Return 'TM' or 'TE' for any accepted alias."""
    t = str(label).strip().upper()
    if t in {"TM", "HH", "H", "HORIZONTAL"}:
        return "TM"
    if t in {"TE", "VV", "V", "VERTICAL"}:
        return "TE"
    raise ValueError(f"unrecognized polarization label {label!r}.")


def _metadata_text(grim: 'Dict[str, Any]', key: 'str', label: 'str',
                   *, required: 'bool' = True) -> 'str':
    if key not in grim:
        if required:
            raise ValueError(
                f"{label}: missing {key!r}; coherent fields cannot be combined "
                "without explicit phase and amplitude conventions.")
        return ""
    arr = np.asarray(grim[key])
    if arr.size != 1:
        raise ValueError(f"{label}: metadata {key!r} must be scalar.")
    value = str(arr.reshape(-1)[0]).strip()
    if required and not value:
        raise ValueError(f"{label}: metadata {key!r} is empty.")
    return value


def _units_metadata(grim: 'Dict[str, Any]', label: 'str') -> 'Dict[str, Any]':
    """Return one explicit GRIM units object for a role-specific check."""
    text = _metadata_text(grim, "units", label)
    try:
        units = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: units metadata is not valid JSON.") from exc
    if not isinstance(units, dict):
        raise ValueError(f"{label}: units metadata must decode to an object.")
    return units


def _require_units(grim: 'Dict[str, Any]', label: 'str', *,
                   linear_quantity: 'str', log_unit: 'str') -> 'Dict[str, Any]':
    units = _units_metadata(grim, label)
    got_quantity = str(
        units.get("rcs_linear_quantity", "")).strip().lower()
    got_log_unit = str(units.get("rcs_log_unit", "")).strip()
    if got_quantity != linear_quantity or got_log_unit != log_unit:
        raise ValueError(
            f"{label}: require {linear_quantity}/{log_unit} normalization; "
            f"got {got_quantity or '<missing>'}/"
            f"{got_log_unit or '<missing>'}.")
    return units


def _require_linear_quantity(grim, label, expected):
    """Require the dimensional linear field; display-unit labels are optional."""
    units = _units_metadata(grim, label)
    got = str(units.get("rcs_linear_quantity", "")).strip().lower()
    if got != expected:
        raise ValueError(
            f"{label}: require units.rcs_linear_quantity={expected!r}; "
            f"got {got or '<missing>'!r}."
        )
    # A GRIM axis is standardized as degrees/GHz. Missing descriptive keys are
    # harmless, but an explicit different unit must never be interpreted as
    # the standard axis silently.
    for key, standard in (
        ("azimuth", "deg"), ("elevation", "deg"), ("frequency", "ghz")
    ):
        if key in units and str(units[key]).strip().lower() != standard:
            raise ValueError(
                f"{label}: GRIM {key} values must be stored in {standard}; "
                f"got units.{key}={units[key]!r}."
            )
    return units


def _require_singleton_zero_elevation(
        grim: 'Dict[str, Any]', label: 'str'
) -> 'None':
    elevations = np.asarray(grim["elevations"], dtype=float)
    if elevations.shape != (1,) or elevations[0] != 0.0:
        raise ValueError(
            f"{label}: this 2-D/BoR artifact requires the singleton "
            "elevation axis [0.0].")


def _require_exact_metadata(
        grim: 'Dict[str, Any]', label: 'str', expected: 'Dict[str, str]') -> 'None':
    for key, want in expected.items():
        got = _metadata_text(grim, key, label)
        if got != want:
            raise ValueError(
                f"{label}: {key} is {got!r}; require {want!r}.")


def _require_2d_source_semantics(
        grim: 'Dict[str, Any]', label: 'str') -> 'None':
    """Require an undifferenced solver coupon/section field."""
    if _metadata_text(grim, "rcs_domain", label) != "power_phase":
        raise ValueError(
            f"{label}: a 2-D source coefficient must have "
            "rcs_domain='power_phase'.")
    if _metadata_text(grim, "power_domain", label) != "linear_rcs":
        raise ValueError(
            f"{label}: a 2-D source coefficient must have "
            "power_domain='linear_rcs'.")
    _require_units(
        grim, label, linear_quantity="sigma_2d", log_unit="dBke")
    _require_singleton_zero_elevation(grim, label)
    _require_exact_metadata(grim, label, {
        "phase_reference": PHYSICAL_2D_PHASE_REFERENCE,
        "amplitude_convention": PHYSICAL_2D_AMPLITUDE_CONVENTION,
        "complex_field_domain": PHYSICAL_2D_FIELD_DOMAIN,
    })


def _canonical_table_channels(
        grim: 'Dict[str, Any]', label: 'str') -> 'Dict[str, int]':
    """Map TM/TE aliases without allowing two labels to overwrite a channel."""
    channels: 'Dict[str, int]' = {}
    for index, raw in enumerate(np.asarray(grim["polarizations"]).ravel()):
        try:
            canonical = _canon_pol(str(raw))
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        if canonical in channels:
            raise ValueError(
                f"{label}: polarization aliases collide on canonical "
                f"channel {canonical}; each physical channel must occur once.")
        channels[canonical] = index
    return channels


def _require_complete_2d_channels(
        channels: 'Dict[str, Any]', label: 'str') -> 'None':
    required = {"TM", "TE"}
    got = set(channels)
    if got != required:
        raise ValueError(
            f"{label}: require exactly the complete TM/TE channel pair "
            f"(HH aliases TM and VV aliases TE); got {sorted(got)}.")


def _coherent_input_convention(
        entries: 'Sequence[Tuple[str, Dict[str, Any]]]'
) -> 'Tuple[str, str, str]':
    """Return one common (phase, amplitude, field-domain) convention."""
    signatures = []
    for label, grim in entries:
        phase = _metadata_text(grim, "phase_reference", label)
        domain = _metadata_text(grim, "complex_field_domain", label)
        # Older solver exports did not have the dedicated key.  An identical,
        # explicit complex_field_domain still lets two legacy files be checked
        # against each other; mixing it with a newly labelled convention does
        # not.
        amplitude = _metadata_text(
            grim, "amplitude_convention", label, required=False)
        signatures.append((phase, amplitude, domain, label))
    reference = signatures[0][:3]
    for signature in signatures[1:]:
        if signature[:3] != reference:
            ref_amplitude = reference[1] or "<legacy unspecified>"
            got_amplitude = signature[1] or "<legacy unspecified>"
            raise ValueError(
                "clean and featured fields have incompatible coherent-field "
                f"conventions: {signatures[0][3]} uses "
                f"(phase={reference[0]!r}, amplitude={ref_amplitude!r}, "
                f"domain={reference[2]!r}), while "
                f"{signature[3]} uses (phase={signature[0]!r}, "
                f"amplitude={got_amplitude!r}, "
                f"domain={signature[2]!r}).")
    return reference


def _amp_tables(grims: 'Sequence[Dict[str, Any]]') -> 'Dict[str, Any]':
    """Merge one or more single-cut grims into per-(pol) amplitude tables keyed
    by 'TM'/'TE', on a shared (azimuth, frequency) grid.  azimuth == 2D cut
    angle, elevation is the singleton 0.
    """
    az = np.asarray(grims[0]["azimuths"], dtype=float)
    fr = np.asarray(grims[0]["frequencies"], dtype=float)
    tables: 'Dict[str, np.ndarray]' = {}
    for source_index, g in enumerate(grims):
        if not (np.array_equal(np.asarray(g["azimuths"], float), az)
                and np.array_equal(np.asarray(g["frequencies"], float), fr)
                and np.array_equal(
                    np.asarray(g["elevations"], float), np.array([0.0]))):
            raise ValueError(
                "grim files do not share one (azimuth, singleton elevation, "
                "frequency) grid; inputs must be solved on identical sweeps.")
        channels = _canonical_table_channels(
            g, f"GRIM input {source_index}")
        amp = g["_amp"]                                   # [az, el, f, pol]
        for key, j in channels.items():
            # per-channel primary alias ('HH'->TM, 'VV'->TE) is always correct,
            # for both single-pol solver grims and multi-pol delta grims
            if key in tables:
                raise ValueError(
                    f"GRIM inputs provide canonical channel {key} more than "
                    "once; duplicate aliases cannot be resolved safely.")
            tables[key] = amp[:, 0, :, j]                 # [az, f]
    return {"azimuths": az, "frequencies": fr, "tables": tables}


def make_delta_grim(clean: 'PathOrList', featured: 'PathOrList', out_path: 'str',
                    history: 'str' = "") -> 'str':
    """Coherently subtract featured - clean 2D amplitudes -> a delta .grim.

    ``clean`` / ``featured`` are each a single .grim or a list of them (e.g. one
    per polarization).  Matching pols on a shared (angle, frequency) grid are
    differenced; the result carries the SAME complex-amplitude layout so it can
    be inspected with the usual grim tools, tagged rcs_domain='delta'.
    """
    clean_paths = [clean] if isinstance(clean, str) else list(clean)
    featured_paths = [featured] if isinstance(featured, str) else list(featured)
    if not clean_paths or not featured_paths:
        raise ValueError("clean and featured must each contain at least one .grim.")
    cg = [_load_grim(p) for p in clean_paths]
    fg = [_load_grim(p) for p in featured_paths]
    for path, grim in zip(clean_paths, cg):
        _require_2d_source_semantics(grim, f"clean {path}")
    for path, grim in zip(featured_paths, fg):
        _require_2d_source_semantics(grim, f"featured {path}")
    phase_ref, amplitude_convention, source_field_domain = (
        _coherent_input_convention(
            [(f"clean {p}", g) for p, g in zip(clean_paths, cg)]
            + [(f"featured {p}", g) for p, g in zip(featured_paths, fg)])
    )
    ct, ft = _amp_tables(cg), _amp_tables(fg)
    if not (np.array_equal(ct["azimuths"], ft["azimuths"])
            and np.array_equal(ct["frequencies"], ft["frequencies"])):
        raise ValueError("clean and featured are on different (angle, frequency) grids.")
    clean_pols = set(ct["tables"])
    featured_pols = set(ft["tables"])
    if clean_pols != featured_pols:
        raise ValueError(
            "clean and featured polarization sets differ after canonical "
            f"TM/HH and TE/VV aliasing: clean={sorted(clean_pols)}, "
            f"featured={sorted(featured_pols)}.")
    _require_complete_2d_channels(ct["tables"], "clean inputs")
    _require_complete_2d_channels(ft["tables"], "featured inputs")
    pols = sorted(clean_pols)

    az, fr = ct["azimuths"], ct["frequencies"]
    shape = (len(az), 1, len(fr), len(pols))
    amp = np.zeros(shape, dtype=complex)
    primaries = []
    for j, p in enumerate(pols):
        amp[:, 0, :, j] = ft["tables"][p] - ct["tables"][p]
        primaries.append("HH" if p == "TM" else "VV")

    units = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
             "rcs_log_unit": "dBke", "rcs_linear_quantity": "sigma_2d"}
    # A delta is a 2-D quantity, so rcs_power follows the 2-D convention like any
    # other 2-D cut: sigma_2d = |amp|^2 / (4k), a scattering width in metres (see
    # rcs_solver.py's normalization block).  There are only TWO conventions in the
    # toolchain -- 2-D (sigma_2d, dBke) and 3-D (sigma = 4 pi |amp|^2, dBsm) -- and
    # 'delta' is a separate, ORTHOGONAL axis: it says the samples are a
    # DIFFERENCE (featured - clean), not what units they are in.  Storing bare
    # |dA|^2 here (as this did before) made a dBke plot read 10log10(4k) high:
    # +24 dB at 3 GHz, +27 at 6, and tilted 3 dB/octave.
    k_per_freq = 2.0 * math.pi * np.asarray(fr, dtype=float) * 1e9 / C0
    sigma_2d = np.abs(amp) ** 2 / (4.0 * k_per_freq)[None, None, :, None]
    out = out_path if out_path.lower().endswith(".grim") else out_path + ".grim"
    payload = dict(
        azimuths=az, elevations=np.array([0.0]), frequencies=fr,
        polarizations=np.asarray(primaries, dtype=str),
        polarization_alias_primary=",".join(pols),
        polarization_aliases_json=json.dumps(pols),
        rcs_power=sigma_2d.astype(np.float32),
        rcs_phase=np.angle(amp).astype(np.float32),
        rcs_domain="delta", power_domain="linear_rcs",
        source_path="", history=(history or "make_delta_grim: featured - clean"),
        units=json.dumps(units),
        phase_reference=phase_ref + DELTA_PHASE_SUFFIX,
        amplitude_convention=(
            amplitude_convention or
            f"legacy convention identified by complex_field_domain="
            f"{source_field_domain}"),
        raw_complex_amplitude_preserved=True,
        rcs_amp_real=amp.real.astype(np.float64),
        rcs_amp_imag=amp.imag.astype(np.float64),
        complex_field_domain=DELTA_FIELD_DOMAIN,
    )
    from grim_io import _save_grim_npz
    return os.path.abspath(_save_grim_npz(payload, out))


def _coeffs_from_tables(tab, frequency_ghz, tol_ghz, label):
    _require_complete_2d_channels(tab["tables"], label)
    fr = tab["frequencies"]
    j = int(np.argmin(np.abs(fr - float(frequency_ghz))))
    if abs(fr[j] - float(frequency_ghz)) > tol_ghz:
        raise ValueError(f"{label}: no frequency {frequency_ghz} GHz (has {fr.tolist()}). "
                         f"Solve the cross-section at the frequencies you combine at.")
    phi = tab["azimuths"]
    a_tm = tab["tables"]["TM"][:, j]
    a_te = tab["tables"]["TE"][:, j]
    return SeamCoefficients(float(fr[j]), phi, a_tm, a_te, label=label)


def load_coefficients_from_grim(paths: 'PathOrList', frequency_ghz: 'float',
                                tol_ghz: 'float' = 1e-6) -> 'SeamCoefficients':
    """Load a FULL-OBJECT coefficient (a wing/fin airfoil 2D solve) from plain
    2D monostatic .grim export(s) -- the wing analog of load_seam_from_grim
    (which is for differential deltas).  Accepts one multi-pol file or a list
    (e.g. one per polarization)."""
    source_paths = [paths] if isinstance(paths, str) else list(paths)
    if not source_paths:
        raise ValueError("coefficient inputs must contain at least one .grim.")
    grims = [_load_grim(p) for p in source_paths]
    for path, grim in zip(source_paths, grims):
        _require_2d_source_semantics(grim, f"coefficient {path}")
    return _coeffs_from_tables(_amp_tables(grims), frequency_ghz, tol_ghz,
                               os.path.basename(str(paths)))


def _signed_seam(coefficients: 'SeamCoefficients', delta_sign: 'float'
                 ) -> 'SeamCoefficients':
    """Apply the declared subtraction order without changing interpolation."""
    sign = float(delta_sign)
    if not math.isfinite(sign) or sign not in (-1.0, 1.0):
        raise ValueError("delta_sign must be exactly +1 or -1.")
    if sign == 1.0:
        return coefficients
    return SeamCoefficients(
        coefficients.frequency_ghz,
        coefficients.phi_deg,
        -coefficients.dA_tm,
        -coefficients.dA_te,
        label=coefficients.label,
    )


def load_seam_from_grim(path: 'str', frequency_ghz: 'float',
                        tol_ghz: 'float' = 1e-6, *,
                        declared_coherent_delta: 'bool' = False,
                        delta_sign: 'float' = 1.0,
                        _grim_payload=None) -> 'SeamCoefficients':
    """Load a delta .grim at one frequency into a SeamCoefficients.

    Both physical channels are required.  HH is the accepted alias for TM and
    VV is the accepted alias for TE.
    """
    g = _load_grim(path) if _grim_payload is None else _grim_payload
    dom = str(g.get("rcs_domain", "")).strip()
    normalized_domain = dom.lower().replace("-", "_")
    if declared_coherent_delta:
        # Listing a file as a LINE_FEATURES dataset is the role declaration.
        # GUI derived grids often preserve the numerical complex field as
        # power+phase while dropping or retaining stale source semantics. The
        # declaration supersedes those descriptive strings. Dimensional units
        # and the numerical power/phase normalization remain strict below.
        pass
    else:
        if normalized_domain != "delta":
            raise ValueError(
                f"{path}: rcs_domain is {dom!r}, not 'delta'. A direct API "
                "call must provide a canonical delta or set "
                "declared_coherent_delta=True after verifying that the file "
                "is featured minus clean."
            )
        if _metadata_text(g, "power_domain", path) != "linear_rcs":
            raise ValueError(
                f"{path}: a production seam delta must have "
                "power_domain='linear_rcs'.")
    if declared_coherent_delta:
        _require_linear_quantity(g, path, "sigma_2d")
    else:
        _require_units(
            g, path, linear_quantity="sigma_2d", log_unit="dBke")
    _require_singleton_zero_elevation(g, path)
    expected_phase_reference = (
        PHYSICAL_2D_PHASE_REFERENCE + DELTA_PHASE_SUFFIX)
    expected_metadata = {
        "complex_field_domain": DELTA_FIELD_DOMAIN,
        "phase_reference": expected_phase_reference,
        "amplitude_convention": PHYSICAL_2D_AMPLITUDE_CONVENTION,
    }
    for key, expected_value in expected_metadata.items():
        if declared_coherent_delta:
            continue
        got = _metadata_text(g, key, path)
        if got != expected_value:
            raise ValueError(
                f"{path}: incompatible line-delta {key}: got {got!r}; "
                f"require {expected_value!r}.")
    # Also rejects absent/unknown dimensional normalization.  Legacy
    # delta_amp_sq artifacts are intentionally not accepted by the production
    # placement path: they are not 2-D scattering widths and must be rebuilt.
    scale = convention_scale(g)
    expected = 1.0 / (
        2.0 * np.sqrt(
            2.0 * math.pi * np.asarray(g["frequencies"], float) * 1.0e9 / C0
        )
    )
    if not np.allclose(scale, expected, rtol=1.0e-14, atol=0.0):
        raise ValueError(f"{path}: a seam delta must use sigma_2d normalization.")
    coefficients = _coeffs_from_tables(
        _amp_tables([g]), frequency_ghz, tol_ghz, os.path.basename(path)
    )
    return _signed_seam(coefficients, delta_sign)


def tag_as_delta(path: 'str', *, source_2d_grim: 'Optional[str]' = None) -> 'str':
    """Mark an existing .grim as a differential (featured - clean) delta.

    For a delta built OUTSIDE this pipeline -- typically a coherent subtract in
    the viewer. A derived viewer grid may lose convention metadata, so
    ``source_2d_grim`` must name one of the verified solver inputs unless the
    target already carries the complete delta convention. No field array is
    touched. Do NOT use this on a whole-object solve.
    """
    loaded = _load_grim(str(path))
    d = {key: value for key, value in loaded.items()
         if not key.startswith("_")}
    # Materialize the reconstructed field so the attested delta remains
    # self-contained and future coherent use does not incur another
    # power/phase round trip.
    target_amp = np.asarray(loaded["_amp"], dtype=complex)
    d["rcs_amp_real"] = target_amp.real.astype(np.float64)
    d["rcs_amp_imag"] = target_amp.imag.astype(np.float64)
    d["raw_complex_amplitude_preserved"] = np.asarray(True)
    if source_2d_grim is not None:
        source = _load_grim(str(source_2d_grim))
        _require_2d_source_semantics(
            source, f"source_2d_grim {source_2d_grim}")
        d["phase_reference"] = np.asarray(
            _metadata_text(source, "phase_reference", str(source_2d_grim))
            + DELTA_PHASE_SUFFIX
        )
        d["amplitude_convention"] = np.asarray(
            _metadata_text(
                source, "amplitude_convention", str(source_2d_grim))
        )
        d["complex_field_domain"] = np.asarray(DELTA_FIELD_DOMAIN)
    else:
        missing = [
            key for key in (
                "phase_reference", "amplitude_convention",
                "complex_field_domain")
            if key not in d or not str(np.asarray(d[key]).reshape(-1)[0]).strip()
        ]
        if missing:
            raise ValueError(
                "tag_as_delta cannot invent coherent-field conventions; pass "
                "source_2d_grim=<one verified clean/featured solver GRIM>. "
                f"Missing {missing}.")
    was = str(d.get("rcs_domain", ""))
    d["rcs_domain"] = np.asarray("delta")
    d["history"] = np.asarray(f"{str(d.get('history', ''))} | tag_as_delta: "
                              f"rcs_domain {was!r} -> 'delta'")
    from grim_io import _save_grim_npz
    saved = _save_grim_npz(d, str(path))
    # Reuse the same strict semantic checks as production loading.
    load_seam_from_grim(saved, float(np.asarray(d["frequencies"], float)[0]))
    return saved


# -----------------------------------------------------------------------------
# Vehicle body from a .geo (WITH MATERIALS) -> combine-ready body + generatrix
# -----------------------------------------------------------------------------

def _stitch_chains(chains):
    """Order directed ChainSpec chains head-to-tail into one (rho, z) polyline.

    The air-facing BoR profile is a physical boundary, so disconnected pieces,
    branches, and ambiguous ordering are fatal.  Silently concatenating them
    would invent straight surface spans that are absent from the geometry and
    would corrupt feature normals/placement.
    """
    chains = list(chains)
    if not chains:
        raise ValueError("cannot stitch an empty set of air-facing chains.")
    if len(chains) == 1:
        return [tuple(p) for p in chains[0].points]
    span = max(1.0, max(abs(v) for c in chains for p in c.points for v in p))
    tol = 1e-6 * span

    def key(p):
        return (round(p[0] / tol), round(p[1] / tol))

    start = {}
    end = {}
    for c in chains:
        if len(c.points) < 2:
            raise ValueError(f"air-facing chain {c.name!r} has fewer than two points.")
        ks, ke = key(c.points[0]), key(c.points[-1])
        if ks in start:
            raise ValueError(
                "air-facing surface branches or is ambiguously ordered: "
                f"{start[ks].name!r} and {c.name!r} start at the same point.")
        if ke in end:
            raise ValueError(
                "air-facing surface branches or is ambiguously ordered: "
                f"{end[ke].name!r} and {c.name!r} end at the same point.")
        start[ks] = c
        end[ke] = c

    heads = [c for c in chains if key(c.points[0]) not in end]
    if len(heads) != 1:
        raise ValueError(
            "air-facing TYPE 2/3 segments do not form one directed head-to-tail "
            "BoR profile; check disconnected pieces and segment endpoint order.")

    order = [heads[0]]
    seen = {id(heads[0])}
    while True:
        nxt = start.get(key(order[-1].points[-1]))
        if nxt is None:
            break
        if id(nxt) in seen:
            raise ValueError(
                "air-facing TYPE 2/3 segments form a closed/looped chain in the "
                "(rho, z) plane; a BoR outer generatrix must be one open run.")
        order.append(nxt)
        seen.add(id(nxt))
    if len(order) != len(chains):
        missing = [c.name for c in chains if id(c) not in seen]
        raise ValueError(
            "air-facing TYPE 2/3 segments split into disconnected profiles; "
            f"unreached chain(s): {missing}.")

    pts = list(order[0].points)
    for c in order[1:]:
        pts += list(c.points[1:])
    return [tuple(p) for p in pts]


def outer_generatrix(snapshot, geometry_units: 'str' = "meters") -> 'np.ndarray':
    """The air-facing surface (rho, z) polyline features sit on, in METRES.
    It is the ordered UNION of TYPE 2 (air|PEC/IBC) and TYPE 3
    (air|dielectric) segments.  A partially coated/banded body contains both;
    preferring TYPE 3 globally would drop every bare exterior span."""
    from geometry_io import chains_from_snapshot_segments
    units = str(geometry_units).strip().lower()
    scales = {"m": 1.0, "meter": 1.0, "meters": 1.0,
              "mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3,
              "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
              "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}
    if units not in scales:
        raise ValueError(
            f"Unsupported geometry units {geometry_units!r}; use meters, "
            "millimeters, inches, or feet.")
    scale = scales[units]
    chains = chains_from_snapshot_segments(snapshot["segments"])
    outer = [c for c in chains if c.seg_type in (2, 3)]
    if not outer:
        raise ValueError("no air-facing (TYPE 2 or 3) surface found in the body .geo.")
    return np.asarray(_stitch_chains(outer), dtype=float) * scale


def surface_of_revolution_distance(generatrix: 'np.ndarray',
                                   points: 'np.ndarray') -> 'np.ndarray':
    """Shortest distance in metres from 3-D points to a revolved profile.

    Unlike ``rho(z)`` interpolation, this remains valid on sloped, vertical,
    and re-entrant generatrix segments.
    """
    gen = np.asarray(generatrix, dtype=float)
    if gen.ndim != 2 or gen.shape[1] != 2 or len(gen) < 2:
        raise ValueError("generatrix must be an (n, 2) array of (rho, z).")
    p0, p1 = gen[:-1], gen[1:]
    seg = p1 - p0
    seg_len2 = np.sum(seg ** 2, axis=1)
    if np.any(seg_len2 <= 0.0):
        raise ValueError("generatrix has a zero-length segment.")
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    q = np.column_stack([np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2]])
    t = np.clip(
        np.sum((q[:, None, :] - p0[None, :, :]) * seg[None, :, :], axis=-1)
        / seg_len2[None, :],
        0.0, 1.0)
    foot = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
    return np.sqrt(np.min(np.sum((q[:, None, :] - foot) ** 2, axis=-1),
                          axis=1))


def solve_vehicle_body(geometry, frequencies_ghz, aspects_deg,
                       geometry_units: 'str' = "meters", cfie_alpha: 'float' = 0.5,
                       workers: 'int' = 4, material_base_dir=None,
                       return_diagnostics: 'bool' = False):
    """Solve a BoR body FROM A .geo (or snapshot) WITH ITS MATERIALS and return
    (bodies, generatrix) ready for sum_features / the exporters.

    Materials are defined in the .geo (TYPE tags + IBCS_Resistances /
    Dielectrics).  Every layout, including bare PEC, is routed through
    ``bor_dispatch`` so its wavelength meshing, N controls, geometry preflight,
    and formulation guards cannot be bypassed.

    ``bodies``      {freq_ghz: {"theta_deg", "amp_vv", "amp_hh"}} (both pols).
    ``generatrix``  the outer air-facing surface (rho, z) in metres, for the
                    feature surface normals.
    ``diagnostics`` optional per-frequency solver metadata when
                    ``return_diagnostics=True``.
    """
    from geometry_io import parse_geometry, build_geometry_snapshot

    if isinstance(geometry, str):
        geometry_path = os.path.abspath(os.path.expanduser(geometry))
        with open(geometry_path, encoding="utf-8") as stream:
            snap = build_geometry_snapshot(*parse_geometry(stream.read()))
        snap["source_path"] = geometry_path
    else:
        snap = geometry
    aspects = [float(a) for a in aspects_deg]
    gen = outer_generatrix(snap, geometry_units)
    bodies = {}
    diagnostics = {}

    from bor_dispatch import solve_monostatic_rcs_bor
    kw = dict(geometry_units=geometry_units, cfie_alpha=cfie_alpha, workers=workers,
              material_base_dir=material_base_dir, expand_to_360=False)
    for f in frequencies_ghz:
        result = solve_monostatic_rcs_bor(
            snap, [float(f)], aspects, "VV", **kw
        )
        co_solved = result.get("co_solved_samples")
        if (
            not isinstance(co_solved, dict)
            or set(co_solved) != {"VV", "HH"}
        ):
            raise RuntimeError(
                "BoR dispatcher did not return its co-solved VV/HH fields; "
                "refusing to repeat or combine mismatched body solves."
            )
        sv = sorted(
            co_solved["VV"], key=lambda sample: sample["theta_inc_deg"]
        )
        sh = sorted(
            co_solved["HH"], key=lambda sample: sample["theta_inc_deg"]
        )
        if [row["theta_inc_deg"] for row in sv] != [
            row["theta_inc_deg"] for row in sh
        ]:
            raise RuntimeError(
                "Co-solved BoR VV/HH aspect grids differ."
            )
        hh = {
            round(sample["theta_inc_deg"], 6):
                complex(
                    sample["rcs_amp_real"],
                    sample["rcs_amp_imag"],
                )
            for sample in sh
        }
        th = [s["theta_inc_deg"] for s in sv]
        bodies[float(f)] = {
            "theta_deg": th,
            "amp_vv": [complex(s["rcs_amp_real"], s["rcs_amp_imag"]) for s in sv],
            "amp_hh": [hh[round(t, 6)] for t in th]}
        diagnostics[float(f)] = {
            "solver": result.get("solver", ""),
            "scattering_mode": result.get("scattering_mode", ""),
            "co_solved_polarizations": ["VV", "HH"],
            "metadata": dict(result.get("metadata", {}) or {}),
        }
    if return_diagnostics:
        return bodies, gen, diagnostics
    return bodies, gen


# -----------------------------------------------------------------------------
# The body solve as a .grim (so it opens in the viewer like every other dataset)
# -----------------------------------------------------------------------------

_BODY_AZ_MEANING = ("BoR aspect from the +z rotation axis (0 = nose-on, "
                    "90 = broadside, 180 = tail-on) -- NOT radar azimuth")
_MONOSTATIC_BODY_MODEL_SCHEMA = "ghost.workflow.embedded-bor-body-model.v1"


def verify_body_artifact_bundle(body_grim: 'str') -> 'Dict[str, Any]':
    """Validate one self-contained body GRIM and its embedded profile.

    Mesh certification is a solve-time accuracy choice, not an authorization
    token.  A base-mesh and a certified body therefore pass the same structural
    checks here and may both be used by downstream feature workflows.
    """

    path = os.path.abspath(str(body_grim))
    load_body_grim(path)
    profile = load_body_profile_grim(path)
    return {
        "schema": "ghost.workflow.self-contained-body-grim.v1",
        "body_grim": path,
        "profile_points": int(len(profile)),
    }


def require_body_mesh_certification(path: 'str') -> 'Dict[str, Any]':
    """Explicitly audit that a body used the refined-mesh path.

    This opt-in audit helper is retained for users who want to enforce that
    policy themselves.  Normal loading and downstream feature operations do
    not call it.
    """

    label = str(path)
    try:
        with np.load(label, allow_pickle=False) as payload:
            frequencies = [
                float(value)
                for value in np.asarray(
                    payload["frequencies"], dtype=float
                ).ravel()
            ]
            raw = np.asarray(
                payload["solver_metadata_json"]
            ).reshape(()).item()
    except KeyError as exc:
        raise ValueError(
            f"{label}: body has no production mesh certification; rerun "
            "step 2a/2b with the current solver."
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label}: body mesh certification cannot be read."
        ) from exc
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        audit = json.loads(str(raw))
        per_frequency = audit["metadata"]["per_frequency"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{label}: body mesh certification is malformed."
        ) from exc
    if not isinstance(per_frequency, dict):
        raise ValueError(
            f"{label}: body mesh certification has no frequency records."
        )

    certified = {}
    for frequency in frequencies:
        record = per_frequency.get(str(float(frequency)))
        if not isinstance(record, dict):
            raise ValueError(
                f"{label}: no mesh certification for {frequency:g} GHz."
            )
        metadata = record.get("metadata")
        mesh = (
            metadata.get("mesh_convergence")
            if isinstance(metadata, dict)
            else None
        )
        if (
            not isinstance(mesh, dict)
            or mesh.get("passed") is not True
            or mesh.get("published_mesh") != "fine"
        ):
            raise ValueError(
                f"{label}: {frequency:g} GHz is not certified as a "
                "passed fine-mesh body result."
            )
        polarizations = mesh.get("polarizations")
        if (
            not isinstance(polarizations, dict)
            or set(polarizations) != {"VV", "HH"}
            or any(
                not isinstance(polarizations[pol], dict)
                or polarizations[pol].get("passed") is not True
                for pol in ("VV", "HH")
            )
        ):
            raise ValueError(
                f"{label}: {frequency:g} GHz lacks passed VV/HH mesh "
                "certification."
            )
        certified[str(float(frequency))] = mesh
    return {
        "schema": "ghost.workflow.body-mesh-certification.v1",
        "passed": True,
        "published_mesh": "fine",
        "frequencies_ghz": frequencies,
        "per_frequency": certified,
    }


def require_delta_mesh_certification(path: 'str') -> 'Dict[str, Any]':
    """Explicitly audit a delta's refined-mesh source chain.

    This is an optional user policy check, not a prerequisite for loading or
    combining the delta.
    """

    label = str(path)
    try:
        with np.load(label, allow_pickle=False) as payload:
            raw = np.asarray(
                payload["production_mesh_certification_json"]
            ).reshape(()).item()
    except KeyError as exc:
        raise ValueError(
            f"{label}: delta has no production mesh certification; rebuild "
            "it through current steps 1a/1b and 1c."
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label}: delta mesh certification cannot be read."
        ) from exc
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        certification = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{label}: delta mesh certification is malformed."
        ) from exc
    sources = (
        certification.get("sources")
        if isinstance(certification, dict)
        else None
    )
    if (
        not isinstance(certification, dict)
        or certification.get("schema")
        != "ghost.workflow.mesh-certified-sources.v1"
        or certification.get("passed") is not True
        or certification.get("published_mesh") != "fine"
        or not isinstance(sources, list)
        or not sources
        or certification.get("source_count") != len(sources)
    ):
        raise ValueError(
            f"{label}: delta mesh certification is incomplete."
        )
    for source in sources:
        mesh = (
            source.get("mesh_convergence")
            if isinstance(source, dict)
            else None
        )
        if (
            not isinstance(mesh, dict)
            or mesh.get("passed") is not True
            or mesh.get("published_mesh") != "fine"
        ):
            raise ValueError(
                f"{label}: delta contains an uncertified source field."
            )
    return certification


def save_body_grim(bodies: 'Dict[float, Dict[str, Any]]', out_path: 'str', *,
                   history: 'str' = "", source_path: 'str' = "",
                   geometry_input_sha256: 'str' = "",
                   solver_source_sha256: 'str' = "",
                   runtime_environment_sha256: 'str' = "",
                   run_solve_spec_sha256: 'str' = "",
                   collection_source_sha256: 'str' = "",
                   body_profile: 'Optional[np.ndarray]' = None,
                   frequency_ghz: 'Optional[float]' = None,
                   solver_diagnostics: 'Optional[Dict[float, Any]]' = None,
                   requested_radar_grid: 'Optional[Dict[str, Any]]' = None) -> 'str':
    """Write a BoR body solve as ONE .grim: aspect x 1 x frequency x [VV, HH].

    The 3-D convention, like every BoR export: ``rcs_power`` = sigma = 4 pi |F|^2
    in m^2 (dBsm), ``rcs_amp_real/imag`` = the field amplitude F, phase preserved.

    THE AZIMUTH AXIS IS THE BoR ASPECT, not radar azimuth.  A body of revolution
    is axisymmetric, so one polar angle from the axis is its whole angular
    dependence and [0, 180] covers the sphere.  That aspect equals radar azimuth
    ONLY when the body axis is horizontal AND you stay in the elevation-0 cut
    (the waterline); off that cut a whole CONE of (az, el) looks shares one
    aspect, which is what lets export_radar_grim fill a 2-D radar grid from this
    1-D sweep.  The meaning is recorded in ``units["azimuth_meaning"]`` and in
    ``history`` so the axis cannot be silently misread as azimuth downstream.
    """
    if not isinstance(bodies, dict) or "theta_deg" in bodies:
        if frequency_ghz is None:
            raise ValueError(
                "A single BoR result has no frequency key; pass its physical "
                "frequency explicitly as frequency_ghz.")
        frequency_ghz = float(frequency_ghz)
        if not math.isfinite(frequency_ghz) or frequency_ghz <= 0.0:
            raise ValueError(
                "frequency_ghz must be positive and finite for a single "
                "BoR result.")
        bodies = {frequency_ghz: bodies}
    freqs = sorted(float(f) for f in bodies)
    th = np.asarray(bodies[freqs[0]]["theta_deg"], dtype=float)
    for f in freqs:
        t = np.asarray(bodies[f]["theta_deg"], dtype=float)
        if not np.array_equal(t, th):
            raise ValueError(f"{f} GHz has a different aspect sweep from "
                             f"{freqs[0]} GHz; one .grim needs one shared axis.")
    amp = np.zeros((len(th), 1, len(freqs), 2), dtype=complex)
    for kf, f in enumerate(freqs):
        amp[:, 0, kf, 0] = np.asarray(bodies[f]["amp_vv"], dtype=complex)
        amp[:, 0, kf, 1] = np.asarray(bodies[f]["amp_hh"], dtype=complex)

    units = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
             "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
             "azimuth_meaning": _BODY_AZ_MEANING}
    amp_real = amp.real.astype(np.float64)
    amp_imag = amp.imag.astype(np.float64)
    stored_power = 4.0 * math.pi * (
        amp_real.astype(float) ** 2 + amp_imag.astype(float) ** 2)
    out = out_path if str(out_path).lower().endswith(".grim") else str(out_path) + ".grim"
    payload = dict(
        azimuths=th, elevations=np.array([0.0]), frequencies=np.asarray(freqs, float),
        polarizations=np.asarray(["VV", "HH"], dtype=str),
        polarization_alias_primary="VV,HH",
        polarization_aliases_json=json.dumps(["VV", "HH"]),
        rcs_power=stored_power.astype(np.float32),
        rcs_phase=np.angle(
            amp_real.astype(float) + 1j * amp_imag.astype(float)).astype(np.float32),
        rcs_domain="power_phase", power_domain="linear_rcs",
        source_path=str(source_path),
        history=(history or "save_body_grim: BoR body solve")
                + f" | axis_frame: azimuth = {_BODY_AZ_MEANING}",
        units=json.dumps(units),
        phase_reference=BOR_BODY_PHASE_REFERENCE,
        amplitude_convention=PHYSICAL_3D_AMPLITUDE_CONVENTION,
        raw_complex_amplitude_preserved=True,
        rcs_amp_real=amp_real,
        rcs_amp_imag=amp_imag,
        complex_field_domain=BOR_BODY_FIELD_DOMAIN)
    if geometry_input_sha256:
        payload["geometry_input_sha256"] = np.asarray(
            str(geometry_input_sha256))
    for key, value in (
        ("solver_source_sha256", solver_source_sha256),
        ("runtime_environment_sha256", runtime_environment_sha256),
        ("run_solve_spec_sha256", run_solve_spec_sha256),
        ("collection_source_sha256", collection_source_sha256),
    ):
        if value:
            payload[key] = np.asarray(str(value))
    if solver_diagnostics is not None:
        from grim_io import _solver_metadata_json
        payload["solver_metadata_json"] = np.asarray(
            _solver_metadata_json({
                "solver": "bor_mom_rcs",
                "scattering_mode": "monostatic",
                "polarization": "VV,HH",
                "polarization_export": "VV,HH",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "metadata": {
                    "co_solved_polarizations": ["VV", "HH"],
                    "per_frequency": solver_diagnostics,
                },
                "samples": [],
            })
        )
    if requested_radar_grid is not None:
        grid = dict(requested_radar_grid)
        required = {
            "azimuths_deg",
            "elevations_deg",
            "frequencies_ghz",
            "axis_az_deg",
            "axis_el_deg",
        }
        if set(grid) != required:
            raise ValueError(
                "requested_radar_grid must contain exactly "
                + ", ".join(sorted(required))
                + "."
            )
        requested_az, requested_el = validate_radar_grid(
            grid["azimuths_deg"], grid["elevations_deg"]
        )
        requested_freqs = [float(value) for value in grid["frequencies_ghz"]]
        if (
            not requested_freqs
            or not all(math.isfinite(value) and value > 0.0
                       for value in requested_freqs)
            or len(set(requested_freqs)) != len(requested_freqs)
            or sorted(requested_freqs) != freqs
        ):
            raise ValueError(
                "requested_radar_grid frequencies must uniquely match the "
                "stored positive body frequencies."
            )
        requested_freqs = freqs
        axis_az = float(grid["axis_az_deg"])
        axis_el = float(grid["axis_el_deg"])
        if (
            not math.isfinite(axis_az)
            or not math.isfinite(axis_el)
            or not -90.0 <= axis_el <= 90.0
        ):
            raise ValueError(
                "requested_radar_grid body-axis angles must be finite and "
                "axis_el_deg must be in [-90, 90]."
            )
        payload["requested_radar_grid_json"] = np.asarray(json.dumps(
            {
                "schema": "ghost.workflow.requested-radar-grid.v1",
                "azimuths_deg": requested_az,
                "elevations_deg": requested_el,
                "frequencies_ghz": requested_freqs,
                "axis_az_deg": axis_az,
                "axis_el_deg": axis_el,
            },
            sort_keys=True,
            separators=(",", ":"),
        ))
    if body_profile is not None:
        profile = np.asarray(body_profile, dtype=float)
        if (
            profile.ndim != 2
            or profile.shape[1] != 2
            or len(profile) < 2
            or not np.all(np.isfinite(profile))
        ):
            raise ValueError(
                "body_profile must contain at least two finite rho_m,z_m rows."
            )
        payload["body_profile_rho_m"] = profile[:, 0].astype(np.float64)
        payload["body_profile_z_m"] = profile[:, 1].astype(np.float64)
    from grim_io import _save_grim_npz
    return _save_grim_npz(payload, out)


def load_body_profile_grim(path: 'str') -> 'np.ndarray':
    """Load the embedded metre-valued ``rho,z`` generatrix from a body GRIM."""
    with np.load(path, allow_pickle=False) as payload:
        if (
            "body_profile_rho_m" not in payload.files
            or "body_profile_z_m" not in payload.files
        ):
            raise ValueError(
                f"{path}: body GRIM has no embedded profile; regenerate it "
                "with the simplified step-2 runner."
            )
        rho = np.asarray(payload["body_profile_rho_m"], dtype=float).ravel()
        z = np.asarray(payload["body_profile_z_m"], dtype=float).ravel()
    profile = np.column_stack((rho, z))
    if (
        len(profile) < 2
        or len(rho) != len(z)
        or not np.all(np.isfinite(profile))
    ):
        raise ValueError(f"{path}: embedded body profile is malformed.")
    return profile


def load_body_requested_radar_grid(
    path: 'str',
) -> 'Optional[Dict[str, Any]]':
    """Read the step-2 radar-grid request embedded for provenance.

    Older or programmatically written body artifacts may not carry this
    optional record. Exact downstream support is always enforced from the
    stored BoR aspect nodes themselves.
    """

    with np.load(path, allow_pickle=False) as payload:
        if "requested_radar_grid_json" not in payload.files:
            return None
        stored_frequencies = [
            float(value)
            for value in np.asarray(payload["frequencies"], dtype=float).ravel()
        ]
        raw = np.asarray(
            payload["requested_radar_grid_json"]
        ).reshape(()).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        grid = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{path}: requested radar-grid metadata is malformed."
        ) from exc
    if (
        not isinstance(grid, dict)
        or grid.get("schema") != "ghost.workflow.requested-radar-grid.v1"
    ):
        raise ValueError(
            f"{path}: requested radar-grid metadata has an unknown schema."
        )
    azimuths, elevations = validate_radar_grid(
        grid.get("azimuths_deg", []),
        grid.get("elevations_deg", []),
    )
    frequencies = [
        float(value) for value in grid.get("frequencies_ghz", [])
    ]
    if (
        not frequencies
        or not all(math.isfinite(value) and value > 0.0
                   for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
        or sorted(frequencies) != sorted(stored_frequencies)
    ):
        raise ValueError(
            f"{path}: requested radar-grid frequencies are invalid or do not "
            "match the body field."
        )
    axis_az = float(grid.get("axis_az_deg", math.nan))
    axis_el = float(grid.get("axis_el_deg", math.nan))
    roll = float(grid.get("roll_deg", 0.0))
    if (
        not math.isfinite(axis_az)
        or not math.isfinite(axis_el)
        or not math.isfinite(roll)
        or not -90.0 <= axis_el <= 90.0
    ):
        raise ValueError(
            f"{path}: requested radar-grid body-axis angles are invalid."
        )
    return {
        "schema": "ghost.workflow.requested-radar-grid.v1",
        "azimuths_deg": azimuths,
        "elevations_deg": elevations,
        "frequencies_ghz": frequencies,
        "axis_az_deg": axis_az,
        "axis_el_deg": axis_el,
        "roll_deg": roll,
    }


def load_body_grim(path: 'str') -> 'Dict[float, Dict[str, Any]]':
    """Read a body .grim back into the ``{frequency: {theta_deg, amp_vv, amp_hh}}``
    dict that sum_features and the exporters consume.

    Current solver deliverables are radar-frame monostatic grids with the
    compact BoR aspect model embedded inside the same file.  Legacy compact
    body GRIMs remain readable so existing validated datasets do not need a
    lossy conversion.
    """
    g = _load_grim(str(path))
    label = str(path)
    if "body_model_metadata_json" in g:
        try:
            raw = np.asarray(g["body_model_metadata_json"]).reshape(()).item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            metadata = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{path}: embedded BoR body-model metadata is malformed."
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != _MONOSTATIC_BODY_MODEL_SCHEMA
            or metadata.get("phase_reference") != BOR_BODY_PHASE_REFERENCE
            or metadata.get("amplitude_convention")
            != PHYSICAL_3D_AMPLITUDE_CONVENTION
        ):
            raise ValueError(
                f"{path}: embedded BoR body model has incompatible field "
                "conventions."
            )
        required = (
            "body_model_aspects_deg",
            "body_model_amp_vv_real",
            "body_model_amp_vv_imag",
            "body_model_amp_hh_real",
            "body_model_amp_hh_imag",
        )
        missing = [key for key in required if key not in g]
        if missing:
            raise ValueError(
                f"{path}: embedded BoR body model is missing {missing}."
            )
        aspects = np.asarray(g["body_model_aspects_deg"], dtype=float)
        frequencies = np.asarray(g["frequencies"], dtype=float)
        expected = (len(aspects), len(frequencies))
        arrays = {
            key: np.asarray(g[key], dtype=float)
            for key in required[1:]
        }
        if (
            aspects.ndim != 1
            or not len(aspects)
            or not np.all(np.isfinite(aspects))
            or np.any(np.diff(aspects) <= 0.0)
            or any(value.shape != expected for value in arrays.values())
            or any(not np.all(np.isfinite(value)) for value in arrays.values())
        ):
            raise ValueError(
                f"{path}: embedded BoR body-model arrays are malformed."
            )
        vv = arrays["body_model_amp_vv_real"] + 1j * arrays[
            "body_model_amp_vv_imag"
        ]
        hh = arrays["body_model_amp_hh_real"] + 1j * arrays[
            "body_model_amp_hh_imag"
        ]
        return {
            float(frequency): {
                "theta_deg": aspects.copy(),
                "amp_vv": vv[:, index].copy(),
                "amp_hh": hh[:, index].copy(),
            }
            for index, frequency in enumerate(frequencies)
        }

    if _metadata_text(g, "rcs_domain", label) != "power_phase":
        raise ValueError(
            f"{path}: a BoR body must have rcs_domain='power_phase'.")
    if _metadata_text(g, "power_domain", label) != "linear_rcs":
        raise ValueError(
            f"{path}: a BoR body must have power_domain='linear_rcs'.")
    units = _require_units(
        g, label, linear_quantity="sigma_3d", log_unit="dBsm")
    if str(units.get("azimuth_meaning", "")).strip() != _BODY_AZ_MEANING:
        raise ValueError(
            f"{path}: units.azimuth_meaning does not identify the BoR "
            "aspect axis.")
    _require_singleton_zero_elevation(g, label)
    _require_exact_metadata(g, label, {
        "phase_reference": BOR_BODY_PHASE_REFERENCE,
        "amplitude_convention": PHYSICAL_3D_AMPLITUDE_CONVENTION,
        "complex_field_domain": BOR_BODY_FIELD_DOMAIN,
    })
    canonical = _canonical_table_channels(g, label)
    _require_complete_2d_channels(canonical, f"{path}: BoR body")
    idx = {"HH" if key == "TM" else "VV": value
           for key, value in canonical.items()}
    th = np.asarray(g["azimuths"], dtype=float)
    amp = g["_amp"]
    return {float(f): {"theta_deg": th,
                       "amp_vv": amp[:, 0, kf, idx["VV"]],
                       "amp_hh": amp[:, 0, kf, idx["HH"]]}
            for kf, f in enumerate(np.asarray(g["frequencies"], dtype=float))}


# -----------------------------------------------------------------------------
# Wing-body (dihedral) corner double-bounce ESTIMATE
# -----------------------------------------------------------------------------

def corner_amplitude(fold, n_wing, n_body, face_width: 'float',
                     directions: 'np.ndarray', frequency_ghz: 'float',
                     internal_phase_deg: 'float' = 0.0,
                     retro_halfwidth_deg: 'float' = 45.0,
                     occluder=None) -> 'Dict[str, np.ndarray]':
    """PO-level estimate of the wing-body dihedral double-bounce.

    The line-expansion sum is SINGLE-bounce: body and wing scatter in isolation
    and their fields add.  The corner where a wing meets the body is a
    DOUBLE-bounce (body -> wing -> radar) that exists in neither isolated solve.
    This adds it as a corner-reflector term, right-angle or canted.

    Physics captured:
      * magnitude: the standard dihedral peak sigma_0 = 8 pi a^2 b^2 / lambda^2
        (b = fold length, a = ``face_width`` = effective double-bounce height),
        with a sinc^2 aperture ALONG the fold and a broad cos^2 retroreflective
        lobe PERPENDICULAR to it (the defining dihedral pattern);
      * polarization: the EXACT ideal-dihedral Jones matrix diag(1,-1) in the
        fold-aligned basis -> co-pol (with a V/H sign flip) when the fold lies
        along the radar V or H, PURE cross-pol when the fold is at 45 deg;
      * placement phase exp(2jk d.r_center) from the fold midpoint;
      * NON-RIGHT dihedral (canted / dihedral / anhedral wing root).  With
        interior angle alpha = 90 + eps the outward normals give
        eps = asin(n_wing . n_body) and delta = |eps| is the deviation from
        square.  Two reflections rotate a ray by 2 alpha, so the double bounce
        leaves the corner deflected 2 delta off the incidence reversal instead of
        retroreflecting; the lobe centre is therefore rotated by 2 eps about the
        fold axis.  BOTH bounce senses exist (body->wing and wing->body deflect
        oppositely, +2 eps and -2 eps about n_wing x n_body), so the lobe is a
        symmetric PAIR and the perpendicular-plane response is the cos^2
        envelope of the two -- at eps = 0 the pair collapses back onto the
        bisector and this is exactly the right-dihedral lobe.  The peak is then
        rolled off by cos^2(2 eps): unity at eps = 0, monotone, and vanishing at
        delta = 45 deg, where the corner has opened into a single plane (or shut
        into a cusp) and no double bounce survives.

    ESTIMATE caveats (this is not a rigorous solve): the internal double-bounce
    constant phase is not tracked (``internal_phase_deg``, default 0) so the
    corner's phase relative to the single-bounce terms is rough -- prefer
    ``mode="hybrid"`` (power-add to the body) over full coherent.  The body is
    treated as locally flat at the root; a real curved body reduces the return.
    The non-right model is a screening HEURISTIC: the 2 delta deflection is a
    BISTATIC ray-geometry statement (the exit beam misses the radar by 2 delta at
    EVERY look angle in the perpendicular plane), so the rigorous monostatic
    answer is two shifted plate lobes whose aperture mismatch attenuates the
    return roughly like sinc^2(k a sin 2 delta) -- tens of dB within a few
    degrees of square for an electrically large face, far sharper than the
    cos^2(2 eps) used here. The smooth cos^2 rolloff keeps the term monotone but
    is not a guaranteed upper or lower bound; it says "the modeled corner stops
    pointing energy back at you and points it 2 delta away", not "this is the
    exact canted-dihedral pattern". Corners more than ~20 deg off square carry a
    warning.

    ``fold``       (2,3) endpoints or (n,2,3) segments of the root/fold line.
    ``n_wing``     outward wing face normal (3,).
    ``n_body``     outward body face normal at the root (3,).
    ``face_width`` effective face width a (m): how far the double bounce reaches
                   up the wing / along the body -- e.g. min(wing height, body
                   extent).  The single biggest modelling knob.
    Returns {"F_vv","F_hh","F_vh"} complex over directions (sigma = 4pi|F|^2).
    """
    fold = np.asarray(fold, dtype=float)
    if (fold.ndim == 2 and fold.shape == (2, 3)):
        p0, p1 = fold[0], fold[1]
    elif (fold.ndim == 3 and fold.shape[1:] == (2, 3)
          and len(fold) > 0):
        seg_lengths = np.linalg.norm(fold[:, 1] - fold[:, 0], axis=1)
        if np.any(seg_lengths <= 0.0):
            raise ValueError("corner fold contains a zero-length segment.")
        if len(fold) > 1:
            scale = max(float(np.max(np.abs(fold))), 1.0)
            if np.any(np.linalg.norm(
                    fold[:-1, 1] - fold[1:, 0], axis=1) > 1e-9 * scale):
                raise ValueError(
                    "corner fold segments must form one head-to-tail chain.")
        p0, p1 = fold[0, 0], fold[-1, 1]
    else:
        raise ValueError(
            "corner fold must have shape (2,3) endpoints or "
            "(n_segments,2,3).")
    if not np.all(np.isfinite(fold)):
        raise ValueError("corner fold contains NaN or infinite coordinates.")
    f = p1 - p0
    Lf = float(np.linalg.norm(f))
    if not math.isfinite(Lf) or Lf <= 0.0:
        raise ValueError("corner fold line must have positive finite length.")
    fhat = f / Lf
    r_c = 0.5 * (p0 + p1)
    nw = np.asarray(n_wing, float)
    nb = np.asarray(n_body, float)
    for label, normal in (("n_wing", nw), ("n_body", nb)):
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError(f"{label} must be a finite 3-vector.")
        if float(np.linalg.norm(normal)) <= 1e-12:
            raise ValueError(f"{label} must be nonzero.")
    nw = nw / np.linalg.norm(nw)
    nb = nb / np.linalg.norm(nb)
    face_width = float(face_width)
    frequency_ghz = float(frequency_ghz)
    retro_halfwidth_deg = float(retro_halfwidth_deg)
    internal_phase_deg = float(internal_phase_deg)
    if not math.isfinite(face_width) or face_width <= 0.0:
        raise ValueError("corner face_width must be positive and finite.")
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0.0:
        raise ValueError("corner frequency_ghz must be positive and finite.")
    if (not math.isfinite(retro_halfwidth_deg)
            or retro_halfwidth_deg <= 0.0
            or retro_halfwidth_deg > 180.0):
        raise ValueError(
            "corner retro_halfwidth_deg must be finite and in (0, 180].")
    if not math.isfinite(internal_phase_deg):
        raise ValueError("corner internal_phase_deg must be finite.")
    # Deviation from square.  A RIGHT dihedral has perpendicular outward normals
    # (nw.nb = 0); with interior angle alpha the outward normals make (180-alpha)
    # with each other, so nw.nb = -cos(alpha) = sin(alpha - 90):
    #   eps = asin(nw.nb) = alpha - 90 (signed, >0 = corner opened out)
    #   delta = |eps|                   (angle off perpendicular)
    dot_n = float(np.clip(nw @ nb, -1.0, 1.0))
    eps = math.asin(dot_n)
    delta = abs(eps)
    # Peak rolloff.  Two reflections rotate a ray by 2*alpha, so the double bounce
    # exits 2*delta off the incidence reversal and the corner stops retro-
    # reflecting; cos^2(2 eps) is a smooth monotone stand-in: exactly 1 at
    # delta=0 (the right dihedral 8 pi a^2 b^2/lam^2), 0 at delta=45 deg where the
    # faces have gone coplanar (or shut into a cusp) and no double bounce is left.
    # An ESTIMATE, and a generous one -- the rigorous monostatic mismatch factor
    # (~sinc^2(k a sin 2 delta), see the docstring) falls far faster.
    roll = max(math.cos(2.0 * eps), 0.0) ** 2
    two_eps = 2.0 * eps                       # lobe-centre deflection (rad)
    # Signed fold axis: both normals are perpendicular to the fold line, so
    # nw x nb IS along it, and it fixes the deflection handedness independently
    # of how the caller ordered the fold-line endpoints.
    cr = np.cross(nw, nb)
    ncr = float(np.linalg.norm(cr))
    ahat = cr / ncr if ncr > 1e-9 else fhat
    warn = None
    if abs(dot_n) > 0.34:                     # >~20 deg from a right dihedral
        warn = (f"dihedral {math.degrees(delta):.0f} deg off square -- double-bounce "
                f"lobe deflected ~{math.degrees(2*delta):.0f} deg off the bisector and "
                f"peak attenuated {10*math.log10(max(roll, 1e-12)):+.1f} dB; "
                f"deflected/attenuated PO estimate.")
    bhat = nw + nb
    bhat_norm = float(np.linalg.norm(bhat))
    if bhat_norm <= 1e-12:
        raise ValueError(
            "corner face normals are antiparallel, so the bisector is "
            "undefined.")
    bhat = bhat / bhat_norm                   # bisector (retro direction at eps=0)

    k = 2.0 * math.pi * frequency_ghz * 1e9 / C0
    lam = C0 / (frequency_ghz * 1e9)
    sigma0 = 8.0 * math.pi * face_width ** 2 * Lf ** 2 / lam ** 2
    retro = math.radians(retro_halfwidth_deg)
    intph = math.radians(internal_phase_deg)

    dirs = np.atleast_2d(np.asarray(directions, float))
    if (dirs.ndim != 2 or dirs.shape[1:] != (3,) or len(dirs) == 0
            or not np.all(np.isfinite(dirs))):
        raise ValueError(
            "corner directions must be a nonempty array of finite 3-vectors.")
    dir_norm = np.linalg.norm(dirs, axis=1)
    if np.any(dir_norm <= 1e-12):
        raise ValueError("corner directions contain a zero vector.")
    dirs = dirs / dir_norm[:, None]
    e_vv, e_hh = _pol_unit_vectors(dirs)
    F = {c: np.zeros(len(dirs), complex) for c in ("F_vv", "F_hh", "F_vh")}

    for i, d in enumerate(dirs):
        if (d @ nw) <= 0.0 or (d @ nb) <= 0.0:      # both faces must be lit
            continue
        if occluder is not None and not bool(occluder.visible(r_c[None, :], d)[0]):
            continue                                # body blocks the corner
        df = float(d @ fhat)
        x = k * Lf * df
        sinc_fold = float(np.sinc(x / math.pi))
        a_fold = sinc_fold ** 2
        d_perp = d - df * fhat
        npn = float(np.linalg.norm(d_perp))
        if npn < 1e-9:
            continue
        dhat_perp = d_perp / npn
        phi = math.acos(np.clip(dhat_perp @ bhat, -1.0, 1.0))
        if two_eps != 0.0:
            # non-right: the lobe centre sits 2*eps off the bisector, one lobe per
            # bounce sense (+2eps body->wing, -2eps wing->body).  Measure phi to
            # the NEARER centre = cos^2 envelope of the pair.  (eps == 0 skips
            # this entirely, so a right dihedral is untouched.)
            phi_s = phi if float(np.cross(bhat, dhat_perp) @ ahat) >= 0.0 else -phi
            phi = min(abs(phi_s - two_eps), abs(phi_s + two_eps))
        if phi > retro:
            continue
        a_perp = math.cos(phi) ** 2
        m = math.sqrt(max(sigma0 * a_fold * a_perp * roll, 0.0) / (4.0 * math.pi))
        # pol: dihedral Jones diag(1,-1) in the fold-aligned basis, rotated to V/H
        phat = fhat - df * d
        if np.linalg.norm(phat) < 1e-9:
            continue
        phat = phat / np.linalg.norm(phat)
        qhat = np.cross(d, phat)
        R = np.array([[phat @ e_vv[i], phat @ e_hh[i]],
                      [qhat @ e_vv[i], qhat @ e_hh[i]]])
        Svh = R.T @ np.array([[1.0, 0.0], [0.0, -1.0]]) @ R
        # sigma follows sinc^2, but the complex aperture field follows sinc and
        # changes sign through every null.  Keeping only sqrt(sinc^2) would turn
        # that sign into |sinc| and erase the physical pi phase reversals.
        fold_sign = 0.0 if sinc_fold == 0.0 else math.copysign(1.0, sinc_fold)
        s = (fold_sign * m * np.exp(2j * k * float(d @ r_c))
             * np.exp(1j * intph))
        F["F_vv"][i] = s * Svh[0, 0]
        F["F_hh"][i] = s * Svh[1, 1]
        F["F_vh"][i] = s * Svh[0, 1]
    if warn:
        F["warning"] = warn
    return F


# -----------------------------------------------------------------------------
# Point scatterer: a precomputed 3-D delta pattern placed at one coordinate
# (e.g. a blind cavity solved by an EXTERNAL 3-D MoM as featured - clean)
# -----------------------------------------------------------------------------


class PreparedPointPattern(NamedTuple):
    """Validated compact pattern cached for reuse at many coordinates."""

    azimuths: 'np.ndarray'
    elevations: 'np.ndarray'
    frequencies: 'np.ndarray'
    amplitude: 'np.ndarray'
    channel_indices: 'Dict[str, int]'


def _validate_point_pattern_metadata(metadata: 'Dict[str, Any]',
                                     label: 'str', *,
                                     declared_coherent_delta: 'bool' = False
                                     ) -> 'None':
    expected = point_pattern_convention_metadata()
    if declared_coherent_delta:
        # Listing this file in COMPACT_FEATURES declares both the operation
        # meaning (installed feature minus matching clean skin) and the cavity
        # frame/origin convention documented by place_features.py. A GUI may
        # drop these strings or carry stale source strings into its derived
        # output, so the explicit declaration supersedes all of them. Units,
        # normalization, finite fields, channels, and angular coverage are
        # independently checked by _load_pattern.
        return
    for key, required in expected.items():
        got = _metadata_text(metadata, key, label)
        if got != required:
            raise ValueError(
                f"{label}: incompatible compact-pattern {key}: got {got!r}; "
                f"require {required!r}.  The feature can be placed coherently "
                "only when its phase origin, time sign, frame, and far-field "
                "normalization are explicit.")


def _load_pattern(pattern, *, declared_coherent_delta=False,
                  assume_missing_cross_pol_zero=False):
    """Return (az_deg, el_deg, freqs_ghz, amp[az,el,freq,pol], {ch:idx}) for a
    3-D delta pattern given as a .grim path or a dict with the same axes.  The
    pattern is the COMPLEX differential scattering (featured - clean) of the
    compact feature in ITS OWN reference frame: az/el are the cavity-frame
    spherical angles of the coming-from look (el measured from the aperture
    plane, +z = aperture outward normal), pols are the cavity meridian basis
    VV = theta-pol, HH = phi-pol about that normal, plus cross-pol VH."""
    if isinstance(pattern, PreparedPointPattern):
        return tuple(pattern)
    if isinstance(pattern, str) and not pattern.lower().endswith(".grim"):
        # not one of our .grim exports -> try the GRIM_Revised_2 viewer's
        # importers (.out / .ss / PIO / theta-phi CSV or TXT) via grim_compat.
        # Those formats cannot prove a phase origin/frame convention, so the
        # returned untagged dict will fail the convention gate below.  Import
        # it explicitly with grim_compat.load_pattern_any(...,
        # convention_metadata=point_pattern_convention_metadata()) after
        # verifying the external solver setup.
        from grim_compat import load_pattern_any
        pattern = load_pattern_any(pattern)
    if isinstance(pattern, str):
        g = _load_grim(pattern)
        _validate_point_pattern_metadata(
            g, pattern, declared_coherent_delta=declared_coherent_delta
        )
        az = np.asarray(g["azimuths"], float); el = np.asarray(g["elevations"], float)
        fr = np.asarray(g["frequencies"], float)
        pols = [str(p) for p in np.asarray(g["polarizations"]).ravel()]
        if "rcs_amp_real" in g and "rcs_amp_imag" in g:
            amp = g["rcs_amp_real"] + 1j * g["rcs_amp_imag"]
        elif declared_coherent_delta and g.get("_amp_from_power_phase", False):
            # A GRIM GUI coherent subtraction stores the same complex field as
            # sigma plus phase. _load_grim reverses the declared sigma_3d
            # normalization, including exact physical nulls.
            amp = np.asarray(g["_amp"], dtype=np.complex128)
        else:
            raise ValueError(
                f"{pattern}: compact-feature patterns require preserved raw "
                "complex amplitudes, or a declared GUI coherent subtraction "
                "with finite power and phase.")
        pattern_units = _require_linear_quantity(g, pattern, "sigma_3d")
        if (not declared_coherent_delta and
                str(pattern_units.get("rcs_log_unit", "")).strip().lower()
                != "dbsm"):
            raise ValueError(
                f"{pattern}: compact-feature pattern must use dBsm display "
                "units for sigma_3d.")
        if "rcs_power" not in g:
            raise ValueError(f"{pattern}: compact-feature pattern has no rcs_power.")
        stored_power = np.asarray(g["rcs_power"], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            amplitude_squared = np.abs(np.asarray(amp, dtype=complex)) ** 2
            predicted_power = 4.0 * math.pi * amplitude_squared
        if stored_power.shape != amp.shape:
            raise ValueError(
                f"{pattern}: rcs_power shape {stored_power.shape} does not "
                f"match complex-field shape {amp.shape}.")
        tolerance = (
            8.0 * np.finfo(np.float32).eps
            * np.maximum(predicted_power, stored_power)
            + np.finfo(np.float32).tiny
        )
        if (not np.all(np.isfinite(stored_power))
                or not np.all(np.isfinite(predicted_power))
                or np.any(stored_power < 0.0)
                or np.any(np.abs(stored_power - predicted_power) > tolerance)):
            raise ValueError(
                f"{pattern}: rcs_power is inconsistent with the 3-D complex "
                "field; require rcs_power=4*pi*|F|^2.")
    else:
        _validate_point_pattern_metadata(pattern, "point pattern")
        az = np.asarray(pattern["azimuths"], float); el = np.asarray(pattern["elevations"], float)
        fr = np.asarray(pattern["frequencies"], float)
        pols = [str(p) for p in np.asarray(pattern["polarizations"]).ravel()]
        amp = np.asarray(pattern["amp"], complex)
    if az.ndim != 1 or el.ndim != 1 or fr.ndim != 1:
        raise ValueError("point pattern axes must be one-dimensional.")
    if min(len(az), len(el), len(fr)) == 0:
        raise ValueError("point pattern axes cannot be empty.")
    if not np.all(np.isfinite(az)) or not np.all(np.isfinite(el)) \
            or not np.all(np.isfinite(fr)):
        raise ValueError("point pattern axes contain NaN or infinite values.")
    if np.any(np.diff(az) <= 0.0) or np.any(np.diff(el) <= 0.0) \
            or np.any(np.diff(fr) <= 0.0):
        raise ValueError("point pattern azimuth, elevation, and frequency axes "
                         "must be strictly increasing.")
    if np.any(fr <= 0.0):
        raise ValueError("point pattern frequencies must be positive.")
    if el[0] < -90.0 - 1e-9 or el[-1] > 90.0 + 1e-9:
        raise ValueError("point pattern elevation must lie in [-90, 90] deg.")
    expected = (len(az), len(el), len(fr), len(pols))
    if amp.shape != expected:
        raise ValueError(
            f"point pattern amplitude shape {amp.shape} does not match axes "
            f"{expected}.")
    if not np.all(np.isfinite(amp.real) & np.isfinite(amp.imag)):
        raise ValueError("point pattern contains NaN or infinite amplitudes.")
    az_span = float(az[-1] - az[0]) if len(az) > 1 else 0.0
    if math.isclose(az_span, 360.0, rel_tol=0.0, abs_tol=1e-6):
        if not np.allclose(amp[0], amp[-1], rtol=2e-5, atol=1e-10):
            raise ValueError(
                "point pattern includes both azimuth seam endpoints but "
                "their complex amplitudes do not agree."
            )
    else:
        # Normal monostatic grids contain one unique revolution (for example
        # 0..359 by 1 deg), not duplicate 0/360 looks. Close that periodic seam
        # internally only when the axis itself proves full uniform coverage:
        # every internal step and the wrap gap must be the same. A partial or
        # irregular cut is still never silently wrapped.
        steps = np.diff(az)
        step = float(np.median(steps)) if len(steps) else float("nan")
        wrap_gap = float(az[0] + 360.0 - az[-1]) if len(az) else float("nan")
        axis_scale = max(1.0, float(np.max(np.abs(az)))) if len(az) else 1.0
        # GRIM axes may have been serialized as float32. Allow several ulps at
        # the largest azimuth while still distinguishing a genuinely missing
        # angular sample from roundoff.
        axis_tol = max(
            1e-9,
            8.0 * abs(float(np.spacing(np.float32(axis_scale)))),
        )
        uniform = (
            len(az) >= 3
            and math.isfinite(step)
            and step > 0.0
            and np.allclose(steps, step, rtol=1e-8, atol=axis_tol)
            and math.isclose(
                wrap_gap, step, rel_tol=1e-8,
                abs_tol=axis_tol,
            )
        )
        if not uniform:
            raise ValueError(
                "point pattern must cover one complete 360-degree azimuth "
                "period. Accepted forms are matching first/last seam "
                "endpoints or one complete uniform unique-look grid; got "
                f"span {az_span:g} deg and wrap gap {wrap_gap:g} deg. "
                "Partial data are not silently wrapped."
            )
        az = np.concatenate([az, [az[0] + 360.0]])
        amp = np.concatenate([amp, amp[:1]], axis=0)

    idx = {}
    for i, p in enumerate(pols):
        P = p.strip().upper()
        key = ("VV" if P in ("VV", "TE", "V")
               else "HH" if P in ("HH", "TM", "H")
               else "VH" if P in ("VH", "HV") else P)
        if key in idx:
            raise ValueError(f"point pattern has duplicate polarization alias "
                             f"for {key}.")
        idx[key] = i
    missing = [p for p in ("VV", "HH", "VH") if p not in idx]
    if missing == ["VH"] and assume_missing_cross_pol_zero:
        amp = np.concatenate(
            [amp, np.zeros(amp.shape[:-1] + (1,), dtype=complex)], axis=-1
        )
        idx["VH"] = amp.shape[-1] - 1
    elif missing:
        raise ValueError(
            f"point pattern is missing {missing}. A general compact 3-D "
            "scatterer requires the full reciprocal Jones matrix VV/HH/VH; "
            "missing channels are not assumed to be zero. For a locally "
            "diagonal reciprocal feature, explicitly set "
            "assume_missing_cross_pol_zero=True.")
    return az, el, fr, amp, idx


def prepare_point_pattern(pattern, *, declared_coherent_delta=False,
                          delta_sign: 'float' = 1.0,
                          assume_missing_cross_pol_zero: 'bool' = False
                          ) -> 'PreparedPointPattern':
    """Validate and load one compact pattern once for repeated placement.

    ``declared_coherent_delta=True`` is for a GUI power/phase result that the
    caller explicitly attests is installed-feature minus clean-skin in the
    documented cavity frame and phase origin. It supplies only convention tags
    lost or copied stale by that GUI operation; units, grid, polarization,
    seam, and numerical normalization remain strict.
    """
    sign = float(delta_sign)
    if not math.isfinite(sign) or sign not in (-1.0, 1.0):
        raise ValueError("delta_sign must be exactly +1 or -1.")
    loaded = pattern if isinstance(pattern, PreparedPointPattern) else PreparedPointPattern(
        *_load_pattern(
            pattern,
            declared_coherent_delta=declared_coherent_delta,
            assume_missing_cross_pol_zero=assume_missing_cross_pol_zero,
        )
    )
    if sign == 1.0:
        return loaded
    return PreparedPointPattern(
        loaded.azimuths,
        loaded.elevations,
        loaded.frequencies,
        -loaded.amplitude,
        loaded.channel_indices,
    )


def point_scatterer_amplitude(pattern, location, aperture_normal, directions,
                              frequency_ghz, roll_ref=None,
                              tol_ghz: 'float' = 1e-6, occluder=None,
                              _interpolator_cache=None) -> 'Dict[str, np.ndarray]':
    """Place a precomputed 3-D delta pattern at a single body coordinate.

    Unlike the line-expanded features (distributed along a perimeter/span), a
    compact feature such as a blind cavity is a POINT scatterer: its full 3-D
    differential far field ``DeltaS(az, el, f)`` is computed once by an external
    3-D solver (featured - clean, same background) and simply relocated:

        F(d) = [ DeltaS(d in cavity frame), rotated into the body pol basis ]
               * exp(+2jk d.r_c) * shadow(d)

    ``pattern``          .grim path or dict of the delta (see _load_pattern).
                         It must carry the exact metadata returned by
                         point_pattern_convention_metadata(), cover a complete
                         360-degree azimuth period (a unique-look grid such as
                         0..359 is closed internally), and support every
                         requested lit elevation.
    ``location``         r_c (3,), the cavity phase centre on the body (place it
                         where the external solver put ITS phase origin).
    ``aperture_normal``  cavity aperture outward normal (3,) in the body frame.
    ``roll_ref``         optional (3,) fixing the cavity-frame azimuth zero
                         (its projection perpendicular to the normal); default
                         is an arbitrary transverse vector.
    ``directions``       (n,3) COMING-FROM look directions in the body frame.

    Returns {"F_vv","F_hh","F_vh"}.  Contributes only where the aperture faces
    the radar (d.normal > 0).  The remaining approximation is single-bounce:
    body<->cavity mutual coupling is not modelled.
    """
    from scipy.interpolate import RegularGridInterpolator

    az, el, fr, amp, idx = _load_pattern(pattern)
    j = int(np.argmin(np.abs(fr - float(frequency_ghz))))
    if abs(fr[j] - float(frequency_ghz)) > tol_ghz:
        raise ValueError(f"point pattern has no {frequency_ghz} GHz (has {fr.tolist()}).")
    cache_key = (id(pattern), j)
    interp = None if _interpolator_cache is None else _interpolator_cache.get(
        cache_key
    )
    if interp is None:
        def _mk(ch):
            if ch not in idx:
                return None
            a2 = amp[:, :, j, idx[ch]]
            return (RegularGridInterpolator(
                        (az, el), a2.real, bounds_error=True),
                    RegularGridInterpolator(
                        (az, el), a2.imag, bounds_error=True))
        interp = {c: _mk(c) for c in ("VV", "HH", "VH")}
        if _interpolator_cache is not None:
            _interpolator_cache[cache_key] = interp

    zc = np.asarray(aperture_normal, float)
    if zc.shape != (3,) or not np.all(np.isfinite(zc)) \
            or np.linalg.norm(zc) <= 1e-12:
        raise ValueError("aperture_normal must be a finite nonzero 3-vector.")
    zc = zc / np.linalg.norm(zc)
    if roll_ref is not None:
        xc = np.asarray(roll_ref, float)
        if xc.shape != (3,) or not np.all(np.isfinite(xc)):
            raise ValueError("roll_ref must be a finite 3-vector.")
    else:
        xc = np.array([1.0, 0.0, 0.0]) if abs(zc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    xc = xc - (xc @ zc) * zc
    if np.linalg.norm(xc) <= 1e-12:
        raise ValueError("roll_ref is parallel to aperture_normal, so the "
                         "cavity azimuth-zero direction is undefined.")
    xc = xc / np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.column_stack([xc, yc, zc])                          # cavity -> body
    rc = np.asarray(location, float)
    if rc.shape != (3,) or not np.all(np.isfinite(rc)):
        raise ValueError("location must be a finite 3-vector.")
    k = 2.0 * math.pi * frequency_ghz * 1e9 / C0

    dirs = np.atleast_2d(np.asarray(directions, float))
    if dirs.shape[1:] != (3,) or not np.all(np.isfinite(dirs)) \
            or np.any(np.linalg.norm(dirs, axis=1) <= 1e-12):
        raise ValueError("directions must contain finite nonzero 3-vectors.")
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    e_vv, e_hh = _pol_unit_vectors(dirs)
    F = {c: np.zeros(len(dirs), complex) for c in ("F_vv", "F_hh", "F_vh")}

    lit = (dirs @ zc) > 0.0
    if not np.any(lit):
        return F
    dc = dirs @ R                                              # look in cavity frame
    az_q = az[0] + (np.degrees(np.arctan2(dc[:, 1], dc[:, 0])) - az[0]) % 360.0
    el_q = np.degrees(np.arcsin(np.clip(dc[:, 2], -1.0, 1.0)))
    support_tol = 1e-9
    outside = lit & ((el_q < el[0] - support_tol)
                     | (el_q > el[-1] + support_tol))
    if np.any(outside):
        bad = el_q[outside]
        raise ValueError(
            "point pattern elevation support is incomplete for the requested "
            f"lit look(s): support is [{el[0]:g}, {el[-1]:g}] deg, queried "
            f"{float(np.min(bad)):g}..{float(np.max(bad)):g} deg. "
            "Out-of-support fields are not assumed to be zero.")
    el_q = np.clip(el_q, el[0], el[-1])
    pts = np.column_stack([az_q, el_q])
    Scav = {}
    for ch in ("VV", "HH", "VH"):
        values = np.zeros(len(dirs), complex)
        if interp[ch] is not None and np.any(lit):
            values[lit] = (
                interp[ch][0](pts[lit]) + 1j * interp[ch][1](pts[lit]))
        Scav[ch] = values
    evc, ehc = _pol_unit_vectors(dc)                           # cavity meridian basis (cavity coords)
    rc2 = rc[None, :]
    for i in np.nonzero(lit)[0]:
        if occluder is not None and not bool(occluder.visible(rc2, dirs[i])[0]):
            continue                                           # body blocks the cavity
        Evv, Ehh = R @ evc[i], R @ ehc[i]                      # -> body coords
        M = np.array([[Evv @ e_vv[i], Evv @ e_hh[i]],
                      [Ehh @ e_vv[i], Ehh @ e_hh[i]]])
        S = np.array([[Scav["VV"][i], Scav["VH"][i]],
                      [Scav["VH"][i], Scav["HH"][i]]])
        Sb = M.T @ S @ M                                       # cavity basis -> body basis
        ph = np.exp(2j * k * float(dirs[i] @ rc))
        F["F_vv"][i] = Sb[0, 0] * ph
        F["F_hh"][i] = Sb[1, 1] * ph
        F["F_vh"][i] = Sb[0, 1] * ph
    return F


# -----------------------------------------------------------------------------
# Placement + multi-feature sum
# -----------------------------------------------------------------------------

def _bor_amp_interp(bor_result: 'Dict[str, Any]', key: 'str',
                    theta_deg: 'np.ndarray') -> 'np.ndarray':
    """Return complex BoR amplitude only at explicitly solved aspect nodes.

    The historical name is retained for compatibility, but production body
    fields are never interpolated. Complex nulls and phase can move sharply
    between aspects, so every requested radar look must map to an aspect that
    was solved and stored in the body artifact.
    """
    if not isinstance(bor_result, dict):
        raise ValueError("BoR result must be a mapping.")
    if "theta_deg" not in bor_result or key not in bor_result:
        raise ValueError(
            f"BoR result must contain 'theta_deg' and {key!r}.")
    th = np.asarray(bor_result["theta_deg"], dtype=float)
    a = np.asarray(bor_result[key], dtype=complex)
    if th.ndim != 1 or a.ndim != 1:
        raise ValueError(
            f"BoR theta_deg and {key} must both be one-dimensional.")
    if len(th) == 0 or len(a) != len(th):
        raise ValueError(
            f"BoR theta_deg and {key} must be nonempty matching arrays "
            f"(got {th.shape} and {a.shape}).")
    if (not np.all(np.isfinite(th))
            or not np.all(np.isfinite(a.real))
            or not np.all(np.isfinite(a.imag))):
        raise ValueError(
            f"BoR theta_deg/{key} contain NaN or infinite values.")
    order = np.argsort(th)
    th, a = th[order], a[order]
    if np.any(np.diff(th) <= 0.0):
        raise ValueError(
            "BoR theta_deg values must be unique.")
    q_raw = np.asarray(theta_deg, dtype=float)
    scalar = q_raw.ndim == 0
    if not np.all(np.isfinite(q_raw)):
        raise ValueError("BoR aspect queries must be finite.")
    q_shape = q_raw.shape
    q = np.atleast_1d(q_raw).ravel()
    out = np.empty(q.shape, dtype=complex)
    missing = []
    for i, qi in enumerate(q):
        hit = np.nonzero(np.isclose(th, qi, rtol=0.0, atol=1e-9))[0]
        if not hit.size:
            missing.append(float(qi))
            continue
        out[i] = a[int(hit[0])]
    if missing:
        unique_missing = np.unique(np.round(missing, 12))
        raise ValueError(
            "BoR body has no explicitly solved aspect for "
            f"{len(unique_missing)} requested look(s); first missing "
            f"{unique_missing[:5].tolist()} deg. No coarse complex-field "
            "interpolation is permitted. Re-solve step 2 with azimuths and "
            "elevations that include these looks."
        )
    return out[0] if scalar else out.reshape(q_shape)


def _pick_body(bor_result, freq_ghz):
    """Resolve the BoR body for one frequency.  ``bor_result`` may be a single
    result (reused at every frequency) or a dict {freq_ghz: result} for a proper
    multi-frequency vehicle signature (the body IS frequency-dependent).  A
    single result is itself a dict, so it is recognised by its solver keys."""
    if bor_result is None:
        return None
    if isinstance(bor_result, dict) and "theta_deg" not in bor_result \
            and "amp_vv" not in bor_result:                 # a {freq: result} map
        for k, v in bor_result.items():
            if abs(float(k) - float(freq_ghz)) < 1e-6:
                return v
        raise ValueError(f"no BoR body for {freq_ghz} GHz (have {list(bor_result)}).")
    return bor_result


def _aspect_of(directions: 'np.ndarray', axis: 'np.ndarray') -> 'np.ndarray':
    d = directions / np.linalg.norm(directions, axis=1)[:, None]
    ax = axis / np.linalg.norm(axis)
    return np.degrees(np.arccos(np.clip(d @ ax, -1.0, 1.0)))


def directions_from_aspect_roll(aspects_deg: 'Sequence[float]',
                                rolls_deg: 'Sequence[float]' = (0.0,),
                                axis: 'Sequence[float]' = (0.0, 0.0, 1.0)
                                ) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Build unit COMING-FROM look directions on an (aspect x roll) grid for a
    body whose axis is ``axis`` (default +z).  Roll spins the look about the
    axis; a feature on one side is only seen over part of the roll circle.

    Returns (directions [n,3], aspect_deg [n], roll_deg [n]) flattened.
    """
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    # a fixed transverse basis (e1, e2) perpendicular to the axis
    seed = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ ax) * ax
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    asp = np.radians(np.asarray(aspects_deg, dtype=float))
    rol = np.radians(np.asarray(rolls_deg, dtype=float))
    A, R = np.meshgrid(asp, rol, indexing="ij")
    A, R = A.ravel(), R.ravel()
    d = (np.cos(A)[:, None] * ax[None, :]
         + np.sin(A)[:, None] * (np.cos(R)[:, None] * e1[None, :]
                                 + np.sin(R)[:, None] * e2[None, :]))
    return d, np.degrees(A), np.degrees(R)


def sum_features(bor_result: 'Dict[str, Any]',
                 placements: 'Sequence[Dict[str, Any]]',
                 directions: 'np.ndarray',
                 frequency_ghz: 'float',
                 normal_fn=None,
                 generatrix: 'Optional[np.ndarray]' = None,
                 mode: 'str' = "coherent",
                 perimeter_scale: 'float' = 1.0,
                 psi_tm_deg: 'float' = PSI_HH_DEG,
                 psi_te_deg: 'float' = PSI_VV_DEG,
                 corners: 'Sequence[Dict[str, Any]]' = (),
                 points: 'Sequence[Dict[str, Any]]' = (),
                 occluder=None,
                 retain_feature_amplitudes: 'bool' = True
                 ) -> 'Dict[str, np.ndarray]':
    """Combine the BoR body with any number of line-expanded features.

    ``bor_result``  a solve_monostatic_rcs_bor / solve_bor result (needs
                    theta_deg + amp_vv + amp_hh), OR None for features only.
    ``placements``  list of dicts, each:
                      {"delta": <path to delta/coef .grim OR SeamCoefficients>,
                       "perimeter": <path OR (n,2,3) array>,
                       "scale": <optional per-feature unit scale>,
                       "normal": <optional constant (3,) outward normal>,
                       "normal_fn": <optional callable overriding the body one>}
                    A WING/FIN is a placement whose ``delta`` is a full-object
                    airfoil coefficient (line_expand.coefficients_from_2d), whose
                    ``perimeter`` is the open span line (root -> tip), and which
                    carries its OWN ``normal`` (the airfoil face normal) instead
                    of the body surface normal.  Body-surface features omit
                    ``normal`` and use the generatrix normal.
    ``directions``  (n_dir, 3) unit COMING-FROM look directions (see
                    directions_from_aspect_roll).
    ``normal_fn``   default surface-normal callable for placements that do not
                    carry their own; if None it is built from ``generatrix``.

    Returns a dict with per-channel sigma (m^2) and dBsm, the per-feature and
    body amplitudes (for auditing interference), and the combine mode.
    """
    dirs = np.atleast_2d(np.asarray(directions, dtype=float))
    if normal_fn is None and generatrix is not None:
        normal_fn = surface_of_revolution_normal(generatrix)

    def _placement_normal_fn(pl):
        if callable(pl.get("normal_fn")):
            return pl["normal_fn"]
        if pl.get("normal") is not None:
            v = np.asarray(pl["normal"], dtype=float)
            v = v / np.linalg.norm(v)
            return lambda pts, _v=v: np.tile(_v, (len(np.atleast_2d(pts)), 1))
        if normal_fn is None:
            raise ValueError("placement has no normal and no body generatrix/"
                             "normal_fn was provided.")
        return normal_fn

    body = {"F_vv": np.zeros(len(dirs), complex),
            "F_hh": np.zeros(len(dirs), complex),
            "F_vh": np.zeros(len(dirs), complex)}
    if bor_result is not None:
        axis = np.array([0.0, 0.0, 1.0])
        theta = _aspect_of(dirs, axis)
        body["F_vv"] = _bor_amp_interp(bor_result, "amp_vv", theta)
        body["F_hh"] = _bor_amp_interp(bor_result, "amp_hh", theta)
        # body cross-pol is identically zero for a monostatic axisymmetric look

    warnings: 'List[str]' = []
    feats: 'List[Dict[str, np.ndarray]]' = []
    stream_features = (
        not bool(retain_feature_amplitudes)
        and str(mode).strip().lower() == "coherent"
    )
    feature_total = {
        key: np.zeros(len(dirs), dtype=complex)
        for key in ("F_vv", "F_hh", "F_vh")
    }

    def _record_feature(feature):
        if stream_features:
            for key in feature_total:
                feature_total[key] += np.asarray(
                    feature.get(key, 0.0), dtype=complex
                )
        else:
            feats.append(feature)

    for pl in placements:
        coef = pl["delta"]
        if isinstance(coef, SeamCoefficients):
            pass
        elif isinstance(coef, (list, tuple)):        # per-pol full-object grims (wing)
            coef = load_coefficients_from_grim(coef, frequency_ghz)
        else:
            # A single .grim path.  DIFFERENTIAL delta or FULL-OBJECT coefficient?
            # Both are 2-D cuts with identical axes and units, and both are line
            # expanded by the same code -- the distinction is what the numbers
            # MEAN, so getting it wrong is silent (a full-object coupon used as a
            # delta double-counts the smooth skin the body already has).
            # Declare it with "kind" when the folder/workflow already knows -- that
            # is a stronger statement than a tag, which travels badly (any derived
            # dataset, e.g. a subtract done in the viewer, loses it).  With no
            # declaration, fall back to sniffing rcs_domain.
            kind = str(pl.get("kind", "") or "").strip().lower()
            dom = str(_load_grim(str(coef)).get("rcs_domain", ""))
            if kind in ("delta", "seam"):
                coef = load_seam_from_grim(
                    str(coef), frequency_ghz,
                    declared_coherent_delta=bool(
                        pl.get("declared_coherent_delta", False)
                    ),
                    delta_sign=float(pl.get("delta_sign", 1.0)),
                )
            elif kind in ("object", "full", "coefficient"):
                if dom == "delta":
                    raise ValueError(
                        f"{os.path.basename(str(coef))}: declared "
                        "kind='object' but the file is a featured-clean delta."
                    )
                coef = load_coefficients_from_grim(str(coef), frequency_ghz)
            elif kind:
                raise ValueError(f"placement kind={kind!r}; use 'delta' or 'object'.")
            else:
                coef = (load_seam_from_grim(str(coef), frequency_ghz) if dom == "delta"
                        else load_coefficients_from_grim(str(coef), frequency_ghz))
        per = pl["perimeter"]
        if not isinstance(per, np.ndarray):
            per = read_perimeter_txt(str(per), scale=float(pl.get("scale", perimeter_scale)))
        _record_feature(expand_perimeter(
            per, coef, _placement_normal_fn(pl), dirs,
            frequency_ghz=frequency_ghz,
            psi_tm_deg=psi_tm_deg, psi_te_deg=psi_te_deg,
            occluder=occluder,
        ))

    for cn in corners:
        cf = corner_amplitude(cn["fold"], cn["n_wing"], cn["n_body"],
                              float(cn["face_width"]), dirs, frequency_ghz,
                              internal_phase_deg=float(cn.get("internal_phase_deg", 0.0)),
                              retro_halfwidth_deg=float(cn.get("retro_halfwidth_deg", 45.0)),
                              occluder=occluder)
        if "warning" in cf:
            warnings.append(cf.pop("warning"))
        _record_feature(cf)

    point_interpolator_cache = {}
    for pt in points:
        _record_feature(point_scatterer_amplitude(
            pt["pattern"], pt["location"], pt["aperture_normal"], dirs, frequency_ghz,
            roll_ref=pt.get("roll_ref"), occluder=occluder,
            _interpolator_cache=point_interpolator_cache))

    combined_features = [feature_total] if stream_features else feats
    out = combine(body, combined_features, mode=mode)
    for ch in ("vv", "hh", "vh"):
        out[f"dbsm_{ch}"] = dbsm(out[f"sigma_{ch}"])
    out["body_amp"] = body
    out["feature_amps"] = feats if retain_feature_amplitudes else None
    if stream_features:
        out["feature_amp_total"] = feature_total
    out["n_corners"] = len(corners)
    out["frequency_ghz"] = float(frequency_ghz)
    if warnings:
        out["warnings"] = warnings
    return out


# -----------------------------------------------------------------------------
# Export a combined vehicle signature to .grim
# -----------------------------------------------------------------------------


def _prepared_line_placements_at_frequency(
    placements, frequency_ghz, payload_cache
):
    """Resolve declared line deltas while loading each source GRIM only once."""
    prepared = []
    for placement in placements:
        coefficient = placement.get("delta")
        kind = str(placement.get("kind", "") or "").strip().lower()
        if (
            kind in {"delta", "seam"}
            and isinstance(coefficient, (str, os.PathLike))
        ):
            source = os.path.abspath(str(coefficient))
            if source not in payload_cache:
                payload_cache[source] = _load_grim(source)
            resolved = dict(placement)
            resolved["delta"] = load_seam_from_grim(
                source,
                float(frequency_ghz),
                declared_coherent_delta=bool(
                    placement.get("declared_coherent_delta", False)
                ),
                delta_sign=float(placement.get("delta_sign", 1.0)),
                _grim_payload=payload_cache[source],
            )
            prepared.append(resolved)
        else:
            prepared.append(placement)
    return prepared

def export_signature_grim(out_path: 'str', *,
                          bor_result: 'Optional[Dict[str, Any]]',
                          placements: 'Sequence[Dict[str, Any]]',
                          generatrix: 'np.ndarray',
                          frequencies_ghz: 'Sequence[float]',
                          aspects_deg: 'Sequence[float]',
                          rolls_deg: 'Sequence[float]' = (0.0,),
                          axis: 'Sequence[float]' = (0.0, 0.0, 1.0),
                          mode: 'str' = "coherent",
                          perimeter_scale: 'float' = 1.0,
                          psi_tm_deg: 'float' = PSI_HH_DEG,
                          psi_te_deg: 'float' = PSI_VV_DEG,
                          corners: 'Sequence[Dict[str, Any]]' = (),
                          points: 'Sequence[Dict[str, Any]]' = (),
                          occluder=None,
                          source_path: 'str' = "", history: 'str' = "") -> 'List[str]':
    """Combine body + features (+ optional wing-body ``corners``) over an
    (aspect x roll x frequency) grid and write one .grim per channel (VV, HH,
    VH), using the same physical field normalization as the radar exporter.

    Axes are BODY-FRAME: ``azimuth`` = roll about the body axis, ``elevation``
    = aspect from +axis (0 = nose-on).  This is NOT a radar az/el frame -- no
    earth-vertical rotation is applied, so the VV/HH labels are the body's
    meridian pols and avoid the radar-frame V/H swap trap. Use
    ``export_radar_grim`` when a true radar-frame product is needed.

    ``rcs_power`` always equals 4*pi times the squared magnitude of the stored
    coherent ``rcs_amp_real/imag`` field.  If ``mode`` is hybrid or envelope,
    that separately requested engineering estimate is stored under
    ``combination_estimate_power``; it never replaces the physical field pair.
    """
    from grim_io import _save_grim_npz         # same NPZ writer as the solvers

    normal_fn = surface_of_revolution_normal(np.asarray(generatrix, dtype=float))
    freqs = np.asarray([float(f) for f in frequencies_ghz], dtype=float)
    asp = np.asarray([float(a) for a in aspects_deg], dtype=float)
    rol = np.asarray([float(r) for r in rolls_deg], dtype=float)
    n_a, n_r, n_f = len(asp), len(rol), len(freqs)

    dirs, asp_flat, _ = directions_from_aspect_roll(asp, rol, axis)   # aspect-major
    chans = ("vv", "hh", "vh")
    # [roll, aspect, freq] per channel
    amp = {c: np.zeros((n_r, n_a, n_f), dtype=complex) for c in chans}
    power = {c: np.zeros((n_r, n_a, n_f), dtype=float) for c in chans}
    line_payload_cache = {}
    for fi, f in enumerate(freqs):
        frequency_placements = _prepared_line_placements_at_frequency(
            placements, float(f), line_payload_cache
        )
        res = sum_features(_pick_body(bor_result, f), frequency_placements, dirs, float(f),
                           normal_fn=normal_fn, mode=mode,
                           perimeter_scale=perimeter_scale,
                           psi_tm_deg=psi_tm_deg, psi_te_deg=psi_te_deg,
                           corners=corners, points=points, occluder=occluder,
                           retain_feature_amplitudes=False)
        for c in chans:
            a = np.asarray(res[f"amp_{c}"]).reshape(n_a, n_r).T       # -> [roll, aspect]
            s = np.asarray(res[f"sigma_{c}"]).reshape(n_a, n_r).T
            amp[c][:, :, fi] = a
            power[c][:, :, fi] = s

    units = json.dumps({"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                        "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d"})
    aliases = {"vv": ["TE", "VV", "V", "VERTICAL"],
               "hh": ["TM", "HH", "H", "HORIZONTAL"], "vh": ["VH", "HV"]}
    root = out_path[:-5] if out_path.lower().endswith(".grim") else out_path
    written: 'List[str]' = []
    for c in chans:
        A = amp[c][..., None]                                          # [roll, asp, f, 1]
        A_real = A.real.astype(np.float64)
        A_imag = A.imag.astype(np.float64)
        A_stored = A_real.astype(float) + 1j * A_imag.astype(float)
        P = 4.0 * math.pi * (
            A_real.astype(float) ** 2 + A_imag.astype(float) ** 2)
        P_est = power[c][..., None]
        payload = {
            "azimuths": rol, "elevations": asp, "frequencies": freqs,
            "polarizations": np.asarray([c.upper()], dtype=str),
            "polarization_alias_primary": c.upper(),
            "polarization_aliases_json": json.dumps(aliases[c]),
            "rcs_power": P.astype(np.float32),
            "combination_estimate_power": P_est.astype(np.float32),
            "combination_estimate_mode": np.asarray(str(mode)),
            "combination_estimate_semantics": np.asarray(
                "engineering/statistical estimate; not represented by rcs_amp"),
            "rcs_phase": np.angle(A_stored).astype(np.float32),
            "rcs_domain": "power_phase", "power_domain": "linear_rcs",
            "source_path": source_path,
            "history": (history + f" | feature_sum mode={mode} "
                        "rcs_power_is_4pi_amp2=True "
                        "estimate_key=combination_estimate_power "
                        f"axis_frame=body(az=roll,el=aspect) "
                        f"axis={tuple(float(x) for x in axis)}").strip(" |"),
            "units": units,
            "phase_reference": "origin=(0,0,0) vehicle frame, convention=exp(+jwt), "
                               "coherent total far-field amplitude (body+features)",
            "amplitude_convention": PHYSICAL_3D_AMPLITUDE_CONVENTION,
            "raw_complex_amplitude_preserved": True,
            "rcs_amp_real": A_real,
            "rcs_amp_imag": A_imag,
            "complex_field_domain": "coherent_body_plus_features_far_field_amplitude",
        }
        written.append(os.path.abspath(_save_grim_npz(payload, f"{root}_{c.upper()}")))
    return written


# -----------------------------------------------------------------------------
# Radar-frame export: monostatic RCS over (azimuth, elevation, frequency, pol)
# -----------------------------------------------------------------------------

def _direction(az_deg: 'float', el_deg: 'float') -> 'np.ndarray':
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def validate_radar_grid(azimuths_deg, elevations_deg):
    """Return a finite, unique physical radar azimuth/elevation grid."""

    azimuths = [float(value) for value in azimuths_deg]
    elevations = [float(value) for value in elevations_deg]
    if (
        not azimuths
        or not all(
            math.isfinite(value) and 0.0 <= value <= 360.0
            for value in azimuths
        )
        or len(set(azimuths)) != len(azimuths)
        or len({round(value % 360.0, 12) for value in azimuths})
        != len(azimuths)
    ):
        raise ValueError(
            "AZIMUTHS_DEG must be physically unique finite values in [0, "
            "360]; do not include both 0 and 360."
        )
    if (
        not elevations
        or not all(
            math.isfinite(value) and -90.0 <= value <= 90.0
            for value in elevations
        )
        or len(set(elevations)) != len(elevations)
    ):
        raise ValueError(
            "ELEVATIONS_DEG must be unique finite values in [-90, 90]."
        )
    return azimuths, elevations


def radar_grid_aspects(azimuths_deg, elevations_deg,
                       axis_az_deg: 'float' = 0.0,
                       axis_el_deg: 'float' = 0.0) -> 'np.ndarray':
    """Exact BoR aspect nodes required by a radar az/el output grid.

    Interpolating a rapidly varying complex body field from a coarse uniform
    aspect sweep can move nulls by tens of dB and rotate phase by many tens of
    degrees.  BoR aspect RHS columns are comparatively cheap, so production
    body solves should include the actual output-grid aspects directly.

    The returned array is sorted/deduplicated and contains only aspects mapped
    from the requested looks. Roll is intentionally absent: it rotates
    features about the BoR axis but cannot change the body aspect.
    """
    azimuths, elevations = validate_radar_grid(
        azimuths_deg, elevations_deg
    )
    _, axis = _attitude(axis_az_deg, axis_el_deg, 0.0)
    vals = []
    for az in azimuths:
        for el in elevations:
            vals.append(float(_aspect_of(_direction(float(az), float(el))[None, :],
                                         axis)[0]))
    # Identical directions can differ by a few ulps after trig/arccos.  Twelve
    # decimals is many orders tighter than any meaningful angular tolerance
    # while giving stable cache/grid membership checks.
    return np.unique(np.round(np.asarray(vals, dtype=float), 12))


def require_body_radar_support(
    body,
    frequencies_ghz,
    azimuths_deg,
    elevations_deg,
    axis_az_deg: 'float' = 0.0,
    axis_el_deg: 'float' = 0.0,
) -> 'Dict[str, Any]':
    """Require exact stored BoR nodes for every requested radar look."""

    bodies = load_body_grim(str(body)) if isinstance(
        body, (str, os.PathLike)
    ) else body
    if not isinstance(bodies, dict) or not bodies:
        raise ValueError("Body radar support requires a nonempty body result.")
    frequencies = [float(value) for value in frequencies_ghz]
    if (
        not frequencies
        or not all(math.isfinite(value) and value > 0.0
                   for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
    ):
        raise ValueError(
            "Requested body frequencies must be positive, finite, and unique."
        )
    azimuths, elevations = validate_radar_grid(
        azimuths_deg, elevations_deg
    )
    required = radar_grid_aspects(
        azimuths,
        elevations,
        axis_az_deg,
        axis_el_deg,
    )
    missing_by_frequency = {}
    for frequency in frequencies:
        result = _pick_body(bodies, frequency)
        stored = np.asarray(result.get("theta_deg", []), dtype=float)
        if (
            stored.ndim != 1
            or not stored.size
            or not np.all(np.isfinite(stored))
        ):
            raise ValueError(
                f"Body {frequency:g} GHz aspect support is malformed."
            )
        missing = [
            float(value) for value in required
            if not np.any(np.isclose(
                stored, value, rtol=0.0, atol=1.0e-9
            ))
        ]
        if missing:
            missing_by_frequency[str(float(frequency))] = missing
    if missing_by_frequency:
        first_frequency = sorted(
            missing_by_frequency, key=float
        )[0]
        first_missing = missing_by_frequency[first_frequency]
        raise ValueError(
            "Body has no explicitly solved BoR aspect for "
            f"{len(first_missing)} requested radar look(s) at "
            f"{float(first_frequency):g} GHz; first missing "
            f"{first_missing[:5]} deg. Re-run step 2 with matching "
            "AZIMUTHS_DEG, ELEVATIONS_DEG, and body-axis settings. No body "
            "complex-field interpolation is permitted."
        )
    return {
        "passed": True,
        "frequencies_ghz": frequencies,
        "azimuths_deg": azimuths,
        "elevations_deg": elevations,
        "required_aspects_deg": required.tolist(),
        "axis_az_deg": float(axis_az_deg),
        "axis_el_deg": float(axis_el_deg),
    }


def _attitude(axis_az_deg: 'float', axis_el_deg: 'float', roll_deg: 'float'):
    """Rotation R (vehicle coords -> earth coords) for a vehicle whose axis
    points (axis_az, axis_el) with the given roll about it, and the vehicle
    axis direction in earth coords.  Roll=0 puts the vehicle x-reference (where
    feature azimuths are measured from) in the vertical plane, upper side."""
    ax = _direction(axis_az_deg, axis_el_deg)
    zhat = np.array([0.0, 0.0, 1.0])
    r0 = zhat - float(zhat @ ax) * ax
    if np.linalg.norm(r0) < 1e-9:                    # vertical axis -> use earth x
        xhat = np.array([1.0, 0.0, 0.0])
        r0 = xhat - float(xhat @ ax) * ax
    u = r0 / np.linalg.norm(r0)
    w = np.cross(ax, u)
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    x_ax = cr * u + sr * w
    y_ax = -sr * u + cr * w
    return np.column_stack([x_ax, y_ax, ax]), ax


def export_radar_grim(out_path: 'str', *,
                      bor_result: 'Optional[Dict[str, Any]]',
                      placements: 'Sequence[Dict[str, Any]]',
                      generatrix: 'Optional[np.ndarray]' = None,
                      normal_fn=None,
                      frequencies_ghz: 'Sequence[float]',
                      azimuths_deg: 'Sequence[float]',
                      elevations_deg: 'Sequence[float]',
                      axis_az_deg: 'float' = 0.0,
                      axis_el_deg: 'float' = 0.0,
                      roll_deg: 'float' = 0.0,
                      perimeter_scale: 'float' = 1.0,
                      psi_tm_deg: 'float' = PSI_HH_DEG,
                      psi_te_deg: 'float' = PSI_VV_DEG,
                      corners: 'Sequence[Dict[str, Any]]' = (),
                      points: 'Sequence[Dict[str, Any]]' = (),
                      occluder=None,
                      source_path: 'str' = "", history: 'str' = "") -> 'str':
    """Monostatic radar-frame RCS -> ONE .grim with axes
    (azimuth, elevation, frequency, polarization=[VV,HH,VH]).

    The vehicle sits at attitude (axis_az, axis_el, roll) in the earth frame.
    For each radar (az, el) look this evaluates the COHERENT body+feature
    scattering in the vehicle meridian basis, then rotates the full 2x2
    scattering matrix into the radar's earth-vertical V/H basis, extended to
    the non-diagonal matrix the features produce and to a full 3-DOF attitude:

        S_radar = M^T S_vehicle M,   M[i,j] = (vehicle meridian basis_i . radar basis_j)

    This is the internally field-consistent COHERENT product represented by the
    reduced-order model (phase-summed). The canonical PEC-groove embedding
    envelope is documented in FEATURE_SUM_GUIDE.md; other features need their
    own evidence. VV/HH are the radar's earth V/H; VH is the radar-frame
    cross-pol present in the modeled component Jones matrices plus basis
    rotation. LABEL NOTE: for a horizontal axis the waterline
    radar-VV equals the vehicle's HH (handled here; don't relabel by hand).
    """
    from grim_io import _save_grim_npz

    if normal_fn is None and generatrix is not None:
        normal_fn = surface_of_revolution_normal(
            np.asarray(generatrix, dtype=float)
        )
    freqs = np.asarray([float(f) for f in frequencies_ghz], dtype=float)
    requested_az, requested_el = validate_radar_grid(
        azimuths_deg, elevations_deg
    )
    az = np.asarray(requested_az, dtype=float)
    el = np.asarray(requested_el, dtype=float)
    R, _ax = _attitude(axis_az_deg, axis_el_deg, roll_deg)

    # Earth-frame look directions and the radar's spherical V/H basis.  Build
    # H directly from the requested azimuth rather than cross(z,d), which is
    # singular at vertical looks even though the radar coordinate still
    # supplies a definite azimuthal polarization reference.
    d_e = np.zeros((len(az), len(el), 3))
    v_r = np.zeros_like(d_e)
    h_r = np.zeros_like(d_e)
    for i, a in enumerate(az):
        ar = math.radians(float(a))
        h = np.array([-math.sin(ar), math.cos(ar), 0.0])
        for j, e in enumerate(el):
            d = _direction(a, e)
            d_e[i, j] = d
            h_r[i, j] = h
            v_r[i, j] = np.cross(h, d)

    d_e_flat = d_e.reshape(-1, 3)
    d_v_flat = d_e_flat @ R              # R^T @ d_e per row  (earth -> vehicle)
    # Use exactly the vehicle-frame basis in which sum_features returns its
    # Jones matrix, then rotate that basis into earth coordinates.  This also
    # preserves vehicle roll for a look exactly along the BoR axis, where a
    # meridian chosen only from cross(axis, look) is mathematically singular
    # but off-axis features remain polarization-anisotropic.
    v_t_v, h_t_v = _pol_unit_vectors(d_v_flat)
    v_t_e = v_t_v @ R.T
    h_t_e = h_t_v @ R.T
    vrf = v_r.reshape(-1, 3)
    hrf = h_r.reshape(-1, 3)
    Mf = np.empty((len(d_v_flat), 2, 2), dtype=float)
    Mf[:, 0, 0] = np.sum(v_t_e * vrf, axis=1)
    Mf[:, 0, 1] = np.sum(v_t_e * hrf, axis=1)
    Mf[:, 1, 0] = np.sum(h_t_e * vrf, axis=1)
    Mf[:, 1, 1] = np.sum(h_t_e * hrf, axis=1)

    n_pol = 3
    shape = (len(az), len(el), len(freqs), n_pol)
    amp = np.zeros(shape, dtype=complex)
    line_payload_cache = {}
    for fi, f in enumerate(freqs):
        frequency_placements = _prepared_line_placements_at_frequency(
            placements, float(f), line_payload_cache
        )
        res = sum_features(_pick_body(bor_result, f), frequency_placements, d_v_flat, float(f),
                           normal_fn=normal_fn, mode="coherent",
                           perimeter_scale=perimeter_scale,
                           psi_tm_deg=psi_tm_deg, psi_te_deg=psi_te_deg,
                           corners=corners, points=points, occluder=occluder,
                           retain_feature_amplitudes=False)
        S = np.zeros((len(d_v_flat), 2, 2), dtype=complex)
        S[:, 0, 0] = res["amp_vv"]
        S[:, 1, 1] = res["amp_hh"]
        S[:, 0, 1] = res["amp_vh"]       # reciprocity: S_hv = S_vh
        S[:, 1, 0] = res["amp_vh"]
        Sr = np.einsum("nai,nab,nbj->nij", Mf, S, Mf)      # M^T S M
        vv = Sr[:, 0, 0].reshape(len(az), len(el))
        hh = Sr[:, 1, 1].reshape(len(az), len(el))
        vh = Sr[:, 0, 1].reshape(len(az), len(el))
        amp[:, :, fi, 0] = vv
        amp[:, :, fi, 1] = hh
        amp[:, :, fi, 2] = vh

    amp_real = amp.real.astype(np.float64)
    amp_imag = amp.imag.astype(np.float64)
    amp_stored = amp_real.astype(float) + 1j * amp_imag.astype(float)
    power = (4.0 * math.pi * (
        amp_real.astype(float) ** 2 + amp_imag.astype(float) ** 2)
    ).astype(np.float32)
    out = out_path if out_path.lower().endswith(".grim") else out_path + ".grim"
    payload = {
        "azimuths": az, "elevations": el, "frequencies": freqs,
        "polarizations": np.asarray(["VV", "HH", "VH"], dtype=str),
        "polarization_alias_primary": "VV",
        "polarization_aliases_json": json.dumps(["VV", "HH", "VH"]),
        "combine_role": np.asarray("coherent"),
        "rcs_power": power,
        "rcs_phase": np.angle(amp_stored).astype(np.float32),
        "rcs_domain": "power_phase", "power_domain": "linear_rcs",
        "source_path": source_path,
        "history": (history + f" | feature_sum radar-frame coherent "
                    f"axis_az={axis_az_deg:g} axis_el={axis_el_deg:g} "
                    f"roll={roll_deg:g}").strip(" |"),
        "units": json.dumps({"azimuth": "deg", "elevation": "deg",
                             "frequency": "GHz", "rcs_log_unit": "dBsm",
                             "rcs_linear_quantity": "sigma_3d"}),
        "phase_reference": RADAR_COMPONENT_PHASE_REFERENCE,
        "amplitude_convention": PHYSICAL_3D_AMPLITUDE_CONVENTION,
        "raw_complex_amplitude_preserved": True,
        "rcs_amp_real": amp_real,
        "rcs_amp_imag": amp_imag,
        "complex_field_domain": RADAR_COMPONENT_FIELD_DOMAIN,
    }
    saved = os.path.abspath(_save_grim_npz(payload, out))
    # grim_io intentionally writes a fixed cross-tool core schema, so stamp
    # the component-only combining semantic after that writer returns.
    from components import tag_component
    tag_component(saved, "coherent")
    return saved


def _attach_body_model_payload(
    payload: 'Dict[str, Any]',
    bodies: 'Dict[float, Dict[str, Any]]',
    generatrix: 'np.ndarray',
    *,
    azimuths_deg: 'Sequence[float]',
    elevations_deg: 'Sequence[float]',
    axis_az_deg: 'float',
    axis_el_deg: 'float',
    roll_deg: 'float',
) -> 'Dict[str, Any]':
    """Embed the compact reusable BoR model inside a radar-frame product."""

    frequencies = sorted(float(value) for value in bodies)
    if not frequencies:
        raise ValueError("Cannot embed an empty BoR body model.")
    first = bodies[frequencies[0]]
    aspects = np.asarray(first.get("theta_deg", []), dtype=float)
    if (
        aspects.ndim != 1
        or not len(aspects)
        or not np.all(np.isfinite(aspects))
        or np.any(np.diff(aspects) <= 0.0)
    ):
        raise ValueError("BoR body aspects must be finite and increasing.")
    vv = np.empty((len(aspects), len(frequencies)), dtype=np.complex128)
    hh = np.empty_like(vv)
    for index, frequency in enumerate(frequencies):
        body = bodies[frequency]
        current = np.asarray(body.get("theta_deg", []), dtype=float)
        av = np.asarray(body.get("amp_vv", []), dtype=np.complex128)
        ah = np.asarray(body.get("amp_hh", []), dtype=np.complex128)
        if (
            not np.array_equal(current, aspects)
            or av.shape != aspects.shape
            or ah.shape != aspects.shape
            or not np.all(np.isfinite(av.real) & np.isfinite(av.imag))
            or not np.all(np.isfinite(ah.real) & np.isfinite(ah.imag))
        ):
            raise ValueError(
                f"BoR body model at {frequency:g} GHz does not share one "
                "finite aspect grid."
            )
        vv[:, index] = av
        hh[:, index] = ah

    profile = np.asarray(generatrix, dtype=float)
    if (
        profile.ndim != 2
        or profile.shape[1] != 2
        or len(profile) < 2
        or not np.all(np.isfinite(profile))
    ):
        raise ValueError(
            "The embedded BoR profile must contain finite rho,z rows."
        )
    requested_azimuths, requested_elevations = validate_radar_grid(
        azimuths_deg, elevations_deg
    )
    primary_frequencies = np.asarray(payload["frequencies"], dtype=float)
    if not np.array_equal(primary_frequencies, np.asarray(frequencies)):
        raise ValueError(
            "Radar-frame and embedded body-model frequencies differ."
        )

    payload["body_model_metadata_json"] = np.asarray(json.dumps({
        "schema": _MONOSTATIC_BODY_MODEL_SCHEMA,
        "phase_reference": BOR_BODY_PHASE_REFERENCE,
        "amplitude_convention": PHYSICAL_3D_AMPLITUDE_CONVENTION,
        "complex_field_domain": BOR_BODY_FIELD_DOMAIN,
        "axis_meaning": _BODY_AZ_MEANING,
    }, sort_keys=True, separators=(",", ":")))
    payload["body_model_aspects_deg"] = aspects.astype(np.float64)
    payload["body_model_amp_vv_real"] = vv.real.astype(np.float64)
    payload["body_model_amp_vv_imag"] = vv.imag.astype(np.float64)
    payload["body_model_amp_hh_real"] = hh.real.astype(np.float64)
    payload["body_model_amp_hh_imag"] = hh.imag.astype(np.float64)
    payload["body_profile_rho_m"] = profile[:, 0].astype(np.float64)
    payload["body_profile_z_m"] = profile[:, 1].astype(np.float64)
    payload["requested_radar_grid_json"] = np.asarray(json.dumps({
        "schema": "ghost.workflow.requested-radar-grid.v1",
        "azimuths_deg": requested_azimuths,
        "elevations_deg": requested_elevations,
        "frequencies_ghz": frequencies,
        "axis_az_deg": float(axis_az_deg),
        "axis_el_deg": float(axis_el_deg),
        "roll_deg": float(roll_deg),
    }, sort_keys=True, separators=(",", ":")))
    return payload


def save_monostatic_grim(
    bodies: 'Dict[float, Dict[str, Any]]',
    generatrix: 'np.ndarray',
    out_path: 'str',
    *,
    azimuths_deg: 'Sequence[float]',
    elevations_deg: 'Sequence[float]',
    axis_az_deg: 'float' = 0.0,
    axis_el_deg: 'float' = 0.0,
    roll_deg: 'float' = 0.0,
    source_path: 'str' = "",
    history: 'str' = "",
    artifact_metadata: 'Optional[Dict[str, Any]]' = None,
) -> 'str':
    """Publish one complete BoR monostatic deliverable.

    Its primary arrays are the requested radar-frame azimuth/elevation VV, HH,
    and VH response. The exact body-frame aspect amplitudes and profile needed
    for later coherent feature placement travel inside the same GRIM, so users
    never have to manage separate radar-grid and body-model products.
    """

    frequencies = sorted(float(value) for value in bodies)
    require_body_radar_support(
        bodies,
        frequencies,
        azimuths_deg,
        elevations_deg,
        axis_az_deg,
        axis_el_deg,
    )
    destination = os.path.abspath(
        out_path if str(out_path).lower().endswith(".grim")
        else str(out_path) + ".grim"
    )
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = os.path.join(
        os.path.dirname(destination),
        f".{os.path.basename(destination)}.tmp.{os.getpid()}.grim",
    )
    try:
        export_radar_grim(
            temporary,
            bor_result=bodies,
            placements=[],
            generatrix=generatrix,
            frequencies_ghz=frequencies,
            azimuths_deg=azimuths_deg,
            elevations_deg=elevations_deg,
            axis_az_deg=axis_az_deg,
            axis_el_deg=axis_el_deg,
            roll_deg=roll_deg,
            source_path=source_path,
            history=(history or "BoR monostatic response"),
        )
        with np.load(temporary, allow_pickle=False) as stored:
            payload = {
                key: np.array(stored[key], copy=True) for key in stored.files
            }
        _attach_body_model_payload(
            payload,
            bodies,
            generatrix,
            azimuths_deg=azimuths_deg,
            elevations_deg=elevations_deg,
            axis_az_deg=axis_az_deg,
            axis_el_deg=axis_el_deg,
            roll_deg=roll_deg,
        )
        for key, value in dict(artifact_metadata or {}).items():
            payload[str(key)] = np.asarray(value)
        from grim_io import _save_grim_npz
        _save_grim_npz(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _validate_declared_coherent_base(base_payload, label):
    """Validate a GUI-derived platform field under an explicit declaration.

    GRIM may retain only sigma and phase after a derived operation. Selecting
    that file as BASE_MONOSTATIC_GRIM attests the missing common-origin,
    radar-frame coherent semantics; metadata that remains must not contradict
    the declaration. The returned payload is a canonical in-memory view used
    only for validation and summation.
    """
    from components import (
        COMPONENT_AMPLITUDE_CONVENTION,
        COMPONENT_COMPLEX_FIELD_DOMAIN,
        COMPONENT_PHASE_REFERENCE,
        validate_component_schema,
    )

    candidate = dict(base_payload)
    if "combine_role" in candidate:
        role = _metadata_text(candidate, "combine_role", label).strip().lower()
        if role != "coherent":
            raise ValueError(
                f"{label}: BASE_MONOSTATIC_GRIM is explicitly tagged "
                f"combine_role={role!r}; a power-only field cannot receive "
                "coherent placed features."
            )
    candidate["combine_role"] = np.asarray("coherent")

    units = _require_linear_quantity(candidate, label, "sigma_3d")
    units["rcs_log_unit"] = "dBsm"
    units.setdefault("azimuth", "deg")
    units.setdefault("elevation", "deg")
    units.setdefault("frequency", "GHz")
    candidate["units"] = np.asarray(json.dumps(units, sort_keys=True))

    expected = {
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "phase_reference": COMPONENT_PHASE_REFERENCE,
        "amplitude_convention": COMPONENT_AMPLITUDE_CONVENTION,
        "complex_field_domain": COMPONENT_COMPLEX_FIELD_DOMAIN,
    }
    for key, required in expected.items():
        # The BASE_MONOSTATIC_GRIM selection supersedes descriptive metadata
        # that a GUI may omit or copy from an input grid. The loaded numerical
        # sigma/phase normalization and combine_role='power' remain hard gates.
        candidate[key] = np.asarray(required)

    amplitude = np.asarray(candidate["_amp"], dtype=np.complex128)
    candidate["rcs_amp_real"] = amplitude.real.astype(np.float64)
    candidate["rcs_amp_imag"] = amplitude.imag.astype(np.float64)
    candidate["raw_complex_amplitude_preserved"] = np.asarray(True)
    validate_component_schema(candidate, label)
    return candidate


def _canonical_3d_channel_indices(polarizations, label, *, require_all=True):
    """Return (channel names, indices) in canonical radar order."""
    indices = {}
    for index, raw in enumerate(np.asarray(polarizations).ravel()):
        value = str(raw).strip().upper()
        canonical = (
            "VV" if value in {"VV", "V", "VERTICAL"}
            else "HH" if value in {"HH", "H", "HORIZONTAL"}
            else "VH" if value in {"VH", "HV"}
            else value
        )
        if canonical not in {"VV", "HH", "VH"}:
            raise ValueError(
                f"{label}: unsupported polarization label {raw!r}; use VV, "
                "HH, or reciprocal VH/HV."
            )
        if canonical in indices:
            raise ValueError(
                f"{label}: duplicate polarization alias for {canonical}."
            )
        indices[canonical] = index
    required = {"VV", "HH", "VH"}
    if require_all and set(indices) != required:
        raise ValueError(
            f"{label}: require exactly VV, HH, and reciprocal VH/HV; got "
            f"{[str(value) for value in np.asarray(polarizations).ravel()]}."
        )
    channels = [channel for channel in ("VV", "HH", "VH") if channel in indices]
    if not channels:
        raise ValueError(f"{label}: no usable radar polarization channels.")
    return channels, [indices[channel] for channel in channels]


def add_features_to_monostatic_grim(
    base_path: 'str',
    out_path: 'str',
    *,
    placements: 'Sequence[Dict[str, Any]]' = (),
    points: 'Sequence[Dict[str, Any]]' = (),
    corners: 'Sequence[Dict[str, Any]]' = (),
    occluder=None,
    radar_grid: 'Optional[Dict[str, Any]]' = None,
    surface_normal_fn=None,
    declared_coherent_base: 'bool' = False,
    feature_provenance: 'Optional[Dict[str, Any]]' = None,
    history: 'str' = "",
) -> 'str':
    """Coherently add placed features to one monostatic deliverable.

    The existing radar-frame field is retained and the newly evaluated feature
    field is added sample-by-sample.  Writing is atomic and may target a new
    path or intentionally replace ``base_path``.
    """

    base = os.path.abspath(str(base_path))
    if not os.path.isfile(base):
        raise FileNotFoundError(f"Base monostatic GRIM does not exist: {base}")
    base_payload = _load_grim(base)
    label = os.path.basename(base)
    if declared_coherent_base:
        validated_base = _validate_declared_coherent_base(base_payload, label)
    else:
        from components import validate_component_schema
        validate_component_schema(base_payload, label)
        validated_base = base_payload
    embedded_grid = load_body_requested_radar_grid(base)
    grid = dict(radar_grid) if radar_grid is not None else embedded_grid
    if grid is None:
        raise ValueError(
            f"{base}: this is not a self-contained BoR result. Supply "
            "radar_grid with frequencies_ghz, azimuths_deg, elevations_deg, "
            "axis_az_deg, axis_el_deg, and roll_deg for an external platform."
        )
    required_grid = {
        "frequencies_ghz", "azimuths_deg", "elevations_deg",
        "axis_az_deg", "axis_el_deg",
    }
    missing_grid = sorted(required_grid - set(grid))
    if missing_grid:
        raise ValueError(f"radar_grid is missing {missing_grid}.")

    profile = None
    if embedded_grid is not None:
        # Structural validation of the embedded body model; no certification
        # policy is imposed here.
        load_body_grim(base)
        profile = load_body_profile_grim(base)

    destination = os.path.abspath(
        out_path if str(out_path).lower().endswith(".grim")
        else str(out_path) + ".grim"
    )
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    component_tmp = os.path.join(
        os.path.dirname(destination),
        f".{os.path.basename(destination)}.features.{os.getpid()}.grim",
    )
    output_tmp = os.path.join(
        os.path.dirname(destination),
        f".{os.path.basename(destination)}.tmp.{os.getpid()}.grim",
    )
    try:
        export_radar_grim(
            component_tmp,
            bor_result=None,
            placements=placements,
            points=points,
            corners=corners,
            generatrix=profile,
            normal_fn=surface_normal_fn,
            occluder=occluder,
            frequencies_ghz=grid["frequencies_ghz"],
            azimuths_deg=grid["azimuths_deg"],
            elevations_deg=grid["elevations_deg"],
            axis_az_deg=grid["axis_az_deg"],
            axis_el_deg=grid["axis_el_deg"],
            roll_deg=grid.get("roll_deg", 0.0),
            history="placed coherent feature field",
        )
        component = _load_grim(component_tmp)
        for key in ("azimuths", "elevations", "frequencies"):
            if not np.array_equal(base_payload[key], component[key]):
                raise ValueError(
                    f"Feature field {key} does not match the base monostatic "
                    "grid."
                )
        base_channels, base_order = _canonical_3d_channel_indices(
            base_payload["polarizations"], label, require_all=False
        )
        component_channels, component_order = _canonical_3d_channel_indices(
            component["polarizations"], "placed feature field"
        )
        feature_lookup = dict(zip(component_channels, component_order))
        component_order = [feature_lookup[channel] for channel in base_channels]
        base_amplitude = np.asarray(
            validated_base["_amp"], dtype=np.complex128
        )[..., base_order]
        feature_amplitude = np.asarray(
            component["_amp"], dtype=np.complex128
        )[..., component_order]
        total = base_amplitude + feature_amplitude
        with np.load(base, allow_pickle=False) as stored:
            payload = {
                key: np.array(stored[key], copy=True) for key in stored.files
            }
        real = total.real.astype(np.float64)
        imag = total.imag.astype(np.float64)
        payload["rcs_amp_real"] = real
        payload["rcs_amp_imag"] = imag
        payload["polarizations"] = np.asarray(base_channels)
        for key in list(payload):
            if key.startswith("polarization_alias"):
                payload.pop(key, None)
        payload["rcs_power"] = (
            4.0 * math.pi * (real * real + imag * imag)
        ).astype(np.float32)
        payload["rcs_phase"] = np.angle(total).astype(np.float32)
        from components import (
            COMPONENT_AMPLITUDE_CONVENTION,
            COMPONENT_COMPLEX_FIELD_DOMAIN,
            COMPONENT_PHASE_REFERENCE,
        )
        payload["combine_role"] = np.asarray("coherent")
        payload["combine_role_note"] = np.asarray(
            "base platform plus coherently placed feature deltas"
        )
        payload["rcs_domain"] = np.asarray("power_phase")
        payload["power_domain"] = np.asarray("linear_rcs")
        payload["phase_reference"] = np.asarray(COMPONENT_PHASE_REFERENCE)
        payload["amplitude_convention"] = np.asarray(
            COMPONENT_AMPLITUDE_CONVENTION
        )
        payload["complex_field_domain"] = np.asarray(
            COMPONENT_COMPLEX_FIELD_DOMAIN
        )
        payload["raw_complex_amplitude_preserved"] = np.asarray(True)
        payload["history"] = (
            str(np.asarray(payload.get("history", "")).reshape(-1)[0])
            + " | " + (history or "coherently added placed features")
        ).strip(" |")
        for key in (
            "combination_estimate_power",
            "combination_estimate_mode",
            "combination_estimate_semantics",
        ):
            payload.pop(key, None)

        from workflow_provenance import sha256_file
        records = []
        if "feature_provenance_json" in payload:
            raw = np.asarray(payload["feature_provenance_json"]).reshape(()).item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            decoded = json.loads(str(raw))
            records = list(decoded) if isinstance(decoded, list) else [decoded]
        records.append({
            "schema": "ghost.workflow.coherent-feature-addition.v1",
            "source_monostatic_sha256": sha256_file(base),
            "line_feature_count": int(len(placements)),
            "compact_feature_count": int(len(points)),
            "corner_estimate_count": int(len(corners)),
            "details": dict(feature_provenance or {}),
        })
        payload["feature_provenance_json"] = np.asarray(json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ))
        from grim_io import _save_grim_npz
        _save_grim_npz(payload, output_tmp)
        os.replace(output_tmp, destination)
    finally:
        for path in (component_tmp, output_tmp):
            if os.path.exists(path):
                os.unlink(path)
    return destination
