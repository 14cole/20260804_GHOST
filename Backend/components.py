#!/usr/bin/env python3
"""
Component files: how step 4 knows what may interfere with what.

Steps 3a / 3b / 3c each export ONE COMPONENT ALONE as an az x el x freq x pol
.grim.  Step 4 sums them.  But they are not all the same KIND of number:

  role = "coherent"   the complex amplitude is trustworthy against other
                      components.  Doors and wings: the line expansion is
                      linear and shares one far-field phase reference, so
                      summing separately-solved placements equals solving them
                      together (exact, ~1e-8).

  role = "power"      the amplitude's PHASE is not trustworthy against other
                      components, so step 4 includes this one's POWER only in a
                      separately labelled statistical/engineering estimate.
                      The primary rcs_power remains the power of the stored
                      coherent field.  A PO-level wing-root corner whose
                      internal double-bounce phase is not tracked is this kind.

The role is written INTO the file, not inferred from the folder, so a component
carries its own combining rule wherever it is moved.  A file with no tag is
rejected: field metadata cannot reveal whether an engineering term has a
trustworthy phase relative to the other components.

Do not place a phase-unknown term in the same component file as a coherent
term: their internal interference would already be invented before the role
tag could help.  Step 3b therefore writes the wing and corner estimate as
separate files.
"""

import json
import math
import os
from typing import Any, Dict

import numpy as np

ROLES = ("coherent", "power")
_KEY = "combine_role"

# Component files consumed by step 4 are radar-frame, physical 3-D far-field
# amplitudes.  Coherent addition is meaningful only when every file uses this
# exact common origin, time convention, polarization frame, and normalization.
COMPONENT_PHASE_REFERENCE = (
    "origin=(0,0,0) vehicle frame, convention=exp(+jwt), "
    "radar earth-frame V/H monostatic amplitude"
)
COMPONENT_AMPLITUDE_CONVENTION = (
    "F physical far-field amplitude; sigma_3d=4*pi*|F|^2"
)
COMPONENT_COMPLEX_FIELD_DOMAIN = (
    "coherent_radar_frame_far_field_amplitude"
)


def _scalar_text(value: 'Any', key: 'str', label: 'str') -> 'str':
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(
            f"{label}: metadata {key!r} must contain exactly one value.")
    return str(arr.reshape(-1)[0])


def _schema_error(label: 'str', detail: 'str') -> 'ValueError':
    return ValueError(f"{label}: invalid component .grim schema: {detail}")


def combine_component_fields(body_amp, coherent_amps=(), power_amps=(),
                             mode: 'str' = "coherent"):
    """Combine component arrays without conflating fields and estimates.

    Returns ``(total_amp, coherent_power, estimate_power)``.  The first two
    always obey ``coherent_power = 4*pi*abs(total_amp)**2``.  Phase-unknown
    ``power_amps`` affect only the separately labelled estimate.
    """
    mode = str(mode).strip().lower()
    if mode not in ("coherent", "hybrid", "envelope"):
        raise ValueError(f"unknown combination estimate mode {mode!r}.")
    body = np.asarray(body_amp, dtype=complex)
    coherent = [np.asarray(a, dtype=complex) for a in coherent_amps]
    unknown = [np.asarray(a, dtype=complex) for a in power_amps]
    for label, arrays in (("coherent", coherent), ("power", unknown)):
        for a in arrays:
            if a.shape != body.shape:
                raise ValueError(
                    f"{label} component shape {a.shape} != body shape "
                    f"{body.shape}.")
            if not np.all(np.isfinite(a.real) & np.isfinite(a.imag)):
                raise ValueError(f"{label} component contains non-finite values.")
    if not np.all(np.isfinite(body.real) & np.isfinite(body.imag)):
        raise ValueError("body amplitude contains non-finite values.")

    cluster = sum(coherent, np.zeros_like(body))
    total = body + cluster
    coherent_power = 4.0 * np.pi * np.abs(total) ** 2
    unknown_power = sum(
        (np.abs(a) ** 2 for a in unknown), np.zeros(body.shape, dtype=float))
    if mode == "coherent":
        estimate = coherent_power + 4.0 * np.pi * unknown_power
    elif mode == "hybrid":
        estimate = 4.0 * np.pi * (
            np.abs(body) ** 2 + np.abs(cluster) ** 2 + unknown_power)
    else:
        estimate = 4.0 * np.pi * (
            np.abs(body) ** 2
            + sum((np.abs(a) ** 2 for a in coherent),
                  np.zeros(body.shape, dtype=float))
            + unknown_power)
    return total, coherent_power, estimate


def tag_component(path: 'str', role: 'str', note: 'str' = "") -> 'str':
    """Stamp a component .grim with how step 4 must combine it."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}.")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d[_KEY] = np.asarray(role)
    if note:
        d["combine_role_note"] = np.asarray(note)
    for key in ("rcs_amp_real", "rcs_amp_imag"):
        if key in d:
            d[key] = np.asarray(d[key], dtype=np.float64)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **d)
    return path


def component_role(grim: 'Dict[str, Any]') -> 'str':
    """Read the explicit role back and reject missing/unknown semantics."""
    if _KEY not in grim:
        raise ValueError(
            "missing combine_role; explicitly tag the component 'coherent' "
            "or 'power' before combining it."
        )
    role = _scalar_text(grim[_KEY], _KEY, "component").strip().lower()
    if role not in ROLES:
        raise ValueError(
            f"unknown combine_role {role!r}; expected one of {ROLES}.")
    return role


def validate_component_schema(grim: 'Dict[str, Any]',
                              label: 'str' = "component") -> 'str':
    """Fail closed unless a step-4 input is a compatible physical 3-D field.

    Returns the validated combine role.  The primary linear array must carry
    sigma_3d in dBsm display units, while the complex arrays must be the common
    vehicle-origin radar-frame far-field amplitude ``F``.  Stored power is
    checked against the *stored* complex samples, including exact nulls; a
    display floor in ``rcs_power`` is therefore rejected.
    """
    try:
        role = component_role(grim)
    except ValueError as exc:
        raise _schema_error(label, str(exc)) from exc

    try:
        units_raw = _scalar_text(grim["units"], "units", label)
        units = json.loads(units_raw)
    except KeyError as exc:
        raise _schema_error(label, "missing required metadata 'units'.") from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise _schema_error(label, "'units' is not valid JSON.") from exc
    if not isinstance(units, dict):
        raise _schema_error(label, "'units' must decode to an object.")
    if str(units.get("rcs_linear_quantity", "")).strip().lower() != "sigma_3d":
        raise _schema_error(
            label, "units.rcs_linear_quantity must be 'sigma_3d'.")
    if str(units.get("rcs_log_unit", "")).strip().lower() != "dbsm":
        raise _schema_error(label, "units.rcs_log_unit must be 'dBsm'.")

    expected_metadata = {
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "phase_reference": COMPONENT_PHASE_REFERENCE,
        "amplitude_convention": COMPONENT_AMPLITUDE_CONVENTION,
        "complex_field_domain": COMPONENT_COMPLEX_FIELD_DOMAIN,
    }
    for key, expected in expected_metadata.items():
        if key not in grim:
            raise _schema_error(label, f"missing required metadata {key!r}.")
        got = _scalar_text(grim[key], key, label)
        if got != expected:
            raise _schema_error(
                label, f"{key}={got!r}; require {expected!r}.")

    if "raw_complex_amplitude_preserved" not in grim:
        raise _schema_error(
            label, "missing 'raw_complex_amplitude_preserved'.")
    raw = np.asarray(grim["raw_complex_amplitude_preserved"])
    raw_value = raw.reshape(-1)[0] if raw.size == 1 else None
    if (raw.size != 1
            or not isinstance(raw_value, (bool, np.bool_))
            or not bool(raw_value)):
        raise _schema_error(
            label, "raw_complex_amplitude_preserved must be true.")

    axis_keys = ("azimuths", "elevations", "frequencies", "polarizations")
    for key in axis_keys:
        if key not in grim:
            raise _schema_error(label, f"missing required array {key!r}.")
    az = np.asarray(grim["azimuths"], dtype=float).ravel()
    el = np.asarray(grim["elevations"], dtype=float).ravel()
    fr = np.asarray(grim["frequencies"], dtype=float).ravel()
    pol = np.asarray(grim["polarizations"]).ravel()
    if not len(az) or not len(el) or not len(fr) or not len(pol):
        raise _schema_error(label, "coordinate and polarization axes must be nonempty.")
    if not (np.all(np.isfinite(az)) and np.all(np.isfinite(el))
            and np.all(np.isfinite(fr)) and np.all(fr > 0.0)):
        raise _schema_error(
            label, "coordinate axes must be finite and frequencies positive.")
    shape = (len(az), len(el), len(fr), len(pol))

    field_keys = ("rcs_amp_real", "rcs_amp_imag", "rcs_phase", "rcs_power")
    for key in field_keys:
        if key not in grim:
            raise _schema_error(label, f"missing required array {key!r}.")
        if np.asarray(grim[key]).shape != shape:
            raise _schema_error(
                label, f"{key} shape {np.asarray(grim[key]).shape} != {shape}.")
    real = np.asarray(grim["rcs_amp_real"], dtype=float)
    imag = np.asarray(grim["rcs_amp_imag"], dtype=float)
    phase = np.asarray(grim["rcs_phase"], dtype=float)
    power = np.asarray(grim["rcs_power"], dtype=float)
    if not (np.all(np.isfinite(real)) and np.all(np.isfinite(imag))
            and np.all(np.isfinite(phase)) and np.all(np.isfinite(power))):
        raise _schema_error(label, "field arrays contain NaN or infinity.")
    if np.any(power < 0.0):
        raise _schema_error(label, "rcs_power contains negative values.")

    amp = real + 1j * imag
    live = (
        np.abs(real) > np.finfo(np.float32).tiny
    ) | (
        np.abs(imag) > np.finfo(np.float32).tiny
    )
    if np.any(live):
        phase_error = np.abs(np.angle(
            np.exp(1j * (phase[live] - np.angle(amp[live])))))
        if np.max(phase_error) > 2.0e-5:
            raise _schema_error(
                label, "rcs_phase is inconsistent with rcs_amp_real/imag.")

    with np.errstate(over="ignore", invalid="ignore"):
        amplitude_squared = real * real + imag * imag
        expected_power = 4.0 * math.pi * amplitude_squared
    if not np.all(np.isfinite(amplitude_squared)) or not np.all(
        np.isfinite(expected_power)
    ):
        raise _schema_error(
            label,
            "complex amplitude is too large to form finite physical power.",
        )
    # Allow only float32 serialization roundoff (and unavoidable subnormal
    # underflow), never an engineering display floor at a physical null.
    tolerance = (
        8.0 * np.finfo(np.float32).eps
        * np.maximum(expected_power, power)
        + np.finfo(np.float32).tiny
    )
    if np.any(np.abs(power - expected_power) > tolerance):
        worst = float(np.max(np.abs(power - expected_power)))
        raise _schema_error(
            label, "require rcs_power=4*pi*|F|^2 for the stored amplitude "
            f"(worst absolute mismatch {worst:.3e}).")
    return role


def keep_pols(path: 'str', wanted) -> 'str':
    """Trim a component file to POLARIZATIONS, in that order.

    The exporter always writes VV, HH, VH; a study that only wants co-pol should
    not carry a cross-pol column around, and step 4 requires every component to
    agree on the pol list.
    """
    import json
    wanted = list(wanted)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    have = [str(p) for p in np.asarray(d["polarizations"]).ravel()]
    gone = [p for p in wanted if p not in have]
    if gone:
        raise SystemExit(f"{os.path.basename(path)}: POLARIZATIONS {gone} not "
                         f"produced (have {have}).")
    idx = [have.index(p) for p in wanted]
    if idx == list(range(len(have))):
        return path
    for k in ("rcs_power", "rcs_phase", "rcs_amp_real", "rcs_amp_imag"):
        if k in d:
            d[k] = np.asarray(d[k])[:, :, :, idx]
    for key in ("rcs_amp_real", "rcs_amp_imag"):
        if key in d:
            d[key] = np.asarray(d[key], dtype=np.float64)
    d["polarizations"] = np.asarray(wanted, dtype=str)
    d["polarization_alias_primary"] = ",".join(wanted)
    d["polarization_aliases_json"] = json.dumps(wanted)
    with open(path, "wb") as fh:
        np.savez(fh, **d)
    return path
