
import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np

EPS = 1e-12

PRODUCTION_MESH_CONVERGENCE_DEFAULTS = {
    "fine_factor": 1.5,
    "rms_limit_db": 1.0,
    "max_abs_limit_db": 3.0,
    "complex_rms_limit": 0.02,
    "complex_max_limit": 0.05,
    "phase_rms_limit_deg": 5.0,
    "phase_max_limit_deg": 15.0,
    "phase_floor_relative": 1.0e-6,
}


def _sample_key(row: 'Dict[str, Any]') -> 'Tuple[float, float, float]':
    return (
        round(float(row.get("frequency_ghz", 0.0)), 9),
        round(float(row.get("theta_inc_deg", 0.0)), 9),
        round(float(row.get("theta_scat_deg", 0.0)), 9),
    )


def _ensure_properties_len(props: 'List[Any]', n: 'int' = 6) -> 'List[str]':
    out = [str(p) for p in list(props or [])]
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out



def scale_snapshot_panel_density(snapshot: 'Dict[str, Any]', fine_factor: 'float') -> 'Dict[str, Any]':
    """
    Return a deep-copied geometry snapshot with increased panel density.

    This scales the per-segment N/discretization property in `properties[1]` while
    preserving all geometric/material flags.

    When ``properties[1]`` is empty or zero (auto-density mode), the solver uses
    DEFAULT_PANELS_PER_WAVELENGTH.  The fine snapshot switches to an explicit
    panels-per-wavelength value (negative N) scaled by ``fine_factor`` so the
    convergence comparison is meaningful.
    """

    try:
        from rcs_solver import DEFAULT_PANELS_PER_WAVELENGTH
    except ImportError:
        DEFAULT_PANELS_PER_WAVELENGTH = 20

    factor = float(fine_factor)
    if factor <= 1.0:
        raise ValueError("fine_factor must be > 1.0.")

    out = copy.deepcopy(snapshot)
    segments = list(out.get("segments", []) or [])
    for seg in segments:
        props = _ensure_properties_len(seg.get("properties", []), 6)
        raw = str(props[1]).strip()
        try:
            base_n = int(round(float(raw or 0)))
        except Exception:
            base_n = 0

        if base_n == 0:
            # Auto-density mode: solver uses DEFAULT_PANELS_PER_WAVELENGTH ppw.
            # Switch to explicit negative-N (panels-per-wavelength) scaled by factor.
            fine_ppw = max(2, int(math.ceil(DEFAULT_PANELS_PER_WAVELENGTH * factor)))
            props[1] = str(-fine_ppw)
        elif base_n > 0:
            # Explicit panel count: scale directly.
            fine_n = max(base_n + 1, int(math.ceil(base_n * factor)))
            props[1] = str(fine_n)
        else:
            # Already in panels-per-wavelength mode (negative N): scale the ppw value.
            base_ppw = abs(base_n)
            fine_ppw = max(base_ppw + 1, int(math.ceil(base_ppw * factor)))
            props[1] = str(-fine_ppw)

        seg["properties"] = props
        seg["seg_type"] = props[0] if props[0] else seg.get("seg_type")
    out["segments"] = segments
    return out


def validate_mesh_convergence_policy(
    policy: 'Dict[str, Any]' = None,
) -> 'Dict[str, float]':
    """Return a complete, finite, fail-closed production mesh policy."""

    supplied = dict(policy or {})
    unknown = sorted(
        set(supplied) - set(PRODUCTION_MESH_CONVERGENCE_DEFAULTS)
    )
    if unknown:
        raise ValueError(
            "Unknown mesh convergence policy field(s): "
            + ", ".join(unknown)
        )
    complete = dict(PRODUCTION_MESH_CONVERGENCE_DEFAULTS)
    complete.update(supplied)
    validated = {}
    for name, raw_value in complete.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be a finite number."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        validated[name] = value

    if validated["fine_factor"] <= 1.0:
        raise ValueError("fine_factor must be > 1.0.")
    for name in (
        "rms_limit_db",
        "max_abs_limit_db",
        "complex_rms_limit",
        "complex_max_limit",
        "phase_rms_limit_deg",
        "phase_max_limit_deg",
    ):
        if validated[name] < 0.0:
            raise ValueError(f"{name} must be nonnegative.")
    if not 0.0 <= validated["phase_floor_relative"] < 1.0:
        raise ValueError(
            "phase_floor_relative must be in [0, 1)."
        )
    return validated



def evaluate_mesh_convergence(
    base_result: 'Dict[str, Any]',
    fine_result: 'Dict[str, Any]',
    rms_limit_db: 'float',
    max_abs_limit_db: 'float',
    complex_rms_limit: 'float' = 0.02,
    complex_max_limit: 'float' = 0.05,
    phase_rms_limit_deg: 'float' = 5.0,
    phase_max_limit_deg: 'float' = 15.0,
    phase_floor_relative: 'float' = 1.0e-6,
) -> 'Dict[str, Any]':
    """
    Compare two solve results point-by-point in magnitude and complex field.

    Matching is done on (frequency_ghz, theta_inc_deg, theta_scat_deg).

    The normalized complex error uses the peak field over the compared grid as
    its reference.  This remains meaningful at physical nulls, where a
    pointwise relative error and phase are undefined.  Phase is checked only
    where both meshes are above ``phase_floor_relative`` times that peak.
    """

    policy = validate_mesh_convergence_policy({
        "rms_limit_db": rms_limit_db,
        "max_abs_limit_db": max_abs_limit_db,
        "complex_rms_limit": complex_rms_limit,
        "complex_max_limit": complex_max_limit,
        "phase_rms_limit_deg": phase_rms_limit_deg,
        "phase_max_limit_deg": phase_max_limit_deg,
        "phase_floor_relative": phase_floor_relative,
    })
    phase_floor_fraction = policy["phase_floor_relative"]

    base_samples = list(base_result.get("samples", []) or [])
    fine_samples = list(fine_result.get("samples", []) or [])
    if not base_samples or not fine_samples:
        raise ValueError("Both base_result and fine_result must contain samples.")

    def unique_by_key(samples, label):
        indexed = {}
        for row in samples:
            key = _sample_key(row)
            if key in indexed:
                raise ValueError(
                    f"Mesh convergence {label} contains duplicate sample "
                    f"key {key}."
                )
            indexed[key] = row
        return indexed

    base_by_key = unique_by_key(base_samples, "base_result")
    fine_by_key = unique_by_key(fine_samples, "fine_result")
    base_keys = set(base_by_key)
    fine_keys = set(fine_by_key)
    missing = sorted(base_keys - fine_keys)
    extra = sorted(fine_keys - base_keys)
    if missing or extra:
        raise ValueError(
            "Mesh convergence sample grids differ: "
            f"{len(missing)} missing and {len(extra)} extra fine-result "
            "sample point(s)."
        )

    matched: 'List[Tuple[Dict[str, Any], Dict[str, Any]]]' = []

    for row in base_samples:
        key = _sample_key(row)
        matched.append((row, fine_by_key[key]))
    if not matched:
        raise ValueError("Mesh convergence comparison produced no overlapping samples.")

    def finite_float(row, key):
        if key not in row:
            raise ValueError(
                "Mesh convergence requires authoritative complex amplitudes; "
                f"sample {_sample_key(row)} is missing {key!r}."
            )
        try:
            value = float(row[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Mesh convergence sample {_sample_key(row)} has invalid "
                f"{key}={row[key]!r}."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Mesh convergence sample {_sample_key(row)} has non-finite "
                f"{key}={row[key]!r}."
            )
        return value

    def sample_db(row):
        if "rcs_db" in row:
            value = float(row["rcs_db"])
        else:
            linear = finite_float(row, "rcs_linear")
            if linear < 0.0:
                raise ValueError(
                    f"Mesh convergence sample {_sample_key(row)} has negative "
                    f"rcs_linear={linear!r}."
                )
            value = 10.0 * math.log10(max(linear, EPS))
        if not math.isfinite(value):
            raise ValueError(
                f"Mesh convergence sample {_sample_key(row)} has non-finite "
                f"rcs_db={value!r}."
            )
        return value

    base_amp = np.asarray([
        complex(
            finite_float(row, "rcs_amp_real"),
            finite_float(row, "rcs_amp_imag"),
        )
        for row, _other in matched
    ], dtype=np.complex128)
    fine_amp = np.asarray([
        complex(
            finite_float(other, "rcs_amp_real"),
            finite_float(other, "rcs_amp_imag"),
        )
        for _row, other in matched
    ], dtype=np.complex128)
    deltas = np.asarray([
        sample_db(row) - sample_db(other)
        for row, other in matched
    ], dtype=float)

    abs_deltas = np.abs(deltas)
    rms_db = float(np.sqrt(np.mean(deltas * deltas)))
    max_abs_db = float(np.max(abs_deltas))

    # Normalize independently per frequency. A strong return in one band must
    # not hide a phase or complex-field failure in a weaker band.
    frequency_groups = {}
    for index, (row, _other) in enumerate(matched):
        frequency = round(float(row.get("frequency_ghz", 0.0)), 9)
        frequency_groups.setdefault(frequency, []).append(index)
    complex_errors = np.zeros(len(matched), dtype=float)
    phase_error_groups = []
    peak_by_frequency = {}
    for frequency, raw_indices in sorted(frequency_groups.items()):
        indices = np.asarray(raw_indices, dtype=int)
        peak = float(max(
            np.max(np.abs(base_amp[indices])),
            np.max(np.abs(fine_amp[indices])),
        ))
        peak_by_frequency[str(frequency)] = peak
        if not math.isfinite(peak) or peak <= 0.0:
            # Two exact-zero fields agree exactly and have no meaningful phase.
            continue
        complex_errors[indices] = (
            np.abs(base_amp[indices] - fine_amp[indices]) / peak
        )
        phase_floor = phase_floor_fraction * peak
        phase_mask = (
            (np.abs(base_amp[indices]) > phase_floor)
            & (np.abs(fine_amp[indices]) > phase_floor)
        )
        if np.any(phase_mask):
            phase_error_groups.append(np.abs(np.degrees(np.angle(
                fine_amp[indices][phase_mask]
                * np.conj(base_amp[indices][phase_mask])
            ))))
    phase_errors_deg = (
        np.concatenate(phase_error_groups)
        if phase_error_groups
        else np.zeros(0, dtype=float)
    )
    peak_amp = float(max(peak_by_frequency.values()))

    complex_rms = float(np.sqrt(np.mean(complex_errors ** 2)))
    complex_max = float(np.max(complex_errors))
    if phase_errors_deg.size:
        phase_rms_deg = float(np.sqrt(np.mean(phase_errors_deg ** 2)))
        phase_max_deg = float(np.max(phase_errors_deg))
    else:
        phase_rms_deg = 0.0
        phase_max_deg = 0.0

    violations: 'List[str]' = []
    if rms_db > float(rms_limit_db):
        violations.append(f"RMS dB delta {rms_db:.6g} exceeds limit {float(rms_limit_db):.6g}")
    if max_abs_db > float(max_abs_limit_db):
        violations.append(
            f"Max |dB| delta {max_abs_db:.6g} exceeds limit {float(max_abs_limit_db):.6g}"
        )
    if complex_rms > float(complex_rms_limit):
        violations.append(
            f"RMS normalized complex-field delta {complex_rms:.6g} exceeds "
            f"limit {float(complex_rms_limit):.6g}"
        )
    if complex_max > float(complex_max_limit):
        violations.append(
            f"Max normalized complex-field delta {complex_max:.6g} exceeds "
            f"limit {float(complex_max_limit):.6g}"
        )
    if phase_rms_deg > float(phase_rms_limit_deg):
        violations.append(
            f"RMS phase delta {phase_rms_deg:.6g} deg exceeds limit "
            f"{float(phase_rms_limit_deg):.6g} deg"
        )
    if phase_max_deg > float(phase_max_limit_deg):
        violations.append(
            f"Max phase delta {phase_max_deg:.6g} deg exceeds limit "
            f"{float(phase_max_limit_deg):.6g} deg"
        )
    passed = not violations

    return {
        "passed": bool(passed),
        "sample_count": int(len(deltas)),
        "rms_db": rms_db,
        "max_abs_db": max_abs_db,
        "mean_db": float(np.mean(deltas)),
        "median_abs_db": float(np.median(abs_deltas)),
        "complex_rms_normalized": complex_rms,
        "complex_max_normalized": complex_max,
        "complex_reference_peak_amplitude": peak_amp,
        "complex_reference_peak_amplitude_by_frequency":
            peak_by_frequency,
        "phase_sample_count": int(phase_errors_deg.size),
        "phase_rms_deg": phase_rms_deg,
        "phase_max_deg": phase_max_deg,
        "limits": {
            "rms_db": float(rms_limit_db),
            "max_abs_db": float(max_abs_limit_db),
            "complex_rms_normalized": float(complex_rms_limit),
            "complex_max_normalized": float(complex_max_limit),
            "phase_rms_deg": float(phase_rms_limit_deg),
            "phase_max_deg": float(phase_max_limit_deg),
            "phase_floor_relative": float(phase_floor_relative),
        },
        "violations": violations,
        "reason": "; ".join(violations) if violations else "mesh convergence passed",
    }
