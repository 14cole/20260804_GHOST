#!/usr/bin/env python3
"""Compare direct full-wave featured bodies with GHOST feature reconstruction.

The comparison is deliberately performed on the complex far-field amplitude,
not on dBsm alone. Edit REFERENCE_PAIRS and the release gates, then run:

    python Backend/validate_feature_reconstruction.py

No fitted phase, amplitude, or coordinate correction is applied. A best-fit
global phase is reported only as a diagnostic because applying it would hide a
phase-origin or time-sign error in the external reference.
"""

import json
import math
from pathlib import Path

import numpy as np

from feature_sum import _load_grim

# =============================================================================
# USER SETTINGS
# =============================================================================

REFERENCE_PAIRS = [
    # First validate the clean-body agreement, then the featured result:
    # {"name": "clean baseline", "truth": "3d_clean.grim",
    #  "prediction": "bor_clean.grim"},
    # {"name": "door reconstruction", "truth": "3d_door.grim",
    #  "prediction": "bor_with_door.grim"},
]

# Samples more than this far below the truth-field peak are excluded from
# pointwise magnitude/phase statistics because phase at a null is undefined.
# They remain included in normalized complex-field error.
ACTIVE_FIELD_FLOOR_DB = -40.0

# Initial engineering release gates only. No supporting calibration fixture is
# shipped; projects must justify or tighten them with their own independent
# reference family before releasing a feature library.
MAX_NORMALIZED_COMPLEX_RMS = 0.25
MAX_MAGNITUDE_ERROR_P95_DB = 3.5
MAX_PHASE_ERROR_RMS_DEG = 25.0
MIN_COMPLEX_COHERENCE = 0.95

REPORT_JSON = "feature_validation_report.json"

# =============================================================================


def _text(payload, key, label):
    if key not in payload:
        raise ValueError(f"{label}: missing coherent metadata {key!r}.")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{label}: metadata {key!r} is not scalar.")
    return str(value.reshape(-1)[0])


def _require_compatible(truth, prediction, truth_label, prediction_label):
    for key in ("azimuths", "elevations", "frequencies", "polarizations"):
        if not np.array_equal(np.asarray(truth[key]), np.asarray(prediction[key])):
            raise ValueError(
                f"{truth_label} and {prediction_label} have different {key}; "
                "export the exact same monostatic grid without interpolation."
            )
    for key in (
        "phase_reference", "amplitude_convention", "complex_field_domain"
    ):
        expected = _text(truth, key, truth_label)
        actual = _text(prediction, key, prediction_label)
        if expected != actual:
            raise ValueError(
                f"{truth_label} and {prediction_label} disagree on {key}: "
                f"{expected!r} versus {actual!r}."
            )
    for label, payload in ((truth_label, truth), (prediction_label, prediction)):
        try:
            units = json.loads(_text(payload, "units", label))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}: units metadata is invalid JSON.") from exc
        if (
            units.get("rcs_linear_quantity") != "sigma_3d"
            or str(units.get("rcs_log_unit", "")).lower() != "dbsm"
        ):
            raise ValueError(f"{label}: reference must be physical sigma_3d/dBsm.")


def compare_grims(
    truth_path,
    prediction_path,
    *,
    active_floor_db=ACTIVE_FIELD_FLOOR_DB,
    max_normalized_rms=MAX_NORMALIZED_COMPLEX_RMS,
    max_magnitude_p95_db=MAX_MAGNITUDE_ERROR_P95_DB,
    max_phase_rms_deg=MAX_PHASE_ERROR_RMS_DEG,
    min_coherence=MIN_COMPLEX_COHERENCE,
):
    """Return unfitted complex-field agreement metrics and a pass/fail result."""

    truth_label = str(Path(truth_path).resolve())
    prediction_label = str(Path(prediction_path).resolve())
    truth = _load_grim(truth_label)
    prediction = _load_grim(prediction_label)
    _require_compatible(truth, prediction, truth_label, prediction_label)
    reference = np.asarray(truth["_amp"], dtype=np.complex128)
    estimate = np.asarray(prediction["_amp"], dtype=np.complex128)
    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("Truth and prediction complex fields have incompatible shapes.")

    error = estimate - reference
    reference_rms = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    error_rms = float(np.sqrt(np.mean(np.abs(error) ** 2)))
    tiny = np.finfo(float).tiny
    normalized_rms = error_rms / max(reference_rms, tiny)
    reference_peak = float(np.max(np.abs(reference)))
    active_threshold = reference_peak * 10.0 ** (float(active_floor_db) / 20.0)
    active = np.abs(reference) >= active_threshold
    if not np.any(active):
        raise ValueError("Truth field has no finite nonzero samples.")
    magnitude_error_db = 20.0 * np.log10(
        np.maximum(np.abs(estimate[active]), tiny)
        / np.maximum(np.abs(reference[active]), tiny)
    )
    phase_error_deg = np.degrees(np.angle(
        estimate[active] * np.conjugate(reference[active])
    ))
    magnitude_p95 = float(np.percentile(np.abs(magnitude_error_db), 95.0))
    magnitude_max = float(np.max(np.abs(magnitude_error_db)))
    phase_rms = float(np.sqrt(np.mean(phase_error_deg ** 2)))
    phase_p95 = float(np.percentile(np.abs(phase_error_deg), 95.0))

    flat_reference = reference.ravel()
    flat_estimate = estimate.ravel()
    inner = np.vdot(flat_reference, flat_estimate)
    coherence = float(
        abs(inner)
        / max(
            float(np.linalg.norm(flat_reference) * np.linalg.norm(flat_estimate)),
            tiny,
        )
    )
    best_global_phase_deg = float(np.degrees(np.angle(inner)))

    polarizations = [str(value) for value in np.asarray(
        truth["polarizations"]
    ).ravel()]
    per_channel = {}
    for index, polarization in enumerate(polarizations):
        channel_reference = reference[..., index]
        channel_error = error[..., index]
        denominator = float(np.sqrt(np.mean(np.abs(channel_reference) ** 2)))
        per_channel[polarization] = {
            "normalized_complex_rms": float(
                np.sqrt(np.mean(np.abs(channel_error) ** 2))
                / max(denominator, tiny)
            ),
            "truth_peak_dbsm": float(
                10.0 * math.log10(
                    max(4.0 * math.pi * float(np.max(np.abs(channel_reference)) ** 2), tiny)
                )
            ),
        }

    gates = {
        "normalized_complex_rms": normalized_rms <= float(max_normalized_rms),
        "magnitude_error_p95_db": magnitude_p95 <= float(max_magnitude_p95_db),
        "phase_error_rms_deg": phase_rms <= float(max_phase_rms_deg),
        "complex_coherence": coherence >= float(min_coherence),
    }
    return {
        "schema": "ghost.validation.feature-reconstruction.v1",
        "truth": truth_label,
        "prediction": prediction_label,
        "shape": list(reference.shape),
        "active_field_floor_db": float(active_floor_db),
        "active_sample_count": int(np.count_nonzero(active)),
        "normalized_complex_rms": normalized_rms,
        "magnitude_error_p95_db": magnitude_p95,
        "magnitude_error_max_db": magnitude_max,
        "phase_error_rms_deg": phase_rms,
        "phase_error_p95_deg": phase_p95,
        "complex_coherence": coherence,
        "best_fit_global_phase_diagnostic_deg": best_global_phase_deg,
        "per_channel": per_channel,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main():
    if not REFERENCE_PAIRS:
        raise SystemExit("Configure at least one REFERENCE_PAIRS entry.")
    comparisons = []
    failed = False
    for pair in REFERENCE_PAIRS:
        result = compare_grims(pair["truth"], pair["prediction"])
        result["name"] = str(pair["name"])
        comparisons.append(result)
        failed = failed or not result["passed"]
        print(
            f"{result['name']}: {'PASS' if result['passed'] else 'FAIL'} | "
            f"complex RMS={result['normalized_complex_rms']:.4f}, "
            f"|dB| p95={result['magnitude_error_p95_db']:.2f}, "
            f"phase RMS={result['phase_error_rms_deg']:.2f} deg, "
            f"coherence={result['complex_coherence']:.5f}"
        )
        print(
            "  diagnostic best-fit global phase (not applied): "
            f"{result['best_fit_global_phase_diagnostic_deg']:+.2f} deg"
        )
    report = {
        "schema": "ghost.validation.feature-reconstruction-report.v1",
        "passed": not failed,
        "comparisons": comparisons,
    }
    destination = Path(REPORT_JSON).expanduser().resolve()
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
