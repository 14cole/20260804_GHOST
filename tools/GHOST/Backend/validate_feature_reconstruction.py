#!/usr/bin/env python3
"""Compare direct full-wave featured bodies with GHOST feature reconstruction.

The comparison is deliberately performed on the complex far-field amplitude,
not on dBsm alone. Edit REFERENCE_PAIRS and the release gates, then run:

    python Backend/validate_feature_reconstruction.py

No fitted phase, amplitude, or coordinate correction is applied. A best-fit
global phase is reported only as a diagnostic because applying it would hide a
phase-origin or time-sign error in the external reference.
"""

import argparse
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

CASE_MANIFEST_SCHEMA = "ghost.validation.feature-cases.v1"
CASE_REQUIRED_PATHS = (
    "clean_truth",
    "clean_prediction",
    "featured_truth",
    "featured_prediction",
)

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


def _comparison_metrics(
    reference,
    estimate,
    polarizations,
    *,
    active_floor_db=ACTIVE_FIELD_FLOOR_DB,
    max_normalized_rms=MAX_NORMALIZED_COMPLEX_RMS,
    max_magnitude_p95_db=MAX_MAGNITUDE_ERROR_P95_DB,
    max_phase_rms_deg=MAX_PHASE_ERROR_RMS_DEG,
    min_coherence=MIN_COMPLEX_COHERENCE,
):
    """Return unfitted metrics for two already compatible complex fields."""

    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("Truth and prediction complex fields have incompatible shapes.")
    if (
        not np.all(np.isfinite(reference.real) & np.isfinite(reference.imag))
        or not np.all(np.isfinite(estimate.real) & np.isfinite(estimate.imag))
    ):
        raise ValueError("Truth and prediction complex fields must be finite.")
    settings = {
        "active_floor_db": float(active_floor_db),
        "max_normalized_rms": float(max_normalized_rms),
        "max_magnitude_p95_db": float(max_magnitude_p95_db),
        "max_phase_rms_deg": float(max_phase_rms_deg),
        "min_coherence": float(min_coherence),
    }
    if not all(math.isfinite(value) for value in settings.values()):
        raise ValueError("Feature-validation thresholds must be finite.")
    if settings["active_floor_db"] > 0.0:
        raise ValueError("active_floor_db must be non-positive.")
    if (
        settings["max_normalized_rms"] < 0.0
        or settings["max_magnitude_p95_db"] < 0.0
        or settings["max_phase_rms_deg"] < 0.0
        or not 0.0 <= settings["min_coherence"] <= 1.0
    ):
        raise ValueError("Feature-validation gate limits are out of range.")

    error = estimate - reference
    reference_rms = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    error_rms = float(np.sqrt(np.mean(np.abs(error) ** 2)))
    tiny = np.finfo(float).tiny
    normalized_rms = error_rms / max(reference_rms, tiny)
    reference_peak = float(np.max(np.abs(reference)))
    if reference_peak <= tiny:
        raise ValueError("Truth field has no finite nonzero samples.")
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

    polarizations = [str(value) for value in np.asarray(polarizations).ravel()]
    if reference.shape[-1] != len(polarizations):
        raise ValueError(
            "Complex-field polarization dimension does not match its labels."
        )
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
    result = _comparison_metrics(
        truth["_amp"],
        prediction["_amp"],
        truth["polarizations"],
        active_floor_db=active_floor_db,
        max_normalized_rms=max_normalized_rms,
        max_magnitude_p95_db=max_magnitude_p95_db,
        max_phase_rms_deg=max_phase_rms_deg,
        min_coherence=min_coherence,
    )
    return {
        "schema": "ghost.validation.feature-reconstruction.v1",
        "truth": truth_label,
        "prediction": prediction_label,
        **result,
    }


def compare_feature_case(
    *,
    clean_truth,
    clean_prediction,
    featured_truth,
    featured_prediction,
    active_floor_db=ACTIVE_FIELD_FLOOR_DB,
    max_normalized_rms=MAX_NORMALIZED_COMPLEX_RMS,
    max_magnitude_p95_db=MAX_MAGNITUDE_ERROR_P95_DB,
    max_phase_rms_deg=MAX_PHASE_ERROR_RMS_DEG,
    min_coherence=MIN_COMPLEX_COHERENCE,
):
    """Grade a clean/featured four-artifact reconstruction case.

    In addition to the two whole-body comparisons, this computes the physically
    important isolated feature comparison::

        (featured_prediction - clean_prediction)
            versus
        (featured_truth - clean_truth)

    All subtraction is on signed complex far-field amplitude. No amplitude,
    phase, range, or coordinate fit is applied.
    """

    paths = {
        "clean_truth": str(Path(clean_truth).resolve()),
        "clean_prediction": str(Path(clean_prediction).resolve()),
        "featured_truth": str(Path(featured_truth).resolve()),
        "featured_prediction": str(Path(featured_prediction).resolve()),
    }
    payloads = {name: _load_grim(path) for name, path in paths.items()}
    anchor_name = "clean_truth"
    anchor = payloads[anchor_name]
    for name in ("clean_prediction", "featured_truth", "featured_prediction"):
        _require_compatible(
            anchor,
            payloads[name],
            paths[anchor_name],
            paths[name],
        )

    fields = {
        name: np.asarray(payload["_amp"], dtype=np.complex128)
        for name, payload in payloads.items()
    }
    settings = {
        "active_floor_db": active_floor_db,
        "max_normalized_rms": max_normalized_rms,
        "max_magnitude_p95_db": max_magnitude_p95_db,
        "max_phase_rms_deg": max_phase_rms_deg,
        "min_coherence": min_coherence,
    }
    polarizations = anchor["polarizations"]
    clean_result = _comparison_metrics(
        fields["clean_truth"],
        fields["clean_prediction"],
        polarizations,
        **settings,
    )
    featured_result = _comparison_metrics(
        fields["featured_truth"],
        fields["featured_prediction"],
        polarizations,
        **settings,
    )
    truth_delta = fields["featured_truth"] - fields["clean_truth"]
    prediction_delta = (
        fields["featured_prediction"] - fields["clean_prediction"]
    )
    delta_result = _comparison_metrics(
        truth_delta,
        prediction_delta,
        polarizations,
        **settings,
    )
    return {
        "schema": "ghost.validation.feature-case.v1",
        "paths": paths,
        "clean_baseline": clean_result,
        "featured_total": featured_result,
        "isolated_feature_delta": delta_result,
        "passed": bool(
            clean_result["passed"]
            and featured_result["passed"]
            and delta_result["passed"]
        ),
    }


def _metric_summary(label, result):
    return (
        f"  {label}: {'PASS' if result['passed'] else 'FAIL'} | "
        f"complex RMS={result['normalized_complex_rms']:.4f}, "
        f"|dB| p95={result['magnitude_error_p95_db']:.2f}, "
        f"phase RMS={result['phase_error_rms_deg']:.2f} deg, "
        f"coherence={result['complex_coherence']:.5f}"
    )


def load_case_manifest(path):
    """Load non-BoR validation cases, resolving paths beside the manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: manifest root must be an object.")
    if manifest.get("schema") != CASE_MANIFEST_SCHEMA:
        raise ValueError(
            f"{manifest_path}: schema must be {CASE_MANIFEST_SCHEMA!r}."
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{manifest_path}: cases must be a nonempty array.")
    defaults = {
        "active_floor_db": ACTIVE_FIELD_FLOOR_DB,
        "max_normalized_rms": MAX_NORMALIZED_COMPLEX_RMS,
        "max_magnitude_p95_db": MAX_MAGNITUDE_ERROR_P95_DB,
        "max_phase_rms_deg": MAX_PHASE_ERROR_RMS_DEG,
        "min_coherence": MIN_COMPLEX_COHERENCE,
    }
    top_gates = manifest.get("gates", {})
    if not isinstance(top_gates, dict):
        raise ValueError(f"{manifest_path}: gates must be an object.")
    unknown = sorted(set(top_gates) - set(defaults))
    if unknown:
        raise ValueError(f"{manifest_path}: unknown gate setting(s) {unknown}.")
    defaults.update({key: float(value) for key, value in top_gates.items()})

    resolved = []
    for index, raw in enumerate(cases, start=1):
        if not isinstance(raw, dict):
            raise ValueError(
                f"{manifest_path}: case {index} must be an object."
            )
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"{manifest_path}: case {index} has no name.")
        missing = [key for key in CASE_REQUIRED_PATHS if not raw.get(key)]
        if missing:
            raise ValueError(
                f"{manifest_path}: case {name!r} is missing {missing}."
            )
        gates = dict(defaults)
        case_gates = raw.get("gates", {})
        if not isinstance(case_gates, dict):
            raise ValueError(
                f"{manifest_path}: case {name!r} gates must be an object."
            )
        unknown = sorted(set(case_gates) - set(defaults))
        if unknown:
            raise ValueError(
                f"{manifest_path}: case {name!r} has unknown gate(s) {unknown}."
            )
        gates.update({key: float(value) for key, value in case_gates.items()})
        resolved.append({
            "name": name,
            "paths": {
                key: str((manifest_path.parent / str(raw[key])).resolve())
                for key in CASE_REQUIRED_PATHS
            },
            "gates": gates,
            "body": str(raw.get("body", "")).strip(),
            "feature": str(raw.get("feature", "")).strip(),
        })
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare clean and explicitly featured full-wave bodies against "
            "clean and reduced-order reconstructed GRIM fields."
        )
    )
    parser.add_argument(
        "--manifest",
        help=(
            "JSON feature-case manifest. Paths are resolved relative to the "
            "manifest, and each case grades clean, featured, and delta fields."
        ),
    )
    parser.add_argument(
        "--report",
        default=REPORT_JSON,
        help=f"output JSON report (default: {REPORT_JSON})",
    )
    args = parser.parse_args(argv)

    if args.manifest:
        cases = load_case_manifest(args.manifest)
        comparisons = []
        failed = False
        for case in cases:
            result = compare_feature_case(**case["paths"], **case["gates"])
            result["name"] = case["name"]
            result["body"] = case["body"]
            result["feature"] = case["feature"]
            comparisons.append(result)
            failed = failed or not result["passed"]
            print(f"{case['name']}: {'PASS' if result['passed'] else 'FAIL'}")
            print(_metric_summary("clean baseline", result["clean_baseline"]))
            print(_metric_summary("featured total", result["featured_total"]))
            print(_metric_summary(
                "isolated feature delta", result["isolated_feature_delta"]
            ))
        report = {
            "schema": "ghost.validation.feature-case-report.v1",
            "passed": not failed,
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "comparisons": comparisons,
        }
        destination = Path(args.report).expanduser().resolve()
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {destination}")
        return 1 if failed else 0

    if not REFERENCE_PAIRS:
        parser.error(
            "provide --manifest, or configure at least one REFERENCE_PAIRS entry"
        )
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
    destination = Path(args.report).expanduser().resolve()
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {destination}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
