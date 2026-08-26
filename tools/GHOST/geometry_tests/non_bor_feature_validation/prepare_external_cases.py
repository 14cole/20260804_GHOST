#!/usr/bin/env python3
"""Prepare and preflight deterministic external full-wave feature cases.

This utility creates only case specifications and a validator manifest.  It
never creates, copies, or substitutes electromagnetic results.  The four GRIM
artifacts in every case directory must come from the external-solver and GHOST
workflows described in README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parents[1] / "Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from components import (  # noqa: E402
    COMPONENT_AMPLITUDE_CONVENTION,
    COMPONENT_COMPLEX_FIELD_DOMAIN,
    COMPONENT_PHASE_REFERENCE,
)
from feature_sum import _load_grim  # noqa: E402
from validate_feature_reconstruction import (  # noqa: E402
    CASE_MANIFEST_SCHEMA,
    CASE_REQUIRED_PATHS,
    load_case_manifest,
)


PLAN_SCHEMA = "ghost.validation.external-feature-plan.v1"
CASE_SPEC_SCHEMA = "ghost.validation.external-feature-case.v1"
PREFLIGHT_SCHEMA = "ghost.validation.external-feature-preflight.v1"
DEFAULT_PLAN = THIS_DIR / "external_case_plan.json"
_CASE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_GATE_KEYS = {
    "active_floor_db",
    "max_normalized_rms",
    "max_magnitude_p95_db",
    "max_phase_rms_deg",
    "min_coherence",
}


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _number_axis(
    value: Any,
    label: str,
    *,
    nonempty: bool = True,
    unique: bool = True,
) -> list[float]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label} must be a nonempty array.")
    result = [_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicate samples.")
    return result


def _validate_gates(gates: Any, label: str) -> dict[str, float]:
    if not isinstance(gates, dict) or set(gates) != _GATE_KEYS:
        raise ValueError(f"{label} must contain exactly {sorted(_GATE_KEYS)}.")
    clean = {key: _finite_number(value, f"{label}.{key}") for key, value in gates.items()}
    if clean["active_floor_db"] > 0.0:
        raise ValueError(f"{label}.active_floor_db must be non-positive.")
    if any(
        clean[key] < 0.0
        for key in (
            "max_normalized_rms",
            "max_magnitude_p95_db",
            "max_phase_rms_deg",
        )
    ):
        raise ValueError(f"{label} error limits must be non-negative.")
    if not 0.0 <= clean["min_coherence"] <= 1.0:
        raise ValueError(f"{label}.min_coherence must be between zero and one.")
    return clean


def load_external_plan(path: str | os.PathLike[str] = DEFAULT_PLAN) -> dict[str, Any]:
    """Load and strictly validate the external-solver handoff plan."""

    plan_path = Path(path).expanduser().resolve()
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{plan_path}: invalid JSON: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{plan_path}: schema must be {PLAN_SCHEMA!r}.")

    artifacts = plan.get("artifact_filenames")
    if not isinstance(artifacts, dict) or set(artifacts) != set(CASE_REQUIRED_PATHS):
        raise ValueError(
            f"{plan_path}: artifact_filenames must contain exactly "
            f"{list(CASE_REQUIRED_PATHS)}."
        )
    filenames: list[str] = []
    for role in CASE_REQUIRED_PATHS:
        filename = str(artifacts[role]).strip()
        if not filename or Path(filename).name != filename or filename in (".", ".."):
            raise ValueError(
                f"{plan_path}: artifact filename for {role} must be one plain filename."
            )
        filenames.append(filename)
    if len(set(filenames)) != len(filenames):
        raise ValueError(f"{plan_path}: artifact filenames must be unique.")

    contract = plan.get("solve_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{plan_path}: solve_contract must be an object.")
    frame = contract.get("coordinate_frame")
    if not isinstance(frame, dict) or frame != {
        "x": "right",
        "y": "nose",
        "z": "up",
        "length_unit": "m",
    }:
        raise ValueError(f"{plan_path}: coordinate_frame must match the GHOST vehicle frame.")
    origin = _number_axis(
        contract.get("global_phase_origin_m"),
        "global_phase_origin_m",
        unique=False,
    )
    if origin != [0.0, 0.0, 0.0]:
        raise ValueError(f"{plan_path}: global_phase_origin_m must be [0, 0, 0].")
    expected_text = {
        "time_convention": "exp(+jwt)",
        "outgoing_wave": "exp(-jkr)",
        "look_vector": "target_to_radar",
        "amplitude_normalization": "F physical far-field amplitude; sigma_3d=4*pi*|F|^2",
    }
    for key, expected in expected_text.items():
        if contract.get(key) != expected:
            raise ValueError(f"{plan_path}: {key} must be {expected!r}.")
    if contract.get("polarizations") != ["VV", "HH", "VH"]:
        raise ValueError(f"{plan_path}: polarizations must be ['VV', 'HH', 'VH'].")

    grid = contract.get("acceptance_grid")
    if not isinstance(grid, dict):
        raise ValueError(f"{plan_path}: acceptance_grid must be an object.")
    frequencies = _number_axis(grid.get("frequencies_GHz"), "frequencies_GHz")
    azimuths = _number_axis(grid.get("azimuths_deg"), "azimuths_deg")
    elevations = _number_axis(grid.get("elevations_deg"), "elevations_deg")
    if frequencies != sorted(frequencies) or elevations != sorted(elevations):
        raise ValueError(f"{plan_path}: frequency and elevation axes must be ascending.")
    if azimuths != sorted(azimuths) or any(not 0.0 <= value < 360.0 for value in azimuths):
        raise ValueError(
            f"{plan_path}: azimuths must be ascending in [0, 360) with no duplicate seam."
        )

    feature_references = plan.get("feature_reference_contracts")
    if not isinstance(feature_references, dict) or not feature_references:
        raise ValueError(
            f"{plan_path}: feature_reference_contracts must be a nonempty object."
        )

    bodies = plan.get("bodies")
    if not isinstance(bodies, dict) or not bodies:
        raise ValueError(f"{plan_path}: bodies must be a nonempty object.")
    for body_id, body in bodies.items():
        if not _CASE_ID.fullmatch(str(body_id)) or not isinstance(body, dict):
            raise ValueError(f"{plan_path}: invalid body {body_id!r}.")
        for key in ("name", "clean_geometry", "surface_normal_contract"):
            if not body.get(key):
                raise ValueError(f"{plan_path}: body {body_id!r} is missing {key!r}.")
        source_spec = str(body.get("canonical_source_spec", "")).strip()
        if source_spec and not (plan_path.parent / source_spec).is_file():
            raise ValueError(
                f"{plan_path}: body {body_id!r} canonical source spec "
                f"{source_spec!r} does not exist."
            )
    execution_order = plan.get("recommended_execution_order")
    if (
        not isinstance(execution_order, list)
        or len(execution_order) != len(set(execution_order))
        or set(execution_order) != set(bodies)
    ):
        raise ValueError(
            f"{plan_path}: recommended_execution_order must name every body "
            "exactly once."
        )

    raw_cases = plan.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{plan_path}: cases must be a nonempty array.")
    ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{plan_path}: case {index} must be an object.")
        case_id = str(raw.get("id", "")).strip()
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError(f"{plan_path}: case {index} has invalid id {case_id!r}.")
        if case_id in ids:
            raise ValueError(f"{plan_path}: duplicate case id {case_id!r}.")
        ids.add(case_id)
        body_id = str(raw.get("body_id", "")).strip()
        if body_id not in bodies:
            raise ValueError(f"{plan_path}: case {case_id!r} references unknown body {body_id!r}.")
        for key in ("name", "feature_summary", "parameter_sweep", "feature_spec"):
            if not raw.get(key):
                raise ValueError(f"{plan_path}: case {case_id!r} is missing {key!r}.")
        if not isinstance(raw["parameter_sweep"], dict):
            raise ValueError(f"{plan_path}: case {case_id!r} parameter_sweep must be an object.")
        if not isinstance(raw["feature_spec"], dict):
            raise ValueError(f"{plan_path}: case {case_id!r} feature_spec must be an object.")
        for key, reference_id in raw["feature_spec"].items():
            if (key == "reference_contract" or key.endswith("_reference_contract")) and (
                reference_id not in feature_references
            ):
                raise ValueError(
                    f"{plan_path}: case {case_id!r} references unknown feature "
                    f"contract {reference_id!r}."
                )
        case = dict(raw)
        case["id"] = case_id
        case["body_id"] = body_id
        case["gates"] = _validate_gates(raw.get("gates"), f"case {case_id} gates")
        cases.append(case)

    order_index = {body_id: index for index, body_id in enumerate(execution_order)}
    cases.sort(key=lambda case: order_index[case["body_id"]])

    return {
        **plan,
        "_path": str(plan_path),
        "artifact_filenames": {role: str(artifacts[role]) for role in CASE_REQUIRED_PATHS},
        "recommended_execution_order": list(execution_order),
        "cases": cases,
    }


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_validator_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    """Build the existing four-artifact validator manifest without results."""

    filenames = plan["artifact_filenames"]
    cases = []
    for case in plan["cases"]:
        prefix = Path("cases") / case["id"]
        entry = {
            "name": case["name"],
            "body": plan["bodies"][case["body_id"]]["name"],
            "feature": case["feature_summary"],
            "gates": case["gates"],
        }
        entry.update({role: (prefix / filenames[role]).as_posix() for role in CASE_REQUIRED_PATHS})
        cases.append(entry)
    return {"schema": CASE_MANIFEST_SCHEMA, "cases": cases}


def prepare_run(plan_path: str | os.PathLike[str], output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Create deterministic handoff metadata while leaving GRIM slots empty."""

    plan = load_external_plan(plan_path)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_validator_manifest(plan)
    _atomic_write(root / "feature_cases.json", _json_text(manifest))
    for case in plan["cases"]:
        body = plan["bodies"][case["body_id"]]
        spec = {
            "schema": CASE_SPEC_SCHEMA,
            "case_id": case["id"],
            "name": case["name"],
            "body_id": case["body_id"],
            "body": body,
            "feature_summary": case["feature_summary"],
            "parameter_sweep": case["parameter_sweep"],
            "feature_spec": case["feature_spec"],
            "feature_reference_contracts": plan["feature_reference_contracts"],
            "solve_contract": plan["solve_contract"],
            "artifact_filenames": plan["artifact_filenames"],
            "acceptance_gates": case["gates"],
            "artifact_contract": {
                "clean_truth": "independent full-wave clean-body solve",
                "clean_prediction": "the same clean field after import/load into GHOST",
                "featured_truth": "independent full-wave body with this feature modeled explicitly",
                "featured_prediction": "clean_prediction plus this case's placed GHOST feature delta",
            },
            "prohibited_adjustments": [
                "fitted global phase",
                "fitted amplitude scale",
                "range or phase-center correction after the solve",
                "angle interpolation between any of the four artifacts",
            ],
        }
        case_dir = root / "cases" / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(case_dir / "case_spec.json", _json_text(spec))
    # Confirm that the generated document is consumable by the existing
    # validator before handing it to an external solver operator.
    load_case_manifest(root / "feature_cases.json")
    return {"root": str(root), "manifest": str(root / "feature_cases.json"), "case_count": len(plan["cases"])}


def _scalar_text(payload: dict[str, Any], key: str) -> str:
    value = np.asarray(payload.get(key))
    if value.size != 1:
        return ""
    return str(value.reshape(-1)[0])


def _artifact_errors(path: Path, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        payload = _load_grim(str(path))
    except Exception as exc:  # a preflight report should cover all cases
        return [f"cannot load coherent GRIM: {exc}"]
    grid = contract["acceptance_grid"]
    expected_axes = {
        "frequencies": np.asarray(grid["frequencies_GHz"], dtype=float),
        "azimuths": np.asarray(grid["azimuths_deg"], dtype=float),
        "elevations": np.asarray(grid["elevations_deg"], dtype=float),
        "polarizations": np.asarray(contract["polarizations"], dtype=str),
    }
    for key, expected in expected_axes.items():
        actual = np.asarray(payload.get(key))
        if not np.array_equal(actual, expected):
            errors.append(
                f"{key} do not exactly match the acceptance grid: "
                f"expected {expected.tolist()}, got {actual.tolist()}"
            )
    expected_metadata = {
        "phase_reference": COMPONENT_PHASE_REFERENCE,
        "amplitude_convention": COMPONENT_AMPLITUDE_CONVENTION,
        "complex_field_domain": COMPONENT_COMPLEX_FIELD_DOMAIN,
    }
    for key, expected in expected_metadata.items():
        actual = _scalar_text(payload, key)
        if actual != expected:
            errors.append(f"{key} must be {expected!r}; got {actual!r}")
    if bool(payload.get("_amp_from_power_phase", False)):
        errors.append(
            "signed rcs_amp_real/rcs_amp_imag are required; a power/phase "
            "reconstruction is not an external complex-field artifact"
        )
    if not bool(payload.get("raw_complex_amplitude_preserved", False)):
        errors.append("raw_complex_amplitude_preserved must be true")
    try:
        units = json.loads(_scalar_text(payload, "units"))
    except json.JSONDecodeError:
        units = {}
        errors.append("units metadata must be valid JSON")
    if not isinstance(units, dict) or (
        units.get("frequency") != "GHz"
        or units.get("rcs_linear_quantity") != "sigma_3d"
        or str(units.get("rcs_log_unit", "")).lower() != "dbsm"
    ):
        errors.append("units must declare frequency GHz and physical sigma_3d/dBsm")
    amplitude = np.asarray(payload.get("_amp"), dtype=np.complex128)
    expected_shape = (
        len(expected_axes["azimuths"]),
        len(expected_axes["elevations"]),
        len(expected_axes["frequencies"]),
        len(expected_axes["polarizations"]),
    )
    if amplitude.shape != expected_shape:
        errors.append(f"complex amplitude shape must be {expected_shape}; got {amplitude.shape}")
    elif not np.all(np.isfinite(amplitude.real) & np.isfinite(amplitude.imag)):
        errors.append("complex amplitude contains a NaN or infinity")
    return errors


def preflight_run(plan_path: str | os.PathLike[str], output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Check artifact presence, exact grid, and coherent metadata for all cases."""

    plan = load_external_plan(plan_path)
    root = Path(output_root).expanduser().resolve()
    filenames = plan["artifact_filenames"]
    case_reports = []
    for case in plan["cases"]:
        case_dir = root / "cases" / case["id"]
        artifacts = {}
        case_passed = True
        for role in CASE_REQUIRED_PATHS:
            path = case_dir / filenames[role]
            exists = path.is_file()
            errors = [] if not exists else _artifact_errors(path, plan["solve_contract"])
            if not exists:
                errors.append("required artifact is missing")
            passed = exists and not errors
            case_passed = case_passed and passed
            artifacts[role] = {
                "path": str(path),
                "exists": exists,
                "passed": passed,
                "errors": errors,
            }
        case_reports.append({
            "case_id": case["id"],
            "name": case["name"],
            "passed": case_passed,
            "artifacts": artifacts,
        })
    return {
        "schema": PREFLIGHT_SCHEMA,
        "plan": str(Path(plan_path).expanduser().resolve()),
        "root": str(root),
        "passed": all(case["passed"] for case in case_reports),
        "case_count": len(case_reports),
        "cases": case_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic external full-wave case folders and/or "
            "preflight their four coherent GRIM artifacts."
        )
    )
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="external case-plan JSON")
    parser.add_argument("--output", required=True, help="run directory outside source control")
    parser.add_argument("--prepare", action="store_true", help="write case specs and validator manifest")
    parser.add_argument("--preflight", action="store_true", help="check every required GRIM artifact")
    parser.add_argument("--report", help="preflight report path (default: OUTPUT/preflight_report.json)")
    args = parser.parse_args(argv)
    if not args.prepare and not args.preflight:
        parser.error("choose --prepare, --preflight, or both")
    if args.prepare:
        prepared = prepare_run(args.plan, args.output)
        print(f"Prepared {prepared['case_count']} cases in {prepared['root']}")
        print(f"Validator manifest: {prepared['manifest']}")
    if args.preflight:
        report = preflight_run(args.plan, args.output)
        report_path = Path(args.report).expanduser().resolve() if args.report else Path(args.output).expanduser().resolve() / "preflight_report.json"
        _atomic_write(report_path, _json_text(report))
        missing_or_invalid = sum(
            not artifact["passed"]
            for case in report["cases"]
            for artifact in case["artifacts"].values()
        )
        print(
            f"Preflight {'PASS' if report['passed'] else 'FAIL'}: "
            f"{missing_or_invalid} missing or invalid artifact(s)."
        )
        print(f"Report: {report_path}")
        if not report["passed"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
