#!/usr/bin/env python3
"""Create/check evidence-bound Assembly response and surface manifests.

The feature command consumes a report from validate_feature_reconstruction.py
and proves that each selected passing case exercised the exact response being
certified. Surface registration remains a separately reviewed attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from feature_workflow import (
    FEATURE_LIBRARY_MANIFEST_KEY,
    FEATURE_LIBRARY_MANIFEST_SCHEMA,
    FEATURE_VALIDATION_ARTIFACT_ROLES,
    FEATURE_VALIDATION_EVIDENCE_SCHEMA,
    FEATURE_VALIDATION_MAX_ACTIVE_FLOOR_DB,
    FEATURE_VALIDATION_RELEASE_CEILINGS,
    GRAZING_TAPER_DEG,
    LINE_PHASE_CALIBRATION_SCHEMA,
    PSI_HH_DEG,
    PSI_VV_DEG,
    _FEATURE_FRAME_CONVENTIONS,
    _FEATURE_PHASE_ORIGINS,
    check_surface_binding as _check_backend_surface_binding,
    feature_response_content_sha256,
    load_feature_library_manifest,
    resolve_path,
    validate_declared_feature_delta_response,
    validate_feature_library_manifest,
    write_surface_binding as _write_backend_surface_binding,
)


_FEATURE_CASE_REPORT_SCHEMA = "ghost.validation.feature-case-report.v1"


_SURFACE_UNIT_ALIASES = {
    "m": "meters",
    "meter": "meters",
    "meters": "meters",
    "mm": "millimeters",
    "millimeter": "millimeters",
    "millimeters": "millimeters",
    "in": "inches",
    "inch": "inches",
    "inches": "inches",
    "ft": "feet",
    "foot": "feet",
    "feet": "feet",
}


def _legal_sidecars(response: Path) -> tuple[Path, ...]:
    return tuple(dict.fromkeys((
        Path(str(response) + ".feature.json"),
        response.with_suffix(".feature.json"),
    )))


def _decode_manifest(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to a JSON object.")
    return value


def _embedded_manifest(response: Path) -> dict[str, Any] | None:
    """Read only an advertised embedded manifest; malformed data fails hard."""

    try:
        stored_context = np.load(response, allow_pickle=False)
    except (OSError, EOFError, ValueError):
        return None
    with stored_context as stored:
        if FEATURE_LIBRARY_MANIFEST_KEY not in stored.files:
            return None
        try:
            raw = np.asarray(stored[FEATURE_LIBRARY_MANIFEST_KEY])
            if raw.size != 1:
                raise ValueError(f"{FEATURE_LIBRARY_MANIFEST_KEY} must be scalar.")
            return _decode_manifest(
                raw.reshape(()).item(),
                label=f"{response}:{FEATURE_LIBRARY_MANIFEST_KEY}",
            )
        except Exception as exc:
            raise ValueError(
                f"{response}: embedded feature-library manifest is unreadable "
                "or malformed."
            ) from exc


def _raw_manifest_candidates(response: Path) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    embedded = _embedded_manifest(response)
    if embedded is not None:
        candidates.append(("embedded manifest", embedded))
    for sidecar in _legal_sidecars(response):
        if not sidecar.is_file():
            continue
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            candidates.append((str(sidecar), _decode_manifest(value, label=str(sidecar))))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{sidecar}: sidecar is not valid UTF-8 JSON.") from exc
    return candidates


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_text(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256 digest.")
    return digest


def _comparison_sha256(comparison: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(comparison),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validated_gate_limits(
    comparison: Mapping[str, Any], *, label: str
) -> dict[str, float]:
    """Require all three comparisons and every polarization to pass safe gates."""

    sections = (
        "clean_baseline",
        "featured_total",
        "isolated_feature_delta",
    )
    common_limits = None
    required_limits = {"active_floor_db", *FEATURE_VALIDATION_RELEASE_CEILINGS}
    for section_name in sections:
        section = comparison.get(section_name)
        if not isinstance(section, dict) or section.get("passed") is not True:
            raise ValueError(f"{label}.{section_name} did not pass.")
        gates = section.get("gates")
        if not isinstance(gates, dict) or not gates or any(
            value is not True for value in gates.values()
        ):
            raise ValueError(
                f"{label}.{section_name} contains a failed or malformed gate."
            )
        if gates.get("every_polarization_channel") is not True:
            raise ValueError(
                f"{label}.{section_name} predates per-polarization release "
                "gating; regenerate it with the current validator."
            )
        raw_limits = section.get("gate_limits")
        if not isinstance(raw_limits, dict) or set(raw_limits) != required_limits:
            raise ValueError(
                f"{label}.{section_name}.gate_limits must contain exactly "
                f"{sorted(required_limits)}."
            )
        try:
            limits = {key: float(raw_limits[key]) for key in required_limits}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label}.{section_name}.gate_limits must be numeric."
            ) from exc
        if not all(math.isfinite(value) for value in limits.values()):
            raise ValueError(
                f"{label}.{section_name}.gate_limits must be finite."
            )
        if common_limits is None:
            common_limits = limits
        elif limits != common_limits:
            raise ValueError(
                f"{label} uses different gate limits across clean, featured, "
                "and isolated-delta comparisons."
            )
    assert common_limits is not None
    if common_limits["active_floor_db"] > FEATURE_VALIDATION_MAX_ACTIVE_FLOOR_DB:
        raise ValueError(
            f"{label} active_floor_db={common_limits['active_floor_db']:g} "
            "excludes more weak-field samples than the Production maximum "
            f"{FEATURE_VALIDATION_MAX_ACTIVE_FLOOR_DB:g} dB."
        )
    for key in (
        "max_normalized_rms",
        "max_magnitude_p95_db",
        "max_phase_rms_deg",
    ):
        if not 0.0 <= common_limits[key] <= FEATURE_VALIDATION_RELEASE_CEILINGS[key]:
            raise ValueError(
                f"{label} {key}={common_limits[key]:g} is looser than the "
                f"Production ceiling {FEATURE_VALIDATION_RELEASE_CEILINGS[key]:g}."
            )
    if not FEATURE_VALIDATION_RELEASE_CEILINGS[
        "min_coherence"
    ] <= common_limits["min_coherence"] <= 1.0:
        raise ValueError(
            f"{label} min_coherence={common_limits['min_coherence']:g} is "
            "looser than the Production floor "
            f"{FEATURE_VALIDATION_RELEASE_CEILINGS['min_coherence']:g}."
        )
    return {key: common_limits[key] for key in sorted(common_limits)}


def _validation_evidence_from_reports(
    report_paths: Sequence[str],
    *,
    requested_case_ids: Sequence[str],
    response_content_sha256: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract passing, artifact-bound evidence for one exact feature response."""

    if not report_paths:
        raise ValueError(
            "A validated feature manifest requires at least one "
            "--validation-report produced by validate_feature_reconstruction.py."
        )
    requested = [str(value).strip() for value in requested_case_ids]
    if any(not value for value in requested) or len(set(requested)) != len(requested):
        raise ValueError("--validation-case-id values must be nonempty and unique.")
    requested_set = set(requested)
    evidence: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for raw_report_path in report_paths:
        report_path = Path(raw_report_path).expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"Validation report not found: {report_path}")
        report_bytes = report_path.read_bytes()
        try:
            report = json.loads(report_bytes.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{report_path}: validation report is not valid UTF-8 JSON."
            ) from exc
        if not isinstance(report, dict) or report.get("schema") != (
            _FEATURE_CASE_REPORT_SCHEMA
        ):
            raise ValueError(
                f"{report_path}: report schema must be "
                f"{_FEATURE_CASE_REPORT_SCHEMA!r}."
            )
        comparisons = report.get("comparisons")
        if not isinstance(comparisons, list):
            raise ValueError(f"{report_path}: comparisons must be an array.")
        report_sha256 = _sha256_bytes(report_bytes)
        for index, comparison in enumerate(comparisons):
            label = f"{report_path}: comparison {index + 1}"
            if not isinstance(comparison, dict):
                raise ValueError(f"{label} must be an object.")
            case_id = str(comparison.get("case_id", "")).strip()
            if not case_id:
                raise ValueError(
                    f"{label} has no stable case_id; regenerate the report "
                    "with the current validator."
                )
            if requested_set and case_id not in requested_set:
                continue
            response_hashes = comparison.get(
                "feature_response_content_sha256"
            )
            if not isinstance(response_hashes, list) or any(
                not isinstance(value, str) for value in response_hashes
            ):
                raise ValueError(
                    f"{label} has no valid feature-response provenance list."
                )
            response_hashes = {
                _digest_text(value, label=f"{label} feature response hash")
                for value in response_hashes
            }
            if response_content_sha256 not in response_hashes:
                if requested_set and case_id in requested_set:
                    raise ValueError(
                        f"{label} did not exercise this exact feature response "
                        f"({response_content_sha256})."
                    )
                continue
            if len(response_hashes) != 1:
                raise ValueError(
                    f"{label} combines {len(response_hashes)} distinct reusable "
                    "feature responses. Its aggregate pass cannot certify any "
                    "one library response because errors can cancel. Add an "
                    "isolated case that uses only this response."
                )
            if comparison.get("passed") is not True:
                raise ValueError(
                    f"{label} exercised this response but did not pass every "
                    "clean, featured-total, and isolated-delta gate."
                )
            gate_limits = _validated_gate_limits(comparison, label=label)
            if case_id in seen_case_ids:
                raise ValueError(
                    f"Validation case_id {case_id!r} appears more than once "
                    "across the selected reports."
                )
            artifacts = comparison.get("artifact_sha256")
            paths = comparison.get("paths")
            if not isinstance(artifacts, dict) or set(artifacts) != set(
                FEATURE_VALIDATION_ARTIFACT_ROLES
            ):
                raise ValueError(
                    f"{label}.artifact_sha256 must contain exactly "
                    f"{list(FEATURE_VALIDATION_ARTIFACT_ROLES)}."
                )
            if not isinstance(paths, dict) or set(paths) != set(
                FEATURE_VALIDATION_ARTIFACT_ROLES
            ):
                raise ValueError(
                    f"{label}.paths must contain exactly "
                    f"{list(FEATURE_VALIDATION_ARTIFACT_ROLES)}."
                )
            checked_artifacts = {}
            for role in FEATURE_VALIDATION_ARTIFACT_ROLES:
                expected = _digest_text(
                    artifacts[role], label=f"{label}.artifact_sha256.{role}"
                )
                artifact = Path(str(paths[role])).expanduser().resolve()
                if not artifact.is_file():
                    raise FileNotFoundError(
                        f"{label}: evidence artifact is missing: {artifact}"
                    )
                actual = _sha256_file(artifact)
                if actual != expected:
                    raise ValueError(
                        f"{label}: {role} changed after validation; expected "
                        f"{expected}, got {actual}. Regenerate the report."
                    )
                checked_artifacts[role] = actual
            seen_case_ids.add(case_id)
            evidence.append({
                "schema": FEATURE_VALIDATION_EVIDENCE_SCHEMA,
                "case_id": case_id,
                "passed": True,
                "report_sha256": report_sha256,
                "comparison_sha256": _comparison_sha256(comparison),
                "feature_response_content_sha256": response_content_sha256,
                "artifact_sha256": checked_artifacts,
                "gate_limits": gate_limits,
            })
    if requested_set - seen_case_ids:
        raise ValueError(
            "Selected validation case ID(s) were not found as passing evidence "
            "for this exact response: " + ", ".join(sorted(requested_set - seen_case_ids))
        )
    if not evidence:
        raise ValueError(
            "No passing validation case in the selected report(s) exercised "
            "this exact feature response."
        )
    evidence.sort(key=lambda item: item["case_id"])
    return evidence, [item["case_id"] for item in evidence]


def _normalized_surface_units(value: Any) -> str:
    key = str(value or "").strip().casefold()
    try:
        return _SURFACE_UNIT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "surface_units must be meters, millimeters, inches, or feet "
            "(standard abbreviations are accepted)."
        ) from exc


def _check_line_extensions(manifest: Mapping[str, Any], *, label: str) -> None:
    """Check line fields that travel with the current fixed solver behavior."""

    applicability = manifest.get("applicability")
    if not isinstance(applicability, Mapping):
        raise ValueError(f"{label}: applicability must be an object.")
    maximum_turn = _finite_number(
        applicability.get("maximum_path_vertex_turn_deg"),
        f"{label}: applicability.maximum_path_vertex_turn_deg",
    )
    if not 0.0 <= maximum_turn <= 180.0:
        raise ValueError(
            f"{label}: maximum_path_vertex_turn_deg must lie in [0, 180]."
        )
    calibration = manifest.get("line_phase_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(f"{label}: line_phase_calibration must be an object.")
    taper = _finite_number(
        calibration.get("grazing_taper_deg"),
        f"{label}: line_phase_calibration.grazing_taper_deg",
    )
    if not math.isclose(taper, GRAZING_TAPER_DEG, abs_tol=1.0e-12):
        raise ValueError(
            f"{label}: grazing_taper_deg must be {GRAZING_TAPER_DEG:g}, "
            "matching the Assembly line-expansion implementation."
        )


def _validate_cli_manifest(
    manifest: Mapping[str, Any], *, dataset_id: str, feature_kind: str, label: str
) -> dict[str, Any]:
    normalized = validate_feature_library_manifest(
        manifest, dataset_id=dataset_id, feature_kind=feature_kind
    )
    if feature_kind == "line":
        _check_line_extensions(manifest, label=label)
    return normalized


def _manifest_from_args(args: argparse.Namespace, response: Path) -> dict[str, Any]:
    dataset_id = str(args.dataset_id).strip()
    host_material = " ".join(str(args.host_material).split())
    validation_case_ids = [
        str(value).strip() for value in args.validation_case_id
    ]
    phase_case_ids = [
        str(value).strip() for value in args.phase_calibration_case_id
    ]
    if not dataset_id:
        raise ValueError("--dataset-id must not be blank.")
    if not host_material:
        raise ValueError("--host-material must not be blank.")
    response_content_sha256 = feature_response_content_sha256(response)
    validation_evidence: list[dict[str, Any]] = []
    if args.validation_status == "validated":
        validation_evidence, validation_case_ids = (
            _validation_evidence_from_reports(
                args.validation_report,
                requested_case_ids=validation_case_ids,
                response_content_sha256=response_content_sha256,
            )
        )
    elif args.validation_report:
        raise ValueError(
            "--validation-report is accepted only with "
            "--validation-status validated."
        )

    applicability: dict[str, Any] = {
        "frequency_ghz": {
            "min": args.frequency_min_ghz,
            "max": args.frequency_max_ghz,
        },
        "footprint_radius_m": args.footprint_radius_m,
    }
    manifest: dict[str, Any] = {
        "schema": FEATURE_LIBRARY_MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "feature_kind": args.feature_kind,
        "subtraction_order": "featured_minus_clean",
        "phase_origin": _FEATURE_PHASE_ORIGINS[args.feature_kind],
        "frame_convention": _FEATURE_FRAME_CONVENTIONS[args.feature_kind],
        "time_convention": "exp(+jwt)",
        "response_content_sha256": response_content_sha256,
        "host": {"material": host_material},
        "applicability": applicability,
        "validation": {
            "status": args.validation_status,
            "case_ids": validation_case_ids,
            "evidence": validation_evidence,
        },
    }

    line_values = (
        args.minimum_along_line_normal_turn_radius_m,
        args.maximum_conical_incidence_deg,
        args.maximum_path_vertex_turn_deg,
    )
    if args.host_stack_id:
        manifest["host"]["stack_id"] = args.host_stack_id.strip()
    if args.minimum_principal_radius_m is not None:
        applicability["minimum_principal_radius_m"] = args.minimum_principal_radius_m
    if args.feature_kind == "line":
        if any(value is None for value in line_values):
            raise ValueError(
                "A line manifest requires --minimum-along-line-normal-turn-"
                "radius-m, --maximum-conical-incidence-deg, and "
                "--maximum-path-vertex-turn-deg."
            )
        if not phase_case_ids and validation_evidence:
            phase_case_ids = list(validation_case_ids)
        if not phase_case_ids:
            raise ValueError(
                "A line manifest requires at least one independent "
                "--phase-calibration-case-id."
            )
        if validation_evidence and not set(phase_case_ids).issubset(
            validation_case_ids
        ):
            raise ValueError(
                "Every --phase-calibration-case-id must name a selected "
                "passing full-wave validation case for this exact response."
            )
        applicability.update({
            "minimum_along_line_normal_turn_radius_m": line_values[0],
            "maximum_conical_incidence_deg": line_values[1],
            "maximum_path_vertex_turn_deg": line_values[2],
        })
        manifest["line_phase_calibration"] = {
            "schema": LINE_PHASE_CALIBRATION_SCHEMA,
            "tm_deg": float(PSI_HH_DEG),
            "te_deg": float(PSI_VV_DEG),
            "grazing_taper_deg": GRAZING_TAPER_DEG,
            "case_ids": phase_case_ids,
        }
    elif any(value is not None for value in line_values) or (
        phase_case_ids
    ):
        raise ValueError(
            "Line-only applicability/calibration options cannot be used for "
            "point responses."
        )

    _validate_cli_manifest(
        manifest,
        dataset_id=dataset_id,
        feature_kind=args.feature_kind,
        label="new manifest",
    )
    return manifest


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, force: bool) -> None:
    if not path.parent.is_dir():
        raise ValueError(f"Manifest parent directory does not exist: {path.parent}")
    if path.exists() and not force:
        raise ValueError(f"Manifest already exists: {path}. Use --force to replace it.")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resolve_response(value: str) -> Path:
    response = resolve_path(value)
    if not response.is_file():
        raise FileNotFoundError(f"Feature response not found: {response}")
    validate_declared_feature_delta_response(response)
    return response


def _create(args: argparse.Namespace) -> Path:
    response = _resolve_response(args.response)
    legal_sidecars = _legal_sidecars(response)
    output = (
        resolve_path(args.output)
        if args.output is not None
        else legal_sidecars[0]
    )
    if output.resolve() not in {path.resolve() for path in legal_sidecars}:
        raise ValueError(
            "--output must be one of the two sidecars discovered by Assembly: "
            + " or ".join(str(path) for path in legal_sidecars)
        )
    embedded = _embedded_manifest(response)
    if embedded is not None:
        raise ValueError(
            f"{response} already embeds a manifest. This command never mutates "
            "response archives; check the embedded declaration instead."
        )
    for sidecar in legal_sidecars:
        if sidecar.is_file() and sidecar.resolve() != output.resolve():
            raise ValueError(
                f"Alternate sidecar already exists: {sidecar}. Keep exactly one "
                "supported sidecar."
            )
    manifest = _manifest_from_args(args, response)
    _atomic_write_json(output, manifest, force=bool(args.force))
    loaded, _sources = load_feature_library_manifest(
        response,
        dataset_id=manifest["dataset_id"],
        feature_kind=manifest["feature_kind"],
    )
    if loaded is None:
        raise RuntimeError("Assembly could not discover the manifest just written.")
    return output


def _check(args: argparse.Namespace) -> dict[str, Any]:
    response = _resolve_response(args.response)
    manifest, _sources = load_feature_library_manifest(
        response,
        dataset_id=args.dataset_id,
        feature_kind=args.feature_kind,
    )
    if manifest is None:
        raise ValueError(
            f"{response}: no embedded or adjacent feature-library manifest found."
        )
    raw_candidates = _raw_manifest_candidates(response)
    if not raw_candidates:
        raise RuntimeError("Manifest discovery succeeded without a readable source.")
    for label, raw in raw_candidates:
        _validate_cli_manifest(
            raw,
            dataset_id=args.dataset_id,
            feature_kind=args.feature_kind,
            label=label,
        )
    return manifest


def _resolve_surface(value: str) -> Path:
    surface = resolve_path(value)
    if not surface.is_file():
        raise FileNotFoundError(f"Assembly surface not found: {surface}")
    if surface.suffix.casefold() not in {".stl", ".facet"}:
        raise ValueError(
            "Assembly surface must be an STL or indexed ASCII .facet file."
        )
    return surface


def _resolve_base_grim(value: str) -> Path:
    base_grim = resolve_path(value)
    if not base_grim.is_file():
        raise FileNotFoundError(f"External clean-body GRIM not found: {base_grim}")
    if base_grim.suffix.casefold() != ".grim":
        raise ValueError("External clean-body response must use the .grim extension.")
    return base_grim


def _create_surface_binding(args: argparse.Namespace) -> Path:
    base_grim = _resolve_base_grim(args.base_grim)
    surface = _resolve_surface(args.surface)
    geometry_id = str(args.geometry_id).strip()
    attestation_case_id = str(args.attestation_case_id).strip()
    if not geometry_id:
        raise ValueError("--geometry-id must not be blank.")
    if not attestation_case_id:
        raise ValueError("--attestation-case-id must not be blank.")
    units = _normalized_surface_units(args.surface_units)
    _manifest, output = _write_backend_surface_binding(
        base_grim,
        surface,
        surface_units=units,
        geometry_id=geometry_id,
        attestation_case_id=attestation_case_id,
        attest_reviewed_registration=bool(args.attest_reviewed_registration),
        overwrite=bool(args.force),
    )
    return output


def _check_surface_binding(args: argparse.Namespace) -> dict[str, str]:
    base_grim = _resolve_base_grim(args.base_grim)
    surface = _resolve_surface(args.surface)
    manifest, _binding = _check_backend_surface_binding(
        base_grim,
        surface,
        surface_units=args.surface_units,
    )
    recorded_geometry = str(manifest["geometry_id"])
    recorded_case = str(manifest["attestation_case_id"])
    if args.geometry_id is not None and str(args.geometry_id).strip() != recorded_geometry:
        raise ValueError(
            f"Surface binding geometry_id is {recorded_geometry!r}, not "
            f"{str(args.geometry_id).strip()!r}."
        )
    if (
        args.attestation_case_id is not None
        and str(args.attestation_case_id).strip() != recorded_case
    ):
        raise ValueError(
            "Surface binding attestation_case_id is "
            f"{recorded_case!r}, not {str(args.attestation_case_id).strip()!r}."
        )
    return dict(manifest)


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", required=True, help="Exact placement CSV dataset_id.")
    parser.add_argument(
        "--feature-kind", required=True, choices=("point", "line"),
        help="Response library kind used by Assembly.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser(
        "create",
        help="Write one content-bound adjacent manifest after team review.",
    )
    create.add_argument("response", help="Exact feature-response GRIM/NPZ file.")
    _add_identity_arguments(create)
    create.add_argument(
        "--host-material",
        required=True,
        help="Clean local material/coating stack ID.",
    )
    create.add_argument("--frequency-min-ghz", required=True, type=float)
    create.add_argument("--host-stack-id", default="", help="Exact coating/material stack identity used by the reference coupon.")
    create.add_argument("--minimum-principal-radius-m", type=float, help="Validated lower bound for both principal curvature radii over the entire mounted footprint.")
    create.add_argument("--frequency-max-ghz", required=True, type=float)
    create.add_argument("--footprint-radius-m", required=True, type=float)
    create.add_argument(
        "--validation-status", required=True,
        choices=("validated", "provisional", "uncertified"),
    )
    create.add_argument(
        "--validation-case-id", action="append", default=[],
        help=(
            "Case ID to select from --validation-report; repeat for multiple "
            "cases. If omitted, every passing case bound to this response is used."
        ),
    )
    create.add_argument(
        "--validation-report", action="append", default=[],
        help=(
            "Passing JSON report from validate_feature_reconstruction.py; "
            "required for validated status and repeatable across case families."
        ),
    )
    create.add_argument("--minimum-along-line-normal-turn-radius-m", type=float)
    create.add_argument("--maximum-conical-incidence-deg", type=float)
    create.add_argument("--maximum-path-vertex-turn-deg", type=float)
    create.add_argument(
        "--phase-calibration-case-id", action="append", default=[],
        help=(
            "Selected passing full-wave case that checks the line phase mapping; "
            "defaults to all selected evidence cases."
        ),
    )
    create.add_argument(
        "--attest-reviewed-evidence", action="store_true", required=True,
        help=(
            "Confirm a responsible team member reviewed the declared response, "
            "host, envelope, and machine-bound evidence. This does not replace "
            "independent solver convergence review."
        ),
    )
    create.add_argument(
        "--output",
        help="One legal adjacent sidecar name; default is RESPONSE.feature.json.",
    )
    create.add_argument("--force", action="store_true", help="Replace the selected sidecar.")

    check = commands.add_parser(
        "check",
        help="Verify discovery, schema, identity, applicability, and content binding.",
    )
    check.add_argument("response", help="Exact feature-response GRIM/NPZ file.")
    _add_identity_arguments(check)

    create_surface = commands.add_parser(
        "create-surface-binding",
        help="Bind one external base GRIM to its exact placement surface.",
    )
    create_surface.add_argument("base_grim", help="External clean-body GRIM.")
    create_surface.add_argument("surface", help="Matching STL or .facet surface.")
    create_surface.add_argument("--surface-units", required=True)
    create_surface.add_argument(
        "--geometry-id",
        required=True,
        help="Team-controlled ID for this exact CAD/mesh revision.",
    )
    create_surface.add_argument(
        "--attestation-case-id",
        required=True,
        help="Reviewed solve-to-surface registration evidence/case ID.",
    )
    create_surface.add_argument(
        "--attest-reviewed-registration",
        action="store_true",
        required=True,
        help=(
            "Confirm a responsible team member established that this exact "
            "surface, unit choice, origin, and frame match the body solve."
        ),
    )
    create_surface.add_argument(
        "--force",
        action="store_true",
        help="Replace the canonical <surface>.assembly.json sidecar.",
    )

    check_surface = commands.add_parser(
        "check-surface-binding",
        help="Verify exact base/surface hashes, units, frame, and attestation IDs.",
    )
    check_surface.add_argument("base_grim", help="External clean-body GRIM.")
    check_surface.add_argument("surface", help="Matching STL or .facet surface.")
    check_surface.add_argument("--surface-units", required=True)
    check_surface.add_argument(
        "--geometry-id",
        help="Optional expected geometry revision ID.",
    )
    check_surface.add_argument(
        "--attestation-case-id",
        help="Optional expected registration evidence/case ID.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            output = _create(args)
            print(f"Wrote team-attested feature manifest: {output}")
            print(
                "The manifest records reviewed evidence; it does not replace "
                "independent full-wave validation."
            )
        elif args.command == "check":
            manifest = _check(args)
            frequency = manifest["applicability"]["frequency_ghz"]
            print(
                "Manifest OK: "
                f"{manifest['feature_kind']}:{manifest['dataset_id']}, "
                f"status={manifest['validation']['status']}, "
                f"host={manifest['host']['material']!r}, "
                f"frequency={frequency['min']:g}-{frequency['max']:g} GHz, "
                f"response_sha256={manifest['response_content_sha256']}"
            )
        elif args.command == "create-surface-binding":
            output = _create_surface_binding(args)
            print(f"Wrote team-attested Assembly surface binding: {output}")
            print(
                "The binding proves exact file identity; it records, but does "
                "not independently prove, solve-to-CAD registration."
            )
        else:
            binding = _check_surface_binding(args)
            print(
                "Surface binding OK: "
                f"geometry_id={binding['geometry_id']!r}, "
                f"case_id={binding['attestation_case_id']!r}, "
                f"units={binding['surface_units']}, "
                f"frame={binding['frame_convention']}"
            )
    except (FileNotFoundError, OSError, RuntimeError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
