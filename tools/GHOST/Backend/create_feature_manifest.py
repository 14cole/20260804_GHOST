#!/usr/bin/env python3
"""Create/check team-attested Assembly response and surface manifests.

This command records reviewed engineering claims and binds them to the exact
GRIM/geometry payloads. It does not run a full-wave comparison and does not
machine-certify electromagnetic accuracy or CAD registration.
"""

from __future__ import annotations

import argparse
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
    if args.validation_status == "validated" and not validation_case_ids:
        raise ValueError(
            "A validated declaration requires at least one "
            "--validation-case-id from reviewed evidence."
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
        "response_content_sha256": feature_response_content_sha256(response),
        "host": {"material": host_material},
        "applicability": applicability,
        "validation": {
            "status": args.validation_status,
            "case_ids": validation_case_ids,
        },
    }

    line_values = (
        args.minimum_along_line_normal_turn_radius_m,
        args.maximum_conical_incidence_deg,
        args.maximum_path_vertex_turn_deg,
    )
    if args.feature_kind == "line":
        if any(value is None for value in line_values):
            raise ValueError(
                "A line manifest requires --minimum-along-line-normal-turn-"
                "radius-m, --maximum-conical-incidence-deg, and "
                "--maximum-path-vertex-turn-deg."
            )
        if not phase_case_ids:
            raise ValueError(
                "A line manifest requires at least one independent "
                "--phase-calibration-case-id."
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
    create.add_argument("--frequency-max-ghz", required=True, type=float)
    create.add_argument("--footprint-radius-m", required=True, type=float)
    create.add_argument(
        "--validation-status", required=True,
        choices=("validated", "provisional", "uncertified"),
    )
    create.add_argument(
        "--validation-case-id", action="append", default=[],
        help="Reviewed full-wave validation case ID; repeat for multiple cases.",
    )
    create.add_argument("--minimum-along-line-normal-turn-radius-m", type=float)
    create.add_argument("--maximum-conical-incidence-deg", type=float)
    create.add_argument("--maximum-path-vertex-turn-deg", type=float)
    create.add_argument(
        "--phase-calibration-case-id", action="append", default=[],
        help="Independent line phase-calibration case ID; repeat as needed.",
    )
    create.add_argument(
        "--attest-reviewed-evidence", action="store_true", required=True,
        help=(
            "Confirm a responsible team member reviewed the declared response, "
            "host, envelope, and case IDs. This is an attestation, not a solver "
            "certification."
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
