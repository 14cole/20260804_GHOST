"""Lossless operations on solver-compatible GRIM/NPZ datasets.

This module deliberately avoids the viewer's float32 power/phase reconstruction
for coherent work. Solver ``rcs_amp_real``/``rcs_amp_imag`` arrays are the
authoritative fields and remain float64 through joins and subtraction.
"""

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

from .errors import CemToolError


C0 = 299_792_458.0
AXIS_KEYS = ("azimuths", "elevations", "frequencies", "polarizations")
REQUIRED_KEYS = AXIS_KEYS + ("rcs_power", "rcs_phase")
CRITICAL_METADATA = (
    "rcs_domain",
    "power_domain",
    "phase_reference",
    "amplitude_convention",
    "complex_field_domain",
    "units",
)
DELTA_FIELD_DOMAIN = "featured_minus_clean_far_field_amplitude_delta"
DELTA_PHASE_SUFFIX = (
    "; coherent subtraction=featured-clean; placement phase center is the "
    "seam line on the coupon outer face y=0"
)
PHYSICAL_2D_PHASE_REFERENCE = (
    "origin=(0,0), convention=exp(+jwt); stored complex field is the "
    "2D layer-potential bare-integral amplitude B. The coefficient "
    "in u_s~exp(-j(kr-pi/4))/sqrt(8*pi*k*r)*A is A=j*B."
)
PHYSICAL_2D_AMPLITUDE_CONVENTION = "A_physical_asymptotic = +j * B_stored"
PHYSICAL_2D_FIELD_DOMAIN = "2d_layer_potential_bare_integral_amplitude_B"
# Legacy/advisory evidence retained for explicit source-file inspection only.
# Numerical CEM transforms neither interpret nor copy this field.
MESH_CERTIFICATION_KEY = "production_mesh_certification_json"
MESH_CERTIFICATION_SCHEMA = "ghost.workflow.mesh-certified-sources.v1"
DUAL_OUTPUT_POLARIZATIONS = ("VV", "HH")
EMBEDDED_ATTESTATION_SCHEMA = "ghost.workflow.embedded-attestation.v1"
_ATTESTATION_SHA256_FIELDS = (
    "solver_source_sha256",
    "runtime_environment_sha256",
    "geometry_input_sha256",
    "run_solve_spec_sha256",
    "unit_solve_spec_sha256",
    "solver_config_sha256",
    "angular_grid_sha256",
)
_POLARIZATION_2D = {
    "TM": "TM", "HH": "TM", "H": "TM", "HORIZONTAL": "TM",
    "TE": "TE", "VV": "TE", "V": "TE", "VERTICAL": "TE",
}


def _scalar_text(value: 'Any') -> 'str':
    array = np.asarray(value)
    if array.size != 1:
        raise CemToolError("expected scalar GRIM metadata")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _canonical_output_polarization(value: 'Any') -> 'str':
    key = str(value).strip().upper()
    internal = _POLARIZATION_2D.get(key)
    if internal == "TE":
        return "VV"
    if internal == "TM":
        return "HH"
    raise CemToolError(f"unsupported 2-D polarization in certification: {value!r}")


def _require_passed_quality_gate(gate: 'Any', label: 'str') -> 'dict[str, Any]':
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise CemToolError(f"{label}: solver quality gate did not pass")
    return gate


def _require_passed_mesh_gate(gate: 'Any', label: 'str') -> 'dict[str, Any]':
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or gate.get("published_mesh") != "fine"
    ):
        raise CemToolError(
            f"{label}: mesh-convergence gate did not publish a passing fine mesh"
        )
    _require_passed_quality_gate(
        gate.get("base_quality_gate"), f"{label} base mesh"
    )
    _require_passed_quality_gate(
        gate.get("fine_quality_gate"), f"{label} fine mesh"
    )
    return gate


def _attestation_frequency(attestation: 'dict[str, Any]', label: 'str') -> 'float':
    try:
        value = float(attestation["frequency_ghz"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CemToolError(
            f"{label}: embedded output attestation has no finite frequency_ghz"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise CemToolError(
            f"{label}: embedded output attestation has no finite frequency_ghz"
        )
    return value


def _require_embedded_attestation(
    attestation: 'Any',
    label: 'str',
    *,
    frequency_ghz: 'float',
    polarizations: 'tuple[str, ...]',
) -> 'dict[str, Any]':
    """Validate the exact local/HPC embedded output-attestation contract."""

    if (
        not isinstance(attestation, dict)
        or attestation.get("schema") != EMBEDDED_ATTESTATION_SCHEMA
    ):
        raise CemToolError(
            f"{label}: missing current embedded output attestation"
        )
    if not str(attestation.get("run_id", "")).strip():
        raise CemToolError(f"{label}: embedded output attestation has no run_id")
    for field in _ATTESTATION_SHA256_FIELDS:
        digest = str(attestation.get(field, "")).strip()
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise CemToolError(
                f"{label}: embedded output attestation has invalid {field}"
            )
    if attestation.get("angular_grid_kind") != "azimuths_deg":
        raise CemToolError(
            f"{label}: embedded output attestation has the wrong angular grid kind"
        )
    recorded_frequency = _attestation_frequency(attestation, label)
    if not math.isclose(
        recorded_frequency, float(frequency_ghz), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise CemToolError(
            f"{label}: embedded output attestation frequency does not match "
            "the GRIM frequency axis"
        )

    if polarizations == DUAL_OUTPUT_POLARIZATIONS:
        recorded = attestation.get("polarizations")
        if not isinstance(recorded, list) or tuple(recorded) != polarizations:
            raise CemToolError(
                f"{label}: embedded output attestation must declare exactly "
                "polarizations=['VV','HH']"
            )
        if "polarization" in attestation:
            raise CemToolError(
                f"{label}: dual-channel output attestation must not select one "
                "polarization"
            )
    elif len(polarizations) == 1:
        if "polarizations" in attestation:
            raise CemToolError(
                f"{label}: legacy single-channel output attestation must not "
                "declare a channel collection"
            )
        try:
            recorded = _canonical_output_polarization(
                attestation.get("polarization")
            )
        except CemToolError as exc:
            raise CemToolError(
                f"{label}: embedded output attestation has no valid polarization"
            ) from exc
        if recorded != polarizations[0]:
            raise CemToolError(
                f"{label}: embedded output attestation polarization does not "
                "match the GRIM channel"
            )
    else:
        raise CemToolError(f"{label}: unsupported attested polarization contract")
    return attestation


def _require_channel_dictionary(
    value: 'Any', label: 'str'
) -> 'dict[str, Any]':
    if not isinstance(value, dict) or set(value) != set(DUAL_OUTPUT_POLARIZATIONS):
        raise CemToolError(
            f"{label}: certification must contain exactly VV and HH channels"
        )
    return value


def _validate_dual_channel_evidence(
    *,
    mesh: 'Any',
    quality: 'Any',
    channels: 'Any',
    label: 'str',
) -> 'None':
    """Require independent mesh and algebraic evidence for VV and HH."""

    aggregate_mesh = _require_passed_mesh_gate(mesh, f"{label} aggregate")
    aggregate_quality = _require_passed_quality_gate(
        quality, f"{label} aggregate"
    )
    channel_meshes = _require_channel_dictionary(
        aggregate_mesh.get("channels"), f"{label} aggregate mesh"
    )
    quality_channels = _require_channel_dictionary(
        aggregate_quality.get("channels"), f"{label} aggregate quality"
    )
    channel_records = _require_channel_dictionary(channels, label)

    for phase in ("base_quality_gate", "fine_quality_gate"):
        phase_gate = _require_passed_quality_gate(
            aggregate_mesh.get(phase), f"{label} aggregate {phase}"
        )
        phase_channels = _require_channel_dictionary(
            phase_gate.get("channels"), f"{label} aggregate {phase}"
        )
        for polarization in DUAL_OUTPUT_POLARIZATIONS:
            _require_passed_quality_gate(
                phase_channels[polarization],
                f"{label} {polarization} aggregate {phase}",
            )

    for polarization in DUAL_OUTPUT_POLARIZATIONS:
        channel_label = f"{label} {polarization}"
        _require_passed_quality_gate(
            quality_channels[polarization], channel_label
        )
        _require_passed_mesh_gate(
            channel_meshes[polarization], channel_label
        )
        record = channel_records[polarization]
        if not isinstance(record, dict):
            raise CemToolError(f"{channel_label}: malformed channel metadata")
        _require_passed_quality_gate(record.get("quality_gate"), channel_label)
        _require_passed_mesh_gate(
            record.get("mesh_convergence"), channel_label
        )


def _validate_propagated_source(source: 'Any', label: 'str') -> 'None':
    if not isinstance(source, dict):
        raise CemToolError(f"{label}: malformed source-chain record")
    contract = source.get("contract")
    if contract == "embedded-dual-output-attestation.v1":
        polarizations = tuple(source.get("polarizations", ()))
        if polarizations != DUAL_OUTPUT_POLARIZATIONS:
            raise CemToolError(
                f"{label}: dual source-chain record does not contain VV and HH"
            )
        try:
            frequency_value = float(source.get("frequency_ghz"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CemToolError(
                f"{label}: dual source-chain record has invalid frequency"
            ) from exc
        _require_embedded_attestation(
            source.get("output_attestation"),
            label,
            frequency_ghz=frequency_value,
            polarizations=DUAL_OUTPUT_POLARIZATIONS,
        )
        _validate_dual_channel_evidence(
            mesh=source.get("mesh_convergence"),
            quality=source.get("quality_gate"),
            channels=source.get("channels"),
            label=label,
        )
        return
    if contract == "embedded-single-output-attestation.v1":
        polarization = _canonical_output_polarization(
            source.get("polarization")
        )
        try:
            frequency_value = float(source.get("frequency_ghz"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CemToolError(
                f"{label}: single-channel source-chain record has invalid frequency"
            ) from exc
        _require_embedded_attestation(
            source.get("output_attestation"),
            label,
            frequency_ghz=frequency_value,
            polarizations=(polarization,),
        )
        _require_passed_mesh_gate(source.get("mesh_convergence"), label)
        _require_passed_quality_gate(source.get("quality_gate"), label)
        return

    # Records emitted by the original workflow-unit contract did not carry a
    # contract discriminator or a top-level quality gate. Its base/fine gates
    # are nevertheless preserved and still validated here.
    if contract not in (None, "legacy-workflow-unit.v1"):
        raise CemToolError(f"{label}: unsupported source-chain contract")
    _require_passed_mesh_gate(source.get("mesh_convergence"), label)


def _mesh_certification_sources(
    payload: 'dict[str, np.ndarray]',
    label: 'str',
) -> 'list[dict[str, Any]] | None':
    """Explicitly audit optional source evidence, including legacy CEM chains.

    Normal joins and subtraction deliberately do not call this helper; mesh
    certification is advisory and never authorizes a numerical operation.
    """

    if MESH_CERTIFICATION_KEY in payload:
        try:
            decoded = json.loads(
                _scalar_text(payload[MESH_CERTIFICATION_KEY])
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CemToolError(
                f"{label}: malformed production mesh certification"
            ) from exc
        sources = decoded.get("sources") if isinstance(decoded, dict) else None
        if (
            not isinstance(decoded, dict)
            or decoded.get("schema") != MESH_CERTIFICATION_SCHEMA
            or decoded.get("passed") is not True
            or decoded.get("published_mesh") != "fine"
            or not isinstance(sources, list)
            or not sources
            or (
                "source_count" in decoded
                and decoded.get("source_count") != len(sources)
            )
        ):
            raise CemToolError(
                f"{label}: unsupported production mesh certification"
            )
        if decoded.get("certifies_coherent_delta_convergence") is True:
            raise CemToolError(
                f"{label}: invalid source-chain certificate claims coherent "
                "delta convergence"
            )
        for index, source in enumerate(sources):
            _validate_propagated_source(
                source, f"{label} source-chain item {index + 1}"
            )
        return sources

    if "solver_metadata_json" not in payload:
        return None
    try:
        audit = json.loads(_scalar_text(payload["solver_metadata_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CemToolError(
            f"{label}: malformed solver metadata"
        ) from exc
    metadata = audit.get("metadata") if isinstance(audit, dict) else None
    mesh = (
        metadata.get("mesh_convergence")
        if isinstance(metadata, dict)
        else None
    )
    if mesh is None:
        return None
    output_attestation = metadata.get("output_attestation")
    payload_polarizations = tuple(
        str(value).strip().upper()
        for value in np.asarray(payload["polarizations"]).ravel()
    )
    frequencies = np.asarray(payload["frequencies"], dtype=float).ravel()

    if output_attestation is not None and "polarizations" in output_attestation:
        if payload_polarizations != DUAL_OUTPUT_POLARIZATIONS:
            raise CemToolError(
                f"{label}: an attested dual-channel artifact must store exactly "
                "polarizations ['VV', 'HH'] in that order"
            )
        if frequencies.size != 1:
            raise CemToolError(
                f"{label}: an attested solver unit must contain one frequency"
            )
        if tuple(audit.get("polarizations", ())) != DUAL_OUTPUT_POLARIZATIONS:
            raise CemToolError(
                f"{label}: solver audit does not declare exactly VV and HH"
            )
        if tuple(metadata.get("polarizations", ())) != DUAL_OUTPUT_POLARIZATIONS:
            raise CemToolError(
                f"{label}: solver metadata does not declare exactly VV and HH"
            )
        if metadata.get("polarization_mapping") != {"VV": "TE", "HH": "TM"}:
            raise CemToolError(
                f"{label}: solver metadata has the wrong VV/HH scalar mapping"
            )
        _require_embedded_attestation(
            output_attestation,
            label,
            frequency_ghz=float(frequencies[0]),
            polarizations=DUAL_OUTPUT_POLARIZATIONS,
        )
        if (
            metadata.get("mesh_convergence_certified") is not True
            or metadata.get("certified_entry_point") is not True
            or metadata.get("published_mesh") != "fine"
        ):
            raise CemToolError(
                f"{label}: dual-channel solver result was not published by the "
                "certified fine-mesh entry point"
            )
        quality = metadata.get("quality_gate")
        channel_metadata = _require_channel_dictionary(
            metadata.get("channel_metadata"), label
        )
        _validate_dual_channel_evidence(
            mesh=mesh,
            quality=quality,
            channels=channel_metadata,
            label=label,
        )
        for polarization in DUAL_OUTPUT_POLARIZATIONS:
            channel = channel_metadata[polarization]
            if (
                channel.get("mesh_convergence_certified") is not True
                or channel.get("certified_entry_point") is not True
                or channel.get("published_mesh") != "fine"
            ):
                raise CemToolError(
                    f"{label} {polarization}: channel was not published by the "
                    "certified fine-mesh entry point"
                )
        return [{
            "contract": "embedded-dual-output-attestation.v1",
            "label": str(label),
            "frequency_ghz": float(frequencies[0]),
            "polarizations": list(DUAL_OUTPUT_POLARIZATIONS),
            "output_attestation": output_attestation,
            "mesh_convergence": mesh,
            "quality_gate": quality,
            "channels": {
                polarization: {
                    "mesh_convergence": channel_metadata[polarization][
                        "mesh_convergence"
                    ],
                    "quality_gate": channel_metadata[polarization][
                        "quality_gate"
                    ],
                }
                for polarization in DUAL_OUTPUT_POLARIZATIONS
            },
        }]

    if output_attestation is not None:
        if len(payload_polarizations) != 1 or frequencies.size != 1:
            raise CemToolError(
                f"{label}: legacy attested solver unit must contain one channel "
                "and one frequency"
            )
        polarization = _canonical_output_polarization(payload_polarizations[0])
        _require_embedded_attestation(
            output_attestation,
            label,
            frequency_ghz=float(frequencies[0]),
            polarizations=(polarization,),
        )
        _require_passed_mesh_gate(mesh, label)
        quality = _require_passed_quality_gate(
            metadata.get("quality_gate"), label
        )
        return [{
            "contract": "embedded-single-output-attestation.v1",
            "label": str(label),
            "frequency_ghz": float(frequencies[0]),
            "polarization": polarization,
            "output_attestation": output_attestation,
            "mesh_convergence": mesh,
            "quality_gate": quality,
        }]

    # Backward compatibility for artifacts produced before embedded output
    # attestations. These carried a workflow_unit beside a scalar mesh gate.
    workflow = metadata.get("workflow_unit")
    try:
        _require_passed_mesh_gate(mesh, label)
    except CemToolError as exc:
        raise CemToolError(
            f"{label}: solver mesh certification did not pass the strict "
            "base/fine production policy"
        ) from exc
    if (
        not isinstance(workflow, dict)
        or workflow.get("published_mesh") != "fine"
        or not isinstance(workflow.get("mesh_convergence_policy"), dict)
    ):
        raise CemToolError(
            f"{label}: solver mesh certification did not pass the strict "
            "base/fine production policy"
        )
    return [{
        "contract": "legacy-workflow-unit.v1",
        "label": str(label),
        "workflow_unit_sha256": str(workflow.get("unit_sha256", "")),
        "frequency_ghz": workflow.get("frequency_ghz"),
        "polarization": workflow.get("polarization"),
        "mesh_convergence": mesh,
    }]


def _grid_shape(payload: 'dict[str, np.ndarray]') -> 'tuple[int, int, int, int]':
    return tuple(len(np.asarray(payload[key]).ravel()) for key in AXIS_KEYS)


def load_grim(path: 'str | os.PathLike[str]') -> 'dict[str, np.ndarray]':
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".grim" or not source.is_file():
        raise CemToolError(f"not a readable .grim file: {source}")
    try:
        with np.load(source, allow_pickle=False) as data:
            payload = {key: np.array(data[key], copy=True) for key in data.files}
    except (OSError, ValueError, TypeError) as exc:
        raise CemToolError(f"cannot read {source.name}: {exc}") from exc
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise CemToolError(
            f"{source.name} is missing required GRIM fields {missing}"
        )
    for key in AXIS_KEYS:
        original = np.asarray(payload[key])
        if original.ndim != 1 or original.size == 0:
            raise CemToolError(
                f"{source.name}: axis {key} must be a nonempty 1-D array"
            )
        axis = original.ravel()
        if key == "polarizations":
            labels = [str(value).strip() for value in axis]
            if any(not value for value in labels) or len(labels) != len(set(labels)):
                raise CemToolError(
                    f"{source.name}: polarization labels must be nonempty and unique"
                )
            axis = np.asarray(labels, dtype=str)
        else:
            try:
                axis = np.asarray(axis, dtype=float)
            except (TypeError, ValueError) as exc:
                raise CemToolError(
                    f"{source.name}: axis {key} must be numeric"
                ) from exc
            if not np.all(np.isfinite(axis)):
                raise CemToolError(f"{source.name}: axis {key} is nonfinite")
            if len(np.unique(axis)) != len(axis):
                raise CemToolError(f"{source.name}: axis {key} contains duplicates")
            if key == "frequencies" and np.any(axis <= 0.0):
                raise CemToolError(
                    f"{source.name}: frequencies must be positive"
                )
        payload[key] = axis
    shape = _grid_shape(payload)
    for key in ("rcs_power", "rcs_phase"):
        values = np.asarray(payload[key])
        if values.shape != shape:
            raise CemToolError(
                f"{source.name}: {key} shape {np.shape(payload[key])} != {shape}"
            )
        if not np.all(np.isfinite(values)):
            raise CemToolError(f"{source.name}: {key} contains nonfinite values")
        if key == "rcs_power" and np.any(values < 0.0):
            raise CemToolError(f"{source.name}: rcs_power contains negative values")
    for key in ("rcs_amp_real", "rcs_amp_imag"):
        if key in payload:
            values = np.asarray(payload[key])
            if values.shape != shape:
                raise CemToolError(
                    f"{source.name}: {key} shape {np.shape(payload[key])} != {shape}"
                )
            if not np.all(np.isfinite(values)):
                raise CemToolError(f"{source.name}: {key} contains nonfinite values")
    return payload


def save_grim_atomic(
    payload: 'dict[str, Any]',
    path: 'str | os.PathLike[str]',
    *,
    overwrite: 'bool' = False,
) -> 'Path':
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".grim":
        destination = destination.with_suffix(".grim")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise CemToolError(f"output exists: {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def raw_amplitude(payload: 'dict[str, np.ndarray]', label: 'str') -> 'np.ndarray':
    if "rcs_amp_real" not in payload or "rcs_amp_imag" not in payload:
        raise CemToolError(
            f"{label}: coherent operations require preserved "
            "rcs_amp_real/rcs_amp_imag arrays"
        )
    amplitude = (
        np.asarray(payload["rcs_amp_real"], dtype=np.float64)
        + 1j * np.asarray(payload["rcs_amp_imag"], dtype=np.float64)
    )
    if not np.all(np.isfinite(amplitude)):
        raise CemToolError(f"{label}: raw complex amplitude is nonfinite")
    return amplitude


def _axis_equal(left: 'np.ndarray', right: 'np.ndarray', categorical: 'bool') -> 'bool':
    if categorical:
        return np.array_equal(
            np.asarray(left).astype(str), np.asarray(right).astype(str)
        )
    return np.array_equal(np.asarray(left, float), np.asarray(right, float))


def _critical_metadata_equal(
    payloads: 'list[dict[str, np.ndarray]]', labels: 'list[str]'
) -> 'None':
    for key in CRITICAL_METADATA:
        present = [key in payload for payload in payloads]
        if any(present) and not all(present):
            raise CemToolError(
                f"{key} metadata is missing from part of the group: {labels}"
            )
        if all(present):
            values = [_scalar_text(payload[key]) for payload in payloads]
            if len(set(values)) != 1:
                raise CemToolError(
                    f"incompatible {key} metadata in {labels}: {values}"
                )


def _union_axis(arrays: 'Iterable[np.ndarray]', categorical: 'bool') -> 'np.ndarray':
    if categorical:
        values: 'list[str]' = []
        for array in arrays:
            for raw in np.asarray(array).ravel():
                value = str(raw)
                if value not in values:
                    values.append(value)
        return np.asarray(values, dtype=str)
    values = sorted(
        {float(raw) for array in arrays for raw in np.asarray(array).ravel()}
    )
    return np.asarray(values, dtype=float)


def _indices(union: 'np.ndarray', values: 'np.ndarray', categorical: 'bool') -> 'list[int]':
    result: 'list[int]' = []
    for raw in np.asarray(values).ravel():
        if categorical:
            matches = np.flatnonzero(union.astype(str) == str(raw))
        else:
            matches = np.flatnonzero(
                np.isclose(union.astype(float), float(raw), rtol=0.0, atol=1e-9)
            )
        if len(matches) != 1:
            raise CemToolError(f"cannot uniquely align axis value {raw!r}")
        result.append(int(matches[0]))
    return result


def _merge_metadata(
    payloads: 'list[dict[str, np.ndarray]]',
    shape: 'tuple[int, int, int, int]',
    history: 'str',
) -> 'dict[str, Any]':
    first = payloads[0]
    merged: 'dict[str, Any]' = {}
    excluded = set(AXIS_KEYS) | {
        "rcs_power",
        "rcs_phase",
        "rcs_amp_real",
        "rcs_amp_imag",
        "source_path",
        "history",
        "solver_metadata_json",
        MESH_CERTIFICATION_KEY,
        "polarization_alias_primary",
        "polarization_aliases_json",
    }
    for key, value in first.items():
        if key in excluded:
            continue
        array = np.asarray(value)
        if array.shape == _grid_shape(first):
            continue
        candidates = [payload.get(key) for payload in payloads]
        if all(candidate is not None for candidate in candidates):
            try:
                if all(
                    np.array_equal(np.asarray(candidates[0]), np.asarray(candidate))
                    for candidate in candidates[1:]
                ):
                    merged[key] = value
            except (TypeError, ValueError):
                pass
    merged["source_path"] = np.asarray("")
    merged["history"] = np.asarray(history)
    return merged


def join_payloads(
    payloads: 'list[dict[str, np.ndarray]]',
    *,
    axis: 'str',
    labels: 'list[str] | None' = None,
) -> 'dict[str, Any]':
    if not payloads:
        raise CemToolError("no GRIM datasets to concatenate")
    labels = labels or [f"input {index}" for index in range(len(payloads))]
    if len(labels) != len(payloads):
        raise CemToolError("join labels do not match input count")
    for payload, label in zip(payloads, labels):
        has_real = "rcs_amp_real" in payload
        has_imag = "rcs_amp_imag" in payload
        if has_real != has_imag:
            raise CemToolError(
                f"{label}: raw complex amplitude must contain both real and imaginary arrays"
            )
    raw_presence = ["rcs_amp_real" in payload for payload in payloads]
    if any(raw_presence) and not all(raw_presence):
        raise CemToolError(
            "cannot concatenate a mixture of raw solver fields and "
            "magnitude/phase-only datasets"
        )
    _critical_metadata_equal(payloads, labels)
    join_index = AXIS_KEYS.index(axis)
    for key_index, key in enumerate(AXIS_KEYS):
        if key_index == join_index:
            continue
        categorical = key == "polarizations"
        reference = payloads[0][key]
        for payload, label in zip(payloads[1:], labels[1:]):
            if not _axis_equal(reference, payload[key], categorical):
                raise CemToolError(
                    f"{label}: {key} differs from the other files in its group"
                )
    axes = {
        key: (
            _union_axis(
                [payload[key] for payload in payloads],
                key == "polarizations",
            )
            if key == axis
            else np.array(payloads[0][key], copy=True)
        )
        for key in AXIS_KEYS
    }
    shape = tuple(len(axes[key]) for key in AXIS_KEYS)
    common_grid_keys = {
        key
        for key in payloads[0]
        if np.asarray(payloads[0][key]).shape == _grid_shape(payloads[0])
        and all(
            key in payload
            and np.asarray(payload[key]).shape == _grid_shape(payload)
            for payload in payloads
        )
    }
    common_grid_keys.update({"rcs_power", "rcs_phase"})
    merged_arrays: 'dict[str, np.ndarray]' = {}
    filled: 'dict[str, np.ndarray]' = {}
    for key in common_grid_keys:
        dtype = np.result_type(*[np.asarray(payload[key]).dtype for payload in payloads])
        if np.issubdtype(dtype, np.floating):
            array = np.full(shape, np.nan, dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            array = np.full(shape, np.nan + 1j * np.nan, dtype=dtype)
        else:
            continue
        merged_arrays[key] = array
        filled[key] = np.zeros(shape, dtype=bool)
    for payload, label in zip(payloads, labels):
        selections = [
            _indices(
                axes[key],
                payload[key],
                key == "polarizations",
            )
            for key in AXIS_KEYS
        ]
        destination = np.ix_(*selections)
        for key, output in merged_arrays.items():
            incoming = np.asarray(payload[key])
            occupied = filled[key][destination]
            if np.any(occupied):
                existing = output[destination]
                if not np.allclose(
                    existing[occupied],
                    incoming[occupied],
                    rtol=1e-12,
                    atol=0.0,
                    equal_nan=True,
                ):
                    raise CemToolError(
                        f"{label}: overlapping {key} samples disagree"
                    )
            output[destination] = incoming
            filled[key][destination] = True
    for key, mask in filled.items():
        if not np.all(mask):
            raise CemToolError(f"concatenation left missing {key} cells")
    history = (
        f"CEM Tools concatenated {axis}: "
        + ", ".join(Path(label).name for label in labels)
    )
    result = _merge_metadata(payloads, shape, history)
    result.update(axes)
    result.update(merged_arrays)
    if "rcs_amp_real" in result and "rcs_amp_imag" in result:
        result["rcs_amp_real"] = np.asarray(result["rcs_amp_real"], np.float64)
        result["rcs_amp_imag"] = np.asarray(result["rcs_amp_imag"], np.float64)
        result["raw_complex_amplitude_preserved"] = np.asarray(True)
    return result


def _normalization_per_frequency(
    payloads: 'list[dict[str, np.ndarray]]',
    amplitudes: 'list[np.ndarray]',
    labels: 'list[str]',
) -> 'np.ndarray':
    """Validate and return the exact solver 2-D power normalization.

    The current solver defines ``sigma_2d = |B|^2 / (4 k)``.  Deriving that
    factor from nonzero samples made an all-zero field impossible to subtract
    and let float32 rounding in ``rcs_power`` perturb the delta normalization.
    The convention metadata has already established the field as the canonical
    solver ``B`` amplitude, so use the analytic factor and treat power only as
    a consistency check.
    """
    frequencies = np.asarray(payloads[0]["frequencies"], dtype=float)
    wave_numbers = 2.0 * math.pi * frequencies * 1.0e9 / C0
    factors = 1.0 / (4.0 * wave_numbers)
    for frequency_index, frequency in enumerate(frequencies):
        for payload, amplitude, label in zip(payloads, amplitudes, labels):
            power = np.asarray(payload["rcs_power"], dtype=float)[
                :, :, frequency_index, :
            ]
            with np.errstate(over="ignore", invalid="ignore"):
                predicted = (
                    np.abs(amplitude[:, :, frequency_index, :]) ** 2
                    * factors[frequency_index]
                )
            tolerance = (
                8.0 * np.finfo(np.float32).eps
                * np.maximum(np.abs(predicted), np.abs(power))
                + np.finfo(np.float32).tiny
            )
            if (
                not np.all(np.isfinite(power))
                or not np.all(np.isfinite(predicted))
                or np.any(power < 0.0)
                or np.any(np.abs(power - predicted) > tolerance)
            ):
                raise CemToolError(
                    f"{label}: rcs_power is inconsistent with canonical "
                    f"sigma_2d=|B|^2/(4k) at {frequency:g} GHz"
                )
    return factors


def _canonical_2d_indices(
    payload: 'dict[str, np.ndarray]', label: 'str'
) -> 'list[int]':
    """Return source indices in canonical VV/HH (TE/TM) order."""

    channels: 'dict[str, int]' = {}
    for index, raw in enumerate(np.asarray(payload["polarizations"]).ravel()):
        key = str(raw).strip().upper()
        if key not in _POLARIZATION_2D:
            raise CemToolError(
                f"{label}: unsupported 2D polarization {raw!r}; require the "
                "complete VV/HH pair (aliases TE/TM are accepted)"
            )
        canonical = _POLARIZATION_2D[key]
        if canonical in channels:
            raise CemToolError(
                f"{label}: polarization aliases collide on {canonical}"
            )
        channels[canonical] = index
    if set(channels) != {"TE", "TM"}:
        raise CemToolError(
            f"{label}: coherent line-feature deltas require exactly both "
            f"VV/TE and HH/TM; got {list(np.asarray(payload['polarizations']).astype(str))}"
        )
    return [channels["TE"], channels["TM"]]


def _validate_2d_source(payload: 'dict[str, np.ndarray]', label: 'str') -> 'None':
    expected = {
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "phase_reference": PHYSICAL_2D_PHASE_REFERENCE,
        "amplitude_convention": PHYSICAL_2D_AMPLITUDE_CONVENTION,
        "complex_field_domain": PHYSICAL_2D_FIELD_DOMAIN,
    }
    for key, wanted in expected.items():
        if key not in payload:
            raise CemToolError(
                f"{label}: missing required 2D solver metadata {key!r}; "
                "legacy/ambiguous coherent fields cannot be subtracted safely"
            )
        actual = _scalar_text(payload[key])
        if actual != wanted:
            raise CemToolError(
                f"{label}: {key}={actual!r}; expected canonical 2D value {wanted!r}"
            )
    if not np.array_equal(
        np.asarray(payload["elevations"], dtype=float), np.asarray([0.0])
    ):
        raise CemToolError(f"{label}: a 2D source requires elevation axis [0.0]")
    try:
        units = json.loads(_scalar_text(payload["units"]))
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise CemToolError(f"{label}: invalid or missing units metadata") from exc
    if (
        units.get("rcs_linear_quantity") != "sigma_2d"
        or units.get("rcs_log_unit") != "dBke"
    ):
        raise CemToolError(
            f"{label}: coherent feature subtraction requires sigma_2d/dBke units"
        )


def subtract_payloads(
    featured: 'dict[str, np.ndarray]',
    clean: 'dict[str, np.ndarray]',
    *,
    featured_label: 'str',
    clean_label: 'str',
) -> 'dict[str, Any]':
    _validate_2d_source(featured, featured_label)
    _validate_2d_source(clean, clean_label)
    _critical_metadata_equal(
        [featured, clean], [featured_label, clean_label]
    )
    for key in AXIS_KEYS[:-1]:
        if not _axis_equal(
            featured[key], clean[key], False
        ):
            raise CemToolError(
                f"featured and clean {key} axes differ for "
                f"{featured_label} / {clean_label}"
            )
    featured_indices = _canonical_2d_indices(featured, featured_label)
    clean_indices = _canonical_2d_indices(clean, clean_label)
    featured_amp_raw = raw_amplitude(featured, featured_label)
    clean_amp_raw = raw_amplitude(clean, clean_label)
    factors = _normalization_per_frequency(
        [featured, clean],
        [featured_amp_raw, clean_amp_raw],
        [featured_label, clean_label],
    )
    featured_amp = featured_amp_raw[..., featured_indices]
    clean_amp = clean_amp_raw[..., clean_indices]
    delta = featured_amp - clean_amp
    power = (
        np.abs(delta) ** 2
        * factors.reshape(1, 1, -1, 1)
    )
    result = _merge_metadata(
        [featured, clean],
        delta.shape,
        f"CEM Tools coherent subtraction: {featured_label} - {clean_label}",
    )
    for key in AXIS_KEYS[:-1]:
        result[key] = np.array(featured[key], copy=True)
    result["polarizations"] = np.asarray(["VV", "HH"], dtype=str)
    result.update(
        rcs_power=power.astype(np.float32),
        rcs_phase=np.angle(delta).astype(np.float32),
        rcs_amp_real=delta.real.astype(np.float64),
        rcs_amp_imag=delta.imag.astype(np.float64),
        raw_complex_amplitude_preserved=np.asarray(True),
        rcs_domain=np.asarray("delta"),
        power_domain=np.asarray("linear_rcs"),
        complex_field_domain=np.asarray(DELTA_FIELD_DOMAIN),
        polarization_alias_primary=np.asarray("TE,TM"),
        polarization_aliases_json=np.asarray(json.dumps(["TE", "TM"])),
    )
    phase_reference = _scalar_text(featured.get("phase_reference", ""))
    if phase_reference:
        result["phase_reference"] = np.asarray(
            phase_reference + DELTA_PHASE_SUFFIX
        )
    return result
