"""Stable provenance helpers for numbered solver workflows.

The workflow cache may reuse a coherent field only when the exact
Python/native implementation and relevant numerical runtime match the
recorded run. These helpers intentionally over-invalidate: rebuilding is
safer than accepting a field made by an unrecorded implementation.
"""

import glob
import hashlib
import json
import math
import ntpath
import os
import platform
import posixpath
import sys
import tempfile
from typing import Any, Dict, List, Sequence


_BACKEND_SOURCE_SUFFIXES = (
    ".py",
    ".pyx",
    ".pxd",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".f",
    ".f90",
    ".so",
    ".dylib",
    ".dll",
)


def backend_source_paths(backend_dir: 'str') -> 'List[str]':
    """All top-level Python and native artifacts that can affect a solve."""

    paths: 'List[str]' = []
    for suffix in _BACKEND_SOURCE_SUFFIXES:
        paths.extend(glob.glob(os.path.join(backend_dir, f"*{suffix}")))
    return sorted(set(os.path.abspath(path) for path in paths))


def sha256_file(path: 'str') -> 'str':
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_bundle_fingerprint(records: 'Dict[str, str]') -> 'str':
    """Hash logical source names and exact bytes, independent of location."""

    payload = [
        {
            "path": str(logical_name),
            "sha256": sha256_file(os.path.abspath(path)),
        }
        for logical_name, path in sorted(records.items())
    ]
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def backend_source_records(
    backend_dir: 'str',
    extra_records: 'Dict[str, str]' = None,
) -> 'Dict[str, str]':
    """Logical name -> path for everything a solve's identity depends on."""

    records = {
        f"Backend/{os.path.basename(path)}": path
        for path in backend_source_paths(backend_dir)
    }
    records.update(extra_records or {})
    return records


def backend_source_inventory(
    backend_dir: 'str',
    extra_records: 'Dict[str, str]' = None,
) -> 'Dict[str, str]':
    """Per-file hashes behind `backend_source_fingerprint`.

    The fingerprint alone can only say that *something* under Backend/ differs
    from what a run recorded, which is not enough to act on -- the usual cause
    is a partially updated tree, and the useful question is which file.
    Recording the inventory beside the fingerprint lets the mismatch name the
    files that were added, removed, or edited.
    """

    return {
        name: sha256_file(os.path.abspath(path))
        for name, path in sorted(backend_source_records(
            backend_dir, extra_records
        ).items())
    }


def compare_source_inventories(
    expected: 'Dict[str, str]',
    actual: 'Dict[str, str]',
) -> 'Dict[str, List[str]]':
    """(changed, added, removed) logical names between two inventories."""

    expected = dict(expected or {})
    actual = dict(actual or {})
    return {
        "changed": sorted(
            name for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        ),
        "added": sorted(set(actual) - set(expected)),
        "removed": sorted(set(expected) - set(actual)),
    }


def describe_source_mismatch(
    expected: 'Dict[str, str]',
    actual: 'Dict[str, str]',
) -> 'str':
    """One-line summary of an inventory difference, or '' when identical."""

    diff = compare_source_inventories(expected, actual)
    parts = []
    for label in ("changed", "added", "removed"):
        names = diff[label]
        if names:
            shown = ", ".join(names[:6])
            if len(names) > 6:
                shown += f", ... (+{len(names) - 6} more)"
            parts.append(f"{label}: {shown}")
    return "; ".join(parts)


def backend_source_fingerprint(
    backend_dir: 'str',
    extra_records: 'Dict[str, str]' = None,
) -> 'str':
    """Hash a backend tree plus logically named runner/config files."""

    return source_bundle_fingerprint(
        backend_source_records(backend_dir, extra_records)
    )


def write_output_attestation(
    output_path: 'str',
    provenance: 'Dict[str, Any]',
) -> 'str':
    """Atomically bind one resumable output's exact bytes to its run state."""

    path = os.path.abspath(output_path)
    payload = dict(provenance)
    payload.update(
        schema="ghost.workflow.output-attestation.v1",
        output_name=os.path.basename(path),
        output_sha256=sha256_file(path),
    )
    attestation_path = path + ".provenance.json"
    temporary = attestation_path + f".tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, attestation_path)
    return attestation_path


def verify_output_attestation(
    output_path: 'str',
    expected: 'Dict[str, Any]',
) -> 'Dict[str, Any]':
    """Verify exact output bytes and every caller-specified provenance field."""

    path = os.path.abspath(output_path)
    attestation_path = path + ".provenance.json"
    try:
        with open(attestation_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"{os.path.basename(path)} has no readable output attestation."
        ) from exc
    if payload.get("schema") != "ghost.workflow.output-attestation.v1":
        raise ValueError(
            f"{os.path.basename(path)} has a legacy output attestation."
        )
    if payload.get("output_name") != os.path.basename(path):
        raise ValueError(
            f"{os.path.basename(path)} attestation names a different output."
        )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{os.path.basename(path)} attestation does not match {key}."
            )
    if payload.get("output_sha256") != sha256_file(path):
        raise ValueError(
            f"{os.path.basename(path)} bytes differ from its attestation."
        )
    return payload


def _artifact_path(
    root_abs: 'str',
    raw_name: 'str',
    *,
    label: 'str' = "Artifact output",
) -> 'tuple':
    """Return one canonical bundle name/path, rejecting path aliases."""

    if not isinstance(raw_name, str):
        raise ValueError(f"{label} names must be strings.")
    name = raw_name.replace("\\", "/")
    drive, _tail = ntpath.splitdrive(name)
    if (
        not name
        or drive
        or name.startswith("/")
        or posixpath.normpath(name) != name
        or name in (".", "..")
    ):
        raise ValueError(
            f"{label} {raw_name!r} is not a canonical relative path."
        )
    candidate = os.path.abspath(
        os.path.join(root_abs, *name.split("/"))
    )
    root_real = os.path.realpath(root_abs)
    candidate_real = os.path.realpath(candidate)
    try:
        contained = (
            os.path.commonpath([root_real, candidate_real]) == root_real
        )
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(
            f"{label} {raw_name!r} escapes its manifest root."
        )
    return name, candidate


def write_artifact_manifest(
    root: 'str',
    schema: 'str',
    expected_outputs: 'Sequence[str]',
    provenance: 'Dict[str, Any]' = None,
    manifest_name: 'str' = "collection_manifest.json",
) -> 'str':
    """Atomically commit an exact multi-file artifact bundle.

    The data files must already be in place.  Writing the manifest last makes
    it a commit marker: an interrupted replacement leaves either no manifest
    or hashes that no longer match, so a downstream consumer fails closed.
    """

    root_abs = os.path.abspath(root)
    if not os.path.isdir(root_abs):
        raise ValueError(
            f"Artifact manifest root is not a directory: {root_abs}"
        )
    manifest_rel, manifest_path = _artifact_path(
        root_abs, manifest_name, label="Artifact manifest"
    )
    if "/" in manifest_rel:
        raise ValueError("Artifact manifests must live directly in their root.")
    normalized = [
        _artifact_path(root_abs, name) for name in expected_outputs
    ]
    names = [name for name, _path in normalized]
    if not names or len(set(names)) != len(names):
        raise ValueError(
            "An artifact manifest needs a non-empty, unique output list."
        )
    if manifest_rel in names:
        raise ValueError(
            "An artifact manifest cannot include its own commit marker."
        )
    paths: 'Dict[str, str]' = {}
    real_targets = set()
    for name, candidate in normalized:
        if not os.path.isfile(candidate):
            raise ValueError(
                f"Cannot commit missing artifact output {name!r}."
            )
        target = os.path.realpath(candidate)
        if target in real_targets:
            raise ValueError(
                "Artifact outputs must identify unique physical files."
            )
        real_targets.add(target)
        paths[name] = candidate
    payload = dict(provenance or {})
    payload.update(
        schema=str(schema),
        status="complete",
        expected_outputs=names,
        output_sha256={
            name: sha256_file(path) for name, path in paths.items()
        },
    )
    fd, temporary = tempfile.mkstemp(
        prefix=f".{manifest_rel}.tmp.", dir=root_abs
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, manifest_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return manifest_path


def write_artifact_in_progress(
    root: 'str',
    schema: 'str',
    expected_outputs: 'Sequence[str]',
    provenance: 'Dict[str, Any]' = None,
    manifest_name: 'str' = "collection_manifest.json",
) -> 'str':
    """Atomically invalidate an older completed artifact commit.

    Long-running builders call this immediately before changing any output.
    If the process then fails, the old output bytes remain available for
    inspection, but no downstream consumer can mistake the mixed/incomplete
    directory for a completed bundle.
    """

    root_abs = os.path.abspath(root)
    if not os.path.isdir(root_abs):
        raise ValueError(
            f"Artifact manifest root is not a directory: {root_abs}"
        )
    manifest_rel, manifest_path = _artifact_path(
        root_abs, manifest_name, label="Artifact manifest"
    )
    if "/" in manifest_rel:
        raise ValueError("Artifact manifests must live directly in their root.")
    normalized = [
        _artifact_path(root_abs, name) for name in expected_outputs
    ]
    names = [name for name, _path in normalized]
    if not names or len(set(names)) != len(names):
        raise ValueError(
            "An artifact manifest needs a non-empty, unique output list."
        )
    if manifest_rel in names:
        raise ValueError(
            "An artifact manifest cannot include its own commit marker."
        )
    payload = dict(provenance or {})
    payload.update(
        schema=str(schema),
        status="in_progress",
        expected_outputs=names,
    )
    payload.pop("output_sha256", None)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{manifest_rel}.tmp.", dir=root_abs
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, manifest_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return manifest_path


def verify_artifact_manifest(
    root: 'str',
    required_outputs: 'Sequence[str]',
    manifest_names: 'Sequence[str]' = (
        "cache_manifest.json",
        "collection_manifest.json",
    ),
    exact_grim_set: 'bool' = False,
    expected_schema: 'str' = None,
) -> 'Dict[str, Any]':
    """Verify a local or collected bundle before downstream physical use."""

    root_abs = os.path.abspath(root)
    if not os.path.isdir(root_abs):
        raise ValueError(
            f"Artifact manifest root is not a directory: {root_abs}"
        )
    normalized_manifests = [
        _artifact_path(root_abs, name, label="Artifact manifest")
        for name in manifest_names
    ]
    if len({name for name, _path in normalized_manifests}) != len(
        normalized_manifests
    ):
        raise ValueError("Artifact manifest candidate names must be unique.")
    present = [
        path for _name, path in normalized_manifests
        if os.path.isfile(path)
    ]
    if len(present) != 1:
        raise ValueError(
            f"{root_abs} must contain exactly one readable cache/collection "
            "manifest; legacy or ambiguous artifact bundles are refused."
        )
    try:
        with open(present[0], encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"{present[0]} is unreadable; artifact bundle is uncommitted."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{present[0]} does not contain a manifest object."
        )
    schema = payload.get("schema")
    if (
        not isinstance(schema, str)
        or not schema.startswith("ghost.")
        or (expected_schema is not None and schema != expected_schema)
        or payload.get("status") != "complete"
    ):
        raise ValueError(
            f"{present[0]} is not a completed recognized artifact manifest."
        )
    raw_expected = payload.get("expected_outputs")
    if not isinstance(raw_expected, list):
        raise ValueError(
            f"{present[0]} has no explicit output inventory."
        )
    normalized_outputs = [
        _artifact_path(root_abs, name) for name in raw_expected
    ]
    expected = [name for name, _path in normalized_outputs]
    hashes = payload.get("output_sha256", {})
    if (
        not expected
        or len(set(expected)) != len(expected)
        or not isinstance(hashes, dict)
        or set(hashes) != set(expected)
    ):
        raise ValueError(
            f"{present[0]} has an incomplete or ambiguous output inventory."
        )
    required = {
        _artifact_path(root_abs, name, label="Required artifact")[0]
        for name in required_outputs
    }
    if not required.issubset(set(expected)):
        raise ValueError(
            f"{present[0]} does not commit every required artifact "
            f"(missing {sorted(required - set(expected))})."
        )
    real_targets = set()
    for name, candidate in normalized_outputs:
        target = os.path.realpath(candidate)
        if (
            target in real_targets
            or not os.path.isfile(candidate)
            or hashes.get(name) != sha256_file(candidate)
        ):
            raise ValueError(
                f"Artifact {name!r} is missing, outside its bundle, or differs "
                "from the committed bytes."
            )
        real_targets.add(target)
    if exact_grim_set:
        actual_grims = set()
        for directory, _subdirs, files in os.walk(root_abs):
            for filename in files:
                if filename.lower().endswith(".grim"):
                    actual_grims.add(
                        os.path.relpath(
                            os.path.join(directory, filename), root_abs
                        ).replace(os.sep, "/")
                    )
        expected_grims = {
            name for name in expected
            if name.lower().endswith(".grim")
        }
        if actual_grims != expected_grims:
            raise ValueError(
                "The artifact directory's .grim set differs from its "
                "committed inventory."
            )
    return payload


def verify_component_output_manifest(
    output_dir: 'str',
) -> 'Dict[str, Any]':
    """Verify one step-3 component directory and its exact direct GRIM set.

    Step 3b predates :func:`write_artifact_manifest` and records its diagnostic
    ``../wing_dbsm.csv`` in ``output_sha256`` in addition to the component
    GRIMs.  This verifier deliberately accepts that established schema while
    retaining strict path and byte checks; all other supported step-3 schemas
    require an exact hash map for their component outputs.
    """

    root_abs = os.path.abspath(output_dir)
    manifest_path = os.path.join(root_abs, "provenance_manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"{root_abs} has no readable component provenance manifest."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{manifest_path} does not contain a manifest object."
        )
    supported = {
        "ghost.workflow.doors-output-provenance.v1",
        "ghost.workflow.wing-output-provenance.v1",
        "ghost.workflow.cavity-output-provenance.v1",
    }
    schema = payload.get("schema")
    if schema not in supported or payload.get("status") != "complete":
        raise ValueError(
            f"{manifest_path} is not a completed supported component manifest."
        )
    raw_expected = payload.get("expected_outputs")
    if not isinstance(raw_expected, list) or not raw_expected:
        raise ValueError(
            f"{manifest_path} has no explicit component output inventory."
        )
    expected = []
    for raw_name in raw_expected:
        name, candidate = _artifact_path(
            root_abs, raw_name, label="Component output"
        )
        if "/" in name or not name.lower().endswith(".grim"):
            raise ValueError(
                "Component manifests may name only direct .grim outputs."
            )
        if not os.path.isfile(candidate):
            raise ValueError(f"Committed component {name!r} is missing.")
        expected.append(name)
    if len(set(expected)) != len(expected):
        raise ValueError("Component output inventory contains duplicates.")
    actual = {
        name
        for name in os.listdir(root_abs)
        if name.lower().endswith(".grim")
        and os.path.isfile(os.path.join(root_abs, name))
    }
    if actual != set(expected):
        raise ValueError(
            "Component directory .grim set differs from its committed "
            f"inventory (missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))})."
        )
    hashes = payload.get("output_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Component manifest has no output hash map.")
    allowed_hash_names = set(expected)
    if schema == "ghost.workflow.wing-output-provenance.v1":
        # Preserve the existing 3b schema exactly.  Its diagnostic is adjacent
        # to Output/, and is byte-attested even though step 4 does not consume it.
        allowed_hash_names.add("../wing_dbsm.csv")
    if set(hashes) != allowed_hash_names:
        raise ValueError(
            "Component manifest hash map differs from its schema's exact "
            "output set."
        )
    for name in expected:
        path = os.path.join(root_abs, name)
        if hashes.get(name) != sha256_file(path):
            raise ValueError(
                f"Component {name!r} differs from its committed bytes."
            )
    if schema == "ghost.workflow.wing-output-provenance.v1":
        summary = os.path.abspath(
            os.path.join(root_abs, os.pardir, "wing_dbsm.csv")
        )
        workflow_root = os.path.realpath(os.path.dirname(root_abs))
        if (
            os.path.commonpath([workflow_root, os.path.realpath(summary)])
            != workflow_root
            or not os.path.isfile(summary)
            or hashes.get("../wing_dbsm.csv") != sha256_file(summary)
        ):
            raise ValueError(
                "Wing diagnostic differs from its committed bytes."
            )
    return payload


def _stable_json_value(value: 'Any') -> 'Any':
    """Convert build-configuration objects to deterministic JSON values."""

    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    return str(value)


def stable_json_fingerprint(value: 'Any') -> 'str':
    """Hash a JSON-like solve specification deterministically."""

    raw = json.dumps(
        _stable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def manifest_solve_spec_fingerprint(manifest: 'Dict[str, Any]') -> 'str':
    """Hash every immutable field that defines the underlying unit solves.

    Completion status and byte inventories are written after individual unit
    attestations, so they are deliberately excluded. Editing a grid, unit,
    solver control, or source/runtime hash changes this fingerprint and
    invalidates every stale per-unit output. Derived az/el configuration has a
    separate attestation because it does not change the underlying VV/HH solve.
    """

    mutable_fields = {
        "status",
        "output_sha256",
        "collection",
        "collection_manifest",
        # Derived radar-grid settings do not change the underlying VV/HH
        # aspect solves; their own output attestations bind this separately.
        "azel_config",
    }
    solve_spec = {
        str(key): value
        for key, value in manifest.items()
        if str(key) not in mutable_fields
    }
    return stable_json_fingerprint(solve_spec)


def unit_solve_spec_fingerprint(unit: 'Dict[str, Any]') -> 'str':
    """Hash the complete per-unit record, including its angular grid."""

    return stable_json_fingerprint(unit)


def _package_configuration(package: 'Any') -> 'Dict[str, Any]':
    config_module = getattr(package, "__config__", None)
    config = getattr(config_module, "CONFIG", None)
    return _stable_json_value(config) if isinstance(config, dict) else {}


def runtime_environment_payload() -> 'Dict[str, Any]':
    """Numerically relevant interpreter/platform/library configuration."""

    import numpy as np
    try:
        import scipy
    except Exception:  # Production solvers reject a missing required backend.
        scipy = None
    implementation = getattr(sys, "implementation", None)
    return {
        "python_version": sys.version,
        "python_implementation": getattr(implementation, "name", ""),
        "python_cache_tag": getattr(implementation, "cache_tag", ""),
        "byteorder": sys.byteorder,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "numpy_version": np.__version__,
        "numpy_config": _package_configuration(np),
        "scipy_version": getattr(scipy, "__version__", "unavailable"),
        "scipy_config": (
            _package_configuration(scipy) if scipy is not None else {}
        ),
    }


def runtime_environment_fingerprint() -> 'str':
    raw = json.dumps(
        runtime_environment_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
