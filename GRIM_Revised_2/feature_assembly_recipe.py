"""Portable Assembly recipe serialization and atomic file publication.

Form values and preflight policies are resolved lazily from the Qt-free model.
"""
from __future__ import annotations

from feature_assembly_values import FeatureAssemblyValues, LoadedFeatureAssemblyRecipe

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def _recipe_target_path(value: str | Path) -> Path:
    from feature_assembly_model import (
        FEATURE_RECIPE_SUFFIX,
        _clean_path,
    )
    raw = _clean_path(value)
    if not raw:
        raise ValueError("Choose where to save the Assembly recipe.")
    target = Path(raw).expanduser()
    if target.suffix.casefold() != ".json":
        target = Path(str(target) + FEATURE_RECIPE_SUFFIX)
    return target.resolve()

def _recipe_relative_path(
    value: Any,
    *,
    source_base_dir: Any,
    recipe_dir: Path,
) -> str:
    """Store one effective path relative to the recipe when possible."""
    from feature_assembly_model import (
        _clean_path,
        _resolved_user_path,
    )

    if not _clean_path(value):
        return ""
    resolved = _resolved_user_path(value, base_dir=source_base_dir)
    try:
        relative = os.path.relpath(str(resolved), str(recipe_dir))
    except ValueError:  # Different Windows drives cannot form a relative path.
        return str(resolved)
    return Path(relative).as_posix()

def _recipe_absolute_path(value: Any, *, recipe_dir: Path) -> str:
    from feature_assembly_model import (
        _clean_path,
    )
    raw = _clean_path(value)
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = recipe_dir / path
    return str(path.resolve())

def _recipe_source_items(
    values: FeatureAssemblyValues,
) -> tuple[tuple[str, str | None, str], ...]:
    from feature_assembly_model import (
        _clean_path,
    )
    items: list[tuple[str, str | None, str]] = [
        ("base_grim", None, _clean_path(values.base_grim)),
        ("surface_mesh", None, _clean_path(values.surface_mesh)),
        ("point_locations_csv", None, _clean_path(values.point_locations_csv)),
        ("line_locations_csv", None, _clean_path(values.line_locations_csv)),
    ]
    items.extend(
        ("point_dataset", str(dataset_id), _clean_path(path))
        for dataset_id, path in sorted(values.point_datasets.items())
    )
    items.extend(
        ("line_dataset", str(dataset_id), _clean_path(path))
        for dataset_id, path in sorted(values.line_datasets.items())
    )
    return tuple(item for item in items if item[2])

def feature_assembly_recipe_payload(
    values: FeatureAssemblyValues,
    *,
    recipe_path: str | Path,
    name: str,
    variant: str,
) -> dict[str, Any]:
    """Return a portable, versioned recipe with lightweight source identity."""
    from feature_assembly_model import (
        FEATURE_RECIPE_HASH_LIMIT_BYTES,
        FEATURE_RECIPE_SCHEMA,
        FEATURE_RECIPE_VERSION,
        _FileFingerprint,
        _fingerprint_file,
        _path_key,
        _resolved_user_path,
    )

    if not isinstance(values, FeatureAssemblyValues):
        raise TypeError("values must be FeatureAssemblyValues")
    target = _recipe_target_path(recipe_path)
    recipe_dir = target.parent
    clean_name = str(name).strip()
    clean_variant = str(variant).strip()
    if not clean_name:
        raise ValueError("Enter an Assembly recipe name.")
    if not clean_variant:
        raise ValueError("Enter a variant name, such as Baseline or Option A.")

    def relative(path: Any) -> str:
        return _recipe_relative_path(
            path,
            source_base_dir=values.base_dir,
            recipe_dir=recipe_dir,
        )

    serialized_values: dict[str, Any] = {
        **{key: getattr(values, key) for key in ("study_frequencies_ghz", "study_azimuths_deg", "study_elevations_deg")},
        "host_material": values.host_material,
        "host_stack_id": values.host_stack_id,
        "host_minimum_radius_m": values.host_minimum_radius_m,
        "base_grim": relative(values.base_grim),
        "output_grim": relative(values.output_grim),
        "coordinate_units": str(values.coordinate_units),
        "surface_mesh": relative(values.surface_mesh),
        "surface_units": str(values.surface_units),
        "flip_surface_normals": bool(values.flip_surface_normals),
        "shadow": bool(values.shadow),
        "shadow_bias_m": (
            None
            if values.shadow_bias_m is None
            else float(values.shadow_bias_m)
        ),
        "point_locations_csv": relative(values.point_locations_csv),
        "line_locations_csv": relative(values.line_locations_csv),
        "skin_tol_m": float(values.skin_tol_m),
        "skin_phase_tol_deg": float(values.skin_phase_tol_deg),
        "normal_tol_deg": float(values.normal_tol_deg),
        "allow_legacy_base_metadata": bool(values.allow_legacy_base_metadata),
        "require_feature_manifests": bool(values.require_feature_manifests),
        "require_body_mesh_certification": bool(
            values.require_body_mesh_certification
        ),
        # Every effective path above is rebased to this recipe directory.
        "base_dir": ".",
        "point_datasets": {
            str(dataset_id): relative(path)
            for dataset_id, path in sorted(values.point_datasets.items())
        },
        "line_datasets": {
            str(dataset_id): relative(path)
            for dataset_id, path in sorted(values.line_datasets.items())
        },
        "excluded_point_placement_ids": sorted(
            str(value) for value in values.excluded_point_placement_ids
        ),
        "excluded_line_ids": sorted(
            str(value) for value in values.excluded_line_ids
        ),
    }

    manifest: list[dict[str, Any]] = []
    for role, dataset_id, path in _recipe_source_items(values):
        resolved = _resolved_user_path(path, base_dir=values.base_dir)
        try:
            size = int(resolved.stat().st_size) if resolved.is_file() else None
            include_hash = bool(
                size is not None and size <= FEATURE_RECIPE_HASH_LIMIT_BYTES
            )
            fingerprint = _fingerprint_file(
                path,
                base_dir=values.base_dir,
                include_hash=include_hash,
            )
        except OSError:
            fingerprint = _FileFingerprint(
                resolved_path=_path_key(resolved), exists=False
            )
        record: dict[str, Any] = {
            "role": role,
            "path": relative(path),
            "exists": fingerprint.exists,
            "size": fingerprint.size,
            "mtime_ns": fingerprint.mtime_ns,
        }
        if dataset_id is not None:
            record["dataset_id"] = dataset_id
        if fingerprint.sha256 is not None:
            record["sha256"] = fingerprint.sha256
        manifest.append(record)

    return {
        "schema": FEATURE_RECIPE_SCHEMA,
        "version": FEATURE_RECIPE_VERSION,
        "name": clean_name,
        "variant": clean_variant,
        "path_policy": "relative-to-recipe",
        "values": serialized_values,
        "source_manifest": manifest,
    }

def write_feature_assembly_recipe(
    values: FeatureAssemblyValues,
    path: str | Path,
    *,
    name: str,
    variant: str,
) -> Path:
    """Atomically save one portable Assembly recipe."""

    target = _recipe_target_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"Assembly recipe folder does not exist: {target.parent}"
        )
    payload = feature_assembly_recipe_payload(
        values,
        recipe_path=target,
        name=name,
        variant=variant,
    )
    serialized = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return target

def _recipe_string_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Assembly recipe {label} must be an object.")
    result: dict[str, str] = {}
    for key, path in value.items():
        dataset_id = str(key).strip()
        if not dataset_id or not isinstance(path, str):
            raise ValueError(
                f"Assembly recipe {label} must map nonempty IDs to paths."
            )
        result[dataset_id] = path
    return result

def _recipe_id_set(value: Any, label: str) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"Assembly recipe {label} must be a list.")
    result = {str(item).strip() for item in value}
    if "" in result or len(result) != len(value):
        raise ValueError(
            f"Assembly recipe {label} contains a blank or duplicate ID."
        )
    return result

def read_feature_assembly_recipe(
    path: str | Path,
) -> LoadedFeatureAssemblyRecipe:
    """Load one recipe and report missing or changed referenced inputs."""
    from feature_assembly_model import (
        FEATURE_RECIPE_SCHEMA,
        FEATURE_RECIPE_VERSION,
        UNIT_CHOICES,
        _clean_path,
        _fingerprint_file,
        _require_finite_nonnegative,
        parse_study_samples,
    )

    source = Path(_clean_path(path)).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Assembly recipe is not valid JSON ({exc.msg} at line {exc.lineno})."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Assembly recipe root must be a JSON object.")
    if payload.get("schema") != FEATURE_RECIPE_SCHEMA:
        raise ValueError(
            f"Not a {FEATURE_RECIPE_SCHEMA!r} Assembly recipe."
        )
    version = payload.get("version")
    if type(version) is not int or version != FEATURE_RECIPE_VERSION:
        raise ValueError(
            f"Unsupported Assembly recipe version {version!r}; this GRIM build "
            f"requires version {FEATURE_RECIPE_VERSION}."
        )
    name = payload.get("name")
    variant = payload.get("variant")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Assembly recipe name must be a nonempty string.")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("Assembly recipe variant must be a nonempty string.")
    raw_values = payload.get("values")
    if not isinstance(raw_values, Mapping):
        raise ValueError("Assembly recipe values must be a JSON object.")
    for key in ("host_material", "host_stack_id"):
        if not isinstance(raw_values.get(key, ""), str):
            raise ValueError(f"Recipe {key} must be a string.")

    required = {
        "base_grim",
        "output_grim",
        "coordinate_units",
        "surface_mesh",
        "surface_units",
        "flip_surface_normals",
        "shadow",
        "shadow_bias_m",
        "point_locations_csv",
        "line_locations_csv",
        "skin_tol_m",
        "skin_phase_tol_deg",
        "normal_tol_deg",
        "allow_legacy_base_metadata",
        "require_feature_manifests",
        "require_body_mesh_certification",
        "base_dir",
        "point_datasets",
        "line_datasets",
        "excluded_point_placement_ids",
        "excluded_line_ids",
    }
    missing = sorted(required - set(raw_values))
    if missing:
        raise ValueError(
            "Assembly recipe is missing value(s): " + ", ".join(missing)
        )
    for key in (
        "base_grim",
        "output_grim",
        "surface_mesh",
        "point_locations_csv",
        "line_locations_csv",
    ):
        if not isinstance(raw_values[key], str):
            raise ValueError(f"Assembly recipe {key} must be a path string.")
    if any(
        not isinstance(raw_values[key], bool)
        for key in (
            "flip_surface_normals",
            "shadow",
            "allow_legacy_base_metadata",
            "require_feature_manifests",
            "require_body_mesh_certification",
        )
    ):
        raise ValueError("Assembly recipe boolean settings must be true or false.")
    coordinate_units = str(raw_values["coordinate_units"])
    surface_units = str(raw_values["surface_units"])
    supported_units = {value for _label, value in UNIT_CHOICES}
    if coordinate_units and coordinate_units not in supported_units:
        raise ValueError("Assembly recipe contains unsupported coordinate units.")
    if surface_units and surface_units not in supported_units:
        raise ValueError("Assembly recipe contains unsupported surface units.")
    skin_tol = _require_finite_nonnegative(
        raw_values["skin_tol_m"], "Recipe skin distance tolerance"
    )
    phase_tol = _require_finite_nonnegative(
        raw_values["skin_phase_tol_deg"], "Recipe skin phase tolerance"
    )
    normal_tol = _require_finite_nonnegative(
        raw_values["normal_tol_deg"], "Recipe normal tolerance"
    )
    if skin_tol > 0.1:
        raise ValueError(
            "Assembly recipe skin distance tolerance must not exceed 100 mm."
        )
    if not 0.0 < phase_tol <= 90.0:
        raise ValueError(
            "Assembly recipe skin phase tolerance must be above 0 and at most "
            "90 degrees."
        )
    if normal_tol >= 90.0:
        raise ValueError("Assembly recipe normal tolerance must be below 90 degrees.")
    shadow_bias_raw = raw_values["shadow_bias_m"]
    shadow_bias = (
        None
        if shadow_bias_raw is None
        else _require_finite_nonnegative(shadow_bias_raw, "Recipe shadow bias")
    )

    point_paths = _recipe_string_mapping(
        raw_values["point_datasets"], "point_datasets"
    )
    line_paths = _recipe_string_mapping(
        raw_values["line_datasets"], "line_datasets"
    )
    recipe_dir = source.parent
    values = FeatureAssemblyValues(
        base_grim=_recipe_absolute_path(raw_values["base_grim"], recipe_dir=recipe_dir),
        output_grim=_recipe_absolute_path(raw_values["output_grim"], recipe_dir=recipe_dir),
        coordinate_units=coordinate_units,
        surface_mesh=_recipe_absolute_path(raw_values["surface_mesh"], recipe_dir=recipe_dir),
        surface_units=surface_units,
        flip_surface_normals=raw_values["flip_surface_normals"],
        shadow=raw_values["shadow"],
        shadow_bias_m=shadow_bias,
        point_locations_csv=_recipe_absolute_path(
            raw_values["point_locations_csv"], recipe_dir=recipe_dir
        ),
        line_locations_csv=_recipe_absolute_path(
            raw_values["line_locations_csv"], recipe_dir=recipe_dir
        ),
        skin_tol_m=skin_tol,
        skin_phase_tol_deg=phase_tol,
        normal_tol_deg=normal_tol,
        allow_legacy_base_metadata=raw_values["allow_legacy_base_metadata"],
        **{key: parse_study_samples(raw_values.get(key)) for key in ("study_frequencies_ghz", "study_azimuths_deg", "study_elevations_deg")},
        host_material=raw_values.get("host_material", ""),
        host_stack_id=raw_values.get("host_stack_id", ""),
        host_minimum_radius_m=(None if raw_values.get("host_minimum_radius_m") is None else _require_finite_nonnegative(raw_values["host_minimum_radius_m"], "Host minimum radius")),
        require_feature_manifests=raw_values["require_feature_manifests"],
        require_body_mesh_certification=raw_values["require_body_mesh_certification"],
        base_dir=None,
        point_datasets={
            dataset_id: _recipe_absolute_path(value, recipe_dir=recipe_dir)
            for dataset_id, value in point_paths.items()
        },
        line_datasets={
            dataset_id: _recipe_absolute_path(value, recipe_dir=recipe_dir)
            for dataset_id, value in line_paths.items()
        },
        excluded_point_placement_ids=_recipe_id_set(
            raw_values["excluded_point_placement_ids"],
            "excluded_point_placement_ids",
        ),
        excluded_line_ids=_recipe_id_set(
            raw_values["excluded_line_ids"], "excluded_line_ids"
        ),
    )

    current_sources = {
        (role, dataset_id): path_value
        for role, dataset_id, path_value in _recipe_source_items(values)
    }
    warnings: list[str] = []
    raw_manifest = payload.get("source_manifest", [])
    if not isinstance(raw_manifest, list):
        raise ValueError("Assembly recipe source_manifest must be a list.")
    seen_manifest_keys: set[tuple[str, str | None]] = set()
    for index, record in enumerate(raw_manifest):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Assembly recipe source_manifest entry {index} must be an object."
            )
        role = str(record.get("role", "")).strip()
        dataset_raw = record.get("dataset_id")
        dataset_id = None if dataset_raw is None else str(dataset_raw).strip()
        key = (role, dataset_id)
        if not role or key in seen_manifest_keys:
            raise ValueError(
                "Assembly recipe source_manifest contains a blank or duplicate role."
            )
        seen_manifest_keys.add(key)
        current_path = current_sources.get(key)
        if not current_path:
            warnings.append(f"{role}: referenced source is no longer configured")
            continue
        display = role if dataset_id is None else f"{role} {dataset_id!r}"
        saved_exists = record.get("exists")
        if not isinstance(saved_exists, bool):
            raise ValueError(
                f"Assembly recipe source_manifest {display} has invalid exists state."
            )
        try:
            current = _fingerprint_file(
                current_path,
                include_hash=isinstance(record.get("sha256"), str),
            )
        except OSError as exc:
            warnings.append(f"{display}: could not verify source ({exc})")
            continue
        if not current.exists:
            warnings.append(f"{display}: file is missing")
            continue
        if not saved_exists:
            warnings.append(f"{display}: file was missing when this recipe was saved")
            continue
        saved_hash = record.get("sha256")
        if isinstance(saved_hash, str):
            if current.sha256 != saved_hash:
                warnings.append(f"{display}: file content changed since recipe save")
            continue
        saved_size = record.get("size")
        if isinstance(saved_size, int) and current.size != saved_size:
            warnings.append(f"{display}: file size changed since recipe save")
        elif (
            isinstance(record.get("mtime_ns"), int)
            and current.mtime_ns != record["mtime_ns"]
        ):
            warnings.append(
                f"{display}: timestamp changed; large-file content was not hashed"
            )

    return LoadedFeatureAssemblyRecipe(
        path=source,
        name=name.strip(),
        variant=variant.strip(),
        values=values,
        source_warnings=tuple(warnings),
    )
