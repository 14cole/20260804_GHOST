"""Durable, non-destructive ISAR formation result files.

An ``.isar.npz`` artifact is deliberately separate from :class:`RcsGrid`:
its axes are image-space distance, not azimuth/elevation/frequency/polarization.
The archive contains numeric arrays plus a versioned JSON manifest binding the
result to its selected source samples and complete formation recipe.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import zipfile
import zlib

import numpy as np
from numpy.lib import format as npformat


ISAR_ARTIFACT_SCHEMA = "grim.isar-result.v2"
_SUPPORTED_ISAR_ARTIFACT_SCHEMAS = {
    "grim.isar-result.v1",
    ISAR_ARTIFACT_SCHEMA,
}
_COMPRESSION_SAMPLE_BYTES = 2 * 1024**2
_MAX_ARTIFACT_BANDS = 128
_MAX_ARTIFACT_ARRAY_CELLS = 250_000_000
_MAX_ARTIFACT_NUMERIC_BYTES = 2 * 1024**3
_MAX_ARTIFACT_MANIFEST_BYTES = 64 * 1024**2
_MAX_NPY_HEADER_BYTES = 64 * 1024
_MAX_ARCHIVE_MEMBERS = 1 + 4 * _MAX_ARTIFACT_BANDS

_MAX_METADATA_DEPTH = 8
_MAX_METADATA_COLLECTION_ITEMS = 256
_MAX_METADATA_TOTAL_ITEMS = 2_048
_MAX_METADATA_STRING_CHARS = 32_768
_MAX_METADATA_TOTAL_STRING_CHARS = 131_072
_MAX_METADATA_JSON_BYTES = 512 * 1024
_OMIT_METADATA = object()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, complex):
        return {
            "__complex__": [
                _json_safe(float(value.real)),
                _json_safe(float(value.imag)),
            ]
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _bounded_metadata_value(value: Any, state: dict[str, int], depth: int = 0):
    """Return JSON-safe small metadata or a sentinel when a bound is exceeded."""

    if depth > _MAX_METADATA_DEPTH:
        return _OMIT_METADATA
    if state["items"] >= _MAX_METADATA_TOTAL_ITEMS:
        return _OMIT_METADATA
    state["items"] += 1

    if isinstance(value, np.generic):
        return _bounded_metadata_value(value.item(), state, depth)
    if isinstance(value, np.ndarray):
        if value.size > _MAX_METADATA_COLLECTION_ITEMS:
            return _OMIT_METADATA
        return _bounded_metadata_value(value.tolist(), state, depth + 1)
    if isinstance(value, complex):
        return {
            "__complex__": [float(value.real), float(value.imag)],
        }
    if isinstance(value, dict):
        if len(value) > _MAX_METADATA_COLLECTION_ITEMS:
            return _OMIT_METADATA
        result = {}
        for key, item in value.items():
            key_text = str(key)
            if len(key_text) > _MAX_METADATA_STRING_CHARS:
                return _OMIT_METADATA
            if (
                state["string_chars"] + len(key_text)
                > _MAX_METADATA_TOTAL_STRING_CHARS
            ):
                return _OMIT_METADATA
            state["string_chars"] += len(key_text)
            converted = _bounded_metadata_value(item, state, depth + 1)
            if converted is _OMIT_METADATA:
                return _OMIT_METADATA
            result[key_text] = converted
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_METADATA_COLLECTION_ITEMS:
            return _OMIT_METADATA
        result = []
        for item in value:
            converted = _bounded_metadata_value(item, state, depth + 1)
            if converted is _OMIT_METADATA:
                return _OMIT_METADATA
            result.append(converted)
        return result
    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, (bytes, bytearray, memoryview)):
        if len(value) > _MAX_METADATA_STRING_CHARS:
            return _OMIT_METADATA
        text = repr(bytes(value))
    else:
        text = value if isinstance(value, str) else str(value)

    if len(text) > _MAX_METADATA_STRING_CHARS:
        return _OMIT_METADATA
    if (
        state["string_chars"] + len(text)
        > _MAX_METADATA_TOTAL_STRING_CHARS
    ):
        return _OMIT_METADATA
    state["string_chars"] += len(text)
    return text


def _small_metadata(container: dict | None) -> dict[str, Any]:
    """Keep bounded scalar/list metadata without copying grid-sized payloads."""

    result: dict[str, Any] = {}
    state = {"items": 0, "string_chars": 0}
    for index, (key, value) in enumerate((container or {}).items()):
        if index >= _MAX_METADATA_COLLECTION_ITEMS:
            break
        if state["items"] >= _MAX_METADATA_TOTAL_ITEMS:
            break
        key_text = str(key)
        if len(key_text) > _MAX_METADATA_STRING_CHARS:
            continue
        trial = dict(state)
        if (
            trial["string_chars"] + len(key_text)
            > _MAX_METADATA_TOTAL_STRING_CHARS
        ):
            continue
        trial["string_chars"] += len(key_text)
        converted = _bounded_metadata_value(value, trial)
        if converted is _OMIT_METADATA:
            continue
        candidate = dict(result)
        candidate[key_text] = converted
        encoded = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > _MAX_METADATA_JSON_BYTES:
            continue
        result = candidate
        state = trial
    return result


def _numerical_magnitude(band: dict) -> np.ndarray:
    """Return full-resolution magnitude, independent of GUI decimation."""

    if "complex_image" in band:
        return np.abs(np.asarray(band["complex_image"])).astype(np.float32)
    raw = np.asarray(band["magnitude"])
    if np.iscomplexobj(raw):
        raise ValueError("ISAR artifact magnitude must not be complex")
    return raw.astype(np.float32, copy=False)


def _numeric_array(name: str, value: Any, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"ISAR artifact {name} must be {ndim}-D")
    if array.dtype.kind not in "fciu":
        raise ValueError(f"ISAR artifact {name} must be numeric")
    if array.size == 0:
        raise ValueError(f"ISAR artifact {name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"ISAR artifact {name} contains non-finite values")
    return array


def _validated_axis(name: str, value: Any) -> np.ndarray:
    raw_axis = _numeric_array(name, value, ndim=1)
    if np.iscomplexobj(raw_axis):
        raise ValueError(f"ISAR artifact {name} must be real")
    with np.errstate(over="ignore", invalid="ignore"):
        axis = raw_axis.astype(np.float64, copy=False)
    if not np.all(np.isfinite(axis)):
        raise ValueError(
            f"ISAR artifact {name} cannot be represented as finite float64 values"
        )
    if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"ISAR artifact {name} must be strictly increasing")
    return axis


def _manifest_band_list(manifest: dict, expected_count: int) -> list[dict]:
    if not isinstance(manifest, dict):
        raise ValueError("ISAR artifact manifest must be a JSON object")
    schema = str(manifest.get("schema", ""))
    if schema not in _SUPPORTED_ISAR_ARTIFACT_SCHEMAS:
        raise ValueError("ISAR artifact manifest has an unsupported schema")
    try:
        json.dumps(_json_safe(manifest), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("ISAR artifact manifest is not finite valid JSON") from exc
    band_manifests = manifest.get("bands")
    if not isinstance(band_manifests, list) or not band_manifests:
        raise ValueError("ISAR artifact manifest needs at least one band")
    if len(band_manifests) != expected_count:
        raise ValueError(
            "ISAR artifact manifest band count does not match stored arrays"
        )
    if not all(isinstance(item, dict) for item in band_manifests):
        raise ValueError("ISAR artifact band manifests must be JSON objects")
    if len(band_manifests) > _MAX_ARTIFACT_BANDS:
        raise ValueError(
            f"ISAR artifact has {len(band_manifests):,} bands; the safety limit "
            f"is {_MAX_ARTIFACT_BANDS:,}"
        )
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("ISAR artifact manifest is missing source metadata")
    digests = source.get("band_content_digests")
    if not isinstance(digests, list) or len(digests) != expected_count:
        raise ValueError(
            "ISAR artifact source digest count does not match stored bands"
        )
    for index, (digest, band_manifest) in enumerate(zip(digests, band_manifests)):
        declared = str(band_manifest.get("source_selection_digest", "")).strip()
        if not declared or str(digest).strip() != declared:
            raise ValueError(
                f"ISAR artifact band {index} source digest disagrees with manifest"
            )
        if schema == ISAR_ARTIFACT_SCHEMA:
            has_complex = bool(
                band_manifest.get("complex_image_available", False)
            )
            expected_storage = "derived_from_complex" if has_complex else "stored"
            if band_manifest.get("magnitude_storage") != expected_storage:
                raise ValueError(
                    f"ISAR artifact band {index} magnitude storage disagrees "
                    "with complex-image availability"
                )
    return band_manifests


def _validated_band_payload(
    index: int,
    band: dict,
    band_manifest: dict,
    *,
    derive_magnitude: bool = True,
    validate_redundant_magnitude: bool = False,
) -> dict[str, np.ndarray]:
    x_axis = _validated_axis(f"band {index} x_range", band["x_range"])
    y_axis = _validated_axis(f"band {index} y_range", band["y_range"])
    expected_shape = (x_axis.size, y_axis.size)
    if list(band_manifest.get("shape", [])) != list(expected_shape):
        raise ValueError(
            f"ISAR artifact band {index} shape disagrees with manifest"
        )
    for key, axis in (("x_extent", x_axis), ("y_extent", y_axis)):
        extent = np.asarray(band_manifest.get(key, []), dtype=float)
        expected_extent = np.asarray([axis[0], axis[-1]], dtype=float)
        if extent.shape != (2,) or not np.all(np.isfinite(extent)) or not np.allclose(
            extent, expected_extent, rtol=1.0e-12, atol=1.0e-12
        ):
            raise ValueError(
                f"ISAR artifact band {index} {key} disagrees with stored axis"
            )
    payload: dict[str, np.ndarray] = {"x_range": x_axis, "y_range": y_axis}
    has_complex = "complex_image" in band
    if bool(band_manifest.get("complex_image_available", False)) != has_complex:
        raise ValueError(
            f"ISAR artifact band {index} complex-image availability disagrees "
            "with manifest"
        )
    if has_complex:
        raw_complex = _numeric_array(
            f"band {index} complex_image", band["complex_image"], ndim=2
        )
        with np.errstate(over="ignore", invalid="ignore"):
            complex_image = raw_complex.astype(np.complex64, copy=False)
        if not np.all(np.isfinite(complex_image)):
            raise ValueError(
                f"ISAR artifact band {index} complex image cannot be represented "
                "as finite complex64 values"
            )
        if complex_image.shape != expected_shape:
            raise ValueError(
                f"ISAR artifact band {index} complex image does not match axes"
            )
        if validate_redundant_magnitude:
            if "magnitude" not in band:
                raise ValueError(
                    f"ISAR artifact band {index} is missing its legacy magnitude"
                )
            raw_magnitude = _numeric_array(
                f"band {index} magnitude", band["magnitude"], ndim=2
            )
            if np.iscomplexobj(raw_magnitude):
                raise ValueError(
                    f"ISAR artifact band {index} magnitude must be real and nonnegative"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                legacy_magnitude = raw_magnitude.astype(np.float32, copy=False)
            if not np.all(np.isfinite(legacy_magnitude)):
                raise ValueError(
                    f"ISAR artifact band {index} magnitude cannot be represented "
                    "as finite float32 values"
                )
            if legacy_magnitude.shape != expected_shape:
                raise ValueError(
                    f"ISAR artifact band {index} magnitude does not match axes"
                )
            if np.any(legacy_magnitude < 0.0):
                raise ValueError(
                    f"ISAR artifact band {index} magnitude must be real and nonnegative"
                )
            expected_magnitude = np.abs(complex_image).astype(
                np.float32, copy=False
            )
            if not np.allclose(
                legacy_magnitude,
                expected_magnitude,
                rtol=2.0e-5,
                atol=np.finfo(np.float32).tiny * 8.0,
            ):
                raise ValueError(
                    f"ISAR artifact band {index} legacy magnitude disagrees "
                    "with its complex image"
                )
        payload["complex_image"] = complex_image
        if derive_magnitude:
            payload["magnitude"] = np.abs(complex_image).astype(
                np.float32, copy=False
            )
    else:
        if "magnitude" not in band:
            raise ValueError(f"ISAR artifact band {index} is missing magnitude")
        raw_magnitude = _numeric_array(
            f"band {index} magnitude", band["magnitude"], ndim=2
        )
        if np.iscomplexobj(raw_magnitude):
            raise ValueError(
                f"ISAR artifact band {index} magnitude must be real and nonnegative"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = raw_magnitude.astype(np.float32, copy=False)
        if not np.all(np.isfinite(magnitude)):
            raise ValueError(
                f"ISAR artifact band {index} magnitude cannot be represented "
                "as finite float32 values"
            )
        if magnitude.shape != expected_shape:
            raise ValueError(
                f"ISAR artifact band {index} image shape {magnitude.shape} does "
                f"not match axes {expected_shape}"
            )
        if np.any(magnitude < 0.0):
            raise ValueError(
                f"ISAR artifact band {index} magnitude must be real and nonnegative"
            )
        payload["magnitude"] = magnitude
    source_digest = str(band.get("source_selection_digest", "")).strip()
    manifest_digest = str(band_manifest["source_selection_digest"]).strip()
    if source_digest and source_digest != manifest_digest:
        raise ValueError(
            f"ISAR artifact band {index} source digest disagrees with manifest"
        )
    return payload


def build_isar_manifest(
    dataset,
    params: dict,
    band_results: Iterable[dict],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the reproducibility manifest for one completed formation."""

    bands = list(band_results)
    azimuth_indices = sorted(
        {int(index) for band in params["bands"] for index in band}
    )
    frequency_indices = [int(index) for index in params["freq_indices_sorted"]]
    target = params.get("az_target_deg")
    selected_azimuth_native = np.asarray(
        dataset.azimuths, dtype=float
    )[azimuth_indices]
    selected_frequency_native = np.asarray(
        dataset.frequencies, dtype=float
    )[frequency_indices]
    selected_elevation_native = float(
        np.asarray(dataset.elevations, dtype=float)[int(params["elev_idx"])]
    )
    selected_polarization = str(
        np.asarray(dataset.polarizations)[int(params["pol_idx"])]
    )
    source_digests = [
        str(band.get("source_selection_digest", "")) for band in bands
    ]
    if any(not digest for digest in source_digests):
        raise ValueError(
            "ISAR result is missing its selected-source digest; recompute before export"
        )

    diagnostics = []
    for index, band in enumerate(bands):
        x_axis = np.asarray(band["x_range"], dtype=float)
        y_axis = np.asarray(band["y_range"], dtype=float)
        has_complex = "complex_image" in band
        image_shape = np.asarray(
            band["complex_image"] if has_complex else band["magnitude"]
        ).shape
        diagnostics.append(
            {
                "band": index,
                "source_selection_digest": source_digests[index],
                "azimuth_values_degrees": np.asarray(
                    band.get("az_values", []), dtype=float
                ).tolist(),
                "shape": list(image_shape),
                "x_extent": [float(x_axis[0]), float(x_axis[-1])],
                "y_extent": [float(y_axis[0]), float(y_axis[-1])],
                "phase_coverage": float(band.get("phase_coverage", 1.0)),
                "sampling": _json_safe(band.get("sampling", {})),
                "azimuth_spacing_spread": float(
                    band.get("az_nonuniformity", 0.0)
                ),
                "frequency_spacing_spread": float(
                    band.get("freq_nonuniformity", 0.0)
                ),
                "composite_looks": int(band.get("composite", 0)),
                "sparse": _json_safe(band.get("sparse_diagnostics", {})),
                "complex_image_available": has_complex,
                "magnitude_storage": (
                    "derived_from_complex" if has_complex else "stored"
                ),
            }
        )

    manifest = {
        "schema": ISAR_ARTIFACT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"application": "GRIM", "isar_engine_schema": 1},
        "source": {
            "path": str(getattr(dataset, "source_path", "") or ""),
            "history": str(getattr(dataset, "history", "") or ""),
            "shape": list(np.asarray(dataset.rcs_power).shape),
            "units": _small_metadata(getattr(dataset, "units", {})),
            "metadata": _small_metadata(getattr(dataset, "extra", {})),
            "selected_azimuth_indices": azimuth_indices,
            "selected_azimuth_values_native": selected_azimuth_native.tolist(),
            "selected_frequency_indices": frequency_indices,
            "selected_frequency_values_native": selected_frequency_native.tolist(),
            "selected_frequency_values_hz": np.asarray(
                params["freq_hz"], dtype=float
            ).tolist(),
            "elevation_index": int(params["elev_idx"]),
            "selected_elevation_value_native": selected_elevation_native,
            "selected_elevation_degrees": float(params["elevation_deg"]),
            "polarization_index": int(params["pol_idx"]),
            "selected_polarization": selected_polarization,
            "band_content_digests": source_digests,
        },
        "formation": {
            "reconstruction": str(params["recon"]),
            "window": str(params["window_name"]),
            "length_unit": str(params["unit_name"]),
            "azimuth_target_degrees": (
                None if target is None else np.asarray(target, dtype=float).tolist()
            ),
            "aperture_center_degrees": params.get("az_center_deg"),
            "l1_strength": float(params["l1_strength"]),
            "l1_iterations": int(params["l1_iters"]),
            "flip_x": bool(params["flip_x"]),
            "flip_y": bool(params["flip_y"]),
            "legacy_metadata_attested": bool(
                params.get("legacy_metadata_attested", False)
            ),
            "elapsed_seconds": float(elapsed_seconds),
        },
        "bands": diagnostics,
    }
    safe_manifest = _json_safe(manifest)
    band_manifests = _manifest_band_list(safe_manifest, len(bands))
    for index, (band, band_manifest) in enumerate(zip(bands, band_manifests)):
        _validated_band_payload(
            index, band, band_manifest, derive_magnitude=False
        )
    return safe_manifest


def _representative_bytes(array: np.ndarray, budget: int) -> bytes:
    value = np.asarray(array)
    if budget <= 0 or value.nbytes == 0:
        return b""
    if value.flags.c_contiguous:
        raw = memoryview(value).cast("B")
        if len(raw) <= budget:
            return bytes(raw)
        block = max(1, budget // 3)
        middle = max(0, (len(raw) - block) // 2)
        end = max(0, len(raw) - block)
        sample = bytes(raw[:block]) + bytes(raw[middle : middle + block])
        remaining = budget - len(sample)
        if remaining > 0:
            sample += bytes(raw[end : end + remaining])
        return sample[:budget]

    # Axis flips intentionally return negative-stride views. Sampling their
    # flat iterator copies only the bounded selected elements; making the
    # entire image contiguous here would defeat the 2 MiB decision budget.
    element_bytes = max(int(value.dtype.itemsize), 1)
    sample_count = min(
        int(value.size),
        max(1, int(budget) // element_bytes),
    )
    if sample_count >= value.size:
        selected = np.ascontiguousarray(value)
    else:
        indices = np.linspace(
            0,
            int(value.size) - 1,
            sample_count,
            dtype=np.intp,
        )
        selected = np.ascontiguousarray(value.flat[indices])
    return selected.view(np.uint8).tobytes()[:budget]


def _compression_decision(arrays: Iterable[np.ndarray]) -> dict[str, Any]:
    values = [np.asarray(value) for value in arrays]
    payload_bytes = sum(int(value.nbytes) for value in values)
    candidates = [value for value in values if value.nbytes]
    per_array = max(1, _COMPRESSION_SAMPLE_BYTES // max(1, len(candidates)))
    remaining = _COMPRESSION_SAMPLE_BYTES
    samples = []
    for value in candidates:
        if remaining <= 0:
            break
        sample = _representative_bytes(value, min(per_array, remaining))
        samples.append(sample)
        remaining -= len(sample)
    combined = b"".join(samples)
    ratio = (
        len(zlib.compress(combined, level=1)) / len(combined)
        if combined else 1.0
    )
    savings = max(0.0, 1.0 - float(ratio))
    minimum = 0.10
    return {
        "compressed": bool(savings >= minimum),
        "estimated_savings_fraction": savings,
        "sample_bytes": len(combined),
        "payload_bytes": payload_bytes,
        "minimum_savings_fraction": minimum,
    }


def _validate_numerical_payload_budget(
    arrays: Iterable[np.ndarray],
) -> tuple[int, int]:
    """Bound numerical artifact allocation before save or decompression."""

    total_cells = 0
    total_bytes = 0
    for value in arrays:
        array = np.asarray(value)
        cells = int(array.size)
        payload_bytes = int(array.nbytes)
        if cells > _MAX_ARTIFACT_ARRAY_CELLS:
            raise ValueError(
                f"ISAR artifact array has {cells:,} cells; the safety limit is "
                f"{_MAX_ARTIFACT_ARRAY_CELLS:,}"
            )
        total_cells += cells
        total_bytes += payload_bytes
        if total_bytes > _MAX_ARTIFACT_NUMERIC_BYTES:
            raise ValueError(
                "ISAR artifact numerical payload exceeds the "
                f"{_MAX_ARTIFACT_NUMERIC_BYTES / 1024**3:g} GiB safety limit"
            )
    return total_cells, total_bytes


def _normalized_array_bytes(
    key: str,
    dtype: np.dtype,
    cells: int,
    payload_bytes: int,
) -> int:
    """Estimate resident bytes after loader normalization, not just file dtype."""

    if str(key).endswith("_complex_image"):
        target_dtype = np.dtype(np.complex64)
    elif str(key).endswith("_magnitude"):
        target_dtype = np.dtype(np.float32)
    elif str(key).endswith(("_x_range", "_y_range")):
        target_dtype = np.dtype(np.float64)
    else:
        # Unknown members are rejected after manifest parsing. Until then use
        # the largest normal artifact scalar so a tiny on-disk dtype cannot
        # evade the pre-allocation working-set budget.
        target_dtype = np.dtype(np.complex64)
    target_bytes = int(cells) * int(target_dtype.itemsize)
    if np.dtype(dtype) == target_dtype:
        return int(payload_bytes)
    # During dtype normalization the source and destination coexist briefly.
    return int(payload_bytes) + target_bytes


def _read_npy_member_header(stream, member_name: str):
    try:
        version = npformat.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = npformat.read_array_header_1_0(
                stream,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = npformat.read_array_header_2_0(
                stream,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        else:
            raise ValueError(
                f"ISAR artifact member {member_name!r} uses unsupported NPY "
                f"version {version!r}"
            )
    except (EOFError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("ISAR artifact"):
            raise
        raise ValueError(
            f"ISAR artifact member {member_name!r} has an invalid NPY header"
        ) from exc
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError(
            f"ISAR artifact member {member_name!r} must not contain object data"
        )
    shape = tuple(int(value) for value in shape)
    if any(value < 0 for value in shape):
        raise ValueError(
            f"ISAR artifact member {member_name!r} has an invalid array shape"
        )
    cells = int(math.prod(shape)) if shape else 1
    payload_bytes = cells * int(dtype.itemsize)
    return shape, bool(fortran_order), dtype, cells, payload_bytes, int(stream.tell())


def _preflight_npz_archive(source) -> dict[str, dict[str, Any]]:
    """Inspect ZIP/NPY declarations before NumPy allocates artifact arrays."""

    try:
        archive = zipfile.ZipFile(source, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("ISAR artifact is not a valid NPZ archive") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) > _MAX_ARCHIVE_MEMBERS:
            raise ValueError(
                f"ISAR artifact has {len(infos):,} archive members; the safety "
                f"limit is {_MAX_ARCHIVE_MEMBERS:,}"
            )
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ValueError("ISAR artifact contains duplicate archive members")
        if "manifest_json.npy" not in names:
            raise ValueError("ISAR artifact is missing manifest_json")

        headers: dict[str, dict[str, Any]] = {}
        total_numeric_bytes = 0
        for info in infos:
            member_name = str(info.filename)
            if (
                info.is_dir()
                or not member_name.endswith(".npy")
                or "/" in member_name
                or "\\" in member_name
                or member_name in {".npy", "..npy"}
            ):
                raise ValueError(
                    f"ISAR artifact contains an invalid member name {member_name!r}"
                )
            if info.flag_bits & 0x1:
                raise ValueError("ISAR artifact must not contain encrypted members")
            key = member_name[:-4]
            try:
                with archive.open(info, mode="r") as stream:
                    (
                        shape,
                        fortran_order,
                        dtype,
                        cells,
                        payload_bytes,
                        header_bytes,
                    ) = _read_npy_member_header(stream, member_name)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise ValueError(
                    f"ISAR artifact member {member_name!r} cannot be inspected"
                ) from exc
            if header_bytes + payload_bytes != int(info.file_size):
                raise ValueError(
                    f"ISAR artifact member {member_name!r} size disagrees with "
                    "its NPY header"
                )
            if key == "manifest_json":
                if shape != () or dtype.kind not in "US":
                    raise ValueError(
                        "ISAR artifact manifest_json must be a scalar string"
                    )
                if payload_bytes > _MAX_ARTIFACT_MANIFEST_BYTES:
                    raise ValueError(
                        "ISAR artifact manifest exceeds the "
                        f"{_MAX_ARTIFACT_MANIFEST_BYTES / 1024**2:g} MiB safety limit"
                    )
            else:
                if cells > _MAX_ARTIFACT_ARRAY_CELLS:
                    raise ValueError(
                        f"ISAR artifact member {member_name!r} declares {cells:,} "
                        f"cells; the safety limit is {_MAX_ARTIFACT_ARRAY_CELLS:,}"
                    )
                total_numeric_bytes += _normalized_array_bytes(
                    key,
                    dtype,
                    cells,
                    payload_bytes,
                )
                if total_numeric_bytes > _MAX_ARTIFACT_NUMERIC_BYTES:
                    raise ValueError(
                        "ISAR artifact numerical payload exceeds the "
                        f"{_MAX_ARTIFACT_NUMERIC_BYTES / 1024**3:g} GiB safety limit"
                    )
            headers[key] = {
                "shape": shape,
                "fortran_order": fortran_order,
                "dtype": dtype,
                "cells": cells,
                "payload_bytes": payload_bytes,
            }
        return headers


def save_isar_artifact(
    path,
    band_results: Iterable[dict],
    manifest: dict,
    *,
    compressed: bool | None = None,
) -> Path:
    """Transactionally save an ISAR artifact with adaptive NPZ compression.

    ``compressed=None`` samples at most 2 MiB and avoids expensive DEFLATE for
    noise-like coherent images. Version-2 artifacts store a complex image only
    once and derive its magnitude on load.
    """

    destination = Path(path).expanduser().resolve()
    if not str(destination).lower().endswith(".isar.npz"):
        destination = Path(f"{destination}.isar.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bands = list(band_results)
    if not bands:
        raise ValueError("ISAR artifact needs at least one band")
    band_manifests = _manifest_band_list(manifest, len(bands))
    schema = str(manifest.get("schema", ""))
    payloads = [
        _validated_band_payload(
            index,
            band,
            band_manifest,
            derive_magnitude=not (
                schema == ISAR_ARTIFACT_SCHEMA
                and bool(band_manifest.get("complex_image_available", False))
            ),
            validate_redundant_magnitude=(
                schema == "grim.isar-result.v1"
                and bool(band_manifest.get("complex_image_available", False))
            ),
        )
        for index, (band, band_manifest) in enumerate(zip(bands, band_manifests))
    ]

    numerical_arrays: dict[str, np.ndarray] = {}
    for index, payload in enumerate(payloads):
        prefix = f"band_{index}_"
        if "magnitude" in payload:
            numerical_arrays[prefix + "magnitude"] = payload["magnitude"]
        numerical_arrays[prefix + "x_range"] = payload["x_range"]
        numerical_arrays[prefix + "y_range"] = payload["y_range"]
        if "complex_image" in payload:
            numerical_arrays[prefix + "complex_image"] = payload["complex_image"]

    _validate_numerical_payload_budget(numerical_arrays.values())

    if compressed is not None and not isinstance(compressed, (bool, np.bool_)):
        raise TypeError("compressed must be True, False, or None")
    decision = _compression_decision(numerical_arrays.values())
    use_compression = (
        bool(decision["compressed"]) if compressed is None else bool(compressed)
    )
    manifest_to_write = dict(_json_safe(manifest))
    manifest_to_write["storage"] = {
        "npz_compression": "deflate" if use_compression else "stored",
        "adaptive": compressed is None,
        "estimated_savings_fraction": float(
            decision["estimated_savings_fraction"]
        ),
        "sample_bytes": int(decision["sample_bytes"]),
        "payload_bytes": int(decision["payload_bytes"]),
    }
    manifest_text = json.dumps(
        manifest_to_write,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(manifest_text.encode("utf-8")) > _MAX_ARTIFACT_MANIFEST_BYTES:
        raise ValueError(
            "ISAR artifact manifest exceeds the "
            f"{_MAX_ARTIFACT_MANIFEST_BYTES / 1024**2:g} MiB safety limit"
        )
    manifest_array = np.asarray(manifest_text)
    if manifest_array.nbytes > _MAX_ARTIFACT_MANIFEST_BYTES:
        raise ValueError(
            "ISAR artifact stored manifest exceeds the "
            f"{_MAX_ARTIFACT_MANIFEST_BYTES / 1024**2:g} MiB safety limit"
        )
    arrays: dict[str, np.ndarray] = {
        "manifest_json": manifest_array,
        **numerical_arrays,
    }

    handle = tempfile.NamedTemporaryFile(
        prefix=".isar-stage-",
        suffix=".npz",
        dir=destination.parent,
        delete=False,
    )
    stage = Path(handle.name)
    handle.close()
    try:
        writer = np.savez_compressed if use_compression else np.savez
        writer(stage, **arrays)
        # Windows requires a writable descriptor for ``fsync`` even though no
        # further bytes are changed here.
        with stage.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(stage, destination)
    except Exception:
        try:
            stage.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def load_isar_artifact(path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """Load and validate an ISAR artifact without enabling pickle."""

    source = Path(path).expanduser().resolve()
    source_stream = source.open("rb")
    try:
        headers = _preflight_npz_archive(source_stream)
        source_stream.seek(0)
    except Exception:
        source_stream.close()
        raise
    with source_stream, np.load(source_stream, allow_pickle=False) as archive:
        if set(archive.files) != set(headers):
            raise ValueError(
                "ISAR artifact archive index changed during validation"
            )
        if "manifest_json" not in archive:
            raise ValueError("ISAR artifact is missing manifest_json")
        manifest_array = np.asarray(archive["manifest_json"])
        if manifest_array.ndim != 0 or manifest_array.dtype.kind not in "US":
            raise ValueError("ISAR artifact manifest_json must be a scalar string")
        raw_manifest = manifest_array.item()
        if isinstance(raw_manifest, bytes):
            try:
                manifest_text = raw_manifest.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "ISAR artifact manifest_json is not valid UTF-8"
                ) from exc
        else:
            manifest_text = str(raw_manifest)
        try:
            manifest = json.loads(manifest_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("ISAR artifact manifest_json is invalid") from exc
        raw_band_manifests = manifest.get("bands") if isinstance(manifest, dict) else None
        expected_count = len(raw_band_manifests) if isinstance(raw_band_manifests, list) else 0
        band_manifests = _manifest_band_list(manifest, expected_count)
        schema = str(manifest.get("schema", ""))
        allowed_keys = {"manifest_json"}
        for index in range(expected_count):
            prefix = f"band_{index}_"
            has_complex = bool(
                band_manifests[index].get("complex_image_available", False)
            )
            stores_magnitude = schema != ISAR_ARTIFACT_SCHEMA or not has_complex
            allowed_keys.update({prefix + "x_range", prefix + "y_range"})
            if stores_magnitude:
                allowed_keys.add(prefix + "magnitude")
            if has_complex:
                allowed_keys.add(prefix + "complex_image")
        unexpected = sorted(set(archive.files) - allowed_keys)
        if unexpected:
            raise ValueError(
                "ISAR artifact contains arrays not declared by its manifest: "
                + ", ".join(unexpected)
            )
        bands = []
        for index, band_manifest in enumerate(band_manifests):
            prefix = f"band_{index}_"
            has_complex = bool(
                band_manifest.get("complex_image_available", False)
            )
            stores_magnitude = schema != ISAR_ARTIFACT_SCHEMA or not has_complex
            required = [prefix + name for name in ("x_range", "y_range")]
            if stores_magnitude:
                required.append(prefix + "magnitude")
            if has_complex:
                required.append(prefix + "complex_image")
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(
                    "ISAR artifact is missing array(s): " + ", ".join(missing)
                )
            raw_band = {
                # NpzFile indexing already materializes an independent ndarray;
                # copying it again would briefly double every loaded image.
                "x_range": np.asarray(archive[prefix + "x_range"]),
                "y_range": np.asarray(archive[prefix + "y_range"]),
            }
            magnitude_key = prefix + "magnitude"
            if magnitude_key in archive:
                raw_band["magnitude"] = np.asarray(archive[magnitude_key])
            complex_key = prefix + "complex_image"
            if complex_key in archive:
                raw_band["complex_image"] = np.asarray(archive[complex_key])
            payload = _validated_band_payload(
                index,
                raw_band,
                band_manifest,
                validate_redundant_magnitude=(
                    schema == "grim.isar-result.v1" and has_complex
                ),
            )
            band = dict(payload)
            band["manifest"] = dict(band_manifest)
            bands.append(band)
    return manifest, bands


__all__ = [
    "ISAR_ARTIFACT_SCHEMA",
    "build_isar_manifest",
    "load_isar_artifact",
    "save_isar_artifact",
]
