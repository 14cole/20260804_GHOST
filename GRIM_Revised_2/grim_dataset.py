from grim_format_io import RcsGridFormatMixin
import copy
from dataclasses import replace
import hashlib
import json
import csv
import math
import operator
import os
import re
import tempfile
import unicodedata
import warnings
import zipfile
import numpy as np

from grim_metadata import (
    ADVISORY_METADATA_KEYS, canonical_time_convention, inspect_scalar_metadata,
)

C0 = 299_792_458.0

# SENTRi's documented export convention: global origin, exp(+jwt), with the
# outgoing exp(-jkr)/r factor removed. Incoming propagation reverses the look
# vector without changing theta/phi polarization vectors. These are the same
# V=theta, H=phi field conventions used by Assembly. RcsGrid still stores sigma
# and phase; the backend recovers F by dividing sqrt(sigma) by sqrt(4*pi).
SENTRI_FAR_FIELD_METADATA = {
    "phase_reference": (
        "origin=(0,0,0) vehicle frame, convention=exp(+jwt), "
        "radar earth-frame V/H monostatic amplitude"
    ),
    "amplitude_convention": "F physical far-field amplitude; sigma_3d=4*pi*|F|^2",
    "complex_field_domain": "coherent_radar_frame_far_field_amplitude",
    "time_convention": "exp(+jwt)",
    "sentri_far_field_reference": (
        "SENTRi export: exp(+jwt); global coordinate origin; "
        "outgoing exp(-jkr)/r removed; incident propagation opposite look; "
        "unchanged theta/phi polarization vectors"
    ),
}

# The legacy PTM bytes do not define the sign/origin of their aspect axis or
# the H/V basis used along a great-circle cut.  GRIM therefore distinguishes
# its explicit convention from an unmarked legacy PTM instead of silently
# treating every file as the same coordinate chart.
GRIM_GC_CONVENTION = "grim_gc_v1"
LEGACY_PTM_GC_CONVENTION = "legacy_ptm_unspecified"
_PTM_GRIM_GC_MARKER = "GRIM_GC_V1"
WEDGE_TURNTABLE_CONVENTION = "grim_vertical_turntable_body_y_pitch_v1"
CONIC_VH_BASIS_CONVENTION = "grim_conic_spherical_vh_v1"


_PIO_ASCII_METADATA_REPLACEMENTS = str.maketrans(
    {
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "⇄": "<->",
        "⇒": "=>",
        "⇐": "<=",
        "⇔": "<=>",
        "°": " deg",
        "Δ": "Delta",
        "δ": "delta",
        "Σ": "Sum",
        "∑": "sum",
        "⊕": "+",
        "σ": "sigma",
        "λ": "lambda",
        "π": "pi",
        "θ": "theta",
        "φ": "phi",
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "x",
        "÷": "/",
        "·": "*",
        "≥": ">=",
        "≤": "<=",
        "≈": "~",
        "…": "...",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def _pio_ascii_metadata(value):
    """Return readable, single-line ASCII for Pioneer header metadata.

    Pioneer headers are byte-level ASCII even though GRIM histories and file
    names are UTF-8. Common engineering symbols are transliterated for human
    readability; any uncommon character is retained as an ASCII ``\\u`` or
    ``\\U`` escape instead of aborting an otherwise valid complex-data export.
    """

    text = str(value or "").translate(_PIO_ASCII_METADATA_REPLACEMENTS)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in text
    )
    text = " ".join(text.split())
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def wedge_to_conic_geometry_deg(phi_deg, tau_deg):
    """Map vertical-turntable/body-wedge coordinates to conic directions.

    ``phi`` is rotation about the fixed world ``+z`` turntable axis. ``tau``
    is a body pitch about body ``+y`` applied before that rotation, so the
    body-to-world attitude is ``Rz(phi) @ Ry(tau)``.  The returned longitude
    and latitude describe the same world line of sight in body coordinates.
    """

    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    tau = np.deg2rad(np.asarray(tau_deg, dtype=float))
    phi, tau = np.broadcast_arrays(phi, tau)
    direction = np.stack(
        (
            np.cos(tau) * np.cos(phi),
            -np.sin(phi),
            np.sin(tau) * np.cos(phi),
        ),
        axis=-1,
    )
    longitude = np.arctan2(direction[..., 1], direction[..., 0])
    latitude = np.arcsin(np.clip(direction[..., 2], -1.0, 1.0))
    return np.rad2deg(longitude), np.rad2deg(latitude)


def conic_to_wedge_geometry_deg(longitude_deg, latitude_deg):
    """Inverse of :func:`wedge_to_conic_geometry_deg` for |tau| <= 90 deg."""

    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=float))
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    longitude, latitude = np.broadcast_arrays(longitude, latitude)
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)
    sin_phi = np.clip(-y, -1.0, 1.0)
    cos_phi_magnitude = np.sqrt(np.maximum(0.0, 1.0 - sin_phi * sin_phi))
    # cos(tau) is nonnegative on the supported wedge interval, so x fixes the
    # branch of cos(phi). At x=0 choose the + branch; tau then correctly tends
    # to +/-90 degrees for non-equatorial side looks.
    cos_phi = np.where(x < 0.0, -cos_phi_magnitude, cos_phi_magnitude)
    phi = np.arctan2(sin_phi, cos_phi)
    branch = np.where(cos_phi < 0.0, -1.0, 1.0)
    tau = np.arctan2(branch * z, branch * x)
    return np.rad2deg(phi), np.rad2deg(tau)


def wedge_to_conic_basis_change(phi_deg, tau_deg):
    """Return old-basis coordinates of the conic ``(V,H)`` basis.

    If ``S_w`` is a monostatic Jones matrix in the range's vertical/horizontal
    basis for the vertical-turntable wedge setup, the normal conic-range
    matrix is ``C.T @ S_w @ C``.  The last two axes of the result are ordered
    old ``(V,H)`` by new ``(V,H)``.
    """

    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    tau = np.deg2rad(np.asarray(tau_deg, dtype=float))
    phi, tau = np.broadcast_arrays(phi, tau)
    longitude_deg, latitude_deg = wedge_to_conic_geometry_deg(
        np.rad2deg(phi), np.rad2deg(tau)
    )
    longitude = np.deg2rad(longitude_deg)
    latitude = np.deg2rad(latitude_deg)

    wedge_v = np.stack(
        (-np.sin(tau), np.zeros_like(tau), np.cos(tau)), axis=-1
    )
    wedge_h = np.stack(
        (
            np.cos(tau) * np.sin(phi),
            np.cos(phi),
            np.sin(tau) * np.sin(phi),
        ),
        axis=-1,
    )
    conic_v = np.stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ),
        axis=-1,
    )
    conic_h = np.stack(
        (-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)),
        axis=-1,
    )
    old_basis = np.stack((wedge_v, wedge_h), axis=-2)
    new_basis = np.stack((conic_v, conic_h), axis=-2)
    return np.einsum("...ia,...ja->...ij", old_basis, new_basis)


def rotate_wedge_jones_to_conic(jones, phi_deg, tau_deg):
    """Rotate monostatic Jones matrices from wedge-range to conic V/H."""

    matrix = np.asarray(jones)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("Jones data must end with a 2x2 (receive, transmit) matrix")
    change = wedge_to_conic_basis_change(phi_deg, tau_deg)
    return np.einsum("...ia,...ij,...jb->...ab", change, matrix, change)


def _jones_from_polarization_channels(
    field, polarizations, *, assume_missing_cross_pol_zero=False
):
    """Build ``[..., receive(V,H), transmit(V,H)]`` from named channels."""

    labels = [str(value).strip().upper() for value in polarizations]
    if len(set(labels)) != len(labels):
        raise ValueError("Wedge-to-Conic requires unique polarization labels")
    unsupported = sorted(set(labels) - {"VV", "VH", "HV", "HH"})
    if unsupported:
        raise ValueError(
            "Wedge-to-Conic Jones rotation supports VV/VH/HV/HH labels only; got "
            + ", ".join(unsupported)
        )
    index = {label: position for position, label in enumerate(labels)}
    if "VV" not in index or "HH" not in index:
        raise ValueError(
            "Wedge-to-Conic Jones rotation requires both VV and HH channels"
        )
    values = np.asarray(field)
    matrix = np.empty(values.shape[:-1] + (2, 2), dtype=values.dtype)
    matrix[..., 0, 0] = values[..., index["VV"]]
    matrix[..., 1, 1] = values[..., index["HH"]]
    if "VH" in index and "HV" in index:
        matrix[..., 0, 1] = values[..., index["VH"]]
        matrix[..., 1, 0] = values[..., index["HV"]]
        cross_note = "measured VH and HV"
    elif "VH" in index:
        matrix[..., 0, 1] = values[..., index["VH"]]
        matrix[..., 1, 0] = values[..., index["VH"]]
        cross_note = "monostatic reciprocity: HV=VH"
    elif "HV" in index:
        matrix[..., 1, 0] = values[..., index["HV"]]
        matrix[..., 0, 1] = values[..., index["HV"]]
        cross_note = "monostatic reciprocity: VH=HV"
    elif assume_missing_cross_pol_zero:
        matrix[..., 0, 1] = 0.0
        matrix[..., 1, 0] = 0.0
        cross_note = "explicit assumption: missing VH=HV=0"
    else:
        raise ValueError(
            "Wedge-to-Conic changes the V/H basis and cannot rotate VV/HH "
            "alone. Supply VH or HV (monostatic reciprocity supplies the "
            "other channel), or explicitly assume missing cross-pol is zero."
        )
    return matrix, labels, cross_note


def _polarization_channels_from_jones(matrix, labels):
    channel = {"VV": (0, 0), "VH": (0, 1), "HV": (1, 0), "HH": (1, 1)}
    return np.stack(
        [matrix[..., channel[label][0], channel[label][1]] for label in labels],
        axis=-1,
    )


def _ptm_configuration_has_grim_gc_marker(value):
    return bool(
        re.search(
            r"(?<![A-Z0-9_])GRIM_GC_V1(?![A-Z0-9_])",
            str(value or "").upper(),
        )
    )


def _ptm_configuration_with_grim_gc_marker(value):
    """Embed GRIM's coordinate convention without discarding legacy text."""

    text = str(value or "").strip()
    if _ptm_configuration_has_grim_gc_marker(text):
        return text
    if not text:
        return _PTM_GRIM_GC_MARKER
    available = 50 - len(_PTM_GRIM_GC_MARKER) - 1
    return f"{_PTM_GRIM_GC_MARKER};{text[:available]}"


def _ptm_configuration_without_grim_gc_marker(value):
    """Remove only GRIM's semicolon-delimited convention marker."""

    parts = [part.strip() for part in str(value or "").split(";")]
    return ";".join(
        part for part in parts
        if part and part.upper() != _PTM_GRIM_GC_MARKER
    )

_FREQUENCY_UNITS = {
    "hz": "Hz",
    "khz": "kHz",
    "mhz": "MHz",
    "ghz": "GHz",
}
_ANGLE_UNITS = {
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
}

# Metadata producers have historically used several names for the same
# acquisition fact.  These are semantic families, not independent optional
# strings: a calibration ID under one alias must be checked against the other
# aliases, and a monostatic declaration must not coexist with a bistatic one.
# Identity-like fields that describe different facts (for example a
# calibration run and a calibration version) deliberately remain separate.
_ACQUISITION_METADATA_FAMILIES = (
    (
        "amplitude_convention",
        "amplitude convention",
        "text",
        ("amplitude_convention",),
    ),
    (
        "complex_field_domain",
        "complex-field domain",
        "text",
        ("complex_field_domain",),
    ),
    (
        "range_phase_law",
        "two-way range-phase convention",
        "range_phase",
        ("range_phase_convention", "phase_law"),
    ),
    (
        "acquisition_geometry",
        "measurement geometry",
        "geometry",
        (
            "measurement_geometry",
            "acquisition_geometry",
            "scattering_geometry",
            "radar_geometry",
            "measurement_domain",
            "field_domain",
            "range_type",
            "wavefront_geometry",
        ),
    ),
    (
        "motion_state",
        "motion-compensation/phase-center state",
        "motion",
        (
            "motion_compensation",
            "motion_compensated",
            "phase_center_stability",
            "phase_center_motion",
            "range_alignment",
        ),
    ),
    (
        "calibration_identifier",
        "calibration ID",
        "identity",
        ("calibration_id", "calibration_identifier"),
    ),
    (
        "calibration_chain_id",
        "calibration-chain ID",
        "identity",
        ("calibration_chain_id",),
    ),
    (
        "calibration_run_id",
        "calibration-run ID",
        "identity",
        ("calibration_run_id",),
    ),
    (
        "calibration_version",
        "calibration version",
        "identity",
        ("calibration_version",),
    ),
    (
        "measurement_setup_identifier",
        "measurement-setup ID",
        "identity",
        ("measurement_setup_id", "radar_setup_id"),
    ),
    (
        "fixture_id",
        "fixture ID",
        "identity",
        ("fixture_id",),
    ),
    (
        "static_setup",
        "static-setup declaration",
        "setup_state",
        ("static_setup",),
    ),
)

_SUPPORT_REFERENCE_METADATA_FIELDS = tuple(
    key
    for _family, _label, _kind, keys in _ACQUISITION_METADATA_FAMILIES
    for key in keys
)

# Dense joins use a bounded advanced-index block and then transfer ownership of
# their already-sanitized output into RcsGrid. The singleton prevents callers
# from bypassing constructor sanitation with a public-looking boolean switch.
_JOIN_MERGE_BLOCK_CELLS = 262_144
_PIO_WRITE_BLOCK_CELLS = 262_144
_RAW_COMPLEX_VALIDATION_BLOCK_CELLS = 262_144
_COHERENT_OPERATION_BLOCK_CELLS = 262_144
# Text/binary legacy readers expand sparse row sets into dense Cartesian grids.
# Keep a direct-call safety ceiling even when the GUI's broader batch-memory
# planner is bypassed. Callers handling a deliberately larger verified file
# may raise the explicit per-load limit.
_DENSE_IMPORT_FALLBACK_LIMIT_BYTES = 2 * 1024**3
_ADOPT_CLEAN_ARRAYS_TOKEN = object()


def _pio_remove_closed_azimuth_endpoint(values, unit):
    """Return one physical turn of a PIO azimuth axis, without its closer.

    Range data often starts at an arbitrary turntable position and repeats that
    direction after one revolution.  The closing coordinate may therefore be
    ``181.16`` for an opening coordinate of ``-178.84``, or a writer may wrap
    it back to ``-178.84``.  Detect the full-turn trajectory rather than
    special-casing 0/360 or -180/+180.  The opening sample is authoritative;
    the repeated closing row is removed even when a second measurement differs.
    """

    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        return axis, False

    period = 2.0 * np.pi if unit == "rad" else 360.0
    half_period = 0.5 * period
    scale = max(1.0, float(np.max(np.abs(axis))), period)
    tolerance = max(
        float(np.deg2rad(1.0e-7)) if unit == "rad" else 1.0e-7,
        8.0 * np.finfo(np.float32).eps * scale,
    )

    raw_steps = np.diff(axis)
    periodic_steps = np.remainder(raw_steps + half_period, period) - half_period
    # At exactly half a turn, modulo chooses the negative representative.
    # Retain the acquisition direction expressed by the source axis.
    positive_half_turn = (
        np.isclose(periodic_steps, -half_period, rtol=0.0, atol=tolerance)
        & (raw_steps > 0.0)
    )
    periodic_steps[positive_half_turn] = half_period

    increasing = bool(np.all(periodic_steps > 0.0))
    descending = bool(np.all(periodic_steps < 0.0))
    unwrapped = np.concatenate(
        (axis[:1], axis[0] + np.cumsum(periodic_steps, dtype=float))
    )
    trajectory_span = float(unwrapped[-1] - unwrapped[0])
    raw_span = float(axis[-1] - axis[0])
    endpoint_residual = float(
        np.remainder(raw_span + half_period, period) - half_period
    )
    same_direction = abs(endpoint_residual) <= tolerance
    full_turn = (
        abs(abs(trajectory_span) - period) <= tolerance
        or abs(abs(raw_span) - period) <= tolerance
    )
    direct_two_point_turn = (
        axis.size == 2 and abs(abs(raw_span) - period) <= tolerance
    )
    if not (
        same_direction
        and full_turn
        and (increasing or descending or direct_two_point_turn)
    ):
        return axis, False

    raw_prefix_steps = np.diff(axis[:-1])
    raw_prefix_monotonic = bool(
        axis.size <= 2
        or np.all(raw_prefix_steps > 0.0)
        or np.all(raw_prefix_steps < 0.0)
    )
    unique_axis = axis[:-1] if raw_prefix_monotonic else unwrapped[:-1]
    return np.asarray(unique_axis, dtype=float).copy(), True


def _available_import_memory_bytes():
    """Best-effort available physical memory without a hard dependency."""

    try:
        import psutil

        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                available = int(status.ullAvailPhys)
                if available > 0:
                    return available
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        available = page_size * available_pages
        return available if available > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _default_dense_import_limit_bytes():
    available = _available_import_memory_bytes()
    if available is None:
        return _DENSE_IMPORT_FALLBACK_LIMIT_BYTES
    # Match GRIM's other dense dataset workflows: never let one legacy import
    # reserve more than half of currently available physical memory.
    return max(1, int(available * 0.5))


def _coherent_working_set_limit_bytes(maximum_working_bytes):
    """Return the reviewed cap for a dense coherent arithmetic operation."""

    if maximum_working_bytes is None:
        raw_limit_mb = os.environ.get("GRIM_COHERENT_WORKING_SET_MB")
        if raw_limit_mb is None:
            return _default_dense_import_limit_bytes()
        try:
            limit = operator.index(int(raw_limit_mb)) * 1024**2
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "GRIM_COHERENT_WORKING_SET_MB must be a positive integer"
            ) from exc
    else:
        if isinstance(maximum_working_bytes, (bool, np.bool_)):
            raise TypeError("maximum_working_bytes must be a positive integer")
        try:
            limit = operator.index(maximum_working_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "maximum_working_bytes must be a positive integer"
            ) from exc
    if limit <= 0:
        raise ValueError("coherent-operation working-set limit must be positive")
    return int(limit)


def _bounded_grid_selections(shape, maximum_cells):
    """Yield basic-slice tiles whose Cartesian size is bounded."""

    dimensions = tuple(int(value) for value in shape)
    if not dimensions or any(value <= 0 for value in dimensions):
        raise ValueError("bounded grid selection requires positive dimensions")
    maximum_cells = int(maximum_cells)
    if maximum_cells <= 0:
        raise ValueError("maximum_cells must be positive")

    block_shape = [1] * len(dimensions)
    remaining = maximum_cells
    for axis in range(len(dimensions) - 1, -1, -1):
        width = min(dimensions[axis], max(1, remaining))
        block_shape[axis] = width
        remaining = max(1, remaining // width)
    block_counts = tuple(
        (extent + width - 1) // width
        for extent, width in zip(dimensions, block_shape)
    )
    for block_index in np.ndindex(*block_counts):
        yield tuple(
            slice(index * width, min(extent, (index + 1) * width))
            for index, width, extent in zip(
                block_index, block_shape, dimensions
            )
        )


def _checked_dense_import_allocation(
    shape,
    dtypes,
    *,
    source,
    max_output_bytes=None,
    resident_bytes=0,
):
    """Validate one planned dense import before any output array allocation.

    ``dense_bytes`` is exact for the declared arrays. ``resident_bytes`` lets
    readers add the exact NumPy payloads that remain live while those outputs
    are created. Python row/dict overhead is deliberately not guessed;
    this guard's job is to stop sparse-axis Cartesian-product bombs before
    ``np.full`` commits the dense storage.
    """

    dimensions = []
    cell_count = 1
    for raw_count in tuple(shape):
        if isinstance(raw_count, (bool, np.bool_)) or not isinstance(
            raw_count, (int, np.integer)
        ):
            raise ValueError(f"{source}: dense axis counts must be integers")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"{source}: dense axis counts must be positive")
        dimensions.append(count)
        cell_count *= count

    dtype_list = [np.dtype(value) for value in tuple(dtypes)]
    if not dtype_list:
        raise ValueError(f"{source}: dense allocation must declare an array dtype")
    bytes_per_cell = sum(dtype.itemsize for dtype in dtype_list)
    dense_bytes = cell_count * bytes_per_cell
    try:
        resident = operator.index(resident_bytes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source}: resident_bytes must be a nonnegative integer") from exc
    if resident < 0:
        raise ValueError(f"{source}: resident_bytes must be nonnegative")
    peak_bytes = dense_bytes + resident

    if max_output_bytes is None:
        limit = _default_dense_import_limit_bytes()
    else:
        if isinstance(max_output_bytes, (bool, np.bool_)):
            raise ValueError(f"{source}: max_output_bytes must be a positive integer")
        try:
            limit = operator.index(max_output_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{source}: max_output_bytes must be a positive integer"
            ) from exc
        if limit <= 0:
            raise ValueError(f"{source}: max_output_bytes must be a positive integer")

    if dense_bytes > np.iinfo(np.intp).max or peak_bytes > np.iinfo(np.intp).max:
        raise MemoryError(
            f"{source}: dense grid {tuple(dimensions)} exceeds this Python/NumPy "
            "build's addressable allocation size"
        )
    if peak_bytes > limit:
        raise MemoryError(
            f"{source}: dense grid {tuple(dimensions)} has {cell_count:,} cells; "
            f"the planned arrays require {dense_bytes / 1024**2:.1f} MiB"
            + (
                f" plus {resident / 1024**2:.1f} MiB of live parsed NumPy payload"
                if resident
                else ""
            )
            + f", exceeding the {limit / 1024**2:.1f} MiB import limit. "
            "Split the source sweep or pass a larger explicit max_output_bytes "
            "only after verifying available memory."
        )
    return {
        "shape": tuple(dimensions),
        "cell_count": int(cell_count),
        "dense_bytes": int(dense_bytes),
        "resident_bytes": int(resident),
        "peak_bytes": int(peak_bytes),
        "limit_bytes": int(limit),
    }


def _preflight_native_archive_allocation(
    path,
    *,
    allow_legacy_pickle=False,
    max_output_bytes=None,
):
    """Validate NPZ member framing and peak eager-load bytes before extraction.

    ``np.load`` defers each compressed member until subscription.  Without a
    central-directory/header preflight, a tiny archive can declare a huge NPY
    shape and trigger a multi-gigabyte allocation before GRIM sees the axes.
    This reads only bounded NPY headers, verifies their declared data lengths,
    and applies the same configurable memory policy as direct dense imports.
    """

    if max_output_bytes is None:
        limit = _default_dense_import_limit_bytes()
    else:
        if isinstance(max_output_bytes, (bool, np.bool_)):
            raise ValueError("native .grim max_output_bytes must be a positive integer")
        try:
            limit = operator.index(max_output_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "native .grim max_output_bytes must be a positive integer"
            ) from exc
        if limit <= 0:
            raise ValueError("native .grim max_output_bytes must be a positive integer")

    member_payload_bytes: dict[str, int] = {}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            seen_names: set[str] = set()
            duplicate_names: set[str] = set()
            for name in names:
                if name in seen_names:
                    duplicate_names.add(name)
                seen_names.add(name)
            if duplicate_names:
                raise ValueError(
                    f"{path} contains duplicate archive member(s): "
                    + ", ".join(sorted(duplicate_names))
                )
            unexpected = [name for name in names if not name.endswith(".npy")]
            if unexpected:
                raise ValueError(
                    f"{path} contains unsupported non-NPY archive member(s): "
                    + ", ".join(unexpected[:5])
                )
            encrypted = [info.filename for info in infos if info.flag_bits & 0x1]
            if encrypted:
                raise ValueError(
                    f"{path} contains encrypted archive member(s), which GRIM "
                    "does not support"
                )

            declared_uncompressed = sum(max(0, int(info.file_size)) for info in infos)
            if declared_uncompressed > limit:
                raise MemoryError(
                    f"{path} declares {declared_uncompressed / 1024**3:.2f} GiB "
                    "of uncompressed native members, above the current "
                    f"{limit / 1024**3:.2f} GiB load limit. Use a machine with "
                    "more available memory or pass a larger reviewed "
                    "max_output_bytes value."
                )

            for info in infos:
                with archive.open(info, "r") as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _fortran, dtype = (
                            np.lib.format.read_array_header_1_0(
                                member, max_header_size=10_000
                            )
                        )
                    elif version in {(2, 0), (3, 0)}:
                        # Formats 2 and 3 share the four-byte header-length
                        # framing. GRIM writes ASCII dtype descriptors, so the
                        # public v2 reader safely validates either one.
                        shape, _fortran, dtype = (
                            np.lib.format.read_array_header_2_0(
                                member, max_header_size=10_000
                            )
                        )
                    else:
                        raise ValueError(
                            f"{path} contains unsupported NPY version "
                            f"{version!r} in {info.filename}"
                        )
                    header_bytes = int(member.tell())

                dtype = np.dtype(dtype)
                remaining = int(info.file_size) - header_bytes
                if remaining < 0:
                    raise ValueError(
                        f"{path} contains a truncated NPY header in {info.filename}"
                    )
                if dtype.hasobject:
                    if not bool(allow_legacy_pickle):
                        raise ValueError(
                            f"{path} contains object-typed member {info.filename}; "
                            "legacy pickle loading must be explicitly enabled"
                        )
                    payload_bytes = remaining
                else:
                    element_count = 1
                    for raw_dimension in tuple(shape):
                        dimension = int(raw_dimension)
                        if dimension < 0:
                            raise ValueError(
                                f"{path} contains a negative NPY dimension in "
                                f"{info.filename}"
                            )
                        element_count *= dimension
                    payload_bytes = element_count * int(dtype.itemsize)
                    if payload_bytes != remaining:
                        raise ValueError(
                            f"{path} contains inconsistent NPY shape/data framing "
                            f"in {info.filename}"
                        )
                member_payload_bytes[info.filename[:-4]] = payload_bytes
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"{path} is not a valid native .grim/NPZ archive") from exc

    required = {
        "azimuths",
        "elevations",
        "frequencies",
        "polarizations",
        "rcs_power",
        "rcs_phase",
    }
    missing = sorted(required.difference(member_payload_bytes))
    if missing:
        raise ValueError(
            f"{path} is not a supported .grim file (missing keys: "
            + ", ".join(missing)
            + ")"
        )
    retained_bytes = sum(member_payload_bytes.values())
    # During RcsGrid construction, sanitized power and phase arrays coexist
    # with the eager NPZ members. This is the dominant exact load peak.
    peak_bytes = max(
        declared_uncompressed,
        retained_bytes
        + member_payload_bytes["rcs_power"]
        + member_payload_bytes["rcs_phase"],
    )
    if peak_bytes > limit:
        raise MemoryError(
            f"{path} needs about {peak_bytes / 1024**3:.2f} GiB peak memory "
            "for eager native load and validation, above the current "
            f"{limit / 1024**3:.2f} GiB limit. Close other datasets, use a "
            "machine with more available memory, or pass a larger reviewed "
            "max_output_bytes value."
        )
    return {
        "member_payload_bytes": member_payload_bytes,
        "declared_uncompressed_bytes": declared_uncompressed,
        "retained_bytes": retained_bytes,
        "estimated_peak_bytes": peak_bytes,
        "limit_bytes": limit,
    }


def canonical_angular_coordinate_system(value):
    """Normalize scalar angular-coordinate metadata without guessing."""

    raw = value
    if isinstance(raw, np.ndarray) and raw.size == 1:
        raw = raw.reshape(-1)[0]
    text = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "": "conic",
        "az_el": "conic",
        "azimuth_elevation": "conic",
        "spherical": "conic",
        "gc": "great_circle",
        "greatcircle": "great_circle",
    }
    return aliases.get(text, text)


def _read_cst_delimited_rows(path):
    """Read a CST text export while retaining any leading metadata rows."""

    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = max((",", "\t", ";"), key=sample.count)
        return list(csv.reader(stream, delimiter=delimiter))


def _cst_compact_header(value):
    """Normalize CST/MATLAB table headings without losing their unit text."""

    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _cst_frequency_unit(value):
    """Return an exact supported CST frequency unit, never a substring guess."""

    compact = _cst_compact_header(value)
    for prefix in ("frequency", "freq"):
        if compact.startswith(prefix):
            suffix = compact[len(prefix):]
            return suffix if suffix in {"hz", "khz", "mhz", "ghz"} else None
    return None


def _cst_frequency_scale_to_ghz(value):
    unit = _cst_frequency_unit(value)
    scales = {"hz": 1.0e-9, "khz": 1.0e-6, "mhz": 1.0e-3, "ghz": 1.0}
    if unit is None:
        raise ValueError(
            "CST frequency header must explicitly end in exactly Hz, kHz, "
            "MHz, or GHz; other prefixes and unit guessing are unsupported"
        )
    return scales[unit], unit


def _wrap_cst_azimuth_deg(value):
    """Use GRIM's canonical half-open azimuth interval, [-180, 180)."""

    wrapped = float(np.mod(float(value) + 180.0, 360.0) - 180.0)
    return 0.0 if abs(wrapped) < 1.0e-12 else wrapped


def _parse_cst_iq(value):
    """Parse common Python/MATLAB spellings of one complex IQ sample."""

    text = str(value or "").strip()
    if not text:
        return None
    token = text.replace(" ", "").strip("()[]{}")
    token = token.replace("*", "").replace("I", "j").replace("i", "j")
    token = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", token)
    try:
        result = complex(token)
    except ValueError as exc:
        raise ValueError(f"unsupported IQ value {text!r}") from exc
    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError(f"IQ value must be finite, got {text!r}")
    return result


def _cst_dbsm_to_power(value, *, context="CST magnitude"):
    """Convert dBsm to finite float64 power with an actionable overflow error."""

    value = float(value)
    if np.isneginf(value):
        return 0.0
    if not np.isfinite(value):
        raise ValueError(f"{context} must be finite or -Inf, got {value!r}")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.power(10.0, value / 10.0))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            f"{context}={value:g} dBsm overflows finite linear power"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"{context}={value:g} dBsm does not produce finite linear power"
        )
    return result


def _cst_iq_to_power(value, *, context="CST IQ"):
    """Return finite |IQ|^2 without allowing float64 overflow."""

    amplitude = float(abs(value))
    if amplitude > float(np.sqrt(np.finfo(np.float64).max)):
        raise ValueError(f"{context} magnitude overflows finite linear power")
    result = amplitude * amplitude
    if not np.isfinite(result):
        raise ValueError(f"{context} does not produce finite linear power")
    return result


def _cst_samples_equivalent(left_power, left_phase, right_power, right_phase):
    """Return True when two seam/duplicate rows encode the same field."""

    if not np.isclose(
        float(left_power), float(right_power), rtol=1.0e-8, atol=1.0e-12
    ):
        return False
    # Phase is undefined for an exactly zero complex sample. Vendor exporters
    # commonly fill that redundant column with arbitrary values at a null.
    if float(left_power) == 0.0 and float(right_power) == 0.0:
        return True
    left_phase = float(left_phase)
    right_phase = float(right_phase)
    if np.isnan(left_phase) and np.isnan(right_phase):
        return True
    if not (np.isfinite(left_phase) and np.isfinite(right_phase)):
        return False
    phase_error = np.angle(np.exp(1j * (left_phase - right_phase)))
    return abs(float(phase_error)) <= 1.0e-8


def _real_storage_dtype(*values):
    """Return float32 unless any supplied numeric array carries >32-bit precision."""
    dtypes = [np.asarray(value).dtype for value in values if value is not None]
    if any(
        (dtype.kind == "f" and dtype.itemsize > 4)
        or (dtype.kind == "c" and dtype.itemsize > 8)
        for dtype in dtypes
    ):
        return np.float64
    return np.float32


def _physical_grid_content_sha256(grid, *, namespace):
    """Bind provenance to axes, conventions, and the complex field actually used.

    The authoritative GHOST real/imaginary payload can differ from the modeled
    power/phase pair (for example after float32 underflow), so hash the complex
    response through :meth:`RcsGrid.rcs_slice` in bounded azimuth blocks.  This
    helper deliberately excludes arbitrary descriptive metadata: labels and
    history may change without changing the physical subtraction/calibration.
    """

    digest = hashlib.sha256()
    digest.update(str(namespace).encode("utf-8") + b"\0")

    def _update_array(label, values):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(label).encode("ascii") + b"\0")
        digest.update(str(contiguous.shape).encode("ascii") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(contiguous.tobytes(order="C"))

    for label, values in (
        ("azimuth", np.asarray(grid.azimuths, dtype=np.float64)),
        ("elevation", np.asarray(grid.elevations, dtype=np.float64)),
        ("frequency", np.asarray(grid.frequencies, dtype=np.float64)),
    ):
        _update_array(label, values)

    digest.update(b"authoritative-complex-field\0")
    digest.update(str(grid.rcs_power.shape).encode("ascii") + b"\0")
    read_complex, _real_dtype = grid._bounded_complex_slice_reader()
    for selection in _bounded_grid_selections(
        grid.rcs_power.shape, _COHERENT_OPERATION_BLOCK_CELLS
    ):
        field = np.asarray(
            read_complex(selection),
            dtype=np.complex128,
        )
        _update_array("field-real", field.real)
        _update_array("field-imag", field.imag)

    digest.update(
        json.dumps(
            [str(value) for value in grid.polarizations.tolist()],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(
        json.dumps(
            dict(grid.units or {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    convention_metadata = {
        key: grid._declared_scalar_metadata(key)
        for key in (
            "phase_reference",
            "time_convention",
            "polarization_basis",
            "amplitude_convention",
        )
    }
    digest.update(
        json.dumps(
            convention_metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _support_reference_qa(target_with_support, support_reference, difference):
    """Return bounded-memory diagnostics for an exact complex difference.

    These are unweighted sums over the common finite sample support, not a
    physical angular/frequency integral.  The complex coherence is therefore
    an acquisition-similarity diagnostic; it is deliberately not labelled a
    measure of how much support scattering was physically removed.
    """

    energy_terms = {
        "pre_target_plus_support": [],
        "subtracted_support_reference": [],
        "post_support_referenced_difference": [],
        "algebraic_closure_residual": [],
    }
    cross_real_terms = []
    cross_imag_terms = []
    common_count = 0
    read_target, _target_dtype = target_with_support._bounded_complex_slice_reader()
    read_support, _support_dtype = support_reference._bounded_complex_slice_reader()
    read_difference, _difference_dtype = difference._bounded_complex_slice_reader()
    for selection in _bounded_grid_selections(
        target_with_support.rcs_power.shape, _COHERENT_OPERATION_BLOCK_CELLS
    ):
        target = np.asarray(read_target(selection), dtype=np.complex128)
        support = np.asarray(read_support(selection), dtype=np.complex128)
        post = np.asarray(read_difference(selection), dtype=np.complex128)
        common = np.isfinite(target) & np.isfinite(support) & np.isfinite(post)
        if not np.any(common):
            continue
        target = target[common]
        support = support[common]
        post = post[common]
        closure = target - support - post
        common_count += int(target.size)
        energy_terms["pre_target_plus_support"].append(
            float(np.vdot(target, target).real)
        )
        energy_terms["subtracted_support_reference"].append(
            float(np.vdot(support, support).real)
        )
        energy_terms["post_support_referenced_difference"].append(
            float(np.vdot(post, post).real)
        )
        energy_terms["algebraic_closure_residual"].append(
            float(np.vdot(closure, closure).real)
        )
        cross = np.vdot(support, target)
        cross_real_terms.append(float(cross.real))
        cross_imag_terms.append(float(cross.imag))

    def _bounded_fsum(values):
        try:
            total = float(math.fsum(values))
        except (OverflowError, ValueError):
            return float("nan")
        return total if np.isfinite(total) else float("nan")

    energies = {
        key: _bounded_fsum(values) for key, values in energy_terms.items()
    }
    cross = complex(
        _bounded_fsum(cross_real_terms), _bounded_fsum(cross_imag_terms)
    )
    pre_energy = energies["pre_target_plus_support"]
    support_energy = energies["subtracted_support_reference"]
    post_energy = energies["post_support_referenced_difference"]

    def _safe_db_ratio(numerator, denominator):
        if not (
            np.isfinite(numerator)
            and np.isfinite(denominator)
            and numerator > 0.0
            and denominator > 0.0
        ):
            return None
        return float(10.0 * np.log10(numerator / denominator))

    coherence = None
    coherence_phase_deg = None
    coherence_meaningful = bool(
        common_count >= 2
        and np.isfinite(pre_energy)
        and np.isfinite(support_energy)
        and pre_energy > 0.0
        and support_energy > 0.0
        and np.isfinite(cross)
    )
    if coherence_meaningful:
        normalization = np.sqrt(pre_energy) * np.sqrt(support_energy)
        coherence = float(abs(cross) / normalization)
        # Roundoff can put a mathematically bounded value a few ulps over one.
        coherence = min(1.0, max(0.0, coherence))
        coherence_phase_deg = float(np.rad2deg(np.angle(cross)))

    def _finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    return {
        "energy_metric": (
            "unweighted_sum_of_squared_complex_sample_magnitudes_on_common_"
            "finite_support"
        ),
        "total_sample_count": int(target_with_support.rcs_power.size),
        "common_finite_sample_count": int(common_count),
        "excluded_sample_count": int(
            target_with_support.rcs_power.size - common_count
        ),
        "energy_sum_linear": {
            key: _finite_or_none(value) for key, value in energies.items()
        },
        "post_to_pre_energy_db": _safe_db_ratio(post_energy, pre_energy),
        "reference_to_pre_energy_db": _safe_db_ratio(
            support_energy, pre_energy
        ),
        "complex_coherence": coherence,
        "complex_coherence_phase_deg": coherence_phase_deg,
        "complex_coherence_meaningful": coherence_meaningful,
        "complex_coherence_definition": (
            "abs(sum(conj(support_reference)*target_plus_support)) / "
            "sqrt(sum(abs(support_reference)^2)*"
            "sum(abs(target_plus_support)^2))"
        ),
    }


class RcsGrid(RcsGridFormatMixin):
    """Container for gridded RCS data with axis metadata and helpers."""

    def __init__(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=None,
        rcs_power=None,
        rcs_phase=None,
        rcs_domain: str | None = None,
        source_path: str | None = None,
        history: str | None = None,
        units: dict | None = None,
        extra: dict | None = None,
        _adopt_clean_arrays=None,
    ):
        """Build a grid from axis arrays and power/phase-backed RCS samples.

        Use when loading data from files or constructing an in-memory grid.

        Args:
            azimuths: 1D sequence of azimuth values (deg).
            elevations: 1D sequence of elevation values (deg).
            frequencies: 1D sequence of frequency values (GHz or Hz).
            polarizations: 1D sequence of polarization labels.
            rcs: Optional complex field samples shaped (az, el, f, pol).
            rcs_power: Optional linear-power samples shaped (az, el, f, pol).
            rcs_phase: Optional phase samples (radians) shaped (az, el, f, pol).
                Use NaN where phase is unknown.
            rcs_domain: Optional domain tag metadata.
            source_path: Optional source path for provenance.
            history: Optional history string.
            units: Optional units dict (e.g., {"azimuth": "deg", "frequency": "GHz"}).
            extra: Optional passthrough metadata from the source file -- keys this
                class does not model, carried so save() can write them back.
                Producers that store the raw complex far-field amplitude
                (rcs_amp_real / rcs_amp_imag, as GHOST .grim
                exports do) rely on this: without it a load/save round-trip
                silently drops the amplitude and those tools can no longer read
                the file.  Array entries are only re-emitted while their shape
                still matches the grid. Exact slice/reorder/relabel operations
                explicitly transform the raw field with the samples; nonexact
                magnitude/statistical operations drop it rather than writing a
                stale array. Exact axis-union joins preserve it only when every
                input carries a complete, compatible raw field.

        Raises:
            ValueError: if shapes do not match the expected grid.
        """

        self.azimuths = self._clean_axis(azimuths)
        self.elevations = self._clean_axis(elevations)
        self.frequencies = self._clean_axis(frequencies)
        self.polarizations = self._canonical_polarization_axis(polarizations)

        expected = (len(self.azimuths), len(self.elevations), len(self.frequencies), len(self.polarizations))

        complex_arr = None
        real_dtype = _real_storage_dtype(rcs, rcs_power, rcs_phase)
        complex_dtype = np.complex128 if real_dtype == np.float64 else np.complex64
        if rcs is not None:
            rcs_arr = np.asarray(rcs)
            if rcs_arr.shape == expected + (2,):
                complex_arr = np.asarray(
                    rcs_arr[..., 0] + 1j * rcs_arr[..., 1], dtype=complex_dtype
                )
            elif rcs_arr.shape == expected:
                if np.iscomplexobj(rcs_arr):
                    complex_arr = np.asarray(rcs_arr, dtype=complex_dtype)
                elif rcs_power is None:
                    # Real-valued rcs input is treated as linear power when explicit power is not provided.
                    rcs_power = np.asarray(rcs_arr, dtype=real_dtype)
            else:
                raise ValueError(f"rcs shape {rcs_arr.shape} != {expected}")

        if rcs_power is not None:
            power_arr = np.asarray(rcs_power, dtype=real_dtype)
            if power_arr.shape != expected:
                raise ValueError(f"rcs_power shape {power_arr.shape} != {expected}")
        elif complex_arr is not None:
            power_arr = np.abs(complex_arr) ** 2
        else:
            raise ValueError("provide complex rcs samples and/or rcs_power")

        if rcs_phase is not None:
            phase_arr = np.asarray(rcs_phase, dtype=real_dtype)
            if phase_arr.shape != expected:
                raise ValueError(f"rcs_phase shape {phase_arr.shape} != {expected}")
        elif complex_arr is not None:
            phase_arr = np.angle(complex_arr).astype(real_dtype)
        else:
            phase_arr = np.full(expected, np.nan, dtype=real_dtype)

        if (
            _adopt_clean_arrays is not None
            and _adopt_clean_arrays is not False
            and _adopt_clean_arrays is not _ADOPT_CLEAN_ARRAYS_TOKEN
        ):
            raise ValueError("_adopt_clean_arrays is reserved for internal operations")
        if _adopt_clean_arrays is _ADOPT_CLEAN_ARRAYS_TOKEN:
            # Internal ownership-transfer path for dense operations that
            # allocated and sanitised fresh arrays themselves. Re-copying a
            # multi-gigabyte result here can double the operation's peak RSS.
            power_clean = power_arr
            phase_clean = phase_arr
        else:
            power_clean = self._clean_power(power_arr)
            phase_clean = self._clean_phase(phase_arr)
            phase_clean[~np.isfinite(power_clean)] = np.nan

        self.rcs_power = power_clean
        self.rcs_phase = phase_clean
        domain = str(rcs_domain or "").strip().lower()
        if domain not in {"complex_amplitude", "linear_rcs", "power_phase"}:
            domain = "power_phase"
        self.rcs_domain = domain
        self.power_domain = "linear_rcs"
        self.source_path = source_path
        self.history = history
        self.units = dict(units or {})
        self.extra = dict(extra or {})
        # Physical amplitude identity is independent of the source solver's
        # grid-bound audit envelope, which derived operations rightly discard.
        envelope = self.extra.get("solver_metadata_json")
        if envelope is not None and self.linear_quantity() == "sigma_2d":
            try:
                parsed = json.loads(str(np.asarray(envelope).reshape(()).item()))
                version = parsed.get("amplitude_version")
                if version is not None and "amplitude_version" not in self.extra:
                    self.extra["amplitude_version"] = version
                elif version is not None and str(self.extra["amplitude_version"]) != str(version):
                    self.extra["solver_metadata_advisory"] = "Conflicting amplitude-version annotations; supplied samples retained."
            except (ValueError, TypeError, AttributeError):
                self.extra["solver_metadata_advisory"] = "Unreadable solver annotation; supplied samples retained."

        # A declared phase-wrap marker describes the stored phase
        # representation, not merely a plotting preference.  Normalize here
        # so every constructor path (including _new_grid derivatives whose
        # complex interpolation naturally returns np.angle's signed range)
        # remains internally consistent without changing the complex field.
        phase_wrap = str(self.units.get("phase_wrap", "")).strip()
        if phase_wrap:
            if phase_wrap not in {"0_360", "-180_180"}:
                raise ValueError(
                    "phase_wrap must be '0_360' or '-180_180' when declared"
                )
            # Work entirely in place.  _clean_phase has already converted
            # nonfinite values to NaN, and NumPy modulo preserves NaN, so a
            # grid-sized finite mask and masked-value temporary are unnecessary.
            with np.errstate(invalid="ignore"):
                if phase_wrap == "0_360":
                    np.remainder(self.rcs_phase, 2.0 * np.pi, out=self.rcs_phase)
                else:
                    np.add(self.rcs_phase, np.pi, out=self.rcs_phase)
                    np.remainder(self.rcs_phase, 2.0 * np.pi, out=self.rcs_phase)
                    np.subtract(self.rcs_phase, np.pi, out=self.rcs_phase)

        # Migrate supported legacy/fallback angular metadata into the modeled
        # units dictionary.  Derived grids copy units, whereas arbitrary extra
        # arrays are intentionally not propagated, so leaving physical tags
        # only in extra would let a transform silently turn GC data into conic.
        unit_coordinate = self.units.get("angular_coordinate_system")
        extra_coordinate = self.extra.get("angular_coordinate_system")
        if (
            (unit_coordinate is None or str(unit_coordinate).strip() == "")
            and extra_coordinate is not None
            and str(extra_coordinate).strip() != ""
        ):
            self.units["angular_coordinate_system"] = (
                canonical_angular_coordinate_system(extra_coordinate)
            )
        if self.angular_coordinate_system() == "great_circle":
            self.units.setdefault(
                "great_circle_coordinate_convention",
                self.great_circle_coordinate_convention(),
            )
            roll, tilt = self.angular_frame_orientation_deg()
            self.units.setdefault("angular_roll_deg", roll)
            self.units.setdefault("angular_tilt_deg", tilt)

    @staticmethod
    def _clean_power(power_value):
        dtype = _real_storage_dtype(power_value)
        power = np.asarray(power_value, dtype=dtype)
        finite = np.isfinite(power)
        out = np.full(power.shape, np.nan, dtype=dtype)
        out[finite] = np.maximum(power[finite], 0.0)
        return out

    @staticmethod
    def _clean_phase(phase_value):
        dtype = _real_storage_dtype(phase_value)
        phase = np.array(phase_value, dtype=dtype, copy=True)
        phase[~np.isfinite(phase)] = np.nan
        return phase

    @staticmethod
    def _complex_from_power_phase(power_value, phase_value):
        real_dtype = _real_storage_dtype(power_value, phase_value)
        complex_dtype = np.complex128 if real_dtype == np.float64 else np.complex64
        power = np.asarray(power_value, dtype=real_dtype)
        phase = np.asarray(phase_value, dtype=real_dtype)
        if power.shape != phase.shape:
            raise ValueError(f"power/phase shapes {power.shape}/{phase.shape} do not match")
        out = np.full(power.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
        valid = np.isfinite(power) & np.isfinite(phase)
        if np.any(valid):
            out[valid] = (
                np.sqrt(power[valid]) * np.exp(1j * phase[valid])
            ).astype(complex_dtype)
        return out

    def _complete_authoritative_raw_arrays(self):
        """Return a complete raw pair, or ``None`` for malformed/partial data."""

        real = self.extra.get("rcs_amp_real")
        imag = self.extra.get("rcs_amp_imag")
        if real is None or imag is None:
            return None
        real = np.asarray(real)
        imag = np.asarray(imag)
        if real.shape != self.rcs_power.shape or imag.shape != self.rcs_power.shape:
            return None
        # A raw pair is all-or-nothing authority.  Missing raw data must not
        # turn an otherwise valid power/phase cell into NaN, and stray raw data
        # must not resurrect a response cell explicitly marked missing.
        # Validate in bounded flat blocks.  The former whole-grid masks could
        # transiently reserve three extra bytes per sample, and callers such as
        # content hashing may validate very large solver grids repeatedly.
        for start in range(
            0, self.rcs_power.size, _RAW_COMPLEX_VALIDATION_BLOCK_CELLS
        ):
            stop = min(
                self.rcs_power.size,
                start + _RAW_COMPLEX_VALIDATION_BLOCK_CELLS,
            )
            # ``flat`` produces at most one bounded copy even for a transformed
            # negative-stride passthrough array; ``reshape`` could silently copy
            # the entire non-contiguous field before validation began.
            modeled = np.isfinite(self.rcs_power.flat[start:stop])
            raw_finite = np.isfinite(real.flat[start:stop])
            raw_finite &= np.isfinite(imag.flat[start:stop])
            if np.any(modeled != raw_finite):
                return None
        return real, imag

    def _drop_malformed_raw_metadata(self, extra):
        """Remove a partial raw pair before it can become derived authority."""

        if self._complete_authoritative_raw_arrays() is None:
            for key in (
                "rcs_amp_real",
                "rcs_amp_imag",
                "raw_complex_amplitude_preserved",
            ):
                extra.pop(key, None)
        return extra

    def _authoritative_raw_amplitude_from_pair(self, pair, selection=None):
        """Normalize one previously validated raw real/imaginary pair."""

        real, imag = pair
        quantity = self.linear_quantity()
        if selection is None:
            real_values = real.astype(np.float64, copy=False)
            imag_values = imag.astype(np.float64, copy=False)
        else:
            real_values = real[selection].astype(np.float64, copy=False)
            imag_values = imag[selection].astype(np.float64, copy=False)
        raw = real_values + 1j * imag_values
        if quantity == "sigma_2d":
            freq_hz = self._frequency_value_to_hz(self.frequencies)
            k0 = (2.0 * np.pi * np.asarray(freq_hz, dtype=float)) / C0
            if np.any(~np.isfinite(k0)) or np.any(k0 <= 0.0):
                return None
            scale = 1.0 / (2.0 * np.sqrt(k0))
            if selection is None:
                scale = scale[None, None, :, None]
            else:
                # Apply exactly the same NumPy indexing semantics to the
                # frequency normalization as to the raw response.  Indexing
                # only the 1-D frequency vector is incorrect for a retained
                # 4-D slice: NumPy then broadcasts frequency along the final
                # polarization axis.  A zero-stride broadcast view costs no
                # grid-sized storage for basic slices and also gives advanced
                # selections (including np.ix_) the exact response shape.
                scale = np.broadcast_to(
                    scale[None, None, :, None], self.rcs_power.shape
                )[selection]
            return raw * scale
        if quantity == "sigma_3d":
            return raw * np.sqrt(4.0 * np.pi)
        return None

    def _authoritative_raw_amplitude(self, selection=None):
        """Return a solver-provided raw field when its normalization is known."""

        pair = self._complete_authoritative_raw_arrays()
        if pair is None:
            return None
        return self._authoritative_raw_amplitude_from_pair(pair, selection)

    def _bounded_complex_slice_reader(self):
        """Return a bounded field reader and its real precision.

        The authoritative raw-pair contract is validated once when the reader
        is created, instead of rescanning the entire grid for every azimuth
        block.  Every returned slice is newly allocated and may be used as an
        in-place arithmetic work buffer by internal dense operations.
        """

        pair = self._complete_authoritative_raw_arrays()
        quantity = self.linear_quantity()
        if pair is not None and quantity in {"sigma_2d", "sigma_3d"}:
            if quantity == "sigma_2d":
                freq_hz = self._frequency_value_to_hz(self.frequencies)
                k0 = (2.0 * np.pi * np.asarray(freq_hz, dtype=float)) / C0
                if np.any(~np.isfinite(k0)) or np.any(k0 <= 0.0):
                    pair = None
            if pair is not None:
                return (
                    lambda selection: self._authoritative_raw_amplitude_from_pair(
                        pair, selection
                    ),
                    np.dtype(np.float64),
                )
        real_dtype = np.dtype(
            _real_storage_dtype(self.rcs_power, self.rcs_phase)
        )
        return (
            lambda selection: self._complex_from_power_phase(
                self.rcs_power[selection], self.rcs_phase[selection]
            ),
            real_dtype,
        )

    @property
    def rcs(self):
        """Complex RCS values derived from stored linear power and phase."""
        authoritative = self._authoritative_raw_amplitude()
        if authoritative is not None:
            return authoritative
        return self._complex_from_power_phase(self.rcs_power, self.rcs_phase)

    def rcs_slice(self, selection):
        """Reconstruct only a requested complex slice, avoiding a whole-grid allocation."""
        authoritative = self._authoritative_raw_amplitude(selection)
        if authoritative is not None:
            return authoritative
        return self._complex_from_power_phase(
            self.rcs_power[selection], self.rcs_phase[selection]
        )

    def __len__(self):
        """Return total number of complex samples in the grid."""
        return self.rcs_power.size

    def get(self, az_idx, el_idx, f_idx, p_idx):
        """Fetch a single sample by axis indices.

        Args:
            az_idx: Azimuth index.
            el_idx: Elevation index.
            f_idx: Frequency index.
            p_idx: Polarization index.

        Returns:
            dict with axis values and complex RCS sample.
        """
        return {
            "azimuth": self.azimuths[az_idx],
            "elevation": self.elevations[el_idx],
            "frequency": self.frequencies[f_idx],
            "polarization": self.polarizations[p_idx],
            "rcs": self.rcs_slice((az_idx, el_idx, f_idx, p_idx)),
        }

    def get_axis(self, name):
        """Return a single axis array by name.

        Use when you need a specific axis without unpacking all axes.

        Args:
            name: One of "azimuth", "elevation", "frequency", "polarization".

        Returns:
            Numpy array for the requested axis.
        """
        if name == "azimuth":
            return self.azimuths
        if name == "elevation":
            return self.elevations
        if name == "frequency":
            return self.frequencies
        if name == "polarization":
            return self.polarizations
        raise ValueError(f"unknown axis name: {name}")

    def get_axes(self):
        """Return all axis arrays in a dict."""
        return {
            "azimuths": self.azimuths,
            "elevations": self.elevations,
            "frequencies": self.frequencies,
            "polarizations": self.polarizations,
        }

    def audit(self):
        """Return a non-mutating, JSON-serializable dataset health report.

        The report always contains ``status``, ``errors``, ``warnings``,
        ``info``, and ``metrics``.  Grid samples are scanned in bounded blocks;
        the audit never constructs a second full-size power, phase, or complex
        grid.  This method is deliberately diagnostic: it reports malformed
        public mutations instead of repairing them.
        """

        from grim_dataset_audit import audit_dataset
        return audit_dataset(self)

    def edit_axis_value(self, name, index, value):
        """Return a grid with one safely edited axis value.

        Numeric coordinate edits are kept finite and unique, then the edited
        axis is stable-sorted.  Every sample array follows the same permutation,
        including passthrough arrays whose leading four dimensions match the
        RCS grid.  Polarization edits are label-only: surrounding whitespace is
        removed, blank labels and case-insensitive duplicates are rejected, and
        channel order is preserved.

        The operation is transactional: validation and all reordered arrays are
        prepared before a new :class:`RcsGrid` is returned, so ``self`` is never
        partially mutated when an edit is invalid.
        """

        axis_specs = {
            "azimuth": ("azimuths", 0),
            "elevation": ("elevations", 1),
            "frequency": ("frequencies", 2),
            "polarization": ("polarizations", 3),
        }
        try:
            attribute, axis_index = axis_specs[str(name).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unknown axis name: {name}") from exc

        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("axis index must be an integer")
        item_index = int(index)
        source_axis = np.asarray(getattr(self, attribute))
        if item_index < 0 or item_index >= source_axis.size:
            raise IndexError(
                f"{name} axis index {item_index} is outside 0..{source_axis.size - 1}"
            )

        axes = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        order = np.arange(source_axis.size, dtype=int)
        reordered = False

        if axis_index == 3:
            old_value = str(source_axis[item_index])
            new_value = str(value).strip().upper()
            if not new_value:
                raise ValueError("polarization label must not be blank")
            duplicate_key = new_value.casefold()
            if any(
                str(label).strip().casefold() == duplicate_key
                for candidate_index, label in enumerate(source_axis)
                if candidate_index != item_index
            ):
                raise ValueError(
                    f"polarization label {new_value!r} duplicates another channel"
                )
            if new_value == old_value:
                return self
            # Building a fresh unicode array is important here: assigning a
            # longer label into an existing fixed-width dtype (for example
            # '<U2') would silently truncate it.
            labels = [str(label) for label in source_axis.tolist()]
            labels[item_index] = new_value
            axes[axis_index] = np.asarray(labels, dtype=str)
            old_text = repr(old_value)
            new_text = repr(new_value)
        else:
            old_value = float(source_axis[item_index])
            try:
                new_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} axis value must be numeric") from exc
            if not np.isfinite(new_value):
                raise ValueError(f"{name} axis value must be finite")
            if axis_index == 2 and new_value <= 0.0:
                raise ValueError("frequency axis value must be greater than zero")
            if new_value == old_value:
                return self

            edited_axis = np.asarray(source_axis, dtype=float).copy()
            edited_axis[item_index] = new_value
            if np.unique(edited_axis).size != edited_axis.size:
                raise ValueError(
                    f"{name} axis value {new_value:g} duplicates another coordinate"
                )
            order = np.argsort(edited_axis, kind="stable")
            reordered = not np.array_equal(
                order, np.arange(edited_axis.size, dtype=int)
            )
            axes[axis_index] = edited_axis[order]
            old_text = f"{old_value:g}"
            new_text = f"{new_value:g}"

        if reordered:
            power = np.take(self.rcs_power, order, axis=axis_index)
            phase = np.take(self.rcs_phase, order, axis=axis_index)
        else:
            # The internal adoption token below is only safe for arrays owned
            # by this result.  A metadata-only/non-reordering edit used to
            # hand the new grid the source object's public mutable arrays,
            # allowing a later mutation of either dataset to change both.
            # Copy once here, then transfer those fresh buffers into the
            # constructor without its usual second sanitation copy.
            power = np.array(self.rcs_power, copy=True)
            phase = np.array(self.rcs_phase, copy=True)

        original_shape = tuple(self.rcs_power.shape)
        stale_grid_metadata = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
        }
        if axis_index == 3:
            stale_grid_metadata.update(
                {"polarization_alias_primary", "polarization_aliases_json"}
            )

        edited_extra = {}
        for key, extra_value in self.extra.items():
            if key in stale_grid_metadata:
                continue
            if reordered:
                extra_array = np.asarray(extra_value)
                if (
                    extra_array.ndim >= 4
                    and tuple(extra_array.shape[:4]) == original_shape
                ):
                    edited_extra[key] = np.take(
                        extra_array, order, axis=axis_index
                    )
                    continue
            # RcsGrid treats passthrough values as immutable.  Sharing an
            # unchanged value keeps a metadata-only edit from duplicating
            # potentially multi-gigabyte embedded body-model arrays; the
            # containing ``extra`` dictionary itself is still new.
            edited_extra[key] = extra_value
        self._drop_malformed_raw_metadata(edited_extra)
        self._invalidate_assembly_sampling_hash(
            edited_extra, f"edit-{str(name).strip().lower()}-axis"
        )
        if axis_index in (0, 1):
            edited_extra.pop("assembly_angular_coordinate_contract", None)

        history_entry = (
            f"Edit {name} axis[{item_index}]: {old_text} -> {new_text}"
        )
        if reordered:
            history_entry += "; stable-sorted axis and sample arrays"
        prior_history = str(self.history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )
        return RcsGrid(
            axes[0],
            axes[1],
            axes[2],
            axes[3],
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=copy.deepcopy(self.units),
            extra=edited_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @staticmethod
    def _canonical_unit(value, aliases, default):
        text = str(value or default).strip().lower()
        return aliases.get(text, text)

    def _supported_unit(self, axis_name, aliases, default):
        """Return one canonical modeled unit, rejecting explicit unknowns.

        Older GRIM files omitted unit metadata and historically used degrees
        and GHz, so a missing/blank value still receives that documented
        default.  A present but unknown unit must not quietly take the same
        conversion path as the default.
        """

        raw = (self.units or {}).get(axis_name)
        canonical = self._canonical_unit(raw, aliases, default)
        supported = set(aliases.values())
        if canonical not in supported:
            raise ValueError(
                f"unsupported {axis_name} unit {raw!r}; expected one of "
                + ", ".join(sorted(supported))
            )
        return canonical

    def _angle_value_from_degrees(self, value, axis_name):
        """Convert a degree-valued operation argument to an axis's unit."""

        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{axis_name} angle must be finite")
        unit = self._supported_unit(axis_name, _ANGLE_UNITS, "deg")
        return float(np.deg2rad(numeric)) if unit == "rad" else numeric

    def inspect_scalar_metadata(self, key):
        """Expose metadata evidence without changing numerical eligibility."""
        return inspect_scalar_metadata(
            key, self.units, self.extra,
            canonicalizer=(
                self._canonical_time_convention if key == "time_convention" else None
            ),
        )

    def _declared_scalar_metadata(self, key):
        return self.inspect_scalar_metadata(key).scalar(
            advisory=key in ADVISORY_METADATA_KEYS
        )

    _canonical_time_convention = staticmethod(canonical_time_convention)

    def linear_quantity(self):
        """Physical meaning of ``rcs_power`` (sigma_2d, sigma_3d, or ratio)."""
        raw = str((self.units or {}).get("rcs_linear_quantity", "")).strip().lower()
        if raw:
            return raw
        return "sigma_2d" if self.default_log_unit().lower() == "dbke" else "sigma_3d"

    def _phase_reference(self):
        return self._declared_scalar_metadata("phase_reference")

    def angular_coordinate_system(self):
        """Return the physical angular convention used by the two angle axes.

        GRIM's native convention is conic azimuth/elevation.  Legacy PTM cuts
        use great-circle aspect/pitch; identical numeric axes from those two
        conventions are not physically interchangeable.
        """
        raw = (self.units or {}).get("angular_coordinate_system")
        if raw is None or str(raw).strip() == "":
            raw = (self.extra or {}).get("angular_coordinate_system", "")
        return canonical_angular_coordinate_system(raw)

    def angular_frame_orientation_deg(self):
        """Return stored great-circle/PTM roll and tilt scalar metadata.

        They remain compatibility fields.  The supplied legacy PTM reference
        does not define their Euler order or enough geometry to apply them as
        rotations, so conversion code must require both to be zero.
        """

        values = []
        for unit_key, extra_key in (
            ("angular_roll_deg", "ptm_roll"),
            ("angular_tilt_deg", "ptm_tilt"),
        ):
            raw = (self.units or {}).get(unit_key)
            if raw is None or str(raw).strip() == "":
                raw = (self.extra or {}).get(extra_key, 0.0)
            array = np.asarray(raw)
            if array.size != 1:
                raise ValueError(f"{unit_key} must be scalar")
            value = float(array.reshape(-1)[0])
            if not np.isfinite(value):
                raise ValueError(f"{unit_key} must be finite")
            values.append(value)
        return tuple(values)

    def set_angular_coordinate_system(
        self, coordinate_system, *,
        gc_convention=LEGACY_PTM_GC_CONVENTION, roll_deg=0.0, tilt_deg=0.0,
    ):
        """Declare the meaning of existing angles and return an independent copy.

        This explicitly overrides import assumptions; it is not a geometric
        conversion. Numeric axes, sample order, polarizations, power, and phase
        are preserved exactly, including nonzero cuts and cross-polar channels.
        Great-circle declarations also specify their convention and frame.
        """
        target = canonical_angular_coordinate_system(coordinate_system)
        if not str(coordinate_system or "").strip() or target not in {
            "conic", "great_circle"
        }:
            raise ValueError("coordinate_system must be conic or great_circle")
        if target == "great_circle":
            gc_convention = str(gc_convention).strip().lower()
            if gc_convention not in {LEGACY_PTM_GC_CONVENTION, GRIM_GC_CONVENTION}:
                raise ValueError("unsupported great-circle convention")
            roll_deg, tilt_deg = float(roll_deg), float(tilt_deg)
            if not np.isfinite([roll_deg, tilt_deg]).all():
                raise ValueError("great-circle roll and tilt must be finite")

        source_system = self.angular_coordinate_system()
        declaration = {
            "schema": "grim.angular-coordinate-declaration.v1",
            "source_system": source_system,
            "source_gc_convention": (
                self.great_circle_coordinate_convention()
                if source_system == "great_circle" else None
            ),
            "source_orientation_deg": self.angular_frame_orientation_deg(),
            "target_system": target,
            "numeric_data_changed": False,
        }
        # A pure declaration must not reclean arrays, wrap phase, or reorder
        # coordinates through the constructor. Own the copied arrays so later
        # edits cannot mutate the imported source.
        result = copy.deepcopy(self)
        for container in (result.units, result.extra):
            for key in (
                "great_circle_coordinate_convention", "angular_roll_deg",
                "angular_tilt_deg", "ptm_roll", "ptm_tilt", "ptm_cut_type",
                "ptm_cut_type_source", "elevation_coordinate_convention",
                "sentri_elevation_convention", "sentri_coordinate_mapping",
                "assembly_angular_coordinate_contract",
            ):
                container.pop(key, None)
            container["angular_coordinate_system"] = target
        if target == "great_circle":
            for container in (result.units, result.extra):
                container["great_circle_coordinate_convention"] = gc_convention
            result.units.update(angular_roll_deg=roll_deg, angular_tilt_deg=tilt_deg)
            declaration.update(
                gc_convention=gc_convention, roll_deg=roll_deg, tilt_deg=tilt_deg
            )
        for key in (
            "solver_metadata_json", "production_mesh_certification_json",
            "source_body_mesh_certification_json",
        ):
            result.extra.pop(key, None)
        self._invalidate_assembly_sampling_hash(result.extra, "set-angular-coordinates")
        result.extra["angular_coordinate_declaration_json"] = json.dumps(
            declaration, sort_keys=True, separators=(",", ":")
        )
        label = "azimuth/elevation (conic)" if target == "conic" else (
            f"aspect/pitch (great_circle; {gc_convention}; "
            f"roll={roll_deg:g}, tilt={tilt_deg:g} deg)"
        )
        entry = (
            f"User declared coordinates: {source_system} -> {label}; "
            "numeric axes and samples unchanged; no coordinate conversion"
        )
        result.history = f"{self.history}\n{entry}" if self.history else entry
        return result

    def great_circle_coordinate_convention(self):
        """Return the declared great-circle chart/basis convention.

        A GRIM-created equatorial great-circle grid is tagged ``grim_gc_v1``.
        An imported PTM without GRIM's marker is deliberately reported as
        ``legacy_ptm_unspecified`` because its byte header does not establish
        aspect sign/origin or the polarization basis.
        """

        raw = (self.units or {}).get("great_circle_coordinate_convention")
        if raw is None or str(raw).strip() == "":
            raw = (self.extra or {}).get(
                "great_circle_coordinate_convention", ""
            )
        text = str(raw or "").strip().lower().replace("-", "_")
        aliases = {
            "": LEGACY_PTM_GC_CONVENTION,
            "grim": GRIM_GC_CONVENTION,
            "grim_gc": GRIM_GC_CONVENTION,
            "legacy": LEGACY_PTM_GC_CONVENTION,
            "unknown": LEGACY_PTM_GC_CONVENTION,
            "unspecified": LEGACY_PTM_GC_CONVENTION,
        }
        return aliases.get(text, text)

    def convert_wedge_to_conic(
        self,
        *,
        attest_wedge_axes=False,
        assume_missing_cross_pol_zero=False,
    ):
        """Convert a vertical-turntable/body-wedge acquisition to conic V/H.

        This produces the normal-range grid for a pylon/article assembly that
        is tilted together and then rotated.  Direction queries are inverse-
        mapped into the measured ``(turntable phi, body wedge tau)`` grid,
        interpolated as a full complex Jones matrix, and congruence-rotated
        into the conic spherical V/H basis. Unsupported parts of the normal
        conic grid remain NaN; they are never extrapolated.

        A single wedge tilt is only a curved one-dimensional cut and cannot
        determine a constant-elevation normal azimuth cut, so at least two
        measured wedge tilts and a complete turntable revolution are required.
        """

        declared_source_system = self._declared_scalar_metadata(
            "angular_coordinate_system"
        )
        source_system = self.angular_coordinate_system()
        if source_system == "great_circle":
            raise ValueError(
                "Wedge-to-Conic requires turntable-angle/wedge-tilt axes, not "
                "a great-circle dataset"
            )
        if declared_source_system and source_system != "wedge_turntable":
            raise ValueError(
                "Wedge-to-Conic cannot override an explicit non-wedge angular "
                f"coordinate system {declared_source_system!r}"
            )
        assumed_wedge_axes = not bool(declared_source_system)

        az_unit = self._canonical_unit(
            (self.units or {}).get("azimuth"), _ANGLE_UNITS, "deg"
        )
        el_unit = self._canonical_unit(
            (self.units or {}).get("elevation"), _ANGLE_UNITS, "deg"
        )
        if az_unit not in {"deg", "rad"} or el_unit not in {"deg", "rad"}:
            raise ValueError(
                "Wedge-to-Conic requires degree or radian angle axes; got "
                f"azimuth={az_unit!r}, elevation={el_unit!r}"
            )
        phi = np.asarray(self.azimuths, dtype=float)
        tau = np.asarray(self.elevations, dtype=float)
        if phi.size < 4 or not np.all(np.isfinite(phi)):
            raise ValueError(
                "Wedge-to-Conic requires at least four finite turntable angles"
            )
        if tau.size < 2 or not np.all(np.isfinite(tau)):
            raise ValueError(
                "One fixed wedge tilt traces a curved cut and cannot be "
                "converted into a normal constant-elevation azimuth sweep. "
                "Supply at least two measured wedge tilts."
            )
        phi_deg = np.rad2deg(phi) if az_unit == "rad" else phi
        tau_deg = np.rad2deg(tau) if el_unit == "rad" else tau
        if np.any(np.abs(tau_deg) >= 90.0 - 1.0e-9):
            raise ValueError(
                "Wedge-to-Conic requires body wedge tilts strictly between "
                "-90 and +90 degrees"
            )

        wrapped_phi = np.mod(phi_deg + 180.0, 360.0) - 180.0
        wrapped_phi[np.abs(wrapped_phi) <= 1.0e-12] = 0.0
        phi_order = np.argsort(wrapped_phi, kind="stable")
        wrapped_phi = wrapped_phi[phi_order]
        if np.any(np.diff(wrapped_phi) <= 1.0e-9):
            raise ValueError(
                "Wedge turntable axis contains duplicate or seam-alias angles"
            )
        circular_gaps = np.diff(
            np.concatenate((wrapped_phi, [wrapped_phi[0] + 360.0]))
        )
        typical_gap = float(np.median(circular_gaps))
        if (
            not np.isfinite(typical_gap)
            or typical_gap <= 0.0
            or float(np.max(circular_gaps)) > 2.5 * typical_gap + 1.0e-7
        ):
            raise ValueError(
                "Wedge-to-Conic normal-azimuth conversion requires a complete "
                "turntable revolution without a large unmeasured angular gap"
            )

        tau_order = np.argsort(tau_deg, kind="stable")
        tau_sorted = tau_deg[tau_order]
        if np.any(np.diff(tau_sorted) <= 1.0e-9):
            raise ValueError("Wedge tilt axis contains duplicate coordinates")
        if np.any(
            np.isfinite(self.rcs_power) & ~np.isfinite(self.rcs_phase)
        ):
            raise ValueError(
                "Wedge-to-Conic Jones rotation requires finite phase for every "
                "finite polarization sample; power-only data cannot be rotated"
            )

        source_field = np.asarray(self.rcs)[phi_order, ...][:, tau_order, ...]
        source_jones, labels, cross_note = _jones_from_polarization_channels(
            source_field,
            self.polarizations,
            assume_missing_cross_pol_zero=assume_missing_cross_pol_zero,
        )

        # Periodic interpolation in measured wedge coordinates.  The first
        # turntable slice is repeated at +360; no unmeasured sector is bridged
        # because the complete-revolution gate above has already passed.
        from scipy.interpolate import RegularGridInterpolator

        phi_interp = np.concatenate((wrapped_phi, [wrapped_phi[0] + 360.0]))
        jones_interp = np.concatenate(
            (source_jones, source_jones[:1, ...]), axis=0
        )
        interpolator = RegularGridInterpolator(
            (phi_interp, tau_sorted),
            jones_interp,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        target_lon = np.mod(-wrapped_phi + 180.0, 360.0) - 180.0
        target_lon[np.abs(target_lon) <= 1.0e-12] = 0.0
        target_lon = np.sort(target_lon, kind="stable")
        target_lat = np.array(tau_sorted, copy=True)
        lon_mesh, lat_mesh = np.meshgrid(target_lon, target_lat, indexing="ij")
        query_phi, query_tau = conic_to_wedge_geometry_deg(lon_mesh, lat_mesh)
        query_phi = (
            np.mod(query_phi - wrapped_phi[0], 360.0) + wrapped_phi[0]
        )
        query = np.column_stack((query_phi.ravel(), query_tau.ravel()))
        wedge_jones = interpolator(query)
        change = wedge_to_conic_basis_change(query[:, 0], query[:, 1])
        conic_jones = np.einsum(
            "qia,qfij,qjb->qfab", change, wedge_jones, change
        )
        conic_channels = _polarization_channels_from_jones(
            conic_jones, labels
        )
        output_shape = (
            target_lon.size,
            target_lat.size,
            self.frequencies.size,
            self.polarizations.size,
        )
        conic_channels = conic_channels.reshape(output_shape)

        converted_units = copy.deepcopy(self.units or {})
        converted_units["angular_coordinate_system"] = "conic"
        converted_units["polarization_basis"] = CONIC_VH_BASIS_CONVENTION
        converted_units.pop("wedge_coordinate_convention", None)
        output_lon = np.deg2rad(target_lon) if az_unit == "rad" else target_lon
        output_lat = np.deg2rad(target_lat) if el_unit == "rad" else target_lat

        converted_extra = {}
        original_shape = tuple(self.rcs_power.shape)
        stale = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
            "rcs_amp_real",
            "rcs_amp_imag",
        }
        for key, value in (self.extra or {}).items():
            if key in stale:
                continue
            array = np.asarray(value)
            if array.ndim >= 4 and tuple(array.shape[:4]) == original_shape:
                continue
            converted_extra[key] = copy.deepcopy(value)
        converted_extra.update(
            {
                "source_angular_coordinate_system": "wedge_turntable",
                "wedge_coordinate_convention": WEDGE_TURNTABLE_CONVENTION,
                "polarization_basis": CONIC_VH_BASIS_CONVENTION,
                "wedge_to_conic_cross_pol_treatment": cross_note,
            }
        )
        if assumed_wedge_axes:
            converted_extra["wedge_axes_assumption_json"] = json.dumps(
                {
                    "schema": "grim.wedge-axes-assumption.v1",
                    "operation_requested": True,
                    "source_coordinate_declaration_missing": True,
                    "assumed_axes": WEDGE_TURNTABLE_CONVENTION,
                    "legacy_user_attested": bool(attest_wedge_axes),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        geometric_supported = (
            (query_tau >= tau_sorted[0] - 1.0e-9)
            & (query_tau <= tau_sorted[-1] + 1.0e-9)
        )
        coverage = 100.0 * float(np.count_nonzero(geometric_supported)) / float(
            geometric_supported.size
        )
        history_entry = (
            "Wedge->Conic physical regrid: inverse direction map; complex "
            f"Jones C^T*S*C rotation ({cross_note}); no extrapolation; "
            f"normal-grid geometric coverage {coverage:.1f}%"
        )
        if assumed_wedge_axes:
            history_entry += "; untagged source axes assumed from requested operation"
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return RcsGrid(
            output_lon,
            output_lat,
            self.frequencies,
            self.polarizations,
            rcs=conic_channels,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=converted_units,
            extra=converted_extra,
        )

    def convert_equatorial_conic_gc(
        self,
        direction,
        *,
        attest_legacy_ptm_convention=False,
    ):
        """Losslessly change the angular tag for the exact zero-plane case.

        No direction interpolation or polarization rotation occurs.  Under
        GRIM's declared convention, signed great-circle aspect equals conic
        azimuth at zero pitch and the V/H bases coincide.  General nonzero-cut
        conversion is intentionally unsupported because its path is curved in
        conic coordinates and it requires a full complex scattering-matrix
        basis rotation.

        Selecting the conversion for an unmarked legacy PTM records the GRIM
        aspect/basis convention as an operation assumption. Explicitly tagged
        incompatible great-circle conventions remain unsupported.
        """

        direction = str(direction or "").strip().lower()
        if direction not in {"conic_to_gc", "gc_to_conic"}:
            raise ValueError(
                "direction must be 'conic_to_gc' or 'gc_to_conic'"
            )
        source_system = self.angular_coordinate_system()
        expected_source = "conic" if direction == "conic_to_gc" else "great_circle"
        if source_system not in {"conic", "great_circle"}:
            raise ValueError(
                "equatorial Conic/GC conversion does not support angular "
                f"coordinate system {source_system!r}"
            )
        if source_system != expected_source:
            arrow = "Conic→GC" if direction == "conic_to_gc" else "GC→Conic"
            raise ValueError(
                f"{arrow} requires a source tagged {expected_source}; got "
                f"{source_system}"
            )

        az_unit = self._canonical_unit(
            (self.units or {}).get("azimuth"), _ANGLE_UNITS, "deg"
        )
        el_unit = self._canonical_unit(
            (self.units or {}).get("elevation"), _ANGLE_UNITS, "deg"
        )
        if az_unit not in {"deg", "rad"} or el_unit not in {"deg", "rad"}:
            raise ValueError(
                "equatorial Conic/GC conversion requires degree or radian "
                f"angle axes; got azimuth={az_unit!r}, elevation={el_unit!r}"
            )
        azimuths = np.asarray(self.azimuths, dtype=float)
        elevations = np.asarray(self.elevations, dtype=float)
        if azimuths.size == 0 or not np.all(np.isfinite(azimuths)):
            raise ValueError("equatorial Conic/GC conversion needs a finite aspect axis")
        if elevations.size != 1 or not np.all(np.isfinite(elevations)):
            raise ValueError("exact Conic/GC conversion requires exactly one finite cut")
        elevation_deg = (
            np.rad2deg(elevations) if el_unit == "rad" else elevations
        )
        if not np.isclose(elevation_deg[0], 0.0, rtol=0.0, atol=1.0e-7):
            label = "elevation" if source_system == "conic" else "pitch"
            raise ValueError(
                f"exact Conic/GC conversion requires one 0 degree {label} cut"
            )

        roll, tilt = self.angular_frame_orientation_deg()
        if not np.allclose((roll, tilt), (0.0, 0.0), rtol=0.0, atol=1.0e-7):
            raise ValueError(
                "exact Conic/GC conversion requires stored roll=tilt=0 "
                f"degrees; got roll={roll:g}, tilt={tilt:g}"
            )
        polarizations = [
            str(value).strip().upper() for value in self.polarizations
        ]
        unsupported = sorted(set(polarizations) - {"VV", "HH"})
        if unsupported:
            raise ValueError(
                "exact Conic/GC conversion currently supports VV/HH only; "
                "legacy PTM cross-polar basis signs are unspecified; got "
                + ", ".join(unsupported)
            )

        if direction == "gc_to_conic":
            convention = self.great_circle_coordinate_convention()
            if convention != GRIM_GC_CONVENTION:
                if convention != LEGACY_PTM_GC_CONVENTION:
                    raise ValueError(
                        "unsupported great-circle coordinate convention "
                        f"{convention!r}; only GRIM_GC_V1 or an unmarked "
                        "legacy PTM is supported"
                    )
                convention_note = "unmarked legacy PTM assumed GRIM_GC_V1"
            else:
                convention_note = "declared GRIM_GC_V1"
        else:
            convention_note = "created with GRIM_GC_V1"

        # Canonicalize the periodic primary axis while carrying every sample
        # and any grid-shaped passthrough array through the same stable order.
        period = 2.0 * np.pi if az_unit == "rad" else 360.0
        half_period = 0.5 * period
        wrapped = np.mod(azimuths + half_period, period) - half_period
        wrapped[np.isclose(wrapped, 0.0, rtol=0.0, atol=1.0e-12)] = 0.0
        order = np.argsort(wrapped, kind="stable")
        wrapped = wrapped[order]
        tolerance = np.deg2rad(1.0e-7) if az_unit == "rad" else 1.0e-7
        if wrapped.size > 1 and np.any(np.diff(wrapped) <= tolerance):
            raise ValueError(
                "aspect axis contains duplicate or seam-alias directions after wrapping"
            )

        expected_shape = self.rcs_power.shape
        converted_extra = {}
        for key, value in self._extra_to_write().items():
            array = np.asarray(value)
            if array.ndim >= 4 and array.shape[:4] == expected_shape:
                converted_extra[key] = np.array(array[order, ...], copy=True)
            else:
                converted_extra[key] = copy.deepcopy(value)
        # A coordinate-derived file is not the original solver artifact.  Do
        # not carry a stale grid-bound attestation/certification through the
        # axis normalization even though those scalar blobs fit structurally.
        for key in (
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
        ):
            converted_extra.pop(key, None)
        self._drop_malformed_raw_metadata(converted_extra)
        self._invalidate_assembly_sampling_hash(
            converted_extra, "convert-equatorial-conic-great-circle"
        )
        converted_extra.pop("assembly_angular_coordinate_contract", None)
        if (
            direction == "gc_to_conic"
            and self.great_circle_coordinate_convention()
            == LEGACY_PTM_GC_CONVENTION
        ):
            converted_extra["great_circle_conversion_assumption_json"] = json.dumps(
                {
                    "schema": "grim.great-circle-conversion-assumption.v1",
                    "operation_requested": True,
                    "source_convention_unmarked": True,
                    "assumed_convention": GRIM_GC_CONVENTION,
                    "legacy_user_attested": bool(attest_legacy_ptm_convention),
                },
                sort_keys=True,
                separators=(",", ":"),
            )

        converted_units = copy.deepcopy(self.units or {})
        if direction == "conic_to_gc":
            converted_units["angular_coordinate_system"] = "great_circle"
            converted_units["great_circle_coordinate_convention"] = GRIM_GC_CONVENTION
            converted_units["angular_roll_deg"] = 0.0
            converted_units["angular_tilt_deg"] = 0.0
            converted_extra["angular_coordinate_system"] = "great_circle"
            converted_extra["great_circle_coordinate_convention"] = GRIM_GC_CONVENTION
        else:
            converted_units["angular_coordinate_system"] = "conic"
            for key in (
                "great_circle_coordinate_convention",
                "angular_roll_deg",
                "angular_tilt_deg",
            ):
                converted_units.pop(key, None)
            for key in (
                "angular_coordinate_system",
                "great_circle_coordinate_convention",
                "ptm_cut_type",
                "ptm_roll",
                "ptm_tilt",
            ):
                converted_extra.pop(key, None)

        arrow = "Conic->GC" if direction == "conic_to_gc" else "GC->Conic"
        history_entry = (
            f"{arrow} exact equatorial relabel; no interpolation; "
            f"{convention_note}; VV/HH only"
        )
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return RcsGrid(
            wrapped,
            np.asarray([0.0], dtype=self.elevations.dtype),
            self.frequencies,
            self.polarizations,
            rcs=None,
            rcs_power=np.asarray(self.rcs_power)[order, ...],
            rcs_phase=np.asarray(self.rcs_phase)[order, ...],
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=converted_units,
            extra=converted_extra,
        )

    def _assert_axis_metadata_compatible(self, other):
        """Require coordinates to share units and the same angular frame.

        This is the compatibility contract for operations that only align or
        crop coordinates and keep each dataset's response values separate.
        Response quantity and logarithmic display metadata are intentionally
        irrelevant to those operations.
        """

        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        for key, aliases, default in (
            ("azimuth", _ANGLE_UNITS, "deg"),
            ("elevation", _ANGLE_UNITS, "deg"),
            ("frequency", _FREQUENCY_UNITS, "GHz"),
        ):
            left = self._supported_unit(key, aliases, default)
            right = other._supported_unit(key, aliases, default)
            if left != right:
                raise ValueError(f"{key} unit mismatch: {left} != {right}")
        left_angles = self.angular_coordinate_system()
        right_angles = other.angular_coordinate_system()
        if left_angles != right_angles:
            raise ValueError(
                "angular coordinate system mismatch: "
                f"{left_angles} != {right_angles}"
            )
        if left_angles == "great_circle":
            left_convention = self.great_circle_coordinate_convention()
            right_convention = other.great_circle_coordinate_convention()
            if left_convention != right_convention:
                raise ValueError(
                    "great-circle coordinate convention mismatch: "
                    f"{left_convention} != {right_convention}"
                )
            left_orientation = self.angular_frame_orientation_deg()
            right_orientation = other.angular_frame_orientation_deg()
            if not np.allclose(
                left_orientation, right_orientation, rtol=0.0, atol=1.0e-7
            ):
                raise ValueError(
                    "great-circle frame orientation mismatch: "
                    f"roll/tilt {left_orientation} != {right_orientation} deg"
                )

    def _assert_physical_metadata_compatible(self, other):
        self._assert_axis_metadata_compatible(other)
        if self.linear_quantity() != other.linear_quantity():
            raise ValueError(
                "RCS linear quantity mismatch: "
                f"{self.linear_quantity()} != {other.linear_quantity()}"
            )
        if self.default_log_unit().lower() != other.default_log_unit().lower():
            raise ValueError(
                f"RCS log unit mismatch: {self.default_log_unit()} != "
                f"{other.default_log_unit()}"
            )

    @staticmethod
    def _metadata_placeholder(value):
        words = set(
            re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split()
        )
        return bool(
            words.intersection(
                {"unknown", "unspecified", "undetermined", "unverified", "arbitrary"}
            )
        )

    def _coherent_source_convention_values(self, key):
        """Return explicit source declarations without promoting one-sided data."""

        values = []
        direct = self._declared_scalar_metadata(key)
        if direct and not self._metadata_placeholder(direct):
            values.append(direct)
        raw = (self.extra or {}).get("coherent_source_conventions_json")
        if raw is not None:
            try:
                if isinstance(raw, np.ndarray):
                    raw = raw.reshape(()).item()
                record = json.loads(str(raw))
                stored = record.get("declared_values", {}).get(key, [])
                if isinstance(stored, list):
                    values.extend(
                        str(value).strip()
                        for value in stored
                        if str(value).strip()
                        and not self._metadata_placeholder(value)
                    )
            except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
                # Malformed lineage is non-authoritative provenance. Direct
                # declarations still participate in the physical comparison.
                pass
        return values

    def _assert_coherent_metadata_compatible(
        self, other, *, metadata_attested=False
    ):
        """Return advisory convention differences; field operations stay usable."""

        if not isinstance(metadata_attested, (bool, np.bool_)):
            raise TypeError("metadata_attested must be True or False")
        issues = []
        if self.linear_quantity() == "sigma_2d":
            versions = [grid._declared_scalar_metadata("amplitude_version") for grid in (self, other)]
            if any(versions) and versions != ["2", "2"]:
                issues.append("2-D amplitude_version annotations differ or are unverified; using supplied complex samples.")
        fields = (
            (
                "phase_reference",
                "phase references",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "time_convention",
                "time conventions",
                self._canonical_time_convention,
            ),
            (
                "polarization_basis",
                "polarization bases",
                lambda value: " ".join(value.split()).casefold(),
            ),
            ("amplitude_convention", "amplitude conventions", lambda value: " ".join(value.split()).casefold()),
            ("complex_field_domain", "complex field domains", lambda value: " ".join(value.split()).casefold()),
        )
        for key, label, canonicalize in fields:
            left_values = self._coherent_source_convention_values(key)
            right_values = other._coherent_source_convention_values(key)
            normalized = {
                canonicalize(value) for value in (*left_values, *right_values)
            }
            if len(normalized) > 1:
                issues.append(
                    f"coherent operation uses supplied samples despite different {label} ({key}): "
                    f"{left_values or ['<unspecified>']!r} and "
                    f"{right_values or ['<unspecified>']!r}"
                )
        return issues

    def _assert_acquisition_metadata_compatible(
        self,
        other,
        *,
        operation_label,
        left_role,
        right_role,
        schema,
        excluded_families=(),
    ):
        """Compare acquisition declarations by physical meaning, not key name.

        Several importers use equivalent aliases (for example
        ``calibration_id`` and ``calibration_identifier``).  Treating each key
        independently lets crossed aliases evade comparison and also misses
        contradictions inside one dataset.  This helper consolidates every
        family into canonical semantic dimensions before comparing inputs.
        """

        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        operation = str(operation_label or "").strip()
        left_name = str(left_role or "").strip()
        right_name = str(right_role or "").strip()
        contract_schema = str(schema or "").strip()
        if not operation or not left_name or not right_name or not contract_schema:
            raise ValueError("acquisition metadata contract labels cannot be blank")
        if left_name == right_name:
            raise ValueError("acquisition metadata contract roles must be distinct")
        excluded = {str(value).strip() for value in tuple(excluded_families)}
        known_families = {
            family for family, _label, _kind, _keys in _ACQUISITION_METADATA_FAMILIES
        }
        unknown_exclusions = sorted(excluded - known_families)
        if unknown_exclusions:
            raise ValueError(
                "unknown acquisition metadata families excluded from contract: "
                + ", ".join(unknown_exclusions)
            )

        def normalized(value: str) -> str:
            return re.sub(
                r"[^a-z0-9]+", " ", str(value or "").strip().casefold()
            ).strip()

        def identity(value: str) -> str:
            # Punctuation can be significant in an externally assigned ID.
            return " ".join(str(value or "").strip().casefold().split())

        def canonical(key: str, value: str, kind: str, role: str) -> dict[str, str]:
            semantic = normalized(value)
            words = set(semantic.split())
            if words.intersection(
                {
                    "unknown",
                    "unspecified",
                    "undetermined",
                    "unverified",
                    "arbitrary",
                }
            ):
                # Foreign producers commonly serialize an explicit placeholder
                # instead of omitting an unavailable declaration. It carries no
                # physical assertion and is therefore treated as missing, not as
                # a contradiction.
                return {}

            if kind == "identity":
                return {"identity": identity(value)}
            if kind == "text":
                return {"value": semantic}
            if kind == "range_phase":
                compact = str(value).casefold()
                compact = compact.replace("−", "-").replace("–", "-")
                compact = re.sub(r"[\s*·^{}()\[\]_=~]+", "", compact)
                matches = re.findall(
                    r"(?:exp|e)([+-])j2(?:\.0+)?kr", compact
                )
                if "negativetwowayrangephase" in compact:
                    matches.append("-")
                if "positivetwowayrangephase" in compact:
                    matches.append("+")
                signs = {"negative" if match == "-" else "positive" for match in matches}
                if len(signs) > 1:
                    raise ValueError(
                        f"{role} contains a contradictory two-way range-phase "
                        f"declaration: {key}={value!r}"
                    )
                if signs:
                    return {"two_way_sign": next(iter(signs))}
                if key == "range_phase_convention":
                    return {}
                # ``phase_law`` often carries time-convention text only. That
                # evidence is checked by the coherent metadata contract, but it
                # does not establish a two-way range sign here.
                return {}
            if kind == "geometry":
                dimensions: dict[str, str] = {}
                topologies = set()
                if "multistatic" in words:
                    topologies.add("multistatic")
                if "bistatic" in words:
                    topologies.add("bistatic")
                if "quasi monostatic" in semantic or "quasimonostatic" in words:
                    topologies.add("quasi_monostatic")
                elif "not monostatic" in semantic:
                    topologies.add("non_monostatic")
                elif "monostatic" in words:
                    topologies.add("monostatic")
                if len(topologies) > 1:
                    raise ValueError(
                        f"{role} contains a contradictory measurement geometry "
                        f"declaration: {key}={value!r}"
                    )
                if topologies:
                    dimensions["scattering_configuration"] = next(iter(topologies))

                far_field = bool(
                    "far field" in semantic
                    or "farfield" in words
                    or "far zone" in semantic
                    or "farzone" in words
                    or "fraunhofer" in words
                    or "plane wave" in semantic
                    or "radiation zone" in semantic
                    or semantic in {"far", "ff"}
                )
                near_field = bool(
                    "near field" in semantic
                    or "nearfield" in words
                    or "near zone" in semantic
                    or "nearzone" in words
                    or "fresnel" in words
                    or "reactive near" in semantic
                    or semantic in {"near", "nf"}
                )
                if far_field and near_field:
                    raise ValueError(
                        f"{role} contains a contradictory measurement geometry "
                        f"declaration: {key}={value!r}"
                    )
                if far_field or near_field:
                    dimensions["propagation_regime"] = (
                        "far_field" if far_field else "near_field"
                    )
                return dimensions

            if kind == "motion":
                positive_state_key = key != "phase_center_motion"
                if semantic in {"1", "true", "yes"}:
                    return {"state": "stable" if positive_state_key else "unsafe"}
                if semantic in {"0", "false", "no", "none", "n a", "na"}:
                    return {"state": "unsafe" if positive_state_key else "stable"}
                safe = bool(
                    "no motion" in semantic
                    or "without motion" in semantic
                    or "no drift" in semantic
                    or words.intersection(
                        {
                            "compensated",
                            "stable",
                            "static",
                            "fixed",
                            "aligned",
                            "corrected",
                        }
                    )
                )
                unsafe = bool(
                    words.intersection(
                        {
                            "uncompensated",
                            "unstable",
                            "moving",
                            "varying",
                            "variable",
                            "misaligned",
                        }
                    )
                    or (
                        any(word.startswith("drift") for word in words)
                        and "no drift" not in semantic
                        and "without drift" not in semantic
                    )
                    or "not compensated" in semantic
                    or "not stable" in semantic
                    or "not static" in semantic
                    or "not fixed" in semantic
                    or "not aligned" in semantic
                    or "motion present" in semantic
                )
                if safe and unsafe:
                    raise ValueError(
                        f"{role} contains a contradictory motion-state declaration: "
                        f"{key}={value!r}"
                    )
                if safe or unsafe:
                    return {"state": "stable" if safe else "unsafe"}
                return {}

            if kind == "setup_state":
                if semantic in {"1", "true", "yes", "same", "unchanged"}:
                    return {"state": "stable"}
                if semantic in {"0", "false", "no", "different", "changed"}:
                    return {"state": "unsafe"}
                safe = bool(words.intersection({"static", "fixed", "unchanged"}))
                unsafe = bool(
                    words.intersection({"changed", "different", "reconfigured"})
                    or "not static" in semantic
                    or "not fixed" in semantic
                )
                if safe and unsafe:
                    raise ValueError(
                        f"{role} contains a contradictory static-setup declaration: "
                        f"{key}={value!r}"
                    )
                if safe or unsafe:
                    return {"state": "stable" if safe else "unsafe"}
                return {}
            raise RuntimeError(f"unsupported acquisition metadata kind {kind!r}")

        def collect(grid, family, label, kind, keys, role):
            raw_by_key: dict[str, str] = {}
            canonical_by_key: dict[str, dict[str, str]] = {}
            dimensions: dict[str, str] = {}
            dimension_sources: dict[str, str] = {}
            for key in keys:
                raw = grid._declared_scalar_metadata(key)
                if not raw:
                    continue
                values = canonical(key, raw, kind, role)
                raw_by_key[key] = raw
                canonical_by_key[key] = values
                for dimension, value in values.items():
                    prior = dimensions.get(dimension)
                    if prior is not None and prior != value:
                        prior_key = dimension_sources[dimension]
                        raise ValueError(
                            f"{role} contains contradictory {label} declarations: "
                            f"{prior_key}={raw_by_key[prior_key]!r} conflicts with "
                            f"{key}={raw!r}"
                        )
                    dimensions[dimension] = value
                    dimension_sources[dimension] = key
            if kind == "motion" and dimensions.get("state") == "unsafe":
                raise ValueError(
                    f"{operation} requires stable/aligned acquisitions; "
                    f"{role} declares {label}: {raw_by_key!r}"
                )
            if kind == "setup_state" and dimensions.get("state") == "unsafe":
                raise ValueError(
                    f"{operation} requires an unchanged static setup; "
                    f"{role} declares {label}: {raw_by_key!r}"
                )
            return {
                "declared_by_key": raw_by_key,
                "canonical_by_key": canonical_by_key,
                "canonical_dimensions": dimensions,
            }

        matching: dict[str, str] = {}
        missing: dict[str, list[str]] = {}
        semantic_families = {}
        for family, label, kind, keys in _ACQUISITION_METADATA_FAMILIES:
            if family in excluded:
                continue
            left = collect(self, family, label, kind, keys, left_name)
            right = collect(other, family, label, kind, keys, right_name)
            left_dimensions = left["canonical_dimensions"]
            right_dimensions = right["canonical_dimensions"]
            for dimension in sorted(set(left_dimensions).intersection(right_dimensions)):
                if left_dimensions[dimension] != right_dimensions[dimension]:
                    raise ValueError(
                        f"{operation} requires matching explicit {label}; "
                        f"{left_name} declares {left['declared_by_key']!r}, while "
                        f"{right_name} declares {right['declared_by_key']!r} "
                        f"({dimension} mismatch)"
                    )

            all_dimensions_set = set(left_dimensions).union(right_dimensions)
            if kind == "range_phase":
                # The family is physically incomplete until the two-way sign is
                # explicit on each acquisition; a general assumptions
                # attestation may cover absence but never an opposite sign.
                all_dimensions_set.add("two_way_sign")
            all_dimensions = sorted(all_dimensions_set)
            if not all_dimensions:
                missing[family] = [left_name, right_name]
            else:
                for dimension in all_dimensions:
                    absent = []
                    if dimension not in left_dimensions:
                        absent.append(left_name)
                    if dimension not in right_dimensions:
                        absent.append(right_name)
                    if absent:
                        missing[f"{family}.{dimension}"] = absent

            # Propagate only declarations from the left input whose complete
            # semantic content is explicitly present and equal on the right.
            # The general acquisition attestation may cover missing facts, but
            # it must not fabricate a declaration on the result.
            for key, raw in left["declared_by_key"].items():
                values = left["canonical_by_key"][key]
                if values and all(
                    right_dimensions.get(dimension) == value
                    for dimension, value in values.items()
                ):
                    matching[key] = raw

            semantic_families[family] = {
                "label": label,
                "aliases": list(keys),
                "declarations_by_role": {
                    left_name: left["declared_by_key"],
                    right_name: right["declared_by_key"],
                },
                "canonical_dimensions_by_role": {
                    left_name: left_dimensions,
                    right_name: right_dimensions,
                },
            }

        return {
            "schema": contract_schema,
            "checked_fields": [
                key
                for family, _label, _kind, keys in _ACQUISITION_METADATA_FAMILIES
                if family not in excluded
                for key in keys
            ],
            "excluded_families": sorted(excluded),
            "semantic_families": semantic_families,
            "matching_explicit_declarations": matching,
            "missing_declarations_by_role": missing,
            "missing_declarations_covered_by_operation_assumption": bool(missing),
            "missing_declarations_covered_by_user_attestation": False,
            "explicit_contradictions_allowed": False,
        }

    def _assert_support_reference_metadata_compatible(self, other):
        """Reject explicit acquisition/setup contradictions for support subtraction.

        Missing declarations are recorded as assumptions. Explicit declarations
        are compared across their semantic alias families and can never be
        waived by an operation choice.
        """

        return self._assert_acquisition_metadata_compatible(
            other,
            operation_label="support-referenced subtraction",
            left_role="target_plus_support",
            right_role="support_only_reference",
            schema="grim.support-reference-metadata-contract.v3",
        )

    def _coherent_attestation_provenance(
        self,
        others,
        *,
        operation,
        metadata_attested,
    ):
        """Return durable metadata for coherent assumptions or legacy attestation.

        Missing declarations never manufacture convention values. The record
        states that the requested coherent operation proceeded with those facts
        unspecified; the legacy ``metadata_attested=True`` API remains accepted
        for saved scripts and records its stronger user statement.
        """

        if not isinstance(metadata_attested, (bool, np.bool_)):
            raise TypeError("metadata_attested must be True or False")
        inputs = (self, *tuple(others))
        fields = (
            "phase_reference",
            "time_convention",
            "polarization_basis",
            "amplitude_version",
            "amplitude_convention",
            "complex_field_domain",
        )
        advisories = [issue for grid in inputs[1:] for issue in self._assert_coherent_metadata_compatible(grid)]
        missing = {}
        for key in fields:
            missing_indices = [
                index
                for index, grid in enumerate(inputs, start=1)
                if (
                    not grid._declared_scalar_metadata(key)
                    or grid._metadata_placeholder(
                        grid._declared_scalar_metadata(key)
                    )
                )
            ]
            if missing_indices:
                missing[key] = missing_indices

        if not missing and not metadata_attested and not advisories:
            return None, None

        operation_name = str(operation).strip().lower().replace("_", "-")
        user_attested = bool(metadata_attested)
        record = {
            "metadata_policy": "advisory; no phase or amplitude conversion applied",
            "advisories": advisories,
            "schema": (
                "grim.coherent-metadata-attestation.v1"
                if user_attested
                else "grim.coherent-metadata-assumption.v1"
            ),
            "operation": operation_name,
            "input_count": len(inputs),
            "user_attested": user_attested,
            "operation_requested_with_unspecified_metadata": bool(missing),
            "assumed_scope": [
                "phase_reference_or_center",
                "phasor_time_convention",
                "polarization_basis",
            ],
            "missing_declarations_by_input": missing,
            "declarations_inferred": False,
        }
        if user_attested:
            history_entry = (
                f"User-attested coherent metadata compatibility ({operation_name}, "
                f"{len(inputs)} inputs): compatible phase reference/center, phasor "
                "time convention, and polarization basis where declarations were "
                "unspecified; no convention values inferred"
            )
        else:
            history_entry = (
                f"Coherent operation used available complex samples ({operation_name}, "
                f"{len(inputs)} inputs): missing convention metadata recorded as "
                "unspecified; no convention values inferred"
            )
        prior_history = str(self.history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )

        extra = {}
        source_declarations = {}
        for key in fields:
            declared_values = []
            direct_values = []
            for grid in inputs:
                declared_values.extend(
                    grid._coherent_source_convention_values(key)
                )
                direct = grid._declared_scalar_metadata(key)
                if direct and not grid._metadata_placeholder(direct):
                    direct_values.append(direct)
            if key == "time_convention":
                normalized = {
                    self._canonical_time_convention(value)
                    for value in declared_values
                }
            else:
                normalized = {
                    " ".join(value.split()).casefold()
                    for value in declared_values
                }
            if declared_values:
                unique_values = []
                seen = set()
                for value in declared_values:
                    canonical = (
                        self._canonical_time_convention(value)
                        if key == "time_convention"
                        else " ".join(value.split()).casefold()
                    )
                    if canonical not in seen:
                        unique_values.append(value)
                        seen.add(canonical)
                source_declarations[key] = unique_values
            if len(direct_values) == len(inputs) and len(normalized) == 1:
                # This is an exact declaration from an input, not a value
                # manufactured by the assumption. Carrying it forward also
                # prevents a later chained operation from hiding a conflict.
                extra[key] = direct_values[0]
        if source_declarations:
            extra["coherent_source_conventions_json"] = json.dumps(
                {
                    "schema": "grim.coherent-source-conventions.v1",
                    "declared_values": source_declarations,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        record_key = (
            "coherent_metadata_attestation_json"
            if user_attested
            else "coherent_metadata_assumption_json"
        )
        extra[record_key] = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return history, extra

    def _assert_compatible(
        self,
        other,
        *,
        coherent=False,
        coherent_metadata_attested=False,
        _scan_phase_samples=True,
    ):
        """Validate another grid for element-wise operations.

        Use before coherent/incoherent add/subtract operations.

        Args:
            other: Another RcsGrid instance.

        Raises:
            TypeError: if other is not an RcsGrid.
            ValueError: if axes or shapes differ.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        if self.rcs_power.shape != other.rcs_power.shape:
            raise ValueError(f"rcs shape {other.rcs_power.shape} != {self.rcs_power.shape}")
        if not np.array_equal(self.azimuths, other.azimuths):
            raise ValueError("azimuth axis mismatch")
        if not np.array_equal(self.elevations, other.elevations):
            raise ValueError("elevation axis mismatch")
        if not np.array_equal(self.frequencies, other.frequencies):
            raise ValueError("frequency axis mismatch")
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch")
        self._assert_physical_metadata_compatible(other)
        if coherent:
            if not isinstance(_scan_phase_samples, (bool, np.bool_)):
                raise TypeError("_scan_phase_samples must be True or False")
            if _scan_phase_samples:
                for label, grid in (("left", self), ("right", other)):
                    missing = np.isfinite(grid.rcs_power) & ~np.isfinite(grid.rcs_phase)
                    if np.any(missing):
                        raise ValueError(
                            f"coherent operation requires phase; {label} grid has "
                            f"{int(np.count_nonzero(missing))} finite-power sample(s) "
                            "with unknown phase"
                        )
            self._assert_coherent_metadata_compatible(
                other, metadata_attested=coherent_metadata_attested
            )

    def range_calibrate(
        self,
        measured_calibration,
        exact_reference,
        range_offset_m,
        *,
        allow_singleton_angular_broadcast=False,
        convention_attested=False,
        measured_label=None,
        exact_label=None,
        maximum_correction_gain_db=60.0,
    ):
        """Apply complex substitution calibration at a signed range offset.

        The stored field is ``A = sqrt(sigma) * exp(1j*phase)``.  For GRIM's
        ``exp(+j*omega*t)`` convention, with a monostatic range response
        proportional to ``exp(-j*2*k*R)``, the operation is

        ``A_out = A_dut * A_exact * exp(-j*4*pi*f*dR/c) / A_measured``.

        ``dR`` is positive when the measured calibration target is farther
        from the radar than the DUT/reference plane. Selecting the measured
        and exact roles expresses the user's calibration intent; unavailable
        acquisition metadata is recorded as assumed rather than blocking the
        calculation. No frequency or angular interpolation is performed.
        """

        measured_calibration, exact_reference = self._ensure_grids(
            (measured_calibration, exact_reference)
        )
        for option_name, option_value in (
            ("convention_attested", convention_attested),
            (
                "allow_singleton_angular_broadcast",
                allow_singleton_angular_broadcast,
            ),
        ):
            if not isinstance(option_value, (bool, np.bool_)):
                raise TypeError(f"{option_name} must be True or False")
        try:
            offset_m = float(range_offset_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("range offset must be a finite distance in meters") from exc
        if not np.isfinite(offset_m):
            raise ValueError("range offset must be a finite distance in meters")
        prior_calibration_raw = (self.extra or {}).get("range_calibration_json")
        prior_calibration = None
        if prior_calibration_raw is not None:
            try:
                if isinstance(prior_calibration_raw, np.ndarray):
                    prior_calibration_raw = prior_calibration_raw.reshape(()).item()
                prior_calibration = json.loads(str(prior_calibration_raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                prior_calibration = {"unparsed_record": str(prior_calibration_raw)}
        if maximum_correction_gain_db is None:
            gain_limit_db = None
        else:
            try:
                gain_limit_db = float(maximum_correction_gain_db)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "maximum correction gain must be a finite nonnegative dB value"
                ) from exc
            if not np.isfinite(gain_limit_db) or gain_limit_db < 0.0:
                raise ValueError(
                    "maximum correction gain must be a finite nonnegative dB value"
                )

        grids = (
            ("DUT", self),
            ("measured calibration", measured_calibration),
            ("exact reference", exact_reference),
        )
        for label, grid in grids:
            raw_frequency_unit = str(
                (grid.units or {}).get("frequency", "GHz")
            ).strip().lower()
            if raw_frequency_unit not in _FREQUENCY_UNITS:
                raise ValueError(
                    f"{label} has unsupported frequency unit "
                    f"{(grid.units or {}).get('frequency')!r}; Range Cal "
                    "requires Hz, kHz, MHz, or GHz"
                )
            if grid.linear_quantity() != "sigma_3d":
                raise ValueError(
                    f"{label} must contain sigma_3d/dBsm data, got "
                    f"{grid.linear_quantity()}"
                )
            raw_log_unit = (grid.units or {}).get("rcs_log_unit")
            if raw_log_unit is not None and str(raw_log_unit).strip().lower() not in {
                "dbsm",
                "dbm2",
            }:
                raise ValueError(
                    f"{label} has unsupported RCS log unit {raw_log_unit!r}; "
                    "Range Cal requires dBsm"
                )
            if grid.default_log_unit().lower() != "dbsm":
                raise ValueError(
                    f"{label} must use dBsm, got {grid.default_log_unit()}"
                )
        self._assert_physical_metadata_compatible(measured_calibration)
        self._assert_physical_metadata_compatible(exact_reference)
        measured_calibration._assert_physical_metadata_compatible(exact_reference)

        def _declared_time_sign(grid, label):
            values = []
            for container in (grid.units or {}, grid.extra or {}):
                for key in (
                    "time_convention",
                    "phase_reference",
                    "amplitude_convention",
                ):
                    raw = container.get(key)
                    if raw is not None:
                        array = np.asarray(raw)
                        if array.size != 1:
                            raise ValueError(
                                f"{label} metadata {key!r} must be scalar"
                            )
                        values.append(str(array.reshape(-1)[0].item()))
            signs = set()
            for value in values:
                compact = (
                    value.lower()
                    .replace("ω", "omega")
                    .replace("*", "")
                    .replace(" ", "")
                )
                if re.search(r"exp\(\+?j(?:omega|w)t\)", compact):
                    signs.add("+jwt")
                if re.search(r"exp\(-j(?:omega|w)t\)", compact):
                    signs.add("-jwt")
            if len(signs) > 1:
                raise ValueError(
                    f"{label} contains contradictory declared time conventions"
                )
            return next(iter(signs)) if signs else None

        declared_time_signs = {
            label: _declared_time_sign(grid, label) for label, grid in grids
        }
        incompatible_signs = {
            label: sign
            for label, sign in declared_time_signs.items()
            if sign is not None and sign != "+jwt"
        }
        if incompatible_signs:
            details = ", ".join(
                f"{label}={sign}" for label, sign in incompatible_signs.items()
            )
            raise ValueError(
                "Range Cal uses GRIM's exp(+j*omega*t) phase law and will not "
                f"override contradictory declared metadata ({details})"
            )

        def _canonical_polarizations(grid, label):
            labels = [str(value).strip().upper() for value in grid.polarizations]
            if any(not value for value in labels):
                raise ValueError(f"{label} contains a blank polarization label")
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"{label} contains duplicate polarization labels after normalization"
                )
            return labels

        dut_pols = _canonical_polarizations(self, "DUT")
        measured_pols = _canonical_polarizations(
            measured_calibration, "measured calibration"
        )
        exact_pols = _canonical_polarizations(exact_reference, "exact reference")
        missing_measured_pols = [
            value for value in dut_pols if value not in measured_pols
        ]
        missing_exact_pols = [value for value in dut_pols if value not in exact_pols]
        if missing_measured_pols or missing_exact_pols:
            missing_parts = []
            if missing_measured_pols:
                missing_parts.append(
                    "measured calibration: " + ", ".join(missing_measured_pols)
                )
            if missing_exact_pols:
                missing_parts.append(
                    "exact reference: " + ", ".join(missing_exact_pols)
                )
            raise ValueError(
                "calibration references are missing DUT polarization(s) in "
                + "; ".join(missing_parts)
            )

        if not np.array_equal(
            measured_calibration.frequencies, exact_reference.frequencies
        ):
            raise ValueError(
                "measured-calibration and exact-reference frequency axes differ"
            )
        if not np.array_equal(self.frequencies, measured_calibration.frequencies):
            raise ValueError(
                "DUT and calibration frequency axes differ; align them explicitly first"
            )

        for axis_name in ("azimuths", "elevations"):
            measured_axis = np.asarray(getattr(measured_calibration, axis_name))
            exact_axis = np.asarray(getattr(exact_reference, axis_name))
            dut_axis = np.asarray(getattr(self, axis_name))
            if not np.array_equal(measured_axis, exact_axis):
                raise ValueError(
                    "measured-calibration and exact-reference "
                    f"{axis_name[:-1]} axes differ"
                )
            if np.array_equal(dut_axis, measured_axis):
                continue
            if len(measured_axis) == 1 and bool(
                allow_singleton_angular_broadcast
            ):
                continue
            if len(measured_axis) == 1:
                raise ValueError(
                    f"singleton calibration {axis_name[:-1]} requires explicit "
                    "broadcast confirmation"
                )
            raise ValueError(
                f"DUT and calibration {axis_name[:-1]} axes differ; no angular "
                "interpolation or averaging is performed"
            )

        frequency_hz = np.asarray(
            self._frequency_value_to_hz(self.frequencies), dtype=np.float64
        )
        if np.any(~np.isfinite(frequency_hz)) or np.any(frequency_hz <= 0.0):
            raise ValueError("range calibration requires positive finite frequencies")

        measured_pol_index = [measured_pols.index(label) for label in dut_pols]
        exact_pol_index = [exact_pols.index(label) for label in dut_pols]
        measured_amp = np.asarray(
            measured_calibration.rcs[..., measured_pol_index], dtype=np.complex128
        )
        exact_amp = np.asarray(
            exact_reference.rcs[..., exact_pol_index], dtype=np.complex128
        )
        dut_amp = np.asarray(self.rcs, dtype=np.complex128)
        measured_power = np.asarray(
            measured_calibration.rcs_power[..., measured_pol_index]
        )
        exact_power = np.asarray(exact_reference.rcs_power[..., exact_pol_index])
        dut_power = np.asarray(self.rcs_power)

        # Phase is immaterial for an exact zero field. Reconstruct zero from
        # power only when there is no finite authoritative raw amplitude.  A
        # GHOST raw field can remain finite and nonzero after float32 power has
        # underflowed to zero, and that field must remain authoritative.
        exact_zero_power = np.isfinite(exact_power) & (exact_power == 0.0)
        dut_zero_power = np.isfinite(dut_power) & (dut_power == 0.0)
        exact_amp = np.array(exact_amp, copy=True)
        dut_amp = np.array(dut_amp, copy=True)
        exact_amp[exact_zero_power & ~np.isfinite(exact_amp)] = 0.0 + 0.0j
        dut_amp[dut_zero_power & ~np.isfinite(dut_amp)] = 0.0 + 0.0j

        # A measured zero cannot define a correction denominator.  Synthesize
        # an explicit zero for magnitude-only nulls, but preserve a finite
        # authoritative raw field even if its separately stored power rounded
        # or underflowed to zero.
        measured_amp = np.array(measured_amp, copy=True)
        measured_zero_power = np.isfinite(measured_power) & (
            measured_power == 0.0
        )
        measured_amp[
            measured_zero_power & ~np.isfinite(measured_amp)
        ] = 0.0 + 0.0j

        range_phase = np.exp(
            -1j * (4.0 * np.pi * frequency_hz * offset_m / C0)
        ).reshape(1, 1, -1, 1)
        measured_finite = np.isfinite(measured_amp)
        exact_finite = np.isfinite(exact_amp)
        dut_finite = np.isfinite(dut_amp)
        measured_zero = np.isfinite(measured_amp) & (np.abs(measured_amp) == 0.0)

        # A zero exact response is valid: it means the calibrated field is
        # exactly zero at that bin. Missing DUT/reference bins remain sparse
        # NaNs rather than aborting an otherwise usable calibration grid.
        valid_reference = measured_finite & exact_finite & ~measured_zero
        correction = np.full(
            np.broadcast_shapes(exact_amp.shape, measured_amp.shape),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        correction_numerator = exact_amp * range_phase
        np.divide(
            correction_numerator,
            measured_amp,
            out=correction,
            where=valid_reference & ~measured_zero,
        )
        valid_correction = np.isfinite(correction)
        correction_magnitude = np.abs(correction)
        positive_correction = valid_correction & (correction_magnitude > 0.0)
        correction_gain_db = np.full(correction.shape, np.nan, dtype=np.float64)
        correction_gain_db[positive_correction] = (
            20.0 * np.log10(correction_magnitude[positive_correction])
        )
        excessive = np.zeros(correction.shape, dtype=bool)
        if gain_limit_db is not None:
            excessive = correction_gain_db > gain_limit_db
            correction[excessive] = np.nan + 1j * np.nan
            valid_correction &= ~excessive
        try:
            correction_for_dut = np.broadcast_to(correction, dut_amp.shape)
            with np.errstate(over="ignore", invalid="ignore"):
                output_amp = dut_amp * correction_for_dut
        except ValueError as exc:
            raise ValueError(
                "calibration angular axes cannot broadcast to the DUT grid"
            ) from exc
        if output_amp.shape != dut_amp.shape:
            raise ValueError(
                f"calibration produced shape {output_amp.shape}, expected {dut_amp.shape}"
            )
        candidate_output = dut_finite & np.isfinite(correction_for_dut)
        finite_output_amp = np.isfinite(output_amp)
        overflowed_complex_count = int(
            np.count_nonzero(candidate_output & ~finite_output_amp)
        )
        valid_output = candidate_output & finite_output_amp
        output_power = np.full(output_amp.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            output_power[valid_output] = np.abs(output_amp[valid_output]) ** 2
        overflowed_power = valid_output & ~np.isfinite(output_power)
        overflowed_power_count = int(np.count_nonzero(overflowed_power))
        valid_output &= ~overflowed_power
        output_amp = np.array(output_amp, copy=True)
        output_amp[~valid_output] = np.nan + 1j * np.nan
        output_power[~valid_output] = np.nan
        if not np.any(valid_output):
            raise ValueError(
                "range calibration has no calibratable bins after masking "
                "missing, zero-denominator, over-limit, or overflowed samples"
            )

        finite_gain = correction_gain_db[np.isfinite(correction_gain_db)]
        gain_summary = {
            "minimum": float(np.min(finite_gain)) if finite_gain.size else None,
            "median": float(np.median(finite_gain)) if finite_gain.size else None,
            "maximum": float(np.max(finite_gain)) if finite_gain.size else None,
            "zero_factor_count": int(
                np.count_nonzero(valid_correction & (correction_magnitude == 0.0))
            ),
            "missing_reference_bin_count": int(
                np.count_nonzero(~valid_reference)
            ),
            "zero_measured_denominator_bin_count": int(
                np.count_nonzero(measured_zero)
            ),
            "over_limit_correction_bin_count": int(
                np.count_nonzero(excessive)
            ),
            "overflowed_complex_output_bin_count": overflowed_complex_count,
            "overflowed_power_output_bin_count": overflowed_power_count,
            "masked_output_bin_count": int(np.count_nonzero(~valid_output)),
        }
        measured_name = str(
            measured_label or measured_calibration.source_path or "measured calibration"
        )
        exact_name = str(
            exact_label or exact_reference.source_path or "exact reference"
        )

        def _grid_content_sha256(grid):
            """Bind provenance to the physical complex field Range Cal uses."""

            digest = hashlib.sha256()
            digest.update(b"grim.range-calibration-grid-id.v2\0")

            def _update_array(label, values):
                contiguous = np.ascontiguousarray(values)
                digest.update(label.encode("ascii") + b"\0")
                digest.update(str(contiguous.shape).encode("ascii") + b"\0")
                digest.update(contiguous.tobytes(order="C"))

            for values in (
                np.asarray(grid.azimuths, dtype=np.float64),
                np.asarray(grid.elevations, dtype=np.float64),
                np.asarray(grid.frequencies, dtype=np.float64),
                np.asarray(grid.rcs_power, dtype=np.float64),
                np.asarray(grid.rcs_phase, dtype=np.float64),
            ):
                _update_array("modeled-array", values)

            # Hash the authoritative complex field in bounded azimuth chunks.
            # This distinguishes files whose modeled power/phase match but
            # whose GHOST raw fields differ, without allocating another whole
            # complex128 grid solely for provenance.
            cells_per_azimuth = int(np.prod(grid.rcs_power.shape[1:]))
            azimuth_block = max(1, 262_144 // max(1, cells_per_azimuth))
            digest.update(b"authoritative-complex-field\0")
            digest.update(str(grid.rcs_power.shape).encode("ascii") + b"\0")
            for start in range(0, len(grid.azimuths), azimuth_block):
                field = np.asarray(
                    grid.rcs_slice(
                        (
                            slice(start, start + azimuth_block),
                            slice(None),
                            slice(None),
                            slice(None),
                        )
                    ),
                    dtype=np.complex128,
                )
                _update_array("field-real", field.real)
                _update_array("field-imag", field.imag)
            digest.update(
                json.dumps(
                    [str(value) for value in grid.polarizations.tolist()],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(
                json.dumps(
                    dict(grid.units or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
            convention_metadata = {
                key: grid._declared_scalar_metadata(key)
                for key in (
                    "phase_reference",
                    "time_convention",
                    "polarization_basis",
                    "amplitude_convention",
                )
            }
            digest.update(
                json.dumps(
                    convention_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            return digest.hexdigest()

        measured_sha256 = _grid_content_sha256(measured_calibration)
        exact_sha256 = _grid_content_sha256(exact_reference)
        provenance = {
            "schema": "grim.range-calibration.v1",
            "mode": "complex_substitution",
            "formula": (
                "A_out=A_dut*A_exact*exp(-j*4*pi*f*delta_R/c)/A_measured_cal"
            ),
            "range_offset_m": offset_m,
            "range_offset_positive_direction": "away_from_radar",
            "phase_law": "exp(+j*omega*t); S(range) proportional to exp(-j*2*k*R)",
            "axis_policy": (
                "exact_frequency_and_polarization; exact_or_explicit_singleton_"
                "broadcast_angular_axes; no_interpolation"
            ),
            "singleton_angular_broadcast": bool(
                bool(allow_singleton_angular_broadcast)
            ),
            "operation_selected_as_convention_assumption": True,
            "user_convention_attested": bool(convention_attested),
            "input_was_previously_range_calibrated": prior_calibration is not None,
            "prior_range_calibration": prior_calibration,
            "measured_calibration": measured_name,
            "measured_calibration_content_sha256": measured_sha256,
            "exact_reference": exact_name,
            "exact_reference_content_sha256": exact_sha256,
            "declared_time_conventions": declared_time_signs,
            "maximum_correction_gain_db": gain_limit_db,
            "correction_gain_db": gain_summary,
        }

        # A calibrated field is a new measurement-domain artifact. Carrying
        # arbitrary DUT extras can resurrect stale solver, delta, raw-field, or
        # certification semantics because save() deliberately round-trips those
        # keys. Rebuild only the truthful calibrated metadata below.
        extra = {}
        extra["range_calibration_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        exact_phase_reference = exact_reference._phase_reference()
        extra["phase_reference"] = (
            "range-calibrated complex substitution; exact reference="
            f"{exact_phase_reference or '<unspecified phase center>'}; "
            f"exact_content_sha256={exact_sha256}; "
            f"delta_R={offset_m:.12g} m positive away from radar; "
            "exp(+j*omega*t), S(range)~exp(-j*2*k*R)"
        )
        extra["amplitude_convention"] = (
            "stored complex field magnitude=sqrt(sigma_3d); calibrated by "
            "complex substitution"
        )
        history_entry = (
            f"Range Cal{' re-calibration' if prior_calibration is not None else ''} "
            f"complex substitution: measured={measured_name}; "
            f"exact={exact_name}; delta_R={offset_m:.12g} m positive away "
            "from radar; no interpolation; "
            f"masked_output_bins={gain_summary['masked_output_bin_count']}"
        )
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return RcsGrid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs=output_amp,
            rcs_power=output_power,
            rcs_phase=np.where(valid_output, np.angle(output_amp), np.nan),
            rcs_domain="complex_amplitude",
            source_path=None,
            history=history,
            units=dict(self.units),
            extra=extra,
        )

    def coherent_add(self, other, *, metadata_attested=False):
        """Coherently add two grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            other: Another RcsGrid with identical axes.
            metadata_attested: Optional user-attestation record. Missing and
                conflicting convention annotations are advisory without it.

        Returns:
            New RcsGrid with rcs = self.rcs + other.rcs.
        """
        self._assert_compatible(
            other,
            coherent=True,
            coherent_metadata_attested=metadata_attested,
            _scan_phase_samples=False,
        )
        rcs_out = self.rcs + other.rcs
        usable = np.isfinite(rcs_out.real) & np.isfinite(rcs_out.imag)
        usable_count = int(np.count_nonzero(usable))
        if usable_count == 0:
            raise ValueError(
                "coherent addition has no common usable complex samples"
            )
        history, attestation_extra = self._coherent_attestation_provenance(
            (other,),
            operation="coherent-add",
            metadata_attested=metadata_attested,
        )
        extra = self._derived_response_extra(
            (other,),
            operation="coherent-add",
            coherent=True,
            attestation_extra=attestation_extra,
        )
        extra["coherent_sample_qa_json"] = json.dumps(
            {
                "schema": "grim.coherent-sample-qa.v1",
                "operation": "coherent-add",
                "total_sample_count": int(rcs_out.size),
                "usable_sample_count": usable_count,
                "masked_sample_count": int(rcs_out.size - usable_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_domain="power_phase",
            history=history,
            extra=extra,
        )

    def coherent_add_many(self, *grids, metadata_attested=False):
        """Coherently add multiple grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            *grids: One or more RcsGrid instances.
            metadata_attested: Optional user-attestation record. Missing and
                conflicting convention annotations are advisory without it.

        Returns:
            New RcsGrid with rcs = self.rcs + sum(grid.rcs).
        """
        if not grids:
            return self
        total = np.array(self.rcs, copy=True)
        for grid in grids:
            self._assert_compatible(
                grid,
                coherent=True,
                coherent_metadata_attested=metadata_attested,
                _scan_phase_samples=False,
            )
            total = total + grid.rcs
        usable = np.isfinite(total.real) & np.isfinite(total.imag)
        usable_count = int(np.count_nonzero(usable))
        if usable_count == 0:
            raise ValueError(
                "coherent addition has no common usable complex samples"
            )
        history, attestation_extra = self._coherent_attestation_provenance(
            grids,
            operation="coherent-add-many",
            metadata_attested=metadata_attested,
        )
        extra = self._derived_response_extra(
            grids,
            operation="coherent-add-many",
            coherent=True,
            attestation_extra=attestation_extra,
        )
        extra["coherent_sample_qa_json"] = json.dumps(
            {
                "schema": "grim.coherent-sample-qa.v1",
                "operation": "coherent-add-many",
                "total_sample_count": int(total.size),
                "usable_sample_count": usable_count,
                "masked_sample_count": int(total.size - usable_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            total,
            rcs_domain="power_phase",
            history=history,
            extra=extra,
        )

    def coherent_subtract(
        self,
        other,
        *,
        metadata_attested=False,
        maximum_working_bytes=None,
    ):
        """Coherently subtract two grids (complex difference).

        Use when phases are aligned and you want field-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.
            metadata_attested: Optional user-attestation record. Missing and
                conflicting convention annotations are advisory without it.
            maximum_working_bytes: Optional cap for the newly retained result
                arrays plus bounded arithmetic/QA scratch. By default GRIM uses
                half of currently available physical memory (or the reviewed
                fallback); ``GRIM_COHERENT_WORKING_SET_MB`` can set a process-
                wide cap.

        Returns:
            New RcsGrid with rcs = self.rcs - other.rcs.
        """
        self._assert_compatible(
            other,
            coherent=True,
            coherent_metadata_attested=metadata_attested,
            _scan_phase_samples=False,
        )
        def response_precision_upper_bound(grid):
            raw_real = (grid.extra or {}).get("rcs_amp_real")
            raw_imag = (grid.extra or {}).get("rcs_amp_imag")
            if raw_real is not None and raw_imag is not None:
                if (
                    np.asarray(raw_real).shape == grid.rcs_power.shape
                    and np.asarray(raw_imag).shape == grid.rcs_power.shape
                ):
                    # Authoritative solver amplitude is normalized in float64.
                    # This shape-only upper bound deliberately runs before the
                    # O(N) finite-pair validation and is conservative if a
                    # malformed pair later falls back to float32 power/phase.
                    return np.dtype(np.float64)
            return np.dtype(_real_storage_dtype(grid.rcs_power, grid.rcs_phase))

        left_real_dtype = response_precision_upper_bound(self)
        right_real_dtype = response_precision_upper_bound(other)
        real_dtype = np.dtype(
            np.float64
            if max(left_real_dtype.itemsize, right_real_dtype.itemsize) > 4
            else np.float32
        )
        complex_dtype = np.dtype(
            np.complex128 if real_dtype == np.dtype(np.float64) else np.complex64
        )
        cell_count = int(self.rcs_power.size)
        block_cells = min(cell_count, _COHERENT_OPERATION_BLOCK_CELLS)
        retained_bytes = 2 * real_dtype.itemsize * cell_count
        # The block estimate covers both complex operands, an optional writable
        # copy, finite masks, magnitude/phase ufunc scratch, and the larger
        # post-subtraction QA/hash tile.  Inputs are already resident and are
        # intentionally excluded from this incremental operation budget.
        scratch_bytes = block_cells * (
            12 * complex_dtype.itemsize + 8 * real_dtype.itemsize
        )
        estimated_peak_bytes = retained_bytes + scratch_bytes
        limit_bytes = _coherent_working_set_limit_bytes(maximum_working_bytes)
        if (
            retained_bytes > np.iinfo(np.intp).max
            or estimated_peak_bytes > np.iinfo(np.intp).max
        ):
            raise MemoryError(
                "coherent subtraction result exceeds this Python/NumPy build's "
                "addressable allocation size"
            )
        if estimated_peak_bytes > limit_bytes:
            raise MemoryError(
                "coherent subtraction needs an estimated "
                f"{estimated_peak_bytes / 1024**2:.1f} MiB working set "
                f"({retained_bytes / 1024**2:.1f} MiB retained result plus "
                f"{scratch_bytes / 1024**2:.1f} MiB bounded scratch), above "
                f"the {limit_bytes / 1024**2:.1f} MiB limit. Crop the common "
                "grid or deliberately raise maximum_working_bytes / "
                "GRIM_COHERENT_WORKING_SET_MB on a machine with verified "
                "headroom."
            )

        # Validate authoritative raw-pair semantics only after the byte gate;
        # a rejected oversized operation must not scan either numerical field.
        read_left, _left_reader_dtype = self._bounded_complex_slice_reader()
        read_right, _right_reader_dtype = other._bounded_complex_slice_reader()
        power_out = np.empty(self.rcs_power.shape, dtype=real_dtype)
        phase_out = np.empty(self.rcs_power.shape, dtype=real_dtype)
        usable_count = 0
        for selection in _bounded_grid_selections(
            self.rcs_power.shape, _COHERENT_OPERATION_BLOCK_CELLS
        ):
            left = np.asarray(read_left(selection), dtype=complex_dtype)
            if not left.flags.writeable or not left.flags.owndata:
                left = np.array(left, dtype=complex_dtype, copy=True)
            right = np.asarray(read_right(selection), dtype=complex_dtype)
            np.subtract(left, right, out=left)
            power_block = power_out[selection]
            phase_block = phase_out[selection]
            with np.errstate(invalid="ignore", over="ignore"):
                np.hypot(left.real, left.imag, out=power_block)
                np.multiply(power_block, power_block, out=power_block)
                np.arctan2(left.imag, left.real, out=phase_block)
            invalid = ~np.isfinite(left.real)
            invalid |= ~np.isfinite(left.imag)
            invalid |= ~np.isfinite(power_block)
            power_block[invalid] = np.nan
            phase_block[invalid] = np.nan
            usable_count += int(np.count_nonzero(~invalid))

        if usable_count == 0:
            raise ValueError(
                "coherent subtraction has no common usable complex samples"
            )

        history, attestation_extra = self._coherent_attestation_provenance(
            (other,),
            operation="coherent-subtract",
            metadata_attested=metadata_attested,
        )
        extra = self._derived_response_extra(
            (other,),
            operation="coherent-subtract",
            coherent=True,
            attestation_extra=attestation_extra,
        )
        extra["coherent_sample_qa_json"] = json.dumps(
            {
                "schema": "grim.coherent-sample-qa.v1",
                "operation": "coherent-subtract",
                "total_sample_count": int(cell_count),
                "usable_sample_count": int(usable_count),
                "masked_sample_count": int(cell_count - usable_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_out,
            rcs_phase=phase_out,
            rcs_domain="power_phase",
            history=history,
            extra=extra,
            _adopt_clean_arrays=True,
        )

    def support_referenced_difference(
        self,
        support_reference,
        *,
        metadata_attested=False,
        assumptions_attested=False,
        target_label=None,
        support_label=None,
        maximum_working_bytes=None,
    ):
        """Return an exact target-plus-support minus support-only field.

        This guided wrapper intentionally delegates all numerical work and
        compatibility enforcement to :meth:`coherent_subtract`.  It adds role-
        explicit, content-bound provenance and QA diagnostics so the result
        cannot be mistaken for a generic operand-order subtraction.

        The result is a *support-referenced difference*.  It is not guaranteed
        to equal the target's free-space response because target/support
        coupling, shadowing, and multiple-bounce terms are not recoverable from
        two measurements by subtraction.
        """

        for option_name, option_value in (
            ("metadata_attested", metadata_attested),
            ("assumptions_attested", assumptions_attested),
        ):
            if not isinstance(option_value, (bool, np.bool_)):
                raise TypeError(f"{option_name} must be True or False")
        if not isinstance(support_reference, RcsGrid):
            raise TypeError("support_reference must be an RcsGrid")
        if support_reference is self:
            raise ValueError(
                "target+support and support-only roles must use different datasets"
            )
        chained_inputs = []
        if "support_reference_difference_json" in (self.extra or {}):
            chained_inputs.append("target_plus_support")
        if "support_reference_difference_json" in (support_reference.extra or {}):
            chained_inputs.append("support_only_reference")

        support_metadata_contract = (
            self._assert_support_reference_metadata_compatible(support_reference)
        )

        # This is the sole numerical operation. It enforces exact axes,
        # quantities/units, coordinate conventions, finite coherent phase, and
        # declared field-convention compatibility before subtracting.
        difference = self.coherent_subtract(
            support_reference,
            metadata_attested=bool(metadata_attested),
            maximum_working_bytes=maximum_working_bytes,
        )
        target_name = str(
            target_label or self.source_path or "target+support acquisition"
        )
        support_name = str(
            support_label
            or support_reference.source_path
            or "support-only reference"
        )
        qa = _support_reference_qa(self, support_reference, difference)
        if int(qa["common_finite_sample_count"]) == 0:
            raise ValueError(
                "support-referenced difference has no common finite complex "
                "samples after exact subtraction"
            )
        content_namespace = "grim.physical-grid-content.v1"
        target_sha256 = _physical_grid_content_sha256(
            self, namespace=content_namespace
        )
        support_sha256 = _physical_grid_content_sha256(
            support_reference, namespace=content_namespace
        )
        result_sha256 = _physical_grid_content_sha256(
            difference, namespace=content_namespace
        )
        provenance = {
            "schema": "grim.support-reference-difference.v1",
            "mode": "exact_complex_subtraction",
            "formula": "A_difference=A_target_plus_support-A_support_reference",
            "axis_policy": (
                "identical_axes_units_quantities_and_noncontradictory_explicit_"
                "acquisition_metadata; no_interpolation"
            ),
            "target_plus_support": target_name,
            "target_plus_support_content_sha256": target_sha256,
            "support_only_reference": support_name,
            "support_only_reference_content_sha256": support_sha256,
            "result_content_sha256": result_sha256,
            "content_hash_schema": content_namespace,
            "operation_selected_as_assumption_of_compatible_acquisition": True,
            "user_assumptions_attested": bool(assumptions_attested),
            "metadata_attestation_used": bool(metadata_attested),
            "chained_support_difference_input_roles": chained_inputs,
            "support_metadata_contract": support_metadata_contract,
            "interpretation": "support_referenced_complex_difference",
            "not_free_space_target": True,
            "unrecoverable_effects": [
                "target_support_coupling",
                "support_shadowing",
                "target_support_multiple_bounce_scattering",
                "acquisition_drift_or_misregistration",
            ],
            "qa": qa,
        }
        difference.extra = dict(difference.extra or {})
        for key, value in support_metadata_contract[
            "matching_explicit_declarations"
        ].items():
            if key != "complex_field_domain":
                difference.extra[key] = value
        difference.extra["support_reference_difference_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        difference.extra["complex_field_domain"] = (
            "support_referenced_complex_difference"
        )
        history_entry = (
            "Support-referenced difference (exact complex subtraction): "
            f"target+support={target_name}; support-only={support_name}; "
            "identical axes, no interpolation; QA/content hashes recorded; "
            "not a reconstructed free-space target response"
        )
        if chained_inputs:
            history_entry += (
                "; warning: chained support-difference input role(s)="
                + ",".join(chained_inputs)
            )
        difference.history = (
            f"{difference.history}\n{history_entry}"
            if difference.history
            else history_entry
        )
        difference.source_path = None
        return difference

    def incoherent_add(self, other):
        """Incoherently add two grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = self.rcs_power + other.rcs_power.
        """
        self._assert_compatible(other)
        power_sum = self.rcs_power + other.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_sum,
            rcs_phase=np.full(power_sum.shape, np.nan, dtype=power_sum.dtype),
            rcs_domain="power_phase",
            extra=self._derived_response_extra(
                (other,), operation="incoherent-add", coherent=False
            ),
        )

    def incoherent_add_many(self, *grids):
        """Incoherently add multiple grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with linear power = self.rcs_power + sum(grid.rcs_power).
        """
        if not grids:
            return self
        total = np.array(self.rcs_power, copy=True)
        for grid in grids:
            self._assert_compatible(grid)
            total = total + grid.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=total,
            rcs_phase=np.full(total.shape, np.nan, dtype=total.dtype),
            rcs_domain="power_phase",
            extra=self._derived_response_extra(
                grids, operation="incoherent-add-many", coherent=False
            ),
        )

    def incoherent_subtract(self, other):
        """Incoherently subtract two grids (magnitude difference).

        Use when phases are unrelated and you want power-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = self.rcs_power - other.rcs_power.

        A physically negative power result is rejected.  Only a negative
        residual consistent with floating-point subtraction roundoff is
        replaced by exact zero; this prevents a materially invalid
        subtraction from being silently clipped into a plausible dataset.
        """
        self._assert_compatible(other)
        left = np.asarray(self.rcs_power)
        right = np.asarray(other.rcs_power)
        calculation_dtype = np.result_type(left.dtype, right.dtype, np.float64)
        left_calc = left.astype(calculation_dtype, copy=False)
        right_calc = right.astype(calculation_dtype, copy=False)
        power_diff = left_calc - right_calc

        # The least precise input bounds the significance of the subtraction,
        # even though the calculation itself is promoted to float64.  Eight
        # ulps comfortably covers one subtraction plus ordinary upstream
        # representation noise without introducing an arbitrary absolute
        # floor that could erase meaningful low-power negatives.
        input_epsilons = [
            np.finfo(dtype).eps
            for dtype in (left.dtype, right.dtype)
            if np.issubdtype(dtype, np.floating)
        ]
        input_epsilon = max(input_epsilons, default=np.finfo(np.float64).eps)
        scale = np.maximum(np.abs(left_calc), np.abs(right_calc))
        roundoff_limit = 8.0 * float(input_epsilon) * scale
        finite_negative = np.isfinite(power_diff) & (power_diff < 0.0)
        material_negative = finite_negative & (power_diff < -roundoff_limit)
        if np.any(material_negative):
            count = int(np.count_nonzero(material_negative))
            minimum = float(np.min(power_diff[material_negative]))
            raise ValueError(
                "incoherent subtraction would produce materially negative "
                f"linear power in {count} cell(s); minimum difference is "
                f"{minimum:.17g}"
            )
        power_diff[finite_negative] = 0.0
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_diff,
            rcs_phase=np.full(power_diff.shape, np.nan, dtype=power_diff.dtype),
            rcs_domain="power_phase",
            extra=self._derived_response_extra(
                (other,), operation="incoherent-subtract", coherent=False
            ),
        )

    def arithmetic_db_subtract(self, other):
        """Return the dimensionless power ratio represented by a dB difference.

        Returns a grid whose dB display equals ``self_dB - other_dB``. For two
        constant lines at 30 and 25 dBsm, the result displays as 5 dB. Phase is
        meaningless for this magnitude-domain operation and is set to NaN.

        Both grids must share the same ``default_log_unit`` (dBsm or dBke).
        """
        self._assert_compatible(other)
        unit_a = self.default_log_unit()
        unit_b = other.default_log_unit()
        if unit_a != unit_b:
            raise ValueError(
                f"dB arithmetic requires matching log units; got {unit_a} vs {unit_b}"
            )

        # For matching physical units, dB subtraction is exactly the linear
        # power ratio.  Computing it in the dB display domain used to apply the
        # plotting floor to zero, turning 0/0 into a plausible 0 dB and 1/0
        # into an arbitrary finite 120 dB.  Undefined denominators remain NaN;
        # an exact zero numerator over a positive denominator remains zero.
        numerator = np.asarray(self.rcs_power)
        denominator = np.asarray(other.rcs_power)
        # Always divide in at least float64.  A representable ratio such as
        # float32(1e30) / float32(1e-30) is 1e60; performing that division in
        # float32 first would overflow and incorrectly turn it into NaN.
        output_dtype = np.result_type(
            numerator.dtype, denominator.dtype, np.float64
        )
        numerator = numerator.astype(output_dtype, copy=False)
        denominator = denominator.astype(output_dtype, copy=False)
        output_power = np.full(numerator.shape, np.nan, dtype=output_dtype)
        valid = (
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator > 0.0)
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(numerator, denominator, out=output_power, where=valid)
        output_power[~np.isfinite(output_power)] = np.nan
        ratio_units = dict(self.units)
        ratio_units["rcs_log_unit"] = "dB"
        ratio_units["rcs_linear_quantity"] = "power_ratio"

        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=output_power,
            rcs_phase=np.full(output_power.shape, np.nan, dtype=output_power.dtype),
            rcs_domain="power_phase",
            units=ratio_units,
            extra=self._derived_response_extra(
                (other,), operation="arithmetic-db-subtract", coherent=False
            ),
        )

    def align_to(self, other, mode="exact"):
        """Align this grid to another grid's axes.

        Modes:
            exact: require identical axes (returns self on success).
            intersect: keep only axis values present in both grids.
            interp: interpolate numeric axes to match other (no extrapolation).

        Args:
            other: Another RcsGrid instance.
            mode: "exact", "intersect", or "interp".

        Returns:
            New RcsGrid aligned to other's axes.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        self._assert_axis_metadata_compatible(other)

        if mode == "exact":
            if self.rcs_power.shape != other.rcs_power.shape:
                raise ValueError(
                    f"rcs shape {other.rcs_power.shape} != {self.rcs_power.shape}"
                )
            for name, left, right in (
                ("azimuth", self.azimuths, other.azimuths),
                ("elevation", self.elevations, other.elevations),
                ("frequency", self.frequencies, other.frequencies),
                ("polarization", self.polarizations, other.polarizations),
            ):
                if not np.array_equal(left, right):
                    raise ValueError(f"{name} axis mismatch")
            return self
        if mode not in ("intersect", "interp"):
            raise ValueError("mode must be 'exact', 'intersect', or 'interp'")

        if mode == "intersect":
            def _match_axis(axis_self, axis_other, tol=1e-6):
                axis_self = np.asarray(axis_self).ravel()
                axis_other = np.asarray(axis_other).ravel()
                _common, matched = self._common_axis_alignment(
                    (axis_self, axis_other), tol=tol
                )
                indices_self = [int(value) for value in matched[0]]
                indices_other = [int(value) for value in matched[1]]
                if not indices_self:
                    raise ValueError("no overlapping axis values for intersect")
                # Preserve the target grid's physical coordinates while using
                # the one-to-one matcher.  A source sample can no longer be
                # duplicated into two nearby target bins.  The symmetric
                # matcher emits numeric matches in value order, so reorder the
                # pairs back into the target grid's original axis order.
                target_source_pairs = sorted(
                    zip(indices_other, indices_self), key=lambda pair: pair[0]
                )
                target_indices = [pair[0] for pair in target_source_pairs]
                source_indices = [pair[1] for pair in target_source_pairs]
                return axis_other[target_indices], source_indices

            az_unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
            el_unit = self._supported_unit("elevation", _ANGLE_UNITS, "deg")
            frequency_unit = self._supported_unit(
                "frequency", _FREQUENCY_UNITS, "GHz"
            )
            az_tol = float(np.deg2rad(1.0e-6)) if az_unit == "rad" else 1.0e-6
            el_tol = float(np.deg2rad(1.0e-6)) if el_unit == "rad" else 1.0e-6
            # Preserve the historical 1e-6-GHz (1 kHz) physical tolerance
            # across every supported native frequency unit.
            f_tol = {
                "Hz": 1.0e3,
                "kHz": 1.0,
                "MHz": 1.0e-3,
                "GHz": 1.0e-6,
            }[frequency_unit]
            az_new, az_idx = _match_axis(
                self.azimuths, other.azimuths, tol=az_tol
            )
            el_new, el_idx = _match_axis(
                self.elevations, other.elevations, tol=el_tol
            )
            f_new, f_idx = _match_axis(
                self.frequencies, other.frequencies, tol=f_tol
            )
            pol_new, pol_idx = _match_axis(self.polarizations, other.polarizations, tol=0.0)
            pwr_new = self.rcs_power[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            phs_new = self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            selection = np.ix_(az_idx, el_idx, f_idx, pol_idx)
            source_axes = (
                np.asarray(self.azimuths)[az_idx],
                np.asarray(self.elevations)[el_idx],
                np.asarray(self.frequencies)[f_idx],
                np.asarray(self.polarizations)[pol_idx],
            )
            relabeled = not all(
                np.array_equal(source, target)
                for source, target in zip(
                    source_axes, (az_new, el_new, f_new, pol_new)
                )
            )
            return self._new_grid(
                az_new,
                el_new,
                f_new,
                pol_new,
                rcs_power=pwr_new,
                rcs_phase=phs_new,
                rcs_domain="power_phase",
                extra=self._exact_transform_extra(
                    lambda value: value[selection],
                    coordinate_change=("align-intersect-relabel" if relabeled else None),
                ),
            )

        # interp mode
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch for interp")

        self._check_axis_sorted(self.azimuths, "azimuth")
        self._check_axis_sorted(self.elevations, "elevation")
        self._check_axis_sorted(self.frequencies, "frequency")
        self._check_axis_sorted(other.azimuths, "azimuth")
        self._check_axis_sorted(other.elevations, "elevation")
        self._check_axis_sorted(other.frequencies, "frequency")

        # Interpolate the authoritative complex field directly.  GHOST files
        # intentionally retain a float64 raw amplitude alongside a float32
        # display power/phase pair; rebuilding the field from the latter first
        # can erase small but physically important coherent deltas.
        power_interp = np.asarray(self.rcs_power)
        complex_interp = np.asarray(self.rcs, dtype=np.complex128)
        for axis, old, new in (
            (0, self.azimuths, other.azimuths),
            (1, self.elevations, other.elevations),
            (2, self.frequencies, other.frequencies),
        ):
            power_interp = self._interp_real_axis(
                power_interp, old, new, axis
            )
            complex_interp = self._interp_complex_axis(
                complex_interp, old, new, axis
            )
        complex_valid = np.isfinite(complex_interp.real) & np.isfinite(
            complex_interp.imag
        )
        phase_interp = np.full(power_interp.shape, np.nan, dtype=np.float64)
        power_interp = np.asarray(power_interp, dtype=np.float64)
        power_interp[complex_valid] = np.abs(complex_interp[complex_valid]) ** 2
        phase_interp[complex_valid] = np.angle(complex_interp[complex_valid])
        interp_extra = self._derived_response_extra(
            operation="align-interpolate", coherent=True
        )
        self._invalidate_assembly_sampling_hash(
            interp_extra, "align-interpolate"
        )
        return self._new_grid(
            other.azimuths,
            other.elevations,
            other.frequencies,
            other.polarizations,
            rcs_power=power_interp,
            rcs_phase=phase_interp,
            rcs_domain="power_phase",
            extra=interp_extra,
        )

    @staticmethod
    def _check_axis_sorted(axis, name):
        axis = np.asarray(axis)
        if axis.size < 2:
            return
        if not np.all(np.diff(axis) > 0):
            raise ValueError(f"{name} axis must be strictly increasing for interp")

    @staticmethod
    def _interp_complex_axis(data, x_old, x_new, axis):
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        return RcsGrid._interp_linear_axis(data, x_old, x_new, axis)

    @staticmethod
    def _interp_real_axis(data, x_old, x_new, axis):
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        return RcsGrid._interp_linear_axis(data, x_old, x_new, axis)

    @staticmethod
    def _interp_linear_axis(data, x_old, x_new, axis):
        """Vectorized adjacent-bin interpolation; NaNs remain local."""
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        moved = np.moveaxis(np.asarray(data), axis, 0)
        right = np.searchsorted(x_old, x_new, side="left")
        right = np.clip(right, 0, len(x_old) - 1)
        exact = x_old[right] == x_new
        left = np.where(exact, right, np.maximum(right - 1, 0))
        denom = x_old[right] - x_old[left]
        weight = np.divide(
            x_new - x_old[left],
            denom,
            out=np.zeros_like(x_new, dtype=float),
            where=denom != 0.0,
        )
        reshape = (len(x_new),) + (1,) * (moved.ndim - 1)
        w = weight.reshape(reshape)
        out = moved[left] * (1.0 - w) + moved[right] * w
        return np.moveaxis(out.astype(moved.dtype, copy=False), 0, axis)


    def interpolate_axis(self, axis_name, new_values):
        """Linearly interpolate the grid onto new values along one numeric axis.

        Other axes are left unchanged. Raises if `new_values` extends beyond
        the existing axis range (no extrapolation).
        """
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2}
        key = str(axis_name).strip().lower()
        if key not in axis_map:
            raise ValueError(f"axis must be one of {list(axis_map)}")
        axis_idx = axis_map[key]
        new_arr = np.asarray(new_values, dtype=float).ravel()
        if new_arr.size == 0:
            raise ValueError("new axis must have at least one value")
        if new_arr.size > 1 and not np.all(np.diff(new_arr) > 0):
            raise ValueError("new axis must be strictly increasing")

        old_axes = [self.azimuths, self.elevations, self.frequencies]
        self._check_axis_sorted(old_axes[axis_idx], key)

        new_axes = list(old_axes)
        new_axes[axis_idx] = new_arr

        power_interp = self._interp_real_axis(
            self.rcs_power, old_axes[axis_idx], new_arr, axis_idx
        )
        complex_interp = self._interp_complex_axis(
            np.asarray(self.rcs, dtype=np.complex128),
            old_axes[axis_idx],
            new_arr,
            axis_idx,
        )
        complex_valid = np.isfinite(complex_interp.real) & np.isfinite(
            complex_interp.imag
        )
        power_interp = np.asarray(power_interp, dtype=np.float64)
        phase_interp = np.full(power_interp.shape, np.nan, dtype=np.float64)
        power_interp[complex_valid] = np.abs(complex_interp[complex_valid]) ** 2
        phase_interp[complex_valid] = np.angle(complex_interp[complex_valid])
        interp_extra = self._derived_response_extra(
            operation=f"interpolate-{key}", coherent=True
        )
        self._invalidate_assembly_sampling_hash(
            interp_extra, f"interpolate-{key}"
        )
        return self._new_grid(
            new_axes[0],
            new_axes[1],
            new_axes[2],
            self.polarizations,
            rcs_power=power_interp,
            rcs_phase=phase_interp,
            rcs_domain="power_phase",
            extra=interp_extra,
        )

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _clean_axis(axis):
        """Normalize an axis to float64 (numeric) or keep dtype (non-numeric).

        For float32 input, round-trips each value through its shortest-decimal
        repr so that user-intended values like 0.1 stay as 0.1 in float64
        instead of inheriting the float32 quantization noise (0.10000000149...).
        That way later ops like `shift_azimuth(180)` produce clean values
        (180.1 instead of 180.10000001).
        """
        arr = np.asarray(axis)
        if not np.issubdtype(arr.dtype, np.number):
            return arr
        if arr.dtype == np.float32:
            return arr.astype(str).astype(np.float64)
        return arr.astype(np.float64, copy=False)

    @staticmethod
    def _canonical_polarization_axis(polarizations):
        """Return stripped uppercase polarization identities without aliases.

        Polarization labels are identifiers, not display prose. Treating
        ``HH`` and ``hh`` as distinct allowed joins to create an axis that a
        later native save correctly rejected as a case-folded duplicate.
        Canonicalizing at every construction boundary keeps lookup, union, and
        serialization behavior consistent. Distinct input channels that fold
        to one identity are rejected rather than silently merged.
        """

        values = np.asarray(polarizations)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "polarizations must be a nonempty one-dimensional string axis"
            )
        labels = []
        for raw_value in values.tolist():
            if isinstance(raw_value, bytes):
                try:
                    label = raw_value.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("polarization labels must be UTF-8 strings") from exc
            elif isinstance(raw_value, (str, np.str_)):
                label = str(raw_value)
            else:
                raise ValueError(
                    f"polarization labels must be strings; got {raw_value!r}"
                )
            label = label.strip().upper()
            if not label:
                raise ValueError("polarization labels must not be blank")
            labels.append(label)
        if len(set(labels)) != len(labels):
            raise ValueError(
                "polarization labels must be unique after case normalization"
            )
        # Native unicode prevents object/pickle storage in .grim archives.
        return np.asarray(labels, dtype=str)

    @staticmethod
    def _axis_value_match(axis_arr, value, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        if np.issubdtype(axis_arr.dtype, np.number) and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return np.where(np.isclose(axis_arr, float(value), atol=tol, rtol=0.0))[0]
        return np.where(axis_arr == value)[0]

    @staticmethod
    def _indices_for_axis_values(axis_arr, values, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        values_arr = np.asarray(values)
        if values_arr.size == 0:
            return []
        if axis_arr.size == 0:
            return None
        if np.issubdtype(axis_arr.dtype, np.number) and np.issubdtype(
            values_arr.dtype, np.number
        ):
            axis_f = axis_arr.astype(float, copy=False).ravel()
            values_f = values_arr.astype(float, copy=False).ravel()
            order = np.argsort(axis_f, kind="stable")
            sorted_axis = axis_f[order]
            pos = np.searchsorted(sorted_axis, values_f)
            n = sorted_axis.size
            left = np.clip(pos - 1, 0, n - 1)
            right = np.clip(pos, 0, n - 1)
            d_left = np.abs(sorted_axis[left] - values_f)
            d_right = np.abs(sorted_axis[right] - values_f)
            use_right = d_right <= d_left
            sorted_idx = np.where(use_right, right, left)
            dist = np.where(use_right, d_right, d_left)
            if np.any(dist > tol):
                return None
            orig_idx = order[sorted_idx]
            seen = set()
            out = []
            for i in orig_idx.tolist():
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            return out
        idx_map = {}
        for i in range(axis_arr.size):
            v = axis_arr[i]
            key = v.item() if isinstance(v, np.generic) else v
            if key not in idx_map:
                idx_map[key] = i
        seen = set()
        out = []
        for value in values_arr:
            key = value.item() if isinstance(value, np.generic) else value
            if key not in idx_map:
                return None
            idx = idx_map[key]
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    @staticmethod
    def _axis_union(axis_arrays, tol=1e-6):
        if not axis_arrays:
            return np.asarray([])
        first_dtype = np.asarray(axis_arrays[0]).dtype
        numeric_axis = np.issubdtype(first_dtype, np.number)
        if not numeric_axis:
            seen = {}
            for axis_arr in axis_arrays:
                for value in np.asarray(axis_arr):
                    key = value.item() if isinstance(value, np.generic) else value
                    if key not in seen:
                        seen[key] = None
            return np.asarray(list(seen))
        parts = [np.asarray(a, dtype=float).ravel() for a in axis_arrays]
        combined = np.concatenate(parts) if parts else np.asarray([], dtype=float)
        if combined.size == 0:
            return np.asarray([])
        combined.sort(kind="mergesort")
        keep = np.ones(combined.size, dtype=bool)
        if tol <= 0:
            keep[1:] = combined[1:] != combined[:-1]
        else:
            last_kept = combined[0]
            for i in range(1, combined.size):
                if combined[i] - last_kept > tol:
                    last_kept = combined[i]
                else:
                    keep[i] = False
        return combined[keep]

    @staticmethod
    def _common_axis_alignment(axis_arrays, tol=1e-6):
        """Return a symmetric axis intersection and indices into every input.

        Numeric values match only when one value from every axis fits in a
        window no wider than ``tol``.  The lowest value in that window is the
        canonical output coordinate.  Sorting the values before matching makes
        both the coordinates and the chosen samples independent of which
        dataset happened to be selected first.
        """
        arrays = [np.asarray(axis).ravel() for axis in axis_arrays]
        if not arrays:
            return np.asarray([]), []

        try:
            tol = float(tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("tol must be a finite nonnegative number") from exc
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be a finite nonnegative number")

        numeric_flags = [np.issubdtype(array.dtype, np.number) for array in arrays]
        if any(numeric_flags) and not all(numeric_flags):
            raise TypeError("axis inputs must all be numeric or all be nonnumeric")

        if all(numeric_flags):
            sorted_values = []
            sorted_indices = []
            for array in arrays:
                values = array.astype(float, copy=False)
                finite_indices = np.flatnonzero(np.isfinite(values))
                order = np.argsort(values[finite_indices], kind="stable")
                original_indices = finite_indices[order]
                sorted_values.append(values[original_indices])
                sorted_indices.append(original_indices)

            if any(values.size == 0 for values in sorted_values):
                return np.asarray([], dtype=float), [[] for _ in arrays]

            pointers = np.zeros(len(arrays), dtype=np.int64)
            common_values = []
            matched_indices = [[] for _ in arrays]
            while all(
                pointer < values.size
                for pointer, values in zip(pointers, sorted_values)
            ):
                current = np.asarray(
                    [values[pointer] for values, pointer in zip(sorted_values, pointers)],
                    dtype=float,
                )
                low = float(np.min(current))
                high = float(np.max(current))
                if high - low <= tol:
                    common_values.append(low)
                    for axis_idx in range(len(arrays)):
                        matched_indices[axis_idx].append(
                            int(sorted_indices[axis_idx][pointers[axis_idx]])
                        )
                        pointers[axis_idx] += 1
                    continue

                # A lowest value cannot match the current maximum or any later
                # value from that maximum's axis, so discard every tied low.
                for axis_idx, value in enumerate(current):
                    if value == low:
                        pointers[axis_idx] += 1

            return np.asarray(common_values, dtype=float), matched_indices

        indices_by_value = []
        for array in arrays:
            mapping = {}
            for index, raw_value in enumerate(array):
                value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                mapping.setdefault(value, []).append(index)
            indices_by_value.append(mapping)

        common_values = set(indices_by_value[0])
        for mapping in indices_by_value[1:]:
            common_values.intersection_update(mapping)

        def _canonical_key(value):
            value_type = type(value)
            return (value_type.__module__, value_type.__qualname__, repr(value))

        output_values = []
        matched_indices = [[] for _ in arrays]
        for value in sorted(common_values, key=_canonical_key):
            occurrences = min(len(mapping[value]) for mapping in indices_by_value)
            output_values.extend([value] * occurrences)
            for axis_idx, mapping in enumerate(indices_by_value):
                matched_indices[axis_idx].extend(mapping[value][:occurrences])
        return np.asarray(output_values), matched_indices

    @classmethod
    def _axis_intersection(cls, axis_arrays, tol=1e-6):
        common, _indices = cls._common_axis_alignment(axis_arrays, tol=tol)
        return common

    @classmethod
    def _ensure_grids(cls, grids):
        checked = []
        for grid in grids:
            if not isinstance(grid, cls):
                raise TypeError("all inputs must be RcsGrid instances")
            checked.append(grid)
        if not checked:
            raise ValueError("at least one grid is required")
        return checked

    # Scalar declarations that remain meaningful on a derived response.  The
    # lists are intentionally narrow: solver certificates and arbitrary source
    # payloads must not be promoted to statements about a transformed grid.
    _FIELD_CONVENTION_EXTRA_KEYS = frozenset({
        "amplitude_version",
        "phase_reference",
        "time_convention",
        "polarization_basis",
        "amplitude_convention",
        "complex_field_domain",
    })
    _COORDINATE_LINEAGE_EXTRA_KEYS = frozenset({
        "source_format",
        "angular_coordinate_declaration_json",
        "angular_coordinate_system",
        "great_circle_coordinate_convention",
        "elevation_coordinate_convention",
        "ptm_cut_type",
        "ptm_roll",
        "ptm_tilt",
        "assembly_angular_coordinate_contract",
    })
    _ASSEMBLY_LINEAGE_EXTRA_KEYS = frozenset({
        "combine_role",
        "combine_role_note",
        "assembly_response_role",
        "assembly_base_sha256",
        "assembly_source_base_sha256_json",
        "assembly_base_response_sha256",
        "assembly_source_base_response_sha256",
        "assembly_base_response_transform",
        "source_monostatic_sha256",
        "feature_provenance_json",
        "assembly_provenance_json",
        "coherent_metadata_attestation_json",
        "coherent_metadata_assumption_json",
        "coherent_source_conventions_json",
    })
    _DERIVED_PROVENANCE_EXTRA_KEYS = frozenset({
        # Durable lineage for the guided exact complex subtraction workflow.
        # The record is intentionally retained by later sample transforms, but
        # its content hashes continue to identify the original two inputs.
        "support_reference_difference_json",
        "coherent_sample_qa_json",
        "coherent_metadata_assumption_json",
        "merge_metadata_assumption_json",
        "coherent_source_conventions_json",
        "elevation_pair_to_azimuth_json",
        "decimation_json",
        "statistics_reduction_json",
    })
    _RAW_AMPLITUDE_EXTRA_KEYS = ("rcs_amp_real", "rcs_amp_imag")

    def _safe_derived_scalar_extra(self, *, include_field_conventions=True):
        """Copy only scalar metadata with durable derived-grid semantics.

        Every ``sentri_*`` scalar is retained conservatively.  That prefix is
        vendor provenance as well as UI information, and losing it could make
        a native polar-theta table look like canonical signed elevation after
        an otherwise unrelated dataset operation.
        """

        allowed = set(self._COORDINATE_LINEAGE_EXTRA_KEYS)
        allowed.update(self._ASSEMBLY_LINEAGE_EXTRA_KEYS)
        allowed.update(self._DERIVED_PROVENANCE_EXTRA_KEYS)
        if include_field_conventions:
            allowed.update(self._FIELD_CONVENTION_EXTRA_KEYS)
        result = {}
        for key, value in self.extra.items():
            if key not in allowed and not str(key).startswith("sentri_"):
                continue
            array = np.asarray(value)
            if array.size != 1:
                continue
            result[key] = copy.deepcopy(value)
        return result

    def _has_native_sentri_coordinate_hazard(self):
        """Return whether this grid still uses vendor polar theta as elevation."""

        convention_values = []
        for container in (self.units or {}, self.extra or {}):
            value = container.get("elevation_coordinate_convention")
            if value is not None:
                convention_values.append(str(value).strip().casefold())
            value = container.get("sentri_elevation_convention")
            if value is not None:
                convention_values.append(str(value).strip().casefold())
        if "sentri_theta_top_zero" in convention_values:
            return True
        mapping = str(
            (self.extra or {}).get("sentri_coordinate_mapping", "") or ""
        ).casefold().replace(" ", "")
        return "elevation=theta" in mapping and "elevation=90-theta" not in mapping

    @classmethod
    def _carry_native_sentri_hazard(cls, extra, sources):
        """Make a native-SENTRi source impossible to hide in a derived grid."""

        if not any(grid._has_native_sentri_coordinate_hazard() for grid in sources):
            return extra
        # A mixture containing native theta cannot truthfully retain a
        # canonical Assembly angular contract from another source.
        extra.pop("assembly_angular_coordinate_contract", None)
        extra["source_format"] = "derived response includes native SENTRi coordinates"
        extra["sentri_elevation_convention"] = "sentri_theta_top_zero"
        extra["sentri_coordinate_mapping"] = (
            "elevation=theta; native SENTRi polar coordinates retained"
        )
        return extra

    @staticmethod
    def _invalidate_assembly_sampling_hash(extra, operation):
        """Retain source lineage while invalidating a sampling-bound base hash."""

        digest = extra.pop("assembly_base_response_sha256", None)
        has_assembly_lineage = digest is not None or any(
            key in extra
            for key in (
                "assembly_response_role",
                "assembly_base_sha256",
                "source_monostatic_sha256",
                "feature_provenance_json",
                "assembly_provenance_json",
            )
        )
        if digest is not None and str(np.asarray(digest).reshape(-1)[0]).strip():
            prior = extra.get("assembly_source_base_response_sha256")
            if prior is None:
                extra["assembly_source_base_response_sha256"] = copy.deepcopy(digest)
            elif str(np.asarray(prior).reshape(-1)[0]).strip().casefold() != str(
                np.asarray(digest).reshape(-1)[0]
            ).strip().casefold():
                # Contradictory lineage must stay visibly contradictory rather
                # than choosing one digest.  This value is provenance only;
                # Assembly deliberately does not accept it as a current hash.
                extra["assembly_source_base_response_sha256"] = json.dumps(
                    sorted({
                        str(np.asarray(prior).reshape(-1)[0]).strip().lower(),
                        str(np.asarray(digest).reshape(-1)[0]).strip().lower(),
                    }),
                    separators=(",", ":"),
                )
        if has_assembly_lineage:
            extra["assembly_base_response_transform"] = str(operation)
        return extra

    def _exact_transform_extra(
        self,
        array_transform=None,
        *,
        preserve_all=False,
        preserve_raw=True,
        coordinate_change=None,
        preserve_angular_contract=True,
    ):
        """Metadata policy for an exact sample-preserving transform.

        ``array_transform`` is applied only to the authoritative raw solver
        field.  Other grid-shaped producer arrays are deliberately not guessed
        at.  ``preserve_all`` is reserved for representation-only operations
        such as phase wrapping where neither axes nor samples change.
        """

        if preserve_all:
            extra = {
                key: copy.deepcopy(value)
                for key, value in self._extra_to_write().items()
            }
            if self._complete_authoritative_raw_arrays() is None:
                for key in (
                    *self._RAW_AMPLITUDE_EXTRA_KEYS,
                    "raw_complex_amplitude_preserved",
                ):
                    extra.pop(key, None)
        else:
            extra = self._safe_derived_scalar_extra(
                include_field_conventions=True
            )
            pair = self._complete_authoritative_raw_arrays()
            if preserve_raw and pair is not None:
                real_array, imag_array = pair
                transform = array_transform or (
                    lambda value: np.array(value, copy=True)
                )
                transformed_real = np.asarray(transform(real_array))
                transformed_imag = np.asarray(transform(imag_array))
                if np.shares_memory(transformed_real, real_array):
                    transformed_real = np.array(transformed_real, copy=True)
                if np.shares_memory(transformed_imag, imag_array):
                    transformed_imag = np.array(transformed_imag, copy=True)
                extra["rcs_amp_real"] = transformed_real
                extra["rcs_amp_imag"] = transformed_imag
                extra["raw_complex_amplitude_preserved"] = True
        if not preserve_all:
            self._carry_native_sentri_hazard(extra, (self,))
        if coordinate_change:
            self._invalidate_assembly_sampling_hash(extra, coordinate_change)
        if not preserve_angular_contract:
            extra.pop("assembly_angular_coordinate_contract", None)
        return extra

    def _derived_response_extra(
        self,
        others=(),
        *,
        operation,
        coherent,
        attestation_extra=None,
    ):
        """Metadata policy for arithmetic/interpolated/statistical responses."""

        sources = (self, *tuple(others))
        extra = self._safe_derived_scalar_extra(
            include_field_conventions=bool(coherent)
        )
        self._carry_native_sentri_hazard(extra, sources)

        def declared_values(key):
            values = []
            for grid in sources:
                value = grid._declared_scalar_metadata(key)
                if value:
                    values.append(value)
            return values

        source_roles = [
            grid._declared_scalar_metadata(
                "assembly_response_role"
            ).strip().casefold()
            or None
            for grid in sources
        ]
        roles = [role for role in source_roles if role is not None]
        base_hashes = {
            value.strip().casefold()
            for value in declared_values("assembly_base_sha256")
        }
        response_hashes = {
            value.strip().casefold()
            for value in declared_values("assembly_base_response_sha256")
        }
        if len(base_hashes) == 1:
            extra["assembly_base_sha256"] = next(iter(base_hashes))
        elif len(base_hashes) > 1:
            extra.pop("assembly_base_sha256", None)
            extra["assembly_source_base_sha256_json"] = json.dumps(
                sorted(base_hashes), separators=(",", ":")
            )
        if len(response_hashes) == 1:
            extra["assembly_base_response_sha256"] = next(iter(response_hashes))
        elif len(response_hashes) > 1:
            extra.pop("assembly_base_response_sha256", None)
            extra["assembly_source_base_response_sha256"] = json.dumps(
                sorted(response_hashes), separators=(",", ":")
            )

        if coherent:
            # A convention declaration belongs on the combined artifact only
            # when every source declared the same value. A one-sided value is
            # evidence about that source, not proof about all output samples.
            for key in (
                "phase_reference",
                "time_convention",
                "polarization_basis",
                "amplitude_version",
                "amplitude_convention",
                "complex_field_domain",
            ):
                values = [
                    grid._declared_scalar_metadata(key) for grid in sources
                ]
                if any(not value for value in values):
                    extra.pop(key, None)
                    continue
                if key == "time_convention":
                    normalized = {
                        self._canonical_time_convention(value)
                        for value in values
                    }
                else:
                    normalized = {
                        " ".join(value.split()).casefold() for value in values
                    }
                if len(normalized) == 1:
                    extra[key] = values[0]
                else:
                    # The compatibility preflight normally rejects this. Keep
                    # the metadata policy defensive for internal callers.
                    extra.pop(key, None)
            if "body_plus_features" in roles:
                extra["assembly_response_role"] = "body_plus_features"
            elif source_roles and all(
                role == "features_only_delta" for role in source_roles
            ) and len(base_hashes) == 1 and len(response_hashes) == 1:
                extra["assembly_response_role"] = "features_only_delta"
            elif roles:
                extra["assembly_response_role"] = "coherent_field_sum"
            if roles or any(
                grid._declared_scalar_metadata("combine_role") for grid in sources
            ):
                extra["combine_role"] = "coherent"
        else:
            # A power/statistical result is never a reusable coherent feature
            # delta.  A body-bearing source remains explicitly body-bearing so
            # downstream duplicate-body guards cannot be bypassed.
            if "body_plus_features" in roles:
                extra["assembly_response_role"] = "body_plus_features"
            elif roles:
                extra["assembly_response_role"] = "incoherent_power_sum"
            extra["combine_role"] = "power"
            for key in (
                "phase_reference",
                "time_convention",
                "amplitude_convention",
                "complex_field_domain",
            ):
                extra.pop(key, None)
            self._invalidate_assembly_sampling_hash(extra, operation)

        if attestation_extra:
            extra.update(copy.deepcopy(attestation_extra))
        return extra

    def _new_grid(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=None,
        *,
        rcs_power=None,
        rcs_phase=None,
        rcs_domain=None,
        history=None,
        units=None,
        extra=None,
        _adopt_clean_arrays=False,
    ):
        if extra is None:
            # Safe future-proof fallback.  Current operations pass one of the
            # explicit policies above, but a new caller must never erase native
            # SENTRi or Assembly response semantics merely by omission.
            extra = self._safe_derived_scalar_extra(
                include_field_conventions=True
            )
            self._carry_native_sentri_hazard(extra, (self,))
        return RcsGrid(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs,
            rcs_power=rcs_power,
            rcs_phase=rcs_phase,
            rcs_domain=(self.rcs_domain if rcs_domain is None else rcs_domain),
            source_path=self.source_path,
            history=history if history is not None else self.history,
            units=dict(self.units if units is None else units),
            extra=extra,
            _adopt_clean_arrays=(
                _ADOPT_CLEAN_ARRAYS_TOKEN if _adopt_clean_arrays else None
            ),
        )

    def _power_from_values(self, rcs_value):
        values_raw = np.asarray(rcs_value)
        if np.iscomplexobj(values_raw):
            values = np.asarray(values_raw, dtype=np.complex128)
            power = np.abs(values) ** 2
        else:
            power = np.asarray(values_raw, dtype=float)
        power = np.asarray(power, dtype=float)
        finite = np.isfinite(power)
        out = np.zeros_like(power, dtype=float)
        out[finite] = np.maximum(power[finite], 0.0)
        out[~finite] = np.nan
        return out


    def rcs_to_linear(self, rcs_value):
        """Convert complex field or real-power values to linear power."""
        return self._power_from_values(rcs_value)

    def linear_to_dbsm(self, linear_value, eps=1e-12):
        linear = np.asarray(linear_value, dtype=float)
        linear = np.where(np.isfinite(linear), linear, np.nan)
        linear = np.maximum(linear, eps)
        return 10.0 * np.log10(linear)

    def _frequency_value_to_hz(self, frequency_value):
        freq = np.asarray(frequency_value, dtype=float)
        unit = self._supported_unit("frequency", _FREQUENCY_UNITS, "GHz")
        if unit == "Hz":
            return freq
        if unit == "MHz":
            return freq * 1.0e6
        if unit == "kHz":
            return freq * 1.0e3
        if unit == "GHz":
            return freq * 1.0e9
        raise AssertionError(f"unhandled canonical frequency unit: {unit}")

    def linear_to_dbke(self, linear_value, frequency_value, eps=1e-12):
        linear = np.asarray(linear_value, dtype=float)
        linear = np.where(np.isfinite(linear), linear, np.nan)
        linear = np.maximum(linear, eps)
        freq_hz = self._frequency_value_to_hz(frequency_value)
        freq_hz = np.asarray(freq_hz, dtype=float)
        freq_hz = np.where(np.isfinite(freq_hz) & (freq_hz > 0.0), freq_hz, np.nan)
        factor = (2.0 * np.pi * freq_hz) / C0
        return 10.0 * np.log10(factor * linear)

    def dbke_to_linear(self, dbke_value, frequency_value):
        dbke = np.asarray(dbke_value, dtype=float)
        freq_hz = self._frequency_value_to_hz(frequency_value)
        freq_hz = np.asarray(freq_hz, dtype=float)
        factor = np.where(np.isfinite(freq_hz) & (freq_hz > 0.0), C0 / (2.0 * np.pi * freq_hz), np.nan)
        return factor * (10.0 ** (dbke / 10.0))

    def default_log_unit(self):
        raw = str((self.units or {}).get("rcs_log_unit", "dBsm")).strip().lower()
        if raw == "dbke":
            return "dBke"
        if raw == "db":
            return "dB"
        return "dBsm"

    def linear_to_default_db(self, linear_value, frequency_value=None, eps=1e-12):
        if self.default_log_unit().lower() == "dbke":
            if frequency_value is None:
                raise ValueError("frequency_value is required for dBke conversion")
            return self.linear_to_dbke(linear_value, frequency_value, eps=eps)
        return self.linear_to_dbsm(linear_value, eps=eps)

    def default_db_to_linear(self, db_value, frequency_value=None):
        """Inverse of ``linear_to_default_db`` — convert dB display values back
        to linear power using the dataset's default log unit (dBsm or dBke).
        """
        if self.default_log_unit().lower() == "dbke":
            if frequency_value is None:
                raise ValueError("frequency_value is required for dBke conversion")
            return self.dbke_to_linear(db_value, frequency_value)
        return 10.0 ** (np.asarray(db_value, dtype=float) / 10.0)

    def axis_crop(
        self,
        *,
        azimuths=None,
        elevations=None,
        frequencies=None,
        polarizations=None,
        azimuth_range=None,
        elevation_range=None,
        frequency_range=None,
        azimuth_min=None,
        azimuth_max=None,
        elevation_min=None,
        elevation_max=None,
        frequency_min=None,
        frequency_max=None,
        tol=1e-6,
    ):
        """Return a grid cropped by explicit axis values and/or numeric ranges."""

        def _resolve_range(raw_range, vmin, vmax):
            if raw_range is not None:
                if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                    raise ValueError("axis range must be a 2-item [min, max] sequence")
                return raw_range[0], raw_range[1]
            if vmin is None and vmax is None:
                return None
            return vmin, vmax

        azimuth_range = _resolve_range(azimuth_range, azimuth_min, azimuth_max)
        elevation_range = _resolve_range(elevation_range, elevation_min, elevation_max)
        frequency_range = _resolve_range(frequency_range, frequency_min, frequency_max)

        def _axis_indices(axis_arr, axis_values, axis_range, axis_name, axis_tol):
            all_indices = list(range(len(axis_arr)))
            values = self._as_list(axis_values)
            if values is not None:
                if axis_name == "polarization":
                    values = [str(value).strip().upper() for value in values]
                selected = self._indices_for_axis_values(axis_arr, values, tol=axis_tol)
                if selected is None:
                    raise ValueError(f"{axis_name} contains value(s) not present in dataset")
                indices = selected
            else:
                indices = all_indices

            if axis_range is not None:
                lo, hi = axis_range
                if lo is not None:
                    lo = float(lo)
                if hi is not None:
                    hi = float(hi)
                if lo is not None and hi is not None and lo > hi:
                    lo, hi = hi, lo

                axis_num = np.asarray(axis_arr, dtype=float)
                range_mask = np.ones(axis_num.shape[0], dtype=bool)
                if lo is not None:
                    range_mask &= axis_num >= (lo - axis_tol)
                if hi is not None:
                    range_mask &= axis_num <= (hi + axis_tol)
                range_idx = set(np.where(range_mask)[0].tolist())
                indices = [idx for idx in indices if idx in range_idx]

            if not indices:
                raise ValueError(f"{axis_name} crop produced no samples")
            return indices

        az_idx = _axis_indices(self.azimuths, azimuths, azimuth_range, "azimuth", tol)
        el_idx = _axis_indices(self.elevations, elevations, elevation_range, "elevation", tol)
        f_idx = _axis_indices(self.frequencies, frequencies, frequency_range, "frequency", tol)
        p_idx = _axis_indices(self.polarizations, polarizations, None, "polarization", 0.0)
        selection = np.ix_(az_idx, el_idx, f_idx, p_idx)

        return self._new_grid(
            self.azimuths[az_idx],
            self.elevations[el_idx],
            self.frequencies[f_idx],
            self.polarizations[p_idx],
            rcs_power=self.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)],
            rcs_phase=self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)],
            extra=self._exact_transform_extra(
                lambda value: value[selection]
            ),
        )

    @staticmethod
    def _merge_equivalent_sample_blocks(
        existing_power,
        existing_phase,
        incoming_power,
        incoming_phase,
        *,
        context,
    ):
        """Merge complementary/equivalent samples or reject a seam conflict."""

        existing_power = np.asarray(existing_power)
        existing_phase = np.asarray(existing_phase)
        incoming_power = np.asarray(incoming_power)
        incoming_phase = np.asarray(incoming_phase)
        existing_finite = np.isfinite(existing_power)
        incoming_finite = np.isfinite(incoming_power)
        both = existing_finite & incoming_finite
        power_equal = both & np.isclose(
            existing_power, incoming_power, rtol=1.0e-6, atol=1.0e-12
        )
        power_conflict = both & ~power_equal

        both_phase = (
            power_equal
            & np.isfinite(existing_phase)
            & np.isfinite(incoming_phase)
        )
        both_zero = both & (existing_power == 0.0) & (incoming_power == 0.0)
        phase_delta = np.abs(
            np.angle(np.exp(1j * (existing_phase - incoming_phase)))
        )
        phase_conflict = both_phase & ~both_zero & (phase_delta > 1.0e-5)
        if np.any(power_conflict) or np.any(phase_conflict):
            raise ValueError(
                f"{context}: conflicting finite seam samples would overlap"
            )

        take_power = ~existing_finite & incoming_finite
        if np.any(take_power):
            existing_power[take_power] = incoming_power[take_power]
            existing_phase[take_power] = np.where(
                np.isfinite(incoming_phase[take_power]),
                incoming_phase[take_power],
                np.nan,
            )
        fill_phase = (
            power_equal
            & ~np.isfinite(existing_phase)
            & np.isfinite(incoming_phase)
        )
        existing_phase[fill_phase] = incoming_phase[fill_phase]

    @staticmethod
    def _merge_equivalent_raw_blocks(
        existing_real,
        existing_imag,
        incoming_real,
        incoming_imag,
        *,
        context,
    ):
        """Merge authoritative float64 seam fields or reject hidden conflicts."""

        existing_real = np.asarray(existing_real)
        existing_imag = np.asarray(existing_imag)
        incoming_real = np.asarray(incoming_real)
        incoming_imag = np.asarray(incoming_imag)
        existing_finite = np.isfinite(existing_real) & np.isfinite(existing_imag)
        incoming_finite = np.isfinite(incoming_real) & np.isfinite(incoming_imag)
        both = existing_finite & incoming_finite
        equivalent = (
            np.isclose(
                existing_real,
                incoming_real,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            & np.isclose(
                existing_imag,
                incoming_imag,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        )
        if np.any(both & ~equivalent):
            raise ValueError(
                f"{context}: conflicting authoritative raw seam samples would overlap"
            )
        take = ~existing_finite & incoming_finite
        existing_real[take] = incoming_real[take]
        existing_imag[take] = incoming_imag[take]

    def mirror_about_azimuth(self, azimuth_deg: float):
        """Mirror azimuth axis about a reference angle and return a new grid.

        The transformed axis is `az' = 2*azimuth_deg - az`. Output azimuths are
        sorted ascending, with samples reordered to match.
        """
        about = self._angle_value_from_degrees(azimuth_deg, "azimuth")

        az = np.asarray(self.azimuths, dtype=float)
        mirrored_az = (2.0 * about) - az
        order = np.argsort(mirrored_az, kind="stable")

        return self._new_grid(
            mirrored_az[order],
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=self.rcs_power[order, :, :, :],
            rcs_phase=self.rcs_phase[order, :, :, :],
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                lambda value: np.take(value, order, axis=0),
                coordinate_change="mirror-azimuth",
                preserve_angular_contract=False,
            ),
        )

    def swap_elevation_azimuth(self):
        """Swap the elevation and azimuth axes and return a new grid."""
        swapped_units = copy.deepcopy(self.units)
        azimuth_unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        elevation_unit = self._supported_unit("elevation", _ANGLE_UNITS, "deg")
        swapped_units["azimuth"] = elevation_unit
        swapped_units["elevation"] = azimuth_unit
        return self._new_grid(
            np.array(self.elevations, copy=True),
            np.array(self.azimuths, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.swapaxes(self.rcs_power, 0, 1).copy(),
            rcs_phase=np.swapaxes(self.rcs_phase, 0, 1).copy(),
            rcs_domain="power_phase",
            units=swapped_units,
            extra=self._exact_transform_extra(
                lambda value: np.swapaxes(value, 0, 1),
                coordinate_change="swap-elevation-azimuth",
                preserve_angular_contract=False,
            ),
        )

    def convert_axis_units(
        self,
        *,
        azimuth="deg",
        elevation="deg",
        frequency="GHz",
    ):
        """Convert numeric-axis storage units without changing physical samples."""

        target_az = self._canonical_unit(azimuth, _ANGLE_UNITS, "deg")
        target_el = self._canonical_unit(elevation, _ANGLE_UNITS, "deg")
        target_frequency = self._canonical_unit(
            frequency, _FREQUENCY_UNITS, "GHz"
        )
        source_az = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        source_el = self._supported_unit("elevation", _ANGLE_UNITS, "deg")
        source_frequency = self._supported_unit(
            "frequency", _FREQUENCY_UNITS, "GHz"
        )

        def convert_angle(values, source, target):
            values = np.asarray(values, dtype=float)
            degrees = np.rad2deg(values) if source == "rad" else values
            return np.deg2rad(degrees) if target == "rad" else degrees.copy()

        frequency_to_hz = {
            "Hz": 1.0,
            "kHz": 1.0e3,
            "MHz": 1.0e6,
            "GHz": 1.0e9,
        }
        frequency_hz = (
            np.asarray(self.frequencies, dtype=float)
            * frequency_to_hz[source_frequency]
        )
        converted_frequency = frequency_hz / frequency_to_hz[target_frequency]
        converted_units = copy.deepcopy(self.units or {})
        converted_units.update(
            azimuth=target_az,
            elevation=target_el,
            frequency=target_frequency,
        )
        history_entry = (
            "Convert axis storage units without interpolation: "
            f"azimuth {source_az}->{target_az}, elevation {source_el}->{target_el}, "
            f"frequency {source_frequency}->{target_frequency}"
        )
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return self._new_grid(
            convert_angle(self.azimuths, source_az, target_az),
            convert_angle(self.elevations, source_el, target_el),
            converted_frequency,
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            units=converted_units,
            history=history,
            extra=self._exact_transform_extra(
                coordinate_change="convert-axis-storage-units"
            ),
        )

    def shift_azimuth(self, delta_deg: float):
        """Shift azimuth axis by a constant offset and return a new grid."""
        delta = self._angle_value_from_degrees(delta_deg, "azimuth")
        shifted_az = np.asarray(self.azimuths, dtype=float) + delta
        return self._new_grid(
            shifted_az,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="shift-azimuth",
                preserve_angular_contract=False,
            ),
        )

    def wrap_azimuth(self, mode: str):
        """Wrap azimuth axis into the given range and return a new grid.

        ``mode`` is ``"0_360"`` for [0, 360) or ``"-180_180"`` for [-180, 180).
        Output azimuths are sorted ascending; samples are reordered to match.
        Degree axes use 360/180 and radian axes use 2*pi/pi.  If wrapping
        collapses distinct inputs onto one seam coordinate, complementary or
        equivalent samples are merged; conflicting finite samples are rejected
        instead of silently discarding one.
        """
        az = np.asarray(self.azimuths, dtype=float)
        unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        period = (2.0 * np.pi) if unit == "rad" else 360.0
        half_period = 0.5 * period
        seam_tol = float(np.deg2rad(1.0e-9)) if unit == "rad" else 1.0e-9
        if mode == "0_360":
            wrapped = np.mod(az, period)
            wrapped[np.isclose(wrapped, period, atol=seam_tol, rtol=0.0)] = 0.0
            wrapped[np.isclose(wrapped, 0.0, atol=seam_tol, rtol=0.0)] = 0.0
        elif mode == "-180_180":
            wrapped = np.mod(az + half_period, period) - half_period
            wrapped[
                np.isclose(wrapped, half_period, atol=seam_tol, rtol=0.0)
                | np.isclose(wrapped, -half_period, atol=seam_tol, rtol=0.0)
            ] = -half_period
        else:
            raise ValueError(f"unknown wrap mode: {mode!r}")

        order = np.argsort(wrapped, kind="stable")
        groups = []
        for source_index in order.tolist():
            if not groups or (
                wrapped[source_index] - wrapped[groups[-1][0]] > seam_tol
            ):
                groups.append([source_index])
            else:
                groups[-1].append(source_index)
        unique_vals = np.asarray(
            [wrapped[group[0]] for group in groups], dtype=float
        )
        output_shape = (len(groups),) + self.rcs_power.shape[1:]
        output_power = np.full(output_shape, np.nan, dtype=self.rcs_power.dtype)
        output_phase = np.full(output_shape, np.nan, dtype=self.rcs_phase.dtype)
        raw_pair = self._complete_authoritative_raw_arrays()
        preserve_raw = raw_pair is not None
        if preserve_raw:
            raw_real = np.asarray(raw_pair[0], dtype=np.float64)
            raw_imag = np.asarray(raw_pair[1], dtype=np.float64)
            output_raw_real = np.full(output_shape, np.nan, dtype=np.float64)
            output_raw_imag = np.full(output_shape, np.nan, dtype=np.float64)
        for output_index, group in enumerate(groups):
            for source_index in group:
                context = (
                    f"azimuth wrap at {unique_vals[output_index]:.12g} {unit}"
                )
                self._merge_equivalent_sample_blocks(
                    output_power[output_index],
                    output_phase[output_index],
                    self.rcs_power[source_index],
                    self.rcs_phase[source_index],
                    context=context,
                )
                if preserve_raw:
                    self._merge_equivalent_raw_blocks(
                        output_raw_real[output_index],
                        output_raw_imag[output_index],
                        raw_real[source_index],
                        raw_imag[source_index],
                        context=context,
                    )
        if preserve_raw:
            unmodeled = ~np.isfinite(output_power)
            output_raw_real[unmodeled] = np.nan
            output_raw_imag[unmodeled] = np.nan
        wrapped_extra = self._exact_transform_extra(
            coordinate_change="wrap-azimuth", preserve_raw=False
        )
        if preserve_raw:
            wrapped_extra["rcs_amp_real"] = output_raw_real
            wrapped_extra["rcs_amp_imag"] = output_raw_imag
            wrapped_extra["raw_complex_amplitude_preserved"] = True
        return self._new_grid(
            unique_vals,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=output_power,
            rcs_phase=output_phase,
            rcs_domain="power_phase",
            extra=wrapped_extra,
        )

    def wrap_phase(self, mode: str):
        """Wrap stored phase while preserving power and the complex field.

        ``mode`` is ``"0_360"`` for [0, 360) degrees or ``"-180_180"``
        for [-180, 180) degrees.  Phase is stored in radians, so only a
        modulo-2*pi representation change is made.  Missing phase remains
        missing and power samples are copied without modification.
        """

        if mode not in {"0_360", "-180_180"}:
            raise ValueError(
                "phase wrap mode must be '0_360' or '-180_180'"
            )

        wrapped_phase = np.array(self.rcs_phase, copy=True)
        period = 2.0 * np.pi
        with np.errstate(invalid="ignore"):
            if mode == "0_360":
                np.remainder(wrapped_phase, period, out=wrapped_phase)
                range_label = "[0, 360) deg"
            else:
                np.add(wrapped_phase, np.pi, out=wrapped_phase)
                np.remainder(wrapped_phase, period, out=wrapped_phase)
                np.subtract(wrapped_phase, np.pi, out=wrapped_phase)
                range_label = "[-180, 180) deg"

        history_entry = f"Wrap phase to {range_label}; complex field unchanged"
        prior_history = str(self.history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )
        wrapped_units = dict(self.units)
        wrapped_units["phase_wrap"] = mode
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=wrapped_phase,
            history=history,
            units=wrapped_units,
            extra=self._exact_transform_extra(preserve_all=True),
        )

    def round_azimuths(self, decimals: int):
        """Round azimuth axis values to ``decimals`` decimal places (no resampling).

        Use to clean up floating-point noise like 180.0001 -> 180.0.
        Raises if rounding collapses two distinct azimuths into the same value.
        """
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.azimuths, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding azimuths to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            rounded,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="round-azimuth"
            ),
        )

    def round_elevations(self, decimals: int):
        """Round elevation axis values to ``decimals`` decimal places (no resampling)."""
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.elevations, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding elevations to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            rounded,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="round-elevation"
            ),
        )

    def round_frequencies(self, decimals: int):
        """Round frequency axis values to ``decimals`` decimal places (no resampling)."""
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.frequencies, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding frequencies to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            rounded,
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="round-frequency"
            ),
        )

    def shift_elevation(self, delta_deg: float):
        """Shift elevation axis by a constant offset and return a new grid."""
        delta = self._angle_value_from_degrees(delta_deg, "elevation")
        shifted_el = np.asarray(self.elevations, dtype=float) + delta
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            shifted_el,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
            extra=self._exact_transform_extra(
                coordinate_change="shift-elevation",
                preserve_angular_contract=False,
            ),
        )

    def convert_sentri_elevation_to_grim(self):
        """Convert native SENTRi polar theta to GRIM signed elevation.

        SENTRi uses a polar angle measured down from the top look: 0 degrees is
        top-down, 90 degrees is waterline, and 180 degrees is bottom-up.  GRIM's
        conic elevation is positive above waterline, so the exact relabel is
        ``elevation = 90 - theta``.  The transformed elevation axis is
        stable-sorted and every grid-shaped sample/provenance array follows the
        same permutation.  SENTRi phi is also wrapped into GRIM's canonical
        [0, 360) degree azimuth axis and sorted.  There is no interpolation and
        no phase change.

        Explicitly incompatible elevation metadata is rejected. If convention
        metadata is absent, selecting this format-specific operation records
        the native-SENTRi assumption instead of blocking the conversion.
        """

        native_tag = "sentri_theta_top_zero"
        grim_tag = "grim_elevation_waterline_zero_top_positive"
        units = copy.deepcopy(self.units)
        extra = dict(self.extra)
        convention = str(
            units.get(
                "elevation_coordinate_convention",
                extra.get("sentri_elevation_convention", ""),
            )
            or ""
        ).strip().lower()
        source_format = str(extra.get("source_format", "") or "").strip()

        if convention == grim_tag:
            raise ValueError("dataset already uses GRIM signed elevation")
        if convention and convention != native_tag:
            raise ValueError(
                "SENTRi coordinate conversion cannot override explicit "
                f"elevation convention {convention!r}"
            )
        assumed_native_convention = not bool(convention)

        elevation_unit = self._canonical_unit(
            units.get("elevation"), _ANGLE_UNITS, "deg"
        )
        if elevation_unit != "deg":
            raise ValueError(
                "SENTRi elevation conversion requires a degree-valued "
                f"elevation axis; got {units.get('elevation')!r}"
            )
        azimuth_unit = self._canonical_unit(
            units.get("azimuth"), _ANGLE_UNITS, "deg"
        )
        if azimuth_unit != "deg":
            raise ValueError(
                "SENTRi azimuth conversion requires a degree-valued azimuth "
                f"axis; got {units.get('azimuth')!r}"
            )

        native_theta = np.asarray(self.elevations, dtype=float)
        if np.any(~np.isfinite(native_theta)):
            raise ValueError("SENTRi theta axis must contain only finite values")
        tolerance = 1.0e-9
        if np.any(native_theta < -tolerance) or np.any(
            native_theta > 180.0 + tolerance
        ):
            raise ValueError(
                "native SENTRi theta must be within 0..180 degrees before "
                "conversion"
            )

        converted_elevation = 90.0 - native_theta
        converted_elevation[np.abs(converted_elevation) <= tolerance] = 0.0
        order = np.argsort(converted_elevation, kind="stable")
        converted_elevation = converted_elevation[order]
        if (
            converted_elevation.size > 1
            and np.any(np.diff(converted_elevation) <= tolerance)
        ):
            raise ValueError(
                "SENTRi elevation conversion would create duplicate or "
                "near-duplicate GRIM elevation coordinates within 1e-9 deg"
            )

        native_azimuth = np.asarray(self.azimuths, dtype=float)
        if np.any(~np.isfinite(native_azimuth)):
            raise ValueError("SENTRi phi axis must contain only finite values")
        converted_azimuth = np.mod(native_azimuth, 360.0)
        converted_azimuth[
            np.isclose(converted_azimuth, 360.0, atol=tolerance, rtol=0.0)
            | np.isclose(converted_azimuth, 0.0, atol=tolerance, rtol=0.0)
        ] = 0.0
        azimuth_order = np.argsort(converted_azimuth, kind="stable")
        converted_azimuth = converted_azimuth[azimuth_order]
        if (
            converted_azimuth.size > 1
            and np.any(np.diff(converted_azimuth) <= tolerance)
        ):
            raise ValueError(
                "SENTRi azimuth wrapping would create duplicate or "
                "near-duplicate coordinates within 1e-9 deg; reload the "
                "source with read_SENTRi so its seam-precedence policy can "
                "resolve the duplicate first"
            )

        power = np.take(self.rcs_power, azimuth_order, axis=0)
        power = np.take(power, order, axis=1)
        phase = np.take(self.rcs_phase, azimuth_order, axis=0)
        phase = np.take(phase, order, axis=1)
        original_shape = tuple(self.rcs_power.shape)
        stale_grid_metadata = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
        }
        converted_extra = {}
        for key, value in extra.items():
            if key in stale_grid_metadata:
                continue
            value_array = np.asarray(value)
            if (
                value_array.ndim >= 4
                and tuple(value_array.shape[:4]) == original_shape
            ):
                converted_value = np.take(value_array, azimuth_order, axis=0)
                converted_extra[key] = np.take(converted_value, order, axis=1)
            else:
                converted_extra[key] = value

        self._drop_malformed_raw_metadata(converted_extra)
        self._invalidate_assembly_sampling_hash(
            converted_extra, "convert-native-sentri-to-grim"
        )

        units["elevation_coordinate_convention"] = grim_tag
        converted_extra["sentri_elevation_convention"] = grim_tag
        converted_extra["assembly_angular_coordinate_contract"] = (
            "ghost.radar-azimuth-elevation.coming-from.deg.v1"
        )
        converted_extra["sentri_coordinate_mapping"] = (
            "GRIM elevation = 90 deg - native SENTRi theta; "
            "azimuth=wrapped phi"
        )
        if assumed_native_convention:
            converted_extra["sentri_coordinate_assumption_json"] = json.dumps(
                {
                    "schema": "grim.sentri-coordinate-assumption.v1",
                    "operation_requested": True,
                    "source_format": source_format or None,
                    "source_elevation_convention_missing": True,
                    "assumed_convention": native_tag,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        prior_history = str(self.history or "").strip()
        history_entry = (
            "Convert native SENTRi coordinates to GRIM conic angles "
            "(elevation=90-theta; azimuth=phi wrapped to [0,360) deg); "
            "stable-sorted axes and sample arrays; no interpolation or phase "
            "change"
        )
        if assumed_native_convention:
            history_entry += "; untagged source convention assumed from operation"
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )

        return RcsGrid(
            converted_azimuth,
            converted_elevation,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=units,
            extra=converted_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    def combine_elevation_pair_to_azimuth_360(
        self,
        elevation_lo: float | None = None,
        elevation_hi: float | None = None,
        *,
        azimuth_shift_deg: float = 180.0,
        tol: float = 1e-6,
        assumptions_attested: bool = False,
    ):
        """Stitch two elevation cuts into one 0-360 azimuth cut.

        The lower-elevation cut keeps its original azimuth values. The higher
        cut is shifted by the degree-valued `azimuth_shift_deg` and merged onto
        the same output elevation plane. On radian-native data the shift and
        tolerance are converted internally. Equivalent/complementary overlap
        bins merge; conflicting finite seam samples are rejected. Because this
        is an acquisition-specific relabel rather than a general spherical
        coordinate transform, untagged inputs require explicit attestation and
        the two elevations must be equal and opposite about zero.
        """

        if not isinstance(assumptions_attested, (bool, np.bool_)):
            raise TypeError("assumptions_attested must be True or False")

        el_axis = np.asarray(self.elevations, dtype=float)
        if el_axis.size < 2:
            raise ValueError("need at least 2 elevation values to combine into 360 azimuth")

        if elevation_lo is None or elevation_hi is None:
            finite = el_axis[np.isfinite(el_axis)]
            if finite.size < 2:
                raise ValueError("elevation axis has fewer than 2 finite values")
            lo_value = float(np.min(finite))
            hi_value = float(np.max(finite))
        else:
            lo_value = float(elevation_lo)
            hi_value = float(elevation_hi)

        if not np.isfinite(lo_value) or not np.isfinite(hi_value):
            raise ValueError("elevation pair values must be finite")
        try:
            tolerance_deg = float(tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("combine tolerance must be finite and nonnegative") from exc
        if not np.isfinite(tolerance_deg) or tolerance_deg < 0.0:
            raise ValueError("combine tolerance must be finite and nonnegative")
        elevation_unit = self._supported_unit(
            "elevation", _ANGLE_UNITS, "deg"
        )
        azimuth_unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        elevation_tol = (
            float(np.deg2rad(tolerance_deg))
            if elevation_unit == "rad"
            else tolerance_deg
        )
        azimuth_tol = (
            float(np.deg2rad(tolerance_deg))
            if azimuth_unit == "rad"
            else tolerance_deg
        )
        if np.isclose(lo_value, hi_value, atol=elevation_tol, rtol=0.0):
            raise ValueError("elevation pair values must be distinct")
        lo_deg = float(np.rad2deg(lo_value)) if elevation_unit == "rad" else lo_value
        hi_deg = float(np.rad2deg(hi_value)) if elevation_unit == "rad" else hi_value
        if not np.isclose(
            lo_deg,
            -hi_deg,
            atol=max(tolerance_deg, 1.0e-9),
            rtol=0.0,
        ):
            raise ValueError(
                "El->Az360 requires equal-and-opposite elevation cuts about "
                f"zero; got {lo_deg:.12g} and {hi_deg:.12g} deg"
            )
        if not np.isclose(
            float(azimuth_shift_deg), 180.0, atol=max(tolerance_deg, 1.0e-9), rtol=0.0
        ):
            raise ValueError(
                "El->Az360 requires a 180 degree second-half azimuth shift"
            )
        declared_contract = self._declared_scalar_metadata(
            "elevation_pair_azimuth_contract"
        )
        if not declared_contract and not bool(assumptions_attested):
            raise ValueError(
                "El->Az360 is an acquisition-specific relabel. Confirm the "
                "opposite-elevation/180-degree acquisition assumption before "
                "creating the result."
            )

        lo_matches = self._axis_value_match(
            self.elevations, lo_value, tol=elevation_tol
        )
        hi_matches = self._axis_value_match(
            self.elevations, hi_value, tol=elevation_tol
        )
        if lo_matches.size == 0 or hi_matches.size == 0:
            raise ValueError("requested elevation pair not found in dataset")

        lo_idx = int(lo_matches[0])
        hi_idx = int(hi_matches[0])
        az_shift = self._angle_value_from_degrees(
            azimuth_shift_deg, "azimuth"
        )

        az_base = np.asarray(self.azimuths, dtype=float)
        if az_base.size == 0:
            raise ValueError("dataset has no azimuth samples")

        az_lo = np.array(az_base, copy=True)
        az_hi = np.array(az_base, copy=True) + az_shift
        az_merged = self._axis_union([az_lo, az_hi], tol=azimuth_tol)
        if az_merged.size == 0:
            raise ValueError("combined azimuth axis is empty")

        out_shape = (len(az_merged), 1, len(self.frequencies), len(self.polarizations))
        out_power = np.full(out_shape, np.nan, dtype=self.rcs_power.dtype)
        out_phase = np.full(out_shape, np.nan, dtype=self.rcs_phase.dtype)
        raw_pair = self._complete_authoritative_raw_arrays()
        preserve_raw = raw_pair is not None
        if preserve_raw:
            raw_real = np.asarray(raw_pair[0], dtype=np.float64)
            raw_imag = np.asarray(raw_pair[1], dtype=np.float64)
            out_raw_real = np.full(out_shape, np.nan, dtype=np.float64)
            out_raw_imag = np.full(out_shape, np.nan, dtype=np.float64)

        lo_target_idx = self._indices_for_axis_values(
            az_merged, az_lo, tol=azimuth_tol
        )
        hi_target_idx = self._indices_for_axis_values(
            az_merged, az_hi, tol=azimuth_tol
        )
        if lo_target_idx is None or hi_target_idx is None:
            raise ValueError("failed to align azimuth bins during elevation combine")
        if (
            len(lo_target_idx) != az_lo.size
            or len(hi_target_idx) != az_hi.size
        ):
            raise ValueError(
                "cannot combine elevation cuts: the input azimuth axis contains "
                "coordinates closer than the matching tolerance "
                f"({tolerance_deg:g} deg); "
                "deduplicate the azimuth axis or use a smaller tolerance"
            )

        lo_power = self.rcs_power[:, lo_idx, :, :]
        lo_phase = self.rcs_phase[:, lo_idx, :, :]
        hi_power = self.rcs_power[:, hi_idx, :, :]
        hi_phase = self.rcs_phase[:, hi_idx, :, :]
        if preserve_raw:
            lo_raw_real = raw_real[:, lo_idx, :, :]
            lo_raw_imag = raw_imag[:, lo_idx, :, :]
            hi_raw_real = raw_real[:, hi_idx, :, :]
            hi_raw_imag = raw_imag[:, hi_idx, :, :]

        for label, target_indices, source_power, source_phase, source_real, source_imag in (
            (
                "lower elevation",
                lo_target_idx,
                lo_power,
                lo_phase,
                lo_raw_real if preserve_raw else None,
                lo_raw_imag if preserve_raw else None,
            ),
            (
                "shifted higher elevation",
                hi_target_idx,
                hi_power,
                hi_phase,
                hi_raw_real if preserve_raw else None,
                hi_raw_imag if preserve_raw else None,
            ),
        ):
            for src_idx, dst_idx in enumerate(target_indices):
                context = (
                    f"El->Az360 {label} at azimuth "
                    f"{az_merged[dst_idx]:.12g} {azimuth_unit}"
                )
                self._merge_equivalent_sample_blocks(
                    out_power[dst_idx, 0, :, :],
                    out_phase[dst_idx, 0, :, :],
                    source_power[src_idx, :, :],
                    source_phase[src_idx, :, :],
                    context=context,
                )
                if preserve_raw:
                    self._merge_equivalent_raw_blocks(
                        out_raw_real[dst_idx, 0, :, :],
                        out_raw_imag[dst_idx, 0, :, :],
                        source_real[src_idx, :, :],
                        source_imag[src_idx, :, :],
                        context=context,
                    )

        if preserve_raw:
            unmodeled = ~np.isfinite(out_power)
            out_raw_real[unmodeled] = np.nan
            out_raw_imag[unmodeled] = np.nan

        combined_extra = self._exact_transform_extra(
            coordinate_change="combine-elevation-pair-to-azimuth-360",
            preserve_angular_contract=False,
            preserve_raw=False,
        )
        if preserve_raw:
            combined_extra["rcs_amp_real"] = out_raw_real
            combined_extra["rcs_amp_imag"] = out_raw_imag
            combined_extra["raw_complex_amplitude_preserved"] = True
        combined_extra["elevation_pair_to_azimuth_json"] = json.dumps(
            {
                "schema": "grim.elevation-pair-to-azimuth.v1",
                "operation": "acquisition_specific_relabel",
                "elevation_pair_deg": [lo_deg, hi_deg],
                "azimuth_shift_deg": float(azimuth_shift_deg),
                "declared_contract": declared_contract or None,
                "user_assumptions_attested": bool(assumptions_attested),
                "interpolation": False,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return self._new_grid(
            az_merged,
            np.asarray([el_axis[lo_idx]], dtype=float),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=out_power,
            rcs_phase=out_phase,
            rcs_domain="power_phase",
            extra=combined_extra,
        )

    @classmethod
    def stitch_many(
        cls,
        *grids,
        policy="priority-first",
        tol=1e-6,
        metadata_attested=False,
        max_output_bytes=None,
        return_report=False,
    ):
        """Stitch union-grid samples using one explicit overlap policy.

        Policies are ``"priority-first"``, ``"priority-last"``,
        ``"power-mean"``, and ``"coherent-mean"``.  Priority policies choose
        an entire power/phase sample atomically in input order. ``power-mean``
        averages finite linear power and makes phase unknown only at cells
        with multiple contributors. ``coherent-mean`` averages complex fields
        and therefore requires finite phase plus compatible coherent metadata.

        This is intentionally separate from :meth:`join_many`: strict Join
        continues to reject conflicting finite overlaps.  Stitch resolves
        them only under the caller-selected policy and records a durable
        report.  Processing uses bounded source/target blocks and
        ``max_output_bytes`` caps the estimated peak retained/working memory.

        When ``return_report`` is true, return ``(grid, report)``; otherwise
        return only the stitched grid.  Report counts are union-grid cell
        counts except ``contributing_count``, which counts all finite input
        contributions.
        """

        policies = {
            "priority-first",
            "priority-last",
            "power-mean",
            "coherent-mean",
        }
        policy = str(policy).strip().lower().replace("_", "-")
        if policy not in policies:
            raise ValueError(
                "policy must be 'priority-first', 'priority-last', "
                "'power-mean', or 'coherent-mean'"
            )
        if not isinstance(metadata_attested, (bool, np.bool_)):
            raise TypeError("metadata_attested must be True or False")
        if not isinstance(return_report, (bool, np.bool_)):
            raise TypeError("return_report must be True or False")
        try:
            tolerance = float(tol)
        except (TypeError, ValueError) as exc:
            raise TypeError("tol must be a finite nonnegative number") from exc
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tol must be a finite nonnegative number")
        if max_output_bytes is not None:
            try:
                memory_limit = int(max_output_bytes)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("max_output_bytes must be a nonnegative integer") from exc
            if memory_limit < 0:
                raise ValueError("max_output_bytes must be nonnegative")
        else:
            memory_limit = None

        grids = cls._ensure_grids(grids)
        if len(grids) > np.iinfo(np.uint32).max:
            raise ValueError("too many input grids for stitch contributor counts")
        ref = grids[0]

        convention_fields = (
            (
                "phase_reference",
                "phase references",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "time_convention",
                "time conventions",
                ref._canonical_time_convention,
            ),
            (
                "polarization_basis",
                "polarization bases",
                lambda value: " ".join(value.split()).casefold(),
            ),
        )

        metadata_assumptions = {}
        preserved_conventions = {}
        for key, label, canonicalize in convention_fields:
            values = [grid._declared_scalar_metadata(key) for grid in grids]
            values = [
                "" if grid._metadata_placeholder(value) else value
                for grid, value in zip(grids, values)
            ]
            nonblank = [value for value in values if value]
            normalized = {canonicalize(value) for value in nonblank}
            if len(normalized) > 1:
                rendered = ", ".join(
                    f"input {index}={value or '<unspecified>'!r}"
                    for index, value in enumerate(values, start=1)
                )
                metadata_assumptions[f"{key} (conflicting annotations; samples unchanged)"] = rendered
            if nonblank and len(nonblank) == len(grids) and len(normalized) == 1:
                preserved_conventions[key] = nonblank[0]
            elif len(nonblank) != len(grids):
                metadata_assumptions[key] = [
                    index
                    for index, value in enumerate(values, start=1)
                    if not value
                ]

        def scalar_convention_extra():
            return dict(preserved_conventions)

        # Stitch creates one physical dataset. Missing declarations remain
        # unspecified on that result; explicit physical conflicts still stop.
        for grid in grids[1:]:
            ref._assert_physical_metadata_compatible(grid)

        if policy == "coherent-mean":
            for grid in grids[1:]:
                ref._assert_coherent_metadata_compatible(
                    grid, metadata_attested=bool(metadata_attested)
                )

        expected_shapes = []
        coherent_missing_phase_count = 0
        for input_index, grid in enumerate(grids, start=1):
            input_phase_wrap = str(
                (grid.units or {}).get("phase_wrap", "")
            ).strip()
            if input_phase_wrap not in {"", "0_360", "-180_180"}:
                raise ValueError(
                    f"stitch input {input_index} has unsupported phase_wrap "
                    f"{input_phase_wrap!r}"
                )
            numeric_axes = (
                ("azimuth", np.asarray(grid.azimuths)),
                ("elevation", np.asarray(grid.elevations)),
                ("frequency", np.asarray(grid.frequencies)),
            )
            for axis_name, axis in numeric_axes:
                if (
                    axis.ndim != 1
                    or axis.size == 0
                    or axis.dtype.kind not in "iuf"
                    or np.any(~np.isfinite(axis))
                    or np.unique(axis).size != axis.size
                ):
                    raise ValueError(
                        f"stitch input {input_index} has an invalid {axis_name} axis"
                    )
                if axis_name == "frequency" and np.any(axis <= 0.0):
                    raise ValueError(
                        f"stitch input {input_index} has nonpositive frequencies"
                    )
            polarizations = np.asarray(grid.polarizations)
            labels = [str(value).strip() for value in polarizations.tolist()]
            if (
                polarizations.ndim != 1
                or polarizations.size == 0
                or any(not label for label in labels)
                or len({label.casefold() for label in labels}) != len(labels)
            ):
                raise ValueError(
                    f"stitch input {input_index} has an invalid polarization axis"
                )
            expected = tuple(len(axis) for _name, axis in numeric_axes) + (
                len(polarizations),
            )
            expected_shapes.append(expected)
            power = np.asarray(grid.rcs_power)
            phase = np.asarray(grid.rcs_phase)
            if power.shape != expected or phase.shape != expected:
                raise ValueError(
                    f"stitch input {input_index} sample shape does not match its axes"
                )
            if power.dtype.kind not in "iuf" or phase.dtype.kind not in "iuf":
                raise ValueError(
                    f"stitch input {input_index} power and phase must be real numeric"
                )

            infinite_power_count = 0
            negative_power_count = 0
            minimum_negative = None
            infinite_phase_count = 0
            missing_coherent_phase_count = 0
            iterator = np.nditer(
                (power, phase),
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"]],
                order="K",
                buffersize=_JOIN_MERGE_BLOCK_CELLS,
            )
            for power_block, phase_block in iterator:
                power_block = np.asarray(power_block)
                phase_block = np.asarray(phase_block)
                infinite_power_count += int(np.count_nonzero(np.isinf(power_block)))
                finite_negative = np.isfinite(power_block) & (power_block < 0.0)
                negative_power_count += int(np.count_nonzero(finite_negative))
                if np.any(finite_negative):
                    block_minimum = float(np.min(power_block[finite_negative]))
                    minimum_negative = (
                        block_minimum
                        if minimum_negative is None
                        else min(minimum_negative, block_minimum)
                    )
                infinite_phase_count += int(np.count_nonzero(np.isinf(phase_block)))
                if policy == "coherent-mean":
                    missing_coherent_phase_count += int(
                        np.count_nonzero(
                            np.isfinite(power_block) & ~np.isfinite(phase_block)
                        )
                    )
            if infinite_power_count or infinite_phase_count:
                raise ValueError(
                    f"stitch input {input_index} contains infinite samples "
                    f"(power={infinite_power_count}, phase={infinite_phase_count})"
                )
            if negative_power_count:
                raise ValueError(
                    f"stitch input {input_index} contains {negative_power_count} "
                    "negative power sample(s); minimum is "
                    f"{minimum_negative:.17g}"
                )
            coherent_missing_phase_count += missing_coherent_phase_count

        if len(grids) == 1:
            az_union = np.array(ref.azimuths, copy=True)
            el_union = np.array(ref.elevations, copy=True)
            f_union = np.array(ref.frequencies, copy=True)
            p_union = np.array(ref.polarizations, copy=True)
        else:
            az_union = cls._axis_union(
                [grid.azimuths for grid in grids], tol=tolerance
            )
            el_union = cls._axis_union(
                [grid.elevations for grid in grids], tol=tolerance
            )
            f_union = cls._axis_union(
                [grid.frequencies for grid in grids], tol=tolerance
            )
            p_union = cls._axis_union(
                [grid.polarizations for grid in grids], tol=0.0
            )

        shape = (len(az_union), len(el_union), len(f_union), len(p_union))
        cell_count = 1
        for dimension in shape:
            cell_count *= int(dimension)
        if policy in {"power-mean", "coherent-mean"}:
            output_dtype = np.result_type(
                *[grid.rcs_power.dtype for grid in grids], np.float64
            )
        else:
            output_dtype = np.result_type(
                *[grid.rcs_power.dtype for grid in grids]
            )
        itemsize = np.dtype(output_dtype).itemsize
        output_bytes = cell_count * itemsize * 2
        state_bytes = cell_count * (
            np.dtype(np.uint32).itemsize + np.dtype(np.bool_).itemsize
        )
        merge_block_cells = min(cell_count, _JOIN_MERGE_BLOCK_CELLS)
        merge_scratch_bytes = merge_block_cells * (12 * itemsize + 64)
        estimated_peak_bytes = output_bytes + state_bytes + merge_scratch_bytes
        if memory_limit is not None and estimated_peak_bytes > memory_limit:
            raise MemoryError(
                "dense stitched grid needs about "
                f"{estimated_peak_bytes / (1024**3):.2f} GiB peak "
                f"({(output_bytes + state_bytes) / (1024**3):.2f} GiB retained "
                "during construction), above the configured limit of "
                f"{memory_limit / (1024**3):.2f} GiB"
            )

        stitched_power = np.full(shape, np.nan, dtype=output_dtype)
        stitched_phase = np.full(shape, np.nan, dtype=output_dtype)
        contributor_counts = np.zeros(shape, dtype=np.uint32)
        conflict_flags = np.zeros(shape, dtype=bool)

        mapped_indices = []
        for grid in grids:
            indices = (
                cls._indices_for_axis_values(
                    az_union, grid.azimuths, tol=tolerance
                ),
                cls._indices_for_axis_values(
                    el_union, grid.elevations, tol=tolerance
                ),
                cls._indices_for_axis_values(
                    f_union, grid.frequencies, tol=tolerance
                ),
                cls._indices_for_axis_values(
                    p_union, grid.polarizations, tol=0.0
                ),
            )
            if any(value is None for value in indices):
                raise ValueError("failed to align a dataset during stitch")
            for axis_name, axis_indices, source_axis, axis_tol in (
                ("azimuth", indices[0], grid.azimuths, tolerance),
                ("elevation", indices[1], grid.elevations, tolerance),
                ("frequency", indices[2], grid.frequencies, tolerance),
                ("polarization", indices[3], grid.polarizations, 0.0),
            ):
                if len(axis_indices) != np.asarray(source_axis).size:
                    raise ValueError(
                        f"cannot stitch: an input {axis_name} axis contains "
                        "coordinates that collapse within the matching "
                        f"tolerance ({axis_tol:g}); deduplicate that axis or "
                        "use a smaller tolerance"
                    )
            mapped_indices.append(indices)

        for grid, indices in zip(grids, mapped_indices):
            az_idx, el_idx, f_idx, p_idx = indices
            incoming_power = np.asarray(grid.rcs_power)
            incoming_phase = np.asarray(grid.rcs_phase)
            pol_block = max(1, min(len(p_idx), _JOIN_MERGE_BLOCK_CELLS))
            freq_block = max(
                1, min(len(f_idx), _JOIN_MERGE_BLOCK_CELLS // pol_block)
            )
            remaining = max(
                1, _JOIN_MERGE_BLOCK_CELLS // (pol_block * freq_block)
            )
            elev_block = max(1, min(len(el_idx), remaining))
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS
                // (pol_block * freq_block * elev_block),
            )
            az_block = max(1, min(len(az_idx), remaining))
            for a_start in range(0, len(az_idx), az_block):
                a_stop = min(a_start + az_block, len(az_idx))
                union_a = az_idx[a_start:a_stop]
                for e_start in range(0, len(el_idx), elev_block):
                    e_stop = min(e_start + elev_block, len(el_idx))
                    union_e = el_idx[e_start:e_stop]
                    for f_start in range(0, len(f_idx), freq_block):
                        f_stop = min(f_start + freq_block, len(f_idx))
                        union_f = f_idx[f_start:f_stop]
                        for p_start in range(0, len(p_idx), pol_block):
                            p_stop = min(p_start + pol_block, len(p_idx))
                            union_p = p_idx[p_start:p_stop]
                            target = np.ix_(union_a, union_e, union_f, union_p)
                            source = (
                                slice(a_start, a_stop),
                                slice(e_start, e_stop),
                                slice(f_start, f_stop),
                                slice(p_start, p_stop),
                            )
                            block_power = incoming_power[source]
                            block_phase = incoming_phase[source]
                            valid = np.isfinite(block_power)
                            if policy == "coherent-mean":
                                valid &= np.isfinite(block_phase)
                            if not np.any(valid):
                                continue

                            existing_power = stitched_power[target]
                            existing_phase = stitched_phase[target]
                            count_block = contributor_counts[target]
                            conflict_block = conflict_flags[target]
                            overlap = valid & (count_block > 0)

                            incoming_field = None
                            if policy == "coherent-mean":
                                incoming_field = np.asarray(grid.rcs_slice(source))
                                invalid_field = valid & (
                                    ~np.isfinite(incoming_field.real)
                                    | ~np.isfinite(incoming_field.imag)
                                )
                                if np.any(invalid_field):
                                    valid &= ~invalid_field
                                    if not np.any(valid):
                                        continue

                            if np.any(overlap):
                                if policy == "power-mean":
                                    reference_power = np.divide(
                                        existing_power,
                                        count_block,
                                        out=np.full_like(existing_power, np.nan),
                                        where=count_block > 0,
                                    )
                                    power_equal = overlap & np.isclose(
                                        reference_power,
                                        block_power,
                                        rtol=1.0e-6,
                                        atol=1.0e-12,
                                    )
                                    conflict_block |= overlap & ~power_equal
                                elif policy == "coherent-mean":
                                    reference_field = np.divide(
                                        existing_power + 1j * existing_phase,
                                        count_block,
                                        out=np.full(
                                            existing_power.shape,
                                            np.nan + 1j * np.nan,
                                            dtype=np.complex128,
                                        ),
                                        where=count_block > 0,
                                    )
                                    reference_power = np.abs(reference_field) ** 2
                                    incoming_field_power = np.abs(incoming_field) ** 2
                                    power_equal = overlap & np.isclose(
                                        reference_power,
                                        incoming_field_power,
                                        rtol=1.0e-6,
                                        atol=1.0e-12,
                                    )
                                    both_zero = (
                                        power_equal
                                        & (reference_power == 0.0)
                                        & (incoming_field_power == 0.0)
                                    )
                                    phase_delta = np.abs(
                                        np.angle(reference_field / incoming_field)
                                    )
                                    phase_conflict = (
                                        overlap
                                        & power_equal
                                        & ~both_zero
                                        & (phase_delta > 1.0e-5)
                                    )
                                    conflict_block |= (
                                        (overlap & ~power_equal) | phase_conflict
                                    )
                                else:
                                    power_equal = overlap & np.isclose(
                                        existing_power,
                                        block_power,
                                        rtol=1.0e-6,
                                        atol=1.0e-12,
                                    )
                                    both_phase = (
                                        power_equal
                                        & np.isfinite(existing_phase)
                                        & np.isfinite(block_phase)
                                    )
                                    both_zero = (
                                        power_equal
                                        & (existing_power == 0.0)
                                        & (block_power == 0.0)
                                    )
                                    phase_delta = np.abs(
                                        np.angle(
                                            np.exp(1j * (existing_phase - block_phase))
                                        )
                                    )
                                    phase_conflict = (
                                        both_phase
                                        & ~both_zero
                                        & (phase_delta > 1.0e-5)
                                    )
                                    conflict_block |= (
                                        (overlap & ~power_equal) | phase_conflict
                                    )

                            first_contribution = valid & (count_block == 0)
                            repeated_contribution = valid & (count_block > 0)
                            if policy == "priority-first":
                                existing_power[first_contribution] = block_power[
                                    first_contribution
                                ]
                                existing_phase[first_contribution] = block_phase[
                                    first_contribution
                                ]
                            elif policy == "priority-last":
                                existing_power[valid] = block_power[valid]
                                existing_phase[valid] = block_phase[valid]
                            elif policy == "power-mean":
                                existing_power[first_contribution] = block_power[
                                    first_contribution
                                ]
                                existing_phase[first_contribution] = block_phase[
                                    first_contribution
                                ]
                                existing_power[repeated_contribution] += block_power[
                                    repeated_contribution
                                ]
                                existing_phase[repeated_contribution] = np.nan
                            else:
                                field_real = incoming_field.real
                                field_imag = incoming_field.imag
                                existing_power[first_contribution] = field_real[
                                    first_contribution
                                ]
                                existing_phase[first_contribution] = field_imag[
                                    first_contribution
                                ]
                                existing_power[repeated_contribution] += field_real[
                                    repeated_contribution
                                ]
                                existing_phase[repeated_contribution] += field_imag[
                                    repeated_contribution
                                ]

                            count_block[valid] += np.uint32(1)
                            stitched_power[target] = existing_power
                            stitched_phase[target] = existing_phase
                            contributor_counts[target] = count_block
                            conflict_flags[target] = conflict_block

        flat_power = stitched_power.reshape(-1)
        flat_phase = stitched_phase.reshape(-1)
        flat_counts = contributor_counts.reshape(-1)
        if policy in {"power-mean", "coherent-mean"}:
            for start in range(0, cell_count, _JOIN_MERGE_BLOCK_CELLS):
                stop = min(start + _JOIN_MERGE_BLOCK_CELLS, cell_count)
                counts_block = flat_counts[start:stop]
                valid = counts_block > 0
                if policy == "power-mean":
                    power_block = flat_power[start:stop]
                    power_block[valid] /= counts_block[valid]
                    if np.any(~np.isfinite(power_block[valid])):
                        raise ValueError(
                            "power-mean stitch produced nonfinite output power"
                        )
                else:
                    real_block = flat_power[start:stop]
                    imag_block = flat_phase[start:stop]
                    mean_real = real_block[valid] / counts_block[valid]
                    mean_imag = imag_block[valid] / counts_block[valid]
                    result_power = mean_real * mean_real + mean_imag * mean_imag
                    if np.any(~np.isfinite(result_power)):
                        raise ValueError(
                            "coherent-mean stitch produced nonfinite output power"
                        )
                    real_block[valid] = result_power
                    result_phase = np.arctan2(mean_imag, mean_real)
                    result_phase[result_power == 0.0] = np.nan
                    imag_block[valid] = result_phase
                flat_power[start:stop][~valid] = np.nan
                flat_phase[start:stop][~valid] = np.nan

        output_phase_wrap = str(
            (ref.units or {}).get("phase_wrap", "")
        ).strip() or "-180_180"
        for start in range(0, cell_count, _JOIN_MERGE_BLOCK_CELLS):
            stop = min(start + _JOIN_MERGE_BLOCK_CELLS, cell_count)
            phase_block = flat_phase[start:stop]
            finite_phase = np.isfinite(phase_block)
            if output_phase_wrap == "0_360":
                phase_block[finite_phase] = np.mod(
                    phase_block[finite_phase], 2.0 * np.pi
                )
            else:
                phase_block[finite_phase] = (
                    np.mod(phase_block[finite_phase] + np.pi, 2.0 * np.pi)
                    - np.pi
                )

        contributing_count = 0
        output_finite_count = 0
        overlap_count = 0
        conflict_count = 0
        max_contributors = 0
        flat_conflicts = conflict_flags.reshape(-1)
        for start in range(0, cell_count, _JOIN_MERGE_BLOCK_CELLS):
            stop = min(start + _JOIN_MERGE_BLOCK_CELLS, cell_count)
            counts_block = flat_counts[start:stop]
            conflicts_block = flat_conflicts[start:stop]
            contributing_count += int(np.sum(counts_block, dtype=np.uint64))
            output_finite_count += int(np.count_nonzero(counts_block > 0))
            overlap_count += int(np.count_nonzero(counts_block > 1))
            conflict_count += int(
                np.count_nonzero((counts_block > 1) & conflicts_block)
            )
            if counts_block.size:
                max_contributors = max(
                    max_contributors, int(np.max(counts_block))
                )
        equal_count = int(overlap_count - conflict_count)
        report = {
            "schema": "grim.stitch-report.v1",
            "policy": policy,
            "selected_policy": policy,
            "input_count": int(len(grids)),
            "output_cell_count": int(cell_count),
            "output_finite_count": int(output_finite_count),
            "missing_count": int(cell_count - output_finite_count),
            "single_source_count": int(output_finite_count - overlap_count),
            "contributing_count": int(contributing_count),
            "overlap_count": int(overlap_count),
            "equal_count": int(equal_count),
            "conflict_count": int(conflict_count),
            "max_contributors": int(max_contributors),
            "masked_missing_phase_sample_count": int(
                coherent_missing_phase_count
            ),
            "metadata_assumptions": metadata_assumptions,
            "tolerance": float(tolerance),
            "estimated_peak_bytes": int(estimated_peak_bytes),
            "count_semantics": (
                "union-grid cells; contributing_count is finite input samples"
            ),
        }

        if policy == "coherent-mean" and output_finite_count == 0:
            raise ValueError(
                "coherent-mean stitch has no usable complex samples"
            )

        if policy == "coherent-mean":
            attested_history, attested_extra = ref._coherent_attestation_provenance(
                grids[1:],
                operation="coherent-mean-stitch",
                metadata_attested=bool(metadata_attested),
            )
            base_history = (
                attested_history
                if attested_history is not None
                else str(ref.history or "").strip()
            )
            output_extra = (
                dict(attested_extra)
                if attested_extra is not None
                else scalar_convention_extra()
            )
        else:
            base_history = str(ref.history or "").strip()
            output_extra = scalar_convention_extra()
        if metadata_assumptions:
            output_extra["merge_metadata_assumption_json"] = json.dumps(
                {
                    "schema": "grim.merge-metadata-assumption.v1",
                    "operation": "overlap-merge",
                    "policy": policy,
                    "input_count": len(grids),
                    "unspecified_declarations_by_input": metadata_assumptions,
                    "declarations_inferred": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        history_entry = (
            f"Stitch ({policy}, inputs={len(grids)}): "
            f"overlap={overlap_count}, equal={equal_count}, "
            f"conflict={conflict_count}, contributors={contributing_count}"
        )
        history = (
            f"{base_history}\n{history_entry}" if base_history else history_entry
        )
        if metadata_assumptions:
            assumption_entry = (
                "Overlap merge retained unspecified metadata: "
                + ", ".join(sorted(metadata_assumptions))
            )
            history = f"{history}\n{assumption_entry}"
        provenance = dict(report)
        provenance.update(
            {
                "schema": "grim.stitch-provenance.v1",
                "metadata_attested": bool(metadata_attested),
                "input_sources": [str(grid.source_path or "") for grid in grids],
            }
        )
        output_extra["stitch_provenance_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        output_units = dict(ref.units)
        output_units["phase_wrap"] = output_phase_wrap
        stitched = cls(
            az_union,
            el_union,
            f_union,
            p_union,
            rcs_power=stitched_power,
            rcs_phase=stitched_phase,
            rcs_domain="power_phase",
            source_path=ref.source_path,
            history=history,
            units=output_units,
            extra=output_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )
        if bool(return_report):
            return stitched, report
        return stitched

    @classmethod
    def join_many(cls, *grids, tol=1e-6, overlap="error", max_output_bytes=None):
        """Join datasets on union axes without silently replacing finite data.

        ``overlap`` may be ``"error"`` (default), ``"first"``, or ``"last"``.
        Equal finite samples are accepted in all modes. ``max_output_bytes`` can
        cap the estimated peak allocation for memory-aware folder workflows.
        """
        grids = cls._ensure_grids(grids)
        if overlap not in {"error", "first", "last"}:
            raise ValueError("overlap must be 'error', 'first', or 'last'")
        ref = grids[0]

        def canonical_json_scalar(value):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return value.strip()
            return json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )

        scalar_metadata_fields = (
            (
                "phase_reference",
                "phase references",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "time_convention",
                "time conventions",
                ref._canonical_time_convention,
            ),
            (
                "polarization_basis",
                "polarization bases",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "amplitude_convention",
                "amplitude conventions",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "complex_field_domain",
                "complex-field domains",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "combine_role",
                "combination roles",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "assembly_response_role",
                "Assembly response roles",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "assembly_base_sha256",
                "Assembly base identities",
                lambda value: value.strip().casefold(),
            ),
            (
                "assembly_base_response_sha256",
                "Assembly base response identities",
                lambda value: value.strip().casefold(),
            ),
            (
                "assembly_angular_coordinate_contract",
                "Assembly angular-coordinate contracts",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "elevation_coordinate_convention",
                "elevation-coordinate conventions",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "sentri_elevation_convention",
                "SENTRi elevation conventions",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "sentri_coordinate_mapping",
                "SENTRi coordinate mappings",
                lambda value: " ".join(value.split()).casefold(),
            ),
            (
                "feature_provenance_json",
                "feature provenance records",
                canonical_json_scalar,
            ),
        )
        for grid in grids[1:]:
            ref._assert_physical_metadata_compatible(grid)

        metadata_assumptions = {}
        for key, label, canonicalize in scalar_metadata_fields:
            declared = [grid._declared_scalar_metadata(key) for grid in grids]
            declared = [
                "" if grid._metadata_placeholder(value) else value
                for grid, value in zip(grids, declared)
            ]
            nonblank = [value for value in declared if value]
            normalized = {canonicalize(value) for value in nonblank}
            if len(normalized) > 1:
                rendered = ", ".join(
                    f"input {index}={value or '<unspecified>'!r}"
                    for index, value in enumerate(declared, start=1)
                )
                if key in {"phase_reference", "time_convention", "polarization_basis", "amplitude_convention", "complex_field_domain", "feature_provenance_json", "amplitude_version"}:
                    metadata_assumptions[f"{key} (conflicting annotations; samples unchanged)"] = rendered
                else:
                    raise ValueError(
                        f"cannot join grids with different explicit {label}: {rendered}"
                    )
            if nonblank and len(nonblank) != len(grids):
                metadata_assumptions[key] = {
                    "declared_input_indices": [
                        index
                        for index, value in enumerate(declared, start=1)
                        if value
                    ],
                    "unspecified_input_indices": [
                        index
                        for index, value in enumerate(declared, start=1)
                        if not value
                    ],
                }

        preserved_scalar_extra = {}
        for key, _label, canonicalize in scalar_metadata_fields:
            declared = [grid._declared_scalar_metadata(key) for grid in grids]
            declared = [
                "" if grid._metadata_placeholder(value) else value
                for grid, value in zip(grids, declared)
            ]
            nonblank = [value for value in declared if value]
            if not nonblank:
                continue
            if any(not value for value in declared):
                # A one-sided declaration cannot safely be promoted to a
                # statement about the complete joined artifact.
                continue
            normalized = {canonicalize(value) for value in nonblank}
            if len(normalized) == 1:
                preserved_scalar_extra[key] = nonblank[0]

        raw_inputs = []
        preserve_raw_amplitude = True
        for grid in grids:
            raw_real = grid.extra.get("rcs_amp_real")
            raw_imag = grid.extra.get("rcs_amp_imag")
            if raw_real is None or raw_imag is None:
                preserve_raw_amplitude = False
                break
            try:
                raw_real = np.asarray(raw_real, dtype=np.float64)
                raw_imag = np.asarray(raw_imag, dtype=np.float64)
            except (TypeError, ValueError):
                preserve_raw_amplitude = False
                break
            if (
                raw_real.shape != grid.rcs_power.shape
                or raw_imag.shape != grid.rcs_power.shape
            ):
                preserve_raw_amplitude = False
                break
            # A finite stored power cell is a modeled sample.  If its raw
            # field is incomplete, retaining the raw arrays would make
            # ``rcs`` authoritative but turn that valid sample into NaN.
            modeled = np.isfinite(grid.rcs_power)
            raw_finite = np.isfinite(raw_real) & np.isfinite(raw_imag)
            if np.any(modeled & ~raw_finite):
                preserve_raw_amplitude = False
                break
            raw_inputs.append((raw_real, raw_imag))
        if not preserve_raw_amplitude:
            raw_inputs = []
        if len(grids) == 1:
            # Preserve clone semantics, including original axis order, while
            # still using the bounded allocation/ownership-transfer path.
            az_union = np.array(ref.azimuths, copy=True)
            el_union = np.array(ref.elevations, copy=True)
            f_union = np.array(ref.frequencies, copy=True)
            p_union = np.array(ref.polarizations, copy=True)
        else:
            az_union = cls._axis_union([grid.azimuths for grid in grids], tol=tol)
            el_union = cls._axis_union([grid.elevations for grid in grids], tol=tol)
            f_union = cls._axis_union([grid.frequencies for grid in grids], tol=tol)
            p_union = cls._axis_union([grid.polarizations for grid in grids], tol=0.0)

        shape = (len(az_union), len(el_union), len(f_union), len(p_union))
        out_dtype = np.result_type(*[g.rcs_power.dtype for g in grids])
        cell_count = 1
        for dimension in shape:
            cell_count *= int(dimension)
        itemsize = np.dtype(out_dtype).itemsize
        raw_output_bytes = (
            cell_count * 2 * np.dtype(np.float64).itemsize
            if preserve_raw_amplitude else 0
        )
        output_bytes = cell_count * itemsize * 2 + raw_output_bytes
        # The bounded merge below avoids full input-sized advanced-index
        # copies. Count the two retained arrays, one union-sized sanitation
        # mask, and a conservative allowance for a bounded merge block.
        merge_block_cells = min(cell_count, _JOIN_MERGE_BLOCK_CELLS)
        merge_scratch_bytes = merge_block_cells * (
            8 * itemsize + (48 if preserve_raw_amplitude else 32)
        )
        estimated_peak_bytes = output_bytes + cell_count + merge_scratch_bytes
        if max_output_bytes is not None and estimated_peak_bytes > int(max_output_bytes):
            raise MemoryError(
                f"dense joined grid needs about {estimated_peak_bytes / (1024**3):.2f} GiB peak "
                f"({output_bytes / (1024**3):.2f} GiB retained), "
                f"above the configured limit of {int(max_output_bytes) / (1024**3):.2f} GiB"
            )
        joined_power = np.full(shape, np.nan, dtype=out_dtype)
        joined_phase = np.full(shape, np.nan, dtype=out_dtype)
        joined_raw_real = (
            np.full(shape, np.nan, dtype=np.float64)
            if preserve_raw_amplitude else None
        )
        joined_raw_imag = (
            np.full(shape, np.nan, dtype=np.float64)
            if preserve_raw_amplitude else None
        )

        for grid_index, grid in enumerate(grids):
            az_idx = cls._indices_for_axis_values(az_union, grid.azimuths, tol=tol)
            el_idx = cls._indices_for_axis_values(el_union, grid.elevations, tol=tol)
            f_idx = cls._indices_for_axis_values(f_union, grid.frequencies, tol=tol)
            p_idx = cls._indices_for_axis_values(p_union, grid.polarizations, tol=0.0)
            if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
                raise ValueError("failed to align a dataset during join")
            for axis_name, indices, source_axis, axis_tol in (
                ("azimuth", az_idx, grid.azimuths, tol),
                ("elevation", el_idx, grid.elevations, tol),
                ("frequency", f_idx, grid.frequencies, tol),
                ("polarization", p_idx, grid.polarizations, 0.0),
            ):
                if len(indices) != np.asarray(source_axis).size:
                    raise ValueError(
                        f"cannot join: an input {axis_name} axis contains "
                        "coordinates that collapse within the matching "
                        f"tolerance ({axis_tol:g}); deduplicate that axis or "
                        "use a smaller tolerance"
                    )
            # Keep source dtypes as views. NumPy promotes only the bounded block
            # expressions below; casting an entire lower-precision grid here
            # would reintroduce two input-sized peak allocations.
            incoming_power = np.asarray(grid.rcs_power)
            incoming_phase = np.asarray(grid.rcs_phase)
            # np.ix_ over all four complete axes materialises full input-sized
            # copies. Tile every axis so each advanced-index block stays
            # bounded without degenerating into a Python loop per scalar cell.
            pol_block = max(1, min(len(p_idx), _JOIN_MERGE_BLOCK_CELLS))
            freq_block = max(
                1,
                min(len(f_idx), _JOIN_MERGE_BLOCK_CELLS // pol_block),
            )
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS // (pol_block * freq_block),
            )
            elev_block = max(1, min(len(el_idx), remaining))
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS
                // (pol_block * freq_block * elev_block),
            )
            az_block = max(1, min(len(az_idx), remaining))
            for a_start in range(0, len(az_idx), az_block):
                a_stop = min(a_start + az_block, len(az_idx))
                union_a = az_idx[a_start:a_stop]
                for e_start in range(0, len(el_idx), elev_block):
                    e_stop = min(e_start + elev_block, len(el_idx))
                    union_e = el_idx[e_start:e_stop]
                    for f_start in range(0, len(f_idx), freq_block):
                        f_stop = min(f_start + freq_block, len(f_idx))
                        union_f = f_idx[f_start:f_stop]
                        for p_start in range(0, len(p_idx), pol_block):
                            p_stop = min(p_start + pol_block, len(p_idx))
                            union_p = p_idx[p_start:p_stop]
                            target = np.ix_(union_a, union_e, union_f, union_p)
                            existing_power = joined_power[target]
                            existing_phase = joined_phase[target]
                            block_selection = (
                                slice(a_start, a_stop),
                                slice(e_start, e_stop),
                                slice(f_start, f_stop),
                                slice(p_start, p_stop),
                            )
                            block_power = incoming_power[block_selection]
                            block_phase = incoming_phase[block_selection]
                            if preserve_raw_amplitude:
                                incoming_raw_real, incoming_raw_imag = raw_inputs[
                                    grid_index
                                ]
                                existing_raw_real = joined_raw_real[target]
                                existing_raw_imag = joined_raw_imag[target]
                                block_raw_real = incoming_raw_real[block_selection]
                                block_raw_imag = incoming_raw_imag[block_selection]

                            both = np.isfinite(existing_power) & np.isfinite(block_power)
                            power_conflict = both & ~np.isclose(
                                existing_power, block_power, rtol=1e-6, atol=1e-12
                            )
                            both_phase = (
                                both
                                & np.isfinite(existing_phase)
                                & np.isfinite(block_phase)
                            )
                            phase_delta = np.abs(
                                np.angle(np.exp(1j * (existing_phase - block_phase)))
                            )
                            both_zero = (
                                both
                                & (existing_power == 0.0)
                                & (block_power == 0.0)
                            )
                            # Phase is undefined at an exact zero field, so
                            # arbitrary exporter phase fillers there cannot be
                            # a physically meaningful overlap conflict.
                            phase_conflict = (
                                both_phase
                                & ~both_zero
                                & (phase_delta > 1e-5)
                            )
                            raw_conflict = np.zeros_like(both)
                            if preserve_raw_amplitude:
                                both_raw = (
                                    both
                                    & np.isfinite(existing_raw_real)
                                    & np.isfinite(existing_raw_imag)
                                    & np.isfinite(block_raw_real)
                                    & np.isfinite(block_raw_imag)
                                )
                                raw_equal = (
                                    np.isclose(
                                        existing_raw_real,
                                        block_raw_real,
                                        rtol=1.0e-12,
                                        atol=1.0e-15,
                                    )
                                    & np.isclose(
                                        existing_raw_imag,
                                        block_raw_imag,
                                        rtol=1.0e-12,
                                        atol=1.0e-15,
                                    )
                                )
                                raw_conflict = both_raw & ~raw_equal
                            if overlap == "error" and (
                                np.any(power_conflict)
                                or np.any(phase_conflict)
                                or np.any(raw_conflict)
                            ):
                                raise ValueError(
                                    "conflicting finite samples overlap during join"
                                )

                            if overlap == "last":
                                take_power = np.isfinite(block_power)
                                take_phase = take_power
                            else:
                                take_power = (
                                    ~np.isfinite(existing_power)
                                    & np.isfinite(block_power)
                                )
                                # Equal finite power with a missing earlier
                                # phase is complementary data, not a replacement.
                                fill_phase = (
                                    both
                                    & ~power_conflict
                                    & ~np.isfinite(existing_phase)
                                    & np.isfinite(block_phase)
                                )
                                take_phase = take_power | fill_phase
                            existing_power[take_power] = block_power[take_power]
                            existing_phase[take_phase] = block_phase[take_phase]
                            joined_power[target] = existing_power
                            joined_phase[target] = existing_phase
                            if preserve_raw_amplitude:
                                existing_raw_real[take_power] = block_raw_real[
                                    take_power
                                ]
                                existing_raw_imag[take_power] = block_raw_imag[
                                    take_power
                                ]
                                joined_raw_real[target] = existing_raw_real
                                joined_raw_imag[target] = existing_raw_imag

        # Inputs are normally already clean, but RcsGrid arrays are public and
        # may have been mutated. Preserve constructor sanitation in place, then
        # transfer ownership of these newly allocated arrays without copying.
        finite = np.empty(shape, dtype=bool)
        np.isfinite(joined_power, out=finite)
        np.maximum(joined_power, 0.0, out=joined_power, where=finite)
        np.logical_not(finite, out=finite)
        joined_power[finite] = np.nan
        np.isfinite(joined_phase, out=finite)
        np.logical_not(finite, out=finite)
        joined_phase[finite] = np.nan
        np.isfinite(joined_power, out=finite)
        np.logical_not(finite, out=finite)
        joined_phase[finite] = np.nan
        if preserve_raw_amplitude:
            joined_raw_real[finite] = np.nan
            joined_raw_imag[finite] = np.nan
        del finite

        output_extra = dict(preserved_scalar_extra)
        if metadata_assumptions:
            output_extra["merge_metadata_assumption_json"] = json.dumps(
                {
                    "schema": "grim.merge-metadata-assumption.v1",
                    "operation": "strict-merge",
                    "input_count": len(grids),
                    "one_sided_declarations": metadata_assumptions,
                    "declarations_inferred": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        if preserve_raw_amplitude:
            output_extra["rcs_amp_real"] = joined_raw_real
            output_extra["rcs_amp_imag"] = joined_raw_imag
            output_extra["raw_complex_amplitude_preserved"] = True

        history = str(ref.history or "").strip()
        if metadata_assumptions:
            note = (
                "Join allowed one-sided metadata as unspecified: "
                + ", ".join(sorted(metadata_assumptions))
            )
            history = f"{history}\n{note}" if history else note

        return cls(
            az_union,
            el_union,
            f_union,
            p_union,
            rcs_power=joined_power,
            rcs_phase=joined_phase,
            rcs_domain="power_phase",
            source_path=ref.source_path,
            history=history,
            units=dict(ref.units),
            extra=output_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def overlap_many(cls, *grids, tol=1e-6):
        """Return one cropped dataset per input, all on common overlap axes.

        Every input participates equally in one all-selected intersection; no
        input is treated as a reference grid.  Numeric matching and the output
        axis coordinates are therefore independent of input selection order.

        Overlap is enforced cell-wise: if any input is missing data (NaN) at a
        given (az, el, freq, pol) cell, that cell is set to NaN in every output.
        Axis values whose entire slice becomes NaN after this intersection are
        dropped — so e.g. a frequency that one dataset lacks for HH but all
        datasets have for VV will stay on the axis, with HH masked to NaN.
        """
        grids = cls._ensure_grids(grids)
        if len(grids) == 1:
            return [grids[0]]
        for grid in grids[1:]:
            grids[0]._assert_axis_metadata_compatible(grid)

        az_common, az_indices = cls._common_axis_alignment(
            [grid.azimuths for grid in grids], tol=tol
        )
        el_common, el_indices = cls._common_axis_alignment(
            [grid.elevations for grid in grids], tol=tol
        )
        f_common, f_indices = cls._common_axis_alignment(
            [grid.frequencies for grid in grids], tol=tol
        )
        p_common, p_indices = cls._common_axis_alignment(
            [grid.polarizations for grid in grids], tol=0.0
        )

        if (
            az_common.size == 0
            or el_common.size == 0
            or f_common.size == 0
            or p_common.size == 0
        ):
            raise ValueError("no overlap across one or more axes")

        aligned_power = []
        aligned_phase = []
        for grid_idx, grid in enumerate(grids):
            az_idx = az_indices[grid_idx]
            el_idx = el_indices[grid_idx]
            f_idx = f_indices[grid_idx]
            p_idx = p_indices[grid_idx]
            aligned_power.append(grid.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)].copy())
            aligned_phase.append(grid.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)].copy())

        missing_any = np.zeros(aligned_power[0].shape, dtype=bool)
        for power in aligned_power:
            missing_any |= ~np.isfinite(power)
        for power, phase in zip(aligned_power, aligned_phase):
            power[missing_any] = np.nan
            phase[missing_any] = np.nan

        finite = ~missing_any
        az_keep = finite.any(axis=(1, 2, 3))
        el_keep = finite.any(axis=(0, 2, 3))
        f_keep = finite.any(axis=(0, 1, 3))
        p_keep = finite.any(axis=(0, 1, 2))

        if not (az_keep.any() and el_keep.any() and f_keep.any() and p_keep.any()):
            raise ValueError("no overlap across one or more axes")

        az_sel = np.where(az_keep)[0]
        el_sel = np.where(el_keep)[0]
        f_sel = np.where(f_keep)[0]
        p_sel = np.where(p_keep)[0]
        az_common = az_common[az_sel]
        el_common = el_common[el_sel]
        f_common = f_common[f_sel]
        p_common = p_common[p_sel]

        overlap_grids = []
        final_common_selection = np.ix_(az_sel, el_sel, f_sel, p_sel)
        final_missing = missing_any[final_common_selection]
        for grid_index, (grid, power, phase) in enumerate(
            zip(grids, aligned_power, aligned_phase)
        ):
            source_indices = (
                np.asarray(az_indices[grid_index], dtype=int)[az_sel],
                np.asarray(el_indices[grid_index], dtype=int)[el_sel],
                np.asarray(f_indices[grid_index], dtype=int)[f_sel],
                np.asarray(p_indices[grid_index], dtype=int)[p_sel],
            )
            source_selection = np.ix_(*source_indices)
            source_axes = (
                np.asarray(grid.azimuths)[source_indices[0]],
                np.asarray(grid.elevations)[source_indices[1]],
                np.asarray(grid.frequencies)[source_indices[2]],
                np.asarray(grid.polarizations)[source_indices[3]],
            )
            relabeled = not all(
                np.array_equal(source, target)
                for source, target in zip(
                    source_axes, (az_common, el_common, f_common, p_common)
                )
            )
            overlap_extra = grid._exact_transform_extra(
                lambda value, selection=source_selection: value[selection],
                coordinate_change=("overlap-axis-relabel" if relabeled else None),
            )
            for raw_key in grid._RAW_AMPLITUDE_EXTRA_KEYS:
                if raw_key in overlap_extra:
                    raw = np.asarray(overlap_extra[raw_key]).copy()
                    raw[final_missing] = np.nan
                    overlap_extra[raw_key] = raw
            overlap_grids.append(
                cls(
                    az_common,
                    el_common,
                    f_common,
                    p_common,
                    rcs_power=power[final_common_selection],
                    rcs_phase=phase[final_common_selection],
                    rcs_domain="power_phase",
                    source_path=grid.source_path,
                    history=grid.history,
                    units=dict(grid.units),
                    extra=overlap_extra,
                )
            )

        return overlap_grids

    def statistics_dataset(
        self,
        statistic="mean",
        axes=("azimuth", "elevation", "frequency"),
        *,
        domain="magnitude",
        percentile=50.0,
        broadcast_reduced=False,
    ):
        """Compute a statistic over selected axes and return a dataset."""
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2, "polarization": 3}
        axis_alias = {
            "azimuths": "azimuth",
            "elevations": "elevation",
            "frequencies": "frequency",
            "polarizations": "polarization",
            "az": "azimuth",
            "el": "elevation",
            "freq": "frequency",
            "pol": "polarization",
        }

        axes_list = self._as_list(axes)
        if axes_list is None:
            raise ValueError("axes must include at least one axis")
        reduce_axes = []
        for axis_name in axes_list:
            key = str(axis_name).strip().lower()
            key = axis_alias.get(key, key)
            if key not in axis_map:
                raise ValueError(f"unknown axis: {axis_name}")
            idx = axis_map[key]
            if idx not in reduce_axes:
                reduce_axes.append(idx)
        if not reduce_axes:
            raise ValueError("axes must include at least one axis")
        reduce_axes = tuple(sorted(reduce_axes))

        if domain == "complex":
            values = self.rcs
        elif domain == "magnitude":
            values = self.rcs_power
        elif domain in ("db", "dbsm"):
            values = self.linear_to_dbsm(self.rcs_power)
        elif domain == "dbke":
            # Conversion helpers accept values in the dataset's declared unit.
            # Passing preconverted Hz here caused a second unit conversion.
            freq_grid = np.asarray(self.frequencies, dtype=float).reshape(1, 1, -1, 1)
            values = self.linear_to_dbke(self.rcs_power, freq_grid)
        else:
            raise ValueError("domain must be 'complex', 'magnitude', 'dbsm', or 'dbke'")

        stat_key = str(statistic).strip().lower()
        if stat_key.startswith("p") and stat_key[1:].replace(".", "", 1).isdigit():
            percentile = float(stat_key[1:])
            stat_key = "percentile"

        if domain == "complex" and stat_key == "percentile":
            raise ValueError("percentile on complex values is not supported; use magnitude, dbsm, or dbke domain")
        if stat_key == "std" and domain in {"db", "dbsm", "dbke"}:
            raise ValueError(
                "standard deviation in a logarithmic domain is a dB spread, "
                "not an absolute RCS dataset. Use domain='magnitude' for a "
                "linear-power RCS standard deviation."
            )

        if stat_key == "mean":
            reduced = np.nanmean(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "median":
            reduced = np.nanmedian(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "min":
            reduced = np.nanmin(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "max":
            reduced = np.nanmax(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "std":
            reduced = np.nanstd(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "percentile":
            reduced = np.nanpercentile(values, float(percentile), axis=reduce_axes, keepdims=True)
        else:
            raise ValueError(
                "statistic must be mean, median, min, max, std, percentile, or pXX (for percentile XX)"
            )

        axis_values = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        if broadcast_reduced:
            # Repeat the reduced result across each reduced axis so the output
            # keeps original axis lengths for downstream plotting.
            reduced = np.broadcast_to(reduced, values.shape).copy()
        else:
            for axis_idx in reduce_axes:
                original = axis_values[axis_idx]
                if axis_idx == 3:
                    axis_values[axis_idx] = np.asarray(["ALL"])
                else:
                    numeric = np.asarray(original, dtype=float)
                    rep = float(np.nanmean(numeric)) if numeric.size else 0.0
                    axis_values[axis_idx] = np.asarray([rep], dtype=float)

        statistics_coherent = domain == "complex"
        statistics_extra = self._derived_response_extra(
            operation=f"statistics-{stat_key}-{domain}",
            coherent=statistics_coherent,
        )
        axis_names = ("azimuth", "elevation", "frequency", "polarization")
        axis_summaries = {}
        source_axes = (
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
        )
        for axis_idx in reduce_axes:
            axis_name = axis_names[axis_idx]
            original = np.asarray(source_axes[axis_idx])
            summary = {"source_count": int(original.size)}
            if axis_idx == 3:
                summary["representative"] = (
                    None if broadcast_reduced else "ALL"
                )
            else:
                numeric = np.asarray(original, dtype=float)
                finite = numeric[np.isfinite(numeric)]
                summary.update(
                    source_min=(float(np.min(finite)) if finite.size else None),
                    source_max=(float(np.max(finite)) if finite.size else None),
                    representative=(
                        None
                        if broadcast_reduced
                        else float(axis_values[axis_idx][0])
                    ),
                )
            axis_summaries[axis_name] = summary
        statistics_extra["statistics_reduction_json"] = json.dumps(
            {
                "schema": "grim.statistics-reduction.v1",
                "statistic": stat_key,
                "percentile": (
                    float(percentile) if stat_key == "percentile" else None
                ),
                "domain": domain,
                "reduced_axes": [axis_names[index] for index in reduce_axes],
                "broadcast_reduced": bool(broadcast_reduced),
                "coordinate_semantics": (
                    "original coordinates with repeated aggregate value"
                    if broadcast_reduced
                    else "representative aggregate label; not an observed sample"
                ),
                "axes": axis_summaries,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        source_role = self._declared_scalar_metadata(
            "assembly_response_role"
        ).strip().casefold()
        if source_role == "features_only_delta":
            statistics_extra["assembly_response_role"] = (
                "coherent_field_sum"
                if statistics_coherent
                else "incoherent_power_sum"
            )
        if statistics_coherent:
            self._invalidate_assembly_sampling_hash(
                statistics_extra, f"statistics-{stat_key}-{domain}"
            )

        if domain == "complex":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                reduced,
                rcs_domain="power_phase",
                extra=statistics_extra,
            )
        if domain == "magnitude":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                rcs_power=np.asarray(reduced, dtype=self.rcs_power.dtype),
                rcs_phase=np.full(reduced.shape, np.nan, dtype=self.rcs_phase.dtype),
                rcs_domain="power_phase",
                extra=statistics_extra,
            )
        # db domain: compute in a log domain, then store as linear so future conversion reproduces the reduced values.
        if domain == "dbke":
            freq_grid = np.asarray(axis_values[2], dtype=float).reshape(1, 1, -1, 1)
            reduced_linear = np.asarray(
                self.dbke_to_linear(np.asarray(reduced, dtype=float), freq_grid),
                dtype=self.rcs_power.dtype,
            )
        else:
            reduced_linear = np.asarray(
                10.0 ** (np.asarray(reduced, dtype=float) / 10.0),
                dtype=self.rcs_power.dtype,
            )
        return self._new_grid(
            axis_values[0],
            axis_values[1],
            axis_values[2],
            axis_values[3],
            rcs_power=reduced_linear,
            rcs_phase=np.full(reduced_linear.shape, np.nan, dtype=self.rcs_phase.dtype),
            rcs_domain="power_phase",
            extra=statistics_extra,
        )

    def _index_for_value(self, axis, value, tol=0.0):
        """Find the first index of a value on an axis.

        Args:
            axis: 1D array to search.
            value: Value to find.
            tol: Absolute tolerance for numeric matching. Text axes such as
                polarization are always matched exactly.

        Returns:
            Integer index of the first match.

        Raises:
            ValueError: if no match is found.
        """
        axis_arr = np.asarray(axis)
        if tol > 0.0 and axis_arr.dtype.kind in "biufc":
            matches = np.where(np.isclose(axis_arr, value, atol=tol, rtol=0.0))[0]
        else:
            matches = np.where(axis_arr == value)[0]
        if matches.size == 0:
            raise ValueError(f"value {value} not found on axis")
        return int(matches[0])

    def get_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0):
        """Fetch a single sample by axis values.

        Use when you have physical axis values rather than indices.

        Args:
            azimuth: Azimuth value.
            elevation: Elevation value.
            frequency: Frequency value.
            polarization: Polarization label.
            tol: Absolute tolerance for numeric matching.

        Returns:
            Complex RCS sample.
        """
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(
            self.polarizations, str(polarization).strip().upper(), tol=tol
        )
        return self.rcs_slice((az_idx, el_idx, f_idx, p_idx))

    def rcs_to_dbsm(self, rcs_value, eps=1e-12):
        """Convert linear RCS to dBsm.

        Args:
            rcs_value: Complex or real RCS value(s).
            eps: Floor to avoid log(0).

        Returns:
            dBsm value(s) as float or ndarray.
        """
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_dbsm(linear, eps=eps)

    def rcs_to_dbke(self, rcs_value, frequency_value, eps=1e-12):
        """Convert linear 2D scattering width to absolute dBke."""
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_dbke(linear, frequency_value, eps=eps)

    def rcs_to_display_db(self, rcs_value, frequency_value=None, eps=1e-12):
        """Convert to the dataset's preferred log-power display unit."""
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_default_db(linear, frequency_value=frequency_value, eps=eps)

    def get_dbsm(self, az_idx, el_idx, f_idx, p_idx, eps=1e-12):
        """Fetch a sample by indices and return dBsm."""
        return self.linear_to_dbsm(self.rcs_power[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbke(self, az_idx, el_idx, f_idx, p_idx, eps=1e-12):
        """Fetch a sample by indices and return dBke."""
        freq_value = self.frequencies[f_idx]
        return self.linear_to_dbke(self.rcs_power[az_idx, el_idx, f_idx, p_idx], freq_value, eps=eps)

    def get_dbsm_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBsm."""
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(
            self.polarizations, str(polarization).strip().upper(), tol=tol
        )
        return self.linear_to_dbsm(self.rcs_power[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbke_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBke."""
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(
            self.polarizations, str(polarization).strip().upper(), tol=tol
        )
        return self.linear_to_dbke(self.rcs_power[az_idx, el_idx, f_idx, p_idx], self.frequencies[f_idx], eps=eps)

    # keys this class fully models and always rewrites itself.  rcs_domain and
    # power_domain are deliberately NOT here: a producer may tag a file with a
    # domain word outside this class's 3-value vocabulary (the GHOST backend
    # writes rcs_domain='delta' / power_domain='delta_amp_sq', and routes on it),
    # so those tags are captured in `extra` and re-emitted verbatim by save().
    _RESERVED_KEYS = ("azimuths", "elevations", "frequencies", "polarizations",
                      "rcs_power", "rcs_phase", "source_path", "history", "units")

    def _extra_to_write(self):
        """Passthrough keys to re-emit, minus anything whose shape no longer fits.

        A grid-sized array (for example ``rcs_amp_real``) is emitted only while
        its leading dimensions match the current grid. Exact transforms create
        a correspondingly transformed array; nonexact transforms omit it. A
        mismatched four-dimensional shape is always dropped instead of written
        stale.
        Lower-dimensional arrays are independent ancillary models, not partial
        RCS grids: GHOST's BoR profiles, body-model amplitudes, and surface
        triangles must survive an unchanged load/save round-trip.
        """
        expected = (len(self.azimuths), len(self.elevations),
                    len(self.frequencies), len(self.polarizations))
        out = {}
        for key, value in self.extra.items():
            if key in self._RESERVED_KEYS:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 4 and arr.shape[:4] != expected:
                continue
            out[key] = value
        return out

    def save(self, path, *, compressed=True):
        """Save the grid to a .grim (npz) file.

        Passthrough metadata from ``extra`` is written first (so the grid's own
        axes and samples always win on a name clash), which is what lets a file
        carrying a raw complex amplitude survive a load/save round-trip.
        The archive is fully written and flushed to a same-directory staging
        file before ``os.replace`` publishes it, so a failed save leaves an
        existing artifact intact.

        Args:
            path: Output path, with or without .grim.
            compressed: ``True`` (default) writes a compact ZIP-compressed
                archive for distribution/storage. ``False`` uses the faster
                uncompressed NPZ path for temporary high-throughput work.

        Returns:
            The actual path written (always ends with .grim).
        """
        if not isinstance(compressed, (bool, np.bool_)):
            raise TypeError("compressed must be True or False")
        path = os.fspath(path)
        if not path.casefold().endswith(".grim"):
            path = f"{path}.grim"
        extra_to_write = self._extra_to_write()
        self._validate_native_payload(
            path=path,
            azimuths=self.azimuths,
            elevations=self.elevations,
            frequencies=self.frequencies,
            polarizations=self.polarizations,
            rcs_power=self.rcs_power,
            rcs_phase=self.rcs_phase,
            units=self.units,
            extra=extra_to_write,
        )
        directory = os.path.dirname(os.path.abspath(path)) or os.curdir
        fd, stage_path = tempfile.mkstemp(
            prefix=".grim-write-",
            suffix=".staging",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                fd = -1
                units_payload = json.dumps(self.units) if self.units else ""
                payload = dict(extra_to_write)                  # passthrough first
                payload.update(
                    azimuths=self.azimuths,
                    elevations=self.elevations,
                    frequencies=self.frequencies,
                    polarizations=self.polarizations,
                    rcs_power=self.rcs_power,
                    rcs_phase=self.rcs_phase,
                    rcs_domain="power_phase",
                    power_domain=self.power_domain,
                    source_path=self.source_path if self.source_path is not None else "",
                    history=self.history if self.history is not None else "",
                    units=units_payload,
                )
                # a source domain tag we could not represent wins back its own slot:
                # cropping or joining does not change what the samples MEAN, and a
                # consumer may route on it (see _RESERVED_KEYS)
                for tag in ("rcs_domain", "power_domain"):
                    if tag in self.extra:
                        payload[tag] = self.extra[tag]
                object_keys = [
                    str(key)
                    for key, value in payload.items()
                    if np.asarray(value).dtype.hasobject
                ]
                if object_keys:
                    raise ValueError(
                        "cannot save a pickle-free .grim archive because metadata "
                        "contains object-typed value(s): "
                        + ", ".join(sorted(object_keys))
                        + ". Convert those values to numeric/string arrays or JSON text."
                    )
                writer = np.savez_compressed if bool(compressed) else np.savez
                writer(f, **payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(stage_path, path)
        finally:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(stage_path):
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
        return path

    @classmethod
    def _raw_complex_consistency_report(
        cls,
        *,
        expected_shape,
        frequencies,
        rcs_power,
        rcs_phase,
        units,
        extra,
    ):
        """Check a solver raw-field pair against displayed power and phase.

        GHOST stores its unnormalised far-field amplitude alongside physical
        ``rcs_power``.  The relationship is quantity dependent: sigma3D is
        ``4*pi*|A|^2`` and sigma2D is ``|A|^2/(4*k0)``.  This routine mirrors
        the producer-side tolerances while scanning bounded blocks, so loading
        or auditing a large archive never constructs another grid-sized
        complex or expected-power array.
        """

        metadata = dict(extra or {})
        has_real = "rcs_amp_real" in metadata
        has_imag = "rcs_amp_imag" in metadata
        report = {
            "present": bool(has_real or has_imag),
            "complete_pair": bool(has_real and has_imag),
            "finite_pair_count": 0,
            "missing_pair_count": 0,
            "invalid_pair_count": 0,
            "coverage_mismatch_count": 0,
            "live_phase_missing_count": 0,
            "phase_mismatch_count": 0,
            "power_mismatch_count": 0,
            "normalization_overflow_count": 0,
            "maximum_phase_error_rad": None,
            "maximum_power_absolute_error": None,
            "normalization": None,
            "issues": [],
        }

        def issue(code, message, **details):
            item = {"code": str(code), "message": str(message)}
            item.update(details)
            report["issues"].append(item)

        raw_flag = metadata.get("raw_complex_amplitude_preserved")
        flag_true = False
        if raw_flag is not None:
            raw_flag_array = np.asarray(raw_flag)
            if raw_flag_array.size != 1:
                issue(
                    "invalid_raw_complex_preserved_flag",
                    "raw_complex_amplitude_preserved must be scalar",
                )
            else:
                flag_value = raw_flag_array.reshape(-1)[0]
                if isinstance(flag_value, (str, np.str_, bytes, np.bytes_)):
                    if isinstance(flag_value, (bytes, np.bytes_)):
                        try:
                            flag_value = bytes(flag_value).decode("ascii")
                        except UnicodeDecodeError:
                            flag_value = ""
                    flag_true = str(flag_value).strip().casefold() in {
                        "1", "true", "yes", "on"
                    }
                else:
                    try:
                        flag_true = bool(flag_value)
                    except (TypeError, ValueError):
                        flag_true = False

        if has_real != has_imag:
            issue(
                "partial_raw_complex_pair",
                "raw complex amplitude must provide both rcs_amp_real and rcs_amp_imag",
            )
            return report
        if not has_real:
            if flag_true:
                issue(
                    "missing_raw_complex_pair",
                    "raw_complex_amplitude_preserved is true but the raw amplitude grids are absent",
                )
            return report

        raw_real = np.asarray(metadata["rcs_amp_real"])
        raw_imag = np.asarray(metadata["rcs_amp_imag"])
        for name, values in (
            ("rcs_amp_real", raw_real),
            ("rcs_amp_imag", raw_imag),
        ):
            if values.shape != tuple(expected_shape):
                issue(
                    "raw_complex_shape_mismatch",
                    f"{name} shape {values.shape} does not match axes {tuple(expected_shape)}",
                    field=name,
                )
            if values.dtype.kind not in "iuf":
                issue(
                    "non_numeric_raw_complex",
                    f"{name} must be real numeric",
                    field=name,
                )
        if any(item["code"] != "invalid_raw_complex_preserved_flag" for item in report["issues"]):
            return report

        power = np.asarray(rcs_power)
        phase = np.asarray(rcs_phase)
        if power.shape != tuple(expected_shape) or phase.shape != tuple(expected_shape):
            return report
        if power.dtype.kind not in "iuf" or phase.dtype.kind not in "iuf":
            return report

        normalized_units = dict(units or {})
        quantity = str(
            normalized_units.get("rcs_linear_quantity", "")
        ).strip().casefold()
        if not quantity:
            log_unit = str(
                normalized_units.get("rcs_log_unit", "dBsm")
            ).strip().casefold()
            quantity = "sigma_2d" if log_unit == "dbke" else "sigma_3d"
        if quantity not in {"sigma_2d", "sigma_3d"}:
            issue(
                "unsupported_raw_complex_normalization",
                "raw complex amplitude requires rcs_linear_quantity sigma_2d or sigma_3d",
                linear_quantity=quantity,
            )
            return report
        report["normalization"] = quantity

        frequency_values = np.asarray(frequencies, dtype=np.float64)
        frequency_unit = cls._canonical_unit(
            normalized_units.get("frequency"), _FREQUENCY_UNITS, "GHz"
        )
        frequency_scale = {
            "Hz": 1.0,
            "kHz": 1.0e3,
            "MHz": 1.0e6,
            "GHz": 1.0e9,
        }.get(frequency_unit)
        if frequency_scale is None:
            issue(
                "unsupported_raw_complex_frequency_unit",
                "raw complex amplitude normalization requires a supported frequency unit",
                frequency_unit=str(normalized_units.get("frequency")),
            )
            return report

        max_phase_error = 0.0
        max_power_error = 0.0
        float32_epsilon = np.finfo(np.float32).eps
        float32_tiny = np.finfo(np.float32).tiny

        for frequency_index, frequency_value in enumerate(frequency_values):
            if not np.isfinite(frequency_value) or frequency_value <= 0.0:
                issue(
                    "invalid_raw_complex_frequency",
                    "raw complex amplitude normalization requires positive finite frequencies",
                )
                return report
            if quantity == "sigma_2d":
                k0 = (
                    2.0
                    * np.pi
                    * float(frequency_value)
                    * float(frequency_scale)
                    / C0
                )
                power_scale = 1.0 / (4.0 * k0)
            else:
                power_scale = 4.0 * np.pi

            iterator = np.nditer(
                (
                    power[:, :, frequency_index, :],
                    phase[:, :, frequency_index, :],
                    raw_real[:, :, frequency_index, :],
                    raw_imag[:, :, frequency_index, :],
                ),
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"], ["readonly"], ["readonly"]],
                order="K",
                buffersize=_RAW_COMPLEX_VALIDATION_BLOCK_CELLS,
            )
            for power_block, phase_block, real_block, imag_block in iterator:
                power_block = np.asarray(power_block, dtype=np.float64)
                phase_block = np.asarray(phase_block, dtype=np.float64)
                real_block = np.asarray(real_block, dtype=np.float64)
                imag_block = np.asarray(imag_block, dtype=np.float64)

                real_finite = np.isfinite(real_block)
                imag_finite = np.isfinite(imag_block)
                raw_finite = real_finite & imag_finite
                raw_missing = np.isnan(real_block) & np.isnan(imag_block)
                invalid_pair = ~(raw_finite | raw_missing)
                power_finite = np.isfinite(power_block)

                report["finite_pair_count"] += int(np.count_nonzero(raw_finite))
                report["missing_pair_count"] += int(np.count_nonzero(raw_missing))
                report["invalid_pair_count"] += int(np.count_nonzero(invalid_pair))
                report["coverage_mismatch_count"] += int(
                    np.count_nonzero(raw_finite != power_finite)
                )

                comparable = raw_finite & power_finite
                if not np.any(comparable):
                    continue

                comparable_real = real_block[comparable]
                comparable_imag = imag_block[comparable]
                comparable_power = power_block[comparable]
                with np.errstate(over="ignore", invalid="ignore"):
                    amp_abs2 = (
                        comparable_real * comparable_real
                        + comparable_imag * comparable_imag
                    )
                    expected_power = amp_abs2 * power_scale
                expected_finite = np.isfinite(expected_power)
                overflow_count = int(np.count_nonzero(~expected_finite))
                report["normalization_overflow_count"] += overflow_count
                if np.any(expected_finite):
                    expected_values = expected_power[expected_finite]
                    stored_values = comparable_power[expected_finite]
                    absolute_error = np.abs(stored_values - expected_values)
                    tolerance = (
                        16.0
                        * float32_epsilon
                        * np.maximum(expected_values, stored_values)
                        + float32_tiny
                    )
                    report["power_mismatch_count"] += int(
                        np.count_nonzero(absolute_error > tolerance)
                    )
                    if absolute_error.size:
                        max_power_error = max(
                            max_power_error, float(np.max(absolute_error))
                        )

                live = comparable & (
                    (np.abs(real_block) > float32_tiny)
                    | (np.abs(imag_block) > float32_tiny)
                )
                live_phase_missing = live & ~np.isfinite(phase_block)
                report["live_phase_missing_count"] += int(
                    np.count_nonzero(live_phase_missing)
                )
                phase_comparable = live & np.isfinite(phase_block)
                if np.any(phase_comparable):
                    raw_angle = np.arctan2(
                        imag_block[phase_comparable], real_block[phase_comparable]
                    )
                    phase_difference = phase_block[phase_comparable] - raw_angle
                    phase_error = np.abs(
                        np.arctan2(np.sin(phase_difference), np.cos(phase_difference))
                    )
                    report["phase_mismatch_count"] += int(
                        np.count_nonzero(phase_error > 2.0e-5)
                    )
                    if phase_error.size:
                        max_phase_error = max(
                            max_phase_error, float(np.max(phase_error))
                        )

        report["maximum_phase_error_rad"] = float(max_phase_error)
        report["maximum_power_absolute_error"] = float(max_power_error)
        if report["invalid_pair_count"]:
            issue(
                "invalid_raw_complex_pair",
                "raw complex amplitude contains one-sided NaN or infinite component samples",
                count=report["invalid_pair_count"],
            )
        if report["coverage_mismatch_count"]:
            issue(
                "raw_complex_coverage_mismatch",
                "raw complex amplitude finite coverage does not match rcs_power",
                count=report["coverage_mismatch_count"],
            )
        if report["normalization_overflow_count"]:
            issue(
                "raw_complex_normalization_overflow",
                f"raw complex amplitude is too large to form finite {quantity} power",
                count=report["normalization_overflow_count"],
            )
        if report["power_mismatch_count"]:
            issue(
                "raw_complex_power_mismatch",
                "rcs_power is inconsistent with the stored raw complex amplitude "
                f"under {quantity} normalization",
                count=report["power_mismatch_count"],
                maximum_absolute_error=report["maximum_power_absolute_error"],
            )
        if report["live_phase_missing_count"]:
            issue(
                "raw_complex_phase_missing",
                "nonzero raw complex amplitude has no finite stored rcs_phase",
                count=report["live_phase_missing_count"],
            )
        if report["phase_mismatch_count"]:
            issue(
                "raw_complex_phase_mismatch",
                "rcs_phase is inconsistent with the stored raw complex amplitude",
                count=report["phase_mismatch_count"],
                maximum_phase_error_rad=report["maximum_phase_error_rad"],
            )
        return report

    @classmethod
    def _validate_native_payload(
        cls,
        *,
        path,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs_power,
        rcs_phase,
        units,
        extra=None,
    ):
        """Validate the native archive before constructor sanitation.

        NaN RCS cells are intentional sparse-grid markers and are retained.
        Infinities, negative finite power, malformed axes, and ambiguous unit
        declarations are rejected rather than silently repaired.
        """

        numeric_axes = {}
        for name, raw_values in (
            ("azimuth", azimuths),
            ("elevation", elevations),
            ("frequency", frequencies),
        ):
            values = np.asarray(raw_values)
            if (
                values.ndim != 1
                or values.size == 0
                or values.dtype.kind not in "iuf"
            ):
                raise ValueError(
                    f"{path} contains an invalid {name} axis; expected a "
                    "nonempty one-dimensional real numeric array"
                )
            checked_values = values.astype(np.float64, copy=False)
            if np.any(~np.isfinite(checked_values)):
                raise ValueError(f"{path} contains a nonfinite {name} coordinate")
            if np.unique(checked_values).size != checked_values.size:
                raise ValueError(f"{path} contains duplicate {name} coordinates")
            if name == "frequency" and np.any(checked_values <= 0.0):
                raise ValueError(
                    f"{path} contains a nonpositive frequency coordinate"
                )
            # Preserve constructor normalization for float32 axes: values such
            # as float32(0.1) represent the user's decimal 0.1, not its binary
            # quantization noise widened verbatim into float64.
            numeric_axes[name] = cls._clean_axis(values)

        raw_polarizations = np.asarray(polarizations)
        if raw_polarizations.ndim != 1 or raw_polarizations.size == 0:
            raise ValueError(
                f"{path} contains an invalid polarization axis; expected a "
                "nonempty one-dimensional string array"
            )
        labels = []
        for raw_label in raw_polarizations.tolist():
            if isinstance(raw_label, bytes):
                try:
                    label = raw_label.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"{path} contains a non-UTF-8 polarization label"
                    ) from exc
            elif isinstance(raw_label, (str, np.str_)):
                label = str(raw_label)
            else:
                raise ValueError(
                    f"{path} contains a non-string polarization label "
                    f"{raw_label!r}"
                )
            label = label.strip()
            if not label:
                raise ValueError(f"{path} contains a blank polarization label")
            labels.append(label)
        normalized_labels = [label.casefold() for label in labels]
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError(
                f"{path} contains duplicate polarization labels after normalization"
            )

        expected = (
            len(numeric_axes["azimuth"]),
            len(numeric_axes["elevation"]),
            len(numeric_axes["frequency"]),
            len(labels),
        )
        power = np.asarray(rcs_power)
        phase = np.asarray(rcs_phase)
        for name, values in (("rcs_power", power), ("rcs_phase", phase)):
            if values.shape != expected:
                raise ValueError(
                    f"{path} contains {name} shape {values.shape}; expected {expected}"
                )
            if values.dtype.kind not in "iuf":
                raise ValueError(f"{path} contains non-real-numeric {name}")
            if np.any(np.isinf(values)):
                raise ValueError(f"{path} contains infinite {name} samples")
        # NaN compares false, so this catches every finite negative without
        # materialising a second advanced-index copy of a potentially huge
        # sparse grid. Negative infinity was already rejected above.
        if np.any(power < 0.0):
            raise ValueError(f"{path} contains negative finite rcs_power samples")

        normalized_units = dict(units or {})
        for key, aliases, default in (
            ("azimuth", _ANGLE_UNITS, "deg"),
            ("elevation", _ANGLE_UNITS, "deg"),
            ("frequency", _FREQUENCY_UNITS, "GHz"),
        ):
            raw_unit = normalized_units.get(key)
            canonical = cls._canonical_unit(raw_unit, aliases, default)
            if canonical not in set(aliases.values()):
                raise ValueError(
                    f"{path} contains unsupported {key} unit {raw_unit!r}"
                )
            if raw_unit is not None and str(raw_unit).strip():
                normalized_units[key] = canonical

        raw_report = cls._raw_complex_consistency_report(
            expected_shape=expected,
            frequencies=numeric_axes["frequency"],
            rcs_power=power,
            rcs_phase=phase,
            units=normalized_units,
            extra=extra,
        )
        numerical_issues = [issue for issue in raw_report["issues"] if issue["code"] not in {
            "invalid_raw_complex_preserved_flag", "missing_raw_complex_pair",
        }]
        if numerical_issues:
            first_issue = numerical_issues[0]
            raise ValueError(f"{path} contains {first_issue['message']}")

        return (
            numeric_axes["azimuth"],
            numeric_axes["elevation"],
            numeric_axes["frequency"],
            np.asarray(labels, dtype=str),
            power,
            phase,
            normalized_units,
        )
