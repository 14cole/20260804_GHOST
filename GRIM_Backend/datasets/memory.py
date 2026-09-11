"""Allocation limits and bounded iteration for dense RCS grids."""
from __future__ import annotations

import operator
import os
import zipfile

import numpy as np

from GRIM_Backend.datasets.constants import _DENSE_IMPORT_FALLBACK_LIMIT_BYTES


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
    """Check the planned dense arrays against the import memory limit.

    ``dense_bytes`` describes output arrays; ``resident_bytes`` describes arrays
    retained during allocation. Python container overhead is excluded.
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
