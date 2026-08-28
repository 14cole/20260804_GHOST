"""Dependency-light reader for GRIM's versioned flat RCS interchange table.

The module intentionally depends only on the standard library and NumPy so
the standalone CEM Tools package can reuse the exact same parser on its older
Python/HPC installations.  ``grid_class`` is injected by the caller; this
module does not import Qt or the rest of the GRIM application.
"""

import csv
import ctypes
import math
import os
import tempfile

import numpy as np


FLAT_CSV_SCHEMA = "grim.flat-rcs.v1"

_FREQUENCY_FACTORS = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}
_MAGNITUDE_COLUMNS = (
    "magnitude_power_linear",
    "magnitude_linear",
    "magnitude_db",
    "magnitude_dbsm",
    "magnitude_dbke",
)
_QUANTITY_LOG_UNITS = {
    "sigma_3d": "dBsm",
    "sigma_2d": "dBke",
    "power_ratio": "dB",
}
_V1_REQUIRED_COLUMNS = (
    "grim_csv_schema",
    "azimuth",
    "azimuth_unit",
    "elevation",
    "elevation_unit",
    "frequency",
    "frequency_unit",
    "polarization",
    "rcs_linear_quantity",
    "rcs_log_unit",
    "angular_coordinate_system",
)


def _field_map(fieldnames):
    mapped = {}
    duplicates = []
    for raw_name in fieldnames or ():
        if raw_name is None:
            continue
        key = str(raw_name).strip().lower()
        if not key:
            continue
        if key in mapped:
            duplicates.append(str(raw_name))
        else:
            mapped[key] = raw_name
    if duplicates:
        raise ValueError(
            "duplicate CSV column name(s), ignoring case: "
            + ", ".join(duplicates)
        )
    return mapped


def has_flat_csv_signature(path):
    """Return whether *path* has a deliberate GRIM/CEM flat-table header."""

    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
            reader = csv.reader(stream, delimiter=delimiter)
            header = next(reader, None)
    except (OSError, UnicodeError, csv.Error):
        return False
    try:
        fields = _field_map(header)
    except ValueError:
        return True
    plain_axes = {"azimuth", "elevation", "frequency", "polarization"}
    cem_axes = {
        "azimuth_deg",
        "elevation_deg",
        "frequency_ghz",
        "polarization",
    }
    return bool(
        (plain_axes.issubset(fields) or cem_axes.issubset(fields))
        and any(name in fields for name in _MAGNITUDE_COLUMNS)
    )


def _canonical_frequency_unit(value):
    aliases = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError("unsupported frequency unit {!r}".format(value))
    return aliases[text]


def _canonical_angle_unit(value):
    aliases = {
        "deg": "deg",
        "degree": "deg",
        "degrees": "deg",
        "rad": "rad",
        "radian": "rad",
        "radians": "rad",
    }
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError("unsupported angular unit {!r}; use deg or rad".format(value))
    return aliases[text]


def _canonical_log_unit(value):
    aliases = {"db": "dB", "dbsm": "dBsm", "dbke": "dBke"}
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError("unsupported RCS logarithmic unit {!r}".format(value))
    return aliases[text]


def _canonical_quantity(value):
    aliases = {
        "sigma_3d": "sigma_3d",
        "sigma3d": "sigma_3d",
        "sigma_2d": "sigma_2d",
        "sigma2d": "sigma_2d",
        "power_ratio": "power_ratio",
        "ratio": "power_ratio",
    }
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError(
            "unsupported rcs_linear_quantity {!r}; use sigma_3d, sigma_2d, "
            "or power_ratio".format(value)
        )
    return aliases[text]


def _canonical_phase_wrap(value):
    aliases = {
        "": "-180_180",
        "-180_180": "-180_180",
        "-180-180": "-180_180",
        "pm180": "-180_180",
        "0_360": "0_360",
        "0-360": "0_360",
    }
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError(
            "unsupported phase_wrap {!r}; use -180_180 or 0_360".format(value)
        )
    return aliases[text]


def _legacy_frequency_unit(values):
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    typical = float(np.median(np.abs(finite))) if finite.size else 0.0
    if typical >= 1.0e6:
        unit = "Hz"
    elif typical >= 1.0e3:
        unit = "MHz"
    else:
        unit = "GHz"
    rule = (
        "legacy magnitude heuristic selected {} from median |frequency|={:.9g}; "
        "add frequency_unit to remove this inference"
    ).format(unit, typical)
    return unit, rule


def _available_memory_bytes():
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == "nt":
        try:
            class MemoryStatusEx(ctypes.Structure):
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

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _allocation_budget_bytes():
    declared = str(os.environ.get("GRIM_MAX_CSV_GRID_GB", "") or "").strip()
    if declared:
        try:
            gib = float(declared)
        except ValueError as exc:
            raise ValueError("GRIM_MAX_CSV_GRID_GB must be a positive number") from exc
        if not np.isfinite(gib) or gib <= 0.0:
            raise ValueError("GRIM_MAX_CSV_GRID_GB must be a positive number")
        return int(gib * 1024.0 ** 3), "GRIM_MAX_CSV_GRID_GB"
    available = _available_memory_bytes()
    if available is not None and available > 0:
        return int(available * 0.5), "50% of currently available RAM"
    return 2 * 1024 ** 3, "2 GiB fallback because available RAM is unknown"


def _preflight_dense_grid(axis_lengths, record_count):
    cells = 1
    for length in axis_lengths:
        cells *= int(length)
    float_bytes = np.dtype(np.float64).itemsize
    line_bytes = np.dtype(np.int64).itemsize
    bool_bytes = np.dtype(np.bool_).itemsize
    # The parser owns float64 power/phase plus the dense first-line tracker.
    # Before constructing the grid the tracker is released. RcsGrid's
    # sanitation then has the two input arrays, one output power array, its
    # finite mask, and (at worst) both the boolean-gather and ufunc result used
    # by maximum(). Count that exact conservative live-array peak instead of
    # assuming only one clean copy of power/phase.
    dense_payload = cells * 2 * float_bytes
    parser_peak = dense_payload + cells * line_bytes
    constructor_peak = dense_payload + cells * (
        float_bytes + bool_bytes + float_bytes + float_bytes
    )
    estimated_peak = max(parser_peak, constructor_peak)
    budget, source = _allocation_budget_bytes()
    if estimated_peak > budget:
        raise ValueError(
            "flat CSV axes form a dense grid of {:,} cells ({} x {} x {} x {}), "
            "requiring about {:.2f} GiB peak for power/phase arrays; {} allows "
            "{:.2f} GiB. The file contains only {:,} row(s), so verify that the "
            "axes were intended to form a Cartesian grid. Increase "
            "GRIM_MAX_CSV_GRID_GB only after confirming sufficient RAM."
            .format(
                cells,
                axis_lengths[0],
                axis_lengths[1],
                axis_lengths[2],
                axis_lengths[3],
                estimated_peak / 1024.0 ** 3,
                source,
                budget / 1024.0 ** 3,
                record_count,
            )
        )
    return cells, dense_payload, estimated_peak


def _grid_scalar(value, label):
    if value is None:
        return ""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("flat CSV {} must be scalar".format(label))
    return str(array.reshape(-1)[0].item()).strip()


def _declared_grid_metadata(grid, key):
    """Return one scalar convention declared in either units or extra.

    RcsGrid owns the canonical compatibility rules, including equivalent time
    convention spellings, so use its helper when available. The dependency-
    light fallback still validates scalar shape and refuses two different
    nonblank declarations rather than silently preferring one container.
    """

    declared_helper = getattr(grid, "_declared_scalar_metadata", None)
    if callable(declared_helper):
        return _grid_scalar(declared_helper(key), key)

    declarations = []
    for container_name in ("units", "extra"):
        container = dict(getattr(grid, container_name, {}) or {})
        if key not in container:
            continue
        value = _grid_scalar(container[key], key)
        if value:
            declarations.append(value)
    normalized = {
        " ".join(value.split()).casefold() for value in declarations
    }
    if len(normalized) > 1:
        raise ValueError(
            "dataset contains contradictory {} metadata".format(key)
        )
    return declarations[0] if declarations else ""


def _finite_csv_number(value):
    number = float(value)
    if np.isnan(number):
        return ""
    if np.isneginf(number):
        return "-inf"
    if not np.isfinite(number):
        raise ValueError("flat CSV cannot represent positive infinity")
    return format(number, ".17g")


def _write_flat_csv_direct(
    grid, path, scale="linear", delimiter=",", include_phase=False
):
    """Stream *grid* using ``grim.flat-rcs.v1`` and return *path*.

    ``scale`` mirrors the Plotting export choices: ``linear``, ``db``,
    ``dbsm``, ``dbke``, or ``both``.  Linear output is always stored power,
    never field amplitude.  ``both`` writes power plus the only logarithmic
    column compatible with the declared physical quantity.
    """

    scale = str(scale or "linear").strip().lower()
    if scale not in ("linear", "db", "dbsm", "dbke", "both"):
        raise ValueError("CSV scale must be linear, db, dbsm, dbke, or both")
    if delimiter not in (",", "\t"):
        raise ValueError("flat CSV delimiter must be a comma or tab")

    units = dict(getattr(grid, "units", {}) or {})
    azimuth_unit = _canonical_angle_unit(units.get("azimuth", "deg"))
    elevation_unit = _canonical_angle_unit(units.get("elevation", "deg"))
    frequency_unit = _canonical_frequency_unit(units.get("frequency", "GHz"))
    quantity = (
        _canonical_quantity(grid.linear_quantity())
        if hasattr(grid, "linear_quantity")
        else _canonical_quantity(units.get("rcs_linear_quantity", "sigma_3d"))
    )
    log_unit = (
        _canonical_log_unit(grid.default_log_unit())
        if hasattr(grid, "default_log_unit")
        else _canonical_log_unit(units.get("rcs_log_unit", "dBsm"))
    )
    expected_log = _QUANTITY_LOG_UNITS[quantity]
    if log_unit != expected_log:
        raise ValueError(
            "inconsistent RCS metadata: {} requires rcs_log_unit={}, not {}"
            .format(quantity, expected_log, log_unit)
        )
    required_quantity = {"db": "power_ratio", "dbsm": "sigma_3d", "dbke": "sigma_2d"}
    if scale in required_quantity and quantity != required_quantity[scale]:
        if quantity == "sigma_3d" and scale == "dbke":
            raise ValueError(
                "sigma_3d datasets cannot be labeled dBke; convert to sigma_2d first"
            )
        if quantity == "sigma_2d" and scale == "dbsm":
            raise ValueError(
                "sigma_2d datasets cannot be labeled dBsm; convert to sigma_3d first"
            )
        if quantity == "power_ratio":
            raise ValueError(
                "dimensionless power ratios can be exported only as Linear or dB"
            )
        raise ValueError(
            "{} export requires {}, but the dataset contains {}"
            .format(scale, required_quantity[scale], quantity)
        )

    angular_coordinate_system = (
        str(grid.angular_coordinate_system()).strip()
        if hasattr(grid, "angular_coordinate_system")
        else str(units.get("angular_coordinate_system", "conic")).strip()
    )
    if not angular_coordinate_system:
        angular_coordinate_system = "conic"
    gc_convention = (
        str(grid.great_circle_coordinate_convention()).strip()
        if angular_coordinate_system == "great_circle"
        and hasattr(grid, "great_circle_coordinate_convention")
        else ""
    )
    if hasattr(grid, "angular_frame_orientation_deg"):
        angular_roll_deg, angular_tilt_deg = grid.angular_frame_orientation_deg()
    else:
        angular_roll_deg = units.get("angular_roll_deg", 0.0)
        angular_tilt_deg = units.get("angular_tilt_deg", 0.0)
    for label, value in (
        ("angular_roll_deg", angular_roll_deg),
        ("angular_tilt_deg", angular_tilt_deg),
    ):
        if not np.isfinite(float(value)):
            raise ValueError("{} must be finite".format(label))
    polarization_basis = _declared_grid_metadata(grid, "polarization_basis")
    time_convention = _declared_grid_metadata(grid, "time_convention")
    phase_reference = _declared_grid_metadata(grid, "phase_reference")
    phase_wrap = _canonical_phase_wrap((getattr(grid, "units", {}) or {}).get(
        "phase_wrap", "-180_180"
    ))

    magnitude_columns = []
    if scale in ("linear", "both"):
        magnitude_columns.append("magnitude_power_linear")
    if scale == "db":
        magnitude_columns.append("magnitude_db")
    elif scale == "dbsm":
        magnitude_columns.append("magnitude_dbsm")
    elif scale == "dbke":
        magnitude_columns.append("magnitude_dbke")
    elif scale == "both":
        magnitude_columns.append({
            "sigma_3d": "magnitude_dbsm",
            "sigma_2d": "magnitude_dbke",
            "power_ratio": "magnitude_db",
        }[quantity])

    header = [
        "grim_csv_schema", "azimuth", "azimuth_unit", "elevation",
        "elevation_unit", "frequency", "frequency_unit", "polarization",
        "rcs_linear_quantity", "rcs_log_unit", "angular_coordinate_system",
        "great_circle_coordinate_convention", "angular_roll_deg",
        "angular_tilt_deg", "polarization_basis", "time_convention",
        "phase_reference", "phase_wrap",
    ] + magnitude_columns
    if include_phase:
        header.append("phase_deg")

    power_array = np.asarray(grid.rcs_power)
    phase_array = np.asarray(grid.rcs_phase)
    expected_shape = (
        len(grid.azimuths), len(grid.elevations), len(grid.frequencies),
        len(grid.polarizations),
    )
    if power_array.shape != expected_shape or phase_array.shape != expected_shape:
        raise ValueError("dataset power/phase arrays do not match its four axes")

    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(header)
        for azimuth_index, azimuth in enumerate(grid.azimuths):
            for elevation_index, elevation in enumerate(grid.elevations):
                for frequency_index, frequency in enumerate(grid.frequencies):
                    frequency_hz = float(frequency) * _FREQUENCY_FACTORS[frequency_unit]
                    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
                        raise ValueError("flat CSV frequencies must be positive and finite")
                    for polarization_index, polarization in enumerate(grid.polarizations):
                        index = (
                            azimuth_index, elevation_index, frequency_index,
                            polarization_index,
                        )
                        power = float(power_array[index])
                        row = [
                            FLAT_CSV_SCHEMA,
                            _finite_csv_number(azimuth), azimuth_unit,
                            _finite_csv_number(elevation), elevation_unit,
                            _finite_csv_number(frequency), frequency_unit,
                            str(polarization), quantity, log_unit,
                            angular_coordinate_system, gc_convention,
                            _finite_csv_number(angular_roll_deg),
                            _finite_csv_number(angular_tilt_deg),
                            polarization_basis, time_convention, phase_reference,
                            phase_wrap,
                        ]
                        for column in magnitude_columns:
                            if not np.isfinite(power):
                                row.append("")
                            elif column == "magnitude_power_linear":
                                row.append(_finite_csv_number(power))
                            elif power == 0.0:
                                row.append("-inf")
                            elif column == "magnitude_dbke":
                                wavelength = 299792458.0 / frequency_hz
                                row.append(_finite_csv_number(
                                    10.0 * np.log10(power * 2.0 * math.pi / wavelength)
                                ))
                            else:
                                row.append(_finite_csv_number(10.0 * np.log10(power)))
                        if include_phase:
                            phase_rad = float(phase_array[index])
                            if phase_wrap == "0_360":
                                phase_deg = np.mod(np.degrees(phase_rad), 360.0)
                            else:
                                phase_deg = (
                                    np.mod(np.degrees(phase_rad) + 180.0, 360.0)
                                    - 180.0
                                )
                            row.append(
                                _finite_csv_number(phase_deg)
                                if np.isfinite(phase_rad) else ""
                            )
                        writer.writerow(row)
    return path


def write_flat_csv(grid, path, scale="linear", delimiter=",", include_phase=False):
    """Atomically publish a versioned flat-RCS table.

    Validation, formatting, or I/O failures leave an existing destination
    untouched.  The streaming implementation still retains only one CSV row
    beyond the dataset arrays.
    """

    requested_path = os.fspath(path)
    output_path = os.path.abspath(requested_path)
    directory = os.path.dirname(output_path) or os.curdir
    fd, stage_path = tempfile.mkstemp(
        prefix=".grim-flat-csv-", suffix=".staging", dir=directory
    )
    os.close(fd)
    try:
        _write_flat_csv_direct(
            grid,
            stage_path,
            scale=scale,
            delimiter=delimiter,
            include_phase=include_phase,
        )
        os.replace(stage_path, output_path)
    finally:
        if os.path.lexists(stage_path):
            try:
                os.unlink(stage_path)
            except OSError:
                pass
    return requested_path


def _one_value(values, label, default=None, required=False):
    if len(values) > 1:
        rendered = ", ".join(repr(value) for value in sorted(values, key=str))
        raise ValueError("one grid cannot contain multiple {} values: {}".format(label, rendered))
    value = next(iter(values)) if values else default
    if required and (value is None or str(value).strip() == ""):
        raise ValueError("{} is required and cannot be blank".format(label))
    return value


def _parse_number(text, label, line_no, strict):
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("line {}: invalid {} ({})".format(line_no, label, exc)) from exc
    if np.isnan(value):
        if strict:
            raise ValueError("line {}: {} cannot be NaN; leave the cell blank".format(line_no, label))
        return None
    if np.isposinf(value):
        raise ValueError("line {}: {} cannot be +infinity".format(line_no, label))
    return value


def _power_candidates(record, mode, quantity, frequency_unit, c0):
    line_no = record["line_no"]
    values = record["magnitudes"]
    strict = mode == "v1"
    candidates = []
    for column, text in values.items():
        value = _parse_number(text, column, line_no, strict)
        if value is None:
            continue
        if column in ("magnitude_power_linear", "magnitude_linear"):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "line {}: {} must be finite and >= 0".format(line_no, column)
                )
            if column == "magnitude_linear" and mode == "legacy_cem_amplitude":
                power = value * value
                semantic = "legacy CEM field amplitude squared"
            else:
                power = value
                semantic = "linear power"
        else:
            if np.isneginf(value):
                power = 0.0
            elif not np.isfinite(value):
                raise ValueError(
                    "line {}: {} must be finite or -inf".format(line_no, column)
                )
            else:
                ratio = 10.0 ** (value / 10.0)
                if column == "magnitude_dbke":
                    frequency_hz = record["frequency"] * _FREQUENCY_FACTORS[frequency_unit]
                    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
                        raise ValueError(
                            "line {}: positive frequency is required to convert dBke"
                            .format(line_no)
                        )
                    power = (c0 / (2.0 * math.pi * frequency_hz)) * ratio
                else:
                    power = ratio
            semantic = column
        if not np.isfinite(power) or power < 0.0:
            raise ValueError(
                "line {}: {} converts to an invalid linear power".format(line_no, column)
            )
        candidates.append((column, float(power), semantic))

    if not candidates:
        return float("nan")
    reference_column, reference, _semantic = candidates[0]
    for column, candidate, _semantic in candidates[1:]:
        if not np.isclose(reference, candidate, rtol=5.0e-6, atol=0.0):
            raise ValueError(
                "line {}: redundant magnitude columns conflict: {}={} W-like "
                "power but {}={} W-like power".format(
                    line_no, reference_column, reference, column, candidate
                )
            )
    return reference


def load_flat_csv(path, grid_class, canonical_angular_coordinate_system, c0=299792458.0):
    """Load versioned and deliberate legacy flat RCS tables into *grid_class*."""

    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("missing CSV header row")
        fields = _field_map(reader.fieldnames)

        if "grim_csv_schema" in fields:
            mode = "v1"
            missing = [name for name in _V1_REQUIRED_COLUMNS if name not in fields]
            if missing:
                raise ValueError(
                    "{} is missing required column(s): {}".format(
                        FLAT_CSV_SCHEMA, ", ".join(missing)
                    )
                )
            if "magnitude_linear" in fields:
                raise ValueError(
                    "{} does not permit ambiguous magnitude_linear; use "
                    "magnitude_power_linear".format(FLAT_CSV_SCHEMA)
                )
            axis_fields = {
                "azimuth": "azimuth",
                "elevation": "elevation",
                "frequency": "frequency",
            }
        elif all(name in fields for name in (
            "azimuth_deg", "elevation_deg", "frequency_ghz", "polarization"
        )):
            mode = "legacy_cem_amplitude"
            axis_fields = {
                "azimuth": "azimuth_deg",
                "elevation": "elevation_deg",
                "frequency": "frequency_ghz",
            }
        else:
            mode = "legacy_grim_power"
            required = ("azimuth", "elevation", "frequency", "polarization")
            missing = [name for name in required if name not in fields]
            if missing:
                raise ValueError("missing required column(s): {}".format(", ".join(missing)))
            axis_fields = {
                "azimuth": "azimuth",
                "elevation": "elevation",
                "frequency": "frequency",
            }

        magnitude_columns = [name for name in _MAGNITUDE_COLUMNS if name in fields]
        if not magnitude_columns:
            raise ValueError("missing magnitude column")

        def cell(row, key):
            raw = row.get(fields[key], "")
            return str(raw).strip() if raw is not None else ""

        metadata_sets = {
            "schema": set(),
            "azimuth_unit": set(),
            "elevation_unit": set(),
            "frequency_unit": set(),
            "rcs_linear_quantity": set(),
            "rcs_log_unit": set(),
            "angular_coordinate_system": set(),
            "great_circle_coordinate_convention": set(),
            "angular_roll_deg": set(),
            "angular_tilt_deg": set(),
            "polarization_basis": set(),
            "time_convention": set(),
            "phase_reference": set(),
            "phase_wrap": set(),
        }
        record_count = 0
        azimuth_values = set()
        elevation_values = set()
        frequency_values = set()
        polarization_order = []

        for line_no, row in enumerate(reader, start=2):
            if None in row and any(str(value or "").strip() for value in row[None]):
                raise ValueError("line {}: more values than header columns".format(line_no))
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                azimuth = float(cell(row, axis_fields["azimuth"]))
                elevation = float(cell(row, axis_fields["elevation"]))
                frequency = float(cell(row, axis_fields["frequency"]))
            except ValueError as exc:
                raise ValueError("line {}: invalid axis value ({})".format(line_no, exc)) from exc
            if not np.all(np.isfinite([azimuth, elevation, frequency])):
                raise ValueError("line {}: axis values must be finite".format(line_no))
            polarization = cell(row, "polarization")
            if not polarization:
                raise ValueError("line {}: polarization is blank".format(line_no))

            for name in metadata_sets:
                if name in fields:
                    metadata_sets[name].add(cell(row, name))
            if mode == "v1":
                schema_value = cell(row, "grim_csv_schema")
                metadata_sets["schema"].add(schema_value)
                if schema_value != FLAT_CSV_SCHEMA:
                    raise ValueError(
                        "line {}: unsupported grim_csv_schema {!r}; expected {!r}"
                        .format(line_no, schema_value, FLAT_CSV_SCHEMA)
                    )
                for name in (
                    "azimuth_unit", "elevation_unit", "frequency_unit",
                    "rcs_linear_quantity", "rcs_log_unit",
                    "angular_coordinate_system",
                ):
                    if not cell(row, name):
                        raise ValueError("line {}: {} is blank".format(line_no, name))

            azimuth_values.add(azimuth)
            elevation_values.add(elevation)
            frequency_values.add(frequency)
            if polarization not in polarization_order:
                polarization_order.append(polarization)
            record_count += 1

    if not record_count:
        raise ValueError("CSV contains no data rows")

    inferred_frequency = False
    inference_rule = ""
    if mode == "legacy_cem_amplitude":
        azimuth_unit = "deg"
        elevation_unit = "deg"
        frequency_unit = "GHz"
        for name, expected, canonicalizer in (
            ("azimuth_unit", "deg", _canonical_angle_unit),
            ("elevation_unit", "deg", _canonical_angle_unit),
            ("frequency_unit", "GHz", _canonical_frequency_unit),
        ):
            declared = _one_value(metadata_sets[name], name, default="")
            if str(declared or "").strip() and canonicalizer(declared) != expected:
                raise ValueError(
                    "legacy CEM column suffix declares {} but {} says {}"
                    .format(expected, name, declared)
                )
    else:
        azimuth_text = _one_value(
            metadata_sets["azimuth_unit"], "azimuth units",
            default="deg", required=mode == "v1",
        )
        elevation_text = _one_value(
            metadata_sets["elevation_unit"], "elevation units",
            default="deg", required=mode == "v1",
        )
        azimuth_unit = _canonical_angle_unit(azimuth_text)
        elevation_unit = _canonical_angle_unit(elevation_text)
        frequency_text = _one_value(
            metadata_sets["frequency_unit"], "frequency units",
            default=None, required=mode == "v1",
        )
        if frequency_text is None or not str(frequency_text).strip():
            frequency_unit, inference_rule = _legacy_frequency_unit(frequency_values)
            inferred_frequency = True
        else:
            frequency_unit = _canonical_frequency_unit(frequency_text)

    log_text = _one_value(
        metadata_sets["rcs_log_unit"], "RCS logarithmic units",
        default=None, required=mode == "v1",
    )
    quantity_text = _one_value(
        metadata_sets["rcs_linear_quantity"], "RCS linear quantities",
        default=None, required=mode == "v1",
    )
    if log_text is None or not str(log_text).strip():
        db_columns = [
            name for name in ("magnitude_db", "magnitude_dbsm", "magnitude_dbke")
            if name in magnitude_columns
        ]
        if len(db_columns) > 1:
            raise ValueError(
                "legacy CSV mixes incompatible logarithmic magnitude columns: {}"
                .format(", ".join(db_columns))
            )
        log_unit = {
            "magnitude_db": "dB",
            "magnitude_dbsm": "dBsm",
            "magnitude_dbke": "dBke",
        }.get(db_columns[0] if db_columns else "", "dBsm")
    else:
        log_unit = _canonical_log_unit(log_text)
    quantity = (
        _canonical_quantity(quantity_text)
        if quantity_text is not None and str(quantity_text).strip()
        else {"dB": "power_ratio", "dBke": "sigma_2d", "dBsm": "sigma_3d"}[log_unit]
    )
    expected_log = _QUANTITY_LOG_UNITS[quantity]
    if log_unit != expected_log:
        raise ValueError(
            "inconsistent physical magnitude metadata: {} requires "
            "rcs_log_unit={}, not {}".format(quantity, expected_log, log_unit)
        )
    required_quantity = {
        "magnitude_db": "power_ratio",
        "magnitude_dbsm": "sigma_3d",
        "magnitude_dbke": "sigma_2d",
    }
    for column, column_quantity in required_quantity.items():
        if column in magnitude_columns and quantity != column_quantity:
            raise ValueError(
                "{} is incompatible with rcs_linear_quantity={}; use the "
                "matching logarithmic column or magnitude_power_linear"
                .format(column, quantity)
            )

    angular_text = _one_value(
        metadata_sets["angular_coordinate_system"],
        "angular coordinate systems", default="conic", required=mode == "v1",
    )
    angular_coordinate_system = canonical_angular_coordinate_system(angular_text)
    gc_convention = _one_value(
        metadata_sets["great_circle_coordinate_convention"],
        "great-circle coordinate conventions", default="",
    )
    if angular_coordinate_system == "great_circle" and not str(gc_convention or "").strip():
        gc_convention = "legacy_ptm_unspecified"

    def numeric_metadata(name, default):
        text = _one_value(metadata_sets[name], name, default=str(default))
        if text is None or str(text).strip() == "":
            return float(default)
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError("invalid {} ({})".format(name, exc)) from exc
        if not np.isfinite(value):
            raise ValueError("{} must be finite".format(name))
        return value

    angular_roll_deg = numeric_metadata("angular_roll_deg", 0.0)
    angular_tilt_deg = numeric_metadata("angular_tilt_deg", 0.0)
    polarization_basis = _one_value(
        metadata_sets["polarization_basis"], "polarization bases", default=""
    )
    time_convention = _one_value(
        metadata_sets["time_convention"], "time conventions", default=""
    )
    phase_reference = _one_value(
        metadata_sets["phase_reference"], "phase references", default=""
    )
    phase_wrap = _canonical_phase_wrap(_one_value(
        metadata_sets["phase_wrap"], "phase wrap declarations", default="-180_180"
    ))

    axis_lengths = (
        len(azimuth_values),
        len(elevation_values),
        len(frequency_values),
        len(polarization_order),
    )
    _preflight_dense_grid(axis_lengths, record_count)

    azimuth_axis = np.asarray(sorted(azimuth_values), dtype=float)
    elevation_axis = np.asarray(sorted(elevation_values), dtype=float)
    frequency_axis = np.asarray(sorted(frequency_values), dtype=float)
    polarization_axis = np.asarray(polarization_order)
    shape = tuple(axis_lengths)
    power = np.full(shape, np.nan, dtype=np.float64)
    phase = np.full(shape, np.nan, dtype=np.float64)
    axis_maps = (
        dict((value, index) for index, value in enumerate(azimuth_axis)),
        dict((value, index) for index, value in enumerate(elevation_axis)),
        dict((value, index) for index, value in enumerate(frequency_axis)),
        dict((value, index) for index, value in enumerate(polarization_axis)),
    )
    # Re-read rows after the axis-product preflight. A dense int64 array stores
    # the first CSV line for every populated Cartesian cell (zero means
    # unseen). Its memory is bounded by the already-preflighted grid rather
    # than growing as a Python tuple/dict entry for every input record.
    first_line = np.zeros(shape, dtype=np.int64)
    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        second_fields = _field_map(reader.fieldnames)
        if second_fields != fields:
            raise ValueError("CSV header changed while the file was being loaded")
        for line_no, row in enumerate(reader, start=2):
            if None in row and any(str(value or "").strip() for value in row[None]):
                raise ValueError("line {}: more values than header columns".format(line_no))
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                record = {
                    "azimuth": float(cell(row, axis_fields["azimuth"])),
                    "elevation": float(cell(row, axis_fields["elevation"])),
                    "frequency": float(cell(row, axis_fields["frequency"])),
                    "polarization": cell(row, "polarization"),
                    "magnitudes": dict(
                        (name, cell(row, name)) for name in magnitude_columns
                    ),
                    "phase_deg": (
                        cell(row, "phase_deg") if "phase_deg" in fields else ""
                    ),
                    "line_no": line_no,
                }
            except ValueError as exc:
                raise ValueError(
                    "line {}: invalid axis value ({})".format(line_no, exc)
                ) from exc
            if not np.all(np.isfinite([
                record["azimuth"], record["elevation"], record["frequency"]
            ])):
                raise ValueError("line {}: axis values must be finite".format(line_no))
            if not record["polarization"]:
                raise ValueError("line {}: polarization is blank".format(line_no))

            sample_power = _power_candidates(
                record, mode, quantity, frequency_unit, c0
            )
            phase_value = float("nan")
            if record["phase_deg"]:
                parsed_phase = _parse_number(
                    record["phase_deg"], "phase_deg", record["line_no"], mode == "v1"
                )
                if parsed_phase is not None:
                    if not np.isfinite(parsed_phase):
                        raise ValueError(
                            "line {}: phase_deg must be finite".format(
                                record["line_no"]
                            )
                        )
                    phase_value = float(np.deg2rad(parsed_phase))
            try:
                index = (
                    axis_maps[0][record["azimuth"]],
                    axis_maps[1][record["elevation"]],
                    axis_maps[2][record["frequency"]],
                    axis_maps[3][record["polarization"]],
                )
            except KeyError as exc:
                raise ValueError(
                    "CSV axes changed while the file was being loaded"
                ) from exc
            previous_line = int(first_line[index])
            if previous_line:
                previous_power = float(power[index])
                previous_phase = float(phase[index])
                same_power = bool(np.isclose(
                    sample_power, previous_power,
                    rtol=1.0e-12, atol=0.0, equal_nan=True,
                ))
                zero_power = same_power and sample_power == 0.0
                same_phase = zero_power or (
                    (np.isnan(phase_value) and np.isnan(previous_phase))
                    or (
                        np.isfinite(phase_value)
                        and np.isfinite(previous_phase)
                        and abs(np.angle(np.exp(
                            1j * (phase_value - previous_phase)
                        ))) <= 1.0e-12
                    )
                )
                if not (same_power and same_phase):
                    raise ValueError(
                        "line {}: conflicting duplicate CSV sample; first "
                        "defined on line {}".format(line_no, previous_line)
                    )
                continue
            first_line[index] = line_no
            power[index] = sample_power
            phase[index] = phase_value

    if not np.isfinite(power).any():
        raise ValueError("CSV contains no finite magnitude values")
    # Duplicate diagnostics are complete; release this dense tracker before
    # grid_class sanitizes/copies power and phase.
    del first_line

    units = {
        "azimuth": azimuth_unit,
        "elevation": elevation_unit,
        "frequency": frequency_unit,
        "rcs_log_unit": log_unit,
        "rcs_linear_quantity": quantity,
        "angular_coordinate_system": angular_coordinate_system,
        "angular_roll_deg": angular_roll_deg,
        "angular_tilt_deg": angular_tilt_deg,
    }
    if angular_coordinate_system == "great_circle":
        units["great_circle_coordinate_convention"] = str(gc_convention)
    if str(polarization_basis or "").strip():
        units["polarization_basis"] = str(polarization_basis).strip()
    if str(time_convention or "").strip():
        units["time_convention"] = str(time_convention).strip()
    units["phase_wrap"] = phase_wrap

    extra = {
        "flat_csv_schema": FLAT_CSV_SCHEMA if mode == "v1" else mode,
        "source_format": "GRIM flat RCS CSV" if mode == "v1" else mode,
    }
    if str(phase_reference or "").strip():
        extra["phase_reference"] = str(phase_reference).strip()
    if inferred_frequency:
        extra["frequency_unit_inferred"] = True
        extra["frequency_unit_inference"] = inference_rule

    history = "Loaded flat CSV ({}): {}".format(
        FLAT_CSV_SCHEMA if mode == "v1" else mode.replace("_", " "), path
    )
    if inferred_frequency:
        history += "\nWARNING: " + inference_rule
    return grid_class(
        azimuth_axis,
        elevation_axis,
        frequency_axis,
        polarization_axis,
        rcs_power=power,
        rcs_phase=phase,
        source_path=str(path),
        history=history,
        units=units,
        extra=extra,
    )
