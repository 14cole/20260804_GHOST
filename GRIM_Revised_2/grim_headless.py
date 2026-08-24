"""Qt-free GRIM loading, folder operations, and command-line entry point."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from grim_dataset import C0, RcsGrid, canonical_angular_coordinate_system
from plot_modes.isar_mode import form_isar


SUPPORTED_EXTENSIONS = (
    ".grim",
    ".csv",
    ".cst_data",
    ".txt",
    ".out",
    ".pio",
    ".cmplx_di",
    ".ptm",
    ".ss",
)
_FREQUENCY_FACTORS = {"Hz": 1.0, "kHz": 1.0e3, "MHz": 1.0e6, "GHz": 1.0e9}


def is_supported_path(path: str) -> bool:
    return str(path).lower().endswith(SUPPORTED_EXTENSIONS)


def _frequency_unit(value: object, values=()) -> str:
    aliases = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}
    text = str(value or "").strip().lower()
    if text:
        if text not in aliases:
            raise ValueError(f"unsupported frequency unit {value!r}")
        return aliases[text]
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    typical = float(np.median(np.abs(finite))) if finite.size else 0.0
    return "Hz" if typical >= 1.0e6 else "MHz" if typical >= 1.0e3 else "GHz"


def _log_unit(value: object, default="dBsm") -> str:
    aliases = {"db": "dB", "dbsm": "dBsm", "dbke": "dBke"}
    text = str(value or default).strip().lower()
    if text not in aliases:
        raise ValueError(f"unsupported logarithmic unit {value!r}")
    return aliases[text]


def load_flat_csv(path: str) -> RcsGrid:
    """Load the flat CSV schema written by GRIM's dataset exporter."""
    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("missing CSV header row")
        fields = {str(name).strip().lower(): name for name in reader.fieldnames if name}
        required = ("azimuth", "elevation", "frequency", "polarization")
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(missing)}")
        magnitude_keys = [
            key for key in ("magnitude_linear", "magnitude_db", "magnitude_dbsm", "magnitude_dbke")
            if key in fields
        ]
        if not magnitude_keys:
            raise ValueError("missing magnitude column")

        def cell(row, key):
            raw = row.get(fields[key], "")
            return str(raw).strip() if raw is not None else ""

        records = []
        frequency_units = set()
        log_units = set()
        angular_coordinate_systems = set()
        gc_coordinate_conventions = set()
        angular_rolls = set()
        angular_tilts = set()
        pol_order = []
        for line_no, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                az = float(cell(row, "azimuth"))
                el = float(cell(row, "elevation"))
                freq = float(cell(row, "frequency"))
            except ValueError as exc:
                raise ValueError(f"line {line_no}: invalid axis value ({exc})") from exc
            if not np.all(np.isfinite([az, el, freq])):
                raise ValueError(f"line {line_no}: axis values must be finite")
            pol = cell(row, "polarization")
            if not pol:
                raise ValueError(f"line {line_no}: polarization is blank")
            if pol not in pol_order:
                pol_order.append(pol)
            freq_unit_text = cell(row, "frequency_unit") if "frequency_unit" in fields else ""
            if freq_unit_text:
                frequency_units.add(_frequency_unit(freq_unit_text))
            log_text = cell(row, "rcs_log_unit") if "rcs_log_unit" in fields else ""
            if log_text:
                log_units.add(_log_unit(log_text))
            angular_text = (
                cell(row, "angular_coordinate_system")
                if "angular_coordinate_system" in fields else ""
            )
            if "angular_coordinate_system" in fields and not angular_text:
                raise ValueError(
                    f"line {line_no}: angular_coordinate_system is blank"
                )
            if angular_text:
                angular_coordinate_systems.add(
                    canonical_angular_coordinate_system(angular_text)
                )
                if (
                    canonical_angular_coordinate_system(angular_text)
                    == "great_circle"
                ):
                    convention_text = (
                        cell(row, "great_circle_coordinate_convention")
                        if "great_circle_coordinate_convention" in fields
                        else ""
                    )
                    gc_coordinate_conventions.add(
                        convention_text or "legacy_ptm_unspecified"
                    )
            for key, target in (
                ("angular_roll_deg", angular_rolls),
                ("angular_tilt_deg", angular_tilts),
            ):
                if key not in fields:
                    continue
                value_text = cell(row, key)
                if not value_text:
                    raise ValueError(f"line {line_no}: {key} is blank")
                try:
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_no}: invalid {key} ({exc})"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"line {line_no}: {key} must be finite")
                target.add(value)

            linear = None
            if "magnitude_linear" in fields and cell(row, "magnitude_linear"):
                linear = float(cell(row, "magnitude_linear"))
            if linear is None and "magnitude_dbsm" in fields and cell(row, "magnitude_dbsm"):
                linear = 10.0 ** (float(cell(row, "magnitude_dbsm")) / 10.0)
            if linear is None and "magnitude_db" in fields and cell(row, "magnitude_db"):
                linear = 10.0 ** (float(cell(row, "magnitude_db")) / 10.0)
            dbke = None
            if linear is None and "magnitude_dbke" in fields and cell(row, "magnitude_dbke"):
                dbke = float(cell(row, "magnitude_dbke"))
            phase = np.nan
            if "phase_deg" in fields and cell(row, "phase_deg"):
                phase = np.deg2rad(float(cell(row, "phase_deg")))
            records.append((az, el, freq, pol, linear, dbke, phase))

    if not records:
        raise ValueError("CSV contains no data rows")
    if (
        len(frequency_units) > 1
        or len(log_units) > 1
        or len(angular_coordinate_systems) > 1
        or len(gc_coordinate_conventions) > 1
        or len(angular_rolls) > 1
        or len(angular_tilts) > 1
    ):
        raise ValueError(
            "one grid cannot contain multiple frequency units, RCS units, or "
            "angular coordinate systems/frame orientations"
        )
    freq_unit = next(iter(frequency_units)) if frequency_units else _frequency_unit("", [r[2] for r in records])
    log_unit = next(iter(log_units)) if log_units else (
        "dB" if "magnitude_db" in magnitude_keys
        else "dBke" if "magnitude_dbke" in magnitude_keys and "magnitude_dbsm" not in magnitude_keys
        else "dBsm"
    )
    angular_coordinate_system = (
        next(iter(angular_coordinate_systems))
        if angular_coordinate_systems else "conic"
    )
    gc_coordinate_convention = (
        next(iter(gc_coordinate_conventions))
        if gc_coordinate_conventions else "legacy_ptm_unspecified"
    )
    angular_roll_deg = next(iter(angular_rolls)) if angular_rolls else 0.0
    angular_tilt_deg = next(iter(angular_tilts)) if angular_tilts else 0.0
    az_axis = np.asarray(sorted({r[0] for r in records}), dtype=float)
    el_axis = np.asarray(sorted({r[1] for r in records}), dtype=float)
    freq_axis = np.asarray(sorted({r[2] for r in records}), dtype=float)
    pol_axis = np.asarray(pol_order)
    shape = (len(az_axis), len(el_axis), len(freq_axis), len(pol_axis))
    power = np.full(shape, np.nan, dtype=np.float64)
    phase = np.full(shape, np.nan, dtype=np.float64)
    ai = {v: i for i, v in enumerate(az_axis)}
    ei = {v: i for i, v in enumerate(el_axis)}
    fi = {v: i for i, v in enumerate(freq_axis)}
    pi = {v: i for i, v in enumerate(pol_axis)}
    for az, el, freq, pol, linear, dbke, phase_value in records:
        if linear is None and dbke is not None:
            freq_hz = freq * _FREQUENCY_FACTORS[freq_unit]
            linear = (C0 / (2.0 * np.pi * freq_hz)) * 10.0 ** (dbke / 10.0)
        idx = (ai[az], ei[el], fi[freq], pi[pol])
        power[idx] = np.nan if linear is None else max(float(linear), 0.0)
        phase[idx] = phase_value
    if not np.isfinite(power).any():
        raise ValueError("CSV contains no finite magnitude values")
    quantity = "power_ratio" if log_unit == "dB" else "sigma_2d" if log_unit == "dBke" else "sigma_3d"
    units = {
        "azimuth": "deg", "elevation": "deg", "frequency": freq_unit,
        "rcs_log_unit": log_unit, "rcs_linear_quantity": quantity,
        "angular_coordinate_system": angular_coordinate_system,
        "angular_roll_deg": angular_roll_deg,
        "angular_tilt_deg": angular_tilt_deg,
    }
    if angular_coordinate_system == "great_circle":
        units["great_circle_coordinate_convention"] = gc_coordinate_convention
    return RcsGrid(
        az_axis, el_axis, freq_axis, pol_axis,
        rcs_power=power, rcs_phase=phase, source_path=str(path),
        history=f"Loaded flat CSV: {path}",
        units=units,
    )


def read_CST(path: str) -> RcsGrid:
    """Read a supported CST far-field table.

    This is the deliberately named public entry point.  ``RcsGrid.read_CST``
    recognizes both CST's wide theta/phi CSV export and the row-oriented
    ``.cst_data`` schema used by the team's MATLAB workflow.  Native GRIM flat
    CSV remains a separate format handled by :func:`load_flat_csv`.
    """
    return RcsGrid.read_CST(path)


def read_SENTRi(path: str) -> RcsGrid:
    """Read a CREATE-RF SENTRi RCS table with its vendor conventions."""

    return RcsGrid.read_SENTRi(path)


def load_dataset(path: str, *, allow_legacy_pickle=False) -> RcsGrid:
    """Load any GRIM-supported dataset without importing Qt."""
    path = str(path)
    lower = path.lower()
    if lower.endswith(".grim"):
        return RcsGrid.load(path, allow_legacy_pickle=allow_legacy_pickle)
    if lower.endswith(".out"):
        return RcsGrid.load_out(path)
    if lower.endswith(".ss"):
        return RcsGrid.load_ss(path)
    if lower.endswith(".ptm"):
        return RcsGrid.load_ptm(path)
    if lower.endswith((".pio", ".cmplx_di")):
        return RcsGrid.load_pio(path)
    if lower.endswith(".cst_data"):
        return read_CST(path)
    if lower.endswith((".csv", ".txt")) and RcsGrid.has_SENTRi_signature(path):
        # A recognized vendor header commits the file to the SENTRi parser.
        # Propagate corrupt-data errors instead of falling through to a loose
        # legacy numeric reader that could reinterpret the same row.
        return read_SENTRi(path)
    loaders = (
        (load_flat_csv, read_CST)
        if lower.endswith(".csv")
        else (
            RcsGrid.load_theta_phi_txt,
            load_flat_csv,
            read_CST,
        )
    )
    errors = []
    for loader in loaders:
        try:
            return loader(path)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise ValueError("; ".join(errors))


def combine_datasets(grids, operation: str, *, overlap="error", max_output_bytes=None):
    grids = list(grids)
    if not grids:
        raise ValueError("at least one dataset is required")
    operation = str(operation).strip().lower().replace("_", "-")
    if operation == "join":
        return RcsGrid.join_many(
            *grids, overlap=overlap, max_output_bytes=max_output_bytes
        )
    result = grids[0]
    for grid in grids[1:]:
        if operation == "coherent-add":
            result = result.coherent_add(grid)
        elif operation == "incoherent-add":
            result = result.incoherent_add(grid)
        else:
            raise ValueError("operation must be join, coherent-add, or incoherent-add")
    return result


def load_folder(
    folder: str,
    *,
    pattern="*",
    recursive=False,
    operation="join",
    workers=1,
    overlap="error",
    max_output_bytes=None,
):
    """Load matching files and combine them in deterministic pathname order."""
    search = os.path.join(str(folder), "**", pattern) if recursive else os.path.join(str(folder), pattern)
    paths = [path for path in sorted(glob.glob(search, recursive=recursive)) if os.path.isfile(path) and is_supported_path(path)]
    if not paths:
        raise ValueError(f"no supported datasets matched {search!r}")
    workers = max(1, int(workers))
    if workers == 1:
        grids = [load_dataset(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            grids = list(pool.map(load_dataset, paths))
    return combine_datasets(
        grids, operation, overlap=overlap, max_output_bytes=max_output_bytes
    )


def _parser():
    parser = argparse.ArgumentParser(description="Headless GRIM dataset operations")
    parser.add_argument("inputs", nargs="*", help="input files")
    parser.add_argument("-o", "--output", help="output .grim path")
    parser.add_argument("--folder", help="load a folder instead of explicit inputs")
    parser.add_argument("--pattern", default="*", help="folder glob pattern")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--operation", choices=("join", "coherent-add", "incoherent-add"), default="join")
    parser.add_argument("--overlap", choices=("error", "first", "last"), default="error")
    parser.add_argument("--max-gib", type=float, default=None, help="maximum dense output allocation")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    limit = None if args.max_gib is None else int(args.max_gib * 1024**3)
    if args.folder:
        result = load_folder(
            args.folder, pattern=args.pattern, recursive=args.recursive,
            operation=args.operation, workers=args.workers, overlap=args.overlap,
            max_output_bytes=limit,
        )
    else:
        if not args.inputs:
            raise SystemExit("provide input files or --folder")
        result = combine_datasets(
            [load_dataset(path) for path in args.inputs], args.operation,
            overlap=args.overlap, max_output_bytes=limit,
        )
    if not args.output:
        raise SystemExit("--output is required")
    output = result.save(args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
