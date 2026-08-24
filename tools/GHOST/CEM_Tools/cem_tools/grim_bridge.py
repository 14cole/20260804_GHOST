"""Adapter to the standalone GRIM viewer's dataset readers and writers."""

import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .errors import CemToolError


INPUT_EXTENSIONS = (".grim", ".out", ".pio", ".cmplx_di", ".csv", ".txt", ".ss")
OUTPUT_EXTENSIONS = (".grim", ".pio", ".cmplx_di", ".csv", ".txt", ".out")


def grim_project_path() -> 'Path':
    configured = os.environ.get("GRIM_REVISED_2_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    # Support a sibling GRIM checkout without embedding a developer-specific
    # absolute path. Conversion is optional; coherent GRIM joins/subtraction do
    # not import this external library.
    for parent in Path(__file__).resolve().parents:
        for name in ("GRIM_Revised_2", "GRIM"):
            candidate = parent / name
            if (candidate / "grim_dataset.py").is_file():
                return candidate
    return Path.cwd() / "GRIM_Revised_2"


def _rcs_grid_class() -> 'type':
    module_path = grim_project_path() / "grim_dataset.py"
    if not module_path.is_file():
        raise CemToolError(
            f"GRIM dataset library not found at {module_path}; set "
            "GRIM_REVISED_2_PATH to its folder"
        )
    module_name = "_cem_tools_external_grim_dataset"
    existing = sys.modules.get(module_name)
    if existing is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise CemToolError(f"cannot import {module_path}")
        existing = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = existing
        spec.loader.exec_module(existing)
    return existing.RcsGrid


def load_dataset(path: 'str | os.PathLike[str]') -> 'Any':
    source = Path(path).expanduser().resolve()
    extension = source.suffix.lower()
    if extension not in INPUT_EXTENSIONS:
        raise CemToolError(f"unsupported input extension: {extension}")
    grid_class = _rcs_grid_class()
    if extension in {".csv", ".txt"}:
        try:
            with source.open("r", encoding="utf-8", errors="replace") as stream:
                first_line = stream.readline()
        except OSError as exc:
            raise CemToolError(f"cannot read {source.name}: {exc}") from exc
        if "azimuth_deg" in first_line and "magnitude_linear" in first_line:
            return _load_flat_table(source, grid_class, "," if extension == ".csv" else "\t")
    if extension in {".csv", ".txt"}:
        fallback = (
            grid_class.load_theta_phi_csv
            if extension == ".csv"
            else grid_class.load_theta_phi_txt
        )
        try:
            if grid_class.has_SENTRi_signature(str(source)):
                return grid_class.read_SENTRi(str(source))
            return fallback(str(source))
        except Exception as exc:
            raise CemToolError(f"cannot load {source.name}: {exc}") from exc
    loaders = {
        ".grim": grid_class.load,
        ".out": grid_class.load_out,
        ".pio": grid_class.load_pio,
        ".cmplx_di": grid_class.load_pio,
        ".ss": grid_class.load_ss,
    }
    try:
        return loaders[extension](str(source))
    except Exception as exc:
        raise CemToolError(f"cannot load {source.name}: {exc}") from exc


def _load_flat_table(source: 'Path', grid_class: 'type', delimiter: 'str') -> 'Any':
    import csv

    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise CemToolError(f"{source.name}: empty table")
    azimuths = np.asarray(sorted({float(row["azimuth_deg"]) for row in rows}))
    elevations = np.asarray(sorted({float(row["elevation_deg"]) for row in rows}))
    frequencies = np.asarray(sorted({float(row["frequency_GHz"]) for row in rows}))
    polarizations = np.asarray(list(dict.fromkeys(row["polarization"] for row in rows)))
    shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
    power = np.full(shape, np.nan, dtype=np.float32)
    phase = np.full(shape, np.nan, dtype=np.float32)
    maps = [
        {value: index for index, value in enumerate(axis)}
        for axis in (azimuths, elevations, frequencies, polarizations)
    ]
    for row in rows:
        index = (
            maps[0][float(row["azimuth_deg"])],
            maps[1][float(row["elevation_deg"])],
            maps[2][float(row["frequency_GHz"])],
            maps[3][row["polarization"]],
        )
        magnitude = float(row["magnitude_linear"])
        if np.isfinite(power[index]):
            raise CemToolError(f"{source.name}: duplicate sample at {index}")
        power[index] = magnitude * magnitude
        phase[index] = np.radians(float(row["phase_deg"]))
    if not np.all(np.isfinite(power)) or not np.all(np.isfinite(phase)):
        raise CemToolError(f"{source.name}: table does not form a complete grid")
    return grid_class(
        azimuths, elevations, frequencies, polarizations,
        rcs_power=power, rcs_phase=phase,
        source_path=str(source),
        history="Loaded CEM Tools flat table",
        units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
    )


def _available_path(path: 'Path', overwrite: 'bool') -> 'None':
    if path.exists() and not overwrite:
        raise CemToolError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_table(grid: 'Any', destination: 'Path', delimiter: 'str') -> 'Path':
    header = delimiter.join(
        ("azimuth_deg", "elevation_deg", "frequency_GHz", "polarization",
         "magnitude_linear", "phase_deg")
    )
    with destination.open("w", encoding="utf-8", newline="") as stream:
        stream.write(header + "\n")
        for ia, azimuth in enumerate(grid.azimuths):
            for ie, elevation in enumerate(grid.elevations):
                for iff, frequency in enumerate(grid.frequencies):
                    for ip, polarization in enumerate(grid.polarizations):
                        magnitude = np.sqrt(max(float(grid.rcs_power[ia, ie, iff, ip]), 0.0))
                        phase = np.degrees(float(grid.rcs_phase[ia, ie, iff, ip]))
                        values = (
                            f"{float(azimuth):.17g}",
                            f"{float(elevation):.17g}",
                            f"{float(frequency):.17g}",
                            str(polarization),
                            f"{magnitude:.17g}",
                            f"{phase:.17g}",
                        )
                        stream.write(delimiter.join(values) + "\n")
    return destination


def _safe_token(value: 'object') -> 'str':
    return str(value).replace(" ", "").replace("/", "-")


def export_dataset(
    grid: 'Any',
    destination: 'str | os.PathLike[str]',
    extension: 'str',
    *,
    overwrite: 'bool' = False,
) -> 'tuple[Path, ...]':
    extension = extension.lower()
    if not extension.startswith("."):
        extension = "." + extension
    if extension not in OUTPUT_EXTENSIONS:
        if extension == ".ss":
            raise CemToolError(".ss is currently read-only and cannot be an output")
        raise CemToolError(f"unsupported output extension: {extension}")
    requested = Path(destination).expanduser().resolve()
    base = (
        requested
        if requested.suffix.lower() == extension
        else requested.with_name(requested.name + extension)
    )

    if extension == ".grim":
        _available_path(base, overwrite)
        try:
            actual = Path(grid.save(str(base))).resolve()
        except Exception as exc:
            raise CemToolError(f"cannot write {base.name}: {exc}") from exc
        return (actual,)
    if extension in {".csv", ".txt"}:
        _available_path(base, overwrite)
        return (_write_table(grid, base, "," if extension == ".csv" else "\t"),)

    if extension in {".pio", ".cmplx_di"}:
        outputs: 'list[Path]' = []
        split = len(grid.elevations) * len(grid.polarizations) > 1
        for ie, elevation in enumerate(grid.elevations):
            for ip, polarization in enumerate(grid.polarizations):
                suffix = (
                    f"_EL{_safe_token(elevation)}_{_safe_token(polarization)}"
                    if split else ""
                )
                target = base.with_name(base.stem + suffix + extension)
                _available_path(target, overwrite)
                try:
                    grid.save_pio(str(target), el_idx=ie, pol_idx=ip)
                except Exception as exc:
                    raise CemToolError(f"cannot write {target.name}: {exc}") from exc
                outputs.append(target)
        return tuple(outputs)

    # The GRIM .out convention is sigma_2d expressed as dBke.
    metadata = dict(getattr(grid, "units", {}) or {})
    metadata.update(getattr(grid, "extra", {}) or {})
    quantity = str(metadata.get("rcs_linear_quantity", ""))
    log_unit = str(metadata.get("rcs_log_unit", ""))
    if "sigma_2d" not in quantity and "dBke" not in log_unit:
        raise CemToolError(
            ".out represents 2D sigma in dBke; this dataset is not tagged as sigma_2d"
        )
    outputs = []
    split = len(grid.elevations) * len(grid.polarizations) > 1
    for ie, elevation in enumerate(grid.elevations):
        for ip, polarization in enumerate(grid.polarizations):
            suffix = f"_EL{_safe_token(elevation)}_{_safe_token(polarization)}" if split else ""
            target = base.with_name(base.stem + suffix + extension)
            _available_path(target, overwrite)
            with target.open("w", encoding="utf-8") as stream:
                stream.write("# frequency_GHz azimuth_deg rcs_dBke phase_deg\n")
                for iff, frequency in enumerate(grid.frequencies):
                    wavelength = 299_792_458.0 / (float(frequency) * 1e9)
                    for ia, azimuth in enumerate(grid.azimuths):
                        sigma = float(grid.rcs_power[ia, ie, iff, ip])
                        dbke = 10.0 * np.log10(sigma * 2.0 * np.pi / wavelength)
                        phase = np.degrees(float(grid.rcs_phase[ia, ie, iff, ip]))
                        stream.write(
                            f"{float(frequency):.17g} {float(azimuth):.17g} "
                            f"{dbke:.17g} {phase:.17g}\n"
                        )
            outputs.append(target)
    return tuple(outputs)


def convert_dataset(
    source: 'str | os.PathLike[str]',
    output_dir: 'str | os.PathLike[str]',
    extension: 'str',
    *,
    overwrite: 'bool' = False,
) -> 'tuple[Path, ...]':
    grid = load_dataset(source)
    destination = Path(output_dir).expanduser().resolve() / Path(source).stem
    return export_dataset(grid, destination, extension, overwrite=overwrite)
