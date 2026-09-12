"""Build GRIM grids from labeled rows with GHz frequencies and degree angles.

The standard CSV reader owns memory admission, sparse cells, duplicate checks
and logarithmic RCS conventions. This adapter has no Qt dependency.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile

from GRIM_Backend.io.csv import FLAT_CSV_SCHEMA
from GRIM_Backend.io.loaders import load_flat_csv


MAGNITUDE_FORMATS = {
    "RCS power (m²)": ("magnitude_power_linear", "sigma_3d", "dBsm"),
    "RCS amplitude (sqrt(m²))": ("magnitude_power_linear", "sigma_3d", "dBsm"),
    "RCS (dBsm)": ("magnitude_dbsm", "sigma_3d", "dBsm"),
    "Scattering width (m)": ("magnitude_power_linear", "sigma_2d", "dBke"),
    "Scattering width (dBke)": ("magnitude_dbke", "sigma_2d", "dBke"),
    "Power ratio": ("magnitude_power_linear", "power_ratio", "dB"),
    "Amplitude ratio": ("magnitude_power_linear", "power_ratio", "dB"),
    "Power ratio (dB)": ("magnitude_db", "power_ratio", "dB"),
}


def grid_from_mapped_rows(rows, source_path, magnitude_format, *, mapping=None):
    if magnitude_format not in MAGNITUDE_FORMATS:
        raise ValueError("Choose what the magnitude column represents.")
    magnitude_name, quantity, log_unit = MAGNITUDE_FORMATS[magnitude_format]
    metadata = {
        "grim_csv_schema": FLAT_CSV_SCHEMA,
        "frequency_unit": "GHz", "azimuth_unit": "deg", "elevation_unit": "deg",
        "rcs_linear_quantity": quantity, "rcs_log_unit": log_unit,
        "angular_coordinate_system": "conic",
    }
    headers = ["frequency", "azimuth", "elevation", "polarization", magnitude_name,
               "phase_deg", *metadata]
    with tempfile.TemporaryDirectory(prefix="grim-column-import-") as directory:
        canonical = Path(directory) / "mapped.csv"
        with canonical.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, headers)
            writer.writeheader()
            count = 0
            for line_no, values in rows:
                try:
                    row = {key: values[key] for key in ("frequency", "azimuth", "elevation", "polarization")}
                    for key in ("frequency", "azimuth", "elevation"):
                        row[key] = float(row[key])
                        if not math.isfinite(row[key]):
                            raise ValueError(f"{key} must be finite")
                    if row["frequency"] <= 0:
                        raise ValueError("frequency must be positive")
                    if not str(row["polarization"]).strip():
                        raise ValueError("polarization must not be blank")
                    magnitude = float(str(values["magnitude"]).replace("D", "E").replace("d", "e"))
                    if "amplitude" in magnitude_format.lower():
                        if not math.isfinite(magnitude) or magnitude < 0:
                            raise ValueError("amplitude must be finite and nonnegative")
                        magnitude *= magnitude
                    if magnitude_name == "magnitude_power_linear":
                        if not math.isfinite(magnitude) or magnitude < 0:
                            raise ValueError("linear magnitude must be finite and nonnegative")
                    elif math.isnan(magnitude) or magnitude == math.inf:
                        raise ValueError("logarithmic magnitude must be finite or -inf (zero power)")
                    row[magnitude_name] = magnitude
                    row["phase_deg"] = values.get("phase", "")
                    writer.writerow(dict(row, **metadata))
                    count += 1
                except (KeyError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Line {line_no}: {exc}") from exc
        if not count:
            raise ValueError("The file contains no data rows.")
        grid = load_flat_csv(str(canonical))
    grid.source_path = str(source_path)
    grid.history = f"Imported labeled table: {source_path}\nMagnitude: {magnitude_format}"
    grid.extra["source_format"] = "User-mapped delimited table"
    grid.extra.pop("flat_csv_schema", None)
    if mapping is not None:
        provenance = json.dumps(mapping, sort_keys=True, ensure_ascii=True)
        grid.extra["column_mapping_json"] = provenance
        grid.history += "\nColumn mapping: " + provenance
    return grid
