#!/usr/bin/env python3
"""Find one physical coordinate in a GRIM dataset and inspect its sample.

This example shows the normal four-axis access pattern:

1. load a supported dataset;
2. convert requested angle/frequency values into its native axis units;
3. find the azimuth, elevation, frequency, and polarization indices;
4. index ``rcs_power``/``rcs_phase`` and request the corresponding complex
   value without allocating a complex copy of the full dataset.

Example::

    python query_dataset.py example.grim \
        --azimuth 45 --elevation 0 --frequency 10 --polarization VV
    python query_dataset.py example.grim \
        --azimuth 0.5 --elevation 0 --angle-unit rad \
        --frequency 9500 --frequency-unit MHz --polarization HH --nearest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grim_dataset import RcsGrid
from grim_headless import load_dataset
from plot_modes.common import (
    axis_matching_tolerance,
    axis_unit,
    convert_axis_values,
)


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def find_numeric_index(
    dataset: RcsGrid,
    axis_name: str,
    requested_value: float,
    *,
    requested_unit: str,
    nearest: bool = False,
    tolerance: float | None = None,
) -> dict[str, int | float | str]:
    """Convert a physical value and return its unique or nearest axis index."""

    axis = np.asarray(dataset.get_axis(axis_name), dtype=float)
    if axis.size == 0:
        raise ValueError(f"dataset {axis_name} axis is empty")
    native_unit = axis_unit(dataset, axis_name)
    query = float(requested_value)
    if not math.isfinite(query):
        raise ValueError(f"requested {axis_name} must be finite")
    native_query = float(
        convert_axis_values([query], axis_name, requested_unit, native_unit)[0]
    )
    if tolerance is None:
        native_tolerance = axis_matching_tolerance(dataset, axis_name)
    else:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{axis_name} tolerance must be finite and nonnegative")
        native_tolerance = abs(
            float(
                convert_axis_values(
                    [tolerance], axis_name, requested_unit, native_unit
                )[0]
            )
        )

    differences = np.abs(axis - native_query)
    if nearest:
        index = int(np.argmin(differences))
    else:
        matches = np.flatnonzero(differences <= native_tolerance)
        if matches.size != 1:
            closest = int(np.argmin(differences))
            closest_requested = float(
                convert_axis_values(
                    [axis[closest]], axis_name, native_unit, requested_unit
                )[0]
            )
            if matches.size == 0:
                raise ValueError(
                    f"{axis_name} {query:g} {requested_unit} was not found within "
                    f"{float(tolerance) if tolerance is not None else 'the default'} "
                    f"tolerance; closest is {closest_requested:g} {requested_unit} "
                    f"at index {closest}. Use --nearest to select it explicitly."
                )
            raise ValueError(
                f"{axis_name} {query:g} {requested_unit} matches multiple axis "
                "entries; use a smaller tolerance"
            )
        index = int(matches[0])

    matched_requested = float(
        convert_axis_values([axis[index]], axis_name, native_unit, requested_unit)[0]
    )
    return {
        "index": index,
        "requested_value": query,
        "requested_unit": requested_unit,
        "native_value": float(axis[index]),
        "native_unit": native_unit,
        "matched_value_in_requested_unit": matched_requested,
        "difference_in_requested_unit": abs(matched_requested - query),
    }


def find_polarization_index(dataset: RcsGrid, requested: str) -> dict[str, int | str]:
    """Return one case-insensitive polarization match and its exact label."""

    labels = np.asarray(dataset.polarizations).astype(str)
    query = str(requested).strip()
    matches = np.flatnonzero(np.char.upper(np.char.strip(labels)) == query.upper())
    if matches.size != 1:
        available = ", ".join(labels.tolist()) or "<none>"
        if matches.size == 0:
            raise ValueError(
                f"polarization {requested!r} was not found; available: {available}"
            )
        raise ValueError(f"polarization {requested!r} is ambiguous")
    index = int(matches[0])
    return {"index": index, "requested_value": query, "matched_value": labels[index]}


def query_sample(
    dataset: RcsGrid,
    *,
    azimuth: float,
    elevation: float,
    frequency: float,
    polarization: str,
    angle_unit: str = "deg",
    frequency_unit: str = "GHz",
    nearest: bool = False,
    angle_tolerance: float | None = None,
    frequency_tolerance: float | None = None,
) -> dict[str, object]:
    """Resolve coordinate values to indices and return one JSON-safe sample."""

    az_match = find_numeric_index(
        dataset,
        "azimuth",
        azimuth,
        requested_unit=angle_unit,
        nearest=nearest,
        tolerance=angle_tolerance,
    )
    el_match = find_numeric_index(
        dataset,
        "elevation",
        elevation,
        requested_unit=angle_unit,
        nearest=nearest,
        tolerance=angle_tolerance,
    )
    freq_match = find_numeric_index(
        dataset,
        "frequency",
        frequency,
        requested_unit=frequency_unit,
        nearest=nearest,
        tolerance=frequency_tolerance,
    )
    pol_match = find_polarization_index(dataset, polarization)
    indices = (
        int(az_match["index"]),
        int(el_match["index"]),
        int(freq_match["index"]),
        int(pol_match["index"]),
    )

    power = float(dataset.rcs_power[indices])
    phase_rad = float(dataset.rcs_phase[indices])
    complex_value = complex(dataset.rcs_slice(indices))
    native_frequency = float(dataset.frequencies[indices[2]])
    display_db = (
        float(dataset.linear_to_default_db(power, native_frequency))
        if math.isfinite(power)
        else math.nan
    )
    phase_degrees = math.degrees(phase_rad) if math.isfinite(phase_rad) else math.nan
    phase_wrap = str((dataset.units or {}).get("phase_wrap", "-180_180")).strip()
    if math.isfinite(phase_degrees):
        if phase_wrap == "0_360":
            phase_degrees %= 360.0
        else:
            phase_degrees = (phase_degrees + 180.0) % 360.0 - 180.0

    complex_is_finite = math.isfinite(complex_value.real) and math.isfinite(
        complex_value.imag
    )
    return {
        "source_path": (
            None if dataset.source_path is None else str(dataset.source_path)
        ),
        "angular_coordinate_system": dataset.angular_coordinate_system(),
        "match_mode": "nearest" if nearest else "within_tolerance",
        "indices": {
            "azimuth": indices[0],
            "elevation": indices[1],
            "frequency": indices[2],
            "polarization": indices[3],
        },
        "matches": {
            "azimuth": az_match,
            "elevation": el_match,
            "frequency": freq_match,
            "polarization": pol_match,
        },
        "sample": {
            "linear_power": _finite_or_none(power),
            "field_magnitude": (
                _finite_or_none(math.sqrt(power))
                if math.isfinite(power) and power >= 0.0
                else None
            ),
            "display_magnitude_db": _finite_or_none(display_db),
            "display_magnitude_unit": dataset.default_log_unit(),
            "phase_degrees": _finite_or_none(phase_degrees),
            "phase_wrap": phase_wrap or "-180_180",
            "complex": {
                "real": float(complex_value.real) if complex_is_finite else None,
                "imag": float(complex_value.imag) if complex_is_finite else None,
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve azimuth/elevation/frequency/polarization values to GRIM "
            "indices and print the selected sample as JSON."
        )
    )
    parser.add_argument("dataset", help="any dataset format supported by GRIM")
    parser.add_argument("--azimuth", required=True, type=float)
    parser.add_argument("--elevation", required=True, type=float)
    parser.add_argument("--frequency", required=True, type=float)
    parser.add_argument("--polarization", required=True)
    parser.add_argument(
        "--angle-unit",
        choices=("deg", "rad"),
        default="deg",
        help="unit of --azimuth/--elevation: deg or rad (default: deg)",
    )
    parser.add_argument(
        "--frequency-unit",
        choices=("Hz", "kHz", "MHz", "GHz"),
        default="GHz",
        help="unit of --frequency: Hz, kHz, MHz, or GHz (default: GHz)",
    )
    parser.add_argument(
        "--nearest",
        action="store_true",
        help="select the nearest coordinate instead of requiring a tolerance match",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=None,
        help="matching tolerance in --angle-unit (strict mode only)",
    )
    parser.add_argument(
        "--frequency-tolerance",
        type=float,
        default=None,
        help="matching tolerance in --frequency-unit (strict mode only)",
    )
    parser.add_argument(
        "--output", help="optional JSON output path; stdout is always populated"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing JSON output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        result = query_sample(
            dataset,
            azimuth=args.azimuth,
            elevation=args.elevation,
            frequency=args.frequency,
            polarization=args.polarization,
            angle_unit=args.angle_unit,
            frequency_unit=args.frequency_unit,
            nearest=args.nearest,
            angle_tolerance=args.angle_tolerance,
            frequency_tolerance=args.frequency_tolerance,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"query failed: {exc}") from exc

    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists() and output.samefile(Path(args.dataset).resolve()):
            raise SystemExit("--output must not overwrite the input dataset")
        if output.exists() and not args.overwrite:
            raise SystemExit(
                f"JSON output already exists: {output}; pass --overwrite to replace it"
            )
        if not output.parent.is_dir():
            raise SystemExit(f"JSON output directory does not exist: {output.parent}")
        output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
