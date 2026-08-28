#!/usr/bin/env python3
"""Find one physical coordinate in a GRIM dataset and inspect its sample.

This example shows the normal four-axis access pattern:

1. load a supported dataset;
2. convert requested angle/frequency values into its native axis units;
3. find the azimuth, elevation, frequency, and polarization indices;
4. index ``rcs_power``/``rcs_phase`` and request the corresponding complex
   value without allocating a complex copy of the full dataset.

Edit the clearly marked configuration block below, then run this file.  The
reusable :func:`query_sample` function can also be imported by another script.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

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


# =============================================================================
# EDIT THESE SETTINGS, THEN RUN THIS SCRIPT
# =============================================================================
DATASET_PATH = Path(r"C:\path\to\dataset.grim")

QUERY_AZIMUTH = 45.0
QUERY_ELEVATION = 0.0
QUERY_FREQUENCY = 10.0
QUERY_POLARIZATION = "VV"

# Angles may use "deg" or "rad". Frequency may use "Hz", "kHz", "MHz",
# or "GHz"; these are the units of the requested values above, not necessarily
# the dataset's stored units.
QUERY_ANGLE_UNIT = "deg"
QUERY_FREQUENCY_UNIT = "GHz"

# False requires a unique match within tolerance. True deliberately selects
# the closest stored coordinate. None uses GRIM's scale-aware default tolerance.
NEAREST_MATCH = False
ANGLE_TOLERANCE: float | None = None
FREQUENCY_TOLERANCE: float | None = None

# The result is always printed. Set this to a Path to also save JSON.
JSON_OUTPUT_PATH: Path | None = None
OVERWRITE_JSON = False
# =============================================================================


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
                    f"at index {closest}. Call with nearest=True to select it "
                    "explicitly."
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


def main() -> int:
    """Run the query using the editable constants at the top of this file."""

    dataset_path = Path(DATASET_PATH).expanduser().resolve()
    try:
        dataset = load_dataset(str(dataset_path))
        result = query_sample(
            dataset,
            azimuth=QUERY_AZIMUTH,
            elevation=QUERY_ELEVATION,
            frequency=QUERY_FREQUENCY,
            polarization=QUERY_POLARIZATION,
            angle_unit=QUERY_ANGLE_UNIT,
            frequency_unit=QUERY_FREQUENCY_UNIT,
            nearest=NEAREST_MATCH,
            angle_tolerance=ANGLE_TOLERANCE,
            frequency_tolerance=FREQUENCY_TOLERANCE,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"query failed: {exc}") from exc

    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if JSON_OUTPUT_PATH is not None:
        output = Path(JSON_OUTPUT_PATH).expanduser().resolve()
        if output.exists() and output.samefile(dataset_path):
            raise SystemExit("JSON_OUTPUT_PATH must not overwrite DATASET_PATH")
        if output.exists() and not OVERWRITE_JSON:
            raise SystemExit(
                f"JSON output already exists: {output}; set OVERWRITE_JSON = True "
                "to replace it"
            )
        if not output.parent.is_dir():
            raise SystemExit(f"JSON output directory does not exist: {output.parent}")
        output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
