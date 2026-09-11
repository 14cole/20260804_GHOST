#!/usr/bin/env python3
"""Render headless frequency sweeps from every supported dataset in a folder.

Edit the clearly marked configuration block below, then run this file directly::

    python plot_folder_frequency_sweeps.py

Use an exact stored azimuth, or set ``AZIMUTH_BAND`` and its optional
``AZIMUTH_PERCENTILE``. A descending band crosses the periodic seam (for
example, ``(170, -170)`` on a signed degree axis). The percentile calculation
follows GRIM's PowerPoint path: it uses identical common stored azimuth
samples, counts a duplicated periodic seam only once, and calculates magnitude
percentiles in the displayed logarithmic RCS unit. Wrapped phase uses exact
azimuth cuts only.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np


_PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from GRIM_Backend.io.loaders import is_supported_path, load_dataset
from GRIM_Backend.reports.plot_data import build_frequency_specs, get_plot_availability
from GRIM_Backend.reports.report import render_plot_png

# Compatibility imports retain the established module entrypoints.
from GRIM_Backend.examples._folder_common import discover_dataset_paths, _validated_axis_limits


# =============================================================================
# EDIT THESE SETTINGS, THEN RUN THIS SCRIPT. No command-line arguments are used.
# =============================================================================
INPUT_FOLDER = Path(r"C:\data\trade_study")
INPUT_PATTERN = "*"                 # Example: "*.grim" or "*.csv"
SEARCH_SUBFOLDERS = False
OUTPUT_FOLDER: Path | None = None    # None -> <INPUT_FOLDER>/grim_frequency_plots

# Exact-cut mode: set AZIMUTH and leave AZIMUTH_BAND as None.
# Band mode: set AZIMUTH_BAND=(start, end); AZIMUTH is then ignored. A descending
# band crosses the periodic seam. None uses P50 when a band is active.
# All selection values below are in each dataset's stored/native units.
AZIMUTH: float | None = 0.0
AZIMUTH_BAND: tuple[float, float] | None = None
AZIMUTH_PERCENTILE: float | None = None
ELEVATION: float | None = None       # None -> first common elevation/pitch
POLARIZATIONS: tuple[str, ...] | None = None  # None -> first common polarization

QUANTITY = "magnitude"              # "magnitude" or "phase"; bands require magnitude
ANGLE_DISPLAY_UNIT = "deg"          # "deg" or "rad"
FREQUENCY_DISPLAY_UNIT = "GHz"      # "Hz", "kHz", "MHz", or "GHz"
Y_LIMITS: tuple[float, float] | None = None
FIGURE_WIDTH_INCHES = 10.0
FIGURE_HEIGHT_INCHES = 6.0
DPI = 160
AXIS_MATCH_TOLERANCE = 1.0e-6
SHOW_LEGEND = True
SKIP_LOAD_ERRORS = False
OVERWRITE_EXISTING = False
# =============================================================================


def load_named_datasets(
    paths: Sequence[Path],
    *,
    root: Path,
    skip_errors: bool = False,
):
    """Load datasets and retain relative paths as unique plot labels."""

    loaded = []
    failures: list[tuple[Path, Exception]] = []
    for path in paths:
        try:
            loaded.append((path.relative_to(root).as_posix(), load_dataset(str(path))))
        except Exception as exc:
            if not skip_errors:
                raise RuntimeError(f"Failed to load dataset {path}: {exc}") from exc
            failures.append((path, exc))
    for path, exc in failures:
        print(f"Skipped {path}: {exc}", file=sys.stderr)
    if not loaded:
        raise ValueError("No datasets loaded successfully")
    return loaded


def _safe_destination(
    output_dir: Path,
    stem: str,
    *,
    overwrite: bool,
) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "plot"
    destination = output_dir / f"{safe_stem}.png"
    if overwrite or not destination.exists():
        return destination
    suffix = 2
    while True:
        candidate = output_dir / f"{safe_stem}_{suffix}.png"
        if not candidate.exists():
            return candidate
        suffix += 1


def run(
    folder: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    output_folder: str | Path | None = None,
    azimuth: float | None = None,
    azimuth_band: tuple[float, float] | None = None,
    percentile: float | None = None,
    elevation: float | None = None,
    polarizations: Sequence[str] | None = None,
    quantity: str = "magnitude",
    angle_display_unit: str = "deg",
    frequency_display_unit: str = "GHz",
    y_limits: tuple[float, float] | None = None,
    width_inches: float = 10.0,
    height_inches: float = 6.0,
    dpi: int = 160,
    tol: float = 1.0e-6,
    show_legend: bool = True,
    skip_errors: bool = False,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Render exact-azimuth or band-percentile frequency sweeps."""

    if quantity not in {"magnitude", "phase"}:
        raise ValueError("QUANTITY must be 'magnitude' or 'phase'")
    if angle_display_unit not in {"deg", "rad"}:
        raise ValueError("ANGLE_DISPLAY_UNIT must be 'deg' or 'rad'")
    if frequency_display_unit not in {"Hz", "kHz", "MHz", "GHz"}:
        raise ValueError(
            "FREQUENCY_DISPLAY_UNIT must be 'Hz', 'kHz', 'MHz', or 'GHz'"
        )
    if not np.isfinite(width_inches) or not np.isfinite(height_inches):
        raise ValueError("Figure width and height must be finite")
    if width_inches <= 0.0 or height_inches <= 0.0:
        raise ValueError("Figure width and height must be positive")
    if dpi < 72:
        raise ValueError("DPI must be at least 72")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("AXIS_MATCH_TOLERANCE must be finite and nonnegative")

    root = Path(folder).expanduser().resolve()
    paths = discover_dataset_paths(root, pattern=pattern, recursive=recursive)
    datasets = load_named_datasets(paths, root=root, skip_errors=skip_errors)
    availability = get_plot_availability(
        datasets,
        tol=tol,
        evaluate_phase=quantity == "phase",
    )
    if not availability.azimuths:
        raise ValueError("Loaded datasets have no common azimuth/aspect sample")
    if not availability.elevations:
        raise ValueError("Loaded datasets have no common elevation/pitch sample")
    if not availability.polarizations:
        raise ValueError("Loaded datasets have no common polarization")

    if percentile is not None and azimuth_band is None:
        raise ValueError("AZIMUTH_PERCENTILE requires AZIMUTH_BAND")
    selected_percentile = (
        (50.0 if percentile is None else float(percentile))
        if azimuth_band is not None
        else None
    )
    if selected_percentile is not None and not 0.0 <= selected_percentile <= 100.0:
        raise ValueError("AZIMUTH_PERCENTILE must be between 0 and 100")
    if azimuth_band is not None and quantity == "phase":
        raise ValueError(
            "Phase sweeps require an exact AZIMUTH; ordinary percentiles of "
            "wrapped phase are invalid"
        )
    if azimuth_band is not None and len(azimuth_band) != 2:
        raise ValueError("AZIMUTH_BAND must contain exactly (start, end)")

    selected_azimuth = (
        None
        if azimuth_band is not None
        else (
            float(azimuth)
            if azimuth is not None
            else float(availability.azimuths[0])
        )
    )
    selected_elevation = (
        float(elevation)
        if elevation is not None
        else float(availability.elevations[0])
    )
    selected_polarizations = (
        list(polarizations) if polarizations else [availability.polarizations[0]]
    )
    validated_y_limits = _validated_axis_limits(y_limits)

    specs = build_frequency_specs(
        datasets,
        azimuth=selected_azimuth,
        elevation=selected_elevation,
        polarization=selected_polarizations,
        quantity=quantity,
        angle_display_unit=angle_display_unit,
        frequency_display_unit=frequency_display_unit,
        azimuth_band=(
            None
            if azimuth_band is None
            else (float(azimuth_band[0]), float(azimuth_band[1]))
        ),
        azimuth_percentile=selected_percentile,
        y_limits=validated_y_limits,
        show_legend=show_legend,
        tol=tol,
    )
    output_dir = (
        Path(output_folder).expanduser().resolve()
        if output_folder is not None
        else root / "grim_frequency_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for spec in specs:
        destination = _safe_destination(
            output_dir, spec.plot_id, overwrite=overwrite
        )
        rendered.append(
            render_plot_png(
                spec,
                destination,
                width_points=width_inches * 72.0,
                height_points=height_inches * 72.0,
                dpi=dpi,
            )
        )
    print(f"Loaded {len(datasets)} dataset(s); wrote {len(rendered)} plot(s) to {output_dir}")
    return tuple(rendered)


def main() -> int:
    try:
        run(
            INPUT_FOLDER,
            pattern=INPUT_PATTERN,
            recursive=SEARCH_SUBFOLDERS,
            output_folder=OUTPUT_FOLDER,
            azimuth=AZIMUTH,
            azimuth_band=AZIMUTH_BAND,
            percentile=AZIMUTH_PERCENTILE,
            elevation=ELEVATION,
            polarizations=POLARIZATIONS,
            quantity=QUANTITY,
            angle_display_unit=ANGLE_DISPLAY_UNIT,
            frequency_display_unit=FREQUENCY_DISPLAY_UNIT,
            y_limits=Y_LIMITS,
            width_inches=FIGURE_WIDTH_INCHES,
            height_inches=FIGURE_HEIGHT_INCHES,
            dpi=DPI,
            tol=AXIS_MATCH_TOLERANCE,
            show_legend=SHOW_LEGEND,
            skip_errors=SKIP_LOAD_ERRORS,
            overwrite=OVERWRITE_EXISTING,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
