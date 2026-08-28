#!/usr/bin/env python3
"""Render Cartesian azimuth sweeps from every supported dataset in a folder.

Edit the clearly marked configuration block below, then run this file directly::

    python plot_folder_azimuth_sweeps.py

The script uses the same exact-axis matching, unit conversion, logarithmic RCS
conversion, and headless Matplotlib renderer as GRIM's PowerPoint workflow.
It never imports Qt and does not modify its input datasets.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np


# Make this source-tree example runnable from any current working directory.
_GRIM_MODULE_DIR = Path(__file__).resolve().parents[1]
if str(_GRIM_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_GRIM_MODULE_DIR))

from grim_headless import is_supported_path, load_dataset
from ppt_plot_data import build_azimuth_specs, get_plot_availability
from ppt_report import render_plot_png


# =============================================================================
# EDIT THESE SETTINGS, THEN RUN THIS SCRIPT. No command-line arguments are used.
# =============================================================================
INPUT_FOLDER = Path(r"C:\data\trade_study")
INPUT_PATTERN = "*"                 # Example: "*.grim" or "*.csv"
SEARCH_SUBFOLDERS = False
OUTPUT_FOLDER: Path | None = None    # None -> <INPUT_FOLDER>/grim_azimuth_plots

# Values below are in each dataset's stored/native units. None selects every
# common frequency or the first common elevation/polarization, respectively.
FREQUENCIES: tuple[float, ...] | None = None
ELEVATION: float | None = None
POLARIZATIONS: tuple[str, ...] | None = None

QUANTITY = "magnitude"              # "magnitude" or "phase"
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


def discover_dataset_paths(
    folder: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
) -> tuple[Path, ...]:
    """Return supported regular files below *folder* in stable path order."""

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {root}")
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError(
            "INPUT_PATTERN must be a relative glob that stays inside INPUT_FOLDER"
        )
    candidates = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = {
        candidate.resolve()
        for candidate in candidates
        if candidate.is_file()
        and candidate.resolve().is_relative_to(root)
        and is_supported_path(str(candidate))
    }
    result = tuple(sorted(paths, key=lambda value: str(value).casefold()))
    if not result:
        scope = "recursively" if recursive else "at the folder's top level"
        raise ValueError(
            f"No GRIM-supported dataset files matched {pattern!r} {scope} in {root}"
        )
    return result


def load_named_datasets(
    paths: Sequence[Path],
    *,
    root: Path,
    skip_errors: bool = False,
):
    """Load datasets and label them by relative path for unambiguous legends."""

    loaded = []
    failures: list[tuple[Path, Exception]] = []
    for path in paths:
        try:
            loaded.append((path.relative_to(root).as_posix(), load_dataset(str(path))))
        except Exception as exc:  # a batch example should identify the exact file
            if not skip_errors:
                raise RuntimeError(f"Failed to load dataset {path}: {exc}") from exc
            failures.append((path, exc))
    for path, exc in failures:
        print(f"Skipped {path}: {exc}", file=sys.stderr)
    if not loaded:
        raise ValueError("No datasets loaded successfully")
    return loaded


def _validated_axis_limits(
    limits: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if limits is None:
        return None
    if len(limits) != 2:
        raise ValueError("Y_LIMITS must contain exactly (minimum, maximum)")
    low, high = (float(value) for value in limits)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Y_LIMITS values must be finite")
    if low >= high:
        raise ValueError("Y_LIMITS minimum must be less than its maximum")
    return low, high


def _safe_destination(
    output_dir: Path,
    stem: str,
    *,
    overwrite: bool,
) -> Path:
    """Choose a portable PNG name without overwriting unless requested."""

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
    frequencies: Sequence[float] | None = None,
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
    """Render the selected sweeps; arguments use dataset-native axis units."""

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
    if not availability.elevations:
        raise ValueError("Loaded datasets have no common elevation/pitch sample")
    if not availability.frequencies:
        raise ValueError("Loaded datasets have no common frequency sample")
    if not availability.polarizations:
        raise ValueError("Loaded datasets have no common polarization")

    selected_frequencies = (
        [float(value) for value in frequencies]
        if frequencies
        else list(availability.frequencies)
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

    specs = build_azimuth_specs(
        datasets,
        frequencies=selected_frequencies,
        elevation=selected_elevation,
        polarization=selected_polarizations,
        kind="azimuth_rect",
        quantity=quantity,
        angle_display_unit=angle_display_unit,
        frequency_display_unit=frequency_display_unit,
        y_limits=validated_y_limits,
        show_legend=show_legend,
        tol=tol,
    )
    output_dir = (
        Path(output_folder).expanduser().resolve()
        if output_folder is not None
        else root / "grim_azimuth_plots"
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
            frequencies=FREQUENCIES,
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
