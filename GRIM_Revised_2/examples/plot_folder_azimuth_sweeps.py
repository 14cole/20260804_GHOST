#!/usr/bin/env python3
"""Render Cartesian azimuth sweeps from every supported dataset in a folder.

Examples
--------
Render every common frequency at the first common elevation/polarization::

    python plot_folder_azimuth_sweeps.py C:\\data\\trade_study

Select two frequencies (in the files' stored frequency unit) and overlay HH::

    python plot_folder_azimuth_sweeps.py C:\\data\\trade_study \
        --frequency 8 --frequency 10 --elevation 0 --polarization HH

The script uses the same exact-axis matching, unit conversion, logarithmic RCS
conversion, and headless Matplotlib renderer as GRIM's PowerPoint workflow.
It never imports Qt and does not modify its input datasets.
"""

from __future__ import annotations

import argparse
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
        raise ValueError("--pattern must be a relative glob that stays inside the input folder")
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


def _axis_limits(parser: argparse.ArgumentParser, low, high):
    if (low is None) != (high is None):
        parser.error("--y-min and --y-max must be supplied together")
    if low is not None and low >= high:
        parser.error("--y-min must be less than --y-max")
    return None if low is None else (float(low), float(high))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay all supported datasets in a folder and render one Cartesian "
            "azimuth sweep PNG per selected frequency and polarization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="folder containing .grim/.csv/.ptm/etc. datasets")
    parser.add_argument("--pattern", default="*", help="relative input filename glob")
    parser.add_argument("--recursive", action="store_true", help="search subfolders")
    parser.add_argument(
        "--output-dir",
        help="PNG destination (default: <folder>/grim_azimuth_plots)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        action="append",
        help=(
            "exact common frequency in the datasets' stored unit; repeat for more "
            "frequencies (default: every common frequency)"
        ),
    )
    parser.add_argument(
        "--elevation",
        type=float,
        help="exact common elevation/pitch in its stored unit (default: first common value)",
    )
    parser.add_argument(
        "--polarization",
        action="append",
        help="common polarization; repeat to create separate plots (default: first common value)",
    )
    parser.add_argument("--quantity", choices=("magnitude", "phase"), default="magnitude")
    parser.add_argument(
        "--angle-unit",
        choices=("deg", "rad"),
        default="deg",
        help=(
            "output-axis display unit; --elevation remains in the stored "
            "angle unit"
        ),
    )
    parser.add_argument(
        "--frequency-unit",
        choices=("Hz", "kHz", "MHz", "GHz"),
        default="GHz",
        help="display unit only; --frequency remains in the stored unit",
    )
    parser.add_argument("--y-min", type=float, help="fixed lower response-axis limit")
    parser.add_argument("--y-max", type=float, help="fixed upper response-axis limit")
    parser.add_argument("--width", type=float, default=10.0, help="image width in inches")
    parser.add_argument("--height", type=float, default=6.0, help="image height in inches")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--tol", type=float, default=1.0e-6, help="native-axis match tolerance")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="report unreadable matching files and continue with the rest",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace same-named PNGs instead of adding a numeric suffix",
    )
    return parser


def run(args: argparse.Namespace, *, parser: argparse.ArgumentParser) -> tuple[Path, ...]:
    root = Path(args.folder).expanduser().resolve()
    paths = discover_dataset_paths(root, pattern=args.pattern, recursive=args.recursive)
    datasets = load_named_datasets(paths, root=root, skip_errors=args.skip_errors)
    availability = get_plot_availability(
        datasets,
        tol=args.tol,
        evaluate_phase=args.quantity == "phase",
    )
    if not availability.elevations:
        raise ValueError("Loaded datasets have no common elevation/pitch sample")
    if not availability.frequencies:
        raise ValueError("Loaded datasets have no common frequency sample")
    if not availability.polarizations:
        raise ValueError("Loaded datasets have no common polarization")

    frequencies = args.frequency or list(availability.frequencies)
    elevation = (
        float(args.elevation)
        if args.elevation is not None
        else float(availability.elevations[0])
    )
    polarizations = args.polarization or [availability.polarizations[0]]
    y_limits = _axis_limits(parser, args.y_min, args.y_max)
    if not np.isfinite(args.width) or not np.isfinite(args.height):
        parser.error("--width and --height must be finite")
    if args.width <= 0.0 or args.height <= 0.0:
        parser.error("--width and --height must be positive")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    if not np.isfinite(args.tol) or args.tol < 0.0:
        parser.error("--tol must be finite and nonnegative")

    specs = build_azimuth_specs(
        datasets,
        frequencies=frequencies,
        elevation=elevation,
        polarization=polarizations,
        kind="azimuth_rect",
        quantity=args.quantity,
        angle_display_unit=args.angle_unit,
        frequency_display_unit=args.frequency_unit,
        y_limits=y_limits,
        show_legend=not args.no_legend,
        tol=args.tol,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "grim_azimuth_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for spec in specs:
        destination = _safe_destination(
            output_dir, spec.plot_id, overwrite=args.overwrite
        )
        rendered.append(
            render_plot_png(
                spec,
                destination,
                width_points=args.width * 72.0,
                height_points=args.height * 72.0,
                dpi=args.dpi,
            )
        )
    print(f"Loaded {len(datasets)} dataset(s); wrote {len(rendered)} plot(s) to {output_dir}")
    return tuple(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args, parser=parser)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
