#!/usr/bin/env python3
"""Render headless frequency sweeps from every supported dataset in a folder.

The ordinary form selects an exact stored azimuth::

    python plot_folder_frequency_sweeps.py C:\\data\\trade_study \
        --azimuth 0 --elevation 0 --polarization VV

To summarize an inclusive azimuth band at every frequency, supply its two
stored-axis limits and a display-domain percentile::

    python plot_folder_frequency_sweeps.py C:\\data\\trade_study \
        --azimuth-band -10 10 --percentile 90 --elevation 0 --polarization VV

A descending band crosses the periodic seam (for example, ``170 -170`` on a
signed degree axis). The percentile calculation follows GRIM's PowerPoint
path: it uses identical common stored azimuth samples, counts a duplicated
periodic seam only once, and calculates magnitude percentiles in the displayed
logarithmic RCS unit. Wrapped phase uses exact azimuth cuts only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np


_GRIM_MODULE_DIR = Path(__file__).resolve().parents[1]
if str(_GRIM_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_GRIM_MODULE_DIR))

from grim_headless import is_supported_path, load_dataset
from ppt_plot_data import build_frequency_specs, get_plot_availability
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
            "Overlay all supported datasets in a folder and render frequency "
            "sweeps at an exact azimuth or a percentile across an azimuth band."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="folder containing .grim/.csv/.ptm/etc. datasets")
    parser.add_argument("--pattern", default="*", help="relative input filename glob")
    parser.add_argument("--recursive", action="store_true", help="search subfolders")
    parser.add_argument(
        "--output-dir",
        help="PNG destination (default: <folder>/grim_frequency_plots)",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--azimuth",
        type=float,
        help="exact common azimuth/aspect in its stored unit (default: first common value)",
    )
    selector.add_argument(
        "--azimuth-band",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help=(
            "inclusive common-sample band in the stored azimuth unit; START > END "
            "selects across the periodic seam"
        ),
    )
    parser.add_argument(
        "--percentile",
        type=float,
        help="percentile for --azimuth-band (default with a band: 50)",
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
            "output display unit; --azimuth, --azimuth-band, and --elevation "
            "remain in the stored angle unit"
        ),
    )
    parser.add_argument(
        "--frequency-unit",
        choices=("Hz", "kHz", "MHz", "GHz"),
        default="GHz",
        help="output-axis display unit",
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
    if not availability.azimuths:
        raise ValueError("Loaded datasets have no common azimuth/aspect sample")
    if not availability.elevations:
        raise ValueError("Loaded datasets have no common elevation/pitch sample")
    if not availability.polarizations:
        raise ValueError("Loaded datasets have no common polarization")

    if args.percentile is not None and args.azimuth_band is None:
        parser.error("--percentile requires --azimuth-band")
    percentile = (
        (50.0 if args.percentile is None else float(args.percentile))
        if args.azimuth_band is not None
        else None
    )
    if percentile is not None and not 0.0 <= percentile <= 100.0:
        parser.error("--percentile must be between 0 and 100")
    if args.azimuth_band is not None and args.quantity == "phase":
        parser.error(
            "phase sweeps require --azimuth; ordinary percentiles of wrapped "
            "phase are invalid"
        )

    azimuth = (
        None
        if args.azimuth_band is not None
        else (
            float(args.azimuth)
            if args.azimuth is not None
            else float(availability.azimuths[0])
        )
    )
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

    specs = build_frequency_specs(
        datasets,
        azimuth=azimuth,
        elevation=elevation,
        polarization=polarizations,
        quantity=args.quantity,
        angle_display_unit=args.angle_unit,
        frequency_display_unit=args.frequency_unit,
        azimuth_band=(
            None
            if args.azimuth_band is None
            else (float(args.azimuth_band[0]), float(args.azimuth_band[1]))
        ),
        azimuth_percentile=percentile,
        y_limits=y_limits,
        show_legend=not args.no_legend,
        tol=args.tol,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "grim_frequency_plots"
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
