#!/usr/bin/env python3
"""Join every supported GRIM dataset in a folder into one ``.grim`` file.

Edit the clearly marked configuration block below, then run this file.  The
default overlap policy is deliberately strict: equal finite samples may
overlap, but conflicting finite samples stop the operation.  Input files are
always sorted by full path, so ``first``/``last`` priority is repeatable.
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

# Permit running this file directly from a source checkout.  An installed GRIM
# environment already exposes these modules and does not need this fallback.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grim_dataset import RcsGrid
from grim_headless import is_supported_path, load_dataset


# =============================================================================
# EDIT THESE SETTINGS, THEN RUN THIS SCRIPT
# =============================================================================
INPUT_FOLDER = Path(r"C:\path\to\dataset_folder")
OUTPUT_FILE = Path(r"C:\path\to\combined.grim")

# Use "*" for every supported dataset or, for example, "*.csv".
FILE_PATTERN = "*"
SEARCH_SUBFOLDERS = False
PARALLEL_LOADERS = 1

# "error" is safest. "first" and "last" resolve conflicts using sorted
# full-path order and should only be used when that priority is intentional.
OVERLAP_POLICY = "error"
COORDINATE_TOLERANCE = 1.0e-6

# Set a positive value to cap GRIM's estimated peak join allocation, or None
# for no explicit cap.  Existing output is protected unless this is True.
MAX_OUTPUT_GIB: float | None = None
OVERWRITE_OUTPUT = False
# =============================================================================


def normalized_grim_path(path: str | os.PathLike[str]) -> Path:
    """Return the output path that ``RcsGrid.save`` will actually publish."""

    output = Path(path).expanduser().resolve()
    if not str(output).casefold().endswith(".grim"):
        output = Path(str(output) + ".grim")
    return output


def discover_dataset_files(
    folder: str | os.PathLike[str],
    *,
    pattern: str = "*",
    recursive: bool = False,
    exclude: Iterable[str | os.PathLike[str]] = (),
) -> list[Path]:
    """Return supported regular files in deterministic pathname order."""

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"input folder does not exist or is not a directory: {root}")
    excluded = {Path(item).expanduser().resolve() for item in exclude}
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = sorted(
        (
            path.resolve()
            for path in iterator
            if path.is_file()
            and is_supported_path(str(path))
            and path.resolve() not in excluded
        ),
        key=lambda value: os.path.normcase(str(value)),
    )
    if not paths:
        scope = "recursively " if recursive else ""
        raise ValueError(
            f"no supported dataset files matched {pattern!r} {scope}under {root}"
        )
    return paths


def load_files(paths: Sequence[Path], *, workers: int = 1) -> list[RcsGrid]:
    """Load paths in order, optionally parsing independent files in threads."""

    worker_count = max(1, int(workers))
    if worker_count == 1:
        return [load_dataset(str(path)) for path in paths]
    with ThreadPoolExecutor(max_workers=min(worker_count, len(paths))) as pool:
        # executor.map preserves input ordering even when parsing completes out
        # of order.  That property matters for first/last overlap priority.
        return list(pool.map(lambda path: load_dataset(str(path)), paths))


def join_folder(
    folder: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    pattern: str = "*",
    recursive: bool = False,
    workers: int = 1,
    overlap: str = "error",
    tolerance: float = 1.0e-6,
    max_output_bytes: int | None = None,
    overwrite: bool = False,
) -> tuple[RcsGrid, Path, list[Path]]:
    """Load, strictly join, and atomically save all matching folder datasets.

    ``overlap`` is passed directly to :meth:`RcsGrid.join_many`:

    * ``error`` accepts identical samples and rejects conflicting samples.
    * ``first`` keeps the finite sample from the first sorted input path.
    * ``last`` keeps the finite sample from the last sorted input path.
    """

    output_path = normalized_grim_path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; set overwrite=True to replace it"
        )
    if not output_path.parent.is_dir():
        raise ValueError(f"output directory does not exist: {output_path.parent}")
    tolerance = float(tolerance)
    if not tolerance >= 0.0 or not tolerance < float("inf"):
        raise ValueError("tolerance must be a finite nonnegative number")
    if max_output_bytes is not None and int(max_output_bytes) <= 0:
        raise ValueError("max_output_bytes must be positive when supplied")

    paths = discover_dataset_files(
        folder,
        pattern=pattern,
        recursive=recursive,
        # An earlier output inside the input folder must never become an input
        # on a repeat run.
        exclude=(output_path,),
    )
    grids = load_files(paths, workers=workers)
    joined = RcsGrid.join_many(
        *grids,
        tol=tolerance,
        overlap=overlap,
        max_output_bytes=max_output_bytes,
    )
    written = Path(joined.save(output_path)).resolve()
    return joined, written, paths


def main() -> int:
    """Run the join using the editable constants at the top of this file."""

    workers = int(PARALLEL_LOADERS)
    if workers < 1:
        raise SystemExit("PARALLEL_LOADERS must be at least 1")
    if MAX_OUTPUT_GIB is None:
        maximum = None
    else:
        maximum_gib = float(MAX_OUTPUT_GIB)
        if not math.isfinite(maximum_gib) or maximum_gib <= 0.0:
            raise SystemExit("MAX_OUTPUT_GIB must be a finite positive number")
        maximum = int(maximum_gib * 1024**3)
    try:
        joined, written, paths = join_folder(
            INPUT_FOLDER,
            OUTPUT_FILE,
            pattern=FILE_PATTERN,
            recursive=SEARCH_SUBFOLDERS,
            workers=workers,
            overlap=OVERLAP_POLICY,
            tolerance=COORDINATE_TOLERANCE,
            max_output_bytes=maximum,
            overwrite=OVERWRITE_OUTPUT,
        )
    except (OSError, TypeError, ValueError, MemoryError) as exc:
        raise SystemExit(f"join failed: {exc}") from exc

    print(f"Loaded {len(paths)} dataset(s) in sorted pathname order:")
    for path in paths:
        print(f"  {path}")
    print(f"Overlap policy: {OVERLAP_POLICY}")
    print(f"Joined shape: {joined.rcs_power.shape}")
    print(f"Wrote: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
