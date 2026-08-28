#!/usr/bin/env python3
"""Join every supported GRIM dataset in a folder into one ``.grim`` file.

The default overlap policy is deliberately strict: equal finite samples may
overlap, but conflicting finite samples stop the operation.  Use
``--overlap first`` or ``--overlap last`` only when pathname-order priority is
intentional.  Input files are always sorted by their full path, so priority is
repeatable.

Example::

    python join_folder.py ./cuts ./combined.grim --pattern "*.csv"
    python join_folder.py ./cuts ./combined.grim --recursive --workers 4
"""

from __future__ import annotations

import argparse
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
            f"output already exists: {output_path}; pass --overwrite to replace it"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join supported datasets from a folder into one GRIM file.",
        epilog=(
            "Strict --overlap error is safest. first/last use sorted full-path "
            "order and should only be selected when that priority is intended."
        ),
    )
    parser.add_argument("folder", help="folder containing input datasets")
    parser.add_argument("output", help="output .grim file")
    parser.add_argument(
        "--pattern", default="*", help="filename glob within the folder (default: *)"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="search matching subfolders too"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="parallel file loaders (default: 1)"
    )
    parser.add_argument(
        "--overlap",
        choices=("error", "first", "last"),
        default="error",
        help="finite-overlap policy (default: error)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-6,
        help=(
            "same numeric coordinate matching tolerance on each native axis "
            "(default: 1e-6)"
        ),
    )
    parser.add_argument(
        "--max-output-gib",
        type=float,
        default=None,
        help="optional upper bound for GRIM's estimated peak join allocation",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.max_output_gib is not None and args.max_output_gib <= 0.0:
        raise SystemExit("--max-output-gib must be positive")
    maximum = (
        None
        if args.max_output_gib is None
        else int(float(args.max_output_gib) * 1024**3)
    )
    try:
        joined, written, paths = join_folder(
            args.folder,
            args.output,
            pattern=args.pattern,
            recursive=args.recursive,
            workers=args.workers,
            overlap=args.overlap,
            tolerance=args.tolerance,
            max_output_bytes=maximum,
            overwrite=args.overwrite,
        )
    except (OSError, TypeError, ValueError, MemoryError) as exc:
        raise SystemExit(f"join failed: {exc}") from exc

    print(f"Loaded {len(paths)} dataset(s) in sorted pathname order:")
    for path in paths:
        print(f"  {path}")
    print(f"Overlap policy: {args.overlap}")
    print(f"Joined shape: {joined.rcs_power.shape}")
    print(f"Wrote: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
