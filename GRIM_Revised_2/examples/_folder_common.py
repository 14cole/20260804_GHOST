"""Shared discovery and axis validation for the editable folder examples."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from grim_backend.io.loaders import is_supported_path


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
