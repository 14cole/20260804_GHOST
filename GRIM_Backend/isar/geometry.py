"""Physical selection and image-coordinate helpers shared by GUI and scripts."""
from __future__ import annotations

import numpy as np


def angular_bands(indices, degrees, *, gap_factor=2.5):
    """Group selected samples by angular support, preserving regular strides.

    The cadence is local on either side of a candidate gap, so a legitimate
    change in acquisition density is not mistaken for many isolated samples.
    Singleton sectors are retained for callers to explain, never discarded.
    """
    ids = np.asarray(indices, dtype=np.int64)
    if not ids.size:
        return []
    angles = np.asarray(degrees, dtype=float)[ids]
    if not np.all(np.isfinite(angles)):
        raise ValueError("ISAR azimuth samples must be finite")
    order = np.argsort(angles, kind="stable")
    ids, angles = ids[order], angles[order]
    steps = np.diff(angles)
    if np.any(steps <= 0):
        raise ValueError("Selected ISAR azimuths must be distinct")
    if len(steps) < 2:
        return [ids.tolist()]
    # Three-neighbor medians in one array operation. Per-sample Python medians
    # otherwise dominate warm formation for long azimuth histories.
    left = np.full((len(steps), 3), np.nan)
    right = np.full_like(left, np.nan)
    for offset in range(1, 4):
        left[offset:, offset-1] = steps[:-offset]
        right[:-offset, offset-1] = steps[offset:]
    left[0] = right[0]
    right[-1] = left[-1]
    cadence = np.maximum(np.nanmedian(left, axis=1), np.nanmedian(right, axis=1))
    cuts = np.flatnonzero(steps > gap_factor * cadence) + 1
    return [part.tolist() for part in np.split(ids, cuts)]


def angular_sublooks(indices, degrees, *, maximum_span=10.0, overlap=0.0):
    """Partition in physical angle, with bounded optional overlap and no loss.

    A final singleton may share the previous sample. An isolated sample that
    cannot form a look inside the angular bound is an explicit error.
    """
    if not np.isfinite(maximum_span) or maximum_span <= 0:
        raise ValueError("Sublook width must be finite and positive")
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Sublook overlap must be in [0, 1)")
    angles = np.asarray(degrees, dtype=float)
    looks = []
    for sector in angular_bands(indices, angles):
        values = angles[sector]
        start = 0
        while start < len(sector):
            stop = int(np.searchsorted(values, values[start] + maximum_span + 1e-10, side="right"))
            if stop - start < 2:
                if start > 0 and values[start] - values[start - 1] <= maximum_span + 1e-10:
                    looks.append(sector[start - 1:stop])
                    start = stop
                    continue
                raise ValueError(
                    f"Azimuth {values[start]:g}° has no second sample within the "
                    f"{maximum_span:g}° sublook bound; select a denser angular sector"
                )
            looks.append(sector[start:stop])
            if stop == len(sector):
                break
            if overlap:
                next_angle = values[start] + maximum_span * (1.0 - overlap)
                start = max(start + 1, min(stop, int(np.searchsorted(values, next_angle))))
            else:
                start = stop
    return looks


def axis_edges(axis):
    """Outer edges of a regular center-coordinate axis, including descending axes."""
    values = np.asarray(axis, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Image axes need at least two finite centers")
    steps = np.diff(values)
    if steps[0] == 0 or not np.allclose(steps, steps[0], rtol=1e-7, atol=abs(steps[0]) * 1e-9):
        raise ValueError("Image display requires uniformly spaced centers")
    return float(values[0] - steps[0] / 2), float(values[-1] + steps[0] / 2)


def image_extent(band):
    """Full image boundary, retained even when magnitude is display-decimated."""
    return [*axis_edges(band["x_range"]), *axis_edges(band["y_range"])]
