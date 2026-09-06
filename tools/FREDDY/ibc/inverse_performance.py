"""Display metrics for sampled PEC reflection; independent of search ranking."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class BandPerformance:
    deepest_db: float
    worst_db: float
    coverage_pct: float
    widest_band: tuple[float, float] | None
    passing_bands: tuple[tuple[float, float], ...]

    @property
    def widest_ghz(self) -> float | None:
        if self.widest_band is None:
            return None
        return self.widest_band[1] - self.widest_band[0]


def response_curve(samples: Sequence[Sequence[float]], percentile: float = 100.) -> list[float]:
    """Pointwise percentile over analyzed angles/corners, not a yield statistic."""
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError('Percentile must be between 0 and 100.')
    curve = []
    for values in samples:
        ordered = sorted(float(v) for v in values)
        if not ordered or not all(math.isfinite(v) for v in ordered):
            raise ValueError('Response samples must be complete and finite.')
        position = percentile / 100 * (len(ordered) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        fraction = position - lo
        curve.append(ordered[lo] * (1 - fraction) + ordered[hi] * fraction)
    return curve


def band_performance(frequencies: Sequence[float], curve: Sequence[float],
                     threshold_db: float, *, continuous: bool) -> BandPerformance:
    """Estimate passing intervals by linear interpolation in dB on a band sweep.

    Discrete targets and single frequencies get point coverage only: no width
    is inferred between independent targets. Never extrapolate beyond the
    sampled range. A coarse sweep can miss narrow features between samples.
    """
    if len(frequencies) != len(curve) or not frequencies:
        raise ValueError('Frequency and response arrays must have equal, nonzero lengths.')
    if not math.isfinite(threshold_db):
        raise ValueError('Reflection target must be finite.')
    pairs = sorted((float(f), float(v)) for f, v in zip(frequencies, curve))
    if not all(math.isfinite(f) and f > 0 and math.isfinite(v) for f, v in pairs):
        raise ValueError('Frequencies must be positive and all data must be finite.')
    if any(a[0] == b[0] for a, b in zip(pairs, pairs[1:])):
        raise ValueError('Frequencies must be unique.')
    values = [v for _, v in pairs]
    if not continuous or len(pairs) == 1:
        return BandPerformance(min(values), max(values),
                               100 * sum(v <= threshold_db for v in values) / len(values),
                               None, ())
    bands: list[tuple[float, float]] = []
    for (a, va), (b, vb) in zip(pairs, pairs[1:]):
        if va > threshold_db and vb > threshold_db:
            continue
        start, end = a, b
        if (va <= threshold_db) != (vb <= threshold_db):
            crossing = a + (threshold_db - va) / (vb - va) * (b - a)
            if va <= threshold_db:
                end = crossing
            else:
                start = crossing
        if bands and start <= bands[-1][1]:
            bands[-1] = bands[-1][0], end
        else:
            bands.append((start, end))
    coverage = 100 * sum(b - a for a, b in bands) / (pairs[-1][0] - pairs[0][0])
    # Zero width means no passing bandwidth; None is reserved for discrete data.
    widest = max(bands, key=lambda ab: ab[1] - ab[0]) if bands else (pairs[0][0], pairs[0][0])
    return BandPerformance(min(values), max(values), coverage, widest, tuple(bands))


def search_progress(scores: Sequence[float]) -> tuple[list[float], list[float]]:
    """One point per completed unique evaluation, including local refinement."""
    values = [float(v) for v in scores]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Search scores must be finite.')
    best = []
    for value in values:
        best.append(min(best[-1], value) if best else value)
    return values, best
