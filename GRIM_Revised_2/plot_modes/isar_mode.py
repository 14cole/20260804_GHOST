from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
import re
import threading
import time

import numpy as np

from . import common

try:  # scipy.fft is multithreaded and preserves single precision
    from scipy import fft as _sp_fft
except ImportError:  # pragma: no cover - scipy is normally present
    _sp_fft = None


_FFT_WORKERS = int(os.environ.get("GRIM_FFT_WORKERS", "-1"))


def _ifft(a: np.ndarray, n: int, axis: int) -> np.ndarray:
    if _sp_fft is not None:
        return _sp_fft.ifft(a, n=n, axis=axis, workers=_FFT_WORKERS)
    return np.fft.ifft(a, n=n, axis=axis)


def _fft2(a: np.ndarray) -> np.ndarray:
    if _sp_fft is not None:
        return _sp_fft.fft2(a, workers=_FFT_WORKERS)
    return np.fft.fft2(a)


def _ifft2(a: np.ndarray) -> np.ndarray:
    if _sp_fft is not None:
        return _sp_fft.ifft2(a, workers=_FFT_WORKERS)
    return np.fft.ifft2(a)


def _lerp_along_last(src_x: np.ndarray, src_y: np.ndarray, tgt_x: np.ndarray) -> np.ndarray:
    """Linear interpolation of `src_y` (..., n_src), sampled at ascending
    `src_x`, onto `tgt_x` — vectorized over ALL leading rows at once (the
    per-row np.interp loop this replaces was the bottleneck for 0.01°-step
    files: 36k Python-level interp calls). Complex-safe; targets outside the
    source support clamp to the edge samples (callers zero-fill if needed)."""
    j = np.clip(np.searchsorted(src_x, tgt_x, side="left"), 1, src_x.size - 1)
    x0 = src_x[j - 1]
    span = src_x[j] - x0
    span[span == 0.0] = 1.0
    w = np.clip((tgt_x - x0) / span, 0.0, 1.0)
    if src_y.dtype in (np.complex64, np.float32):
        w = w.astype(np.float32)  # keep single precision from upcasting
    y0 = src_y[..., j - 1]
    return y0 + (src_y[..., j] - y0) * w


def _unit_to_hz_scale(unit: str) -> float:
    unit = unit.strip().lower()
    if unit == "hz":
        return 1.0
    if unit == "khz":
        return 1e3
    if unit == "mhz":
        return 1e6
    if unit == "ghz":
        return 1e9
    raise ValueError(
        f"unsupported frequency unit {unit!r}; expected Hz, kHz, MHz, or GHz"
    )


def _angle_values_to_degrees(dataset, axis: str, values) -> np.ndarray:
    """Normalize native degree/radian axes before ISAR trigonometry."""

    return common.convert_axis_values(
        values, axis, common.axis_unit(dataset, axis), "deg"
    )


_LENGTH_UNIT_FACTORS = {
    "m": 1.0,
    "in": 1.0 / 0.0254,
    "ft": 1.0 / 0.3048,
}

_ISAR_WINDOW_NAMES = (
    "Hanning",
    "Hamming",
    "Blackman",
    "Blackman-Harris",
    "Kaiser β=15",
    "Rectangular",
)
_ISAR_WINDOW_LOOKUP = {name.casefold(): name for name in _ISAR_WINDOW_NAMES}
_ISAR_WINDOW_LOOKUP["kaiser beta=15"] = "Kaiser β=15"

# A Fourier image needs a rectangular, uniformly sampled k-space grid.  A
# missing block may be represented by zero measurement weights, but it must
# never be interpolated into apparently observed samples.  A separation over
# 2.5 times the robust local cadence identifies at least two omitted nominal
# samples while still allowing meaningful nonuniform sampling jitter.
_ISAR_GAP_FACTOR = 2.5
_MAX_INTERP_AZIMUTH_SAMPLES = 1_000_000
_MAX_INTERP_COMPLEX_CELLS = 16_000_000


def _length_unit(name: str | None) -> tuple[str, float]:
    key = str(name).strip().lower() if name is not None else ""
    if key not in _LENGTH_UNIT_FACTORS:
        expected = ", ".join(_LENGTH_UNIT_FACTORS)
        raise ValueError(
            f"unsupported ISAR length unit {name!r}; expected one of: {expected}"
        )
    return key, _LENGTH_UNIT_FACTORS[key]


def _window_name(name: str) -> str:
    """Return the canonical public ISAR taper name or reject a typo."""

    key = str(name).strip().casefold()
    if key not in _ISAR_WINDOW_LOOKUP:
        # Keep public exception text ASCII-safe for Windows batch/headless
        # consoles even though the GUI's canonical Kaiser label uses beta.
        expected = ", ".join(
            value.replace("β", "beta") for value in _ISAR_WINDOW_NAMES
        )
        raise ValueError(
            f"unsupported ISAR window {name!r}; expected one of: {expected}"
        )
    return _ISAR_WINDOW_LOOKUP[key]


def _sparse_parameters(strength, n_iters) -> tuple[float, int]:
    """Validate the fixed-lambda LASSO controls used by public/headless ISAR."""

    try:
        strength_value = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("l1_strength must be a finite number between 0 and 1") from exc
    if not np.isfinite(strength_value) or not 0.0 < strength_value < 1.0:
        raise ValueError("l1_strength must be a finite number strictly between 0 and 1")

    if isinstance(n_iters, (bool, np.bool_)):
        raise ValueError("l1_iterations must be an integer between 10 and 10000")
    try:
        iterations_value = int(n_iters)
        numeric_iterations = float(n_iters)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("l1_iterations must be an integer between 10 and 10000") from exc
    if (
        not np.isfinite(numeric_iterations)
        or numeric_iterations != iterations_value
        or not 10 <= iterations_value <= 10_000
    ):
        raise ValueError("l1_iterations must be an integer between 10 and 10000")
    return strength_value, iterations_value


def _spacing_summary(differences: np.ndarray) -> tuple[float, float, float]:
    """Return robust nominal step, relative spread, and large-gap threshold."""

    positive = np.asarray(differences, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return 0.0, 0.0, 0.0
    median = float(np.median(positive))
    spread = (
        float(np.max(positive) - np.min(positive)) / median
        if positive.size > 1 and median > 0.0
        else 0.0
    )
    # A single missing sector pushes the ordinary median toward the gap when
    # only three or four samples are selected.  Estimate the acquired cadence
    # from the lower half of the spacings instead.
    ordered = np.sort(positive)
    lower_count = max(1, (ordered.size + 1) // 2)
    nominal = float(np.median(ordered[:lower_count]))
    return nominal, spread, _ISAR_GAP_FACTOR * nominal


def _interpolation_support(
    source: np.ndarray,
    target: np.ndarray,
    *,
    periodic_degrees: bool = False,
) -> tuple[np.ndarray, dict]:
    """Identify target samples supported by a nearby measured interval.

    Targets exactly coincident with a source sample are always supported.
    Intervals wider than ``_ISAR_GAP_FACTOR`` times the robust local cadence
    are holes, not interpolation support.  The returned mask is therefore a
    measurement-operator mask, not merely an interpolation-domain check.
    """

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 1 or target.ndim != 1 or source.size < 2:
        raise ValueError("gap-aware interpolation requires 1-D axes with at least 2 samples")
    if not np.all(np.isfinite(source)) or np.any(np.diff(source) <= 0.0):
        raise ValueError("gap-aware interpolation source must be finite and strictly increasing")

    if periodic_degrees:
        source_eval = np.concatenate((
            [source[-1] - 360.0], source, [source[0] + 360.0],
        ))
        target_eval = source[0] + np.mod(target - source[0], 360.0)
    else:
        source_eval = source
        target_eval = target

    diffs = np.diff(source_eval)
    nominal, spread, gap_limit = _spacing_summary(diffs)
    scale = max(float(np.max(np.abs(source_eval))), 1.0)
    exact_tolerance = max(nominal * 1.0e-7, np.finfo(float).eps * scale * 32.0)

    right = np.searchsorted(source_eval, target_eval, side="left")
    right_clip = np.clip(right, 0, source_eval.size - 1)
    left_clip = np.clip(right - 1, 0, source_eval.size - 1)
    exact = (
        np.abs(target_eval - source_eval[right_clip]) <= exact_tolerance
    ) | (
        np.abs(target_eval - source_eval[left_clip]) <= exact_tolerance
    )
    between = (right > 0) & (right < source_eval.size)
    bridge = np.zeros(target.size, dtype=bool)
    if gap_limit > 0.0 and np.any(between):
        bridge[between] = diffs[right[between] - 1] <= gap_limit
    support = exact | (between & bridge)

    positive_diffs = diffs[diffs > exact_tolerance]
    gap_count = int(np.count_nonzero(positive_diffs > gap_limit)) if gap_limit > 0.0 else 0
    largest_gap = float(np.max(positive_diffs)) if positive_diffs.size else 0.0
    info = {
        "non_uniformity": float(spread),
        "nominal_step": float(nominal),
        "largest_gap": largest_gap,
        "gap_count": gap_count,
        "unsupported_fraction": float(np.mean(~support)) if support.size else 0.0,
    }
    return support, info


def _uniform_resample_plan(
    values: np.ndarray,
    *,
    max_output_samples: int | None = None,
    rel_tol: float = 1.0e-3,
) -> dict:
    """Build a uniform target axis and an honest measurement-support mask."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("uniform ISAR resampling requires at least 2 axis samples")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("axis samples must be finite and strictly increasing")

    diffs = np.diff(values)
    nominal, spread, gap_limit = _spacing_summary(diffs)
    large_gaps = diffs > gap_limit if gap_limit > 0.0 else np.zeros_like(diffs, dtype=bool)
    if spread < rel_tol and not np.any(large_gaps):
        target = values
    elif np.any(large_gaps):
        # Preserve the local acquired cadence so samples on either side of a
        # missing sector are retained.  The sector itself is zero-weighted by
        # the support mask below.
        required_intervals = float(values[-1] - values[0]) / nominal
        effective_limit = (
            _MAX_INTERP_AZIMUTH_SAMPLES
            if max_output_samples is None else int(max_output_samples)
        )
        if (
            not np.isfinite(required_intervals)
            or required_intervals + 1.0 > effective_limit
        ):
            largest = float(np.max(diffs))
            requested = (
                "an unbounded number of"
                if not np.isfinite(required_intervals)
                else f"approximately {int(np.ceil(required_intervals)) + 1:,}"
            )
            raise ValueError(
                f"disjoint samples contain a {largest:g} gap "
                f"({largest / nominal:.1f}x nominal {nominal:g}); preserving "
                f"the missing sector would require {requested} uniform "
                "samples. Select one contiguous band/aperture instead"
            )
        intervals = max(values.size - 1, int(np.rint(required_intervals)))
        target_count = intervals + 1
        if target_count > effective_limit and target_count > values.size:
            largest = float(np.max(diffs))
            raise ValueError(
                f"disjoint samples contain a {largest:g} gap "
                f"({largest / nominal:.1f}x nominal {nominal:g}); preserving "
                f"the missing sector would require {target_count:,} uniform "
                "samples. Select one contiguous band/aperture instead"
            )
        target = np.linspace(values[0], values[-1], target_count)
    else:
        target = np.linspace(values[0], values[-1], values.size)

    support, info = _interpolation_support(values, target)
    info["resampled"] = not (
        target.shape == values.shape and np.array_equal(target, values)
    )
    return {"target": target, "support": support, "info": info}


def _apply_resample_plan(
    source: np.ndarray,
    samples: np.ndarray,
    axis: int,
    plan: dict,
) -> np.ndarray:
    """Interpolate onto a plan and zero targets outside measured support."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(plan["target"], dtype=float)
    support = np.asarray(plan["support"], dtype=bool)
    if source.shape == target.shape and np.array_equal(source, target):
        out = samples
    else:
        moved = np.moveaxis(samples, axis, -1)
        out = np.moveaxis(_lerp_along_last(source, moved, target), -1, axis)
    if not np.all(support):
        out = np.array(out, copy=True)
        moved = np.moveaxis(out, axis, -1)
        moved[..., ~support] = 0
    return out


def _resample_azimuth_to_target(
    source_deg: np.ndarray,
    samples: np.ndarray,
    target_deg: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resample complex `samples` along `axis` from azimuth `source_deg` onto `target_deg`.

    Treats azimuth as a periodic (360°) axis when the source covers ≥359°,
    which is the case the reference ISAR program is built around — a full sweep
    sampled non-uniformly. For partial apertures, falls back to linear
    interpolation with zero-fill outside the source support so the missing
    arcs don't get aliased by a long way-round wrap.

    Real and imaginary parts are interpolated independently — interpolating
    magnitude or argument introduces phase-wrap artefacts.
    """
    source = np.asarray(source_deg, dtype=float)
    target = np.asarray(target_deg, dtype=float)
    span = float(source.max() - source.min())
    use_period = span >= 359.0

    order = np.argsort(source)
    source_sorted = source[order]
    samples_moved = np.moveaxis(np.take(samples, order, axis=axis), axis, -1)

    if use_period:
        # Periodic interpolation: wrap targets into [src_min, src_min+360) and
        # extend one wrap sample on each side so the seam interpolates between
        # the last and (first + 360°) source samples — same result as
        # np.interp(..., period=360), but vectorized across all rows.
        src_ext = np.concatenate((
            [source_sorted[-1] - 360.0], source_sorted, [source_sorted[0] + 360.0],
        ))
        samp_ext = np.concatenate(
            (samples_moved[..., -1:], samples_moved, samples_moved[..., :1]), axis=-1
        )
        tgt_wrapped = source_sorted[0] + np.mod(target - source_sorted[0], 360.0)
        out = _lerp_along_last(src_ext, samp_ext, tgt_wrapped)
        support, gap_info = _interpolation_support(
            source_sorted, target, periodic_degrees=True
        )
    else:
        # Linear interp with zero-fill outside [source_min, source_max].
        out = _lerp_along_last(source_sorted, samples_moved, target)
        outside = (target < source_sorted[0]) | (target > source_sorted[-1])
        if np.any(outside):
            out[..., outside] = 0.0
        support, gap_info = _interpolation_support(source_sorted, target)

    # A target between two widely separated acquired sectors is not an
    # observation.  Zero it in both the field and the separately-resampled
    # validity weights so downstream FFT/L1 paths see a missing measurement.
    if not np.all(support):
        out[..., ~support] = 0.0

    out = np.moveaxis(out, -1, axis)
    return target, out, gap_info


def _resample_complex_uniform(
    values: np.ndarray,
    samples: np.ndarray,
    axis: int,
    rel_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Linearly resample complex `samples` onto a uniform grid along `axis`.

    Returns (uniform_values, resampled_samples, non_uniformity).
    `non_uniformity` is the relative spread of the original spacings —
    `(max_diff - min_diff) / median_diff`. When this is below `rel_tol`,
    the inputs are returned unchanged (no work done) and the value is
    reported so callers can warn the user.

    Complex values are interpolated linearly (equivalent to interpolating
    real and imaginary parts independently) — for ISAR that's what you want,
    since interpolating |z| or arg(z) introduces phase-wrap artefacts.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values, samples, 0.0
    plan = _uniform_resample_plan(values, rel_tol=rel_tol)
    target = np.asarray(plan["target"], dtype=float)
    out = _apply_resample_plan(values, samples, axis=axis, plan=plan)
    return target, out, float(plan["info"]["non_uniformity"])


def _block_reduce_max(a: np.ndarray, factor: int, axis: int) -> np.ndarray:
    """Max-pool `a` along `axis` by `factor` (edge-padded to a multiple)."""
    n = a.shape[axis]
    n_blocks = -(-n // factor)
    pad = n_blocks * factor - n
    if pad:
        widths = [(0, 0)] * a.ndim
        widths[axis] = (0, pad)
        a = np.pad(a, widths, mode="edge")
    shape = list(a.shape)
    shape[axis] = n_blocks
    shape.insert(axis + 1, factor)
    return a.reshape(shape).max(axis=axis + 1)


def _decimate_display_max(img: np.ndarray, max_side: int = 4096) -> np.ndarray:
    """Reduce an oversized display image with per-block MAX so bright
    scatterers survive. imshow's nearest-neighbour draw-time resampling drops
    ~97% of the pixels of a 36000-wide image in a ~1000-px viewport — point
    responses visibly blink in and out while panning. Max-pooling to a
    screen-comparable size keeps every peak (max in dB == max in linear).
    The extent is unchanged, so axes and cursor readout stay correct."""
    for axis in (0, 1):
        n = img.shape[axis]
        if n > max_side:
            img = _block_reduce_max(img, -(-n // max_side), axis)
    return img


def _split_into_bands(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    bands: list[list[int]] = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            bands.append(current)
            current = [idx]
    bands.append(current)
    return bands


def _unwrap_degrees(values: np.ndarray, center_deg: float) -> np.ndarray:
    """Return angles on the continuous branch centred on ``center_deg``."""
    values = np.asarray(values, dtype=float)
    return center_deg + np.mod(values - center_deg + 180.0, 360.0) - 180.0




def _window_array(name: str, n: int) -> np.ndarray:
    """Aperture window by combo-box name. Module-level and Qt-free so the
    worker thread can build windows without touching widgets (the mixin's
    `_isar_window` delegates here after reading the combo on the GUI thread)."""
    name = _window_name(name)
    # Hann/Blackman evaluate to all zeros at n=2, which makes a perfectly
    # valid minimum-size aperture look like zero weighted coverage.  There is
    # no meaningful endpoint taper with only two samples, so use rectangular.
    if n <= 2:
        return np.ones(n)
    if name == "Hamming":
        return np.hamming(n)
    if name == "Blackman":
        return np.blackman(n)
    if name == "Blackman-Harris":
        # 4-term Blackman-Harris, peak sidelobe ~ -92 dB.
        x = 2.0 * np.pi * np.arange(n) / (n - 1)
        return (
            0.35875
            - 0.48829 * np.cos(x)
            + 0.14128 * np.cos(2.0 * x)
            - 0.01168 * np.cos(3.0 * x)
        )
    if name.startswith("Kaiser"):
        # Kaiser β=15 — peak sidelobe ~ -110 dB.
        return np.kaiser(n, 15.0)
    if name == "Rectangular":
        return np.ones(n)
    return np.hanning(n)


def _next_fast_len(n: int) -> int:
    """Smallest 5-smooth number (2^a·3^b·5^c) >= n — the FFT sizes pocketfft
    computes fastest (numpy has no next_fast_len; scipy's isn't a dependency)."""
    if n <= 6:
        return max(n, 1)
    best = 1 << (n - 1).bit_length()          # next power of two always works
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n:                       # smallest p35·2^k >= n
                q <<= 1
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


def _compute_band_polar_format(
    window_name: str,
    rcs_polar: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
    *,
    sample_weights: np.ndarray | None = None,
    elevation_deg: float = 0.0,
    cancel_check=None,
):
    """Range-Doppler / Polar Format ISAR image (decoupled 2-D IFFT) — the
    industry-standard formation used by FFT-based ISAR tools.

    Treats `S(θ, f)` *as if* `(θ, f)` were Cartesian k-space coordinates
    (`k_x ∝ 2 f_c sin θ ≈ 2 f_c θ`, `k_y ∝ 2 f - 2 f_c`) and runs two
    independent 1-D IFFTs — over frequency to get range, then over azimuth
    to get cross-range. No polar-to-Cartesian remap, so the algorithm has
    no `tan(θ)` step and tolerates any aperture (including full 360°).
    The small-angle identification distorts scatterer positions away from
    broadside — the standard trade all FFT-based ISAR tools share.

    Both axes are zero-padded to `next_fast_len` (with a floor for display
    smoothness). Padding never changes the scene extent — only the pixel
    pitch — and the pad-ratio rescale below keeps amplitudes on the
    canonical unpadded `1 / (n_az · n_freq)` ifft2 convention, so absolute
    dB values line up with other FFT-based tools without an offset.

    np.fft.ifft's `exp(+j·2π·m·n/N)` kernel matches the physics convention
    `S(θ, f) ∝ exp(-j·2k·r)`, so peak positions land at +(x, y).
    """
    n_az = theta.size
    n_freq = freq_hz.size
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."

    # Single precision throughout: halves memory traffic on 0.01°-step files
    # (36000×1701 slices) and scipy's pocketfft keeps complex64 native.
    win_az = _window_array(window_name, n_az).astype(np.float32)
    win_freq = _window_array(window_name, n_freq).astype(np.float32)
    rcs_windowed = np.asarray(rcs_polar, dtype=np.complex64) * win_az[:, None]
    rcs_windowed *= win_freq[None, :]
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    if sample_weights is None:
        coherent_gain = float(win_az.sum()) * float(win_freq.sum())
    else:
        weights = np.asarray(sample_weights, dtype=np.float32)
        if weights.shape != rcs_windowed.shape:
            raise ValueError("ISAR sample weights must match the phase-history shape")
        coherent_gain = float(np.sum(weights * win_az[:, None] * win_freq[None, :]))
    if coherent_gain <= 0.0:
        return "ISAR selected data has zero weighted aperture coverage."

    # Pad to fast FFT lengths: primes (e.g. 1601 frequencies) fall back to
    # Bluestein and are several times slower; the display floor keeps small
    # selections from rendering as a handful of blocky pixels.
    n_az_fft = _next_fast_len(max(n_az, 256))
    n_freq_fft = _next_fast_len(max(n_freq, 256))

    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    range_az = _ifft(rcs_windowed, n=n_freq_fft, axis=1)
    del rcs_windowed
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    isar_complex = _ifft(range_az, n=n_az_fft, axis=0)
    del range_az
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    isar_complex = np.fft.fftshift(isar_complex, axes=(0, 1))
    # Undo the padded 1/(n_az_fft·n_freq_fft) so amplitudes match the
    # canonical unpadded ifft2 normalisation.
    # Normalize by the actual coherent aperture gain, not merely the number
    # of samples. This keeps a unit point target at 0 dB for every taper and
    # when gridding/missing-data masks reduce the usable k-space support.
    isar_complex *= (n_az_fft * n_freq_fft) / coherent_gain

    x_range, y_range = _scene_axes(
        n_az_fft, n_freq_fft, theta, freq_hz, df, unit_scale,
        elevation_deg=elevation_deg,
    )
    return isar_complex, x_range, y_range


def _scene_axes(
    n_az_fft: int,
    n_freq_fft: int,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
    *,
    elevation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-range / range axes of the (padded, fftshifted) image grid —
    shared by the FFT and sparse reconstructions, which image onto the same
    k-space geometry."""
    c0 = 299_792_458.0
    projection = abs(float(np.cos(np.deg2rad(elevation_deg))))
    if projection < 1.0e-6:
        raise ValueError(
            "Azimuth ISAR is degenerate at elevation ±90° for a horizontal 2-D image plane."
        )
    dtheta = float(np.mean(np.diff(theta)))
    f_c = float(np.mean(freq_hz))
    y_range = (
        np.fft.fftshift(np.fft.fftfreq(n_freq_fft, d=df))
        * (c0 / (2.0 * projection)) * unit_scale
    )
    cross_freq_grid_d = (np.arange(n_az_fft) - n_az_fft // 2) / (n_az_fft * dtheta)
    x_range = (
        cross_freq_grid_d
        * (c0 / (2.0 * max(f_c, 1.0) * projection)) * unit_scale
    )
    return x_range, y_range


def _pfa_regrid_azimuth(
    S: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    target_q: np.ndarray | None = None,
    *,
    cancel_check=None,
) -> np.ndarray:
    """Keystone / polar-format regrid: per frequency row, resample the azimuth
    axis so the cross-range wavenumber k_x = 2·k_n·sin(ψ) (ψ measured from the
    aperture center) lands on the SAME uniform grid at every frequency.

    Without this, the decoupled Fourier model is only exact at the band
    center: at 20% fractional bandwidth the k_x mismatch reaches ~10%, which
    smears a point scatterer over several pixels no matter how good the
    solver is. With it, the separable model the sparse solver inverts matches
    the physics to second order (residual: range curvature, O(ψ²)).

    Rows lose a sliver at the ends where k_n < k̄ shrinks the source support;
    those samples zero-fill (standard PFA behavior). Skipped by the caller
    for apertures ≥ 90° where sin(ψ) stops being monotonic.
    """
    theta = np.asarray(theta, dtype=float)
    psi = theta - float(np.mean(theta))
    if target_q is None:
        target_q = psi
    else:
        target_q = np.asarray(target_q, dtype=float)
    k_ratio = np.asarray(freq_hz, dtype=float) / max(float(np.mean(freq_hz)), 1.0)
    sin_psi = np.sin(psi)
    out = np.empty_like(S)
    for n in range(S.shape[1]):
        if n % 16 == 0 and _cancel_requested(cancel_check):
            raise InterruptedError("ISAR computation superseded")
        src = k_ratio[n] * sin_psi
        row = S[:, n]
        if np.iscomplexobj(row):
            out[:, n] = (
                np.interp(target_q, src, row.real, left=0.0, right=0.0)
                + 1j * np.interp(target_q, src, row.imag, left=0.0, right=0.0)
            )
        else:
            out[:, n] = np.interp(target_q, src, row, left=0.0, right=0.0)
    if _cancel_requested(cancel_check):
        raise InterruptedError("ISAR computation superseded")
    return out


def _interp_uniform_axis0(data: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Four-point interpolation along axis 0 at per-column sample coordinates.

    Cubic Lagrange interpolation is used in the interior and linear at the
    two boundary intervals. Unlike phase/angle interpolation, operating on
    the complex field is continuous through phase wraps.
    """
    data = np.asarray(data)
    coordinates = np.asarray(coordinates, dtype=float)
    if data.ndim != 2 or coordinates.ndim != 2 or data.shape[1] != coordinates.shape[1]:
        raise ValueError("Uniform interpolation expects (samples, columns) arrays")
    n = data.shape[0]
    valid = (coordinates >= 0.0) & (coordinates <= n - 1)
    base = np.clip(np.floor(coordinates).astype(np.int64), 0, n - 2)
    t = coordinates - base
    if data.dtype in (np.complex64, np.float32):
        t = t.astype(np.float32)
    y0 = np.take_along_axis(data, base, axis=0)
    y1 = np.take_along_axis(data, base + 1, axis=0)
    out = y0 + (y1 - y0) * t
    interior = (base >= 1) & (base <= n - 3)
    if np.any(interior):
        ym1 = np.take_along_axis(data, np.clip(base - 1, 0, n - 1), axis=0)
        y2 = np.take_along_axis(data, np.clip(base + 2, 0, n - 1), axis=0)
        wm1 = -t * (t - 1.0) * (t - 2.0) / 6.0
        w0 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
        w1 = -(t + 1.0) * t * (t - 2.0) / 2.0
        w2 = (t + 1.0) * t * (t - 1.0) / 6.0
        cubic = ym1 * wm1 + y0 * w0 + y1 * w1 + y2 * w2
        out[interior] = cubic[interior]
    out[~valid] = 0
    return out


def _cartesian_pfa_axes(
    theta: np.ndarray,
    freq_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interpolation q and exact output axes for Cartesian PFA."""

    theta = np.asarray(theta, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float)
    if theta.size < 2 or freq_hz.size < 2:
        raise ValueError("Accurate PFA requires at least two azimuth and frequency samples")
    psi = theta - float(np.mean(theta))
    if float(psi[-1] - psi[0]) >= np.pi / 2.0:
        raise ValueError("Accurate Cartesian PFA requires an aperture narrower than 90°")
    fc = float(np.mean(freq_hz))
    ratio_min = float(freq_hz[0] / fc)
    q = np.linspace(
        ratio_min * np.sin(psi[0]), ratio_min * np.sin(psi[-1]), theta.size
    )
    u = fc * q
    max_abs_u = float(np.max(np.abs(u)))
    v_max_sq = float(freq_hz[-1] ** 2 - max_abs_u**2)
    if v_max_sq <= float(freq_hz[0] ** 2):
        raise ValueError("Selected aperture/band has no common Cartesian PFA support")
    v = np.linspace(float(freq_hz[0]), np.sqrt(v_max_sq), freq_hz.size)
    # _scene_axes derives cross-range spacing from mean(frequency)*dtheta.
    # Preserve the true U spacing when its returned frequency coordinate is V.
    axis_q = q * fc / float(np.mean(v))
    return q, axis_q, v


def _pfa_regrid_cartesian(
    S: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    *,
    row_block: int = 128,
    cancel_check=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid polar samples onto an inscribed uniform Cartesian k-space grid.

    The fast keystone path makes ``U = f_c q`` uniform but leaves
    ``V = sqrt(f**2-U**2)`` coupled to cross-range. This second interpolation
    makes V uniform as well, removing the O(psi**2) range-curvature defocus.
    The grid is inscribed in the measured polar support so no artificial
    zero-filled wedge changes its coherent gain.
    """
    theta = np.asarray(theta, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float)
    if _cancel_requested(cancel_check):
        raise InterruptedError("ISAR computation superseded")
    psi = theta - float(np.mean(theta))
    fc = float(np.mean(freq_hz))
    q, axis_q, v = _cartesian_pfa_axes(theta, freq_hz)
    az_grid = np.empty_like(S)
    dpsi = float(np.mean(np.diff(psi)))
    ratios = freq_hz / fc
    # Invert q=(f/fc)sin(psi) analytically, then interpolate the complex
    # field with a four-point stencil. This avoids the amplitude droop of
    # piecewise-linear gridding when phase advances appreciably per sample.
    for start in range(0, freq_hz.size, max(int(row_block), 1)):
        if _cancel_requested(cancel_check):
            raise InterruptedError("ISAR computation superseded")
        stop = min(start + max(int(row_block), 1), freq_hz.size)
        arg = q[:, None] / ratios[None, start:stop]
        required_psi = np.arcsin(np.clip(arg, -1.0, 1.0))
        coordinates = (required_psi - psi[0]) / dpsi
        block = _interp_uniform_axis0(np.asarray(S)[:, start:stop], coordinates)
        block[np.abs(arg) > 1.0] = 0
        az_grid[:, start:stop] = block

    u = fc * q
    out = np.empty_like(az_grid)
    for start in range(0, q.size, max(int(row_block), 1)):
        if _cancel_requested(cancel_check):
            raise InterruptedError("ISAR computation superseded")
        stop = min(start + max(int(row_block), 1), q.size)
        required_f = np.sqrt(v[None, :] ** 2 + u[start:stop, None] ** 2)
        coordinates = (required_f - freq_hz[0]) / float(np.mean(np.diff(freq_hz)))
        out[start:stop] = _interp_uniform_axis0(
            az_grid[start:stop].T, coordinates.T
        ).T
    if _cancel_requested(cancel_check):
        raise InterruptedError("ISAR computation superseded")
    return out, axis_q, v


def _soft_threshold_complex(x: np.ndarray, t: float) -> np.ndarray:
    """Complex soft-thresholding: shrink magnitudes by `t`, keep phases —
    the proximal operator of t·Σ|x_i| for complex x."""
    mag = np.abs(x)
    scale = np.maximum(mag - t, 0.0) / np.maximum(mag, 1e-30)
    return x * scale.astype(np.float32)


# Sparse reconstructions beyond this many image pixels would need tens of
# seconds per solve; steer the user to a sub-aperture / sub-band instead.
_SPARSE_MAX_PIXELS = 16_000_000


def _environment_mebibytes(name: str, default: int, minimum: int) -> int:
    """Read a bounded MiB setting without making a bad environment fatal."""

    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(value, minimum)


# This is a peak *additional working-set* budget, not an RSS claim.  It covers
# the selected complex slice, regridding temporaries, FFT/FISTA buffers, and
# already-retained results.  The preprocess cache has its own independent byte
# bound below.  Workstations with deliberately larger RAM can opt in via the
# environment rather than silently attempting a multi-gigabyte allocation.
_ISAR_WORKING_SET_LIMIT = (
    _environment_mebibytes("GRIM_ISAR_WORKING_SET_MB", 2048, 256) * 1024**2
)
_COMPOSITE_GRID_SIDE = 1024
_COMPOSITE_SCRATCH_BYTES = 64 * 1024**2


def _format_binary_bytes(value: int) -> str:
    value = max(int(value), 0)
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def _estimate_band_working_set_bytes(
    n_azimuth: int,
    n_frequency: int,
    *,
    reconstruction: str,
    retain_complex: bool,
) -> int:
    """Conservative byte estimate for one coherent band formation.

    Coefficients account for simultaneously-live arrays, not file size.  They
    deliberately include FFT-library scratch and the float64 source slice that
    may exist briefly while RcsGrid constructs an authoritative complex slice.
    """

    n_azimuth = max(int(n_azimuth), 1)
    n_frequency = max(int(n_frequency), 1)
    source_cells = n_azimuth * n_frequency
    n_az_fft = _next_fast_len(max(n_azimuth, 256))
    n_freq_fft = _next_fast_len(max(n_frequency, 256))
    image_cells = n_az_fft * n_freq_fft
    reconstruction = str(reconstruction).strip().lower()

    # Complex source, finite mask, weights, conversion and one simultaneous
    # regrid/PFA output.  Accurate PFA's block temporaries fit inside this
    # conservative per-source-cell allowance.
    preprocess_peak = 64 * source_cells
    retained_output = (12 if retain_complex else 4) * image_cells
    if reconstruction == "sparse":
        # FISTA and support debias can simultaneously hold X/Y/gradient/new
        # iterates plus adjoint, residual and CG buffers.
        formation_peak = 24 * source_cells + 144 * image_cells
    elif reconstruction in {"fft", "fast", "accurate"}:
        formation_peak = 32 * source_cells + 40 * image_cells
    else:
        raise ValueError(f"unsupported ISAR reconstruction {reconstruction!r}")
    return int(max(preprocess_peak, formation_peak + retained_output))


def _validate_isar_working_set(
    estimated_bytes: int,
    *,
    operation: str,
    resident_bytes: int = 0,
    limit_bytes: int | None = None,
) -> int:
    """Validate estimated peak bytes and return the total used for reporting."""

    estimated_bytes = max(int(estimated_bytes), 0)
    resident_bytes = max(int(resident_bytes), 0)
    total = estimated_bytes + resident_bytes
    limit = _ISAR_WORKING_SET_LIMIT if limit_bytes is None else int(limit_bytes)
    if total > limit:
        raise ValueError(
            f"{operation} needs an estimated {_format_binary_bytes(total)} peak "
            f"working set ({_format_binary_bytes(resident_bytes)} already retained; "
            f"limit {_format_binary_bytes(limit)}). Narrow the aperture/frequency "
            "band, export fewer panels, or raise GRIM_ISAR_WORKING_SET_MB only "
            "on a workstation with verified available RAM"
        )
    return total


def _result_array_bytes(results) -> int:
    """Count distinct retained NumPy buffers across completed band results."""

    total = 0
    seen: set[int] = set()
    for result in results:
        for value in result.values():
            if not isinstance(value, np.ndarray):
                continue
            root = value
            while isinstance(getattr(root, "base", None), np.ndarray):
                root = root.base
            identity = id(root)
            if identity in seen:
                continue
            seen.add(identity)
            total += int(root.nbytes)
    return total


def _cancel_requested(cancel_check) -> bool:
    return bool(cancel_check is not None and cancel_check())

# Keep only reusable, post-gridding phase histories. The byte bound prevents
# cine mode from retaining several platform-sized arrays. Set to zero to
# disable, or tune for a workstation with GRIM_ISAR_CACHE_MB.
_PREPROCESS_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_PREPROCESS_CACHE_LIMIT = max(int(os.environ.get("GRIM_ISAR_CACHE_MB", "512")), 0) * 1024**2
_PREPROCESS_CACHE_BYTES = 0
_PREPROCESS_CACHE_LOCK = threading.RLock()

_ISAR_TIME_CONVENTION = "+jwt"
_SENTRI_NATIVE_ELEVATION = "sentri_theta_top_zero"
_GRIM_ELEVATION = "grim_elevation_waterline_zero_top_positive"
_ISAR_GEOMETRY_IDENTITY_KEYS = (
    "measurement_geometry",
    "acquisition_geometry",
    "scattering_geometry",
    "radar_geometry",
)
_ISAR_PROPAGATION_METADATA_KEYS = (
    "measurement_domain",
    "field_domain",
    "range_type",
    "wavefront_geometry",
)
_ISAR_GEOMETRY_METADATA_KEYS = (
    *_ISAR_GEOMETRY_IDENTITY_KEYS,
    *_ISAR_PROPAGATION_METADATA_KEYS,
)
_ISAR_MOTION_METADATA_KEYS = (
    "motion_compensation",
    "motion_compensated",
    "phase_center_stability",
    "phase_center_motion",
    "range_alignment",
)
_ISAR_RANGE_PHASE_METADATA_KEYS = (
    "range_phase_convention",
    "phase_law",
    "amplitude_convention",
    "phase_reference",
)


def _declared_scalar_metadata(dataset, key: str) -> str:
    """Read one convention declaration without guessing between containers."""

    getter = getattr(dataset, "_declared_scalar_metadata", None)
    if callable(getter):
        declared_by_dataset = str(getter(key) or "").strip()
        if declared_by_dataset:
            return declared_by_dataset
        # Some legacy implementations collapse explicit scalar False/0 to a
        # blank string.  Fall through and preserve those values: for motion
        # metadata they mean "not compensated", not "undeclared".
    declared = []
    for container_name in ("units", "extra"):
        container = getattr(dataset, container_name, None) or {}
        if key not in container:
            continue
        value = np.asarray(container[key])
        if value.size != 1:
            raise ValueError(f"metadata {key!r} must be scalar")
        scalar = value.reshape(-1)[0]
        text = "" if scalar is None else str(scalar).strip()
        if text:
            declared.append(text)
    normalized = {" ".join(value.split()).casefold() for value in declared}
    if len(normalized) > 1:
        raise ValueError(f"dataset contains contradictory {key} metadata")
    return declared[0] if declared else ""


def _canonical_time_convention(dataset, value: str) -> str:
    canonicalizer = getattr(dataset, "_canonical_time_convention", None)
    if callable(canonicalizer):
        return str(canonicalizer(value))
    compact = (
        str(value or "").strip().casefold().replace("omega", "w")
        .replace("ω", "w").replace("*", "").replace(" ", "")
    )
    if "exp(+jwt)" in compact or "exp(jwt)" in compact:
        return "+jwt"
    if "exp(-jwt)" in compact:
        return "-jwt"
    return compact


def _declared_range_phase_sign(dataset) -> int | None:
    """Return -1/+1 for an explicit ``exp(±j*2*k*R)`` declaration.

    Generic amplitude labels and one-way ``exp(±j*k*R)`` references do not
    establish the two-way range law and therefore return ``None``.
    """

    signs = set()
    for key in _ISAR_RANGE_PHASE_METADATA_KEYS:
        declared = _declared_scalar_metadata(dataset, key)
        if not declared:
            continue
        compact = str(declared).casefold()
        compact = compact.replace("−", "-").replace("–", "-")
        compact = re.sub(r"[\s*·^{}()\[\]_=~]+", "", compact)
        matches = re.findall(r"(?:exp|e)([+-])j2(?:\.0+)?kr", compact)
        if "negativetwowayrangephase" in compact:
            matches.append("-")
        if "positivetwowayrangephase" in compact:
            matches.append("+")
        signs.update(-1 if match == "-" else 1 for match in matches)
    if len(signs) > 1:
        raise ValueError(
            "dataset contains contradictory two-way range-phase declarations"
        )
    return next(iter(signs), None)


def _isar_preflight_error(
    dataset,
    *,
    legacy_metadata_attested=False,
    flip_x: bool = False,
    flip_y: bool = False,
    undeclared_out: list[str] | None = None,
) -> str | None:
    """Return a user-facing reason that ``dataset`` is unsafe for ISAR.

    The implemented k-space geometry is GRIM conic azimuth/elevation and the
    IFFT sign assumes ``exp(+j*omega*t)`` with a monostatic phase history
    proportional to ``exp(-j*2*k*R)``. Missing declarations on a legacy file
    are recorded as user assumptions instead of blocking formation;
    contradictory or unsupported declarations still fail closed.

    ``legacy_metadata_attested`` remains accepted for compatibility with
    previously recorded scripts but is no longer required. When supplied,
    ``undeclared_out`` receives human-readable names for contract declarations
    that were absent from the source dataset.
    """

    if not isinstance(legacy_metadata_attested, (bool, np.bool_)):
        raise TypeError("legacy_metadata_attested must be True or False")
    if undeclared_out is not None and not isinstance(undeclared_out, list):
        raise TypeError("undeclared_out must be a list or None")
    if not isinstance(flip_x, (bool, np.bool_)) or not isinstance(
        flip_y, (bool, np.bool_)
    ):
        raise TypeError("flip_x and flip_y must be True or False")

    units = getattr(dataset, "units", None) or {}
    extra = getattr(dataset, "extra", None) or {}

    raw_frequency_unit = units.get("frequency")
    if raw_frequency_unit is None or not str(raw_frequency_unit).strip():
        return (
            "frequency units are missing. Declare Hz, kHz, MHz, or GHz; "
            "ISAR will not infer units from numeric magnitude"
        )
    try:
        _unit_to_hz_scale(str(raw_frequency_unit))
    except ValueError as exc:
        return str(exc)

    def normalized_semantics(value: str) -> str:
        return re.sub(
            r"[^a-z0-9]+", " ", str(value or "").strip().casefold()
        ).strip()

    def unsafe_semantics(value: str) -> bool:
        words = normalized_semantics(value).split()
        word_set = set(words)
        joined = " ".join(words)
        return bool(
            word_set.intersection({
                "none", "unfixed", "unstable", "varying", "variable",
                "moving", "uncompensated", "arbitrary",
            })
            or (
                any(word.startswith("drift") for word in words)
                and "no drift" not in joined
                and "without drift" not in joined
            )
            or any(
                phrase in joined
                for phrase in ("not fixed", "not compensated", "not aligned")
            )
        )

    def placeholder_semantics(value: str) -> bool:
        """Treat common vendor placeholders as absent, not physical claims."""

        return normalized_semantics(value) in {"na", "n a", "not applicable"}

    def recognized_far_field(value: str) -> bool:
        semantic = normalized_semantics(value)
        words = set(semantic.split())
        return bool(
            "far field" in semantic
            or "farfield" in words
            or "far zone" in semantic
            or "farzone" in words
            or "fraunhofer" in words
            or "plane wave" in semantic
            or "radiation zone" in semantic
            or semantic in {"far", "ff"}
        )

    def recognized_frequency_domain(value: str) -> bool:
        """Return whether a domain tag describes data representation only.

        ``measurement_domain=frequency-domain`` says that samples are indexed
        in frequency; it does not contradict, or establish, a separately
        declared near-/far-field propagation zone.
        """

        semantic = normalized_semantics(value)
        return bool(
            semantic in {"frequency", "frequency domain", "spectral domain"}
            or "frequency sweep" in semantic
            or "stepped frequency" in semantic
            or "frequency response" in semantic
        )

    def incompatible_geometry(value: str) -> bool:
        semantic = normalized_semantics(value)
        words = set(semantic.split())
        return bool(
            unsafe_semantics(value)
            or "not monostatic" in semantic
            or words.intersection(
                {
                    "bistatic",
                    "multistatic",
                    "fresnel",
                    "quasi",
                    "quasimonostatic",
                    "pseudo",
                    "pseudomonostatic",
                }
            )
            or "near field" in semantic
            or "nearfield" in words
            or "reactive near" in semantic
        )

    # Explicit incompatible geometry remains authoritative. Missing or
    # placeholder geometry is recorded below as a user-owned assumption.
    monostatic_declared = False
    far_field_declared = False
    for key in _ISAR_GEOMETRY_METADATA_KEYS:
        declared = _declared_scalar_metadata(dataset, key)
        if not declared:
            continue
        if placeholder_semantics(declared):
            continue
        semantic = normalized_semantics(declared)
        if any(word in semantic.split() for word in ("bistatic", "multistatic")):
            return (
                f"{key} declares {declared!r}; this ISAR implementation requires "
                "monostatic phase history"
            )
        if (
            "near field" in semantic
            or "nearfield" in semantic
            or "fresnel" in semantic
        ):
            return (
                f"{key} declares {declared!r}; this ISAR implementation requires "
                "far-field phase history"
            )
        if incompatible_geometry(declared):
            return (
                f"{key} is explicitly {declared!r}; verify and declare far-field "
                "monostatic geometry before ISAR"
            )
        geometry_identity_key = key in _ISAR_GEOMETRY_IDENTITY_KEYS
        if geometry_identity_key:
            if "monostatic" not in semantic.split():
                # Producer-specific, unknown, or otherwise unrecognized text
                # is ambiguous rather than a known physical contradiction.
                # Record the standard monostatic assumption below.
                continue
            monostatic_declared = True
            if recognized_far_field(declared):
                far_field_declared = True
        elif recognized_far_field(declared):
            far_field_declared = True
        elif recognized_frequency_domain(declared):
            # Neutral data-domain evidence. A separate declaration must still
            # establish far-field propagation below.
            continue
        else:
            # Unknown producer vocabulary is not proof of an incompatible
            # propagation domain. Treat it as undeclared and user-assumed.
            continue
    # ``complex_field_domain`` records the algebraic meaning of a complex
    # response (for example a support-referenced difference or a feature
    # delta), not necessarily its propagation zone.  Preserve those valid
    # downstream workflows while still rejecting an explicitly unsafe domain.
    complex_field_domain = _declared_scalar_metadata(
        dataset, "complex_field_domain"
    )
    if complex_field_domain and incompatible_geometry(complex_field_domain):
        return (
            "complex_field_domain declares an incompatible or unverified "
            f"response {complex_field_domain!r}; ISAR requires a verified "
            "far-field response"
        )
    if complex_field_domain and recognized_far_field(complex_field_domain):
        far_field_declared = True
    if any(
        key in extra
        for key in (
            "fixed_incident_azimuth_deg",
            "fixed_incident_elevation_deg",
            "fixed_observation_azimuth_deg",
            "fixed_observation_elevation_deg",
        )
    ):
        return (
            "dataset carries fixed incident/observation coordinates from a "
            "bistatic acquisition; monostatic ISAR cannot use that geometry"
        )

    stable_motion_declared = False
    for key in _ISAR_MOTION_METADATA_KEYS:
        declared = _declared_scalar_metadata(dataset, key)
        if not declared:
            continue
        if placeholder_semantics(declared):
            continue
        semantic = normalized_semantics(declared)
        # Most keys assert a positive safe state (compensated/stable/aligned),
        # while phase_center_motion asserts the opposite physical quantity.
        # Preserve that polarity: False/none means a fixed phase center, and
        # True means motion is present.
        inverse_polarity = key == "phase_center_motion"
        true_value = semantic in {"1", "true", "yes"}
        false_value = semantic in {"0", "false", "no", "none", "n a", "na"}
        explicitly_unsafe = bool(
            (true_value if inverse_polarity else false_value)
            or unsafe_semantics(declared)
        )
        if explicitly_unsafe:
            return (
                f"{key} declares {declared!r}; ISAR requires a stable, "
                "motion-compensated phase center"
            )
        motion_words = set(semantic.split())
        explicitly_safe = bool(
            (false_value if inverse_polarity else true_value)
            or "no motion" in semantic
            or "without motion" in semantic
            or "no drift" in semantic
            or motion_words.intersection(
                {"compensated", "stable", "static", "fixed", "aligned", "corrected"}
            )
        )
        if not explicitly_safe:
            # Unknown producer vocabulary is not proof of actual motion.
            # Treat it as an undeclared stable-center assumption.
            continue
        stable_motion_declared = True

    def canonical_angular(value) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        return {
            "": "conic",
            "az_el": "conic",
            "azimuth_elevation": "conic",
            "spherical": "conic",
            "gc": "great_circle",
            "greatcircle": "great_circle",
        }.get(text, text)

    angular_declarations = {
        canonical_angular(container.get("angular_coordinate_system"))
        for container in (units, extra)
        if str(container.get("angular_coordinate_system", "") or "").strip()
    }
    if len(angular_declarations) > 1:
        return "units and extra contain contradictory angular coordinate systems"
    angular_getter = getattr(dataset, "angular_coordinate_system", None)
    angular_system = next(iter(angular_declarations), None) or (
        canonical_angular(angular_getter()) if callable(angular_getter) else "conic"
    )
    if angular_system != "conic":
        if angular_system == "great_circle":
            return (
                "great-circle aspect/pitch is not an azimuth/elevation ISAR "
                "aperture. Convert the dataset to GRIM conic coordinates first; "
                "non-equatorial PTM cuts require a physically defined conversion"
            )
        return (
            f"angular coordinate system {angular_system or '<unspecified>'!r} is "
            "unsupported; convert the dataset to GRIM conic azimuth/elevation first"
        )

    elevation_declarations = {
        str(value or "").strip().lower()
        for value in (
            units.get("elevation_coordinate_convention", ""),
            extra.get("sentri_elevation_convention", ""),
        )
        if str(value or "").strip()
    }
    if len(elevation_declarations) > 1:
        return "units and extra contain contradictory elevation conventions"
    elevation_convention = next(iter(elevation_declarations), "")
    if elevation_convention == _SENTRI_NATIVE_ELEVATION:
        return (
            "native SENTRi theta uses 0 deg top-down and 90 deg waterline. "
            "Run Geometry & Units > Convert SENTRi Coordinates before ISAR"
        )
    if elevation_convention and elevation_convention != _GRIM_ELEVATION:
        return (
            f"elevation convention {elevation_convention!r} is unsupported; "
            "convert or relabel it to GRIM signed elevation before ISAR"
        )

    phase_reference = _declared_scalar_metadata(dataset, "phase_reference")
    if placeholder_semantics(phase_reference):
        phase_reference = ""
    declared_time = _declared_scalar_metadata(dataset, "time_convention")
    declared_time_key = (
        _canonical_time_convention(dataset, declared_time) if declared_time else ""
    )
    phase_time_key = (
        _canonical_time_convention(dataset, phase_reference)
        if phase_reference else ""
    )
    phase_declares_time = phase_time_key in {"+jwt", "-jwt"}
    semantic_phase = normalized_semantics(phase_reference)
    phase_words = set(semantic_phase.split())
    phase_has_fixed_reference = bool(
        phase_words.intersection({"origin", "center", "centre"})
        or "fixed reference" in semantic_phase
        or "reference plane" in semantic_phase
        or "calibrated reference" in semantic_phase
    )

    if phase_reference and (
        semantic_phase in {
            "none", "na", "arbitrary", "n a",
        }
        or unsafe_semantics(phase_reference)
    ):
        return (
            f"phase reference {phase_reference!r} is not a verified fixed "
            "phase center; calibrate/motion-compensate before ISAR"
        )

    if declared_time and declared_time_key == "-jwt":
        return (
            "ISAR requires the exp(+j*omega*t) time convention; dataset declares "
            f"{declared_time!r}. Convert/conjugate the phase history explicitly "
            "before imaging"
        )
    if phase_declares_time and phase_time_key == "-jwt":
        return (
            "ISAR requires the exp(+j*omega*t) time convention, but the phase "
            f"reference declares {phase_reference!r}. Convert/conjugate the phase "
            "history explicitly before imaging"
        )
    if (
        declared_time_key == _ISAR_TIME_CONVENTION
        and phase_declares_time
        and phase_time_key != declared_time_key
    ):
        return "time-convention and phase-reference declarations contradict each other"

    range_phase_sign = _declared_range_phase_sign(dataset)
    if range_phase_sign == 1 and not (bool(flip_x) and bool(flip_y)):
        return (
            "dataset explicitly declares S~exp(+j*2*k*R), while the default ISAR "
            "axes assume S~exp(-j*2*k*R). Convert/conjugate the phase history, "
            "or deliberately enable both Flip X and Flip Y after validating a "
            "known asymmetric target"
        )

    missing = []
    if not monostatic_declared:
        missing.append("far-field monostatic acquisition geometry")
    elif not far_field_declared:
        missing.append("a far-field/Fraunhofer measurement domain")
    if not stable_motion_declared:
        missing.append("a stable or motion-compensated phase center")
    if not phase_has_fixed_reference:
        missing.append("a fixed phase reference/center")
    if declared_time_key != _ISAR_TIME_CONVENTION and not (
        phase_declares_time and phase_time_key == _ISAR_TIME_CONVENTION
    ):
        missing.append("the exp(+j*omega*t) time convention")
    if range_phase_sign is None:
        missing.append("the S~exp(-j*2*k*R) two-way range-phase convention")
    if undeclared_out is not None:
        undeclared_out.extend(missing)
    return None


def _isar_metadata_token(dataset, elevation_index: int, polarization_index: int) -> tuple:
    """Cache identity for physical conventions not encoded in numeric axes."""

    units = getattr(dataset, "units", None) or {}
    extra = getattr(dataset, "extra", None) or {}
    angular_getter = getattr(dataset, "angular_coordinate_system", None)
    angular_system = (
        angular_getter() if callable(angular_getter)
        else units.get("angular_coordinate_system", "conic")
    )
    gc_getter = getattr(dataset, "great_circle_coordinate_convention", None)
    gc_convention = gc_getter() if callable(gc_getter) else units.get(
        "great_circle_coordinate_convention", ""
    )
    orientation_getter = getattr(dataset, "angular_frame_orientation_deg", None)
    orientation = orientation_getter() if callable(orientation_getter) else (
        units.get("angular_roll_deg", extra.get("ptm_roll", 0.0)),
        units.get("angular_tilt_deg", extra.get("ptm_tilt", 0.0)),
    )
    scalar_fields = tuple(
        _declared_scalar_metadata(dataset, key)
        for key in (
            "phase_reference", "time_convention", "polarization_basis",
            *_ISAR_GEOMETRY_METADATA_KEYS,
            *_ISAR_MOTION_METADATA_KEYS,
            *_ISAR_RANGE_PHASE_METADATA_KEYS,
        )
    )
    return (
        str(angular_system),
        str(gc_convention),
        tuple(float(value) for value in orientation),
        str(units.get("elevation_coordinate_convention", extra.get(
            "sentri_elevation_convention", ""
        ))),
        str(units.get("azimuth", "deg")),
        str(units.get("elevation", "deg")),
        scalar_fields,
        float(np.asarray(dataset.elevations)[int(elevation_index)]),
        str(np.asarray(dataset.polarizations)[int(polarization_index)]),
    )


def _array_token(values) -> tuple:
    arr = np.ascontiguousarray(values)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(repr(arr.shape).encode("ascii"))
    digest.update(memoryview(arr).cast("B"))
    return (arr.dtype.str, arr.shape, digest.digest())


def _selected_data_token(
    dataset,
    azimuth_indices,
    elevation_index: int,
    frequency_indices,
    polarization_index: int,
    *,
    cancel_check=None,
) -> bytes | None:
    """Exact digest of the selected authoritative complex source.

    RcsGrid arrays are intentionally public and may be changed in place, so an
    object id or axis-only cache key is not a safe data revision. Hashing one
    frequency row at a time bounds temporary memory while detecting any selected
    sample change and also makes reuse across object-id recycling harmless. When
    a complete raw real/imaginary pair is authoritative, hash that pair instead
    of the display power/phase arrays; this mirrors ``RcsGrid.rcs_slice``.
    """

    digest = hashlib.blake2b(digest_size=20)
    if _cancel_requested(cancel_check):
        return None
    azimuth_indices = np.asarray(azimuth_indices, dtype=np.intp)
    frequency_indices = np.asarray(frequency_indices, dtype=np.intp)
    digest.update(b"grim-isar-selected-source-v2\0")

    def update_array(label: bytes, values) -> None:
        array = np.ascontiguousarray(values)
        digest.update(label + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(repr(array.shape).encode("ascii") + b"\0")
        digest.update(memoryview(array).cast("B"))

    update_array(
        b"azimuth-values-native",
        np.asarray(dataset.azimuths)[azimuth_indices],
    )
    update_array(
        b"frequency-values-native",
        np.asarray(dataset.frequencies)[frequency_indices],
    )
    update_array(
        b"elevation-value-native",
        np.asarray(dataset.elevations)[[int(elevation_index)]],
    )
    polarization = str(
        np.asarray(dataset.polarizations)[int(polarization_index)]
    ).encode("utf-8")
    digest.update(b"polarization\0" + len(polarization).to_bytes(8, "little"))
    digest.update(polarization)
    for key in (
        "azimuth",
        "elevation",
        "frequency",
        "rcs_linear_quantity",
        "rcs_log_unit",
        "phase_reference",
        "time_convention",
        "polarization_basis",
        "range_phase_convention",
        "phase_law",
        "amplitude_convention",
    ):
        if key in {"phase_reference", "time_convention", "polarization_basis"}:
            value = _declared_scalar_metadata(dataset, key)
        else:
            value = str((getattr(dataset, "units", None) or {}).get(key, ""))
        encoded = str(value).strip().encode("utf-8")
        digest.update(key.encode("ascii") + b"\0")
        digest.update(len(encoded).to_bytes(8, "little") + encoded)
    if _cancel_requested(cancel_check):
        return None
    raw_pair_getter = getattr(dataset, "_complete_authoritative_raw_arrays", None)
    raw_pair = raw_pair_getter() if callable(raw_pair_getter) else None
    if raw_pair is None:
        sources = (
            (b"power", dataset.rcs_power),
            (b"phase", dataset.rcs_phase),
        )
        digest.update(b"power-phase")
    else:
        sources = (
            (b"raw-real", raw_pair[0]),
            (b"raw-imag", raw_pair[1]),
        )
        digest.update(b"authoritative-raw")
        linear_quantity = getattr(dataset, "linear_quantity", None)
        digest.update(str(
            linear_quantity() if callable(linear_quantity) else ""
        ).encode("utf-8"))

    for label, values in sources:
        if _cancel_requested(cancel_check):
            return None
        source = np.asarray(values)
        digest.update(label)
        digest.update(source.dtype.str.encode("ascii"))
        digest.update(
            repr((len(azimuth_indices), len(frequency_indices))).encode("ascii")
        )
        for row_number, azimuth_index in enumerate(azimuth_indices):
            if row_number % 16 == 0 and _cancel_requested(cancel_check):
                return None
            row = np.ascontiguousarray(
                source[
                    int(azimuth_index),
                    int(elevation_index),
                    frequency_indices,
                    int(polarization_index),
                ]
            )
            digest.update(memoryview(row).cast("B"))
    if _cancel_requested(cancel_check):
        return None
    return digest.digest()


def _preprocess_cache_get(key: tuple):
    # Headless callers may form independent images concurrently.  OrderedDict
    # mutation and the separate byte ledger must move as one critical section.
    with _PREPROCESS_CACHE_LOCK:
        value = _PREPROCESS_CACHE.get(key)
        if value is not None:
            _PREPROCESS_CACHE.move_to_end(key)
        return value


def _preprocess_cache_put(key: tuple, value: dict) -> None:
    global _PREPROCESS_CACHE_BYTES
    if _PREPROCESS_CACHE_LIMIT <= 0:
        return
    nbytes = sum(v.nbytes for v in value.values() if isinstance(v, np.ndarray))
    if nbytes > _PREPROCESS_CACHE_LIMIT // 2:
        return
    with _PREPROCESS_CACHE_LOCK:
        previous = _PREPROCESS_CACHE.pop(key, None)
        if previous is not None:
            _PREPROCESS_CACHE_BYTES -= int(previous.get("_cache_nbytes", 0))
        value["_cache_nbytes"] = nbytes
        _PREPROCESS_CACHE[key] = value
        _PREPROCESS_CACHE_BYTES += nbytes
        while _PREPROCESS_CACHE_BYTES > _PREPROCESS_CACHE_LIMIT and _PREPROCESS_CACHE:
            _, evicted = _PREPROCESS_CACHE.popitem(last=False)
            _PREPROCESS_CACHE_BYTES -= int(evicted.get("_cache_nbytes", 0))


def _compute_band_sparse_l1(
    rcs_polar: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
    strength: float,
    n_iters: int,
    *,
    sample_weights: np.ndarray | None = None,
    elevation_deg: float = 0.0,
    cancel_check=None,
    convergence_tol: float = 1.0e-4,
):
    """Experimental fixed-λ complex LASSO image reconstruction.

    The gridded phase history ``S`` is modelled by the same partial Fourier
    operator used by the fast PFA path and FISTA solves

        min_X  ½‖W(A·X − S)‖₂² + λ‖X‖₁,
        λ = strength · ‖AᴴW²S‖_∞.

    This is a penalised LASSO problem, *not* residual-constrained BPDN and not
    target/contaminant decomposition.  Pixel sparsity may suppress sidelobes,
    noise, weak real returns, or distributed scattering alike; it cannot
    identify or remove a pylon, bird, cavity return, or other physical source.

    FISTA uses an objective-triggered adaptive restart and a primal/dual-gap
    certificate.  A support-restricted least-squares refit can remove the
    LASSO amplitude bias after safeguards against underdetermined or unstable
    supports.  The taper selection intentionally does not apply to this mode.
    """
    strength, n_iters = _sparse_parameters(strength, n_iters)
    try:
        convergence_tol = float(convergence_tol)
    except (TypeError, ValueError) as exc:
        raise ValueError("sparse convergence_tol must be a finite number in (0, 1)") from exc
    if not np.isfinite(convergence_tol) or not 0.0 < convergence_tol < 1.0:
        raise ValueError("sparse convergence_tol must be a finite number in (0, 1)")

    n_az = theta.size
    n_freq = freq_hz.size
    n_az_fft = _next_fast_len(max(n_az, 256))
    n_freq_fft = _next_fast_len(max(n_freq, 256))
    n_pad = n_az_fft * n_freq_fft
    if n_pad > _SPARSE_MAX_PIXELS:
        return (
            f"Sparse L1 grid too large ({n_az_fft}×{n_freq_fft}). Narrow the "
            "selection (Aperture / Freq Band) or switch Recon to FFT."
        )

    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    S = np.ascontiguousarray(rcs_polar, dtype=np.complex64)
    if sample_weights is None:
        weights = np.ones(S.shape, dtype=np.float32)
    else:
        weights = np.clip(np.asarray(sample_weights, dtype=np.float32), 0.0, 1.0)
        if weights.shape != S.shape:
            raise ValueError("Sparse ISAR sample weights must match the phase-history shape")
    weights_sq = weights * weights
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."

    def forward(X: np.ndarray) -> np.ndarray:
        # image -> predicted phase history at the measured (θ, f) samples
        return _fft2(np.fft.ifftshift(X))[:n_az, :n_freq]

    def adjoint(y: np.ndarray) -> np.ndarray:
        Z = np.zeros((n_az_fft, n_freq_fft), dtype=np.complex64)
        Z[:n_az, :n_freq] = y
        return np.fft.fftshift(_ifft2(Z)).astype(np.complex64) * np.float32(n_pad)

    # The sampling mask is part of the measurement operator. Missing samples
    # must not be interpreted as measured zero fields.
    lip = float(n_pad)
    matched = adjoint(weights_sq * S)
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    matched_peak = float(np.abs(matched).max())
    lam = strength * matched_peak
    if lam <= 0.0:
        return "Sparse L1 reconstruction: selected data is identically zero."

    weighted_data = weights * S
    weighted_data_norm = float(
        np.sqrt(np.sum(np.abs(weighted_data) ** 2, dtype=np.float64))
    )

    def lasso_diagnostics(value: np.ndarray) -> dict:
        predicted = forward(value)
        residual = weights * (predicted - S)
        residual_sq = float(np.sum(np.abs(residual) ** 2, dtype=np.float64))
        l1_norm = float(np.sum(np.abs(value), dtype=np.float64))
        objective = 0.5 * residual_sq + lam * l1_norm

        # For min .5||B x-b||^2 + lambda||x||_1, a scaled residual is dual
        # feasible when ||B^H u||_inf <= lambda.  Its objective supplies an
        # actual optimality certificate; small iterate changes alone can be
        # misleading for the highly underdetermined partial-Fourier operator.
        dual_gradient = adjoint(weights * residual)
        dual_norm = float(np.abs(dual_gradient).max())
        dual_scale = min(1.0, lam / max(dual_norm, 1.0e-30))
        dual_value = residual * np.float32(dual_scale)
        dual_objective = -0.5 * float(
            np.sum(np.abs(dual_value) ** 2, dtype=np.float64)
        ) - float(
            np.real(
                np.sum(
                    np.conjugate(weighted_data) * dual_value,
                    dtype=np.complex128,
                )
            )
        )
        duality_gap = max(objective - dual_objective, 0.0)
        relative_gap = duality_gap / max(
            abs(objective), abs(dual_objective), 1.0e-30
        )
        residual_norm = float(np.sqrt(residual_sq))
        return {
            "objective": objective,
            "duality_gap": duality_gap,
            "relative_duality_gap": relative_gap,
            "residual_norm": residual_norm,
            "relative_residual_norm": residual_norm
            / max(weighted_data_norm, 1.0e-30),
        }

    X = np.zeros((n_az_fft, n_freq_fft), dtype=np.complex64)
    Y = X
    t_momentum = 1.0
    iterations_used = 0
    restart_count = 0
    previous_checkpoint_objective = float("inf")
    converged = False
    diagnostics = None
    for iteration in range(n_iters):
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        grad = adjoint(weights_sq * (forward(Y) - S))
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        X_new = _soft_threshold_complex(Y - grad / np.float32(lip), lam / lip)
        if not np.all(np.isfinite(X_new)):
            return "Sparse L1 reconstruction failed: solver produced non-finite values."
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_momentum**2))
        Y_new = X_new + np.float32((t_momentum - 1.0) / t_new) * (X_new - X)
        iterations_used = iteration + 1
        checkpoint = (
            iterations_used >= 10 and iterations_used % 5 == 0
        ) or iterations_used == n_iters
        if checkpoint:
            diagnostics = lasso_diagnostics(X_new)
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            objective = diagnostics["objective"]
            if objective > previous_checkpoint_objective * (1.0 + 1.0e-7):
                # FISTA can oscillate after it identifies a small support.
                # Drop the extrapolation when the certified objective rises.
                Y_new = X_new
                t_new = 1.0
                restart_count += 1
            previous_checkpoint_objective = objective
            if diagnostics["relative_duality_gap"] <= convergence_tol:
                X = X_new
                converged = True
                break
        X, Y, t_momentum = X_new, Y_new, t_new

    if diagnostics is None or not converged:
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        diagnostics = lasso_diagnostics(X)
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."

    lasso_residual_norm = diagnostics["residual_norm"]
    lasso_peak = float(np.abs(X).max())
    # Do not let floating-point/proximal crumbs make the restricted normal
    # equations nearly singular.  The -60 dB amplitude floor is well below
    # the weakest coefficient normally retained by the public lambda range.
    support_threshold = lasso_peak * 1.0e-3
    support = np.abs(X) > support_threshold
    support_size = int(np.count_nonzero(support))
    measured_samples = int(np.count_nonzero(weights > 1.0e-6))
    debias_applied = False
    debias_iterations = 0
    debias_status = "skipped: empty recovered support"

    # Debias: the l1 shrinkage biases recovered amplitudes low.
    # Re-fit least squares ON THE RECOVERED SUPPORT ONLY, penalty removed, so
    # off-support pixels stay zero.  Never attempt an underdetermined refit,
    # and retain the LASSO result if CG breaks down, explodes, or makes the
    # weighted data residual worse.
    if support_size > measured_samples:
        debias_status = (
            f"skipped: support ({support_size}) exceeds measured samples "
            f"({measured_samples})"
        )
    elif support_size > 0:
        def normal_op(v: np.ndarray) -> np.ndarray:
            out = adjoint(weights_sq * forward(v))
            out[~support] = 0
            return out

        b = matched.copy()
        b[~support] = 0
        candidate = X.copy()
        candidate[~support] = 0
        r = b - normal_op(candidate)
        p = r.copy()
        rs_old = float(np.vdot(r, r).real)
        b_norm = float(np.vdot(b, b).real)
        cg_ok = np.isfinite(rs_old) and np.isfinite(b_norm)
        for cg_iteration in range(50):
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            if not cg_ok or rs_old <= 1.0e-10 * max(b_norm, 1.0e-30):
                break
            Ap = normal_op(p)
            denominator = float(np.vdot(p, Ap).real)
            scale_floor = (
                np.finfo(np.float32).eps
                * max(float(np.linalg.norm(p)) * float(np.linalg.norm(Ap)), 1.0e-30)
            )
            if not np.isfinite(denominator) or denominator <= scale_floor:
                cg_ok = False
                break
            alpha = rs_old / denominator
            candidate = candidate + np.complex64(alpha) * p
            r = r - np.complex64(alpha) * Ap
            rs_new = float(np.vdot(r, r).real)
            debias_iterations = cg_iteration + 1
            if not np.isfinite(rs_new):
                cg_ok = False
                break
            p = r + np.complex64(rs_new / max(rs_old, 1e-30)) * p
            rs_old = rs_new
        candidate[~support] = 0

        if cg_ok and np.all(np.isfinite(candidate)):
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            candidate_residual = weights * (forward(candidate) - S)
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            candidate_residual_norm = float(
                np.sqrt(
                    np.sum(np.abs(candidate_residual) ** 2, dtype=np.float64)
                )
            )
            candidate_peak = float(np.abs(candidate).max())
            stable_amplitude = candidate_peak <= max(lasso_peak * 1.0e3, 1.0e-20)
            residual_improved = candidate_residual_norm <= lasso_residual_norm * (1.0 + 1.0e-5)
            if stable_amplitude and residual_improved:
                X = candidate
                debias_applied = True
                debias_status = "applied"
            elif not stable_amplitude:
                debias_status = "skipped: unstable amplitude growth"
            else:
                debias_status = "skipped: refit did not improve weighted residual"
        else:
            debias_status = "skipped: conjugate-gradient breakdown"

    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    output_residual = weights * (forward(X) - S)
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    output_residual_norm = float(
        np.sqrt(np.sum(np.abs(output_residual) ** 2, dtype=np.float64))
    )
    sparse_status = (
        "converged by relative primal/dual gap"
        if converged
        else "maximum iterations reached; convergence not certified"
    )
    sparse_diagnostics = {
        "sparse_converged": converged,
        "sparse_status": sparse_status,
        "sparse_iterations": iterations_used,
        "sparse_restarts": restart_count,
        "sparse_lambda": lam,
        "sparse_objective": diagnostics["objective"],
        "sparse_duality_gap": diagnostics["duality_gap"],
        "sparse_relative_duality_gap": diagnostics["relative_duality_gap"],
        "sparse_lasso_residual_norm": lasso_residual_norm,
        "sparse_lasso_relative_residual_norm": diagnostics["relative_residual_norm"],
        "sparse_output_residual_norm": output_residual_norm,
        "sparse_output_relative_residual_norm": output_residual_norm
        / max(weighted_data_norm, 1.0e-30),
        "sparse_support_size": support_size,
        "sparse_support_threshold": support_threshold,
        "sparse_debias_applied": debias_applied,
        "sparse_debias_iterations": debias_iterations,
        "sparse_debias_status": debias_status,
    }

    # No rescale needed: a unit-amplitude scatterer produces |S| = 1 under the
    # forward model, so the recovered X IS the physical reflectivity (≈ 1 →
    # 0 dB), matching the FFT path's normalisation convention.
    x_range, y_range = _scene_axes(
        n_az_fft, n_freq_fft, theta, freq_hz, df, unit_scale,
        elevation_deg=elevation_deg,
    )
    return X, x_range, y_range, sparse_diagnostics


def _compute_band(
    dataset,
    window_name: str,
    band_az_indices: list[int],
    freq_indices_sorted: list[int],
    elev_idx: int,
    pol_idx: int,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
    *,
    az_target_deg: np.ndarray | None = None,
    az_center_deg: float | None = None,
    recon: str = "fft",
    l1_strength: float = 0.05,
    l1_iters: int = 300,
    elevation_deg: float = 0.0,
    retain_complex: bool = False,
    resident_bytes: int = 0,
    cancel_check=None,
):
    band_az_values = _angle_values_to_degrees(
        dataset, "azimuth", dataset.azimuths[band_az_indices]
    )
    if az_center_deg is not None:
        band_az_values = _unwrap_degrees(band_az_values, az_center_deg)
    order = np.argsort(band_az_values)
    sorted_band_indices = [band_az_indices[i] for i in order]
    az_values = band_az_values[order].astype(float)
    initial_azimuth_count = (
        int(np.asarray(az_target_deg).size)
        if az_target_deg is not None
        else len(sorted_band_indices)
    )
    try:
        _validate_isar_working_set(
            _estimate_band_working_set_bytes(
                initial_azimuth_count,
                len(freq_indices_sorted),
                reconstruction=recon,
                retain_complex=retain_complex,
            ),
            operation="ISAR band",
            resident_bytes=resident_bytes,
        )
    except ValueError as exc:
        return f"ISAR working-set preflight blocked: {exc}"
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    target_token = None if az_target_deg is None else _array_token(np.asarray(az_target_deg))
    preprocess_mode = "accurate" if recon == "accurate" else "fast"
    source_token = _selected_data_token(
        dataset,
        sorted_band_indices,
        elev_idx,
        freq_indices_sorted,
        pol_idx,
        cancel_check=cancel_check,
    )
    if source_token is None:
        return "ISAR computation superseded."
    cache_key = (
        _array_token(az_values),
        _array_token(np.asarray(freq_hz, dtype=float)),
        source_token,
        _isar_metadata_token(dataset, elev_idx, pol_idx),
        target_token,
        az_center_deg,
        preprocess_mode,
    )
    prep = _preprocess_cache_get(cache_key)
    cache_hit = prep is not None
    if prep is None:
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        # Slice first; constructing the whole complex grid can require GBs.
        sel = np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
        rcs_slice = dataset.rcs_slice(sel)[:, 0, :, 0]
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        valid = np.isfinite(rcs_slice)
        if not np.any(valid):
            return "ISAR imaging requires phase-aware samples; selected data has no finite complex phase history."
        weights = valid.astype(np.float32)
        rcs_slice = np.nan_to_num(
            rcs_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.complex64, copy=False)
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."

        if az_target_deg is not None:
            target = np.asarray(az_target_deg, dtype=float)
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            az_uniform, rcs_slice, az_gap_info = _resample_azimuth_to_target(
                az_values, rcs_slice, target, axis=0
            )
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            _, weights, _ = _resample_azimuth_to_target(
                az_values, weights, target, axis=0
            )
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            if az_uniform.size < 2:
                return "ISAR azimuth target grid must have ≥2 samples."
            az_nonuniformity = float(az_gap_info["non_uniformity"])
        else:
            theta_native = np.deg2rad(az_values)
            if not np.all(np.isfinite(theta_native)) or np.any(np.diff(theta_native) <= 0):
                axis_name = common.angular_axis_name(dataset, "azimuth")
                return f"{axis_name} samples must be strictly increasing within a band."
            max_az_samples = max(
                az_values.size,
                _MAX_INTERP_COMPLEX_CELLS // max(freq_hz.size, 1),
            )
            try:
                az_plan = _uniform_resample_plan(
                    az_values, max_output_samples=max_az_samples
                )
            except ValueError as exc:
                return f"ISAR azimuth resampling blocked: {exc}."
            az_uniform = np.asarray(az_plan["target"], dtype=float)
            try:
                _validate_isar_working_set(
                    _estimate_band_working_set_bytes(
                        az_uniform.size,
                        freq_hz.size,
                        reconstruction=recon,
                        retain_complex=retain_complex,
                    ),
                    operation="ISAR azimuth regridding",
                    resident_bytes=resident_bytes,
                )
            except ValueError as exc:
                return f"ISAR working-set preflight blocked: {exc}"
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            rcs_slice = _apply_resample_plan(
                az_values, rcs_slice, axis=0, plan=az_plan
            )
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            weights = _apply_resample_plan(
                az_values, weights, axis=0, plan=az_plan
            )
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            az_gap_info = az_plan["info"]
            az_nonuniformity = float(az_gap_info["non_uniformity"])

        try:
            _validate_isar_working_set(
                _estimate_band_working_set_bytes(
                    az_uniform.size,
                    freq_hz.size,
                    reconstruction=recon,
                    retain_complex=retain_complex,
                ),
                operation="ISAR azimuth regridding",
                resident_bytes=resident_bytes,
            )
        except ValueError as exc:
            return f"ISAR working-set preflight blocked: {exc}"
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."

        max_freq_samples = max(
            freq_hz.size,
            _MAX_INTERP_COMPLEX_CELLS // max(az_uniform.size, 1),
        )
        try:
            freq_plan = _uniform_resample_plan(
                freq_hz, max_output_samples=max_freq_samples
            )
        except ValueError as exc:
            return f"ISAR frequency resampling blocked: {exc}."
        freq_uniform = np.asarray(freq_plan["target"], dtype=float)
        try:
            _validate_isar_working_set(
                _estimate_band_working_set_bytes(
                    az_uniform.size,
                    freq_uniform.size,
                    reconstruction=recon,
                    retain_complex=retain_complex,
                ),
                operation="ISAR frequency regridding/formation",
                resident_bytes=resident_bytes,
            )
        except ValueError as exc:
            return f"ISAR working-set preflight blocked: {exc}"
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        rcs_slice = _apply_resample_plan(
            freq_hz, rcs_slice, axis=1, plan=freq_plan
        )
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        weights = _apply_resample_plan(
            freq_hz, weights, axis=1, plan=freq_plan
        )
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        freq_gap_info = freq_plan["info"]
        fr_nonuniformity = float(freq_gap_info["non_uniformity"])
        theta = np.deg2rad(az_uniform)
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."

        if float(theta.max() - theta.min()) < np.pi / 2.0:
            try:
                if preprocess_mode == "accurate":
                    theta_input = theta.copy()
                    freq_input = np.asarray(freq_uniform, dtype=float).copy()
                    rcs_slice, theta, freq_uniform = _pfa_regrid_cartesian(
                        rcs_slice,
                        theta_input,
                        freq_input,
                        cancel_check=cancel_check,
                    )
                    weights, _, _ = _pfa_regrid_cartesian(
                        weights,
                        theta_input,
                        freq_input,
                        cancel_check=cancel_check,
                    )
                else:
                    rcs_slice = _pfa_regrid_azimuth(
                        rcs_slice, theta, freq_uniform, cancel_check=cancel_check
                    )
                    weights = _pfa_regrid_azimuth(
                        weights, theta, freq_uniform, cancel_check=cancel_check
                    )
            except InterruptedError:
                return "ISAR computation superseded."
        elif preprocess_mode == "accurate":
            return "Accurate Cartesian PFA requires an aperture narrower than 90°."

        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."

        weights = np.clip(np.asarray(weights, dtype=np.float32), 0.0, 1.0)
        observed = np.zeros_like(rcs_slice, dtype=np.complex64)
        np.divide(rcs_slice, weights, out=observed, where=weights > 1.0e-6)
        df_eff = float(np.mean(np.diff(freq_uniform)))
        prep = {
            "observed": observed,
            "weights": weights,
            "theta": np.asarray(theta, dtype=float),
            "freq": np.asarray(freq_uniform, dtype=float),
            "df": df_eff,
            "az_values": np.asarray(az_uniform, dtype=float),
            "az_nonuniformity": float(az_nonuniformity),
            "freq_nonuniformity": float(fr_nonuniformity),
            "az_gap_count": int(az_gap_info["gap_count"]),
            "freq_gap_count": int(freq_gap_info["gap_count"]),
            "az_gap_fraction": float(az_gap_info["unsupported_fraction"]),
            "freq_gap_fraction": float(freq_gap_info["unsupported_fraction"]),
            "az_largest_gap": float(az_gap_info["largest_gap"]),
            "freq_largest_gap": float(freq_gap_info["largest_gap"]),
            "coverage": float(np.mean(weights)),
        }
        _preprocess_cache_put(cache_key, prep)

    observed = prep["observed"]
    weights = prep["weights"]
    theta = prep["theta"]
    freq_uniform = prep["freq"]
    df_eff = float(prep["df"])
    az_values = prep["az_values"]
    az_nonuniformity = float(prep["az_nonuniformity"])
    fr_nonuniformity = float(prep["freq_nonuniformity"])
    try:
        _validate_isar_working_set(
            _estimate_band_working_set_bytes(
                observed.shape[0],
                observed.shape[1],
                reconstruction=recon,
                retain_complex=retain_complex,
            ),
            operation="ISAR formation",
            resident_bytes=resident_bytes,
        )
    except ValueError as exc:
        return f"ISAR working-set preflight blocked: {exc}"
    if _cancel_requested(cancel_check):
        return "ISAR computation superseded."
    projection = abs(float(np.cos(np.deg2rad(elevation_deg))))
    theta_span = float(theta[-1] - theta[0])
    dtheta_eff = float(np.mean(np.diff(theta)))
    bandwidth = float(freq_uniform[-1] - freq_uniform[0])
    fc_eff = float(np.mean(freq_uniform))
    sampling = {
        "range_resolution": 299_792_458.0 / (2.0 * projection * bandwidth) * unit_scale,
        "cross_resolution": 299_792_458.0 / (2.0 * projection * fc_eff * theta_span) * unit_scale,
        "range_half_extent": 299_792_458.0 / (4.0 * projection * df_eff) * unit_scale,
        "cross_half_extent": 299_792_458.0 / (4.0 * projection * fc_eff * dtheta_eff) * unit_scale,
    }

    if recon == "sparse":
        out = _compute_band_sparse_l1(
            observed, theta, freq_uniform, df_eff, unit_scale, l1_strength, l1_iters,
            sample_weights=weights, elevation_deg=elevation_deg,
            cancel_check=cancel_check,
        )
        if isinstance(out, str):
            return out
        complex_image, x_range, y_range, sparse_diagnostics = out
        magnitude = np.asarray(np.abs(complex_image), dtype=np.float32)
    else:
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        weighted_observed = observed * weights
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        out = _compute_band_polar_format(
            window_name, weighted_observed, theta, freq_uniform, df_eff, unit_scale,
            sample_weights=weights, elevation_deg=elevation_deg,
            cancel_check=cancel_check,
        )
        if isinstance(out, str):
            return out
        complex_image, x_range, y_range = out
        magnitude = np.asarray(np.abs(complex_image), dtype=np.float32)

    # Sanity-check the computed scene extent.
    x_max_m = (
        float(np.max(np.abs(x_range))) / unit_scale
        if np.all(np.isfinite(x_range)) and unit_scale > 0.0 else float("inf")
    )
    y_max_m = (
        float(np.max(np.abs(y_range))) / unit_scale
        if np.all(np.isfinite(y_range)) and unit_scale > 0.0 else float("inf")
    )
    if (
        not np.all(np.isfinite(x_range))
        or not np.all(np.isfinite(y_range))
        or x_max_m > 1.0e4
        or y_max_m > 1.0e4
    ):
        th_max_deg = float(np.rad2deg(np.max(np.abs(theta))))
        dth_deg = float(np.rad2deg(np.mean(np.diff(theta)))) if theta.size > 1 else 0.0
        f_min_ghz = float(np.min(freq_uniform)) / 1e9
        f_max_ghz = float(np.max(freq_uniform)) / 1e9
        return (
            f"ISAR produced a degenerate scene extent: "
            f"x≈±{x_max_m:.1e} m, y≈±{y_max_m:.1e} m. "
            f"Inputs: θ_max={th_max_deg:.4f}°, dθ={dth_deg:.6f}°, "
            f"f∈[{f_min_ghz:.3f}, {f_max_ghz:.3f}] GHz. "
            f"Likely a too-narrow azimuth selection or unit mismatch."
        )

    return {
        "az_values": az_values,
        "magnitude": magnitude,
        "x_range": x_range,
        "y_range": y_range,
        "az_nonuniformity": az_nonuniformity,
        "freq_nonuniformity": fr_nonuniformity,
        "az_gap_count": int(prep.get("az_gap_count", 0)),
        "freq_gap_count": int(prep.get("freq_gap_count", 0)),
        "az_gap_fraction": float(prep.get("az_gap_fraction", 0.0)),
        "freq_gap_fraction": float(prep.get("freq_gap_fraction", 0.0)),
        "az_largest_gap": float(prep.get("az_largest_gap", 0.0)),
        "freq_largest_gap": float(prep.get("freq_largest_gap", 0.0)),
        "phase_coverage": float(prep["coverage"]),
        "preprocess_cache_hit": cache_hit,
        "source_selection_digest": source_token.hex(),
        **(sparse_diagnostics if recon == "sparse" else {}),
        **(
            {"sparse_diagnostics": dict(sparse_diagnostics)}
            if recon == "sparse"
            else {}
        ),
        "accurate_pfa": recon == "accurate",
        "sampling": sampling,
        **({"complex_image": complex_image} if retain_complex else {}),
    }


# Beyond this azimuth span, a single coherent decoupled-FFT look is
# physically invalid (rotational migration smears scatterers along arcs);
# switch to the sub-aperture composite that wide-angle ISAR tools use.
_COMPOSITE_SPAN_DEG = 20.0
_COMPOSITE_SUB_DEG = 10.0


def _bounded_uniform_azimuth_grid(
    start: float,
    stop: float,
    step: float,
    *,
    frequency_count: int,
) -> np.ndarray:
    """Build an inclusive-when-exact target grid after allocation preflight."""

    start = float(start)
    stop = float(stop)
    step = float(step)
    if not np.all(np.isfinite([start, stop, step])):
        raise ValueError("limits/step must be finite")
    if step <= 0.0:
        raise ValueError("step must be positive")
    if stop <= start:
        raise ValueError("max must exceed min")
    frequency_count = int(frequency_count)
    if frequency_count < 1:
        raise ValueError("at least one frequency sample is required")

    span = stop - start
    tolerance = np.finfo(float).eps * max(span, step, 1.0) * 16.0
    count = int(np.floor((span + tolerance) / step)) + 1
    cells = count * frequency_count
    if count > _MAX_INTERP_AZIMUTH_SAMPLES:
        raise ValueError(
            f"requested {count:,} azimuth samples exceeds the safety limit "
            f"{_MAX_INTERP_AZIMUTH_SAMPLES:,}; increase the azimuth step"
        )
    if cells > _MAX_INTERP_COMPLEX_CELLS:
        raise ValueError(
            f"requested {count:,} azimuth samples x {frequency_count:,} "
            f"frequencies ({cells:,} complex cells) exceeds the working-set "
            f"limit {_MAX_INTERP_COMPLEX_CELLS:,}; increase the azimuth step "
            "or select fewer frequencies"
        )
    return start + step * np.arange(count, dtype=float)


def _validated_azimuth_target(
    dataset,
    azimuth_indices,
    azimuth_target_degrees,
    *,
    aperture_center_degrees: float | None = None,
) -> np.ndarray | None:
    """Validate an explicit Fourier aperture against measured angular support.

    Both the GUI and headless API must pass through this function. Interpolation
    may regularize samples inside a measured aperture; it must never manufacture
    a wider coherent aperture from zeros.
    """

    if azimuth_target_degrees is None:
        return None
    target = np.array(azimuth_target_degrees, dtype=float, copy=True)
    if target.ndim != 1 or target.size < 2:
        raise ValueError(
            "azimuth_target_degrees must be a 1-D grid with at least two samples"
        )
    if not np.all(np.isfinite(target)) or np.any(np.diff(target) <= 0.0):
        raise ValueError(
            "azimuth_target_degrees must be finite and strictly increasing"
        )
    target_steps = np.diff(target)
    target_step = float(np.median(target_steps))
    if not np.allclose(
        target_steps,
        target_step,
        rtol=1.0e-6,
        atol=max(abs(target_step) * 1.0e-9, 1.0e-12),
    ):
        raise ValueError(
            "azimuth_target_degrees must be uniformly spaced for Fourier imaging"
        )

    indices = [int(index) for index in azimuth_indices]
    if len(indices) < 2:
        raise ValueError(
            "at least two measured azimuth samples are required to validate "
            "azimuth_target_degrees"
        )
    source = _angle_values_to_degrees(
        dataset,
        "azimuth",
        np.asarray(dataset.azimuths, dtype=float)[indices],
    )
    if aperture_center_degrees is not None:
        center = float(aperture_center_degrees)
        if not np.isfinite(center):
            raise ValueError("aperture_center_degrees must be finite")
        source = _unwrap_degrees(source, center)
        target = _unwrap_degrees(target, center)
        if np.any(np.diff(target) <= 0.0):
            raise ValueError(
                "azimuth_target_degrees crosses its aperture-center branch; "
                "supply one strictly increasing unwrapped aperture"
            )
    source = np.sort(np.asarray(source, dtype=float))
    if not np.all(np.isfinite(source)) or np.any(np.diff(source) <= 0.0):
        raise ValueError("selected measured azimuth samples must be finite and unique")

    source_span = float(source[-1] - source[0])
    target_span = float(target[-1] - target[0])
    if target_span >= 360.0 - 1.0e-9:
        raise ValueError(
            "azimuth_target_degrees must not include duplicate endpoints one "
            "full revolution apart"
        )
    if source_span < 359.0:
        tolerance = max(float(np.median(np.diff(source))) * 1.0e-6, 1.0e-9)
        if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
            raise ValueError(
                "azimuth_target_degrees extends outside the measured aperture; "
                "GRIM will not turn unmeasured angles into a coherent image"
            )
    return target


def _predict_sublook_scene_axes(
    dataset,
    band_az_indices: list[int],
    freq_hz: np.ndarray,
    unit_scale: float,
    *,
    reconstruction: str,
    elevation_deg: float,
    az_center_deg: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the exact coherent image axes without slicing field data."""

    az_values = _angle_values_to_degrees(
        dataset,
        "azimuth",
        np.asarray(dataset.azimuths, dtype=float)[band_az_indices],
    )
    if az_center_deg is not None:
        az_values = _unwrap_degrees(az_values, az_center_deg)
    az_values = np.sort(np.asarray(az_values, dtype=float))
    max_az_samples = max(
        az_values.size,
        _MAX_INTERP_COMPLEX_CELLS // max(freq_hz.size, 1),
    )
    az_plan = _uniform_resample_plan(
        az_values, max_output_samples=max_az_samples
    )
    az_uniform = np.asarray(az_plan["target"], dtype=float)
    max_freq_samples = max(
        freq_hz.size,
        _MAX_INTERP_COMPLEX_CELLS // max(az_uniform.size, 1),
    )
    freq_plan = _uniform_resample_plan(
        freq_hz, max_output_samples=max_freq_samples
    )
    freq_uniform = np.asarray(freq_plan["target"], dtype=float)
    theta = np.deg2rad(az_uniform)
    if reconstruction == "accurate":
        _q, theta, freq_uniform = _cartesian_pfa_axes(theta, freq_uniform)
    df_eff = float(np.mean(np.diff(freq_uniform)))
    return _scene_axes(
        _next_fast_len(max(theta.size, 256)),
        _next_fast_len(max(freq_uniform.size, 256)),
        theta,
        freq_uniform,
        df_eff,
        unit_scale,
        elevation_deg=elevation_deg,
    )


def _compute_band_composite(
    dataset,
    window_name: str,
    band_az_indices: list[int],
    freq_indices_sorted: list[int],
    elev_idx: int,
    pol_idx: int,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
    *,
    recon: str,
    l1_strength: float,
    l1_iters: int,
    elevation_deg: float,
    retain_complex: bool = False,
    resident_bytes: int = 0,
    az_center_deg: float | None = None,
    cancel_check=None,
):
    """Stream a qualitative max-look composite of coherent narrow looks."""
    az_all = _angle_values_to_degrees(dataset, "azimuth", dataset.azimuths)
    az_vals = az_all[band_az_indices]
    if az_center_deg is not None:
        az_vals = _unwrap_degrees(az_vals, az_center_deg)
    order = np.argsort(az_vals)
    idx_sorted = [band_az_indices[i] for i in order]
    az_sorted = az_vals[order]
    span = float(az_sorted[-1] - az_sorted[0])
    n_sub = max(2, int(np.ceil(span / _COMPOSITE_SUB_DEG)))
    chunks = np.array_split(np.asarray(idx_sorted, dtype=np.int64), n_sub)
    chunk_specs: list[tuple[list[int], float]] = []
    common_half = float("inf")
    for raw_chunk in chunks:
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        chunk = [int(value) for value in raw_chunk]
        if len(chunk) < 2:
            continue
        try:
            x_axis, y_axis = _predict_sublook_scene_axes(
                dataset,
                chunk,
                freq_hz,
                unit_scale,
                reconstruction=recon,
                elevation_deg=elevation_deg,
                az_center_deg=az_center_deg,
            )
        except ValueError as exc:
            return f"Wide-aperture composite preflight failed: {exc}"
        common_half = min(
            common_half,
            float(np.max(np.abs(x_axis))),
            float(np.max(np.abs(y_axis))),
        )
        chunk_az = az_all[chunk]
        if az_center_deg is not None:
            chunk_az = _unwrap_degrees(chunk_az, az_center_deg)
        chunk_specs.append((chunk, np.deg2rad(float(np.mean(chunk_az)))))
    if not chunk_specs or not np.isfinite(common_half):
        return "Wide-aperture composite: no sub-aperture had ≥2 azimuth samples."

    full_source_token = _selected_data_token(
        dataset,
        idx_sorted,
        elev_idx,
        freq_indices_sorted,
        pol_idx,
        cancel_check=cancel_check,
    )
    if full_source_token is None:
        return "ISAR computation superseded."
    composite_bytes = (
        _COMPOSITE_SCRATCH_BYTES
        + _COMPOSITE_GRID_SIDE * _COMPOSITE_GRID_SIDE * np.dtype(np.float32).itemsize
    )
    try:
        _validate_isar_working_set(
            composite_bytes,
            operation="wide-aperture composite",
            resident_bytes=resident_bytes,
        )
    except ValueError as exc:
        return f"ISAR working-set preflight blocked: {exc}"

    axis = np.linspace(-common_half, common_half, _COMPOSITE_GRID_SIDE)
    xq = axis[:, None].astype(np.float32)
    yq = axis[None, :].astype(np.float32)
    comp = np.zeros(
        (_COMPOSITE_GRID_SIDE, _COMPOSITE_GRID_SIDE), dtype=np.float32
    )
    az_nonuni = 0.0
    fr_nonuni = 0.0
    az_gap_fraction = 0.0
    fr_gap_fraction = 0.0
    az_gap_count = 0
    fr_gap_count = 0
    az_largest_gap = 0.0
    fr_largest_gap = 0.0
    phase_coverage = 1.0
    processed_looks = 0
    sparse_converged_looks = 0
    sparse_max_iterations = 0
    sparse_restarts = 0
    sparse_max_gap = 0.0
    sparse_max_residual = 0.0
    sparse_support_size = 0
    sparse_all_debiased = True

    # Form, rotate, and discard one sublook at a time.  Only the final
    # float32 composite and bounded interpolation scratch remain resident.
    for chunk, theta_c in chunk_specs:
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        r = _compute_band(
            dataset, window_name, chunk, freq_indices_sorted, elev_idx, pol_idx,
            freq_hz, df, unit_scale,
            az_target_deg=None, recon=recon,
            l1_strength=l1_strength, l1_iters=l1_iters,
            az_center_deg=az_center_deg, elevation_deg=elevation_deg,
            retain_complex=False,
            resident_bytes=resident_bytes + composite_bytes,
            cancel_check=cancel_check,
        )
        if isinstance(r, str):
            return r
        processed_looks += 1
        az_nonuni = max(az_nonuni, r.get("az_nonuniformity", 0.0))
        fr_nonuni = max(fr_nonuni, r.get("freq_nonuniformity", 0.0))
        az_gap_fraction = max(az_gap_fraction, r.get("az_gap_fraction", 0.0))
        fr_gap_fraction = max(fr_gap_fraction, r.get("freq_gap_fraction", 0.0))
        az_gap_count = max(az_gap_count, r.get("az_gap_count", 0))
        fr_gap_count = max(fr_gap_count, r.get("freq_gap_count", 0))
        az_largest_gap = max(az_largest_gap, r.get("az_largest_gap", 0.0))
        fr_largest_gap = max(fr_largest_gap, r.get("freq_largest_gap", 0.0))
        phase_coverage = min(phase_coverage, r.get("phase_coverage", 1.0))
        if recon == "sparse":
            sparse_converged_looks += int(bool(r.get("sparse_converged", False)))
            sparse_max_iterations = max(
                sparse_max_iterations, int(r.get("sparse_iterations", 0))
            )
            sparse_restarts += int(r.get("sparse_restarts", 0))
            sparse_max_gap = max(
                sparse_max_gap,
                float(r.get("sparse_relative_duality_gap", float("inf"))),
            )
            sparse_max_residual = max(
                sparse_max_residual,
                float(r.get("sparse_output_relative_residual_norm", float("inf"))),
            )
            sparse_support_size += int(r.get("sparse_support_size", 0))
            sparse_all_debiased = sparse_all_debiased and bool(
                r.get("sparse_debias_applied", False)
            )

        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        mag = np.asarray(r["magnitude"], dtype=np.float32)
        xa, ya = r["x_range"], r["y_range"]
        dx = float(xa[1] - xa[0])
        dy = float(ya[1] - ya[0])
        ct, st = np.float32(np.cos(theta_c)), np.float32(np.sin(theta_c))
        # body (x, y) -> this look's rotated-frame (cross-range, range)
        fx = (xq * ct - yq * st - np.float32(xa[0])) / np.float32(dx)
        fy = (xq * st + yq * ct - np.float32(ya[0])) / np.float32(dy)
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        ix = np.floor(fx).astype(np.int32)
        iy = np.floor(fy).astype(np.int32)
        valid = (ix >= 0) & (ix < mag.shape[0] - 1) & (iy >= 0) & (iy < mag.shape[1] - 1)
        wx = (fx - ix).astype(np.float32)
        wy = (fy - iy).astype(np.float32)
        np.clip(ix, 0, mag.shape[0] - 2, out=ix)
        np.clip(iy, 0, mag.shape[1] - 2, out=iy)
        val = (
            mag[ix, iy] * (1 - wx) * (1 - wy)
            + mag[ix + 1, iy] * wx * (1 - wy)
            + mag[ix, iy + 1] * (1 - wx) * wy
            + mag[ix + 1, iy + 1] * wx * wy
        )
        val[~valid] = np.float32(0.0)
        np.maximum(comp, val, out=comp)
        del r, mag, fx, fy, ix, iy, valid, wx, wy, val
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."

    result = {
        "az_values": az_sorted,
        "magnitude": comp,
        "x_range": axis,
        "y_range": axis.copy(),
        "az_nonuniformity": az_nonuni,
        "freq_nonuniformity": fr_nonuni,
        "az_gap_fraction": az_gap_fraction,
        "freq_gap_fraction": fr_gap_fraction,
        "az_gap_count": az_gap_count,
        "freq_gap_count": fr_gap_count,
        "az_largest_gap": az_largest_gap,
        "freq_largest_gap": fr_largest_gap,
        "phase_coverage": phase_coverage,
        "source_selection_digest": full_source_token.hex(),
        "composite": processed_looks,
        "composite_streamed": True,
    }
    if retain_complex:
        result["complex_image_unavailable_reason"] = (
            "wide-aperture max-look composite is incoherent and has no "
            "physically defined complex image"
        )
    if recon == "sparse":
        aggregate_sparse_diagnostics = {
            "sparse_converged": sparse_converged_looks == processed_looks,
            "sparse_status": (
                f"{sparse_converged_looks}/{processed_looks} sub-apertures "
                "converged by relative primal/dual gap"
            ),
            "sparse_iterations": sparse_max_iterations,
            "sparse_restarts": sparse_restarts,
            "sparse_relative_duality_gap": sparse_max_gap,
            "sparse_output_relative_residual_norm": sparse_max_residual,
            "sparse_support_size": sparse_support_size,
            "sparse_debias_applied": sparse_all_debiased,
        }
        result.update(aggregate_sparse_diagnostics)
        result["sparse_diagnostics"] = dict(aggregate_sparse_diagnostics)
    return result


def compute_bands(params: dict):
    """Heavy half of the ISAR render: slicing, resampling, FFTs, display
    decimation. Pure numpy over the arrays captured in `params` — no Qt
    access — so the mixin runs it on a worker thread and the GUI stays live.
    Returns (band_results, elapsed_seconds), or an error-message string."""
    t_start = time.perf_counter()
    band_results = []
    cancel_check = params.get("cancel_check")
    for band_az_indices in params["bands"]:
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        resident_bytes = _result_array_bytes(band_results)
        az_band = _angle_values_to_degrees(
            params["dataset"],
            "azimuth",
            np.asarray(params["dataset"].azimuths, dtype=float)[band_az_indices],
        )
        if params.get("az_center_deg") is not None:
            az_band = _unwrap_degrees(az_band, params["az_center_deg"])
        span = float(az_band.max() - az_band.min())
        if span > _COMPOSITE_SPAN_DEG:
            # A single coherent look is invalid this wide — build the
            # sub-aperture body-frame composite instead (az-interp targets
            # don't apply to composites and are ignored here).
            result = _compute_band_composite(
                params["dataset"],
                params["window_name"],
                band_az_indices,
                params["freq_indices_sorted"],
                params["elev_idx"],
                params["pol_idx"],
                params["freq_hz"],
                params["df"],
                params["unit_scale"],
                recon=params.get("recon", "fft"),
                l1_strength=params.get("l1_strength", 0.05),
                l1_iters=params.get("l1_iters", 300),
                elevation_deg=params.get("elevation_deg", 0.0),
                retain_complex=bool(params.get("retain_complex", False)),
                resident_bytes=resident_bytes,
                az_center_deg=params.get("az_center_deg"),
                cancel_check=cancel_check,
            )
        else:
            result = _compute_band(
                params["dataset"],
                params["window_name"],
                band_az_indices,
                params["freq_indices_sorted"],
                params["elev_idx"],
                params["pol_idx"],
                params["freq_hz"],
                params["df"],
                params["unit_scale"],
                az_target_deg=params["az_target_deg"],
                az_center_deg=params.get("az_center_deg"),
                recon=params.get("recon", "fft"),
                l1_strength=params.get("l1_strength", 0.05),
                l1_iters=params.get("l1_iters", 300),
                elevation_deg=params.get("elevation_deg", 0.0),
                retain_complex=bool(params.get("retain_complex", False)),
                resident_bytes=resident_bytes,
                cancel_check=cancel_check,
            )
        if isinstance(result, str):
            return result
        if _cancel_requested(cancel_check):
            return "ISAR computation superseded."
        # Convention flips, applied to the FINAL image so they are exactly
        # equivalent for single looks and composites (a global mirror commutes
        # through the sub-aperture rotations): Flip X mirrors about x=0
        # (opposite azimuth rotation direction), Flip Y about y=0 (opposite
        # down-range sign). Both checked = the e^{+j2kr} phase convention.
        if params.get("flip_x"):
            result["magnitude"] = result["magnitude"][::-1, :]
            if "complex_image" in result:
                result["complex_image"] = result["complex_image"][::-1, :]
            result["x_range"] = -np.asarray(result["x_range"])[::-1]
        if params.get("flip_y"):
            result["magnitude"] = result["magnitude"][:, ::-1]
            if "complex_image" in result:
                result["complex_image"] = result["complex_image"][:, ::-1]
            result["y_range"] = -np.asarray(result["y_range"])[::-1]
        # GUI rendering can reduce oversized images after formation; headless
        # callers keep the full numerical grid by disabling this parameter.
        if params.get("decimate_display", True):
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
            max_side = min(common.MAX_IMAGE_SIDE, int(np.sqrt(common.MAX_IMAGE_CELLS)))
            original_shape = result["magnitude"].shape
            result["magnitude"] = _decimate_display_max(
                result["magnitude"], max_side=max_side
            )
            result["display_decimated"] = result["magnitude"].shape != original_shape
            if _cancel_requested(cancel_check):
                return "ISAR computation superseded."
        contract_assumptions = tuple(
            str(value) for value in params.get("isar_contract_assumptions", ())
        )
        result["isar_contract_user_assumed"] = bool(contract_assumptions)
        result["isar_contract_undeclared_fields"] = list(contract_assumptions)
        band_results.append(result)
    return band_results, time.perf_counter() - t_start


def form_isar(
    dataset,
    *,
    azimuth_indices=None,
    frequency_indices=None,
    elevation_index: int = 0,
    polarization_index: int = 0,
    window: str = "Hanning",
    reconstruction: str = "fast",
    length_unit: str = "m",
    aperture_center_degrees: float | None = None,
    azimuth_target_degrees: np.ndarray | None = None,
    l1_strength: float = 0.05,
    l1_iterations: int = 300,
    flip_x: bool = False,
    flip_y: bool = False,
    retain_complex: bool = False,
    decimate_display: bool = False,
    legacy_metadata_attested: bool = False,
):
    """Form ISAR images without Qt.

    Returns ``(band_results, elapsed_seconds)`` using the same physical path as
    the GUI. ``reconstruction`` accepts ``fast``, ``accurate``, or ``sparse``.
    Sparse mode is an experimental fixed-lambda LASSO image reconstruction;
    it does not perform target/contaminant separation or emit cleaned phase
    history. ``l1_strength`` must be in ``(0, 1)`` and ``l1_iterations`` is a
    certified-solver iteration ceiling, not a promise of convergence.
    ``retain_complex=True`` keeps each coherent band's complex image for
    scientific export; the default returns magnitude only to limit memory.
    ``decimate_display=True`` applies the GUI's peak-preserving display-only
    reduction to magnitude after full-resolution formation. It must remain
    false for scientific result export.
    ``legacy_metadata_attested`` is retained only for backward compatibility
    with older recorded scripts. Missing legacy convention metadata is now a
    recorded user assumption; explicit incompatibilities are still rejected.
    """
    contract_assumptions: list[str] = []
    preflight_error = _isar_preflight_error(
        dataset,
        legacy_metadata_attested=legacy_metadata_attested,
        flip_x=bool(flip_x),
        flip_y=bool(flip_y),
        undeclared_out=contract_assumptions,
    )
    if preflight_error is not None:
        raise ValueError(f"ISAR blocked: {preflight_error}")

    az_indices = list(range(len(dataset.azimuths))) if azimuth_indices is None else sorted(
        int(i) for i in azimuth_indices
    )
    freq_indices = list(range(len(dataset.frequencies))) if frequency_indices is None else sorted(
        int(i) for i in frequency_indices
    )
    if len(az_indices) < 2 or len(freq_indices) < 2:
        axis_name = common.angular_axis_name(dataset, "azimuth").lower()
        raise ValueError(
            f"ISAR requires at least two {axis_name} and two frequency samples"
        )
    if len(set(az_indices)) != len(az_indices):
        raise ValueError("azimuth_indices must not contain duplicates")
    if len(set(freq_indices)) != len(freq_indices):
        raise ValueError("frequency_indices must not contain duplicates")
    if any(index < 0 or index >= len(dataset.azimuths) for index in az_indices):
        raise IndexError("azimuth_indices contains an out-of-range index")
    if any(index < 0 or index >= len(dataset.frequencies) for index in freq_indices):
        raise IndexError("frequency_indices contains an out-of-range index")
    if not (0 <= elevation_index < len(dataset.elevations)):
        raise IndexError("elevation_index is out of range")
    if not (0 <= polarization_index < len(dataset.polarizations)):
        raise IndexError("polarization_index is out of range")

    frequency_values = np.asarray(dataset.frequencies, dtype=float)[freq_indices]
    order = np.argsort(frequency_values)
    freq_indices = [freq_indices[int(i)] for i in order]
    frequency_values = frequency_values[order]
    if not np.all(np.isfinite(frequency_values)) or np.any(np.diff(frequency_values) <= 0.0):
        raise ValueError("ISAR frequency samples must be finite and strictly increasing")
    frequency_hz = frequency_values * _unit_to_hz_scale(
        str((dataset.units or {}).get("frequency", ""))
    )
    window_name = _window_name(window)
    _, unit_scale = _length_unit(length_unit)
    l1_strength, l1_iterations = _sparse_parameters(
        l1_strength, l1_iterations
    )
    recon_key = str(reconstruction).strip().lower()
    if recon_key in {"accurate", "cartesian", "pfa-accurate"}:
        recon_key = "accurate"
    elif recon_key in {"sparse", "l1", "sparse-l1"}:
        recon_key = "sparse"
    elif recon_key in {"fast", "fft", "pfa", "fast-pfa"}:
        recon_key = "fft"
    else:
        raise ValueError("reconstruction must be 'fast', 'accurate', or 'sparse'")

    if aperture_center_degrees is not None:
        try:
            aperture_center_degrees = float(aperture_center_degrees)
        except (TypeError, ValueError) as exc:
            raise ValueError("aperture_center_degrees must be finite") from exc
        if not np.isfinite(aperture_center_degrees):
            raise ValueError("aperture_center_degrees must be finite")

    azimuth_target = _validated_azimuth_target(
        dataset,
        az_indices,
        azimuth_target_degrees,
        aperture_center_degrees=aperture_center_degrees,
    )

    if aperture_center_degrees is not None or azimuth_target_degrees is not None:
        bands = [az_indices]
    else:
        bands = [b for b in _split_into_bands(az_indices) if len(b) >= 2]
    result = compute_bands({
        "dataset": dataset,
        "bands": bands,
        "freq_indices_sorted": freq_indices,
        "elev_idx": int(elevation_index),
        "elevation_deg": float(
            _angle_values_to_degrees(
                dataset, "elevation", [dataset.elevations[elevation_index]]
            )[0]
        ),
        "pol_idx": int(polarization_index),
        "freq_hz": frequency_hz,
        "df": float(np.mean(np.diff(frequency_hz))),
        "unit_scale": unit_scale,
        "az_target_deg": azimuth_target,
        "az_center_deg": aperture_center_degrees,
        "window_name": window_name,
        "recon": recon_key,
        "l1_strength": l1_strength,
        "l1_iters": l1_iterations,
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
        "retain_complex": bool(retain_complex),
        "decimate_display": bool(decimate_display),
        "isar_contract_assumptions": contract_assumptions,
    })
    if isinstance(result, str):
        raise ValueError(result)
    return result


def render(self) -> None:
    """GUI-thread half: validate the selections, capture everything the worker
    needs into a params dict, and hand off to the mixin's async submit. The
    finished computation comes back through `display_results`."""
    self.last_plot_mode = "isar_image"
    self._start_plot_render()
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return
    if self._preflight_plot_datasets([("Dataset", self.active_dataset)]) is None:
        return

    flip_x_widget = getattr(self, "chk_isar_flip_x", None)
    flip_x = bool(flip_x_widget.isChecked()) if flip_x_widget is not None else False
    flip_y_widget = getattr(self, "chk_isar_flip_y", None)
    flip_y = bool(flip_y_widget.isChecked()) if flip_y_widget is not None else False
    contract_assumptions: list[str] = []
    try:
        preflight_error = _isar_preflight_error(
            self.active_dataset,
            flip_x=flip_x,
            flip_y=flip_y,
            undeclared_out=contract_assumptions,
        )
    except (TypeError, ValueError) as exc:
        self.status.showMessage(f"ISAR blocked: invalid convention metadata: {exc}.")
        return
    if preflight_error is not None:
        self.status.showMessage(f"ISAR blocked: {preflight_error}.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    if not az_indices:
        self.status.showMessage("Select one or more azimuths/aspects to plot.")
        return

    # Optional aperture window (the scrub workflow): keep only selected
    # azimuths within ±width/2 of the center look angle, with 0/360 wrap.
    ap_widget = getattr(self, "chk_isar_aperture", None)
    az_center_deg: float | None = None
    if ap_widget is not None and ap_widget.isChecked():
        ap_center = float(self.spin_isar_ap_center.value())
        ap_width = float(self.spin_isar_ap_width.value())
        if not np.isfinite(ap_center) or not np.isfinite(ap_width) or ap_width <= 0.0:
            self.status.showMessage("ISAR aperture: center/width must be finite and width positive.")
            return
        az_arr = _angle_values_to_degrees(
            self.active_dataset,
            "azimuth",
            np.asarray(self.active_dataset.azimuths, dtype=float)[az_indices],
        )
        dist = np.abs(np.mod(az_arr - ap_center + 180.0, 360.0) - 180.0)
        keep = dist <= ap_width / 2.0 + 1e-9
        az_indices = [i for i, k in zip(az_indices, keep) if k]
        az_center_deg = ap_center
        if len(az_indices) < 2:
            self.status.showMessage(
                f"ISAR aperture {ap_center:g}° ± {ap_width / 2.0:g}° contains fewer "
                "than 2 selected azimuth samples."
            )
            return

    freq_indices = sorted(self._selected_indices(self.list_freq))
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return

    # Optional frequency sub-band: keep only selected frequencies inside
    # [min, max] so an engineer can sweep bands numerically instead of
    # re-selecting thousands of list entries per image.
    band_widget = getattr(self, "chk_isar_freq_band", None)
    if band_widget is not None and band_widget.isChecked():
        f_lo = float(self.spin_isar_freq_min.value())
        f_hi = float(self.spin_isar_freq_max.value())
        if f_hi <= f_lo:
            self.status.showMessage("ISAR freq band: max must exceed min.")
            return
        fvals = self.active_dataset.frequencies
        freq_indices = [i for i in freq_indices if f_lo <= float(fvals[i]) <= f_hi]
        if len(freq_indices) < 2:
            self.status.showMessage(
                f"ISAR freq band [{f_lo:g}, {f_hi:g}] contains fewer than 2 "
                "of the selected frequency samples."
            )
            return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for ISAR imaging.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return
    elevation_deg = float(
        _angle_values_to_degrees(
            self.active_dataset,
            "elevation",
            [self.active_dataset.elevations[elev_idx]],
        )[0]
    )
    if abs(float(np.cos(np.deg2rad(elevation_deg)))) < 1.0e-6:
        self.status.showMessage(
            "Azimuth ISAR is degenerate at elevation ±90° for the horizontal image plane."
        )
        return

    az_interp_widget = getattr(self, "chk_isar_az_interp", None)
    az_interp_on = bool(az_interp_widget.isChecked()) if az_interp_widget is not None else False
    az_target_deg: np.ndarray | None = None
    if az_interp_on:
        az_min = float(self.spin_isar_az_min.value())
        az_max = float(self.spin_isar_az_max.value())
        az_step = float(self.spin_isar_az_step.value())
        try:
            az_target_deg = _bounded_uniform_azimuth_grid(
                az_min,
                az_max,
                az_step,
                frequency_count=len(freq_indices),
            )
            az_target_deg = _validated_azimuth_target(
                self.active_dataset,
                az_indices,
                az_target_deg,
                aperture_center_degrees=az_center_deg,
            )
        except ValueError as exc:
            self.status.showMessage(f"ISAR azimuth interp blocked: {exc}.")
            return
        if az_target_deg.size < 2:
            self.status.showMessage("ISAR azimuth interp grid needs ≥2 samples.")
            return

    if az_interp_on or az_center_deg is not None:
        # Explicit resample collapses the multi-band view into a single image —
        # and circular aperture mode must keep a 0°/360° crossing coherent.
        bands: list[list[int]] = [az_indices] if len(az_indices) >= 2 else []
    else:
        bands = _split_into_bands(az_indices)
        bands = [b for b in bands if len(b) >= 2]
    if not bands:
        self.status.showMessage(
            "Each azimuth/aspect band needs at least 2 contiguous samples for ISAR imaging."
        )
        return
    if len(bands) > common.MAX_WATERFALL_PANELS:
        self.status.showMessage(
            f"ISAR blocked: selection would create {len(bands)} panels (limit "
            f"{common.MAX_WATERFALL_PANELS}). Select contiguous azimuth/aspect "
            "samples or enable one aperture/interpolation window."
        )
        return

    # The per-image reducer is not an aggregate memory limit: several allowed
    # panels can otherwise retain tens of millions of pixels at once. Estimate
    # each post-decimation image before launching the worker and apply one
    # figure-wide budget. Wide-aperture composites use their fixed 1024² grid.
    n_freq_fft_estimate = _next_fast_len(max(len(freq_indices), 256))
    total_display_cells = 0
    for band in bands:
        band_degrees = _angle_values_to_degrees(
            self.active_dataset,
            "azimuth",
            np.asarray(self.active_dataset.azimuths, dtype=float)[band],
        )
        if float(np.max(band_degrees) - np.min(band_degrees)) > _COMPOSITE_SPAN_DEG:
            total_display_cells += 1024 * 1024
            continue
        az_count = az_target_deg.size if az_target_deg is not None else len(band)
        n_az_fft_estimate = _next_fast_len(max(int(az_count), 256))
        total_display_cells += common.bounded_image_cell_count(
            n_az_fft_estimate, n_freq_fft_estimate
        )
    try:
        common.validate_aggregate_image_cells(
            total_display_cells,
            panel_count=len(bands),
            operation="ISAR image",
        )
    except ValueError as exc:
        self.status.showMessage(f"ISAR blocked: {exc}.")
        return

    freq_values_full = self.active_dataset.frequencies[freq_indices]
    freq_order = np.argsort(freq_values_full)
    freq_indices_sorted = [freq_indices[i] for i in freq_order]
    freq_values = freq_values_full[freq_order].astype(float)
    if np.any(np.diff(freq_values) <= 0) or not np.all(np.isfinite(freq_values)):
        self.status.showMessage(
            "Frequency samples must be finite and strictly increasing for ISAR imaging."
        )
        return

    freq_unit = str((self.active_dataset.units or {}).get("frequency", ""))
    freq_hz = freq_values * _unit_to_hz_scale(freq_unit)
    df = float(np.mean(np.diff(freq_hz)))
    if df <= 0.0:
        self.status.showMessage("ISAR imaging requires increasing frequency samples.")
        return

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = _length_unit(units_combo.currentText() if units_combo else "m")

    recon_combo = getattr(self, "combo_isar_recon", None)
    recon_text = recon_combo.currentText() if recon_combo is not None else "FFT"
    recon_lower = recon_text.lower()
    if recon_lower.startswith("sparse"):
        recon = "sparse"
    elif "accurate" in recon_lower or "cartesian" in recon_lower:
        recon = "accurate"
    else:
        recon = "fft"
    l1_strength_spin = getattr(self, "spin_isar_l1_strength", None)
    l1_iters_spin = getattr(self, "spin_isar_l1_iters", None)
    l1_strength = float(l1_strength_spin.value()) if l1_strength_spin is not None else 0.05
    l1_iters = int(l1_iters_spin.value()) if l1_iters_spin is not None else 300
    # Byte-based worker preflight. Display-cell caps alone do not account for
    # the complex source slice, FFT/FISTA temporaries, or full-resolution
    # complex results retained for Export ISAR Result.
    estimated_resident = 0
    estimated_peak = 0
    for band in bands:
        band_degrees = _angle_values_to_degrees(
            self.active_dataset,
            "azimuth",
            np.asarray(self.active_dataset.azimuths, dtype=float)[band],
        )
        if az_center_deg is not None:
            band_degrees = _unwrap_degrees(band_degrees, az_center_deg)
        band_span = float(np.max(band_degrees) - np.min(band_degrees))
        if band_span > _COMPOSITE_SPAN_DEG:
            look_count = max(2, int(np.ceil(band_span / _COMPOSITE_SUB_DEG)))
            sublook_count = max(2, int(np.ceil(len(band) / look_count)))
            working = _estimate_band_working_set_bytes(
                sublook_count,
                len(freq_indices_sorted),
                reconstruction=recon,
                retain_complex=False,
            ) + _COMPOSITE_SCRATCH_BYTES + (
                _COMPOSITE_GRID_SIDE**2 * np.dtype(np.float32).itemsize
            )
            retained = _COMPOSITE_GRID_SIDE**2 * np.dtype(np.float32).itemsize
        else:
            az_count = int(az_target_deg.size) if az_target_deg is not None else len(band)
            working = _estimate_band_working_set_bytes(
                az_count,
                len(freq_indices_sorted),
                reconstruction=recon,
                retain_complex=True,
            )
            n_az_fft = _next_fast_len(max(az_count, 256))
            n_freq_fft = _next_fast_len(max(len(freq_indices_sorted), 256))
            image_cells = n_az_fft * n_freq_fft
            display_cells = common.bounded_image_cell_count(n_az_fft, n_freq_fft)
            retained = 8 * image_cells + 4 * display_cells
        estimated_peak = max(estimated_peak, estimated_resident + working)
        estimated_resident += retained
    try:
        _validate_isar_working_set(
            estimated_peak,
            operation="ISAR selection",
        )
    except ValueError as exc:
        self.status.showMessage(f"ISAR blocked: {exc}.")
        return

    self._isar_submit({
        "dataset": self.active_dataset,
        # Identity token: if the active figure changed while computing (user
        # switched tabs), the finished result is dropped instead of being
        # painted onto whatever tab is now in front.
        "figure_token": self.plot_figure,
        "render_generation": getattr(self, "_plot_render_generation", 0),
        "bands": bands,
        "freq_indices_sorted": freq_indices_sorted,
        "elev_idx": elev_idx,
        "elevation_deg": elevation_deg,
        "pol_idx": pol_idx,
        "freq_hz": freq_hz,
        "df": df,
        "unit_scale": unit_scale,
        "unit_name": unit_name,
        "az_target_deg": az_target_deg,
        "az_center_deg": az_center_deg,
        "window_name": str(self.combo_isar_window.currentText()),
        "recon": recon,
        "l1_strength": l1_strength,
        "l1_iters": l1_iters,
        "flip_x": flip_x,
        "flip_y": flip_y,
        # The ISAR tab exposes Export ISAR Result, so retain the coherent
        # worker result. Wide max-look composites explicitly report that no
        # physically meaningful complex image exists.
        "retain_complex": True,
        "isar_contract_assumptions": contract_assumptions,
    })


def display_results(self, params: dict, band_results: list, elapsed: float) -> None:
    """GUI-thread half two: draw the computed band images. Runs from the
    mixin's worker-finished slot; everything Qt/matplotlib happens here."""
    dataset = params["dataset"]
    unit_name = params["unit_name"]
    az_target_deg = params["az_target_deg"]

    # Convert coherent magnitude to generic image intensity. Image formation
    # does not guarantee an absolute square-metre normalization, so neither
    # branch claims dBsm/dBke. (Magnitude was already max-pool decimated on the
    # worker; block-max commutes with squaring and log.)
    for br in band_results:
        magnitude = np.asarray(br["magnitude"], dtype=np.float32)
        intensity = np.empty_like(magnitude, dtype=np.float32)
        np.multiply(magnitude, magnitude, out=intensity)
        if self._plot_scale_is_linear():
            br["isar_display"] = intensity
        else:
            np.maximum(intensity, np.float32(1.0e-12), out=intensity)
            np.log10(intensity, out=intensity)
            intensity *= np.float32(10.0)
            br["isar_display"] = intensity

    n_bands = len(band_results)

    self._remove_colorbar()
    self.plot_figure.clear()
    if n_bands == 1:
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        active_axes = [self.plot_ax]
    else:
        ax_array = self.plot_figure.subplots(1, n_bands, sharey=True)
        if not isinstance(ax_array, np.ndarray):
            ax_array = np.array([ax_array])
        active_axes = list(ax_array.ravel())
        self.plot_axes = active_axes
        self.plot_ax = active_axes[0]
    self._style_plot_axes()

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax
    shared_scale = bool(self.chk_colorbar_shared.isChecked())
    shared_limits = (
        common.finite_data_limits(br["isar_display"] for br in band_results)
        if shared_scale and not use_clamp
        else None
    )
    plot_vmin = zmin if use_clamp else (
        shared_limits[0] if shared_limits is not None else None
    )
    plot_vmax = zmax if use_clamp else (
        shared_limits[1] if shared_limits is not None else None
    )

    square_widget = getattr(self, "chk_isar_square", None)
    square_aspect = bool(square_widget.isChecked()) if square_widget is not None else False

    last_mesh = None
    self._isar_meshes = []
    overall_x_min = float("inf")
    overall_x_max = float("-inf")
    overall_y_min = float("inf")
    overall_y_max = float("-inf")
    for ax, br in zip(active_axes, band_results):
        x_min = float(br["x_range"].min())
        x_max = float(br["x_range"].max())
        y_min = float(br["y_range"].min())
        y_max = float(br["y_range"].max())
        # imshow on a uniform grid is several times faster than pcolormesh
        # for big arrays (1601-frequency datasets feel laggy with pcolormesh).
        mesh = ax.imshow(
            br["isar_display"].T,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=plot_vmin,
            vmax=plot_vmax,
        )
        if square_aspect:
            # adjustable="datalim" keeps the plot box at its current size and
            # *expands* the visible data limits to maintain 1:1 cross-range /
            # range scale. The opposite ("box") shrinks the box, which is
            # what we don't want when range and cross-range extents differ.
            ax.set_aspect("equal", adjustable="datalim")
        last_mesh = mesh
        self._isar_meshes.append(mesh)
        overall_x_min = min(overall_x_min, x_min)
        overall_x_max = max(overall_x_max, x_max)
        overall_y_min = min(overall_y_min, y_min)
        overall_y_max = max(overall_y_max, y_max)
        if n_bands > 1:
            ax.set_title(
                f"{float(br['az_values'][0]):g}°–{float(br['az_values'][-1]):g}°",
                color=self._current_plot_text(),
            )

    elev_value = float(params["elevation_deg"])
    elev_name = common.angular_axis_name(dataset, "elevation")
    pol_value = dataset.polarizations[params["pol_idx"]]
    if params.get("recon") == "sparse":
        recon_label = " | Sparse L1 (Experimental)"
    elif params.get("recon") == "accurate":
        recon_label = " | Cartesian PFA"
    else:
        recon_label = " | Fast PFA"
    composite_subs = max((br.get("composite", 0) for br in band_results), default=0)
    if composite_subs:
        recon_label += f" | Wide-Aperture Composite ({composite_subs} looks)"
    fig_title = (
        f"ISAR Image | {elev_name} {elev_value:g} deg | Pol {pol_value}{recon_label}"
    )
    if n_bands > 1:
        self.plot_figure.suptitle(fig_title, color=self._current_plot_text())
    else:
        active_axes[0].set_title(fig_title, color=self._current_plot_text())

    # Composite images live in the body frame (θ=0 radar frame), not a single
    # look's cross-range/range frame — label accordingly.
    horizontal_projection = abs(float(params.get("elevation_deg", 0.0))) > 1.0e-9
    if composite_subs:
        x_label = f"Cross-Range at 0° ({unit_name})"
        y_label = f"Down-Range at 0° ({unit_name})"
    elif horizontal_projection:
        x_label = f"Horizontal Cross-Range ({unit_name})"
        y_label = f"Horizontal Range ({unit_name})"
    else:
        x_label = f"Cross-Range ({unit_name})"
        y_label = f"Range ({unit_name})"
    for ax in active_axes:
        ax.set_xlabel(x_label)
    active_axes[0].set_ylabel(y_label)

    if self.chk_colorbar.isChecked() and last_mesh is not None:
        if shared_scale:
            self.plot_colorbars = [
                self.plot_figure.colorbar(last_mesh, ax=active_axes)
            ]
        else:
            self.plot_colorbars = [
                self.plot_figure.colorbar(mesh, ax=ax)
                for ax, mesh in zip(active_axes, self._isar_meshes)
            ]
        for colorbar in self.plot_colorbars:
            self._apply_colorbar_ticks(colorbar)
            if self._plot_scale_is_linear():
                colorbar.set_label(
                    "Image Intensity (linear)", color=self._current_plot_text()
                )
            else:
                colorbar.set_label(
                    "Image Intensity (dB)", color=self._current_plot_text()
                )
            colorbar.ax.tick_params(colors=self._current_plot_text())
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self._current_plot_text())

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(overall_x_min)
    self.spin_plot_xmax.setValue(overall_x_max)
    self.spin_plot_ymin.setValue(overall_y_min)
    self.spin_plot_ymax.setValue(overall_y_max)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    # Auto-fit the z (dB) spinboxes only on the *first* render of a new
    # dataset. Re-running this on every render — which the per-keystroke
    # `valueChanged` signal triggers — would clobber the user's typing
    # whenever zmin transiently exceeds zmax mid-keystroke.
    state_key = id(dataset)
    last_state = getattr(self, "_isar_last_autofit_state", None)
    autofit_limits = None
    if state_key != last_state:
        img_min = float("inf")
        img_max = float("-inf")
        for br in band_results:
            finite = br["isar_display"][np.isfinite(br["isar_display"])]
            if finite.size:
                img_min = min(img_min, float(finite.min()))
                img_max = max(img_max, float(finite.max()))
        if np.isfinite(img_min) and np.isfinite(img_max) and img_max > img_min:
            cur_zmin = self.spin_plot_zmin.value()
            cur_zmax = self.spin_plot_zmax.value()
            clamp_active = cur_zmin < cur_zmax
            clamp_dead = clamp_active and (cur_zmax < img_min or cur_zmin > img_max)
            if not clamp_active or clamp_dead:
                display_floor = img_max - 60.0 if not self._plot_scale_is_linear() else img_min
                self.spin_plot_zmin.blockSignals(True)
                self.spin_plot_zmax.blockSignals(True)
                self.spin_plot_zmin.setValue(display_floor)
                self.spin_plot_zmax.setValue(img_max)
                self.spin_plot_zmin.blockSignals(False)
                self.spin_plot_zmax.blockSignals(False)
                autofit_limits = (display_floor, img_max)
        self._isar_last_autofit_state = state_key

    # Signals are blocked to avoid recursive rendering, so apply a new first-
    # render auto-fit directly. This keeps the visible canvas, controls, export,
    # and frozen headless recipe on the same global normalization.
    if autofit_limits is not None:
        for mesh in self._isar_meshes:
            mesh.set_clim(*autofit_limits)
        for colorbar in self.plot_colorbars or []:
            self._apply_colorbar_ticks(colorbar)

    self._apply_plot_limits()

    # Surface any resampling that happened so the user knows their input
    # wasn't on a uniform grid. The number is the relative spread of native
    # spacings ((max-min)/median); anything > ~0.001 was actually resampled.
    az_max = max(br.get("az_nonuniformity", 0.0) for br in band_results)
    fr_max = max(br.get("freq_nonuniformity", 0.0) for br in band_results)
    if params.get("recon") == "sparse":
        mode_label = "Sparse L1 (Experimental)"
    elif params.get("recon") == "accurate":
        mode_label = "Cartesian PFA"
    else:
        mode_label = "Fast PFA"
    parts = [f"ISAR image updated in {elapsed:.2f}s ({mode_label})"]
    if n_bands > 1:
        parts.append(f" ({n_bands} bands)")
    notes = []
    contract_assumptions = tuple(params.get("isar_contract_assumptions", ()))
    if contract_assumptions:
        notes.append(
            "undeclared ISAR conventions user-assumed: "
            + ", ".join(str(value) for value in contract_assumptions)
        )
    if composite_subs:
        notes.append(
            f"wide aperture — composited {composite_subs} × ~{_COMPOSITE_SUB_DEG:g}° looks "
            "into the 0°-azimuth body frame"
        )
    if composite_subs and az_target_deg is not None:
        notes.append("az interp grid ignored (composite mode)")
    elif az_target_deg is not None:
        notes.append(
            f"az interp {az_target_deg[0]:g}→{az_target_deg[-1]:g}° step "
            f"{float(np.mean(np.diff(az_target_deg))):g}° ({az_target_deg.size} samples)"
        )
    elif az_max >= 1e-3:
        notes.append(f"resampled azimuth (Δ-spread {az_max*100:.1f}%)")
    if fr_max >= 1e-3:
        notes.append(f"resampled frequency (Δ-spread {fr_max*100:.1f}%)")
    az_gap_count = max((br.get("az_gap_count", 0) for br in band_results), default=0)
    fr_gap_count = max((br.get("freq_gap_count", 0) for br in band_results), default=0)
    if az_gap_count:
        az_gap_fraction = max(
            br.get("az_gap_fraction", 0.0) for br in band_results
        )
        largest_az_gap = max(br.get("az_largest_gap", 0.0) for br in band_results)
        notes.append(
            f"{az_gap_count} missing azimuth sector(s) zero-weighted, not "
            f"interpolated (largest {largest_az_gap:g}°, "
            f"{az_gap_fraction*100:.1f}% of uniform grid)"
        )
    if fr_gap_count:
        fr_gap_fraction = max(
            br.get("freq_gap_fraction", 0.0) for br in band_results
        )
        largest_fr_gap = max(br.get("freq_largest_gap", 0.0) for br in band_results)
        notes.append(
            f"{fr_gap_count} missing frequency band(s) zero-weighted, not "
            f"interpolated (largest {largest_fr_gap/1.0e9:g} GHz, "
            f"{fr_gap_fraction*100:.1f}% of uniform grid)"
        )
    coverage = min(br.get("phase_coverage", 1.0) for br in band_results)
    if coverage < 0.9995:
        notes.append(f"weighted phase coverage {coverage*100:.1f}%")
    if any(br.get("preprocess_cache_hit", False) for br in band_results):
        notes.append("reused gridded phase history")
    if any(br.get("display_decimated", False) for br in band_results):
        notes.append(
            "peak-preserving display decimation applied; narrow the aperture/band "
            "for full display resolution"
        )
    sparse_results = [
        br for br in band_results if br.get("sparse_iterations") is not None
    ]
    if sparse_results:
        converged_count = sum(
            bool(br.get("sparse_converged", False)) for br in sparse_results
        )
        max_iterations = max(
            int(br.get("sparse_iterations", 0)) for br in sparse_results
        )
        max_gap = max(
            float(br.get("sparse_relative_duality_gap", float("inf")))
            for br in sparse_results
        )
        if converged_count == len(sparse_results):
            notes.append(
                f"sparse solver converged ({max_iterations} iterations, "
                f"relative gap ≤{max_gap:.2g})"
            )
        else:
            notes.append(
                f"SPARSE NOT CONVERGED: {converged_count}/{len(sparse_results)} "
                f"images certified at {max_iterations} iterations "
                f"(worst relative gap {max_gap:.2g})"
            )
        max_relative_residual = max(
            float(br.get("sparse_output_relative_residual_norm", float("inf")))
            for br in sparse_results
        )
        notes.append(
            f"sparse weighted residual ≤{max_relative_residual:.3g} relative"
        )
    if band_results:
        sampling = band_results[0].get("sampling", {})
        if sampling:
            notes.append(
                f"nominal resolution Δx≈{sampling['cross_resolution']:.3g} {unit_name}, "
                f"Δr≈{sampling['range_resolution']:.3g} {unit_name}; "
                f"unambiguous |x|≤{sampling['cross_half_extent']:.3g}, "
                f"|r|≤{sampling['range_half_extent']:.3g} {unit_name}"
            )
    if notes:
        parts.append(" — " + ", ".join(notes))
    self._show_plot_status("".join(parts))
