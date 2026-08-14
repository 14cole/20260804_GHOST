from __future__ import annotations

from collections import OrderedDict
import os
import time

import numpy as np

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
    return 1e9


_LENGTH_UNIT_FACTORS = {
    "m": 1.0,
    "in": 1.0 / 0.0254,
    "ft": 1.0 / 0.3048,
}


def _length_unit(name: str | None) -> tuple[str, float]:
    key = (name or "m").strip().lower()
    return (key, _LENGTH_UNIT_FACTORS[key]) if key in _LENGTH_UNIT_FACTORS else ("m", 1.0)


def _resample_azimuth_to_target(
    source_deg: np.ndarray,
    samples: np.ndarray,
    target_deg: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    else:
        # Linear interp with zero-fill outside [source_min, source_max].
        out = _lerp_along_last(source_sorted, samples_moved, target)
        outside = (target < source_sorted[0]) | (target > source_sorted[-1])
        if np.any(outside):
            out[..., outside] = 0.0

    out = np.moveaxis(out, -1, axis)
    return target, out


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
    if values.size < 3:
        return values, samples, 0.0
    diffs = np.diff(values)
    median_diff = float(np.median(diffs))
    if median_diff <= 0.0:
        return values, samples, 0.0
    non_uniformity = float(np.max(diffs) - np.min(diffs)) / median_diff
    if non_uniformity < rel_tol:
        return values, samples, non_uniformity

    target = np.linspace(values[0], values[-1], values.size)
    moved = np.moveaxis(samples, axis, -1)
    out = np.moveaxis(_lerp_along_last(values, moved, target), -1, axis)
    return target, out, non_uniformity


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
    if n <= 1:
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

    # Single precision throughout: halves memory traffic on 0.01°-step files
    # (36000×1701 slices) and scipy's pocketfft keeps complex64 native.
    win_az = _window_array(window_name, n_az).astype(np.float32)
    win_freq = _window_array(window_name, n_freq).astype(np.float32)
    rcs_windowed = np.asarray(rcs_polar, dtype=np.complex64) * win_az[:, None]
    rcs_windowed *= win_freq[None, :]
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

    range_az = _ifft(rcs_windowed, n=n_freq_fft, axis=1)
    del rcs_windowed
    isar_complex = _ifft(range_az, n=n_az_fft, axis=0)
    del range_az
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
        src = k_ratio[n] * sin_psi
        row = S[:, n]
        if np.iscomplexobj(row):
            out[:, n] = (
                np.interp(target_q, src, row.real, left=0.0, right=0.0)
                + 1j * np.interp(target_q, src, row.imag, left=0.0, right=0.0)
            )
        else:
            out[:, n] = np.interp(target_q, src, row, left=0.0, right=0.0)
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


def _pfa_regrid_cartesian(
    S: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    *,
    row_block: int = 128,
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
    if theta.size < 2 or freq_hz.size < 2:
        raise ValueError("Accurate PFA requires at least two azimuth and frequency samples")
    psi = theta - float(np.mean(theta))
    if float(psi[-1] - psi[0]) >= np.pi / 2.0:
        raise ValueError("Accurate Cartesian PFA requires an aperture narrower than 90°")
    fc = float(np.mean(freq_hz))
    ratio_min = float(freq_hz[0] / fc)
    q = np.linspace(ratio_min * np.sin(psi[0]), ratio_min * np.sin(psi[-1]), theta.size)
    az_grid = np.empty_like(S)
    dpsi = float(np.mean(np.diff(psi)))
    ratios = freq_hz / fc
    # Invert q=(f/fc)sin(psi) analytically, then interpolate the complex
    # field with a four-point stencil. This avoids the amplitude droop of
    # piecewise-linear gridding when phase advances appreciably per sample.
    for start in range(0, freq_hz.size, max(int(row_block), 1)):
        stop = min(start + max(int(row_block), 1), freq_hz.size)
        arg = q[:, None] / ratios[None, start:stop]
        required_psi = np.arcsin(np.clip(arg, -1.0, 1.0))
        coordinates = (required_psi - psi[0]) / dpsi
        block = _interp_uniform_axis0(np.asarray(S)[:, start:stop], coordinates)
        block[np.abs(arg) > 1.0] = 0
        az_grid[:, start:stop] = block

    u = fc * q
    max_abs_u = float(np.max(np.abs(u)))
    v_max_sq = float(freq_hz[-1] ** 2 - max_abs_u**2)
    if v_max_sq <= float(freq_hz[0] ** 2):
        raise ValueError("Selected aperture/band has no common Cartesian PFA support")
    v = np.linspace(float(freq_hz[0]), np.sqrt(v_max_sq), freq_hz.size)
    out = np.empty_like(az_grid)
    for start in range(0, q.size, max(int(row_block), 1)):
        stop = min(start + max(int(row_block), 1), q.size)
        required_f = np.sqrt(v[None, :] ** 2 + u[start:stop, None] ** 2)
        coordinates = (required_f - freq_hz[0]) / float(np.mean(np.diff(freq_hz)))
        out[start:stop] = _interp_uniform_axis0(
            az_grid[start:stop].T, coordinates.T
        ).T
    return out, q, v


def _soft_threshold_complex(x: np.ndarray, t: float) -> np.ndarray:
    """Complex soft-thresholding: shrink magnitudes by `t`, keep phases —
    the proximal operator of t·Σ|x_i| for complex x."""
    mag = np.abs(x)
    scale = np.maximum(mag - t, 0.0) / np.maximum(mag, 1e-30)
    return x * scale.astype(np.float32)


# Sparse reconstructions beyond this many image pixels would need tens of
# seconds per solve; steer the user to a sub-aperture / sub-band instead.
_SPARSE_MAX_PIXELS = 16_000_000

# Keep only reusable, post-gridding phase histories. The byte bound prevents
# cine mode from retaining several platform-sized arrays. Set to zero to
# disable, or tune for a workstation with GRIM_ISAR_CACHE_MB.
_PREPROCESS_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_PREPROCESS_CACHE_LIMIT = max(int(os.environ.get("GRIM_ISAR_CACHE_MB", "512")), 0) * 1024**2
_PREPROCESS_CACHE_BYTES = 0


def _array_token(values) -> tuple:
    arr = np.ascontiguousarray(values)
    return (arr.size, hash(arr.view(np.uint8).tobytes()))


def _preprocess_cache_get(key: tuple):
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
    """Sparse (ℓ1-regularised) ISAR image — the compressed-sensing / basis
    pursuit denoise formulation (van den Berg & Friedlander 2008, "Probing the
    Pareto Frontier for Basis Pursuit Solutions", SIAM J. Sci. Comput. 31(2)).

    The measured phase history S(θ, f) is modelled as a partial 2-D Fourier
    transform of the reflectivity image X (the SAME decoupled k-space model
    the FFT path inverts):  S = A·X,  A = crop ∘ fft2 ∘ ifftshift.  Instead of
    the matched filter AᴴS (whose point-spread sidelobes smear every
    scatterer into crosses and haze), we solve the ℓ1-regularised least
    squares / BPDN Lagrangian

        min_X  ½‖A·X − S‖₂² + λ‖X‖₁,   λ = strength · ‖AᴴS‖_∞

    with FISTA (accelerated proximal gradient; same objective family the
    referenced SPGL1 paper solves via its Pareto root-finding). The ℓ1 prior
    drives sidelobes and noise to exactly zero, so only genuine scatterers
    survive — the "clean" look of sparse-imaging ISAR tools. Every operator
    application is one FFT, so a sub-aperture solve stays interactive.

    NO taper window is applied here: windows trade resolution for sidelobe
    suppression, and the sparsity prior already does the suppression. The
    user's window combo only affects the FFT mode.

    Amplitude convention matches the FFT path (unit scatterer → |X| ≈ 1 →
    0 dB); the soft-threshold bias is ≲ `strength` (≈0.4 dB at 0.05).
    """
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

    S = np.ascontiguousarray(rcs_polar, dtype=np.complex64)
    if sample_weights is None:
        weights = np.ones(S.shape, dtype=np.float32)
    else:
        weights = np.clip(np.asarray(sample_weights, dtype=np.float32), 0.0, 1.0)
        if weights.shape != S.shape:
            raise ValueError("Sparse ISAR sample weights must match the phase-history shape")
    weights_sq = weights * weights

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
    lam = float(strength) * float(np.abs(matched).max())
    if lam <= 0.0:
        return "Sparse L1: selected data is identically zero."

    X = np.zeros((n_az_fft, n_freq_fft), dtype=np.complex64)
    Y = X
    t_momentum = 1.0
    iterations_used = 0
    for iteration in range(int(n_iters)):
        if cancel_check is not None and cancel_check():
            return "ISAR computation superseded."
        grad = adjoint(weights_sq * (forward(Y) - S))
        X_new = _soft_threshold_complex(Y - grad / np.float32(lip), lam / lip)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_momentum**2))
        Y = X_new + np.float32((t_momentum - 1.0) / t_new) * (X_new - X)
        iterations_used = iteration + 1
        if iterations_used >= 10 and iterations_used % 5 == 0:
            rel_change = float(np.linalg.norm(X_new - X)) / max(float(np.linalg.norm(X_new)), 1e-30)
            if rel_change <= float(convergence_tol):
                X = X_new
                break
        X, t_momentum = X_new, t_new

    # Debias (standard companion step to BPDN, cf. §1 of the referenced
    # paper): the ℓ1 shrinkage biases every recovered amplitude low by ~λ.
    # Re-fit least squares ON THE RECOVERED SUPPORT ONLY, penalty removed, so
    # scatterer amplitudes read true dB while off-support pixels stay exactly
    # zero. Conjugate gradient on the support-restricted normal equations —
    # adjacent Fourier columns are strongly correlated, which stalls plain
    # gradient (Landweber) steps but CG handles well.
    support = np.abs(X) > 0
    if support.any():
        def normal_op(v: np.ndarray) -> np.ndarray:
            out = adjoint(weights_sq * forward(v))
            out[~support] = 0
            return out

        b = adjoint(weights_sq * S)
        b[~support] = 0
        X[~support] = 0
        r = b - normal_op(X)
        p = r.copy()
        rs_old = float(np.vdot(r, r).real)
        b_norm = float(np.vdot(b, b).real)
        for _ in range(30):
            if cancel_check is not None and cancel_check():
                return "ISAR computation superseded."
            if rs_old <= 1e-12 * max(b_norm, 1e-30):
                break
            Ap = normal_op(p)
            alpha = rs_old / max(float(np.vdot(p, Ap).real), 1e-30)
            X = X + np.complex64(alpha) * p
            r = r - np.complex64(alpha) * Ap
            rs_new = float(np.vdot(r, r).real)
            p = r + np.complex64(rs_new / max(rs_old, 1e-30)) * p
            rs_old = rs_new
        X[~support] = 0

    # No rescale needed: a unit-amplitude scatterer produces |S| = 1 under the
    # forward model, so the recovered X IS the physical reflectivity (≈ 1 →
    # 0 dB), matching the FFT path's normalisation convention.
    x_range, y_range = _scene_axes(
        n_az_fft, n_freq_fft, theta, freq_hz, df, unit_scale,
        elevation_deg=elevation_deg,
    )
    return X, x_range, y_range, iterations_used


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
    l1_iters: int = 100,
    elevation_deg: float = 0.0,
    cancel_check=None,
):
    band_az_values = np.asarray(dataset.azimuths[band_az_indices], dtype=float)
    if az_center_deg is not None:
        band_az_values = _unwrap_degrees(band_az_values, az_center_deg)
    order = np.argsort(band_az_values)
    sorted_band_indices = [band_az_indices[i] for i in order]
    az_values = band_az_values[order].astype(float)
    target_token = None if az_target_deg is None else _array_token(np.asarray(az_target_deg))
    preprocess_mode = "accurate" if recon == "accurate" else "fast"
    cache_key = (
        id(dataset), _array_token(np.asarray(sorted_band_indices, dtype=np.int64)),
        _array_token(np.asarray(freq_indices_sorted, dtype=np.int64)), elev_idx, pol_idx,
        target_token, az_center_deg, preprocess_mode,
    )
    prep = _preprocess_cache_get(cache_key)
    cache_hit = prep is not None
    if prep is None:
        if cancel_check is not None and cancel_check():
            return "ISAR computation superseded."
        # Slice first; constructing the whole complex grid can require GBs.
        sel = np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
        rcs_slice = dataset.rcs_slice(sel)[:, 0, :, 0]
        valid = np.isfinite(rcs_slice)
        if not np.any(valid):
            return "ISAR imaging requires phase-aware samples; selected data has no finite complex phase history."
        weights = valid.astype(np.float32)
        rcs_slice = np.nan_to_num(
            rcs_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.complex64, copy=False)

        if az_target_deg is not None:
            target = np.asarray(az_target_deg, dtype=float)
            az_uniform, rcs_slice = _resample_azimuth_to_target(
                az_values, rcs_slice, target, axis=0
            )
            _, weights = _resample_azimuth_to_target(az_values, weights, target, axis=0)
            if az_uniform.size < 2:
                return "ISAR azimuth target grid must have ≥2 samples."
            az_nonuniformity = 0.0
        else:
            theta_native = np.deg2rad(az_values)
            if not np.all(np.isfinite(theta_native)) or np.any(np.diff(theta_native) <= 0):
                return "Azimuth samples must be strictly increasing within a band."
            az_uniform, rcs_slice, az_nonuniformity = _resample_complex_uniform(
                az_values, rcs_slice, axis=0
            )
            _, weights, _ = _resample_complex_uniform(az_values, weights, axis=0)

        freq_uniform, rcs_slice, fr_nonuniformity = _resample_complex_uniform(
            freq_hz, rcs_slice, axis=1
        )
        _, weights, _ = _resample_complex_uniform(freq_hz, weights, axis=1)
        theta = np.deg2rad(az_uniform)

        if float(theta.max() - theta.min()) < np.pi / 2.0:
            if preprocess_mode == "accurate":
                theta_input = theta.copy()
                freq_input = np.asarray(freq_uniform, dtype=float).copy()
                rcs_slice, theta, freq_uniform = _pfa_regrid_cartesian(
                    rcs_slice, theta_input, freq_input
                )
                weights, _, _ = _pfa_regrid_cartesian(weights, theta_input, freq_input)
            else:
                rcs_slice = _pfa_regrid_azimuth(rcs_slice, theta, freq_uniform)
                weights = _pfa_regrid_azimuth(weights, theta, freq_uniform)
        elif preprocess_mode == "accurate":
            return "Accurate Cartesian PFA requires an aperture narrower than 90°."

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
        complex_image, x_range, y_range, iterations_used = out
        magnitude = np.abs(complex_image)
        peak = float(magnitude.max())
        if peak > 0.0:
            # The ℓ1 solution is exactly zero off the scatterers; floor at
            # peak−120 dB so the dB conversion stays finite for display.
            np.maximum(magnitude, peak * 1e-6, out=magnitude)
    else:
        out = _compute_band_polar_format(
            window_name, observed * weights, theta, freq_uniform, df_eff, unit_scale,
            sample_weights=weights, elevation_deg=elevation_deg,
        )
        if isinstance(out, str):
            return out
        complex_image, x_range, y_range = out
        magnitude = np.abs(complex_image)

    # Sanity-check the computed scene extent.
    if (
        not np.all(np.isfinite(x_range))
        or not np.all(np.isfinite(y_range))
        or float(np.max(np.abs(x_range))) > 1.0e4
        or float(np.max(np.abs(y_range))) > 1.0e4
    ):
        x_max_abs = float(np.max(np.abs(x_range))) if np.all(np.isfinite(x_range)) else float("inf")
        y_max_abs = float(np.max(np.abs(y_range))) if np.all(np.isfinite(y_range)) else float("inf")
        th_max_deg = float(np.rad2deg(np.max(np.abs(theta))))
        dth_deg = float(np.rad2deg(np.mean(np.diff(theta)))) if theta.size > 1 else 0.0
        f_min_ghz = float(np.min(freq_uniform)) / 1e9
        f_max_ghz = float(np.max(freq_uniform)) / 1e9
        return (
            f"ISAR produced a degenerate scene extent: "
            f"x≈±{x_max_abs:.1e}m, y≈±{y_max_abs:.1e}m. "
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
        "phase_coverage": float(prep["coverage"]),
        "preprocess_cache_hit": cache_hit,
        "sparse_iterations": iterations_used if recon == "sparse" else None,
        "accurate_pfa": recon == "accurate",
        "sampling": sampling,
    }


# Beyond this azimuth span, a single coherent decoupled-FFT look is
# physically invalid (rotational migration smears scatterers along arcs);
# switch to the sub-aperture composite that wide-angle ISAR tools use.
_COMPOSITE_SPAN_DEG = 20.0
_COMPOSITE_SUB_DEG = 10.0


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
    az_center_deg: float | None = None,
    cancel_check=None,
):
    """Wide-aperture image as an incoherent composite of narrow looks — the
    processing that paints the full object OUTLINE from a 360° turntable
    sweep (each look angle contributes its specular glints; their union traces
    the body shape). Splits the band into ~10° sub-apertures, images each
    coherently in its own rotated frame (keystone-corrected FFT or Sparse L1),
    rotates every look into the common body frame (the θ=0 radar frame:
    +Y = down-range at 0° azimuth), and max-combines magnitudes so each
    pixel keeps its brightest look."""
    az_all = np.asarray(dataset.azimuths, dtype=float)
    az_vals = az_all[band_az_indices]
    if az_center_deg is not None:
        az_vals = _unwrap_degrees(az_vals, az_center_deg)
    order = np.argsort(az_vals)
    idx_sorted = [band_az_indices[i] for i in order]
    az_sorted = az_vals[order]
    span = float(az_sorted[-1] - az_sorted[0])
    n_sub = max(2, int(np.ceil(span / _COMPOSITE_SUB_DEG)))
    chunks = np.array_split(np.asarray(idx_sorted, dtype=np.int64), n_sub)

    subs = []
    az_nonuni = 0.0
    fr_nonuni = 0.0
    for chunk in chunks:
        if cancel_check is not None and cancel_check():
            return "ISAR computation superseded."
        chunk = [int(c) for c in chunk]
        if len(chunk) < 2:
            continue
        r = _compute_band(
            dataset, window_name, chunk, freq_indices_sorted, elev_idx, pol_idx,
            freq_hz, df, unit_scale,
            az_target_deg=None, recon=recon,
            l1_strength=l1_strength, l1_iters=l1_iters,
            az_center_deg=az_center_deg, elevation_deg=elevation_deg,
            cancel_check=cancel_check,
        )
        if isinstance(r, str):
            return r
        chunk_az = az_all[chunk]
        if az_center_deg is not None:
            chunk_az = _unwrap_degrees(chunk_az, az_center_deg)
        theta_c = np.deg2rad(float(np.mean(chunk_az)))
        subs.append((r, theta_c))
        az_nonuni = max(az_nonuni, r.get("az_nonuniformity", 0.0))
        fr_nonuni = max(fr_nonuni, r.get("freq_nonuniformity", 0.0))
    if not subs:
        return "Wide-aperture composite: no sub-aperture had ≥2 azimuth samples."

    # Body-frame grid: bounded by the smallest sub-image extent so looks
    # cover it from every angle; pixels a given look can't see contribute 0.
    half = min(
        min(float(np.max(np.abs(r["x_range"]))), float(np.max(np.abs(r["y_range"]))))
        for r, _ in subs
    )
    n_grid = 1024
    axis = np.linspace(-half, half, n_grid)
    xq = axis[:, None].astype(np.float32)
    yq = axis[None, :].astype(np.float32)
    comp = np.zeros((n_grid, n_grid), dtype=np.float32)
    for r, theta_c in subs:
        mag = np.asarray(r["magnitude"], dtype=np.float32)
        xa, ya = r["x_range"], r["y_range"]
        dx = float(xa[1] - xa[0])
        dy = float(ya[1] - ya[0])
        ct, st = np.float32(np.cos(theta_c)), np.float32(np.sin(theta_c))
        # body (x, y) -> this look's rotated-frame (cross-range, range)
        fx = (xq * ct - yq * st - np.float32(xa[0])) / np.float32(dx)
        fy = (xq * st + yq * ct - np.float32(ya[0])) / np.float32(dy)
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        valid = (ix >= 0) & (ix < mag.shape[0] - 1) & (iy >= 0) & (iy < mag.shape[1] - 1)
        ixc = np.clip(ix, 0, mag.shape[0] - 2)
        iyc = np.clip(iy, 0, mag.shape[1] - 2)
        wx = (fx - ix).astype(np.float32)
        wy = (fy - iy).astype(np.float32)
        val = (
            mag[ixc, iyc] * (1 - wx) * (1 - wy)
            + mag[ixc + 1, iyc] * wx * (1 - wy)
            + mag[ixc, iyc + 1] * (1 - wx) * wy
            + mag[ixc + 1, iyc + 1] * wx * wy
        )
        np.maximum(comp, np.where(valid, val, np.float32(0.0)), out=comp)

    peak = float(comp.max())
    if peak > 0.0:
        # keep uncovered corners off the -inf dB floor
        np.maximum(comp, peak * 1e-6, out=comp)

    return {
        "az_values": az_sorted,
        "magnitude": comp,
        "x_range": axis,
        "y_range": axis.copy(),
        "az_nonuniformity": az_nonuni,
        "freq_nonuniformity": fr_nonuni,
        "composite": len(subs),
    }


def compute_bands(params: dict):
    """Heavy half of the ISAR render: slicing, resampling, FFTs, display
    decimation. Pure numpy over the arrays captured in `params` — no Qt
    access — so the mixin runs it on a worker thread and the GUI stays live.
    Returns (band_results, elapsed_seconds), or an error-message string."""
    t_start = time.perf_counter()
    band_results = []
    cancel_check = params.get("cancel_check")
    for band_az_indices in params["bands"]:
        if cancel_check is not None and cancel_check():
            return "ISAR computation superseded."
        az_band = np.asarray(params["dataset"].azimuths, dtype=float)[band_az_indices]
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
                l1_iters=params.get("l1_iters", 100),
                elevation_deg=params.get("elevation_deg", 0.0),
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
                l1_iters=params.get("l1_iters", 100),
                elevation_deg=params.get("elevation_deg", 0.0),
                cancel_check=cancel_check,
            )
        if isinstance(result, str):
            return result
        # Convention flips, applied to the FINAL image so they are exactly
        # equivalent for single looks and composites (a global mirror commutes
        # through the sub-aperture rotations): Flip X mirrors about x=0
        # (opposite azimuth rotation direction), Flip Y about y=0 (opposite
        # down-range sign). Both checked = the e^{+j2kr} phase convention.
        if params.get("flip_x"):
            result["magnitude"] = result["magnitude"][::-1, :]
            result["x_range"] = -np.asarray(result["x_range"])[::-1]
        if params.get("flip_y"):
            result["magnitude"] = result["magnitude"][:, ::-1]
            result["y_range"] = -np.asarray(result["y_range"])[::-1]
        # GUI rendering can reduce oversized images after formation; headless
        # callers keep the full numerical grid by disabling this parameter.
        if params.get("decimate_display", True):
            result["magnitude"] = _decimate_display_max(result["magnitude"])
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
    l1_iterations: int = 100,
    flip_x: bool = False,
    flip_y: bool = False,
):
    """Form full-resolution ISAR images without Qt or display decimation.

    Returns ``(band_results, elapsed_seconds)`` using the same physical path as
    the GUI. ``reconstruction`` accepts ``fast``, ``accurate``, or ``sparse``.
    """
    az_indices = list(range(len(dataset.azimuths))) if azimuth_indices is None else sorted(
        int(i) for i in azimuth_indices
    )
    freq_indices = list(range(len(dataset.frequencies))) if frequency_indices is None else sorted(
        int(i) for i in frequency_indices
    )
    if len(az_indices) < 2 or len(freq_indices) < 2:
        raise ValueError("ISAR requires at least two azimuth and two frequency samples")
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
        str(dataset.units.get("frequency", "ghz"))
    )
    _, unit_scale = _length_unit(length_unit)
    recon_key = str(reconstruction).strip().lower()
    if recon_key in {"accurate", "cartesian", "pfa-accurate"}:
        recon_key = "accurate"
    elif recon_key in {"sparse", "l1", "sparse-l1"}:
        recon_key = "sparse"
    elif recon_key in {"fast", "fft", "pfa", "fast-pfa"}:
        recon_key = "fft"
    else:
        raise ValueError("reconstruction must be 'fast', 'accurate', or 'sparse'")

    if aperture_center_degrees is not None or azimuth_target_degrees is not None:
        bands = [az_indices]
    else:
        bands = [b for b in _split_into_bands(az_indices) if len(b) >= 2]
    result = compute_bands({
        "dataset": dataset,
        "bands": bands,
        "freq_indices_sorted": freq_indices,
        "elev_idx": int(elevation_index),
        "elevation_deg": float(dataset.elevations[elevation_index]),
        "pol_idx": int(polarization_index),
        "freq_hz": frequency_hz,
        "df": float(np.mean(np.diff(frequency_hz))),
        "unit_scale": unit_scale,
        "az_target_deg": None if azimuth_target_degrees is None else np.asarray(
            azimuth_target_degrees, dtype=float
        ),
        "az_center_deg": aperture_center_degrees,
        "window_name": window,
        "recon": recon_key,
        "l1_strength": float(l1_strength),
        "l1_iters": int(l1_iterations),
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
        "decimate_display": False,
    })
    if isinstance(result, str):
        raise ValueError(result)
    return result


def render(self) -> None:
    """GUI-thread half: validate the selections, capture everything the worker
    needs into a params dict, and hand off to the mixin's async submit. The
    finished computation comes back through `display_results`."""
    self.last_plot_mode = "isar_image"
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    if not az_indices:
        self.status.showMessage("Select one or more azimuths to plot.")
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
        az_arr = np.asarray(self.active_dataset.azimuths, dtype=float)[az_indices]
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
    elevation_deg = float(self.active_dataset.elevations[elev_idx])
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
        if not np.isfinite(az_min) or not np.isfinite(az_max) or not np.isfinite(az_step):
            self.status.showMessage("ISAR azimuth interp: limits/step must be finite.")
            return
        if az_step <= 0.0:
            self.status.showMessage("ISAR azimuth interp: step must be positive.")
            return
        if az_max <= az_min:
            self.status.showMessage("ISAR azimuth interp: max must exceed min.")
            return
        # arange-with-half-step so the inclusive upper bound lands on the grid
        # when (max-min) is an integer multiple of step (the common case).
        az_target_deg = np.arange(az_min, az_max + az_step * 0.5, az_step, dtype=float)
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
            "Each azimuth band needs at least 2 contiguous samples for ISAR imaging."
        )
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

    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
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
    l1_iters = int(l1_iters_spin.value()) if l1_iters_spin is not None else 100
    flip_x_widget = getattr(self, "chk_isar_flip_x", None)
    flip_x = bool(flip_x_widget.isChecked()) if flip_x_widget is not None else False
    flip_y_widget = getattr(self, "chk_isar_flip_y", None)
    flip_y = bool(flip_y_widget.isChecked()) if flip_y_widget is not None else False

    self._isar_submit({
        "dataset": self.active_dataset,
        # Identity token: if the active figure changed while computing (user
        # switched tabs), the finished result is dropped instead of being
        # painted onto whatever tab is now in front.
        "figure_token": self.plot_figure,
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
    })


def display_results(self, params: dict, band_results: list, elapsed: float) -> None:
    """GUI-thread half two: draw the computed band images. Runs from the
    mixin's worker-finished slot; everything Qt/matplotlib happens here."""
    dataset = params["dataset"]
    unit_name = params["unit_name"]
    az_target_deg = params["az_target_deg"]

    # Convert linear magnitudes to display values. ISAR image intensity is
    # 10·log₁₀(|I|²) = 20·log₁₀(|I|) — feed *power* (|I|²) to the dB converter,
    # not magnitude. `rcs_to_dbsm` interprets real inputs as already-linear
    # power, so passing the bare magnitude would give 10·log₁₀(|I|) (half the
    # correct dB value). Labelled "dB" rather than "dBsm" because the linear
    # value isn't necessarily in m². (The magnitude was already max-pool
    # decimated on the worker; block-max commutes with log.)
    for br in band_results:
        if self._plot_scale_is_linear():
            br["isar_display"] = br["magnitude"]
        else:
            br["isar_display"] = dataset.rcs_to_dbsm(br["magnitude"] ** 2)

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
            vmin=zmin if use_clamp else None,
            vmax=zmax if use_clamp else None,
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

    elev_value = dataset.elevations[params["elev_idx"]]
    pol_value = dataset.polarizations[params["pol_idx"]]
    if params.get("recon") == "sparse":
        recon_label = " | Sparse L1"
    elif params.get("recon") == "accurate":
        recon_label = " | Cartesian PFA"
    else:
        recon_label = " | Fast PFA"
    composite_subs = max((br.get("composite", 0) for br in band_results), default=0)
    if composite_subs:
        recon_label += f" | Wide-Aperture Composite ({composite_subs} looks)"
    fig_title = f"ISAR Image | Elevation {elev_value} deg | Pol {pol_value}{recon_label}"
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
        colorbar = self.plot_figure.colorbar(last_mesh, ax=active_axes)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label("Image Intensity (linear)", color=self._current_plot_text())
        else:
            colorbar.set_label("Image Intensity (dB)", color=self._current_plot_text())
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
        self._isar_last_autofit_state = state_key

    self._apply_plot_limits()

    # Surface any resampling that happened so the user knows their input
    # wasn't on a uniform grid. The number is the relative spread of native
    # spacings ((max-min)/median); anything > ~0.001 was actually resampled.
    az_max = max(br.get("az_nonuniformity", 0.0) for br in band_results)
    fr_max = max(br.get("freq_nonuniformity", 0.0) for br in band_results)
    if params.get("recon") == "sparse":
        mode_label = "Sparse L1"
    elif params.get("recon") == "accurate":
        mode_label = "Cartesian PFA"
    else:
        mode_label = "Fast PFA"
    parts = [f"ISAR image updated in {elapsed:.2f}s ({mode_label})"]
    if n_bands > 1:
        parts.append(f" ({n_bands} bands)")
    notes = []
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
    coverage = min(br.get("phase_coverage", 1.0) for br in band_results)
    if coverage < 0.9995:
        notes.append(f"weighted phase coverage {coverage*100:.1f}%")
    if any(br.get("preprocess_cache_hit", False) for br in band_results):
        notes.append("reused gridded phase history")
    sparse_used = [br.get("sparse_iterations") for br in band_results
                   if br.get("sparse_iterations") is not None]
    if sparse_used:
        notes.append(f"sparse convergence {max(sparse_used)} iterations")
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
    self.status.showMessage("".join(parts))
