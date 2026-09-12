"""Static far-field ISAR planning, phase contracts, and origin PSF metrics."""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0
ENGINE_VERSION = "2.0.0"


def scene_extents(value):
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Scene half extents must be two finite positive distances in metres")
    return tuple(float(v) for v in values)


def aperture_mode(value):
    value = str(value).strip().lower()
    if value not in {"auto", "coherent", "composite"}:
        raise ValueError("aperture_mode must be 'auto', 'coherent', or 'composite'")
    return value


def plan_isar(azimuth_degrees, frequency_hz, *, elevation_degrees=0.0,
              scene_half_extent_m=None, reconstruction="fast", mode="auto"):
    """Estimate scene-dependent phase errors from acquired samples, before interpolation.

    Per-step phase checks are local sampling diagnostics, not a proof of
    invertibility or an interpolation-accuracy certificate.
    """
    az = np.sort(np.asarray(azimuth_degrees, dtype=float))
    freq = np.sort(np.asarray(frequency_hz, dtype=float))
    if az.ndim != 1 or freq.ndim != 1 or min(az.size, freq.size) < 2:
        raise ValueError("ISAR planning needs at least two azimuths and frequencies")
    if not np.all(np.isfinite(az)) or not np.all(np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("Planning axes must be finite and frequencies positive")
    if np.any(np.diff(az) <= 0) or np.any(np.diff(freq) <= 0):
        raise ValueError("Planning axes must contain distinct samples")
    elevation = float(elevation_degrees)
    projection = abs(float(np.cos(np.deg2rad(elevation))))
    if not np.isfinite(elevation) or projection < 1e-6:
        raise ValueError("Azimuth ISAR is degenerate at elevation ±90°")
    mode = aperture_mode(mode)
    if reconstruction not in {'fast', 'fft', 'accurate', 'sparse', 'auto'}:
        raise ValueError('Unknown ISAR reconstruction')
    span = float(np.ptp(az))
    composite = mode == "composite" or (mode == "auto" and span > 20)
    # Composite crop bounds use the fixed body frame. Coherent bounds use
    # the mean-look frame; native phase steps must use the matching basis.
    theta = np.deg2rad(az if composite else az - np.mean(az))
    half = scene_extents(scene_half_extent_m)
    explicit = half is not None
    if half is None:
        # Conservative advisory scene, derived from nominal periodic intervals.
        half = (C0 / (4 * projection * np.mean(freq) * np.median(np.diff(theta))),
                C0 / (4 * projection * np.median(np.diff(freq))))
    x, y = half
    factor = 4 * np.pi * projection / C0
    az_step = factor * freq[-1] * float(np.max(x * abs(np.diff(np.sin(theta))) + y * abs(np.diff(np.cos(theta)))))
    freq_step = factor * float(np.max(np.diff(freq))) * float(np.max(x * abs(np.sin(theta)) + y * abs(np.cos(theta))))
    center = np.deg2rad(np.mean(az))
    range_half = abs(np.sin(center))*x + abs(np.cos(center))*y if composite else y
    curvature = factor * freq[-1] * range_half * (1 - np.cos(np.deg2rad(span / 2)))
    recommended = "accurate" if curvature > np.pi / 4 or max(az_step, freq_step) > np.pi / 2 else "fft"
    selected = recommended if reconstruction == "auto" else ("fft" if reconstruction == "fast" else reconstruction)
    warnings = []
    if not explicit:
        warnings.append("Scene bounds use nominal periodic limits. Enter the occupied scene extent for a useful focus estimate.")
    if max(az_step, freq_step) >= np.pi:
        warnings.append("The requested scene exceeds the native per-step sampling limit. Interpolation does not recover missing information.")
    elif max(az_step, freq_step) > np.pi / 2:
        warnings.append("Phase changes exceed π/2 per sample near the scene edge; interpolation amplitude loss can be significant.")
    if selected in {"fft", "sparse"} and curvature > np.pi / 4:
        warnings.append("Fast-grid range-curvature error exceeds π/4 at the scene edge. Accurate PFA is recommended for coherent imaging.")
    if composite:
        warnings.append("This selection produces a qualitative maximum-magnitude composite with no coherent complex image.")
    if not composite and span >= 90:
        raise ValueError("A coherent PFA aperture must be narrower than 90°; choose composite or a narrower sector")
    return {
        "aperture_degrees": span, "center_degrees": float(np.mean(az)),
        "frequency_min_hz": float(freq[0]), "frequency_max_hz": float(freq[-1]),
        "elevation_degrees": elevation, "image_plane": "horizontal",
        "scene_half_extent_m": list(half), "scene_extent_source": "requested" if explicit else "nominal_periodic_limits",
        "scene_frame": "body_at_zero_azimuth" if composite else "mean_look",
        "fast_range_curvature_edge_rad": float(curvature),
        "max_native_azimuth_phase_step_rad": float(az_step),
        "max_native_frequency_phase_step_rad": float(freq_step),
        "native_step_check_passed": bool(max(az_step, freq_step) < np.pi),
        "recommended_reconstruction": recommended, "selected_reconstruction": selected,
        "aperture_mode": mode, "image_kind": "maximum_magnitude_composite" if composite else "coherent",
        "warnings": warnings,
    }


def image_contract(theta, frequency_hz, center_degrees, elevation_degrees, unit_scale):
    angle = np.deg2rad(center_degrees)
    theta = np.asarray(theta, dtype=float)
    freq = np.asarray(frequency_hz, dtype=float)
    return {
        "version": 1, "engine_version": ENGINE_VERSION,
        "image_plane": "horizontal", "distance_scale_per_metre": float(unit_scale),
        "image_kind": "coherent",
        "phase_convention": "spatial_frequency_origin_demodulated",
        "range_phase_convention": "S~exp(-j*2*k*R)",
        "phase_center": "dataset_fixed_origin",
        "spatial_frequency_origin_hz": [float(freq.mean() * theta[0]), float(freq[0])],
        "spatial_frequency_step_hz": [float(freq.mean() * np.mean(np.diff(theta))), float(np.mean(np.diff(freq)))],
        "projection_cos_elevation": abs(float(np.cos(np.deg2rad(elevation_degrees)))),
        "look_center_degrees": float(center_degrees), "elevation_degrees": float(elevation_degrees),
        "cross_range_basis": [float(np.cos(angle)), float(-np.sin(angle)), 0.0],
        "range_basis": [float(np.sin(angle)), float(np.cos(angle)), 0.0],
        "coordinate_signs": [1, 1],
        "amplitude_normalization": "unit_point_coherent_gain",
    }


def physical_coefficients(image, x_axis, y_axis, contract):
    """Restore physical point-coefficient phases without changing magnitude."""
    if contract.get("phase_convention") != "spatial_frequency_origin_demodulated":
        raise ValueError("Unsupported or missing complex image phase convention")
    scale = float(contract["distance_scale_per_metre"])
    u0, v0 = contract["spatial_frequency_origin_hz"]
    factor = 4j * np.pi / C0 * contract["projection_cos_elevation"]
    phase_x = np.exp(factor * u0 * np.asarray(x_axis) / scale)
    phase_y = np.exp(factor * v0 * np.asarray(y_axis) / scale)
    return np.asarray(image) * phase_x[:, None] * phase_y[None, :]


def _profile_metrics(aperture, spatial_step, projection, scale):
    n = len(aperture)
    count = min(65536, max(512, 8 * n))
    if count < n:
        return {"status": "not_computed", "reason": "PSF profile exceeds bounded diagnostic grid"}
    response = abs(np.fft.fftshift(np.fft.ifft(aperture, n=count)))**2
    center = count // 2
    peak = float(response[center])
    if peak <= 0:
        return {"status": "not_computed", "reason": "zero support"}
    response /= peak
    right = response[center:]
    below = np.flatnonzero(right <= .5)
    fwhm = None
    step = C0 * scale / (2 * projection * spatial_step * count)
    if below.size and below[0] > 0:
        i = int(below[0])
        crossing = i - 1 + (right[i - 1] - .5) / max(right[i - 1] - right[i], 1e-30)
        fwhm = float(2 * crossing * step)
    minima = np.flatnonzero((right[1:-1] <= right[:-2]) & (right[1:-1] < right[2:])) + 1
    if not minima.size:
        return {"status": "partial", "power_fwhm": fwhm, "reason": "no first PSF minimum resolved"}
    half = int(minima[0])
    main = np.zeros(count, dtype=bool)
    main[max(0, center - half):min(count, center + half + 1)] = True
    sidelobe = response[~main]
    return {"status": "computed", "power_fwhm": fwhm,
            "pslr_db": float(10 * np.log10(max(float(sidelobe.max(initial=0)), 1e-30))),
            "islr_db": float(10 * np.log10(max(float(sidelobe.sum()), 1e-30) / float(response[main].sum()))),
            "mainlobe_definition": "between first minima about origin peak"}


def psf_metrics(weights, window_az, window_freq, theta, frequency_hz, elevation_degrees, scale):
    """Actual origin PSF cuts for the windowed, regridded measurement support.

    Cut metrics are one-dimensional. They are neither a full 2D integrated
    sidelobe metric nor a guarantee of off-center focus.
    """
    projection = abs(float(np.cos(np.deg2rad(elevation_degrees))))
    theta, freq = np.asarray(theta), np.asarray(frequency_hz)
    cross = np.einsum('ij,j->i', weights, window_freq) * window_az
    down = np.einsum('ij,i->j', weights, window_az) * window_freq
    return {"definition": "origin PSF cuts on realized gridded support",
            "cross_range": _profile_metrics(cross, freq.mean() * np.mean(np.diff(theta)), projection, scale),
            "range": _profile_metrics(down, np.mean(np.diff(freq)), projection, scale)}
