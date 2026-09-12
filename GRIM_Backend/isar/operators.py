"""Bounded native polar forward/adjoint reference and image residual diagnostics."""
from __future__ import annotations

import numpy as np
from .quality import C0


class PolarPointOperator:
    """Exact two-way far-field point operator with bounded phase blocks.

    Rows correspond to paired acquired azimuth/frequency coordinates. Columns
    are physical planar points in the look frame, with physical coefficients.
    The adjoint is not an inverse. Large products fail before phase allocation.
    """
    def __init__(self, azimuth_radians, frequency_hz, points_m, *, elevation_degrees=0.,
                 maximum_interactions=32_000_000, maximum_block_bytes=4 * 1024**2, cancel_check=None):
        az = np.asarray(azimuth_radians, dtype=float).reshape(-1)
        freq = np.asarray(frequency_hz, dtype=float).reshape(-1)
        points = np.asarray(points_m, dtype=float)
        if az.shape != freq.shape or points.ndim != 2 or points.shape[1] != 2 or not len(az) or not len(points):
            raise ValueError("Native polar operator requires paired samples and (x,y) points")
        if not all(np.all(np.isfinite(v)) for v in (az, freq, points)) or np.any(freq <= 0):
            raise ValueError("Native operator coordinates must be finite and frequencies positive")
        if maximum_interactions < 1 or maximum_block_bytes < 48:
            raise ValueError('Native operator budgets must allow at least one interaction')
        if len(az) * len(points) > int(maximum_interactions):
            raise ValueError("Native direct operator exceeds its interaction budget; use a smaller scene or a validated accelerated operator")
        elevation = float(elevation_degrees)
        if not np.isfinite(elevation):
            raise ValueError("Elevation must be finite")
        self.points = points.copy()
        factor = 4 * np.pi / C0 * abs(np.cos(np.deg2rad(elevation)))
        self.kx = factor * freq * np.sin(az)
        self.ky = factor * freq * np.cos(az)
        self.shape = (len(az), len(points))
        self.block_cells = max(1, int(maximum_block_bytes) // 48)
        self.cancel_check = cancel_check

    def _blocks(self):
        columns = min(self.shape[1], max(1, int(np.sqrt(self.block_cells))))
        rows = max(1, self.block_cells // columns)
        for start in range(0, self.shape[0], rows):
            stop = min(start + rows, self.shape[0])
            for first in range(0, self.shape[1], columns):
                if self.cancel_check is not None and self.cancel_check():
                    raise InterruptedError("Native polar operation cancelled")
                last = min(first + columns, self.shape[1])
                phase = self.kx[start:stop, None] * self.points[None, first:last, 0]
                phase += self.ky[start:stop, None] * self.points[None, first:last, 1]
                yield slice(start, stop), slice(first, last), np.exp(-1j * phase)

    def forward(self, coefficients):
        value = np.asarray(coefficients, dtype=np.complex128)
        if value.shape != (self.shape[1],) or not np.all(np.isfinite(value)):
            raise ValueError("Native coefficient vector has wrong size or nonfinite values")
        result = np.zeros(self.shape[0], dtype=np.complex128)
        for rows, cols, phase in self._blocks():
            result[rows] += phase @ value[cols]
        return result

    def adjoint(self, samples):
        value = np.asarray(samples, dtype=np.complex128)
        if value.shape != (self.shape[0],) or not np.all(np.isfinite(value)):
            raise ValueError("Native sample vector has wrong size or nonfinite values")
        result = np.zeros(self.shape[1], dtype=np.complex128)
        for rows, cols, phase in self._blocks():
            result[cols] += phase.conj().T @ value[rows]
        return result


def native_image_residual(image, x_axis, y_axis, contract, azimuth_degrees, frequency_hz,
                          source, *, support_threshold=0., maximum_samples=4096,
                          maximum_points=512, maximum_interactions=8_000_000, cancel_check=None):
    """Fit check on original samples with explicit sampling/truncation semantics."""
    image = np.asarray(image)
    magnitude = abs(image)
    ix, iy = np.nonzero(magnitude > max(float(support_threshold), 0.))
    if len(ix) > maximum_points:
        return {"status": "not_computed", "reason": f"Image support exceeds {maximum_points} diagnostic points", "support_pixels": len(ix)}
    if not len(ix):
        return {"status": "not_computed", "reason": "No retained image support"}
    az, freq = np.asarray(azimuth_degrees), np.asarray(frequency_hz)
    total = len(az) * len(freq)
    sample_limit = min(maximum_samples, max(1, maximum_interactions // len(ix)))
    ids = np.unique(np.linspace(0, total - 1, min(total, sample_limit), dtype=np.int64))
    ia, jf = np.unravel_index(ids, (len(az), len(freq)))
    measured = np.asarray(source(ia, jf), dtype=np.complex128)
    finite = np.isfinite(measured)
    ia, jf, measured = ia[finite], jf[finite], measured[finite]
    if not len(measured) or np.linalg.norm(measured) == 0:
        return {"status": "not_computed", "reason": "No finite nonzero native diagnostic samples"}
    scale = contract['distance_scale_per_metre']
    x, y = np.asarray(x_axis)[ix] / scale, np.asarray(y_axis)[iy] / scale
    u0, v0 = contract['spatial_frequency_origin_hz']
    coefficients = image[ix, iy] * np.exp(4j * np.pi / C0 * contract['projection_cos_elevation'] * (u0 * x + v0 * y))
    op = PolarPointOperator(np.deg2rad(az[ia] - contract['look_center_degrees']), freq[jf], np.column_stack((x, y)),
        elevation_degrees=contract['elevation_degrees'], maximum_interactions=maximum_interactions, cancel_check=cancel_check)
    predicted = op.forward(coefficients)
    residual = float(np.linalg.norm(predicted - measured) / np.linalg.norm(measured))
    retained_energy = float(np.sum(magnitude[ix, iy].astype(float)**2))
    total_energy = float(np.sum(magnitude.astype(float)**2))
    return {"status": "computed", "relative_complex_l2_residual": residual,
            "sample_count": len(measured), "source_sample_count": total,
            "sampled": len(ids) < total, "sample_selection": "uniform source-index subset" if len(ids) < total else "all source samples",
            "support_pixels": len(ix), "support_threshold": float(support_threshold),
            "omitted_image_energy_fraction": max(0., 1 - retained_energy / max(total_energy, 1e-30)),
            "operator": "native_polar_point_sum", "high_model_mismatch": residual > .2,
            "scope": "full formed image before optional scene crop and display flips",
            "warning": 'Native residual exceeds 20%; gridded convergence does not certify physical agreement.' if residual > .2 else ''}
