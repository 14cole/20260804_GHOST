"""Bounded Cartesian interpolation with consistent measured support.

Cubic field interpolation is used only where its entire stencil is observed.
At a hole, both the zero-weighted field and coverage use the same positive
linear stencil. An origin point therefore has identical numerator/weight.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
import numpy as np

_PLAN_BYTES = 8 * 1024**2
_PLANS = OrderedDict()
_LOCK = threading.RLock()
_BYTES = 0


def plan_cache_bytes():
    with _LOCK:
        return _BYTES


def _plan(coordinates, n):
    global _BYTES
    # Cache small blocks only. Coordinate hashes bind reuse to actual geometry.
    raw = np.ascontiguousarray(coordinates, dtype=np.float64)
    cacheable = raw.size * 10 <= _PLAN_BYTES // 4
    key = (n, raw.shape, hashlib.blake2b(memoryview(raw).cast('B'), digest_size=16).digest()) if cacheable else None
    with _LOCK:
        cached = _PLANS.get(key)
        if cached is not None:
            _PLANS.move_to_end(key)
            return cached
    base = np.clip(np.floor(raw), 0, n - 2).astype(np.int32)
    t = (raw - base).astype(np.float32)
    valid = (raw >= -1e-8) & (raw <= n - 1 + 1e-8)
    np.clip(t, 0, 1, out=t)
    interior = (base >= 1) & (base <= n - 3)
    value = (base, t, valid, interior)
    nbytes = sum(a.nbytes for a in value)
    if nbytes <= _PLAN_BYTES // 4:
        for a in value:
            a.flags.writeable = False
        with _LOCK:
            old = _PLANS.pop(key, None)
            if old is not None:
                _BYTES -= sum(a.nbytes for a in old)
            _PLANS[key] = value
            _BYTES += nbytes
            while _BYTES > _PLAN_BYTES:
                _, old = _PLANS.popitem(last=False)
                _BYTES -= sum(a.nbytes for a in old)
    return value


def interpolate_pair(field, coverage, coordinates):
    """Apply one geometry plan to a complex numerator and positive coverage."""
    n = field.shape[0]
    base, t, valid, interior = _plan(coordinates, n)
    take = lambda a, j: np.take_along_axis(a, j, axis=0)
    f0, f1 = take(field, base), take(field, base + 1)
    w0, w1 = take(coverage, base), take(coverage, base + 1)
    out = f0 + (f1 - f0) * t
    weight = w0 + (w1 - w0) * t
    if np.any(interior):
        im1, i2 = np.maximum(base - 1, 0), np.minimum(base + 2, n - 1)
        wm1, w2 = take(coverage, im1), take(coverage, i2)
        complete = interior & (w0 == 1) & (w1 == 1) & (wm1 == 1) & (w2 == 1)
        if np.any(complete):
            cm1 = -t * (t - 1) * (t - 2) / 6
            c1 = -(t + 1) * t * (t - 2) / 2
            c2 = (t + 1) * t * (t - 1) / 6
            # Difference form preserves constant fields exactly despite roundoff.
            cubic = f0 + (take(field, im1) - f0) * cm1 + (f1 - f0) * c1 + (take(field, i2) - f0) * c2
            out[complete] = cubic[complete]
            weight[complete] = 1
    out[~valid] = 0
    weight[~valid] = 0
    return out, weight


def cartesian_pair(field, coverage, theta, frequency, axes, *, block_size=128, cancel_check=None):
    """Two Cartesian passes; share coordinate construction across both arrays."""
    q, axis_q, v = axes
    theta, frequency = np.asarray(theta), np.asarray(frequency)
    psi = theta - theta.mean()
    fc = frequency.mean()
    az_field = np.empty_like(field)
    az_weight = np.empty_like(coverage)
    az_block = min(block_size, max(1, 65536 // len(theta)))
    for start in range(0, len(frequency), az_block):
        if cancel_check is not None and cancel_check():
            raise InterruptedError('ISAR computation cancelled')
        end = min(start + az_block, len(frequency))
        arg = q[:, None] / (frequency[None, start:end] / fc)
        coordinates = (np.arcsin(np.clip(arg, -1, 1)) - psi[0]) / np.mean(np.diff(psi))
        f, w = interpolate_pair(field[:, start:end], coverage[:, start:end], coordinates)
        f[np.abs(arg) > 1] = 0
        w[np.abs(arg) > 1] = 0
        az_field[:, start:end], az_weight[:, start:end] = f, w
    out, weight = np.empty_like(field), np.empty_like(coverage)
    u = fc * q
    range_block = min(block_size, max(1, 65536 // len(frequency)))
    for start in range(0, len(q), range_block):
        if cancel_check is not None and cancel_check():
            raise InterruptedError('ISAR computation cancelled')
        end = min(start + range_block, len(q))
        required_f = np.sqrt(v[None, :]**2 + u[start:end, None]**2)
        coordinates = (required_f - frequency[0]) / np.mean(np.diff(frequency))
        f, w = interpolate_pair(az_field[start:end].T, az_weight[start:end].T, coordinates.T)
        out[start:end], weight[start:end] = f.T, w.T
    return out, weight, axis_q, v
