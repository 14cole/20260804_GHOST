"""Shared conservative mesh-bias selection for step-3 feature shadowing."""

import numpy as np

from occluder import Occluder


def _visibility_signature(occluder, points, normals, directions):
    chunks = []
    for direction in np.asarray(directions, dtype=float):
        lit = (normals @ direction) > 1e-3
        if np.any(lit):
            chunks.append(occluder.visible(points[lit], direction))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=bool)


def conservative_occluder(
    triangles,
    *,
    scale,
    points,
    normals,
    directions,
    override_m=None,
):
    """Return an occluder and audit record without enlarging its safe default.

    The half-default check can remove unnecessary offset. The validation point
    above the default is diagnostic only: automatically increasing the offset
    could step over a real nearby blocker.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("shadow calibration points must have shape (n, 3).")
    if normals.shape != points.shape:
        raise ValueError("shadow calibration normals must match points.")
    if directions.ndim != 2 or directions.shape[1] != 3 or len(directions) == 0:
        raise ValueError("shadow calibration directions must have shape (n, 3).")

    baseline = Occluder(triangles, scale=scale)
    if override_m is not None:
        selected = float(override_m)
        if not np.isfinite(selected) or selected < 0.0:
            raise ValueError("shadow bias override must be finite and nonnegative.")
        return Occluder(triangles, scale=scale, bias=selected), {
            "mode": "advanced_override",
            "selected_bias_m": selected,
            "mesh_default_bias_m": float(baseline.bias),
        }

    default = float(baseline.bias)
    half = 0.5 * default
    validation = min(1.25 * default, 1e-2 * float(baseline.diag))
    signatures = {
        bias: _visibility_signature(
            Occluder(triangles, scale=scale, bias=bias),
            points,
            normals,
            directions,
        )
        for bias in sorted({half, default, validation})
    }
    half_stable = bool(np.array_equal(signatures[half], signatures[default]))
    selected = half if half_stable else default
    validation_changes = int(
        np.count_nonzero(signatures[default] != signatures[validation])
    )
    return Occluder(triangles, scale=scale, bias=selected), {
        "mode": "automatic_conservative",
        "selected_bias_m": selected,
        "mesh_default_bias_m": default,
        "half_default_stable": half_stable,
        "changes_default_to_validation": validation_changes,
        "calibration_direction_count": int(len(directions)),
        "calibration_point_count": int(len(points)),
    }
