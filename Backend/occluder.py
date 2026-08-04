#!/usr/bin/env python3
"""
Self-contained STL body-shadowing (geometric occlusion) for feature placement.

The line-expanded / point components mask themselves on their own surface normal
(d.n > 0), which is the correct terminator for a CONVEX body.  For a NON-convex
body (a boattail step, a recessed bay, one part hiding another) a feature can
face the radar yet be geometrically BLOCKED by the body.  This module adds that
missing test: read the clean vehicle STL, and for each look mask any feature
point whose straight line to the radar passes through the body.

Monostatic, so one ray per (point, direction) covers illumination AND return
(reciprocity).  This is geometric-optics (PO-level) blockage -- it captures a
feature going dark behind the body, NOT diffraction/creeping waves into the
shadow (a soft floor, same caveat as the grazing taper).  No dependencies:
Moller-Trumbore is replaced by an equivalent vectorised projected-occlusion
test (project along the look, 2-D point-in-triangle + depth), which is faster
here because all monostatic rays for one look are parallel.

    occ = Occluder.from_stl("vehicle_clean.stl", units="inches")
    sum_features(..., occluder=occ)          # threads into every component
"""

import struct
from typing import Optional

import numpy as np


def read_stl(path: 'str') -> 'np.ndarray':
    """Read a binary or ASCII STL into an (n_tri, 3, 3) array of vertices."""
    with open(path, "rb") as fh:
        raw = fh.read()
    # binary if the size matches the 84 + 50*n_tri layout
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if len(raw) == 84 + 50 * n:
            dt = np.dtype([("n", "<3f4"), ("v1", "<3f4"), ("v2", "<3f4"),
                           ("v3", "<3f4"), ("attr", "<u2")])
            recs = np.frombuffer(raw, dtype=dt, count=n, offset=84)
            return np.stack([recs["v1"], recs["v2"], recs["v3"]], axis=1).astype(float)
    # ASCII fallback
    verts = []
    for line in raw.decode("ascii", errors="ignore").splitlines():
        s = line.strip().split()
        if len(s) == 4 and s[0] == "vertex":
            verts.append([float(s[1]), float(s[2]), float(s[3])])
    v = np.asarray(verts, dtype=float)
    if v.size == 0 or v.shape[0] % 3 != 0:
        raise ValueError(f"{path}: could not parse as binary or ASCII STL.")
    return v.reshape(-1, 3, 3)


class Occluder:
    """Geometric shadow tester built from a triangle mesh (the clean body)."""

    def __init__(self, triangles: 'np.ndarray', scale: 'float' = 1.0,
                 bias: 'Optional[float]' = None):
        tris = np.asarray(triangles, dtype=float)
        if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
            raise ValueError("triangles must have shape (n, 3, 3) with n > 0.")
        if not np.all(np.isfinite(tris)):
            raise ValueError("STL triangles contain NaN or infinite coordinates.")
        scl = float(scale)
        if not np.isfinite(scl) or scl <= 0.0:
            raise ValueError("STL scale must be finite and positive.")
        self.tris = tris * scl                                      # (n,3,3)
        lo = self.tris.reshape(-1, 3).min(0)
        hi = self.tris.reshape(-1, 3).max(0)
        self.diag = float(np.linalg.norm(hi - lo)) or 1.0
        # typical facet size (median over NON-degenerate edges: a revolved mesh has
        # zero-length edges at the axis, and a few huge triangles should not set it)
        _e = np.linalg.norm(self.tris[:, [1, 2, 0], :] - self.tris, axis=2).ravel()
        _e = _e[_e > 0.0]
        self.median_edge = float(np.median(_e)) if _e.size else 0.0
        # Offset along the ray to lift a surface point off its own facet.  It must
        # exceed the SAG between the true surface and the facets that approximate
        # it -- a feature drawn on a curved surface (and the polyline that traces
        # it) is inscribed in that surface, so the mesh sits a hair IN FRONT of it
        # and the body appears to block its own features ("shadow acne", tens of dB
        # on a coarse mesh).  The sag scales with facet size, so the default does
        # too; it is capped at 1% of the body size so a mesh made of a few huge
        # facets cannot get a bias large enough to swallow real blockage, and
        # floored at the old 1e-4 x diagonal for very fine meshes.
        # Calibrate on your own mesh: on a CONVEX body the occluder must change
        # nothing (calibrate it with 0_calibrate_shadowing).
        self.bias = (float(bias) if bias is not None
                     else max(1e-4 * self.diag,
                              min(0.2 * self.median_edge, 1e-2 * self.diag)))
        if not np.isfinite(self.bias) or self.bias < 0.0:
            raise ValueError("occlusion bias must be finite and non-negative.")

    @classmethod
    def from_stl(cls, path: 'str', units: 'str' = "meters", bias: 'Optional[float]' = None):
        scales = {"m": 1.0, "meter": 1.0, "meters": 1.0,
                  "mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3,
                  "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
                  "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}
        key = str(units).strip().lower()
        if key not in scales:
            raise ValueError(
                f"Unsupported STL units {units!r}; use meters, millimeters, "
                "inches, or feet.")
        scale = scales[key]
        return cls(read_stl(path), scale=scale, bias=bias)

    def visible(self, points: 'np.ndarray', direction: 'np.ndarray',
                bias: 'Optional[float]' = None) -> 'np.ndarray':
        """Boolean visibility of ``points`` for a COMING-FROM look ``direction``.

        A point is False (shadowed) if the body blocks the segment from the
        point toward the radar (along +direction, to infinity)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        e = self.bias if bias is None else float(bias)
        if not np.isfinite(e) or e < 0.0:
            raise ValueError("occlusion bias must be finite and non-negative.")
        # orthonormal frame with w = +d (toward the radar)
        seed = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = seed - (seed @ d) * d
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)

        A = self.tris[:, 0, :]; B = self.tris[:, 1, :]; C = self.tris[:, 2, :]
        Au = np.column_stack([A @ e1, A @ e2]); Aw = A @ d
        Bu = np.column_stack([B @ e1, B @ e2]); Bw = B @ d
        Cu = np.column_stack([C @ e1, C @ e2]); Cw = C @ d
        v0 = Bu - Au; v1 = Cu - Au
        det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]           # 2*signed area
        okdet = np.abs(det) > 1e-14 * (self.diag ** 2)           # drop edge-on tris
        det_s = np.where(okdet, det, 1.0)

        Pu = np.column_stack([pts @ e1, pts @ e2])
        Pw = pts @ d
        vis = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            v2 = Pu[i] - Au
            b = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / det_s
            c = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / det_s
            a = 1.0 - b - c
            inside = okdet & (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
            w_at = a * Aw + b * Bw + c * Cw                       # body depth under P
            if np.any(inside & (w_at > Pw[i] + e)):               # body is in front -> blocked
                vis[i] = False
        return vis
