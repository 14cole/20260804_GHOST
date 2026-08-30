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
Queries use an exact double-sided Moller-Trumbore ray/triangle test accelerated
by a lazily built Morton-ordered bounding-volume hierarchy (BVH).  The same BVH
is reused for every radar direction, and leaf triangles are evaluated in
bounded vectorized batches so fine vehicle meshes do not create one enormous
point-by-triangle temporary array.

    occ = Occluder.from_stl("vehicle_clean.stl", units="inches")
    sum_features(..., occluder=occ)          # threads into every component
"""

import math
import inspect
import struct
import threading
from typing import Callable, Optional

import numpy as np


def visible_query_adapter(occluder):
    """Bind a visibility query across current and historical integrations.

    Older third-party occluders expose only ``visible(points, direction)``.
    Inspecting once avoids both per-look signature overhead and the unsafe
    pattern of catching a ``TypeError`` that may have originated inside the
    third-party implementation rather than at its call boundary.
    """

    visible = getattr(occluder, "visible", None)
    if not callable(visible):
        raise TypeError("occluder must expose callable visible(points, direction).")
    try:
        parameters = inspect.signature(visible).parameters.values()
    except (TypeError, ValueError):
        accepts_cancel = False
    else:
        accepts_cancel = any(
            parameter.name == "cancel_check"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    if accepts_cancel:
        return lambda points, direction, cancel_check=None: visible(
            points, direction, cancel_check=cancel_check
        )
    return lambda points, direction, cancel_check=None: visible(
        points, direction
    )


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


class PackedVisibilityRow:
    """Read-only point visibility over looks without unpacking a dense row."""

    __slots__ = ("_packed", "_n_directions")

    def __init__(self, packed: 'np.ndarray', n_directions: int):
        self._packed = packed
        self._n_directions = int(n_directions)

    @property
    def shape(self) -> 'tuple':
        return (self._n_directions,)

    def __len__(self) -> int:
        return self._n_directions

    def __getitem__(self, direction_index: int) -> bool:
        index = int(direction_index)
        if index < 0:
            index += self._n_directions
        if index < 0 or index >= self._n_directions:
            raise IndexError("visibility direction index is out of range")
        return bool(
            int(self._packed[index // 8]) & (1 << (index % 8))
        )

    def to_dense(self) -> 'np.ndarray':
        """Materialize one row for compatibility/debugging, never implicitly."""

        return np.unpackbits(
            self._packed, bitorder="little"
        )[:self._n_directions].astype(bool, copy=False)


class PackedVisibility:
    """Immutable point-major visibility stored as one bit per point/look pair.

    ``shape`` is ``(n_points, n_directions)``. Production point evaluation
    obtains a :class:`PackedVisibilityRow` and tests only requested lit looks;
    it never materializes the full Boolean matrix.
    """

    __slots__ = ("_packed", "_n_points", "_n_directions")

    def __init__(
        self,
        packed: 'np.ndarray',
        *,
        n_points: int,
        n_directions: int,
    ):
        point_count = int(n_points)
        direction_count = int(n_directions)
        if point_count < 0 or direction_count < 0:
            raise ValueError("Packed visibility dimensions must be nonnegative.")
        byte_count = (direction_count + 7) // 8
        raw = np.asarray(packed)
        if raw.dtype != np.uint8 or raw.shape != (point_count, byte_count):
            raise ValueError(
                "packed visibility bytes must have uint8 shape "
                "(n_points, ceil(n_directions / 8))."
            )
        unused = direction_count % 8
        if unused and point_count and np.any(
            raw[:, -1] & np.uint8(0xff ^ ((1 << unused) - 1))
        ):
            raise ValueError("Packed visibility has nonzero unused trailing bits.")
        immutable = np.frombuffer(
            np.ascontiguousarray(raw).tobytes(), dtype=np.uint8
        ).reshape(raw.shape)
        self._packed = immutable
        self._n_points = point_count
        self._n_directions = direction_count

    @property
    def shape(self) -> 'tuple':
        return (self._n_points, self._n_directions)

    @property
    def nbytes(self) -> int:
        return int(self._packed.nbytes)

    def row(self, point_index: int) -> 'PackedVisibilityRow':
        index = int(point_index)
        if index < 0:
            index += self._n_points
        if index < 0 or index >= self._n_points:
            raise IndexError("visibility point index is out of range")
        return PackedVisibilityRow(self._packed[index], self._n_directions)

    def column(self, direction_index: int) -> 'np.ndarray':
        """Materialize one look over points without unpacking other looks."""

        index = int(direction_index)
        if index < 0:
            index += self._n_directions
        if index < 0 or index >= self._n_directions:
            raise IndexError("visibility direction index is out of range")
        return (
            self._packed[:, index // 8] & np.uint8(1 << (index % 8))
        ).astype(bool, copy=False)

    def to_dense(self) -> 'np.ndarray':
        """Materialize the point-major matrix for tests/legacy adapters only."""

        if self._n_directions == 0:
            return np.empty((self._n_points, 0), dtype=bool)
        return np.unpackbits(
            self._packed, axis=1, bitorder="little"
        )[:, :self._n_directions].astype(bool, copy=False)


class Occluder:
    """Geometric shadow tester built from a triangle mesh (the clean body).

    Visibility queries use a lazily-built, direction-independent bounding-volume
    hierarchy (BVH).  The previous implementation projected *every* triangle for
    *every* feature point and look, which made vehicle-scale meshes effectively
    unusable.  The BVH keeps the same exact ray/triangle decision while reducing
    the normal query from ``O(n_triangles)`` to a small tree traversal.  Building
    it lazily also keeps Assembly validation and preview responsive when body
    shadowing has not actually been requested by a solve.  BVH boxes are
    conservatively padded to cover the leaf test's barycentric edge tolerance;
    the leaf ray/triangle decision remains authoritative.
    """

    # A moderately wide leaf is intentional: intersection within a leaf is a
    # NumPy vector operation, while every BVH node visit is Python work.  This
    # value performed best across locally tessellated vehicle panels and sparse
    # random triangle soups without materially increasing retained memory.
    _LEAF_TRIANGLES = 256

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
        # Back the public read-only view with an immutable ``bytes`` object.
        # ``setflags(write=False)`` alone is reversible when an ndarray owns
        # writable memory; a caller holding a reviewed plan could otherwise
        # re-enable writes from a progress callback after the plan hash check.
        scaled = np.ascontiguousarray(tris if scl == 1.0 else tris * scl)
        self._tris = np.frombuffer(
            scaled.tobytes(), dtype=scaled.dtype
        ).reshape(scaled.shape)                                      # (n,3,3)
        lo = self._tris.reshape(-1, 3).min(0)
        hi = self._tris.reshape(-1, 3).max(0)
        self.diag = float(np.linalg.norm(hi - lo)) or 1.0
        # typical facet size (median over NON-degenerate edges: a revolved mesh has
        # zero-length edges at the axis, and a few huge triangles should not set it)
        _e = np.linalg.norm(
            self._tris[:, [1, 2, 0], :] - self._tris, axis=2
        ).ravel()
        _e = _e[_e > 0.0]
        self.median_edge = float(np.median(_e)) if _e.size else 0.0
        # This is only a floating-point self-hit tolerance, not a geometric
        # offset intended to bridge CAD-to-mesh sag.  The previous edge-scaled
        # default could reach centimetres on a vehicle and silently skip a real
        # nearby blocker.  Placements must therefore be registered to the same
        # mesh; a larger deliberate bias remains available and is recorded by
        # the Assembly workflow.
        self._bias = (float(bias) if bias is not None
                      else max(
                          1e-9 * self.diag,
                          min(1e-6 * self.median_edge, 1e-6 * self.diag),
                      ))
        if not np.isfinite(self._bias) or self._bias < 0.0:
            raise ValueError("occlusion bias must be finite and non-negative.")
        self._bvh_lock = threading.Lock()
        self._bvh_ready = False
        self._bvh_base = 0
        self._bvh_lo = None
        self._bvh_hi = None
        self._tri_edge1 = None
        self._tri_edge2 = None

    @property
    def tris(self) -> 'np.ndarray':
        """Read-only triangle geometry used for visibility decisions."""

        return self._tris

    @property
    def bias(self) -> float:
        """Immutable self-hit distance established at validation."""

        return float(self._bias)

    def execution_snapshot(self) -> 'Occluder':
        """Return isolated execution state sharing only immutable geometry.

        Triangle coordinates are backed by ``bytes`` and a completed BVH is
        published with read-only arrays, so those large buffers are safe to
        share.  Scalar state and the lock remain private to the snapshot: even
        a caller that mutates a reviewed object's private bias from a progress
        callback cannot alter the in-flight visibility calculation.
        """

        clone = object.__new__(type(self))
        with self._bvh_lock:
            clone._tris = self._tris
            clone.diag = float(self.diag)
            clone.median_edge = float(self.median_edge)
            clone._bias = float(self._bias)
            clone._bvh_ready = bool(self._bvh_ready)
            clone._bvh_base = int(self._bvh_base)
            clone._bvh_lo = self._bvh_lo
            clone._bvh_hi = self._bvh_hi
            clone._tri_edge1 = self._tri_edge1
            clone._tri_edge2 = self._tri_edge2
        clone._bvh_lock = threading.Lock()
        return clone

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

    @staticmethod
    def _morton_codes(centres: 'np.ndarray') -> 'np.ndarray':
        """Return 30-bit Morton codes used only to make spatial BVH leaves.

        Morton ordering is inexpensive, deterministic, and avoids retaining a
        large centroid tree.  It is not part of the physical answer; the BVH
        still performs exact AABB and triangle intersection tests.
        """
        lo = centres.min(axis=0)
        span = centres.max(axis=0) - lo
        safe_span = np.where(span > 0.0, span, 1.0)
        q = np.floor(np.clip((centres - lo) / safe_span, 0.0, 1.0)
                     * 1023.0).astype(np.uint64)

        def _spread(v):
            v = v & np.uint64(0x3ff)
            v = (v | (v << np.uint64(16))) & np.uint64(0x30000ff)
            v = (v | (v << np.uint64(8))) & np.uint64(0x300f00f)
            v = (v | (v << np.uint64(4))) & np.uint64(0x30c30c3)
            v = (v | (v << np.uint64(2))) & np.uint64(0x9249249)
            return v

        return _spread(q[:, 0]) | (_spread(q[:, 1]) << np.uint64(1)) \
            | (_spread(q[:, 2]) << np.uint64(2))

    @staticmethod
    def _cancelled(cancel_check: 'Optional[Callable[[], bool]]') -> bool:
        return bool(cancel_check is not None and cancel_check())

    def prepare_acceleration(
            self,
            cancel_check: 'Optional[Callable[[], bool]]' = None) -> None:
        """Build the immutable BVH, if needed.

        This method is safe to call more than once and from competing worker
        threads.  Callers may provide a cooperative cancellation predicate;
        cancellation never publishes a partial index.
        """
        if self._bvh_ready:
            return
        with self._bvh_lock:
            if self._bvh_ready:
                return
            if self._cancelled(cancel_check):
                raise InterruptedError("Body-shadow acceleration build cancelled.")

            centres = self.tris.mean(axis=1)
            order = np.argsort(self._morton_codes(centres), kind="stable")
            del centres
            if self._cancelled(cancel_check):
                raise InterruptedError("Body-shadow acceleration build cancelled.")

            # Reordering is internal-only and makes every leaf a contiguous
            # slice, avoiding an additional per-triangle index allocation.
            ordered = self.tris[order]
            leaf_size = int(self._LEAF_TRIANGLES)
            n_triangles = len(ordered)
            n_leaves = int(math.ceil(n_triangles / leaf_size))
            base = 1 << max(0, (n_leaves - 1).bit_length())
            bvh_lo = np.full((2 * base, 3), np.inf, dtype=float)
            bvh_hi = np.full((2 * base, 3), -np.inf, dtype=float)

            starts = np.arange(0, n_triangles, leaf_size, dtype=np.intp)
            tri_lo = ordered.min(axis=1)
            tri_hi = ordered.max(axis=1)
            bvh_lo[base:base + n_leaves] = np.minimum.reduceat(tri_lo, starts)
            bvh_hi[base:base + n_leaves] = np.maximum.reduceat(tri_hi, starts)
            del tri_lo, tri_hi

            level = base // 2
            while level:
                nodes = np.arange(level, 2 * level, dtype=np.intp)
                bvh_lo[nodes] = np.minimum(bvh_lo[2 * nodes],
                                           bvh_lo[2 * nodes + 1])
                bvh_hi[nodes] = np.maximum(bvh_hi[2 * nodes],
                                           bvh_hi[2 * nodes + 1])
                level //= 2
            if self._cancelled(cancel_check):
                raise InterruptedError("Body-shadow acceleration build cancelled.")

            # Publish together only after every array is complete.
            immutable_ordered = np.frombuffer(
                np.ascontiguousarray(ordered).tobytes(),
                dtype=ordered.dtype,
            ).reshape(ordered.shape)
            self._tris = immutable_ordered
            self._tri_edge1 = np.frombuffer(
                np.ascontiguousarray(ordered[:, 1] - ordered[:, 0]).tobytes(),
                dtype=ordered.dtype,
            ).reshape((n_triangles, 3))
            self._tri_edge2 = np.frombuffer(
                np.ascontiguousarray(ordered[:, 2] - ordered[:, 0]).tobytes(),
                dtype=ordered.dtype,
            ).reshape((n_triangles, 3))
            self._bvh_base = base
            # As with triangles, a write=False flag on an owning ndarray can be
            # reversed by outside code. Back the published boxes with immutable
            # bytes so execution snapshots may safely share the acceleration
            # structure with a reviewed plan.
            self._bvh_lo = np.frombuffer(
                bvh_lo.tobytes(), dtype=bvh_lo.dtype
            ).reshape(bvh_lo.shape)
            self._bvh_hi = np.frombuffer(
                bvh_hi.tobytes(), dtype=bvh_hi.dtype
            ).reshape(bvh_hi.shape)
            self._bvh_ready = True

    @property
    def acceleration_info(self) -> 'dict':
        """Small diagnostic summary suitable for logs and support bundles."""
        n_triangles = int(len(self.tris))
        return {
            "kind": "morton_bvh",
            "ready": bool(self._bvh_ready),
            "triangle_count": n_triangles,
            "leaf_triangles": int(self._LEAF_TRIANGLES),
            "leaf_count": int(math.ceil(n_triangles / self._LEAF_TRIANGLES)),
            "node_slots": int(2 * self._bvh_base) if self._bvh_ready else 0,
        }

    def _aabb_entry(self, node: int, origin: 'np.ndarray',
                    direction: 'np.ndarray', minimum_t: float) -> 'Optional[float]':
        # Moller-Trumbore accepts barycentric coordinates through 1e-9 at
        # edges/vertices.  Such an accepted hit can lie just outside a raw
        # coordinate AABB.  Pad every axis (not only parallel-ray axes) by a
        # conservative bound so acceleration can never cull a leaf the exact
        # triangle test would accept.
        coordinate_tol = 4e-9 * self.diag
        lo_values = self._bvh_lo[node]
        hi_values = self._bvh_hi[node]
        near = -np.inf
        far = np.inf
        for axis in range(3):
            lo = float(lo_values[axis]) - coordinate_tol
            hi = float(hi_values[axis]) + coordinate_tol
            if lo > hi:
                return None
            component = float(direction[axis])
            if abs(component) <= 1e-15:
                if origin[axis] < lo or origin[axis] > hi:
                    return None
                continue
            t0 = (lo - origin[axis]) / component
            t1 = (hi - origin[axis]) / component
            if t0 > t1:
                t0, t1 = t1, t0
            near = max(near, t0)
            far = min(far, t1)
            if far < max(near, minimum_t):
                return None
        return max(near, minimum_t)

    def _leaf_hit(self, leaf: int, origin: 'np.ndarray',
                  direction: 'np.ndarray', minimum_t: float) -> bool:
        leaf_index = leaf - self._bvh_base
        start = leaf_index * self._LEAF_TRIANGLES
        stop = min(start + self._LEAF_TRIANGLES, len(self.tris))
        if start >= stop:
            return False
        tri0 = self.tris[start:stop, 0]
        edge1 = self._tri_edge1[start:stop]
        edge2 = self._tri_edge2[start:stop]
        h = np.cross(direction[None, :], edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        determinant_tol = 1e-14 * (self.diag ** 2)
        valid = np.abs(determinant) > determinant_tol
        if not np.any(valid):
            return False
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        s = origin[None, :] - tri0
        u = inverse * np.einsum("ij,ij->i", s, h)
        valid &= (u >= -1e-9) & (u <= 1.0 + 1e-9)
        if not np.any(valid):
            return False
        q = np.cross(s, edge1)
        v = inverse * (q @ direction)
        valid &= (v >= -1e-9) & ((u + v) <= 1.0 + 1e-9)
        if not np.any(valid):
            return False
        distance = inverse * np.einsum("ij,ij->i", edge2, q)
        return bool(np.any(valid & (distance > minimum_t)))

    def _ray_hits_mesh(self, origin: 'np.ndarray', direction: 'np.ndarray',
                       minimum_t: float,
                       cancel_check: 'Optional[Callable[[], bool]]' = None) -> bool:
        root_entry = self._aabb_entry(1, origin, direction, minimum_t)
        if root_entry is None:
            return False
        stack = [(1, root_entry)]
        visited = 0
        while stack:
            node, _entry = stack.pop()
            visited += 1
            if visited % 256 == 0 and self._cancelled(cancel_check):
                raise InterruptedError("Body-shadow query cancelled.")
            if node >= self._bvh_base:
                if self._leaf_hit(node, origin, direction, minimum_t):
                    return True
                continue
            left = 2 * node
            right = left + 1
            left_entry = self._aabb_entry(left, origin, direction, minimum_t)
            right_entry = self._aabb_entry(right, origin, direction, minimum_t)
            # Visit the nearer child first.  The stack is LIFO, hence far first.
            if left_entry is None:
                if right_entry is not None:
                    stack.append((right, right_entry))
            elif right_entry is None:
                stack.append((left, left_entry))
            elif left_entry <= right_entry:
                stack.append((right, right_entry))
                stack.append((left, left_entry))
            else:
                stack.append((left, left_entry))
                stack.append((right, right_entry))
        return False

    def visible(self, points: 'np.ndarray', direction: 'np.ndarray',
                bias: 'Optional[float]' = None, *,
                cancel_check: 'Optional[Callable[[], bool]]' = None) -> 'np.ndarray':
        """Boolean visibility of ``points`` for a COMING-FROM look ``direction``.

        A point is False (shadowed) if the body blocks the segment from the
        point toward the radar (along +direction, to infinity)."""
        raw_points = np.asarray(points, dtype=float)
        pts = np.atleast_2d(raw_points)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (n, 3) or (3,).")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points contain NaN or infinite coordinates.")
        d = np.asarray(direction, dtype=float)
        if d.shape != (3,) or not np.all(np.isfinite(d)):
            raise ValueError("direction must be a finite 3-vector.")
        direction_norm = float(np.linalg.norm(d))
        if direction_norm <= 1e-15:
            raise ValueError("direction must be nonzero.")
        d = d / direction_norm
        e = self.bias if bias is None else float(bias)
        if not np.isfinite(e) or e < 0.0:
            raise ValueError("occlusion bias must be finite and non-negative.")
        if len(pts) == 0:
            return np.ones(0, dtype=bool)
        self.prepare_acceleration(cancel_check=cancel_check)
        vis = np.ones(len(pts), dtype=bool)
        for i, point in enumerate(pts):
            if i % 64 == 0 and self._cancelled(cancel_check):
                raise InterruptedError("Body-shadow query cancelled.")
            vis[i] = not self._ray_hits_mesh(
                point, d, e, cancel_check=cancel_check)
        return vis

    def visible_many(self, points: 'np.ndarray', directions: 'np.ndarray',
                     bias: 'Optional[float]' = None, *,
                     cancel_check: 'Optional[Callable[[], bool]]' = None,
                     progress_callback: 'Optional[Callable[[int, int], None]]' = None
                     ) -> 'np.ndarray':
        """Return a ``(n_directions, n_points)`` visibility matrix.

        This convenience API centralizes progress and cancellation for worker
        jobs without allocating any triangle-by-ray intermediate arrays.
        """
        dirs = np.atleast_2d(np.asarray(directions, dtype=float))
        if dirs.ndim != 2 or dirs.shape[1] != 3:
            raise ValueError("directions must have shape (n, 3) or (3,).")
        result = np.empty((len(dirs), len(np.atleast_2d(points))), dtype=bool)
        for index, direction in enumerate(dirs):
            if self._cancelled(cancel_check):
                raise InterruptedError("Body-shadow query cancelled.")
            result[index] = self.visible(
                points, direction, bias=bias, cancel_check=cancel_check)
            if progress_callback is not None:
                progress_callback(index + 1, len(dirs))
        return result

    def visible_many_packed(
            self,
            points: 'np.ndarray',
            directions: 'np.ndarray',
            bias: 'Optional[float]' = None,
            *,
            facing_normals: 'Optional[np.ndarray]' = None,
            cancel_check: 'Optional[Callable[[], bool]]' = None,
            progress_callback: 'Optional[Callable[[int, int], None]]' = None,
            ) -> 'PackedVisibility':
        """Trace and pack point/look visibility in bounded direction batches.

        The result is point-major and occupies one bit per pair, constructed
        one direction at a time. When ``facing_normals`` is supplied, pairs
        with ``direction.normal <= 0`` remain False without entering the
        BVH/ray-triangle path. This matches the point-scatterer's strict
        front-face gate and avoids tracing work whose field contribution is
        known to be zero.
        """

        raw_points = np.asarray(points, dtype=float)
        pts = np.atleast_2d(raw_points)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (n, 3) or (3,).")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points contain NaN or infinite coordinates.")

        raw_directions = np.asarray(directions, dtype=float)
        dirs = np.atleast_2d(raw_directions)
        if dirs.ndim != 2 or dirs.shape[1] != 3:
            raise ValueError("directions must have shape (n, 3) or (3,).")
        if not np.all(np.isfinite(dirs)):
            raise ValueError("directions contain NaN or infinite coordinates.")
        direction_norms = np.linalg.norm(dirs, axis=1)
        if np.any(direction_norms <= 1e-15):
            raise ValueError("directions must contain nonzero 3-vectors.")
        dirs = dirs / direction_norms[:, None]
        query_bias = self.bias if bias is None else float(bias)
        if not np.isfinite(query_bias) or query_bias < 0.0:
            raise ValueError("occlusion bias must be finite and non-negative.")

        normals = None
        if facing_normals is not None:
            normals = np.atleast_2d(
                np.asarray(facing_normals, dtype=float)
            )
            if normals.shape != pts.shape:
                raise ValueError(
                    "facing_normals must have shape (n_points, 3)."
                )
            if not np.all(np.isfinite(normals)):
                raise ValueError(
                    "facing_normals contain NaN or infinite coordinates."
                )
            normal_norms = np.linalg.norm(normals, axis=1)
            if np.any(normal_norms <= 1e-15):
                raise ValueError(
                    "facing_normals must contain nonzero 3-vectors."
                )
            normals = normals / normal_norms[:, None]

        packed = np.zeros(
            (len(pts), (len(dirs) + 7) // 8), dtype=np.uint8
        )
        all_indices = np.arange(len(pts), dtype=np.intp)
        for direction_index, direction in enumerate(dirs):
            if self._cancelled(cancel_check):
                raise InterruptedError("Body-shadow query cancelled.")
            active = (
                all_indices
                if normals is None
                else np.flatnonzero((normals @ direction) > 0.0)
            )
            if active.size:
                visible = self.visible(
                    pts[active],
                    direction,
                    bias=query_bias,
                    cancel_check=cancel_check,
                )
                visible_points = active[np.asarray(visible, dtype=bool)]
                if visible_points.size:
                    packed[visible_points, direction_index // 8] |= np.uint8(
                        1 << (direction_index % 8)
                    )
            if progress_callback is not None:
                progress_callback(direction_index + 1, len(dirs))
        if self._cancelled(cancel_check):
            raise InterruptedError("Body-shadow query cancelled.")
        return PackedVisibility(
            packed,
            n_points=len(pts),
            n_directions=len(dirs),
        )
