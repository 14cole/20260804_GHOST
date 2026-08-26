#!/usr/bin/env python3
"""Deterministic topology QA for Assembly placement/shadow meshes.

The placement solver needs trustworthy local face normals, while geometric
shadowing needs a reasonably closed body.  STL itself carries neither topology
nor reliable normal records, so this module reconstructs edge incidence from
the actual triangle vertices and reports the conditions that matter to those
two uses.  It does not mutate or repair user geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MeshTopologyReport:
    schema: str
    triangle_count: int
    welded_vertex_count: int
    unique_edge_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    inconsistent_winding_edge_count: int
    duplicate_triangle_count: int
    weld_tolerance_m: float
    watertight: bool
    edge_manifold: bool
    consistently_wound: bool
    connected_component_count: int
    closed_component_count: int
    outward_closed_component_count: int
    inward_closed_component_count: int
    indeterminate_component_count: int
    signed_component_volumes_m3: tuple[float, ...]
    global_orientation: str

    def as_dict(self) -> dict:
        return asdict(self)

    def messages(
        self, *, shadow_requested: bool = False, normals_flipped: bool = False
    ) -> tuple[str, ...]:
        messages = []
        if self.duplicate_triangle_count:
            messages.append(
                f"surface mesh contains {self.duplicate_triangle_count} duplicate "
                "triangle(s); remove duplicate faces before placement"
            )
        if self.nonmanifold_edge_count:
            messages.append(
                f"surface mesh contains {self.nonmanifold_edge_count} non-manifold "
                "edge(s); local skin ownership and normals are ambiguous"
            )
        if self.inconsistent_winding_edge_count:
            messages.append(
                f"surface mesh contains {self.inconsistent_winding_edge_count} "
                "same-direction shared edge(s); repair mixed face winding (the "
                "global Flip normals option cannot repair mixed winding)"
            )
        if self.boundary_edge_count:
            use = (
                "body-shadow rays can leak through these openings"
                if shadow_requested
                else "this is acceptable only when the file intentionally represents "
                "an open placement patch"
            )
            messages.append(
                f"surface mesh has {self.boundary_edge_count} open boundary edge(s); "
                + use
            )
        if (
            self.inward_closed_component_count
            and self.outward_closed_component_count
        ):
            messages.append(
                "closed surface components have mixed global orientation; a "
                "single Flip normals option cannot make every component outward"
            )
        elif self.inward_closed_component_count and not normals_flipped:
            messages.append(
                "every closed surface component is consistently wound inward; "
                "review and enable the global Flip normals option"
            )
        elif self.outward_closed_component_count and normals_flipped:
            messages.append(
                "closed surface components are already wound outward, but the "
                "global Flip normals option makes their effective normals inward"
            )
        return tuple(messages)


def _validated_triangles(triangles: np.ndarray) -> np.ndarray:
    values = np.asarray(triangles, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) == 0:
        raise ValueError("triangles must have shape (n, 3, 3) with n > 0.")
    if not np.all(np.isfinite(values)):
        raise ValueError("surface triangles contain NaN or infinite coordinates.")
    return values


def audit_triangle_topology(
    triangles: np.ndarray, *, weld_tolerance_m: Optional[float] = None
) -> MeshTopologyReport:
    """Reconstruct mesh edge incidence and return production-facing QA.

    Vertices closer than ``weld_tolerance_m`` are treated as one topological
    vertex.  The conservative default is one part per billion of the largest
    mesh span (with a 1 pm floor), enough to join repeated float STL vertices
    without merging ordinary modeled gaps.
    """
    tris = _validated_triangles(triangles)
    flat = tris.reshape(-1, 3)
    lo = flat.min(axis=0)
    span = flat.max(axis=0) - lo
    extent = max(float(np.max(span)), 1.0e-12)
    tolerance = (
        max(1.0e-12, 1.0e-9 * extent)
        if weld_tolerance_m is None
        else float(weld_tolerance_m)
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("mesh weld tolerance must be finite and positive.")

    # Collapse exact duplicates first (the normal STL case), then union every
    # pair within the actual Euclidean tolerance. Coordinate rounding alone is
    # not a distance test: near vertices on opposite cell boundaries can be
    # missed, while diagonal points farther than the tolerance can be merged.
    exact_vertices, exact_inverse = np.unique(
        flat, axis=0, return_inverse=True
    )
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # scipy is a declared GHOST dependency
        raise RuntimeError(
            "scipy is required for exact surface-mesh topology welding."
        ) from exc
    near_pairs = cKDTree(exact_vertices).query_pairs(
        tolerance, output_type="ndarray"
    )
    parent = np.arange(len(exact_vertices), dtype=np.intp)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in np.asarray(near_pairs, dtype=np.intp).reshape(-1, 2):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            # Deterministic representative independent of pair traversal.
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
    for index in range(len(parent)):
        parent[index] = find(index)
    _vertices, welded_inverse = np.unique(parent, return_inverse=True)
    inverse = welded_inverse[exact_inverse]
    vertex_ids = inverse.reshape(-1, 3).astype(np.uint64, copy=False)
    if np.any(
        (vertex_ids[:, 0] == vertex_ids[:, 1])
        | (vertex_ids[:, 1] == vertex_ids[:, 2])
        | (vertex_ids[:, 2] == vertex_ids[:, 0])
    ):
        raise ValueError(
            "mesh weld tolerance collapses at least one triangle; reduce the "
            "tolerance or repair near-degenerate faces."
        )

    canonical_triangles = np.sort(vertex_ids, axis=1)
    duplicate_count = int(
        len(canonical_triangles)
        - len(np.unique(canonical_triangles, axis=0))
    )

    starts = vertex_ids[:, [0, 1, 2]].reshape(-1)
    stops = vertex_ids[:, [1, 2, 0]].reshape(-1)
    edge_lo = np.minimum(starts, stops)
    edge_hi = np.maximum(starts, stops)
    # Vertex count may exceed 2**32 in theory; a structured pair keeps this
    # exact without relying on a fixed-width packed key.
    keys = np.empty(len(edge_lo), dtype=[("lo", "<u8"), ("hi", "<u8")])
    keys["lo"] = edge_lo
    keys["hi"] = edge_hi
    orientation = np.where(starts == edge_lo, 1, -1).astype(np.int8)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_orientation = orientation[order]
    group_start = np.r_[
        0,
        1 + np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]),
    ]
    counts = np.diff(np.r_[group_start, len(sorted_keys)])
    orientation_sum = np.add.reduceat(
        sorted_orientation.astype(np.int64), group_start
    )

    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    inconsistent = int(np.count_nonzero((counts == 2) & (orientation_sum != 0)))
    manifold = nonmanifold == 0
    consistent = inconsistent == 0

    # Connected face components and their signed enclosed volumes distinguish
    # a consistently *inward* closed vehicle from a correctly outward one.
    # Relative edge winding alone cannot see that global reversal.
    face_parent = np.arange(len(tris), dtype=np.intp)

    def face_find(index: int) -> int:
        while face_parent[index] != index:
            face_parent[index] = face_parent[face_parent[index]]
            index = int(face_parent[index])
        return index

    def face_union(left: int, right: int) -> None:
        left_root = face_find(left)
        right_root = face_find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        face_parent[right_root] = left_root

    face_open = np.zeros(len(tris), dtype=bool)
    face_invalid = np.zeros(len(tris), dtype=bool)
    for edge_index, start_index in enumerate(group_start):
        stop_index = (
            int(group_start[edge_index + 1])
            if edge_index + 1 < len(group_start) else len(order)
        )
        faces = np.unique(order[start_index:stop_index] // 3)
        for face in faces[1:]:
            face_union(int(faces[0]), int(face))
        if counts[edge_index] == 1:
            face_open[faces] = True
        if counts[edge_index] != 2 or orientation_sum[edge_index] != 0:
            face_invalid[faces] = True
    roots = np.asarray([face_find(index) for index in range(len(tris))])
    component_roots = np.unique(roots)
    signed_face_volume = np.einsum(
        "ij,ij->i",
        tris[:, 0],
        np.cross(tris[:, 1], tris[:, 2]),
    ) / 6.0
    volume_tolerance = 1.0e-12 * max(extent ** 3, 1.0e-30)
    signed_component_volumes = []
    outward_components = 0
    inward_components = 0
    closed_components = 0
    indeterminate_components = 0
    for root in component_roots:
        members = roots == root
        volume = float(np.sum(signed_face_volume[members]))
        signed_component_volumes.append(volume)
        closed_valid = not np.any(face_open[members] | face_invalid[members])
        if not closed_valid or abs(volume) <= volume_tolerance:
            indeterminate_components += 1
            continue
        closed_components += 1
        if volume > 0.0:
            outward_components += 1
        else:
            inward_components += 1
    if closed_components and not indeterminate_components:
        if outward_components == closed_components:
            global_orientation = "outward"
        elif inward_components == closed_components:
            global_orientation = "inward"
        else:
            global_orientation = "mixed"
    elif outward_components and inward_components:
        global_orientation = "mixed"
    else:
        global_orientation = "open_or_indeterminate"
    return MeshTopologyReport(
        schema="ghost.assembly-mesh-topology.v1",
        triangle_count=int(len(tris)),
        welded_vertex_count=int(len(_vertices)),
        unique_edge_count=int(len(counts)),
        boundary_edge_count=boundary,
        nonmanifold_edge_count=nonmanifold,
        inconsistent_winding_edge_count=inconsistent,
        duplicate_triangle_count=duplicate_count,
        weld_tolerance_m=float(tolerance),
        watertight=bool(boundary == 0 and manifold),
        edge_manifold=bool(manifold),
        consistently_wound=bool(consistent),
        connected_component_count=int(len(component_roots)),
        closed_component_count=int(closed_components),
        outward_closed_component_count=int(outward_components),
        inward_closed_component_count=int(inward_components),
        indeterminate_component_count=int(indeterminate_components),
        signed_component_volumes_m3=tuple(
            float(value) for value in signed_component_volumes
        ),
        global_orientation=global_orientation,
    )


__all__ = ["MeshTopologyReport", "audit_triangle_topology"]
