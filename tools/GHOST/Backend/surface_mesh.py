#!/usr/bin/env python3
"""Triangulated platform surfaces for feature placement and shadowing.

The supported ``.facet`` representation is the simple indexed ASCII form used
by CUBIT/Coreform::

    n_vertices n_facets
    vertex_id x y z                 # repeated n_vertices times
    facet_id vertex_id ...          # 3 (triangle) or 4 (quad) vertex ids

Quads are split along the 1-3 diagonal without changing their winding.  STL is
also accepted through :func:`read_surface_mesh`.  Coordinates are returned in
the file's native units; callers own the unit and frame conversion.
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np


def _mesh_extent(triangles):
    """Largest Cartesian span, compatible with old NumPy releases."""
    vertices = np.asarray(triangles, dtype=float).reshape(-1, 3)
    spans = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    return max(float(np.max(spans)), 1.0)


def _data_lines(path: str):
    with open(path, "r", encoding="utf-8-sig") as stream:
        for number, raw in enumerate(stream, 1):
            text = raw.split("#", 1)[0].strip()
            if text:
                yield number, text.split()


def read_facet(path: str) -> np.ndarray:
    """Read an indexed ASCII CUBIT/Coreform ``.facet`` triangle/quad mesh."""
    rows = iter(_data_lines(path))
    try:
        first_number, first = next(rows)
    except StopIteration:
        raise ValueError(f"{path}: empty .facet file.")
    if len(first) != 2:
        raise ValueError(
            f"{path}:{first_number}: expected 'n_vertices n_facets'. This "
            "reader supports indexed ASCII CUBIT/Coreform .facet files."
        )
    try:
        n_vertices, n_facets = (int(value) for value in first)
    except ValueError as exc:
        raise ValueError(
            f"{path}:{first_number}: vertex/facet counts must be integers."
        ) from exc
    if n_vertices < 3 or n_facets < 1:
        raise ValueError(f"{path}: .facet counts must be at least 3 vertices/1 facet.")

    def required_row(section, index):
        try:
            return next(rows)
        except StopIteration as exc:
            raise ValueError(
                f"{path}: ended before declared {section} {index + 1}."
            ) from exc

    vertices = {}
    for index in range(n_vertices):
        number, tokens = required_row("vertex", index)
        if len(tokens) != 4:
            raise ValueError(f"{path}:{number}: expected 'vertex_id x y z'.")
        try:
            identifier = int(tokens[0])
            point = np.asarray([float(value) for value in tokens[1:]], dtype=float)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid vertex row.") from exc
        if identifier in vertices:
            raise ValueError(f"{path}:{number}: duplicate vertex id {identifier}.")
        if not np.all(np.isfinite(point)):
            raise ValueError(f"{path}:{number}: vertex contains NaN/infinity.")
        vertices[identifier] = point

    triangles = []
    facet_ids = set()
    for index in range(n_facets):
        number, tokens = required_row("facet", index)
        if len(tokens) not in (4, 5):
            raise ValueError(
                f"{path}:{number}: facets require an id plus 3 or 4 vertex ids."
            )
        try:
            facet_id = int(tokens[0])
            ids = [int(value) for value in tokens[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid facet connectivity.") from exc
        if facet_id in facet_ids:
            raise ValueError(f"{path}:{number}: duplicate facet id {facet_id}.")
        facet_ids.add(facet_id)
        try:
            polygon = [vertices[value] for value in ids]
        except KeyError as exc:
            raise ValueError(
                f"{path}:{number}: facet references unknown vertex id {exc.args[0]}."
            ) from exc
        triangles.append([polygon[0], polygon[1], polygon[2]])
        if len(polygon) == 4:
            triangles.append([polygon[0], polygon[2], polygon[3]])
    try:
        extra_number, _extra = next(rows)
    except StopIteration:
        pass
    else:
        raise ValueError(
            f"{path}:{extra_number}: data follows the declared facet count."
        )

    result = np.asarray(triangles, dtype=float)
    cross = np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0])
    scale = _mesh_extent(result)
    if np.any(np.linalg.norm(cross, axis=1) <= 1e-14 * scale * scale):
        raise ValueError(f"{path}: mesh contains a degenerate triangle.")
    return result


def read_surface_mesh(path: str) -> np.ndarray:
    """Read a supported STL or indexed ASCII ``.facet`` surface."""
    suffix = Path(path).suffix.lower()
    if suffix == ".facet":
        return read_facet(path)
    if suffix == ".stl":
        from occluder import read_stl
        return read_stl(path)
    raise ValueError(f"{path}: supported surface extensions are .facet and .stl.")


def _closest_on_segments(point, starts, ends):
    edges = ends - starts
    lengths_squared = np.einsum("ij,ij->i", edges, edges)
    parameter = np.divide(
        np.einsum("ij,ij->i", point - starts, edges),
        lengths_squared,
        out=np.zeros_like(lengths_squared),
        where=lengths_squared > 0.0,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    return starts + parameter[:, None] * edges


class TriangleSurface:
    """Exact closest-point/face-normal queries on a triangle surface."""

    def __init__(self, triangles, *, flip_normals=False, triangle_chunk=32768):
        self.triangles = np.asarray(triangles, dtype=float)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape (n, 3, 3).")
        if len(self.triangles) == 0 or not np.all(np.isfinite(self.triangles)):
            raise ValueError("triangle surface must be finite and nonempty.")
        raw = np.cross(
            self.triangles[:, 1] - self.triangles[:, 0],
            self.triangles[:, 2] - self.triangles[:, 0],
        )
        magnitude = np.linalg.norm(raw, axis=1)
        extent = _mesh_extent(self.triangles)
        if np.any(magnitude <= 1e-14 * extent * extent):
            raise ValueError("triangle surface contains a degenerate triangle.")
        sign = -1.0 if bool(flip_normals) else 1.0
        self.face_normals = sign * raw / magnitude[:, None]
        self.triangle_chunk = max(1, int(triangle_chunk))
        self.centroids = np.mean(self.triangles, axis=1)
        self.centroid_radius = np.max(
            np.linalg.norm(
                self.triangles - self.centroids[:, None, :], axis=2
            ),
            axis=1,
        )
        self.max_centroid_radius = float(np.max(self.centroid_radius))
        try:
            from scipy.spatial import cKDTree
            self._centroid_tree = cKDTree(self.centroids)
        except ImportError:  # exact brute-force fallback for minimal installs
            self._centroid_tree = None

    @staticmethod
    def _closest_on_triangles(point, tris):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        raw = np.cross(b - a, c - a)
        raw_squared = np.einsum("ij,ij->i", raw, raw)
        projected = point - (
            np.einsum("ij,ij->i", point - a, raw) / raw_squared
        )[:, None] * raw

        v0, v1, v2 = b - a, c - a, projected - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denominator = d00 * d11 - d01 * d01
        bary_b = (d11 * d20 - d01 * d21) / denominator
        bary_c = (d00 * d21 - d01 * d20) / denominator
        inside = (
            (bary_b >= -1e-12) & (bary_c >= -1e-12)
            & (bary_b + bary_c <= 1.0 + 1e-12)
        )

        candidates = [
            np.where(inside[:, None], projected, np.inf),
            _closest_on_segments(point, a, b),
            _closest_on_segments(point, b, c),
            _closest_on_segments(point, c, a),
        ]
        candidate_distance = np.stack([
            np.einsum("ij,ij->i", value - point, value - point)
            for value in candidates
        ])
        choice = np.argmin(candidate_distance, axis=0)
        distance = candidate_distance[choice, np.arange(len(tris))]
        closest = np.stack(candidates)[choice, np.arange(len(tris))]
        return distance, closest

    def nearest(self, points) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return distance, closest point, wound face normal, and facet index."""
        query = np.atleast_2d(np.asarray(points, dtype=float))
        if query.shape[1] != 3 or not np.all(np.isfinite(query)):
            raise ValueError("surface query points must have shape (n,3) and be finite.")
        best_distance_squared = np.full(len(query), np.inf)
        best_point = np.empty_like(query)
        best_index = np.full(len(query), -1, dtype=int)

        for point_index, point in enumerate(query):
            if self._centroid_tree is None:
                possible = np.arange(len(self.triangles), dtype=int)
            else:
                _centroid_distance, seed_index = self._centroid_tree.query(point)
                seed_distance, _seed_point = self._closest_on_triangles(
                    point, self.triangles[[int(seed_index)]]
                )
                # Exact pruning: every point on a triangle lies within its
                # centroid radius. A centroid farther than the current upper
                # bound plus the largest such radius cannot own the nearest
                # surface point.
                radius = math.sqrt(float(seed_distance[0])) + self.max_centroid_radius
                radius += 32.0 * np.finfo(float).eps * max(
                    1.0, radius, float(np.linalg.norm(point))
                )
                possible = np.asarray(
                    self._centroid_tree.query_ball_point(point, radius), dtype=int
                )
            for start in range(0, len(possible), self.triangle_chunk):
                indices = possible[start:start + self.triangle_chunk]
                triangle_distance, closest = self._closest_on_triangles(
                    point, self.triangles[indices]
                )
                local_triangle = int(np.argmin(triangle_distance))
                distance = float(triangle_distance[local_triangle])
                if distance < best_distance_squared[point_index]:
                    best_distance_squared[point_index] = distance
                    best_index[point_index] = int(indices[local_triangle])
                    best_point[point_index] = closest[local_triangle]

        return (
            np.sqrt(best_distance_squared),
            best_point,
            self.face_normals[best_index],
            best_index,
        )

    def distance(self, points):
        return self.nearest(points)[0]

    def normal(self, points):
        return self.nearest(points)[2]
