"""Machine-independent workload preflight for spatial feature assembly.

The counts in this module describe work that the current point/line assembly
actually performs.  They deliberately do not predict elapsed time: body-ray
cost can vary by orders of magnitude with mesh/ray geometry, CPU, and native
acceleration availability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# These are conservative operator-review thresholds, not performance claims.
# Crossing one makes a valid build require deliberate acknowledgement; it does
# not reject the physics or reduce geometric shadowing.
ASSEMBLY_REVIEW_RADAR_GRID_CELLS = 10_000_000
ASSEMBLY_REVIEW_POINT_FIELD_CELLS = 250_000_000
ASSEMBLY_REVIEW_LINE_FIELD_CELLS = 500_000_000
ASSEMBLY_REVIEW_SHADOW_RAYS = 5_000_000
ASSEMBLY_REVIEW_LARGE_MESH_TRIANGLES = 1_000_000
ASSEMBLY_REVIEW_LARGE_MESH_SHADOW_RAYS = 100_000

WORKLOAD_REVIEW_WARNING_PREFIX = "Assembly workload review required"


@dataclass(frozen=True)
class AssemblyWorkload:
    """Auditable operation counts for one Assembly plan."""

    available: bool
    quantities_validated: bool = False
    look_count: int = 0
    frequency_count: int = 0
    point_count: int = 0
    line_path_count: int = 0
    line_segment_count: int = 0
    line_piece_count: int = 0
    line_piece_count_exact: bool = False
    mesh_triangle_count: int = 0
    mesh_triangle_count_exact: bool = False
    shadow_enabled: bool = False
    radar_grid_cell_count: int = 0
    point_field_cell_count: int = 0
    line_field_cell_count: int = 0
    shadow_ray_upper_bound: int = 0
    packed_visibility_bytes_upper_bound: int = 0
    review_reasons: tuple[str, ...] = ()

    @property
    def review_required(self) -> bool:
        return bool(self.available and self.review_reasons)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, result)


def estimate_assembly_workload(
    *,
    look_count: int,
    frequency_count: int,
    point_count: int,
    line_path_count: int,
    line_segment_count: int,
    line_piece_count: int,
    mesh_triangle_count: int = 0,
    shadow_enabled: bool = False,
    quantities_validated: bool = False,
    line_piece_count_exact: bool = False,
    mesh_triangle_count_exact: bool = False,
) -> AssemblyWorkload:
    """Return exact/upper-bound counts without turning them into an ETA."""

    looks = _count(look_count)
    frequencies = _count(frequency_count)
    if looks <= 0 or frequencies <= 0:
        return AssemblyWorkload(available=False)
    points = _count(point_count)
    paths = _count(line_path_count)
    segments = _count(line_segment_count)
    pieces = _count(line_piece_count)
    triangles = _count(mesh_triangle_count)

    radar_cells = looks * frequencies
    point_cells = radar_cells * points
    line_cells = radar_cells * pieces
    shadow_rays = looks * (points + pieces) if shadow_enabled else 0
    # PackedVisibility stores one byte row rounded to complete look bits for
    # every point/line piece.  Back-facing pairs stay zero and are not traced,
    # but the immutable cache still reserves their bit.
    packed_bytes = (
        (points + pieces) * ((looks + 7) // 8) if shadow_enabled else 0
    )

    reasons: list[str] = []
    if radar_cells >= ASSEMBLY_REVIEW_RADAR_GRID_CELLS:
        reasons.append(
            f"{radar_cells:,} radar look-frequency cells "
            f"(review threshold {ASSEMBLY_REVIEW_RADAR_GRID_CELLS:,})"
        )
    if point_cells >= ASSEMBLY_REVIEW_POINT_FIELD_CELLS:
        reasons.append(
            f"{point_cells:,} point look-frequency evaluations "
            f"(review threshold {ASSEMBLY_REVIEW_POINT_FIELD_CELLS:,})"
        )
    if line_cells >= ASSEMBLY_REVIEW_LINE_FIELD_CELLS:
        reasons.append(
            f"{line_cells:,} line-piece look-frequency evaluations "
            f"(review threshold {ASSEMBLY_REVIEW_LINE_FIELD_CELLS:,})"
        )
    if shadow_rays >= ASSEMBLY_REVIEW_SHADOW_RAYS:
        reasons.append(
            f"up to {shadow_rays:,} body-shadow candidate rays "
            f"(review threshold {ASSEMBLY_REVIEW_SHADOW_RAYS:,})"
        )
    if (
        shadow_enabled
        and triangles >= ASSEMBLY_REVIEW_LARGE_MESH_TRIANGLES
        and shadow_rays >= ASSEMBLY_REVIEW_LARGE_MESH_SHADOW_RAYS
    ):
        reasons.append(
            f"{triangles:,}-triangle body mesh with up to "
            f"{shadow_rays:,} shadow candidates"
        )

    return AssemblyWorkload(
        available=True,
        quantities_validated=bool(quantities_validated),
        look_count=looks,
        frequency_count=frequencies,
        point_count=points,
        line_path_count=paths,
        line_segment_count=segments,
        line_piece_count=pieces,
        line_piece_count_exact=bool(line_piece_count_exact),
        mesh_triangle_count=triangles,
        mesh_triangle_count_exact=bool(mesh_triangle_count_exact),
        shadow_enabled=bool(shadow_enabled),
        radar_grid_cell_count=radar_cells,
        point_field_cell_count=point_cells,
        line_field_cell_count=line_cells,
        shadow_ray_upper_bound=shadow_rays,
        packed_visibility_bytes_upper_bound=packed_bytes,
        review_reasons=tuple(reasons),
    )


def workload_review_warning(workload: AssemblyWorkload) -> str | None:
    """Return the sealed-plan warning used by GUI and headless review gates."""

    if not workload.review_required or not workload.quantities_validated:
        return None
    detail = "; ".join(workload.review_reasons)
    shadow_note = ""
    if workload.shadow_enabled:
        shadow_note = (
            " Shadow candidates are computed once and reused across all "
            f"{workload.frequency_count:,} frequencies; front-facing culling "
            "can reduce the rays actually traced. BVH traversal cost still "
            "depends strongly on the mesh and ray geometry."
        )
    return (
        f"{WORKLOAD_REVIEW_WARNING_PREFIX} (operation counts only; no "
        f"elapsed-time estimate): {detail}.{shadow_note} Review the exact "
        "counts before execution and acknowledge this sealed plan."
    )


def warnings_require_workload_acknowledgement(warnings: Any) -> bool:
    """Whether a warning collection contains the count-based review gate."""

    return any(
        str(value).startswith(WORKLOAD_REVIEW_WARNING_PREFIX)
        for value in (warnings or ())
    )


__all__ = [
    "ASSEMBLY_REVIEW_LARGE_MESH_SHADOW_RAYS",
    "ASSEMBLY_REVIEW_LARGE_MESH_TRIANGLES",
    "ASSEMBLY_REVIEW_LINE_FIELD_CELLS",
    "ASSEMBLY_REVIEW_POINT_FIELD_CELLS",
    "ASSEMBLY_REVIEW_RADAR_GRID_CELLS",
    "ASSEMBLY_REVIEW_SHADOW_RAYS",
    "WORKLOAD_REVIEW_WARNING_PREFIX",
    "AssemblyWorkload",
    "estimate_assembly_workload",
    "warnings_require_workload_acknowledgement",
    "workload_review_warning",
]
