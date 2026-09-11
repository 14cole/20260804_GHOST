"""Shared Qt-free form values and loaded Assembly recipe records."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureAssemblyValues:
    """User-editable values, independent of any GUI toolkit."""

    base_grim: str = ""
    output_grim: str = ""
    # Placement and mesh formats do not reliably encode length units.  Empty
    # defaults force a deliberate choice instead of silently assuming inches.
    coordinate_units: str = ""
    surface_mesh: str = ""
    surface_units: str = ""
    flip_surface_normals: bool = False
    shadow: bool = False
    shadow_bias_m: float | None = None
    point_locations_csv: str = ""
    line_locations_csv: str = ""
    skin_tol_m: float = 1.0e-3
    skin_phase_tol_deg: float = 15.0
    normal_tol_deg: float = 15.0
    allow_legacy_base_metadata: bool = True
    require_feature_manifests: bool = False
    require_body_mesh_certification: bool = False
    host_material: str = ""
    host_stack_id: str = ""
    host_minimum_radius_m: float | None = None
    study_frequencies_ghz: tuple[float, ...] | None = None
    study_azimuths_deg: tuple[float, ...] | None = None
    study_elevations_deg: tuple[float, ...] | None = None
    base_dir: str | None = None
    point_datasets: dict[str, str] = field(default_factory=dict)
    line_datasets: dict[str, str] = field(default_factory=dict)
    # Spatial feature-definition state. These stable CSV IDs are independent
    # from both preview visibility and whole-response dataset arithmetic.
    excluded_point_placement_ids: set[str] = field(default_factory=set)
    excluded_line_ids: set[str] = field(default_factory=set)

@dataclass(frozen=True)
class LoadedFeatureAssemblyRecipe:
    """One validated recipe plus non-fatal source-integrity observations."""

    path: Path
    name: str
    variant: str
    values: FeatureAssemblyValues
    source_warnings: tuple[str, ...] = ()
