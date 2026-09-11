"""Source snapshots and placement preparation for feature Assembly."""
from ghost_backend.execution.runtime import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from ghost_backend.geometry.surface import TriangleSurface

@dataclass
class AssemblySources:
    active_features: 'bool'
    base: 'Path'
    base_sha256: 'str | None'
    coordinate_scale: 'float'
    features_only_output: 'Path'
    line_coordinates_path: 'Path | None'
    output: 'Path'
    point_coordinates_path: 'Path | None'
    prepared_features_only_output_absent: 'bool'
    prepared_features_only_output_sha256: 'str | None'
    prepared_input_sources: 'dict[str, dict[str, str]]'
    prepared_output_absent: 'bool'
    prepared_output_sha256: 'str | None'
    prepared_source_sha256: 'dict[str, str]'
    surface_path: 'Path | None'
    surface_scale: 'float | None'


@dataclass
class PreparedPlacements:
    line_preview_endpoint_normals: 'dict[str, Any]'
    line_preview_paths: 'dict[str, Any]'
    line_records: 'list[dict[str, Any]]'
    lines: 'list[dict[str, Any]]'
    mesh_topology_report: 'dict[str, Any] | None'
    point_preview_ids: 'dict[str, Any]'
    point_preview_lists: 'dict[str, Any]'
    point_preview_normals: 'dict[str, Any]'
    point_preview_roll_references: 'dict[str, Any]'
    point_records: 'list[dict[str, Any]]'
    points: 'list[dict[str, Any]]'
    skin_limit: 'float'
    surface: 'TriangleSurface | None'
    surface_geometry_contract: 'dict[str, Any]'
    surface_triangles_cad_m: 'np.ndarray | None'
    wavelength: 'float'


def capture_assembly_sources(request, *, cancel_check=None, progress_callback=None) -> 'AssemblySources':
    from ghost_backend.assembly.workflow import (
        FeatureAssemblyRequest,
        Optional,
        Path,
        PathValue,
        _canonical_grim_output_path,
        _reject_output_aliases,
        _required_unit_scale,
        feature_only_output_path,
        resolve_path,
        sha256_file,
    )
    if not isinstance(request, FeatureAssemblyRequest):
        raise TypeError("request must be a FeatureAssemblyRequest.")
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")
    if progress_callback is not None:
        progress_callback(0, 100, "Checking Assembly inputs")
    base = resolve_path(request.base_grim, base_dir=request.base_dir)
    output = _canonical_grim_output_path(
        request.output_grim, base_dir=request.base_dir
    )
    features_only_output = Path(feature_only_output_path(str(output)))
    _reject_output_aliases(request, base=base, output=output)
    active_features = bool(
        (request.point_locations_csv is not None and request.enabled_point_placement_ids != ())
        or (request.line_locations_csv is not None and request.enabled_line_ids != ())
    )
    coordinate_scale = _required_unit_scale(
        request.coordinate_units,
        label="coordinate_units",
        used_for="a point or line placement CSV",
    ) if request.point_locations_csv is not None or request.line_locations_csv is not None else 1.0
    surface_scale: 'Optional[float]' = None
    if request.surface_mesh is not None:
        surface_scale = _required_unit_scale(
            request.surface_units,
            label="surface_units",
            used_for="surface_mesh",
        )
    if not base.is_file():
        raise FileNotFoundError(f"Base monostatic GRIM not found: {base}")
    if output.exists() and not output.is_file():
        raise ValueError(f"Assembly output exists but is not a file: {output}")
    if features_only_output.exists() and not features_only_output.is_file():
        raise ValueError(
            "Feature-only Assembly output exists but is not a file: "
            f"{features_only_output}"
        )
    prepared_output_absent = not output.is_file()
    prepared_output_sha256 = (
        None if prepared_output_absent else sha256_file(str(output))
    )
    prepared_features_only_output_absent = not features_only_output.is_file()
    prepared_features_only_output_sha256 = (
        None
        if prepared_features_only_output_absent
        else sha256_file(str(features_only_output))
    )
    base_sha256 = sha256_file(str(base))
    prepared_source_sha256 = {str(base): base_sha256}
    prepared_input_sources: 'dict[str, dict[str, str]]' = {
        "base_grim": {"path": str(base), "sha256": base_sha256}
    }
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")
    if progress_callback is not None:
        progress_callback(12, 100, "Reading clean-body response")

    def snapshot_input_source(
        role: 'str',
        value: 'Optional[PathValue]',
        *,
        label: 'str',
    ) -> 'Optional[Path]':
        if value is None:
            return None
        source = resolve_path(value, base_dir=request.base_dir)
        if not source.is_file():
            raise FileNotFoundError(f"{label} not found: {source}")
        digest = sha256_file(str(source))
        previous = prepared_source_sha256.get(str(source))
        if previous is not None and previous != digest:
            raise RuntimeError(
                f"Feature-assembly source changed while input files were "
                f"being snapshotted: {source}. Revalidate the assembly."
            )
        prepared_source_sha256[str(source)] = digest
        prepared_input_sources[role] = {
            "path": str(source),
            "sha256": digest,
        }
        return source


    surface_path = snapshot_input_source(
        "surface_mesh", request.surface_mesh, label="Surface mesh"
    )
    line_coordinates_path = snapshot_input_source(
        "line_locations_csv",
        request.line_locations_csv,
        label="Line-placement CSV",
    )
    point_coordinates_path = snapshot_input_source(
        "point_locations_csv",
        request.point_locations_csv,
        label="Point-placement CSV",
    )
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")

    return AssemblySources(
        active_features=active_features,
        base=base,
        base_sha256=base_sha256,
        coordinate_scale=coordinate_scale,
        features_only_output=features_only_output,
        line_coordinates_path=line_coordinates_path,
        output=output,
        point_coordinates_path=point_coordinates_path,
        prepared_features_only_output_absent=prepared_features_only_output_absent,
        prepared_features_only_output_sha256=prepared_features_only_output_sha256,
        prepared_input_sources=prepared_input_sources,
        prepared_output_absent=prepared_output_absent,
        prepared_output_sha256=prepared_output_sha256,
        prepared_source_sha256=prepared_source_sha256,
        surface_path=surface_path,
        surface_scale=surface_scale,
    )


def prepare_assembly_placements(request, sources: 'AssemblySources', *, embedded_grid, grid,
                                profile, pre_validation_warnings, surface_geometry_contract,
                                cancel_check=None, progress_callback=None) -> 'PreparedPlacements':
    from ghost_backend.assembly.workflow import (
        CAD2AXIS,
        Optional,
        SURFACE_BINDING_SCHEMA,
        TriangleSurface,
        _validate_bor_surface_agreement,
        bor_shadow_triangles,
        compute_skin_limit,
        np,
        prepare_line_placements,
        prepare_point_placements,
        read_surface_mesh,
        to_axis_frame,
        validate_normal_tolerance,
    )
    skin_limit, wavelength = compute_skin_limit(grid["frequencies_ghz"], skin_tol_m=request.skin_tol_m, skin_phase_tol_deg=request.skin_phase_tol_deg)
    normal_tolerance = validate_normal_tolerance(request.normal_tol_deg)
    auto_shadow_report = None
    surface: 'Optional[TriangleSurface]' = None
    surface_triangles_cad_m: 'Optional[np.ndarray]' = None
    mesh_topology_report = None
    if sources.surface_path is not None:
        assert sources.surface_scale is not None
        surface_triangles_cad_m = (
            np.asarray(read_surface_mesh(str(sources.surface_path)), dtype=float)
            * sources.surface_scale
        )
        surface_triangles_cad_m.setflags(write=False)
        triangles = to_axis_frame(surface_triangles_cad_m)
        surface = TriangleSurface(
            triangles,
            flip_normals=bool(request.flip_surface_normals),
        )


        mesh_topology_report = surface.topology_report
    elif embedded_grid is not None and sources.active_features and request.shadow:
        triangles, auto_shadow_report = bor_shadow_triangles(profile, max_sag_m=skin_limit/4, normal_tolerance_deg=max(normal_tolerance/2, 1e-6))
        surface = TriangleSurface(triangles)
        surface_triangles_cad_m = triangles @ CAD2AXIS
        surface_triangles_cad_m.setflags(write=False)
        mesh_topology_report = surface.topology_report
    elif embedded_grid is None and sources.active_features:
        raise ValueError(
            "A non-BoR base requires surface_mesh=.facet or .stl for skin "
            "validation and outward normals."
        )
    if request.shadow and sources.active_features and surface is None:
        raise ValueError("shadow=True requires surface_mesh.")
    if surface is not None and sources.active_features and not request.shadow:
        pre_validation_warnings.append(
            "Geometric body shadowing is OFF while a body mesh is selected. "
            "Hidden point and line features are not occlusion-tested and can "
            "contribute at full modeled amplitude whenever they are front-face "
            "illuminated. Enable body shadowing for vehicle placement work, or "
            "review and accept the existing one-time release-warning waiver if "
            "this no-shadow trade study is intentional."
        )
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Feature placement validation cancelled.")
    if progress_callback is not None:
        progress_callback(35, 100, "Checking body surface and topology")

    if embedded_grid is not None:
        if surface is None:
            surface_geometry_contract = {
                "schema": SURFACE_BINDING_SCHEMA,
                "status": "embedded_bor_profile_is_authoritative_surface",
            }
        else:
            surface_geometry_contract = _validate_bor_surface_agreement(
                profile,
                surface,
                skin_limit_m=skin_limit,
                shadow_requested=bool(request.shadow),
                cancel_check=cancel_check,
            )
            surface_geometry_contract["surface_mesh"] = str(sources.surface_path)
            if auto_shadow_report is not None:
                surface_geometry_contract["surface_mesh"] = None
                surface_geometry_contract["generated_shadow_surface"] = auto_shadow_report
    point_preview_lists: 'dict[str, list[np.ndarray]]' = {}
    point_preview_normals: 'dict[str, list[np.ndarray]]' = {}
    point_preview_roll_references: 'dict[str, list[np.ndarray]]' = {}
    line_preview_paths: 'dict[str, dict[str, np.ndarray]]' = {}
    line_preview_endpoint_normals: 'dict[str, dict[str, np.ndarray]]' = {}
    lines, line_records = prepare_line_placements(
        profile,
        surface,
        coordinate_scale=sources.coordinate_scale,
        skin_limit_m=skin_limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=normal_tolerance,
        locations_csv=sources.line_coordinates_path,
        datasets=request.line_datasets,
        enabled_line_ids=request.enabled_line_ids,
        base_dir=request.base_dir,
        preview_paths_cad_m=line_preview_paths,
        preview_endpoint_normals_cad=line_preview_endpoint_normals,
        prepare_shadow_origins=bool(request.shadow),
        cancel_check=cancel_check,
    )
    if progress_callback is not None:
        progress_callback(55, 100, "Checking line paths")
    points, point_records = prepare_point_placements(
        profile,
        surface,
        coordinate_scale=sources.coordinate_scale,
        skin_limit_m=skin_limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=normal_tolerance,
        locations_csv=sources.point_coordinates_path,
        datasets=request.point_datasets,
        enabled_point_placement_ids=request.enabled_point_placement_ids,
        base_dir=request.base_dir,
        preview_locations_cad_m=point_preview_lists,
        preview_normals_cad=point_preview_normals,
        preview_roll_references_cad=point_preview_roll_references,
        prepare_shadow_origins=bool(request.shadow),
        cancel_check=cancel_check,
    )
    if progress_callback is not None:
        progress_callback(72, 100, "Checking point placements")

    point_preview_ids: 'dict[str, list[str]]' = {}
    for record in point_records:
        point_preview_ids.setdefault(str(record["dataset_id"]), []).append(
            str(record["placement_id"])
        )

    return PreparedPlacements(
        line_preview_endpoint_normals=line_preview_endpoint_normals,
        line_preview_paths=line_preview_paths,
        line_records=line_records,
        lines=lines,
        mesh_topology_report=mesh_topology_report,
        point_preview_ids=point_preview_ids,
        point_preview_lists=point_preview_lists,
        point_preview_normals=point_preview_normals,
        point_preview_roll_references=point_preview_roll_references,
        point_records=point_records,
        points=points,
        skin_limit=skin_limit,
        surface=surface,
        surface_geometry_contract=surface_geometry_contract,
        surface_triangles_cad_m=surface_triangles_cad_m,
        wavelength=wavelength,
    )
