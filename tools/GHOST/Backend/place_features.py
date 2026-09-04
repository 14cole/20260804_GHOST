#!/usr/bin/env python3
"""Validate, then coherently place point and line features on a platform.

Edit only the USER SETTINGS block, then run:

    python place_features.py

Metadata is advisory by default. Imported complex fields need no solver
certificate or feature-library manifest. The optional ``production`` and
``external`` profiles enable strict metadata auditing. An external monostatic
GRIM needs its matching indexed ASCII ``.facet``/STL platform surface when
placing features. Point and
line datasets use the single canonical OPN-FRD (featured-clean) differential
response.

The executable settings wrapper delegates to :mod:`feature_workflow`, which
is the Qt-free API for programmatic and GUI callers.
"""

from pathlib import Path

# =============================================================================
# USER SETTINGS
# =============================================================================

# Advisory records assumptions without blocking on missing or conflicting
# annotations. Strict metadata checks are available through "external";
# "production" additionally requires a certified local GHOST body.
VALIDATION_PROFILE = "advisory"         # advisory (default), external, or production

# Large workload reviews (and warnings in an explicitly selected strict
# profile) require the printed plan digest here. Metadata advisories in the
# default profile need no acknowledgement. Input changes invalidate a digest.
ACKNOWLEDGED_PLAN_SHA256 = None

# The Production profile verifies that this file declares the coherent
# monostatic platform field in the GHOST global-origin radar VV/HH/VH
# convention. The default profile assumes the selected body frame; an explicitly
# power-only file is always refused.
BASE_MONOSTATIC_GRIM = "rcs_runs_bor/run_x/results/body.grim"
OUTPUT_MONOSTATIC_GRIM = "rcs_runs_bor/run_x/results/body_with_features.grim"
COORDINATE_UNITS = "inches"
HOST_MATERIAL = ""               # match the library's host.material declaration
HOST_STACK_ID = ""               # required when the library declares stack_id
HOST_MINIMUM_RADIUS_M = None      # lower bound for both principal radii over every footprint
STUDY_FREQUENCIES_GHZ = None      # e.g. (1.0, 2.0); exact stored samples only
STUDY_AZIMUTHS_DEG = None
STUDY_ELEVATIONS_DEG = None

# Required for a non-BoR base. Production also requires the canonical
# <surface>.assembly.json binding made by create_feature_manifest.py after the
# exact base/surface/units/frame registration is reviewed. A surface may also
# be supplied for a BoR result when mesh-based skin checks and shadowing are
# desired. The mesh and coordinate files must use the same CAD frame/origin
# (+y nose, +x right, +z up).
SURFACE_MESH = None               # e.g. "platform.facet" or "platform.stl"
SURFACE_UNITS = "inches"
FLIP_SURFACE_NORMALS = False      # True only when mesh winding points inward

# Geometric-optics blockage. False keeps the local outward-facing test but
# does not hide an otherwise facing feature behind another part of the body.
SHADOW = True
SHADOW_BIAS_M = None              # normally leave None for mesh-scaled default

# One strict line-placement table is used for every line-expanded feature.
# The CSV header must be exactly:
#
# line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,n1x,n1y,n1z,n2x,n2y,n2z
#
# Rows for each line_id must be contiguous and numbered from 1. Segments must
# chain head-to-tail in the CAD frame (+y nose, +x right, +z up). Endpoint
# normals are explicit so curved-skin frames are interpolated without guessing.
# The tangent and outward normal fully orient the local 2-D response, so no
# separate roll vector is needed.
LINE_FEATURE_LOCATIONS_CSV = None       # e.g. "line_features.csv"
LINE_FEATURE_DATASETS = {
    # "panel_gap": "panel_gap_opn_minus_frd.grim",
    # "door_seam": "door_seam_opn_minus_frd.grim",
}

# One strict point-placement table is used for every compact feature type.
# The CSV header must be exactly:
#
# placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z
#
# Coordinates and vectors use the CAD frame (+y nose, +x right, +z up).
# ``dataset_id`` selects one entry below. The normal and roll reference are
# always explicit: this avoids file-shape inference and makes the full 3-D
# orientation of an asymmetric antenna unambiguous. For an axisymmetric
# fastener, choose any stable roll vector not parallel to its normal.
POINT_FEATURE_LOCATIONS_CSV = None       # e.g. "point_features.csv"
POINT_FEATURE_DATASETS = {
    # "fastener": "fastener_opn_minus_frd.grim",
    # "antenna": "antenna_opn_minus_frd.grim",
}

# In Production, each listed response needs a team-attested manifest created
# and checked with create_feature_manifest.py. The manifest binds the exact
# response bytes and records the reviewed subtraction, phase/frame, host, and
# applicability evidence; it does not independently certify electromagnetic
# accuracy. Incorrect declarations produce incorrect coherent phase; do not
# list standalone fields.
# Every point dataset must contain VV, HH, and reciprocal VH/HV. Missing point
# cross-polarization is not guessed to be zero. Every line dataset must contain
# the 2-D TE and TM complex responses. OPN-FRD is the only accepted delta order.
# Convert an FRD-OPN file once when building the dataset; placement never infers
# or repairs its sign.

# Coordinate-to-skin validation. The tighter of the distance and two-way
# phase limits is enforced at the highest frequency in the body file.
SKIN_TOL_M = 1.0e-3
SKIN_PHASE_TOL_DEG = 15.0
NORMAL_TOL_DEG = 15.0

# =============================================================================

from feature_sum import prepare_point_pattern  # noqa: E402
from feature_workflow import (  # noqa: E402
    LINE_CSV_COLUMNS,
    LINE_PLACEMENT_SCHEMA,
    POINT_CSV_COLUMNS,
    POINT_PLACEMENT_SCHEMA,
    FeatureAssemblyRequest,
    _sample_perimeter as _workflow_sample_perimeter,
    compute_skin_limit,
    execute_feature_assembly,
    prepare_feature_assembly,
    prepare_line_placements,
    prepare_point_placements,
    read_line_placement_csv,
    read_point_placement_csv,
    resolve_path,
    unit_vector,
    validate_normal_tolerance,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(value):
    """Compatibility path resolver: settings remain repository-relative."""

    return resolve_path(value, base_dir=PROJECT_ROOT)


def _unit(value, label):
    """Compatibility alias for the shared finite-vector validation."""

    return unit_vector(value, label)


def _skin_limit(frequencies_ghz):
    """Compatibility wrapper using the settings-block tolerances."""

    return compute_skin_limit(
        frequencies_ghz,
        skin_tol_m=SKIN_TOL_M,
        skin_phase_tol_deg=SKIN_PHASE_TOL_DEG,
    )


def _normal_tolerance():
    """Compatibility wrapper using the settings-block normal tolerance."""

    return validate_normal_tolerance(NORMAL_TOL_DEG)


def _placement_rows(path):
    """Read the strict point CSV relative to the legacy project root."""

    return read_point_placement_csv(path, base_dir=PROJECT_ROOT)


def _line_rows(path):
    """Read the strict line CSV relative to the legacy project root."""

    return read_line_placement_csv(path, base_dir=PROJECT_ROOT)


def _sample_perimeter(perimeter, samples_per_segment=33):
    return _workflow_sample_perimeter(perimeter, samples_per_segment)


def _line_placements(profile, surface, scale, limit, wavelength):
    """Legacy private API backed by the reusable request-independent core."""

    return prepare_line_placements(
        profile,
        surface,
        coordinate_scale=scale,
        skin_limit_m=limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=_normal_tolerance(),
        locations_csv=LINE_FEATURE_LOCATIONS_CSV,
        datasets=LINE_FEATURE_DATASETS,
        base_dir=PROJECT_ROOT,
    )


def _compact_points(profile, surface, scale, limit, wavelength):
    """Legacy private API backed by the reusable request-independent core."""

    return prepare_point_placements(
        profile,
        surface,
        coordinate_scale=scale,
        skin_limit_m=limit,
        wavelength_m=wavelength,
        normal_tolerance_deg=_normal_tolerance(),
        locations_csv=POINT_FEATURE_LOCATIONS_CSV,
        datasets=POINT_FEATURE_DATASETS,
        base_dir=PROJECT_ROOT,
        # Passing the module attribute preserves existing callers/tests that
        # patch place_features.prepare_point_pattern.
        pattern_loader=prepare_point_pattern,
    )


def _validation_policy():
    """Return strict backend flags for the one explicit profile setting."""

    profile = str(VALIDATION_PROFILE).strip().casefold()
    if profile == "production":
        return profile, False, True, True
    if profile == "external":
        return profile, False, True, False
    if profile in {"advisory", "legacy"}:
        return profile, True, False, False
    raise ValueError(
        "VALIDATION_PROFILE must be 'advisory', 'production', or 'external'."
    )


def main():
    try:
        (
            profile,
            allow_legacy_metadata,
            require_manifests,
            require_body_certification,
        ) = _validation_policy()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Validation profile: {profile.title()} "
        f"(strict base metadata={'yes' if not allow_legacy_metadata else 'no'}, "
        f"validated feature manifests={'yes' if require_manifests else 'no'}, "
        "certified body mesh="
        f"{'yes' if require_body_certification else 'not required'})"
    )
    request = FeatureAssemblyRequest(
        base_grim=BASE_MONOSTATIC_GRIM,
        output_grim=OUTPUT_MONOSTATIC_GRIM,
        coordinate_units=COORDINATE_UNITS,
        host_material=HOST_MATERIAL,
        host_stack_id=HOST_STACK_ID,
        host_minimum_radius_m=HOST_MINIMUM_RADIUS_M,
        study_frequencies_ghz=STUDY_FREQUENCIES_GHZ,
        study_azimuths_deg=STUDY_AZIMUTHS_DEG,
        study_elevations_deg=STUDY_ELEVATIONS_DEG,
        surface_mesh=SURFACE_MESH,
        surface_units=SURFACE_UNITS,
        flip_surface_normals=FLIP_SURFACE_NORMALS,
        shadow=SHADOW,
        shadow_bias_m=SHADOW_BIAS_M,
        line_locations_csv=LINE_FEATURE_LOCATIONS_CSV,
        line_datasets=LINE_FEATURE_DATASETS,
        point_locations_csv=POINT_FEATURE_LOCATIONS_CSV,
        point_datasets=POINT_FEATURE_DATASETS,
        skin_tol_m=SKIN_TOL_M,
        skin_phase_tol_deg=SKIN_PHASE_TOL_DEG,
        normal_tol_deg=NORMAL_TOL_DEG,
        allow_legacy_base_metadata=allow_legacy_metadata,
        require_feature_manifests=require_manifests,
        require_body_mesh_certification=require_body_certification,
        base_dir=PROJECT_ROOT,
        history="place_features.py coherent platform line/compact placement",
    )
    try:
        plan = prepare_feature_assembly(request)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if plan.validation_warnings:
        print(
            f"Validation completed with {len(plan.validation_warnings)} "
            "warning(s):"
        )
        for index, warning in enumerate(plan.validation_warnings, start=1):
            print(f"  {index}. {warning}")
    from assembly_workload import warnings_require_workload_acknowledgement
    review_required = bool(plan.validation_warnings) and (
        require_manifests or require_body_certification
        or warnings_require_workload_acknowledgement(plan.validation_warnings)
    )
    if review_required:
        plan_sha256 = str(plan.prepared_plan_sha256).strip()
        if len(plan_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in plan_sha256
        ):
            raise SystemExit(
                "Output was not published because the warning-bearing plan "
                "does not carry a valid sealed SHA-256. Validate it again with "
                "the current backend."
            )
        print(f"Prepared warning-bearing plan SHA-256: {plan_sha256}")
        if str(ACKNOWLEDGED_PLAN_SHA256 or "").strip() != plan_sha256:
            raise SystemExit(
                "Output was not published. Review the warnings above, then "
                "copy this exact digest into ACKNOWLEDGED_PLAN_SHA256 to waive "
                "them for this sealed configuration."
            )
    if plan.occluder is not None:
        print(
            "Geometric shadowing enabled: "
            f"{len(plan.surface.triangles)} triangles, "
            f"ray bias {plan.occluder.bias * 1e3:.4g} mm"
        )
    try:
        saved = execute_feature_assembly(
            plan,
            acknowledged_plan_sha256=(
                ACKNOWLEDGED_PLAN_SHA256 if review_required else None
            ),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote one combined monostatic dataset: {saved}")


if __name__ == "__main__":
    main()
