"""Physical constants, unit labels, metadata fields, and array block limits."""
from __future__ import annotations


C0 = 299_792_458.0


SENTRI_FAR_FIELD_METADATA = {
    "phase_reference": (
        "origin=(0,0,0) vehicle frame, convention=exp(+jwt), "
        "radar earth-frame V/H monostatic amplitude"
    ),
    "amplitude_convention": "F physical far-field amplitude; sigma_3d=4*pi*|F|^2",
    "complex_field_domain": "coherent_radar_frame_far_field_amplitude",
    "time_convention": "exp(+jwt)",
    "sentri_far_field_reference": (
        "SENTRi export: exp(+jwt); global coordinate origin; "
        "outgoing exp(-jkr)/r removed; incident propagation opposite look; "
        "unchanged theta/phi polarization vectors"
    ),
}


GRIM_GC_CONVENTION = "grim_gc_v1"


LEGACY_PTM_GC_CONVENTION = "legacy_ptm_unspecified"


_PTM_GRIM_GC_MARKER = "GRIM_GC_V1"


WEDGE_TURNTABLE_CONVENTION = "grim_vertical_turntable_body_y_pitch_v1"


CONIC_VH_BASIS_CONVENTION = "grim_conic_spherical_vh_v1"


_FREQUENCY_UNITS = {
    "hz": "Hz",
    "khz": "kHz",
    "mhz": "MHz",
    "ghz": "GHz",
}


_ANGLE_UNITS = {
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
}


_ACQUISITION_METADATA_FAMILIES = (
    (
        "amplitude_convention",
        "amplitude convention",
        "text",
        ("amplitude_convention",),
    ),
    (
        "complex_field_domain",
        "complex-field domain",
        "text",
        ("complex_field_domain",),
    ),
    (
        "range_phase_law",
        "two-way range-phase convention",
        "range_phase",
        ("range_phase_convention", "phase_law"),
    ),
    (
        "acquisition_geometry",
        "measurement geometry",
        "geometry",
        (
            "measurement_geometry",
            "acquisition_geometry",
            "scattering_geometry",
            "radar_geometry",
            "measurement_domain",
            "field_domain",
            "range_type",
            "wavefront_geometry",
        ),
    ),
    (
        "motion_state",
        "motion-compensation/phase-center state",
        "motion",
        (
            "motion_compensation",
            "motion_compensated",
            "phase_center_stability",
            "phase_center_motion",
            "range_alignment",
        ),
    ),
    (
        "calibration_identifier",
        "calibration ID",
        "identity",
        ("calibration_id", "calibration_identifier"),
    ),
    (
        "calibration_chain_id",
        "calibration-chain ID",
        "identity",
        ("calibration_chain_id",),
    ),
    (
        "calibration_run_id",
        "calibration-run ID",
        "identity",
        ("calibration_run_id",),
    ),
    (
        "calibration_version",
        "calibration version",
        "identity",
        ("calibration_version",),
    ),
    (
        "measurement_setup_identifier",
        "measurement-setup ID",
        "identity",
        ("measurement_setup_id", "radar_setup_id"),
    ),
    (
        "fixture_id",
        "fixture ID",
        "identity",
        ("fixture_id",),
    ),
    (
        "static_setup",
        "static-setup declaration",
        "setup_state",
        ("static_setup",),
    ),
)


_SUPPORT_REFERENCE_METADATA_FIELDS = tuple(
    key
    for _family, _label, _kind, keys in _ACQUISITION_METADATA_FAMILIES
    for key in keys
)


_JOIN_MERGE_BLOCK_CELLS = 262_144


_PIO_WRITE_BLOCK_CELLS = 262_144


_RAW_COMPLEX_VALIDATION_BLOCK_CELLS = 262_144


_COHERENT_OPERATION_BLOCK_CELLS = 262_144


_DENSE_IMPORT_FALLBACK_LIMIT_BYTES = 2 * 1024**3


_ADOPT_CLEAN_ARRAYS_TOKEN = object()
