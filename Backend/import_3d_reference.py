#!/usr/bin/env python3
"""Import an external complex monostatic 3-D result for validation.

This stamps coherent conventions only after the user explicitly attests that
the external solve used them. It does not transform an unknown phase origin,
time sign, angular frame, or polarization basis.
"""

import json
from pathlib import Path

import numpy as np

# =============================================================================
# USER SETTINGS
# =============================================================================

SOURCE_FILE = "external_3d_featured.out"
OUTPUT_GRIM = "external_3d_featured.grim"

# Map the external solver's polarization labels to radar-frame VV/HH/VH.
POLARIZATION_MAP = {
    # "ThetaTheta": "VV",
    # "PhiPhi": "HH",
    # "ThetaPhi": "VH",
}

# Set True only after completing FEATURE_VALIDATION_GUIDE.md's convention
# checklist. This is an attestation, not an automatic phase conversion.
ATTEST_GLOBAL_ORIGIN_EXP_PLUS_JWT_RADAR_VH = False

# =============================================================================

from components import (  # noqa: E402
    COMPONENT_AMPLITUDE_CONVENTION,
    COMPONENT_COMPLEX_FIELD_DOMAIN,
    COMPONENT_PHASE_REFERENCE,
)
from feature_sum import validate_radar_grid  # noqa: E402
from grim_compat import load_pattern_any  # noqa: E402
from grim_io import _save_grim_npz  # noqa: E402


def main():
    if not ATTEST_GLOBAL_ORIGIN_EXP_PLUS_JWT_RADAR_VH:
        raise SystemExit(
            "Refusing to stamp coherent metadata. Complete "
            "FEATURE_VALIDATION_GUIDE.md, then set "
            "ATTEST_GLOBAL_ORIGIN_EXP_PLUS_JWT_RADAR_VH=True."
        )
    source = Path(SOURCE_FILE).expanduser().resolve()
    imported = load_pattern_any(str(source), pol_map=POLARIZATION_MAP)
    azimuths, elevations = validate_radar_grid(
        imported["azimuths"], imported["elevations"]
    )
    frequencies = np.asarray(imported["frequencies"], dtype=float)
    polarizations = [
        str(value).strip().upper()
        for value in np.asarray(imported["polarizations"]).ravel()
    ]
    if set(polarizations) != {"VV", "HH", "VH"} or len(polarizations) != 3:
        raise SystemExit(
            "External whole-body reference must provide exactly VV, HH, and "
            f"reciprocal VH after POLARIZATION_MAP; got {polarizations}."
        )
    amplitude = np.asarray(imported["amp"], dtype=np.complex128)
    order = [polarizations.index(channel) for channel in ("VV", "HH", "VH")]
    amplitude = amplitude[..., order]
    polarizations = ["VV", "HH", "VH"]
    expected = (
        len(azimuths), len(elevations), len(frequencies), len(polarizations)
    )
    if amplitude.shape != expected or not np.all(
        np.isfinite(amplitude.real) & np.isfinite(amplitude.imag)
    ):
        raise SystemExit(
            f"External complex field shape/values are invalid: "
            f"{amplitude.shape}, expected {expected}."
        )
    real = amplitude.real.astype(np.float64)
    imag = amplitude.imag.astype(np.float64)
    payload = {
        "azimuths": np.asarray(azimuths, dtype=float),
        "elevations": np.asarray(elevations, dtype=float),
        "frequencies": frequencies,
        "polarizations": np.asarray(polarizations, dtype=str),
        "polarization_alias_primary": ",".join(polarizations),
        "polarization_aliases_json": json.dumps(polarizations),
        "rcs_power": (
            4.0 * np.pi * (real * real + imag * imag)
        ).astype(np.float32),
        "rcs_phase": np.angle(real + 1j * imag).astype(np.float32),
        "rcs_domain": "power_phase",
        "power_domain": "linear_rcs",
        "source_path": str(source),
        "history": (
            "import_3d_reference.py: user-attested external full-wave "
            "monostatic reference; no fitted phase/amplitude correction"
        ),
        "units": json.dumps({
            "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
            "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
        }),
        "phase_reference": COMPONENT_PHASE_REFERENCE,
        "amplitude_convention": COMPONENT_AMPLITUDE_CONVENTION,
        "complex_field_domain": COMPONENT_COMPLEX_FIELD_DOMAIN,
        "raw_complex_amplitude_preserved": True,
        "rcs_amp_real": real,
        "rcs_amp_imag": imag,
        "combine_role": "coherent",
        "combine_role_note": (
            "external direct full-wave truth; convention attested by user"
        ),
    }
    destination = Path(OUTPUT_GRIM).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = _save_grim_npz(payload, str(destination))
    print(f"Wrote attested external reference: {saved}")


if __name__ == "__main__":
    main()
