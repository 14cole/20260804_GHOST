"""OUT table loading into frequency and azimuth RCS grids."""
from __future__ import annotations

import os
import re

import numpy as np


class OutFormatMixin:
    """OUT table loading into frequency and azimuth RCS grids."""

    @classmethod
    def load_out(cls, path):
        """Load whitespace-delimited `.out` data into an RcsGrid.

        Expected columns per non-comment line:
            frequency_ghz  azimuth_deg  rcs_dbke  phase_deg

        Parsing rules:
            - Lines starting with `#` (or text after `#`) are ignored.
            - Values are whitespace-delimited.
            - Polarization is inferred from filename (`HH` or `VV`);
              if not present, polarization is `NA`.
            - The third column is interpreted as absolute dBke and converted to
              linear 2D scattering width using sigma_2d = (lambda / 2pi) * 10^(dBke/10).

        Output mapping:
            - azimuth axis   <- angle column
            - elevation axis <- single value [0.0]
            - frequency axis <- frequency_ghz column
            - polarization   <- inferred filename polarization
            - stored power   <- linear 2D scattering width (matches .grim storage)
        """
        from grim_backend.datasets.constants import C0
        from grim_backend.io.samples import _cst_samples_equivalent

        file_name = os.path.basename(str(path))
        stem_upper = os.path.splitext(file_name)[0].upper()
        pol_matches = set(
            re.findall(r"(?<![A-Z0-9])(HH|VV)(?![A-Z0-9])", stem_upper)
        )
        if len(pol_matches) > 1:
            raise ValueError(
                f"OUT filename {file_name!r} ambiguously declares both HH and VV"
            )
        pol_label = next(iter(pol_matches)) if pol_matches else "NA"

        records: list[tuple[float, float, float, float]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"line {line_no}: expected exactly 4 columns "
                        "(frequency_ghz azimuth_deg rcs_dbke phase_deg)"
                    )
                try:
                    freq_ghz = float(parts[0])
                    azimuth_deg = float(parts[1])
                    rcs_dbke = float(parts[2])
                    phase_deg = float(parts[3])
                except ValueError as exc:
                    raise ValueError(f"line {line_no}: invalid numeric value ({exc})") from exc

                if not np.isfinite(freq_ghz) or freq_ghz <= 0.0:
                    raise ValueError(
                        f"line {line_no}: frequency_ghz must be positive and finite"
                    )
                if not np.isfinite(azimuth_deg):
                    raise ValueError(
                        f"line {line_no}: azimuth_deg must be finite"
                    )
                if np.isnan(rcs_dbke) or np.isposinf(rcs_dbke):
                    raise ValueError(
                        f"line {line_no}: rcs_dbke must be finite or -Inf"
                    )
                if np.isinf(phase_deg):
                    raise ValueError(
                        f"line {line_no}: phase_deg must be finite or NaN"
                    )
                records.append((freq_ghz, azimuth_deg, rcs_dbke, phase_deg))

        if not records:
            raise ValueError("OUT contains no data rows")

        frequencies = np.asarray(sorted({r[0] for r in records}), dtype=float)
        azimuths = np.asarray(sorted({r[1] for r in records}), dtype=float)
        elevations = np.asarray([0.0], dtype=float)
        polarizations = np.asarray([pol_label], dtype=object)

        f_idx = {float(v): i for i, v in enumerate(frequencies.tolist())}
        az_idx = {float(v): i for i, v in enumerate(azimuths.tolist())}

        shape = (len(azimuths), 1, len(frequencies), 1)
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)

        for freq_ghz, azimuth_deg, rcs_dbke, phase_deg in records:
            ai = az_idx[float(azimuth_deg)]
            fi = f_idx[float(freq_ghz)]
            lambda_m = C0 / (float(freq_ghz) * 1.0e9)
            if np.isneginf(rcs_dbke):
                sigma_2d = 0.0
            else:
                with np.errstate(over="raise", invalid="raise"):
                    try:
                        sigma_2d = (lambda_m / (2.0 * np.pi)) * (
                            10.0 ** (rcs_dbke / 10.0)
                        )
                    except (FloatingPointError, OverflowError) as exc:
                        raise ValueError(
                            "OUT dBke magnitude overflows finite linear power at "
                            f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                        ) from exc
            incoming_power = float(sigma_2d)
            if not np.isfinite(incoming_power):
                raise ValueError(
                    "OUT dBke magnitude does not produce finite linear power at "
                    f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                )
            incoming_phase = (
                float(np.deg2rad(phase_deg)) if np.isfinite(phase_deg) else np.nan
            )
            existing_power = float(power[ai, 0, fi, 0])
            if np.isfinite(existing_power):
                existing_phase = float(phase[ai, 0, fi, 0])
                if not _cst_samples_equivalent(
                    existing_power,
                    existing_phase,
                    incoming_power,
                    incoming_phase,
                ):
                    raise ValueError(
                        "conflicting duplicate OUT sample at "
                        f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                    )
                if not np.isfinite(existing_phase) and np.isfinite(incoming_phase):
                    phase[ai, 0, fi, 0] = incoming_phase
                continue
            power[ai, 0, fi, 0] = incoming_power
            phase[ai, 0, fi, 0] = incoming_phase

        if not np.isfinite(power).any():
            raise ValueError("OUT parsed, but no finite RCS magnitude values were found")

        return cls(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded OUT (dBke -> linear sigma_2d): {path}",
            units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz", "rcs_log_unit": "dBke"},
        )
