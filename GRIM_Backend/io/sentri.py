"""SENTRi table parsing and RCS grid conversion."""
from __future__ import annotations

import csv
import re

import numpy as np


class SentriFormatMixin:
    """SENTRi signature detection and complex-field table loading."""

    @classmethod
    def read_SENTRi(cls, path, *, max_output_bytes=None):
        """Read either RCS table schema emitted by CREATE-RF SENTRi.

        Supported header families:

        * compact ``freq_MHz`` / ``theta_deg`` / ``rcs_pp_dBsm`` columns;
        * descriptive ``Frequency`` / ``Theta`` /
          ``RCSPhiScat_PhiInc`` columns.

        SENTRi's reported polar ``Theta`` is stored unchanged so importing a
        file never silently changes its geometry.  The explicit
        :meth:`convert_sentri_elevation_to_grim` operation maps that native
        top-down convention to GRIM elevation when requested.  Phi sweeps
        contained in [0, 180] retain their positive endpoint; other sweeps
        use the signed [-180, 180) azimuth interval.  Reported
        E-field phase is stored with its original sign, so each sample is
        reconstructed as
        ``10**(dBsm/20) * exp(+1j*deg2rad(phase_deg))``.  The far-field phase convention is recorded for coherent Assembly;
        native theta still requires the explicit signed-elevation conversion.
        """
        from GRIM_Backend.datasets.constants import SENTRI_FAR_FIELD_METADATA, _ADOPT_CLEAN_ARRAYS_TOKEN
        from GRIM_Backend.datasets.memory import _checked_dense_import_allocation
        from GRIM_Backend.io.cst import _cst_compact_header, _cst_dbsm_to_power, _read_cst_delimited_rows, _wrap_cst_azimuth_deg
        from GRIM_Backend.io.samples import _cst_samples_equivalent

        rows = _read_cst_delimited_rows(path)
        if not rows:
            raise ValueError("SENTRi table is empty")

        compact_schema = {
            "freqmhz": "frequency",
            "thetadeg": "theta",
            "phideg": "phi",
            "rcsppdbsm": "rcs_hh",
            "efieldphaseppdeg": "phase_hh",
            "rcsttdbsm": "rcs_vv",
            "efieldphasettdeg": "phase_vv",
            "rcsptdbsm": "rcs_hv",
            "efieldphaseptdeg": "phase_hv",
            "rcstpdbsm": "rcs_vh",
            "efieldphasetpdeg": "phase_vh",
        }
        descriptive_schema = {
            "frequency": "frequency",
            "theta": "theta",
            "phi": "phi",
            "rcsphiscatphiinc": "rcs_hh",
            "phasephiphi": "phase_hh",
            "rcsthetascatthetainc": "rcs_vv",
            "phasethetatheta": "phase_vv",
            "rcsphiscatthetainc": "rcs_hv",
            "phasephitheta": "phase_hv",
            "rcsthetascatphiinc": "rcs_vh",
            "phasethetaphi": "phase_vh",
        }
        required = {
            "frequency", "theta", "phi",
            "rcs_vv", "phase_vv", "rcs_hv", "phase_hv",
            "rcs_vh", "phase_vh", "rcs_hh", "phase_hh",
        }

        header_idx = None
        columns = None
        frequency_scale = None
        schema_name = None
        for row_idx, row in enumerate(rows):
            normalized = [_cst_compact_header(cell) for cell in row]
            for aliases, scale, name in (
                (compact_schema, 1.0e-3, "compact MHz"),
                (descriptive_schema, 1.0e-9, "descriptive Hz"),
            ):
                mapped = {}
                for column_idx, token in enumerate(normalized):
                    key = aliases.get(token)
                    if key is not None and key not in mapped:
                        mapped[key] = column_idx
                if required.issubset(mapped):
                    header_idx = row_idx
                    columns = mapped
                    frequency_scale = scale
                    schema_name = name
                    break
            if header_idx is not None:
                break

        if header_idx is None or columns is None or frequency_scale is None:
            raise ValueError(
                "Could not find a complete SENTRi RCS header. Expected either "
                "freq_MHz/theta_deg/phi_deg with pp/tt/pt/tp magnitude and "
                "phase columns, or Frequency/Theta/Phi with the four "
                "Scat/Inc magnitude and phase pairs."
            )

        def _canonical_sentri_unit(raw_value):
            text = str(raw_value or "").strip().lower().replace("²", "2")
            if text == "°":
                return "deg"
            compact = re.sub(r"[^a-z0-9]+", "", text)
            aliases = {
                "hz": "hz",
                "hertz": "hz",
                "mhz": "mhz",
                "megahertz": "mhz",
                "deg": "deg",
                "degree": "deg",
                "degrees": "deg",
                "dbsm": "dbsm",
                "dbm2": "dbsm",
                "dbsqm": "dbsm",
                "dbsquaremeter": "dbsm",
                "dbsquaremetre": "dbsm",
            }
            return aliases.get(compact, compact)


        data_start_idx = header_idx + 1
        while data_start_idx < len(rows) and (
            not rows[data_start_idx]
            or all(not str(cell).strip() for cell in rows[data_start_idx])
        ):
            data_start_idx += 1
        has_units_row = False
        if data_start_idx < len(rows):
            candidate = rows[data_start_idx]
            frequency_cell_idx = columns["frequency"]
            frequency_cell = (
                str(candidate[frequency_cell_idx]).strip()
                if frequency_cell_idx < len(candidate)
                else ""
            )
            try:
                float(frequency_cell)
            except ValueError:
                expected_frequency_unit = (
                    "mhz" if schema_name == "compact MHz" else "hz"
                )
                expected_units = {
                    "frequency": expected_frequency_unit,
                    "theta": "deg",
                    "phi": "deg",
                    "rcs_vv": "dbsm",
                    "phase_vv": "deg",
                    "rcs_hv": "dbsm",
                    "phase_hv": "deg",
                    "rcs_vh": "dbsm",
                    "phase_vh": "deg",
                    "rcs_hh": "dbsm",
                    "phase_hh": "deg",
                }
                bad_units = []
                for key, expected_unit in expected_units.items():
                    column_idx = columns[key]
                    raw_unit = (
                        candidate[column_idx]
                        if column_idx < len(candidate)
                        else ""
                    )
                    actual_unit = _canonical_sentri_unit(raw_unit)
                    if actual_unit != expected_unit:
                        bad_units.append(
                            f"{key}={str(raw_unit).strip()!r} "
                            f"(expected {expected_unit})"
                        )
                if bad_units:
                    raise ValueError(
                        f"line {data_start_idx + 1}: invalid SENTRi units row: "
                        + "; ".join(bad_units)
                    )
                has_units_row = True
                data_start_idx += 1

        def _number(row, key, line_no, *, allow_negative_infinity=False):
            idx = columns[key]
            text = str(row[idx]).strip() if idx < len(row) else ""
            if not text:
                raise ValueError(f"line {line_no}: {key} is blank")
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_no}: invalid {key} value {text!r}"
                ) from exc
            valid = np.isfinite(value) or (
                allow_negative_infinity and np.isneginf(value)
            )
            if not valid:
                expected = "finite or -Inf" if allow_negative_infinity else "finite"
                raise ValueError(f"line {line_no}: {key} must be {expected}")
            return float(value)

        channel_specs = (
            ("VV", "rcs_vv", "phase_vv"),
            ("HV", "rcs_hv", "phase_hv"),
            ("VH", "rcs_vh", "phase_vh"),
            ("HH", "rcs_hh", "phase_hh"),
        )
        records = []
        seen = {}
        seen_source = {}
        used_zero_360_precedence = False
        used_signed_180_precedence = False
        positive_half_sweep = True
        for row_idx, row in enumerate(
            rows[data_start_idx:], start=data_start_idx + 1
        ):
            if not row or all(not str(cell).strip() for cell in row):
                continue
            raw_frequency = _number(row, "frequency", row_idx)
            frequency_ghz = raw_frequency * frequency_scale
            if not np.isfinite(frequency_ghz) or frequency_ghz <= 0.0:
                raise ValueError(f"line {row_idx}: frequency must be positive")
            theta_deg = _number(row, "theta", row_idx)
            coordinate_tolerance = 1.0e-9
            if (
                theta_deg < -coordinate_tolerance
                or theta_deg > 180.0 + coordinate_tolerance
            ):
                raise ValueError(
                    f"line {row_idx}: SENTRi theta must be in [0, 180] deg"
                )


            if abs(theta_deg) <= coordinate_tolerance:
                theta_deg = 0.0
            elif abs(theta_deg - 180.0) <= coordinate_tolerance:
                theta_deg = 180.0
            elevation_deg = float(theta_deg)
            raw_phi_deg = _number(row, "phi", row_idx)
            if abs(raw_phi_deg) <= coordinate_tolerance:
                raw_phi_deg = 0.0
            elif abs(raw_phi_deg - 360.0) <= coordinate_tolerance:
                raw_phi_deg = 360.0
            elif abs(raw_phi_deg + 180.0) <= coordinate_tolerance:
                raw_phi_deg = -180.0
            elif abs(raw_phi_deg - 180.0) <= coordinate_tolerance:
                raw_phi_deg = 180.0
            positive_half_sweep = (
                positive_half_sweep and 0.0 <= raw_phi_deg <= 180.0
            )
            azimuth_deg = _wrap_cst_azimuth_deg(raw_phi_deg)
            if abs(azimuth_deg) <= coordinate_tolerance:
                azimuth_deg = 0.0
            elif abs(azimuth_deg + 180.0) <= coordinate_tolerance:
                azimuth_deg = -180.0

            for polarization, magnitude_key, phase_key in channel_specs:
                magnitude_dbsm = _number(
                    row, magnitude_key, row_idx, allow_negative_infinity=True
                )
                reported_phase_deg = _number(row, phase_key, row_idx)
                power = _cst_dbsm_to_power(
                    magnitude_dbsm,
                    context=f"line {row_idx} {magnitude_key}",
                )
                phase = float(np.deg2rad(reported_phase_deg))
                key = (azimuth_deg, elevation_deg, frequency_ghz, polarization)
                record = (
                    azimuth_deg, elevation_deg, float(frequency_ghz),
                    polarization, power, phase,
                )
                source_key = (
                    raw_phi_deg, elevation_deg, frequency_ghz, polarization
                )
                if source_key in seen_source:
                    prior_source_line, prior_source_power, prior_source_phase = (
                        seen_source[source_key]
                    )
                    if _cst_samples_equivalent(
                        prior_source_power, prior_source_phase, power, phase
                    ):
                        continue
                    raise ValueError(
                        f"line {row_idx}: conflicting duplicate SENTRi sample "
                        f"at source phi={raw_phi_deg:g}; first defined on line "
                        f"{prior_source_line}"
                    )
                seen_source[source_key] = (row_idx, power, phase)
                if key in seen:
                    (
                        prior_line,
                        prior_power,
                        prior_phase,
                        prior_raw_phi,
                        prior_record_index,
                    ) = seen[key]


                    zero_360_pair = (
                        abs(azimuth_deg) <= coordinate_tolerance
                        and (
                            (
                                abs(prior_raw_phi) <= coordinate_tolerance
                                and abs(raw_phi_deg - 360.0)
                                <= coordinate_tolerance
                            )
                            or (
                                abs(prior_raw_phi - 360.0)
                                <= coordinate_tolerance
                                and abs(raw_phi_deg) <= coordinate_tolerance
                            )
                        )
                    )
                    signed_180_pair = (
                        abs(azimuth_deg + 180.0) <= coordinate_tolerance
                        and (
                            (
                                abs(prior_raw_phi + 180.0)
                                <= coordinate_tolerance
                                and abs(raw_phi_deg - 180.0)
                                <= coordinate_tolerance
                            )
                            or (
                                abs(prior_raw_phi - 180.0)
                                <= coordinate_tolerance
                                and abs(raw_phi_deg + 180.0)
                                <= coordinate_tolerance
                            )
                        )
                    )
                    if zero_360_pair or signed_180_pair:
                        if zero_360_pair:
                            used_zero_360_precedence = True
                            current_is_authoritative = (
                                abs(raw_phi_deg - 360.0)
                                <= coordinate_tolerance
                            )
                        else:
                            used_signed_180_precedence = True
                            current_is_authoritative = (
                                abs(raw_phi_deg - 180.0)
                                <= coordinate_tolerance
                            )
                        if current_is_authoritative:
                            records[prior_record_index] = record
                            seen[key] = (
                                row_idx,
                                power,
                                phase,
                                raw_phi_deg,
                                prior_record_index,
                            )
                        continue
                    if _cst_samples_equivalent(
                        prior_power, prior_phase, power, phase
                    ):
                        continue
                    raise ValueError(
                        f"line {row_idx}: conflicting duplicate SENTRi sample "
                        f"after azimuth wrapping; first defined on line {prior_line}"
                    )
                record_index = len(records)
                seen[key] = (
                    row_idx, power, phase, raw_phi_deg, record_index
                )
                records.append(record)

        if not records:
            raise ValueError("SENTRi table contains no data rows")


        rows.clear()
        del seen, seen_source


        if positive_half_sweep:
            for index, record in enumerate(records):
                if record[0] == -180.0:
                    records[index] = (180.0, *record[1:])

        azimuths = np.asarray(sorted({row[0] for row in records}), dtype=float)
        elevations = np.asarray(sorted({row[1] for row in records}), dtype=float)
        frequencies = np.asarray(sorted({row[2] for row in records}), dtype=float)
        polarizations = np.asarray([spec[0] for spec in channel_specs])
        shape = (
            len(azimuths), len(elevations), len(frequencies), len(polarizations)
        )
        allocation = _checked_dense_import_allocation(
            shape,
            (np.float64, np.float64),
            source=f"SENTRi import {path}",
            max_output_bytes=max_output_bytes,
        )
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)
        az_index = {value: idx for idx, value in enumerate(azimuths.tolist())}
        el_index = {value: idx for idx, value in enumerate(elevations.tolist())}
        freq_index = {value: idx for idx, value in enumerate(frequencies.tolist())}
        pol_index = {value: idx for idx, value in enumerate(polarizations.tolist())}
        for azimuth, elevation, frequency, polarization, sample_power, sample_phase in records:
            index = (
                az_index[azimuth], el_index[elevation], freq_index[frequency],
                pol_index[polarization],
            )
            power[index] = sample_power
            phase[index] = sample_phase

        phi_mapping = (
            "phi retained in [0, 180]" if positive_half_sweep
            else "phi wrapped to [-180, 180)"
        )
        mapping = (
            f"elevation=theta; {phi_mapping}; "
            "VV=tt/theta-theta, HV=pt/phi-theta, "
            "VH=tp/theta-phi, HH=pp/phi-phi; "
            "stored phase=reported E-field phase"
        )
        return cls(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=str(path),
            history=f"Loaded SENTRi {schema_name} RCS table; {mapping}: {path}",
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "conic",
                "elevation_coordinate_convention": "sentri_theta_top_zero",
            },
            extra={
                "source_format": f"SENTRi {schema_name} RCS table",
                **SENTRI_FAR_FIELD_METADATA,
                "sentri_coordinate_mapping": (
                    "elevation=theta; azimuth=phi retained in [0, 180]"
                    if positive_half_sweep
                    else "elevation=theta; azimuth=wrapped phi"
                ),
                "sentri_elevation_convention": "sentri_theta_top_zero",
                "sentri_zero_360_seam_policy": (
                    "source phi=360 supplies canonical azimuth 0 when both "
                    "phi=0 and phi=360 are present"
                ),
                "sentri_zero_360_precedence_used": bool(
                    used_zero_360_precedence
                ),
                "sentri_signed_180_seam_policy": (
                    "source phi=+180 supplies canonical azimuth -180 when "
                    "both phi=-180 and phi=+180 are present"
                ),
                "sentri_signed_180_precedence_used": bool(
                    used_signed_180_precedence
                ),
                "sentri_polarization_mapping": (
                    "VV=tt/theta-theta; HV=pt/phi-theta; "
                    "VH=tp/theta-phi; HH=pp/phi-phi"
                ),
                "sentri_phase_mapping": (
                    "GRIM complex amplitude = 10^(dBsm/20) "
                    "* exp(+j*deg2rad(reported_phase_deg))"
                ),
                "sentri_units_row_present": bool(has_units_row),
                "dense_import_allocation_bytes": allocation["dense_bytes"],
                "dense_import_peak_bytes": allocation["peak_bytes"],
                "dense_import_limit_bytes": allocation["limit_bytes"],
            },
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def has_SENTRi_signature(cls, path):
        """Return whether a delimited file has a supported SENTRi header.

        A matching signature selects the SENTRi parser; malformed data raises a
        format error.
        """
        from GRIM_Backend.io.cst import _cst_compact_header


        with open(path, "r", newline="", encoding="utf-8-sig") as stream:
            sample = stream.read(8192)
            stream.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = max((",", "\t", ";"), key=sample.count)
            rows = []
            for row_index, row in enumerate(csv.reader(stream, delimiter=delimiter)):
                rows.append(row)
                if row_index >= 255:
                    break
        compact_required = {
            "freqmhz",
            "thetadeg",
            "phideg",
            "rcsppdbsm",
            "efieldphaseppdeg",
            "rcsttdbsm",
            "efieldphasettdeg",
            "rcsptdbsm",
            "efieldphaseptdeg",
            "rcstpdbsm",
            "efieldphasetpdeg",
        }
        descriptive_required = {
            "frequency",
            "theta",
            "phi",
            "rcsphiscatphiinc",
            "phasephiphi",
            "rcsthetascatthetainc",
            "phasethetatheta",
            "rcsphiscatthetainc",
            "phasephitheta",
            "rcsthetascatphiinc",
            "phasethetaphi",
        }
        for row in rows:
            tokens = {_cst_compact_header(cell) for cell in row}
            if compact_required.issubset(tokens) or descriptive_required.issubset(tokens):
                return True
            compact_family = {"freqmhz", "thetadeg", "phideg"}.issubset(tokens) and any(
                token.startswith(("rcspp", "rcstt", "rcspt", "rcstp"))
                or token.startswith("efieldphase")
                for token in tokens
            )
            descriptive_family = {"frequency", "theta", "phi"}.issubset(tokens) and any(
                token.startswith(("rcsphiscat", "rcsthetascat"))
                for token in tokens
            )
            if compact_family or descriptive_family:
                return True
        return False
