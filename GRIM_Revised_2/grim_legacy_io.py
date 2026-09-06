"""OUT, Xpatch SS, and PTM adapters for RcsGrid."""
from __future__ import annotations

from dataclasses import replace
import os
import re
import numpy as np


class LegacyFormatMixin:
    """OUT, Xpatch SS, and PTM adapters for RcsGrid."""

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
        from grim_dataset import (
            C0,
            _cst_samples_equivalent,
        )

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

    @classmethod
    def load_ss(cls, path, *, max_output_bytes=None):
        """Load an Xpatch ``.ss`` signature file into an RcsGrid.

        Delegates the binary parse to :mod:`read_ss` (a pure-Python port of the
        MATLAB ``ssread.m`` / ``xpheaders.m`` readers), then maps its output
        onto the grid:

            - each signal is one (azimuth, elevation) look;
            - the four polarizations VV/VH/HV/HH become the polarization axis;
            - complex scattering samples retain relative magnitude and phase;
            - Xpatch frequencies are retained in their documented GHz unit.

        The checked-in reader is a hand-transcribed structural port without a
        trusted Xpatch/MATLAB absolute-normalization fixture.  Consequently its
        samples are deliberately tagged as a dimensionless power ratio, not
        sigma3D/dBsm. This keeps plotting and stored-phase inspection available.
        Missing convention metadata is recorded as an assumption; the unresolved
        dimensional quantity still prevents PTM/PIO export, range calibration,
        and coherent vehicle Assembly until a reviewed conversion establishes
        sigma_3d or sigma_2d units.
        """
        from grim_dataset import (
            _ADOPT_CLEAN_ARRAYS_TOKEN,
            _checked_dense_import_allocation,
        )
        import read_ss

        data = read_ss.read_ss(path, verbose=False)

        az = np.round(np.asarray(data["az"], dtype=float), 4)
        el = np.round(np.asarray(data["el"], dtype=float), 4)
        # MATLAB ssread/xpheaders document both uniform and discrete Xpatch
        # frequency values in GHz. Do not use a magnitude heuristic here: a
        # converter must preserve the source convention deterministically.
        freq = np.asarray(data["freq"], dtype=float)

        n_sig = int(az.size)
        n_freq = int(freq.size)
        if el.size != n_sig:
            raise ValueError(
                f"SS elevation axis has {el.size} signal values; expected {n_sig}"
            )
        data_nf = int(np.asarray(data["vv"]).shape[1]) if n_sig else 0
        if not data.get("freq_axis_ok", True):
            raise ValueError(
                "SS header-C looks mislocated (maxfreq != framing freq count), so the "
                "frequency axis is unreliable. Run `python read_ss.py <file>` to inspect "
                "(check the 'header-C offset' / 'match' lines)."
            )
        if n_freq != data_nf:
            raise ValueError(
                f"SS frequency axis ({n_freq}) != per-signal sample count ({data_nf}); "
                "header-C is likely misread (run read_ss.py directly and check 'match')."
            )
        if np.any(~np.isfinite(az)) or np.any(~np.isfinite(el)):
            raise ValueError("SS angular coordinates must be finite")
        if (
            np.any(~np.isfinite(freq))
            or np.any(freq <= 0.0)
            or np.unique(freq).size != freq.size
        ):
            raise ValueError(
                "SS frequency axis must contain unique positive finite GHz values"
            )
        if freq.size > 1 and np.any(np.diff(freq) <= 0.0):
            raise ValueError("SS frequency axis must be strictly increasing")

        az_axis = np.asarray(sorted(set(az.tolist())), dtype=float)
        el_axis = np.asarray(sorted(set(el.tolist())), dtype=float)
        pols = np.asarray(["VV", "VH", "HV", "HH"], dtype=str)
        pol_data = [
            np.asarray(data[name]) for name in ("vv", "vh", "hv", "hh")
        ]
        expected_signal_shape = (n_sig, n_freq)
        for name, samples in zip(("VV", "VH", "HV", "HH"), pol_data):
            if samples.shape != expected_signal_shape:
                raise ValueError(
                    f"SS {name} samples have shape {samples.shape}; expected "
                    f"{expected_signal_shape} from record framing"
                )

        ss_imono = int(data.get("imono", 1))
        ss_angle_source = str(data.get("angle_source", "incident"))
        ss_azimuth_seam_restored = bool(data.get("azimuth_seam_restored", False))
        extra = {}
        if ss_imono == 2:
            if ss_angle_source == "observation":
                extra["fixed_incident_azimuth_deg"] = float(
                    np.asarray(data["az_inc"])[0]
                )
                extra["fixed_incident_elevation_deg"] = float(
                    np.asarray(data["el_inc"])[0]
                )
            else:
                extra["fixed_observation_azimuth_deg"] = float(
                    np.asarray(data["az_obs"])[0]
                )
                extra["fixed_observation_elevation_deg"] = float(
                    np.asarray(data["el_obs"])[0]
                )
        # The four complex matrices and selected axes are all that must stay
        # live for grid construction. Release the parser's duplicate angle and
        # header arrays before the dense allocation.
        del data

        coordinate_owner = {}
        for signal_index, (azimuth, elevation) in enumerate(zip(az, el)):
            key = (float(azimuth), float(elevation))
            previous = coordinate_owner.get(key)
            if previous is not None:
                raise ValueError(
                    "SS angular coordinate collision: signals "
                    f"{previous + 1} and {signal_index + 1} both map to "
                    f"azimuth={key[0]:g}, elevation={key[1]:g} after the "
                    "format's four-decimal coordinate normalization"
                )
            coordinate_owner[key] = signal_index

        az_index = {v: i for i, v in enumerate(az_axis.tolist())}
        el_index = {v: i for i, v in enumerate(el_axis.tolist())}

        shape = (len(az_axis), len(el_axis), n_freq, len(pols))
        resident_bytes = sum(
            int(samples.nbytes)
            for samples in (
                az,
                el,
                freq,
                az_axis,
                el_axis,
                pols,
                *pol_data,
            )
        )
        allocation = _checked_dense_import_allocation(
            shape,
            (np.float32, np.float32),
            source=f"SS import {path}",
            max_output_bytes=max_output_bytes,
            resident_bytes=resident_bytes,
        )
        power = np.full(shape, np.nan, dtype=np.float32)
        phase = np.full(shape, np.nan, dtype=np.float32)
        for s in range(n_sig):
            ai = az_index[float(az[s])]
            ei = el_index[float(el[s])]
            for pj, samples in enumerate(pol_data):
                row = np.asarray(samples[s], dtype=np.complex64)
                finite = np.isfinite(row.real) & np.isfinite(row.imag)
                missing = np.isnan(row.real) & np.isnan(row.imag)
                if np.any(~(finite | missing)):
                    raise ValueError(
                        f"SS {pols[pj]} signal {s + 1} contains an infinite "
                        "or one-sided missing complex sample"
                    )
                if np.any(finite):
                    finite_row = row[finite]
                    real64 = finite_row.real.astype(np.float64)
                    imag64 = finite_row.imag.astype(np.float64)
                    sample_power = real64 * real64 + imag64 * imag64
                    if np.any(sample_power > np.finfo(np.float32).max):
                        raise ValueError(
                            f"SS {pols[pj]} signal {s + 1} magnitude is too "
                            "large for finite relative-power storage"
                        )
                    power[ai, ei, finite, pj] = sample_power.astype(np.float32)
                    phase[ai, ei, finite, pj] = np.arctan2(
                        imag64, real64
                    ).astype(np.float32)

        if not np.isfinite(power).any():
            raise ValueError("SS parsed, but no finite scattering samples were found")

        extra.update(
            {
                "source_format": "Xpatch SS",
                "ss_azimuth_seam_restored": ss_azimuth_seam_restored,
                "ss_absolute_normalization_status": (
                    "unverified; loaded as dimensionless relative power"
                ),
                "ss_reader_validation_scope": (
                    "record framing and axes only; absolute field/RCS normalization "
                    "requires an independent Xpatch or MATLAB ssread fixture"
                ),
                "dense_import_allocation_bytes": allocation["dense_bytes"],
                "dense_import_peak_bytes": allocation["peak_bytes"],
                "dense_import_limit_bytes": allocation["limit_bytes"],
            }
        )

        return cls(
            az_axis,
            el_axis,
            freq,
            pols,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=(f"Loaded Xpatch .ss ({n_sig} signals, {n_freq} freqs, "
                     f"{ss_angle_source} angles, imono={ss_imono}"
                     f"{', restored +180 azimuth seam' if ss_azimuth_seam_restored else ''}"
                     f"): {path}"),
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dB", "rcs_linear_quantity": "power_ratio",
            },
            extra=extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def load_ptm(cls, path):
        """Load one legacy PTM great-circle RCS cut.

        PTM stores a single polarization and pitch/elevation per file, with
        uniformly implied aspect and GHz frequency axes.  Its complex float32
        IQ samples are mapped to GRIM's 3-D RCS power/phase representation.
        The great-circle coordinate convention is retained explicitly in
        ``extra`` so it is not silently mistaken for a conic cut.
        """
        from grim_dataset import (
            GRIM_GC_CONVENTION,
            LEGACY_PTM_GC_CONVENTION,
            _ptm_configuration_has_grim_gc_marker,
        )
        import ptm_io

        parsed = ptm_io.read_ptm(path)
        header = parsed.header
        gc_convention = (
            GRIM_GC_CONVENTION
            if _ptm_configuration_has_grim_gc_marker(header.configuration)
            else LEGACY_PTM_GC_CONVENTION
        )
        header_extra = ptm_io.header_to_extra(header)
        header_extra["great_circle_coordinate_convention"] = gc_convention
        header_extra["ptm_cut_type_source"] = "legacy_reader_assumption_not_header"
        complex_grid = parsed.iq[:, np.newaxis, :, np.newaxis]
        history = (
            f"Loaded PTM great-circle cut ({header.num_aspects} aspects, "
            f"{header.num_frequencies} freqs, {header.polarity}, "
            f"{header.byte_order}-endian): {path}"
        )
        return cls(
            parsed.aspects_deg,
            np.asarray([header.pitch], dtype=np.float32),
            parsed.frequencies_ghz,
            np.asarray([header.polarity]),
            rcs=complex_grid,
            rcs_domain="complex_amplitude",
            source_path=str(path),
            history=history,
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "great_circle",
                "great_circle_coordinate_convention": gc_convention,
                "angular_roll_deg": float(header.roll),
                "angular_tilt_deg": float(header.tilt),
            },
            extra=header_extra,
        )

    def save_ptm(self, path, *, el_idx=None, pol_idx=None):
        """Save one (elevation, polarization) slice as little-endian PTM.

        PTM is a complex 3-D RCS format.  It cannot represent 2-D scattering
        width, missing phase, nonuniform axes, multiple elevations, or multiple
        polarizations in one file.  Callers must select one slice when the grid
        contains more than one elevation or polarization.
        """
        from grim_dataset import (
            GRIM_GC_CONVENTION,
            _ptm_configuration_with_grim_gc_marker,
            _ptm_configuration_without_grim_gc_marker,
        )
        import ptm_io

        if self.linear_quantity() != "sigma_3d":
            raise ValueError(
                "save_ptm: PTM stores 3-D RCS (sigma_3d/dBsm); "
                f"dataset quantity is {self.linear_quantity()!r}"
            )
        if el_idx is None:
            if len(self.elevations) == 1:
                el_idx = 0
            else:
                raise ValueError(
                    f"save_ptm: el_idx required ({len(self.elevations)} elevations present)"
                )
        if pol_idx is None:
            if len(self.polarizations) == 1:
                pol_idx = 0
            else:
                raise ValueError(
                    f"save_ptm: pol_idx required "
                    f"({len(self.polarizations)} polarizations present)"
                )
        el_idx = int(el_idx)
        pol_idx = int(pol_idx)
        if not 0 <= el_idx < len(self.elevations):
            raise IndexError(f"save_ptm: el_idx {el_idx} is out of range")
        if not 0 <= pol_idx < len(self.polarizations):
            raise IndexError(f"save_ptm: pol_idx {pol_idx} is out of range")

        def _angle_axis_to_deg(values, unit_key):
            unit = str((self.units or {}).get(unit_key, "deg")).strip().lower()
            array = np.asarray(values, dtype=float)
            if unit in ("deg", "degree", "degrees", ""):
                return array
            if unit in ("rad", "radian", "radians"):
                return np.rad2deg(array)
            raise ValueError(f"save_ptm: unsupported {unit_key} unit {unit!r}")

        aspects_deg = _angle_axis_to_deg(self.azimuths, "azimuth")
        elevations_deg = _angle_axis_to_deg(self.elevations, "elevation")
        pitch_deg = float(elevations_deg[el_idx])
        frequencies_ghz = np.asarray(
            self._frequency_value_to_hz(self.frequencies), dtype=float
        ) / 1.0e9

        coordinate_system = self.angular_coordinate_system()
        if coordinate_system not in {"conic", "great_circle"}:
            raise ValueError(
                "save_ptm: angular coordinate system must be explicitly "
                f"conic or great_circle; got {coordinate_system!r}"
            )
        is_great_circle = coordinate_system == "great_circle"
        selected_polarity = str(self.polarizations[pol_idx]).strip().upper()
        if not is_great_circle:
            if not np.isclose(pitch_deg, 0.0, atol=1.0e-9, rtol=0.0):
                raise ValueError(
                    "save_ptm: PTM uses great-circle aspect/pitch coordinates; "
                    "a nonzero-elevation conic/untagged slice cannot be "
                    "exported without a physical basis/path conversion"
                )
            roll_deg, tilt_deg = self.angular_frame_orientation_deg()
            if not np.allclose(
                (roll_deg, tilt_deg), (0.0, 0.0), rtol=0.0, atol=1.0e-9
            ):
                raise ValueError(
                    "save_ptm: direct conic-equator export requires "
                    "roll=tilt=0 degrees"
                )
            if selected_polarity in {"VH", "HV"}:
                raise ValueError(
                    "save_ptm: direct conic-equator PTM export supports VV/HH "
                    "only; cross-polar data requires explicit polarization-basis "
                    "rotation"
                )

        power_slice = self.rcs_power[:, el_idx, :, pol_idx]
        phase_slice = self.rcs_phase[:, el_idx, :, pol_idx]
        power_missing = ~np.isfinite(power_slice)
        if np.any(power_missing):
            raise ValueError(
                "save_ptm: PTM has no documented missing-sample marker; "
                f"{int(np.count_nonzero(power_missing))} sample(s) lack finite power"
            )
        phase_missing = (power_slice > 0.0) & ~np.isfinite(phase_slice)
        if np.any(phase_missing):
            raise ValueError(
                "save_ptm: complex PTM export requires phase for every positive-power "
                f"sample; {int(np.count_nonzero(phase_missing))} sample(s) lack phase"
            )
        zero_without_phase = (power_slice == 0.0) & ~np.isfinite(phase_slice)
        complex_slice = np.asarray(
            self.rcs_slice((slice(None), el_idx, slice(None), pol_idx))
        )
        if np.any(zero_without_phase):
            complex_slice = np.array(complex_slice, copy=True)
            complex_slice[zero_without_phase] = 0.0 + 0.0j
        expected_shape = (len(self.azimuths), len(self.frequencies))
        if complex_slice.shape != expected_shape:
            raise ValueError(
                f"save_ptm: slice shape {complex_slice.shape} != {expected_shape}"
            )

        header_extra = dict(self.extra or {})
        roll_deg, tilt_deg = self.angular_frame_orientation_deg()
        header_extra["ptm_roll"] = roll_deg
        header_extra["ptm_tilt"] = tilt_deg
        header = ptm_io.header_from_extra(header_extra)
        # Only the tested 0-degree, zero-roll/tilt, co-pol subset defines
        # GRIM's signed aspect and V/H convention.  Preserve that declaration
        # in the otherwise free-form configuration field.  Strip the marker
        # from every wider case so a later import cannot overclaim certainty.
        convention_is_known = (
            not is_great_circle
            or self.great_circle_coordinate_convention() == GRIM_GC_CONVENTION
        )
        marker_scope_is_trusted = (
            convention_is_known
            and np.isclose(pitch_deg, 0.0, atol=1.0e-9, rtol=0.0)
            and np.allclose(
                (roll_deg, tilt_deg), (0.0, 0.0), rtol=0.0, atol=1.0e-9
            )
            and selected_polarity in {"VV", "HH"}
        )
        configuration = (
            _ptm_configuration_with_grim_gc_marker(header.configuration)
            if marker_scope_is_trusted
            else _ptm_configuration_without_grim_gc_marker(header.configuration)
        )
        header = replace(header, configuration=configuration)
        return ptm_io.write_ptm(
            path,
            aspects_deg,
            frequencies_ghz,
            complex_slice,
            polarity=selected_polarity,
            pitch_deg=pitch_deg,
            header=header,
        )
