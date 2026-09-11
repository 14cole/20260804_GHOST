"""Range calibration against reference RCS measurements."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from GRIM_Backend.datasets.constants import C0, _FREQUENCY_UNITS


class GridCalibrationMixin:
    """Range calibration against reference RCS measurements."""

    def range_calibrate(
        self,
        measured_calibration,
        exact_reference,
        range_offset_m,
        *,
        allow_singleton_angular_broadcast=False,
        convention_attested=False,
        measured_label=None,
        exact_label=None,
        maximum_correction_gain_db=60.0,
    ):
        """Apply complex substitution calibration at a signed range offset.

        The stored field is ``A = sqrt(sigma) * exp(1j*phase)``.  For GRIM's
        ``exp(+j*omega*t)`` convention, with a monostatic range response
        proportional to ``exp(-j*2*k*R)``, the operation is

        ``A_out = A_dut * A_exact * exp(-j*4*pi*f*dR/c) / A_measured``.

        ``dR`` is positive when the measured calibration target is farther
        from the radar than the DUT/reference plane. Selecting the measured
        and exact roles expresses the user's calibration intent; unavailable
        acquisition metadata is recorded as assumed rather than blocking the
        calculation. No frequency or angular interpolation is performed.
        """
        from GRIM_Backend.datasets.grid import RcsGrid

        measured_calibration, exact_reference = self._ensure_grids(
            (measured_calibration, exact_reference)
        )
        for option_name, option_value in (
            ("convention_attested", convention_attested),
            (
                "allow_singleton_angular_broadcast",
                allow_singleton_angular_broadcast,
            ),
        ):
            if not isinstance(option_value, (bool, np.bool_)):
                raise TypeError(f"{option_name} must be True or False")
        try:
            offset_m = float(range_offset_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("range offset must be a finite distance in meters") from exc
        if not np.isfinite(offset_m):
            raise ValueError("range offset must be a finite distance in meters")
        prior_calibration_raw = (self.extra or {}).get("range_calibration_json")
        prior_calibration = None
        if prior_calibration_raw is not None:
            try:
                if isinstance(prior_calibration_raw, np.ndarray):
                    prior_calibration_raw = prior_calibration_raw.reshape(()).item()
                prior_calibration = json.loads(str(prior_calibration_raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                prior_calibration = {"unparsed_record": str(prior_calibration_raw)}
        if maximum_correction_gain_db is None:
            gain_limit_db = None
        else:
            try:
                gain_limit_db = float(maximum_correction_gain_db)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "maximum correction gain must be a finite nonnegative dB value"
                ) from exc
            if not np.isfinite(gain_limit_db) or gain_limit_db < 0.0:
                raise ValueError(
                    "maximum correction gain must be a finite nonnegative dB value"
                )

        grids = (
            ("DUT", self),
            ("measured calibration", measured_calibration),
            ("exact reference", exact_reference),
        )
        for label, grid in grids:
            raw_frequency_unit = str(
                (grid.units or {}).get("frequency", "GHz")
            ).strip().lower()
            if raw_frequency_unit not in _FREQUENCY_UNITS:
                raise ValueError(
                    f"{label} has unsupported frequency unit "
                    f"{(grid.units or {}).get('frequency')!r}; Range Cal "
                    "requires Hz, kHz, MHz, or GHz"
                )
            if grid.linear_quantity() != "sigma_3d":
                raise ValueError(
                    f"{label} must contain sigma_3d/dBsm data, got "
                    f"{grid.linear_quantity()}"
                )
            raw_log_unit = (grid.units or {}).get("rcs_log_unit")
            if raw_log_unit is not None and str(raw_log_unit).strip().lower() not in {
                "dbsm",
                "dbm2",
            }:
                raise ValueError(
                    f"{label} has unsupported RCS log unit {raw_log_unit!r}; "
                    "Range Cal requires dBsm"
                )
            if grid.default_log_unit().lower() != "dbsm":
                raise ValueError(
                    f"{label} must use dBsm, got {grid.default_log_unit()}"
                )
        self._assert_physical_metadata_compatible(measured_calibration)
        self._assert_physical_metadata_compatible(exact_reference)
        measured_calibration._assert_physical_metadata_compatible(exact_reference)

        def _declared_time_sign(grid, label):
            values = []
            for container in (grid.units or {}, grid.extra or {}):
                for key in (
                    "time_convention",
                    "phase_reference",
                    "amplitude_convention",
                ):
                    raw = container.get(key)
                    if raw is not None:
                        array = np.asarray(raw)
                        if array.size != 1:
                            raise ValueError(
                                f"{label} metadata {key!r} must be scalar"
                            )
                        values.append(str(array.reshape(-1)[0].item()))
            signs = set()
            for value in values:
                compact = (
                    value.lower()
                    .replace("ω", "omega")
                    .replace("*", "")
                    .replace(" ", "")
                )
                if re.search(r"exp\(\+?j(?:omega|w)t\)", compact):
                    signs.add("+jwt")
                if re.search(r"exp\(-j(?:omega|w)t\)", compact):
                    signs.add("-jwt")
            if len(signs) > 1:
                raise ValueError(
                    f"{label} contains contradictory declared time conventions"
                )
            return next(iter(signs)) if signs else None

        declared_time_signs = {
            label: _declared_time_sign(grid, label) for label, grid in grids
        }
        incompatible_signs = {
            label: sign
            for label, sign in declared_time_signs.items()
            if sign is not None and sign != "+jwt"
        }
        if incompatible_signs:
            details = ", ".join(
                f"{label}={sign}" for label, sign in incompatible_signs.items()
            )
            raise ValueError(
                "Range Cal uses GRIM's exp(+j*omega*t) phase law and will not "
                f"override contradictory declared metadata ({details})"
            )

        def _canonical_polarizations(grid, label):
            labels = [str(value).strip().upper() for value in grid.polarizations]
            if any(not value for value in labels):
                raise ValueError(f"{label} contains a blank polarization label")
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"{label} contains duplicate polarization labels after normalization"
                )
            return labels

        dut_pols = _canonical_polarizations(self, "DUT")
        measured_pols = _canonical_polarizations(
            measured_calibration, "measured calibration"
        )
        exact_pols = _canonical_polarizations(exact_reference, "exact reference")
        missing_measured_pols = [
            value for value in dut_pols if value not in measured_pols
        ]
        missing_exact_pols = [value for value in dut_pols if value not in exact_pols]
        if missing_measured_pols or missing_exact_pols:
            missing_parts = []
            if missing_measured_pols:
                missing_parts.append(
                    "measured calibration: " + ", ".join(missing_measured_pols)
                )
            if missing_exact_pols:
                missing_parts.append(
                    "exact reference: " + ", ".join(missing_exact_pols)
                )
            raise ValueError(
                "calibration references are missing DUT polarization(s) in "
                + "; ".join(missing_parts)
            )

        if not np.array_equal(
            measured_calibration.frequencies, exact_reference.frequencies
        ):
            raise ValueError(
                "measured-calibration and exact-reference frequency axes differ"
            )
        if not np.array_equal(self.frequencies, measured_calibration.frequencies):
            raise ValueError(
                "DUT and calibration frequency axes differ; align them explicitly first"
            )

        for axis_name in ("azimuths", "elevations"):
            measured_axis = np.asarray(getattr(measured_calibration, axis_name))
            exact_axis = np.asarray(getattr(exact_reference, axis_name))
            dut_axis = np.asarray(getattr(self, axis_name))
            if not np.array_equal(measured_axis, exact_axis):
                raise ValueError(
                    "measured-calibration and exact-reference "
                    f"{axis_name[:-1]} axes differ"
                )
            if np.array_equal(dut_axis, measured_axis):
                continue
            if len(measured_axis) == 1 and bool(
                allow_singleton_angular_broadcast
            ):
                continue
            if len(measured_axis) == 1:
                raise ValueError(
                    f"singleton calibration {axis_name[:-1]} requires explicit "
                    "broadcast confirmation"
                )
            raise ValueError(
                f"DUT and calibration {axis_name[:-1]} axes differ; no angular "
                "interpolation or averaging is performed"
            )

        frequency_hz = np.asarray(
            self._frequency_value_to_hz(self.frequencies), dtype=np.float64
        )
        if np.any(~np.isfinite(frequency_hz)) or np.any(frequency_hz <= 0.0):
            raise ValueError("range calibration requires positive finite frequencies")

        measured_pol_index = [measured_pols.index(label) for label in dut_pols]
        exact_pol_index = [exact_pols.index(label) for label in dut_pols]
        measured_amp = np.asarray(
            measured_calibration.rcs[..., measured_pol_index], dtype=np.complex128
        )
        exact_amp = np.asarray(
            exact_reference.rcs[..., exact_pol_index], dtype=np.complex128
        )
        dut_amp = np.asarray(self.rcs, dtype=np.complex128)
        measured_power = np.asarray(
            measured_calibration.rcs_power[..., measured_pol_index]
        )
        exact_power = np.asarray(exact_reference.rcs_power[..., exact_pol_index])
        dut_power = np.asarray(self.rcs_power)


        exact_zero_power = np.isfinite(exact_power) & (exact_power == 0.0)
        dut_zero_power = np.isfinite(dut_power) & (dut_power == 0.0)
        exact_amp = np.array(exact_amp, copy=True)
        dut_amp = np.array(dut_amp, copy=True)
        exact_amp[exact_zero_power & ~np.isfinite(exact_amp)] = 0.0 + 0.0j
        dut_amp[dut_zero_power & ~np.isfinite(dut_amp)] = 0.0 + 0.0j


        measured_amp = np.array(measured_amp, copy=True)
        measured_zero_power = np.isfinite(measured_power) & (
            measured_power == 0.0
        )
        measured_amp[
            measured_zero_power & ~np.isfinite(measured_amp)
        ] = 0.0 + 0.0j

        range_phase = np.exp(
            -1j * (4.0 * np.pi * frequency_hz * offset_m / C0)
        ).reshape(1, 1, -1, 1)
        measured_finite = np.isfinite(measured_amp)
        exact_finite = np.isfinite(exact_amp)
        dut_finite = np.isfinite(dut_amp)
        measured_zero = np.isfinite(measured_amp) & (np.abs(measured_amp) == 0.0)


        valid_reference = measured_finite & exact_finite & ~measured_zero
        correction = np.full(
            np.broadcast_shapes(exact_amp.shape, measured_amp.shape),
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )
        correction_numerator = exact_amp * range_phase
        np.divide(
            correction_numerator,
            measured_amp,
            out=correction,
            where=valid_reference & ~measured_zero,
        )
        valid_correction = np.isfinite(correction)
        correction_magnitude = np.abs(correction)
        positive_correction = valid_correction & (correction_magnitude > 0.0)
        correction_gain_db = np.full(correction.shape, np.nan, dtype=np.float64)
        correction_gain_db[positive_correction] = (
            20.0 * np.log10(correction_magnitude[positive_correction])
        )
        excessive = np.zeros(correction.shape, dtype=bool)
        if gain_limit_db is not None:
            excessive = correction_gain_db > gain_limit_db
            correction[excessive] = np.nan + 1j * np.nan
            valid_correction &= ~excessive
        try:
            correction_for_dut = np.broadcast_to(correction, dut_amp.shape)
            with np.errstate(over="ignore", invalid="ignore"):
                output_amp = dut_amp * correction_for_dut
        except ValueError as exc:
            raise ValueError(
                "calibration angular axes cannot broadcast to the DUT grid"
            ) from exc
        if output_amp.shape != dut_amp.shape:
            raise ValueError(
                f"calibration produced shape {output_amp.shape}, expected {dut_amp.shape}"
            )
        candidate_output = dut_finite & np.isfinite(correction_for_dut)
        finite_output_amp = np.isfinite(output_amp)
        overflowed_complex_count = int(
            np.count_nonzero(candidate_output & ~finite_output_amp)
        )
        valid_output = candidate_output & finite_output_amp
        output_power = np.full(output_amp.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            output_power[valid_output] = np.abs(output_amp[valid_output]) ** 2
        overflowed_power = valid_output & ~np.isfinite(output_power)
        overflowed_power_count = int(np.count_nonzero(overflowed_power))
        valid_output &= ~overflowed_power
        output_amp = np.array(output_amp, copy=True)
        output_amp[~valid_output] = np.nan + 1j * np.nan
        output_power[~valid_output] = np.nan
        if not np.any(valid_output):
            raise ValueError(
                "range calibration has no calibratable bins after masking "
                "missing, zero-denominator, over-limit, or overflowed samples"
            )

        finite_gain = correction_gain_db[np.isfinite(correction_gain_db)]
        gain_summary = {
            "minimum": float(np.min(finite_gain)) if finite_gain.size else None,
            "median": float(np.median(finite_gain)) if finite_gain.size else None,
            "maximum": float(np.max(finite_gain)) if finite_gain.size else None,
            "zero_factor_count": int(
                np.count_nonzero(valid_correction & (correction_magnitude == 0.0))
            ),
            "missing_reference_bin_count": int(
                np.count_nonzero(~valid_reference)
            ),
            "zero_measured_denominator_bin_count": int(
                np.count_nonzero(measured_zero)
            ),
            "over_limit_correction_bin_count": int(
                np.count_nonzero(excessive)
            ),
            "overflowed_complex_output_bin_count": overflowed_complex_count,
            "overflowed_power_output_bin_count": overflowed_power_count,
            "masked_output_bin_count": int(np.count_nonzero(~valid_output)),
        }
        measured_name = str(
            measured_label or measured_calibration.source_path or "measured calibration"
        )
        exact_name = str(
            exact_label or exact_reference.source_path or "exact reference"
        )

        def _grid_content_sha256(grid):
            """Bind provenance to the physical complex field Range Cal uses."""

            digest = hashlib.sha256()
            digest.update(b"grim.range-calibration-grid-id.v2\0")

            def _update_array(label, values):
                contiguous = np.ascontiguousarray(values)
                digest.update(label.encode("ascii") + b"\0")
                digest.update(str(contiguous.shape).encode("ascii") + b"\0")
                digest.update(contiguous.tobytes(order="C"))

            for values in (
                np.asarray(grid.azimuths, dtype=np.float64),
                np.asarray(grid.elevations, dtype=np.float64),
                np.asarray(grid.frequencies, dtype=np.float64),
                np.asarray(grid.rcs_power, dtype=np.float64),
                np.asarray(grid.rcs_phase, dtype=np.float64),
            ):
                _update_array("modeled-array", values)


            cells_per_azimuth = int(np.prod(grid.rcs_power.shape[1:]))
            azimuth_block = max(1, 262_144 // max(1, cells_per_azimuth))
            digest.update(b"authoritative-complex-field\0")
            digest.update(str(grid.rcs_power.shape).encode("ascii") + b"\0")
            for start in range(0, len(grid.azimuths), azimuth_block):
                field = np.asarray(
                    grid.rcs_slice(
                        (
                            slice(start, start + azimuth_block),
                            slice(None),
                            slice(None),
                            slice(None),
                        )
                    ),
                    dtype=np.complex128,
                )
                _update_array("field-real", field.real)
                _update_array("field-imag", field.imag)
            digest.update(
                json.dumps(
                    [str(value) for value in grid.polarizations.tolist()],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(
                json.dumps(
                    dict(grid.units or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
            convention_metadata = {
                key: grid._declared_scalar_metadata(key)
                for key in (
                    "phase_reference",
                    "time_convention",
                    "polarization_basis",
                    "amplitude_convention",
                )
            }
            digest.update(
                json.dumps(
                    convention_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            return digest.hexdigest()

        measured_sha256 = _grid_content_sha256(measured_calibration)
        exact_sha256 = _grid_content_sha256(exact_reference)
        provenance = {
            "schema": "grim.range-calibration.v1",
            "mode": "complex_substitution",
            "formula": (
                "A_out=A_dut*A_exact*exp(-j*4*pi*f*delta_R/c)/A_measured_cal"
            ),
            "range_offset_m": offset_m,
            "range_offset_positive_direction": "away_from_radar",
            "phase_law": "exp(+j*omega*t); S(range) proportional to exp(-j*2*k*R)",
            "axis_policy": (
                "exact_frequency_and_polarization; exact_or_explicit_singleton_"
                "broadcast_angular_axes; no_interpolation"
            ),
            "singleton_angular_broadcast": bool(
                bool(allow_singleton_angular_broadcast)
            ),
            "operation_selected_as_convention_assumption": True,
            "user_convention_attested": bool(convention_attested),
            "input_was_previously_range_calibrated": prior_calibration is not None,
            "prior_range_calibration": prior_calibration,
            "measured_calibration": measured_name,
            "measured_calibration_content_sha256": measured_sha256,
            "exact_reference": exact_name,
            "exact_reference_content_sha256": exact_sha256,
            "declared_time_conventions": declared_time_signs,
            "maximum_correction_gain_db": gain_limit_db,
            "correction_gain_db": gain_summary,
        }


        extra = {}
        extra["range_calibration_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        exact_phase_reference = exact_reference._phase_reference()
        extra["phase_reference"] = (
            "range-calibrated complex substitution; exact reference="
            f"{exact_phase_reference or '<unspecified phase center>'}; "
            f"exact_content_sha256={exact_sha256}; "
            f"delta_R={offset_m:.12g} m positive away from radar; "
            "exp(+j*omega*t), S(range)~exp(-j*2*k*R)"
        )
        extra["amplitude_convention"] = (
            "stored complex field magnitude=sqrt(sigma_3d); calibrated by "
            "complex substitution"
        )
        history_entry = (
            f"Range Cal{' re-calibration' if prior_calibration is not None else ''} "
            f"complex substitution: measured={measured_name}; "
            f"exact={exact_name}; delta_R={offset_m:.12g} m positive away "
            "from radar; no interpolation; "
            f"masked_output_bins={gain_summary['masked_output_bin_count']}"
        )
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return RcsGrid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs=output_amp,
            rcs_power=output_power,
            rcs_phase=np.where(valid_output, np.angle(output_amp), np.nan),
            rcs_domain="complex_amplitude",
            source_path=None,
            history=history,
            units=dict(self.units),
            extra=extra,
        )
