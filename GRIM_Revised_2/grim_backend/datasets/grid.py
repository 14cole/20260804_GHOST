"""RCS grid arrays, sample access, and physical-unit conversion."""
from __future__ import annotations

import json

import numpy as np

from grim_backend.datasets.arithmetic import GridArithmeticMixin
from grim_backend.datasets.axes import GridAxesMixin
from grim_backend.datasets.calibration import GridCalibrationMixin
from grim_backend.datasets.combine import GridCombineMixin
from grim_backend.datasets.constants import (
    C0,
    _ADOPT_CLEAN_ARRAYS_TOKEN,
    _FREQUENCY_UNITS,
    _RAW_COMPLEX_VALIDATION_BLOCK_CELLS,
)
from grim_backend.datasets.coordinates import (
    GridCoordinatesMixin,
    canonical_angular_coordinate_system,
)
from grim_backend.datasets.memory import _real_storage_dtype
from grim_backend.datasets.metadata import GridMetadataMixin
from grim_backend.io.cst import CstFormatMixin
from grim_backend.io.native import NativeFormatMixin
from grim_backend.io.out import OutFormatMixin
from grim_backend.io.pioneer import PioFormatMixin
from grim_backend.io.ptm import PtmFormatMixin
from grim_backend.io.sentri import SentriFormatMixin
from grim_backend.io.xpatch import XpatchFormatMixin


class RcsGrid(
    GridAxesMixin,
    GridCoordinatesMixin,
    GridMetadataMixin,
    GridCalibrationMixin,
    GridArithmeticMixin,
    GridCombineMixin,
    NativeFormatMixin,
    CstFormatMixin,
    SentriFormatMixin,
    PioFormatMixin,
    OutFormatMixin,
    XpatchFormatMixin,
    PtmFormatMixin,
):
    """RCS grid arrays, sample access, and physical-unit conversion."""

    def __init__(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=None,
        rcs_power=None,
        rcs_phase=None,
        rcs_domain: str | None = None,
        source_path: str | None = None,
        history: str | None = None,
        units: dict | None = None,
        extra: dict | None = None,
        _adopt_clean_arrays=None,
    ):
        """Build a grid from axis arrays and power/phase-backed RCS samples.

        Use when loading data from files or constructing an in-memory grid.

        Args:
            azimuths: 1D sequence of azimuth values (deg).
            elevations: 1D sequence of elevation values (deg).
            frequencies: 1D sequence of frequency values (GHz or Hz).
            polarizations: 1D sequence of polarization labels.
            rcs: Optional complex field samples shaped (az, el, f, pol).
            rcs_power: Optional linear-power samples shaped (az, el, f, pol).
            rcs_phase: Optional phase samples (radians) shaped (az, el, f, pol).
                Use NaN where phase is unknown.
            rcs_domain: Optional domain tag metadata.
            source_path: Optional source path for provenance.
            history: Optional history string.
            units: Optional units dict (e.g., {"azimuth": "deg", "frequency": "GHz"}).
            extra: Metadata and ancillary arrays retained for native export.
                Raw real/imaginary fields follow exact sample transforms.
                Magnitude and statistical transforms drop raw fields. Axis-union
                joins retain them when all inputs supply compatible raw fields.

        Raises:
            ValueError: if shapes do not match the expected grid.
        """

        self.azimuths = self._clean_axis(azimuths)
        self.elevations = self._clean_axis(elevations)
        self.frequencies = self._clean_axis(frequencies)
        self.polarizations = self._canonical_polarization_axis(polarizations)

        expected = (len(self.azimuths), len(self.elevations), len(self.frequencies), len(self.polarizations))

        complex_arr = None
        real_dtype = _real_storage_dtype(rcs, rcs_power, rcs_phase)
        complex_dtype = np.complex128 if real_dtype == np.float64 else np.complex64
        if rcs is not None:
            rcs_arr = np.asarray(rcs)
            if rcs_arr.shape == expected + (2,):
                complex_arr = np.asarray(
                    rcs_arr[..., 0] + 1j * rcs_arr[..., 1], dtype=complex_dtype
                )
            elif rcs_arr.shape == expected:
                if np.iscomplexobj(rcs_arr):
                    complex_arr = np.asarray(rcs_arr, dtype=complex_dtype)
                elif rcs_power is None:

                    rcs_power = np.asarray(rcs_arr, dtype=real_dtype)
            else:
                raise ValueError(f"rcs shape {rcs_arr.shape} != {expected}")

        if rcs_power is not None:
            power_arr = np.asarray(rcs_power, dtype=real_dtype)
            if power_arr.shape != expected:
                raise ValueError(f"rcs_power shape {power_arr.shape} != {expected}")
        elif complex_arr is not None:
            power_arr = np.abs(complex_arr) ** 2
        else:
            raise ValueError("provide complex rcs samples and/or rcs_power")

        if rcs_phase is not None:
            phase_arr = np.asarray(rcs_phase, dtype=real_dtype)
            if phase_arr.shape != expected:
                raise ValueError(f"rcs_phase shape {phase_arr.shape} != {expected}")
        elif complex_arr is not None:
            phase_arr = np.angle(complex_arr).astype(real_dtype)
        else:
            phase_arr = np.full(expected, np.nan, dtype=real_dtype)

        if (
            _adopt_clean_arrays is not None
            and _adopt_clean_arrays is not False
            and _adopt_clean_arrays is not _ADOPT_CLEAN_ARRAYS_TOKEN
        ):
            raise ValueError("_adopt_clean_arrays is reserved for internal operations")
        if _adopt_clean_arrays is _ADOPT_CLEAN_ARRAYS_TOKEN:


            power_clean = power_arr
            phase_clean = phase_arr
        else:
            power_clean = self._clean_power(power_arr)
            phase_clean = self._clean_phase(phase_arr)
            phase_clean[~np.isfinite(power_clean)] = np.nan

        self.rcs_power = power_clean
        self.rcs_phase = phase_clean
        domain = str(rcs_domain or "").strip().lower()
        if domain not in {"complex_amplitude", "linear_rcs", "power_phase"}:
            domain = "power_phase"
        self.rcs_domain = domain
        self.power_domain = "linear_rcs"
        self.source_path = source_path
        self.history = history
        self.units = dict(units or {})
        self.extra = dict(extra or {})


        envelope = self.extra.get("solver_metadata_json")
        if envelope is not None and self.linear_quantity() == "sigma_2d":
            try:
                parsed = json.loads(str(np.asarray(envelope).reshape(()).item()))
                version = parsed.get("amplitude_version")
                if version is not None and "amplitude_version" not in self.extra:
                    self.extra["amplitude_version"] = version
                elif version is not None and str(self.extra["amplitude_version"]) != str(version):
                    self.extra["solver_metadata_advisory"] = "Conflicting amplitude-version annotations; supplied samples retained."
            except (ValueError, TypeError, AttributeError):
                self.extra["solver_metadata_advisory"] = "Unreadable solver annotation; supplied samples retained."


        phase_wrap = str(self.units.get("phase_wrap", "")).strip()
        if phase_wrap:
            if phase_wrap not in {"0_360", "-180_180"}:
                raise ValueError(
                    "phase_wrap must be '0_360' or '-180_180' when declared"
                )


            with np.errstate(invalid="ignore"):
                if phase_wrap == "0_360":
                    np.remainder(self.rcs_phase, 2.0 * np.pi, out=self.rcs_phase)
                else:
                    np.add(self.rcs_phase, np.pi, out=self.rcs_phase)
                    np.remainder(self.rcs_phase, 2.0 * np.pi, out=self.rcs_phase)
                    np.subtract(self.rcs_phase, np.pi, out=self.rcs_phase)


        unit_coordinate = self.units.get("angular_coordinate_system")
        extra_coordinate = self.extra.get("angular_coordinate_system")
        if (
            (unit_coordinate is None or str(unit_coordinate).strip() == "")
            and extra_coordinate is not None
            and str(extra_coordinate).strip() != ""
        ):
            self.units["angular_coordinate_system"] = (
                canonical_angular_coordinate_system(extra_coordinate)
            )
        if self.angular_coordinate_system() == "great_circle":
            self.units.setdefault(
                "great_circle_coordinate_convention",
                self.great_circle_coordinate_convention(),
            )
            roll, tilt = self.angular_frame_orientation_deg()
            self.units.setdefault("angular_roll_deg", roll)
            self.units.setdefault("angular_tilt_deg", tilt)

    @staticmethod
    def _clean_power(power_value):
        dtype = _real_storage_dtype(power_value)
        power = np.asarray(power_value, dtype=dtype)
        finite = np.isfinite(power)
        out = np.full(power.shape, np.nan, dtype=dtype)
        out[finite] = np.maximum(power[finite], 0.0)
        return out

    @staticmethod
    def _clean_phase(phase_value):
        dtype = _real_storage_dtype(phase_value)
        phase = np.array(phase_value, dtype=dtype, copy=True)
        phase[~np.isfinite(phase)] = np.nan
        return phase

    @staticmethod
    def _complex_from_power_phase(power_value, phase_value):
        real_dtype = _real_storage_dtype(power_value, phase_value)
        complex_dtype = np.complex128 if real_dtype == np.float64 else np.complex64
        power = np.asarray(power_value, dtype=real_dtype)
        phase = np.asarray(phase_value, dtype=real_dtype)
        if power.shape != phase.shape:
            raise ValueError(f"power/phase shapes {power.shape}/{phase.shape} do not match")
        out = np.full(power.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
        valid = np.isfinite(power) & np.isfinite(phase)
        if np.any(valid):
            out[valid] = (
                np.sqrt(power[valid]) * np.exp(1j * phase[valid])
            ).astype(complex_dtype)
        return out

    def _complete_authoritative_raw_arrays(self):
        """Return a complete raw pair, or ``None`` for malformed/partial data."""

        real = self.extra.get("rcs_amp_real")
        imag = self.extra.get("rcs_amp_imag")
        if real is None or imag is None:
            return None
        real = np.asarray(real)
        imag = np.asarray(imag)
        if real.shape != self.rcs_power.shape or imag.shape != self.rcs_power.shape:
            return None


        for start in range(
            0, self.rcs_power.size, _RAW_COMPLEX_VALIDATION_BLOCK_CELLS
        ):
            stop = min(
                self.rcs_power.size,
                start + _RAW_COMPLEX_VALIDATION_BLOCK_CELLS,
            )


            modeled = np.isfinite(self.rcs_power.flat[start:stop])
            raw_finite = np.isfinite(real.flat[start:stop])
            raw_finite &= np.isfinite(imag.flat[start:stop])
            if np.any(modeled != raw_finite):
                return None
        return real, imag

    def _drop_malformed_raw_metadata(self, extra):
        """Remove a partial raw pair before it can become derived authority."""

        if self._complete_authoritative_raw_arrays() is None:
            for key in (
                "rcs_amp_real",
                "rcs_amp_imag",
                "raw_complex_amplitude_preserved",
            ):
                extra.pop(key, None)
        return extra

    def _authoritative_raw_amplitude_from_pair(self, pair, selection=None):
        """Normalize one previously validated raw real/imaginary pair."""

        real, imag = pair
        quantity = self.linear_quantity()
        if selection is None:
            real_values = real.astype(np.float64, copy=False)
            imag_values = imag.astype(np.float64, copy=False)
        else:
            real_values = real[selection].astype(np.float64, copy=False)
            imag_values = imag[selection].astype(np.float64, copy=False)
        raw = real_values + 1j * imag_values
        if quantity == "sigma_2d":
            freq_hz = self._frequency_value_to_hz(self.frequencies)
            k0 = (2.0 * np.pi * np.asarray(freq_hz, dtype=float)) / C0
            if np.any(~np.isfinite(k0)) or np.any(k0 <= 0.0):
                return None
            scale = 1.0 / (2.0 * np.sqrt(k0))
            if selection is None:
                scale = scale[None, None, :, None]
            else:


                scale = np.broadcast_to(
                    scale[None, None, :, None], self.rcs_power.shape
                )[selection]
            return raw * scale
        if quantity == "sigma_3d":
            return raw * np.sqrt(4.0 * np.pi)
        return None

    def _authoritative_raw_amplitude(self, selection=None):
        """Return a solver-provided raw field when its normalization is known."""

        pair = self._complete_authoritative_raw_arrays()
        if pair is None:
            return None
        return self._authoritative_raw_amplitude_from_pair(pair, selection)

    def _bounded_complex_slice_reader(self):
        """Return a bounded field reader and its real precision.

        The authoritative raw-pair contract is validated once when the reader
        is created, instead of rescanning the entire grid for every azimuth
        block.  Every returned slice is newly allocated and may be used as an
        in-place arithmetic work buffer by internal dense operations.
        """

        pair = self._complete_authoritative_raw_arrays()
        quantity = self.linear_quantity()
        if pair is not None and quantity in {"sigma_2d", "sigma_3d"}:
            if quantity == "sigma_2d":
                freq_hz = self._frequency_value_to_hz(self.frequencies)
                k0 = (2.0 * np.pi * np.asarray(freq_hz, dtype=float)) / C0
                if np.any(~np.isfinite(k0)) or np.any(k0 <= 0.0):
                    pair = None
            if pair is not None:
                return (
                    lambda selection: self._authoritative_raw_amplitude_from_pair(
                        pair, selection
                    ),
                    np.dtype(np.float64),
                )
        real_dtype = np.dtype(
            _real_storage_dtype(self.rcs_power, self.rcs_phase)
        )
        return (
            lambda selection: self._complex_from_power_phase(
                self.rcs_power[selection], self.rcs_phase[selection]
            ),
            real_dtype,
        )

    @property
    def rcs(self):
        """Complex RCS values derived from stored linear power and phase."""
        authoritative = self._authoritative_raw_amplitude()
        if authoritative is not None:
            return authoritative
        return self._complex_from_power_phase(self.rcs_power, self.rcs_phase)

    def rcs_slice(self, selection):
        """Reconstruct only a requested complex slice, avoiding a whole-grid allocation."""
        authoritative = self._authoritative_raw_amplitude(selection)
        if authoritative is not None:
            return authoritative
        return self._complex_from_power_phase(
            self.rcs_power[selection], self.rcs_phase[selection]
        )

    def __len__(self):
        """Return total number of complex samples in the grid."""
        return self.rcs_power.size

    def get(self, az_idx, el_idx, f_idx, p_idx):
        """Fetch a single sample by axis indices.

        Args:
            az_idx: Azimuth index.
            el_idx: Elevation index.
            f_idx: Frequency index.
            p_idx: Polarization index.

        Returns:
            dict with axis values and complex RCS sample.
        """
        return {
            "azimuth": self.azimuths[az_idx],
            "elevation": self.elevations[el_idx],
            "frequency": self.frequencies[f_idx],
            "polarization": self.polarizations[p_idx],
            "rcs": self.rcs_slice((az_idx, el_idx, f_idx, p_idx)),
        }

    def get_axis(self, name):
        """Return a single axis array by name.

        Use when you need a specific axis without unpacking all axes.

        Args:
            name: One of "azimuth", "elevation", "frequency", "polarization".

        Returns:
            Numpy array for the requested axis.
        """
        if name == "azimuth":
            return self.azimuths
        if name == "elevation":
            return self.elevations
        if name == "frequency":
            return self.frequencies
        if name == "polarization":
            return self.polarizations
        raise ValueError(f"unknown axis name: {name}")

    def get_axes(self):
        """Return all axis arrays in a dict."""
        return {
            "azimuths": self.azimuths,
            "elevations": self.elevations,
            "frequencies": self.frequencies,
            "polarizations": self.polarizations,
        }

    def audit(self):
        """Return a non-mutating, JSON-serializable dataset health report.

        The report always contains ``status``, ``errors``, ``warnings``,
        ``info``, and ``metrics``.  Grid samples are scanned in bounded blocks;
        the audit never constructs a second full-size power, phase, or complex
        grid.  This method is deliberately diagnostic: it reports malformed
        public mutations instead of repairing them.
        """

        from grim_backend.datasets.audit import audit_dataset
        return audit_dataset(self)

    def _new_grid(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=None,
        *,
        rcs_power=None,
        rcs_phase=None,
        rcs_domain=None,
        history=None,
        units=None,
        extra=None,
        _adopt_clean_arrays=False,
    ):
        if extra is None:


            extra = self._safe_derived_scalar_extra(
                include_field_conventions=True
            )
            self._carry_native_sentri_hazard(extra, (self,))
        return RcsGrid(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs,
            rcs_power=rcs_power,
            rcs_phase=rcs_phase,
            rcs_domain=(self.rcs_domain if rcs_domain is None else rcs_domain),
            source_path=self.source_path,
            history=history if history is not None else self.history,
            units=dict(self.units if units is None else units),
            extra=extra,
            _adopt_clean_arrays=(
                _ADOPT_CLEAN_ARRAYS_TOKEN if _adopt_clean_arrays else None
            ),
        )

    def _power_from_values(self, rcs_value):
        values_raw = np.asarray(rcs_value)
        if np.iscomplexobj(values_raw):
            values = np.asarray(values_raw, dtype=np.complex128)
            power = np.abs(values) ** 2
        else:
            power = np.asarray(values_raw, dtype=float)
        power = np.asarray(power, dtype=float)
        finite = np.isfinite(power)
        out = np.zeros_like(power, dtype=float)
        out[finite] = np.maximum(power[finite], 0.0)
        out[~finite] = np.nan
        return out

    def rcs_to_linear(self, rcs_value):
        """Convert complex field or real-power values to linear power."""
        return self._power_from_values(rcs_value)

    def linear_to_dbsm(self, linear_value, eps=1e-12):
        linear = np.asarray(linear_value, dtype=float)
        linear = np.where(np.isfinite(linear), linear, np.nan)
        linear = np.maximum(linear, eps)
        return 10.0 * np.log10(linear)

    def _frequency_value_to_hz(self, frequency_value):
        freq = np.asarray(frequency_value, dtype=float)
        unit = self._supported_unit("frequency", _FREQUENCY_UNITS, "GHz")
        if unit == "Hz":
            return freq
        if unit == "MHz":
            return freq * 1.0e6
        if unit == "kHz":
            return freq * 1.0e3
        if unit == "GHz":
            return freq * 1.0e9
        raise AssertionError(f"unhandled canonical frequency unit: {unit}")

    def linear_to_dbke(self, linear_value, frequency_value, eps=1e-12):
        linear = np.asarray(linear_value, dtype=float)
        linear = np.where(np.isfinite(linear), linear, np.nan)
        linear = np.maximum(linear, eps)
        freq_hz = self._frequency_value_to_hz(frequency_value)
        freq_hz = np.asarray(freq_hz, dtype=float)
        freq_hz = np.where(np.isfinite(freq_hz) & (freq_hz > 0.0), freq_hz, np.nan)
        factor = (2.0 * np.pi * freq_hz) / C0
        return 10.0 * np.log10(factor * linear)

    def dbke_to_linear(self, dbke_value, frequency_value):
        dbke = np.asarray(dbke_value, dtype=float)
        freq_hz = self._frequency_value_to_hz(frequency_value)
        freq_hz = np.asarray(freq_hz, dtype=float)
        factor = np.where(np.isfinite(freq_hz) & (freq_hz > 0.0), C0 / (2.0 * np.pi * freq_hz), np.nan)
        return factor * (10.0 ** (dbke / 10.0))

    def default_log_unit(self):
        raw = str((self.units or {}).get("rcs_log_unit", "dBsm")).strip().lower()
        if raw == "dbke":
            return "dBke"
        if raw == "db":
            return "dB"
        return "dBsm"

    def linear_to_default_db(self, linear_value, frequency_value=None, eps=1e-12):
        if self.default_log_unit().lower() == "dbke":
            if frequency_value is None:
                raise ValueError("frequency_value is required for dBke conversion")
            return self.linear_to_dbke(linear_value, frequency_value, eps=eps)
        return self.linear_to_dbsm(linear_value, eps=eps)

    def default_db_to_linear(self, db_value, frequency_value=None):
        """Inverse of ``linear_to_default_db`` — convert dB display values back
        to linear power using the dataset's default log unit (dBsm or dBke).
        """
        if self.default_log_unit().lower() == "dbke":
            if frequency_value is None:
                raise ValueError("frequency_value is required for dBke conversion")
            return self.dbke_to_linear(db_value, frequency_value)
        return 10.0 ** (np.asarray(db_value, dtype=float) / 10.0)

    def _index_for_value(self, axis, value, tol=0.0):
        """Find the first index of a value on an axis.

        Args:
            axis: 1D array to search.
            value: Value to find.
            tol: Absolute tolerance for numeric matching. Text axes such as
                polarization are always matched exactly.

        Returns:
            Integer index of the first match.

        Raises:
            ValueError: if no match is found.
        """
        axis_arr = np.asarray(axis)
        if tol > 0.0 and axis_arr.dtype.kind in "biufc":
            matches = np.where(np.isclose(axis_arr, value, atol=tol, rtol=0.0))[0]
        else:
            matches = np.where(axis_arr == value)[0]
        if matches.size == 0:
            raise ValueError(f"value {value} not found on axis")
        return int(matches[0])

    def get_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0):
        """Fetch a single sample by axis values.

        Use when you have physical axis values rather than indices.

        Args:
            azimuth: Azimuth value.
            elevation: Elevation value.
            frequency: Frequency value.
            polarization: Polarization label.
            tol: Absolute tolerance for numeric matching.

        Returns:
            Complex RCS sample.
        """
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(
            self.polarizations, str(polarization).strip().upper(), tol=tol
        )
        return self.rcs_slice((az_idx, el_idx, f_idx, p_idx))

    def rcs_to_dbsm(self, rcs_value, eps=1e-12):
        """Convert linear RCS to dBsm.

        Args:
            rcs_value: Complex or real RCS value(s).
            eps: Floor to avoid log(0).

        Returns:
            dBsm value(s) as float or ndarray.
        """
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_dbsm(linear, eps=eps)

    def rcs_to_dbke(self, rcs_value, frequency_value, eps=1e-12):
        """Convert linear 2D scattering width to absolute dBke."""
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_dbke(linear, frequency_value, eps=eps)

    def rcs_to_display_db(self, rcs_value, frequency_value=None, eps=1e-12):
        """Convert to the dataset's preferred log-power display unit."""
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_default_db(linear, frequency_value=frequency_value, eps=eps)

    def get_dbsm(self, az_idx, el_idx, f_idx, p_idx, eps=1e-12):
        """Fetch a sample by indices and return dBsm."""
        return self.linear_to_dbsm(self.rcs_power[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbke(self, az_idx, el_idx, f_idx, p_idx, eps=1e-12):
        """Fetch a sample by indices and return dBke."""
        freq_value = self.frequencies[f_idx]
        return self.linear_to_dbke(self.rcs_power[az_idx, el_idx, f_idx, p_idx], freq_value, eps=eps)

    def get_dbsm_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBsm."""
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(
            self.polarizations, str(polarization).strip().upper(), tol=tol
        )
        return self.linear_to_dbsm(self.rcs_power[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbke_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBke."""
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(
            self.polarizations, str(polarization).strip().upper(), tol=tol
        )
        return self.linear_to_dbke(self.rcs_power[az_idx, el_idx, f_idx, p_idx], self.frequencies[f_idx], eps=eps)
