import copy
from dataclasses import replace
import hashlib
import json
import csv
import os
import re
import tempfile
import warnings
import numpy as np

C0 = 299_792_458.0

# The legacy PTM bytes do not define the sign/origin of their aspect axis or
# the H/V basis used along a great-circle cut.  GRIM therefore distinguishes
# its explicit convention from an unmarked legacy PTM instead of silently
# treating every file as the same coordinate chart.
GRIM_GC_CONVENTION = "grim_gc_v1"
LEGACY_PTM_GC_CONVENTION = "legacy_ptm_unspecified"
_PTM_GRIM_GC_MARKER = "GRIM_GC_V1"


def _ptm_configuration_has_grim_gc_marker(value):
    return bool(
        re.search(
            r"(?<![A-Z0-9_])GRIM_GC_V1(?![A-Z0-9_])",
            str(value or "").upper(),
        )
    )


def _ptm_configuration_with_grim_gc_marker(value):
    """Embed GRIM's coordinate convention without discarding legacy text."""

    text = str(value or "").strip()
    if _ptm_configuration_has_grim_gc_marker(text):
        return text
    if not text:
        return _PTM_GRIM_GC_MARKER
    available = 50 - len(_PTM_GRIM_GC_MARKER) - 1
    return f"{_PTM_GRIM_GC_MARKER};{text[:available]}"


def _ptm_configuration_without_grim_gc_marker(value):
    """Remove only GRIM's semicolon-delimited convention marker."""

    parts = [part.strip() for part in str(value or "").split(";")]
    return ";".join(
        part for part in parts
        if part and part.upper() != _PTM_GRIM_GC_MARKER
    )

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

# Dense joins use a bounded advanced-index block and then transfer ownership of
# their already-sanitized output into RcsGrid. The singleton prevents callers
# from bypassing constructor sanitation with a public-looking boolean switch.
_JOIN_MERGE_BLOCK_CELLS = 262_144
_ADOPT_CLEAN_ARRAYS_TOKEN = object()


def canonical_angular_coordinate_system(value):
    """Normalize scalar angular-coordinate metadata without guessing."""

    raw = value
    if isinstance(raw, np.ndarray) and raw.size == 1:
        raw = raw.reshape(-1)[0]
    text = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "": "conic",
        "az_el": "conic",
        "azimuth_elevation": "conic",
        "spherical": "conic",
        "gc": "great_circle",
        "greatcircle": "great_circle",
    }
    return aliases.get(text, text)


def _read_cst_delimited_rows(path):
    """Read a CST text export while retaining any leading metadata rows."""

    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = max((",", "\t", ";"), key=sample.count)
        return list(csv.reader(stream, delimiter=delimiter))


def _cst_compact_header(value):
    """Normalize CST/MATLAB table headings without losing their unit text."""

    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _cst_frequency_unit(value):
    """Return an exact supported CST frequency unit, never a substring guess."""

    compact = _cst_compact_header(value)
    for prefix in ("frequency", "freq"):
        if compact.startswith(prefix):
            suffix = compact[len(prefix):]
            return suffix if suffix in {"hz", "khz", "mhz", "ghz"} else None
    return None


def _cst_frequency_scale_to_ghz(value):
    unit = _cst_frequency_unit(value)
    scales = {"hz": 1.0e-9, "khz": 1.0e-6, "mhz": 1.0e-3, "ghz": 1.0}
    if unit is None:
        raise ValueError(
            "CST frequency header must explicitly end in exactly Hz, kHz, "
            "MHz, or GHz; other prefixes and unit guessing are unsupported"
        )
    return scales[unit], unit


def _wrap_cst_azimuth_deg(value):
    """Use GRIM's canonical half-open azimuth interval, [-180, 180)."""

    wrapped = float(np.mod(float(value) + 180.0, 360.0) - 180.0)
    return 0.0 if abs(wrapped) < 1.0e-12 else wrapped


def _parse_cst_iq(value):
    """Parse common Python/MATLAB spellings of one complex IQ sample."""

    text = str(value or "").strip()
    if not text:
        return None
    token = text.replace(" ", "").strip("()[]{}")
    token = token.replace("*", "").replace("I", "j").replace("i", "j")
    token = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", token)
    try:
        result = complex(token)
    except ValueError as exc:
        raise ValueError(f"unsupported IQ value {text!r}") from exc
    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError(f"IQ value must be finite, got {text!r}")
    return result


def _cst_dbsm_to_power(value, *, context="CST magnitude"):
    """Convert dBsm to finite float64 power with an actionable overflow error."""

    value = float(value)
    if np.isneginf(value):
        return 0.0
    if not np.isfinite(value):
        raise ValueError(f"{context} must be finite or -Inf, got {value!r}")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.power(10.0, value / 10.0))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            f"{context}={value:g} dBsm overflows finite linear power"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"{context}={value:g} dBsm does not produce finite linear power"
        )
    return result


def _cst_iq_to_power(value, *, context="CST IQ"):
    """Return finite |IQ|^2 without allowing float64 overflow."""

    amplitude = float(abs(value))
    if amplitude > float(np.sqrt(np.finfo(np.float64).max)):
        raise ValueError(f"{context} magnitude overflows finite linear power")
    result = amplitude * amplitude
    if not np.isfinite(result):
        raise ValueError(f"{context} does not produce finite linear power")
    return result


def _cst_samples_equivalent(left_power, left_phase, right_power, right_phase):
    """Return True when two seam/duplicate rows encode the same field."""

    if not np.isclose(
        float(left_power), float(right_power), rtol=1.0e-8, atol=1.0e-12
    ):
        return False
    # Phase is undefined for an exactly zero complex sample. Vendor exporters
    # commonly fill that redundant column with arbitrary values at a null.
    if float(left_power) == 0.0 and float(right_power) == 0.0:
        return True
    left_phase = float(left_phase)
    right_phase = float(right_phase)
    if np.isnan(left_phase) and np.isnan(right_phase):
        return True
    if not (np.isfinite(left_phase) and np.isfinite(right_phase)):
        return False
    phase_error = np.angle(np.exp(1j * (left_phase - right_phase)))
    return abs(float(phase_error)) <= 1.0e-8


def _real_storage_dtype(*values):
    """Return float32 unless any supplied numeric array carries >32-bit precision."""
    dtypes = [np.asarray(value).dtype for value in values if value is not None]
    if any(
        (dtype.kind == "f" and dtype.itemsize > 4)
        or (dtype.kind == "c" and dtype.itemsize > 8)
        for dtype in dtypes
    ):
        return np.float64
    return np.float32


class RcsGrid:
    """Container for gridded RCS data with axis metadata and helpers."""

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
            extra: Optional passthrough metadata from the source file -- keys this
                class does not model, carried so save() can write them back.
                Producers that store the raw complex far-field amplitude
                (rcs_amp_real / rcs_amp_imag, as the Claude21 RCS solver's .grim
                exports do) rely on this: without it a load/save round-trip
                silently drops the amplitude and those tools can no longer read
                the file.  Array entries are only re-emitted while their shape
                still matches the grid, so a cropped or joined grid drops them
                rather than writing a stale array.

        Raises:
            ValueError: if shapes do not match the expected grid.
        """

        self.azimuths = self._clean_axis(azimuths)
        self.elevations = self._clean_axis(elevations)
        self.frequencies = self._clean_axis(frequencies)
        pol_arr = np.asarray(polarizations)
        if pol_arr.dtype.kind == "O":
            # Normalize object arrays of strings to native unicode dtype so
            # np.savez stores them without pickle (round-trips with allow_pickle=False).
            pol_arr = np.asarray([str(p) for p in pol_arr.tolist()])
        self.polarizations = pol_arr

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
                    # Real-valued rcs input is treated as linear power when explicit power is not provided.
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
            # Internal ownership-transfer path for dense operations that
            # allocated and sanitised fresh arrays themselves. Re-copying a
            # multi-gigabyte result here can double the operation's peak RSS.
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

        # Migrate supported legacy/fallback angular metadata into the modeled
        # units dictionary.  Derived grids copy units, whereas arbitrary extra
        # arrays are intentionally not propagated, so leaving physical tags
        # only in extra would let a transform silently turn GC data into conic.
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

    def _authoritative_raw_amplitude(self, selection=None):
        """Return a solver-provided raw field when its normalization is known."""
        real = self.extra.get("rcs_amp_real")
        imag = self.extra.get("rcs_amp_imag")
        if real is None or imag is None:
            return None
        real = np.asarray(real)
        imag = np.asarray(imag)
        if real.shape != self.rcs_power.shape or imag.shape != self.rcs_power.shape:
            return None
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
                scale = scale[selection[2]]
            return raw * scale
        if quantity == "sigma_3d":
            return raw * np.sqrt(4.0 * np.pi)
        return None

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

    def edit_axis_value(self, name, index, value):
        """Return a grid with one safely edited axis value.

        Numeric coordinate edits are kept finite and unique, then the edited
        axis is stable-sorted.  Every sample array follows the same permutation,
        including passthrough arrays whose leading four dimensions match the
        RCS grid.  Polarization edits are label-only: surrounding whitespace is
        removed, blank labels and case-insensitive duplicates are rejected, and
        channel order is preserved.

        The operation is transactional: validation and all reordered arrays are
        prepared before a new :class:`RcsGrid` is returned, so ``self`` is never
        partially mutated when an edit is invalid.
        """

        axis_specs = {
            "azimuth": ("azimuths", 0),
            "elevation": ("elevations", 1),
            "frequency": ("frequencies", 2),
            "polarization": ("polarizations", 3),
        }
        try:
            attribute, axis_index = axis_specs[str(name).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unknown axis name: {name}") from exc

        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("axis index must be an integer")
        item_index = int(index)
        source_axis = np.asarray(getattr(self, attribute))
        if item_index < 0 or item_index >= source_axis.size:
            raise IndexError(
                f"{name} axis index {item_index} is outside 0..{source_axis.size - 1}"
            )

        axes = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        order = np.arange(source_axis.size, dtype=int)
        reordered = False

        if axis_index == 3:
            old_value = str(source_axis[item_index])
            new_value = str(value).strip()
            if not new_value:
                raise ValueError("polarization label must not be blank")
            duplicate_key = new_value.casefold()
            if any(
                str(label).strip().casefold() == duplicate_key
                for candidate_index, label in enumerate(source_axis)
                if candidate_index != item_index
            ):
                raise ValueError(
                    f"polarization label {new_value!r} duplicates another channel"
                )
            if new_value == old_value:
                return self
            # Building a fresh unicode array is important here: assigning a
            # longer label into an existing fixed-width dtype (for example
            # '<U2') would silently truncate it.
            labels = [str(label) for label in source_axis.tolist()]
            labels[item_index] = new_value
            axes[axis_index] = np.asarray(labels, dtype=str)
            old_text = repr(old_value)
            new_text = repr(new_value)
        else:
            old_value = float(source_axis[item_index])
            try:
                new_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} axis value must be numeric") from exc
            if not np.isfinite(new_value):
                raise ValueError(f"{name} axis value must be finite")
            if axis_index == 2 and new_value <= 0.0:
                raise ValueError("frequency axis value must be greater than zero")
            if new_value == old_value:
                return self

            edited_axis = np.asarray(source_axis, dtype=float).copy()
            edited_axis[item_index] = new_value
            if np.unique(edited_axis).size != edited_axis.size:
                raise ValueError(
                    f"{name} axis value {new_value:g} duplicates another coordinate"
                )
            order = np.argsort(edited_axis, kind="stable")
            reordered = not np.array_equal(
                order, np.arange(edited_axis.size, dtype=int)
            )
            axes[axis_index] = edited_axis[order]
            old_text = f"{old_value:g}"
            new_text = f"{new_value:g}"

        if reordered:
            power = np.take(self.rcs_power, order, axis=axis_index)
            phase = np.take(self.rcs_phase, order, axis=axis_index)
        else:
            power = self.rcs_power
            phase = self.rcs_phase

        original_shape = tuple(self.rcs_power.shape)
        stale_grid_metadata = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
        }
        if axis_index == 3:
            stale_grid_metadata.update(
                {"polarization_alias_primary", "polarization_aliases_json"}
            )

        edited_extra = {}
        for key, extra_value in self.extra.items():
            if key in stale_grid_metadata:
                continue
            if reordered:
                extra_array = np.asarray(extra_value)
                if (
                    extra_array.ndim >= 4
                    and tuple(extra_array.shape[:4]) == original_shape
                ):
                    edited_extra[key] = np.take(
                        extra_array, order, axis=axis_index
                    )
                    continue
            # RcsGrid treats passthrough values as immutable.  Sharing an
            # unchanged value keeps a metadata-only edit from duplicating
            # potentially multi-gigabyte embedded body-model arrays; the
            # containing ``extra`` dictionary itself is still new.
            edited_extra[key] = extra_value

        history_entry = (
            f"Edit {name} axis[{item_index}]: {old_text} -> {new_text}"
        )
        if reordered:
            history_entry += "; stable-sorted axis and sample arrays"
        prior_history = str(self.history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )
        return RcsGrid(
            axes[0],
            axes[1],
            axes[2],
            axes[3],
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=copy.deepcopy(self.units),
            extra=edited_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @staticmethod
    def _canonical_unit(value, aliases, default):
        text = str(value or default).strip().lower()
        return aliases.get(text, text)

    def linear_quantity(self):
        """Physical meaning of ``rcs_power`` (sigma_2d, sigma_3d, or ratio)."""
        raw = str((self.units or {}).get("rcs_linear_quantity", "")).strip().lower()
        if raw:
            return raw
        return "sigma_2d" if self.default_log_unit().lower() == "dbke" else "sigma_3d"

    def _phase_reference(self):
        raw = self.extra.get("phase_reference", "")
        if isinstance(raw, np.ndarray) and raw.size == 1:
            raw = raw.reshape(-1)[0]
        return str(raw or "").strip()

    def angular_coordinate_system(self):
        """Return the physical angular convention used by the two angle axes.

        GRIM's native convention is conic azimuth/elevation.  Legacy PTM cuts
        use great-circle aspect/pitch; identical numeric axes from those two
        conventions are not physically interchangeable.
        """
        raw = (self.units or {}).get("angular_coordinate_system")
        if raw is None or str(raw).strip() == "":
            raw = (self.extra or {}).get("angular_coordinate_system", "")
        return canonical_angular_coordinate_system(raw)

    def angular_frame_orientation_deg(self):
        """Return stored great-circle/PTM roll and tilt scalar metadata.

        They remain compatibility fields.  The supplied legacy PTM reference
        does not define their Euler order or enough geometry to apply them as
        rotations, so conversion code must require both to be zero.
        """

        values = []
        for unit_key, extra_key in (
            ("angular_roll_deg", "ptm_roll"),
            ("angular_tilt_deg", "ptm_tilt"),
        ):
            raw = (self.units or {}).get(unit_key)
            if raw is None or str(raw).strip() == "":
                raw = (self.extra or {}).get(extra_key, 0.0)
            array = np.asarray(raw)
            if array.size != 1:
                raise ValueError(f"{unit_key} must be scalar")
            value = float(array.reshape(-1)[0])
            if not np.isfinite(value):
                raise ValueError(f"{unit_key} must be finite")
            values.append(value)
        return tuple(values)

    def great_circle_coordinate_convention(self):
        """Return the declared great-circle chart/basis convention.

        A GRIM-created equatorial great-circle grid is tagged ``grim_gc_v1``.
        An imported PTM without GRIM's marker is deliberately reported as
        ``legacy_ptm_unspecified`` because its byte header does not establish
        aspect sign/origin or the polarization basis.
        """

        raw = (self.units or {}).get("great_circle_coordinate_convention")
        if raw is None or str(raw).strip() == "":
            raw = (self.extra or {}).get(
                "great_circle_coordinate_convention", ""
            )
        text = str(raw or "").strip().lower().replace("-", "_")
        aliases = {
            "": LEGACY_PTM_GC_CONVENTION,
            "grim": GRIM_GC_CONVENTION,
            "grim_gc": GRIM_GC_CONVENTION,
            "legacy": LEGACY_PTM_GC_CONVENTION,
            "unknown": LEGACY_PTM_GC_CONVENTION,
            "unspecified": LEGACY_PTM_GC_CONVENTION,
        }
        return aliases.get(text, text)

    def convert_equatorial_conic_gc(
        self,
        direction,
        *,
        attest_legacy_ptm_convention=False,
    ):
        """Losslessly change the angular tag for the exact zero-plane case.

        No direction interpolation or polarization rotation occurs.  Under
        GRIM's declared convention, signed great-circle aspect equals conic
        azimuth at zero pitch and the V/H bases coincide.  General nonzero-cut
        conversion is intentionally unsupported because its path is curved in
        conic coordinates and it requires a full complex scattering-matrix
        basis rotation.

        An unmarked legacy PTM is accepted only when the caller explicitly
        attests that it uses GRIM's aspect sign/origin and V/H convention.
        """

        direction = str(direction or "").strip().lower()
        if direction not in {"conic_to_gc", "gc_to_conic"}:
            raise ValueError(
                "direction must be 'conic_to_gc' or 'gc_to_conic'"
            )
        source_system = self.angular_coordinate_system()
        expected_source = "conic" if direction == "conic_to_gc" else "great_circle"
        if source_system not in {"conic", "great_circle"}:
            raise ValueError(
                "equatorial Conic/GC conversion does not support angular "
                f"coordinate system {source_system!r}"
            )
        if source_system != expected_source:
            arrow = "Conic→GC" if direction == "conic_to_gc" else "GC→Conic"
            raise ValueError(
                f"{arrow} requires a source tagged {expected_source}; got "
                f"{source_system}"
            )

        az_unit = self._canonical_unit(
            (self.units or {}).get("azimuth"), _ANGLE_UNITS, "deg"
        )
        el_unit = self._canonical_unit(
            (self.units or {}).get("elevation"), _ANGLE_UNITS, "deg"
        )
        if az_unit not in {"deg", "rad"} or el_unit not in {"deg", "rad"}:
            raise ValueError(
                "equatorial Conic/GC conversion requires degree or radian "
                f"angle axes; got azimuth={az_unit!r}, elevation={el_unit!r}"
            )
        azimuths = np.asarray(self.azimuths, dtype=float)
        elevations = np.asarray(self.elevations, dtype=float)
        if azimuths.size == 0 or not np.all(np.isfinite(azimuths)):
            raise ValueError("equatorial Conic/GC conversion needs a finite aspect axis")
        if elevations.size != 1 or not np.all(np.isfinite(elevations)):
            raise ValueError("exact Conic/GC conversion requires exactly one finite cut")
        elevation_deg = (
            np.rad2deg(elevations) if el_unit == "rad" else elevations
        )
        if not np.isclose(elevation_deg[0], 0.0, rtol=0.0, atol=1.0e-7):
            label = "elevation" if source_system == "conic" else "pitch"
            raise ValueError(
                f"exact Conic/GC conversion requires one 0 degree {label} cut"
            )

        roll, tilt = self.angular_frame_orientation_deg()
        if not np.allclose((roll, tilt), (0.0, 0.0), rtol=0.0, atol=1.0e-7):
            raise ValueError(
                "exact Conic/GC conversion requires stored roll=tilt=0 "
                f"degrees; got roll={roll:g}, tilt={tilt:g}"
            )
        polarizations = [
            str(value).strip().upper() for value in self.polarizations
        ]
        unsupported = sorted(set(polarizations) - {"VV", "HH"})
        if unsupported:
            raise ValueError(
                "exact Conic/GC conversion currently supports VV/HH only; "
                "legacy PTM cross-polar basis signs are unspecified; got "
                + ", ".join(unsupported)
            )

        if direction == "gc_to_conic":
            convention = self.great_circle_coordinate_convention()
            if convention != GRIM_GC_CONVENTION:
                if convention != LEGACY_PTM_GC_CONVENTION:
                    raise ValueError(
                        "unsupported great-circle coordinate convention "
                        f"{convention!r}; only GRIM_GC_V1 or an explicitly "
                        "attested unmarked legacy PTM is supported"
                    )
                if not attest_legacy_ptm_convention:
                    raise ValueError(
                        "legacy PTM great-circle convention is unmarked: its "
                        "aspect sign/origin and V/H basis are unspecified. "
                        "Explicitly attest the GRIM GC convention before "
                        "performing this relabel"
                    )
                convention_note = "user-attested legacy PTM as GRIM_GC_V1"
            else:
                convention_note = "declared GRIM_GC_V1"
        else:
            convention_note = "created with GRIM_GC_V1"

        # Canonicalize the periodic primary axis while carrying every sample
        # and any grid-shaped passthrough array through the same stable order.
        period = 2.0 * np.pi if az_unit == "rad" else 360.0
        half_period = 0.5 * period
        wrapped = np.mod(azimuths + half_period, period) - half_period
        wrapped[np.isclose(wrapped, 0.0, rtol=0.0, atol=1.0e-12)] = 0.0
        order = np.argsort(wrapped, kind="stable")
        wrapped = wrapped[order]
        tolerance = np.deg2rad(1.0e-7) if az_unit == "rad" else 1.0e-7
        if wrapped.size > 1 and np.any(np.diff(wrapped) <= tolerance):
            raise ValueError(
                "aspect axis contains duplicate or seam-alias directions after wrapping"
            )

        expected_shape = self.rcs_power.shape
        converted_extra = {}
        for key, value in self._extra_to_write().items():
            array = np.asarray(value)
            if array.ndim >= 4 and array.shape[:4] == expected_shape:
                converted_extra[key] = np.array(array[order, ...], copy=True)
            else:
                converted_extra[key] = copy.deepcopy(value)
        # A coordinate-derived file is not the original solver artifact.  Do
        # not carry a stale grid-bound attestation/certification through the
        # axis normalization even though those scalar blobs fit structurally.
        for key in (
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
        ):
            converted_extra.pop(key, None)

        converted_units = copy.deepcopy(self.units or {})
        if direction == "conic_to_gc":
            converted_units["angular_coordinate_system"] = "great_circle"
            converted_units["great_circle_coordinate_convention"] = GRIM_GC_CONVENTION
            converted_units["angular_roll_deg"] = 0.0
            converted_units["angular_tilt_deg"] = 0.0
            converted_extra["angular_coordinate_system"] = "great_circle"
            converted_extra["great_circle_coordinate_convention"] = GRIM_GC_CONVENTION
        else:
            converted_units["angular_coordinate_system"] = "conic"
            for key in (
                "great_circle_coordinate_convention",
                "angular_roll_deg",
                "angular_tilt_deg",
            ):
                converted_units.pop(key, None)
            for key in (
                "angular_coordinate_system",
                "great_circle_coordinate_convention",
                "ptm_cut_type",
                "ptm_roll",
                "ptm_tilt",
            ):
                converted_extra.pop(key, None)

        arrow = "Conic->GC" if direction == "conic_to_gc" else "GC->Conic"
        history_entry = (
            f"{arrow} exact equatorial relabel; no interpolation; "
            f"{convention_note}; VV/HH only"
        )
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return RcsGrid(
            wrapped,
            np.asarray([0.0], dtype=self.elevations.dtype),
            self.frequencies,
            self.polarizations,
            rcs=None,
            rcs_power=np.asarray(self.rcs_power)[order, ...],
            rcs_phase=np.asarray(self.rcs_phase)[order, ...],
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=converted_units,
            extra=converted_extra,
        )

    def _assert_physical_metadata_compatible(self, other):
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        for key, aliases, default in (
            ("azimuth", _ANGLE_UNITS, "deg"),
            ("elevation", _ANGLE_UNITS, "deg"),
            ("frequency", _FREQUENCY_UNITS, "GHz"),
        ):
            left = self._canonical_unit((self.units or {}).get(key), aliases, default)
            right = self._canonical_unit((other.units or {}).get(key), aliases, default)
            if left != right:
                raise ValueError(f"{key} unit mismatch: {left} != {right}")
        if self.linear_quantity() != other.linear_quantity():
            raise ValueError(
                "RCS linear quantity mismatch: "
                f"{self.linear_quantity()} != {other.linear_quantity()}"
            )
        if self.default_log_unit().lower() != other.default_log_unit().lower():
            raise ValueError(
                f"RCS log unit mismatch: {self.default_log_unit()} != "
                f"{other.default_log_unit()}"
            )
        left_angles = self.angular_coordinate_system()
        right_angles = other.angular_coordinate_system()
        if left_angles != right_angles:
            raise ValueError(
                "angular coordinate system mismatch: "
                f"{left_angles} != {right_angles}"
            )
        if left_angles == "great_circle":
            left_convention = self.great_circle_coordinate_convention()
            right_convention = other.great_circle_coordinate_convention()
            if left_convention != right_convention:
                raise ValueError(
                    "great-circle coordinate convention mismatch: "
                    f"{left_convention} != {right_convention}"
                )
            left_orientation = self.angular_frame_orientation_deg()
            right_orientation = other.angular_frame_orientation_deg()
            if not np.allclose(
                left_orientation, right_orientation, rtol=0.0, atol=1.0e-7
            ):
                raise ValueError(
                    "great-circle frame orientation mismatch: "
                    f"roll/tilt {left_orientation} != {right_orientation} deg"
                )

    def _assert_compatible(self, other, *, coherent=False):
        """Validate another grid for element-wise operations.

        Use before coherent/incoherent add/subtract operations.

        Args:
            other: Another RcsGrid instance.

        Raises:
            TypeError: if other is not an RcsGrid.
            ValueError: if axes or shapes differ.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        if self.rcs_power.shape != other.rcs_power.shape:
            raise ValueError(f"rcs shape {other.rcs_power.shape} != {self.rcs_power.shape}")
        if not np.array_equal(self.azimuths, other.azimuths):
            raise ValueError("azimuth axis mismatch")
        if not np.array_equal(self.elevations, other.elevations):
            raise ValueError("elevation axis mismatch")
        if not np.array_equal(self.frequencies, other.frequencies):
            raise ValueError("frequency axis mismatch")
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch")
        self._assert_physical_metadata_compatible(other)
        if coherent:
            for label, grid in (("left", self), ("right", other)):
                missing = np.isfinite(grid.rcs_power) & ~np.isfinite(grid.rcs_phase)
                if np.any(missing):
                    raise ValueError(
                        f"coherent operation requires phase; {label} grid has "
                        f"{int(np.count_nonzero(missing))} finite-power sample(s) "
                        "with unknown phase"
                    )
            left_ref = self._phase_reference()
            right_ref = other._phase_reference()
            if left_ref != right_ref and (left_ref or right_ref):
                raise ValueError(
                    "coherent operation requires matching phase references; "
                    f"got {left_ref or '<unspecified>'!r} and "
                    f"{right_ref or '<unspecified>'!r}"
                )

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
        from the radar than the DUT/reference plane.  The caller must attest
        that the DUT and measured calibration share an acquisition chain and
        that the exact response has the intended phase center.  No frequency
        or angular interpolation is performed.
        """

        measured_calibration, exact_reference = self._ensure_grids(
            (measured_calibration, exact_reference)
        )
        try:
            offset_m = float(range_offset_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("range offset must be a finite distance in meters") from exc
        if not np.isfinite(offset_m):
            raise ValueError("range offset must be a finite distance in meters")
        if not convention_attested:
            raise ValueError(
                "range calibration requires confirmation that DUT/measured-cal "
                "share one acquisition and phase convention and that the exact "
                "reference uses the intended phase center"
            )
        if "range_calibration_json" in self.extra:
            raise ValueError(
                "dataset already contains Range Cal provenance; use the original "
                "measurement to avoid accidental double calibration"
            )
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
            if len(measured_axis) == 1 and allow_singleton_angular_broadcast:
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

        if np.any(~np.isfinite(dut_amp)):
            raise ValueError(
                "DUT contains missing/nonfinite complex samples; trim or repair it first"
            )
        for label, values in (
            ("measured calibration", measured_amp),
            ("exact reference", exact_amp),
        ):
            if np.any(~np.isfinite(values)):
                raise ValueError(
                    f"{label} contains missing/nonfinite complex samples"
                )
            if np.any(np.abs(values) == 0.0):
                raise ValueError(
                    f"{label} contains a zero/null sample and cannot define a "
                    "complex calibration factor"
                )

        range_phase = np.exp(
            -1j * (4.0 * np.pi * frequency_hz * offset_m / C0)
        ).reshape(1, 1, -1, 1)
        correction = exact_amp * range_phase / measured_amp
        if np.any(~np.isfinite(correction)):
            raise ValueError("range calibration factor is nonfinite")
        correction_gain_db = 20.0 * np.log10(np.abs(correction))
        if np.any(~np.isfinite(correction_gain_db)):
            raise ValueError("range calibration gain is nonfinite")
        if gain_limit_db is not None and np.any(
            correction_gain_db > gain_limit_db
        ):
            count = int(np.count_nonzero(correction_gain_db > gain_limit_db))
            observed = float(np.max(correction_gain_db))
            raise ValueError(
                f"{count} calibration factor(s) exceed the user limit of "
                f"{gain_limit_db:g} dB (maximum {observed:g} dB); inspect the "
                "measured-calibration noise floor/nulls or raise the limit explicitly"
            )
        try:
            output_amp = dut_amp * correction
        except ValueError as exc:
            raise ValueError(
                "calibration angular axes cannot broadcast to the DUT grid"
            ) from exc
        if output_amp.shape != dut_amp.shape:
            raise ValueError(
                f"calibration produced shape {output_amp.shape}, expected {dut_amp.shape}"
            )
        if np.any(~np.isfinite(output_amp)):
            raise ValueError("range calibration produced a nonfinite complex sample")
        try:
            with np.errstate(over="raise", invalid="raise"):
                output_power = np.abs(output_amp) ** 2
        except FloatingPointError as exc:
            raise ValueError(
                "range calibration magnitude overflows finite sigma_3d power"
            ) from exc
        if np.any(~np.isfinite(output_power)):
            raise ValueError(
                "range calibration magnitude does not produce finite sigma_3d power"
            )

        gain_summary = {
            "minimum": float(np.min(correction_gain_db)),
            "median": float(np.median(correction_gain_db)),
            "maximum": float(np.max(correction_gain_db)),
        }
        measured_name = str(
            measured_label or measured_calibration.source_path or "measured calibration"
        )
        exact_name = str(
            exact_label or exact_reference.source_path or "exact reference"
        )

        def _grid_content_sha256(grid):
            digest = hashlib.sha256()
            digest.update(b"grim.range-calibration-grid-id.v1\0")
            for values in (
                np.asarray(grid.azimuths, dtype=np.float64),
                np.asarray(grid.elevations, dtype=np.float64),
                np.asarray(grid.frequencies, dtype=np.float64),
                np.asarray(grid.rcs_power, dtype=np.float64),
                np.asarray(grid.rcs_phase, dtype=np.float64),
            ):
                contiguous = np.ascontiguousarray(values)
                digest.update(str(contiguous.shape).encode("ascii"))
                digest.update(contiguous.tobytes(order="C"))
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
            digest.update(grid._phase_reference().encode("utf-8"))
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
                allow_singleton_angular_broadcast
            ),
            "user_convention_attested": True,
            "measured_calibration": measured_name,
            "measured_calibration_content_sha256": measured_sha256,
            "exact_reference": exact_name,
            "exact_reference_content_sha256": exact_sha256,
            "declared_time_conventions": declared_time_signs,
            "maximum_correction_gain_db": gain_limit_db,
            "correction_gain_db": gain_summary,
        }

        # A calibrated field is a new measurement-domain artifact. Carrying
        # arbitrary DUT extras can resurrect stale solver, delta, raw-field, or
        # certification semantics because save() deliberately round-trips those
        # keys. Rebuild only the truthful calibrated metadata below.
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
            f"{exact_phase_reference or '<user-attested phase center>'}; "
            f"exact_content_sha256={exact_sha256}; "
            f"delta_R={offset_m:.12g} m positive away from radar; "
            "exp(+j*omega*t), S(range)~exp(-j*2*k*R)"
        )
        extra["amplitude_convention"] = (
            "stored complex field magnitude=sqrt(sigma_3d); calibrated by "
            "complex substitution"
        )
        history_entry = (
            f"Range Cal complex substitution: measured={measured_name}; "
            f"exact={exact_name}; delta_R={offset_m:.12g} m positive away "
            "from radar; no interpolation"
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
            rcs_phase=np.angle(output_amp),
            rcs_domain="complex_amplitude",
            source_path=None,
            history=history,
            units=dict(self.units),
            extra=extra,
        )

    def coherent_add(self, other):
        """Coherently add two grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with rcs = self.rcs + other.rcs.
        """
        self._assert_compatible(other, coherent=True)
        rcs_out = self.rcs + other.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_domain="power_phase",
        )

    def coherent_add_many(self, *grids):
        """Coherently add multiple grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with rcs = self.rcs + sum(grid.rcs).
        """
        if not grids:
            return self
        total = np.array(self.rcs, copy=True)
        for grid in grids:
            self._assert_compatible(grid, coherent=True)
            total = total + grid.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            total,
            rcs_domain="power_phase",
        )

    def coherent_subtract(self, other):
        """Coherently subtract two grids (complex difference).

        Use when phases are aligned and you want field-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with rcs = self.rcs - other.rcs.
        """
        self._assert_compatible(other, coherent=True)
        rcs_out = self.rcs - other.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_domain="power_phase",
        )

    def incoherent_add(self, other):
        """Incoherently add two grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = self.rcs_power + other.rcs_power.
        """
        self._assert_compatible(other)
        power_sum = self.rcs_power + other.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_sum,
            rcs_phase=np.full(power_sum.shape, np.nan, dtype=power_sum.dtype),
            rcs_domain="power_phase",
        )

    def incoherent_add_many(self, *grids):
        """Incoherently add multiple grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with linear power = self.rcs_power + sum(grid.rcs_power).
        """
        if not grids:
            return self
        total = np.array(self.rcs_power, copy=True)
        for grid in grids:
            self._assert_compatible(grid)
            total = total + grid.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=total,
            rcs_phase=np.full(total.shape, np.nan, dtype=total.dtype),
            rcs_domain="power_phase",
        )

    def incoherent_subtract(self, other):
        """Incoherently subtract two grids (magnitude difference).

        Use when phases are unrelated and you want power-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = max(self.rcs_power - other.rcs_power, 0).
        """
        self._assert_compatible(other)
        power_diff = self.rcs_power - other.rcs_power
        power_diff = np.maximum(power_diff, 0.0)
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_diff,
            rcs_phase=np.full(power_diff.shape, np.nan, dtype=power_diff.dtype),
            rcs_domain="power_phase",
        )

    def arithmetic_db_subtract(self, other):
        """Return the dimensionless power ratio represented by a dB difference.

        Returns a grid whose dB display equals ``self_dB - other_dB``. For two
        constant lines at 30 and 25 dBsm, the result displays as 5 dB. Phase is
        meaningless for this magnitude-domain operation and is set to NaN.

        Both grids must share the same ``default_log_unit`` (dBsm or dBke).
        """
        self._assert_compatible(other)
        unit_a = self.default_log_unit()
        unit_b = other.default_log_unit()
        if unit_a != unit_b:
            raise ValueError(
                f"dB arithmetic requires matching log units; got {unit_a} vs {unit_b}"
            )

        freq_bcast = None
        if unit_a.lower() == "dbke":
            # rcs_power shape is (az, el, freq, pol); reshape freq so it
            # broadcasts across the freq axis only.
            freq_bcast = np.asarray(self.frequencies, dtype=float)[None, None, :, None]

        db_a = self.linear_to_default_db(self.rcs_power, frequency_value=freq_bcast)
        db_b = other.linear_to_default_db(other.rcs_power, frequency_value=freq_bcast)
        diff_db = db_a - db_b
        output_power = np.power(10.0, np.asarray(diff_db, dtype=float) / 10.0)
        ratio_units = dict(self.units)
        ratio_units["rcs_log_unit"] = "dB"
        ratio_units["rcs_linear_quantity"] = "power_ratio"

        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=output_power,
            rcs_phase=np.full(output_power.shape, np.nan, dtype=output_power.dtype),
            rcs_domain="power_phase",
            units=ratio_units,
        )

    def align_to(self, other, mode="exact"):
        """Align this grid to another grid's axes.

        Modes:
            exact: require identical axes (returns self on success).
            intersect: keep only axis values present in both grids.
            interp: interpolate numeric axes to match other (no extrapolation).

        Args:
            other: Another RcsGrid instance.
            mode: "exact", "intersect", or "interp".

        Returns:
            New RcsGrid aligned to other's axes.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        self._assert_physical_metadata_compatible(other)

        if mode == "exact":
            self._assert_compatible(other)
            return self
        if mode not in ("intersect", "interp"):
            raise ValueError("mode must be 'exact', 'intersect', or 'interp'")

        if mode == "intersect":
            def _match_axis(axis_self, axis_other, tol=1e-6):
                axis_self = np.asarray(axis_self)
                axis_other = np.asarray(axis_other)
                is_numeric = np.issubdtype(axis_self.dtype, np.number) and np.issubdtype(
                    axis_other.dtype, np.number
                )
                if is_numeric and axis_self.size and axis_other.size:
                    self_f = axis_self.astype(float, copy=False).ravel()
                    other_f = axis_other.astype(float, copy=False).ravel()
                    order = np.argsort(self_f, kind="stable")
                    sorted_self = self_f[order]
                    pos = np.searchsorted(sorted_self, other_f)
                    n = sorted_self.size
                    left = np.clip(pos - 1, 0, n - 1)
                    right = np.clip(pos, 0, n - 1)
                    d_left = np.abs(sorted_self[left] - other_f)
                    d_right = np.abs(sorted_self[right] - other_f)
                    use_right = d_right <= d_left
                    sorted_idx = np.where(use_right, right, left)
                    dist = np.where(use_right, d_right, d_left)
                    keep_mask = dist <= tol
                    keep_other = axis_other[keep_mask]
                    indices_self = order[sorted_idx[keep_mask]].astype(int).tolist()
                else:
                    keep_other_list = []
                    indices_self = []
                    for value in axis_other:
                        matches = np.where(axis_self == value)[0]
                        if matches.size > 0:
                            keep_other_list.append(value)
                            indices_self.append(int(matches[0]))
                    keep_other = np.asarray(keep_other_list)
                if not indices_self:
                    raise ValueError("no overlapping axis values for intersect")
                return keep_other, indices_self

            az_new, az_idx = _match_axis(self.azimuths, other.azimuths)
            el_new, el_idx = _match_axis(self.elevations, other.elevations)
            f_new, f_idx = _match_axis(self.frequencies, other.frequencies)
            pol_new, pol_idx = _match_axis(self.polarizations, other.polarizations, tol=0.0)
            pwr_new = self.rcs_power[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            phs_new = self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            return self._new_grid(
                az_new,
                el_new,
                f_new,
                pol_new,
                rcs_power=pwr_new,
                rcs_phase=phs_new,
                rcs_domain="power_phase",
            )

        # interp mode
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch for interp")

        self._check_axis_sorted(self.azimuths, "azimuth")
        self._check_axis_sorted(self.elevations, "elevation")
        self._check_axis_sorted(self.frequencies, "frequency")
        self._check_axis_sorted(other.azimuths, "azimuth")
        self._check_axis_sorted(other.elevations, "elevation")
        self._check_axis_sorted(other.frequencies, "frequency")

        power_interp = self.rcs_power
        phase_interp = self.rcs_phase
        for axis, old, new in (
            (0, self.azimuths, other.azimuths),
            (1, self.elevations, other.elevations),
            (2, self.frequencies, other.frequencies),
        ):
            power_interp, phase_interp = self._interp_power_phase_axis(
                power_interp, phase_interp, old, new, axis
            )
        return self._new_grid(
            other.azimuths,
            other.elevations,
            other.frequencies,
            other.polarizations,
            rcs_power=power_interp,
            rcs_phase=phase_interp,
            rcs_domain="power_phase",
        )

    @staticmethod
    def _check_axis_sorted(axis, name):
        axis = np.asarray(axis)
        if axis.size < 2:
            return
        if not np.all(np.diff(axis) > 0):
            raise ValueError(f"{name} axis must be strictly increasing for interp")

    @staticmethod
    def _interp_complex_axis(data, x_old, x_new, axis):
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        return RcsGrid._interp_linear_axis(data, x_old, x_new, axis)

    @staticmethod
    def _interp_real_axis(data, x_old, x_new, axis):
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        return RcsGrid._interp_linear_axis(data, x_old, x_new, axis)

    @staticmethod
    def _interp_linear_axis(data, x_old, x_new, axis):
        """Vectorized adjacent-bin interpolation; NaNs remain local."""
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        if x_new.min() < x_old.min() or x_new.max() > x_old.max():
            raise ValueError("interp would require extrapolation")
        moved = np.moveaxis(np.asarray(data), axis, 0)
        right = np.searchsorted(x_old, x_new, side="left")
        right = np.clip(right, 0, len(x_old) - 1)
        exact = x_old[right] == x_new
        left = np.where(exact, right, np.maximum(right - 1, 0))
        denom = x_old[right] - x_old[left]
        weight = np.divide(
            x_new - x_old[left],
            denom,
            out=np.zeros_like(x_new, dtype=float),
            where=denom != 0.0,
        )
        reshape = (len(x_new),) + (1,) * (moved.ndim - 1)
        w = weight.reshape(reshape)
        out = moved[left] * (1.0 - w) + moved[right] * w
        return np.moveaxis(out.astype(moved.dtype, copy=False), 0, axis)

    @staticmethod
    def _interp_power_phase_axis(power, phase, x_old, x_new, axis):
        power_out = RcsGrid._interp_real_axis(power, x_old, x_new, axis)
        complex_in = RcsGrid._complex_from_power_phase(power, phase)
        complex_out = RcsGrid._interp_complex_axis(complex_in, x_old, x_new, axis)
        complex_valid = np.isfinite(complex_out.real) & np.isfinite(complex_out.imag)
        phase_out = np.full(power_out.shape, np.nan, dtype=power_out.dtype)
        if np.any(complex_valid):
            power_out = np.array(power_out, copy=True)
            power_out[complex_valid] = np.abs(complex_out[complex_valid]) ** 2
            phase_out[complex_valid] = np.angle(complex_out[complex_valid])
        return power_out, phase_out

    def interpolate_axis(self, axis_name, new_values):
        """Linearly interpolate the grid onto new values along one numeric axis.

        Other axes are left unchanged. Raises if `new_values` extends beyond
        the existing axis range (no extrapolation).
        """
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2}
        key = str(axis_name).strip().lower()
        if key not in axis_map:
            raise ValueError(f"axis must be one of {list(axis_map)}")
        axis_idx = axis_map[key]
        new_arr = np.asarray(new_values, dtype=float).ravel()
        if new_arr.size == 0:
            raise ValueError("new axis must have at least one value")
        if new_arr.size > 1 and not np.all(np.diff(new_arr) > 0):
            raise ValueError("new axis must be strictly increasing")

        old_axes = [self.azimuths, self.elevations, self.frequencies]
        self._check_axis_sorted(old_axes[axis_idx], key)

        new_axes = list(old_axes)
        new_axes[axis_idx] = new_arr

        power_interp, phase_interp = self._interp_power_phase_axis(
            self.rcs_power,
            self.rcs_phase,
            old_axes[axis_idx],
            new_arr,
            axis_idx,
        )
        return self._new_grid(
            new_axes[0],
            new_axes[1],
            new_axes[2],
            self.polarizations,
            rcs_power=power_interp,
            rcs_phase=phase_interp,
            rcs_domain="power_phase",
        )

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _clean_axis(axis):
        """Normalize an axis to float64 (numeric) or keep dtype (non-numeric).

        For float32 input, round-trips each value through its shortest-decimal
        repr so that user-intended values like 0.1 stay as 0.1 in float64
        instead of inheriting the float32 quantization noise (0.10000000149...).
        That way later ops like `shift_azimuth(180)` produce clean values
        (180.1 instead of 180.10000001).
        """
        arr = np.asarray(axis)
        if not np.issubdtype(arr.dtype, np.number):
            return arr
        if arr.dtype == np.float32:
            return arr.astype(str).astype(np.float64)
        return arr.astype(np.float64, copy=False)

    @staticmethod
    def _axis_value_match(axis_arr, value, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        if np.issubdtype(axis_arr.dtype, np.number) and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return np.where(np.isclose(axis_arr, float(value), atol=tol, rtol=0.0))[0]
        return np.where(axis_arr == value)[0]

    @staticmethod
    def _indices_for_axis_values(axis_arr, values, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        values_arr = np.asarray(values)
        if values_arr.size == 0:
            return []
        if axis_arr.size == 0:
            return None
        if np.issubdtype(axis_arr.dtype, np.number) and np.issubdtype(
            values_arr.dtype, np.number
        ):
            axis_f = axis_arr.astype(float, copy=False).ravel()
            values_f = values_arr.astype(float, copy=False).ravel()
            order = np.argsort(axis_f, kind="stable")
            sorted_axis = axis_f[order]
            pos = np.searchsorted(sorted_axis, values_f)
            n = sorted_axis.size
            left = np.clip(pos - 1, 0, n - 1)
            right = np.clip(pos, 0, n - 1)
            d_left = np.abs(sorted_axis[left] - values_f)
            d_right = np.abs(sorted_axis[right] - values_f)
            use_right = d_right <= d_left
            sorted_idx = np.where(use_right, right, left)
            dist = np.where(use_right, d_right, d_left)
            if np.any(dist > tol):
                return None
            orig_idx = order[sorted_idx]
            seen = set()
            out = []
            for i in orig_idx.tolist():
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            return out
        idx_map = {}
        for i in range(axis_arr.size):
            v = axis_arr[i]
            key = v.item() if isinstance(v, np.generic) else v
            if key not in idx_map:
                idx_map[key] = i
        seen = set()
        out = []
        for value in values_arr:
            key = value.item() if isinstance(value, np.generic) else value
            if key not in idx_map:
                return None
            idx = idx_map[key]
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    @staticmethod
    def _axis_union(axis_arrays, tol=1e-6):
        if not axis_arrays:
            return np.asarray([])
        first_dtype = np.asarray(axis_arrays[0]).dtype
        numeric_axis = np.issubdtype(first_dtype, np.number)
        if not numeric_axis:
            seen = {}
            for axis_arr in axis_arrays:
                for value in np.asarray(axis_arr):
                    key = value.item() if isinstance(value, np.generic) else value
                    if key not in seen:
                        seen[key] = None
            return np.asarray(list(seen))
        parts = [np.asarray(a, dtype=float).ravel() for a in axis_arrays]
        combined = np.concatenate(parts) if parts else np.asarray([], dtype=float)
        if combined.size == 0:
            return np.asarray([])
        combined.sort(kind="mergesort")
        keep = np.ones(combined.size, dtype=bool)
        if tol <= 0:
            keep[1:] = combined[1:] != combined[:-1]
        else:
            last_kept = combined[0]
            for i in range(1, combined.size):
                if combined[i] - last_kept > tol:
                    last_kept = combined[i]
                else:
                    keep[i] = False
        return combined[keep]

    @staticmethod
    def _common_axis_alignment(axis_arrays, tol=1e-6):
        """Return a symmetric axis intersection and indices into every input.

        Numeric values match only when one value from every axis fits in a
        window no wider than ``tol``.  The lowest value in that window is the
        canonical output coordinate.  Sorting the values before matching makes
        both the coordinates and the chosen samples independent of which
        dataset happened to be selected first.
        """
        arrays = [np.asarray(axis).ravel() for axis in axis_arrays]
        if not arrays:
            return np.asarray([]), []

        try:
            tol = float(tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("tol must be a finite nonnegative number") from exc
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be a finite nonnegative number")

        numeric_flags = [np.issubdtype(array.dtype, np.number) for array in arrays]
        if any(numeric_flags) and not all(numeric_flags):
            raise TypeError("axis inputs must all be numeric or all be nonnumeric")

        if all(numeric_flags):
            sorted_values = []
            sorted_indices = []
            for array in arrays:
                values = array.astype(float, copy=False)
                finite_indices = np.flatnonzero(np.isfinite(values))
                order = np.argsort(values[finite_indices], kind="stable")
                original_indices = finite_indices[order]
                sorted_values.append(values[original_indices])
                sorted_indices.append(original_indices)

            if any(values.size == 0 for values in sorted_values):
                return np.asarray([], dtype=float), [[] for _ in arrays]

            pointers = np.zeros(len(arrays), dtype=np.int64)
            common_values = []
            matched_indices = [[] for _ in arrays]
            while all(
                pointer < values.size
                for pointer, values in zip(pointers, sorted_values)
            ):
                current = np.asarray(
                    [values[pointer] for values, pointer in zip(sorted_values, pointers)],
                    dtype=float,
                )
                low = float(np.min(current))
                high = float(np.max(current))
                if high - low <= tol:
                    common_values.append(low)
                    for axis_idx in range(len(arrays)):
                        matched_indices[axis_idx].append(
                            int(sorted_indices[axis_idx][pointers[axis_idx]])
                        )
                        pointers[axis_idx] += 1
                    continue

                # A lowest value cannot match the current maximum or any later
                # value from that maximum's axis, so discard every tied low.
                for axis_idx, value in enumerate(current):
                    if value == low:
                        pointers[axis_idx] += 1

            return np.asarray(common_values, dtype=float), matched_indices

        indices_by_value = []
        for array in arrays:
            mapping = {}
            for index, raw_value in enumerate(array):
                value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                mapping.setdefault(value, []).append(index)
            indices_by_value.append(mapping)

        common_values = set(indices_by_value[0])
        for mapping in indices_by_value[1:]:
            common_values.intersection_update(mapping)

        def _canonical_key(value):
            value_type = type(value)
            return (value_type.__module__, value_type.__qualname__, repr(value))

        output_values = []
        matched_indices = [[] for _ in arrays]
        for value in sorted(common_values, key=_canonical_key):
            occurrences = min(len(mapping[value]) for mapping in indices_by_value)
            output_values.extend([value] * occurrences)
            for axis_idx, mapping in enumerate(indices_by_value):
                matched_indices[axis_idx].extend(mapping[value][:occurrences])
        return np.asarray(output_values), matched_indices

    @classmethod
    def _axis_intersection(cls, axis_arrays, tol=1e-6):
        common, _indices = cls._common_axis_alignment(axis_arrays, tol=tol)
        return common

    @classmethod
    def _ensure_grids(cls, grids):
        checked = []
        for grid in grids:
            if not isinstance(grid, cls):
                raise TypeError("all inputs must be RcsGrid instances")
            checked.append(grid)
        if not checked:
            raise ValueError("at least one grid is required")
        return checked

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
    ):
        if extra is None:
            # Preserve scalar phase-reference metadata across derived grids, but
            # never carry shape-dependent raw amplitudes through a transform.
            extra = {}
            if "phase_reference" in self.extra:
                extra["phase_reference"] = self.extra["phase_reference"]
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

    def _amplitude_from_power(self, power_value):
        power = self._clean_power(power_value)
        zero_phase = np.zeros(power.shape, dtype=power.dtype)
        return self._complex_from_power_phase(power, zero_phase)

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
        unit = str((self.units or {}).get("frequency", "GHz")).strip().lower()
        if unit == "hz":
            return freq
        if unit == "mhz":
            return freq * 1.0e6
        if unit == "khz":
            return freq * 1.0e3
        return freq * 1.0e9

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

    def axis_crop(
        self,
        *,
        azimuths=None,
        elevations=None,
        frequencies=None,
        polarizations=None,
        azimuth_range=None,
        elevation_range=None,
        frequency_range=None,
        azimuth_min=None,
        azimuth_max=None,
        elevation_min=None,
        elevation_max=None,
        frequency_min=None,
        frequency_max=None,
        tol=1e-6,
    ):
        """Return a grid cropped by explicit axis values and/or numeric ranges."""

        def _resolve_range(raw_range, vmin, vmax):
            if raw_range is not None:
                if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                    raise ValueError("axis range must be a 2-item [min, max] sequence")
                return raw_range[0], raw_range[1]
            if vmin is None and vmax is None:
                return None
            return vmin, vmax

        azimuth_range = _resolve_range(azimuth_range, azimuth_min, azimuth_max)
        elevation_range = _resolve_range(elevation_range, elevation_min, elevation_max)
        frequency_range = _resolve_range(frequency_range, frequency_min, frequency_max)

        def _axis_indices(axis_arr, axis_values, axis_range, axis_name, axis_tol):
            all_indices = list(range(len(axis_arr)))
            values = self._as_list(axis_values)
            if values is not None:
                selected = self._indices_for_axis_values(axis_arr, values, tol=axis_tol)
                if selected is None:
                    raise ValueError(f"{axis_name} contains value(s) not present in dataset")
                indices = selected
            else:
                indices = all_indices

            if axis_range is not None:
                lo, hi = axis_range
                if lo is not None:
                    lo = float(lo)
                if hi is not None:
                    hi = float(hi)
                if lo is not None and hi is not None and lo > hi:
                    lo, hi = hi, lo

                axis_num = np.asarray(axis_arr, dtype=float)
                range_mask = np.ones(axis_num.shape[0], dtype=bool)
                if lo is not None:
                    range_mask &= axis_num >= (lo - axis_tol)
                if hi is not None:
                    range_mask &= axis_num <= (hi + axis_tol)
                range_idx = set(np.where(range_mask)[0].tolist())
                indices = [idx for idx in indices if idx in range_idx]

            if not indices:
                raise ValueError(f"{axis_name} crop produced no samples")
            return indices

        az_idx = _axis_indices(self.azimuths, azimuths, azimuth_range, "azimuth", tol)
        el_idx = _axis_indices(self.elevations, elevations, elevation_range, "elevation", tol)
        f_idx = _axis_indices(self.frequencies, frequencies, frequency_range, "frequency", tol)
        p_idx = _axis_indices(self.polarizations, polarizations, None, "polarization", 0.0)

        return self._new_grid(
            self.azimuths[az_idx],
            self.elevations[el_idx],
            self.frequencies[f_idx],
            self.polarizations[p_idx],
            rcs_power=self.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)],
            rcs_phase=self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)],
        )

    def mirror_about_azimuth(self, azimuth_deg: float):
        """Mirror azimuth axis about a reference angle and return a new grid.

        The transformed axis is `az' = 2*azimuth_deg - az`. Output azimuths are
        sorted ascending, with samples reordered to match.
        """
        about = float(azimuth_deg)
        if not np.isfinite(about):
            raise ValueError("mirror azimuth must be finite")

        az = np.asarray(self.azimuths, dtype=float)
        mirrored_az = (2.0 * about) - az
        order = np.argsort(mirrored_az, kind="stable")

        return self._new_grid(
            mirrored_az[order],
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=self.rcs_power[order, :, :, :],
            rcs_phase=self.rcs_phase[order, :, :, :],
            rcs_domain="power_phase",
        )

    def swap_elevation_azimuth(self):
        """Swap the elevation and azimuth axes and return a new grid."""
        return self._new_grid(
            np.array(self.elevations, copy=True),
            np.array(self.azimuths, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.swapaxes(self.rcs_power, 0, 1).copy(),
            rcs_phase=np.swapaxes(self.rcs_phase, 0, 1).copy(),
            rcs_domain="power_phase",
        )

    def shift_azimuth(self, delta_deg: float):
        """Shift azimuth axis by a constant offset and return a new grid."""
        delta = float(delta_deg)
        if not np.isfinite(delta):
            raise ValueError("azimuth shift must be finite")
        shifted_az = np.asarray(self.azimuths, dtype=float) + delta
        return self._new_grid(
            shifted_az,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    def wrap_azimuth(self, mode: str):
        """Wrap azimuth axis into the given range and return a new grid.

        ``mode`` is ``"0_360"`` for [0, 360) or ``"-180_180"`` for [-180, 180).
        Output azimuths are sorted ascending; samples are reordered to match.
        If wrapping collapses distinct input azimuths onto the same value
        (e.g. 0° and 360° both map to 0° in "0_360"), only the first
        occurrence in the original azimuth order is kept.
        """
        az = np.asarray(self.azimuths, dtype=float)
        if mode == "0_360":
            wrapped = np.mod(az, 360.0)
        elif mode == "-180_180":
            wrapped = np.mod(az + 180.0, 360.0) - 180.0
        else:
            raise ValueError(f"unknown wrap mode: {mode!r}")

        # np.unique returns sorted unique values and the index of the first
        # occurrence of each in the original array — exactly the "drop dupes,
        # keep first, sort ascending" behaviour we want.
        unique_vals, keep_idx = np.unique(wrapped, return_index=True)
        return self._new_grid(
            unique_vals,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=self.rcs_power[keep_idx, :, :, :],
            rcs_phase=self.rcs_phase[keep_idx, :, :, :],
            rcs_domain="power_phase",
        )

    def round_azimuths(self, decimals: int):
        """Round azimuth axis values to ``decimals`` decimal places (no resampling).

        Use to clean up floating-point noise like 180.0001 -> 180.0.
        Raises if rounding collapses two distinct azimuths into the same value.
        """
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.azimuths, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding azimuths to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            rounded,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    def round_elevations(self, decimals: int):
        """Round elevation axis values to ``decimals`` decimal places (no resampling)."""
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.elevations, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding elevations to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            rounded,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    def round_frequencies(self, decimals: int):
        """Round frequency axis values to ``decimals`` decimal places (no resampling)."""
        decimals = int(decimals)
        rounded = np.round(np.asarray(self.frequencies, dtype=float), decimals)
        if rounded.size != np.unique(rounded).size:
            raise ValueError(
                f"Rounding frequencies to {decimals} decimal(s) would create duplicate "
                "values. Use a higher decimal count."
            )
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            rounded,
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    def shift_elevation(self, delta_deg: float):
        """Shift elevation axis by a constant offset and return a new grid."""
        delta = float(delta_deg)
        if not np.isfinite(delta):
            raise ValueError("elevation shift must be finite")
        shifted_el = np.asarray(self.elevations, dtype=float) + delta
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            shifted_el,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    def combine_elevation_pair_to_azimuth_360(
        self,
        elevation_lo: float | None = None,
        elevation_hi: float | None = None,
        *,
        azimuth_shift_deg: float = 180.0,
        tol: float = 1e-6,
    ):
        """Stitch two elevation cuts into one 0-360 azimuth cut.

        The lower-elevation cut keeps its original azimuth values. The higher
        cut is shifted by `azimuth_shift_deg` and merged onto the same output
        elevation plane. Overlap bins keep the lower-elevation data.
        """

        el_axis = np.asarray(self.elevations, dtype=float)
        if el_axis.size < 2:
            raise ValueError("need at least 2 elevation values to combine into 360 azimuth")

        if elevation_lo is None or elevation_hi is None:
            finite = el_axis[np.isfinite(el_axis)]
            if finite.size < 2:
                raise ValueError("elevation axis has fewer than 2 finite values")
            lo_value = float(np.min(finite))
            hi_value = float(np.max(finite))
        else:
            lo_value = float(elevation_lo)
            hi_value = float(elevation_hi)

        if not np.isfinite(lo_value) or not np.isfinite(hi_value):
            raise ValueError("elevation pair values must be finite")
        if np.isclose(lo_value, hi_value, atol=tol, rtol=0.0):
            raise ValueError("elevation pair values must be distinct")

        lo_matches = self._axis_value_match(self.elevations, lo_value, tol=tol)
        hi_matches = self._axis_value_match(self.elevations, hi_value, tol=tol)
        if lo_matches.size == 0 or hi_matches.size == 0:
            raise ValueError("requested elevation pair not found in dataset")

        lo_idx = int(lo_matches[0])
        hi_idx = int(hi_matches[0])
        az_shift = float(azimuth_shift_deg)
        if not np.isfinite(az_shift):
            raise ValueError("azimuth shift must be finite")

        az_base = np.asarray(self.azimuths, dtype=float)
        if az_base.size == 0:
            raise ValueError("dataset has no azimuth samples")

        az_lo = np.array(az_base, copy=True)
        az_hi = np.array(az_base, copy=True) + az_shift
        az_merged = self._axis_union([az_lo, az_hi], tol=tol)
        if az_merged.size == 0:
            raise ValueError("combined azimuth axis is empty")

        out_shape = (len(az_merged), 1, len(self.frequencies), len(self.polarizations))
        out_power = np.full(out_shape, np.nan, dtype=self.rcs_power.dtype)
        out_phase = np.full(out_shape, np.nan, dtype=self.rcs_phase.dtype)

        lo_target_idx = self._indices_for_axis_values(az_merged, az_lo, tol=tol)
        hi_target_idx = self._indices_for_axis_values(az_merged, az_hi, tol=tol)
        if lo_target_idx is None or hi_target_idx is None:
            raise ValueError("failed to align azimuth bins during elevation combine")

        lo_power = self.rcs_power[:, lo_idx, :, :]
        lo_phase = self.rcs_phase[:, lo_idx, :, :]
        hi_power = self.rcs_power[:, hi_idx, :, :]
        hi_phase = self.rcs_phase[:, hi_idx, :, :]

        for src_idx, dst_idx in enumerate(lo_target_idx):
            out_power[dst_idx, 0, :, :] = lo_power[src_idx, :, :]
            out_phase[dst_idx, 0, :, :] = lo_phase[src_idx, :, :]

        for src_idx, dst_idx in enumerate(hi_target_idx):
            existing_power = out_power[dst_idx, 0, :, :]
            existing_phase = out_phase[dst_idx, 0, :, :]
            incoming_power = hi_power[src_idx, :, :]
            incoming_phase = hi_phase[src_idx, :, :]

            take_power = (~np.isfinite(existing_power)) & np.isfinite(incoming_power)
            existing_power[take_power] = incoming_power[take_power]

            take_phase = np.isfinite(incoming_phase) & (
                (~np.isfinite(existing_phase)) | take_power
            )
            existing_phase[take_phase] = incoming_phase[take_phase]

        return self._new_grid(
            az_merged,
            np.asarray([el_axis[lo_idx]], dtype=float),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=out_power,
            rcs_phase=out_phase,
            rcs_domain="power_phase",
        )

    @classmethod
    def join_many(cls, *grids, tol=1e-6, overlap="error", max_output_bytes=None):
        """Join datasets on union axes without silently replacing finite data.

        ``overlap`` may be ``"error"`` (default), ``"first"``, or ``"last"``.
        Equal finite samples are accepted in all modes. ``max_output_bytes`` can
        cap the estimated peak allocation for memory-aware folder workflows.
        """
        grids = cls._ensure_grids(grids)
        if overlap not in {"error", "first", "last"}:
            raise ValueError("overlap must be 'error', 'first', or 'last'")
        ref = grids[0]
        for grid in grids[1:]:
            ref._assert_physical_metadata_compatible(grid)
            left_ref = ref._phase_reference()
            right_ref = grid._phase_reference()
            if left_ref != right_ref and (left_ref or right_ref):
                raise ValueError("cannot join grids with different phase references")
        if len(grids) == 1:
            # Preserve clone semantics, including original axis order, while
            # still using the bounded allocation/ownership-transfer path.
            az_union = np.array(ref.azimuths, copy=True)
            el_union = np.array(ref.elevations, copy=True)
            f_union = np.array(ref.frequencies, copy=True)
            p_union = np.array(ref.polarizations, copy=True)
        else:
            az_union = cls._axis_union([grid.azimuths for grid in grids], tol=tol)
            el_union = cls._axis_union([grid.elevations for grid in grids], tol=tol)
            f_union = cls._axis_union([grid.frequencies for grid in grids], tol=tol)
            p_union = cls._axis_union([grid.polarizations for grid in grids], tol=0.0)

        shape = (len(az_union), len(el_union), len(f_union), len(p_union))
        out_dtype = np.result_type(*[g.rcs_power.dtype for g in grids])
        cell_count = 1
        for dimension in shape:
            cell_count *= int(dimension)
        itemsize = np.dtype(out_dtype).itemsize
        output_bytes = cell_count * itemsize * 2
        # The bounded merge below avoids full input-sized advanced-index
        # copies. Count the two retained arrays, one union-sized sanitation
        # mask, and a conservative allowance for a bounded merge block.
        merge_block_cells = min(cell_count, _JOIN_MERGE_BLOCK_CELLS)
        merge_scratch_bytes = merge_block_cells * (8 * itemsize + 32)
        estimated_peak_bytes = output_bytes + cell_count + merge_scratch_bytes
        if max_output_bytes is not None and estimated_peak_bytes > int(max_output_bytes):
            raise MemoryError(
                f"dense joined grid needs about {estimated_peak_bytes / (1024**3):.2f} GiB peak "
                f"({output_bytes / (1024**3):.2f} GiB retained), "
                f"above the configured limit of {int(max_output_bytes) / (1024**3):.2f} GiB"
            )
        joined_power = np.full(shape, np.nan, dtype=out_dtype)
        joined_phase = np.full(shape, np.nan, dtype=out_dtype)

        for grid in grids:
            az_idx = cls._indices_for_axis_values(az_union, grid.azimuths, tol=tol)
            el_idx = cls._indices_for_axis_values(el_union, grid.elevations, tol=tol)
            f_idx = cls._indices_for_axis_values(f_union, grid.frequencies, tol=tol)
            p_idx = cls._indices_for_axis_values(p_union, grid.polarizations, tol=0.0)
            if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
                raise ValueError("failed to align a dataset during join")
            # Keep source dtypes as views. NumPy promotes only the bounded block
            # expressions below; casting an entire lower-precision grid here
            # would reintroduce two input-sized peak allocations.
            incoming_power = np.asarray(grid.rcs_power)
            incoming_phase = np.asarray(grid.rcs_phase)
            # np.ix_ over all four complete axes materialises full input-sized
            # copies. Tile every axis so each advanced-index block stays
            # bounded without degenerating into a Python loop per scalar cell.
            pol_block = max(1, min(len(p_idx), _JOIN_MERGE_BLOCK_CELLS))
            freq_block = max(
                1,
                min(len(f_idx), _JOIN_MERGE_BLOCK_CELLS // pol_block),
            )
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS // (pol_block * freq_block),
            )
            elev_block = max(1, min(len(el_idx), remaining))
            remaining = max(
                1,
                _JOIN_MERGE_BLOCK_CELLS
                // (pol_block * freq_block * elev_block),
            )
            az_block = max(1, min(len(az_idx), remaining))
            for a_start in range(0, len(az_idx), az_block):
                a_stop = min(a_start + az_block, len(az_idx))
                union_a = az_idx[a_start:a_stop]
                for e_start in range(0, len(el_idx), elev_block):
                    e_stop = min(e_start + elev_block, len(el_idx))
                    union_e = el_idx[e_start:e_stop]
                    for f_start in range(0, len(f_idx), freq_block):
                        f_stop = min(f_start + freq_block, len(f_idx))
                        union_f = f_idx[f_start:f_stop]
                        for p_start in range(0, len(p_idx), pol_block):
                            p_stop = min(p_start + pol_block, len(p_idx))
                            union_p = p_idx[p_start:p_stop]
                            target = np.ix_(union_a, union_e, union_f, union_p)
                            existing_power = joined_power[target]
                            existing_phase = joined_phase[target]
                            block_selection = (
                                slice(a_start, a_stop),
                                slice(e_start, e_stop),
                                slice(f_start, f_stop),
                                slice(p_start, p_stop),
                            )
                            block_power = incoming_power[block_selection]
                            block_phase = incoming_phase[block_selection]

                            both = np.isfinite(existing_power) & np.isfinite(block_power)
                            power_conflict = both & ~np.isclose(
                                existing_power, block_power, rtol=1e-6, atol=1e-12
                            )
                            both_phase = (
                                both
                                & np.isfinite(existing_phase)
                                & np.isfinite(block_phase)
                            )
                            phase_delta = np.abs(
                                np.angle(np.exp(1j * (existing_phase - block_phase)))
                            )
                            phase_conflict = both_phase & (phase_delta > 1e-5)
                            if overlap == "error" and (
                                np.any(power_conflict) or np.any(phase_conflict)
                            ):
                                raise ValueError(
                                    "conflicting finite samples overlap during join"
                                )

                            if overlap == "last":
                                take_power = np.isfinite(block_power)
                                take_phase = take_power
                            else:
                                take_power = (
                                    ~np.isfinite(existing_power)
                                    & np.isfinite(block_power)
                                )
                                # Equal finite power with a missing earlier
                                # phase is complementary data, not a replacement.
                                fill_phase = (
                                    both
                                    & ~np.isfinite(existing_phase)
                                    & np.isfinite(block_phase)
                                )
                                take_phase = take_power | fill_phase
                            existing_power[take_power] = block_power[take_power]
                            existing_phase[take_phase] = block_phase[take_phase]
                            joined_power[target] = existing_power
                            joined_phase[target] = existing_phase

        # Inputs are normally already clean, but RcsGrid arrays are public and
        # may have been mutated. Preserve constructor sanitation in place, then
        # transfer ownership of these newly allocated arrays without copying.
        finite = np.empty(shape, dtype=bool)
        np.isfinite(joined_power, out=finite)
        np.maximum(joined_power, 0.0, out=joined_power, where=finite)
        np.logical_not(finite, out=finite)
        joined_power[finite] = np.nan
        np.isfinite(joined_phase, out=finite)
        np.logical_not(finite, out=finite)
        joined_phase[finite] = np.nan
        np.isfinite(joined_power, out=finite)
        np.logical_not(finite, out=finite)
        joined_phase[finite] = np.nan
        del finite

        return cls(
            az_union,
            el_union,
            f_union,
            p_union,
            rcs_power=joined_power,
            rcs_phase=joined_phase,
            rcs_domain="power_phase",
            source_path=ref.source_path,
            history=ref.history,
            units=dict(ref.units),
            extra={
                "phase_reference": ref.extra["phase_reference"]
                for _ in (0,)
                if "phase_reference" in ref.extra
            },
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def overlap_many(cls, *grids, tol=1e-6):
        """Return one cropped dataset per input, all on common overlap axes.

        Every input participates equally in one all-selected intersection; no
        input is treated as a reference grid.  Numeric matching and the output
        axis coordinates are therefore independent of input selection order.

        Overlap is enforced cell-wise: if any input is missing data (NaN) at a
        given (az, el, freq, pol) cell, that cell is set to NaN in every output.
        Axis values whose entire slice becomes NaN after this intersection are
        dropped — so e.g. a frequency that one dataset lacks for HH but all
        datasets have for VV will stay on the axis, with HH masked to NaN.
        """
        grids = cls._ensure_grids(grids)
        if len(grids) == 1:
            return [grids[0]]
        for grid in grids[1:]:
            grids[0]._assert_physical_metadata_compatible(grid)

        az_common, az_indices = cls._common_axis_alignment(
            [grid.azimuths for grid in grids], tol=tol
        )
        el_common, el_indices = cls._common_axis_alignment(
            [grid.elevations for grid in grids], tol=tol
        )
        f_common, f_indices = cls._common_axis_alignment(
            [grid.frequencies for grid in grids], tol=tol
        )
        p_common, p_indices = cls._common_axis_alignment(
            [grid.polarizations for grid in grids], tol=0.0
        )

        if (
            az_common.size == 0
            or el_common.size == 0
            or f_common.size == 0
            or p_common.size == 0
        ):
            raise ValueError("no overlap across one or more axes")

        aligned_power = []
        aligned_phase = []
        for grid_idx, grid in enumerate(grids):
            az_idx = az_indices[grid_idx]
            el_idx = el_indices[grid_idx]
            f_idx = f_indices[grid_idx]
            p_idx = p_indices[grid_idx]
            aligned_power.append(grid.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)].copy())
            aligned_phase.append(grid.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)].copy())

        missing_any = np.zeros(aligned_power[0].shape, dtype=bool)
        for power in aligned_power:
            missing_any |= ~np.isfinite(power)
        for power, phase in zip(aligned_power, aligned_phase):
            power[missing_any] = np.nan
            phase[missing_any] = np.nan

        finite = ~missing_any
        az_keep = finite.any(axis=(1, 2, 3))
        el_keep = finite.any(axis=(0, 2, 3))
        f_keep = finite.any(axis=(0, 1, 3))
        p_keep = finite.any(axis=(0, 1, 2))

        if not (az_keep.any() and el_keep.any() and f_keep.any() and p_keep.any()):
            raise ValueError("no overlap across one or more axes")

        az_sel = np.where(az_keep)[0]
        el_sel = np.where(el_keep)[0]
        f_sel = np.where(f_keep)[0]
        p_sel = np.where(p_keep)[0]
        az_common = az_common[az_sel]
        el_common = el_common[el_sel]
        f_common = f_common[f_sel]
        p_common = p_common[p_sel]

        overlap_grids = []
        for grid, power, phase in zip(grids, aligned_power, aligned_phase):
            overlap_grids.append(
                cls(
                    az_common,
                    el_common,
                    f_common,
                    p_common,
                    rcs_power=power[np.ix_(az_sel, el_sel, f_sel, p_sel)],
                    rcs_phase=phase[np.ix_(az_sel, el_sel, f_sel, p_sel)],
                    rcs_domain="power_phase",
                    source_path=grid.source_path,
                    history=grid.history,
                    units=dict(grid.units),
                    extra={
                        "phase_reference": grid.extra["phase_reference"]
                        for _ in (0,)
                        if "phase_reference" in grid.extra
                    },
                )
            )

        return overlap_grids

    def statistics_dataset(
        self,
        statistic="mean",
        axes=("azimuth", "elevation", "frequency"),
        *,
        domain="magnitude",
        percentile=50.0,
        broadcast_reduced=False,
    ):
        """Compute a statistic over selected axes and return a dataset."""
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2, "polarization": 3}
        axis_alias = {
            "azimuths": "azimuth",
            "elevations": "elevation",
            "frequencies": "frequency",
            "polarizations": "polarization",
            "az": "azimuth",
            "el": "elevation",
            "freq": "frequency",
            "pol": "polarization",
        }

        axes_list = self._as_list(axes)
        if axes_list is None:
            raise ValueError("axes must include at least one axis")
        reduce_axes = []
        for axis_name in axes_list:
            key = str(axis_name).strip().lower()
            key = axis_alias.get(key, key)
            if key not in axis_map:
                raise ValueError(f"unknown axis: {axis_name}")
            idx = axis_map[key]
            if idx not in reduce_axes:
                reduce_axes.append(idx)
        if not reduce_axes:
            raise ValueError("axes must include at least one axis")
        reduce_axes = tuple(sorted(reduce_axes))

        if domain == "complex":
            values = self.rcs
        elif domain == "magnitude":
            values = self.rcs_power
        elif domain in ("db", "dbsm"):
            values = self.linear_to_dbsm(self.rcs_power)
        elif domain == "dbke":
            # Conversion helpers accept values in the dataset's declared unit.
            # Passing preconverted Hz here caused a second unit conversion.
            freq_grid = np.asarray(self.frequencies, dtype=float).reshape(1, 1, -1, 1)
            values = self.linear_to_dbke(self.rcs_power, freq_grid)
        else:
            raise ValueError("domain must be 'complex', 'magnitude', 'dbsm', or 'dbke'")

        stat_key = str(statistic).strip().lower()
        if stat_key.startswith("p") and stat_key[1:].replace(".", "", 1).isdigit():
            percentile = float(stat_key[1:])
            stat_key = "percentile"

        if domain == "complex" and stat_key == "percentile":
            raise ValueError("percentile on complex values is not supported; use magnitude, dbsm, or dbke domain")

        if stat_key == "mean":
            reduced = np.nanmean(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "median":
            reduced = np.nanmedian(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "min":
            reduced = np.nanmin(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "max":
            reduced = np.nanmax(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "std":
            reduced = np.nanstd(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "percentile":
            reduced = np.nanpercentile(values, float(percentile), axis=reduce_axes, keepdims=True)
        else:
            raise ValueError(
                "statistic must be mean, median, min, max, std, percentile, or pXX (for percentile XX)"
            )

        axis_values = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        if broadcast_reduced:
            # Repeat the reduced result across each reduced axis so the output
            # keeps original axis lengths for downstream plotting.
            reduced = np.broadcast_to(reduced, values.shape).copy()
        else:
            for axis_idx in reduce_axes:
                original = axis_values[axis_idx]
                if axis_idx == 3:
                    axis_values[axis_idx] = np.asarray(["ALL"])
                else:
                    numeric = np.asarray(original, dtype=float)
                    rep = float(np.nanmean(numeric)) if numeric.size else 0.0
                    axis_values[axis_idx] = np.asarray([rep], dtype=float)

        if domain == "complex":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                reduced,
                rcs_domain="power_phase",
            )
        if domain == "magnitude":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                rcs_power=np.asarray(reduced, dtype=self.rcs_power.dtype),
                rcs_phase=np.full(reduced.shape, np.nan, dtype=self.rcs_phase.dtype),
                rcs_domain="power_phase",
            )
        # db domain: compute in a log domain, then store as linear so future conversion reproduces the reduced values.
        if domain == "dbke":
            freq_grid = np.asarray(axis_values[2], dtype=float).reshape(1, 1, -1, 1)
            reduced_linear = np.asarray(
                self.dbke_to_linear(np.asarray(reduced, dtype=float), freq_grid),
                dtype=self.rcs_power.dtype,
            )
        else:
            reduced_linear = np.asarray(
                10.0 ** (np.asarray(reduced, dtype=float) / 10.0),
                dtype=self.rcs_power.dtype,
            )
        return self._new_grid(
            axis_values[0],
            axis_values[1],
            axis_values[2],
            axis_values[3],
            rcs_power=reduced_linear,
            rcs_phase=np.full(reduced_linear.shape, np.nan, dtype=self.rcs_phase.dtype),
            rcs_domain="power_phase",
        )

    def _index_for_value(self, axis, value, tol=0.0):
        """Find the first index of a value on an axis.

        Args:
            axis: 1D array to search.
            value: Value to find.
            tol: Absolute tolerance for numeric matching.

        Returns:
            Integer index of the first match.

        Raises:
            ValueError: if no match is found.
        """
        axis_arr = np.asarray(axis)
        if tol > 0.0:
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
        p_idx = self._index_for_value(self.polarizations, polarization, tol=tol)
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
        p_idx = self._index_for_value(self.polarizations, polarization, tol=tol)
        return self.linear_to_dbsm(self.rcs_power[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbke_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBke."""
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(self.polarizations, polarization, tol=tol)
        return self.linear_to_dbke(self.rcs_power[az_idx, el_idx, f_idx, p_idx], self.frequencies[f_idx], eps=eps)

    # keys this class fully models and always rewrites itself.  rcs_domain and
    # power_domain are deliberately NOT here: a producer may tag a file with a
    # domain word outside this class's 3-value vocabulary (the Claude21 solver
    # writes rcs_domain='delta' / power_domain='delta_amp_sq', and routes on it),
    # so those tags are captured in `extra` and re-emitted verbatim by save().
    _RESERVED_KEYS = ("azimuths", "elevations", "frequencies", "polarizations",
                      "rcs_power", "rcs_phase", "source_path", "history", "units")

    def _extra_to_write(self):
        """Passthrough keys to re-emit, minus anything whose shape no longer fits.

        An array sized to the grid (e.g. rcs_amp_real) is only still valid if the
        grid has not been cropped, joined or interpolated since it was read, so a
        mismatched shape is dropped instead of written stale.  Scalars/strings
        (provenance flags, domain tags) always survive.
        """
        expected = (len(self.azimuths), len(self.elevations),
                    len(self.frequencies), len(self.polarizations))
        out = {}
        for key, value in self.extra.items():
            if key in self._RESERVED_KEYS:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 2 and arr.shape[:4] != expected:
                continue
            out[key] = value
        return out

    def save(self, path):
        """Save the grid to a .grim (npz) file.

        Passthrough metadata from ``extra`` is written first (so the grid's own
        axes and samples always win on a name clash), which is what lets a file
        carrying a raw complex amplitude survive a load/save round-trip.
        The archive is fully written and flushed to a same-directory staging
        file before ``os.replace`` publishes it, so a failed save leaves an
        existing artifact intact.

        Args:
            path: Output path, with or without .grim.

        Returns:
            The actual path written (always ends with .grim).
        """
        path = os.fspath(path)
        if not path.casefold().endswith(".grim"):
            path = f"{path}.grim"
        directory = os.path.dirname(os.path.abspath(path)) or os.curdir
        fd, stage_path = tempfile.mkstemp(
            prefix=".grim-write-",
            suffix=".staging",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                fd = -1
                units_payload = json.dumps(self.units) if self.units else ""
                payload = dict(self._extra_to_write())          # passthrough first
                payload.update(
                    azimuths=self.azimuths,
                    elevations=self.elevations,
                    frequencies=self.frequencies,
                    polarizations=self.polarizations,
                    rcs_power=self.rcs_power,
                    rcs_phase=self.rcs_phase,
                    rcs_domain="power_phase",
                    power_domain=self.power_domain,
                    source_path=self.source_path if self.source_path is not None else "",
                    history=self.history if self.history is not None else "",
                    units=units_payload,
                )
                # a source domain tag we could not represent wins back its own slot:
                # cropping or joining does not change what the samples MEAN, and a
                # consumer may route on it (see _RESERVED_KEYS)
                for tag in ("rcs_domain", "power_domain"):
                    if tag in self.extra:
                        payload[tag] = self.extra[tag]
                np.savez(f, **payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(stage_path, path)
        finally:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(stage_path):
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass
        return path

    @classmethod
    def load(
        cls,
        path,
        mmap_mode: str | None = None,
        *,
        allow_legacy_pickle: bool = False,
    ):
        """Load a grid from a .grim (npz) file.

        Args:
            path: Input path, with or without .grim.
            mmap_mode: Retained for API compatibility. ``.npz`` members cannot
                be memory-mapped; a warning is emitted when this is supplied.
            allow_legacy_pickle: Explicitly opt in to legacy object-array files.
                Never enable this for an untrusted file.

        Returns:
            RcsGrid instance loaded from disk.
        """
        path = os.fspath(path)
        if not path.casefold().endswith(".grim"):
            path = f"{path}.grim"
        if mmap_mode is not None:
            warnings.warn(
                "mmap_mode has no effect for .grim/.npz archives; arrays are loaded eagerly",
                RuntimeWarning,
                stacklevel=2,
            )
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=bool(allow_legacy_pickle))

            units = {}
            if "units" in data:
                raw_units = data["units"]
                if isinstance(raw_units, np.ndarray):
                    raw_units = raw_units.item()
                if isinstance(raw_units, bytes):
                    raw_units = raw_units.decode("utf-8")
                if isinstance(raw_units, str) and raw_units:
                    try:
                        units = json.loads(raw_units)
                    except json.JSONDecodeError:
                        units = {}
                elif isinstance(raw_units, dict):
                    units = raw_units

            source_path_raw = data["source_path"].item() if "source_path" in data else None
            source_path = source_path_raw if source_path_raw else None
            history_raw = data["history"].item() if "history" in data else None
            history = history_raw if history_raw else None
            required = ("azimuths", "elevations", "frequencies", "polarizations", "rcs_power", "rcs_phase")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(
                    f"{path} is not a supported .grim file (missing keys: {', '.join(missing)})"
                )

            # keys this class does not model (e.g. the raw complex amplitude and
            # provenance flags written by the Claude21 solver exports) ride along
            # in `extra` so save() can put them back -- see _extra_to_write
            extra = {k: data[k] for k in getattr(data, "files", [])
                     if k not in cls._RESERVED_KEYS}

            return cls(
                data["azimuths"],
                data["elevations"],
                data["frequencies"],
                data["polarizations"],
                rcs_power=data["rcs_power"],
                rcs_phase=data["rcs_phase"],
                rcs_domain="power_phase",
                source_path=source_path,
                history=history,
                units=units,
                extra=extra,
            )

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

        file_name = os.path.basename(str(path))
        stem_upper = os.path.splitext(file_name)[0].upper()
        pol_match = re.search(r"(?<![A-Z0-9])(HH|VV)(?![A-Z0-9])", stem_upper)
        if pol_match is not None:
            pol_label = pol_match.group(1)
        else:
            idx_hh = stem_upper.find("HH")
            idx_vv = stem_upper.find("VV")
            if idx_hh < 0 and idx_vv < 0:
                pol_label = "NA"
            elif idx_hh >= 0 and (idx_vv < 0 or idx_hh <= idx_vv):
                pol_label = "HH"
            else:
                pol_label = "VV"

        records: list[tuple[float, float, float, float]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(
                        f"line {line_no}: expected 4 columns "
                        "(frequency_ghz azimuth_deg rcs_dbke phase_deg)"
                    )
                try:
                    freq_ghz = float(parts[0])
                    azimuth_deg = float(parts[1])
                    rcs_dbke = float(parts[2])
                    phase_deg = float(parts[3])
                except ValueError as exc:
                    raise ValueError(f"line {line_no}: invalid numeric value ({exc})") from exc

                if not (np.isfinite(freq_ghz) and np.isfinite(azimuth_deg)):
                    continue
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
        power = np.full(shape, np.nan, dtype=np.float32)
        phase = np.full(shape, np.nan, dtype=np.float32)

        for freq_ghz, azimuth_deg, rcs_dbke, phase_deg in records:
            ai = az_idx[float(azimuth_deg)]
            fi = f_idx[float(freq_ghz)]
            if np.isfinite(rcs_dbke):
                lambda_m = C0 / (float(freq_ghz) * 1.0e9) if float(freq_ghz) > 0.0 else float("nan")
                sigma_2d = (lambda_m / (2.0 * np.pi)) * (10.0 ** (rcs_dbke / 10.0)) if np.isfinite(lambda_m) else float("nan")
                power[ai, 0, fi, 0] = np.float32(sigma_2d)
            else:
                power[ai, 0, fi, 0] = np.nan
            if np.isfinite(phase_deg):
                phase[ai, 0, fi, 0] = np.float32(np.deg2rad(phase_deg))
            else:
                phase[ai, 0, fi, 0] = np.nan

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
    def load_ss(cls, path):
        """Load an Xpatch ``.ss`` signature file into an RcsGrid.

        Delegates the binary parse to :mod:`read_ss` (a pure-Python port of the
        MATLAB ``ssread.m`` / ``xpheaders.m`` readers), then maps its output
        onto the grid:

            - each signal is one (azimuth, elevation) look;
            - the four polarizations VV/VH/HV/HH become the polarization axis;
            - the complex scattering samples become ``rcs`` (power = |c|**2);
            - frequencies (stored in Hz) are presented in GHz.
        """
        import read_ss

        data = read_ss.read_ss(path, verbose=False)

        az = np.round(np.asarray(data["az"], dtype=float), 4)
        el = np.round(np.asarray(data["el"], dtype=float), 4)
        freq = np.asarray(data["freq"], dtype=float)
        # Xpatch stores frequency in Hz; present it as GHz (the grid's unit).
        if freq.size and np.nanmedian(np.abs(freq)) >= 1.0e6:
            freq = freq / 1.0e9

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

        az_axis = np.asarray(sorted(set(az.tolist())), dtype=float)
        el_axis = np.asarray(sorted(set(el.tolist())), dtype=float)
        pols = np.asarray(["VV", "VH", "HV", "HH"], dtype=object)
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

        grid = np.full(
            (len(az_axis), len(el_axis), n_freq, len(pols)),
            np.nan + 1j * np.nan,
            dtype=np.complex64,
        )
        for s in range(n_sig):
            ai = az_index[float(az[s])]
            ei = el_index[float(el[s])]
            for pj, samples in enumerate(pol_data):
                grid[ai, ei, :, pj] = np.asarray(samples[s], dtype=np.complex64)

        if not np.isfinite(grid).any():
            raise ValueError("SS parsed, but no finite scattering samples were found")

        extra = {}
        if int(data.get("imono", 1)) == 2:
            if data.get("angle_source") == "observation":
                extra["fixed_incident_azimuth_deg"] = float(np.asarray(data["az_inc"])[0])
                extra["fixed_incident_elevation_deg"] = float(np.asarray(data["el_inc"])[0])
            else:
                extra["fixed_observation_azimuth_deg"] = float(np.asarray(data["az_obs"])[0])
                extra["fixed_observation_elevation_deg"] = float(np.asarray(data["el_obs"])[0])

        return cls(
            az_axis,
            el_axis,
            freq,
            pols,
            rcs=grid,
            rcs_domain="complex_amplitude",
            source_path=path,
            history=(f"Loaded Xpatch .ss ({n_sig} signals, {n_freq} freqs, "
                     f"{data.get('angle_source', 'incident')} angles, "
                     f"imono={data.get('imono', '?')}): {path}"),
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
            extra=extra,
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

    @classmethod
    def read_CST(cls, path):
        """Read a supported CST RCS table into a physically tagged grid.

        Two schemas are recognized:

        * CST's wide spherical table (frequency/theta/phi and one magnitude /
          phase pair per spherical polarization component).
        * The legacy ``.cst_data`` flat table documented by ``Read_CST.m``
          (elevation/azimuth/frequency/polarity/magnitude/phase/IQ).

        Standard CST theta is a colatitude, so the wide form is converted to
        GRIM elevation with ``elevation = 90 - theta``.  Both forms use GRIM's
        canonical azimuth interval ``[-180, 180)``.
        """

        rows = _read_cst_delimited_rows(path)
        if not rows:
            raise ValueError("CST table is empty")

        def _flat_key(cell_value):
            compact = _cst_compact_header(cell_value)
            if compact.startswith("elevation") and (
                "deg" in compact or "degree" in compact
            ):
                return "elevation"
            if compact.startswith("azimuth") and (
                "deg" in compact or "degree" in compact
            ):
                return "azimuth"
            if compact in {"pol", "polarity", "polarization"}:
                return "polarization"
            if compact in {"iq", "complexiq", "complexsample", "complexamplitude"}:
                return "iq"
            if _cst_frequency_unit(cell_value) is not None:
                return "frequency"
            if "magnitude" in compact and (
                "dbsm" in compact or "dbm2" in compact
            ):
                return "magnitude_dbsm"
            if compact.startswith("rcs") and (
                "dbsm" in compact or "dbm2" in compact
            ):
                return "magnitude_dbsm"
            if "phase" in compact and (
                "deg" in compact or "degree" in compact
            ):
                return "phase_deg"
            return None

        required = {"elevation", "azimuth", "frequency", "polarization"}
        for header_idx, row in enumerate(rows):
            mapped = {}
            tokens = {}
            for column_idx, cell in enumerate(row):
                key = _flat_key(cell)
                if key is not None and key not in mapped:
                    mapped[key] = column_idx
                    tokens[key] = str(cell)
            if required.issubset(mapped) and (
                "magnitude_dbsm" in mapped or "iq" in mapped
            ):
                return cls._read_cst_flat_rows(
                    path, rows, header_idx, mapped, tokens
                )

        if str(path).lower().endswith(".cst_data"):
            raise ValueError(
                "Could not find the .cst_data header. Need elevation, azimuth, "
                "frequency, polarity, and magnitude(dBsm) and/or IQ columns."
            )
        return cls._read_cst_theta_phi_csv(path, rows=rows)

    @classmethod
    def read_SENTRi(cls, path):
        """Read either RCS table schema emitted by CREATE-RF SENTRi.

        The supplied team ``READ_SENTRi.m`` documents two header families:

        * compact ``freq_MHz`` / ``theta_deg`` / ``rcs_pp_dBsm`` columns;
        * descriptive ``Frequency`` / ``Theta`` /
          ``RCSPhiScat_PhiInc`` columns.

        SENTRi's reported ``Theta`` is stored directly as GRIM elevation.
        Its reported E-field phase is stored with its original sign, so each
        sample is reconstructed as
        ``10**(dBsm/20) * exp(+1j*deg2rad(phase_deg))``.  These format-specific
        rules are deliberately separate from :meth:`read_CST`.
        """

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

        # CREATE-RF exports may put parameter names on row 1 and their units
        # on row 2.  Preserve compatibility with older header+data tables, but
        # when the first nonblank post-header row is nonnumeric, require it to
        # be the complete, physically consistent units row rather than letting
        # it fall into the numeric parser.
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
            if theta_deg < -1.0e-9 or theta_deg > 180.0 + 1.0e-9:
                raise ValueError(
                    f"line {row_idx}: SENTRi theta must be in [0, 180] deg"
                )
            elevation_deg = float(theta_deg)
            azimuth_deg = _wrap_cst_azimuth_deg(
                _number(row, "phi", row_idx)
            )

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
                if key in seen:
                    prior_line, prior_power, prior_phase = seen[key]
                    if _cst_samples_equivalent(
                        prior_power, prior_phase, power, phase
                    ):
                        continue
                    raise ValueError(
                        f"line {row_idx}: conflicting duplicate SENTRi sample "
                        f"after azimuth wrapping; first defined on line {prior_line}"
                    )
                seen[key] = (row_idx, power, phase)
                records.append(
                    (
                        azimuth_deg, elevation_deg, float(frequency_ghz),
                        polarization, power, phase,
                    )
                )

        if not records:
            raise ValueError("SENTRi table contains no data rows")

        azimuths = np.asarray(sorted({row[0] for row in records}), dtype=float)
        elevations = np.asarray(sorted({row[1] for row in records}), dtype=float)
        frequencies = np.asarray(sorted({row[2] for row in records}), dtype=float)
        polarizations = np.asarray([spec[0] for spec in channel_specs])
        shape = (
            len(azimuths), len(elevations), len(frequencies), len(polarizations)
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

        mapping = (
            "elevation=theta; phi wrapped to [-180, 180); "
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
            },
            extra={
                "source_format": f"SENTRi {schema_name} RCS table",
                "sentri_coordinate_mapping": "elevation=theta; azimuth=wrapped phi",
                "sentri_polarization_mapping": (
                    "VV=tt/theta-theta; HV=pt/phi-theta; "
                    "VH=tp/theta-phi; HH=pp/phi-phi"
                ),
                "sentri_phase_mapping": (
                    "GRIM complex amplitude = 10^(dBsm/20) "
                    "* exp(+j*deg2rad(reported_phase_deg))"
                ),
                "sentri_units_row_present": bool(has_units_row),
            },
        )

    @classmethod
    def has_SENTRi_signature(cls, path):
        """Return whether a delimited file has a recognizable SENTRi family header.

        Dispatchers use this before trying legacy/fallback readers.  Once the
        vendor signature is present, malformed SENTRi data must fail as SENTRi
        instead of being silently reinterpreted as an unrelated numeric TXT.
        """

        # Probe only the header region; full files can be large angular/frequency
        # sweeps and read_SENTRi() will perform the authoritative parse once.
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

    @classmethod
    def load_theta_phi_csv(cls, path):
        """Compatibility name for :meth:`read_CST`."""

        return cls.read_CST(path)

    @classmethod
    def _read_cst_flat_rows(cls, path, rows, header_idx, col_idx, header_tokens):
        """Parse the legacy row-per-polarization ``.cst_data`` schema."""

        def _cell(row, key):
            idx = col_idx.get(key, -1)
            if idx < 0 or idx >= len(row):
                return ""
            return str(row[idx]).strip()

        def _required_float(row, key, line_no):
            text = _cell(row, key)
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_no}: invalid {key} value {text!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"line {line_no}: {key} must be finite")
            return float(value)

        def _optional_float(row, key, line_no):
            text = _cell(row, key)
            if not text:
                return None
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_no}: invalid {key} value {text!r}"
                ) from exc
            if np.isnan(value):
                return None
            if np.isposinf(value):
                raise ValueError(f"line {line_no}: {key} cannot be +Inf")
            return float(value)

        raw_records = []
        pol_order = []
        iq_validated = 0
        iq_only = 0
        iq_unparsed = 0

        for row_idx, row in enumerate(rows[header_idx + 1 :], start=header_idx + 2):
            if not row or all(not str(value).strip() for value in row):
                continue

            elevation = _required_float(row, "elevation", row_idx)
            azimuth = _wrap_cst_azimuth_deg(
                _required_float(row, "azimuth", row_idx)
            )
            raw_frequency = _required_float(row, "frequency", row_idx)
            if raw_frequency <= 0.0:
                raise ValueError(f"line {row_idx}: frequency must be positive")
            polarization = _cell(row, "polarization").upper()
            if not polarization:
                raise ValueError(f"line {row_idx}: polarity is blank")

            magnitude_dbsm = _optional_float(
                row, "magnitude_dbsm", row_idx
            ) if "magnitude_dbsm" in col_idx else None
            phase_deg = _optional_float(
                row, "phase_deg", row_idx
            ) if "phase_deg" in col_idx else None

            iq_text = _cell(row, "iq") if "iq" in col_idx else ""
            iq_value = None
            if iq_text:
                try:
                    iq_value = _parse_cst_iq(iq_text)
                except ValueError as exc:
                    if magnitude_dbsm is None:
                        raise ValueError(f"line {row_idx}: {exc}") from exc
                    iq_unparsed += 1

            if magnitude_dbsm is None and iq_value is None:
                raise ValueError(
                    f"line {row_idx}: need a finite magnitude(dBsm) or parsable IQ sample"
                )

            if magnitude_dbsm is None:
                power = _cst_iq_to_power(
                    iq_value, context=f"line {row_idx} IQ"
                )
            else:
                power = _cst_dbsm_to_power(
                    magnitude_dbsm, context=f"line {row_idx} Magnitude(dBsm)"
                )

            if phase_deg is not None and not np.isfinite(phase_deg):
                raise ValueError(f"line {row_idx}: phase_deg must be finite")
            phase = (
                float(np.deg2rad(phase_deg))
                if phase_deg is not None
                else float(np.angle(iq_value)) if iq_value is not None
                else float("nan")
            )

            if iq_value is not None and magnitude_dbsm is not None:
                iq_power = _cst_iq_to_power(
                    iq_value, context=f"line {row_idx} IQ"
                )
                if power == 0.0:
                    magnitude_matches = iq_power <= 1.0e-20
                elif iq_power == 0.0:
                    magnitude_matches = False
                else:
                    iq_dbsm = 10.0 * np.log10(iq_power)
                    magnitude_matches = abs(iq_dbsm - magnitude_dbsm) <= 0.05
                if not magnitude_matches:
                    raise ValueError(
                        f"line {row_idx}: IQ magnitude disagrees with "
                        "Magnitude(dBsm) by more than 0.05 dB"
                    )

            if (
                iq_value is not None
                and phase_deg is not None
                and abs(iq_value) > 1.0e-15
            ):
                phase_error = np.angle(np.exp(1j * (np.angle(iq_value) - phase)))
                if abs(float(np.rad2deg(phase_error))) > 0.5:
                    raise ValueError(
                        f"line {row_idx}: IQ phase disagrees with Phase(deg) "
                        "by more than 0.5 deg"
                    )

            if iq_value is not None:
                # The team's Read_CST workflow treats IQ as the authoritative
                # coherent field.  Magnitude/phase columns are rounded
                # redundant values: validate them above, but do not replace IQ
                # precision with those display columns.
                power = _cst_iq_to_power(
                    iq_value, context=f"line {row_idx} IQ"
                )
                phase = float(np.angle(iq_value))
                if magnitude_dbsm is None and phase_deg is None:
                    iq_only += 1
                else:
                    iq_validated += 1
            if polarization not in pol_order:
                pol_order.append(polarization)
            raw_records.append(
                (row_idx, azimuth, elevation, raw_frequency, polarization, power, phase)
            )

        if not raw_records:
            raise ValueError("CST flat table contains no data rows")

        frequency_scale, _ = _cst_frequency_scale_to_ghz(
            header_tokens.get("frequency", "")
        )

        records = []
        seen = {}
        for row_idx, azimuth, elevation, raw_frequency, polarization, power, phase in raw_records:
            frequency = float(raw_frequency * frequency_scale)
            key = (azimuth, elevation, frequency, polarization)
            if key in seen:
                prior_line, prior_power, prior_phase = seen[key]
                if _cst_samples_equivalent(
                    prior_power, prior_phase, power, phase
                ):
                    continue
                raise ValueError(
                    f"line {row_idx}: conflicting duplicate CST sample after "
                    f"azimuth wrapping; first defined on line {prior_line}"
                )
            seen[key] = (row_idx, power, phase)
            records.append(
                (azimuth, elevation, frequency, polarization, power, phase)
            )

        azimuths = np.asarray(sorted({record[0] for record in records}), dtype=float)
        elevations = np.asarray(sorted({record[1] for record in records}), dtype=float)
        frequencies = np.asarray(sorted({record[2] for record in records}), dtype=float)
        polarizations = np.asarray(pol_order, dtype=object)
        shape = (
            len(azimuths), len(elevations), len(frequencies), len(polarizations)
        )
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)
        az_index = {value: index for index, value in enumerate(azimuths.tolist())}
        el_index = {value: index for index, value in enumerate(elevations.tolist())}
        freq_index = {value: index for index, value in enumerate(frequencies.tolist())}
        pol_index = {str(value): index for index, value in enumerate(polarizations.tolist())}

        for azimuth, elevation, frequency, polarization, sample_power, sample_phase in records:
            index = (
                az_index[azimuth], el_index[elevation], freq_index[frequency],
                pol_index[polarization],
            )
            power[index] = sample_power
            phase[index] = sample_phase

        iq_summary = (
            f"IQ validated={iq_validated}, IQ-only={iq_only}, "
            f"IQ-unparsed fallback={iq_unparsed}"
        )
        return cls(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=(
                f"Loaded CST flat cst_data; explicit elevation; azimuth wrapped "
                f"to [-180, 180); {iq_summary}: {path}"
            ),
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
            extra={
                "source_format": "CST flat cst_data",
                "cst_angle_mapping": (
                    "explicit elevation; azimuth wrapped to [-180, 180)"
                ),
                "cst_polarization_mapping": "labels supplied by Polarity column",
                "cst_iq_rows_validated": iq_validated,
                "cst_iq_only_rows": iq_only,
                "cst_iq_unparsed_fallback_rows": iq_unparsed,
            },
        )

    @classmethod
    def _read_cst_theta_phi_csv(cls, path, *, rows=None):
        """Load a theta/phi scattering CSV into an RcsGrid.

        Expected layout:
            - Two header rows total (or any leading metadata rows), with one row
              containing column names like:
              frequency(hz), theta(deg), phi(deg),
              rcs theta-theta(dbsm), rcs phi-theta(dbsm),
              rcs theta-phi(dbsm), rcs phi-phi,
              phase theta-theta(...), phase phi-theta(...),
              phase theta-phi(...), phase phi-phi(...)

        Conventions applied:
            - phi(deg), wrapped to [-180, 180), -> azimuth axis
            - standard CST theta colatitude -> elevation = 90 - theta
            - theta -> V, phi -> H
              rcs theta-theta -> VV
              rcs phi-theta   -> HV
              rcs theta-phi   -> VH
              rcs phi-phi     -> HH
            - RCS columns are interpreted as dBsm and converted to linear power.
            - Phase columns are interpreted as degrees and converted to radians.
        """

        def _norm(text: str) -> str:
            s = str(text).strip().lower()
            for ch in (" ", "_", "\t"):
                s = s.replace(ch, "")
            return s

        def _infer_freq_scale_to_ghz(freq_header_token: str) -> tuple[float, str]:
            scale, unit = _cst_frequency_scale_to_ghz(freq_header_token)
            labels = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}
            return scale, labels[unit]

        alias_to_key = {
            "frequency(hz)": "frequency",
            "frequencyhz": "frequency",
            "frequency(ghz)": "frequency",
            "frequencyghz": "frequency",
            "frequency(mhz)": "frequency",
            "frequencymhz": "frequency",
            "frequency(khz)": "frequency",
            "frequencykhz": "frequency",
            "frequency": "frequency",
            "theta(deg)": "theta_deg",
            "phi(deg)": "phi_deg",
            "rcstheta-theta(dbsm)": "rcs_vv_dbsm",
            "rcstheta-thetadbsm": "rcs_vv_dbsm",
            "rcstheta-theta(dbm^2)": "rcs_vv_dbsm",
            "rcstheta-thetadbm2": "rcs_vv_dbsm",
            "rcsphi-theta(dbsm)": "rcs_hv_dbsm",
            "rcsphi-thetadbsm": "rcs_hv_dbsm",
            "rcsphi-theta(dbm^2)": "rcs_hv_dbsm",
            "rcsphi-thetadbm2": "rcs_hv_dbsm",
            "rcstheta-phi(dbsm)": "rcs_vh_dbsm",
            "rcstheta-phidbsm": "rcs_vh_dbsm",
            "rcstheta-phi(dbm^2)": "rcs_vh_dbsm",
            "rcstheta-phidbm2": "rcs_vh_dbsm",
            "rcsphi-phi(dbsm)": "rcs_hh_dbsm",
            "rcsphi-phidbsm": "rcs_hh_dbsm",
            "rcsphi-phi(dbm^2)": "rcs_hh_dbsm",
            "rcsphi-phidbm2": "rcs_hh_dbsm",
            "phasetheta-theta(deg)": "phase_vv_deg",
            "phasephi-theta(deg)": "phase_hv_deg",
            "phasetheta-phi(deg)": "phase_vh_deg",
            "phasephi-phi(deg)": "phase_hh_deg",
        }

        if rows is None:
            rows = _read_cst_delimited_rows(path)
        if not rows:
            raise ValueError("CST theta/phi table is empty")

        def _classify_fuzzy_header(cell_value: str) -> str | None:
            raw = str(cell_value or "").strip().lower()
            if raw == "":
                return None

            key = alias_to_key.get(_norm(raw))
            if key is not None:
                return key

            compact = re.sub(r"[^a-z0-9]+", "", raw)
            if compact in {"f", "freq"} or "frequency" in compact:
                return "frequency"
            if (
                "theta" in compact
                and "phase" not in compact
                and "rcs" not in compact
                and "abs" not in compact
                and ("deg" in compact or "degree" in compact)
            ):
                return "theta_deg"
            if (
                "phi" in compact
                and "phase" not in compact
                and "rcs" not in compact
                and "abs" not in compact
                and ("deg" in compact or "degree" in compact)
            ):
                return "phi_deg"

            has_phase = "phase" in compact and (
                "deg" in compact or "degree" in compact
            )
            has_explicit_rcs_quantity = (
                "rcs" in compact
                or "radarcrosssection" in compact
                or "sigma" in compact
            )
            has_explicit_rcs_unit = "dbsm" in compact or "dbm2" in compact
            has_mag = (
                has_explicit_rcs_quantity
                and has_explicit_rcs_unit
                and not has_phase
            )
            if not has_phase and not has_mag:
                return None

            pair_key: str | None = None
            theta_count = len(re.findall("theta", raw))
            phi_count = len(re.findall("phi", raw))
            if "phi-theta" in raw or re.search(r"phi[^a-z0-9]+theta", raw):
                pair_key = "hv"
            elif "theta-phi" in raw or re.search(r"theta[^a-z0-9]+phi", raw):
                pair_key = "vh"
            elif theta_count >= 2:
                pair_key = "vv"
            elif phi_count >= 2:
                pair_key = "hh"
            elif theta_count == 1 and phi_count == 0:
                pair_key = "vv"
            elif phi_count == 1 and theta_count == 0:
                pair_key = "hh"
            elif theta_count == 1 and phi_count == 1:
                pair_key = "hv" if raw.find("phi") < raw.find("theta") else "vh"

            if pair_key is None:
                return None
            if has_phase:
                return f"phase_{pair_key}_deg"
            return f"rcs_{pair_key}_dbsm"

        header_idx = None
        data_start_idx = 0
        col_idx: dict[str, int] = {}
        header_tokens: dict[str, str] = {}
        required_axes = {"frequency", "theta_deg", "phi_deg"}
        for i, row in enumerate(rows):
            mapped: dict[str, int] = {}
            mapped_tokens: dict[str, str] = {}
            ambiguous_physics_headers: list[str] = []
            for j, cell in enumerate(row):
                key = _classify_fuzzy_header(cell)
                if key is not None and key not in mapped:
                    mapped[key] = j
                    mapped_tokens[key] = str(cell)
                    continue
                compact = re.sub(
                    r"[^a-z0-9]+", "", str(cell or "").strip().lower()
                )
                mentions_basis = "theta" in compact or "phi" in compact
                mentions_rcs = (
                    "rcs" in compact
                    or "radarcrosssection" in compact
                    or "sigma" in compact
                )
                if mentions_basis and ("phase" in compact or mentions_rcs):
                    ambiguous_physics_headers.append(str(cell))
            has_any_rcs = any(k.startswith("rcs_") for k in mapped.keys())
            if required_axes.issubset(mapped.keys()) and has_any_rcs:
                if ambiguous_physics_headers:
                    raise ValueError(
                        "Ambiguous CST wide-table physics header(s): "
                        + ", ".join(repr(value) for value in ambiguous_physics_headers)
                        + ". RCS magnitudes must state dBsm/dBm^2 and phases "
                        "must state degrees."
                    )
                header_idx = i
                data_start_idx = i + 1
                col_idx = mapped
                header_tokens = mapped_tokens
                break

        if header_idx is None:
            raise ValueError(
                "Could not find an explicit CST RCS header. Need frequency "
                "with units, theta/phi axes, and at least one RCS magnitude "
                "column explicitly labeled dBsm or dBm^2. Headerless/order-"
                "guessed and generic Abs(field) tables are not accepted."
            )

        records: list[tuple[float, float, float, float, float, float, float, float, float, float, float]] = []
        for row_index, row in enumerate(rows[data_start_idx:], start=data_start_idx):
            line_no = row_index + 1
            if not row or all(str(cell).strip() == "" for cell in row):
                continue

            def _axis_cell(key: str) -> float:
                idx = col_idx[key]
                raw = row[idx] if idx < len(row) else ""
                text = str(raw).strip()
                if not text:
                    raise ValueError(f"line {line_no}: {key} is blank")
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_no}: invalid {key} value {text!r}"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"line {line_no}: {key} must be finite")
                return value

            f_hz = _axis_cell("frequency")
            if f_hz <= 0.0:
                raise ValueError(f"line {line_no}: frequency must be positive")
            theta_deg = _axis_cell("theta_deg")
            phi_deg = _axis_cell("phi_deg")

            def _cell(key: str) -> float:
                idx = col_idx.get(key, -1)
                if idx < 0 or idx >= len(row):
                    return float("nan")
                text = str(row[idx]).strip()
                if not text:
                    return float("nan")
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_no}: invalid {key} value {text!r}"
                    ) from exc
                if key.startswith("phase_") and not np.isfinite(value):
                    raise ValueError(f"line {line_no}: {key} must be finite")
                if key.startswith("rcs_") and not (
                    np.isfinite(value) or np.isneginf(value)
                ):
                    raise ValueError(
                        f"line {line_no}: {key} must be finite or -Inf"
                    )
                return value

            records.append(
                (
                    float(f_hz),
                    float(theta_deg),
                    float(phi_deg),
                    _cell("rcs_vv_dbsm"),
                    _cell("rcs_hv_dbsm"),
                    _cell("rcs_vh_dbsm"),
                    _cell("rcs_hh_dbsm"),
                    _cell("phase_vv_deg"),
                    _cell("phase_hv_deg"),
                    _cell("phase_vh_deg"),
                    _cell("phase_hh_deg"),
                )
            )

        if not records:
            raise ValueError("CSV contains no data rows after the header")

        freq_scale_to_ghz, _ = _infer_freq_scale_to_ghz(
            header_tokens.get("frequency", "")
        )

        # Preserve the established CST component-name mapping, but do not add
        # all-NaN polarization axes merely because the wide schema permits them.
        channel_specs = (
            ("VV", 3, 7, "theta-theta"),
            ("HV", 4, 8, "phi-theta"),
            ("VH", 5, 9, "theta-phi"),
            ("HH", 6, 10, "phi-phi"),
        )

        def _has_magnitude(value):
            return not bool(np.isnan(value))

        present_specs = [
            spec for spec in channel_specs
            if any(_has_magnitude(record[spec[1]]) for record in records)
        ]
        if not present_specs:
            raise ValueError(
                "CST theta/phi table parsed, but no finite RCS magnitude values were found"
            )

        normalized_records = []
        for record in records:
            f_ghz = float(record[0] * freq_scale_to_ghz)
            theta_deg = float(record[1])
            if theta_deg < -1.0e-9 or theta_deg > 180.0 + 1.0e-9:
                raise ValueError(
                    f"standard CST theta must be within [0, 180] deg, got {theta_deg:g}"
                )
            elevation_deg = float(90.0 - theta_deg)
            azimuth_deg = _wrap_cst_azimuth_deg(record[2])
            if any(_has_magnitude(record[spec[1]]) for spec in present_specs):
                normalized_records.append(
                    (f_ghz, elevation_deg, azimuth_deg, record)
                )

        freqs = np.asarray(
            sorted({record[0] for record in normalized_records}), dtype=float
        )
        elevs = np.asarray(
            sorted({record[1] for record in normalized_records}), dtype=float
        )
        azims = np.asarray(
            sorted({record[2] for record in normalized_records}), dtype=float
        )
        pols = np.asarray([spec[0] for spec in present_specs], dtype=object)

        f_idx = {float(value): index for index, value in enumerate(freqs.tolist())}
        el_idx = {float(value): index for index, value in enumerate(elevs.tolist())}
        az_idx = {float(value): index for index, value in enumerate(azims.tolist())}
        pol_idx = {str(value): index for index, value in enumerate(pols.tolist())}

        shape = (len(azims), len(elevs), len(freqs), len(pols))
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)

        def _dbsm_to_linear(value: float) -> float:
            return _cst_dbsm_to_power(value, context="CST wide-table magnitude")

        def _deg_to_rad(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            return float(np.deg2rad(value))

        seen = {}
        for f_ghz, elevation_deg, azimuth_deg, source_record in normalized_records:
            ai = az_idx[azimuth_deg]
            ei = el_idx[elevation_deg]
            fi = f_idx[f_ghz]
            for pol_label, magnitude_index, phase_index, component_name in present_specs:
                magnitude = source_record[magnitude_index]
                if not _has_magnitude(magnitude):
                    continue
                sample_key = (azimuth_deg, elevation_deg, f_ghz, pol_label)
                sample_power = _dbsm_to_linear(magnitude)
                sample_phase = _deg_to_rad(source_record[phase_index])
                if sample_key in seen:
                    prior_power, prior_phase = seen[sample_key]
                    if _cst_samples_equivalent(
                        prior_power, prior_phase, sample_power, sample_phase
                    ):
                        continue
                    raise ValueError(
                        "conflicting duplicate CST theta/phi sample after "
                        "coordinate conversion: "
                        f"az={azimuth_deg:g}, el={elevation_deg:g}, "
                        f"f={f_ghz:g} GHz, component={component_name}"
                    )
                seen[sample_key] = (sample_power, sample_phase)
                pi = pol_idx[pol_label]
                power[ai, ei, fi, pi] = sample_power
                phase[ai, ei, fi, pi] = sample_phase

        if not np.isfinite(power).any():
            raise ValueError(
                "CST theta/phi table parsed, but no finite RCS magnitude values were found"
            )

        return cls(
            azims,
            elevs,
            freqs,
            pols,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=(
                "Loaded CST theta/phi table; standard theta converted with "
                f"elevation=90-theta; phi wrapped to [-180, 180): {path}"
            ),
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
            extra={
                "source_format": "CST wide theta/phi table",
                "cst_angle_mapping": (
                    "elevation=90-theta; phi wrapped to [-180, 180)"
                ),
                "cst_polarization_mapping": (
                    "theta=V, phi=H; component pair mapped in written order"
                ),
            },
        )

    @classmethod
    def load_theta_phi_txt(cls, path):
        """Load whitespace-delimited theta/phi TXT format into an RcsGrid.

        Expected columns after two header rows:
            theta(deg), phi(deg), abs(rcs)(dbm^2), abs(theta)(dbm^2),
            phase(theta)(deg), abs(phi)(dbm^2), phase(phi)(deg), ax.ratio(db)

        Axis/polarization mapping:
            - theta(deg) -> azimuth
            - phi(deg)   -> elevation
            - theta -> V, phi -> H
              abs(theta), phase(theta) -> VV
              abs(phi),   phase(phi)   -> HH
            - abs(rcs) is loaded as a third polarization channel: TOTAL
        """

        def _norm_token(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())

        def _frequency_from_filename_ghz(file_path: str) -> float | None:
            name = os.path.basename(str(file_path))
            match = re.search(
                r"(?:^|[^a-z0-9])f\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-z]+)?",
                name,
                flags=re.IGNORECASE,
            )
            if match is None:
                return None
            try:
                raw_value = float(match.group(1))
            except (TypeError, ValueError):
                return None
            if not np.isfinite(raw_value):
                return None

            raw_unit = (match.group(2) or "").strip().lower()
            unit = raw_unit
            if unit.startswith("ghz"):
                scale = 1.0
            elif unit.startswith("mhz"):
                scale = 1.0e-3
            elif unit.startswith("khz"):
                scale = 1.0e-6
            elif unit.startswith("hz"):
                scale = 1.0e-9
            else:
                magnitude = abs(raw_value)
                if magnitude >= 1.0e6:
                    scale = 1.0e-9
                elif magnitude >= 1.0e3:
                    scale = 1.0e-3
                else:
                    scale = 1.0
            return float(raw_value * scale)

        alias_to_key = {
            "thetadeg": "theta_deg",
            "phideg": "phi_deg",
            "absrcsdbm2": "abs_rcs_dbm2",
            "absthetadbm2": "abs_theta_dbm2",
            "phasethetadeg": "phase_theta_deg",
            "absphidbm2": "abs_phi_dbm2",
            "phasephideg": "phase_phi_deg",
            "axratiodb": "ax_ratio_db",
        }

        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("TXT is empty")

        header_idx = None
        data_start_idx = 0
        col_idx: dict[str, int] = {}
        required = {
            "theta_deg",
            "phi_deg",
            "abs_theta_dbm2",
            "phase_theta_deg",
            "abs_phi_dbm2",
            "phase_phi_deg",
        }

        fallback_col_idx = {
            "theta_deg": 0,
            "phi_deg": 1,
            "abs_rcs_dbm2": 2,
            "abs_theta_dbm2": 3,
            "phase_theta_deg": 4,
            "abs_phi_dbm2": 5,
            "phase_phi_deg": 6,
            "ax_ratio_db": 7,
        }

        def _tokenize(text: str) -> list[str]:
            return [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]

        def _is_numeric_data_line(tokens: list[str]) -> bool:
            if len(tokens) < 7:
                return False
            numeric_needed = (0, 1, 3, 4, 5, 6)
            for idx in numeric_needed:
                if idx >= len(tokens):
                    return False
                try:
                    float(tokens[idx])
                except ValueError:
                    return False
            return True

        for i, line in enumerate(lines):
            tokens = _tokenize(line)
            mapped: dict[str, int] = {}
            for j, token in enumerate(tokens):
                key = alias_to_key.get(_norm_token(token))
                if key is not None and key not in mapped:
                    mapped[key] = j
            if required.issubset(mapped.keys()):
                header_idx = i
                col_idx = mapped
                data_start_idx = i + 1
                break

        if header_idx is None:
            for i, line in enumerate(lines):
                tokens = _tokenize(line)
                if _is_numeric_data_line(tokens):
                    col_idx = dict(fallback_col_idx)
                    data_start_idx = i
                    break
            else:
                raise ValueError(
                    "Could not parse TXT: expected header columns or numeric rows with at least 7 columns."
                )

        def _parse_float(raw: str) -> float:
            text = str(raw).strip()
            if text == "":
                return float("nan")
            return float(text)

        records: list[tuple[float, float, float, float, float, float, float, float]] = []
        for line in lines[data_start_idx:]:
            tokens = _tokenize(line)
            if not tokens:
                continue

            def _cell(key: str) -> float:
                idx = col_idx.get(key, -1)
                if idx < 0 or idx >= len(tokens):
                    return float("nan")
                try:
                    return _parse_float(tokens[idx])
                except ValueError:
                    return float("nan")

            theta_deg = _cell("theta_deg")
            phi_deg = _cell("phi_deg")
            if not (np.isfinite(theta_deg) and np.isfinite(phi_deg)):
                continue

            records.append(
                (
                    float(theta_deg),
                    float(phi_deg),
                    _cell("abs_theta_dbm2"),
                    _cell("phase_theta_deg"),
                    _cell("abs_phi_dbm2"),
                    _cell("phase_phi_deg"),
                    _cell("abs_rcs_dbm2"),
                    _cell("ax_ratio_db"),
                )
            )

        if not records:
            raise ValueError("TXT contains no data rows after header")

        azims = np.asarray(sorted({r[0] for r in records}), dtype=float)   # theta -> azimuth
        elevs = np.asarray(sorted({r[1] for r in records}), dtype=float)   # phi -> elevation
        freq_ghz = _frequency_from_filename_ghz(path)
        if freq_ghz is None:
            freqs = np.asarray([0.0], dtype=float)
            freq_unit = "arb"
        else:
            freqs = np.asarray([float(freq_ghz)], dtype=float)
            freq_unit = "GHz"
        pols = np.asarray(["VV", "HH", "TOTAL"], dtype=object)

        el_idx = {float(v): i for i, v in enumerate(elevs.tolist())}
        az_idx = {float(v): i for i, v in enumerate(azims.tolist())}

        shape = (len(azims), len(elevs), 1, len(pols))
        power = np.full(shape, np.nan, dtype=np.float32)
        phase = np.full(shape, np.nan, dtype=np.float32)

        def _db_to_linear(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            return float(10.0 ** (value / 10.0))

        def _deg_to_rad(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            return float(np.deg2rad(value))

        for theta_deg, phi_deg, abs_theta_db, ph_theta_deg, abs_phi_db, ph_phi_deg, abs_rcs_db, _ in records:
            ai = az_idx[theta_deg]
            ei = el_idx[phi_deg]
            power[ai, ei, 0, 0] = _db_to_linear(abs_theta_db)   # VV
            phase[ai, ei, 0, 0] = _deg_to_rad(ph_theta_deg)
            power[ai, ei, 0, 1] = _db_to_linear(abs_phi_db)     # HH
            phase[ai, ei, 0, 1] = _deg_to_rad(ph_phi_deg)
            power[ai, ei, 0, 2] = _db_to_linear(abs_rcs_db)     # TOTAL

        if not np.isfinite(power).any():
            raise ValueError("TXT parsed, but no finite magnitude values were found")

        return cls(
            azims,
            elevs,
            freqs,
            pols,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded theta/phi TXT: {path}",
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": freq_unit,
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
        )

    @classmethod
    def load_pio(cls, path):
        """Load a Pioneer (.pio / .cmplx_di) file into an RcsGrid.

        File layout:
            - ASCII header of `key=value` lines, terminated by a line whose
              key is `Offset` (giving the byte offset of the binary block).
            - Binary block of interleaved real/imag floats (single or double
              precision per the `precision` header field) of length
              xsize*ysize*2.
            - Optional ASCII footer of `key=value` lines (e.g. polarity, log).

        Axis convention (this loader):
            - X axis (xname=azimuth/position) -> azimuth in degrees, converted
              exactly from xunits in {deg, rad}
            - Y axis (yname=frequency)        -> frequency in GHz, converted
              exactly from yunits in {Hz, kHz, MHz, GHz}
            - elevation is restored from the optional Elevation field and
              ElevationUnits (defaulting to the X angular unit for legacy files)
            - polarization is taken from the `polarity` header/footer field, or
              inferred from HH/VV/VH/HV in the filename.
        """
        header: dict[str, str] = {}
        footer: dict[str, str] = {}
        first_line: str = ""

        with open(path, "rb") as f:
            raw_first = f.readline()
            first_line = raw_first.decode("ascii", errors="replace").strip()
            if "=" in first_line:
                first_key, _, first_value = first_line.partition("=")
                header[first_key.strip().lower()] = first_value.strip()

            # Read header until a line with key 'offset' (case-insensitive).
            while True:
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError("Unexpected EOF while reading PIO header")
                line = raw_line.decode("ascii", errors="replace").strip()
                if "=" in line:
                    key, _, value = line.partition("=")
                    key_l = key.strip().lower()
                    header[key_l] = value.strip()
                    if key_l == "offset":
                        break

            offset_raw = header.get("offset")
            if offset_raw is None:
                raise ValueError("PIO header missing 'Offset='")
            try:
                offset = int(float(offset_raw))
            except ValueError as exc:
                raise ValueError(f"PIO header has non-numeric Offset: {offset_raw!r}") from exc

            def _int(key: str) -> int | None:
                raw = header.get(key)
                if raw is None:
                    return None
                try:
                    return int(float(raw))
                except ValueError:
                    return None

            xsize = _int("xsize")
            ysize = _int("ysize")
            if xsize is None or ysize is None:
                raise ValueError("PIO header missing xsize/ysize")

            precision = (header.get("precision") or "").strip().lower()
            data_type = (header.get("type") or "complex").strip().lower()
            order_text = (header.get("order") or "little endian").strip().lower()
            if "big" in order_text:
                byte_order = ">"
            elif "little" in order_text or not order_text:
                byte_order = "<"
            else:
                raise ValueError(f"Unsupported PIO byte order: {order_text!r}")

            if precision == "single":
                dtype = np.dtype(f"{byte_order}f4")
            elif precision == "double":
                dtype = np.dtype(f"{byte_order}f8")
            else:
                raise ValueError(f"Unsupported PIO precision: {precision!r}")

            n_floats = int(xsize) * int(ysize) * (2 if data_type == "complex" else 1)
            itemsize = np.dtype(dtype).itemsize

            f.seek(offset, 0)
            raw_buf = f.read(n_floats * itemsize)
            if len(raw_buf) < n_floats * itemsize:
                raise ValueError(
                    f"PIO data block truncated: expected {n_floats * itemsize} bytes, got {len(raw_buf)}"
                )
            rawdata = np.frombuffer(raw_buf, dtype=dtype, count=n_floats)

            # Anything after the data block is treated as the optional footer.
            footer_blob = f.read()

        for raw_line in footer_blob.splitlines():
            line = raw_line.decode("ascii", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            footer[key.strip().lower()] = value.strip()

        def _parse_axis_values(key: str, expected_size: int) -> np.ndarray | None:
            raw = header.get(key)
            if raw is None:
                return None
            tokens = re.split(r"[:\s,]+", raw.strip())
            values: list[float] = []
            for tok in tokens:
                if not tok:
                    continue
                try:
                    values.append(float(tok))
                except ValueError:
                    return None
            if len(values) == expected_size:
                return np.asarray(values, dtype=float)
            return None

        def _build_axis(prefix: str, size: int) -> np.ndarray:
            vals = _parse_axis_values(f"{prefix}vals", size)
            if vals is not None:
                return vals
            start = header.get(f"{prefix}start")
            stop = header.get(f"{prefix}stop")
            step = header.get(f"{prefix}step")
            try:
                start_f = float(start) if start is not None else None
                stop_f = float(stop) if stop is not None else None
                step_f = float(step) if step is not None else None
            except ValueError:
                start_f = stop_f = step_f = None
            if start_f is not None and step_f is not None:
                return start_f + np.arange(size, dtype=float) * step_f
            if start_f is not None and stop_f is not None and size > 1:
                return np.linspace(start_f, stop_f, size)
            if size == 1 and start_f is not None:
                return np.asarray([start_f], dtype=float)
            raise ValueError(f"Could not reconstruct {prefix} axis from PIO header")

        xvals = _build_axis("x", int(xsize))
        yvals = _build_axis("y", int(ysize))

        xname = (header.get("xname") or "").strip().lower()
        yname = (header.get("yname") or "").strip().lower()

        if data_type == "complex":
            complex_arr = rawdata[0::2].astype(np.float64) + 1j * rawdata[1::2].astype(np.float64)
        else:
            complex_arr = rawdata.astype(np.complex128)

        # MATLAB reshape(data, xsize, ysize) is column-major.
        complex_dtype = np.complex128 if precision == "double" else np.complex64
        data_2d = np.asarray(complex_arr, dtype=complex_dtype).reshape(
            (int(xsize), int(ysize)), order="F"
        )

        if not (xname in ("azimuth", "position") and yname == "frequency"):
            raise ValueError(
                f"Unsupported PIO axes (xname={xname!r}, yname={yname!r}); "
                "expected azimuth/position vs frequency"
            )

        xunit = cls._canonical_unit(header.get("xunits"), _ANGLE_UNITS, "deg")
        if xunit not in {"deg", "rad"}:
            raise ValueError(
                f"Unsupported PIO azimuth unit: {header.get('xunits')!r}; "
                "expected degrees or radians"
            )
        yunit = cls._canonical_unit(header.get("yunits"), _FREQUENCY_UNITS, "GHz")
        frequency_to_ghz = {
            "Hz": 1.0e-9,
            "kHz": 1.0e-6,
            "MHz": 1.0e-3,
            "GHz": 1.0,
        }
        if yunit not in frequency_to_ghz:
            raise ValueError(
                f"Unsupported PIO frequency unit: {header.get('yunits')!r}; "
                "expected Hz, kHz, MHz, or GHz"
            )
        freqs_ghz = np.asarray(yvals, dtype=float) * frequency_to_ghz[yunit]

        elevation_raw = header.get("elevation") or footer.get("elevation")
        if elevation_raw is None or str(elevation_raw).strip() == "":
            elevation_native = 0.0
        else:
            try:
                elevation_native = float(elevation_raw)
            except ValueError as exc:
                raise ValueError(
                    f"PIO elevation is not numeric: {elevation_raw!r}"
                ) from exc
        elevation_unit_raw = (
            header.get("elevationunits")
            or header.get("elevation_units")
            or footer.get("elevationunits")
            or footer.get("elevation_units")
        )
        elevation_unit = cls._canonical_unit(
            elevation_unit_raw, _ANGLE_UNITS, xunit
        )
        if elevation_unit not in {"deg", "rad"}:
            raise ValueError(
                f"Unsupported PIO elevation unit: {elevation_unit_raw!r}; "
                "expected degrees or radians"
            )
        elevation_deg = (
            float(np.rad2deg(elevation_native))
            if elevation_unit == "rad"
            else elevation_native
        )

        pol = (header.get("polarity") or footer.get("polarity") or "").strip().upper()
        if not pol:
            stem = os.path.splitext(os.path.basename(str(path)))[0].upper()
            for tag in ("HH", "VV", "VH", "HV"):
                if tag in stem:
                    pol = tag
                    break
        if not pol:
            pol = "NA"

        azimuths = np.asarray(xvals, dtype=float)
        if xunit == "rad":
            azimuths = np.rad2deg(azimuths)
        elevations = np.asarray([elevation_deg], dtype=float)
        polarizations = np.asarray([pol], dtype=object)

        rcs_arr = data_2d[:, np.newaxis, :, np.newaxis]

        prior_log = header.get("log") or footer.get("log") or ""
        history_parts = [f"Loaded Pioneer file: {path}"]
        if prior_log:
            history_parts.append(f"prior log: {prior_log}")
        history = " | ".join(history_parts)

        return cls(
            azimuths,
            elevations,
            freqs_ghz,
            polarizations,
            rcs=rcs_arr,
            rcs_domain="complex_amplitude",
            source_path=str(path),
            history=history,
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
        )

    def save_pio(self, path, *, el_idx=None, pol_idx=None, precision="single"):
        """Save a single (elevation, polarization) slice as a Pioneer .pio file.

        Round-trips with `load_pio`: a grid loaded from a .pio file and saved
        back via this method produces the same complex samples within the
        selected on-disk precision.

        Args:
            path: Output path. `.pio` is appended if missing.
            el_idx: Elevation index to slice. Defaults to 0 if there is exactly
                one elevation; required otherwise.
            pol_idx: Polarization index to slice. Defaults to 0 if there is
                exactly one polarization; required otherwise.
            precision: 'single' (default) or 'double' — width of the on-disk
                interleaved real/imag floats.

        Returns:
            The actual path written.
        """
        if self.angular_coordinate_system() != "conic":
            raise ValueError(
                "save_pio: Pioneer azimuth/elevation output cannot represent "
                f"{self.angular_coordinate_system()!r} angular coordinates; "
                "retain .grim or use PTM for a great-circle cut"
            )
        if el_idx is None:
            if len(self.elevations) == 1:
                el_idx = 0
            else:
                raise ValueError(
                    f"save_pio: el_idx required ({len(self.elevations)} elevations present)"
                )
        if pol_idx is None:
            if len(self.polarizations) == 1:
                pol_idx = 0
            else:
                raise ValueError(
                    f"save_pio: pol_idx required ({len(self.polarizations)} polarizations present)"
                )

        path = str(path)
        if not path.lower().endswith((".pio", ".cmplx_di")):
            path = f"{path}.pio"

        precision_l = (precision or "single").strip().lower()
        if precision_l == "single":
            dtype = np.dtype("<f4")
            precision_label = "Single"
        elif precision_l == "double":
            dtype = np.dtype("<f8")
            precision_label = "Double"
        else:
            raise ValueError(f"save_pio: unsupported precision {precision!r}")

        azimuths = np.asarray(self.azimuths, dtype=float)
        frequencies = np.asarray(self.frequencies, dtype=float)
        xsize = int(azimuths.size)
        ysize = int(frequencies.size)

        # complex_slice[i, j] = complex sample at azimuths[i], frequencies[j]
        power_slice = self.rcs_power[:, el_idx, :, pol_idx]
        phase_slice = self.rcs_phase[:, el_idx, :, pol_idx]
        phase_missing = np.isfinite(power_slice) & ~np.isfinite(phase_slice)
        if np.any(phase_missing):
            raise ValueError(
                "save_pio: complex PIO export requires phase for every finite-power "
                f"sample; {int(np.count_nonzero(phase_missing))} sample(s) lack phase"
            )
        complex_slice = np.asarray(
            self.rcs_slice((slice(None), el_idx, slice(None), pol_idx)),
            dtype=np.complex128,
        )
        if complex_slice.shape != (xsize, ysize):
            raise ValueError(
                f"save_pio: slice shape {complex_slice.shape} != ({xsize}, {ysize})"
            )

        xunits = self._canonical_unit(
            (self.units or {}).get("azimuth"), _ANGLE_UNITS, "deg"
        )
        if xunits not in {"deg", "rad"}:
            raise ValueError(
                "save_pio: azimuth unit must be degrees or radians; got "
                f"{(self.units or {}).get('azimuth')!r}"
            )
        elevation_units = self._canonical_unit(
            (self.units or {}).get("elevation"), _ANGLE_UNITS, "deg"
        )
        if elevation_units not in {"deg", "rad"}:
            raise ValueError(
                "save_pio: elevation unit must be degrees or radians; got "
                f"{(self.units or {}).get('elevation')!r}"
            )
        yunits = self._canonical_unit(
            (self.units or {}).get("frequency"), _FREQUENCY_UNITS, "GHz"
        )
        if yunits not in set(_FREQUENCY_UNITS.values()):
            raise ValueError(
                "save_pio: frequency unit must be Hz, kHz, MHz, or GHz; got "
                f"{(self.units or {}).get('frequency')!r}"
            )
        pol_label = str(self.polarizations[pol_idx]) if len(self.polarizations) else ""
        elevation_value = float(self.elevations[el_idx]) if len(self.elevations) else 0.0

        def _axis_summary(values):
            if len(values) == 1:
                return float(values[0]), float(values[0]), 0.0
            start = float(values[0])
            stop = float(values[-1])
            step = (stop - start) / (len(values) - 1)
            return start, stop, step

        xstart, xstop, xstep = _axis_summary(azimuths)
        ystart, ystop, ystep = _axis_summary(frequencies)

        def _pio_number(value):
            # Preserve a float64 axis through text and any subsequent unit
            # conversion (notably radians -> degrees on import).
            return format(float(value), ".17g")

        def _vals(arr):
            return ":".join(_pio_number(v) for v in arr)

        name_field = os.path.splitext(os.path.basename(path))[0]
        info_field = self.history or ""
        # Newlines would corrupt the header parser; flatten them.
        info_field = info_field.replace("\r", " ").replace("\n", " ")

        header_lines = [
            f"Name={name_field}",
            f"Info={info_field}",
            f"XStart={_pio_number(xstart)}",
            f"XStop={_pio_number(xstop)}",
            f"XStep={_pio_number(xstep)}",
            f"XSize={xsize}",
            "XName=azimuth",
            f"XUnits={xunits}",
            f"XVals={_vals(azimuths)}",
            f"YStart={_pio_number(ystart)}",
            f"YStop={_pio_number(ystop)}",
            f"YStep={_pio_number(ystep)}",
            f"YSize={ysize}",
            "YName=frequency",
            f"YUnits={yunits}",
            f"YVals={_vals(frequencies)}",
            "Type=Complex",
            f"Precision={precision_label}",
            "Order=Little Endian",
            "DataFormat=Binary",
        ]
        if pol_label:
            header_lines.append(f"Polarity={pol_label}")
        header_lines.append(f"Elevation={_pio_number(elevation_value)}")
        header_lines.append(f"ElevationUnits={elevation_units}")

        header_blob = ("\n".join(header_lines) + "\n").encode("ascii")
        # Reserve a fixed-width Offset line so the offset value can be filled
        # in before the binary block is written:
        #   "Offset=" (7) + 10-digit zero-padded offset + "\n" (1) = 18 bytes
        offset_line_bytes = 18
        data_offset = len(header_blob) + offset_line_bytes
        offset_line = f"Offset={data_offset:010d}\n".encode("ascii")
        if len(offset_line) != offset_line_bytes:
            raise RuntimeError(
                f"save_pio: offset line width drift ({len(offset_line)} != {offset_line_bytes})"
            )

        # Loader does reshape((xsize, ysize), order='F'), so we flatten the
        # same way: column-major over (azimuth, frequency).
        flat = complex_slice.flatten(order="F")
        interleaved = np.empty(2 * flat.size, dtype=dtype)
        interleaved[0::2] = flat.real.astype(dtype, copy=False)
        interleaved[1::2] = flat.imag.astype(dtype, copy=False)

        directory = os.path.dirname(os.path.abspath(path)) or os.curdir
        fd, stage_path = tempfile.mkstemp(
            prefix=".pio-write-", suffix=".staging", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as f:
                fd = -1
                f.write(header_blob)
                f.write(offset_line)
                f.write(interleaved.tobytes(order="C"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(stage_path, path)
        finally:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(stage_path):
                try:
                    os.unlink(stage_path)
                except OSError:
                    pass

        return path
