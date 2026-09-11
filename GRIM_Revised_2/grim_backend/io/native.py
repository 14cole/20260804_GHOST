"""Native .grim archive loading, saving, and payload validation."""
from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np

from grim_backend.datasets.constants import (
    C0,
    _ANGLE_UNITS,
    _FREQUENCY_UNITS,
    _RAW_COMPLEX_VALIDATION_BLOCK_CELLS,
)


class NativeFormatMixin:
    """Native .grim archive loading, saving, and payload validation."""

    _RESERVED_KEYS = ("azimuths", "elevations", "frequencies", "polarizations",
                      "rcs_power", "rcs_phase", "source_path", "history", "units")

    def _extra_to_write(self):
        """Return metadata and ancillary arrays for native archive export.

        Arrays with four or more dimensions must match the grid in their first four
        dimensions. Lower-dimensional ancillary arrays are retained independently.
        """
        expected = (len(self.azimuths), len(self.elevations),
                    len(self.frequencies), len(self.polarizations))
        out = {}
        for key, value in self.extra.items():
            if key in self._RESERVED_KEYS:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 4 and arr.shape[:4] != expected:
                continue
            out[key] = value
        return out

    def save(self, path, *, compressed=True):
        """Save the grid to a .grim (npz) file.

        Passthrough metadata from ``extra`` is written first (so the grid's own
        axes and samples always win on a name clash), which is what lets a file
        carrying a raw complex amplitude survive a load/save round-trip.
        The archive is fully written and flushed to a same-directory staging
        file before ``os.replace`` publishes it, so a failed save leaves an
        existing artifact intact.

        Args:
            path: Output path, with or without .grim.
            compressed: ``True`` (default) writes a compact ZIP-compressed
                archive for distribution/storage. ``False`` uses the faster
                uncompressed NPZ path for temporary high-throughput work.

        Returns:
            The actual path written (always ends with .grim).
        """
        if not isinstance(compressed, (bool, np.bool_)):
            raise TypeError("compressed must be True or False")
        path = os.fspath(path)
        if not path.casefold().endswith(".grim"):
            path = f"{path}.grim"
        extra_to_write = self._extra_to_write()
        self._validate_native_payload(
            path=path,
            azimuths=self.azimuths,
            elevations=self.elevations,
            frequencies=self.frequencies,
            polarizations=self.polarizations,
            rcs_power=self.rcs_power,
            rcs_phase=self.rcs_phase,
            units=self.units,
            extra=extra_to_write,
        )
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
                payload = dict(extra_to_write)
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


                for tag in ("rcs_domain", "power_domain"):
                    if tag in self.extra:
                        payload[tag] = self.extra[tag]
                object_keys = [
                    str(key)
                    for key, value in payload.items()
                    if np.asarray(value).dtype.hasobject
                ]
                if object_keys:
                    raise ValueError(
                        "cannot save a pickle-free .grim archive because metadata "
                        "contains object-typed value(s): "
                        + ", ".join(sorted(object_keys))
                        + ". Convert those values to numeric/string arrays or JSON text."
                    )
                writer = np.savez_compressed if bool(compressed) else np.savez
                writer(f, **payload)
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
    def _raw_complex_consistency_report(
        cls,
        *,
        expected_shape,
        frequencies,
        rcs_power,
        rcs_phase,
        units,
        extra,
    ):
        """Check a solver raw-field pair against displayed power and phase.

        GHOST stores its unnormalised far-field amplitude alongside physical
        ``rcs_power``.  The relationship is quantity dependent: sigma3D is
        ``4*pi*|A|^2`` and sigma2D is ``|A|^2/(4*k0)``.  This routine mirrors
        the producer-side tolerances while scanning bounded blocks, so loading
        or auditing a large archive never constructs another grid-sized
        complex or expected-power array.
        """

        metadata = dict(extra or {})
        has_real = "rcs_amp_real" in metadata
        has_imag = "rcs_amp_imag" in metadata
        report = {
            "present": bool(has_real or has_imag),
            "complete_pair": bool(has_real and has_imag),
            "finite_pair_count": 0,
            "missing_pair_count": 0,
            "invalid_pair_count": 0,
            "coverage_mismatch_count": 0,
            "live_phase_missing_count": 0,
            "phase_mismatch_count": 0,
            "power_mismatch_count": 0,
            "normalization_overflow_count": 0,
            "maximum_phase_error_rad": None,
            "maximum_power_absolute_error": None,
            "normalization": None,
            "issues": [],
        }

        def issue(code, message, **details):
            item = {"code": str(code), "message": str(message)}
            item.update(details)
            report["issues"].append(item)

        raw_flag = metadata.get("raw_complex_amplitude_preserved")
        flag_true = False
        if raw_flag is not None:
            raw_flag_array = np.asarray(raw_flag)
            if raw_flag_array.size != 1:
                issue(
                    "invalid_raw_complex_preserved_flag",
                    "raw_complex_amplitude_preserved must be scalar",
                )
            else:
                flag_value = raw_flag_array.reshape(-1)[0]
                if isinstance(flag_value, (str, np.str_, bytes, np.bytes_)):
                    if isinstance(flag_value, (bytes, np.bytes_)):
                        try:
                            flag_value = bytes(flag_value).decode("ascii")
                        except UnicodeDecodeError:
                            flag_value = ""
                    flag_true = str(flag_value).strip().casefold() in {
                        "1", "true", "yes", "on"
                    }
                else:
                    try:
                        flag_true = bool(flag_value)
                    except (TypeError, ValueError):
                        flag_true = False

        if has_real != has_imag:
            issue(
                "partial_raw_complex_pair",
                "raw complex amplitude must provide both rcs_amp_real and rcs_amp_imag",
            )
            return report
        if not has_real:
            if flag_true:
                issue(
                    "missing_raw_complex_pair",
                    "raw_complex_amplitude_preserved is true but the raw amplitude grids are absent",
                )
            return report

        raw_real = np.asarray(metadata["rcs_amp_real"])
        raw_imag = np.asarray(metadata["rcs_amp_imag"])
        for name, values in (
            ("rcs_amp_real", raw_real),
            ("rcs_amp_imag", raw_imag),
        ):
            if values.shape != tuple(expected_shape):
                issue(
                    "raw_complex_shape_mismatch",
                    f"{name} shape {values.shape} does not match axes {tuple(expected_shape)}",
                    field=name,
                )
            if values.dtype.kind not in "iuf":
                issue(
                    "non_numeric_raw_complex",
                    f"{name} must be real numeric",
                    field=name,
                )
        if any(item["code"] != "invalid_raw_complex_preserved_flag" for item in report["issues"]):
            return report

        power = np.asarray(rcs_power)
        phase = np.asarray(rcs_phase)
        if power.shape != tuple(expected_shape) or phase.shape != tuple(expected_shape):
            return report
        if power.dtype.kind not in "iuf" or phase.dtype.kind not in "iuf":
            return report

        normalized_units = dict(units or {})
        quantity = str(
            normalized_units.get("rcs_linear_quantity", "")
        ).strip().casefold()
        if not quantity:
            log_unit = str(
                normalized_units.get("rcs_log_unit", "dBsm")
            ).strip().casefold()
            quantity = "sigma_2d" if log_unit == "dbke" else "sigma_3d"
        if quantity not in {"sigma_2d", "sigma_3d"}:
            issue(
                "unsupported_raw_complex_normalization",
                "raw complex amplitude requires rcs_linear_quantity sigma_2d or sigma_3d",
                linear_quantity=quantity,
            )
            return report
        report["normalization"] = quantity

        frequency_values = np.asarray(frequencies, dtype=np.float64)
        frequency_unit = cls._canonical_unit(
            normalized_units.get("frequency"), _FREQUENCY_UNITS, "GHz"
        )
        frequency_scale = {
            "Hz": 1.0,
            "kHz": 1.0e3,
            "MHz": 1.0e6,
            "GHz": 1.0e9,
        }.get(frequency_unit)
        if frequency_scale is None:
            issue(
                "unsupported_raw_complex_frequency_unit",
                "raw complex amplitude normalization requires a supported frequency unit",
                frequency_unit=str(normalized_units.get("frequency")),
            )
            return report

        max_phase_error = 0.0
        max_power_error = 0.0
        float32_epsilon = np.finfo(np.float32).eps
        float32_tiny = np.finfo(np.float32).tiny

        for frequency_index, frequency_value in enumerate(frequency_values):
            if not np.isfinite(frequency_value) or frequency_value <= 0.0:
                issue(
                    "invalid_raw_complex_frequency",
                    "raw complex amplitude normalization requires positive finite frequencies",
                )
                return report
            if quantity == "sigma_2d":
                k0 = (
                    2.0
                    * np.pi
                    * float(frequency_value)
                    * float(frequency_scale)
                    / C0
                )
                power_scale = 1.0 / (4.0 * k0)
            else:
                power_scale = 4.0 * np.pi

            iterator = np.nditer(
                (
                    power[:, :, frequency_index, :],
                    phase[:, :, frequency_index, :],
                    raw_real[:, :, frequency_index, :],
                    raw_imag[:, :, frequency_index, :],
                ),
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"], ["readonly"], ["readonly"]],
                order="K",
                buffersize=_RAW_COMPLEX_VALIDATION_BLOCK_CELLS,
            )
            for power_block, phase_block, real_block, imag_block in iterator:
                power_block = np.asarray(power_block, dtype=np.float64)
                phase_block = np.asarray(phase_block, dtype=np.float64)
                real_block = np.asarray(real_block, dtype=np.float64)
                imag_block = np.asarray(imag_block, dtype=np.float64)

                real_finite = np.isfinite(real_block)
                imag_finite = np.isfinite(imag_block)
                raw_finite = real_finite & imag_finite
                raw_missing = np.isnan(real_block) & np.isnan(imag_block)
                invalid_pair = ~(raw_finite | raw_missing)
                power_finite = np.isfinite(power_block)

                report["finite_pair_count"] += int(np.count_nonzero(raw_finite))
                report["missing_pair_count"] += int(np.count_nonzero(raw_missing))
                report["invalid_pair_count"] += int(np.count_nonzero(invalid_pair))
                report["coverage_mismatch_count"] += int(
                    np.count_nonzero(raw_finite != power_finite)
                )

                comparable = raw_finite & power_finite
                if not np.any(comparable):
                    continue

                comparable_real = real_block[comparable]
                comparable_imag = imag_block[comparable]
                comparable_power = power_block[comparable]
                with np.errstate(over="ignore", invalid="ignore"):
                    amp_abs2 = (
                        comparable_real * comparable_real
                        + comparable_imag * comparable_imag
                    )
                    expected_power = amp_abs2 * power_scale
                expected_finite = np.isfinite(expected_power)
                overflow_count = int(np.count_nonzero(~expected_finite))
                report["normalization_overflow_count"] += overflow_count
                if np.any(expected_finite):
                    expected_values = expected_power[expected_finite]
                    stored_values = comparable_power[expected_finite]
                    absolute_error = np.abs(stored_values - expected_values)
                    tolerance = (
                        16.0
                        * float32_epsilon
                        * np.maximum(expected_values, stored_values)
                        + float32_tiny
                    )
                    report["power_mismatch_count"] += int(
                        np.count_nonzero(absolute_error > tolerance)
                    )
                    if absolute_error.size:
                        max_power_error = max(
                            max_power_error, float(np.max(absolute_error))
                        )

                live = comparable & (
                    (np.abs(real_block) > float32_tiny)
                    | (np.abs(imag_block) > float32_tiny)
                )
                live_phase_missing = live & ~np.isfinite(phase_block)
                report["live_phase_missing_count"] += int(
                    np.count_nonzero(live_phase_missing)
                )
                phase_comparable = live & np.isfinite(phase_block)
                if np.any(phase_comparable):
                    raw_angle = np.arctan2(
                        imag_block[phase_comparable], real_block[phase_comparable]
                    )
                    phase_difference = phase_block[phase_comparable] - raw_angle
                    phase_error = np.abs(
                        np.arctan2(np.sin(phase_difference), np.cos(phase_difference))
                    )
                    report["phase_mismatch_count"] += int(
                        np.count_nonzero(phase_error > 2.0e-5)
                    )
                    if phase_error.size:
                        max_phase_error = max(
                            max_phase_error, float(np.max(phase_error))
                        )

        report["maximum_phase_error_rad"] = float(max_phase_error)
        report["maximum_power_absolute_error"] = float(max_power_error)
        if report["invalid_pair_count"]:
            issue(
                "invalid_raw_complex_pair",
                "raw complex amplitude contains one-sided NaN or infinite component samples",
                count=report["invalid_pair_count"],
            )
        if report["coverage_mismatch_count"]:
            issue(
                "raw_complex_coverage_mismatch",
                "raw complex amplitude finite coverage does not match rcs_power",
                count=report["coverage_mismatch_count"],
            )
        if report["normalization_overflow_count"]:
            issue(
                "raw_complex_normalization_overflow",
                f"raw complex amplitude is too large to form finite {quantity} power",
                count=report["normalization_overflow_count"],
            )
        if report["power_mismatch_count"]:
            issue(
                "raw_complex_power_mismatch",
                "rcs_power is inconsistent with the stored raw complex amplitude "
                f"under {quantity} normalization",
                count=report["power_mismatch_count"],
                maximum_absolute_error=report["maximum_power_absolute_error"],
            )
        if report["live_phase_missing_count"]:
            issue(
                "raw_complex_phase_missing",
                "nonzero raw complex amplitude has no finite stored rcs_phase",
                count=report["live_phase_missing_count"],
            )
        if report["phase_mismatch_count"]:
            issue(
                "raw_complex_phase_mismatch",
                "rcs_phase is inconsistent with the stored raw complex amplitude",
                count=report["phase_mismatch_count"],
                maximum_phase_error_rad=report["maximum_phase_error_rad"],
            )
        return report

    @classmethod
    def _validate_native_payload(
        cls,
        *,
        path,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs_power,
        rcs_phase,
        units,
        extra=None,
    ):
        """Validate the native archive before constructor sanitation.

        NaN RCS cells are intentional sparse-grid markers and are retained.
        Infinities, negative finite power, malformed axes, and ambiguous unit
        declarations are rejected rather than silently repaired.
        """

        numeric_axes = {}
        for name, raw_values in (
            ("azimuth", azimuths),
            ("elevation", elevations),
            ("frequency", frequencies),
        ):
            values = np.asarray(raw_values)
            if (
                values.ndim != 1
                or values.size == 0
                or values.dtype.kind not in "iuf"
            ):
                raise ValueError(
                    f"{path} contains an invalid {name} axis; expected a "
                    "nonempty one-dimensional real numeric array"
                )
            checked_values = values.astype(np.float64, copy=False)
            if np.any(~np.isfinite(checked_values)):
                raise ValueError(f"{path} contains a nonfinite {name} coordinate")
            if np.unique(checked_values).size != checked_values.size:
                raise ValueError(f"{path} contains duplicate {name} coordinates")
            if name == "frequency" and np.any(checked_values <= 0.0):
                raise ValueError(
                    f"{path} contains a nonpositive frequency coordinate"
                )


            numeric_axes[name] = cls._clean_axis(values)

        raw_polarizations = np.asarray(polarizations)
        if raw_polarizations.ndim != 1 or raw_polarizations.size == 0:
            raise ValueError(
                f"{path} contains an invalid polarization axis; expected a "
                "nonempty one-dimensional string array"
            )
        labels = []
        for raw_label in raw_polarizations.tolist():
            if isinstance(raw_label, bytes):
                try:
                    label = raw_label.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"{path} contains a non-UTF-8 polarization label"
                    ) from exc
            elif isinstance(raw_label, (str, np.str_)):
                label = str(raw_label)
            else:
                raise ValueError(
                    f"{path} contains a non-string polarization label "
                    f"{raw_label!r}"
                )
            label = label.strip()
            if not label:
                raise ValueError(f"{path} contains a blank polarization label")
            labels.append(label)
        normalized_labels = [label.casefold() for label in labels]
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError(
                f"{path} contains duplicate polarization labels after normalization"
            )

        expected = (
            len(numeric_axes["azimuth"]),
            len(numeric_axes["elevation"]),
            len(numeric_axes["frequency"]),
            len(labels),
        )
        power = np.asarray(rcs_power)
        phase = np.asarray(rcs_phase)
        for name, values in (("rcs_power", power), ("rcs_phase", phase)):
            if values.shape != expected:
                raise ValueError(
                    f"{path} contains {name} shape {values.shape}; expected {expected}"
                )
            if values.dtype.kind not in "iuf":
                raise ValueError(f"{path} contains non-real-numeric {name}")
            if np.any(np.isinf(values)):
                raise ValueError(f"{path} contains infinite {name} samples")


        if np.any(power < 0.0):
            raise ValueError(f"{path} contains negative finite rcs_power samples")

        normalized_units = dict(units or {})
        for key, aliases, default in (
            ("azimuth", _ANGLE_UNITS, "deg"),
            ("elevation", _ANGLE_UNITS, "deg"),
            ("frequency", _FREQUENCY_UNITS, "GHz"),
        ):
            raw_unit = normalized_units.get(key)
            canonical = cls._canonical_unit(raw_unit, aliases, default)
            if canonical not in set(aliases.values()):
                raise ValueError(
                    f"{path} contains unsupported {key} unit {raw_unit!r}"
                )
            if raw_unit is not None and str(raw_unit).strip():
                normalized_units[key] = canonical

        raw_report = cls._raw_complex_consistency_report(
            expected_shape=expected,
            frequencies=numeric_axes["frequency"],
            rcs_power=power,
            rcs_phase=phase,
            units=normalized_units,
            extra=extra,
        )
        numerical_issues = [issue for issue in raw_report["issues"] if issue["code"] not in {
            "invalid_raw_complex_preserved_flag", "missing_raw_complex_pair",
        }]
        if numerical_issues:
            first_issue = numerical_issues[0]
            raise ValueError(f"{path} contains {first_issue['message']}")

        return (
            numeric_axes["azimuth"],
            numeric_axes["elevation"],
            numeric_axes["frequency"],
            np.asarray(labels, dtype=str),
            power,
            phase,
            normalized_units,
        )

    @classmethod
    def load(
        cls,
        path,
        mmap_mode: str | None = None,
        *,
        allow_legacy_pickle: bool = False,
        max_output_bytes=None,
    ):
        """Load a grid from a .grim (npz) file.

        Args:
            path: Input path, with or without .grim.
            mmap_mode: Retained for API compatibility. ``.npz`` members cannot
                be memory-mapped; a warning is emitted when this is supplied.
            allow_legacy_pickle: Explicitly opt in to legacy object-array files.
                Never enable this for an untrusted file.
            max_output_bytes: Optional reviewed cap for the exact native NPZ
                payload plus the power/phase sanitation copies. By default one
                load may use at most half of currently available memory (or
                the conservative 2 GiB fallback when memory is unknown).

        Returns:
            RcsGrid instance loaded from disk.
        """
        from grim_backend.datasets.memory import _preflight_native_archive_allocation
        path = os.fspath(path)
        if not path.casefold().endswith(".grim"):
            path = f"{path}.grim"
        _preflight_native_archive_allocation(
            path,
            allow_legacy_pickle=bool(allow_legacy_pickle),
            max_output_bytes=max_output_bytes,
        )
        if mmap_mode is not None:
            warnings.warn(
                "mmap_mode has no effect for .grim/.npz archives; arrays are loaded eagerly",
                RuntimeWarning,
                stacklevel=2,
            )


        with open(path, "rb") as f, np.load(
            f, allow_pickle=bool(allow_legacy_pickle)
        ) as data:

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
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path} contains corrupt units metadata; refusing to "
                            "guess frequency, RCS, or angular conventions"
                        ) from exc
                elif isinstance(raw_units, dict):
                    units = raw_units
                if not isinstance(units, dict):
                    raise ValueError(
                        f"{path} contains invalid units metadata (expected a JSON object)"
                    )

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


            raw_extra = {
                key: data[key]
                for key in (
                    "rcs_amp_real",
                    "rcs_amp_imag",
                    "raw_complex_amplitude_preserved",
                )
                if key in data
            }

            (
                azimuths,
                elevations,
                frequencies,
                polarizations,
                rcs_power,
                rcs_phase,
                units,
            ) = cls._validate_native_payload(
                path=path,
                azimuths=data["azimuths"],
                elevations=data["elevations"],
                frequencies=data["frequencies"],
                polarizations=data["polarizations"],
                rcs_power=data["rcs_power"],
                rcs_phase=data["rcs_phase"],
                units=units,
                extra=raw_extra,
            )


            extra = {
                key: (
                    raw_extra[key]
                    if key in raw_extra
                    else data[key]
                )
                for key in getattr(data, "files", [])
                if key not in cls._RESERVED_KEYS
            }

            return cls(
                azimuths,
                elevations,
                frequencies,
                polarizations,
                rcs_power=rcs_power,
                rcs_phase=rcs_phase,
                rcs_domain="power_phase",
                source_path=source_path,
                history=history,
                units=units,
                extra=extra,
            )
