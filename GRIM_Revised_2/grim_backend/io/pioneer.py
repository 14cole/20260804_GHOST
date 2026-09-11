"""Pioneer PIO and CMPLX_DI loading and export."""
from __future__ import annotations

import os
import re
import tempfile
import unicodedata

import numpy as np


_PIO_ASCII_METADATA_REPLACEMENTS = str.maketrans(
    {
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "⇄": "<->",
        "⇒": "=>",
        "⇐": "<=",
        "⇔": "<=>",
        "°": " deg",
        "Δ": "Delta",
        "δ": "delta",
        "Σ": "Sum",
        "∑": "sum",
        "⊕": "+",
        "σ": "sigma",
        "λ": "lambda",
        "π": "pi",
        "θ": "theta",
        "φ": "phi",
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "x",
        "÷": "/",
        "·": "*",
        "≥": ">=",
        "≤": "<=",
        "≈": "~",
        "…": "...",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def _pio_ascii_metadata(value):
    """Return single-line ASCII for Pioneer header metadata.

    Transliterate engineering symbols and encode remaining characters as ASCII
    Unicode escapes.
    """

    text = str(value or "").translate(_PIO_ASCII_METADATA_REPLACEMENTS)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in text
    )
    text = " ".join(text.split())
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _pio_remove_closed_azimuth_endpoint(values, unit):
    """Return one physical turn of a PIO azimuth axis, without its closer.

    Range data often starts at an arbitrary turntable position and repeats that
    direction after one revolution.  The closing coordinate may therefore be
    ``181.16`` for an opening coordinate of ``-178.84``, or a writer may wrap
    it back to ``-178.84``.  Detect the full-turn trajectory rather than
    special-casing 0/360 or -180/+180.  The opening sample is authoritative;
    the repeated closing row is removed even when a second measurement differs.
    """

    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        return axis, False

    period = 2.0 * np.pi if unit == "rad" else 360.0
    half_period = 0.5 * period
    scale = max(1.0, float(np.max(np.abs(axis))), period)
    tolerance = max(
        float(np.deg2rad(1.0e-7)) if unit == "rad" else 1.0e-7,
        8.0 * np.finfo(np.float32).eps * scale,
    )

    raw_steps = np.diff(axis)
    periodic_steps = np.remainder(raw_steps + half_period, period) - half_period


    positive_half_turn = (
        np.isclose(periodic_steps, -half_period, rtol=0.0, atol=tolerance)
        & (raw_steps > 0.0)
    )
    periodic_steps[positive_half_turn] = half_period

    increasing = bool(np.all(periodic_steps > 0.0))
    descending = bool(np.all(periodic_steps < 0.0))
    unwrapped = np.concatenate(
        (axis[:1], axis[0] + np.cumsum(periodic_steps, dtype=float))
    )
    trajectory_span = float(unwrapped[-1] - unwrapped[0])
    raw_span = float(axis[-1] - axis[0])
    endpoint_residual = float(
        np.remainder(raw_span + half_period, period) - half_period
    )
    same_direction = abs(endpoint_residual) <= tolerance
    full_turn = (
        abs(abs(trajectory_span) - period) <= tolerance
        or abs(abs(raw_span) - period) <= tolerance
    )
    direct_two_point_turn = (
        axis.size == 2 and abs(abs(raw_span) - period) <= tolerance
    )
    if not (
        same_direction
        and full_turn
        and (increasing or descending or direct_two_point_turn)
    ):
        return axis, False

    raw_prefix_steps = np.diff(axis[:-1])
    raw_prefix_monotonic = bool(
        axis.size <= 2
        or np.all(raw_prefix_steps > 0.0)
        or np.all(raw_prefix_steps < 0.0)
    )
    unique_axis = axis[:-1] if raw_prefix_monotonic else unwrapped[:-1]
    return np.asarray(unique_axis, dtype=float).copy(), True


class PioFormatMixin:
    """Bounded PIO binary/text import and export for RcsGrid."""

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

        A closed full-turn azimuth sweep is stored half-open in GRIM.  Its
        repeated closing row is removed based on one-period equivalence to the
        opening angle, so the seam may occur at any measured angle rather than
        only at 0/360 or -180/+180.  The opening measurement is retained.
        """
        from grim_backend.datasets.constants import _ANGLE_UNITS, _FREQUENCY_UNITS
        from grim_backend.io.pioneer import _pio_remove_closed_azimuth_endpoint
        header: dict[str, str] = {}
        footer: dict[str, str] = {}
        first_line: str = ""

        with open(path, "rb") as f:
            file_size = int(os.fstat(f.fileno()).st_size)

            def _decode_header_line(raw_line: bytes, where: str) -> str:
                try:
                    decoded = raw_line.decode("ascii").strip()
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"PIO {where} must contain ASCII key=value text"
                    ) from exc
                if any(
                    ord(character) < 32 and character not in {"\t"}
                    for character in decoded
                ):
                    raise ValueError(
                        f"PIO {where} contains binary control bytes"
                    )
                return decoded

            raw_first = f.readline()
            first_line = _decode_header_line(raw_first, "header")
            if "=" in first_line:
                first_key, _, first_value = first_line.partition("=")
                header[first_key.strip().lower()] = first_value.strip()


            while True:
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError("Unexpected EOF while reading PIO header")
                line = _decode_header_line(raw_line, "header")
                if "=" in line:
                    key, _, value = line.partition("=")
                    key_l = key.strip().lower()
                    header[key_l] = value.strip()
                    if key_l == "offset":
                        break

            header_end = int(f.tell())

            offset_raw = header.get("offset")
            if offset_raw is None:
                raise ValueError("PIO header missing 'Offset='")

            def _integer_field(key: str, *, positive: bool) -> int:
                raw = header.get(key)
                if raw is None:
                    raise ValueError(f"PIO header missing {key}")
                text = str(raw).strip()
                if not re.fullmatch(r"[+-]?\d+", text):
                    raise ValueError(
                        f"PIO header {key} must be an exact integer; got {raw!r}"
                    )
                value = int(text, 10)
                if positive and value <= 0:
                    raise ValueError(f"PIO header {key} must be greater than zero")
                if not positive and value < 0:
                    raise ValueError(f"PIO header {key} must not be negative")
                return value

            offset = _integer_field("offset", positive=False)
            xsize = _integer_field("xsize", positive=True)
            ysize = _integer_field("ysize", positive=True)
            if offset < header_end:
                raise ValueError(
                    f"PIO Offset={offset} precedes the end of the header at byte {header_end}"
                )

            precision = (header.get("precision") or "").strip().lower()
            data_type = (header.get("type") or "complex").strip().lower()
            if data_type not in {"complex", "real"}:
                raise ValueError(
                    f"Unsupported PIO Type: {header.get('type')!r}; expected Complex or Real"
                )
            data_format = (header.get("dataformat") or "binary").strip().lower()
            if data_format != "binary":
                raise ValueError(
                    f"Unsupported PIO DataFormat: {header.get('dataformat')!r}; expected Binary"
                )
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

            cell_count = int(xsize) * int(ysize)
            components_per_cell = 2 if data_type == "complex" else 1
            n_floats = cell_count * components_per_cell
            itemsize = np.dtype(dtype).itemsize
            payload_bytes = n_floats * itemsize
            if payload_bytes > np.iinfo(np.intp).max:
                raise ValueError(
                    "PIO dimensions exceed this Python/NumPy build's addressable payload size"
                )
            payload_end = offset + payload_bytes
            if offset > file_size:
                raise ValueError(
                    f"PIO Offset={offset} lies beyond the {file_size}-byte file"
                )
            if payload_end > file_size:
                available = max(0, file_size - offset)
                raise ValueError(
                    "PIO data block truncated: expected "
                    f"{payload_bytes} bytes at Offset={offset}, got {available}"
                )

            f.seek(offset, 0)
            raw_buf = f.read(payload_bytes)
            if len(raw_buf) != payload_bytes:
                raise ValueError(
                    f"PIO data block truncated: expected {payload_bytes} bytes, got {len(raw_buf)}"
                )
            rawdata = np.frombuffer(raw_buf, dtype=dtype, count=n_floats)


            footer_blob = f.read()

        for raw_line in footer_blob.splitlines():
            line = _decode_header_line(raw_line, "footer")
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    "PIO bytes after the declared data block are not a valid "
                    "ASCII key=value footer; Type/dimensions/Offset may be wrong"
                )
            key, _, value = line.partition("=")
            key = key.strip().lower()
            if not key:
                raise ValueError("PIO footer contains a blank key")
            footer[key] = value.strip()

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
                except ValueError as exc:
                    raise ValueError(
                        f"PIO header {key} contains a non-numeric axis value {tok!r}"
                    ) from exc
            if len(values) != expected_size:
                raise ValueError(
                    f"PIO header {key} contains {len(values)} values; "
                    f"expected {expected_size}"
                )
            result = np.asarray(values, dtype=float)
            if np.any(~np.isfinite(result)):
                raise ValueError(f"PIO header {key} contains a nonfinite axis value")
            return result

        def _build_axis(prefix: str, size: int) -> np.ndarray:
            vals = _parse_axis_values(f"{prefix}vals", size)
            start = header.get(f"{prefix}start")
            stop = header.get(f"{prefix}stop")
            step = header.get(f"{prefix}step")

            def _summary_tolerance(raw_text, *numeric_values):
                """Honor the decimal precision of a redundant PIO summary."""

                text = str(raw_text).strip()
                match = re.fullmatch(
                    r"[+-]?(?:(?:\d+)(?:\.(\d*))?|\.(\d+))"
                    r"(?:[eE]([+-]?\d+))?",
                    text,
                )
                decimal_digits = len(
                    (match.group(1) or match.group(2) or "") if match else ""
                )
                exponent = int(match.group(3) or 0) if match else 0
                printed_resolution = (
                    10.0 ** (exponent - decimal_digits) if match else 0.0
                )
                scale = max(
                    1.0,
                    *(abs(float(value)) for value in numeric_values),
                )
                return max(
                    0.5 * printed_resolution,
                    8.0 * np.finfo(np.float32).eps * scale,
                )

            parsed = {}
            for field_name, raw_value in (
                (f"{prefix}start", start),
                (f"{prefix}stop", stop),
                (f"{prefix}step", step),
            ):
                if raw_value is None:
                    parsed[field_name] = None
                    continue
                try:
                    numeric = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"PIO header {field_name} must be numeric; got {raw_value!r}"
                    ) from exc
                if not np.isfinite(numeric):
                    raise ValueError(
                        f"PIO header {field_name} must be finite; got {raw_value!r}"
                    )
                parsed[field_name] = numeric
            start_f = parsed[f"{prefix}start"]
            stop_f = parsed[f"{prefix}stop"]
            step_f = parsed[f"{prefix}step"]
            if vals is not None:


                comparisons = (
                    (f"{prefix.upper()}Start", start, start_f, float(vals[0])),
                    (f"{prefix.upper()}Stop", stop, stop_f, float(vals[-1])),
                )
                for label, raw_declared, declared, actual in comparisons:
                    if declared is None:
                        continue
                    tolerance = _summary_tolerance(
                        raw_declared, declared, actual
                    )
                    if abs(declared - actual) > tolerance:
                        raise ValueError(
                            f"PIO {label}={declared:.17g} conflicts with explicit "
                            f"{prefix.upper()}Vals endpoint {actual:.17g}; "
                            f"difference {abs(declared - actual):.6g} exceeds "
                            f"the summary precision tolerance {tolerance:.6g}"
                        )
                if step_f is not None:
                    summary_step = (
                        0.0
                        if size == 1
                        else float(vals[-1] - vals[0]) / float(size - 1)
                    )
                    tolerance = _summary_tolerance(
                        step, step_f, summary_step
                    )
                    if abs(step_f - summary_step) > tolerance:
                        raise ValueError(
                            f"PIO {prefix.upper()}Step={step_f:.17g} conflicts "
                            f"with explicit {prefix.upper()}Vals summary step "
                            f"{summary_step:.17g}; difference "
                            f"{abs(step_f - summary_step):.6g} exceeds the "
                            f"summary precision tolerance {tolerance:.6g}"
                        )
                return vals
            if start_f is not None and step_f is not None:
                values = start_f + np.arange(size, dtype=float) * step_f
                if stop_f is not None:
                    scale = max(1.0, abs(stop_f), abs(float(values[-1])))
                    if not np.isclose(
                        values[-1], stop_f, rtol=1.0e-10, atol=1.0e-12 * scale
                    ):
                        raise ValueError(
                            f"PIO {prefix.upper()}Start/{prefix.upper()}Step/"
                            f"{prefix.upper()}Stop are inconsistent with {size} samples"
                        )
                return values
            if start_f is not None and stop_f is not None and size > 1:
                return np.linspace(start_f, stop_f, size)
            if size == 1 and start_f is not None:
                return np.asarray([start_f], dtype=float)
            raise ValueError(f"Could not reconstruct {prefix} axis from PIO header")

        xvals = _build_axis("x", int(xsize))
        yvals = _build_axis("y", int(ysize))

        xname = (header.get("xname") or "").strip().lower()
        yname = (header.get("yname") or "").strip().lower()
        if not (xname in ("azimuth", "position") and yname == "frequency"):
            raise ValueError(
                f"Unsupported PIO axes (xname={xname!r}, yname={yname!r}); "
                "expected azimuth/position vs frequency"
            )

        xunit_raw = header.get("xunits")
        if xunit_raw is None or not str(xunit_raw).strip():
            raise ValueError(
                "PIO header missing XUnits; azimuth values cannot be safely "
                "interpreted as degrees or radians"
            )
        xunit = cls._canonical_unit(xunit_raw, _ANGLE_UNITS, "deg")
        if xunit not in {"deg", "rad"}:
            raise ValueError(
                f"Unsupported PIO azimuth unit: {header.get('xunits')!r}; "
                "expected degrees or radians"
            )

        closing_azimuth = float(xvals[-1])
        opening_azimuth = float(xvals[0])
        xvals, dropped_closing_azimuth = _pio_remove_closed_azimuth_endpoint(
            xvals, xunit
        )

        descending_axes = {}
        for axis_name, values in (("X", xvals), ("Y", yvals)):
            if np.any(~np.isfinite(values)):
                raise ValueError(f"PIO {axis_name} axis contains nonfinite coordinates")
            differences = np.diff(values)
            increasing = bool(values.size <= 1 or np.all(differences > 0.0))
            descending = bool(values.size > 1 and np.all(differences < 0.0))
            if not increasing and not descending:
                raise ValueError(
                    f"PIO {axis_name} axis must be strictly monotonic without duplicates"
                )
            descending_axes[axis_name] = descending

        if data_type == "complex":
            real_samples = rawdata[0::2]
            imag_samples = rawdata[1::2]
            finite_pair = np.isfinite(real_samples) & np.isfinite(imag_samples)
            missing_pair = np.isnan(real_samples) & np.isnan(imag_samples)
            if np.any(~(finite_pair | missing_pair)):
                raise ValueError(
                    "PIO complex data contains an infinite or one-sided missing sample"
                )
            complex_arr = real_samples.astype(np.float64) + 1j * imag_samples.astype(np.float64)
        else:
            if np.any(np.isinf(rawdata)):
                raise ValueError("PIO real data contains an infinite sample")
            complex_arr = rawdata.astype(np.complex128)


        complex_dtype = np.complex128 if precision == "double" else np.complex64
        data_2d = np.asarray(complex_arr, dtype=complex_dtype).reshape(
            (int(xsize), int(ysize)), order="F"
        )
        if dropped_closing_azimuth:
            data_2d = data_2d[:-1, :]


        if descending_axes["X"]:
            xvals = xvals[::-1].copy()
            data_2d = data_2d[::-1, :]
        if descending_axes["Y"]:
            yvals = yvals[::-1].copy()
            data_2d = data_2d[:, ::-1]

        yunit_raw = header.get("yunits")
        if yunit_raw is None or not str(yunit_raw).strip():
            raise ValueError(
                "PIO header missing YUnits; frequency values cannot be safely "
                "interpreted as Hz, kHz, MHz, or GHz"
            )
        yunit = cls._canonical_unit(yunit_raw, _FREQUENCY_UNITS, "GHz")
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
        if np.any(~np.isfinite(freqs_ghz)) or np.any(freqs_ghz <= 0.0):
            raise ValueError("PIO frequency axis must contain positive finite values")

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
        if elevation_raw is not None and (
            elevation_unit_raw is None or not str(elevation_unit_raw).strip()
        ):
            raise ValueError(
                "PIO contains Elevation but no ElevationUnits; the angle cannot "
                "be safely interpreted"
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
        if not np.isfinite(elevation_deg):
            raise ValueError("PIO elevation must be finite")

        pol = (header.get("polarity") or footer.get("polarity") or "").strip().upper()
        if not pol:
            stem = os.path.splitext(os.path.basename(str(path)))[0].upper()
            for tag in ("HH", "VV", "VH", "HV"):
                if tag in stem:
                    pol = tag
                    break
        if not pol:
            pol = "NA"
        if any(ord(character) < 32 or ord(character) == 127 for character in pol):
            raise ValueError("PIO polarity must not contain control characters")

        azimuths = np.asarray(xvals, dtype=float)
        if xunit == "rad":
            azimuths = np.rad2deg(azimuths)
        elevations = np.asarray([elevation_deg], dtype=float)
        polarizations = np.asarray([pol], dtype=object)

        rcs_arr = data_2d[:, np.newaxis, :, np.newaxis]

        prior_log = header.get("log") or footer.get("log") or ""
        history_parts = [f"Loaded Pioneer file: {path}"]
        if dropped_closing_azimuth:
            history_parts.append(
                "removed repeated closing azimuth "
                f"{closing_azimuth:.12g} {xunit} for opening azimuth "
                f"{opening_azimuth:.12g} {xunit}"
            )
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
        selected on-disk precision.  If the input grid itself contains a
        repeated full-turn closing azimuth, export omits that closing row using
        the same angle-independent rule as the loader.

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
        from grim_backend.datasets.constants import _ANGLE_UNITS, _FREQUENCY_UNITS, _PIO_WRITE_BLOCK_CELLS
        from grim_backend.io.pioneer import _pio_ascii_metadata, _pio_remove_closed_azimuth_endpoint
        if self.angular_coordinate_system() != "conic":
            raise ValueError(
                "save_pio: Pioneer azimuth/elevation output cannot represent "
                f"{self.angular_coordinate_system()!r} angular coordinates; "
                "retain .grim or use PTM for a great-circle cut"
            )
        quantity = self.linear_quantity()
        if quantity != "sigma_3d":
            if quantity == "sigma_2d":
                remedy = (
                    "Convert 2-D sigma_2d/dBke data to a physically defined "
                    "3-D quantity before export."
                )
            elif quantity == "power_ratio":
                remedy = (
                    "This relative/dimensionless response has no established "
                    "absolute 3-D RCS normalization; retain .grim/CSV or provide "
                    "a reviewed conversion to sigma_3d first."
                )
            else:
                remedy = (
                    "Establish and record an absolute sigma_3d/dBsm "
                    "normalization before export."
                )
            raise ValueError(
                "save_pio: Pioneer output requires a sigma_3d RCS dataset; "
                f"got {quantity!r}. {remedy}"
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
        for axis_name, values in (
            ("azimuth", azimuths),
            ("frequency", frequencies),
        ):
            if values.ndim != 1 or values.size == 0:
                raise ValueError(
                    f"save_pio: {axis_name} axis must be a nonempty 1-D array"
                )
            if np.any(~np.isfinite(values)):
                raise ValueError(
                    f"save_pio: {axis_name} axis contains nonfinite coordinates"
                )

        xunits = self._canonical_unit(
            (self.units or {}).get("azimuth"), _ANGLE_UNITS, "deg"
        )
        if xunits not in {"deg", "rad"}:
            raise ValueError(
                "save_pio: azimuth unit must be degrees or radians; got "
                f"{(self.units or {}).get('azimuth')!r}"
            )
        azimuths, dropped_closing_azimuth = _pio_remove_closed_azimuth_endpoint(
            azimuths, xunits
        )
        source_azimuth_slice = (
            slice(None, -1) if dropped_closing_azimuth else slice(None)
        )
        xsize = int(azimuths.size)
        ysize = int(frequencies.size)
        for axis_name, values in (
            ("azimuth", azimuths),
            ("frequency", frequencies),
        ):
            differences = np.diff(values)
            if values.size > 1 and not (
                np.all(differences > 0.0) or np.all(differences < 0.0)
            ):
                raise ValueError(
                    f"save_pio: {axis_name} axis must be strictly monotonic"
                )
        if np.any(frequencies <= 0.0):
            raise ValueError(
                "save_pio: frequency axis must contain positive coordinates"
            )


        power_slice = self.rcs_power[source_azimuth_slice, el_idx, :, pol_idx]
        phase_slice = self.rcs_phase[source_azimuth_slice, el_idx, :, pol_idx]
        phase_missing = np.isfinite(power_slice) & ~np.isfinite(phase_slice)
        if np.any(phase_missing):
            raise ValueError(
                "save_pio: complex PIO export requires phase for every finite-power "
                f"sample; {int(np.count_nonzero(phase_missing))} sample(s) lack phase"
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
        pol_label = _pio_ascii_metadata(
            str(self.polarizations[pol_idx]) if len(self.polarizations) else ""
        )
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


            return format(float(value), ".17g")

        def _vals(arr):
            return ":".join(_pio_number(v) for v in arr)

        name_field = _pio_ascii_metadata(
            os.path.splitext(os.path.basename(path))[0]
        )
        info_field = _pio_ascii_metadata(self.history)

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

        try:
            header_blob = ("\n".join(header_lines) + "\n").encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "save_pio: Pioneer header fields must be ASCII; an internal "
                "metadata field was not normalized"
            ) from exc


        offset_line_bytes = 18
        data_offset = len(header_blob) + offset_line_bytes
        offset_line = f"Offset={data_offset:010d}\n".encode("ascii")
        if len(offset_line) != offset_line_bytes:
            raise RuntimeError(
                f"save_pio: offset line width drift ({len(offset_line)} != {offset_line_bytes})"
            )

        directory = os.path.dirname(os.path.abspath(path)) or os.curdir
        fd, stage_path = tempfile.mkstemp(
            prefix=".pio-write-", suffix=".staging", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as f:
                fd = -1
                f.write(header_blob)
                f.write(offset_line)


                frequency_block = max(
                    1, _PIO_WRITE_BLOCK_CELLS // max(1, xsize)
                )
                for start in range(0, ysize, frequency_block):
                    stop = min(ysize, start + frequency_block)
                    complex_block = np.asarray(
                        self.rcs_slice(
                            (
                                source_azimuth_slice,
                                el_idx,
                                slice(start, stop),
                                pol_idx,
                            )
                        )
                    )
                    expected_block_shape = (xsize, stop - start)
                    if complex_block.shape != expected_block_shape:
                        raise ValueError(
                            "save_pio: slice block shape "
                            f"{complex_block.shape} != {expected_block_shape}"
                        )
                    interleaved = np.empty(
                        (stop - start, xsize, 2), dtype=dtype
                    )
                    transposed = complex_block.T
                    interleaved[..., 0] = transposed.real
                    interleaved[..., 1] = transposed.imag
                    f.write(interleaved)
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
