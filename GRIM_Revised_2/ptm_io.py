"""Strict reader and writer for the legacy 291-byte RCS ``.ptm`` format.

The format has no magic number.  Its first 291 bytes are a fixed header,
padded to the size of one complex-frequency data block.  That padded header
block is followed by one equally sized block per aspect angle.  All numeric
values in a file use the same byte order.

This module deliberately contains only format mechanics.  Mapping PTM cuts
onto :class:`grim_dataset.RcsGrid` lives in ``grim_dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import os
import struct
import tempfile

import numpy as np


PTM_HEADER_SIZE = 291
PTM_MAX_ASPECTS = 100_000
PTM_MAX_FREQUENCIES = 10_000_000
PTM_MAX_COMPLEX_SAMPLES = 100_000_000
PTM_WRITABLE_POLARIZATIONS = frozenset({"VV", "HH", "VH", "HV"})

_INT_FIELDS = (
    "num_aspects",
    "corecell",
    "calcell",
    "range_corr",
    "gatemincell",
    "gatemaxcell",
    "cal_type",
    "crosscell",
    "phase_flag",
)
_FLOAT_FIELDS = (
    "start_aspect",
    "aspect_increment",
    "center_frequency",
    "bandwidth",
    "roll",
    "pitch",
    "tilt",
    "cal_dbsm",
    "attenuation",
)


@dataclass(frozen=True)
class PtmHeader:
    """Decoded PTM header plus the frequency count implied by file framing."""

    num_aspects: int = 0
    corecell: int = 0
    calcell: int = 0
    range_corr: int = 0
    gatemincell: int = 0
    gatemaxcell: int = 0
    cal_type: int = 0
    crosscell: int = 0
    phase_flag: int = 0

    start_aspect: float = 0.0
    aspect_increment: float = 0.0
    center_frequency: float = 0.0
    bandwidth: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    tilt: float = 0.0
    cal_dbsm: float = 0.0
    attenuation: float = 0.0

    polarity: str = "NA"
    subject: str = ""
    configuration: str = ""
    operator: str = ""
    filename: str = ""
    date: str = ""
    time: str = ""

    num_frequencies: int = 0
    embedded_num_frequencies: bool = True
    byte_order: str = "little"


@dataclass(frozen=True)
class PtmData:
    """A parsed PTM cut in acquisition order."""

    header: PtmHeader
    aspects_deg: np.ndarray
    frequencies_ghz: np.ndarray
    iq: np.ndarray


def _decode_ascii(raw: bytes, field_name: str) -> str:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"PTM {field_name} is not ASCII") from exc
    return text.rstrip("\x00 *")


def _encode_ascii(value: object, width: int, field_name: str) -> bytes:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"PTM {field_name} must contain ASCII characters only") from exc
    if len(raw) > width:
        raw = raw[:width]
    return raw.ljust(width, b"*")


def _encode_filename(value: object) -> bytes:
    """Match the legacy writer's first-45/last-5 filename truncation."""

    text = str(value or "").replace("\r", " ").replace("\n", " ")
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("PTM filename must contain ASCII characters only") from exc
    if len(raw) > 50:
        raw = raw[:45] + raw[-5:]
    return raw.ljust(50, b"*")


def _validate_polarity(value: object) -> str:
    polarity = str(value or "").strip().upper()
    if len(polarity) != 2 or not polarity.isascii() or not polarity.isalnum():
        raise ValueError(
            f"PTM polarity must be exactly two ASCII letters/digits, got {value!r}"
        )
    return polarity


def _framing_for_size(file_size: int, num_aspects: int) -> tuple[int, int]:
    if not 1 <= int(num_aspects) <= PTM_MAX_ASPECTS:
        raise ValueError(
            f"num_aspects={num_aspects} is outside 1..{PTM_MAX_ASPECTS}"
        )
    block_size, remainder = divmod(int(file_size), int(num_aspects) + 1)
    if remainder:
        raise ValueError(
            f"file size {file_size} is not divisible by num_aspects+1 "
            f"({int(num_aspects) + 1})"
        )
    if block_size < PTM_HEADER_SIZE:
        raise ValueError(
            f"block size {block_size} is smaller than the {PTM_HEADER_SIZE}-byte header"
        )
    if block_size % 8:
        raise ValueError(f"block size {block_size} is not a whole complex-float32 block")
    num_frequencies = block_size // 8
    if not 1 <= num_frequencies <= PTM_MAX_FREQUENCIES:
        raise ValueError(
            f"num_frequencies={num_frequencies} is outside "
            f"1..{PTM_MAX_FREQUENCIES}"
        )
    sample_count = int(num_aspects) * int(num_frequencies)
    if sample_count > PTM_MAX_COMPLEX_SAMPLES:
        raise ValueError(
            f"PTM contains {sample_count} complex samples; safety limit is "
            f"{PTM_MAX_COMPLEX_SAMPLES}"
        )
    return block_size, num_frequencies


def _parse_header_candidate(
    raw_header: bytes,
    file_size: int,
    byte_order: str,
) -> tuple[PtmHeader, int]:
    prefix = "<" if byte_order == "little" else ">"
    numeric = struct.unpack_from(f"{prefix}9i9f", raw_header, 0)
    int_values = numeric[:9]
    float_values = numeric[9:]
    num_aspects = int(int_values[0])
    block_size, num_frequencies = _framing_for_size(file_size, num_aspects)

    if not np.all(np.isfinite(np.asarray(float_values, dtype=float))):
        raise ValueError("numeric header contains a non-finite float")
    center_frequency = float(float_values[2])
    bandwidth = float(float_values[3])
    if center_frequency <= 0.0:
        raise ValueError(f"center_frequency={center_frequency!r} must be positive")
    if bandwidth < 0.0:
        raise ValueError(f"bandwidth={bandwidth!r} must be non-negative")
    if num_frequencies > 1 and bandwidth == 0.0:
        raise ValueError(
            "bandwidth must be positive when PTM contains more than one "
            "frequency sample"
        )
    lowest_frequency = center_frequency - bandwidth / 2.0
    if lowest_frequency <= 0.0:
        raise ValueError(
            "center_frequency - bandwidth/2 must be positive; reconstructed "
            f"lowest frequency is {lowest_frequency!r} GHz"
        )

    polarity = _validate_polarity(_decode_ascii(raw_header[72:74], "polarity"))
    subject = _decode_ascii(raw_header[74:124], "subject")
    configuration = _decode_ascii(raw_header[124:174], "configuration")

    operator_raw = raw_header[174:224]
    embedded_count = struct.unpack(f"{prefix}i", operator_raw[46:50])[0]
    has_embedded_count = embedded_count == num_frequencies
    operator_width = 46 if has_embedded_count else 50
    operator = _decode_ascii(operator_raw[:operator_width], "operator")

    values = dict(zip(_INT_FIELDS, (int(v) for v in int_values)))
    values.update(zip(_FLOAT_FIELDS, (float(v) for v in float_values)))
    header = PtmHeader(
        **values,
        polarity=polarity,
        subject=subject,
        configuration=configuration,
        operator=operator,
        filename=_decode_ascii(raw_header[224:274], "filename"),
        date=_decode_ascii(raw_header[274:283], "date"),
        time=_decode_ascii(raw_header[283:291], "time"),
        num_frequencies=num_frequencies,
        embedded_num_frequencies=has_embedded_count,
        byte_order=byte_order,
    )
    return header, block_size


def read_ptm(path: os.PathLike[str] | str) -> PtmData:
    """Read one PTM cut, accepting little- or big-endian numeric storage."""

    path = os.fspath(path)
    file_size = os.path.getsize(path)
    if file_size < PTM_HEADER_SIZE:
        raise ValueError(
            f"{path}: too small to be PTM ({file_size} < {PTM_HEADER_SIZE} bytes)"
        )
    with open(path, "rb") as stream:
        raw_header = stream.read(PTM_HEADER_SIZE)
    if len(raw_header) != PTM_HEADER_SIZE:
        raise ValueError(f"{path}: truncated PTM header")

    candidates: list[tuple[PtmHeader, int]] = []
    errors: list[str] = []
    for byte_order in ("little", "big"):
        try:
            candidates.append(_parse_header_candidate(raw_header, file_size, byte_order))
        except (ValueError, struct.error) as exc:
            errors.append(f"{byte_order}: {exc}")
    if not candidates:
        raise ValueError(f"{path}: invalid PTM framing/header ({'; '.join(errors)})")

    # The legacy writer is native-endian on Windows and the legacy reader tries
    # little-endian first.  Preserve that deterministic preference in the very
    # unlikely event that both interpretations satisfy every structural check.
    header, block_size = candidates[0]
    prefix = "<" if header.byte_order == "little" else ">"
    sample_count = header.num_aspects * header.num_frequencies
    with open(path, "rb") as stream:
        stream.seek(block_size)
        iq = np.fromfile(stream, dtype=np.dtype(f"{prefix}c8"), count=sample_count)
    if iq.size != sample_count:
        raise ValueError(
            f"{path}: expected {sample_count} complex samples, found {iq.size}"
        )
    iq = np.asarray(iq, dtype=np.complex64).reshape(
        header.num_aspects, header.num_frequencies
    )
    if not np.all(np.isfinite(iq.real) & np.isfinite(iq.imag)):
        raise ValueError(
            f"{path}: PTM IQ contains a non-finite sample; the format has no "
            "documented missing-sample marker"
        )

    raw_aspects = (
        float(header.start_aspect)
        + np.arange(header.num_aspects, dtype=np.float64)
        * float(header.aspect_increment)
    )
    # GRIM interpolation assumes monotonic axes.  PTM acquisition order may
    # cross the +/-180-degree seam, so canonicalize to [-180, 180), reject
    # aliases/duplicates, and carry the IQ rows through the same stable sort.
    aspects = np.mod(raw_aspects + 180.0, 360.0) - 180.0
    aspect_order = np.argsort(aspects, kind="stable")
    aspects = aspects[aspect_order]
    duplicate_tol = max(
        1.0e-7,
        8.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(aspects)))),
    )
    if aspects.size > 1 and np.any(np.diff(aspects) <= duplicate_tol):
        raise ValueError(
            f"{path}: PTM aspect axis contains duplicate/seam-alias coordinates "
            "after wrapping to [-180, 180)"
        )
    iq = iq[aspect_order, :]
    start_frequency = header.center_frequency - header.bandwidth / 2.0
    stop_frequency = header.center_frequency + header.bandwidth / 2.0
    frequencies = np.linspace(
        start_frequency,
        stop_frequency,
        header.num_frequencies,
        dtype=np.float64,
    )
    return PtmData(header, aspects, frequencies, iq)


def _uniform_axis(values: np.ndarray, name: str, *, periodic: bool = False) -> tuple[float, float]:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"PTM {name} axis must be a non-empty 1-D array")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"PTM {name} axis contains a non-finite value")
    working = np.rad2deg(np.unwrap(np.deg2rad(axis))) if periodic else axis
    if working.size == 1:
        return float(working[0]), 0.0
    step = float((working[-1] - working[0]) / (working.size - 1))
    expected = float(working[0]) + step * np.arange(working.size, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(working))))
    atol = max(1.0e-7, 8.0 * np.finfo(np.float32).eps * scale)
    if not np.allclose(working, expected, rtol=1.0e-7, atol=atol):
        max_error = float(np.max(np.abs(working - expected)))
        raise ValueError(
            f"PTM {name} axis must be uniformly spaced; maximum residual is {max_error:g}"
        )
    return float(working[0]), step


def _periodic_aspect_order(
    values: np.ndarray,
    preferred_header: PtmHeader | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return a uniform acquisition ordering for one set of aspect samples."""

    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("PTM aspect axis must be a non-empty 1-D array")
    if not np.all(np.isfinite(axis)):
        raise ValueError("PTM aspect axis contains a non-finite value")
    canonical = np.mod(axis + 180.0, 360.0) - 180.0
    sorted_values = np.sort(canonical, kind="stable")
    tol = max(
        1.0e-7,
        8.0 * np.finfo(np.float32).eps
        * max(1.0, float(np.max(np.abs(canonical)))),
    )
    if canonical.size > 1 and np.any(np.diff(sorted_values) <= tol):
        raise ValueError(
            "PTM aspect axis contains duplicate coordinates or a -180/180 seam alias"
        )

    def _sequence_result(order):
        ordered = canonical[np.asarray(order, dtype=int)]
        try:
            start, step = _uniform_axis(ordered, "aspect", periodic=True)
        except ValueError:
            return None
        if ordered.size > 1 and abs(step) <= tol:
            return None
        return np.asarray(order, dtype=int), ordered, start, step

    # A PTM-origin grid is sorted on load, but the original start/increment are
    # retained.  Prefer that exact acquisition ordering when the current grid
    # still contains the same axis set.
    if (
        preferred_header is not None
        and int(preferred_header.num_aspects) == canonical.size
        and canonical.size > 0
    ):
        preferred_raw = (
            float(preferred_header.start_aspect)
            + np.arange(canonical.size, dtype=np.float64)
            * float(preferred_header.aspect_increment)
        )
        preferred = np.mod(preferred_raw + 180.0, 360.0) - 180.0
        order = []
        used = set()
        for target in preferred:
            matches = np.where(np.isclose(canonical, target, rtol=0.0, atol=tol))[0]
            matches = [int(index) for index in matches if int(index) not in used]
            if len(matches) != 1:
                order = []
                break
            used.add(matches[0])
            order.append(matches[0])
        if len(order) == canonical.size:
            result = _sequence_result(order)
            if result is not None:
                return (
                    result[0],
                    result[1],
                    float(preferred_header.start_aspect),
                    float(preferred_header.aspect_increment),
                )

    result = _sequence_result(np.arange(canonical.size, dtype=int))
    if result is not None:
        return result

    # A seam-crossing cut sorted for GRIM has one large gap.  Rotate the sorted
    # set to start just after that gap, preserving every sample without
    # interpolation.  Ascending order is canonical for a generic grid.
    sorted_order = np.argsort(canonical, kind="stable")
    sorted_axis = canonical[sorted_order]
    gaps = np.diff(np.concatenate((sorted_axis, [sorted_axis[0] + 360.0])))
    rotation = (int(np.argmax(gaps)) + 1) % canonical.size
    rotated_order = np.concatenate((sorted_order[rotation:], sorted_order[:rotation]))
    result = _sequence_result(rotated_order)
    if result is None:
        raise ValueError("PTM aspect axis must be uniformly spaced around its scan arc")
    return result


def _header_blob(header: PtmHeader, num_frequencies: int) -> bytes:
    polarity = _validate_polarity(header.polarity)
    ints = tuple(int(getattr(header, name)) for name in _INT_FIELDS)
    floats = tuple(float(getattr(header, name)) for name in _FLOAT_FIELDS)
    if not np.all(np.isfinite(np.asarray(floats, dtype=float))):
        raise ValueError("PTM numeric header contains a non-finite float")

    blob = bytearray(struct.pack("<9i9f", *ints, *floats))
    blob.extend(_encode_ascii(polarity, 2, "polarity"))
    blob.extend(_encode_ascii(header.subject, 50, "subject"))
    blob.extend(_encode_ascii(header.configuration, 50, "configuration"))
    if header.embedded_num_frequencies:
        blob.extend(_encode_ascii(header.operator, 46, "operator"))
        blob.extend(struct.pack("<i", int(num_frequencies)))
    else:
        blob.extend(_encode_ascii(header.operator, 50, "operator"))
    blob.extend(_encode_filename(header.filename))
    blob.extend(_encode_ascii(header.date, 9, "date"))
    blob.extend(_encode_ascii(header.time, 8, "time"))
    if len(blob) != PTM_HEADER_SIZE:
        raise RuntimeError(f"PTM header width drift ({len(blob)} != {PTM_HEADER_SIZE})")
    return bytes(blob)


def write_ptm(
    path: os.PathLike[str] | str,
    aspects_deg,
    frequencies_ghz,
    iq,
    *,
    polarity: str,
    pitch_deg: float = 0.0,
    header: PtmHeader | None = None,
) -> str:
    """Write one uniformly sampled, complex, great-circle PTM cut.

    New files are always little-endian.  ``header`` supplies optional legacy
    acquisition metadata; axes, dimensions, polarization, pitch, and output
    filename are authoritative arguments and replace those header fields.
    """

    path = os.fspath(path)
    if not path.lower().endswith(".ptm"):
        path = f"{path}.ptm"

    aspects = np.asarray(aspects_deg, dtype=np.float64)
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64)
    base = header or PtmHeader()
    aspect_order, _, start_aspect, aspect_increment = (
        _periodic_aspect_order(aspects, base)
    )
    start_frequency, frequency_increment = _uniform_axis(
        frequencies, "frequency", periodic=False
    )
    if frequencies.size < 37:
        raise ValueError(
            "PTM requires at least 37 frequencies because its 291-byte header "
            "must fit inside one 8-byte-per-frequency block"
        )
    if start_frequency <= 0.0 or np.any(frequencies <= 0.0):
        raise ValueError("PTM frequencies must be positive GHz values")
    if frequencies.size > 1 and frequency_increment <= 0.0:
        raise ValueError("PTM frequency axis must be strictly increasing")
    if aspects.size > PTM_MAX_ASPECTS:
        raise ValueError(
            f"PTM aspect count {aspects.size} exceeds safety limit {PTM_MAX_ASPECTS}"
        )
    if frequencies.size > PTM_MAX_FREQUENCIES:
        raise ValueError(
            f"PTM frequency count {frequencies.size} exceeds safety limit "
            f"{PTM_MAX_FREQUENCIES}"
        )
    sample_count = int(aspects.size) * int(frequencies.size)
    if sample_count > PTM_MAX_COMPLEX_SAMPLES:
        raise ValueError(
            f"PTM contains {sample_count} complex samples; safety limit is "
            f"{PTM_MAX_COMPLEX_SAMPLES}"
        )

    complex_data = np.asarray(iq)
    expected_shape = (int(aspects.size), int(frequencies.size))
    if complex_data.shape != expected_shape:
        raise ValueError(f"PTM IQ shape {complex_data.shape} != {expected_shape}")
    if not np.iscomplexobj(complex_data):
        raise ValueError("PTM IQ data must be complex")
    complex_data = np.asarray(complex_data, dtype=np.complex64)
    if not np.all(np.isfinite(complex_data.real) & np.isfinite(complex_data.imag)):
        raise ValueError(
            "PTM IQ data contains a non-finite sample; the format has no "
            "documented missing-sample marker"
        )
    complex_data = complex_data[aspect_order, :]

    now = datetime.now()
    center_frequency = float((frequencies[0] + frequencies[-1]) / 2.0)
    bandwidth = float(frequencies[-1] - frequencies[0])
    output_polarity = _validate_polarity(polarity)
    if output_polarity not in PTM_WRITABLE_POLARIZATIONS:
        supported = ", ".join(sorted(PTM_WRITABLE_POLARIZATIONS))
        raise ValueError(
            "PTM writer supports only the documented polarizations "
            f"{supported}; got {output_polarity!r}"
        )
    final_header = replace(
        base,
        num_aspects=int(aspects.size),
        start_aspect=start_aspect,
        aspect_increment=aspect_increment,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        pitch=float(pitch_deg),
        polarity=output_polarity,
        filename=os.path.basename(path),
        date=base.date or now.strftime("%d%b%Y").upper(),
        time=base.time or now.strftime("%H:%M:%S"),
        num_frequencies=int(frequencies.size),
        byte_order="little",
    )
    header_bytes = _header_blob(final_header, int(frequencies.size))
    block_size = 8 * int(frequencies.size)
    padding_size = block_size - PTM_HEADER_SIZE
    if padding_size < 0:
        raise RuntimeError("PTM internal framing error: negative header padding")

    expected_size = (int(aspects.size) + 1) * block_size
    destination_dir = os.path.dirname(os.path.abspath(path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_dir,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = stream.name
            stream.write(header_bytes)
            stream.write(b"\x00" * padding_size)
            for row in complex_data:
                stream.write(np.asarray(row, dtype=np.dtype("<c8")).tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        actual_size = os.path.getsize(temp_path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"PTM write size {actual_size} != expected {expected_size}"
            )
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
    return path


def header_to_extra(header: PtmHeader) -> dict[str, object]:
    """Flatten legacy header metadata into pickle-free RcsGrid extras."""

    out: dict[str, object] = {
        "angular_coordinate_system": "great_circle",
        "ptm_cut_type": "GC",
        "ptm_original_byte_order": header.byte_order,
        "ptm_original_num_aspects": int(header.num_aspects),
        "ptm_original_start_aspect": float(header.start_aspect),
        "ptm_original_aspect_increment": float(header.aspect_increment),
        "ptm_embedded_num_frequencies": bool(header.embedded_num_frequencies),
        "ptm_source_filename": header.filename,
    }
    for name in _INT_FIELDS[1:]:
        out[f"ptm_{name}"] = int(getattr(header, name))
    for name in ("roll", "tilt", "cal_dbsm", "attenuation"):
        out[f"ptm_{name}"] = float(getattr(header, name))
    for name in ("subject", "configuration", "operator", "date", "time"):
        out[f"ptm_{name}"] = str(getattr(header, name))
    return out


def _extra_scalar(extra: dict[str, object], key: str, default):
    value = extra.get(key, default)
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return default


def header_from_extra(extra: dict[str, object] | None) -> PtmHeader:
    """Rebuild preserved legacy metadata for a subsequent PTM export."""

    values = dict(extra or {})
    kwargs: dict[str, object] = {}
    kwargs["num_aspects"] = int(
        _extra_scalar(values, "ptm_original_num_aspects", 0)
    )
    kwargs["start_aspect"] = float(
        _extra_scalar(values, "ptm_original_start_aspect", 0.0)
    )
    kwargs["aspect_increment"] = float(
        _extra_scalar(values, "ptm_original_aspect_increment", 0.0)
    )
    for name in _INT_FIELDS[1:]:
        kwargs[name] = int(_extra_scalar(values, f"ptm_{name}", 0))
    for name in ("roll", "tilt", "cal_dbsm", "attenuation"):
        kwargs[name] = float(_extra_scalar(values, f"ptm_{name}", 0.0))
    for name in ("subject", "configuration", "operator", "date", "time"):
        kwargs[name] = str(_extra_scalar(values, f"ptm_{name}", ""))
    kwargs["embedded_num_frequencies"] = bool(
        _extra_scalar(values, "ptm_embedded_num_frequencies", True)
    )
    return PtmHeader(**kwargs)
