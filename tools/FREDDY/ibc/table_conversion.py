"""Streaming delimited tables, explicit column mappings and unit conversion.

This module is Qt-free and usable by standalone FREDDY. No units or column
positions are inferred from numeric values.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile


UNIT_FACTORS = {
    "Hz": ("frequency", 1.0), "kHz": ("frequency", 1e3),
    "MHz": ("frequency", 1e6), "GHz": ("frequency", 1e9),
    "THz": ("frequency", 1e12),
    "deg": ("angle", math.pi / 180), "rad": ("angle", 1.0),
    "m": ("length", 1.0), "cm": ("length", .01),
    "mm": ("length", .001), "in": ("length", .0254),
    "ft": ("length", .3048),
    "s": ("time", 1.0), "ms": ("time", 1e-3),
    "us": ("time", 1e-6), "ns": ("time", 1e-9),
    "ohm": ("impedance", 1.0), "kohm": ("impedance", 1e3),
    "unitless": ("unitless", 1.0),
}
UNITS = ("As written", *UNIT_FACTORS)
DELIMITERS = {"Auto": None, "Comma": ",", "Whitespace": "whitespace",
              "Tab": "\t", "Semicolon": ";"}


@dataclass(frozen=True)
class TableOptions:
    delimiter: str = "Auto"
    header: str = "Auto"
    skip_rows: int = 0
    encoding: str = "utf-8-sig"


@dataclass(frozen=True)
class ColumnMapping:
    name: str
    source: int | None = None
    constant: str = ""
    input_unit: str = "As written"
    output_unit: str = "As written"


def number(text):
    return float(str(text).strip().replace("D", "E").replace("d", "e"))


def _numeric(text):
    try:
        number(text)
        return True
    except ValueError:
        return False


def _records(path, options):
    if options.skip_rows < 0:
        raise ValueError("Skip rows must be nonnegative.")
    if options.delimiter not in DELIMITERS or options.header not in ("Auto", "Yes", "No"):
        raise ValueError("Choose a supported delimiter and header setting.")
    delimiter = DELIMITERS[options.delimiter]
    with open(path, encoding=options.encoding, newline="") as stream:
        for line_no, line in enumerate(stream, 1):
            if "\x00" in line:
                raise ValueError("This file contains binary data; select a text table.")
            if line_no <= options.skip_rows:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%", "!", "//")):
                continue
            if delimiter is None:
                delimiter = next((d for d in (",", ";", "\t") if d in line), "whitespace")
                if delimiter == "\t" and any(
                    len(cell.split()) > 1 and all(_numeric(token) for token in cell.split())
                    for cell in stripped.split("\t")
                ):
                    delimiter = "whitespace"
            try:
                row = (stripped.split() if delimiter == "whitespace" else
                       next(csv.reader([line], delimiter=delimiter, strict=True)))
            except csv.Error as exc:
                raise ValueError(f"Line {line_no}: {exc}") from exc
            yield line_no, tuple(value.strip() for value in row)


def table_rows(path, options=TableOptions()):
    """Yield (line number, row); the first record contains column labels at line 0."""
    records = _records(path, options)
    try:
        line_no, first = next(records)
    except StopIteration:
        raise ValueError("The file contains no table rows.") from None
    has_header = options.header == "Yes" or (
        options.header == "Auto" and all(not _numeric(value) for value in first)
    )
    names = first if has_header else tuple(f"Column {i + 1}" for i in range(len(first)))
    yield 0, names
    count = 0
    if not has_header:
        yield line_no, first
        count += 1
    for line_no, row in records:
        if len(row) != len(names):
            raise ValueError(f"Line {line_no}: expected {len(names)} columns, found {len(row)}. "
                             "Check the delimiter, header and skipped lines.")
        yield line_no, row
        count += 1
    if not count:
        raise ValueError("The file contains a header but no data rows.")


def preview_table(path, options=TableOptions(), limit=20):
    rows = table_rows(path, options)
    try:
        _, names = next(rows)
        sample = []
        for line_no, row in rows:
            sample.append((line_no, row))
            if len(sample) >= limit:
                break
        return names, sample
    finally:
        rows.close()


def validate_mapping(columns, width):
    if not columns:
        raise ValueError("Select at least one output column.")
    names = set()
    for column in columns:
        name = column.name.strip()
        if not name or name.casefold() in names:
            raise ValueError("Output column names must be nonblank and unique.")
        names.add(name.casefold())
        if column.source is not None and not 0 <= column.source < width:
            raise ValueError(f"{name}: select a source column or a constant.")
        if column.source is None and not column.constant.strip():
            raise ValueError(f"{name}: enter a constant value for all rows.")
        src, dst = column.input_unit, column.output_unit
        if src == dst == "As written":
            continue
        if src not in UNIT_FACTORS or dst not in UNIT_FACTORS:
            raise ValueError(f"{name}: select both input and output units.")
        if UNIT_FACTORS[src][0] != UNIT_FACTORS[dst][0]:
            raise ValueError(f"{name}: cannot convert {src} to {dst}.")


def converted_rows(path, options, columns, expected_names=None):
    records = table_rows(path, options)
    try:
        _, names = next(records)
        if expected_names is not None and tuple(expected_names) != names:
            raise ValueError("The source columns changed. Refresh the preview and review the mapping.")
        validate_mapping(columns, len(names))
        for line_no, row in records:
            result = {}
            for column in columns:
                raw = column.constant if column.source is None else row[column.source]
                if column.input_unit == "As written":
                    value = raw
                else:
                    try:
                        value = number(raw) * (UNIT_FACTORS[column.input_unit][1] /
                                               UNIT_FACTORS[column.output_unit][1])
                    except (ValueError, OverflowError) as exc:
                        raise ValueError(f"Line {line_no}, {column.name}: invalid number {raw!r}.") from exc
                    if not math.isfinite(value):
                        raise ValueError(f"Line {line_no}, {column.name}: value must be finite.")
                result[column.name.strip()] = value
            yield line_no, result
    finally:
        records.close()


def output_header(column):
    unit = column.output_unit
    name = column.name.strip()
    if unit in ("As written", "unitless") or name.casefold().endswith("_" + unit.casefold()):
        return name
    name, _old_unit = suggest_column(name)
    return f"{name}_{unit.lower()}"


def suggest_column(label):
    """Use explicit header suffixes only; unitless headers remain unassigned."""
    for unit in UNIT_FACTORS:
        match = re.match(r"^(.*?)(?:[_\s]+|[\[(])" + re.escape(unit) + r"[\])]?$", label, re.I)
        if match and match[1].strip():
            return match[1].strip(), unit
    return label, "As written"


def export_table(path, destination, options, columns, *, delimiter=",", expected_names=None):
    """Validate every row and publish atomically; keep the source untouched."""
    source, target = Path(path).resolve(), Path(destination).resolve()
    if source == target or (target.exists() and os.path.samefile(source, target)):
        raise ValueError("Choose a different output file to keep the source unchanged.")
    if delimiter not in (",", "\t", " ", ";"):
        raise ValueError("Choose a supported output delimiter.")
    headers = [output_header(column) for column in columns]
    if len({name.casefold() for name in headers}) != len(headers):
        raise ValueError("Converted output headers must be unique.")
    fd, staging = tempfile.mkstemp(prefix=".converted-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=delimiter)
            writer.writerow(headers)
            count = 0
            for line_no, row in converted_rows(path, options, columns, expected_names):
                values = [row[column.name.strip()] for column in columns]
                if delimiter == " " and any(any(c.isspace() for c in str(v)) for v in values + headers):
                    raise ValueError(f"Line {line_no}: whitespace output needs single-token labels and values; use CSV.")
                writer.writerow([format(v, ".17g") if isinstance(v, float) else v for v in values])
                count += 1
        os.replace(staging, target)
        return count
    finally:
        if os.path.exists(staging):
            os.unlink(staging)
