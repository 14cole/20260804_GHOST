from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .compute import (
    INCH_TO_M,
    LoadedLayer,
    compute_stack_impedance_many,
    validate_sweep_coverage,
)
from .io import write_impedance_batch


MAX_IBC_BATCH_FILES = 1000
# One million exported frequency rows is roughly tens of megabytes on disk and
# keeps a desktop batch bounded even when the user combines many thicknesses
# with an extremely fine frequency increment.
MAX_IBC_BATCH_TOTAL_POINTS = 1_000_000
THICKNESS_UNITS = ("mil", "in", "mm")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class IbcBatchItem:
    """One planned nominal IBC output and its selected-layer thickness."""

    thickness_value: Decimal
    thickness_unit: str
    thickness_in: float
    path: Path


def ibc_batch_frequency_count(
    start_ghz: float,
    stop_ghz: float,
    step_ghz: float,
) -> int:
    """Return a validated frequency count without allocating the sweep."""

    if not all(
        math.isfinite(value) for value in (start_ghz, stop_ghz, step_ghz)
    ):
        raise ValueError("Frequency start, stop, and step must be finite.")
    if start_ghz <= 0:
        raise ValueError("Frequency start must be > 0 GHz.")
    if step_ghz <= 0:
        raise ValueError("Frequency step must be > 0 GHz.")
    if stop_ghz < start_ghz:
        raise ValueError("Frequency stop must be >= start.")
    count = int(math.floor((stop_ghz - start_ghz) / step_ghz + 1e-12)) + 1
    if count <= 0:
        raise ValueError("Frequency sweep is empty.")
    return count


def validate_ibc_batch_workload(
    file_count: int,
    frequency_count: int,
    *,
    max_total_points: int = MAX_IBC_BATCH_TOTAL_POINTS,
) -> int:
    """Bound total exported rows before allocating or computing a batch."""

    if file_count <= 0:
        raise ValueError("IBC thickness batch is empty.")
    if frequency_count <= 0:
        raise ValueError("IBC batch frequency sweep is empty.")
    if max_total_points <= 0:
        raise ValueError("Maximum IBC batch point count must be > 0.")
    total_points = file_count * frequency_count
    if total_points > max_total_points:
        raise ValueError(
            "IBC batch would export "
            f"{total_points:,} total frequency rows "
            f"({file_count:,} files x {frequency_count:,} points); the safe "
            f"desktop limit is {max_total_points:,}. Use fewer thicknesses or "
            "a coarser frequency step."
        )
    return total_points


def _decimal(value: object, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not parsed.is_finite():
        raise ValueError(f"{label} must be finite.")
    return parsed


def _canonical_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _filename_value(value: Decimal) -> str:
    return _canonical_decimal(value).replace(".", "p")


def _thickness_to_inches(value: Decimal, unit: str) -> float:
    if unit == "mil":
        return float(value / Decimal("1000"))
    if unit == "in":
        return float(value)
    if unit == "mm":
        return float(value / Decimal("25.4"))
    raise ValueError(
        f"Unsupported thickness unit {unit!r}; choose {', '.join(THICKNESS_UNITS)}."
    )


def plan_ibc_thickness_batch(
    output_directory: Path | str,
    file_prefix: str,
    start: object,
    stop: object,
    step: object,
    unit: str,
    *,
    max_files: int = MAX_IBC_BATCH_FILES,
) -> list[IbcBatchItem]:
    """Plan deterministic, collision-free filenames for a thickness sweep.

    The stop value is included when it lies on the requested step grid;
    otherwise the final value is the last grid point below stop.
    """

    directory_text = str(output_directory).strip()
    if not directory_text:
        raise ValueError("Choose an output directory.")
    directory = Path(directory_text).expanduser()
    if not directory.exists():
        raise ValueError(f"Output directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Output directory is not a directory: {directory}")

    prefix = str(file_prefix).strip()
    if not prefix:
        raise ValueError("File prefix is required.")
    if len(prefix) > 80 or _PREFIX_RE.fullmatch(prefix) is None:
        raise ValueError(
            "File prefix must be 1-80 letters, numbers, hyphens, or underscores "
            "and must start with a letter or number. Do not include .csv."
        )

    normalized_unit = str(unit).strip().lower()
    if normalized_unit not in THICKNESS_UNITS:
        raise ValueError(
            f"Unsupported thickness unit {unit!r}; choose {', '.join(THICKNESS_UNITS)}."
        )

    start_value = _decimal(start, "Thickness start")
    stop_value = _decimal(stop, "Thickness stop")
    step_value = _decimal(step, "Thickness step")
    if start_value <= 0:
        raise ValueError("Thickness start must be > 0.")
    if stop_value < start_value:
        raise ValueError("Thickness stop must be >= start.")
    if step_value <= 0:
        raise ValueError("Thickness step must be > 0.")
    if max_files <= 0:
        raise ValueError("Maximum batch file count must be > 0.")

    count = int((stop_value - start_value) // step_value) + 1
    if count > max_files:
        raise ValueError(
            f"Thickness sweep would create {count} files; the limit is {max_files}."
        )

    items: list[IbcBatchItem] = []
    seen_paths: set[str] = set()
    for index in range(count):
        value = start_value + index * step_value
        filename = f"{prefix}_{_filename_value(value)}{normalized_unit}.csv"
        path = directory / filename
        # Fail closed for a release folder that may be planned on a
        # case-sensitive host and then copied to Windows.
        identity = os.path.normcase(str(path.resolve(strict=False))).casefold()
        if identity in seen_paths:
            raise ValueError(f"Thickness naming produced duplicate output {path}.")
        seen_paths.add(identity)
        thickness_in = _thickness_to_inches(value, normalized_unit)
        if not math.isfinite(thickness_in) or thickness_in <= 0:
            raise ValueError(
                f"Thickness {value} {normalized_unit} is outside the supported range."
            )
        items.append(
            IbcBatchItem(
                thickness_value=value,
                thickness_unit=normalized_unit,
                thickness_in=thickness_in,
                path=path,
            )
        )
    return items


def export_pec_ibc_thickness_batch(
    items: list[IbcBatchItem],
    loaded_layers: list[LoadedLayer],
    layer_index: int,
    frequencies_ghz: list[float],
    *,
    max_total_points: int = MAX_IBC_BATCH_TOTAL_POINTS,
) -> int:
    """Stage one computed CSV at a time, then atomically publish the batch."""

    if not items:
        raise ValueError("IBC thickness batch is empty.")
    validate_ibc_batch_workload(
        len(items),
        len(frequencies_ghz),
        max_total_points=max_total_points,
    )
    if layer_index < 0 or layer_index >= len(loaded_layers):
        raise ValueError("Selected IBC batch layer is unavailable.")
    selected = loaded_layers[layer_index]
    if selected.is_sheet:
        raise ValueError("Resistive sheet layers do not have a sweepable thickness.")

    for index, layer in enumerate(loaded_layers, start=1):
        if layer.is_sheet:
            continue
        if layer.table_0deg is None:
            raise ValueError(f"Layer {index} is missing its material table.")
        validate_sweep_coverage(
            frequencies_ghz, layer.table_0deg, f"layer {index} 0 deg/isotropic"
        )
        if layer.anisotropic:
            if layer.table_90deg is None:
                raise ValueError(
                    f"Layer {index}: anisotropic layer is missing a 90 deg table."
                )
            validate_sweep_coverage(
                frequencies_ghz, layer.table_90deg, f"layer {index} 90 deg"
            )

    def outputs():
        for item in items:
            if item.thickness_in <= 0:
                raise ValueError("Every IBC batch thickness must be > 0 in.")
            stack = list(loaded_layers)
            stack[layer_index] = replace(
                selected, thickness_m=item.thickness_in * INCH_TO_M
            )
            impedance = compute_stack_impedance_many(
                frequencies_ghz, stack, "pec"
            )
            rows = [
                (frequency, value.real, value.imag)
                for frequency, value in zip(frequencies_ghz, impedance)
            ]
            yield item.path, rows
            # The downstream writer has staged this file before resuming the
            # generator. Release the computed vector and row table before the
            # next stack solve starts so adjacent outputs never overlap in RAM.
            del impedance, rows, stack

    # The shared writer consumes this generator one file at a time. No
    # destination is published until every thickness has computed and staged;
    # a later failure removes all stages or rolls back the complete publication.
    write_impedance_batch(outputs(), include_header=True)
    return len(items)
