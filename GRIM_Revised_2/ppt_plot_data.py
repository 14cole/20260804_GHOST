"""Physics-safe RCS cut extraction for GRIM PowerPoint reports.

This module deliberately has no Qt dependency.  The GUI supplies named
``RcsGrid`` objects and native-axis selector values; this layer validates the
datasets, extracts exact cuts without interpolation, and returns the
``PlotSpec`` objects shared by slide preview and PowerPoint export.

Selector values are always expressed in the selected datasets' stored units.
``angle_display_unit`` and ``frequency_display_unit`` affect only plot axes,
labels, and titles after the physical selections have been resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Literal, Sequence

import numpy as np

from grim_dataset import (
    GRIM_GC_CONVENTION,
    LEGACY_PTM_GC_CONVENTION,
    RcsGrid,
)
from ppt_report import PlotSeries, PlotSpec


Quantity = Literal["magnitude", "phase"]
AzimuthKind = Literal["azimuth_rect", "azimuth_polar"]

_ANGLE_UNITS = {
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
}
_FREQUENCY_UNITS = {
    "hz": "Hz",
    "khz": "kHz",
    "mhz": "MHz",
    "ghz": "GHz",
}
_FREQUENCY_TO_HZ = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}


@dataclass(frozen=True)
class NamedGrid:
    """One user-visible dataset name paired with its in-memory RCS grid."""

    name: str
    grid: RcsGrid

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Every PowerPoint dataset needs a nonblank name.")
        if not isinstance(self.grid, RcsGrid):
            raise TypeError("PowerPoint datasets must contain RcsGrid objects.")


@dataclass(frozen=True)
class PlotAvailability:
    """Exact common selector values and presentation capabilities.

    Numeric selector tuples use the native units named alongside them.  The
    swept axis for an individual series need not be shared; these intersections
    are the safe fixed-axis choices needed by both report families.
    """

    azimuths: tuple[float, ...]
    elevations: tuple[float, ...]
    frequencies: tuple[float, ...]
    polarizations: tuple[str, ...]
    azimuth_unit: str
    elevation_unit: str
    frequency_unit: str
    rcs_unit: str
    angular_coordinate_system: str
    phase_available: bool
    phase_reference: str
    phase_reason: str
    polar_available: bool
    polar_reason: str


def _coerce_named_grids(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
) -> tuple[NamedGrid, ...]:
    values: list[NamedGrid] = []
    for item in datasets:
        if isinstance(item, NamedGrid):
            values.append(item)
            continue
        try:
            name, grid = item
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Each dataset must be NamedGrid(name, grid) or a (name, grid) pair."
            ) from exc
        values.append(NamedGrid(str(name), grid))
    if not values:
        raise ValueError("Select at least one dataset for the PowerPoint report.")
    return tuple(values)


def _angle_unit(grid: RcsGrid, axis: str) -> str:
    raw = str((grid.units or {}).get(axis, "deg")).strip().lower()
    try:
        return _ANGLE_UNITS[raw]
    except KeyError as exc:
        raise ValueError(
            f"{axis} uses unsupported unit {raw!r}; expected degrees or radians."
        ) from exc


def _frequency_unit(grid: RcsGrid) -> str:
    raw = str((grid.units or {}).get("frequency", "GHz")).strip().lower()
    try:
        return _FREQUENCY_UNITS[raw]
    except KeyError as exc:
        raise ValueError(
            f"frequency uses unsupported unit {raw!r}; expected Hz, kHz, MHz, or GHz."
        ) from exc


def _display_angle_unit(value: str) -> str:
    raw = str(value).strip().lower()
    try:
        return _ANGLE_UNITS[raw]
    except KeyError as exc:
        raise ValueError("Angle display unit must be 'deg' or 'rad'.") from exc


def _display_frequency_unit(value: str) -> str:
    raw = str(value).strip().lower()
    try:
        return _FREQUENCY_UNITS[raw]
    except KeyError as exc:
        raise ValueError(
            "Frequency display unit must be Hz, kHz, MHz, or GHz."
        ) from exc


def _convert_angles(values, source_unit: str, target_unit: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if source_unit == target_unit:
        return result.copy()
    if source_unit == "deg" and target_unit == "rad":
        return np.deg2rad(result)
    if source_unit == "rad" and target_unit == "deg":
        return np.rad2deg(result)
    raise ValueError(f"Unsupported angle conversion: {source_unit} to {target_unit}.")


def _convert_frequencies(values, source_unit: str, target_unit: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return result * (_FREQUENCY_TO_HZ[source_unit] / _FREQUENCY_TO_HZ[target_unit])


def _format_value(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    return f"{value:.8g}"


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "plot"


def _assert_metadata_compatible(datasets: tuple[NamedGrid, ...]) -> None:
    reference = datasets[0]
    # Validate supported units even for a one-dataset report.
    _angle_unit(reference.grid, "azimuth")
    _angle_unit(reference.grid, "elevation")
    _frequency_unit(reference.grid)
    for dataset in datasets[1:]:
        try:
            reference.grid._assert_physical_metadata_compatible(dataset.grid)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset {dataset.name!r} is not physically compatible with "
                f"{reference.name!r}: {exc}"
            ) from exc
        _angle_unit(dataset.grid, "azimuth")
        _angle_unit(dataset.grid, "elevation")
        _frequency_unit(dataset.grid)


def _exact_index(
    axis,
    value,
    *,
    tol: float,
    axis_name: str,
    dataset_name: str,
) -> int:
    values = np.asarray(axis)
    if np.issubdtype(values.dtype, np.number):
        matches = np.flatnonzero(
            np.isclose(values.astype(float, copy=False), float(value), atol=tol, rtol=0.0)
        )
    else:
        matches = np.flatnonzero(values == value)
    if matches.size == 0:
        raise ValueError(
            f"{axis_name} value {value!r} is not present in dataset "
            f"{dataset_name!r} within tolerance {tol:g}; no interpolation is performed."
        )
    if matches.size > 1:
        raise ValueError(
            f"{axis_name} value {value!r} is ambiguous in dataset {dataset_name!r}: "
            f"{matches.size} samples fall within tolerance {tol:g}."
        )
    return int(matches[0])


def _numeric_intersection(axis_values: Sequence[np.ndarray], tol: float) -> tuple[float, ...]:
    if not axis_values:
        return ()
    first = np.asarray(axis_values[0], dtype=float).reshape(-1)
    common: list[float] = []
    for candidate in first:
        candidate = float(candidate)
        if any(math.isclose(candidate, old, rel_tol=0.0, abs_tol=tol) for old in common):
            continue
        if all(
            np.count_nonzero(
                np.isclose(np.asarray(axis, dtype=float), candidate, atol=tol, rtol=0.0)
            )
            == 1
            for axis in axis_values[1:]
        ):
            common.append(candidate)
    return tuple(sorted(common))


def _text_intersection(axis_values: Sequence[np.ndarray]) -> tuple[str, ...]:
    if not axis_values:
        return ()
    common: list[str] = []
    for value in np.asarray(axis_values[0]).reshape(-1):
        text = str(value)
        if text in common:
            continue
        if all(np.count_nonzero(np.asarray(axis).astype(str) == text) == 1 for axis in axis_values[1:]):
            common.append(text)
    return tuple(common)


def _phase_capability(datasets: tuple[NamedGrid, ...]) -> tuple[bool, str, str]:
    references = [dataset.grid._phase_reference() for dataset in datasets]
    if any(not reference for reference in references):
        missing = ", ".join(
            dataset.name
            for dataset, reference in zip(datasets, references)
            if not reference
        )
        return False, "", f"Phase reference is unspecified for: {missing}."
    if len(set(references)) != 1:
        return False, "", "Selected datasets have different phase references."
    missing_phase = [
        dataset.name
        for dataset in datasets
        if not np.any(np.isfinite(np.asarray(dataset.grid.rcs_phase, dtype=float)))
    ]
    if missing_phase:
        return (
            False,
            references[0],
            "No finite stored phase is available for: " + ", ".join(missing_phase) + ".",
        )
    return True, references[0], ""


def _polar_capability(datasets: tuple[NamedGrid, ...]) -> tuple[bool, str]:
    for dataset in datasets:
        system = dataset.grid.angular_coordinate_system()
        if system == "great_circle":
            convention = dataset.grid.great_circle_coordinate_convention()
            if convention == LEGACY_PTM_GC_CONVENTION:
                return (
                    False,
                    f"{dataset.name!r} is unmarked legacy great-circle data; its "
                    "azimuth sign/origin and polarization basis are not established.",
                )
            if convention != GRIM_GC_CONVENTION:
                return False, f"{dataset.name!r} uses unsupported great-circle convention {convention!r}."
        elif system != "conic":
            return False, f"{dataset.name!r} uses unsupported angular coordinate system {system!r}."
    return True, ""


def get_plot_availability(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
    *,
    tol: float = 1.0e-6,
) -> PlotAvailability:
    """Return exact common fixed-axis selectors for the selected overlays."""

    if not math.isfinite(float(tol)) or tol < 0.0:
        raise ValueError("Axis matching tolerance must be a finite nonnegative value.")
    selected = _coerce_named_grids(datasets)
    _assert_metadata_compatible(selected)
    reference = selected[0].grid
    phase_available, phase_reference, phase_reason = _phase_capability(selected)
    polar_available, polar_reason = _polar_capability(selected)
    return PlotAvailability(
        azimuths=_numeric_intersection([item.grid.azimuths for item in selected], tol),
        elevations=_numeric_intersection([item.grid.elevations for item in selected], tol),
        frequencies=_numeric_intersection([item.grid.frequencies for item in selected], tol),
        polarizations=_text_intersection([item.grid.polarizations for item in selected]),
        azimuth_unit=_angle_unit(reference, "azimuth"),
        elevation_unit=_angle_unit(reference, "elevation"),
        frequency_unit=_frequency_unit(reference),
        rcs_unit=reference.default_log_unit(),
        angular_coordinate_system=reference.angular_coordinate_system(),
        phase_available=phase_available,
        phase_reference=phase_reference,
        phase_reason=phase_reason,
        polar_available=polar_available,
        polar_reason=polar_reason,
    )


def _validate_quantity(quantity: str, availability: PlotAvailability) -> Quantity:
    value = str(quantity).strip().lower()
    if value not in {"magnitude", "phase"}:
        raise ValueError("Plot quantity must be 'magnitude' or 'phase'.")
    if value == "phase" and not availability.phase_available:
        raise ValueError(
            "Phase plotting requires finite stored phase and the same explicit, "
            f"nonblank phase reference for every dataset. {availability.phase_reason}"
        )
    return value  # type: ignore[return-value]


def _normalize_y_limits(y_limits) -> tuple[float, float] | None:
    if y_limits is None:
        return None
    try:
        low, high = (float(y_limits[0]), float(y_limits[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Y-axis limits must contain exactly two numeric values.") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("Y-axis limits must be finite and increasing.")
    return low, high


def _series_y(
    dataset: NamedGrid,
    values: np.ndarray,
    *,
    quantity: Quantity,
    frequency_values,
) -> np.ndarray:
    if quantity == "phase":
        result = np.rad2deg(np.asarray(values, dtype=float))
    else:
        # The grid's stored power is already the physical linear RCS quantity.
        # Do not reconstruct a complex field just to display its magnitude.
        result = dataset.grid.linear_to_default_db(
            np.asarray(values, dtype=float),
            frequency_value=frequency_values,
        )
    result = np.asarray(result, dtype=float)
    if not np.any(np.isfinite(result)):
        raise ValueError(
            f"Dataset {dataset.name!r} has no finite {quantity} samples on the selected cut."
        )
    return result


def build_azimuth_specs(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
    *,
    frequencies: Iterable[float],
    elevation: float,
    polarization: str,
    kind: AzimuthKind = "azimuth_rect",
    quantity: Quantity = "magnitude",
    angle_display_unit: str = "deg",
    frequency_display_unit: str = "GHz",
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    tol: float = 1.0e-6,
) -> tuple[PlotSpec, ...]:
    """Build one exact azimuth panel per selected common frequency."""

    selected = _coerce_named_grids(datasets)
    availability = get_plot_availability(selected, tol=tol)
    plot_kind = str(kind).strip().lower()
    if plot_kind not in {"azimuth_rect", "azimuth_polar"}:
        raise ValueError("Azimuth plot kind must be 'azimuth_rect' or 'azimuth_polar'.")
    if plot_kind == "azimuth_polar" and not availability.polar_available:
        raise ValueError(
            "Polar azimuth plotting is unavailable: " + availability.polar_reason
        )
    plot_quantity = _validate_quantity(quantity, availability)
    angle_unit = _display_angle_unit(angle_display_unit)
    frequency_unit = _display_frequency_unit(frequency_display_unit)
    limits = _normalize_y_limits(y_limits)
    requested_frequencies = tuple(float(value) for value in frequencies)
    if not requested_frequencies:
        raise ValueError("Select at least one common frequency for azimuth slides.")

    _exact_index(
        availability.elevations,
        elevation,
        tol=tol,
        axis_name="common elevation",
        dataset_name="selected overlays",
    )
    _exact_index(
        availability.polarizations,
        str(polarization),
        tol=0.0,
        axis_name="common polarization",
        dataset_name="selected overlays",
    )

    reference = selected[0].grid
    native_angle_unit = _angle_unit(reference, "azimuth")
    native_elevation_unit = _angle_unit(reference, "elevation")
    native_frequency_unit = _frequency_unit(reference)
    elevation_display = float(
        _convert_angles([elevation], native_elevation_unit, angle_unit)[0]
    )
    if availability.angular_coordinate_system == "great_circle":
        swept_axis_label = "Aspect"
        fixed_angle_label = "Pitch"
    else:
        swept_axis_label = "Azimuth"
        fixed_angle_label = "Elevation"
    y_label = "Phase (deg)" if plot_quantity == "phase" else f"RCS ({availability.rcs_unit})"

    # The swept azimuth axis, fixed-axis indices, and stable sort order do not
    # change between frequency panels.  Prepare them once per dataset and
    # share the immutable x tuple across PlotSeries values.  This keeps a
    # multi-frequency report from repeatedly allocating the same Python-float
    # coordinate array for every slide.
    target_angle_unit = "deg" if plot_kind == "azimuth_polar" else angle_unit
    prepared_datasets: list[
        tuple[NamedGrid, int, int, np.ndarray, tuple[float, ...], np.ndarray]
    ] = []
    for dataset in selected:
        grid = dataset.grid
        elevation_index = _exact_index(
            grid.elevations,
            elevation,
            tol=tol,
            axis_name="elevation",
            dataset_name=dataset.name,
        )
        polarization_index = _exact_index(
            grid.polarizations,
            str(polarization),
            tol=0.0,
            axis_name="polarization",
            dataset_name=dataset.name,
        )
        order = np.argsort(np.asarray(grid.azimuths, dtype=float), kind="stable")
        native_x = np.asarray(grid.azimuths, dtype=float)[order]
        x_values = tuple(
            float(value)
            for value in _convert_angles(native_x, native_angle_unit, target_angle_unit)
        )
        source = grid.rcs_phase if plot_quantity == "phase" else grid.rcs_power
        prepared_datasets.append(
            (
                dataset,
                elevation_index,
                polarization_index,
                order,
                x_values,
                source,
            )
        )

    specs: list[PlotSpec] = []
    for panel_index, frequency in enumerate(requested_frequencies):
        # Resolve against the common selector set before touching any data.
        _exact_index(
            availability.frequencies,
            frequency,
            tol=tol,
            axis_name="common frequency",
            dataset_name="selected overlays",
        )
        display_frequency = float(
            _convert_frequencies([frequency], native_frequency_unit, frequency_unit)[0]
        )
        series: list[PlotSeries] = []
        for (
            dataset,
            elevation_index,
            polarization_index,
            order,
            x_values,
            source,
        ) in prepared_datasets:
            grid = dataset.grid
            frequency_index = _exact_index(
                grid.frequencies,
                frequency,
                tol=tol,
                axis_name="frequency",
                dataset_name=dataset.name,
            )
            raw_y = np.asarray(
                source[:, elevation_index, frequency_index, polarization_index],
                dtype=float,
            )[order]
            y = _series_y(
                dataset,
                raw_y,
                quantity=plot_quantity,
                frequency_values=float(grid.frequencies[frequency_index]),
            )
            series.append(
                PlotSeries(
                    x=x_values,
                    y=tuple(float(value) for value in y),
                    label=dataset.name,
                )
            )

        title = (
            f"{_format_value(display_frequency)} {frequency_unit} | "
            f"{fixed_angle_label} {_format_value(elevation_display)} "
            f"{angle_unit} | {polarization}"
        )
        specs.append(
            PlotSpec(
                plot_id=_safe_id(
                    f"az_{plot_kind}_{plot_quantity}_{panel_index:03d}_"
                    f"{_format_value(display_frequency)}_{frequency_unit}"
                ),
                kind=plot_kind,  # type: ignore[arg-type]
                title=title,
                x_label=(
                    f"{swept_axis_label} "
                    f"({'deg' if plot_kind == 'azimuth_polar' else angle_unit})"
                ),
                y_label=y_label,
                series=tuple(series),
                y_limits=limits,
                show_legend=bool(show_legend),
            )
        )
    return tuple(specs)


def build_frequency_spec(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
    *,
    azimuth: float,
    elevation: float,
    polarization: str,
    quantity: Quantity = "magnitude",
    angle_display_unit: str = "deg",
    frequency_display_unit: str = "GHz",
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    tol: float = 1.0e-6,
) -> PlotSpec:
    """Build one exact frequency cut at a common azimuth/elevation/polarization."""

    selected = _coerce_named_grids(datasets)
    availability = get_plot_availability(selected, tol=tol)
    plot_quantity = _validate_quantity(quantity, availability)
    angle_unit = _display_angle_unit(angle_display_unit)
    frequency_unit = _display_frequency_unit(frequency_display_unit)
    limits = _normalize_y_limits(y_limits)
    reference = selected[0].grid
    native_azimuth_unit = _angle_unit(reference, "azimuth")
    native_elevation_unit = _angle_unit(reference, "elevation")
    native_frequency_unit = _frequency_unit(reference)

    _exact_index(
        availability.azimuths,
        azimuth,
        tol=tol,
        axis_name="common azimuth",
        dataset_name="selected overlays",
    )
    _exact_index(
        availability.elevations,
        elevation,
        tol=tol,
        axis_name="common elevation",
        dataset_name="selected overlays",
    )
    _exact_index(
        availability.polarizations,
        str(polarization),
        tol=0.0,
        axis_name="common polarization",
        dataset_name="selected overlays",
    )

    azimuth_display = float(
        _convert_angles([azimuth], native_azimuth_unit, angle_unit)[0]
    )
    elevation_display = float(
        _convert_angles([elevation], native_elevation_unit, angle_unit)[0]
    )
    if availability.angular_coordinate_system == "great_circle":
        primary_angle_label = "Aspect"
        secondary_angle_label = "Pitch"
    else:
        primary_angle_label = "Azimuth"
        secondary_angle_label = "Elevation"
    series: list[PlotSeries] = []
    for dataset in selected:
        grid = dataset.grid
        azimuth_index = _exact_index(
            grid.azimuths,
            azimuth,
            tol=tol,
            axis_name="azimuth",
            dataset_name=dataset.name,
        )
        elevation_index = _exact_index(
            grid.elevations,
            elevation,
            tol=tol,
            axis_name="elevation",
            dataset_name=dataset.name,
        )
        polarization_index = _exact_index(
            grid.polarizations,
            str(polarization),
            tol=0.0,
            axis_name="polarization",
            dataset_name=dataset.name,
        )
        order = np.argsort(np.asarray(grid.frequencies, dtype=float), kind="stable")
        native_frequencies = np.asarray(grid.frequencies, dtype=float)[order]
        x = _convert_frequencies(native_frequencies, native_frequency_unit, frequency_unit)
        source = grid.rcs_phase if plot_quantity == "phase" else grid.rcs_power
        raw_y = np.asarray(
            source[azimuth_index, elevation_index, :, polarization_index],
            dtype=float,
        )[order]
        # dBke normalization is frequency dependent, so pass the correspondingly
        # sorted native frequency vector rather than one representative scalar.
        y = _series_y(
            dataset,
            raw_y,
            quantity=plot_quantity,
            frequency_values=native_frequencies,
        )
        series.append(PlotSeries.from_values(x, y, label=dataset.name))

    y_label = "Phase (deg)" if plot_quantity == "phase" else f"RCS ({availability.rcs_unit})"
    title = (
        f"Frequency Sweep | {primary_angle_label} "
        f"{_format_value(azimuth_display)} {angle_unit} | "
        f"{secondary_angle_label} {_format_value(elevation_display)} "
        f"{angle_unit} | {polarization}"
    )
    return PlotSpec(
        plot_id=_safe_id(
            f"frequency_{plot_quantity}_{_format_value(azimuth_display)}_"
            f"{_format_value(elevation_display)}_{polarization}"
        ),
        kind="frequency",
        title=title,
        x_label=f"Frequency ({frequency_unit})",
        y_label=y_label,
        series=tuple(series),
        y_limits=limits,
        show_legend=bool(show_legend),
    )


def build_frequency_specs(*args, **kwargs) -> tuple[PlotSpec, ...]:
    """Tuple-returning companion convenient for presentation-plan builders."""

    return (build_frequency_spec(*args, **kwargs),)


__all__ = [
    "NamedGrid",
    "PlotAvailability",
    "build_azimuth_specs",
    "build_frequency_spec",
    "build_frequency_specs",
    "get_plot_availability",
]
