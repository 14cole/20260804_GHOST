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
PolarizationSelection = str | Sequence[str]

# User-facing shorthand used by the PowerPoint workspace.  Keeping the
# expansion here (rather than only in Qt code) gives previews, exports, and
# headless callers identical behavior.
DUAL_COPOLARIZATION = "VV and HH"

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


def _requested_polarizations(
    selection: PolarizationSelection,
    available: Sequence[str],
) -> tuple[str, ...]:
    """Resolve one or more requested polarizations against the common axes.

    ``VV and HH`` is intentionally a presentation convenience, not a new
    electromagnetic polarization.  It expands to two independent plot specs,
    so no unlike polarization samples are combined in one curve.
    """

    if isinstance(selection, str):
        text = selection.strip()
        compact = re.sub(r"\s+", "", text).lower()
        if compact in {"vvandhh", "vv+hh"}:
            requested = ("VV", "HH")
        elif compact in {"hhandvv", "hh+vv"}:
            requested = ("HH", "VV")
        else:
            requested = (text,)
    else:
        requested = tuple(str(value).strip() for value in selection)

    if not requested or any(not value for value in requested):
        raise ValueError("Select at least one polarization for the PowerPoint report.")

    unique = tuple(dict.fromkeys(requested))
    available_values = tuple(str(value) for value in available)
    missing = tuple(value for value in unique if value not in available_values)
    if missing:
        missing_text = ", ".join(missing)
        available_text = ", ".join(available_values) or "none"
        raise ValueError(
            f"Requested polarization(s) {missing_text} are not common to every "
            f"selected dataset. Common polarizations: {available_text}."
        )
    return unique


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


def _exact_numeric_indices(
    axis,
    requested_values: Sequence[float],
    *,
    tol: float,
    axis_name: str,
    dataset_name: str,
) -> np.ndarray:
    """Resolve many exact-with-tolerance values without repeated axis scans."""

    values = np.asarray(axis, dtype=float).reshape(-1)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    resolved: list[int] = []
    for requested in requested_values:
        requested = float(requested)
        left = int(np.searchsorted(sorted_values, requested - tol, side="left"))
        right = int(np.searchsorted(sorted_values, requested + tol, side="right"))
        count = right - left
        if count == 0:
            raise ValueError(
                f"{axis_name} value {requested!r} is not present in dataset "
                f"{dataset_name!r} within tolerance {tol:g}; no interpolation is performed."
            )
        if count > 1:
            raise ValueError(
                f"{axis_name} value {requested!r} is ambiguous in dataset "
                f"{dataset_name!r}: {count} samples fall within tolerance {tol:g}."
            )
        resolved.append(int(order[left]))
    return np.asarray(resolved, dtype=int)


def _numeric_intersection(axis_values: Sequence[np.ndarray], tol: float) -> tuple[float, ...]:
    if not axis_values:
        return ()
    sorted_axes = [
        np.sort(np.asarray(axis, dtype=float).reshape(-1))
        for axis in axis_values
    ]
    if any(np.any(~np.isfinite(axis)) for axis in sorted_axes):
        raise ValueError("PowerPoint selector axes must contain only finite values.")

    common: list[float] = []
    previous_candidate: float | None = None
    for candidate in sorted_axes[0]:
        candidate = float(candidate)
        if (
            previous_candidate is not None
            and math.isclose(
                candidate,
                previous_candidate,
                rel_tol=0.0,
                abs_tol=tol,
            )
        ):
            continue
        previous_candidate = candidate
        # Every fixed-axis selector must resolve to exactly one value in every
        # overlay. searchsorted keeps this O(N log N) rather than repeatedly
        # scanning each dense axis for every candidate.
        if all(
            int(np.searchsorted(axis, candidate + tol, side="right"))
            - int(np.searchsorted(axis, candidate - tol, side="left"))
            == 1
            for axis in sorted_axes
        ):
            common.append(candidate)
    return tuple(common)


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
    missing_phase = [
        dataset.name
        for dataset in datasets
        if not np.any(np.isfinite(np.asarray(dataset.grid.rcs_phase, dtype=float)))
    ]
    if missing_phase:
        return (
            False,
            "",
            "No finite stored phase is available for: " + ", ".join(missing_phase) + ".",
        )
    references = [dataset.grid._phase_reference() for dataset in datasets]
    missing_reference = [
        dataset.name
        for dataset, reference in zip(datasets, references)
        if not reference
    ]
    declared_references = {reference for reference in references if reference}
    notes = []
    if missing_reference:
        notes.append(
            "Phase reference is unspecified for " + ", ".join(missing_reference)
        )
    if len(declared_references) > 1:
        notes.append(
            "Selected datasets declare different phase references; phase is plotted "
            "as stored without reference conversion"
        )
    common_reference = (
        next(iter(declared_references))
        if len(declared_references) == 1 and not missing_reference
        else ""
    )
    return True, common_reference, ". ".join(notes) + ("." if notes else "")


def _polar_capability(datasets: tuple[NamedGrid, ...]) -> tuple[bool, str]:
    notes = []
    for dataset in datasets:
        system = dataset.grid.angular_coordinate_system()
        if system == "great_circle":
            convention = dataset.grid.great_circle_coordinate_convention()
            if convention == LEGACY_PTM_GC_CONVENTION:
                notes.append(
                    f"{dataset.name!r} is unmarked legacy great-circle data; polar "
                    "placement uses its stored aspect angles without inferring a "
                    "calibrated compass orientation"
                )
                continue
            if convention != GRIM_GC_CONVENTION:
                return False, f"{dataset.name!r} uses unsupported great-circle convention {convention!r}."
        elif system != "conic":
            return False, f"{dataset.name!r} uses unsupported angular coordinate system {system!r}."
    return True, ". ".join(notes) + ("." if notes else "")


def get_plot_availability(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
    *,
    tol: float = 1.0e-6,
    evaluate_phase: bool = True,
) -> PlotAvailability:
    """Return exact common fixed-axis selectors for the selected overlays."""

    if not math.isfinite(float(tol)) or tol < 0.0:
        raise ValueError("Axis matching tolerance must be a finite nonnegative value.")
    selected = _coerce_named_grids(datasets)
    _assert_metadata_compatible(selected)
    reference = selected[0].grid
    if evaluate_phase:
        phase_available, phase_reference, phase_reason = _phase_capability(selected)
    else:
        phase_available, phase_reference, phase_reason = (
            False,
            "",
            "Phase capability was not evaluated for this magnitude-only report.",
        )
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
            "Phase plotting requires finite stored phase. "
            f"{availability.phase_reason}"
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
    polarization: PolarizationSelection,
    kind: AzimuthKind = "azimuth_rect",
    quantity: Quantity = "magnitude",
    angle_display_unit: str = "deg",
    frequency_display_unit: str = "GHz",
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    tol: float = 1.0e-6,
) -> tuple[PlotSpec, ...]:
    """Build exact azimuth panels for every requested polarization/frequency.

    Multiple polarizations are emitted as independent plot specs in
    polarization-major order.  For example, ``("VV", "HH")`` produces all VV
    frequency panels followed by all HH panels; no series combines unlike
    polarization samples.
    """

    selected = _coerce_named_grids(datasets)
    availability = get_plot_availability(
        selected,
        tol=tol,
        evaluate_phase=str(quantity).strip().lower() == "phase",
    )
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
    _exact_numeric_indices(
        availability.frequencies,
        requested_frequencies,
        tol=tol,
        axis_name="common frequency",
        dataset_name="selected overlays",
    )

    _exact_index(
        availability.elevations,
        elevation,
        tol=tol,
        axis_name="common elevation",
        dataset_name="selected overlays",
    )
    requested_polarizations = _requested_polarizations(
        polarization,
        availability.polarizations,
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
        tuple[
            NamedGrid,
            int,
            dict[str, int],
            np.ndarray,
            tuple[float, ...],
            np.ndarray,
            np.ndarray,
        ]
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
        polarization_indices = {
            requested: _exact_index(
                grid.polarizations,
                requested,
                tol=0.0,
                axis_name="polarization",
                dataset_name=dataset.name,
            )
            for requested in requested_polarizations
        }
        order = np.argsort(np.asarray(grid.azimuths, dtype=float), kind="stable")
        native_x = np.asarray(grid.azimuths, dtype=float)[order]
        x_values = tuple(
            float(value)
            for value in _convert_angles(native_x, native_angle_unit, target_angle_unit)
        )
        source = grid.rcs_phase if plot_quantity == "phase" else grid.rcs_power
        frequency_indices = _exact_numeric_indices(
            grid.frequencies,
            requested_frequencies,
            tol=tol,
            axis_name="frequency",
            dataset_name=dataset.name,
        )
        prepared_datasets.append(
            (
                dataset,
                elevation_index,
                polarization_indices,
                order,
                x_values,
                source,
                frequency_indices,
            )
        )

    specs: list[PlotSpec] = []
    for requested_polarization in requested_polarizations:
        for panel_index, frequency in enumerate(requested_frequencies):
            display_frequency = float(
                _convert_frequencies(
                    [frequency], native_frequency_unit, frequency_unit
                )[0]
            )
            series: list[PlotSeries] = []
            for (
                dataset,
                elevation_index,
                polarization_indices,
                order,
                x_values,
                source,
                frequency_indices,
            ) in prepared_datasets:
                grid = dataset.grid
                frequency_index = int(frequency_indices[panel_index])
                raw_y = np.asarray(
                    source[
                        :,
                        elevation_index,
                        frequency_index,
                        polarization_indices[requested_polarization],
                    ],
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
                f"{angle_unit} | {requested_polarization}"
            )
            specs.append(
                PlotSpec(
                    plot_id=_safe_id(
                        f"az_{plot_kind}_{plot_quantity}_"
                        f"{requested_polarization}_{panel_index:03d}_"
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


def build_elevation_specs(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
    *,
    frequencies: Iterable[float],
    azimuth: float,
    polarization: PolarizationSelection,
    quantity: Quantity = "magnitude",
    angle_display_unit: str = "deg",
    frequency_display_unit: str = "GHz",
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    tol: float = 1.0e-6,
) -> tuple[PlotSpec, ...]:
    """Build exact elevation/pitch sweeps at one stored azimuth/aspect cut.

    The fixed selectors must be common to every overlay and are resolved
    without interpolation.  Each dataset keeps its own swept elevation axis,
    allowing compatible grids with different elevation sampling densities to
    be compared without silently resampling either response.
    """

    selected = _coerce_named_grids(datasets)
    undersampled = [
        dataset.name
        for dataset in selected
        if np.asarray(dataset.grid.elevations).size < 2
    ]
    if undersampled:
        raise ValueError(
            "Elevation sweeps require at least two stored elevation samples in "
            "every selected dataset. Insufficient data: " + ", ".join(undersampled)
        )
    availability = get_plot_availability(
        selected,
        tol=tol,
        evaluate_phase=str(quantity).strip().lower() == "phase",
    )
    plot_quantity = _validate_quantity(quantity, availability)
    angle_unit = _display_angle_unit(angle_display_unit)
    frequency_unit = _display_frequency_unit(frequency_display_unit)
    limits = _normalize_y_limits(y_limits)
    requested_frequencies = tuple(float(value) for value in frequencies)
    if not requested_frequencies:
        raise ValueError("Select at least one common frequency for elevation slides.")
    _exact_numeric_indices(
        availability.frequencies,
        requested_frequencies,
        tol=tol,
        axis_name="common frequency",
        dataset_name="selected overlays",
    )

    _exact_index(
        availability.azimuths,
        azimuth,
        tol=tol,
        axis_name="common azimuth",
        dataset_name="selected overlays",
    )
    requested_polarizations = _requested_polarizations(
        polarization,
        availability.polarizations,
    )

    reference = selected[0].grid
    native_azimuth_unit = _angle_unit(reference, "azimuth")
    native_elevation_unit = _angle_unit(reference, "elevation")
    native_frequency_unit = _frequency_unit(reference)
    azimuth_display = float(
        _convert_angles([azimuth], native_azimuth_unit, angle_unit)[0]
    )
    if availability.angular_coordinate_system == "great_circle":
        swept_axis_label = "Pitch"
        fixed_angle_label = "Aspect"
    else:
        swept_axis_label = "Elevation"
        fixed_angle_label = "Azimuth"
    y_label = (
        "Phase (deg)"
        if plot_quantity == "phase"
        else f"RCS ({availability.rcs_unit})"
    )

    prepared_datasets: list[
        tuple[
            NamedGrid,
            int,
            dict[str, int],
            np.ndarray,
            tuple[float, ...],
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    for dataset in selected:
        grid = dataset.grid
        azimuth_index = _exact_index(
            grid.azimuths,
            azimuth,
            tol=tol,
            axis_name="azimuth",
            dataset_name=dataset.name,
        )
        polarization_indices = {
            requested: _exact_index(
                grid.polarizations,
                requested,
                tol=0.0,
                axis_name="polarization",
                dataset_name=dataset.name,
            )
            for requested in requested_polarizations
        }
        order = np.argsort(np.asarray(grid.elevations, dtype=float), kind="stable")
        native_x = np.asarray(grid.elevations, dtype=float)[order]
        x_values = tuple(
            float(value)
            for value in _convert_angles(
                native_x,
                native_elevation_unit,
                angle_unit,
            )
        )
        source = grid.rcs_phase if plot_quantity == "phase" else grid.rcs_power
        frequency_indices = _exact_numeric_indices(
            grid.frequencies,
            requested_frequencies,
            tol=tol,
            axis_name="frequency",
            dataset_name=dataset.name,
        )
        prepared_datasets.append(
            (
                dataset,
                azimuth_index,
                polarization_indices,
                order,
                x_values,
                source,
                frequency_indices,
            )
        )

    specs: list[PlotSpec] = []
    for requested_polarization in requested_polarizations:
        for panel_index, frequency in enumerate(requested_frequencies):
            display_frequency = float(
                _convert_frequencies(
                    [frequency], native_frequency_unit, frequency_unit
                )[0]
            )
            series: list[PlotSeries] = []
            for (
                dataset,
                azimuth_index,
                polarization_indices,
                order,
                x_values,
                source,
                frequency_indices,
            ) in prepared_datasets:
                grid = dataset.grid
                frequency_index = int(frequency_indices[panel_index])
                raw_y = np.asarray(
                    source[
                        azimuth_index,
                        :,
                        frequency_index,
                        polarization_indices[requested_polarization],
                    ],
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
                f"{fixed_angle_label} {_format_value(azimuth_display)} "
                f"{angle_unit} | {requested_polarization}"
            )
            specs.append(
                PlotSpec(
                    plot_id=_safe_id(
                        f"el_{plot_quantity}_{requested_polarization}_"
                        f"{panel_index:03d}_{_format_value(display_frequency)}_"
                        f"{frequency_unit}"
                    ),
                    kind="elevation",
                    title=title,
                    x_label=f"{swept_axis_label} ({angle_unit})",
                    y_label=y_label,
                    series=tuple(series),
                    y_limits=limits,
                    show_legend=bool(show_legend),
                )
            )
    return tuple(specs)


def _normalize_azimuth_band(
    azimuth_band,
    azimuth_percentile,
) -> tuple[tuple[float, float], float] | None:
    if azimuth_band is None and azimuth_percentile is None:
        return None
    if azimuth_band is None or azimuth_percentile is None:
        raise ValueError(
            "Azimuth-band frequency plots require both band limits and a percentile."
        )
    try:
        band_values = tuple(azimuth_band)
        if len(band_values) != 2:
            raise ValueError
        low, high = float(band_values[0]), float(band_values[1])
        percentile = float(azimuth_percentile)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Azimuth band must contain exactly two numeric limits and its "
            "percentile must be numeric."
        ) from exc
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        raise ValueError("Azimuth band limits must be finite and distinct.")
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("Azimuth percentile must be finite and between 0 and 100.")
    return (low, high), percentile


def _common_azimuths_in_band(
    common_azimuths: Sequence[float],
    band: tuple[float, float],
    *,
    angle_unit: str,
    tol: float,
) -> tuple[float, ...]:
    """Return unique common directions in an inclusive, optionally wrapped band.

    Band limits are expressed in the dataset's stored angular convention.  We
    deliberately reject limits outside that convention instead of silently
    interpreting, for example, 350 degrees on a -180..180 axis.  If the stored
    axis contains both representations of the periodic seam (0/360 or
    -180/+180), the numerically greater representation wins so that the same
    physical direction is never counted twice in a percentile.
    """

    low, high = band
    values = np.asarray(common_azimuths, dtype=float)
    if values.size == 0:
        raise ValueError("The selected datasets have no common azimuth samples.")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if low < minimum - tol or low > maximum + tol:
        raise ValueError(
            f"Azimuth-band start {_format_value(low)} is outside the stored "
            f"azimuth range [{_format_value(minimum)}, {_format_value(maximum)}] "
            f"{angle_unit}. Enter limits in the dataset's displayed convention."
        )
    if high < minimum - tol or high > maximum + tol:
        raise ValueError(
            f"Azimuth-band end {_format_value(high)} is outside the stored "
            f"azimuth range [{_format_value(minimum)}, {_format_value(maximum)}] "
            f"{angle_unit}. Enter limits in the dataset's displayed convention."
        )
    if low < high:
        mask = (values >= low - tol) & (values <= high + tol)
    else:
        # A descending pair intentionally crosses the periodic azimuth seam;
        # e.g. 170 to -170 selects the narrow aft-looking sector.
        mask = (values >= low - tol) | (values <= high + tol)
    selected = [float(value) for value in values[mask]]
    if not selected:
        qualifier = "wrapped " if low > high else ""
        raise ValueError(
            f"The {qualifier}azimuth band [{_format_value(low)}, "
            f"{_format_value(high)}] contains no samples common to every "
            "selected dataset; band percentiles do not interpolate."
        )

    period = 360.0 if angle_unit == "deg" else 2.0 * math.pi
    unique: list[float] = []
    for value in selected:
        alias_index = next(
            (
                index
                for index, existing in enumerate(unique)
                if math.isclose(
                    abs(value - existing),
                    period,
                    rel_tol=0.0,
                    abs_tol=tol,
                )
            ),
            None,
        )
        if alias_index is None:
            unique.append(value)
        elif value > unique[alias_index]:
            unique[alias_index] = value
    return tuple(unique)


def _nanpercentile_columns(values: np.ndarray, percentile: float) -> np.ndarray:
    """Calculate a display-domain percentile, retaining all-NaN frequencies."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Azimuth-band percentile input must be a 2-D sample matrix.")
    result = np.full(matrix.shape[1], np.nan, dtype=float)
    for column_index in range(matrix.shape[1]):
        finite = matrix[:, column_index]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            result[column_index] = float(np.percentile(finite, percentile))
    return result


def build_frequency_spec(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
    *,
    azimuth: float | None,
    elevation: float,
    polarization: str,
    quantity: Quantity = "magnitude",
    angle_display_unit: str = "deg",
    frequency_display_unit: str = "GHz",
    azimuth_band: tuple[float, float] | None = None,
    azimuth_percentile: float | None = None,
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    tol: float = 1.0e-6,
) -> PlotSpec:
    """Build one frequency sweep for one polarization.

    With no band arguments, the result is the historical exact azimuth cut.
    When ``azimuth_band`` and ``azimuth_percentile`` are supplied, each plotted
    point is that percentile across the identical common stored azimuth
    samples in the inclusive band.  Periodic seam aliases are counted once,
    with the numerically greater stored representation taking precedence.
    Magnitude percentiles are sample-weighted and calculated in the displayed
    logarithmic RCS unit; phase percentiles are rejected because ordinary
    percentiles are not meaningful for wrapped angular phase.
    """

    selected = _coerce_named_grids(datasets)
    availability = get_plot_availability(
        selected,
        tol=tol,
        evaluate_phase=str(quantity).strip().lower() == "phase",
    )
    plot_quantity = _validate_quantity(quantity, availability)
    angle_unit = _display_angle_unit(angle_display_unit)
    frequency_unit = _display_frequency_unit(frequency_display_unit)
    limits = _normalize_y_limits(y_limits)
    reference = selected[0].grid
    native_azimuth_unit = _angle_unit(reference, "azimuth")
    native_elevation_unit = _angle_unit(reference, "elevation")
    native_frequency_unit = _frequency_unit(reference)
    band_request = _normalize_azimuth_band(azimuth_band, azimuth_percentile)
    if band_request is None:
        if azimuth is None:
            raise ValueError("Select an exact azimuth or enable an azimuth band.")
        _exact_index(
            availability.azimuths,
            azimuth,
            tol=tol,
            axis_name="common azimuth",
            dataset_name="selected overlays",
        )
        selected_band_azimuths: tuple[float, ...] = ()
    else:
        if azimuth is not None:
            raise ValueError(
                "Choose either an exact azimuth or an azimuth percentile band, not both."
            )
        if plot_quantity == "phase":
            raise ValueError(
                "Azimuth-band percentiles are unavailable for wrapped phase; "
                "use an exact azimuth phase cut."
            )
        selected_band_azimuths = _common_azimuths_in_band(
            availability.azimuths,
            band_request[0],
            angle_unit=native_azimuth_unit,
            tol=tol,
        )
    _exact_index(
        availability.elevations,
        elevation,
        tol=tol,
        axis_name="common elevation",
        dataset_name="selected overlays",
    )
    requested_polarizations = _requested_polarizations(
        polarization,
        availability.polarizations,
    )
    if len(requested_polarizations) != 1:
        raise ValueError(
            "build_frequency_spec creates one polarization plot; use "
            "build_frequency_specs for 'VV and HH'."
        )
    requested_polarization = requested_polarizations[0]

    azimuth_display = (
        None
        if azimuth is None
        else float(_convert_angles([azimuth], native_azimuth_unit, angle_unit)[0])
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
        elevation_index = _exact_index(
            grid.elevations,
            elevation,
            tol=tol,
            axis_name="elevation",
            dataset_name=dataset.name,
        )
        polarization_index = _exact_index(
            grid.polarizations,
            requested_polarization,
            tol=0.0,
            axis_name="polarization",
            dataset_name=dataset.name,
        )
        order = np.argsort(np.asarray(grid.frequencies, dtype=float), kind="stable")
        native_frequencies = np.asarray(grid.frequencies, dtype=float)[order]
        x = _convert_frequencies(native_frequencies, native_frequency_unit, frequency_unit)
        source = grid.rcs_phase if plot_quantity == "phase" else grid.rcs_power
        if band_request is None:
            azimuth_index = _exact_index(
                grid.azimuths,
                azimuth,
                tol=tol,
                axis_name="azimuth",
                dataset_name=dataset.name,
            )
            raw_y = np.asarray(
                source[azimuth_index, elevation_index, :, polarization_index],
                dtype=float,
            )[order]
            # dBke normalization is frequency dependent, so pass the
            # correspondingly sorted native frequency vector rather than one
            # representative scalar.
            y = _series_y(
                dataset,
                raw_y,
                quantity=plot_quantity,
                frequency_values=native_frequencies,
            )
        else:
            azimuth_indices = _exact_numeric_indices(
                grid.azimuths,
                selected_band_azimuths,
                tol=tol,
                axis_name="azimuth",
                dataset_name=dataset.name,
            )
            raw_band = np.asarray(
                source[azimuth_indices, elevation_index, :, polarization_index],
                dtype=float,
            )[:, order]
            display_band = _series_y(
                dataset,
                raw_band,
                quantity=plot_quantity,
                frequency_values=native_frequencies,
            )
            y = _nanpercentile_columns(display_band, band_request[1])
            if not np.any(np.isfinite(y)):
                raise ValueError(
                    f"Dataset {dataset.name!r} has no finite magnitude samples "
                    "in the selected azimuth band."
                )
        series.append(PlotSeries.from_values(x, y, label=dataset.name))

    y_label = "Phase (deg)" if plot_quantity == "phase" else f"RCS ({availability.rcs_unit})"
    if band_request is None:
        azimuth_title = (
            f"{primary_angle_label} {_format_value(azimuth_display)} {angle_unit}"
        )
        azimuth_id = _format_value(azimuth_display)
    else:
        native_band, percentile = band_request
        display_band = _convert_angles(native_band, native_azimuth_unit, angle_unit)
        wrap_label = " (wrap)" if native_band[0] > native_band[1] else ""
        azimuth_title = (
            f"P{_format_value(percentile)} across {primary_angle_label} "
            f"[{_format_value(display_band[0])}, {_format_value(display_band[1])}] "
            f"{angle_unit}{wrap_label} "
            f"({len(selected_band_azimuths)} common samples)"
        )
        azimuth_id = (
            f"band_{_format_value(display_band[0])}_"
            f"{_format_value(display_band[1])}_p{_format_value(percentile)}"
        )
    title = (
        f"Frequency Sweep | {azimuth_title} | "
        f"{secondary_angle_label} {_format_value(elevation_display)} "
        f"{angle_unit} | {requested_polarization}"
    )
    return PlotSpec(
        plot_id=_safe_id(
            f"frequency_{plot_quantity}_{azimuth_id}_"
            f"{_format_value(elevation_display)}_{requested_polarization}"
        ),
        kind="frequency",
        title=title,
        x_label=f"Frequency ({frequency_unit})",
        y_label=y_label,
        series=tuple(series),
        y_limits=limits,
        show_legend=bool(show_legend),
    )


def build_frequency_specs(
    datasets: Sequence[NamedGrid | tuple[str, RcsGrid]],
    *,
    azimuth: float | None,
    elevation: float,
    polarization: PolarizationSelection,
    quantity: Quantity = "magnitude",
    angle_display_unit: str = "deg",
    frequency_display_unit: str = "GHz",
    azimuth_band: tuple[float, float] | None = None,
    azimuth_percentile: float | None = None,
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    tol: float = 1.0e-6,
) -> tuple[PlotSpec, ...]:
    """Build one independent frequency plot per requested polarization."""

    selected = _coerce_named_grids(datasets)
    availability = get_plot_availability(
        selected,
        tol=tol,
        evaluate_phase=str(quantity).strip().lower() == "phase",
    )
    requested = _requested_polarizations(polarization, availability.polarizations)
    return tuple(
        build_frequency_spec(
            selected,
            azimuth=azimuth,
            elevation=elevation,
            polarization=value,
            quantity=quantity,
            angle_display_unit=angle_display_unit,
            frequency_display_unit=frequency_display_unit,
            azimuth_band=azimuth_band,
            azimuth_percentile=azimuth_percentile,
            y_limits=y_limits,
            show_legend=show_legend,
            tol=tol,
        )
        for value in requested
    )


__all__ = [
    "DUAL_COPOLARIZATION",
    "NamedGrid",
    "PlotAvailability",
    "PolarizationSelection",
    "build_azimuth_specs",
    "build_elevation_specs",
    "build_frequency_spec",
    "build_frequency_specs",
    "get_plot_availability",
]
