"""Qt-free helpers shared by the Plotting renderers.

The GUI parameter lists contain values in the active dataset's native units.
These helpers make that reference frame explicit, convert selections for each
other dataset, and keep display-only bounding logic out of the data model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


MAX_LINE_SERIES = 128
MAX_LINE_POINTS = 20_000
MAX_WATERFALL_PANELS = 24
MAX_IMAGE_SIDE = 2_048
MAX_IMAGE_CELLS = 2_000_000
MAX_TOTAL_IMAGE_CELLS = 8_000_000
MAX_SYNC_SLICE_CELLS = 4_000_000
MAX_SYNC_TOTAL_CELLS = 16_000_000
MAX_EXPLICIT_TICKS = 200


_FREQUENCY_UNITS = {
    "hz": ("Hz", 1.0),
    "khz": ("kHz", 1.0e3),
    "mhz": ("MHz", 1.0e6),
    "ghz": ("GHz", 1.0e9),
}
_ANGLE_UNITS = {
    "deg": ("deg", np.pi / 180.0),
    "degree": ("deg", np.pi / 180.0),
    "degrees": ("deg", np.pi / 180.0),
    "rad": ("rad", 1.0),
    "radian": ("rad", 1.0),
    "radians": ("rad", 1.0),
}


def axis_unit(dataset, axis: str) -> str:
    """Return a supported canonical unit for an RcsGrid axis."""

    if axis == "frequency":
        raw = str((dataset.units or {}).get(axis, "GHz")).strip().lower()
        entry = _FREQUENCY_UNITS.get(raw)
    elif axis in {"azimuth", "elevation"}:
        raw = str((dataset.units or {}).get(axis, "deg")).strip().lower()
        entry = _ANGLE_UNITS.get(raw)
    else:
        raise ValueError(f"unsupported plot axis {axis!r}")
    if entry is None:
        raise ValueError(f"unsupported {axis} unit {raw!r}")
    return entry[0]


def _unit_scale(axis: str, unit: str) -> float:
    table = _FREQUENCY_UNITS if axis == "frequency" else _ANGLE_UNITS
    entry = table.get(str(unit).strip().lower())
    if entry is None:
        raise ValueError(f"unsupported {axis} unit {unit!r}")
    return float(entry[1])


def convert_axis_values(values, axis: str, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert frequency or angle values without changing the stored dataset."""

    values = np.asarray(values, dtype=float)
    return values * (_unit_scale(axis, from_unit) / _unit_scale(axis, to_unit))


def selection_for_dataset(reference, dataset, axis: str, values) -> tuple[np.ndarray, float]:
    """Convert reference-list values to a dataset's native unit and tolerance."""

    reference_unit = axis_unit(reference, axis)
    dataset_unit = axis_unit(dataset, axis)
    converted = convert_axis_values(values, axis, reference_unit, dataset_unit)
    tolerance = axis_matching_tolerance(dataset, axis)
    return converted, tolerance


def axis_matching_tolerance(dataset, axis: str) -> float:
    """Return GRIM's physical matching tolerance in a dataset's native unit.

    Frequency matching historically uses 1e-6 GHz (1 kHz), while angle
    matching uses 1e-6 degrees. Anchoring the tolerance to those physical
    units makes matching symmetric when the reference grid happens to store
    Hz/radians instead of GHz/degrees.
    """

    target_unit = axis_unit(dataset, axis)
    base_unit = "GHz" if axis == "frequency" else "deg"
    tolerance = 1.0e-6 * abs(
        _unit_scale(axis, base_unit) / _unit_scale(axis, target_unit)
    )
    return max(tolerance, np.finfo(float).eps * 16.0)


def values_for_display(reference, dataset, axis: str, values) -> np.ndarray:
    """Convert a dataset's native axis values to the reference display unit."""

    return convert_axis_values(
        values,
        axis,
        axis_unit(dataset, axis),
        axis_unit(reference, axis),
    )


def reference_dataset(named_datasets, active_dataset=None):
    """Prefer the active dataset when it is among the selected datasets."""

    for _name, dataset in named_datasets:
        if dataset is active_dataset:
            return dataset
    return named_datasets[0][1]


def angular_axis_name(dataset, axis: str) -> str:
    coordinate_system = dataset.angular_coordinate_system()
    if coordinate_system == "great_circle":
        return "Aspect" if axis == "azimuth" else "Pitch"
    return "Azimuth" if axis == "azimuth" else "Elevation"


def axis_label(reference, axis: str) -> str:
    if axis == "frequency":
        return f"Frequency ({axis_unit(reference, axis)})"
    return f"{angular_axis_name(reference, axis)} ({axis_unit(reference, axis)})"


def validate_plot_datasets(named_datasets, *, phase: bool, linear: bool) -> None:
    """Fail before rendering incompatible physical quantities or coordinates.

    Coordinate units may differ because the modes convert them. Coordinate
    *systems* may not: conic azimuth/elevation and great-circle aspect/pitch
    do not describe the same chart even when their numeric arrays happen to
    match. Likewise, unlike linear quantities cannot share one ordinate.
    """

    if not named_datasets:
        return
    for name, dataset in named_datasets:
        for axis in ("azimuth", "elevation", "frequency"):
            try:
                axis_unit(dataset, axis)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc

    # Magnitude ordinates must describe one physical quantity.  Phase itself is
    # dimensionless, however, and the positive real normalization between
    # sigma_3d and sigma_2d does not rotate it.  Blocking a phase-only overlay
    # solely because those magnitude quantities differ prevented otherwise
    # valid phase QA; the coherent convention checks below are the relevant
    # safeguards for that view.
    if not phase:
        quantities = {
            str(dataset.linear_quantity()).strip().lower()
            for _, dataset in named_datasets
        }
        if len(quantities) != 1:
            details = ", ".join(
                f"{name}={dataset.linear_quantity()}" for name, dataset in named_datasets
            )
            raise ValueError(
                f"mixed physical quantities cannot share a plot ({details})"
            )

    coordinate_systems = {dataset.angular_coordinate_system() for _, dataset in named_datasets}
    if len(coordinate_systems) != 1:
        details = ", ".join(
            f"{name}={dataset.angular_coordinate_system()}" for name, dataset in named_datasets
        )
        raise ValueError(f"mixed angular coordinate systems cannot share a plot ({details})")

    if next(iter(coordinate_systems)) == "great_circle":
        conventions = {
            dataset.great_circle_coordinate_convention() for _, dataset in named_datasets
        }
        if len(conventions) != 1:
            raise ValueError("great-circle datasets use different aspect/pitch conventions")
        orientations = [dataset.angular_frame_orientation_deg() for _, dataset in named_datasets]
        if any(
            not np.allclose(orientations[0], orientation, rtol=0.0, atol=1.0e-7)
            for orientation in orientations[1:]
        ):
            raise ValueError("great-circle datasets use different roll/tilt frames")

    # Linear plots already share the same modeled physical quantity. In a dB
    # plot, also require a common log convention so dB, dBsm, and dBke are not
    # presented on one apparently uniform ordinate.
    if not phase and not linear:
        log_units = {dataset.default_log_unit().lower() for _, dataset in named_datasets}
        if len(log_units) != 1:
            details = ", ".join(
                f"{name}={dataset.default_log_unit()}" for name, dataset in named_datasets
            )
            raise ValueError(f"mixed logarithmic quantity units cannot share a plot ({details})")

    if phase and len(named_datasets) > 1:
        # Explicit disagreements are unsafe; absent legacy metadata remains
        # plotable so old files are not turned into an unnecessary blocker.
        for key, label in (
            ("phase_reference", "phase references"),
            ("time_convention", "time conventions"),
            ("polarization_basis", "polarization bases"),
        ):
            values = set()
            for _name, dataset in named_datasets:
                declared_getter = getattr(dataset, "_declared_scalar_metadata", None)
                if callable(declared_getter):
                    raw = declared_getter(key)
                else:
                    raw = (dataset.units or {}).get(
                        key, (dataset.extra or {}).get(key, "")
                    )
                text = str(raw or "").strip()
                if not text:
                    continue
                if key == "time_convention":
                    canonicalizer = getattr(dataset, "_canonical_time_convention", None)
                    text = canonicalizer(text) if callable(canonicalizer) else text.casefold()
                else:
                    text = " ".join(text.split()).casefold()
                values.add(text)
            if len(values) > 1:
                raise ValueError(f"selected datasets declare incompatible {label}")


def missing_coherent_metadata(named_datasets) -> tuple[str, ...]:
    """Return coherent metadata fields missing from at least one dataset."""

    missing = []
    for key in ("phase_reference", "time_convention", "polarization_basis"):
        for _name, dataset in named_datasets:
            getter = getattr(dataset, "_declared_scalar_metadata", None)
            value = getter(key) if callable(getter) else (dataset.units or {}).get(
                key, (dataset.extra or {}).get(key, "")
            )
            if not str(value or "").strip():
                missing.append(key)
                break
    return tuple(missing)


def circular_median_degrees(values, axis=0) -> np.ndarray:
    """Median phase on a local circular branch, robust at the -180/180 seam."""

    angles = np.asarray(values, dtype=float)
    finite = np.isfinite(angles)
    radians = np.deg2rad(np.where(finite, angles, 0.0))
    vector = np.sum(np.where(finite, np.exp(1j * radians), 0.0), axis=axis)
    center = np.rad2deg(np.angle(vector))
    count = np.sum(finite, axis=axis)
    weak_resultant = np.abs(vector) <= np.finfo(float).eps * np.maximum(count, 1) * 8.0
    if np.any(weak_resultant):
        moved_angles = np.moveaxis(angles, axis, 0)
        moved_finite = np.moveaxis(finite, axis, 0)
        first_index = np.argmax(moved_finite, axis=0)
        first_value = np.take_along_axis(
            moved_angles, np.expand_dims(first_index, axis=0), axis=0
        )[0]
        center = np.where(weak_resultant & (count > 0), first_value, center)

    expanded_center = np.expand_dims(center, axis=axis)
    residual = wrap_phase_degrees(angles - expanded_center)
    residual = np.where(finite, residual, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_residual = np.nanmedian(residual, axis=axis)
    result = wrap_phase_degrees(center + median_residual)
    return np.where(count > 0, result, np.nan)


def wrap_phase_degrees(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    wrapped = (values + 180.0) % 360.0 - 180.0
    # Preserve +180 for positive inputs. This makes labels/statistics stable
    # without changing the shortest signed residual convention at -180.
    return np.where((wrapped == -180.0) & (values > 0.0), 180.0, wrapped)


@dataclass
class StreamingEnvelope:
    """Per-X min/max/count accumulator without a duplicate stacked array.

    Phase values are put on the locally nearest 360-degree branch before the
    bounds are updated, preventing 179/-179 degrees from becoming a 358-degree
    point-by-point band.
    """

    phase_degrees: bool = False
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    count: np.ndarray | None = None

    def update(self, values) -> None:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        if self.lower is None:
            self.lower = np.full(values.shape, np.nan, dtype=float)
            self.upper = np.full(values.shape, np.nan, dtype=float)
            self.count = np.zeros(values.shape, dtype=np.int64)
        elif values.shape != self.lower.shape:
            raise ValueError("all envelope series must have the same shape")

        assert self.upper is not None and self.count is not None
        first = finite & (self.count == 0)
        self.lower[first] = values[first]
        self.upper[first] = values[first]

        existing = finite & (self.count > 0)
        if np.any(existing):
            incoming = values[existing]
            if self.phase_degrees:
                midpoint = (self.lower[existing] + self.upper[existing]) * 0.5
                incoming = incoming + 360.0 * np.round((midpoint - incoming) / 360.0)
            self.lower[existing] = np.minimum(self.lower[existing], incoming)
            self.upper[existing] = np.maximum(self.upper[existing], incoming)
        self.count[finite] += 1

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.lower is None or self.upper is None or self.count is None:
            raise ValueError("cannot read an empty envelope")
        return self.lower, self.upper, self.count


def decimate_line(x_values, y_values, max_points: int = MAX_LINE_POINTS):
    """Bound a display line while preserving extrema, endpoints, and gaps.

    A NaN is a semantic break in a measured cut, not merely a value to ignore.
    Keeping one missing sample from every mixed bucket prevents Matplotlib from
    drawing a continuous line across unavailable data after display decimation.
    """

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    if x_values.shape != y_values.shape or x_values.ndim != 1:
        raise ValueError("line x/y arrays must be matching one-dimensional arrays")
    n = x_values.size
    if n <= max_points:
        return x_values, y_values, False
    if max_points < 4:
        raise ValueError("max_points must be at least 4")

    # Min/max envelope decimation retains narrow peaks that simple striding can
    # erase. Allocate room for min, max, and one missing-data marker per
    # interior bucket, plus exact endpoints.
    bucket_count = max(1, (max_points - 2) // 3)
    edges = np.linspace(1, n - 1, bucket_count + 1, dtype=int)
    selected = [0]
    gap_markers: list[int] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        segment = y_values[start:stop]
        finite = np.isfinite(segment)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            local_min = finite_indices[int(np.argmin(segment[finite]))]
            local_max = finite_indices[int(np.argmax(segment[finite]))]
            selected.extend(sorted({start + local_min, start + local_max}))
        if np.any(~finite):
            gap_index = start + int(np.flatnonzero(~finite)[0])
            selected.append(gap_index)
            gap_markers.append(gap_index)
    selected.append(n - 1)
    selected = _bounded_decimation_indices(selected, gap_markers, n, max_points)
    return x_values[selected], y_values[selected], True


def decimate_envelope(
    x_values,
    lower,
    upper,
    count=None,
    max_points: int = MAX_LINE_POINTS,
):
    """Bound a filled envelope using extrema from both of its boundaries."""

    x_values = np.asarray(x_values)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    count_values = None if count is None else np.asarray(count)
    if x_values.shape != lower.shape or x_values.shape != upper.shape or x_values.ndim != 1:
        raise ValueError("envelope arrays must be matching one-dimensional arrays")
    if count_values is not None and count_values.shape != x_values.shape:
        raise ValueError("envelope count must match its coordinate array")
    n = x_values.size
    if n <= max_points:
        return x_values, lower, upper, count_values, False

    # Four extrema plus one missing-data marker per bucket, then exact ends.
    bucket_count = max(1, (max_points - 2) // 5)
    edges = np.linspace(1, n - 1, bucket_count + 1, dtype=int)
    selected = [0]
    gap_markers: list[int] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        valid_pair = np.isfinite(lower[start:stop]) & np.isfinite(upper[start:stop])
        if np.any(~valid_pair):
            gap_index = start + int(np.flatnonzero(~valid_pair)[0])
            selected.append(gap_index)
            gap_markers.append(gap_index)
        for values in (lower, upper):
            segment = values[start:stop]
            finite = np.isfinite(segment)
            if not np.any(finite):
                continue
            finite_indices = np.flatnonzero(finite)
            selected.append(start + finite_indices[int(np.argmin(segment[finite]))])
            selected.append(start + finite_indices[int(np.argmax(segment[finite]))])
    selected.append(n - 1)
    selected = _bounded_decimation_indices(selected, gap_markers, n, max_points)
    count_display = None if count_values is None else count_values[selected]
    return (
        x_values[selected],
        lower[selected],
        upper[selected],
        count_display,
        True,
    )


def _bounded_decimation_indices(selected, gap_markers, size: int, max_points: int) -> np.ndarray:
    """Keep decimator candidates within budget, prioritizing semantic gaps."""

    unique = sorted({int(index) for index in selected})
    if len(unique) <= int(max_points):
        return np.asarray(unique, dtype=int)

    endpoints = [0] if size == 1 else [0, size - 1]
    budget = max(0, int(max_points) - len(endpoints))
    gaps = [index for index in sorted(set(gap_markers)) if index not in endpoints]
    if len(gaps) > budget:
        positions = np.linspace(0, len(gaps) - 1, budget, dtype=int) if budget else []
        kept = endpoints + [gaps[int(position)] for position in positions]
        return np.asarray(sorted(set(kept)), dtype=int)

    kept = endpoints + gaps
    remaining = [index for index in unique if index not in set(kept)]
    slots = int(max_points) - len(kept)
    if slots > 0 and remaining:
        if len(remaining) <= slots:
            kept.extend(remaining)
        else:
            positions = np.linspace(0, len(remaining) - 1, slots, dtype=int)
            kept.extend(remaining[int(position)] for position in positions)
    return np.asarray(sorted(set(kept)), dtype=int)


def _bounded_image_shape(nx: int, ny: int, *, max_side: int, max_cells: int) -> tuple[int, int]:
    """Return a positive display shape that respects both image limits."""

    nx = int(nx)
    ny = int(ny)
    max_side = int(max_side)
    max_cells = int(max_cells)
    if nx < 0 or ny < 0:
        raise ValueError("image dimensions cannot be negative")
    if max_side < 1 or max_cells < 1:
        raise ValueError("image display limits must be positive")
    if nx == 0 or ny == 0:
        return nx, ny

    target_x = min(nx, max_side)
    target_y = min(ny, max_side)
    if target_x * target_y > max_cells:
        scale = np.sqrt(float(max_cells) / float(target_x * target_y))
        target_x = max(1, min(target_x, int(np.floor(target_x * scale))))
        target_y = max(1, min(target_y, int(np.floor(target_y * scale))))
        # Rounding can leave a slightly over-budget product for very small
        # limits. Trim the longer display dimension until the hard cap holds.
        while target_x * target_y > max_cells:
            if target_x >= target_y and target_x > 1:
                target_x -= 1
            elif target_y > 1:
                target_y -= 1
            else:  # pragma: no cover - max_cells >= 1 makes this unreachable
                break
    return target_x, target_y


def image_requires_decimation(
    x_count: int,
    y_count: int,
    *,
    max_side: int | None = None,
    max_cells: int | None = None,
) -> bool:
    """Return whether an image would exceed the interactive display budget."""

    nx = int(x_count)
    ny = int(y_count)
    target_x, target_y = _bounded_image_shape(
        nx,
        ny,
        max_side=MAX_IMAGE_SIDE if max_side is None else max_side,
        max_cells=MAX_IMAGE_CELLS if max_cells is None else max_cells,
    )
    return target_x < nx or target_y < ny


def bounded_image_cell_count(
    x_count: int,
    y_count: int,
    *,
    max_side: int | None = None,
    max_cells: int | None = None,
) -> int:
    """Return the number of cells retained by the interactive image bound."""

    target_x, target_y = _bounded_image_shape(
        int(x_count),
        int(y_count),
        max_side=MAX_IMAGE_SIDE if max_side is None else max_side,
        max_cells=MAX_IMAGE_CELLS if max_cells is None else max_cells,
    )
    return int(target_x) * int(target_y)


def validate_aggregate_image_cells(
    total_cells: int,
    *,
    panel_count: int,
    operation: str,
    max_cells: int = MAX_TOTAL_IMAGE_CELLS,
) -> None:
    """Reject a multi-panel image whose aggregate display storage is unsafe."""

    total_cells = int(total_cells)
    if total_cells <= int(max_cells):
        return
    raise ValueError(
        f"{operation} would retain {total_cells:,} display cells across "
        f"{int(panel_count):,} panels (limit {int(max_cells):,}); select fewer "
        "datasets/panels or narrow the plotted axes"
    )


def validate_synchronous_plot_workload(
    *,
    operation: str,
    peak_slice_cells: int,
    total_cells: int,
    max_slice_cells: int = MAX_SYNC_SLICE_CELLS,
    max_total_cells: int = MAX_SYNC_TOTAL_CELLS,
) -> None:
    """Bound NumPy work that still executes on the Qt GUI thread.

    These renderers use advanced indexing and exact medians, both of which can
    allocate several temporaries per source cell.  The preflight is deliberately
    expressed in cells rather than guessed bytes so it remains conservative for
    either float or complex datasets.
    """

    peak_slice_cells = int(peak_slice_cells)
    total_cells = int(total_cells)
    if peak_slice_cells <= int(max_slice_cells) and total_cells <= int(max_total_cells):
        return
    raise ValueError(
        f"{operation} selection requires a {peak_slice_cells:,}-cell working slice "
        f"and {total_cells:,} total source cells (limits {int(max_slice_cells):,} "
        f"and {int(max_total_cells):,}); crop/slice the dataset or select fewer "
        "axis values before plotting"
    )


def finite_data_limits(arrays) -> tuple[float, float] | None:
    """Return global finite limits without concatenating display arrays."""

    lower = float("inf")
    upper = float("-inf")
    for values in arrays:
        array = np.asarray(values)
        finite = array[np.isfinite(array)]
        if finite.size:
            lower = min(lower, float(np.min(finite)))
            upper = max(upper, float(np.max(finite)))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    return lower, upper


def decimate_image(x_values, y_values, image, *, max_side=MAX_IMAGE_SIDE, max_cells=MAX_IMAGE_CELLS):
    """Bound a display image using peak-preserving block aggregation.

    Uniform striding can completely erase a narrow scattering peak when its
    source cell happens to fall between retained indices.  Each output cell is
    therefore the finite maximum of its source block, which preserves the
    physically important high-return cell in linear and logarithmic intensity
    displays.  Output coordinates are the representative center samples of the
    corresponding source blocks.
    """

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    image = np.asarray(image)
    if image.shape != (x_values.size, y_values.size):
        raise ValueError("image shape must be (len(x_values), len(y_values))")
    nx, ny = image.shape
    target_x, target_y = _bounded_image_shape(
        nx, ny, max_side=max_side, max_cells=max_cells
    )
    if target_x >= nx and target_y >= ny:
        return x_values, y_values, image, False
    if np.iscomplexobj(image):
        raise ValueError("display image decimation requires real-valued intensity")

    # ``fmax.reduceat`` performs variable-width block reduction without a
    # Python loop over up to two million display cells.  It ignores a NaN when
    # the same block contains a finite value and retains NaN for an all-NaN
    # block, matching Matplotlib's missing-data behavior.
    x_starts = (np.arange(target_x, dtype=np.int64) * nx) // target_x
    y_starts = (np.arange(target_y, dtype=np.int64) * ny) // target_y
    image_display = np.fmax.reduceat(image, x_starts, axis=0)
    image_display = np.fmax.reduceat(image_display, y_starts, axis=1)

    x_stops = np.concatenate((x_starts[1:], np.asarray([nx], dtype=np.int64)))
    y_stops = np.concatenate((y_starts[1:], np.asarray([ny], dtype=np.int64)))
    x_centers = (x_starts + x_stops - 1) // 2
    y_centers = (y_starts + y_stops - 1) // 2
    return x_values[x_centers], y_values[y_centers], image_display, True


def bounded_ticks(start: float, stop: float, step: float, *, max_ticks=MAX_EXPLICIT_TICKS):
    """Return explicit ticks or ``None`` when the requested step is excessive."""

    if not np.isfinite(step) or step <= 0.0:
        return None
    if not np.isfinite(start) or not np.isfinite(stop):
        return None
    span = abs(stop - start)
    # Ticks must stay inside an explicit user range. Rounding the quotient to
    # the nearest integer made (0, 1, 0.6) emit 1.2, and Matplotlib then
    # silently expanded the axis beyond the requested maximum.
    tolerance = np.finfo(float).eps * max(span, step, 1.0) * 16.0
    count = int(np.floor((span + tolerance) / step)) + 1
    if count > int(max_ticks):
        return None
    direction = 1.0 if stop >= start else -1.0
    signed_step = direction * step
    ticks = start + signed_step * np.arange(count, dtype=float)
    if direction > 0.0:
        return ticks[ticks <= stop + tolerance]
    return ticks[ticks >= stop - tolerance]


def common_axis_indices(left, right, *, tolerance=1.0e-6):
    """Return one-to-one nearest indices for sorted/unsorted numeric axes."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("comparison axes must be one-dimensional")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("comparison axes must be finite")
    left_order = np.argsort(left, kind="stable")
    right_order = np.argsort(right, kind="stable")
    left_matches: list[int] = []
    right_matches: list[int] = []
    left_pos = 0
    right_pos = 0
    while left_pos < left_order.size and right_pos < right_order.size:
        left_index = int(left_order[left_pos])
        right_index = int(right_order[right_pos])
        delta = float(left[left_index] - right[right_index])
        if abs(delta) <= tolerance:
            left_matches.append(left_index)
            right_matches.append(right_index)
            left_pos += 1
            right_pos += 1
        elif delta < 0.0:
            left_pos += 1
        else:
            right_pos += 1
    return np.asarray(left_matches, dtype=int), np.asarray(right_matches, dtype=int)
