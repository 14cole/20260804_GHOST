"""Matched, slice-only level differences shared by GUI and Python plotting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from . import common


AXES = ("azimuth", "elevation", "frequency")
AXIS_ATTRIBUTES = {"azimuth": "azimuths", "elevation": "elevations", "frequency": "frequencies"}
MAX_DELTA_CELLS = common.MAX_IMAGE_CELLS
MAX_CELL_LABELS = 200


@dataclass
class DeltaMap:
    x: np.ndarray
    y: np.ndarray
    a_db: np.ndarray
    b_db: np.ndarray
    delta_db: np.ndarray
    x_axis: str
    y_axis: str
    fixed_axis: str
    fixed_value: float
    polarization: str
    names: tuple[str, str]
    source_unit: str
    axis_names: dict[str, str]
    axis_units: dict[str, str]

    @property
    def valid_count(self):
        return int(np.count_nonzero(np.isfinite(self.delta_db)))


def _matches(reference, dataset, axis, requested):
    """Match physical coordinates uniquely, without resampling or broadcasting."""
    native = np.asarray(getattr(dataset, AXIS_ATTRIBUTES[axis]), dtype=float)
    values = common.values_for_display(reference, dataset, axis, native)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{axis} coordinates must be finite and one-dimensional")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    tolerance = common.axis_matching_tolerance(reference, axis)
    lo = np.searchsorted(ordered, requested - tolerance, side="left")
    hi = np.searchsorted(ordered, requested + tolerance, side="right")
    if np.any(hi - lo > 1):
        raise ValueError(f"Ambiguous {axis} samples within the coordinate matching tolerance")
    present = np.flatnonzero(hi > lo)
    indices = order[lo[present]]
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Multiple selected {axis} values match the same source sample")
    return present, indices


def prepare(datasets, *, reference, selections, polarization,
            x_axis="azimuth", y_axis="frequency"):
    if len(datasets) != 2:
        raise ValueError("Delta Map requires exactly two datasets")
    if x_axis not in AXES or y_axis not in AXES or x_axis == y_axis:
        raise ValueError("Choose two different axes from frequency, azimuth, and elevation")
    common.validate_plot_datasets(datasets, phase=False, linear=False)
    fixed_axis = next(axis for axis in AXES if axis not in (x_axis, y_axis))
    coordinates = {}
    for axis in AXES:
        values = np.asarray(selections[axis], dtype=float)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError(f"Select finite {axis} values")
        coordinates[axis] = np.unique(values)
    if coordinates[fixed_axis].size != 1:
        raise ValueError(f"Delta Map requires exactly one fixed {fixed_axis} value")
    x, y = coordinates[x_axis], coordinates[y_axis]
    if int(x.size) * int(y.size) > MAX_DELTA_CELLS:
        raise ValueError(
            f"Delta Map is limited to {MAX_DELTA_CELLS:,} cells; reduce the axis selections"
        )
    levels = []
    for name, dataset in datasets:
        pol = np.flatnonzero(np.asarray(dataset.polarizations) == polarization)
        if pol.size != 1:
            raise ValueError(f"{name}: select a polarization present exactly once in both datasets")
        matched = {axis: _matches(reference, dataset, axis, coordinates[axis]) for axis in AXES}
        if matched[fixed_axis][1].size != 1:
            raise ValueError(f"{name}: no matching fixed {fixed_axis} sample")
        plane = np.full((y.size, x.size), np.nan)
        xp, xi = matched[x_axis]
        yp, yi = matched[y_axis]
        if xp.size and yp.size:
            # Two broadcast index vectors plus scalar fixed/polarization indices
            # gather only the selected 2-D plane, regardless of storage axis order.
            key = [None, None, None, int(pol[0])]
            key[AXES.index(x_axis)] = xi[None, :]
            key[AXES.index(y_axis)] = yi[:, None]
            key[AXES.index(fixed_axis)] = int(matched[fixed_axis][1][0])
            power = np.asarray(dataset.rcs_power[tuple(key)], dtype=float)
            if x_axis == "frequency":
                frequency = np.asarray(dataset.frequencies)[xi][None, :]
            elif y_axis == "frequency":
                frequency = np.asarray(dataset.frequencies)[yi][:, None]
            else:
                frequency = float(dataset.frequencies[key[2]])
            # No artificial floor: zeros, negatives, and absent values do not
            # produce a finite logarithmic difference.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                db = dataset.linear_to_default_db(power, frequency_value=frequency, eps=0.0)
            db = np.where(np.isfinite(power) & (power > 0) & np.isfinite(db), db, np.nan)
            plane[np.ix_(yp, xp)] = db
        levels.append(plane)
    delta = levels[0] - levels[1]
    if not np.any(np.isfinite(delta)):
        raise ValueError("No paired finite positive samples for the selected Delta Map slice")
    return DeltaMap(x, y, levels[0], levels[1], delta, x_axis, y_axis,
                    fixed_axis, float(coordinates[fixed_axis][0]), str(polarization),
                    tuple(str(name) for name, _ in datasets),
                    str(datasets[0][1].default_log_unit()),
                    {axis: "Frequency" if axis == "frequency" else common.angular_axis_name(reference, axis) for axis in AXES},
                    {axis: common.axis_unit(reference, axis) for axis in AXES})


def cell_edges(values):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    midpoints = values[:-1] + np.diff(values) / 2
    return np.r_[values[0] - (midpoints[0] - values[0]), midpoints,
                 values[-1] + (values[-1] - midpoints[-1])]


def cell_text(result, x, y):
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return None
    xe, ye = cell_edges(result.x), cell_edges(result.y)
    col = int(np.searchsorted(xe, x, side="right") - 1)
    row = int(np.searchsorted(ye, y, side="right") - 1)
    if not (0 <= col < result.x.size and 0 <= row < result.y.size):
        return None
    coordinates = {result.x_axis: result.x[col], result.y_axis: result.y[row],
                   result.fixed_axis: result.fixed_value}
    labels = []
    for axis in (result.x_axis, result.y_axis, result.fixed_axis):
        labels.append(f"{result.axis_names[axis]} {coordinates[axis]:g} {result.axis_units[axis]}")
    def level(value):
        return f"{value:.3f} {result.source_unit}" if np.isfinite(value) else "unavailable"
    delta = result.delta_db[row, col]
    delta_label = f"{delta:+.3f} dB" if np.isfinite(delta) else "unavailable"
    return " | ".join(labels + [f"Pol {result.polarization}",
        f"A {level(result.a_db[row, col])}", f"B {level(result.b_db[row, col])}",
        f"A - B {delta_label}"])


def draw(figure, axes, result, *, limit=None, show_values=False, show_colorbar=True):
    """Render real coordinate cells with a symmetric diverging scale."""
    if limit is None:
        limit = max(float(np.nanmax(np.abs(result.delta_db))), 0.1)
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Delta color limit must be finite and greater than zero")
    cmap = colormaps["RdBu_r"].with_extremes(bad="#888888")
    mesh = axes.pcolormesh(cell_edges(result.x), cell_edges(result.y),
                          np.ma.masked_invalid(result.delta_db), shading="flat",
                          cmap=cmap, norm=Normalize(-limit, limit), rasterized=True)
    axes.grid(False)
    axes.set_xlabel(f"{result.axis_names[result.x_axis]} ({result.axis_units[result.x_axis]})")
    axes.set_ylabel(f"{result.axis_names[result.y_axis]} ({result.axis_units[result.y_axis]})")
    name = result.axis_names[result.fixed_axis]
    axes.set_title(f"Delta Map: A - B (dB) | {name} {result.fixed_value:g} "
                   f"{result.axis_units[result.fixed_axis]} | {result.polarization}\n"
                   f"{result.valid_count:,}/{result.delta_db.size:,} paired cells; gray = unavailable",
                   fontsize=10)
    axes._grim_delta_map = result
    axes.format_coord = lambda x, y: cell_text(result, x, y) or ""
    if show_values and result.delta_db.size <= MAX_CELL_LABELS:
        for row, y in enumerate(result.y):
            for col, x in enumerate(result.x):
                value = result.delta_db[row, col]
                color = "black"
                if np.isfinite(value):
                    rgba = cmap(mesh.norm(value))
                    color = "white" if np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]) < 0.45 else "black"
                axes.text(x, y, f"{value:+.1f}" if np.isfinite(value) else "--",
                          ha="center", va="center", fontsize=8, color=color, clip_on=True)
    colorbar = None
    if show_colorbar:
        extends = "both" if np.any(np.abs(result.delta_db[np.isfinite(result.delta_db)]) > limit) else "neither"
        colorbar = figure.colorbar(mesh, ax=axes, extend=extends)
        colorbar.set_label("A - B (dB)")
    return mesh, colorbar


def render(self):
    self.last_plot_mode = "delta_map"
    self._start_plot_render()
    self._set_compare_sector_controls_visible(False)
    controls = self.delta_map_controls
    controls.setVisible(True)

    def blocked(message):
        self._clear_plot()
        controls.show()
        self.hover_readout.setText("No Delta Map for the current selection.")
        self.status.showMessage(f"Delta Map: {message}")

    datasets = self._selected_datasets()
    if len(datasets) != 2:
        controls.source_label.clear()
        blocked("select exactly two datasets in the dataset table.")
        return
    reference = common.reference_dataset(datasets, self.active_dataset)
    selections = {axis: self._selected_values(getattr(self, {"azimuth":"list_az", "elevation":"list_elev", "frequency":"list_freq"}[axis])) for axis in AXES}
    try:
        controls.configure(reference, selections, datasets)
    except ValueError as exc:
        blocked(str(exc))
        return
    pol = self._single_selection_value(self.list_pol, "polarization")
    if pol is None:
        blocked("select exactly one polarization.")
        return
    fixed = controls.fixed_value.currentData()
    if fixed is None:
        blocked("select a fixed slice.")
        return
    selections[controls.fixed_axis] = [float(fixed)]
    if controls.reversed:
        datasets = list(reversed(datasets))
    options = controls.options()
    try:
        result = prepare(datasets, reference=reference, selections=selections,
                         polarization=pol, x_axis=options["x_axis"], y_axis=options["y_axis"])
    except ValueError as exc:
        blocked(str(exc))
        return
    self._remove_colorbar()
    self.plot_figure.clear()
    self.plot_ax = self.plot_figure.add_subplot(111)
    self.plot_axes = None
    self.plot_figure.set_facecolor(self._current_plot_bg())
    self._style_axes(self.plot_ax)
    mesh, colorbar = draw(self.plot_figure, self.plot_ax, result,
                         limit=options["limit"], show_values=options["show_values"],
                         show_colorbar=self.chk_colorbar.isChecked())
    self.plot_ax.title.set_color(self._current_plot_text())
    self.plot_figure.text(0.01, 0.01, controls.source_label.text(), fontsize=8,
                          color=self._current_plot_text(), wrap=True)
    self.plot_colorbars = [colorbar] if colorbar is not None else []
    if colorbar is not None:
        colorbar.set_label("A - B (dB)", color=self._current_plot_text())
        colorbar.ax.tick_params(colors=self._current_plot_text())
    self.plot_figure.set_layout_engine("tight", rect=(0, 0.07, 1, 1))
    self.plot_figure._grim_delta_layout = True
    # Reset physical axis bounds on this new slice; later zoom/pan stays local.
    for widget, value in ((self.spin_plot_xmin, cell_edges(result.x)[0]),
                          (self.spin_plot_xmax, cell_edges(result.x)[-1]),
                          (self.spin_plot_ymin, cell_edges(result.y)[0]),
                          (self.spin_plot_ymax, cell_edges(result.y)[-1])):
        was_blocked = widget.blockSignals(True)
        # Native Hz coordinates can exceed the general plot controls' initial
        # range. Keep their editable bounds consistent with the drawn axes.
        widget.setRange(min(widget.minimum(), float(value)), max(widget.maximum(), float(value)))
        widget.setValue(float(value))
        widget.blockSignals(was_blocked)
    self.hover_readout.setText(cell_text(result, result.x[0], result.y[0]))
    self.plot_canvas.draw_idle()
    note = " Cell labels are hidden above 200 cells." if options["show_values"] and result.delta_db.size > MAX_CELL_LABELS else ""
    self.status.showMessage(f"Delta Map updated: {result.valid_count:,}/{result.delta_db.size:,} paired cells; A - B in dB.{note}")
    self._record_python_plot("delta_map", datasets_override=datasets, resolved={
        "azimuths": list(selections["azimuth"]), "elevations": list(selections["elevation"]),
        "frequencies": list(selections["frequency"]), "polarization": pol,
        "phase": False, "scale": "dbsm", "delta_options": options,
        "show_colorbar": self.chk_colorbar.isChecked(),
    }, emit=False)
