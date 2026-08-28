from __future__ import annotations

import numpy as np

from . import common
from .azimuth_rect_mode import _indices, _series


def render(self) -> None:
    self.last_plot_mode = "azimuth_polar"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return

    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    if az_values.size == 0:
        self.status.showMessage("Select one or more azimuths/aspects to plot.")
        return
    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    if freq_values.size == 0:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    if elev_values.size == 0:
        self.status.showMessage("Select one or more elevations/pitches to plot.")
        return
    polarization = self._single_selection_value(self.list_pol, "polarization")
    if polarization is None:
        return

    pbp_active = self._button_checked(self.btn_pbp) and (
        len(datasets) > 1 or freq_values.size > 1 or elev_values.size > 1
    )
    angular_unit = self._plot_axis_unit(reference, "azimuth")
    self._polar_display_unit = angular_unit
    if not self._prepare_line_plot_axes(
        "azimuth_polar",
        "polar",
        reference,
        datasets,
        pbp_active=pbp_active,
    ):
        return

    skipped: list[str] = []
    rendered = 0
    omitted = 0
    envelope = self._new_pbp_envelope() if pbp_active else None
    for name, dataset in datasets:
        selection = _indices(
            self, reference, dataset, az_values, elev_values, freq_values, polarization
        )
        if selection is None:
            skipped.append(name)
            continue
        for x_values, display, label in _series(
            self, reference, dataset, name, selection, polarization
        ):
            if not np.any(np.isfinite(display)):
                continue
            if envelope is not None:
                envelope.update(display)
                rendered += 1
            elif rendered < common.MAX_LINE_SERIES:
                theta = common.convert_axis_values(
                    x_values, "azimuth", angular_unit, "rad"
                )
                self._plot_bounded_line(self.plot_ax, theta, display, label=label)
                rendered += 1
            else:
                omitted += 1

    if envelope is not None and envelope.lower is not None:
        lower, upper, density = envelope.result()
        x_values, lower, upper, density = self._bounded_plot_envelope(
            az_values, lower, upper, density
        )
        theta = common.convert_axis_values(x_values, "azimuth", angular_unit, "rad")
        freq_unit = self._plot_axis_unit(reference, "frequency")
        elev_unit = self._plot_axis_unit(reference, "elevation")
        elev_name = self._plot_axis_name(reference, "elevation")
        freq_label = (
            f"{freq_values[0]:g}-{freq_values[-1]:g} {freq_unit}"
            if freq_values.size > 1
            else f"{freq_values[0]:g} {freq_unit}"
        )
        elev_label = (
            f"{elev_values[0]:g}-{elev_values[-1]:g} {elev_unit}"
            if elev_values.size > 1
            else f"{elev_values[0]:g} {elev_unit}"
        )
        label = f"PBP Pol {polarization}, Freq {freq_label}, {elev_name} {elev_label}"
        self._plot_pbp_fill(theta, lower, upper, label, polar=True, density=density)
        self._plot_bounded_line(
            self.plot_ax, theta, lower, color="#8a8a8a", linewidth=1,
            label="_nolegend_",
        )
        self._plot_bounded_line(
            self.plot_ax, theta, upper, color="#8a8a8a", linewidth=1,
            label="_nolegend_",
        )

    if rendered == 0:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self.status.showMessage(
            "No compatible data for the selected azimuth/aspect, elevation/pitch, "
            f"frequency, and polarization values.{detail}"
        )
        return
    if omitted:
        self._note_plot_render(
            f"Displayed {common.MAX_LINE_SERIES} of {common.MAX_LINE_SERIES + omitted} "
            "series; narrow frequency/elevation selections to show the rest."
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    self.plot_ax.set_ylabel(self._display_axis_label(datasets))
    self._update_legend_visibility()

    half_turn = float(
        common.convert_axis_values([180.0], "azimuth", "deg", angular_unit)[0]
    )
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(-half_turn)
    self.spin_plot_xmax.setValue(half_turn)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)

    self._apply_plot_limits()
    status = "Azimuth/Aspect (Polar) plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
