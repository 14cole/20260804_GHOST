from __future__ import annotations

import warnings

import numpy as np

from . import common


def _frequency_selection(
    self, reference, dataset, freq_values, az_values, elev_values, polarization
):
    freq_indices = self._axis_selection_for_dataset(
        reference, dataset, "frequency", freq_values
    )
    az_indices = self._axis_selection_for_dataset(
        reference, dataset, "azimuth", az_values
    )
    elev_indices = self._axis_selection_for_dataset(
        reference, dataset, "elevation", elev_values
    )
    pol_indices = self._indices_for_values(dataset.polarizations, [polarization], tol=0.0)
    if any(value is None for value in (freq_indices, az_indices, elev_indices, pol_indices)):
        return None
    return freq_indices, az_indices, elev_indices, pol_indices


def _frequency_series(
    self,
    reference,
    dataset,
    name,
    freq_values,
    az_values,
    elev_values,
    polarization,
    *,
    selection=None,
):
    if selection is None:
        selection = _frequency_selection(
            self,
            reference,
            dataset,
            freq_values,
            az_values,
            elev_values,
            polarization,
        )
    if selection is None:
        return None
    freq_indices, az_indices, elev_indices, pol_indices = selection

    native_frequencies = np.asarray(dataset.frequencies[freq_indices], dtype=float)
    display_frequencies = self._plot_axis_values(
        reference, dataset, "frequency", native_frequencies
    )
    pol_value = dataset.polarizations[pol_indices[0]]
    elev_name = self._plot_axis_name(reference, "elevation")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    az_name = self._plot_axis_name(reference, "azimuth")
    az_unit = self._plot_axis_unit(reference, "azimuth")
    az_min, az_max = float(np.min(az_values)), float(np.max(az_values))

    def iter_series():
        for elev_idx in elev_indices:
            native_elevation = float(dataset.elevations[elev_idx])
            elevation = float(
                self._plot_axis_values(
                    reference, dataset, "elevation", [native_elevation]
                )[0]
            )
            if self._button_checked(self.btn_phase):
                raw = dataset.rcs_slice(
                    np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])
                )[:, 0, :, 0]
                phase_degrees = self._phase_display_degrees(dataset, raw)
                display = self._wrap_phase_degrees(
                    dataset, self._phase_p50(phase_degrees, axis=0)
                )
            else:
                power = dataset.rcs_power[
                    np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])
                ][:, 0, :, 0]
                power = np.where(np.isfinite(power), power, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    p50_linear = np.nanmedian(power, axis=0)
                display = self._display_from_linear(
                    dataset, p50_linear, frequency_value=native_frequencies
                )
            label = (
                f"{name} | Pol {pol_value}, {elev_name} {elevation:g} {elev_unit}, "
                f"P50 over {az_name} ({az_min:g},{az_max:g}) {az_unit}"
            )
            yield display_frequencies, np.asarray(display), label

    return iter_series()


def render(self) -> None:
    self.last_plot_mode = "frequency"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return

    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    if freq_values.size == 0:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    if az_values.size == 0:
        self.status.showMessage("Select one or more azimuths/aspects to plot.")
        return
    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    if elev_values.size == 0:
        self.status.showMessage("Select one or more elevations/pitches to plot.")
        return
    polarization = self._single_selection_value(self.list_pol, "polarization")
    if polarization is None:
        return

    pbp_active = self._button_checked(self.btn_pbp) and (
        len(datasets) > 1 or elev_values.size > 1 or az_values.size > 1
    )
    skipped: list[str] = []
    plans = []
    peak_slice_cells = 0
    total_cells = 0
    for name, dataset in datasets:
        selection = _frequency_selection(
            self,
            reference,
            dataset,
            freq_values,
            az_values,
            elev_values,
            polarization,
        )
        if selection is None:
            skipped.append(name)
            continue
        freq_indices, az_indices, elev_indices, _pol_indices = selection
        slice_cells = len(az_indices) * len(freq_indices)
        peak_slice_cells = max(peak_slice_cells, slice_cells)
        total_cells += slice_cells * len(elev_indices)
        plans.append((name, dataset, selection))

    if not plans:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self.status.showMessage(
            "No compatible data for the selected frequency, azimuth/aspect, "
            f"elevation/pitch, and polarization values.{detail}"
        )
        return
    try:
        common.validate_synchronous_plot_workload(
            operation="Frequency P50 plot",
            peak_slice_cells=peak_slice_cells,
            total_cells=total_cells,
        )
    except ValueError as exc:
        self.status.showMessage(f"Plot blocked: {exc}.")
        return
    if not self._prepare_line_plot_axes(
        "frequency",
        "rectilinear",
        reference,
        datasets,
        pbp_active=pbp_active,
    ):
        return

    rendered = 0
    omitted = 0
    envelope = self._new_pbp_envelope() if pbp_active else None
    for name, dataset, selection in plans:
        series = _frequency_series(
            self,
            reference,
            dataset,
            name,
            freq_values,
            az_values,
            elev_values,
            polarization,
            selection=selection,
        )
        assert series is not None
        for x_values, display, label in series:
            if not np.any(np.isfinite(display)):
                continue
            if envelope is not None:
                envelope.update(display)
                rendered += 1
            elif rendered < common.MAX_LINE_SERIES:
                self._plot_bounded_line(self.plot_ax, x_values, display, label=label)
                rendered += 1
            else:
                omitted += 1

    if envelope is not None and envelope.lower is not None:
        lower, upper, density = envelope.result()
        x_values, lower, upper, density = self._bounded_plot_envelope(
            freq_values, lower, upper, density
        )
        elev_name = self._plot_axis_name(reference, "elevation")
        elev_unit = self._plot_axis_unit(reference, "elevation")
        az_name = self._plot_axis_name(reference, "azimuth")
        az_unit = self._plot_axis_unit(reference, "azimuth")
        elev_label = (
            f"{elev_values[0]:g}-{elev_values[-1]:g} {elev_unit}"
            if elev_values.size > 1
            else f"{elev_values[0]:g} {elev_unit}"
        )
        label = (
            f"PBP Pol {polarization}, {elev_name} {elev_label}, "
            f"P50 over {az_name} ({az_values[0]:g},{az_values[-1]:g}) {az_unit}"
        )
        self._plot_pbp_fill(
            x_values, lower, upper, label, polar=False, density=density
        )
        self._plot_bounded_line(
            self.plot_ax, x_values, lower, color="#8a8a8a", linewidth=1,
            label="_nolegend_",
        )
        self._plot_bounded_line(
            self.plot_ax, x_values, upper, color="#8a8a8a", linewidth=1,
            label="_nolegend_",
        )

    if rendered == 0:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self.status.showMessage(
            "No compatible data for the selected frequency, azimuth/aspect, "
            f"elevation/pitch, and polarization values.{detail}"
        )
        return
    if omitted:
        self._note_plot_render(
            f"Displayed {common.MAX_LINE_SERIES} of {common.MAX_LINE_SERIES + omitted} "
            "series; narrow elevation selections to show the rest."
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "frequency"))
    self.plot_ax.set_ylabel(self._display_axis_label(datasets, tag=" P50"))
    self._update_legend_visibility()
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(freq_values[0]))
    self.spin_plot_xmax.setValue(float(freq_values[-1]))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    status = "Frequency plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
