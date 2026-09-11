from __future__ import annotations

import warnings

import numpy as np

from . import common


def render(self) -> None:
    self.last_plot_mode = "elevation_sweep"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return

    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    if elev_values.size == 0:
        self.status.showMessage("Select one or more elevations/pitches to plot.")
        return
    az_values = np.asarray(sorted(self._selected_values(self.list_az)), dtype=float)
    if az_values.size == 0:
        self.status.showMessage("Select one or more azimuths/aspects to plot.")
        return
    freq_values = np.asarray(sorted(self._selected_values(self.list_freq)), dtype=float)
    if freq_values.size == 0:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    polarization = self._single_selection_value(self.list_pol, "polarization")
    if polarization is None:
        return

    p50_mode = az_values.size > 1
    skipped: list[str] = []
    plans = []
    peak_slice_cells = 0
    total_cells = 0
    for name, dataset in datasets:
        freq_indices = self._axis_selection_for_dataset(
            reference, dataset, "frequency", freq_values
        )
        az_indices = self._axis_selection_for_dataset(
            reference, dataset, "azimuth", az_values
        )
        elev_indices = self._axis_selection_for_dataset(
            reference, dataset, "elevation", elev_values
        )
        pol_indices = self._indices_for_values(
            dataset.polarizations, [polarization], tol=0.0
        )
        if any(value is None for value in (freq_indices, az_indices, elev_indices, pol_indices)):
            skipped.append(name)
            continue
        slice_cells = len(az_indices) * len(elev_indices)
        peak_slice_cells = max(peak_slice_cells, slice_cells)
        total_cells += slice_cells * len(freq_indices)
        plans.append(
            (name, dataset, freq_indices, az_indices, elev_indices, pol_indices)
        )

    if not plans:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self.status.showMessage(
            "No compatible data for the selected elevation/pitch, azimuth/aspect, "
            f"frequency, and polarization values.{detail}"
        )
        return
    try:
        common.validate_synchronous_plot_workload(
            operation="Elevation/Pitch sweep",
            peak_slice_cells=peak_slice_cells,
            total_cells=total_cells,
        )
    except ValueError as exc:
        self.status.showMessage(f"Plot blocked: {exc}.")
        return
    if not self._prepare_line_plot_axes(
        "elevation_sweep", "rectilinear", reference, datasets
    ):
        return

    rendered = 0
    omitted = 0
    freq_unit = self._plot_axis_unit(reference, "frequency")
    az_unit = self._plot_axis_unit(reference, "azimuth")
    az_name = self._plot_axis_name(reference, "azimuth")
    az_min, az_max = float(az_values[0]), float(az_values[-1])

    for name, dataset, freq_indices, az_indices, elev_indices, pol_indices in plans:
        x_values = self._plot_axis_values(
            reference, dataset, "elevation", dataset.elevations[elev_indices]
        )
        pol_value = dataset.polarizations[pol_indices[0]]
        for freq_idx in freq_indices:
            native_frequency = float(dataset.frequencies[freq_idx])
            frequency = float(
                self._plot_axis_values(reference, dataset, "frequency", [native_frequency])[0]
            )
            if self._button_checked(self.btn_phase):
                raw = dataset.rcs_slice(
                    np.ix_(az_indices, elev_indices, [freq_idx], [pol_indices[0]])
                )[:, :, 0, 0]
                phase_degrees = self._phase_display_degrees(dataset, raw)
                display = (
                    self._wrap_phase_degrees(
                        dataset, self._phase_p50(phase_degrees, axis=0)
                    )
                    if p50_mode else phase_degrees[0]
                )
            else:
                power = dataset.rcs_power[
                    np.ix_(az_indices, elev_indices, [freq_idx], [pol_indices[0]])
                ][:, :, 0, 0]
                power = np.where(np.isfinite(power), power, np.nan)
                if p50_mode:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        linear = np.nanmedian(power, axis=0)
                else:
                    linear = power[0]
                display = self._display_from_linear(
                    dataset, linear, frequency_value=native_frequency
                )
            if p50_mode:
                label = (
                    f"{name} | Pol {pol_value}, Freq {frequency:g} {freq_unit}, "
                    f"P50 over {az_name} ({az_min:g},{az_max:g}) {az_unit}"
                )
            else:
                label = (
                    f"{name} | Pol {pol_value}, Freq {frequency:g} {freq_unit}, "
                    f"{az_name} {az_min:g} {az_unit}"
                )
            if not np.any(np.isfinite(display)):
                continue
            if rendered < common.MAX_LINE_SERIES:
                self._plot_bounded_line(self.plot_ax, x_values, display, label=label, dataset=dataset)
                rendered += 1
            else:
                omitted += 1

    if rendered == 0:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self.status.showMessage(
            "No compatible data for the selected elevation/pitch, azimuth/aspect, "
            f"frequency, and polarization values.{detail}"
        )
        return
    if omitted:
        self._note_plot_render(
            f"Displayed {common.MAX_LINE_SERIES} of {common.MAX_LINE_SERIES + omitted} "
            "series; narrow frequency selections to show the rest."
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "elevation"))
    self.plot_ax.set_ylabel(
        self._display_axis_label(datasets, tag=" P50" if p50_mode else "")
    )
    self._update_legend_visibility()
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(elev_values[0]))
    self.spin_plot_xmax.setValue(float(elev_values[-1]))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    status = "Elevation/Pitch sweep plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
