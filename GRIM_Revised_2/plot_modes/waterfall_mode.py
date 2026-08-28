from __future__ import annotations

import numpy as np

from . import common


def render(self) -> None:
    self.last_plot_mode = "waterfall"
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

    phase_mode = self._button_checked(self.btn_phase)
    skipped: list[str] = []
    plans = []
    panel_count = 0
    peak_slice_cells = 0
    total_source_cells = 0
    total_display_cells = 0
    for dataset_name, dataset in datasets:
        az_indices = self._axis_selection_for_dataset(
            reference, dataset, "azimuth", az_values
        )
        freq_indices = self._axis_selection_for_dataset(
            reference, dataset, "frequency", freq_values
        )
        elev_indices = self._axis_selection_for_dataset(
            reference, dataset, "elevation", elev_values
        )
        pol_indices = self._indices_for_values(dataset.polarizations, [polarization], tol=0.0)
        if any(value is None for value in (az_indices, freq_indices, elev_indices, pol_indices)):
            skipped.append(dataset_name)
            continue
        panel_count += len(elev_indices)
        if panel_count > common.MAX_WATERFALL_PANELS:
            self.status.showMessage(
                f"Waterfall blocked: selection would create more than "
                f"{common.MAX_WATERFALL_PANELS} panels. Select fewer datasets or "
                "elevations/pitches."
            )
            return
        slice_cells = len(az_indices) * len(freq_indices)
        peak_slice_cells = max(peak_slice_cells, slice_cells)
        total_source_cells += slice_cells * len(elev_indices)
        panel_display_cells = common.bounded_image_cell_count(
            len(az_indices), len(freq_indices)
        )
        total_display_cells += panel_display_cells * len(elev_indices)
        if phase_mode and common.image_requires_decimation(
            len(az_indices), len(freq_indices)
        ):
            self.status.showMessage(
                "Phase waterfall blocked: the selected image exceeds the interactive "
                "display limit, and scalar peak decimation would distort wrapped phase. "
                "Narrow the selected azimuth/aspect or frequency axes and plot again."
            )
            return
        plans.append(
            (dataset_name, dataset, az_indices, freq_indices, elev_indices, pol_indices)
        )

    if not plans:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self.status.showMessage(
            "No compatible data for the selected waterfall axes and polarization."
            f"{detail}"
        )
        return
    try:
        common.validate_synchronous_plot_workload(
            operation="Waterfall plot",
            peak_slice_cells=peak_slice_cells,
            total_cells=total_source_cells,
        )
        common.validate_aggregate_image_cells(
            total_display_cells,
            panel_count=panel_count,
            operation="Waterfall plot",
        )
    except ValueError as exc:
        self.status.showMessage(f"Plot blocked: {exc}.")
        return

    panel_data: list[dict[str, object]] = []
    for dataset_name, dataset, az_indices, freq_indices, elev_indices, pol_indices in plans:
        display_azimuths = self._plot_axis_values(
            reference, dataset, "azimuth", dataset.azimuths[az_indices]
        )
        native_frequencies = np.asarray(dataset.frequencies[freq_indices], dtype=float)
        display_frequencies = self._plot_axis_values(
            reference, dataset, "frequency", native_frequencies
        )
        for elev_idx in elev_indices:
            native_elevation = float(dataset.elevations[elev_idx])
            display_elevation = float(
                self._plot_axis_values(
                    reference, dataset, "elevation", [native_elevation]
                )[0]
            )
            if phase_mode:
                raw = dataset.rcs_slice(
                    np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])
                )[:, 0, :, 0]
            else:
                raw = dataset.rcs_power[
                    np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])
                ][:, 0, :, 0]
            display = self._display_from_values(
                dataset,
                raw,
                frequency_value=native_frequencies.reshape(1, -1),
            )
            display = np.where(np.isfinite(display), display, np.nan)
            if not np.any(np.isfinite(display)):
                continue
            bounded_az, bounded_freq, bounded_display = self._bounded_plot_image(
                display_azimuths, display_frequencies, display
            )
            panel_data.append(
                {
                    "dataset_name": dataset_name,
                    "elevation": display_elevation,
                    "azimuths": bounded_az,
                    "frequencies": bounded_freq,
                    "display": bounded_display,
                }
            )

    if not panel_data:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self.status.showMessage(
            "No compatible data for the selected waterfall axes and polarization."
            f"{detail}"
        )
        return

    self._remove_colorbar()
    self.plot_figure.clear()
    axes = self.plot_figure.subplots(
        nrows=len(panel_data), ncols=1, sharex=False, sharey=False
    )
    if len(panel_data) == 1:
        axes = [axes]
    self.plot_axes = list(axes)
    self.plot_ax = self.plot_axes[0]
    self.plot_figure.set_facecolor(self._current_plot_bg())
    for ax in self.plot_axes:
        self._style_axes(ax)

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax
    shared_scale = bool(self.chk_colorbar_shared.isChecked())
    shared_limits = (
        common.finite_data_limits(panel["display"] for panel in panel_data)
        if shared_scale and not use_clamp
        else None
    )
    plot_vmin = zmin if use_clamp else (
        shared_limits[0] if shared_limits is not None else None
    )
    plot_vmax = zmax if use_clamp else (
        shared_limits[1] if shared_limits is not None else None
    )
    elev_name = self._plot_axis_name(reference, "elevation")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    meshes = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for ax, panel in zip(self.plot_axes, panel_data):
        panel_az = panel["azimuths"]
        panel_freq = panel["frequencies"]
        display = panel["display"]
        mesh = ax.pcolormesh(
            panel_az,
            panel_freq,
            display.T,
            shading="auto",
            cmap=cmap,
            vmin=plot_vmin,
            vmax=plot_vmax,
        )
        meshes.append(mesh)
        ax.set_title(
            f"{panel['dataset_name']} | {elev_name} {panel['elevation']:g} {elev_unit}",
            color=self._current_plot_text(),
        )
        ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
        ax.set_ylabel(self._plot_axis_label(reference, "frequency"))
        xmins.append(float(np.min(panel_az)))
        xmaxs.append(float(np.max(panel_az)))
        ymins.append(float(np.min(panel_freq)))
        ymaxs.append(float(np.max(panel_freq)))

    if self.chk_colorbar.isChecked():
        if shared_scale:
            colorbar = self.plot_figure.colorbar(meshes[-1], ax=self.plot_axes)
            self.plot_colorbars = [colorbar]
        else:
            self.plot_colorbars = [
                self.plot_figure.colorbar(mesh, ax=ax)
                for ax, mesh in zip(self.plot_axes, meshes)
            ]
        for colorbar in self.plot_colorbars:
            self._apply_colorbar_ticks(colorbar)
            colorbar.set_label(
                self._display_axis_label(datasets), color=self._current_plot_text()
            )
            colorbar.ax.tick_params(colors=self._current_plot_text())
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self._current_plot_text())

    for spin in (
        self.spin_plot_xmin,
        self.spin_plot_xmax,
        self.spin_plot_ymin,
        self.spin_plot_ymax,
    ):
        spin.blockSignals(True)
    self.spin_plot_xmin.setValue(min(xmins))
    self.spin_plot_xmax.setValue(max(xmaxs))
    self.spin_plot_ymin.setValue(min(ymins))
    self.spin_plot_ymax.setValue(max(ymaxs))
    for spin in (
        self.spin_plot_xmin,
        self.spin_plot_xmax,
        self.spin_plot_ymin,
        self.spin_plot_ymax,
    ):
        spin.blockSignals(False)

    self._apply_plot_limits()
    status = "Waterfall plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)
