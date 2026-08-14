from __future__ import annotations

import numpy as np


def render(self) -> None:
    self.last_plot_mode = "waterfall"
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return

    az_values_sel = self._selected_values(self.list_az)
    if not az_values_sel:
        self.status.showMessage("Select one or more azimuths to plot.")
        return

    freq_values_sel = self._selected_values(self.list_freq)
    if not freq_values_sel:
        self.status.showMessage("Select one or more frequencies to plot.")
        return

    elev_values_sel = self._selected_values(self.list_elev)
    if not elev_values_sel:
        self.status.showMessage("Select one or more elevations to plot.")
        return

    pol_value_sel = self._single_selection_value(self.list_pol, "polarization")
    if pol_value_sel is None:
        return

    panel_data: list[dict[str, object]] = []
    skipped: list[str] = []

    for dataset_name, dataset in datasets:
        az_indices = self._indices_for_values(dataset.azimuths, az_values_sel)
        freq_indices = self._indices_for_values(dataset.frequencies, freq_values_sel)
        elev_indices = self._indices_for_values(dataset.elevations, elev_values_sel)
        pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
        if (
            az_indices is None
            or freq_indices is None
            or elev_indices is None
            or pol_indices is None
        ):
            skipped.append(dataset_name)
            continue

        az_values = dataset.azimuths[az_indices]
        freq_values = dataset.frequencies[freq_indices]
        az_order = np.argsort(az_values)
        freq_order = np.argsort(freq_values)
        az_values = az_values[az_order]
        freq_values = freq_values[freq_order]

        for elev_idx in elev_indices:
            elev_value = dataset.elevations[elev_idx]
            if self._button_checked(self.btn_phase):
                raw_values = dataset.rcs[np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])]
            else:
                raw_values = dataset.rcs_power[np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])]
            raw_values = raw_values[:, 0, :, 0]
            raw_values = raw_values[np.ix_(az_order, freq_order)]
            rcs_display = self._display_from_values(
                dataset, raw_values,
                frequency_value=np.asarray(freq_values, dtype=float).reshape(1, -1),
            )
            rcs_display = np.where(np.isfinite(rcs_display), rcs_display, np.nan)
            panel_data.append(
                {
                    "dataset_name": dataset_name,
                    "dataset": dataset,
                    "elev_value": elev_value,
                    "az_values": az_values,
                    "freq_values": freq_values,
                    "rcs_display": rcs_display,
                }
            )

    if not panel_data:
        if skipped:
            skipped_list = ", ".join(skipped)
            self.status.showMessage(f"No compatible datasets for waterfall. Skipped: {skipped_list}.")
        else:
            self.status.showMessage("No compatible selections for waterfall plot.")
        return

    self._remove_colorbar()
    self.plot_figure.clear()
    axes = self.plot_figure.subplots(
        nrows=len(panel_data),
        ncols=1,
        sharex=False,
        sharey=False,
    )
    if len(panel_data) == 1:
        axes = [axes]
    self.plot_axes = list(axes)
    self.plot_ax = self.plot_axes[0]
    self.plot_figure.set_facecolor(self._current_plot_bg())
    for ax in self.plot_axes:
        self._style_axes(ax)

    meshes = []
    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax

    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []

    for ax, panel in zip(self.plot_axes, panel_data):
        az_values = panel["az_values"]
        freq_values = panel["freq_values"]
        rcs_display = panel["rcs_display"]
        dataset_name = panel["dataset_name"]
        elev_value = panel["elev_value"]

        mesh = ax.pcolormesh(
            az_values,
            freq_values,
            rcs_display.T,
            shading="auto",
            cmap=cmap,
            vmin=zmin if use_clamp else None,
            vmax=zmax if use_clamp else None,
        )
        meshes.append(mesh)
        ax.set_title(f"{dataset_name} | Elevation {elev_value} deg", color=self._current_plot_text())
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Frequency (GHz)")

        xmins.append(float(np.min(az_values)))
        xmaxs.append(float(np.max(az_values)))
        ymins.append(float(np.min(freq_values)))
        ymaxs.append(float(np.max(freq_values)))

    if self.chk_colorbar.isChecked():
        if self.chk_colorbar_shared.isChecked():
            colorbar = self.plot_figure.colorbar(meshes[-1], ax=self.plot_axes)
            self.plot_colorbars = [colorbar]
        else:
            self.plot_colorbars = []
            for ax, mesh in zip(self.plot_axes, meshes):
                self.plot_colorbars.append(self.plot_figure.colorbar(mesh, ax=ax))
        for colorbar in self.plot_colorbars:
            self._apply_colorbar_ticks(colorbar)
            colorbar.set_label(self._display_axis_label(datasets), color=self._current_plot_text())
            colorbar.ax.tick_params(colors=self._current_plot_text())
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self._current_plot_text())

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(min(xmins))
    self.spin_plot_xmax.setValue(max(xmaxs))
    self.spin_plot_ymin.setValue(min(ymins))
    self.spin_plot_ymax.setValue(max(ymaxs))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    self._apply_plot_limits()
    if skipped:
        skipped_list = ", ".join(skipped)
        self.status.showMessage(f"Waterfall plot updated. Skipped: {skipped_list}.")
    else:
        self.status.showMessage("Waterfall plot updated.")
