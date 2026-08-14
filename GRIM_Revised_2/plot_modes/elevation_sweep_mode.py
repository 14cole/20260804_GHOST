from __future__ import annotations

import numpy as np


def render(self) -> None:
    self.last_plot_mode = "elevation_sweep"
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return

    elev_values_sel = self._selected_values(self.list_elev)
    if not elev_values_sel:
        self.status.showMessage("Select one or more elevations to plot.")
        return

    az_values_sel = self._selected_values(self.list_az)
    if not az_values_sel:
        self.status.showMessage("Select one or more azimuths to plot.")
        return

    freq_values_sel = self._selected_values(self.list_freq)
    if not freq_values_sel:
        self.status.showMessage("Select one or more frequencies to plot.")
        return

    pol_value_sel = self._single_selection_value(self.list_pol, "polarization")
    if pol_value_sel is None:
        return

    elev_values = np.asarray(elev_values_sel, dtype=float)
    order = np.argsort(elev_values)
    elev_values = elev_values[order]
    emin = float(elev_values.min())
    emax = float(elev_values.max())

    p50_mode = len(az_values_sel) > 1
    az_min = float(np.min(az_values_sel))
    az_max = float(np.max(az_values_sel))

    self._ensure_axes("rectilinear")
    if not self._button_checked(self.btn_hold):
        self.plot_ax.clear()
        self._style_plot_axes()

    skipped = []
    for name, dataset in datasets:
        freq_indices = self._indices_for_values(dataset.frequencies, freq_values_sel)
        az_indices = self._indices_for_values(dataset.azimuths, az_values_sel)
        elev_indices = self._indices_for_values(dataset.elevations, elev_values_sel)
        pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
        if (
            freq_indices is None
            or az_indices is None
            or elev_indices is None
            or pol_indices is None
        ):
            skipped.append(name)
            continue

        pol_value = dataset.polarizations[pol_indices[0]]
        frequency_unit = str((dataset.units or {}).get("frequency", "GHz"))
        for f_idx in freq_indices:
            freq_value = float(dataset.frequencies[f_idx])
            if self._button_checked(self.btn_phase):
                rcs_slice = dataset.rcs_slice(np.ix_(az_indices, elev_indices, [f_idx], [pol_indices[0]]))
                rcs_slice = rcs_slice[:, :, 0, 0]  # (az, elev)
                phase_deg = np.degrees(np.angle(rcs_slice))
                phase_deg = np.where(np.isfinite(phase_deg), phase_deg, np.nan)
                y_values = np.nanmedian(phase_deg, axis=0) if p50_mode else phase_deg[0]
            else:
                pwr_slice = dataset.rcs_power[np.ix_(az_indices, elev_indices, [f_idx], [pol_indices[0]])]
                pwr_slice = pwr_slice[:, :, 0, 0]  # (az, elev)
                pwr_slice = np.where(np.isfinite(pwr_slice), pwr_slice, np.nan)
                y_lin = np.nanmedian(pwr_slice, axis=0) if p50_mode else pwr_slice[0]
                y_values = self._display_from_linear(dataset, y_lin, freq_value)
            y_values = np.asarray(y_values)[order]
            if p50_mode:
                label = (
                    f"{name} | Pol {pol_value}, Freq {freq_value:g} {frequency_unit}, "
                    f"P50 over az ({az_min:g},{az_max:g})"
                )
            else:
                label = (
                    f"{name} | Pol {pol_value}, Freq {freq_value:g} {frequency_unit}, "
                    f"Az {az_min:g} deg"
                )
            self.plot_ax.plot(elev_values, y_values, label=label)

    self.plot_ax.set_xlabel("Elevation (deg)")
    self.plot_ax.set_ylabel(self._display_axis_label(datasets, tag=" P50" if p50_mode else ""))
    self._update_legend_visibility()
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(emin)
    self.spin_plot_xmax.setValue(emax)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    if skipped:
        skipped_list = ", ".join(skipped)
        self.status.showMessage(f"Elevation sweep updated. Skipped: {skipped_list}.")
    else:
        self.status.showMessage("Elevation sweep plot updated.")
