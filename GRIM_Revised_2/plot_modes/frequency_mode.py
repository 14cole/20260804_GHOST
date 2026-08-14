from __future__ import annotations

import numpy as np


def render(self) -> None:
    self.last_plot_mode = "frequency"
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return

    freq_values_sel = self._selected_values(self.list_freq)
    if not freq_values_sel:
        self.status.showMessage("Select one or more frequencies to plot.")
        return

    az_values_sel = self._selected_values(self.list_az)
    if not az_values_sel:
        self.status.showMessage("Select one or more azimuths to plot.")
        return

    elev_values_sel = self._selected_values(self.list_elev)
    if not elev_values_sel:
        self.status.showMessage("Select one or more elevations to plot.")
        return

    pol_value_sel = self._single_selection_value(self.list_pol, "polarization")
    if pol_value_sel is None:
        return
    pbp_active = self._button_checked(self.btn_pbp) and (
        len(datasets) > 1 or len(elev_values_sel) > 1 or len(az_values_sel) > 1
    )

    freq_values = np.asarray(freq_values_sel, dtype=float)
    order = np.argsort(freq_values)
    freq_values = freq_values[order]
    fmin = float(freq_values.min())
    fmax = float(freq_values.max())

    self._ensure_axes("rectilinear")
    if not self._button_checked(self.btn_hold):
        self.plot_ax.clear()
        self._style_plot_axes()

    skipped = []
    if pbp_active:
        elev_values = sorted(elev_values_sel, key=float)
        series_list: list[np.ndarray] = []
        for name, dataset in datasets:
            freq_indices = self._indices_for_values(dataset.frequencies, freq_values_sel)
            az_indices = self._indices_for_values(dataset.azimuths, az_values_sel)
            elev_indices = self._indices_for_values(dataset.elevations, elev_values)
            pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
            if (
                freq_indices is None
                or az_indices is None
                or elev_indices is None
                or pol_indices is None
            ):
                skipped.append(name)
                continue

            freq_axis_values = np.asarray([float(dataset.frequencies[idx]) for idx in freq_indices], dtype=float)
            for elev_idx, elev_value in zip(elev_indices, elev_values):
                if self._button_checked(self.btn_phase):
                    rcs_slice = dataset.rcs[np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])]
                    rcs_slice = rcs_slice[:, 0, :, 0]
                    phase_deg = np.degrees(np.angle(rcs_slice))
                    phase_deg = np.where(np.isfinite(phase_deg), phase_deg, np.nan)
                    rcs_p50_values = np.nanmedian(phase_deg, axis=0)
                else:
                    pwr_slice = dataset.rcs_power[np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])]
                    pwr_slice = pwr_slice[:, 0, :, 0]
                    pwr_slice = np.where(np.isfinite(pwr_slice), pwr_slice, np.nan)
                    rcs_p50_lin = np.nanmedian(pwr_slice, axis=0)
                    rcs_p50_values = self._display_from_linear(
                        dataset, rcs_p50_lin,
                        frequency_value=np.asarray(freq_axis_values, dtype=float),
                    )
                rcs_p50_values = np.asarray(rcs_p50_values)[order]
                series_list.append(rcs_p50_values)

        if series_list:
            stacked = np.vstack(series_list)
            y_min = np.nanmin(stacked, axis=0)
            y_max = np.nanmax(stacked, axis=0)
            density = np.sum(np.isfinite(stacked), axis=0)
            az_min = float(np.min(az_values_sel))
            az_max = float(np.max(az_values_sel))
            elev_label = (
                f"{elev_values[0]}-{elev_values[-1]} deg"
                if len(elev_values) > 1
                else f"{elev_values[0]} deg"
            )
            label = (
                f"PBP Pol {pol_value_sel}, El {elev_label}, "
                f"P50 over az ({az_min},{az_max})"
            )
            self._plot_pbp_fill(freq_values, y_min, y_max, label, polar=False, density=density)
            self.plot_ax.plot(freq_values, y_min, color="#8a8a8a", linewidth=1, label="_nolegend_")
            self.plot_ax.plot(freq_values, y_max, color="#8a8a8a", linewidth=1, label="_nolegend_")
    else:
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
            freq_axis_values = np.asarray([float(dataset.frequencies[idx]) for idx in freq_indices], dtype=float)
            for elev_idx in elev_indices:
                elev_value = dataset.elevations[elev_idx]
                if self._button_checked(self.btn_phase):
                    rcs_slice = dataset.rcs[np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])]
                    rcs_slice = rcs_slice[:, 0, :, 0]
                    phase_deg = np.degrees(np.angle(rcs_slice))
                    phase_deg = np.where(np.isfinite(phase_deg), phase_deg, np.nan)
                    rcs_p50_values = np.nanmedian(phase_deg, axis=0)
                else:
                    pwr_slice = dataset.rcs_power[np.ix_(az_indices, [elev_idx], freq_indices, [pol_indices[0]])]
                    pwr_slice = pwr_slice[:, 0, :, 0]
                    pwr_slice = np.where(np.isfinite(pwr_slice), pwr_slice, np.nan)
                    rcs_p50_lin = np.nanmedian(pwr_slice, axis=0)
                    rcs_p50_values = self._display_from_linear(
                        dataset, rcs_p50_lin,
                        frequency_value=np.asarray(freq_axis_values, dtype=float),
                    )
                rcs_p50_values = np.asarray(rcs_p50_values)[order]
                az_min = float(np.min(az_values_sel))
                az_max = float(np.max(az_values_sel))
                label = (
                    f"{name} | Pol {pol_value}, El {elev_value} deg, "
                    f"P50 over az ({az_min},{az_max})"
                )
                self.plot_ax.plot(freq_values, rcs_p50_values, label=label)
    self.plot_ax.set_xlabel("Frequency (GHz)")
    self.plot_ax.set_ylabel(self._display_axis_label(datasets, tag=" P50"))
    self._update_legend_visibility()
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(fmin)
    self.spin_plot_xmax.setValue(fmax)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    if skipped:
        skipped_list = ", ".join(skipped)
        self.status.showMessage(f"Frequency plot updated. Skipped: {skipped_list}.")
    else:
        self.status.showMessage("Frequency plot updated.")
