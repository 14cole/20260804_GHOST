from __future__ import annotations

import numpy as np


def _collect_series(self, dataset, name, az_values_sel, elev_values_sel, freq_values_sel, pol_value_sel):
    az_values = np.asarray(sorted(az_values_sel), dtype=float)
    az_indices = self._indices_for_values(dataset.azimuths, az_values, tol=1e-6)
    freq_values = sorted(freq_values_sel, key=float)
    elev_values = sorted(elev_values_sel, key=float)
    freq_indices = self._indices_for_values(dataset.frequencies, freq_values, tol=1e-6)
    elev_indices = self._indices_for_values(dataset.elevations, elev_values, tol=1e-6)
    pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
    if (
        az_indices is None
        or freq_indices is None
        or elev_indices is None
        or pol_indices is None
    ):
        return None

    pol_value = dataset.polarizations[pol_indices[0]]
    series = []
    for f_idx in freq_indices:
        freq_value = float(dataset.frequencies[f_idx])
        for e_idx in elev_indices:
            elev_value = dataset.elevations[e_idx]
            if self._button_checked(self.btn_phase):
                raw_values = dataset.rcs[az_indices, e_idx, f_idx, pol_indices[0]]
            else:
                raw_values = dataset.rcs_power[az_indices, e_idx, f_idx, pol_indices[0]]
            rcs_display = self._display_from_values(dataset, raw_values, frequency_value=freq_value)
            label = f"{name} | Pol {pol_value}, Freq {freq_value:g} GHz, El {elev_value:g} deg"
            series.append((rcs_display, label))
    return az_values, series


def render(self) -> None:
    self.last_plot_mode = "azimuth_polar"
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
    pbp_active = self._button_checked(self.btn_pbp) and (
        len(datasets) > 1 or len(freq_values_sel) > 1 or len(elev_values_sel) > 1
    )

    self._ensure_axes("polar")
    if not self._button_checked(self.btn_hold):
        self.plot_ax.clear()
        self._style_plot_axes()

    skipped = []
    if pbp_active:
        az_values = np.asarray(sorted(az_values_sel), dtype=float)
        theta = np.deg2rad(az_values)
        freq_values = sorted(freq_values_sel, key=float)
        elev_values = sorted(elev_values_sel, key=float)
        series_list: list[np.ndarray] = []
        for name, dataset in datasets:
            az_indices = self._indices_for_values(dataset.azimuths, az_values, tol=1e-6)
            freq_indices = self._indices_for_values(dataset.frequencies, freq_values, tol=1e-6)
            elev_indices = self._indices_for_values(dataset.elevations, elev_values, tol=1e-6)
            pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
            if (
                az_indices is None
                or freq_indices is None
                or elev_indices is None
                or pol_indices is None
            ):
                skipped.append(name)
                continue
            for f_idx in freq_indices:
                freq_value = float(dataset.frequencies[f_idx])
                for e_idx in elev_indices:
                    if self._button_checked(self.btn_phase):
                        raw_values = dataset.rcs[az_indices, e_idx, f_idx, pol_indices[0]]
                    else:
                        raw_values = dataset.rcs_power[az_indices, e_idx, f_idx, pol_indices[0]]
                    rcs_display = self._display_from_values(dataset, raw_values, frequency_value=freq_value)
                    series_list.append(rcs_display)

        if series_list:
            stacked = np.vstack(series_list)
            y_min = np.nanmin(stacked, axis=0)
            y_max = np.nanmax(stacked, axis=0)
            density = np.sum(np.isfinite(stacked), axis=0)
            freq_label = (
                f"{freq_values[0]}-{freq_values[-1]} GHz"
                if len(freq_values) > 1
                else f"{freq_values[0]} GHz"
            )
            elev_label = (
                f"{elev_values[0]}-{elev_values[-1]} deg"
                if len(elev_values) > 1
                else f"{elev_values[0]} deg"
            )
            label = f"PBP Pol {pol_value_sel}, Freq {freq_label}, El {elev_label}"
            self._plot_pbp_fill(theta, y_min, y_max, label, polar=True, density=density)
            self.plot_ax.plot(theta, y_min, color="#8a8a8a", linewidth=1, label="_nolegend_")
            self.plot_ax.plot(theta, y_max, color="#8a8a8a", linewidth=1, label="_nolegend_")
    else:
        for name, dataset in datasets:
            collected = _collect_series(
                self, dataset, name, az_values_sel, elev_values_sel, freq_values_sel, pol_value_sel
            )
            if collected is None:
                skipped.append(name)
                continue
            az_values, series = collected
            theta = np.deg2rad(az_values)
            for rcs_display, label in series:
                self.plot_ax.plot(theta, rcs_display, label=label)

    self.plot_ax.set_xlabel("Azimuth (deg)")
    self.plot_ax.set_ylabel(self._display_axis_label(datasets))
    self._update_legend_visibility()

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(-180.0)
    self.spin_plot_xmax.setValue(180.0)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)

    self._apply_plot_limits()
    if skipped:
        skipped_list = ", ".join(skipped)
        self.status.showMessage(f"Azimuth (Polar) updated. Skipped: {skipped_list}.")
    else:
        self.status.showMessage("Azimuth (Polar) plot updated.")
