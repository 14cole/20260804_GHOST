from __future__ import annotations

import numpy as np


# When more than one axis has multiple values selected, pick the sweep axis
# by this priority. Azimuth wins first so the original (az-only) compare
# behaviour is preserved for users with existing workflows.
_SWEEP_PRIORITY = ("azimuth", "elevation", "frequency")
_AXIS_LABEL = {
    "azimuth": "Azimuth (deg)",
    "elevation": "Elevation (deg)",
    "frequency": "Frequency (GHz)",
}


def _determine_sweep_axis(az_sel, elev_sel, freq_sel) -> str | None:
    has_multi = {
        "azimuth": len(az_sel) >= 2,
        "elevation": len(elev_sel) >= 2,
        "frequency": len(freq_sel) >= 2,
    }
    for axis in _SWEEP_PRIORITY:
        if has_multi[axis]:
            return axis
    return None


def _collect_series(self, dataset, name, sweep_axis,
                    az_sel, elev_sel, freq_sel, pol_value_sel):
    az_values = np.asarray(sorted(az_sel), dtype=float)
    elev_values = np.asarray(sorted(elev_sel, key=float), dtype=float)
    freq_values = np.asarray(sorted(freq_sel, key=float), dtype=float)
    az_indices = self._indices_for_values(dataset.azimuths, az_values, tol=1e-6)
    elev_indices = self._indices_for_values(dataset.elevations, elev_values, tol=1e-6)
    freq_indices = self._indices_for_values(dataset.frequencies, freq_values, tol=1e-6)
    pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
    if (
        az_indices is None
        or elev_indices is None
        or freq_indices is None
        or pol_indices is None
    ):
        return None

    pol_value = dataset.polarizations[pol_indices[0]]
    frequency_unit = str((dataset.units or {}).get("frequency", "GHz"))
    pol_idx = pol_indices[0]
    use_complex = self._button_checked(self.btn_phase)
    def _values(selection):
        return dataset.rcs_slice(selection) if use_complex else dataset.rcs_power[selection]

    series: list[tuple[np.ndarray, str]] = []
    if sweep_axis == "azimuth":
        sweep_values = az_values
        for f_idx in freq_indices:
            freq_value = float(dataset.frequencies[f_idx])
            for e_idx in elev_indices:
                elev_value = dataset.elevations[e_idx]
                raw = _values((az_indices, e_idx, f_idx, pol_idx))
                rcs_display = self._display_from_values(dataset, raw, frequency_value=freq_value)
                label = f"{name} | Pol {pol_value}, Freq {freq_value:g} {frequency_unit}, El {elev_value:g} deg"
                series.append((rcs_display, label))
    elif sweep_axis == "elevation":
        sweep_values = elev_values
        for f_idx in freq_indices:
            freq_value = float(dataset.frequencies[f_idx])
            for a_idx in az_indices:
                az_value = dataset.azimuths[a_idx]
                raw = _values((a_idx, elev_indices, f_idx, pol_idx))
                rcs_display = self._display_from_values(dataset, raw, frequency_value=freq_value)
                label = f"{name} | Pol {pol_value}, Freq {freq_value:g} {frequency_unit}, Az {az_value:g} deg"
                series.append((rcs_display, label))
    else:  # frequency
        sweep_values = freq_values
        # For frequency sweep, dB conversion needs the per-bin freq, so pass
        # the full freq vector (matches frequency_mode's convention).
        freq_axis_values = np.asarray(
            [float(dataset.frequencies[idx]) for idx in freq_indices], dtype=float
        )
        for e_idx in elev_indices:
            elev_value = dataset.elevations[e_idx]
            for a_idx in az_indices:
                az_value = dataset.azimuths[a_idx]
                raw = _values((a_idx, e_idx, freq_indices, pol_idx))
                rcs_display = self._display_from_values(
                    dataset, raw, frequency_value=freq_axis_values
                )
                label = f"{name} | Pol {pol_value}, El {elev_value:g} deg, Az {az_value:g} deg"
                series.append((rcs_display, label))
    return sweep_values, series


def render(self) -> None:
    self.last_plot_mode = "compare"
    datasets = self._selected_datasets()
    if len(datasets) != 2:
        self.status.showMessage("Compare: select exactly 2 datasets.")
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

    sweep_axis = _determine_sweep_axis(az_values_sel, elev_values_sel, freq_values_sel)
    if sweep_axis is None:
        self.status.showMessage(
            "Compare: select 2+ azimuths, elevations, or frequencies to sweep over."
        )
        return

    name_a, ds_a = datasets[0]
    name_b, ds_b = datasets[1]

    collected_a = _collect_series(
        self, ds_a, name_a, sweep_axis,
        az_values_sel, elev_values_sel, freq_values_sel, pol_value_sel,
    )
    collected_b = _collect_series(
        self, ds_b, name_b, sweep_axis,
        az_values_sel, elev_values_sel, freq_values_sel, pol_value_sel,
    )
    if collected_a is None:
        self.status.showMessage(f"Compare: '{name_a}' missing selected parameters.")
        return
    if collected_b is None:
        self.status.showMessage(f"Compare: '{name_b}' missing selected parameters.")
        return

    x_a, series_a = collected_a
    x_b, series_b = collected_b

    top_ax, res_ax = self._ensure_compare_axes()
    top_ax.clear()
    res_ax.clear()
    self._style_axes(top_ax)
    self._style_axes(res_ax)

    text_color = self._current_plot_text()
    grid_color = self._current_plot_grid()

    color_a = "#4fc3f7"
    color_b = "#ff8a65"
    for rcs_disp, label in series_a:
        top_ax.plot(x_a, rcs_disp, color=color_a, linewidth=1.5, label=label)
    for rcs_disp, label in series_b:
        top_ax.plot(x_b, rcs_disp, color=color_b, linewidth=1.5, linestyle="--", label=label)

    top_ax.set_ylabel(self._display_axis_label(datasets))
    self._update_legend_visibility()

    rcs_a0 = series_a[0][0]
    rcs_b0 = series_b[0][0]

    x_a_r = np.round(x_a, 8)
    x_b_r = np.round(x_b, 8)
    mask_a = np.isin(x_a_r, x_b_r)
    mask_b = np.isin(x_b_r, x_a_r)

    if mask_a.sum() < 2:
        res_ax.text(
            0.5, 0.5,
            f"No common {sweep_axis} points for residual",
            transform=res_ax.transAxes, ha="center", va="center",
            color=text_color, fontsize=8,
        )
    else:
        x_common = x_a[mask_a]
        y_a = rcs_a0[mask_a]
        y_b = rcs_b0[mask_b]
        residual = y_a - y_b

        res_ax.axhline(0, color=grid_color, linewidth=0.8, linestyle="--")
        res_ax.plot(x_common, residual, color="#a5d6a7", linewidth=1.2, label=f"{name_a} − {name_b}")
        res_ax.fill_between(x_common, residual, 0, alpha=0.15, color="#a5d6a7")
        residual_unit = self._display_unit(datasets)
        res_ax.set_ylabel(f"Residual ({residual_unit})", fontsize=8)

        finite = np.isfinite(residual)
        if finite.sum() > 1:
            res_fin = residual[finite]
            mean_err = float(np.mean(res_fin))
            rms_err = float(np.sqrt(np.mean(res_fin ** 2)))
            max_err = float(np.max(np.abs(res_fin)))

            fin_both = finite & np.isfinite(y_a) & np.isfinite(y_b)
            if fin_both.sum() > 1:
                corr = float(np.corrcoef(y_a[fin_both], y_b[fin_both])[0, 1])
                corr_str = f"   Corr: {corr:.4f}"
            else:
                corr_str = ""

            stats_text = (
                f"Mean: {mean_err:+.2f} {residual_unit}   "
                f"RMS: {rms_err:.2f} {residual_unit}   "
                f"Max|err|: {max_err:.2f} {residual_unit}"
                + corr_str
            )
            top_ax.set_title(stats_text, fontsize=8, color=text_color, pad=4)

    if sweep_axis == "frequency":
        frequency_units = {str((dataset.units or {}).get("frequency", "GHz")) for _, dataset in datasets}
        frequency_unit = next(iter(frequency_units)) if len(frequency_units) == 1 else "mixed units"
        res_ax.set_xlabel(f"Frequency ({frequency_unit})")
    else:
        res_ax.set_xlabel(_AXIS_LABEL[sweep_axis])
    self._apply_plot_limits()
    self.status.showMessage(f"Compare plot updated ({sweep_axis} sweep).")
