"""Azimuth vs Down-Range image — partial ISAR.

For each selected azimuth, IFFT over frequency to build a range profile;
stack the profiles side by side. Unlike the ISAR mode this does NOT FFT
across azimuth, so the X axis stays in degrees rather than collapsing to
a spatial cross-range coordinate.

Useful for spotting which look-angles a particular scatterer lights up at,
diagnosing range-walk before doing a full ISAR, or quickly seeing target
extent without committing to a small azimuth window.
"""
from __future__ import annotations

import numpy as np

from .isar_mode import _resample_complex_uniform, _length_unit, _unit_to_hz_scale


def render(self) -> None:
    self.last_plot_mode = "az_vs_range"
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    if not az_indices:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_indices = sorted(self._selected_indices(self.list_freq))
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for range processing.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return

    # Sort axes ascending; build the (n_az, n_freq) complex slice.
    az_values = self.active_dataset.azimuths[az_indices].astype(float)
    az_order = np.argsort(az_values)
    sorted_az_indices = [az_indices[i] for i in az_order]
    az_values = az_values[az_order]
    if not np.all(np.isfinite(az_values)) or np.any(np.diff(az_values) <= 0):
        self.status.showMessage("Azimuth samples must be strictly increasing.")
        return

    freq_values = self.active_dataset.frequencies[freq_indices].astype(float)
    freq_order = np.argsort(freq_values)
    sorted_freq_indices = [freq_indices[i] for i in freq_order]
    freq_values = freq_values[freq_order]
    if np.any(np.diff(freq_values) <= 0):
        self.status.showMessage("Frequency samples must be strictly increasing.")
        return

    rcs_slice = self.active_dataset.rcs[
        np.ix_(sorted_az_indices, [elev_idx], sorted_freq_indices, [pol_idx])
    ][:, 0, :, 0]
    rcs_slice = np.where(np.isfinite(rcs_slice), rcs_slice, 0.0)

    # IFFT along freq requires uniform freq spacing — match what ISAR does.
    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
    freq_hz = freq_values * _unit_to_hz_scale(freq_unit)
    freq_hz_uniform, rcs_slice, fr_nonuniformity = _resample_complex_uniform(
        freq_hz, rcs_slice, axis=1
    )
    n_freq = freq_hz_uniform.size
    df = float(np.mean(np.diff(freq_hz_uniform)))

    # Window over freq (re-uses ISAR window selector).
    win_freq = self._isar_window(n_freq)
    rcs_windowed = rcs_slice * win_freq

    # Range processing: IFFT and shift so range=0 sits at array center.
    range_image = np.fft.ifft(rcs_windowed, axis=1)
    range_image = np.fft.fftshift(range_image, axes=1)

    # Coherent-gain normalisation so a unit-amplitude scatterer reads near
    # 0 dBsm — matches the ISAR magnitude convention so users can re-use
    # the same z-clamp range.
    coh = float(np.mean(win_freq))
    if coh > 0.0:
        range_image = range_image / coh

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = _length_unit(
        units_combo.currentText() if units_combo else "m"
    )
    c0 = 299_792_458.0
    range_axis = (
        np.fft.fftshift(np.fft.fftfreq(n_freq, d=df)) * (c0 / 2.0) * unit_scale
    )

    magnitude = np.abs(range_image)

    # Optional peak normalisation (re-uses ISAR toggle).
    pn_widget = getattr(self, "chk_isar_peak_normalize", None)
    peak_norm = bool(pn_widget.isChecked()) if pn_widget else False
    if peak_norm:
        peak = float(magnitude.max())
        if peak > 0.0:
            magnitude = magnitude / peak

    if self._plot_scale_is_linear():
        display = magnitude
    else:
        display = self.active_dataset.rcs_to_dbsm(magnitude)

    # Build the figure.
    self._remove_colorbar()
    self.plot_figure.clear()
    self.plot_ax = self.plot_figure.add_subplot(111)
    self.plot_axes = None
    self._style_plot_axes()

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax

    # display shape: (n_az, n_freq). imshow wants (n_y, n_x), so transpose.
    mesh = self.plot_ax.imshow(
        display.T,
        extent=[
            float(az_values[0]),
            float(az_values[-1]),
            float(range_axis[0]),
            float(range_axis[-1]),
        ],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=zmin if use_clamp else None,
        vmax=zmax if use_clamp else None,
    )

    self.plot_ax.set_xlabel("Azimuth (deg)")
    self.plot_ax.set_ylabel(f"Down-Range ({unit_name})")
    elev_value = self.active_dataset.elevations[elev_idx]
    pol_value = self.active_dataset.polarizations[pol_idx]
    self.plot_ax.set_title(
        f"Az vs Down-Range | Elevation {elev_value} deg | Pol {pol_value}",
        color=self._current_plot_text(),
    )

    if self.chk_colorbar.isChecked():
        colorbar = self.plot_figure.colorbar(mesh, ax=self.plot_ax)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label("RCS (Linear)", color=self._current_plot_text())
        else:
            colorbar.set_label("RCS (dBsm)", color=self._current_plot_text())
        colorbar.ax.tick_params(colors=self._current_plot_text())
        for label in colorbar.ax.get_yticklabels():
            label.set_color(self._current_plot_text())

    # Update axis spinboxes to match the new view.
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(az_values[0]))
    self.spin_plot_xmax.setValue(float(az_values[-1]))
    self.spin_plot_ymin.setValue(float(range_axis[0]))
    self.spin_plot_ymax.setValue(float(range_axis[-1]))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    self._apply_plot_limits()

    note = ""
    if fr_nonuniformity >= 1e-3:
        note = f" — resampled frequency (Δ-spread {fr_nonuniformity*100:.1f}%)"
    self.status.showMessage(f"Az vs Down-Range updated{note}.")
