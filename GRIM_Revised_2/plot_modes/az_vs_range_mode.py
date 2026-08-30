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

from . import common
from .isar_mode import (
    _MAX_INTERP_COMPLEX_CELLS,
    _apply_resample_plan,
    _decimate_display_max,
    _length_unit,
    _uniform_resample_plan,
    _unit_to_hz_scale,
)


def _prepare_uniform_frequency_history(
    frequency_hz: np.ndarray,
    complex_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Gap-aware range-processing grid and per-row measurement weights.

    Interpolation is allowed only between nearby acquired frequencies. A large
    missing band and any interpolation touching an unknown complex sample stay
    zero-weighted rather than becoming apparently measured phase history.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    history = np.asarray(complex_history)
    if history.ndim != 2 or history.shape[1] != frequency_hz.size:
        raise ValueError(
            "azimuth/range phase history must have shape (azimuth, frequency)"
        )
    maximum_frequency_samples = max(
        frequency_hz.size,
        _MAX_INTERP_COMPLEX_CELLS // max(history.shape[0], 1),
    )
    plan = _uniform_resample_plan(
        frequency_hz,
        max_output_samples=maximum_frequency_samples,
    )
    finite = np.isfinite(history)
    clean = np.nan_to_num(
        history,
        copy=True,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.complex64, copy=False)
    uniform_history = _apply_resample_plan(
        frequency_hz, clean, axis=1, plan=plan
    )
    interpolated_validity = _apply_resample_plan(
        frequency_hz,
        finite.astype(np.float32),
        axis=1,
        plan=plan,
    )
    weights = (interpolated_validity >= 1.0 - 1.0e-6).astype(np.float32)
    uniform_history = np.asarray(uniform_history, dtype=np.complex64) * weights
    return (
        np.asarray(plan["target"], dtype=float),
        uniform_history,
        weights,
        dict(plan["info"]),
    )


def _range_display_values(dataset, magnitude: np.ndarray, *, linear: bool) -> np.ndarray:
    """Convert coherent range amplitude to generic image intensity.

    A frequency IFFT does not, by itself, preserve a calibrated physical RCS
    normalization. Keep the familiar amplitude-dB scaling without claiming
    the result is dBsm/dBke.
    """

    del dataset  # retained in the public helper signature for compatibility
    magnitude = np.asarray(magnitude, dtype=float)
    intensity = magnitude ** 2
    if linear:
        return intensity
    intensity = np.where(np.isfinite(intensity), intensity, np.nan)
    return 10.0 * np.log10(np.maximum(intensity, 1.0e-12))


def render(self) -> None:
    self.last_plot_mode = "az_vs_range"
    self._start_plot_render()
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets([("Dataset", self.active_dataset)])
    if reference is None:
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
        self.status.showMessage(
            f"{self._plot_axis_name(reference, 'azimuth')} samples must be strictly increasing."
        )
        return

    freq_values = self.active_dataset.frequencies[freq_indices].astype(float)
    freq_order = np.argsort(freq_values)
    sorted_freq_indices = [freq_indices[i] for i in freq_order]
    freq_values = freq_values[freq_order]
    if np.any(np.diff(freq_values) <= 0):
        self.status.showMessage("Frequency samples must be strictly increasing.")
        return

    rcs_slice = self.active_dataset.rcs_slice(
        np.ix_(sorted_az_indices, [elev_idx], sorted_freq_indices, [pol_idx])
    )[:, 0, :, 0]
    if not np.any(np.isfinite(rcs_slice)):
        self.status.showMessage(
            "No compatible phase-aware data for the selected azimuth/aspect, "
            "elevation/pitch, frequency, and polarization values."
        )
        return
    # IFFT along frequency requires a uniform grid. Preserve missing bands and
    # unknown phase as zero-weight observations; never bridge them as data.
    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
    freq_hz = freq_values * _unit_to_hz_scale(freq_unit)
    try:
        freq_hz_uniform, rcs_slice, sample_weights, frequency_sampling = (
            _prepare_uniform_frequency_history(freq_hz, rcs_slice)
        )
    except ValueError as exc:
        self.status.showMessage(f"Az vs Down-Range blocked: {exc}")
        return
    fr_nonuniformity = float(frequency_sampling["non_uniformity"])
    n_freq = freq_hz_uniform.size
    df = float(np.mean(np.diff(freq_hz_uniform)))

    # Window over freq (re-uses ISAR window selector).
    win_freq = self._isar_window(n_freq)
    rcs_windowed = rcs_slice * win_freq[None, :] * sample_weights

    # Range processing: IFFT and shift so range=0 sits at array center.
    range_image = np.fft.ifft(rcs_windowed, axis=1)
    range_image = np.fft.fftshift(range_image, axes=1)

    # Coherent-gain normalisation keeps a unit-amplitude point response near
    # 0 dB re 1. It is deliberately not labeled dBsm/dBke: the IFFT image is
    # a processing product, not an RCS sample on the source grid.
    coherent_gain = np.sum(
        sample_weights * win_freq[None, :], axis=1
    ) / float(n_freq)
    usable_rows = coherent_gain > 0.0
    if not np.any(usable_rows):
        self.status.showMessage(
            "No azimuth row has enough finite, supported phase history for range processing."
        )
        return
    range_image[usable_rows] /= coherent_gain[usable_rows, None]
    range_image[~usable_rows] = np.nan + 1j * np.nan

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
        peak = float(np.nanmax(magnitude))
        if peak > 0.0:
            magnitude = magnitude / peak

    display = _range_display_values(
        self.active_dataset,
        magnitude,
        linear=self._plot_scale_is_linear(),
    )
    max_side = min(common.MAX_IMAGE_SIDE, int(np.sqrt(common.MAX_IMAGE_CELLS)))
    display_for_plot = _decimate_display_max(display.T, max_side=max_side)
    if display_for_plot.shape != display.T.shape:
        self._note_plot_render(
            "Large range image was peak-preserving display-decimated for responsive "
            "interaction; narrow the selected axes for full display resolution."
        )

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
        display_for_plot,
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

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    self.plot_ax.set_ylabel(f"Down-Range ({unit_name})")
    elev_value = self.active_dataset.elevations[elev_idx]
    elev_name = self._plot_axis_name(reference, "elevation")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    pol_value = self.active_dataset.polarizations[pol_idx]
    self.plot_ax.set_title(
        f"{self._plot_axis_name(reference, 'azimuth')} vs Down-Range | "
        f"{elev_name} {elev_value:g} {elev_unit} | Pol {pol_value}",
        color=self._current_plot_text(),
    )

    if self.chk_colorbar.isChecked():
        colorbar = self.plot_figure.colorbar(mesh, ax=self.plot_ax)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label(
                "Range image intensity (linear, a.u.)",
                color=self._current_plot_text(),
            )
        else:
            colorbar.set_label(
                "Range image intensity (dB re 1 a.u.)",
                color=self._current_plot_text(),
            )
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
    gap_count = int(frequency_sampling.get("gap_count", 0))
    if gap_count:
        unsupported = 100.0 * float(
            frequency_sampling.get("unsupported_fraction", 0.0)
        )
        note += (
            f" — {gap_count} missing frequency band(s) kept zero-weighted "
            f"({unsupported:.1f}% unsupported grid)"
        )
    self._show_plot_status(f"Az vs Down-Range updated{note}.")
