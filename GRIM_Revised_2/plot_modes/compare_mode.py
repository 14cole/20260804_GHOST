from __future__ import annotations

import numpy as np

from . import common


_SWEEP_PRIORITY = ("azimuth", "elevation", "frequency")


def _determine_sweep_axis(azimuths, elevations, frequencies) -> str | None:
    selections = {
        "azimuth": azimuths,
        "elevation": elevations,
        "frequency": frequencies,
    }
    for axis in _SWEEP_PRIORITY:
        if len(selections[axis]) >= 2:
            return axis
    return None


def _collect_single_series(
    self,
    reference,
    dataset,
    name,
    sweep_axis,
    azimuths,
    elevations,
    frequencies,
    polarization,
):
    selections = {
        "azimuth": np.asarray(sorted(azimuths), dtype=float),
        "elevation": np.asarray(sorted(elevations), dtype=float),
        "frequency": np.asarray(sorted(frequencies), dtype=float),
    }
    indices = {
        axis: self._axis_selection_for_dataset(reference, dataset, axis, values)
        for axis, values in selections.items()
    }
    pol_indices = self._indices_for_values(dataset.polarizations, [polarization], tol=0.0)
    if any(value is None for value in indices.values()) or pol_indices is None:
        return None

    az_indices = indices["azimuth"]
    elev_indices = indices["elevation"]
    freq_indices = indices["frequency"]
    pol_idx = pol_indices[0]
    if sweep_axis == "azimuth":
        native_x = dataset.azimuths[az_indices]
        native_frequency = float(dataset.frequencies[freq_indices[0]])
        raw_selection = (az_indices, elev_indices[0], freq_indices[0], pol_idx)
        fixed = (
            f"Freq {float(self._plot_axis_values(reference, dataset, 'frequency', [native_frequency])[0]):g} "
            f"{self._plot_axis_unit(reference, 'frequency')}, "
            f"{self._plot_axis_name(reference, 'elevation')} "
            f"{float(self._plot_axis_values(reference, dataset, 'elevation', [dataset.elevations[elev_indices[0]]])[0]):g} "
            f"{self._plot_axis_unit(reference, 'elevation')}"
        )
        frequency_value = native_frequency
    elif sweep_axis == "elevation":
        native_x = dataset.elevations[elev_indices]
        native_frequency = float(dataset.frequencies[freq_indices[0]])
        raw_selection = (az_indices[0], elev_indices, freq_indices[0], pol_idx)
        fixed = (
            f"Freq {float(self._plot_axis_values(reference, dataset, 'frequency', [native_frequency])[0]):g} "
            f"{self._plot_axis_unit(reference, 'frequency')}, "
            f"{self._plot_axis_name(reference, 'azimuth')} "
            f"{float(self._plot_axis_values(reference, dataset, 'azimuth', [dataset.azimuths[az_indices[0]]])[0]):g} "
            f"{self._plot_axis_unit(reference, 'azimuth')}"
        )
        frequency_value = native_frequency
    else:
        native_x = dataset.frequencies[freq_indices]
        native_frequencies = np.asarray(native_x, dtype=float)
        raw_selection = (az_indices[0], elev_indices[0], freq_indices, pol_idx)
        fixed = (
            f"{self._plot_axis_name(reference, 'azimuth')} "
            f"{float(self._plot_axis_values(reference, dataset, 'azimuth', [dataset.azimuths[az_indices[0]]])[0]):g} "
            f"{self._plot_axis_unit(reference, 'azimuth')}, "
            f"{self._plot_axis_name(reference, 'elevation')} "
            f"{float(self._plot_axis_values(reference, dataset, 'elevation', [dataset.elevations[elev_indices[0]]])[0]):g} "
            f"{self._plot_axis_unit(reference, 'elevation')}"
        )
        frequency_value = native_frequencies

    if self._button_checked(self.btn_phase):
        raw = dataset.rcs_slice(raw_selection)
    else:
        raw = dataset.rcs_power[raw_selection]
    display = self._display_from_values(
        dataset, raw, frequency_value=frequency_value
    )
    x_values = self._plot_axis_values(reference, dataset, sweep_axis, native_x)
    label = f"{name} | Pol {dataset.polarizations[pol_idx]}, {fixed}"
    return np.asarray(x_values), np.asarray(display), label


def render(self) -> None:
    self.last_plot_mode = "compare"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if len(datasets) != 2:
        self.status.showMessage("Compare: select exactly 2 datasets.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return
    if self._button_checked(self.btn_phase):
        missing_metadata = common.missing_coherent_metadata(datasets)
        if missing_metadata:
            rendered = ", ".join(value.replace("_", " ") for value in missing_metadata)
            self._note_plot_render(
                "Phase residual assumes a common reference because these declarations "
                f"are missing: {rendered}."
            )

    azimuths = self._selected_values(self.list_az)
    elevations = self._selected_values(self.list_elev)
    frequencies = self._selected_values(self.list_freq)
    if not azimuths:
        self.status.showMessage("Compare: select one or more azimuths/aspects.")
        return
    if not elevations:
        self.status.showMessage("Compare: select one or more elevations/pitches.")
        return
    if not frequencies:
        self.status.showMessage("Compare: select one or more frequencies.")
        return
    polarization = self._single_selection_value(self.list_pol, "polarization")
    if polarization is None:
        return

    sweep_axis = _determine_sweep_axis(azimuths, elevations, frequencies)
    if sweep_axis is None:
        self.status.showMessage(
            "Compare: select 2+ azimuths/aspects, elevations/pitches, or frequencies "
            "for the sweep."
        )
        return
    selections = {
        "azimuth": azimuths,
        "elevation": elevations,
        "frequency": frequencies,
    }
    extra_sweeps = [
        axis for axis, values in selections.items()
        if axis != sweep_axis and len(values) != 1
    ]
    if extra_sweeps:
        fixed_names = ", ".join(self._plot_axis_name(reference, axis) for axis in extra_sweeps)
        self.status.showMessage(
            f"Compare requires exactly one series per dataset. Select exactly one "
            f"{fixed_names} value, or make it the only sweep axis."
        )
        return

    collected = []
    for name, dataset in datasets:
        series = _collect_single_series(
            self,
            reference,
            dataset,
            name,
            sweep_axis,
            azimuths,
            elevations,
            frequencies,
            polarization,
        )
        if series is None:
            self.status.showMessage(
                f"No compatible data in '{name}' for the selected comparison parameters."
            )
            return
        collected.append(series)

    (x_a, y_a, label_a), (x_b, y_b, label_b) = collected
    left_indices, right_indices = common.common_axis_indices(
        x_a,
        x_b,
        tolerance=common.axis_matching_tolerance(reference, sweep_axis),
    )
    if left_indices.size < 2:
        self.status.showMessage(
            f"No compatible data: datasets have fewer than 2 common "
            f"{self._plot_axis_name(reference, sweep_axis).lower()} samples."
        )
        return
    x_common = x_a[left_indices]
    left_values = y_a[left_indices]
    right_values = y_b[right_indices]
    finite_common = np.isfinite(left_values) & np.isfinite(right_values)
    if np.count_nonzero(finite_common) < 2:
        self.status.showMessage(
            "No compatible data: datasets have fewer than 2 common finite samples "
            "for the residual."
        )
        return

    top_ax, residual_ax = self._ensure_compare_axes()
    top_ax.clear()
    residual_ax.clear()
    self._style_axes(top_ax)
    self._style_axes(residual_ax)
    self._plot_bounded_line(
        top_ax, x_a, y_a, color="#4fc3f7", linewidth=1.5, label=label_a
    )
    self._plot_bounded_line(
        top_ax, x_b, y_b, color="#ff8a65", linewidth=1.5,
        linestyle="--", label=label_b,
    )
    top_ax.set_ylabel(self._display_axis_label(datasets))
    self._update_legend_visibility()

    residual = left_values - right_values
    phase_mode = self._button_checked(self.btn_phase)
    if phase_mode:
        residual = common.wrap_phase_degrees(residual)
    x_display, residual_display, decimated = common.decimate_line(x_common, residual)
    if decimated:
        self._note_plot_render(
            f"Residual over {common.MAX_LINE_POINTS:,} samples was display-decimated; "
            "statistics still use all common samples."
        )
    grid_color = self._current_plot_grid()
    residual_ax.axhline(0, color=grid_color, linewidth=0.8, linestyle="--")
    residual_ax.plot(
        x_display,
        residual_display,
        color="#a5d6a7",
        linewidth=1.2,
        label=f"{datasets[0][0]} - {datasets[1][0]}",
    )
    residual_ax.fill_between(
        x_display, residual_display, 0, alpha=0.15, color="#a5d6a7"
    )
    residual_unit = self._display_unit(datasets)
    residual_ax.set_ylabel(f"Residual ({residual_unit})", fontsize=8)
    residual_ax.set_xlabel(self._plot_axis_label(reference, sweep_axis))

    finite = np.isfinite(residual)
    if np.count_nonzero(finite) > 1:
        finite_residual = residual[finite]
        if phase_mode:
            mean_error = float(
                np.degrees(np.angle(np.mean(np.exp(1j * np.deg2rad(finite_residual)))))
            )
        else:
            mean_error = float(np.mean(finite_residual))
        rms_error = float(np.sqrt(np.mean(finite_residual ** 2)))
        max_error = float(np.max(np.abs(finite_residual)))
        correlation = ""
        both = finite & np.isfinite(left_values) & np.isfinite(right_values)
        if not phase_mode and np.count_nonzero(both) > 1:
            corr_value = float(np.corrcoef(left_values[both], right_values[both])[0, 1])
            if np.isfinite(corr_value):
                correlation = f"   Corr: {corr_value:.4f}"
        top_ax.set_title(
            f"Mean: {mean_error:+.2f} {residual_unit}   "
            f"RMS: {rms_error:.2f} {residual_unit}   "
            f"Max|err|: {max_error:.2f} {residual_unit}{correlation}",
            fontsize=8,
            color=self._current_plot_text(),
            pad=4,
        )

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(min(np.min(x_a), np.min(x_b))))
    self.spin_plot_xmax.setValue(float(max(np.max(x_a), np.max(x_b))))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    self._show_plot_status(f"Compare plot updated ({sweep_axis} sweep).")
