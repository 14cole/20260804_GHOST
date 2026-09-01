from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import common


_SWEEP_PRIORITY = ("azimuth", "elevation", "frequency")
_MAX_COMPARISON_SECTORS = 6
_MIN_SAMPLES_PER_SECTOR = 3


@dataclass(frozen=True)
class SectorAgreement:
    """Agreement over one contiguous portion of the selected RF sweep."""

    start: float
    stop: float
    sample_count: int
    score: float


@dataclass(frozen=True)
class RFAgreementStatistics:
    """RF agreement metrics computed on physical linear power.

    Pearson correlation only tests whether two centered traces have a linear
    relationship. It is therefore exactly one for traces with a serious gain
    or level error. The RF Agreement Index intentionally combines Lin's
    concordance (shape, scale, and location), log-domain concordance (nulls and
    sidelobes), normalized linear-power error, and local sector agreement. The
    global component is 45% linear CCC, 30% log CCC, and 25% normalized-error
    agreement. The final index is 75% global and 25% the lower quartile of
    contiguous sector scores.
    """

    score: float
    linear_ccc: float
    log_ccc: float
    linear_nrmse: float
    median_db_delta: float
    mae_db: float
    p95_db: float
    peak_offset: float
    sectors: tuple[SectorAgreement, ...]

    @property
    def worst_sector(self) -> SectorAgreement:
        return min(self.sectors, key=lambda sector: sector.score)


@dataclass(frozen=True)
class PhaseAgreementStatistics:
    """Circular phase agreement without a false discontinuity at +/-180 deg."""

    score: float
    mean_error: float
    circular_rms: float
    phasor_agreement: float
    residual_coherence: float
    sectors: tuple[SectorAgreement, ...]

    @property
    def worst_sector(self) -> SectorAgreement:
        return min(self.sectors, key=lambda sector: sector.score)


def _lin_concordance(left, right) -> float:
    """Return Lin's concordance correlation coefficient.

    Unlike Pearson correlation, CCC penalizes both gain and offset errors. A
    pair of equal constant traces is defined as perfect concordance; unequal
    constant traces have zero concordance.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mean_left = float(np.mean(left))
    mean_right = float(np.mean(right))
    centered_left = left - mean_left
    centered_right = right - mean_right
    var_left = float(np.mean(centered_left ** 2))
    var_right = float(np.mean(centered_right ** 2))
    denominator = var_left + var_right + (mean_left - mean_right) ** 2
    scale = max(float(np.mean(left ** 2) + np.mean(right ** 2)), 1.0)
    if denominator <= np.finfo(float).eps * scale:
        return 1.0 if np.array_equal(left, right) else 0.0
    covariance = float(np.mean(centered_left * centered_right))
    return float(np.clip(2.0 * covariance / denominator, -1.0, 1.0))


def _linear_nrmse(left, right) -> float:
    """RMS power error normalized by the pooled RMS power."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    numerator = float(np.mean((left - right) ** 2))
    denominator = float(np.mean((left ** 2 + right ** 2) / 2.0))
    if denominator <= np.finfo(float).tiny:
        return 0.0 if numerator <= np.finfo(float).tiny else float("inf")
    return float(np.sqrt(numerator / denominator))


def _power_db_pair(left, right) -> tuple[np.ndarray, np.ndarray]:
    """Return finite dB traces with a common, data-relative numerical floor."""

    left = np.maximum(np.asarray(left, dtype=float), 0.0)
    right = np.maximum(np.asarray(right, dtype=float), 0.0)
    peak = max(float(np.max(left)), float(np.max(right)), np.finfo(float).tiny)
    floor = peak * 1.0e-12  # -120 dB relative: numerical protection, not censoring.
    return 10.0 * np.log10(np.maximum(left, floor)), 10.0 * np.log10(
        np.maximum(right, floor)
    )


def _sector_slices(sample_count: int) -> tuple[slice, ...]:
    """Partition a sweep into contiguous, sufficiently populated sectors."""

    count = min(
        _MAX_COMPARISON_SECTORS,
        max(1, sample_count // _MIN_SAMPLES_PER_SECTOR),
    )
    return tuple(
        slice(int(bounds[0]), int(bounds[-1]) + 1)
        for bounds in np.array_split(np.arange(sample_count), count)
        if bounds.size
    )


def _magnitude_component_score(left, right) -> tuple[float, float, float, float]:
    linear_ccc = _lin_concordance(left, right)
    left_db, right_db = _power_db_pair(left, right)
    log_ccc = _lin_concordance(left_db, right_db)
    nrmse = _linear_nrmse(left, right)
    error_agreement = 0.0 if not np.isfinite(nrmse) else 1.0 - min(
        nrmse / np.sqrt(2.0), 1.0
    )
    # Negative concordance describes an inverted pattern, not RF agreement.
    score = 100.0 * (
        0.45 * max(linear_ccc, 0.0)
        + 0.30 * max(log_ccc, 0.0)
        + 0.25 * error_agreement
    )
    return float(score), linear_ccc, log_ccc, nrmse


def rf_agreement_statistics(x, left_power, right_power) -> RFAgreementStatistics:
    """Compute an RF-aware agreement summary for one selected sweep.

    Inputs must already be one-to-one coordinate matched. Statistics are
    always evaluated on physical linear power, independent of the plot scale.
    """

    x = np.asarray(x, dtype=float)
    left = np.asarray(left_power, dtype=float)
    right = np.asarray(right_power, dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(left)
        & np.isfinite(right)
        & (left >= 0.0)
        & (right >= 0.0)
    )
    x = x[valid]
    left = left[valid]
    right = right[valid]
    if x.size < 2:
        raise ValueError("RF agreement requires at least two finite matched samples")

    component_score, linear_ccc, log_ccc, nrmse = _magnitude_component_score(
        left, right
    )
    left_db, right_db = _power_db_pair(left, right)
    db_delta = left_db - right_db

    sectors = []
    for sector_slice in _sector_slices(x.size):
        sector_score, _linear, _log, _nrmse = _magnitude_component_score(
            left[sector_slice], right[sector_slice]
        )
        sectors.append(
            SectorAgreement(
                start=float(x[sector_slice.start]),
                stop=float(x[sector_slice.stop - 1]),
                sample_count=sector_slice.stop - sector_slice.start,
                score=sector_score,
            )
        )
    # Local agreement receives enough weight to expose a bad angular sector or
    # frequency band that a globally peak-dominated statistic can conceal.
    lower_quartile_sector = float(
        np.percentile([sector.score for sector in sectors], 25.0)
    )
    score = 0.75 * component_score + 0.25 * lower_quartile_sector
    peak_offset = float(x[int(np.argmax(left))] - x[int(np.argmax(right))])
    return RFAgreementStatistics(
        score=float(np.clip(score, 0.0, 100.0)),
        linear_ccc=linear_ccc,
        log_ccc=log_ccc,
        linear_nrmse=nrmse,
        median_db_delta=float(np.median(db_delta)),
        mae_db=float(np.mean(np.abs(db_delta))),
        p95_db=float(np.percentile(np.abs(db_delta), 95.0)),
        peak_offset=peak_offset,
        sectors=tuple(sectors),
    )


def phase_agreement_statistics(
    x, left_degrees, right_degrees
) -> PhaseAgreementStatistics:
    """Compute phase agreement using circular residuals and phasors."""

    x = np.asarray(x, dtype=float)
    left = np.asarray(left_degrees, dtype=float)
    right = np.asarray(right_degrees, dtype=float)
    valid = np.isfinite(x) & np.isfinite(left) & np.isfinite(right)
    x = x[valid]
    residual = common.wrap_phase_degrees(left[valid] - right[valid])
    if x.size < 2:
        raise ValueError("phase agreement requires at least two finite matched samples")
    radians = np.deg2rad(residual)
    mean_phasor = np.mean(np.exp(1j * radians))
    phasor_agreement = float(np.mean(np.cos(radians)))
    sectors = []
    for sector_slice in _sector_slices(x.size):
        local = radians[sector_slice]
        local_score = 50.0 * (1.0 + float(np.mean(np.cos(local))))
        sectors.append(
            SectorAgreement(
                start=float(x[sector_slice.start]),
                stop=float(x[sector_slice.stop - 1]),
                sample_count=sector_slice.stop - sector_slice.start,
                score=float(np.clip(local_score, 0.0, 100.0)),
            )
        )
    lower_quartile_sector = float(
        np.percentile([sector.score for sector in sectors], 25.0)
    )
    global_score = 50.0 * (1.0 + phasor_agreement)
    return PhaseAgreementStatistics(
        score=float(
            np.clip(
                0.75 * global_score + 0.25 * lower_quartile_sector,
                0.0,
                100.0,
            )
        ),
        mean_error=float(np.degrees(np.angle(mean_phasor))),
        circular_rms=float(np.sqrt(np.mean(residual ** 2))),
        phasor_agreement=phasor_agreement,
        residual_coherence=float(np.abs(mean_phasor)),
        sectors=tuple(sectors),
    )


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


def _comparison_azimuth_sector(self, reference, selected_azimuths):
    """Initialize/read the interactive statistics-sector controls.

    Values are expressed in the active reference dataset's displayed angular
    unit. A changed parameter-list selection resets the bounds to that
    selection's actual minimum and maximum and resets Show all azimuths off.
    """

    selected = np.asarray(selected_azimuths, dtype=float)
    selected_display = self._plot_axis_values(
        reference, reference, "azimuth", selected
    )
    full_display = self._plot_axis_values(
        reference, reference, "azimuth", reference.azimuths
    )
    selected_min = float(np.min(selected_display))
    selected_max = float(np.max(selected_display))
    controls = getattr(self, "compare_sector_bar", None)
    minimum_control = getattr(self, "spin_compare_az_min", None)
    maximum_control = getattr(self, "spin_compare_az_max", None)
    show_all_control = getattr(
        self, "chk_compare_show_all_azimuths", None
    )
    if (
        controls is None
        or minimum_control is None
        or maximum_control is None
        or show_all_control is None
    ):
        return selected_min, selected_max, False

    controls.setVisible(True)
    unit = self._plot_axis_unit(reference, "azimuth")
    signature = (
        id(controls),
        unit,
        tuple(float(value) for value in selected_display),
    )
    if getattr(self, "_compare_sector_selection_signature", None) != signature:
        full_min = float(np.min(full_display))
        full_max = float(np.max(full_display))
        for control, value in (
            (minimum_control, selected_min),
            (maximum_control, selected_max),
        ):
            control.blockSignals(True)
            control.setRange(full_min, full_max)
            control.setSuffix(f" {unit}")
            control.setValue(value)
            control.blockSignals(False)
        show_all_control.blockSignals(True)
        show_all_control.setChecked(False)
        show_all_control.blockSignals(False)
        self._compare_sector_selection_signature = signature

    return (
        float(minimum_control.value()),
        float(maximum_control.value()),
        bool(show_all_control.isChecked()),
    )


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

    phase_mode = self._button_checked(self.btn_phase)
    if phase_mode:
        raw = dataset.rcs_slice(raw_selection)
        analysis = None
    else:
        raw = dataset.rcs_power[raw_selection]
        analysis = np.asarray(raw, dtype=float)
    display = self._display_from_values(
        dataset, raw, frequency_value=frequency_value
    )
    x_values = self._plot_axis_values(reference, dataset, sweep_axis, native_x)
    label = f"{name} | Pol {dataset.polarizations[pol_idx]}, {fixed}"
    return np.asarray(x_values), np.asarray(display), analysis, label


def render(self) -> None:
    self.last_plot_mode = "compare"
    self._start_plot_render()
    controls = getattr(self, "compare_sector_bar", None)
    if controls is not None:
        controls.setVisible(False)
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

    statistics_bounds = None
    show_all_azimuths = False
    highlight_statistics_range = False
    if sweep_axis == "azimuth":
        sector_min, sector_max, show_all_azimuths = _comparison_azimuth_sector(
            self, reference, azimuths
        )
        if sector_min > sector_max:
            self.status.showMessage(
                "Compare: Min azimuth must be less than or equal to Max azimuth."
            )
            return
        reference_azimuths = np.asarray(reference.azimuths, dtype=float)
        reference_display = self._plot_axis_values(
            reference, reference, "azimuth", reference_azimuths
        )
        tolerance = common.axis_matching_tolerance(reference, "azimuth")
        common_reference_mask = np.ones(reference_azimuths.shape, dtype=bool)
        for _name, dataset in datasets:
            dataset_display = self._plot_axis_values(
                reference, dataset, "azimuth", dataset.azimuths
            )
            reference_indices, _dataset_indices = common.common_axis_indices(
                reference_display,
                dataset_display,
                tolerance=tolerance,
            )
            dataset_common_mask = np.zeros(reference_azimuths.shape, dtype=bool)
            dataset_common_mask[reference_indices] = True
            common_reference_mask &= dataset_common_mask
        sector_mask = (
            (reference_display >= sector_min - tolerance)
            & (reference_display <= sector_max + tolerance)
            & common_reference_mask
        )
        if np.count_nonzero(sector_mask) < 2:
            self.status.showMessage(
                "Compare: the Min/Max azimuth sector contains fewer than 2 "
                "reference samples."
            )
            return
        common_display = reference_display[common_reference_mask]
        common_min = float(np.min(common_display))
        common_max = float(np.max(common_display))
        highlight_statistics_range = not (
            sector_min <= common_min + tolerance
            and sector_max >= common_max - tolerance
        )
        azimuths = (
            reference_azimuths[common_reference_mask]
            if show_all_azimuths
            else reference_azimuths[sector_mask]
        )
        statistics_bounds = (sector_min, sector_max)
    else:
        controls = getattr(self, "compare_sector_bar", None)
        if controls is not None:
            controls.setVisible(False)

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

    (x_a, y_a, analysis_a, label_a), (x_b, y_b, analysis_b, label_b) = collected
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
    if not self._button_checked(self.btn_phase):
        finite_common &= (
            np.isfinite(analysis_a[left_indices])
            & np.isfinite(analysis_b[right_indices])
            & (analysis_a[left_indices] >= 0.0)
            & (analysis_b[right_indices] >= 0.0)
        )
    statistics_mask = np.ones(x_common.shape, dtype=bool)
    if statistics_bounds is not None:
        statistics_tolerance = common.axis_matching_tolerance(
            reference, "azimuth"
        )
        statistics_mask = (
            (x_common >= statistics_bounds[0] - statistics_tolerance)
            & (x_common <= statistics_bounds[1] + statistics_tolerance)
        )
    finite_statistics = finite_common & statistics_mask
    if np.count_nonzero(finite_statistics) < 2:
        self.status.showMessage(
            "No compatible data: datasets have fewer than 2 common finite samples "
            "inside the statistics sector."
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
    residual_ax.set_ylabel(f"Difference ({residual_unit})", fontsize=8)
    residual_ax.set_xlabel(self._plot_axis_label(reference, sweep_axis))

    finite = finite_statistics & np.isfinite(residual)
    if np.count_nonzero(finite) > 1:
        sector_title = ""
        if statistics_bounds is not None:
            sector_title = (
                f"Statistics range: {statistics_bounds[0]:g} to "
                f"{statistics_bounds[1]:g} "
                f"{self._plot_axis_unit(reference, 'azimuth')}"
            )
        if phase_mode:
            statistics = phase_agreement_statistics(
                x_common[finite], left_values[finite], right_values[finite]
            )
            phase_alignment = 50.0 * (1.0 + statistics.phasor_agreement)
            top_ax.set_title(
                f"Overall phase match: {statistics.score:.1f}/100   "
                f"Average phase alignment: {phase_alignment:.1f}%   "
                f"Difference consistency: "
                f"{100.0 * statistics.residual_coherence:.1f}%\n"
                f"Average phase difference (first minus second): "
                f"{statistics.mean_error:+.2f} deg   "
                f"Typical phase error: {statistics.circular_rms:.2f} deg\n"
                + sector_title,
                fontsize=8,
                color=self._current_plot_text(),
                pad=4,
            )
        else:
            statistics = rf_agreement_statistics(
                x_common[finite],
                analysis_a[left_indices][finite],
                analysis_b[right_indices][finite],
            )
            axis_unit = self._plot_axis_unit(reference, sweep_axis)
            strong_return_agreement = 100.0 * float(
                np.clip(statistics.linear_ccc, 0.0, 1.0)
            )
            full_pattern_agreement = 100.0 * float(
                np.clip(statistics.log_ccc, 0.0, 1.0)
            )
            top_ax.set_title(
                f"Overall match: {statistics.score:.1f}/100   "
                f"Strong-return agreement: {strong_return_agreement:.1f}%   "
                f"Full-pattern agreement: {full_pattern_agreement:.1f}%\n"
                f"Power error: {100.0 * statistics.linear_nrmse:.1f}%   "
                f"Typical level difference (first minus second): "
                f"{statistics.median_db_delta:+.2f} dB   "
                f"Average level error: {statistics.mae_db:.2f} dB   "
                f"95% of points within: {statistics.p95_db:.2f} dB\n"
                f"Peak shift: {statistics.peak_offset:+g} {axis_unit}"
                + (f"   {sector_title}" if sector_title else ""),
                fontsize=8,
                color=self._current_plot_text(),
                pad=4,
            )
        if statistics_bounds is not None and highlight_statistics_range:
            residual_ax.axvspan(
                statistics_bounds[0],
                statistics_bounds[1],
                color="#ffb74d",
                alpha=0.12,
                linewidth=0,
                label="Statistics range",
            )

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(min(np.min(x_a), np.min(x_b))))
    self.spin_plot_xmax.setValue(float(max(np.max(x_a), np.max(x_b))))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    if statistics_bounds is not None:
        display_note = "all azimuths shown" if show_all_azimuths else "sector shown"
        self._show_plot_status(
            f"Compare plot updated (azimuth statistics "
            f"{statistics_bounds[0]:g} to {statistics_bounds[1]:g}; "
            f"{display_note})."
        )
    else:
        self._show_plot_status(f"Compare plot updated ({sweep_axis} sweep).")
