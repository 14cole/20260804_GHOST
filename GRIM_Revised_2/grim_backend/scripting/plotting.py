"""Render dataset selections to Matplotlib figures."""
from __future__ import annotations

from typing import Iterable, Sequence
import math
import warnings

import numpy as np

from grim_backend.datasets.grid import RcsGrid


def _indices(axis, selected, *, text=False, tolerance=1.0e-6) -> list[int]:
    values = np.asarray(axis)
    result: list[int] = []
    for requested in selected:
        if text:
            matches = np.flatnonzero(values.astype(str) == str(requested))
        else:
            matches = np.flatnonzero(
                np.isclose(values.astype(float), float(requested), atol=tolerance, rtol=0.0)
            )
        if matches.size != 1:
            raise ValueError(f"Selected value {requested!r} is missing or ambiguous")
        result.append(int(matches[0]))
    return result


def _display_values(dataset: RcsGrid, values, *, frequency, phase: bool, scale: str):
    if phase:
        phase_degrees = np.rad2deg(np.angle(values))
        finite = np.isfinite(phase_degrees)
        display = np.full(np.asarray(phase_degrees).shape, np.nan, dtype=float)
        phase_wrap = str(
            (dataset.units or {}).get("phase_wrap", "-180_180")
        ).strip()
        if phase_wrap == "0_360":
            display[finite] = np.mod(phase_degrees[finite], 360.0)
        else:
            display[finite] = (
                np.mod(phase_degrees[finite] + 180.0, 360.0) - 180.0
            )
        return display
    if str(scale).lower() == "linear":
        return dataset.rcs_to_linear(values)
    return dataset.rcs_to_display_db(values, frequency_value=frequency)


def _plot_selection_indices(
    reference: RcsGrid,
    dataset: RcsGrid,
    azimuths,
    elevations,
    frequencies,
    polarization,
):
    """Return converted requested indices, or ``None`` like the GUI policy."""

    from plot_modes import common as plot_common

    try:
        native_azimuths, azimuth_tolerance = plot_common.selection_for_dataset(
            reference, dataset, "azimuth", azimuths
        )
        native_elevations, elevation_tolerance = plot_common.selection_for_dataset(
            reference, dataset, "elevation", elevations
        )
        native_frequencies, frequency_tolerance = plot_common.selection_for_dataset(
            reference, dataset, "frequency", frequencies
        )
        return (
            _indices(
                dataset.azimuths,
                native_azimuths,
                tolerance=azimuth_tolerance,
            ),
            _indices(
                dataset.elevations,
                native_elevations,
                tolerance=elevation_tolerance,
            ),
            _indices(
                dataset.frequencies,
                native_frequencies,
                tolerance=frequency_tolerance,
            ),
            _indices(dataset.polarizations, [polarization], text=True)[0],
        )
    except (TypeError, ValueError):
        return None


def _plot_response_label(
    reference: RcsGrid,
    *,
    phase: bool,
    scale: str,
    p50: bool = False,
) -> str:
    """Match the GUI's response-quantity labels in generated plots."""

    if phase:
        return "Phase P50 (deg)" if p50 else "Phase (deg)"
    quantity_name, linear_unit = {
        "sigma_3d": ("RCS", "m²"),
        "sigma_2d": ("Scattering Width", "m"),
        "power_ratio": ("Power Ratio", "dimensionless"),
        "ratio": ("Power Ratio", "dimensionless"),
    }.get(str(reference.linear_quantity()).strip().lower(), ("Value", "linear"))
    suffix = " P50" if p50 else ""
    if str(scale).strip().lower() == "linear":
        return f"{quantity_name}{suffix} ({linear_unit})"
    return f"{quantity_name}{suffix} ({reference.default_log_unit()})"


def plot_datasets(
    datasets: Sequence[tuple[str, RcsGrid]],
    *,
    mode: str,
    azimuths: Iterable[float],
    elevations: Iterable[float],
    frequencies: Iterable[float],
    polarization: str,
    phase: bool = False,
    scale: str = "dbsm",
    colormap: str = "viridis",
    show_grid: bool = True,
    show_legend: bool = True,
    polar_zero: str = "N",
    reference_index: int = 0,
    waterfall_style: str = "Surface",
    isar_options: dict[str, object] | None = None,
    show_colorbar: bool = True,
    shared_colorbar: bool = True,
    square_aspect: bool = False,
    color_limits: tuple[float, float] | None = None,
    color_step: float = 0.0,
):
    """Create a Matplotlib Figure from resolved physical selector values.

    The implementation uses :class:`~matplotlib.backends.backend_agg.FigureCanvasAgg`
    directly and never imports pyplot or Qt.  Returned figures support ordinary
    ``figure.savefig(...)`` calls in generated scripts.
    """

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    selected = [(str(name), grid) for name, grid in datasets]
    if not selected:
        raise ValueError("plot_datasets needs at least one dataset")
    mode_key = str(mode).strip().lower()
    supported_modes = {
        "azimuth_rect",
        "azimuth_polar",
        "frequency",
        "elevation_sweep",
        "isar_image",
    }
    if mode_key not in supported_modes:
        raise ValueError(
            f"Unsupported headless plot mode {mode!r}; supported modes are "
            "azimuth_rect, azimuth_polar, frequency, elevation_sweep, and "
            "isar_image"
        )
    if mode_key == "isar_image" and len(selected) != 1:
        raise ValueError("ISAR headless plotting requires exactly one dataset")
    try:
        reference_position = int(reference_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_index must identify one selected dataset") from exc
    if reference_position < 0 or reference_position >= len(selected):
        raise ValueError("reference_index must identify one selected dataset")
    reference = selected[reference_position][1]

    from plot_modes import common as plot_common

    plot_common.validate_plot_datasets(
        selected,
        phase=bool(phase),
        linear=str(scale).strip().lower() == "linear",
    )
    az_values = [float(value) for value in azimuths]
    el_values = [float(value) for value in elevations]
    freq_values = [float(value) for value in frequencies]
    if not az_values or not el_values or not freq_values:
        raise ValueError("azimuths, elevations, and frequencies cannot be empty")
    compatible_count = 0
    style_axes = None

    figure = Figure(figsize=(10.0, 6.0), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    if mode_key == "azimuth_polar":
        axes = figure.add_subplot(111, projection="polar")
        axes.set_theta_zero_location(str(polar_zero))
        axes.set_theta_direction(-1)
    elif mode_key == "compare":
        axes, residual_axes = figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    elif mode_key == "waterfall":
        axes = figure.add_subplot(111, projection="3d")
    else:
        axes = figure.add_subplot(111)

    if mode_key in {"azimuth_rect", "azimuth_polar"}:
        for name, dataset in selected:
            indices = _plot_selection_indices(
                reference,
                dataset,
                az_values,
                el_values,
                freq_values,
                polarization,
            )
            if indices is None:
                continue
            az_idx, el_idx, fr_idx, pol_idx = indices
            compatible_count += 1
            native_x = np.asarray(dataset.azimuths)[az_idx]
            x = plot_common.values_for_display(
                reference, dataset, "azimuth", native_x
            )
            order = np.argsort(x)
            x = np.asarray(x)[order]
            if mode_key == "azimuth_polar":
                x = plot_common.convert_axis_values(
                    x,
                    "azimuth",
                    plot_common.axis_unit(reference, "azimuth"),
                    "rad",
                )
            for fi in fr_idx:
                for ei in el_idx:
                    selection = np.ix_(az_idx, [ei], [fi], [pol_idx])
                    raw = (
                        dataset.rcs_slice(selection)[:, 0, 0, 0]
                        if phase
                        else dataset.rcs_power[np.ix_(az_idx, [ei], [fi], [pol_idx])][:, 0, 0, 0]
                    )
                    y = _display_values(
                        dataset,
                        raw,
                        frequency=float(dataset.frequencies[fi]),
                        phase=phase,
                        scale=scale,
                    )[order]
                    axes.plot(
                        x,
                        y,
                        label=(
                            f"{name} | {polarization}, "
                            f"{float(plot_common.values_for_display(reference, dataset, 'frequency', [dataset.frequencies[fi]])[0]):g} "
                            f"{plot_common.axis_unit(reference, 'frequency')}, "
                            f"{plot_common.angular_axis_name(reference, 'elevation')} "
                            f"{float(plot_common.values_for_display(reference, dataset, 'elevation', [dataset.elevations[ei]])[0]):g} "
                            f"{plot_common.axis_unit(reference, 'elevation')}"
                        ),
                    )
        axes.set_xlabel(plot_common.axis_label(reference, "azimuth"))
        axes.set_ylabel(
            _plot_response_label(reference, phase=phase, scale=scale)
        )

    elif mode_key == "frequency":
        for name, dataset in selected:
            indices = _plot_selection_indices(
                reference,
                dataset,
                az_values,
                el_values,
                freq_values,
                polarization,
            )
            if indices is None:
                continue
            az_idx, el_idx, fr_idx, pol_idx = indices
            compatible_count += 1
            native_x = np.asarray(dataset.frequencies)[fr_idx]
            x = plot_common.values_for_display(
                reference, dataset, "frequency", native_x
            )
            order = np.argsort(x)
            for ei in el_idx:
                if phase:
                    block = dataset.rcs_slice(
                        np.ix_(az_idx, [ei], fr_idx, [pol_idx])
                    )[:, 0, :, 0]
                    phase_degrees = np.rad2deg(np.angle(block))
                    phase_degrees = np.where(np.isfinite(block), phase_degrees, np.nan)
                    y = plot_common.circular_median_degrees(
                        phase_degrees, axis=0
                    )
                else:
                    block = dataset.rcs_power[np.ix_(az_idx, [ei], fr_idx, [pol_idx])][:, 0, :, 0]
                    power = np.nanmedian(block, axis=0)
                    y = _display_values(
                        dataset,
                        power,
                        frequency=native_x,
                        phase=False,
                        scale=scale,
                    )
                elevation = float(
                    plot_common.values_for_display(
                        reference,
                        dataset,
                        "elevation",
                        [dataset.elevations[ei]],
                    )[0]
                )
                axes.plot(
                    np.asarray(x)[order],
                    np.asarray(y)[order],
                    label=(
                        f"{name} | "
                        f"{plot_common.angular_axis_name(reference, 'elevation')} "
                        f"{elevation:g} {plot_common.axis_unit(reference, 'elevation')}"
                    ),
                )
        axes.set_xlabel(plot_common.axis_label(reference, "frequency"))
        axes.set_ylabel(
            _plot_response_label(reference, phase=phase, scale=scale, p50=True)
        )

    elif mode_key == "elevation_sweep":
        for name, dataset in selected:
            indices = _plot_selection_indices(
                reference,
                dataset,
                az_values,
                el_values,
                freq_values,
                polarization,
            )
            if indices is None:
                continue
            az_idx, el_idx, fr_idx, pol_idx = indices
            compatible_count += 1
            native_x = np.asarray(dataset.elevations)[el_idx]
            x = plot_common.values_for_display(
                reference, dataset, "elevation", native_x
            )
            order = np.argsort(x)
            for fi in fr_idx:
                if phase:
                    block = dataset.rcs_slice(
                        np.ix_(az_idx, el_idx, [fi], [pol_idx])
                    )[:, :, 0, 0]
                    phase_degrees = np.rad2deg(np.angle(block))
                    phase_degrees = np.where(np.isfinite(block), phase_degrees, np.nan)
                    y = (
                        plot_common.circular_median_degrees(
                            phase_degrees, axis=0
                        )
                        if len(az_idx) > 1
                        else phase_degrees[0]
                    )
                else:
                    block = dataset.rcs_power[np.ix_(az_idx, el_idx, [fi], [pol_idx])][:, :, 0, 0]
                    power = np.nanmedian(block, axis=0) if len(az_idx) > 1 else block[0]
                    y = _display_values(dataset, power, frequency=dataset.frequencies[fi], phase=False, scale=scale)
                frequency = float(
                    plot_common.values_for_display(
                        reference,
                        dataset,
                        "frequency",
                        [dataset.frequencies[fi]],
                    )[0]
                )
                axes.plot(
                    np.asarray(x)[order],
                    np.asarray(y)[order],
                    label=(
                        f"{name} | {frequency:g} "
                        f"{plot_common.axis_unit(reference, 'frequency')}"
                    ),
                )
        axes.set_xlabel(plot_common.axis_label(reference, "elevation"))
        axes.set_ylabel(
            _plot_response_label(
                reference,
                phase=phase,
                scale=scale,
                p50=len(az_values) > 1,
            )
        )

    elif mode_key == "compare":
        if len(selected) != 2:
            raise ValueError("compare requires exactly two datasets")
        varying = "azimuth" if len(az_values) > 1 else "elevation" if len(el_values) > 1 else "frequency"
        series_values = []
        for name, dataset in selected:
            az_idx = _indices(dataset.azimuths, az_values)
            el_idx = _indices(dataset.elevations, el_values)
            fr_idx = _indices(dataset.frequencies, freq_values)
            pol_idx = _indices(dataset.polarizations, [polarization], text=True)[0]
            if varying == "azimuth":
                x = np.asarray(dataset.azimuths)[az_idx]
                selection = np.ix_(az_idx, [el_idx[0]], [fr_idx[0]], [pol_idx])
                raw = (
                    dataset.rcs_slice(selection)
                    if phase
                    else dataset.rcs_power[selection]
                )[:, 0, 0, 0]
                frequency = dataset.frequencies[fr_idx[0]]
            elif varying == "elevation":
                x = np.asarray(dataset.elevations)[el_idx]
                selection = np.ix_([az_idx[0]], el_idx, [fr_idx[0]], [pol_idx])
                raw = (
                    dataset.rcs_slice(selection)
                    if phase
                    else dataset.rcs_power[selection]
                )[0, :, 0, 0]
                frequency = dataset.frequencies[fr_idx[0]]
            else:
                x = np.asarray(dataset.frequencies)[fr_idx]
                selection = np.ix_([az_idx[0]], [el_idx[0]], fr_idx, [pol_idx])
                raw = (
                    dataset.rcs_slice(selection)
                    if phase
                    else dataset.rcs_power[selection]
                )[0, 0, :, 0]
                frequency = x
            y = _display_values(dataset, raw, frequency=frequency, phase=phase, scale=scale)
            order = np.argsort(x)
            x, y = np.asarray(x)[order], np.asarray(y)[order]
            axes.plot(x, y, label=name)
            series_values.append((x, y))
        common, left_idx, right_idx = np.intersect1d(
            np.round(series_values[0][0], 8),
            np.round(series_values[1][0], 8),
            return_indices=True,
        )
        if common.size:
            residual_axes.plot(common, series_values[0][1][left_idx] - series_values[1][1][right_idx])
        residual_axes.axhline(0.0, color="gray", linestyle="--")
        residual_axes.set_xlabel(varying.title())
        residual_axes.set_ylabel("Residual")
        axes.set_ylabel("Phase (deg)" if phase else ("RCS (Linear)" if scale == "linear" else "RCS (dB)"))

    elif mode_key == "waterfall":
        for name, dataset in selected:
            az_idx = _indices(dataset.azimuths, az_values)
            el_idx = _indices(dataset.elevations, el_values)
            fr_idx = _indices(dataset.frequencies, freq_values)
            pol_idx = _indices(dataset.polarizations, [polarization], text=True)[0]
            az_axis = np.asarray(dataset.azimuths)[az_idx]
            fr_axis = np.asarray(dataset.frequencies)[fr_idx]
            x_mesh, y_mesh = np.meshgrid(az_axis, fr_axis, indexing="ij")
            for ei in el_idx:
                selection = np.ix_(az_idx, [ei], fr_idx, [pol_idx])
                raw = (
                    dataset.rcs_slice(selection)
                    if phase
                    else dataset.rcs_power[selection]
                )[:, 0, :, 0]
                z = _display_values(dataset, raw, frequency=fr_axis[None, :], phase=phase, scale=scale)
                if str(waterfall_style).lower().startswith("wire"):
                    axes.plot_wireframe(x_mesh, y_mesh, z, label=name)
                else:
                    axes.plot_surface(x_mesh, y_mesh, z, cmap=colormap, alpha=0.85)
        axes.set_xlabel("Azimuth (deg)")
        axes.set_ylabel(f"Frequency ({selected[0][1].units.get('frequency', 'GHz')})")
        axes.set_zlabel("Phase (deg)" if phase else ("RCS (Linear)" if scale == "linear" else "RCS (dB)"))

    elif mode_key == "az_vs_range":
        dataset = selected[0][1]
        name = selected[0][0]
        az_idx = _indices(dataset.azimuths, az_values)
        fr_idx = _indices(dataset.frequencies, freq_values)
        el_idx = _indices(dataset.elevations, [el_values[0]])[0]
        pol_idx = _indices(dataset.polarizations, [polarization], text=True)[0]
        freq = np.asarray(dataset._frequency_value_to_hz(dataset.frequencies[fr_idx]), dtype=float)
        order_f = np.argsort(freq)
        freq = freq[order_f]
        complex_data = dataset.rcs_slice(
            np.ix_(az_idx, [el_idx], fr_idx, [pol_idx])
        )[:, 0, :, 0][:, order_f]
        window = np.hanning(freq.size)
        image = np.fft.fftshift(np.fft.ifft(complex_data * window[None, :], axis=1), axes=1)
        distance = np.fft.fftshift(np.fft.fftfreq(freq.size, d=float(np.mean(np.diff(freq))))) * 299_792_458.0 / 2.0
        magnitude = np.abs(image)
        display = magnitude if scale == "linear" else 20.0 * np.log10(np.maximum(magnitude, 1.0e-15))
        axes.pcolormesh(np.asarray(dataset.azimuths)[az_idx], distance, display.T, shading="auto", cmap=colormap)
        axes.set_title(name)
        axes.set_xlabel("Azimuth (deg)")
        axes.set_ylabel("Down-Range (m)")

    elif mode_key == "isar_image":
        from plot_modes.isar_mode import _length_unit, form_isar

        dataset = selected[0][1]
        options = dict(isar_options or {})
        options.setdefault("length_unit", "in")


        options["decimate_display"] = True
        options["retain_complex"] = False
        az_idx = _indices(dataset.azimuths, az_values)
        fr_idx = _indices(dataset.frequencies, freq_values)
        el_idx = _indices(dataset.elevations, [el_values[0]])[0]
        pol_idx = _indices(dataset.polarizations, [polarization], text=True)[0]
        bands, _elapsed = form_isar(
            dataset,
            azimuth_indices=az_idx,
            frequency_indices=fr_idx,
            elevation_index=el_idx,
            polarization_index=pol_idx,
            **options,
        )
        compatible_count = 1
        figure.clear()
        axes_values = figure.subplots(1, len(bands), squeeze=False)[0]
        style_axes = list(axes_values)
        displays = []
        for band in bands:
            magnitude = np.asarray(band["magnitude"], dtype=np.float32)
            intensity = np.empty_like(magnitude, dtype=np.float32)
            np.multiply(magnitude, magnitude, out=intensity)
            if str(scale).strip().lower() != "linear":


                np.maximum(intensity, np.float32(1.0e-12), out=intensity)
                np.log10(intensity, out=intensity)
                intensity *= np.float32(10.0)
            displays.append(intensity)

        clamp = None
        if color_limits is not None:
            values = np.asarray(color_limits, dtype=float).reshape(-1)
            if (
                values.size != 2
                or not np.all(np.isfinite(values))
                or not values[0] < values[1]
            ):
                raise ValueError("color_limits must be two finite increasing values")
            clamp = (float(values[0]), float(values[1]))
        shared_limits = (
            plot_common.finite_data_limits(displays)
            if bool(shared_colorbar) and clamp is None
            else None
        )
        plot_vmin = clamp[0] if clamp is not None else (
            shared_limits[0] if shared_limits is not None else None
        )
        plot_vmax = clamp[1] if clamp is not None else (
            shared_limits[1] if shared_limits is not None else None
        )

        meshes = []
        for axis, band, display in zip(axes_values, bands, displays):
            mesh = axis.imshow(
                display.T,
                extent=[band["x_range"][0], band["x_range"][-1], band["y_range"][0], band["y_range"][-1]],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=colormap,
                vmin=plot_vmin,
                vmax=plot_vmax,
            )
            if bool(square_aspect):
                axis.set_aspect("equal", adjustable="datalim")
            meshes.append(mesh)

        unit_name, _unit_scale = _length_unit(options["length_unit"])
        elevation_native = float(np.asarray(dataset.elevations)[el_idx])
        elevation_deg = float(
            plot_common.convert_axis_values(
                [elevation_native],
                "elevation",
                plot_common.axis_unit(dataset, "elevation"),
                "deg",
            )[0]
        )
        elevation_name = plot_common.angular_axis_name(dataset, "elevation")
        reconstruction = str(options.get("reconstruction", "fast")).strip().lower()
        if reconstruction in {"sparse", "l1", "sparse-l1"}:
            reconstruction_label = "Sparse L1 (Experimental)"
        elif reconstruction in {"accurate", "cartesian", "pfa-accurate"}:
            reconstruction_label = "Cartesian PFA"
        else:
            reconstruction_label = "Fast PFA"
        composite_sublooks = max(
            (int(band.get("composite", 0)) for band in bands), default=0
        )
        if composite_sublooks:
            reconstruction_label += (
                f" | Wide-Aperture Composite ({composite_sublooks} looks)"
            )
        title = (
            f"ISAR Image | {elevation_name} {elevation_deg:g} deg | "
            f"Pol {dataset.polarizations[pol_idx]} | {reconstruction_label}"
        )
        if len(bands) > 1:
            figure.suptitle(title)
            for axis, band in zip(axes_values, bands):
                axis.set_title(
                    f"{float(band['az_values'][0]):g}°–"
                    f"{float(band['az_values'][-1]):g}°"
                )
        else:
            axes_values[0].set_title(title)

        horizontal_projection = abs(elevation_deg) > 1.0e-9
        if composite_sublooks:
            x_label = f"Cross-Range at 0° ({unit_name})"
            y_label = f"Down-Range at 0° ({unit_name})"
        elif horizontal_projection:
            x_label = f"Horizontal Cross-Range ({unit_name})"
            y_label = f"Horizontal Range ({unit_name})"
        else:
            x_label = f"Cross-Range ({unit_name})"
            y_label = f"Range ({unit_name})"
        for axis in axes_values:
            axis.set_xlabel(x_label)
        axes_values[0].set_ylabel(y_label)

        colorbars = []
        if bool(show_colorbar) and meshes:
            if bool(shared_colorbar):
                colorbars.append(figure.colorbar(meshes[-1], ax=list(axes_values)))
            else:
                colorbars.extend(
                    figure.colorbar(mesh, ax=axis)
                    for axis, mesh in zip(axes_values, meshes)
                )
            label = (
                "Image Intensity (linear)"
                if str(scale).strip().lower() == "linear"
                else "Image Intensity (dB)"
            )
            try:
                tick_step = float(color_step)
            except (TypeError, ValueError) as exc:
                raise ValueError("color_step must be finite and nonnegative") from exc
            if not np.isfinite(tick_step) or tick_step < 0.0:
                raise ValueError("color_step must be finite and nonnegative")
            for colorbar in colorbars:
                colorbar.set_label(label)
                if tick_step > 0.0:
                    low, high = colorbar.mappable.get_clim()
                    first = math.ceil(low / tick_step) * tick_step
                    tick_count = int(math.floor((high - first) / tick_step)) + 1
                    if 0 < tick_count <= 1000:
                        colorbar.set_ticks(first + tick_step * np.arange(tick_count))

        sparse_results = [
            band for band in bands if band.get("sparse_iterations") is not None
        ]
        if sparse_results and not all(
            bool(band.get("sparse_converged", False)) for band in sparse_results
        ):
            worst_gap = max(
                float(band.get("sparse_relative_duality_gap", float("inf")))
                for band in sparse_results
            )
            warnings.warn(
                "ISAR sparse reconstruction did not certify convergence for "
                f"every band (worst relative duality gap {worst_gap:.3g}); "
                "treat the image as diagnostic, not quantitatively final.",
                RuntimeWarning,
                stacklevel=2,
            )
        axes = axes_values[0]
    else:
        raise ValueError(f"Unsupported plot mode: {mode!r}")

    if compatible_count == 0:
        figure.clear()
        raise ValueError(
            "None of the selected datasets contains every requested plot axis value"
        )

    for axis in (style_axes if style_axes is not None else figure.axes):
        axis.grid(bool(show_grid))
        handles, labels = axis.get_legend_handles_labels()
        if show_legend and handles:
            axis.legend()
    figure.tight_layout()
    return figure
