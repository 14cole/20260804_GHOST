from __future__ import annotations

import threading

import numpy as np

from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QListWidget,
    QToolButton,
)

from grim_dataset import RcsGrid
from plot_modes import (
    az_vs_range_mode,
    azimuth_polar_mode,
    azimuth_rect_mode,
    compare_mode,
    elevation_sweep_mode,
    frequency_mode,
    isar_mode,
    waterfall_mode,
)


class _IsarComputeSignals(QObject):
    """Bridges the ISAR worker thread back to the GUI thread. Emitting from
    the worker queues the slot call onto the GUI event loop (cross-thread
    signals auto-queue), so all Qt/matplotlib work stays on the GUI thread."""

    done = Signal(object, object)  # (params, result)


class PlotOpsMixin:
    def _on_param_selection_changed(self) -> None:
        self._maybe_autoplot()

    def _on_polarization_selection_changed(self) -> None:
        if self.active_dataset is None:
            return
        selected_pol = sorted(self._selected_indices(self.list_pol))
        if not selected_pol:
            self._sync_axis_list(self.list_freq, self.active_dataset.frequencies, None)
            self._sync_axis_list(self.list_elev, self.active_dataset.elevations, None)
            self._sync_axis_list(self.list_az, self.active_dataset.azimuths, None)
            return

        pwr_sel = self.active_dataset.rcs_power[:, :, :, selected_pol]
        if self._button_checked(self.btn_phase):
            phs_sel = self.active_dataset.rcs_phase[:, :, :, selected_pol]
            valid = np.isfinite(pwr_sel) & np.isfinite(phs_sel)
        else:
            valid = np.isfinite(pwr_sel)

        self._sync_axis_list(
            self.list_freq, self.active_dataset.frequencies, valid.any(axis=(0, 1, 3))
        )
        self._sync_axis_list(
            self.list_elev, self.active_dataset.elevations, valid.any(axis=(0, 2, 3))
        )
        self._sync_axis_list(
            self.list_az, self.active_dataset.azimuths, valid.any(axis=(1, 2, 3))
        )
        self._maybe_autoplot()

    def _sync_axis_list(self, widget, values, avail_mask) -> None:
        """Refill an axis list only if the set of displayed indices changed.

        Skipping unchanged rebuilds preserves selection without reselect calls
        and avoids the QListWidget churn that dominates UI lag for large axes
        (e.g. 1601 frequency samples).
        """
        if avail_mask is None:
            new_indices = set(range(len(values)))
            new_index_list = list(range(len(values)))
        else:
            new_index_list = [int(i) for i in np.where(avail_mask)[0]]
            new_indices = set(new_index_list)
        if self._displayed_indices(widget) == new_indices:
            return
        prev_selection = self._selected_indices(widget)
        self._fill_list(widget, values, new_index_list)
        self._reselect_indices(widget, prev_selection)

    def _maybe_autoplot(self) -> None:
        if not self._button_checked(self.btn_auto_plot):
            return
        if self.last_plot_mode is None:
            return
        # Debounce burst selection events (shift-click, ctrl-A) so a tight
        # sequence collapses into a single render. 50 ms feels responsive
        # but groups bursts; matters most for ISAR with many freq samples.
        timer = getattr(self, "_autoplot_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._do_autoplot)
            self._autoplot_timer = timer
        timer.start(50)

    def _do_autoplot(self) -> None:
        if self.last_plot_mode is None:
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
        elif self.last_plot_mode == "elevation_sweep":
            self._plot_elevation_sweep()
        elif self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()
        elif self.last_plot_mode == "compare":
            self._plot_compare()

    def _maybe_autoscale(self) -> None:
        """Auto-fit the view after a render when the Auto Scale toggle is on.

        Mirrors a Fit Both click so axes track the current data without the
        user reaching for the button after every (re)plot.
        """
        if not self._button_checked(getattr(self, "btn_auto_scale", None)):
            return
        if self.last_plot_mode is None:
            return
        self._fit_both()

    def _on_auto_scale_toggled(self) -> None:
        # Apply at once so enabling the toggle fits whatever is already plotted.
        self._maybe_autoscale()

    def _on_pbp_toggled(self) -> None:
        if self.last_plot_mode is None:
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()

    def _on_waterfall_style_changed(self) -> None:
        if self.last_plot_mode not in ("waterfall", "isar_image", "az_vs_range"):
            return
        if self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()
        else:
            self._plot_isar_image()

    def _on_colormap_changed(self) -> None:
        if self.last_plot_mode == "waterfall":
            self._plot_waterfall()
            return
        if self.last_plot_mode == "isar_image":
            self._plot_isar_image()
            return
        if self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()
            return
        if self.pbp_fill_mode not in ("heatmap_rcs", "heatmap_density"):
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()

    def _on_plot_scale_changed(self) -> None:
        if self.last_plot_mode is None:
            self._apply_plot_theme()
            return
        if self.last_plot_mode == "azimuth_rect":
            self._plot_azimuth_rect()
            self._fit_y()
        elif self.last_plot_mode == "azimuth_polar":
            self._plot_azimuth_polar()
            self._fit_y()
        elif self.last_plot_mode == "frequency":
            self._plot_frequency()
            self._fit_y()
        elif self.last_plot_mode == "elevation_sweep":
            self._plot_elevation_sweep()
            self._fit_y()
        elif self.last_plot_mode == "waterfall":
            self._plot_waterfall()
        elif self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()

    def _plot_scale_mode(self) -> str:
        scale = self.combo_plot_scale.currentData()
        if scale in ("dbsm", "linear"):
            return scale
        return "dbsm"

    def _plot_scale_is_linear(self) -> bool:
        return self._plot_scale_mode() == "linear"

    def _rcs_display_values(self, dataset: RcsGrid, rcs_values, frequency_value=None):
        if self._button_checked(self.btn_phase):
            return np.degrees(np.angle(rcs_values))
        if self._plot_scale_is_linear():
            return dataset.rcs_to_linear(rcs_values)
        return dataset.rcs_to_display_db(rcs_values, frequency_value=frequency_value)

    def _rcs_axis_label(self) -> str:
        if self._button_checked(self.btn_phase):
            return "Phase (deg)"
        if self._plot_scale_is_linear():
            return "RCS (Linear)"
        return "RCS (dBsm)"

    def _rcs_p50_axis_label(self) -> str:
        if self._button_checked(self.btn_phase):
            return "Phase P50 (deg)"
        if self._plot_scale_is_linear():
            return "RCS P50 (Linear)"
        return "RCS P50 (dBsm)"

    def _polar_zero_location(self) -> str:
        loc = self.combo_polar_zero.currentData()
        if isinstance(loc, str) and loc:
            return loc
        return "N"

    def _apply_polar_orientation(self, ax) -> None:
        if ax.name != "polar":
            return
        ax.set_theta_zero_location(self._polar_zero_location())
        # Compass convention: azimuth increases clockwise.
        ax.set_theta_direction(-1)
        # Label tick marks in (-180, 180] so -90 shows on the left (W) under
        # the default N-up/CW orientation, matching the compass grid.
        tick_deg = np.arange(0, 360, 30)
        labels = [str(int(t if t <= 180 else t - 360)) for t in tick_deg]
        ax.set_thetagrids(tick_deg, labels=labels)

    def _apply_polar_zero_direction(self) -> None:
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            self._apply_polar_orientation(ax)

    def _on_polar_zero_changed(self) -> None:
        self._apply_polar_zero_direction()
        self.plot_canvas.draw_idle()

    def _edges_from_centers(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            step = 1.0
            return np.array([values[0] - 0.5 * step, values[0] + 0.5 * step], dtype=float)
        diffs = np.diff(values)
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = values[:-1] + diffs / 2.0
        edges[0] = values[0] - diffs[0] / 2.0
        edges[-1] = values[-1] + diffs[-1] / 2.0
        return edges

    def _plot_pbp_heatmap(
        self,
        x_values,
        y_min,
        y_max,
        *,
        density: np.ndarray | None = None,
    ) -> None:
        x_values = np.asarray(x_values, dtype=float)
        y_min = np.asarray(y_min, dtype=float)
        y_max = np.asarray(y_max, dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_min) & np.isfinite(y_max)
        if not np.any(valid):
            return

        def draw_segment(seg_x, seg_min, seg_max, seg_density=None) -> None:
            lower = np.minimum(seg_min, seg_max)
            upper = np.maximum(seg_min, seg_max)
            if seg_x.size == 0:
                return

            x_edges = self._edges_from_centers(seg_x)
            lower_edges = np.interp(x_edges, seg_x, lower, left=lower[0], right=lower[-1])
            upper_edges = np.interp(x_edges, seg_x, upper, left=upper[0], right=upper[-1])

            samples = max(8, int(self.pbp_heatmap_samples))
            y_edges = np.vstack(
                [np.linspace(lo, hi, samples + 1) for lo, hi in zip(lower_edges, upper_edges)]
            ).T
            if self.pbp_fill_mode == "heatmap_density":
                if seg_density is None:
                    return
                values = np.tile(seg_density, (samples, 1))
            else:
                values = np.vstack(
                    [np.linspace(lo, hi, samples) for lo, hi in zip(lower, upper)]
                ).T
            x_grid = np.tile(x_edges, (samples + 1, 1))

            cmap = self._effective_colormap()
            self.plot_ax.pcolormesh(x_grid, y_edges, values, shading="auto", cmap=cmap)

        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            elif not is_valid and start is not None:
                seg = slice(start, idx)
                seg_density = None
                if density is not None:
                    seg_density = np.asarray(density, dtype=float)[seg]
                draw_segment(x_values[seg], y_min[seg], y_max[seg], seg_density)
                start = None
        if start is not None:
            seg = slice(start, len(valid))
            seg_density = None
            if density is not None:
                seg_density = np.asarray(density, dtype=float)[seg]
            draw_segment(x_values[seg], y_min[seg], y_max[seg], seg_density)

    def _plot_pbp_fill(
        self,
        x_values,
        y_min,
        y_max,
        label: str,
        polar: bool,
        *,
        density: np.ndarray | None = None,
    ) -> None:
        if self.pbp_fill_mode in ("heatmap_rcs", "heatmap_density"):
            self._plot_pbp_heatmap(x_values, y_min, y_max, density=density)
            self.plot_ax.plot([], [], color=self.pbp_fill_gray, label=label)
            return
        self.plot_ax.fill_between(
            x_values,
            y_min,
            y_max,
            color=self.pbp_fill_gray,
            alpha=1.0,
            label=label,
        )

    def _style_axes(self, ax) -> None:
        bg = self._current_plot_bg()
        grid = self._current_plot_grid()
        text = self._current_plot_text()
        ax.set_facecolor(bg)
        grid_on = self._plot_grid_enabled()
        ax.grid(grid_on, color=grid, alpha=0.35)
        ax.tick_params(colors=text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        if hasattr(ax, "zaxis") and ax.zaxis is not None:
            ax.zaxis.label.set_color(text)
        if hasattr(ax, "spines"):
            for spine in ax.spines.values():
                spine.set_color(self.palette["border"])
        if ax.name == "polar":
            self._apply_polar_orientation(ax)

    def _style_plot_axes(self) -> None:
        self.plot_figure.set_facecolor(self._current_plot_bg())
        self._style_axes(self.plot_ax)

    def _plot_grid_enabled(self) -> bool:
        checkbox = getattr(self, "chk_plot_grid_visible", None)
        if checkbox is None:
            return True
        return bool(checkbox.isChecked())

    def _current_plot_bg(self) -> str:
        return self.plot_bg_color or self.palette["panel_bg"]

    def _current_plot_grid(self) -> str:
        return self.plot_grid_color or self.palette["grid"]

    def _current_plot_text(self) -> str:
        return self.plot_text_color or self.palette["text"]

    def _apply_plot_theme(self) -> None:
        self.plot_figure.set_facecolor(self._current_plot_bg())
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            self._style_axes(ax)
            legend = ax.get_legend()
            if legend is not None:
                self._configure_legend(legend, ax)
                for text in legend.get_texts():
                    text.set_color(self._current_plot_text())
                legend.get_frame().set_facecolor(self._current_plot_bg())
                legend.get_frame().set_edgecolor(self._current_plot_grid())
        for colorbar in self.plot_colorbars:
            label_text = colorbar.ax.get_ylabel() or self._rcs_axis_label()
            colorbar.set_label(label_text, color=self._current_plot_text())
            colorbar.ax.tick_params(colors=self._current_plot_text())
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self._current_plot_text())
        self.plot_canvas.draw_idle()

    def _apply_colorbar_ticks(self, colorbar) -> None:
        zstep = self.spin_plot_zstep.value()
        if zstep <= 0.0:
            return
        try:
            vmin, vmax = colorbar.mappable.get_clim()
        except Exception:
            return
        if vmin is None or vmax is None:
            return
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        ticks = np.arange(vmin, vmax + zstep * 0.5, zstep)
        if ticks.size == 0:
            return
        colorbar.set_ticks(ticks)

    def _choose_plot_color(self, which: str) -> None:
        if which == "bg":
            current = self._current_plot_bg()
            title = "Select Plot Background Color"
        elif which == "grid":
            current = self._current_plot_grid()
            title = "Select Plot Grid Color"
        else:
            current = self._current_plot_text()
            title = "Select Plot Text Color"
        color = QColorDialog.getColor(QColor(current), self, title)
        if not color.isValid():
            return
        if which == "bg":
            self.plot_bg_color = color.name()
        elif which == "grid":
            self.plot_grid_color = color.name()
        else:
            self.plot_text_color = color.name()
        self._update_plot_color_buttons()
        self._apply_plot_theme()

    def _update_plot_color_buttons(self) -> None:
        self.btn_plot_bg.setStyleSheet(f"background: {self._current_plot_bg()};")
        self.btn_plot_grid.setStyleSheet(f"background: {self._current_plot_grid()};")
        self.btn_plot_text.setStyleSheet(f"background: {self._current_plot_text()};")

    def _remove_colorbar(self) -> None:
        if not self.plot_colorbars:
            return
        for colorbar in self.plot_colorbars:
            try:
                if colorbar.ax is not None:
                    colorbar.remove()
            except Exception:
                pass
        self.plot_colorbars = []

    def _ensure_axes(self, projection: str) -> None:
        desired = "polar" if projection == "polar" else "rectilinear"
        if self.plot_ax.name == desired and self.plot_axes is None:
            return
        self._remove_colorbar()
        self.plot_figure.clear()
        if desired == "polar":
            self.plot_ax = self.plot_figure.add_subplot(111, projection="polar")
        else:
            self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        self._style_plot_axes()

    def _clear_plot(self) -> None:
        self.plot_ax.clear()
        self._remove_colorbar()
        self.plot_axes = None
        self._style_plot_axes()
        self._apply_plot_limits()

    def _single_selection_index(self, widget: QListWidget, label: str) -> int | None:
        selected = sorted(self._selected_indices(widget))
        if len(selected) != 1:
            count = len(selected)
            if count == 0:
                msg = f"Select 1 {label} to plot."
            else:
                msg = f"Select exactly 1 {label} (selected {count})."
            self.status.showMessage(msg)
            return None
        return selected[0]

    def _single_selection_value(self, widget: QListWidget, label: str):
        values = self._selected_values(widget)
        if len(values) != 1:
            count = len(values)
            if count == 0:
                msg = f"Select 1 {label} to plot."
            else:
                msg = f"Select exactly 1 {label} (selected {count})."
            self.status.showMessage(msg)
            return None
        return values[0]

    @staticmethod
    def _button_checked(button: QToolButton | None) -> bool:
        return bool(button.isChecked()) if button is not None else False

    # --- shared display-quantity helpers (used by the plot_modes modules) ----

    def _display_from_values(self, dataset, values, frequency_value):
        """Raw samples -> the current display quantity: phase (deg) when the
        Phase toggle is on, otherwise linear or the dataset's default dB per
        the plot-scale combo. `frequency_value` may be scalar or an array
        broadcastable against `values` (frequency-dependent dB conversions)."""
        if self._button_checked(self.btn_phase):
            phase_deg = np.degrees(np.angle(values))
            return np.where(np.isfinite(phase_deg), phase_deg, np.nan)
        linear = dataset.rcs_to_linear(values)
        if self._plot_scale_is_linear():
            return linear
        return dataset.linear_to_default_db(linear, frequency_value=frequency_value)

    def _display_from_linear(self, dataset, linear_values, frequency_value):
        """Already-linear power values -> current display scale (no phase branch;
        used by the P50/statistics paths that aggregate in linear space)."""
        if self._plot_scale_is_linear():
            return linear_values
        return dataset.linear_to_default_db(linear_values, frequency_value=frequency_value)

    def _display_unit(self, datasets) -> str:
        """Unit string for the current display quantity: deg / Linear / dB unit."""
        if self._button_checked(self.btn_phase):
            return "deg"
        if self._plot_scale_is_linear():
            return "Linear"
        units = {str(ds.default_log_unit()) for _, ds in datasets}
        return next(iter(units)) if len(units) == 1 else "dB"

    def _display_axis_label(self, datasets, tag: str = "") -> str:
        """Axis/colorbar label for the current display quantity; `tag` is an
        optional qualifier inserted after the quantity name (e.g. " P50")."""
        if self._button_checked(self.btn_phase):
            return f"Phase{tag} (deg)"
        if self._plot_scale_is_linear():
            return f"RCS{tag} (Linear)"
        return f"RCS{tag} ({self._display_unit(datasets)})"

    def _zoom_target_axes(self) -> list:
        axes = self.plot_axes or [self.plot_ax]
        return [ax for ax in axes if ax is not None]

    def _sync_plot_limit_spins_from_axes(
        self,
        ax,
        *,
        sync_x: bool = True,
        sync_y: bool = True,
    ) -> None:
        if ax is None:
            return

        if sync_x:
            if ax.name == "polar":
                xmin, xmax = np.degrees(ax.get_xlim())
            else:
                xmin, xmax = ax.get_xlim()
            if np.isfinite(xmin) and np.isfinite(xmax):
                self.spin_plot_xmin.blockSignals(True)
                self.spin_plot_xmax.blockSignals(True)
                self.spin_plot_xmin.setValue(float(xmin))
                self.spin_plot_xmax.setValue(float(xmax))
                self.spin_plot_xmin.blockSignals(False)
                self.spin_plot_xmax.blockSignals(False)

        if sync_y:
            ymin, ymax = ax.get_ylim()
            if np.isfinite(ymin) and np.isfinite(ymax):
                self.spin_plot_ymin.blockSignals(True)
                self.spin_plot_ymax.blockSignals(True)
                self.spin_plot_ymin.setValue(float(ymin))
                self.spin_plot_ymax.setValue(float(ymax))
                self.spin_plot_ymin.blockSignals(False)
                self.spin_plot_ymax.blockSignals(False)

        self._apply_plot_limits()

    def _clear_zoom_box_drag(self) -> None:
        drag = getattr(self, "_zoom_box_drag", None)
        if not isinstance(drag, dict):
            self._zoom_box_drag = None
            return
        patch = drag.get("patch")
        canvas = drag.get("canvas")
        try:
            if patch is not None:
                patch.remove()
        except Exception:
            pass
        self._zoom_box_drag = None
        try:
            if canvas is not None:
                canvas.draw_idle()
        except Exception:
            pass

    def _clear_pan_drag(self, *, sync_limits: bool = False) -> None:
        drag = getattr(self, "_pan_drag", None)
        if not isinstance(drag, dict):
            self._pan_drag = None
            return
        ax = drag.get("ax")
        self._pan_drag = None
        if sync_limits and ax is not None:
            self._sync_plot_limit_spins_from_axes(ax, sync_x=True, sync_y=True)

    @staticmethod
    def _uncheck_silently(btn) -> None:
        if btn is not None and btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def _on_zoom_box_toggled(self, checked: bool) -> None:
        if not checked:
            self._clear_zoom_box_drag()
            return
        if self.plot_ax.name in ("polar", "3d"):
            self._uncheck_silently(getattr(self, "btn_zoom_box", None))
            self.status.showMessage("Box zoom is available on 2D rectilinear plots.")
            return
        # Zoom box and pan both claim the left button — one at a time.
        self._uncheck_silently(getattr(self, "btn_pan", None))
        self.status.showMessage("Box zoom enabled. Drag left mouse on the plot to zoom.")

    def _on_pan_toggled(self, checked: bool) -> None:
        if not checked:
            self._clear_pan_drag()
            return
        if self.plot_ax.name in ("polar", "3d"):
            self._uncheck_silently(getattr(self, "btn_pan", None))
            self.status.showMessage("Pan is available on 2D rectilinear plots.")
            return
        self._uncheck_silently(getattr(self, "btn_zoom_box", None))
        self._clear_zoom_box_drag()
        self.status.showMessage(
            "Pan enabled. Drag left mouse to move around the plot "
            "(middle-drag always pans)."
        )

    def _on_plot_scroll_zoom(self, event) -> None:
        if getattr(event, "canvas", None) is not self.plot_canvas:
            return
        ax = getattr(event, "inaxes", None)
        if ax not in self._zoom_target_axes():
            return
        if ax.name == "3d":
            return
        if getattr(self, "_zoom_box_drag", None):
            return
        pan_drag = getattr(self, "_pan_drag", None)
        if isinstance(pan_drag, dict) and pan_drag.get("canvas") is event.canvas:
            return

        step = getattr(event, "step", None)
        if step is None:
            button = getattr(event, "button", None)
            if button == "up":
                step = 1.0
            elif button == "down":
                step = -1.0
            else:
                return
        if not np.isfinite(step) or np.isclose(step, 0.0):
            return

        zoom_base = 1.2
        zoom_factor = float(np.power(zoom_base, -step))
        if not np.isfinite(zoom_factor) or zoom_factor <= 0.0:
            return

        if ax.name == "polar":
            ymin, ymax = ax.get_ylim()
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                return
            ycenter = getattr(event, "ydata", None)
            if ycenter is None or not np.isfinite(ycenter):
                ycenter = 0.5 * (float(ymin) + float(ymax))
            new_ymin = float(ycenter) - (float(ycenter) - float(ymin)) * zoom_factor
            new_ymax = float(ycenter) + (float(ymax) - float(ycenter)) * zoom_factor
            if not np.isfinite(new_ymin) or not np.isfinite(new_ymax):
                return
            if np.isclose(new_ymin, new_ymax):
                return
            ax.set_ylim(new_ymin, new_ymax)
            self._sync_plot_limit_spins_from_axes(ax, sync_x=False, sync_y=True)
            return

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            return
        xcenter = getattr(event, "xdata", None)
        ycenter = getattr(event, "ydata", None)
        if xcenter is None or not np.isfinite(xcenter):
            xcenter = 0.5 * (float(xmin) + float(xmax))
        if ycenter is None or not np.isfinite(ycenter):
            ycenter = 0.5 * (float(ymin) + float(ymax))

        new_xmin = float(xcenter) - (float(xcenter) - float(xmin)) * zoom_factor
        new_xmax = float(xcenter) + (float(xmax) - float(xcenter)) * zoom_factor
        new_ymin = float(ycenter) - (float(ycenter) - float(ymin)) * zoom_factor
        new_ymax = float(ycenter) + (float(ymax) - float(ycenter)) * zoom_factor
        if not all(np.isfinite(v) for v in (new_xmin, new_xmax, new_ymin, new_ymax)):
            return
        if np.isclose(new_xmin, new_xmax) or np.isclose(new_ymin, new_ymax):
            return

        ax.set_xlim(new_xmin, new_xmax)
        ax.set_ylim(new_ymin, new_ymax)
        self._sync_plot_limit_spins_from_axes(ax, sync_x=True, sync_y=True)

    def _on_plot_mouse_press(self, event) -> None:
        if getattr(event, "canvas", None) is not self.plot_canvas:
            return
        button = getattr(event, "button", None)
        pan_requested = button in (MouseButton.MIDDLE, 2) or (
            button is MouseButton.LEFT
            and self._button_checked(getattr(self, "btn_pan", None))
        )
        if pan_requested:
            self._clear_zoom_box_drag()
            ax = getattr(event, "inaxes", None)
            if ax not in self._zoom_target_axes():
                return
            if ax.name in ("polar", "3d"):
                return
            x0 = getattr(event, "xdata", None)
            y0 = getattr(event, "ydata", None)
            if x0 is None or y0 is None or not np.isfinite(x0) or not np.isfinite(y0):
                return
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            if not all(np.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
                return
            self._pan_drag = {
                "ax": ax,
                "canvas": event.canvas,
                "x0": float(x0),
                "y0": float(y0),
                "xlim0": (float(xmin), float(xmax)),
                "ylim0": (float(ymin), float(ymax)),
            }
            return
        if button is not MouseButton.LEFT:
            return
        if not self._button_checked(getattr(self, "btn_zoom_box", None)):
            return
        ax = getattr(event, "inaxes", None)
        if ax not in self._zoom_target_axes():
            return
        if ax.name in ("polar", "3d"):
            self.status.showMessage("Box zoom is available on 2D rectilinear plots.")
            return
        x0 = getattr(event, "xdata", None)
        y0 = getattr(event, "ydata", None)
        if x0 is None or y0 is None or not np.isfinite(x0) or not np.isfinite(y0):
            return

        self._clear_zoom_box_drag()
        patch = Rectangle(
            (float(x0), float(y0)),
            0.0,
            0.0,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor=self._current_plot_text(),
            alpha=0.9,
        )
        ax.add_patch(patch)
        self._zoom_box_drag = {
            "ax": ax,
            "canvas": event.canvas,
            "x0": float(x0),
            "y0": float(y0),
            "patch": patch,
        }
        event.canvas.draw_idle()

    def _on_plot_mouse_move(self, event) -> None:
        pan_drag = getattr(self, "_pan_drag", None)
        if isinstance(pan_drag, dict) and getattr(event, "canvas", None) is pan_drag.get("canvas"):
            ax = pan_drag.get("ax")
            if ax is None or getattr(event, "inaxes", None) is not ax:
                return
            x1 = getattr(event, "xdata", None)
            y1 = getattr(event, "ydata", None)
            if x1 is None or y1 is None or not np.isfinite(x1) or not np.isfinite(y1):
                return

            dx = float(x1) - float(pan_drag["x0"])
            dy = float(y1) - float(pan_drag["y0"])
            xmin0, xmax0 = pan_drag["xlim0"]
            ymin0, ymax0 = pan_drag["ylim0"]
            ax.set_xlim(float(xmin0) - dx, float(xmax0) - dx)
            ax.set_ylim(float(ymin0) - dy, float(ymax0) - dy)
            event.canvas.draw_idle()
            return

        drag = getattr(self, "_zoom_box_drag", None)
        if not isinstance(drag, dict):
            return
        if getattr(event, "canvas", None) is not drag.get("canvas"):
            return

        ax = drag.get("ax")
        patch = drag.get("patch")
        if ax is None or patch is None:
            return
        if getattr(event, "inaxes", None) is not ax:
            return
        x1 = getattr(event, "xdata", None)
        y1 = getattr(event, "ydata", None)
        if x1 is None or y1 is None or not np.isfinite(x1) or not np.isfinite(y1):
            return

        x0 = float(drag["x0"])
        y0 = float(drag["y0"])
        x_min = min(x0, float(x1))
        x_max = max(x0, float(x1))
        y_min = min(y0, float(y1))
        y_max = max(y0, float(y1))
        patch.set_x(x_min)
        patch.set_y(y_min)
        patch.set_width(x_max - x_min)
        patch.set_height(y_max - y_min)
        event.canvas.draw_idle()

    def _on_plot_mouse_release(self, event) -> None:
        pan_drag = getattr(self, "_pan_drag", None)
        if isinstance(pan_drag, dict) and getattr(event, "canvas", None) is pan_drag.get("canvas"):
            sync = getattr(event, "button", None) in (MouseButton.MIDDLE, 2, MouseButton.LEFT)
            self._clear_pan_drag(sync_limits=sync)
            if getattr(event, "canvas", None) is not None:
                event.canvas.draw_idle()
            return

        drag = getattr(self, "_zoom_box_drag", None)
        if not isinstance(drag, dict):
            return
        if getattr(event, "canvas", None) is not drag.get("canvas"):
            return

        ax = drag.get("ax")
        patch = drag.get("patch")
        if patch is not None:
            try:
                patch.remove()
            except Exception:
                pass
        self._zoom_box_drag = None

        if getattr(event, "button", None) is not MouseButton.LEFT:
            event.canvas.draw_idle()
            return
        if ax is None or getattr(event, "inaxes", None) is not ax:
            event.canvas.draw_idle()
            return
        x1 = getattr(event, "xdata", None)
        y1 = getattr(event, "ydata", None)
        if x1 is None or y1 is None or not np.isfinite(x1) or not np.isfinite(y1):
            event.canvas.draw_idle()
            return

        x0 = float(drag["x0"])
        y0 = float(drag["y0"])
        x_min = min(x0, float(x1))
        x_max = max(x0, float(x1))
        y_min = min(y0, float(y1))
        y_max = max(y0, float(y1))

        cur_xmin, cur_xmax = ax.get_xlim()
        cur_ymin, cur_ymax = ax.get_ylim()
        x_threshold = max(abs(float(cur_xmax) - float(cur_xmin)) * 0.005, 1e-12)
        y_threshold = max(abs(float(cur_ymax) - float(cur_ymin)) * 0.005, 1e-12)
        if (x_max - x_min) <= x_threshold or (y_max - y_min) <= y_threshold:
            event.canvas.draw_idle()
            return

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        self._sync_plot_limit_spins_from_axes(ax, sync_x=True, sync_y=True)

    def _apply_plot_limits(self) -> None:
        xmin = self.spin_plot_xmin.value()
        xmax = self.spin_plot_xmax.value()
        ymin = self.spin_plot_ymin.value()
        ymax = self.spin_plot_ymax.value()
        xstep = self.spin_plot_xstep.value()
        ystep = self.spin_plot_ystep.value()
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            ax.set_autoscale_on(False)
            if ax.name == "polar":
                # Always show the full polar grid; the x limits only choose
                # which azimuths get plotted, never a visible wedge.
                ax.set_thetamin(0.0)
                ax.set_thetamax(360.0)
                theta_step = xstep if xstep > 0.0 else 45.0
                ax.set_thetagrids(np.arange(0.0, 360.0, theta_step))
            else:
                ax.set_xlim(xmin, xmax)
                if xstep > 0.0:
                    ax.set_xticks(np.arange(xmin, xmax + xstep * 0.5, xstep))
            ax.set_ylim(ymin, ymax)
            if ystep > 0.0:
                ax.set_yticks(np.arange(ymin, ymax + ystep * 0.5, ystep))
        self.plot_canvas.draw_idle()

    def _fit_both(self) -> None:
        if self.plot_ax.name == "polar":
            self._fit_y()
            return
        self._fit_x()
        self._fit_y()

    def _effective_colormap(self) -> str:
        name = self.combo_colormap.currentText()
        if self.chk_colormap_invert.isChecked():
            name = name + "_r"
        return name

    def _isar_window(self, n: int) -> np.ndarray:
        # Window math lives in isar_mode._window_array (Qt-free) so the ISAR
        # worker thread can use it; this wrapper just reads the combo.
        return isar_mode._window_array(self.combo_isar_window.currentText(), n)

    # ------------------------------------------------------------------
    # Async ISAR rendering: compute on a worker thread, display on the GUI
    # thread, coalesce bursts of requests (spinbox keystrokes, scrubbing).
    # ------------------------------------------------------------------

    def _isar_submit(self, params: dict) -> None:
        cancel_event = threading.Event()
        params["cancel_check"] = cancel_event.is_set
        if getattr(self, "_isar_busy", False):
            # Latest request wins; it launches as soon as the current one lands.
            current_cancel = getattr(self, "_isar_cancel_event", None)
            if current_cancel is not None:
                current_cancel.set()
            old_pending = getattr(self, "_isar_pending", None)
            if old_pending is not None:
                old_check = old_pending.get("_cancel_event")
                if old_check is not None:
                    old_check.set()
            params["_cancel_event"] = cancel_event
            self._isar_pending = params
            return
        self._isar_busy = True
        self._isar_cancel_event = cancel_event
        params["_cancel_event"] = cancel_event
        signals = getattr(self, "_isar_signals", None)
        if signals is None:
            signals = self._isar_signals = _IsarComputeSignals()
            signals.done.connect(self._on_isar_compute_done)
        self.status.showMessage("Computing ISAR image…")

        def work(params=params, signals=signals):
            try:
                result = isar_mode.compute_bands(params)
            except Exception as exc:  # surface, don't kill the worker silently
                result = f"ISAR computation failed: {exc}"
            signals.done.emit(params, result)

        QThreadPool.globalInstance().start(work)

    def _on_isar_compute_done(self, params: dict, result) -> None:
        self._isar_busy = False
        self._isar_cancel_event = None
        pending = getattr(self, "_isar_pending", None)
        self._isar_pending = None
        ok = not isinstance(result, str)
        superseded = pending is not None and params.get("_cancel_event") is not None \
            and params["_cancel_event"].is_set()
        if superseded:
            ok = False
        elif not ok:
            self.status.showMessage(result)
        elif (
            params.get("dataset") is not self.active_dataset
            or params.get("figure_token") is not self.plot_figure
        ):
            self.status.showMessage("ISAR result discarded (view changed while computing).")
            ok = False
        else:
            band_results, elapsed = result
            isar_mode.display_results(self, params, band_results, elapsed)
        if pending is not None:
            self._isar_submit(pending)
            return
        # Cine mode: keep stepping while Play stays toggled and renders succeed.
        play_btn = getattr(self, "btn_isar_ap_play", None)
        if ok and play_btn is not None and play_btn.isChecked():
            QTimer.singleShot(30, self._isar_play_step)

    # ------------------------------------------------------------------
    # Aperture scrubbing (center/width + step/play) and peak-relative clim
    # ------------------------------------------------------------------

    def _isar_step_aperture(self, direction: int) -> None:
        center_spin = getattr(self, "spin_isar_ap_center", None)
        width_spin = getattr(self, "spin_isar_ap_width", None)
        if center_spin is None or width_spin is None:
            return
        chk = getattr(self, "chk_isar_aperture", None)
        if chk is not None and not chk.isChecked():
            chk.setChecked(True)  # toggling also triggers a render on the ISAR tab
        width = float(width_spin.value())
        step = width / 2.0 if width > 0.0 else 1.0  # 50% overlap between looks
        new_center = float(np.mod(center_spin.value() + direction * step, 360.0))
        old_center = float(center_spin.value())
        center_spin.setValue(new_center)  # valueChanged triggers the render
        if np.isclose(new_center, old_center) or self.last_plot_mode != "isar_image":
            # Value didn't change (no signal) or ISAR was never plotted yet —
            # kick the render explicitly so the button always does something.
            self._plot_isar_image()

    def _on_isar_ap_prev(self) -> None:
        self._isar_step_aperture(-1)

    def _on_isar_ap_next(self) -> None:
        self._isar_step_aperture(+1)

    def _isar_play_step(self) -> None:
        play_btn = getattr(self, "btn_isar_ap_play", None)
        if play_btn is None or not play_btn.isChecked():
            return
        self._isar_step_aperture(+1)

    def _on_isar_ap_play(self, checked: bool) -> None:
        if checked:
            self._isar_play_step()

    def _on_isar_peak_scale(self) -> None:
        """Set the color scale to [peak − N, peak] of the rendered image —
        the standard ISAR dynamic-range convention. Works on the displayed
        data, so it's instant (no recompute)."""
        if self.last_plot_mode != "isar_image":
            self.status.showMessage("Peak scaling applies to a rendered ISAR image.")
            return
        meshes = [m for m in getattr(self, "_isar_meshes", None) or [] if m.axes is not None]
        if not meshes:
            self.status.showMessage("No ISAR image to scale — plot one first.")
            return
        peak = float("-inf")
        for mesh in meshes:
            arr = np.asarray(mesh.get_array(), dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                peak = max(peak, float(finite.max()))
        if not np.isfinite(peak):
            return
        drop = float(self.spin_isar_peak_drop.value())
        if self._plot_scale_is_linear():
            # Linear display shows |I| (amplitude): N dB down = 10^(-N/20).
            zmin = peak * 10.0 ** (-drop / 20.0)
        else:
            zmin = peak - drop
        self.spin_plot_zmin.blockSignals(True)
        self.spin_plot_zmax.blockSignals(True)
        self.spin_plot_zmin.setValue(zmin)
        self.spin_plot_zmax.setValue(peak)
        self.spin_plot_zmin.blockSignals(False)
        self.spin_plot_zmax.blockSignals(False)
        self._on_isar_clim_changed()
        self.status.showMessage(f"ISAR color scale set to peak − {drop:g} dB.")

    def _on_phase_toggled(self) -> None:
        self._on_polarization_selection_changed()
        self._maybe_autoplot()

    def _on_isar_window_changed(self) -> None:
        if self.last_plot_mode == "isar_image":
            self._plot_isar_image()
        elif self.last_plot_mode == "az_vs_range":
            self._plot_az_vs_range()

    def _on_isar_clim_changed(self) -> None:
        """Retune the rendered ISAR image's color scale in place. Unlike the
        Apply button this does NOT re-run the imaging — clim changes only
        touch the existing AxesImage, so they can respond per keystroke."""
        if self.last_plot_mode != "isar_image":
            return
        zmin = self.spin_plot_zmin.value()
        zmax = self.spin_plot_zmax.value()
        if zmin >= zmax:
            return
        meshes = [m for m in getattr(self, "_isar_meshes", None) or [] if m.axes is not None]
        if not meshes:
            return
        for mesh in meshes:
            mesh.set_clim(zmin, zmax)
        for colorbar in self.plot_colorbars or []:
            self._apply_colorbar_ticks(colorbar)
        self.plot_canvas.draw_idle()

    def _fit_polar_x_range(self) -> tuple[float, float]:
        theta_values: list[np.ndarray] = []
        for line in self.plot_ax.lines:
            try:
                x = np.asarray(line.get_xdata(), dtype=float)
            except Exception:
                continue
            if x.size == 0:
                continue
            finite = x[np.isfinite(x)]
            if finite.size:
                theta_values.append(np.degrees(finite))

        if not theta_values:
            xmin, xmax = np.degrees(self.plot_ax.get_xlim())
            xmin = float(xmin)
            xmax = float(xmax)
            if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmin, xmax):
                return -180.0, 180.0
            if xmax < xmin:
                xmax += 360.0
            if (xmax - xmin) >= 359.0:
                return -180.0, 180.0
            return xmin, xmax

        theta = np.mod(np.concatenate(theta_values), 360.0)
        theta.sort()
        if theta.size == 1:
            center = float(theta[0])
            return center - 5.0, center + 5.0

        wrapped = np.concatenate([theta, [theta[0] + 360.0]])
        gaps = np.diff(wrapped)
        gap_idx = int(np.argmax(gaps))
        largest_gap = float(gaps[gap_idx])
        span = 360.0 - largest_gap
        if span >= 359.0:
            return -180.0, 180.0

        start = float(theta[(gap_idx + 1) % theta.size])
        end = start + span
        pad = max(1.0, 0.03 * span)
        xmin = start - pad
        xmax = end + pad
        if (xmax - xmin) >= 359.0:
            return -180.0, 180.0

        while xmin > 180.0:
            xmin -= 360.0
            xmax -= 360.0
        while xmin <= -180.0:
            xmin += 360.0
            xmax += 360.0
        return xmin, xmax

    def _fit_polar_y_range(self) -> tuple[float, float]:
        radial_values: list[np.ndarray] = []
        for line in self.plot_ax.lines:
            try:
                y = np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            if y.size == 0:
                continue
            finite = y[np.isfinite(y)]
            if finite.size:
                radial_values.append(finite)

        if radial_values:
            radial = np.concatenate(radial_values)
            ymin = float(np.nanmin(radial))
            ymax = float(np.nanmax(radial))
        else:
            ymin, ymax = self.plot_ax.get_ylim()
            ymin = float(ymin)
            ymax = float(ymax)

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return -1.0, 1.0
        if np.isclose(ymin, ymax):
            pad = max(1.0, abs(ymin) * 0.05)
            ymin -= pad
            ymax += pad
        return ymin, ymax

    def _fit_x(self) -> None:
        if self.plot_ax.name == "polar":
            return

        self.plot_ax.set_autoscale_on(True)
        self.plot_ax.relim()
        self.plot_ax.autoscale_view(scalex=True, scaley=False)
        xmin, xmax = self.plot_ax.get_xlim()
        self.spin_plot_xmin.blockSignals(True)
        self.spin_plot_xmax.blockSignals(True)
        self.spin_plot_xmin.setValue(float(xmin))
        self.spin_plot_xmax.setValue(float(xmax))
        if self.spin_plot_xstep.value() > 0.0:
            self.spin_plot_xstep.blockSignals(True)
            self.spin_plot_xstep.setValue(0.0)
            self.spin_plot_xstep.blockSignals(False)
        self.spin_plot_xmin.blockSignals(False)
        self.spin_plot_xmax.blockSignals(False)
        self._apply_plot_limits()

    def _fit_y(self) -> None:
        if self.plot_ax.name == "polar":
            ymin, ymax = self._fit_polar_y_range()
            self.spin_plot_ymin.blockSignals(True)
            self.spin_plot_ymax.blockSignals(True)
            self.spin_plot_ymin.setValue(float(ymin))
            self.spin_plot_ymax.setValue(float(ymax))
            if self.spin_plot_ystep.value() > 0.0:
                self.spin_plot_ystep.blockSignals(True)
                self.spin_plot_ystep.setValue(0.0)
                self.spin_plot_ystep.blockSignals(False)
            self.spin_plot_ymin.blockSignals(False)
            self.spin_plot_ymax.blockSignals(False)
            axes = self.plot_axes or [self.plot_ax]
            for ax in axes:
                ax.set_autoscale_on(False)
                ax.set_ylim(ymin, ymax)
            self.plot_canvas.draw_idle()
            return
        else:
            self.plot_ax.set_autoscale_on(True)
            self.plot_ax.relim()
            self.plot_ax.autoscale_view(scalex=False, scaley=True)
            ymin, ymax = self.plot_ax.get_ylim()
        self.spin_plot_ymin.blockSignals(True)
        self.spin_plot_ymax.blockSignals(True)
        self.spin_plot_ymin.setValue(float(ymin))
        self.spin_plot_ymax.setValue(float(ymax))
        if self.spin_plot_ystep.value() > 0.0:
            self.spin_plot_ystep.blockSignals(True)
            self.spin_plot_ystep.setValue(0.0)
            self.spin_plot_ystep.blockSignals(False)
        self.spin_plot_ymin.blockSignals(False)
        self.spin_plot_ymax.blockSignals(False)
        self._apply_plot_limits()

    def _collect_azimuth_series(
        self,
        dataset: RcsGrid,
        dataset_name: str,
        az_values_sel: list,
        elev_values_sel: list,
        freq_values_sel: list,
        pol_value_sel,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, str]]] | None:
        az_indices = self._indices_for_values(dataset.azimuths, az_values_sel)
        elev_indices = self._indices_for_values(dataset.elevations, elev_values_sel)
        freq_indices = self._indices_for_values(dataset.frequencies, freq_values_sel)
        pol_indices = self._indices_for_values(dataset.polarizations, [pol_value_sel], tol=0.0)
        if az_indices is None or elev_indices is None or freq_indices is None or pol_indices is None:
            return None

        az_values = dataset.azimuths[az_indices]
        order = np.argsort(az_values)
        az_values = az_values[order]
        pol_value = dataset.polarizations[pol_indices[0]]
        series: list[tuple[np.ndarray, str]] = []
        for freq_idx in freq_indices:
            freq_value = dataset.frequencies[freq_idx]
            for elev_idx in elev_indices:
                elev_value = dataset.elevations[elev_idx]
                if self._button_checked(self.btn_phase):
                    rcs_values = dataset.rcs_slice((az_indices, elev_idx, freq_idx, pol_indices[0]))
                else:
                    rcs_values = dataset.rcs_power[az_indices, elev_idx, freq_idx, pol_indices[0]]
                rcs_display = self._rcs_display_values(dataset, rcs_values)
                rcs_display = rcs_display[order]
                label = (
                    f"{dataset_name} | Pol {pol_value}, Freq {freq_value} GHz, El {elev_value} deg"
                )
                series.append((rcs_display, label))

        return az_values, series

    def _legend_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "loc": "upper right",
            "frameon": True,
            "framealpha": 0.92,
        }
        if self.last_plot_mode == "compare":
            kwargs["fontsize"] = 8
        return kwargs

    def _configure_legend(self, legend, ax=None) -> None:
        if legend is None:
            return
        if ax is None:
            ax = self.plot_ax
        # Reset a previously dragged/off-canvas legend. Legend.set_loc was
        # added after older Matplotlib releases still found on clusters, and
        # set_bbox_to_anchor's transform keyword also differs by release, so
        # keep the two operations independent and provide the stable numeric
        # upper-right fallback (loc code 1).
        try:
            legend.set_loc("upper right")
        except Exception:
            try:
                legend._loc = 1
            except Exception:
                pass
        try:
            legend.set_bbox_to_anchor(None)
        except Exception:
            try:
                legend._bbox_to_anchor = None
            except Exception:
                pass
        legend.set_visible(True)
        legend.set_zorder(1000)
        try:
            legend.set_in_layout(True)
        except AttributeError:
            pass
        for text in legend.get_texts():
            text.set_color(self._current_plot_text())
        frame = legend.get_frame()
        frame.set_visible(True)
        frame.set_alpha(0.92)
        frame.set_facecolor(self._current_plot_bg())
        frame.set_edgecolor(self._current_plot_grid())
        try:
            legend.set_draggable(True, use_blit=True, update="loc")
        except TypeError:
            try:
                legend.set_draggable(True, use_blit=True)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _format_hover_number(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if not np.isfinite(number):
            return "--"
        magnitude = abs(number)
        if magnitude >= 1e4 or (0.0 < magnitude < 1e-2):
            return f"{number:.2e}"
        return f"{number:.2f}"

    @staticmethod
    def _cursor_data_to_scalar(data) -> float | None:
        if data is None:
            return None
        try:
            values = np.asarray(data)
            if values.size == 0:
                return None
            if np.iscomplexobj(values):
                values = np.real(values)
            flat = np.asarray(values, dtype=float).ravel()
        except Exception:
            return None
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return None
        return float(finite[0])

    def _hover_z_from_axes(self, ax, event) -> float | None:
        artists = []
        artists.extend(reversed(getattr(ax, "collections", [])))
        artists.extend(reversed(getattr(ax, "images", [])))
        for artist in artists:
            getter = getattr(artist, "get_cursor_data", None)
            if getter is None:
                continue
            try:
                value = self._cursor_data_to_scalar(getter(event))
            except Exception:
                continue
            if value is not None:
                return value
        return None

    def _nearest_3d_hover_point(self, ax, event) -> tuple[float, float, float] | None:
        try:
            from mpl_toolkits.mplot3d import proj3d
        except Exception:
            return None
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return None

        view_key = (
            round(float(getattr(ax, "elev", 0.0)), 3),
            round(float(getattr(ax, "azim", 0.0)), 3),
            tuple(np.round(np.asarray(ax.get_xlim3d(), dtype=float), 6)),
            tuple(np.round(np.asarray(ax.get_ylim3d(), dtype=float), 6)),
            tuple(np.round(np.asarray(ax.get_zlim3d(), dtype=float), 6)),
        )
        cache = getattr(ax, "_grim_hover_cache", None)
        if not isinstance(cache, dict) or cache.get("view_key") != view_key:
            xyz_chunks: list[np.ndarray] = []
            xy_chunks: list[np.ndarray] = []
            for artist in getattr(ax, "collections", []):
                offsets3d = getattr(artist, "_offsets3d", None)
                if offsets3d is None:
                    continue
                try:
                    xs = np.asarray(offsets3d[0], dtype=float).ravel()
                    ys = np.asarray(offsets3d[1], dtype=float).ravel()
                    zs = np.asarray(offsets3d[2], dtype=float).ravel()
                except Exception:
                    continue
                finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                if not np.any(finite):
                    continue
                xs = xs[finite]
                ys = ys[finite]
                zs = zs[finite]
                x2d, y2d, _ = proj3d.proj_transform(xs, ys, zs, ax.get_proj())
                finite_2d = np.isfinite(x2d) & np.isfinite(y2d)
                if not np.any(finite_2d):
                    continue
                xs = xs[finite_2d]
                ys = ys[finite_2d]
                zs = zs[finite_2d]
                x2d = x2d[finite_2d]
                y2d = y2d[finite_2d]
                xy_pixels = ax.transData.transform(np.column_stack([x2d, y2d]))
                xyz_chunks.append(np.column_stack([xs, ys, zs]))
                xy_chunks.append(xy_pixels)
            if not xyz_chunks or not xy_chunks:
                return None
            cache = {
                "view_key": view_key,
                "xyz": np.vstack(xyz_chunks),
                "xy": np.vstack(xy_chunks),
            }
            setattr(ax, "_grim_hover_cache", cache)

        xy_pixels = cache.get("xy")
        xyz_points = cache.get("xyz")
        if xy_pixels is None or xyz_points is None or len(xy_pixels) == 0:
            return None

        distances = np.hypot(xy_pixels[:, 0] - event.x, xy_pixels[:, 1] - event.y)
        finite = np.isfinite(distances)
        if not np.any(finite):
            return None
        idx = int(np.argmin(np.where(finite, distances, np.inf)))
        if distances[idx] > 24.0:
            return None
        x_val, y_val, z_val = xyz_points[idx]
        return float(x_val), float(y_val), float(z_val)

    def _reset_hover_readout(self, hover_readout=None) -> None:
        # A leave event supersedes any pending hover update.
        timer = getattr(self, "_hover_timer", None)
        if timer is not None:
            timer.stop()
        self._pending_hover = None
        label = hover_readout or getattr(self, "hover_readout", None)
        if label is None:
            return
        label.setText("x: --   y: --")

    def _schedule_hover(self, event, hover_readout=None) -> None:
        self._pending_hover = (event, hover_readout)
        timer = getattr(self, "_hover_timer", None)
        if timer is None:
            self._flush_hover()
            return
        if not timer.isActive():
            timer.start()

    def _flush_hover(self) -> None:
        pending = self._pending_hover
        self._pending_hover = None
        if pending is None:
            return
        event, label = pending
        self._on_plot_hover(event, label)

    def _on_plot_hover(self, event, hover_readout=None) -> None:
        label = hover_readout or getattr(self, "hover_readout", None)
        if label is None:
            return
        ax = getattr(event, "inaxes", None)
        if ax is None:
            self._reset_hover_readout(label)
            return
        # z stays on the SAME line as x/y: a second line changes the label
        # height and visibly shifts the canvas whenever the cursor crosses
        # onto/off a z-valued artist.
        if ax.name == "3d":
            point = self._nearest_3d_hover_point(ax, event)
            if point is None:
                label.setText("x: --   y: --   z: --")
                return
            x_val, y_val, z_val = point
            label.setText(
                f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}"
                f"   z: {self._format_hover_number(z_val)}"
            )
            return

        x_val = getattr(event, "xdata", None)
        y_val = getattr(event, "ydata", None)
        if (
            x_val is None
            or y_val is None
            or not np.isfinite(x_val)
            or not np.isfinite(y_val)
        ):
            self._reset_hover_readout(label)
            return

        z_val = self._hover_z_from_axes(ax, event)
        if z_val is None:
            label.setText(
                f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}"
            )
            return
        label.setText(
            f"x: {self._format_hover_number(x_val)}   y: {self._format_hover_number(y_val)}"
            f"   z: {self._format_hover_number(z_val)}"
        )

    def _update_legend_visibility(self, checked: bool | None = None) -> None:
        """Create/show/hide legends on every current line-plot axis.

        ``checked`` is accepted directly from the toolbar signal. This avoids
        reading the other tab's toggle during a tab-context transition; plot
        renderers call the method without it and use the active toggle.
        """
        show = bool(self.chk_plot_legend.isChecked()) if checked is None else bool(checked)
        axes = self.plot_axes or [self.plot_ax]
        for ax in axes:
            legend = ax.get_legend()
            handles, labels = ax.get_legend_handles_labels()
            if not show or not handles:
                if legend is not None:
                    legend.set_visible(False)
                continue
            existing_labels = (
                [text.get_text() for text in legend.get_texts()]
                if legend is not None else None
            )
            if legend is None or existing_labels != labels:
                if legend is not None:
                    legend.remove()
                legend = ax.legend(handles, labels, **self._legend_kwargs())
            self._configure_legend(legend, ax)
        # A synchronous draw makes toolbar clicks deterministic even when the
        # Qt event queue is busy loading or replotting a large dataset.
        self.plot_canvas.draw()

    def _plot_azimuth_rect(self) -> None:
        azimuth_rect_mode.render(self)
        self._maybe_autoscale()

    def _plot_azimuth_polar(self) -> None:
        azimuth_polar_mode.render(self)
        self._maybe_autoscale()

    def _plot_frequency(self) -> None:
        frequency_mode.render(self)
        self._maybe_autoscale()

    def _plot_elevation_sweep(self) -> None:
        elevation_sweep_mode.render(self)
        self._maybe_autoscale()

    def _plot_isar_image(self) -> None:
        isar_mode.render(self)
        self._maybe_autoscale()

    def _plot_az_vs_range(self) -> None:
        az_vs_range_mode.render(self)
        self._maybe_autoscale()

    def _plot_waterfall(self) -> None:
        waterfall_mode.render(self)
        self._maybe_autoscale()

    def _plot_compare(self) -> None:
        compare_mode.render(self)
        self._maybe_autoscale()

    def _ensure_compare_axes(self):
        """Return (top_ax, res_ax) for the 2-panel compare layout, recreating if needed."""
        if (
            self.plot_axes is not None
            and len(self.plot_axes) == 1
            and len(self.plot_figure.axes) == 2
        ):
            return self.plot_figure.axes[0], self.plot_figure.axes[1]
        self._remove_colorbar()
        self.plot_figure.clear()
        top_ax, res_ax = self.plot_figure.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        )
        self.plot_ax = top_ax
        # plot_axes = [top_ax] so _apply_plot_limits only touches the top axis
        # (x-limits propagate automatically via sharex; residual y auto-scales)
        self.plot_axes = [top_ax]
        self.plot_figure.set_facecolor(self._current_plot_bg())
        return top_ax, res_ax
