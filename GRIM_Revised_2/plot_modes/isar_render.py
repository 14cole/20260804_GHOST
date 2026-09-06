"""GUI selection handling and rendering for computed ISAR images."""
from __future__ import annotations

import numpy as np
from . import common
from . import isar_mode as computation


def render(self) -> None:
    """GUI-thread half: validate the selections, capture everything the worker
    needs into a params dict, and hand off to the mixin's async submit. The
    finished computation comes back through `display_results`."""
    self.last_plot_mode = "isar_image"
    self._start_plot_render()
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return
    if self._preflight_plot_datasets([("Dataset", self.active_dataset)]) is None:
        return

    flip_x_widget = getattr(self, "chk_isar_flip_x", None)
    flip_x = bool(flip_x_widget.isChecked()) if flip_x_widget is not None else False
    flip_y_widget = getattr(self, "chk_isar_flip_y", None)
    flip_y = bool(flip_y_widget.isChecked()) if flip_y_widget is not None else False
    contract_assumptions: list[str] = []
    try:
        preflight_error = computation._isar_preflight_error(
            self.active_dataset,
            flip_x=flip_x,
            flip_y=flip_y,
            undeclared_out=contract_assumptions,
        )
    except (TypeError, ValueError) as exc:
        self.status.showMessage(f"ISAR blocked: invalid convention metadata: {exc}.")
        return
    if preflight_error is not None:
        self.status.showMessage(f"ISAR blocked: {preflight_error}.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    if not az_indices:
        self.status.showMessage("Select one or more azimuths/aspects to plot.")
        return

    # Optional aperture window (the scrub workflow): keep only selected
    # azimuths within ±width/2 of the center look angle, with 0/360 wrap.
    ap_widget = getattr(self, "chk_isar_aperture", None)
    az_center_deg: float | None = None
    if ap_widget is not None and ap_widget.isChecked():
        ap_center = float(self.spin_isar_ap_center.value())
        ap_width = float(self.spin_isar_ap_width.value())
        if not np.isfinite(ap_center) or not np.isfinite(ap_width) or ap_width <= 0.0:
            self.status.showMessage("ISAR aperture: center/width must be finite and width positive.")
            return
        az_arr = computation._angle_values_to_degrees(
            self.active_dataset,
            "azimuth",
            np.asarray(self.active_dataset.azimuths, dtype=float)[az_indices],
        )
        dist = np.abs(np.mod(az_arr - ap_center + 180.0, 360.0) - 180.0)
        keep = dist <= ap_width / 2.0 + 1e-9
        az_indices = [i for i, k in zip(az_indices, keep) if k]
        az_center_deg = ap_center
        if len(az_indices) < 2:
            self.status.showMessage(
                f"ISAR aperture {ap_center:g}° ± {ap_width / 2.0:g}° contains fewer "
                "than 2 selected azimuth samples."
            )
            return

    freq_indices = sorted(self._selected_indices(self.list_freq))
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return

    # Optional frequency sub-band: keep only selected frequencies inside
    # [min, max] so an engineer can sweep bands numerically instead of
    # re-selecting thousands of list entries per image.
    band_widget = getattr(self, "chk_isar_freq_band", None)
    if band_widget is not None and band_widget.isChecked():
        f_lo = float(self.spin_isar_freq_min.value())
        f_hi = float(self.spin_isar_freq_max.value())
        if f_hi <= f_lo:
            self.status.showMessage("ISAR freq band: max must exceed min.")
            return
        fvals = self.active_dataset.frequencies
        freq_indices = [i for i in freq_indices if f_lo <= float(fvals[i]) <= f_hi]
        if len(freq_indices) < 2:
            self.status.showMessage(
                f"ISAR freq band [{f_lo:g}, {f_hi:g}] contains fewer than 2 "
                "of the selected frequency samples."
            )
            return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for ISAR imaging.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return
    elevation_deg = float(
        computation._angle_values_to_degrees(
            self.active_dataset,
            "elevation",
            [self.active_dataset.elevations[elev_idx]],
        )[0]
    )
    if abs(float(np.cos(np.deg2rad(elevation_deg)))) < 1.0e-6:
        self.status.showMessage(
            "Azimuth ISAR is degenerate at elevation ±90° for the horizontal image plane."
        )
        return

    az_interp_widget = getattr(self, "chk_isar_az_interp", None)
    az_interp_on = bool(az_interp_widget.isChecked()) if az_interp_widget is not None else False
    az_target_deg: np.ndarray | None = None
    if az_interp_on:
        az_min = float(self.spin_isar_az_min.value())
        az_max = float(self.spin_isar_az_max.value())
        az_step = float(self.spin_isar_az_step.value())
        try:
            az_target_deg = computation._bounded_uniform_azimuth_grid(
                az_min,
                az_max,
                az_step,
                frequency_count=len(freq_indices),
            )
            az_target_deg = computation._validated_azimuth_target(
                self.active_dataset,
                az_indices,
                az_target_deg,
                aperture_center_degrees=az_center_deg,
            )
        except ValueError as exc:
            self.status.showMessage(f"ISAR azimuth interp blocked: {exc}.")
            return
        if az_target_deg.size < 2:
            self.status.showMessage("ISAR azimuth interp grid needs ≥2 samples.")
            return

    if az_interp_on or az_center_deg is not None:
        # Explicit resample collapses the multi-band view into a single image —
        # and circular aperture mode must keep a 0°/360° crossing coherent.
        bands: list[list[int]] = [az_indices] if len(az_indices) >= 2 else []
    else:
        bands = computation._split_into_bands(az_indices)
        bands = [b for b in bands if len(b) >= 2]
    if not bands:
        self.status.showMessage(
            "Each azimuth/aspect band needs at least 2 contiguous samples for ISAR imaging."
        )
        return
    if len(bands) > common.MAX_WATERFALL_PANELS:
        self.status.showMessage(
            f"ISAR blocked: selection would create {len(bands)} panels (limit "
            f"{common.MAX_WATERFALL_PANELS}). Select contiguous azimuth/aspect "
            "samples or enable one aperture/interpolation window."
        )
        return

    # The per-image reducer is not an aggregate memory limit: several allowed
    # panels can otherwise retain tens of millions of pixels at once. Estimate
    # each post-decimation image before launching the worker and apply one
    # figure-wide budget. Wide-aperture composites use their fixed 1024² grid.
    n_freq_fft_estimate = computation._next_fast_len(max(len(freq_indices), 256))
    total_display_cells = 0
    for band in bands:
        band_degrees = computation._angle_values_to_degrees(
            self.active_dataset,
            "azimuth",
            np.asarray(self.active_dataset.azimuths, dtype=float)[band],
        )
        if float(np.max(band_degrees) - np.min(band_degrees)) > computation._COMPOSITE_SPAN_DEG:
            total_display_cells += 1024 * 1024
            continue
        az_count = az_target_deg.size if az_target_deg is not None else len(band)
        n_az_fft_estimate = computation._next_fast_len(max(int(az_count), 256))
        total_display_cells += common.bounded_image_cell_count(
            n_az_fft_estimate, n_freq_fft_estimate
        )
    try:
        common.validate_aggregate_image_cells(
            total_display_cells,
            panel_count=len(bands),
            operation="ISAR image",
        )
    except ValueError as exc:
        self.status.showMessage(f"ISAR blocked: {exc}.")
        return

    freq_values_full = self.active_dataset.frequencies[freq_indices]
    freq_order = np.argsort(freq_values_full)
    freq_indices_sorted = [freq_indices[i] for i in freq_order]
    freq_values = freq_values_full[freq_order].astype(float)
    if np.any(np.diff(freq_values) <= 0) or not np.all(np.isfinite(freq_values)):
        self.status.showMessage(
            "Frequency samples must be finite and strictly increasing for ISAR imaging."
        )
        return

    freq_unit = str((self.active_dataset.units or {}).get("frequency", ""))
    freq_hz = freq_values * computation._unit_to_hz_scale(freq_unit)
    df = float(np.mean(np.diff(freq_hz)))
    if df <= 0.0:
        self.status.showMessage("ISAR imaging requires increasing frequency samples.")
        return

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = computation._length_unit(units_combo.currentText() if units_combo else "m")

    recon_combo = getattr(self, "combo_isar_recon", None)
    recon_text = recon_combo.currentText() if recon_combo is not None else "FFT"
    recon_lower = recon_text.lower()
    if recon_lower.startswith("sparse"):
        recon = "sparse"
    elif "accurate" in recon_lower or "cartesian" in recon_lower:
        recon = "accurate"
    else:
        recon = "fft"
    l1_strength_spin = getattr(self, "spin_isar_l1_strength", None)
    l1_iters_spin = getattr(self, "spin_isar_l1_iters", None)
    l1_strength = float(l1_strength_spin.value()) if l1_strength_spin is not None else 0.05
    l1_iters = int(l1_iters_spin.value()) if l1_iters_spin is not None else 300
    # Byte-based worker preflight. Display-cell caps alone do not account for
    # the complex source slice, FFT/FISTA temporaries, or full-resolution
    # complex results retained for Export ISAR Result.
    estimated_resident = 0
    estimated_peak = 0
    for band in bands:
        band_degrees = computation._angle_values_to_degrees(
            self.active_dataset,
            "azimuth",
            np.asarray(self.active_dataset.azimuths, dtype=float)[band],
        )
        if az_center_deg is not None:
            band_degrees = computation._unwrap_degrees(band_degrees, az_center_deg)
        band_span = float(np.max(band_degrees) - np.min(band_degrees))
        if band_span > computation._COMPOSITE_SPAN_DEG:
            look_count = max(2, int(np.ceil(band_span / computation._COMPOSITE_SUB_DEG)))
            sublook_count = max(2, int(np.ceil(len(band) / look_count)))
            working = computation._estimate_band_working_set_bytes(
                sublook_count,
                len(freq_indices_sorted),
                reconstruction=recon,
                retain_complex=False,
            ) + computation._COMPOSITE_SCRATCH_BYTES + (
                computation._COMPOSITE_GRID_SIDE**2 * np.dtype(np.float32).itemsize
            )
            retained = computation._COMPOSITE_GRID_SIDE**2 * np.dtype(np.float32).itemsize
        else:
            az_count = int(az_target_deg.size) if az_target_deg is not None else len(band)
            working = computation._estimate_band_working_set_bytes(
                az_count,
                len(freq_indices_sorted),
                reconstruction=recon,
                retain_complex=True,
            )
            n_az_fft = computation._next_fast_len(max(az_count, 256))
            n_freq_fft = computation._next_fast_len(max(len(freq_indices_sorted), 256))
            image_cells = n_az_fft * n_freq_fft
            display_cells = common.bounded_image_cell_count(n_az_fft, n_freq_fft)
            retained = 8 * image_cells + 4 * display_cells
        estimated_peak = max(estimated_peak, estimated_resident + working)
        estimated_resident += retained
    try:
        computation._validate_isar_working_set(
            estimated_peak,
            operation="ISAR selection",
        )
    except ValueError as exc:
        self.status.showMessage(f"ISAR blocked: {exc}.")
        return

    self._isar_submit({
        "dataset": self.active_dataset,
        # Identity token: if the active figure changed while computing (user
        # switched tabs), the finished result is dropped instead of being
        # painted onto whatever tab is now in front.
        "figure_token": self.plot_figure,
        "render_generation": getattr(self, "_plot_render_generation", 0),
        "bands": bands,
        "freq_indices_sorted": freq_indices_sorted,
        "elev_idx": elev_idx,
        "elevation_deg": elevation_deg,
        "pol_idx": pol_idx,
        "freq_hz": freq_hz,
        "df": df,
        "unit_scale": unit_scale,
        "unit_name": unit_name,
        "az_target_deg": az_target_deg,
        "az_center_deg": az_center_deg,
        "window_name": str(self.combo_isar_window.currentText()),
        "recon": recon,
        "l1_strength": l1_strength,
        "l1_iters": l1_iters,
        "flip_x": flip_x,
        "flip_y": flip_y,
        # The ISAR tab exposes Export ISAR Result, so retain the coherent
        # worker result. Wide max-look composites explicitly report that no
        # physically meaningful complex image exists.
        "retain_complex": True,
        "isar_contract_assumptions": contract_assumptions,
    })

def display_results(self, params: dict, band_results: list, elapsed: float) -> None:
    """GUI-thread half two: draw the computed band images. Runs from the
    mixin's worker-finished slot; everything Qt/matplotlib happens here."""
    dataset = params["dataset"]
    unit_name = params["unit_name"]
    az_target_deg = params["az_target_deg"]

    # Convert coherent magnitude to generic image intensity. Image formation
    # does not guarantee an absolute square-metre normalization, so neither
    # branch claims dBsm/dBke. (Magnitude was already max-pool decimated on the
    # worker; block-max commutes with squaring and log.)
    for br in band_results:
        magnitude = np.asarray(br["magnitude"], dtype=np.float32)
        intensity = np.empty_like(magnitude, dtype=np.float32)
        np.multiply(magnitude, magnitude, out=intensity)
        if self._plot_scale_is_linear():
            br["isar_display"] = intensity
        else:
            np.maximum(intensity, np.float32(1.0e-12), out=intensity)
            np.log10(intensity, out=intensity)
            intensity *= np.float32(10.0)
            br["isar_display"] = intensity

    n_bands = len(band_results)

    self._remove_colorbar()
    self.plot_figure.clear()
    if n_bands == 1:
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        active_axes = [self.plot_ax]
    else:
        ax_array = self.plot_figure.subplots(1, n_bands, sharey=True)
        if not isinstance(ax_array, np.ndarray):
            ax_array = np.array([ax_array])
        active_axes = list(ax_array.ravel())
        self.plot_axes = active_axes
        self.plot_ax = active_axes[0]
    self._style_plot_axes()

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax
    shared_scale = bool(self.chk_colorbar_shared.isChecked())
    shared_limits = (
        common.finite_data_limits(br["isar_display"] for br in band_results)
        if shared_scale and not use_clamp
        else None
    )
    plot_vmin = zmin if use_clamp else (
        shared_limits[0] if shared_limits is not None else None
    )
    plot_vmax = zmax if use_clamp else (
        shared_limits[1] if shared_limits is not None else None
    )

    square_widget = getattr(self, "chk_isar_square", None)
    square_aspect = bool(square_widget.isChecked()) if square_widget is not None else False

    last_mesh = None
    self._isar_meshes = []
    overall_x_min = float("inf")
    overall_x_max = float("-inf")
    overall_y_min = float("inf")
    overall_y_max = float("-inf")
    for ax, br in zip(active_axes, band_results):
        x_min = float(br["x_range"].min())
        x_max = float(br["x_range"].max())
        y_min = float(br["y_range"].min())
        y_max = float(br["y_range"].max())
        # imshow on a uniform grid is several times faster than pcolormesh
        # for big arrays (1601-frequency datasets feel laggy with pcolormesh).
        mesh = ax.imshow(
            br["isar_display"].T,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=plot_vmin,
            vmax=plot_vmax,
        )
        if square_aspect:
            # adjustable="datalim" keeps the plot box at its current size and
            # *expands* the visible data limits to maintain 1:1 cross-range /
            # range scale. The opposite ("box") shrinks the box, which is
            # what we don't want when range and cross-range extents differ.
            ax.set_aspect("equal", adjustable="datalim")
        last_mesh = mesh
        self._isar_meshes.append(mesh)
        overall_x_min = min(overall_x_min, x_min)
        overall_x_max = max(overall_x_max, x_max)
        overall_y_min = min(overall_y_min, y_min)
        overall_y_max = max(overall_y_max, y_max)
        if n_bands > 1:
            ax.set_title(
                f"{float(br['az_values'][0]):g}°–{float(br['az_values'][-1]):g}°",
                color=self._current_plot_text(),
            )

    elev_value = float(params["elevation_deg"])
    elev_name = common.angular_axis_name(dataset, "elevation")
    pol_value = dataset.polarizations[params["pol_idx"]]
    if params.get("recon") == "sparse":
        recon_label = " | Sparse L1 (Experimental)"
    elif params.get("recon") == "accurate":
        recon_label = " | Cartesian PFA"
    else:
        recon_label = " | Fast PFA"
    composite_subs = max((br.get("composite", 0) for br in band_results), default=0)
    if composite_subs:
        recon_label += f" | Wide-Aperture Composite ({composite_subs} looks)"
    fig_title = (
        f"ISAR Image | {elev_name} {elev_value:g} deg | Pol {pol_value}{recon_label}"
    )
    if n_bands > 1:
        self.plot_figure.suptitle(fig_title, color=self._current_plot_text())
    else:
        active_axes[0].set_title(fig_title, color=self._current_plot_text())

    # Composite images live in the body frame (θ=0 radar frame), not a single
    # look's cross-range/range frame — label accordingly.
    horizontal_projection = abs(float(params.get("elevation_deg", 0.0))) > 1.0e-9
    if composite_subs:
        x_label = f"Cross-Range at 0° ({unit_name})"
        y_label = f"Down-Range at 0° ({unit_name})"
    elif horizontal_projection:
        x_label = f"Horizontal Cross-Range ({unit_name})"
        y_label = f"Horizontal Range ({unit_name})"
    else:
        x_label = f"Cross-Range ({unit_name})"
        y_label = f"Range ({unit_name})"
    for ax in active_axes:
        ax.set_xlabel(x_label)
    active_axes[0].set_ylabel(y_label)

    if self.chk_colorbar.isChecked() and last_mesh is not None:
        if shared_scale:
            self.plot_colorbars = [
                self.plot_figure.colorbar(last_mesh, ax=active_axes)
            ]
        else:
            self.plot_colorbars = [
                self.plot_figure.colorbar(mesh, ax=ax)
                for ax, mesh in zip(active_axes, self._isar_meshes)
            ]
        for colorbar in self.plot_colorbars:
            self._apply_colorbar_ticks(colorbar)
            if self._plot_scale_is_linear():
                colorbar.set_label(
                    "Image Intensity (linear)", color=self._current_plot_text()
                )
            else:
                colorbar.set_label(
                    "Image Intensity (dB)", color=self._current_plot_text()
                )
            colorbar.ax.tick_params(colors=self._current_plot_text())
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self._current_plot_text())

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(overall_x_min)
    self.spin_plot_xmax.setValue(overall_x_max)
    self.spin_plot_ymin.setValue(overall_y_min)
    self.spin_plot_ymax.setValue(overall_y_max)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    # Auto-fit the z (dB) spinboxes only on the *first* render of a new
    # dataset. Re-running this on every render — which the per-keystroke
    # `valueChanged` signal triggers — would clobber the user's typing
    # whenever zmin transiently exceeds zmax mid-keystroke.
    state_key = id(dataset)
    last_state = getattr(self, "_isar_last_autofit_state", None)
    autofit_limits = None
    if state_key != last_state:
        img_min = float("inf")
        img_max = float("-inf")
        for br in band_results:
            finite = br["isar_display"][np.isfinite(br["isar_display"])]
            if finite.size:
                img_min = min(img_min, float(finite.min()))
                img_max = max(img_max, float(finite.max()))
        if np.isfinite(img_min) and np.isfinite(img_max) and img_max > img_min:
            cur_zmin = self.spin_plot_zmin.value()
            cur_zmax = self.spin_plot_zmax.value()
            clamp_active = cur_zmin < cur_zmax
            clamp_dead = clamp_active and (cur_zmax < img_min or cur_zmin > img_max)
            if not clamp_active or clamp_dead:
                display_floor = img_max - 60.0 if not self._plot_scale_is_linear() else img_min
                self.spin_plot_zmin.blockSignals(True)
                self.spin_plot_zmax.blockSignals(True)
                self.spin_plot_zmin.setValue(display_floor)
                self.spin_plot_zmax.setValue(img_max)
                self.spin_plot_zmin.blockSignals(False)
                self.spin_plot_zmax.blockSignals(False)
                autofit_limits = (display_floor, img_max)
        self._isar_last_autofit_state = state_key

    # Signals are blocked to avoid recursive rendering, so apply a new first-
    # render auto-fit directly. This keeps the visible canvas, controls, export,
    # and frozen headless recipe on the same global normalization.
    if autofit_limits is not None:
        for mesh in self._isar_meshes:
            mesh.set_clim(*autofit_limits)
        for colorbar in self.plot_colorbars or []:
            self._apply_colorbar_ticks(colorbar)

    self._apply_plot_limits()

    # Surface any resampling that happened so the user knows their input
    # wasn't on a uniform grid. The number is the relative spread of native
    # spacings ((max-min)/median); anything > ~0.001 was actually resampled.
    az_max = max(br.get("az_nonuniformity", 0.0) for br in band_results)
    fr_max = max(br.get("freq_nonuniformity", 0.0) for br in band_results)
    if params.get("recon") == "sparse":
        mode_label = "Sparse L1 (Experimental)"
    elif params.get("recon") == "accurate":
        mode_label = "Cartesian PFA"
    else:
        mode_label = "Fast PFA"
    parts = [f"ISAR image updated in {elapsed:.2f}s ({mode_label})"]
    if n_bands > 1:
        parts.append(f" ({n_bands} bands)")
    notes = []
    contract_assumptions = tuple(params.get("isar_contract_assumptions", ()))
    if contract_assumptions:
        notes.append(
            "undeclared ISAR conventions user-assumed: "
            + ", ".join(str(value) for value in contract_assumptions)
        )
    if composite_subs:
        notes.append(
            f"wide aperture — composited {composite_subs} × ~{computation._COMPOSITE_SUB_DEG:g}° looks "
            "into the 0°-azimuth body frame"
        )
    if composite_subs and az_target_deg is not None:
        notes.append("az interp grid ignored (composite mode)")
    elif az_target_deg is not None:
        notes.append(
            f"az interp {az_target_deg[0]:g}→{az_target_deg[-1]:g}° step "
            f"{float(np.mean(np.diff(az_target_deg))):g}° ({az_target_deg.size} samples)"
        )
    elif az_max >= 1e-3:
        notes.append(f"resampled azimuth (Δ-spread {az_max*100:.1f}%)")
    if fr_max >= 1e-3:
        notes.append(f"resampled frequency (Δ-spread {fr_max*100:.1f}%)")
    az_gap_count = max((br.get("az_gap_count", 0) for br in band_results), default=0)
    fr_gap_count = max((br.get("freq_gap_count", 0) for br in band_results), default=0)
    if az_gap_count:
        az_gap_fraction = max(
            br.get("az_gap_fraction", 0.0) for br in band_results
        )
        largest_az_gap = max(br.get("az_largest_gap", 0.0) for br in band_results)
        notes.append(
            f"{az_gap_count} missing azimuth sector(s) zero-weighted, not "
            f"interpolated (largest {largest_az_gap:g}°, "
            f"{az_gap_fraction*100:.1f}% of uniform grid)"
        )
    if fr_gap_count:
        fr_gap_fraction = max(
            br.get("freq_gap_fraction", 0.0) for br in band_results
        )
        largest_fr_gap = max(br.get("freq_largest_gap", 0.0) for br in band_results)
        notes.append(
            f"{fr_gap_count} missing frequency band(s) zero-weighted, not "
            f"interpolated (largest {largest_fr_gap/1.0e9:g} GHz, "
            f"{fr_gap_fraction*100:.1f}% of uniform grid)"
        )
    coverage = min(br.get("phase_coverage", 1.0) for br in band_results)
    if coverage < 0.9995:
        notes.append(f"weighted phase coverage {coverage*100:.1f}%")
    if any(br.get("preprocess_cache_hit", False) for br in band_results):
        notes.append("reused gridded phase history")
    if any(br.get("display_decimated", False) for br in band_results):
        notes.append(
            "peak-preserving display decimation applied; narrow the aperture/band "
            "for full display resolution"
        )
    sparse_results = [
        br for br in band_results if br.get("sparse_iterations") is not None
    ]
    if sparse_results:
        converged_count = sum(
            bool(br.get("sparse_converged", False)) for br in sparse_results
        )
        max_iterations = max(
            int(br.get("sparse_iterations", 0)) for br in sparse_results
        )
        max_gap = max(
            float(br.get("sparse_relative_duality_gap", float("inf")))
            for br in sparse_results
        )
        if converged_count == len(sparse_results):
            notes.append(
                f"sparse solver converged ({max_iterations} iterations, "
                f"relative gap ≤{max_gap:.2g})"
            )
        else:
            notes.append(
                f"SPARSE NOT CONVERGED: {converged_count}/{len(sparse_results)} "
                f"images certified at {max_iterations} iterations "
                f"(worst relative gap {max_gap:.2g})"
            )
        max_relative_residual = max(
            float(br.get("sparse_output_relative_residual_norm", float("inf")))
            for br in sparse_results
        )
        notes.append(
            f"sparse weighted residual ≤{max_relative_residual:.3g} relative"
        )
    if band_results:
        sampling = band_results[0].get("sampling", {})
        if sampling:
            notes.append(
                f"nominal resolution Δx≈{sampling['cross_resolution']:.3g} {unit_name}, "
                f"Δr≈{sampling['range_resolution']:.3g} {unit_name}; "
                f"unambiguous |x|≤{sampling['cross_half_extent']:.3g}, "
                f"|r|≤{sampling['range_half_extent']:.3g} {unit_name}"
            )
    if notes:
        parts.append(" — " + ", ".join(notes))
    self._show_plot_status("".join(parts))
