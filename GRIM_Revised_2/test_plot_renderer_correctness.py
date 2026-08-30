"""Focused physical-correctness and display-bound regressions for Plotting."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from grim_dataset import GRIM_GC_CONVENTION, RcsGrid
from grim_cut_plot_mixin import (
    PlotOpsMixin,
    _selected_polarization_axis_availability,
)
from grim_python import plot_datasets
from plot_modes import (
    azimuth_polar_mode,
    azimuth_rect_mode,
    common,
    compare_mode,
    elevation_sweep_mode,
    frequency_mode,
    isar_mode,
    waterfall_mode,
)
from plot_modes.az_vs_range_mode import _range_display_values
from plot_modes.isar_mode import form_isar
from plot_modes.isar_mode import _unit_to_hz_scale


def _grid(
    *,
    angle_unit="deg",
    frequency_unit="GHz",
    coordinate_system="conic",
    quantity="sigma_3d",
    log_unit="dBsm",
    phase_reference="",
):
    azimuths = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    elevations = np.asarray([0.0])
    frequencies = np.linspace(9.0, 10.0, 8)
    if angle_unit == "rad":
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)
    if frequency_unit == "Hz":
        frequencies = frequencies * 1.0e9
    shape = (azimuths.size, elevations.size, frequencies.size, 1)
    phase = np.linspace(-0.3, 0.3, np.prod(shape)).reshape(shape)
    units = {
        "azimuth": angle_unit,
        "elevation": angle_unit,
        "frequency": frequency_unit,
        "angular_coordinate_system": coordinate_system,
        "rcs_linear_quantity": quantity,
        "rcs_log_unit": log_unit,
    }
    if coordinate_system == "great_circle":
        units["great_circle_coordinate_convention"] = GRIM_GC_CONVENTION
    return RcsGrid(
        azimuths,
        elevations,
        frequencies,
        ["HH"],
        rcs_power=np.ones(shape),
        rcs_phase=phase,
        units=units,
        extra={"phase_reference": phase_reference} if phase_reference else {},
    )


class PlotRendererHelperTests(unittest.TestCase):
    def test_polarization_availability_scans_bounded_basic_slices(self):
        class RejectAdvancedPolarizationIndex(np.ndarray):
            def __getitem__(self, key):
                if (
                    isinstance(key, tuple)
                    and len(key) == 4
                    and isinstance(key[3], (list, tuple, np.ndarray))
                ):
                    raise AssertionError(
                        "availability must not copy a full advanced-indexed grid"
                    )
                return super().__getitem__(key)

        shape = (4, 3, 2, 2)
        power = np.full(shape, np.nan)
        phase = np.full(shape, np.nan)
        power[0, 1, 0, 0] = 1.0
        phase[0, 1, 0, 0] = 0.0
        power[3, 2, 1, 1] = 2.0
        # This sample is available for magnitude but not coherent phase.
        grid = RcsGrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            ["HH", "VV"],
            rcs_power=power,
            rcs_phase=phase,
        )
        grid.rcs_power = grid.rcs_power.view(RejectAdvancedPolarizationIndex)
        grid.rcs_phase = grid.rcs_phase.view(RejectAdvancedPolarizationIndex)

        magnitude_masks = _selected_polarization_axis_availability(
            grid, [0, 1], require_phase=False, maximum_work_bytes=1
        )
        phase_masks = _selected_polarization_axis_availability(
            grid, [0, 1], require_phase=True, maximum_work_bytes=1
        )

        np.testing.assert_array_equal(magnitude_masks[0], [True, True])
        np.testing.assert_array_equal(magnitude_masks[1], [False, True, True])
        np.testing.assert_array_equal(magnitude_masks[2], [True, False, False, True])
        np.testing.assert_array_equal(phase_masks[0], [True, False])
        np.testing.assert_array_equal(phase_masks[1], [False, True, False])
        np.testing.assert_array_equal(phase_masks[2], [True, False, False, False])

    def test_compatible_frequency_and_angle_units_convert_to_reference(self):
        reference = _grid()
        other = _grid(angle_unit="rad", frequency_unit="Hz")
        common.validate_plot_datasets(
            [("reference", reference), ("other", other)], phase=False, linear=False
        )
        selected_frequency, freq_tolerance = common.selection_for_dataset(
            reference, other, "frequency", [9.0, 10.0]
        )
        selected_azimuth, angle_tolerance = common.selection_for_dataset(
            reference, other, "azimuth", [-2.0, 2.0]
        )
        np.testing.assert_allclose(selected_frequency, [9.0e9, 10.0e9])
        np.testing.assert_allclose(selected_azimuth, np.deg2rad([-2.0, 2.0]))
        self.assertAlmostEqual(freq_tolerance, 1.0e3)
        self.assertAlmostEqual(angle_tolerance, np.deg2rad(1.0e-6))

        reverse_frequency, reverse_tolerance = common.selection_for_dataset(
            other, reference, "frequency", [10.0e9]
        )
        np.testing.assert_allclose(reverse_frequency, [10.0])
        self.assertAlmostEqual(reverse_tolerance, 1.0e-6)
        np.testing.assert_allclose(
            common.values_for_display(reference, other, "frequency", selected_frequency),
            [9.0, 10.0],
        )

    def test_mixed_physical_quantities_are_blocked(self):
        sigma = _grid()
        ratio = _grid(quantity="power_ratio", log_unit="dB")
        with self.assertRaisesRegex(ValueError, "mixed physical quantities"):
            common.validate_plot_datasets(
                [("sigma", sigma), ("ratio", ratio)], phase=False, linear=True
            )

    def test_phase_only_overlay_allows_different_magnitude_quantities(self):
        sigma_3d = _grid(quantity="sigma_3d", log_unit="dBsm")
        sigma_2d = _grid(quantity="sigma_2d", log_unit="dBke")
        common.validate_plot_datasets(
            [("3-D", sigma_3d), ("2-D", sigma_2d)], phase=True, linear=False
        )
        with self.assertRaisesRegex(ValueError, "mixed physical quantities"):
            common.validate_plot_datasets(
                [("3-D", sigma_3d), ("2-D", sigma_2d)],
                phase=False,
                linear=False,
            )

    def test_mixed_coordinate_charts_are_blocked_and_gc_labels_are_dynamic(self):
        conic = _grid()
        great_circle = _grid(coordinate_system="great_circle")
        with self.assertRaisesRegex(ValueError, "mixed angular coordinate systems"):
            common.validate_plot_datasets(
                [("conic", conic), ("gc", great_circle)], phase=False, linear=False
            )
        self.assertEqual(common.axis_label(great_circle, "azimuth"), "Aspect (deg)")
        self.assertEqual(common.axis_label(great_circle, "elevation"), "Pitch (deg)")

    def test_explicit_phase_metadata_disagreement_is_blocked(self):
        left = _grid(phase_reference="nose")
        right = _grid(phase_reference="scene-center")
        with self.assertRaisesRegex(ValueError, "phase references"):
            common.validate_plot_datasets(
                [("left", left), ("right", right)], phase=True, linear=False
            )
        left.units["time_convention"] = "exp(+jwt)"
        right.units["time_convention"] = "exp(-jwt)"
        right.extra["phase_reference"] = "nose"
        with self.assertRaisesRegex(ValueError, "time conventions"):
            common.validate_plot_datasets(
                [("left", left), ("right", right)], phase=True, linear=False
            )

    def test_phase_p50_and_envelope_respect_wrap_seam(self):
        samples = np.asarray([[179.0, -179.0], [-179.0, 179.0]])
        median = common.circular_median_degrees(samples, axis=0)
        np.testing.assert_allclose(np.abs(median), [180.0, 180.0], atol=1.0e-12)
        antipodal = common.circular_median_degrees([[90.0], [-90.0]], axis=0)
        self.assertTrue(np.isfinite(antipodal[0]))

        envelope = common.StreamingEnvelope(phase_degrees=True)
        envelope.update([179.0, -179.0])
        envelope.update([-179.0, 179.0])
        lower, upper, count = envelope.result()
        np.testing.assert_allclose(upper - lower, [2.0, 2.0])
        np.testing.assert_array_equal(count, [2, 2])

    def test_gui_phase_display_honors_dataset_wrap_interval(self):
        dataset = _grid().wrap_phase("0_360")
        renderer = object.__new__(PlotOpsMixin)

        displayed = renderer._phase_display_degrees(
            dataset, np.exp(1j * np.deg2rad([-10.0, 10.0]))
        )

        np.testing.assert_allclose(displayed, [350.0, 10.0], atol=1.0e-12)

    def test_phase_residual_uses_shortest_signed_difference(self):
        residual = common.wrap_phase_degrees(
            np.asarray([179.0, -179.0]) - np.asarray([-179.0, 179.0])
        )
        np.testing.assert_allclose(residual, [-2.0, 2.0])

    def test_compare_sweep_priority_is_explicit(self):
        self.assertEqual(
            compare_mode._determine_sweep_axis([0, 1], [0, 1], [1, 2]),
            "azimuth",
        )
        self.assertIsNone(compare_mode._determine_sweep_axis([0], [0], [1]))

    def test_decimation_caps_lines_and_preserves_narrow_peak(self):
        x = np.arange(100_000, dtype=float)
        y = np.zeros_like(x)
        y[55_555] = 123.0
        x_display, y_display, changed = common.decimate_line(x, y, max_points=2_000)
        self.assertTrue(changed)
        self.assertLessEqual(x_display.size, 2_000)
        self.assertEqual(float(np.max(y_display)), 123.0)
        self.assertEqual(float(x_display[0]), 0.0)
        self.assertEqual(float(x_display[-1]), 99_999.0)

    def test_line_and_envelope_decimation_preserve_missing_data_breaks(self):
        x = np.arange(100, dtype=float)
        line = np.ones_like(x)
        line[51] = np.nan
        _x_out, line_out, changed = common.decimate_line(x, line, max_points=20)
        self.assertTrue(changed)
        self.assertTrue(np.any(np.isnan(line_out)))

        lower = np.zeros_like(x)
        upper = np.ones_like(x)
        lower[51] = np.nan
        upper[51] = np.nan
        _x_out, lower_out, upper_out, _count, changed = common.decimate_envelope(
            x, lower, upper, max_points=30
        )
        self.assertTrue(changed)
        self.assertTrue(np.any(~(np.isfinite(lower_out) & np.isfinite(upper_out))))

    def test_image_and_tick_caps_are_bounded(self):
        x = np.arange(4_000)
        y = np.arange(3_000)
        image = np.zeros((x.size, y.size), dtype=np.uint8)
        image[2_137, 1_733] = 251
        x_out, y_out, image_out, changed = common.decimate_image(x, y, image)
        self.assertTrue(changed)
        self.assertLessEqual(image_out.size, common.MAX_IMAGE_CELLS)
        self.assertEqual(image_out.shape, (x_out.size, y_out.size))
        self.assertEqual(int(np.max(image_out)), 251)
        self.assertIsNone(common.bounded_ticks(0.0, 1.0, 1.0e-6))
        self.assertEqual(common.bounded_ticks(0.0, 1.0, 0.5).tolist(), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            common.bounded_ticks(0.0, 1.0, 0.6), [0.0, 0.6]
        )
        np.testing.assert_allclose(
            common.bounded_ticks(1.0, 0.0, 0.6), [1.0, 0.4]
        )

    def test_image_decimation_preserves_log_scale_peak_and_missing_blocks(self):
        x = np.arange(9)
        y = np.arange(8)
        image = np.full((x.size, y.size), -100.0)
        image[4, 5] = -1.25
        image[0:3, 0:4] = np.nan
        _x_out, _y_out, image_out, changed = common.decimate_image(
            x, y, image, max_side=4, max_cells=12
        )
        self.assertTrue(changed)
        self.assertEqual(float(np.nanmax(image_out)), -1.25)
        self.assertTrue(np.isnan(image_out[0, 0]))

    def test_image_budget_handles_singleton_dimensions_and_hard_cell_cap(self):
        x = np.arange(1)
        y = np.arange(20)
        image = np.arange(20, dtype=float).reshape(1, 20)
        x_out, y_out, image_out, changed = common.decimate_image(
            x, y, image, max_side=20, max_cells=3
        )
        self.assertTrue(changed)
        self.assertEqual(x_out.size, 1)
        self.assertLessEqual(image_out.size, 3)
        self.assertEqual(image_out.shape, (x_out.size, y_out.size))
        self.assertEqual(float(np.max(image_out)), 19.0)

    def test_plot_workload_and_aggregate_image_preflights_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "working slice"):
            common.validate_synchronous_plot_workload(
                operation="test",
                peak_slice_cells=11,
                total_cells=11,
                max_slice_cells=10,
                max_total_cells=100,
            )
        with self.assertRaisesRegex(ValueError, "across 3 panels"):
            common.validate_aggregate_image_cells(
                31, panel_count=3, operation="test", max_cells=30
            )

    def test_common_axis_matching_never_reuses_a_sample(self):
        left, right = common.common_axis_indices(
            [0.0, 0.9e-6], [0.5e-6], tolerance=1.0e-6
        )
        self.assertEqual(left.size, 1)
        self.assertEqual(right.size, 1)

    def test_range_display_is_generic_intensity_in_both_scales(self):
        grid = _grid()
        magnitude = np.asarray([1.0, 0.1])
        np.testing.assert_allclose(
            _range_display_values(grid, magnitude, linear=True), [1.0, 0.01]
        )
        np.testing.assert_allclose(
            _range_display_values(grid, magnitude, linear=False), [0.0, -20.0]
        )

    def test_degree_and_radian_grids_form_equivalent_isar_geometry(self):
        degree = _grid()
        radian = _grid(angle_unit="rad")
        degree_result, _ = form_isar(
            degree, reconstruction="fast", legacy_metadata_attested=True
        )
        radian_result, _ = form_isar(
            radian, reconstruction="fast", legacy_metadata_attested=True
        )
        self.assertEqual(len(degree_result), len(radian_result))
        for left, right in zip(degree_result, radian_result):
            np.testing.assert_allclose(left["az_values"], right["az_values"], atol=1.0e-12)
            np.testing.assert_allclose(left["x_range"], right["x_range"], rtol=1.0e-12)
            np.testing.assert_allclose(left["y_range"], right["y_range"], rtol=1.0e-12)
            np.testing.assert_allclose(left["magnitude"], right["magnitude"], rtol=1.0e-6)

    def test_isar_rejects_unknown_frequency_units_instead_of_guessing_ghz(self):
        with self.assertRaisesRegex(ValueError, "unsupported frequency unit"):
            _unit_to_hz_scale("cycles")

    def test_isar_interpolation_grid_is_preflighted_before_allocation(self):
        exact = isar_mode._bounded_uniform_azimuth_grid(
            0.0, 1.0, 0.5, frequency_count=8
        )
        np.testing.assert_allclose(exact, [0.0, 0.5, 1.0])
        nondivisible = isar_mode._bounded_uniform_azimuth_grid(
            0.0, 1.0, 0.6, frequency_count=8
        )
        np.testing.assert_allclose(nondivisible, [0.0, 0.6])
        with self.assertRaisesRegex(ValueError, "safety limit"):
            isar_mode._bounded_uniform_azimuth_grid(
                -3600.0, 3600.0, 1.0e-4, frequency_count=2
            )
        with self.assertRaisesRegex(ValueError, "complex cells"):
            isar_mode._bounded_uniform_azimuth_grid(
                0.0, 360.0, 0.01, frequency_count=1_000
            )

    def test_headless_rect_replay_normalizes_mixed_axis_units(self):
        reference = _grid()
        converted = _grid(angle_unit="rad", frequency_unit="Hz")
        figure = plot_datasets(
            [("reference", reference), ("converted", converted)],
            mode="azimuth_rect",
            azimuths=reference.azimuths,
            elevations=[0.0],
            frequencies=[9.0],
            polarization="HH",
        )
        axis = figure.axes[0]
        self.assertEqual(axis.get_xlabel(), "Azimuth (deg)")
        self.assertEqual(len(axis.lines), 2)
        np.testing.assert_allclose(
            axis.lines[0].get_xdata(), axis.lines[1].get_xdata(), atol=1.0e-12
        )
        self.assertIn("9 GHz", axis.lines[1].get_label())

    def test_headless_polar_replay_does_not_double_convert_radians(self):
        radian = _grid(angle_unit="rad", frequency_unit="Hz")
        figure = plot_datasets(
            [("radian", radian)],
            mode="azimuth_polar",
            azimuths=radian.azimuths,
            elevations=[0.0],
            frequencies=[9.0e9],
            polarization="HH",
        )
        axis = figure.axes[0]
        self.assertEqual(axis.get_xlabel(), "Azimuth (rad)")
        np.testing.assert_allclose(
            axis.lines[0].get_xdata(), radian.azimuths, atol=1.0e-12
        )

    def test_headless_frequency_phase_p50_is_circular(self):
        dataset = _grid()
        phase_degrees = np.asarray([179.0, -179.0, 179.0, -179.0, 179.0])
        dataset.rcs_phase[:, 0, :, 0] = np.deg2rad(phase_degrees[:, None])
        figure = plot_datasets(
            [("phase", dataset)],
            mode="frequency",
            azimuths=dataset.azimuths,
            elevations=[0.0],
            frequencies=dataset.frequencies,
            polarization="HH",
            phase=True,
        )
        axis = figure.axes[0]
        self.assertEqual(axis.get_xlabel(), "Frequency (GHz)")
        self.assertEqual(axis.get_ylabel(), "Phase P50 (deg)")
        self.assertTrue(np.all(np.abs(axis.lines[0].get_ydata()) > 170.0))

    def test_headless_phase_plots_never_reconstruct_the_whole_complex_grid(self):
        dataset = _grid()
        common_options = {
            "datasets": [("phase", dataset)],
            "azimuths": dataset.azimuths,
            "elevations": dataset.elevations,
            "frequencies": dataset.frequencies,
            "polarization": "HH",
            "phase": True,
        }

        with mock.patch.object(
            RcsGrid,
            "rcs",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("whole-grid complex reconstruction"),
        ):
            for mode in (
                "azimuth_rect",
                "azimuth_polar",
                "frequency",
                "elevation_sweep",
            ):
                with self.subTest(mode=mode):
                    figure = plot_datasets(mode=mode, **common_options)
                    figure.clear()


class _CanvasCounter:
    def __init__(self):
        self.idle_calls = 0
        self.draw_calls = 0

    def draw_idle(self):
        self.idle_calls += 1

    def draw(self):
        self.draw_calls += 1


class _ArtistMixin(PlotOpsMixin):
    def _effective_colormap(self):
        return "plasma"

    def _configure_legend(self, legend, ax=None):
        pass

    def _legend_kwargs(self):
        return {}


class _Checked:
    def __init__(self, checked=True):
        self.checked = checked

    def isChecked(self):
        return self.checked


class _Combo:
    def __init__(self, data):
        self.data = data

    def currentData(self):
        return self.data


class _Spin:
    def __init__(self, value=0.0):
        self._value = float(value)

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = float(value)

    def blockSignals(self, _blocked):
        pass


class _Status:
    def __init__(self):
        self.message = ""

    def showMessage(self, message):
        self.message = str(message)

    def currentMessage(self):
        return self.message


class _RendererHarness(PlotOpsMixin):
    def __init__(self, named_datasets, *, selections, phase=False):
        self._named_datasets = named_datasets
        self.active_dataset = named_datasets[0][1]
        self.list_az = object()
        self.list_elev = object()
        self.list_freq = object()
        self.list_pol = object()
        self._selections = {
            self.list_az: selections["azimuth"],
            self.list_elev: selections["elevation"],
            self.list_freq: selections["frequency"],
            self.list_pol: selections.get("polarization", ["HH"]),
        }
        self.btn_phase = _Checked(phase)
        self.btn_pbp = _Checked(False)
        self.btn_hold = _Checked(False)
        self.btn_auto_plot = _Checked(True)
        self.combo_plot_scale = _Combo("dbsm")
        self.combo_polar_zero = _Combo("N")
        self.chk_plot_legend = _Checked(True)
        self.chk_colorbar = _Checked(False)
        self.chk_colorbar_shared = _Checked(True)
        self.plot_figure = Figure()
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        self.plot_colorbars = []
        self.plot_canvas = _CanvasCounter()
        self.status = _Status()
        self.application_palette = {
            "panel_bg": "#ffffff",
            "grid": "#aaaaaa",
            "text": "#000000",
            "border": "#777777",
        }
        self.plot_bg_color = None
        self.plot_grid_color = None
        self.plot_text_color = None
        self.pbp_fill_mode = "solid"
        self.pbp_fill_gray = "#888888"
        self.pbp_heatmap_samples = 16
        self.spin_plot_xmin = _Spin(-200.0)
        self.spin_plot_xmax = _Spin(200.0)
        self.spin_plot_ymin = _Spin(-200.0)
        self.spin_plot_ymax = _Spin(200.0)
        self.spin_plot_xstep = _Spin(0.0)
        self.spin_plot_ystep = _Spin(0.0)
        self.spin_plot_zmin = _Spin(0.0)
        self.spin_plot_zmax = _Spin(0.0)
        self.spin_plot_zstep = _Spin(0.0)

    def _selected_datasets(self):
        return self._named_datasets

    def _selected_values(self, widget):
        return list(self._selections[widget])

    def _indices_for_values(self, axis, values, tol=1.0e-6):
        return RcsGrid._indices_for_axis_values(axis, values, tol=tol)

    def _effective_colormap(self):
        return "viridis"

    def _configure_legend(self, legend, ax=None):
        pass

    def _legend_kwargs(self):
        return {}


class PlotRendererArtistTests(unittest.TestCase):
    def test_colormap_change_updates_live_mappable_without_recompute(self):
        owner = _ArtistMixin()
        owner.plot_figure = Figure()
        owner.plot_ax = owner.plot_figure.add_subplot(111)
        image = owner.plot_ax.imshow(np.arange(4).reshape(2, 2), cmap="viridis")
        owner.plot_colorbars = []
        owner.plot_canvas = _CanvasCounter()
        owner.last_plot_mode = "isar_image"
        owner._plot_isar_image = lambda: self.fail("style change recomputed ISAR")
        owner._on_colormap_changed()
        self.assertEqual(image.get_cmap().name, "plasma")
        self.assertEqual(owner.plot_canvas.idle_calls, 1)

    def test_renderer_legend_update_does_not_force_a_synchronous_draw(self):
        owner = _ArtistMixin()
        owner.plot_figure = Figure()
        owner.plot_ax = owner.plot_figure.add_subplot(111)
        owner.plot_axes = None
        owner.plot_ax.plot([0, 1], [0, 1], label="line")
        owner.chk_plot_legend = _Checked(True)
        owner.plot_canvas = _CanvasCounter()
        owner._update_legend_visibility()
        self.assertEqual(owner.plot_canvas.draw_calls, 0)
        self.assertEqual(owner.plot_canvas.idle_calls, 0)
        owner._update_legend_visibility(True)
        self.assertEqual(owner.plot_canvas.idle_calls, 1)


class PlotRendererIntegrationTests(unittest.TestCase):
    def test_hold_replaces_identical_series_and_blocks_incompatible_scale(self):
        dataset = _grid()
        harness = _RendererHarness(
            [("dataset", dataset)],
            selections={
                "azimuth": dataset.azimuths,
                "elevation": [0.0],
                "frequency": [9.0],
            },
        )
        azimuth_rect_mode.render(harness)
        self.assertEqual(len(harness.plot_ax.lines), 1)

        harness.btn_hold.checked = True
        azimuth_rect_mode.render(harness)
        self.assertEqual(len(harness.plot_ax.lines), 1)

        harness.combo_plot_scale.data = "linear"
        azimuth_rect_mode.render(harness)
        self.assertIn("hold blocked", harness.status.message.lower())
        self.assertEqual(len(harness.plot_ax.lines), 1)

    def test_hold_phase_signature_includes_coherent_metadata(self):
        first = _grid(phase_reference="nose")
        second = _grid(phase_reference="scene-center")
        harness = _RendererHarness(
            [("first", first)],
            selections={
                "azimuth": first.azimuths,
                "elevation": [0.0],
                "frequency": [9.0],
            },
            phase=True,
        )
        azimuth_rect_mode.render(harness)
        harness.btn_hold.checked = True
        harness._named_datasets = [("second", second)]
        harness.active_dataset = second
        azimuth_rect_mode.render(harness)
        self.assertIn("hold blocked", harness.status.message.lower())
        self.assertEqual(len(harness.plot_ax.lines), 1)

    def test_auto_plot_does_not_append_transient_hold_selections(self):
        dataset = _grid()
        harness = _RendererHarness(
            [("dataset", dataset)],
            selections={
                "azimuth": dataset.azimuths,
                "elevation": [0.0],
                "frequency": [9.0],
            },
        )
        harness.btn_hold.checked = True
        harness.last_plot_mode = "azimuth_rect"
        harness._plot_azimuth_rect = mock.Mock()
        harness._maybe_autoplot()
        harness._plot_azimuth_rect.assert_not_called()
        self.assertFalse(hasattr(harness, "_autoplot_timer"))

    def test_linear_plot_labels_use_physical_units(self):
        cases = (
            ("sigma_3d", "dBsm", "RCS (m²)"),
            ("sigma_2d", "dBke", "Scattering Width (m)"),
            ("power_ratio", "dB", "Power Ratio (dimensionless)"),
        )
        for quantity, log_unit, expected in cases:
            with self.subTest(quantity=quantity):
                dataset = _grid(quantity=quantity, log_unit=log_unit)
                harness = _RendererHarness(
                    [(quantity, dataset)],
                    selections={
                        "azimuth": dataset.azimuths,
                        "elevation": [0.0],
                        "frequency": [9.0],
                    },
                )
                harness.combo_plot_scale.data = "linear"
                azimuth_rect_mode.render(harness)
                self.assertEqual(harness.plot_ax.get_ylabel(), expected)

    def test_large_synchronous_sweep_and_waterfall_work_are_blocked_pre_draw(self):
        dataset = _grid()
        selections = {
            "azimuth": dataset.azimuths,
            "elevation": [0.0],
            "frequency": dataset.frequencies,
        }
        frequency_harness = _RendererHarness(
            [("dataset", dataset)], selections=selections
        )
        waterfall_harness = _RendererHarness(
            [("dataset", dataset)], selections=selections
        )
        elevation_harness = _RendererHarness(
            [("dataset", dataset)], selections=selections
        )
        with mock.patch.object(
            common,
            "validate_synchronous_plot_workload",
            side_effect=ValueError("selection too large"),
        ) as preflight:
            frequency_mode.render(frequency_harness)
            self.assertIn("plot blocked", frequency_harness.status.message.lower())
            self.assertEqual(len(frequency_harness.plot_ax.lines), 0)
            elevation_sweep_mode.render(elevation_harness)
            self.assertIn("plot blocked", elevation_harness.status.message.lower())
            self.assertEqual(len(elevation_harness.plot_ax.lines), 0)
            waterfall_mode.render(waterfall_harness)
            self.assertIn("plot blocked", waterfall_harness.status.message.lower())
            self.assertIsNone(waterfall_harness.plot_axes)
        self.assertEqual(preflight.call_count, 3)

    def test_waterfall_shared_colorbar_uses_one_global_normalization(self):
        low = _grid()
        high = _grid()
        low.rcs_power[:, 0, :, 0] = np.linspace(
            1.0, 10.0, low.rcs_power[:, 0, :, 0].size
        ).reshape(low.rcs_power[:, 0, :, 0].shape)
        high.rcs_power[:, 0, :, 0] = np.linspace(
            100.0, 1_000.0, high.rcs_power[:, 0, :, 0].size
        ).reshape(high.rcs_power[:, 0, :, 0].shape)
        harness = _RendererHarness(
            [("low", low), ("high", high)],
            selections={
                "azimuth": low.azimuths,
                "elevation": [0.0],
                "frequency": low.frequencies,
            },
        )
        waterfall_mode.render(harness)
        clims = [axis.collections[0].get_clim() for axis in harness.plot_axes]
        self.assertEqual(clims[0], clims[1])
        np.testing.assert_allclose(clims[0], [0.0, 30.0], atol=1.0e-12)

    def test_multiband_isar_uses_one_global_normalization(self):
        dataset = _grid()
        harness = _RendererHarness(
            [("dataset", dataset)],
            selections={
                "azimuth": dataset.azimuths,
                "elevation": [0.0],
                "frequency": dataset.frequencies,
            },
        )
        params = {
            "dataset": dataset,
            "unit_name": "m",
            "az_target_deg": None,
            "elevation_deg": 0.0,
            "pol_idx": 0,
            "recon": "fft",
        }
        base = {
            "x_range": np.asarray([-1.0, 1.0]),
            "y_range": np.asarray([-1.0, 1.0]),
            "az_values": np.asarray([-1.0, 1.0]),
            "az_nonuniformity": 0.0,
            "freq_nonuniformity": 0.0,
        }
        bands = [
            dict(base, magnitude=np.ones((2, 2))),
            dict(base, magnitude=np.full((2, 2), 10.0)),
        ]
        isar_mode.display_results(harness, params, bands, 0.01)
        clims = [mesh.get_clim() for mesh in harness._isar_meshes]
        self.assertEqual(clims[0], clims[1])
        np.testing.assert_allclose(clims[0], [0.0, 20.0], atol=1.0e-12)

    def test_stale_isar_result_is_rejected_by_render_generation(self):
        dataset = _grid()
        harness = _RendererHarness(
            [("dataset", dataset)],
            selections={
                "azimuth": dataset.azimuths,
                "elevation": [0.0],
                "frequency": dataset.frequencies,
            },
        )
        harness._start_plot_render()
        params = {
            "dataset": dataset,
            "figure_token": harness.plot_figure,
            "render_generation": harness._plot_render_generation,
        }
        harness._isar_busy = True
        harness._start_plot_render()
        with mock.patch.object(isar_mode, "display_results") as display:
            harness._on_isar_compute_done(params, ([], 0.0))
        display.assert_not_called()
        self.assertIn("discarded", harness.status.message.lower())

    def test_linear_isar_peak_drop_uses_intensity_db_ratio(self):
        dataset = _grid()
        harness = _RendererHarness(
            [("dataset", dataset)],
            selections={
                "azimuth": dataset.azimuths,
                "elevation": [0.0],
                "frequency": dataset.frequencies,
            },
        )
        harness.combo_plot_scale = _Combo("linear")
        harness.last_plot_mode = "isar_image"
        harness.spin_isar_peak_drop = _Spin(20.0)
        harness._isar_meshes = [
            harness.plot_ax.imshow(np.asarray([[1.0, 100.0]]))
        ]
        harness._on_isar_peak_scale()
        self.assertAlmostEqual(harness.spin_plot_zmax.value(), 100.0)
        self.assertAlmostEqual(harness.spin_plot_zmin.value(), 1.0)

    def test_plot_recorder_persists_active_reference_dataset_index(self):
        first = _grid()
        active = _grid(angle_unit="rad", frequency_unit="Hz")
        harness = _RendererHarness(
            [("first", first), ("active", active)],
            selections={
                "azimuth": active.azimuths,
                "elevation": [0.0],
                "frequency": [9.0e9],
            },
        )
        harness.active_dataset = active
        harness.python_recorder = mock.Mock()
        harness._python_reference_for_dataset = lambda dataset: (
            "first-ref" if dataset is first else "active-ref"
        )
        harness._record_python_plot("azimuth_rect")
        parameters = harness.python_recorder.record_plot.call_args.kwargs[
            "parameters"
        ]
        self.assertEqual(parameters["reference_index"], 1)

    def test_rect_overlay_converts_radians_and_hz_before_matching(self):
        reference = _grid()
        converted = _grid(angle_unit="rad", frequency_unit="Hz")
        harness = _RendererHarness(
            [("reference", reference), ("converted", converted)],
            selections={
                "azimuth": reference.azimuths,
                "elevation": [0.0],
                "frequency": [9.0],
            },
        )
        azimuth_rect_mode.render(harness)
        self.assertIn("updated", harness.status.message.lower())
        self.assertEqual(harness.plot_ax.get_xlabel(), "Azimuth (deg)")
        self.assertEqual(len(harness.plot_ax.lines), 2)
        np.testing.assert_allclose(
            harness.plot_ax.lines[0].get_xdata(),
            harness.plot_ax.lines[1].get_xdata(),
            atol=1.0e-12,
        )
        self.assertIn("9 GHz", harness.plot_ax.lines[1].get_label())

    def test_polar_radian_axis_is_not_converted_twice(self):
        radian = _grid(angle_unit="rad")
        harness = _RendererHarness(
            [("radian", radian)],
            selections={
                "azimuth": radian.azimuths,
                "elevation": [0.0],
                "frequency": [9.0],
            },
        )
        azimuth_polar_mode.render(harness)
        self.assertEqual(harness.plot_ax.get_xlabel(), "Azimuth (rad)")
        np.testing.assert_allclose(
            harness.plot_ax.lines[0].get_xdata(), radian.azimuths, atol=1.0e-12
        )
        self.assertAlmostEqual(harness.spin_plot_xmax.value(), np.pi)

    def test_compare_rejects_more_than_one_series_instead_of_using_first(self):
        left = _grid()
        right = _grid()
        harness = _RendererHarness(
            [("left", left), ("right", right)],
            selections={
                "azimuth": left.azimuths,
                "elevation": [0.0, 1.0],
                "frequency": [9.0],
            },
        )
        compare_mode.render(harness)
        self.assertIn("exactly one series", harness.status.message.lower())
        self.assertEqual(len(harness.plot_figure.axes), 1)

    def test_compare_phase_residual_is_wrapped(self):
        left = _grid()
        right = _grid()
        left_degrees = np.asarray([179.0, -179.0, 179.0, -179.0, 179.0])
        right_degrees = -left_degrees
        left.rcs_phase[:, 0, 0, 0] = np.deg2rad(left_degrees)
        right.rcs_phase[:, 0, 0, 0] = np.deg2rad(right_degrees)
        harness = _RendererHarness(
            [("left", left), ("right", right)],
            selections={
                "azimuth": left.azimuths,
                "elevation": [0.0],
                "frequency": [9.0],
            },
            phase=True,
        )
        compare_mode.render(harness)
        self.assertIn("updated", harness.status.message.lower())
        residual = harness.plot_figure.axes[1].lines[-1].get_ydata()
        np.testing.assert_allclose(residual, [-2.0, 2.0, -2.0, 2.0, -2.0])

    def test_frequency_phase_p50_is_circular_across_unit_converted_grids(self):
        reference = _grid()
        converted = _grid(angle_unit="rad", frequency_unit="Hz")
        phase_degrees = np.asarray([179.0, -179.0, 179.0, -179.0, 179.0])
        reference.rcs_phase[:, 0, :, 0] = np.deg2rad(phase_degrees[:, None])
        converted.rcs_phase[:, 0, :, 0] = np.deg2rad(phase_degrees[:, None])
        harness = _RendererHarness(
            [("reference", reference), ("converted", converted)],
            selections={
                "azimuth": reference.azimuths,
                "elevation": [0.0],
                "frequency": reference.frequencies,
            },
            phase=True,
        )
        frequency_mode.render(harness)
        self.assertIn("updated", harness.status.message.lower())
        self.assertEqual(harness.plot_ax.get_xlabel(), "Frequency (GHz)")
        self.assertEqual(len(harness.plot_ax.lines), 2)
        for line in harness.plot_ax.lines:
            self.assertTrue(np.all(np.abs(line.get_ydata()) > 170.0))
        np.testing.assert_allclose(
            harness.plot_ax.lines[0].get_xdata(),
            harness.plot_ax.lines[1].get_xdata(),
        )

    def test_waterfall_uses_one_reference_unit_system_for_every_panel(self):
        reference = _grid()
        converted = _grid(angle_unit="rad", frequency_unit="Hz")
        harness = _RendererHarness(
            [("reference", reference), ("converted", converted)],
            selections={
                "azimuth": reference.azimuths,
                "elevation": [0.0],
                "frequency": reference.frequencies,
            },
        )
        waterfall_mode.render(harness)
        self.assertIn("updated", harness.status.message.lower())
        self.assertEqual(len(harness.plot_axes), 2)
        for axis in harness.plot_axes:
            self.assertEqual(axis.get_xlabel(), "Azimuth (deg)")
            self.assertEqual(axis.get_ylabel(), "Frequency (GHz)")

    def test_oversized_phase_waterfall_is_blocked_instead_of_scalar_decimated(self):
        dataset = _grid()
        harness = _RendererHarness(
            [("phase-grid", dataset)],
            selections={
                "azimuth": dataset.azimuths,
                "elevation": [0.0],
                "frequency": dataset.frequencies,
            },
            phase=True,
        )
        with (
            mock.patch.object(common, "MAX_IMAGE_SIDE", 3),
            mock.patch.object(common, "MAX_IMAGE_CELLS", 8),
        ):
            waterfall_mode.render(harness)
        self.assertIn("phase waterfall blocked", harness.status.message.lower())
        self.assertIn("narrow", harness.status.message.lower())
        self.assertIsNone(harness.plot_axes)


if __name__ == "__main__":
    unittest.main()
