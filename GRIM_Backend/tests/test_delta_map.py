"""Numerical, memory-bound, GUI, and replay regressions for Delta Map."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import delta_map_mode as delta
from GRIM_Backend.scripting.plotting import plot_datasets
from GRIM_Backend.scripting.recorder import DatasetReference, PythonScriptRecorder
from GRIM_Backend.tests import test_gui_shell as shell


def grid(*, offset=0.0, converted=False, log_unit="dBsm", az=None):
    azimuths = np.asarray([-20.0, 0.0, 20.0] if az is None else az)
    elevations = np.asarray([-5.0, 5.0])
    frequencies = np.asarray([8.0, 9.0, 11.0, 12.0])
    db = (azimuths[:, None, None, None] / 10 + elevations[None, :, None, None] / 5
          + frequencies[None, None, :, None] + np.array([0.0, 1.0])[None, None, None, :] - 30 + offset)
    power = 10 ** (db / 10)
    if log_unit == "dBke":
        power = power * 299792458 / (2 * np.pi * frequencies[None, None, :, None] * 1e9)
    return RcsGrid(
        np.deg2rad(azimuths) if converted else azimuths,
        np.deg2rad(elevations) if converted else elevations,
        frequencies * 1e9 if converted else frequencies, ["HH", "VV"],
        rcs_power=power, rcs_phase=np.zeros_like(power),
        units={"azimuth": "rad" if converted else "deg", "elevation": "rad" if converted else "deg",
               "frequency": "Hz" if converted else "GHz", "rcs_log_unit": log_unit,
               "rcs_linear_quantity": "sigma_2d" if log_unit == "dBke" else "sigma_3d"},
    )


def selections(source):
    return {axis: getattr(source, delta.AXIS_ATTRIBUTES[axis]).tolist() for axis in delta.AXES}


class DeltaMapTests(unittest.TestCase):
    def test_all_axis_pairs_preserve_orientation_and_source_levels(self):
        a, b = grid(offset=3.25), grid()
        for x, y in itertools.permutations(delta.AXES, 2):
            with self.subTest(x=x, y=y):
                selected = selections(a)
                fixed = next(v for v in delta.AXES if v not in (x, y))
                selected[fixed] = [selected[fixed][-1]]
                result = delta.prepare([("A", a), ("B", b)], reference=a, selections=selected,
                                       polarization="VV", x_axis=x, y_axis=y)
                self.assertEqual(result.delta_db.shape, (len(selected[y]), len(selected[x])))
                for row, yy in enumerate(result.y):
                    for col, xx in enumerate(result.x):
                        point = {x: xx, y: yy, fixed: selected[fixed][0]}
                        expected = point["azimuth"] / 10 + point["elevation"] / 5 + point["frequency"] - 29
                        self.assertAlmostEqual(result.b_db[row, col], expected)
                np.testing.assert_allclose(result.delta_db, 3.25, atol=1e-12)

    def test_different_units_dbke_and_reference_index(self):
        for log_unit in ("dBsm", "dBke"):
            a, b = grid(offset=-4, log_unit=log_unit), grid(converted=True, log_unit=log_unit)
            selected = selections(b)
            selected["elevation"] = [b.elevations[1]]
            result = delta.prepare([("A", a), ("B", b)], reference=b, selections=selected, polarization="HH")
            np.testing.assert_allclose(result.delta_db, -4, atol=1e-12)
            np.testing.assert_allclose(result.b_db[:, 0], [-23, -22, -20, -19], atol=1e-12)
            self.assertEqual(result.source_unit, log_unit)

    def test_unmatched_axes_and_invalid_values_remain_masked_without_a_floor(self):
        a, b = grid(), grid(az=[-20.0, 20.0])
        a.rcs_power[0, 0, 0, 0] = 0
        b.rcs_power[0, 0, 1, 0] = np.nan
        a.rcs_power[2, 0, 2, 0] = 1e-30
        b.rcs_power[1, 0, 2, 0] = 1e-31
        selected = selections(a)
        selected["elevation"] = [-5]
        result = delta.prepare([("A", a), ("B", b)], reference=a, selections=selected, polarization="HH")
        self.assertTrue(np.all(np.isnan(result.delta_db[:, 1])))
        self.assertTrue(np.isnan(result.delta_db[0, 0]))
        self.assertTrue(np.isnan(result.delta_db[1, 0]))
        self.assertAlmostEqual(result.delta_db[2, 2], 10)
        self.assertAlmostEqual(result.a_db[2, 2], -300)
        self.assertIn("B unavailable", delta.cell_text(result, 0, 8))
        self.assertIsNone(delta.cell_text(result, -999, 8))

    def test_invalid_requests_and_incompatible_physics_are_rejected(self):
        a, b = grid(), grid()
        selected = selections(a)
        selected["elevation"] = [-5]
        kwargs = dict(reference=a, selections=selected, polarization="HH")
        with self.assertRaisesRegex(ValueError, "different axes"):
            delta.prepare([("A", a), ("B", b)], x_axis="frequency", y_axis="frequency", **kwargs)
        with self.assertRaisesRegex(ValueError, "two datasets"):
            delta.prepare([("A", a)], **kwargs)
        with self.assertRaisesRegex(ValueError, "polarization"):
            delta.prepare([("A", a), ("B", b)], **dict(kwargs, polarization="HV"))
        with self.assertRaisesRegex(ValueError, "physical quantities"):
            delta.prepare([("A", a), ("B", grid(log_unit="dBke"))], **kwargs)
        b.units["angular_coordinate_system"] = "great_circle"
        with self.assertRaisesRegex(ValueError, "coordinate systems"):
            delta.prepare([("A", a), ("B", b)], **kwargs)

    def test_missing_fixed_coordinate_and_ambiguous_matches_rejected(self):
        a, b = grid(), grid()
        selected = selections(a)
        selected["elevation"] = [0]
        with self.assertRaisesRegex(ValueError, "fixed elevation"):
            delta.prepare([("A", a), ("B", b)], reference=a, selections=selected, polarization="HH")
        selected["elevation"] = [-5]
        b.azimuths[1] = b.azimuths[0] + 1e-7
        with self.assertRaisesRegex(ValueError, "Ambiguous azimuth"):
            delta.prepare([("A", a), ("B", b)], reference=a, selections=selected, polarization="HH")

    def test_only_two_dimensional_scalar_fixed_polarization_gathers(self):
        class Guard(np.ndarray):
            def __getitem__(self, key):
                if not (isinstance(key, tuple) and len(key) == 4 and isinstance(key[3], int)
                        and sum(isinstance(v, int) for v in key[:3]) == 1):
                    raise AssertionError("attempt to read more than a fixed-polarization slice")
                result = super().__getitem__(key)
                if result.ndim != 2:
                    raise AssertionError("gather created a volume")
                return result
        a, b = grid(), grid(offset=2)
        a.rcs_power = a.rcs_power.view(Guard)
        b.rcs_power = b.rcs_power.view(Guard)
        for x, y in itertools.permutations(delta.AXES, 2):
            selected = selections(a)
            fixed = next(v for v in delta.AXES if v not in (x, y))
            selected[fixed] = [selected[fixed][0]]
            result = delta.prepare([("A", a), ("B", b)], reference=a, selections=selected,
                                   polarization="HH", x_axis=x, y_axis=y)
            np.testing.assert_allclose(result.delta_db, -2, atol=1e-12)
        with mock.patch.object(delta, "MAX_DELTA_CELLS", 1):
            with self.assertRaisesRegex(ValueError, "limited to"):
                delta.prepare([("A", a), ("B", b)], reference=a, selections=selected,
                              polarization="HH", x_axis=x, y_axis=y)

    def test_palette_limits_masking_singletons_and_annotations(self):
        a, b = grid(offset=2), grid()
        selected = {"azimuth": [0], "elevation": [-5], "frequency": [9]}
        result = delta.prepare([("A", a), ("B", b)], reference=a, selections=selected, polarization="HH")
        figure = Figure()
        axes = figure.add_subplot(111)
        mesh, colorbar = delta.draw(figure, axes, result, limit=1, show_values=True)
        self.assertEqual(mesh.get_clim(), (-1, 1))
        self.assertEqual(colorbar.extend, "both")
        self.assertEqual(colorbar.ax.get_ylabel(), "A - B (dB)")
        self.assertEqual(axes.texts[0].get_text(), "+2.0")
        for limit in (0, -1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "limit"):
                delta.draw(figure, axes, result, limit=limit)
        result.delta_db[:] = 0
        mesh, _ = delta.draw(figure, axes, result)
        self.assertEqual(mesh.get_clim(), (-0.1, 0.1))

    def test_headless_replay_has_same_fields_and_no_qt_canvas(self):
        a, b = grid(offset=6, log_unit="dBke"), grid(converted=True, log_unit="dBke")
        figure = plot_datasets([("A", a), ("B", b)], mode="delta_map", reference_index=1,
                               azimuths=b.azimuths, elevations=b.elevations,
                               frequencies=[b.frequencies[1]], polarization="VV",
                               delta_options={"x_axis": "elevation", "y_axis": "azimuth", "limit": 8})
        self.assertEqual(figure.canvas.__class__.__name__, "FigureCanvasAgg")
        result = figure.axes[0]._grim_delta_map
        np.testing.assert_allclose(result.delta_db, 6, atol=1e-12)
        self.assertEqual(figure.axes[0].collections[0].get_clim(), (-8, 8))
        figure.canvas.draw()

    def test_generated_script_exports_delta_map(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            a_path, b_path = directory / "a.grim", directory / "b.grim"
            grid(offset=3).save(a_path)
            grid().save(b_path)
            recorder = PythonScriptRecorder()
            a = DatasetReference("a", "A", str(a_path))
            b = DatasetReference("b", "B", str(b_path))
            recorder.bind_loaded(a)
            recorder.bind_loaded(b)
            variable = recorder.record_plot([a, b], names=["A", "B"], mode="delta_map", parameters={
                "azimuths": [-20, 0, 20], "elevations": [-5], "frequencies": [8, 9, 11, 12],
                "polarization": "HH", "delta_options": {"limit": 4, "show_values": True},
            })
            self.assertIsNotNone(variable)
            output = directory / "delta.png"
            self.assertTrue(recorder.record_plot_save(str(output)))
            namespace = {"__file__": str(directory / "delta-replay.py"), "__name__": "__main__"}
            exec(compile(recorder.script, "delta-replay.py", "exec"), namespace)
            self.assertGreater(output.stat().st_size, 1000)
            np.testing.assert_allclose(namespace[variable].axes[0]._grim_delta_map.delta_db, 3, atol=1e-12)


class DeltaMapGuiTests(unittest.TestCase):
    setUp = shell.UnifiedGuiShellTest.setUp
    tearDown = shell.UnifiedGuiShellTest.tearDown

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        font_path = Path("C:/Windows/Fonts/segoeui.ttf")
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                cls.app.setFont(QFont(families[0], 9))

    @classmethod
    def tearDownClass(cls):
        cls.app.setFont(cls.original_font)

    def load_pair(self, converted=False):
        window = self.window
        a, b = grid(offset=3, converted=converted), grid()
        window._add_dataset_row(a, "Dataset A", "", "a.grim")
        window._add_dataset_row(b, "Dataset B", "", "b.grim")
        index = window.table.model().index(0, 0)
        window.table.setCurrentIndex(index)
        window.table.selectionModel().select(index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        window.table.selectionModel().select(window.table.model().index(1, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        for widget in (window.list_az, window.list_elev, window.list_freq):
            widget.selectAll()
        window.list_pol.clearSelection()
        window.list_pol.item(0).setSelected(True)
        self.app.processEvents()
        return a, b

    def test_controls_swap_axes_inspection_limits_replay_and_plot_switch(self):
        self.load_pair()
        w = self.window
        w._plot_controls_by_tab["plotting"]["delta_map"].click()
        self.assertIn("updated", w.status.currentMessage(), w.status.currentMessage())
        self.assertEqual(w.last_plot_mode, "delta_map")
        controls = w.delta_map_controls
        np.testing.assert_allclose(w.plot_ax._grim_delta_map.delta_db, 3, atol=1e-12)
        controls.swap.click()
        np.testing.assert_allclose(w.plot_ax._grim_delta_map.delta_db, -3, atol=1e-12)
        controls.x_axis.setCurrentIndex(controls.x_axis.findData("elevation"))
        self.assertEqual(w.plot_ax._grim_delta_map.fixed_axis, "azimuth")
        controls.fixed_value.setCurrentIndex(2)
        self.assertEqual(w.plot_ax._grim_delta_map.fixed_value, 20)
        controls.auto_limit.setChecked(False)
        controls.limit.setValue(7)
        self.assertEqual(w.plot_ax.collections[0].get_clim(), (-7, 7))
        event = SimpleNamespace(canvas=w.plot_canvas, inaxes=w.plot_ax, xdata=-5, ydata=9, button=MouseButton.LEFT)
        w._on_plot_mouse_press(event)
        w._reset_hover_readout()
        self.assertIn("A - B -3.000 dB", w.hover_readout.text())
        spec = w.last_python_plot_spec
        self.assertEqual(spec[0], "supported")
        self.assertEqual(spec[3], "delta_map")
        self.assertEqual(spec[4]["delta_options"]["limit"], 7)
        selected = list(reversed(w._selected_datasets()))
        figure = plot_datasets(selected, mode=spec[3], **spec[4])
        np.testing.assert_allclose(figure.axes[0]._grim_delta_map.delta_db, w.plot_ax._grim_delta_map.delta_db)
        w.chk_colorbar.setChecked(False)
        self.assertEqual(len(w.plot_figure.axes), 1)
        w.combo_colormap.setCurrentIndex(1)
        self.assertEqual(w.plot_ax.collections[0].cmap.name, "RdBu_r")
        w.plot_ax.set_xlim(-1, 1)
        w.plot_ax.set_ylim(9, 10)
        w._fit_both()
        np.testing.assert_allclose(w.plot_ax.get_xlim(), [-10, 10])
        np.testing.assert_allclose(w.plot_ax.get_ylim(), [7.5, 12.5])
        for widget in (w.list_elev, w.list_freq):
            widget.clearSelection()
            widget.item(0).setSelected(True)
        w._plot_compare()
        self.assertIn("updated", w.status.currentMessage(), w.status.currentMessage())
        self.assertTrue(controls.isHidden())
        self.assertFalse(hasattr(w.plot_ax, "_grim_delta_map"))
        w._plot_delta_map()
        w.list_freq.selectAll()
        w._plot_frequency()
        self.assertFalse(hasattr(w.plot_ax, "_grim_delta_map"))
        self.assertEqual(len(w.plot_colorbars), 0)
        self.assertEqual(len(w.plot_figure.texts), 0)

    def test_failed_selection_clears_previous_map_and_export_recipe(self):
        self.load_pair()
        w = self.window
        w._plot_delta_map()
        self.assertIsNotNone(w.last_python_plot_spec)
        w.table.selectRow(0)
        w._plot_delta_map()
        self.assertIn("exactly two", w.status.currentMessage())
        self.assertIsNone(w.last_python_plot_spec)
        self.assertFalse(hasattr(w.plot_ax, "_grim_delta_map"))

    def test_hz_reference_keeps_fit_controls_in_physical_range(self):
        self.load_pair(converted=True)
        w = self.window
        w._plot_delta_map()
        self.assertIn("updated", w.status.currentMessage())
        self.assertAlmostEqual(w.spin_plot_ymin.value(), 7.5e9)
        self.assertAlmostEqual(w.spin_plot_ymax.value(), 12.5e9)
        w._fit_both()
        np.testing.assert_allclose(w.plot_ax.get_ylim(), [7.5e9, 12.5e9])

    def test_controls_remain_independent_of_isar_and_fit_compact_width(self):
        self.load_pair()
        w = self.window
        w._plot_delta_map()
        controls = w.delta_map_controls
        controls.auto_limit.setChecked(False)
        controls.limit.setValue(12)
        w.resize(1280, 720)
        w.show()
        self.app.processEvents()
        self.assertEqual(w.width(), 1280)
        self.assertLessEqual(controls.minimumSizeHint().width(), controls.width())
        w._activate_plot_tab("isar")
        self.assertIsNot(w.delta_map_controls, controls)
        self.assertTrue(w.delta_map_controls.isHidden())
        w._activate_plot_tab("plotting")
        self.assertIs(w.delta_map_controls, controls)
        self.assertEqual(controls.limit.value(), 12)


if __name__ == "__main__":
    unittest.main()
