"""Canvas mouse interaction and mixed dataset style regressions."""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

import dataset_plot_style
import grim_cut_gui
from plot_modes import (
    azimuth_rect_mode, azimuth_polar_mode, frequency_mode,
    elevation_sweep_mode, compare_mode,
)
from test_gui_shell import (
    _FakeFeatureWorkflow, _FakeFreddyIntegration, _FakeGhostIntegration,
    _FakeRunsWorkspace, _MemorySettings, _RecordingWindow,
)
from test_plot_renderer_correctness import _grid


class DatasetPlotStyleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        for name, replacement in (
            ("GhostIntegrationWidget", _FakeGhostIntegration),
            ("FreddyIntegrationWidget", _FakeFreddyIntegration),
            ("RunsWorkspace", _FakeRunsWorkspace),
        ):
            patch = mock.patch.object(grim_cut_gui, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(grim_cut_gui, "load_ghost_module", return_value=_FakeFeatureWorkflow())
        patch.start()
        self.addCleanup(patch.stop)
        self.window = _RecordingWindow(settings=_MemorySettings())
        self.window.btn_auto_plot.setChecked(False)
        self.window.btn_zoom_box.setChecked(False)
        self.window.btn_pan.setChecked(False)
        self.datasets = [_grid(), _grid()]
        for index, dataset in enumerate(self.datasets):
            dataset.rcs_power[:] = np.asarray([1, 2, 5, 3, 2])[:, None, None, None] * (index + 1) ** 3
            # Deliberately duplicate names: identity must not come from labels.
            self.window._add_dataset_row(dataset, "Shared name", "", file_name="")
        self.datasets = [self.window.table.item(row, 0).data(Qt.UserRole) for row in range(2)]
        self.window.table.selectAll()
        self.window.list_az.selectAll()
        self.window.list_elev.selectAll()
        self.window.list_freq.clearSelection()
        self.window.list_freq.item(0).setSelected(True)
        self.window.list_freq.item(1).setSelected(True)
        self.window.list_pol.item(0).setSelected(True)
        self.keys = [self.window._dataset_plot_key(dataset) for dataset in self.datasets]
        self.window.resize(1300, 850)
        self.window.show()
        self.app.processEvents()
        self.render()

    def tearDown(self):
        self.window.deleteLater()
        self.app.processEvents()

    def render(self, mode=azimuth_rect_mode):
        mode.render(self.window)
        self.window._fit_both()
        self.window.plot_canvas.draw()

    def lines(self, key):
        return [line for ax in self.window.plot_figure.axes for line in ax.lines
                if getattr(line, "_grim_dataset_key", None) == key]

    def canvas_pos(self, x, y):
        canvas = self.window.plot_canvas
        ratio = canvas.device_pixel_ratio
        return QPoint(round(x / ratio), round((canvas.figure.bbox.height - y) / ratio))

    def line_pos(self, key):
        line = self.lines(key)[0]
        xy = np.column_stack((line.get_xdata(), line.get_ydata()))
        return self.canvas_pos(*line.get_transform().transform(xy)[len(xy) // 2])

    def legend_pos(self, index):
        legend = self.window.plot_ax.get_legend()
        bbox = legend.get_texts()[index].get_window_extent(self.window.plot_canvas.get_renderer())
        return self.canvas_pos(bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2)

    @staticmethod
    def action(menu, text):
        return next(action for action in menu.actions() if action.text() == text)

    def test_canvas_click_highlights_dataset_and_legend_then_clears(self):
        window = self.window
        original_data = self.datasets[0].rcs_power.copy()
        original_colors = [line.get_color() for line in self.lines(self.keys[0])]
        QTest.mouseClick(window.plot_canvas, Qt.LeftButton, pos=self.line_pos(self.keys[0]))
        self.assertEqual(window._highlighted_plot_datasets, {self.keys[0]})
        self.assertTrue(all(line.get_path_effects() for line in self.lines(self.keys[0])))
        self.assertTrue(all(not line.get_path_effects() for line in self.lines(self.keys[1])))
        legend = window.plot_ax.get_legend()
        self.assertTrue(all(text.get_bbox_patch() is not None for text in legend.get_texts()[:2]))
        self.assertTrue(all(text.get_bbox_patch() is None for text in legend.get_texts()[2:]))
        QTest.mouseClick(window.plot_canvas, Qt.LeftButton, pos=self.legend_pos(2))
        self.assertEqual(window._highlighted_plot_datasets, {self.keys[1]})
        QTest.mouseClick(window.plot_canvas, Qt.LeftButton, pos=QPoint(4, 4))
        self.assertEqual(window._highlighted_plot_datasets, set())
        self.assertTrue(all(not line.get_path_effects() for line in self.lines(self.keys[0])))
        self.assertTrue(all(text.get_bbox_patch() is None for text in legend.get_texts()))
        self.assertEqual([line.get_color() for line in self.lines(self.keys[0])], original_colors)
        np.testing.assert_array_equal(self.datasets[0].rcs_power, original_data)
        self.assertEqual(len(window.table.selectionModel().selectedRows()), 2)

    def test_right_click_curve_and_legend_route_to_style_menu(self):
        for pos in (self.line_pos(self.keys[0]), self.legend_pos(0)):
            with mock.patch.object(self.window, "_show_dataset_plot_style_menu") as show:
                self.window._on_plot_context_menu(pos)
            self.assertEqual(show.call_args.args[0]._grim_dataset_key, self.keys[0])

    def test_menu_styles_mix_scatter_and_lines_and_match_legend(self):
        window = self.window
        key = self.keys[0]
        line = self.lines(key)[0]
        menu = QMenu(window)
        window._add_dataset_plot_style_menu(menu, [key], line=line)
        self.action(self.action(menu, "Plot as").menu(), "Scatter").trigger()
        self.action(self.action(menu, "Scatter symbol").menu(), "Diamond").trigger()
        with mock.patch.object(dataset_plot_style.QInputDialog, "getDouble", return_value=(3.5, True)):
            self.action(menu, "Line width…").trigger()
        with mock.patch.object(dataset_plot_style.QInputDialog, "getDouble", return_value=(9.0, True)):
            self.action(menu, "Scatter symbol size…").trigger()
        with mock.patch.object(dataset_plot_style.QColorDialog, "getColor", return_value=QColor("#aa33cc")):
            self.action(menu, "Color…").trigger()
        self.action(self.action(menu, "Line type").menu(), "Dashed").trigger()
        for artist in self.lines(key) + window.plot_ax.get_legend().legend_handles[:2]:
            self.assertEqual(artist.get_linestyle(), "None")
            self.assertEqual(artist.get_marker(), "D")
            self.assertEqual(artist.get_markersize(), 9.0)
            self.assertEqual(artist.get_color(), "#aa33cc")
        self.assertTrue(all(line.get_linestyle() == "-" for line in self.lines(self.keys[1])))
        QTest.mouseClick(window.plot_canvas, Qt.LeftButton, pos=self.line_pos(key))
        self.assertEqual(window._highlighted_plot_datasets, {key})
        self.action(self.action(menu, "Plot as").menu(), "Line").trigger()
        for artist in self.lines(key) + window.plot_ax.get_legend().legend_handles[:2]:
            self.assertEqual(artist.get_linestyle(), "--")
            self.assertEqual(artist.get_linewidth(), 3.5)
            self.assertEqual(artist.get_marker(), "None")
        self.action(menu, "Reset plot style").trigger()
        self.assertEqual(line.get_color(), line._grim_base_style["color"])
        self.assertEqual(line.get_linewidth(), line._grim_base_style["linewidth"])

    def test_replot_hold_and_legend_toggle_preserve_dataset_styles(self):
        window = self.window
        window._set_dataset_plot_style([self.keys[0]], kind="scatter", marker="s", color="#ee2200")
        window._highlight_plot_dataset(self.keys[0])
        self.render()
        window.btn_hold.setChecked(True)
        self.render()
        self.assertEqual(len(window.plot_ax.lines), 4)
        window.chk_plot_legend.setChecked(False)
        self.assertFalse(window.plot_ax.get_legend().get_visible())
        window.chk_plot_legend.setChecked(True)
        self.assertTrue(window.plot_ax.get_legend().get_visible())
        self.assertTrue(all(line.get_marker() == "s" for line in self.lines(self.keys[0])))
        proxies = window.plot_ax.get_legend().legend_handles
        self.assertEqual([proxy.get_marker() for proxy in proxies], ["s", "s", "None", "None"])
        self.assertTrue(all(text.get_bbox_patch() is not None for text in window.plot_ax.get_legend().get_texts()[:2]))
        window.table.item(0, 0).setText("Renamed")
        self.assertEqual(window._dataset_plot_key(self.datasets[0]), self.keys[0])
        window.btn_hold.setChecked(False)
        self.render()
        self.assertEqual(self.lines(self.keys[0])[0].get_marker(), "s")

    def test_all_dataset_curve_renderers_apply_mixed_styles(self):
        self.window._set_dataset_plot_style([self.keys[0]], kind="scatter", marker="^", color="#ee2200")
        self.window._set_dataset_plot_style([self.keys[1]], linewidth=4, linestyle=":")
        for mode in (azimuth_polar_mode, frequency_mode, elevation_sweep_mode, compare_mode):
            with self.subTest(mode=mode.__name__):
                if mode is compare_mode:
                    self.window.list_freq.item(1).setSelected(False)
                self.render(mode)
                self.assertTrue(self.lines(self.keys[0]), self.window.status.currentMessage())
                self.assertTrue(all(line.get_marker() == "^" and line.get_linestyle() == "None" for line in self.lines(self.keys[0])))
                self.assertTrue(all(line.get_linewidth() == 4 and line.get_linestyle() == ":" for line in self.lines(self.keys[1])))

    def test_pan_zoom_and_isar_do_not_select_plot_datasets(self):
        window = self.window
        window._highlight_plot_dataset(None)
        for button in (window.btn_pan, window.btn_zoom_box):
            button.setChecked(True)
            QTest.mouseClick(window.plot_canvas, Qt.LeftButton, pos=self.line_pos(self.keys[0]))
            self.assertEqual(window._highlighted_plot_datasets, set())
            button.setChecked(False)
        window._activate_plot_tab("isar")
        event = MouseEvent("button_press_event", window.plot_canvas, 10, 10, button=MouseButton.LEFT)
        self.assertIsNone(window._dataset_line_at_event(event))


if __name__ == "__main__":
    unittest.main()
