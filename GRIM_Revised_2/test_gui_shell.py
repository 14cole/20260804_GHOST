"""Focused shell regressions for the unified GRIM application."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.colors import to_hex
from PySide6.QtCore import QItemSelectionModel, QMimeData, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import grim_cut_gui
import grim_cut_dataset_mixin
import freddy_integration
import ghost_integration
from grim_cut_dataset_mixin import (
    DATASET_DIRTY_ROLE,
    DATASET_ID_ROLE,
    ConicGCDialog,
    RangeCalibrationDialog,
    SupportReferenceDifferenceDialog,
    WedgeConicDialog,
)
from assembly_tree import (
    AssemblyTreePanel,
    _TYPE_BRANCH,
    _TYPE_ROOT,
    _attach,
    _branch_drop_would_create_cycle,
)
from grim_dataset import GRIM_GC_CONVENTION, RcsGrid


class _FakeGhostIntegration(QWidget):
    files_exported = Signal(list, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.backend_path = None
        self.running = False
        self.focus_called = False
        self.attached_artifacts: list[tuple[str, str]] = []
        self.palette_calls: list[dict[str, object]] = []
        self.setLayout(QVBoxLayout())

    def solve_is_running(self) -> bool:
        return self.running

    def focus_solver(self) -> None:
        self.focus_called = True

    def attach_material_artifact(self, kind: str, path: str) -> None:
        self.attached_artifacts.append((kind, path))

    def apply_application_palette(self, palette) -> bool:
        self.palette_calls.append(dict(palette))
        return True


class _FakeFreddyIntegration(QWidget):
    # Deliberately expose a GHOST-shaped signal: the shell must not connect
    # FREDDY material/IBC CSV exports to GRIM's RCS dataset loader.
    files_exported = Signal(list, str)
    attach_to_ghost_requested = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.running = False
        self.focus_called = False
        self.palette_calls: list[dict[str, object]] = []
        self.setLayout(QVBoxLayout())

    def job_is_running(self) -> bool:
        return self.running

    def focus_workspace(self) -> None:
        self.focus_called = True

    def apply_application_palette(self, palette) -> bool:
        self.palette_calls.append(dict(palette))
        return True


class _FakeRunsWorkspace(QWidget):
    status_changed = Signal(str)
    results_downloaded = Signal(str)

    def __init__(self, parent=None, **_kwargs) -> None:
        super().__init__(parent)
        self.running = False
        self.focus_called = False
        self.save_count = 0
        self.setLayout(QVBoxLayout())

    def job_is_running(self) -> bool:
        return self.running

    def busy_operation(self) -> str | None:
        return "HPC upload and submission" if self.running else None

    def focus_workspace(self) -> None:
        self.focus_called = True

    def save_settings(self) -> None:
        self.save_count += 1


class _MemorySettings:
    def __init__(self, values=None) -> None:
        self.values = dict(values or {})
        self.sync_count = 0

    def value(self, key, default=None):
        return self.values.get(str(key), default)

    def setValue(self, key, value) -> None:
        self.values[str(key)] = value

    def sync(self) -> None:
        self.sync_count += 1


class _RecordingWindow(grim_cut_gui.GrimCutWindow):
    def __init__(self, *, settings=None) -> None:
        self.loaded_path_batches: list[list[str]] = []
        super().__init__(settings=settings)

    def _handle_files_dropped(self, paths) -> None:
        self.loaded_path_batches.append([os.fspath(path) for path in paths])


class _FakeFeatureWorkflow:
    FeatureAssemblyRequest = staticmethod(lambda **values: values)

    @staticmethod
    def discover_feature_dataset_ids(**_values):
        return {"point_dataset_ids": (), "line_dataset_ids": ()}

    @staticmethod
    def prepare_feature_assembly(request):
        return request

    @staticmethod
    def execute_feature_assembly(_plan):
        return "assembled.grim"


def _grid(amplitude: float = 1.0) -> RcsGrid:
    field = np.asarray([[[[complex(amplitude)]]]], dtype=np.complex128)
    return RcsGrid(
        [0.0],
        [0.0],
        [10.0],
        ["VV"],
        rcs=field,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        },
    )


class UnifiedGuiShellTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_application_palette_choices_are_not_plot_colormaps(self) -> None:
        context = self.window._plot_contexts["plotting"]
        palette_names = list(grim_cut_gui.APPLICATION_PALETTES)
        self.assertEqual(
            palette_names,
            ["Colorful", "Light", "Dark", "Raytheon"],
        )
        view_action = next(
            action
            for action in self.window.menuBar().actions()
            if action.text().replace("&", "") == "View"
        )
        self.assertIsNotNone(view_action.menu())
        self.assertTrue(
            any(
                action.menu() is self.window.application_palette_menu
                for action in view_action.menu().actions()
            )
        )
        self.assertEqual(
            [action.text() for action in self.window.application_palette_menu.actions()],
            palette_names,
        )
        self.assertTrue(self.window.application_palette_group.isExclusive())
        for name in palette_names:
            self.assertEqual(context.combo_colormap.findText(name), -1)
        self.assertEqual(
            [
                context.combo_colormap.itemText(index)
                for index in range(context.combo_colormap.count())
            ],
            ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
        )
        for name, palette in grim_cut_gui.APPLICATION_PALETTES.items():
            with self.subTest(palette=name):
                qss = grim_cut_gui.build_qss(palette)
                self.assertIn(str(palette["panel_bg"]), qss)
                freddy_theme = (
                    freddy_integration.freddy_theme_from_application_palette(
                        palette
                    )
                )
                self.assertEqual(
                    freddy_theme["window_bg"], palette["win_bg"]
                )
                self.assertEqual(
                    freddy_theme["plot_grid"], palette["grid"]
                )

        raytheon = grim_cut_gui.APPLICATION_PALETTES["Raytheon"]
        self.assertEqual(
            {
                role: raytheon[role]
                for role in (
                    "win_bg",
                    "panel_bg",
                    "text",
                    "head_bg",
                    "border",
                    "hover",
                    "checked_bg",
                    "checked_border",
                    "grid",
                    "muted",
                    "fg",
                )
            },
            {
                "win_bg": "#d9d9d6",
                "panel_bg": "#ffffff",
                "text": "#000000",
                "head_bg": "#d9d9d6",
                "border": "#63666a",
                "hover": "#63666a",
                "checked_bg": "#ce1126",
                "checked_border": "#ce1126",
                "grid": "#b1b3b3",
                "muted": "#63666a",
                "fg": "#000000",
            },
        )
        self.assertEqual(
            grim_cut_gui.RAYTHEON_TERTIARY_PPT_CHART_COLORS,
            (
                "#7ba7bc",
                "#b7a99a",
                "#908cc2",
                "#9abeaa",
                "#efb661",
            ),
        )
        raytheon_application_colors = {
            str(value).lower()
            for value in raytheon.values()
            if isinstance(value, str)
        }
        raytheon_application_colors.update(
            str(value).lower() for value in raytheon["layer_colors"]
        )
        self.assertTrue(
            raytheon_application_colors.isdisjoint(
                grim_cut_gui.RAYTHEON_TERTIARY_PPT_CHART_COLORS
            )
        )

    def test_invalid_saved_application_palette_falls_back_to_dark(self) -> None:
        settings = _MemorySettings(
            {
                grim_cut_gui.APPLICATION_PALETTE_SETTINGS_KEY:
                    "removed-palette"
            }
        )
        window = _RecordingWindow(settings=settings)
        try:
            self.assertEqual(
                window.application_palette_name,
                grim_cut_gui.DEFAULT_APPLICATION_PALETTE,
            )
            self.assertTrue(
                window._application_palette_actions[
                    grim_cut_gui.DEFAULT_APPLICATION_PALETTE
                ].isChecked()
            )
            self.assertEqual(settings.sync_count, 0)
        finally:
            window.deleteLater()
            self.app.processEvents()

    def test_legacy_raytheon_palette_setting_migrates_to_current_name(self) -> None:
        settings = _MemorySettings(
            {
                grim_cut_gui.APPLICATION_PALETTE_SETTINGS_KEY:
                    "Raytheon-inspired"
            }
        )
        window = _RecordingWindow(settings=settings)
        try:
            self.assertEqual(window.application_palette_name, "Raytheon")
            self.assertTrue(
                window._application_palette_actions["Raytheon"].isChecked()
            )
            self.assertEqual(
                settings.values[
                    grim_cut_gui.APPLICATION_PALETTE_SETTINGS_KEY
                ],
                "Raytheon",
            )
            self.assertEqual(settings.sync_count, 1)
            self.assertEqual(
                grim_cut_gui.normalize_application_palette_name(
                    "Raytheon-inspired"
                ),
                "Raytheon",
            )
        finally:
            window.deleteLater()
            self.app.processEvents()

    def setUp(self) -> None:
        self.feature_service = _FakeFeatureWorkflow()
        self.ghost_patch = mock.patch.object(
            grim_cut_gui, "GhostIntegrationWidget", _FakeGhostIntegration
        )
        self.feature_patch = mock.patch.object(
            grim_cut_gui,
            "load_ghost_module",
            return_value=self.feature_service,
        )
        self.freddy_patch = mock.patch.object(
            grim_cut_gui, "FreddyIntegrationWidget", _FakeFreddyIntegration
        )
        self.runs_patch = mock.patch.object(
            grim_cut_gui, "RunsWorkspace", _FakeRunsWorkspace
        )
        self.ghost_patch.start()
        self.feature_patch.start()
        self.freddy_patch.start()
        self.runs_patch.start()
        self.app_settings = _MemorySettings()
        self.window = _RecordingWindow(settings=self.app_settings)

    def tearDown(self) -> None:
        self.window.ghost_integration.running = False
        self.window.freddy_integration.running = False
        self.window.runs_workspace.running = False
        self.window.deleteLater()
        self.app.processEvents()
        self.runs_patch.stop()
        self.freddy_patch.stop()
        self.feature_patch.stop()
        self.ghost_patch.stop()

    def _wait_for_background(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while self.window._background_job_active() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertFalse(self.window._background_job_active())

    def test_dataset_and_parameters_share_a_draggable_vertical_splitter(self) -> None:
        splitter = self.window.dataset_parameter_splitter
        self.assertEqual(splitter.orientation(), Qt.Vertical)
        self.assertEqual(splitter.count(), 2)
        self.assertEqual(splitter.widget(0).objectName(), "datasetsSection")
        self.assertEqual(splitter.widget(1).objectName(), "parametersSection")
        self.assertFalse(splitter.childrenCollapsible())
        self.assertTrue(splitter.isAncestorOf(self.window.table))
        self.assertTrue(splitter.isAncestorOf(self.window.list_az))
        self.assertFalse(hasattr(self.window, "lbl_dataset_selection_summary"))

        self.window.show()
        self.app.processEvents()
        splitter.setSizes([220, 440])
        self.app.processEvents()
        parameters_larger = splitter.sizes()
        self.assertGreater(parameters_larger[1], parameters_larger[0])
        splitter.setSizes([440, 220])
        self.app.processEvents()
        datasets_larger = splitter.sizes()
        self.assertGreater(datasets_larger[0], datasets_larger[1])

    def test_dataset_operations_hide_inspect_and_audit_controls(self) -> None:
        audit_buttons = [
            button
            for button in self.window.findChildren(QToolButton)
            if button.text() == "Audit / QA"
        ]
        inspect_categories = [
            label
            for label in self.window.findChildren(QLabel)
            if label.objectName() == "opsCategory" and label.text() == "Inspect"
        ]
        self.assertEqual(audit_buttons, [])
        self.assertEqual(inspect_categories, [])
        self.assertFalse(hasattr(self.window, "btn_audit"))

    def test_compare_sector_bar_is_compact_hidden_and_defaults_show_all_off(self) -> None:
        context = self.window._plot_contexts["plotting"]
        self.assertEqual(context.compare_sector_bar.objectName(), "compareSectorBar")
        self.assertTrue(context.compare_sector_bar.isHidden())
        self.assertEqual(context.spin_compare_az_min.objectName(), "compareAzimuthMin")
        self.assertEqual(context.spin_compare_az_max.objectName(), "compareAzimuthMax")
        self.assertEqual(
            context.chk_compare_show_all_azimuths.text(),
            "Show all azimuths",
        )
        self.assertFalse(context.chk_compare_show_all_azimuths.isChecked())

    def test_isar_exports_are_bound_to_isar_inputs_not_other_plot_renders(self) -> None:
        self.window._activate_plot_tab("isar")
        context = self.window._plot_contexts["isar"]
        self.window._isar_input_revision = 10
        self.window._isar_view_revision = 20
        self.window._last_isar_completed_input_revision = 10
        self.window._last_isar_completed_view_revision = 20
        self.window._last_isar_figure_token = self.window.plot_figure
        self.window._last_isar_artifact = ([{"result": True}], {"schema": "test"})
        context.btn_export_isar_result.setEnabled(True)

        # The ordinary plotting generation is global legacy state, but its
        # changes must not make a still-current ISAR canvas look stale.
        self.window._start_plot_render()
        self.assertTrue(self.window._isar_numerical_result_is_current())
        self.assertTrue(self.window._isar_figure_is_current())

        # A deferred numerical setting edit invalidates both exports before
        # Apply, so the old image cannot be mistaken for the new recipe.
        next_index = (context.combo_isar_window.currentIndex() + 1) % max(
            context.combo_isar_window.count(), 1
        )
        context.combo_isar_window.setCurrentIndex(next_index)
        self.assertFalse(self.window._isar_numerical_result_is_current())
        self.assertFalse(self.window._isar_figure_is_current())
        self.assertIsNone(self.window._last_isar_artifact)
        self.assertFalse(context.btn_export_isar_result.isEnabled())

    def test_isar_view_only_edit_preserves_numerical_artifact(self) -> None:
        self.window._activate_plot_tab("isar")
        context = self.window._plot_contexts["isar"]
        self.window._isar_input_revision = 3
        self.window._isar_view_revision = 4
        self.window._last_isar_completed_input_revision = 3
        self.window._last_isar_completed_view_revision = 4
        self.window._last_isar_figure_token = self.window.plot_figure
        artifact = ([{"result": True}], {"schema": "test"})
        self.window._last_isar_artifact = artifact

        context.chk_isar_square.setChecked(not context.chk_isar_square.isChecked())
        self.assertTrue(self.window._isar_numerical_result_is_current())
        self.assertFalse(self.window._isar_figure_is_current())
        self.assertIs(self.window._last_isar_artifact, artifact)

    def test_clear_isar_canvas_invalidates_only_the_figure_export(self) -> None:
        self.window._activate_plot_tab("isar")
        self.window._isar_input_revision = 3
        self.window._isar_view_revision = 4
        self.window._last_isar_completed_input_revision = 3
        self.window._last_isar_completed_view_revision = 4
        self.window._last_isar_figure_token = self.window.plot_figure
        artifact = ([{"result": True}], {"schema": "test"})
        self.window._last_isar_artifact = artifact

        self.window._clear_plot()

        self.assertTrue(self.window._isar_numerical_result_is_current())
        self.assertFalse(self.window._isar_figure_is_current())
        self.assertIs(self.window._last_isar_artifact, artifact)
        self.assertFalse(self.window.plot_ax.images)

    def test_plot_recorder_specs_are_owned_by_each_canvas(self) -> None:
        plotting_spec = (
            "supported", ("plot-ref",), ("Plotting",), "frequency", {}
        )
        isar_spec = (
            "supported", ("isar-ref",), ("ISAR",), "isar_image", {}
        )
        self.window._activate_plot_tab("plotting")
        self.window.last_python_plot_spec = plotting_spec
        self.window._activate_plot_tab("isar")
        self.window.last_python_plot_spec = isar_spec

        with mock.patch.object(
            self.window.python_recorder, "record_plot"
        ) as record_plot:
            self.window._activate_plot_tab("plotting")
            self.assertTrue(self.window._emit_last_successful_python_plot())
            self.assertEqual(record_plot.call_args.kwargs["mode"], "frequency")

            self.window._activate_plot_tab("isar")
            self.assertTrue(self.window._emit_last_successful_python_plot())
            self.assertEqual(record_plot.call_args.kwargs["mode"], "isar_image")

    def test_ctrl_select_same_active_dataset_keeps_isar_result_current(self) -> None:
        first = _grid(1.0)
        second = _grid(2.0)
        self.window._add_dataset_row(first, "First", "", "first.grim")
        self.window._add_dataset_row(second, "Second", "", "second.grim")
        first_row_dataset = self.window.table.item(0, 0).data(Qt.UserRole)
        selection = self.window.table.selectionModel()
        first_index = self.window.table.model().index(0, 0)
        second_index = self.window.table.model().index(1, 0)
        self.window.table.setCurrentIndex(first_index)
        selection.select(
            first_index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )
        self.app.processEvents()
        self.assertIs(self.window.active_dataset, first_row_dataset)
        revision = self.window._isar_input_revision

        selection.select(
            second_index, QItemSelectionModel.Select | QItemSelectionModel.Rows
        )
        self.app.processEvents()

        self.assertIs(self.window.active_dataset, first_row_dataset)
        self.assertEqual(self.window._isar_input_revision, revision)

    def test_tabs_have_one_canonical_assembly_workspace(self) -> None:
        labels = [
            self.window.main_tabs.tabText(index)
            for index in range(self.window.main_tabs.count())
        ]
        self.assertEqual(
            labels,
            [
                "Plotting",
                "ISAR",
                "FREDDY",
                "GHOST",
                "Assembly",
                "PPT",
                "Runs",
                "Python",
            ],
        )
        documented_order = "**" + " | ".join(labels) + "**"
        repository_root = Path(__file__).resolve().parents[1]
        for readme in (repository_root / "README.md", Path(__file__).with_name("README.md")):
            documented_text = " ".join(
                readme.read_text(encoding="utf-8").split()
            )
            self.assertIn(
                documented_order,
                documented_text,
                f"{readme} does not document the application's actual tab order",
            )

        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.freddy_integration), 2
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.ghost_integration), 3
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.assembly_workspace), 4
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.ppt_workspace), 5
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.runs_workspace), 6
        )
        self.assertNotIn("ppt", self.window._plot_contexts)

        panels = self.window.findChildren(AssemblyTreePanel)
        self.assertEqual(panels, [self.window.assembly_workspace.assembly_tree_panel])
        assembly_buttons = [
            button
            for button in self.window.findChildren(QToolButton)
            if button.text() == "Assembly Tree"
        ]
        self.assertEqual(assembly_buttons, [])
        for context in self.window._plot_contexts.values():
            self.assertFalse(hasattr(context, "assembly_tree_panel"))
            self.assertFalse(hasattr(context, "btn_assembly_tree"))
        self.assertIs(
            self.window.feature_assembly_panel.service(), self.feature_service
        )
        left_tabs = self.window.assembly_workspace.left_tabs
        self.assertEqual(
            [left_tabs.tabText(index) for index in range(left_tabs.count())],
            ["Body", "Point Features", "Line Features", "Review"],
        )
        self.assertIs(
            left_tabs.currentWidget(),
            self.window.feature_assembly_panel.body_step_page,
        )

    def test_clear_rebuilds_one_full_plot_axis_and_limits_keep_small_values(self):
        self.window.plot_figure.clear()
        first = self.window.plot_figure.add_subplot(121)
        second = self.window.plot_figure.add_subplot(122)
        self.window.plot_ax = first
        self.window.plot_axes = [first, second]

        self.window._clear_plot()

        self.assertEqual(self.window.plot_figure.axes, [self.window.plot_ax])
        self.assertIsNone(self.window.plot_axes)
        self.assertGreaterEqual(self.window.spin_plot_ymin.decimals(), 12)
        self.assertGreaterEqual(self.window.spin_plot_zmin.decimals(), 12)
        self.window.spin_plot_ymin.setValue(4.0e-4)
        self.window.spin_plot_zmin.setValue(1.0e-8)
        self.assertAlmostEqual(self.window.spin_plot_ymin.value(), 4.0e-4)
        self.assertAlmostEqual(self.window.spin_plot_zmin.value(), 1.0e-8)

    def test_dark_theme_explicitly_styles_logs_and_workspace_scroll_surfaces(self) -> None:
        qss = self.window.styleSheet().lower()
        self.assertIn("qplaintextedit", qss)
        self.assertIn("background: #0b1222", qss)
        self.assertIn("color: #dbeafe", qss)
        self.assertIn("qscrollarea#runscontrolsscroll", qss)
        self.assertIn("qscrollarea#pptcontrolsscroll", qss)
        self.assertEqual(
            self.window.ppt_workspace.controls_content.objectName(),
            "pptControlsContent",
        )

    def test_application_palette_switch_is_global_persistent_and_nonintrusive(self) -> None:
        plotting = self.window._plot_contexts["plotting"]
        isar = self.window._plot_contexts["isar"]
        plotting.combo_colormap.setCurrentText("plasma")
        isar.combo_colormap.setCurrentText("cividis")
        isar.plot_bg_color = "#123456"

        self.window.main_tabs.setCurrentWidget(self.window.tab_simple_plots)
        active_index = self.window.main_tabs.currentIndex()
        plotting.settings_frame.show()
        plotting.settings_frame.filter_edit.setText("plot color")
        self.app.processEvents()

        self.window._application_palette_actions["Light"].trigger()
        self.app.processEvents()

        light = grim_cut_gui.APPLICATION_PALETTES["Light"]
        self.assertEqual(self.window.application_palette_name, "Light")
        self.assertEqual(
            self.app_settings.values[
                grim_cut_gui.APPLICATION_PALETTE_SETTINGS_KEY
            ],
            "Light",
        )
        self.assertEqual(self.app_settings.sync_count, 1)
        self.assertTrue(
            self.window._application_palette_actions["Light"].isChecked()
        )
        self.assertEqual(self.window.main_tabs.currentIndex(), active_index)
        self.assertTrue(plotting.settings_frame.isVisible())
        self.assertEqual(
            plotting.settings_frame.filter_edit.text(), "plot color"
        )
        self.assertEqual(plotting.combo_colormap.currentText(), "plasma")
        self.assertEqual(isar.combo_colormap.currentText(), "cividis")
        self.assertEqual(
            to_hex(plotting.plot_figure.get_facecolor()),
            light["panel_bg"],
        )
        self.assertEqual(
            to_hex(isar.plot_figure.get_facecolor()),
            "#123456",
        )
        self.assertEqual(
            to_hex(
                self.window.assembly_workspace.scene_canvas.figure.get_facecolor()
            ),
            light["panel_bg"],
        )
        self.assertEqual(
            self.window.ghost_integration.palette_calls[-1]["panel_bg"],
            light["panel_bg"],
        )
        self.assertEqual(
            self.window.freddy_integration.palette_calls[-1]["panel_bg"],
            light["panel_bg"],
        )
        self.assertIn(f"background: {light['panel_bg']}", self.window.styleSheet())

        restored = _RecordingWindow(settings=self.app_settings)
        try:
            self.assertEqual(restored.application_palette_name, "Light")
            self.assertTrue(
                restored._application_palette_actions["Light"].isChecked()
            )
            self.assertEqual(self.app_settings.sync_count, 1)
        finally:
            restored.deleteLater()
            self.app.processEvents()

    def test_python_clear_confirms_and_resets_dirty_state(self) -> None:
        self.window.python_recorder._lines.extend(["answer = 42", ""])
        self.window.python_recorder._notify()
        script = self.window.python_recorder.script
        self.assertTrue(self.window._python_script_is_dirty())
        self.assertEqual(
            self.window.main_tabs.tabText(
                self.window.main_tabs.indexOf(self.window.tab_python)
            ),
            "Python*",
        )
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        with mock.patch.object(
            grim_cut_gui.QMessageBox, "question", return_value=buttons.No
        ):
            self.window._clear_python_script()
        self.assertEqual(self.window.python_recorder.script, script)

        with mock.patch.object(
            grim_cut_gui.QMessageBox, "question", return_value=buttons.Yes
        ):
            self.window._clear_python_script()
        self.assertEqual(
            self.window.python_recorder.script, self.window._python_empty_script
        )
        self.assertFalse(self.window._python_script_is_dirty())
        self.assertEqual(
            self.window.main_tabs.tabText(
                self.window.main_tabs.indexOf(self.window.tab_python)
            ),
            "Python",
        )

    def test_python_save_failure_preserves_existing_script(self) -> None:
        self.window.python_recorder._lines.extend(["answer = 42", ""])
        self.window.python_recorder._notify()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "workflow.py"
            target.write_text("original\n", encoding="utf-8")
            with (
                mock.patch.object(
                    grim_cut_gui.QFileDialog,
                    "getSaveFileName",
                    return_value=(str(target), "Python Files (*.py)"),
                ),
                mock.patch.object(
                    grim_cut_gui.os,
                    "replace",
                    side_effect=OSError("publication failed"),
                ),
                mock.patch.object(grim_cut_gui.QMessageBox, "critical"),
            ):
                self.assertFalse(self.window._save_python_script())

            self.assertEqual(target.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(list(Path(temp_dir).glob(".workflow.py.*.tmp")), [])
            self.assertTrue(self.window._python_script_is_dirty())

    def test_ppt_catalog_tracks_stable_dataset_ids_and_main_selection(self) -> None:
        first = _grid(1.0)
        second = _grid(2.0)
        self.window._add_dataset_row(first, "First", "", "first.grim")
        self.window._add_dataset_row(second, "Second", "", "second.grim")
        first_item = self.window.table.item(0, 0)
        second_item = self.window.table.item(1, 0)
        first_id = str(first_item.data(DATASET_ID_ROLE))
        second_id = str(second_item.data(DATASET_ID_ROLE))
        self.assertTrue(first_id)
        self.assertTrue(second_id)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(
            self.window.ppt_workspace.dataset_ids_in_order(),
            (first_id, second_id),
        )

        self.window.ppt_workspace.select_dataset_ids((second_id,))
        first_item.setText("Renamed first")
        self.app.processEvents()
        self.assertEqual(
            self.window.ppt_workspace.selected_dataset_ids(), (second_id,)
        )
        self.assertEqual(
            self.window.ppt_workspace._catalog[first_id].name, "Renamed first"
        )

        self.window.table.clearSelection()
        self.window.table.selectRow(0)
        self.window.ppt_workspace.use_selected_button.click()
        self.assertEqual(
            self.window.ppt_workspace.selected_dataset_ids(), (first_id,)
        )

        self.window.table.clearSelection()
        self.window.table.selectRow(1)
        self.window._delete_selected_datasets()
        self.assertEqual(
            self.window.ppt_workspace.dataset_ids_in_order(), (first_id,)
        )

    def test_feature_input_preview_becomes_visibly_stale_after_edit(self) -> None:
        plan = SimpleNamespace(
            preview_stage="input",
            surface_triangles_cad_m=np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
            ),
            body_profile_rho_z_m=None,
            point_locations_cad_m={"fastener": np.asarray([[0.2, 0.2, 0.0]])},
            line_paths_cad_m={},
        )

        self.window._on_feature_preview_ready(plan)
        self.assertEqual(
            self.window.assembly_workspace.scene_canvas.preview_stage, "input"
        )

        self.window.feature_assembly_panel.preview_stale.emit("CSV changed.")

        self.assertEqual(
            self.window.assembly_workspace.scene_canvas.preview_stage, "stale"
        )
        self.assertIn("CSV changed", self.window.assembly_workspace.lbl_status.text())

    def test_range_cal_button_and_dialog_roles_are_explicit(self) -> None:
        buttons = [
            button
            for button in self.window.findChildren(QToolButton)
            if button.text() == "Range Cal"
        ]
        self.assertEqual(buttons, [self.window.btn_range_cal])
        self.assertIn("signed one-way physical", buttons[0].toolTip())
        self.assertFalse(self.window.btn_dataset_load.isHidden())

        entries = [("DUT", _grid()), ("Measured cylinder", _grid()), ("Exact", _grid())]
        dialog = RangeCalibrationDialog(entries, parent=self.window)
        ok_button = dialog.buttons.button(QDialogButtonBox.Ok)
        self.assertTrue(ok_button.isEnabled())
        dialog.combo_exact.setCurrentIndex(dialog.combo_measured.currentIndex())
        self.assertFalse(ok_button.isEnabled())
        dialog.combo_measured.setCurrentIndex(1)
        dialog.combo_exact.setCurrentIndex(2)
        dialog.spin_offset_m.setValue(0.125)
        dialog.chk_broadcast.setChecked(True)
        self.assertTrue(ok_button.isEnabled())
        params = dialog.get_params()
        self.assertEqual(params["measured"][0], "Measured cylinder")
        self.assertEqual(params["exact"][0], "Exact")
        self.assertAlmostEqual(params["range_offset_m"], 0.125)
        self.assertTrue(params["allow_singleton_angular_broadcast"])
        self.assertFalse(params["convention_attested"])
        dialog.deleteLater()

    def test_support_reference_button_and_dialog_make_roles_and_limits_explicit(self) -> None:
        self.assertEqual(self.window.btn_support_reference.text(), "Support Ref -")
        tooltip = self.window.btn_support_reference.toolTip()
        self.assertIn("target+support minus support-only", tooltip)
        self.assertIn("does not reconstruct a free-space target", tooltip)

        entries = [("Vehicle on support", _grid(3.0)), ("Support only", _grid(1.0))]
        dialog = SupportReferenceDifferenceDialog(entries, parent=self.window)
        ok_button = dialog.buttons.button(QDialogButtonBox.Ok)
        self.assertTrue(ok_button.isEnabled())
        self.assertIn("Exact axes", dialog.compatibility_label.text())
        self.assertIn("calibration ID", dialog.compatibility_label.text())
        self.assertIn(
            "no explicit declaration contradicts", dialog.compatibility_label.text()
        )
        params = dialog.get_params()
        self.assertEqual(params["target"][0], "Vehicle on support")
        self.assertEqual(params["support"][0], "Support only")
        self.assertFalse(params["metadata_attested"])
        self.assertFalse(params["assumptions_attested"])

        dialog.combo_support.setCurrentIndex(0)
        self.assertFalse(ok_button.isEnabled())
        self.assertIn("different rows", dialog.compatibility_label.text())
        dialog.deleteLater()

    def test_support_reference_operation_reviews_and_publishes_unsaved_result(self) -> None:
        combined = _grid(3.0)
        support = _grid(1.0)
        self.window._add_dataset_row(
            combined, "Vehicle on support", "", "combined.grim"
        )
        self.window._add_dataset_row(
            support, "Support only", "", "support.grim"
        )
        selection = self.window.table.selectionModel()
        flags = QItemSelectionModel.Select | QItemSelectionModel.Rows
        selection.select(self.window.table.model().index(0, 0), flags)
        selection.select(self.window.table.model().index(1, 0), flags)
        self.app.processEvents()

        class _AcceptedSupportDialog:
            def __init__(self, entries, parent=None):
                self.parent = parent
                self.entries = list(entries)

            @staticmethod
            def exec():
                return QDialog.Accepted

            def get_params(self):
                return {
                    "target": self.entries[0],
                    "support": self.entries[1],
                    "metadata_attested": True,
                    "assumptions_attested": True,
                }

            @staticmethod
            def deleteLater():
                return None

        def _run_worker_now(_job_name, worker):
            worker.run()
            return True

        with (
            mock.patch.object(
                grim_cut_dataset_mixin,
                "SupportReferenceDifferenceDialog",
                _AcceptedSupportDialog,
            ),
            mock.patch.object(
                self.window,
                "_try_start_background_job",
                side_effect=_run_worker_now,
            ),
        ):
            self.window.btn_support_reference.click()

        self.assertEqual(self.window.table.rowCount(), 3)
        result_item = self.window.table.item(2, 0)
        self.assertEqual(
            result_item.text(),
            "SupportRef[Vehicle on support - Support only]",
        )
        self.assertTrue(result_item.data(DATASET_DIRTY_ROLE))
        self.assertEqual(self.window.table.item(2, 1).text(), "Unsaved")
        result = result_item.data(Qt.UserRole)
        np.testing.assert_allclose(result.rcs, 2.0 + 0.0j)
        provenance = json.loads(result.extra["support_reference_difference_json"])
        self.assertTrue(provenance["not_free_space_target"])
        recorded = self.window.python_recorder.script
        self.assertIn(".support_referenced_difference(", recorded)
        self.assertNotIn("assumptions_attested", recorded)
        self.assertIn("target_label='Vehicle on support'", recorded)

    def test_sentri_elevation_button_is_explicit_and_converts_selected_data(self) -> None:
        self.assertEqual(self.window.btn_sentri_elevation.text(), "SENTRi El→GRIM")
        tooltip = self.window.btn_sentri_elevation.toolTip()
        self.assertIn("elevation = 90° - Theta", tooltip)
        self.assertIn("no interpolation or phase change", tooltip)

        power = np.asarray([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)
        phase = np.asarray([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1)
        source = RcsGrid(
            [0.0],
            [0.0, 90.0, 180.0],
            [10.0],
            ["VV"],
            rcs_power=power,
            rcs_phase=phase,
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "elevation_coordinate_convention": "sentri_theta_top_zero",
            },
            extra={"source_format": "SENTRi compact MHz RCS table"},
        )
        self.window._add_dataset_row(source, "Native SENTRi", "", "sentri.csv")
        self.window.table.selectRow(0)
        self.window.btn_sentri_elevation.click()
        self._wait_for_background()

        self.assertEqual(self.window.table.rowCount(), 2)
        self.assertEqual(
            self.window.table.item(1, 0).text(),
            "Native SENTRi [SENTRi El→GRIM]",
        )
        converted = self.window.table.item(1, 0).data(Qt.UserRole)
        np.testing.assert_allclose(converted.elevations, [-90.0, 0.0, 90.0])
        np.testing.assert_allclose(
            converted.rcs_power, np.take(power, [2, 1, 0], axis=1)
        )
        np.testing.assert_allclose(
            converted.rcs_phase, np.take(phase, [2, 1, 0], axis=1)
        )
        self.assertIn(
            "SENTRi El→GRIM created 1 dataset",
            self.window.status.currentMessage(),
        )

    def test_range_cal_operation_creates_complex_calibrated_dataset(self) -> None:
        truth = _grid(3.0)
        measured_cal = _grid(2.0)
        exact = _grid(1.0)
        dut_measured = _grid(6.0)
        self.window._add_dataset_row(dut_measured, "DUT", "", "dut.grim")
        self.window._add_dataset_row(
            measured_cal, "Measured cylinder", "", "measured.grim"
        )
        self.window._add_dataset_row(exact, "Exact cylinder", "", "exact.grim")
        self.window.table.selectRow(0)
        self.app.processEvents()

        class _AcceptedRangeDialog:
            def __init__(self, _entries, parent=None):
                self.parent = parent

            @staticmethod
            def exec():
                return QDialog.Accepted

            @staticmethod
            def get_params():
                return {
                    "measured": ("Measured cylinder", measured_cal),
                    "exact": ("Exact cylinder", exact),
                    "range_offset_m": 0.0,
                    "allow_singleton_angular_broadcast": False,
                    "convention_attested": True,
                }

            @staticmethod
            def deleteLater():
                return None

        def _run_worker_now(_job_name, worker):
            worker.run()
            return True

        with (
            mock.patch.object(
                grim_cut_dataset_mixin,
                "RangeCalibrationDialog",
                _AcceptedRangeDialog,
            ),
            mock.patch.object(
                self.window,
                "_try_start_background_job",
                side_effect=_run_worker_now,
            ),
        ):
            self.window._range_cal_selected()

        self.assertEqual(self.window.table.rowCount(), 4)
        result_item = self.window.table.item(3, 0)
        self.assertEqual(
            result_item.text(),
            "DUT [Range Cal: Exact cylinder; ΔR +0 m]",
        )
        result = result_item.data(Qt.UserRole)
        np.testing.assert_allclose(result.rcs, truth.rcs)
        self.assertIn("Range Cal created 1 dataset", self.window.status.currentMessage())

    def test_bundled_ghost_backend_is_the_primary_builtin_candidate(self) -> None:
        expected = (
            Path(ghost_integration.__file__).resolve().parents[1]
            / "tools"
            / "GHOST"
            / "Backend"
        ).resolve()
        with mock.patch.dict(
            os.environ, {ghost_integration.GHOST_BACKEND_ENV: ""}, clear=False
        ):
            candidates = list(ghost_integration.ghost_backend_candidates())
            discovered = ghost_integration.discover_ghost_backend()

        self.assertEqual(candidates[0], expected)
        self.assertEqual(discovered, expected)

    def test_bundled_freddy_root_is_the_primary_builtin_candidate(self) -> None:
        expected = (
            Path(freddy_integration.__file__).resolve().parents[1]
            / "tools"
            / "FREDDY"
        ).resolve()
        with mock.patch.dict(
            os.environ, {freddy_integration.FREDDY_ROOT_ENV: ""}, clear=False
        ):
            candidates = list(freddy_integration.freddy_root_candidates())
            discovered = freddy_integration.discover_freddy_root()

        self.assertEqual(candidates[0], expected)
        self.assertEqual(discovered, expected)

    def test_workspace_and_ghost_outputs_enter_existing_dataset_paths(self) -> None:
        self.window.assembly_workspace.files_to_load.emit(["assembly.grim"])
        self.window.ghost_integration.files_exported.emit(
            ["ghost_vv_hh.grim"], "2d"
        )
        self.assertEqual(
            self.window.loaded_path_batches,
            [["assembly.grim"], ["ghost_vv_hh.grim"]],
        )

        start_rows = self.window.table.rowCount()
        self.window.assembly_workspace.platform_built.emit(
            "platform", _grid(1.0), "platform history"
        )
        self.window.assembly_workspace.feature_built.emit(
            "featured", _grid(2.0), "feature history"
        )
        self.assertEqual(self.window.table.rowCount(), start_rows + 2)
        self.assertEqual(
            self.window.table.item(start_rows, 0).text(), "platform"
        )
        self.assertEqual(
            self.window.table.item(start_rows + 1, 0).text(), "featured"
        )

    def test_freddy_outputs_do_not_enter_rcs_dataset_loader(self) -> None:
        self.window.freddy_integration.files_exported.emit(
            ["impedance.csv"], "ibc"
        )
        self.assertEqual(self.window.loaded_path_batches, [])

    def test_freddy_material_handoff_is_typed_directly_to_ghost(self) -> None:
        self.window.freddy_integration.attach_to_ghost_requested.emit(
            "ibc", "nominal_ibc.csv"
        )
        self.assertEqual(
            self.window.ghost_integration.attached_artifacts,
            [("ibc", "nominal_ibc.csv")],
        )
        self.assertEqual(self.window.loaded_path_batches, [])

    def test_busy_imports_queue_fifo_and_deduplicate_casefolded_paths(self) -> None:
        # _RecordingWindow overrides the public hook so ordinary shell-routing
        # tests stay synchronous. Exercise the mixin implementation directly.
        with mock.patch.object(
            self.window, "_background_job_active", return_value=True
        ):
            grim_cut_dataset_mixin.DatasetOpsMixin._handle_files_dropped(
                self.window, ["first.grim", "FIRST.GRIM", "second.grim"]
            )
            grim_cut_dataset_mixin.DatasetOpsMixin._handle_files_dropped(
                self.window, ["SECOND.grim", "third.grim"]
            )

        self.assertEqual(
            [batch[0] for batch in self.window._pending_import_batches],
            [("first.grim", "second.grim"), ("third.grim",)],
        )
        with mock.patch.object(
            self.window, "_start_dataset_import_batch", return_value=True
        ) as start:
            self.window._on_background_thread_finished()
            self.window._on_background_thread_finished()
        self.assertEqual(
            [call.args[0] for call in start.call_args_list],
            [["first.grim", "second.grim"], ["third.grim"]],
        )

    def test_loaded_catalog_prefers_container_over_solver_source_path(self) -> None:
        dataset = _grid()
        dataset.source_path = os.path.abspath("source_geometry.geo")
        container = os.path.abspath("solver_output.grim")
        self.window._ensure_background_worker_state()
        self.window._on_load_worker_finished(
            {
                "loaded": [
                    {
                        "index": 0,
                        "path": container,
                        "file_name": "solver_output.grim",
                        "name": "solver_output",
                        "history": "GHOST solve",
                        "dataset": dataset,
                    }
                ],
                "failed": [],
                "ignored": 0,
                "total_supported": 1,
            }
        )

        self.assertEqual(self.window._dataset_catalog()[0].source, container)
        self.assertEqual(
            self.window.table.item(0, 1).toolTip(), container
        )

    def test_parameter_headers_follow_units_and_axes_are_editable(self) -> None:
        dataset = _grid()
        dataset.units.update(
            {
                "frequency": "Hz",
                "azimuth": "rad",
                "elevation": "rad",
                "angular_coordinate_system": "great_circle",
            }
        )
        self.window._add_dataset_row(dataset, "GC", "Loaded", "gc.grim")
        self.window.table.selectRow(0)
        self.app.processEvents()

        self.assertEqual(self.window.lbl_freq.text(), "Frequency (Hz)")
        self.assertEqual(self.window.lbl_elev.text(), "Pitch (rad)")
        self.assertEqual(self.window.lbl_az.text(), "Aspect (rad)")
        for widget in (
            self.window.list_pol,
            self.window.list_freq,
            self.window.list_elev,
            self.window.list_az,
        ):
            self.assertTrue(
                widget.editTriggers() & QAbstractItemView.DoubleClicked
            )
            self.assertTrue(widget.item(0).flags() & Qt.ItemIsEditable)

        self.window.list_pol.item(0).setText("TOTAL")
        self.app.processEvents()
        edited_item = self.window.table.item(0, 0)
        edited = edited_item.data(Qt.UserRole)
        self.assertEqual(edited.polarizations.tolist(), ["TOTAL"])
        self.assertIs(self.window.active_dataset, edited)
        self.assertTrue(edited_item.data(DATASET_DIRTY_ROLE))
        self.assertEqual(self.window.table.item(0, 1).text(), "Unsaved")
        self.assertIn("Edit polarization axis[0]", edited.history)

    def test_gui_response_edits_preserve_source_history_and_drop_raw_field(self) -> None:
        source = _grid(1.0)
        source.source_path = os.path.abspath("source.grim")
        source.history = "Loaded source"
        source.extra["rcs_amp_real"] = np.ones(source.rcs_power.shape)
        source.extra["body_profile_radius_m"] = np.asarray([1.0, 2.0])

        derived = grim_cut_dataset_mixin._dataset_with_rcs(
            source,
            source.rcs * np.exp(1j * np.deg2rad(15.0)),
            rcs_power=source.rcs_power,
            rcs_domain="complex_amplitude",
        )

        self.assertEqual(derived.source_path, source.source_path)
        self.assertEqual(derived.history, source.history)
        self.assertNotIn("rcs_amp_real", derived.extra)
        np.testing.assert_array_equal(
            derived.extra["body_profile_radius_m"], [1.0, 2.0]
        )

    def test_batch_save_preserves_per_row_provenance_for_shared_grid(self) -> None:
        shared = _grid()
        shared.source_path = os.path.abspath("original_source.grim")
        self.window._add_dataset_row(shared, "First branch", "First operation")
        self.window._add_dataset_row(shared, "Second branch", "Second operation")
        first_history = self.window.table.item(0, 2).text()
        second_history = self.window.table.item(1, 2).text()
        self.assertEqual(first_history, "First operation")
        self.assertEqual(second_history, "Second operation")
        self.assertIsNot(
            self.window.table.item(0, 0).data(Qt.UserRole),
            self.window.table.item(1, 0).data(Qt.UserRole),
        )
        self.assertEqual(
            [entry.source for entry in self.window._dataset_catalog()], ["", ""]
        )

        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(
                self.window._save_rows_to_directory(
                    [0, 1], tmp, dialog_title="Test Save"
                )
            )
            self._wait_for_background()
            first_saved = RcsGrid.load(os.path.join(tmp, "First branch.grim"))
            second_saved = RcsGrid.load(os.path.join(tmp, "Second branch.grim"))
            self.assertEqual(
                [entry.source for entry in self.window._dataset_catalog()],
                [
                    os.path.join(tmp, "First branch.grim"),
                    os.path.join(tmp, "Second branch.grim"),
                ],
            )

        self.assertEqual(first_saved.history, first_history)
        self.assertEqual(second_saved.history, second_history)
        self.assertEqual(self.window.table.item(0, 2).text(), first_history)
        self.assertEqual(self.window.table.item(1, 2).text(), second_history)
        self.assertFalse(self.window._dataset_row_is_dirty(0))
        self.assertFalse(self.window._dataset_row_is_dirty(1))

    def test_dataset_loader_always_finishes_after_pool_setup_failure(self) -> None:
        worker = grim_cut_dataset_mixin._DatasetLoadWorker(
            [(0, "first.grim"), (1, "second.grim")]
        )
        payloads = []
        worker.finished.connect(payloads.append)
        with mock.patch.object(
            grim_cut_dataset_mixin,
            "_recommended_loader_workers",
            side_effect=RuntimeError("pool unavailable"),
        ):
            worker.run()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["loaded"], [])
        self.assertIn("pool unavailable", payloads[0]["failed"][0])

    def test_import_queued_behind_isar_starts_when_isar_becomes_idle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = _grid(1.0).save(os.path.join(tmp, "queued.grim"))
            self.window._isar_busy = True

            grim_cut_dataset_mixin.DatasetOpsMixin._handle_files_dropped(
                self.window, [source_path]
            )

            self.assertEqual(len(self.window._pending_import_batches), 1)
            self.assertEqual(self.window.table.rowCount(), 0)
            self.window._on_isar_compute_done({}, "ISAR cancelled for test")
            self._wait_for_background()

            self.assertEqual(self.window._pending_import_batches, [])
            self.assertEqual(self.window.table.rowCount(), 1)
            loaded = self.window.table.item(0, 0).data(Qt.UserRole)
            self.assertIsInstance(loaded, RcsGrid)
            self.assertEqual(float(loaded.rcs_power.item()), 1.0)

    def test_grim_archive_expansion_caps_parallel_loader_workers(self) -> None:
        tasks = [(0, "first.grim"), (1, "second.grim")]
        expanded = 80 * 1024**2
        with (
            mock.patch.object(os.path, "getsize", return_value=1024**2),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_grim_archive_uncompressed_bytes",
                return_value=expanded,
            ),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_available_memory_bytes",
                return_value=1024**3,
            ),
            mock.patch.object(grim_cut_dataset_mixin.os, "cpu_count", return_value=8),
        ):
            workers = grim_cut_dataset_mixin._recommended_loader_workers(tasks)

        self.assertEqual(workers, 1)

    def test_grim_memory_estimate_reads_uncompressed_archive_metadata(self) -> None:
        payload_size = 2 * 1024**2
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "compressed.grim")
            with zipfile.ZipFile(
                path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("rcs_power.npy", b"\0" * payload_size)

            retained, peak = (
                grim_cut_dataset_mixin._dataset_load_memory_estimate(path)
            )

        self.assertEqual(retained, payload_size)
        self.assertGreater(peak, retained)

    def test_unknown_memory_rejects_unsafe_compressed_grim_batch(self) -> None:
        tasks = [(0, "first.grim"), (1, "second.grim")]
        with (
            mock.patch.object(os.path, "getsize", return_value=1024**2),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_grim_archive_uncompressed_bytes",
                return_value=300 * 1024**2,
            ),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_available_memory_bytes",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(MemoryError, "Load fewer or smaller dataset files"):
                grim_cut_dataset_mixin._recommended_loader_workers(tasks)

    def test_native_gui_save_adapts_compression_from_bounded_sample(self) -> None:
        shape = (1100, 1, 1000, 1)
        rng = np.random.default_rng(1776)
        random_power = rng.random(shape, dtype=np.float32)
        random_phase = rng.uniform(-np.pi, np.pi, size=shape).astype(np.float32)
        noisy = RcsGrid(
            np.arange(shape[0]),
            [0.0],
            np.arange(1, shape[2] + 1),
            ["VV"],
            rcs_power=random_power,
            rcs_phase=random_phase,
        )
        smooth = RcsGrid(
            noisy.azimuths,
            noisy.elevations,
            noisy.frequencies,
            noisy.polarizations,
            rcs_power=np.ones(shape, dtype=np.float32),
            rcs_phase=np.zeros(shape, dtype=np.float32),
        )

        noisy_decision = grim_cut_dataset_mixin._grim_save_compression_decision(noisy)
        smooth_decision = grim_cut_dataset_mixin._grim_save_compression_decision(smooth)
        self.assertFalse(noisy_decision["compressed"])
        self.assertTrue(smooth_decision["compressed"])
        self.assertLessEqual(
            noisy_decision["sample_bytes"],
            grim_cut_dataset_mixin._GRIM_COMPRESSION_SAMPLE_BYTES,
        )
        self.assertLessEqual(
            smooth_decision["sample_bytes"],
            grim_cut_dataset_mixin._GRIM_COMPRESSION_SAMPLE_BYTES,
        )

        with tempfile.TemporaryDirectory() as temporary:
            noisy_path = os.path.join(temporary, "noisy.grim")
            smooth_path = os.path.join(temporary, "smooth.grim")
            decisions: list[dict[str, object]] = []
            grim_cut_dataset_mixin._stage_and_publish_grim_batch(
                [
                    (noisy, noisy_path, "noisy"),
                    (smooth, smooth_path, "smooth"),
                ],
                compression_log=decisions,
            )
            with zipfile.ZipFile(noisy_path) as archive:
                self.assertTrue(
                    all(
                        member.compress_type == zipfile.ZIP_STORED
                        for member in archive.infolist()
                    )
                )
            with zipfile.ZipFile(smooth_path) as archive:
                self.assertTrue(
                    all(
                        member.compress_type == zipfile.ZIP_DEFLATED
                        for member in archive.infolist()
                    )
                )
        self.assertEqual(
            [decision["compressed"] for decision in decisions],
            [False, True],
        )

    def test_batch_save_rejects_sanitized_casefold_collision_before_write(self) -> None:
        self.window._add_dataset_row(_grid(), "Body/A", "first")
        self.window._add_dataset_row(_grid(), "body:a", "second")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                grim_cut_dataset_mixin.QMessageBox, "critical"
            ) as critical:
                result = self.window._save_rows_to_directory(
                    [0, 1], tmp, dialog_title="Test Save"
                )
            self.assertFalse(result)
            self.assertEqual(os.listdir(tmp), [])
        critical.assert_called_once()

    def test_delete_requires_confirmation_for_unsaved_derived_rows(self) -> None:
        start_count = self.window.table.rowCount()
        self.window._add_dataset_row(_grid(), "Unsaved branch", "Derived")
        row = self.window.table.rowCount() - 1
        self.window.table.clearSelection()
        self.window.table.selectRow(row)
        buttons = getattr(
            grim_cut_dataset_mixin.QMessageBox,
            "StandardButton",
            grim_cut_dataset_mixin.QMessageBox,
        )
        with mock.patch.object(
            grim_cut_dataset_mixin.QMessageBox,
            "question",
            return_value=buttons.No,
        ) as question:
            self.window._delete_selected_datasets()
        self.assertEqual(self.window.table.rowCount(), start_count + 1)
        self.assertIn("Unsaved branch", question.call_args.args[2])

        with mock.patch.object(
            grim_cut_dataset_mixin.QMessageBox,
            "question",
            return_value=buttons.Yes,
        ):
            self.window._delete_selected_datasets()
        self.assertEqual(self.window.table.rowCount(), start_count)

    def test_batch_save_confirms_all_existing_replacements_once(self) -> None:
        self.window._add_dataset_row(_grid(1.0), "First", "first")
        self.window._add_dataset_row(_grid(2.0), "Second", "second")
        buttons = getattr(
            grim_cut_dataset_mixin.QMessageBox,
            "StandardButton",
            grim_cut_dataset_mixin.QMessageBox,
        )
        with tempfile.TemporaryDirectory() as tmp:
            for filename in ("First.grim", "Second.grim"):
                with open(os.path.join(tmp, filename), "wb") as stream:
                    stream.write(b"old")
            with mock.patch.object(
                grim_cut_dataset_mixin.QMessageBox,
                "question",
                return_value=buttons.Yes,
            ) as question:
                result = self.window._save_rows_to_directory(
                    [0, 1], tmp, dialog_title="Test Save"
                )

            self.assertTrue(result)
            self._wait_for_background()
            question.assert_called_once()
            self.assertEqual(float(RcsGrid.load(os.path.join(tmp, "First.grim")).rcs_power.item()), 1.0)
            self.assertEqual(float(RcsGrid.load(os.path.join(tmp, "Second.grim")).rcs_power.item()), 4.0)
            self.assertIn("Storage mode:", self.window.status.currentMessage())
            self.assertIn("sampled saving", self.window.status.currentMessage())

    def test_background_save_does_not_mark_a_newer_row_revision_saved(self) -> None:
        original = _grid(1.0)
        self.window._add_dataset_row(original, "Editable", "original")
        row_item = self.window.table.item(0, 0)
        started = threading.Event()
        release = threading.Event()
        real_publish = grim_cut_dataset_mixin._stage_and_publish_grim_batch

        def delayed_publish(entries, **kwargs):
            started.set()
            if not release.wait(5.0):
                raise TimeoutError("test did not release save worker")
            return real_publish(entries, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "Editable.grim")
            with mock.patch.object(
                grim_cut_dataset_mixin,
                "_stage_and_publish_grim_batch",
                side_effect=delayed_publish,
            ):
                self.assertTrue(
                    self.window._save_dataset_plan(
                        [(0, original, target)], dialog_title="Test Save"
                    )
                )
                self.assertTrue(started.wait(2.0))
                newer = _grid(2.0)
                row_item.setData(Qt.UserRole, newer)
                row_item.setData(DATASET_DIRTY_ROLE, True)
                release.set()
                self._wait_for_background()

            self.assertTrue(row_item.data(DATASET_DIRTY_ROLE))
            self.assertIs(row_item.data(Qt.UserRole), newer)
            self.assertEqual(float(RcsGrid.load(target).rcs_power.item()), 1.0)
            self.assertIn("remain unsaved", self.window.status.currentMessage())

    def test_batch_save_retains_prior_backup_when_restore_itself_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "First.grim"
            second = Path(tmp) / "Second.grim"
            first.write_bytes(b"old first")
            second.write_bytes(b"old second")
            original_replace = grim_cut_dataset_mixin.os.replace

            def fail_publish_then_restore(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    destination_path == second
                    and source_path.name.endswith(".staging.grim")
                ):
                    raise OSError("second publication failed")
                if (
                    destination_path == first
                    and source_path.name.endswith(".backup")
                ):
                    raise OSError("first restoration failed")
                return original_replace(source, destination)

            with mock.patch.object(
                grim_cut_dataset_mixin.os,
                "replace",
                side_effect=fail_publish_then_restore,
            ):
                with self.assertRaisesRegex(
                    grim_cut_dataset_mixin._GrimBatchRollbackError,
                    "Retained .grim-backup file",
                ):
                    grim_cut_dataset_mixin._stage_and_publish_grim_batch(
                        [
                            (_grid(1.0), str(first), "first"),
                            (_grid(2.0), str(second), "second"),
                        ]
                    )

            retained = list(Path(tmp).glob(".grim-backup-*.backup"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"old first")
            self.assertEqual(second.read_bytes(), b"old second")

    def test_ptm_and_cst_data_are_accepted_by_main_drop_filter(self) -> None:
        mime = QMimeData()
        mime.setUrls(
            [
                QUrl.fromLocalFile(os.path.abspath("legacy.ptm")),
                QUrl.fromLocalFile(os.path.abspath("far_field.cst_data")),
                QUrl.fromLocalFile(os.path.abspath("notes.docx")),
            ]
        )
        accepted = grim_cut_gui._extract_supported_drop_paths(mime)
        self.assertEqual(
            [os.path.basename(path) for path in accepted],
            ["legacy.ptm", "far_field.cst_data"],
        )

    def test_ptm_export_action_uses_single_slice_writer(self) -> None:
        dataset = _grid(1.0)
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "target.ptm")
            with (
                mock.patch.object(
                    self.window,
                    "_selected_datasets_ordered",
                    return_value=[("target", dataset)],
                ),
                mock.patch.object(
                    grim_cut_gui.QFileDialog,
                    "getSaveFileName",
                    return_value=(destination, "PTM Files (*.ptm)"),
                ),
                mock.patch.object(
                    RcsGrid, "save_ptm", autospec=True, return_value=destination
                ) as save_ptm,
            ):
                self.window._export_ptm_selected()
                self._wait_for_background()

        save_ptm.assert_called_once_with(
            dataset, destination, el_idx=0, pol_idx=0
        )

    def test_pioneer_export_reports_great_circle_incompatibility(self) -> None:
        dataset = _grid(1.0)
        dataset.units["angular_coordinate_system"] = "great_circle"
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "ambiguous.pio")
            with (
                mock.patch.object(
                    self.window,
                    "_selected_datasets_ordered",
                    return_value=[("great-circle", dataset)],
                ),
                mock.patch.object(
                    grim_cut_gui.QFileDialog,
                    "getSaveFileName",
                    return_value=(destination, "Pioneer Files (*.pio)"),
                ),
            ):
                self.window._export_pio_selected()
                self._wait_for_background()

        self.assertIn(
            "cannot represent", self.window.status.currentMessage()
        )

    def test_ptm_batch_rejects_duplicate_targets_before_writing(self) -> None:
        dataset = _grid(1.0)
        plans = [
            ("first/name", dataset, "same_name", 0, 0),
            ("second:name", dataset, "same_name", 0, 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(RcsGrid, "save_ptm", autospec=True) as writer:
                with self.assertRaisesRegex(ValueError, "same file more than once"):
                    self.window._write_ptm_batch(tmp, plans)
            writer.assert_not_called()
            self.assertEqual(os.listdir(tmp), [])

    def test_pioneer_batch_rejects_duplicate_targets_before_writing(self) -> None:
        dataset = _grid(1.0)
        plans = [
            ("first/name", dataset, "same_name", 0, 0),
            ("second:name", dataset, "same_name", 0, 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(RcsGrid, "save_pio", autospec=True) as writer:
                with self.assertRaisesRegex(ValueError, "same file more than once"):
                    self.window._write_pio_batch(tmp, plans)
            writer.assert_not_called()
            self.assertEqual(os.listdir(tmp), [])

    def test_ptm_batch_rolls_back_a_late_publication_failure(self) -> None:
        dataset = _grid(1.0)
        plans = [
            ("first", dataset, "first", 0, 0),
            ("second", dataset, "second", 0, 0),
        ]
        real_replace = grim_cut_dataset_mixin.os.replace
        publications = 0

        def write_stage(_dataset, path, **_kwargs):
            Path(path).write_bytes(b"ptm")
            return path

        def fail_second(source, destination):
            nonlocal publications
            if str(destination).casefold().endswith(".ptm"):
                publications += 1
                if publications == 2:
                    raise OSError("injected PTM publication failure")
            return real_replace(source, destination)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(RcsGrid, "save_ptm", autospec=True, side_effect=write_stage),
                mock.patch.object(grim_cut_dataset_mixin.os, "replace", side_effect=fail_second),
            ):
                with self.assertRaisesRegex(OSError, "injected PTM"):
                    self.window._write_ptm_batch(tmp, plans)
            self.assertEqual(os.listdir(tmp), [])

    def test_pioneer_batch_rolls_back_a_late_publication_failure(self) -> None:
        dataset = _grid(1.0)
        plans = [
            ("first", dataset, "first", 0, 0),
            ("second", dataset, "second", 0, 0),
        ]
        real_replace = grim_cut_dataset_mixin.os.replace
        publications = 0

        def write_stage(_dataset, path, **_kwargs):
            Path(path).write_bytes(b"pio")
            return path

        def fail_second(source, destination):
            nonlocal publications
            if str(destination).casefold().endswith(".pio"):
                publications += 1
                if publications == 2:
                    raise OSError("injected Pioneer publication failure")
            return real_replace(source, destination)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(RcsGrid, "save_pio", autospec=True, side_effect=write_stage),
                mock.patch.object(grim_cut_dataset_mixin.os, "replace", side_effect=fail_second),
            ):
                with self.assertRaisesRegex(OSError, "injected Pioneer"):
                    self.window._write_pio_batch(tmp, plans)
            self.assertEqual(os.listdir(tmp), [])

    def test_duplicate_preserves_ptm_physics_metadata(self) -> None:
        dataset = _grid(1.0)
        dataset.units.update(
            {
                "angular_coordinate_system": "great_circle",
                "angular_roll_deg": 12.5,
                "angular_tilt_deg": -1.0,
            }
        )
        dataset.extra.update(
            {
                "angular_coordinate_system": "great_circle",
                "ptm_roll": 12.5,
                "ptm_tilt": -1.0,
                "test_array": np.asarray([1.0, 2.0]),
            }
        )
        start_row = self.window.table.rowCount()
        with mock.patch.object(
            self.window,
            "_selected_datasets_ordered",
            return_value=[("PTM cut", dataset)],
        ):
            self.window._duplicate_selected()
            self._wait_for_background()

        duplicate = self.window.table.item(start_row, 0).data(Qt.UserRole)
        self.assertEqual(duplicate.angular_coordinate_system(), "great_circle")
        self.assertEqual(
            duplicate.angular_frame_orientation_deg(), (12.5, -1.0)
        )
        np.testing.assert_array_equal(duplicate.extra["test_array"], [1.0, 2.0])
        self.assertIsNot(duplicate.extra["test_array"], dataset.extra["test_array"])

    def test_gc_to_conic_allows_only_exact_equatorial_copol_relabel(self) -> None:
        azimuths = np.asarray([179.0, -179.0, 0.0])
        field = np.arange(12, dtype=float).reshape(3, 1, 2, 2) + 1j
        dataset = RcsGrid(
            azimuths,
            [0.0],
            [9.0, 10.0],
            ["VV", "HH"],
            rcs=field,
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "great_circle",
                "great_circle_coordinate_convention": GRIM_GC_CONVENTION,
                "angular_roll_deg": 0.0,
                "angular_tilt_deg": 0.0,
            },
            extra={
                "angular_coordinate_system": "great_circle",
                "ptm_cut_type": "GC",
                "ptm_roll": 0.0,
                "ptm_tilt": 0.0,
                "phase_reference": "exp(-jkr)",
                "ptm_subject": "archive me",
            },
        )

        converted, suffix, _ = self.window._conic_gc_relabel(
            dataset, "gc_to_conic"
        )
        self.assertEqual(suffix, "equator")
        self.assertEqual(converted.angular_coordinate_system(), "conic")
        np.testing.assert_array_equal(converted.azimuths, [-179.0, 0.0, 179.0])
        np.testing.assert_array_equal(converted.elevations, [0.0])
        np.testing.assert_allclose(
            converted.rcs, field[[1, 2, 0], :, :, :],
            rtol=1.0e-14, atol=1.0e-14,
        )
        self.assertNotIn("angular_roll_deg", converted.units)
        self.assertNotIn("angular_tilt_deg", converted.units)
        self.assertNotIn("angular_coordinate_system", converted.extra)
        self.assertNotIn("ptm_cut_type", converted.extra)
        self.assertEqual(converted.extra["phase_reference"], "exp(-jkr)")
        self.assertEqual(converted.extra["ptm_subject"], "archive me")

        dataset.units["angular_roll_deg"] = 1.0
        with self.assertRaisesRegex(ValueError, "roll=tilt=0"):
            self.window._conic_gc_relabel(dataset, "gc_to_conic")

        dataset.units["angular_roll_deg"] = 0.0
        cross_pol = RcsGrid(
            dataset.azimuths,
            dataset.elevations,
            dataset.frequencies,
            ["VH", "HV"],
            rcs=field,
            units=dict(dataset.units),
        )
        with self.assertRaisesRegex(ValueError, "VV/HH only"):
            self.window._conic_gc_relabel(cross_pol, "gc_to_conic")

        conic_source = RcsGrid(
            [0.0, 90.0, 180.0],
            [0.0],
            dataset.frequencies,
            dataset.polarizations,
            rcs=field,
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "conic",
            },
        )
        converted_gc, _, _ = self.window._conic_gc_relabel(
            conic_source, "conic_to_gc"
        )
        self.assertEqual(converted_gc.angular_coordinate_system(), "great_circle")
        self.assertEqual(
            converted_gc.great_circle_coordinate_convention(), GRIM_GC_CONVENTION
        )

    def test_conic_gc_dialog_is_symmetric_without_attestation_gate(self):
        conic_dialog = ConicGCDialog(source_coordinate_system="conic")
        self.assertEqual(
            conic_dialog.get_params()["direction"], "conic_to_gc"
        )
        self.assertFalse(conic_dialog._radio_regrid.isEnabled())
        self.assertFalse(
            conic_dialog.get_params()["attest_legacy_ptm_convention"]
        )
        conic_dialog.deleteLater()

        legacy_dialog = ConicGCDialog(
            source_coordinate_system="great_circle",
            source_gc_convention="legacy_ptm_unspecified",
        )
        self.assertEqual(
            legacy_dialog.get_params()["direction"], "gc_to_conic"
        )
        self.assertFalse(
            legacy_dialog.get_params()["attest_legacy_ptm_convention"]
        )
        legacy_dialog.deleteLater()

    def test_wedge_dialog_records_axis_assumption_and_only_offers_regrid(self):
        dialog = WedgeConicDialog()
        self.assertEqual(dialog.get_params()["mode"], "regrid")
        self.assertFalse(dialog.get_params()["attest_wedge_axes"])
        self.assertFalse(
            dialog.get_params()["assume_missing_cross_pol_zero"]
        )
        dialog._chk_cross_zero.setChecked(True)
        self.assertFalse(dialog.get_params()["attest_wedge_axes"])
        self.assertTrue(
            dialog.get_params()["assume_missing_cross_pol_zero"]
        )
        dialog.deleteLater()

    def test_feature_panel_preview_and_output_use_workspace_paths(self) -> None:
        plan = SimpleNamespace(
            surface_triangles_cad_m=np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
            ),
            body_profile_rho_z_m=None,
            point_locations_cad_m={"antenna": np.asarray([[0.2, 0.2, 0.0]])},
            line_paths_cad_m={
                "gap": {"g1": np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])}
            },
        )

        self.window.feature_assembly_panel.preview_ready.emit(plan)
        self.assertIn(
            "feature-assembly/points/antenna",
            self.window.assembly_workspace.group_ids,
        )
        self.assertIn(
            "feature-assembly/lines/gap",
            self.window.assembly_workspace.group_ids,
        )

        self.window.feature_assembly_panel.feature_built.emit("assembled.grim")
        self.assertEqual(self.window.loaded_path_batches, [["assembled.grim"]])

    def test_running_ghost_solve_blocks_close_and_focuses_solver(self) -> None:
        self.window.ghost_integration.running = True
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.ghost_integration
        )
        self.assertTrue(self.window.ghost_integration.focus_called)

    def test_native_workspace_cancel_blocks_unified_close(self) -> None:
        request_close = mock.Mock(return_value=False)
        self.window.ghost_integration.workspace = SimpleNamespace(
            request_close=request_close
        )
        event = QCloseEvent()
        self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        request_close.assert_called_once_with(self.window)
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.ghost_integration
        )

    def test_unsaved_feature_recipe_can_cancel_unified_close(self) -> None:
        event = QCloseEvent()
        with mock.patch.object(
            self.window.feature_assembly_panel,
            "request_close",
            return_value=False,
        ) as request_close:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        request_close.assert_called_once_with(self.window)
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.assembly_workspace
        )

    def test_unsaved_python_script_can_cancel_unified_close(self) -> None:
        self.window.python_recorder._lines.extend(["value = 1", ""])
        self.window.python_recorder._notify()
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        event = QCloseEvent()
        with mock.patch.object(
            grim_cut_gui.QMessageBox, "warning", return_value=buttons.Cancel
        ) as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertIn("Python recorder", warning.call_args.args[2])
        self.assertIs(self.window.main_tabs.currentWidget(), self.window.tab_python)

    def test_accepted_unified_close_disposes_ppt_preview_directory(self) -> None:
        preview_directory = Path(self.window.ppt_workspace._preview_temp.name)
        (preview_directory / "marker.png").write_bytes(b"preview")
        event = QCloseEvent()
        self.window.closeEvent(event)

        self.assertTrue(event.isAccepted())
        self.assertFalse(preview_directory.exists())

    def test_running_ppt_export_blocks_close_and_focuses_workspace(self) -> None:
        event = QCloseEvent()
        with (
            mock.patch.object(
                self.window.ppt_workspace, "job_is_running", return_value=True
            ),
            mock.patch.object(
                self.window.ppt_workspace,
                "busy_operation",
                return_value="PowerPoint report export",
            ),
            mock.patch.object(
                self.window.ppt_workspace, "focus_workspace"
            ) as focus_workspace,
            mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        focus_workspace.assert_called_once()
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.ppt_workspace
        )

    def test_running_dataset_job_blocks_close(self) -> None:
        event = QCloseEvent()
        self.window._background_worker_name = "Range Cal"
        with (
            mock.patch.object(
                self.window, "_background_job_active", return_value=True
            ),
            mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertIn("Range Cal", warning.call_args.args[2])
        self.assertIn("Range Cal", self.window.status.currentMessage())

    def test_running_dataset_job_precedes_dirty_discard_close(self) -> None:
        self.window._add_dataset_row(_grid(), "Unsaved result", "Derived")
        self.window._background_worker_name = "Dataset save"
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        event = QCloseEvent()
        with (
            mock.patch.object(
                self.window, "_background_job_active", return_value=True
            ),
            mock.patch.object(
                grim_cut_gui.QMessageBox,
                "warning",
                return_value=buttons.Discard,
            ) as warning,
            mock.patch.object(self.window.ppt_workspace, "dispose") as dispose,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], "Dataset Task Still Running")
        self.assertNotIn("Unsaved Datasets", warning.call_args.args)
        self.assertIn("Dataset save", self.window.status.currentMessage())
        dispose.assert_not_called()

    def test_completed_dataset_thread_does_not_block_close(self) -> None:
        completed_thread = QThread(self.window)
        self.window._background_worker_thread = completed_thread
        self.window._background_worker_name = "Completed dataset save"
        self.assertFalse(completed_thread.isRunning())

        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertTrue(event.isAccepted())
        warning.assert_not_called()

    def test_isar_worker_is_cancelled_and_blocks_close_until_done(self) -> None:
        current_cancel = threading.Event()
        pending_cancel = threading.Event()
        self.window._isar_busy = True
        self.window._isar_cancel_event = current_cancel
        self.window._isar_pending = {"_cancel_event": pending_cancel}
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertTrue(current_cancel.is_set())
        self.assertTrue(pending_cancel.is_set())
        self.assertIsNone(self.window._isar_pending)
        self.assertIs(self.window.main_tabs.currentWidget(), self.window.tab_isar)
        warning.assert_called_once()

    def test_unsaved_derived_dataset_blocks_close_when_cancelled(self) -> None:
        self.window._add_dataset_row(_grid(), "Unsaved result", "Derived")
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        event = QCloseEvent()
        with mock.patch.object(
            grim_cut_gui.QMessageBox, "warning", return_value=buttons.Cancel
        ) as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertIn("Unsaved result", warning.call_args.args[2])

    def test_close_waits_for_async_unsaved_dataset_save(self) -> None:
        self.window._add_dataset_row(_grid(), "Unsaved result", "Derived")
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        event = QCloseEvent()
        with (
            mock.patch.object(
                grim_cut_gui.QMessageBox, "warning", return_value=buttons.Save
            ),
            mock.patch.object(
                grim_cut_gui.QFileDialog,
                "getExistingDirectory",
                return_value="C:/safe-save-target",
            ),
            mock.patch.object(
                self.window, "_save_rows_to_directory", return_value=True
            ) as start_save,
            mock.patch.object(self.window.ppt_workspace, "dispose") as dispose,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        start_save.assert_called_once()
        dispose.assert_not_called()
        self.assertIn(
            "Close GRIM again after the save completes",
            self.window.status.currentMessage(),
        )

    def test_running_feature_job_blocks_close_on_assembly_tab(self) -> None:
        event = QCloseEvent()
        with (
            mock.patch.object(
                self.window.feature_assembly_panel,
                "job_is_running",
                return_value=True,
            ),
            mock.patch.object(
                self.window.feature_assembly_panel,
                "busy_operation",
                return_value="build",
            ),
            mock.patch.object(
                self.window.feature_assembly_panel,
                "request_cancel",
            ) as request_cancel,
            mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        request_cancel.assert_called_once_with()
        self.assertIn("Safe cancellation", warning.call_args.args[2])
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.assembly_workspace
        )

    def test_running_freddy_job_blocks_close_and_focuses_workspace(self) -> None:
        self.window.freddy_integration.running = True
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.freddy_integration
        )
        self.assertTrue(self.window.freddy_integration.focus_called)

    def test_running_hpc_transfer_blocks_close_but_remote_jobs_do_not(self) -> None:
        self.window.runs_workspace.running = True
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertIn("HPC upload", warning.call_args.args[2])
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.runs_workspace
        )
        self.assertTrue(self.window.runs_workspace.focus_called)

        # A tracked SLURM job is owned by the remote scheduler. Once the
        # foreground SSH process is done, it must not trap GRIM open.
        self.window.runs_workspace.running = False
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)
        self.assertTrue(event.isAccepted())
        warning.assert_not_called()
        self.assertGreater(self.window.runs_workspace.save_count, 0)

    def test_downloaded_hpc_result_tree_reuses_dataset_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "results" / "FRD" / "first.grim"
            second = root / "results" / "OPN" / "second.ptm"
            ignored = root / "results" / "solver.log"
            for path in (first, second, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            self.window.runs_workspace.results_downloaded.emit(str(root))

        self.assertEqual(
            self.window.loaded_path_batches,
            [[str(first), str(second)]],
        )

    def test_branch_drop_uses_canonical_workspace_tree(self) -> None:
        tree = self.window.assembly_workspace.assembly_tree_panel.tree
        root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
        branch = tree._make_node("Payload", _TYPE_BRANCH, parent=root, edit=False)
        leaf = tree._make_leaf("part", _grid(1.5))
        _attach(tree, leaf, branch)
        tree._branch_drag_item = branch

        start_rows = self.window.table.rowCount()
        self.window._on_assembly_branch_dropped("Payload", [])

        self.assertEqual(self.window.table.rowCount(), start_rows + 1)
        self.assertEqual(self.window.table.item(start_rows, 0).text(), "Payload")

    def test_internal_branch_drop_blocks_descendants_but_allows_move_up(self) -> None:
        tree = self.window.assembly_workspace.assembly_tree_panel.tree
        root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
        branch = tree._make_node("Payload", _TYPE_BRANCH, parent=root, edit=False)
        child = tree._make_node("Sensors", _TYPE_BRANCH, parent=branch, edit=False)

        self.assertTrue(_branch_drop_would_create_cycle(branch, child))
        self.assertTrue(_branch_drop_would_create_cycle(branch, branch))
        self.assertFalse(_branch_drop_would_create_cycle(branch, root))
        self.assertFalse(_branch_drop_would_create_cycle(branch, None))


if __name__ == "__main__":
    unittest.main()
