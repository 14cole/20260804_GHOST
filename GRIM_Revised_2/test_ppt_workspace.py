"""Focused regressions for GRIM's PowerPoint workspace."""

from __future__ import annotations

import os
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

    from grim_dataset import RcsGrid
    from ppt_report import (
        DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
        DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
        azimuth_3x2_geometry,
    )
    import ppt_workspace
    from ppt_workspace import DatasetCatalogEntry, GUI_AVAILABLE, PptWorkspace
except (ImportError, RuntimeError) as exc:  # pragma: no cover - dependency-specific
    _IMPORT_ERROR = exc
    GUI_AVAILABLE = False


def _grid(
    *,
    frequencies=tuple(range(1, 8)),
    scale: float = 1.0,
    azimuths=(0.0, 90.0, 180.0, 270.0),
    angle_unit: str = "deg",
):
    azimuths = np.asarray(azimuths, dtype=float)
    elevations = np.asarray((0.0,))
    frequencies = np.asarray(frequencies, dtype=float)
    polarizations = np.asarray(("HH", "VV"))
    shape = (
        len(azimuths),
        len(elevations),
        len(frequencies),
        len(polarizations),
    )
    azimuth_factor = 1.0 + 0.15 * np.arange(shape[0])[:, None, None, None]
    frequency_factor = 1.0 + 0.08 * np.arange(shape[2])[None, None, :, None]
    polarization_factor = np.asarray((1.0, 0.7))[None, None, None, :]
    power = scale * azimuth_factor * frequency_factor * polarization_factor
    power = np.broadcast_to(power, shape).copy()
    return RcsGrid(
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs_power=power,
        rcs_phase=np.zeros(shape),
        units={
            "azimuth": angle_unit,
            "elevation": angle_unit,
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
            "angular_coordinate_system": "conic",
        },
        extra={"phase_reference": "common-center"},
    )


@unittest.skipUnless(
    GUI_AVAILABLE,
    f"PPT workspace GUI dependencies are unavailable: {_IMPORT_ERROR}",
)
class PptWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widgets: list[PptWorkspace] = []

    def tearDown(self) -> None:
        for widget in reversed(self.widgets):
            if widget.job_is_running():
                deadline = time.monotonic() + 3.0
                while widget.job_is_running() and time.monotonic() < deadline:
                    QCoreApplication.processEvents()
            widget.close()
            widget.deleteLater()
        QCoreApplication.processEvents()

    def workspace(self, **kwargs) -> PptWorkspace:
        widget = PptWorkspace(**kwargs)
        self.widgets.append(widget)
        return widget

    def test_bundled_template_is_selected_with_named_layout_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "team-template.pptx"
            template.write_bytes(b"temporary test template")
            with mock.patch.object(
                ppt_workspace,
                "DEFAULT_POWERPOINT_TEMPLATE",
                template,
            ):
                widget = self.workspace()

        self.assertEqual(widget.template_edit.text(), str(template))
        self.assertTrue(widget.azimuth_layout_edit.isEnabled())
        self.assertTrue(widget.frequency_layout_edit.isEnabled())
        self.assertEqual(
            widget.azimuth_layout_edit.text(),
            DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
        )
        self.assertEqual(
            widget.frequency_layout_edit.text(),
            DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
        )
        self.assertEqual(
            widget._selected_template_layouts(),
            {
                "azimuth_3x2": DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                "frequency_single": DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
            },
        )

    def test_missing_or_cleared_template_disables_named_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-template.pptx"
            with mock.patch.object(
                ppt_workspace,
                "DEFAULT_POWERPOINT_TEMPLATE",
                missing,
            ):
                widget = self.workspace()

        self.assertEqual(widget.template_edit.text(), "")
        self.assertFalse(widget.azimuth_layout_edit.isEnabled())
        self.assertFalse(widget.frequency_layout_edit.isEnabled())
        self.assertEqual(widget._selected_template_layouts(), {})

        widget.template_edit.setText("custom.pptx")
        self.assertTrue(widget.azimuth_layout_edit.isEnabled())
        widget.template_edit.clear()
        self.assertFalse(widget.azimuth_layout_edit.isEnabled())
        self.assertEqual(widget._selected_template_layouts(), {})

    @staticmethod
    def entries():
        return (
            DatasetCatalogEntry("a", "Baseline", _grid(scale=1.0), "a.grim"),
            DatasetCatalogEntry("b", "Modified", _grid(scale=1.4), "b.grim"),
        )

    def test_controls_surface_is_theme_addressable_and_dispose_is_idempotent(self):
        widget = self.workspace()
        self.assertEqual(widget.controls_scroll.objectName(), "pptControlsScroll")
        self.assertEqual(widget.controls_content.objectName(), "pptControlsContent")
        label_text = " ".join(
            label.text().casefold() for label in widget.findChildren(QLabel)
        )
        self.assertNotIn("bundled temporary template supplies", label_text)

        preview_directory = Path(widget._preview_temp.name)
        marker = preview_directory / "preview_marker.png"
        marker.write_bytes(b"preview")
        self.assertTrue(preview_directory.is_dir())
        widget.dispose()
        self.assertFalse(preview_directory.exists())
        widget.dispose()

    def test_preview_leaves_page_numbering_to_the_slide_master(self):
        widget = self.workspace()
        widget.set_dataset_catalog((self.entries()[0],))
        widget.select_frequencies((1,))

        self.assertTrue(widget.build_preview())

        figure_text = {
            artist.get_text() for artist in widget.preview_canvas.figure.texts
        }
        self.assertNotIn("1 / 1", figure_text)
        self.assertEqual(widget.page_label.text(), "Slide 1 of 1")

    def test_catalog_preserves_check_state_and_user_order_by_stable_id(self):
        widget = self.workspace()
        first = self.entries()
        widget.set_dataset_catalog(first)
        widget.select_dataset_ids(("b",))

        moved = widget.dataset_list.takeItem(1)
        widget.dataset_list.insertItem(0, moved)
        self.assertEqual(widget.dataset_ids_in_order(), ("b", "a"))

        widget.set_dataset_catalog(
            (
                DatasetCatalogEntry("a", "Baseline renamed", first[0].grid),
                DatasetCatalogEntry("b", "Modified renamed", first[1].grid),
                DatasetCatalogEntry("c", "New run", _grid(scale=2.0)),
            )
        )
        self.assertEqual(widget.dataset_ids_in_order(), ("b", "a", "c"))
        self.assertEqual(widget.selected_dataset_ids(), ("b", "c"))
        self.assertEqual(widget.dataset_list.item(0).text(), "Modified renamed")

        widget.set_dataset_catalog(
            (
                DatasetCatalogEntry("b", "Modified final", first[1].grid),
                DatasetCatalogEntry("c", "New run", _grid(scale=2.0)),
            )
        )
        self.assertEqual(widget.dataset_ids_in_order(), ("b", "c"))
        self.assertEqual(widget.selected_dataset_ids(), ("b", "c"))

    def test_drag_reorder_preserves_every_flat_row_and_its_state(self):
        widget = self.workspace()
        entries = (
            *self.entries(),
            DatasetCatalogEntry("c", "Third", _grid(scale=1.8)),
            DatasetCatalogEntry("d", "Fourth", _grid(scale=2.2)),
        )
        widget.set_dataset_catalog(entries)
        widget.select_dataset_ids(("a", "c", "d"))
        widget.dataset_list.item(1).setSelected(True)
        widget.dataset_list.item(2).setSelected(True)
        widget.dataset_list.setCurrentItem(widget.dataset_list.item(2))

        self.assertTrue(widget.dataset_list.move_rows_to_insertion((1, 2), 0))
        self.assertEqual(widget.dataset_ids_in_order(), ("b", "c", "a", "d"))
        self.assertEqual(widget.dataset_list.count(), 4)
        self.assertEqual(set(widget.dataset_ids_in_order()), {"a", "b", "c", "d"})
        self.assertEqual(widget.selected_dataset_ids(), ("c", "a", "d"))
        self.assertEqual(
            {
                str(item.data(ppt_workspace._CATALOG_ID_ROLE))
                for item in widget.dataset_list.selectedItems()
            },
            {"b", "c"},
        )
        self.assertEqual(
            widget.dataset_list.currentItem().data(ppt_workspace._CATALOG_ID_ROLE),
            "c",
        )

        # A drop in viewport space after the final row moves the same objects
        # to the end without losing their check states or stable IDs.
        moved_rows = tuple(
            widget.dataset_list.row(item)
            for item in widget.dataset_list.selectedItems()
        )
        self.assertTrue(
            widget.dataset_list.move_rows_to_insertion(
                moved_rows,
                widget.dataset_list.count(),
            )
        )
        self.assertEqual(widget.dataset_ids_in_order(), ("a", "d", "b", "c"))
        self.assertEqual(widget.dataset_list.count(), 4)
        self.assertEqual(widget.selected_dataset_ids(), ("a", "d", "c"))
        self.assertEqual(
            {
                str(item.data(ppt_workspace._CATALOG_ID_ROLE))
                for item in widget.dataset_list.selectedItems()
            },
            {"b", "c"},
        )

    def test_dataset_rows_are_drag_sources_not_item_drop_parents(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())

        self.assertEqual(
            widget.dataset_list.dragDropMode(),
            widget.dataset_list.DragDropMode.InternalMove,
        )
        self.assertTrue(widget.dataset_list.viewport().acceptDrops())
        for index in range(widget.dataset_list.count()):
            flags = widget.dataset_list.item(index).flags()
            self.assertTrue(flags & Qt.ItemFlag.ItemIsDragEnabled)
            self.assertFalse(flags & Qt.ItemFlag.ItemIsDropEnabled)

    def test_drop_event_path_moves_rows_and_rejects_external_sources(self):
        widget = self.workspace()
        widget.set_dataset_catalog(
            (
                *self.entries(),
                DatasetCatalogEntry("c", "Third", _grid(scale=1.8)),
            )
        )
        catalog = widget.dataset_list
        catalog.clearSelection()
        catalog.item(0).setSelected(True)
        catalog.setCurrentItem(catalog.item(0))
        order_changes: list[bool] = []
        catalog.order_changed.connect(lambda: order_changes.append(True))

        class FakeDropEvent:
            def __init__(self, source):
                self._source = source
                self.action = None
                self.accepted = False
                self.ignored = False

            def source(self):
                return self._source

            def setDropAction(self, action):
                self.action = action

            def accept(self):
                self.accepted = True

            def ignore(self):
                self.ignored = True

        event = FakeDropEvent(catalog)
        with mock.patch.object(
            catalog,
            "_drop_insertion_row",
            return_value=catalog.count(),
        ) as insertion:
            catalog.dropEvent(event)
        insertion.assert_called_once_with(event)
        self.assertEqual(widget.dataset_ids_in_order(), ("b", "c", "a"))
        self.assertEqual(catalog.count(), 3)
        self.assertEqual(
            catalog.currentItem().data(ppt_workspace._CATALOG_ID_ROLE), "a"
        )
        self.assertEqual(event.action, Qt.DropAction.MoveAction)
        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)
        self.assertEqual(order_changes, [True])

        external = FakeDropEvent(object())
        with mock.patch.object(catalog, "_drop_insertion_row") as insertion:
            catalog.dropEvent(external)
        insertion.assert_not_called()
        self.assertTrue(external.ignored)
        self.assertEqual(widget.dataset_ids_in_order(), ("b", "c", "a"))

    def test_seven_frequency_azimuth_preview_pages_six_then_one(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        self.assertEqual(len(widget.selected_frequencies()), 6)
        widget.select_frequencies(range(1, 8))
        self.assertEqual(len(widget.selected_frequencies()), 7)

        with mock.patch.object(widget.preview_canvas, "render_slide") as render:
            self.assertTrue(widget.build_preview())
            render.assert_called_once()
        plan = widget.preview_plan
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(len(plan.slides), 2)
        self.assertEqual([len(slide.plots) for slide in plan.slides], [6, 1])
        geometry = azimuth_3x2_geometry()
        self.assertEqual(
            tuple(placement.frame for placement in plan.slides[0].plots),
            geometry.plot_frames,
        )
        self.assertEqual(
            [placement.slot_index for placement in plan.slides[0].plots],
            list(range(6)),
        )
        self.assertTrue(widget.next_slide_button.isEnabled())
        with mock.patch.object(widget.preview_canvas, "render_slide") as render:
            widget.next_slide()
            render.assert_called_once()
        self.assertEqual(widget.current_slide_index, 1)
        self.assertTrue(widget.previous_slide_button.isEnabled())
        self.assertFalse(widget.next_slide_button.isEnabled())

    def test_vv_and_hh_choice_builds_separate_plots_in_one_report(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        choices = [
            widget.polarization_combo.itemData(index)
            for index in range(widget.polarization_combo.count())
        ]
        self.assertEqual(choices, ["HH", "VV", "VV and HH"])
        widget.polarization_combo.setCurrentIndex(
            widget.polarization_combo.findData("VV and HH")
        )
        widget.select_frequencies(range(1, 8))

        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        self.assertEqual(len(widget.preview_plan.slides), 4)
        self.assertEqual(
            [len(slide.plots) for slide in widget.preview_plan.slides],
            [6, 1, 6, 1],
        )
        self.assertEqual(
            [slide.title for slide in widget.preview_plan.slides],
            [
                "RCS Report — VV",
                "RCS Report — VV",
                "RCS Report — HH",
                "RCS Report — HH",
            ],
        )
        self.assertTrue(
            all(
                placement.plot.title.endswith("| VV")
                for slide in widget.preview_plan.slides[:2]
                for placement in slide.plots
            )
        )
        self.assertTrue(
            all(
                placement.plot.title.endswith("| HH")
                for slide in widget.preview_plan.slides[2:]
                for placement in slide.plots
            )
        )

        widget.set_plot_kind("frequency")
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        self.assertEqual(len(widget.preview_plan.slides), 2)
        self.assertEqual(
            [slide.plots[0].plot.title.rsplit("|", 1)[-1].strip()
             for slide in widget.preview_plan.slides],
            ["VV", "HH"],
        )
        self.assertEqual(
            [slide.title for slide in widget.preview_plan.slides],
            ["RCS Report — VV", "RCS Report — HH"],
        )
        self.assertTrue(
            all(
                slide.plots[0].plot.title not in slide.title
                for slide in widget.preview_plan.slides
            )
        )

    def test_frequency_azimuth_band_controls_build_percentile_trace(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        widget.set_plot_kind("frequency")
        self.assertFalse(widget.frequency_azimuth_mode_combo.isHidden())
        self.assertFalse(widget.azimuth_combo.isHidden())
        self.assertTrue(widget.azimuth_band_widget.isHidden())

        widget.frequency_azimuth_mode_combo.setCurrentIndex(
            widget.frequency_azimuth_mode_combo.findData("band")
        )
        widget.azimuth_band_min_spin.setValue(90.0)
        widget.azimuth_band_max_spin.setValue(270.0)
        widget.azimuth_percentile_spin.setValue(50.0)
        self.assertTrue(widget.azimuth_combo.isHidden())
        self.assertFalse(widget.azimuth_band_widget.isHidden())
        self.assertEqual(widget.azimuth_band_unit_label.text(), "deg")

        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        self.assertEqual(len(widget.preview_plan.slides), 1)
        plot = widget.preview_plan.slides[0].plots[0].plot
        self.assertIn("P50 across Azimuth [90, 270] deg", plot.title)
        self.assertEqual(len(plot.series), 2)

        # Reversed endpoints are an intentional wrapped band, not a validation
        # error or an implicit endpoint swap.
        widget.azimuth_band_min_spin.setValue(270.0)
        widget.azimuth_band_max_spin.setValue(90.0)
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        self.assertIn("(wrap)", widget.preview_plan.slides[0].plots[0].plot.title)

    def test_azimuth_band_controls_fit_sidebar_and_are_accessibly_labeled(self):
        widget = self.workspace()
        widget.resize(750, 700)
        widget.show()
        widget.set_dataset_catalog(self.entries())
        widget.set_plot_kind("frequency")
        widget.frequency_azimuth_mode_combo.setCurrentIndex(
            widget.frequency_azimuth_mode_combo.findData("band")
        )
        QCoreApplication.processEvents()

        self.assertLessEqual(
            widget.azimuth_band_widget.minimumSizeHint().width(),
            widget.controls_scroll.viewport().width(),
        )
        self.assertLessEqual(
            max(
                child.geometry().right()
                for child in (
                    widget.azimuth_band_min_spin,
                    widget.azimuth_band_max_spin,
                    widget.azimuth_percentile_spin,
                    widget.azimuth_band_unit_label,
                )
            ),
            widget.azimuth_band_widget.contentsRect().right(),
        )
        self.assertIs(
            widget.azimuth_band_min_label.buddy(), widget.azimuth_band_min_spin
        )
        self.assertIs(
            widget.azimuth_band_max_label.buddy(), widget.azimuth_band_max_spin
        )
        self.assertIs(
            widget.azimuth_percentile_label.buddy(),
            widget.azimuth_percentile_spin,
        )
        for control in (
            widget.azimuth_band_min_spin,
            widget.azimuth_band_max_spin,
            widget.azimuth_percentile_spin,
        ):
            self.assertTrue(control.accessibleName())
            self.assertTrue(control.accessibleDescription())
        self.assertIn("sample-weighted", widget.azimuth_band_widget.toolTip())
        self.assertIn("common stored angles", widget.azimuth_band_widget.toolTip())

    def test_band_mode_requires_two_common_azimuth_samples(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        widget.set_plot_kind("frequency")
        band_index = widget.frequency_azimuth_mode_combo.findData("band")
        widget.frequency_azimuth_mode_combo.setCurrentIndex(band_index)
        self.assertEqual(widget.frequency_azimuth_mode_combo.currentData(), "band")

        widget.set_dataset_catalog(
            (
                DatasetCatalogEntry(
                    "single",
                    "Single angle",
                    _grid(azimuths=(15.0,)),
                ),
            )
        )
        self.assertFalse(widget._azimuth_band_available)
        self.assertEqual(widget.frequency_azimuth_mode_combo.currentData(), "exact")
        self.assertFalse(
            widget.frequency_azimuth_mode_combo.model().item(band_index).isEnabled()
        )
        self.assertIn(
            "at least two common azimuth samples",
            widget.frequency_azimuth_mode_combo.toolTip().casefold(),
        )
        self.assertFalse(widget.azimuth_combo.isHidden())
        self.assertTrue(widget.azimuth_band_widget.isHidden())

        # Programmatic callers cannot bypass the disabled GUI item and obtain
        # a vague lower-level min/max error.
        widget.frequency_azimuth_mode_combo.setCurrentIndex(band_index)
        with mock.patch.object(widget.preview_canvas, "render_slide") as render:
            self.assertFalse(widget.build_preview())
        render.assert_not_called()
        self.assertIn("at least two common azimuth samples", widget.last_error)

    def test_band_mode_counts_periodic_seam_alias_as_one_sample(self):
        widget = self.workspace()
        widget.set_dataset_catalog(
            (
                DatasetCatalogEntry(
                    "seam",
                    "Duplicate seam direction",
                    _grid(azimuths=(0.0, 360.0)),
                ),
            )
        )

        band_index = widget.frequency_azimuth_mode_combo.findData("band")
        self.assertFalse(widget._azimuth_band_available)
        self.assertFalse(
            widget.frequency_azimuth_mode_combo.model().item(band_index).isEnabled()
        )
        self.assertIn(
            "at least two common azimuth samples",
            widget.frequency_azimuth_mode_combo.toolTip().casefold(),
        )

    def test_radian_band_endpoints_keep_precision_and_native_step(self):
        widget = self.workspace()
        azimuths = (-math.pi, -math.pi / 3.0, 0.0, math.pi / 3.0, math.pi)
        widget.set_dataset_catalog(
            (
                DatasetCatalogEntry(
                    "rad",
                    "Radians",
                    _grid(azimuths=azimuths, angle_unit="rad"),
                ),
            )
        )
        self.assertEqual(widget.azimuth_band_unit_label.text(), "rad")
        self.assertGreaterEqual(widget.azimuth_band_min_spin.decimals(), 10)
        self.assertGreaterEqual(widget.azimuth_band_max_spin.decimals(), 10)
        self.assertAlmostEqual(
            widget.azimuth_band_min_spin.minimum(), -math.pi, places=9
        )
        self.assertAlmostEqual(
            widget.azimuth_band_max_spin.maximum(), math.pi, places=9
        )
        self.assertAlmostEqual(
            widget.azimuth_band_min_spin.singleStep(), math.pi / 3.0, places=9
        )
        widget.azimuth_band_min_spin.setValue(math.pi / 7.0)
        self.assertAlmostEqual(
            widget.azimuth_band_min_spin.value(), math.pi / 7.0, places=9
        )
        self.assertIn("(rad)", widget.azimuth_band_min_spin.accessibleName())

    def test_all_plots_receive_one_shared_or_explicit_fixed_rcs_scale(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        automatic = {
            placement.plot.y_limits
            for slide in widget.preview_plan.slides
            for placement in slide.plots
        }
        self.assertEqual(len(automatic), 1)
        self.assertNotIn(None, automatic)

        widget.scale_mode_combo.setCurrentIndex(
            widget.scale_mode_combo.findData("fixed")
        )
        widget.y_min_spin.setValue(-55.0)
        widget.y_max_spin.setValue(5.0)
        widget.y_step_spin.setValue(5.0)
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        fixed = {
            placement.plot.y_limits
            for slide in widget.preview_plan.slides
            for placement in slide.plots
        }
        self.assertEqual(fixed, {(-55.0, 5.0)})
        self.assertEqual(
            {
                placement.plot.y_tick_step
                for slide in widget.preview_plan.slides
                for placement in slide.plots
            },
            {5.0},
        )

    def test_fixed_axis_limits_and_ticks_apply_uniformly_without_resampling(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        widget.x_scale_mode_combo.setCurrentIndex(
            widget.x_scale_mode_combo.findData("fixed")
        )
        widget.x_min_spin.setValue(0.0)
        widget.x_max_spin.setValue(360.0)
        widget.x_step_spin.setValue(30.0)
        widget.scale_mode_combo.setCurrentIndex(
            widget.scale_mode_combo.findData("fixed")
        )
        widget.y_min_spin.setValue(-70.0)
        widget.y_max_spin.setValue(10.0)
        widget.y_step_spin.setValue(10.0)
        source_x = tuple(float(value) for value in self.entries()[0].grid.azimuths)
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        plots = [
            placement.plot
            for slide in widget.preview_plan.slides
            for placement in slide.plots
        ]
        self.assertTrue(plots)
        self.assertTrue(all(plot.x_limits == (0.0, 360.0) for plot in plots))
        self.assertTrue(all(plot.x_tick_step == 30.0 for plot in plots))
        self.assertTrue(all(plot.y_limits == (-70.0, 10.0) for plot in plots))
        self.assertTrue(all(plot.y_tick_step == 10.0 for plot in plots))
        self.assertTrue(all(plot.series[0].x == source_x for plot in plots))

    def test_horizontal_axis_settings_are_kept_separately_by_plot_family(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        widget.x_scale_mode_combo.setCurrentIndex(
            widget.x_scale_mode_combo.findData("fixed")
        )
        widget.x_min_spin.setValue(-90.0)
        widget.x_max_spin.setValue(90.0)
        widget.x_step_spin.setValue(15.0)

        widget.set_plot_kind("frequency")
        self.assertEqual(widget.x_scale_mode_combo.currentData(), "automatic")
        self.assertEqual(widget.x_min_spin.value(), 1.0)
        self.assertEqual(widget.x_max_spin.value(), 10.0)
        self.assertIn("GHz", widget.x_scale_label.text())
        widget.x_scale_mode_combo.setCurrentIndex(
            widget.x_scale_mode_combo.findData("fixed")
        )
        widget.x_min_spin.setValue(2.0)
        widget.x_max_spin.setValue(7.0)
        widget.x_step_spin.setValue(0.5)
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        frequency_plot = widget.preview_plan.slides[0].plots[0].plot
        self.assertEqual(frequency_plot.x_limits, (2.0, 7.0))
        self.assertEqual(frequency_plot.x_tick_step, 0.5)

        widget.set_plot_kind("azimuth_rect")
        self.assertEqual(widget.x_scale_mode_combo.currentData(), "fixed")
        self.assertEqual(widget.x_min_spin.value(), -90.0)
        self.assertEqual(widget.x_max_spin.value(), 90.0)
        self.assertEqual(widget.x_step_spin.value(), 15.0)
        self.assertIn("deg", widget.x_scale_label.text())

    def test_legend_modes_choose_master_per_plot_or_none(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())

        def plan_for(mode):
            widget.legend_mode_combo.setCurrentIndex(
                widget.legend_mode_combo.findData(mode)
            )
            with mock.patch.object(widget.preview_canvas, "render_slide"):
                self.assertTrue(widget.build_preview())
            assert widget.preview_plan is not None
            return widget.preview_plan

        master = plan_for("master")
        self.assertEqual(
            [entry.label for entry in master.slides[0].master_legend],
            ["Baseline", "Modified"],
        )
        self.assertTrue(
            all(
                not placement.plot.show_legend
                for slide in master.slides
                for placement in slide.plots
            )
        )

        per_plot = plan_for("per_plot")
        self.assertFalse(per_plot.slides[0].master_legend)
        self.assertTrue(
            all(
                placement.plot.show_legend
                for slide in per_plot.slides
                for placement in slide.plots
            )
        )

        no_legend = plan_for("none")
        self.assertFalse(no_legend.slides[0].master_legend)
        self.assertTrue(
            all(
                not placement.plot.show_legend
                for slide in no_legend.slides
                for placement in slide.plots
            )
        )

    def test_preview_master_legend_is_above_every_plot_layer(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        widget.select_frequencies((1.0,))

        self.assertTrue(widget.build_preview())
        legend_axes = [
            axes
            for axes in widget.preview_canvas.figure.axes
            if axes.get_label() == "GRIM master legend"
        ]
        self.assertEqual(len(legend_axes), 1)
        plot_axes = [
            axes
            for axes in widget.preview_canvas.figure.axes
            if axes is not legend_axes[0] and axes.images
        ]
        self.assertTrue(plot_axes)
        self.assertEqual(legend_axes[0].patch.get_alpha(), 0.0)
        self.assertGreater(
            legend_axes[0].get_zorder(),
            max(axes.get_zorder() for axes in plot_axes),
        )

    def test_excessive_fixed_ticks_fail_before_rendering(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        widget.x_scale_mode_combo.setCurrentIndex(
            widget.x_scale_mode_combo.findData("fixed")
        )
        widget.x_min_spin.setValue(0.0)
        widget.x_max_spin.setValue(360.0)
        widget.x_step_spin.setValue(0.1)
        with mock.patch.object(widget.preview_canvas, "render_slide") as render:
            self.assertFalse(widget.build_preview())
        render.assert_not_called()
        self.assertIn("more than 1,000 ticks", widget.last_error)

    def test_large_frequency_catalog_defaults_to_one_slide_and_limits_one_report(self):
        widget = self.workspace()
        entry = DatasetCatalogEntry(
            "many", "Many frequencies", _grid(frequencies=tuple(range(1, 62)))
        )
        widget.set_dataset_catalog((entry,))
        self.assertEqual(widget.selected_frequencies(), tuple(range(1, 7)))
        widget.select_frequencies(range(1, 62))
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertFalse(widget.build_preview())
        self.assertIn("limited to 60", widget.last_error)

    def test_plot_type_switches_between_six_up_and_one_up_controls(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        self.assertFalse(widget.frequency_box.isHidden())
        self.assertTrue(widget.azimuth_combo.isHidden())
        self.assertIsNone(widget.findChild(QLabel, "pptFixedLayoutLabel"))

        widget.set_plot_kind("frequency")
        self.assertTrue(widget.frequency_box.isHidden())
        self.assertFalse(widget.frequency_azimuth_mode_combo.isHidden())
        self.assertFalse(widget.azimuth_combo.isHidden())
        self.assertTrue(widget.azimuth_band_widget.isHidden())
        visible_text = " ".join(
            label.text().casefold() for label in widget.findChildren(QLabel)
        )
        self.assertNotIn("fixed layout:", visible_text)
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        self.assertEqual(len(widget.preview_plan.slides), 1)
        self.assertEqual(widget.preview_plan.slides[0].layout, "frequency_single")
        self.assertEqual(len(widget.preview_plan.slides[0].plots), 1)

    def test_workspace_has_no_footer_control_and_plans_no_custom_footer(self):
        widget = self.workspace()
        widget.set_dataset_catalog(self.entries())
        self.assertFalse(hasattr(widget, "footer_edit"))
        form_labels = " ".join(
            label.text().casefold() for label in widget.findChildren(QLabel)
        )
        self.assertNotIn("footer", form_labels)

        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        self.assertTrue(all(slide.footer == "" for slide in widget.preview_plan.slides))

        widget.set_plot_kind("frequency")
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        assert widget.preview_plan is not None
        self.assertTrue(all(slide.footer == "" for slide in widget.preview_plan.slides))

    def test_async_export_uses_frozen_plan_and_exposes_busy_contract(self):
        started = threading.Event()
        release = threading.Event()
        calls: list[tuple[object, str, str | None, dict[str, str]]] = []

        def fake_export(
            plan,
            destination,
            *,
            template_path=None,
            template_layouts=None,
        ):
            calls.append(
                (
                    plan,
                    str(destination),
                    template_path,
                    dict(template_layouts or {}),
                )
            )
            started.set()
            if not release.wait(3.0):
                raise RuntimeError("test release timed out")
            Path(destination).write_bytes(b"fake pptx")
            return Path(destination)

        widget = self.workspace(exporter=fake_export)
        widget.set_dataset_catalog(self.entries())

        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "team-template.pptx"
            template.write_bytes(b"temporary test template")
            widget.template_edit.setText(str(template))
            widget.azimuth_layout_edit.setText("Team Master :: Six Up")
            widget.frequency_layout_edit.setText("Team Master :: One Up")
            with mock.patch.object(widget.preview_canvas, "render_slide"):
                self.assertTrue(widget.build_preview())
            frozen_plan = widget.preview_plan
            exported: list[str] = []
            widget.report_exported.connect(exported.append)
            output = str(Path(directory) / "uniform_report.pptx")
            widget.output_edit.setText(output)
            self.assertTrue(widget.export_report())
            self.assertTrue(started.wait(2.0))
            widget.azimuth_layout_edit.setText("Changed after start")
            widget.frequency_layout_edit.clear()
            self.assertTrue(widget.job_is_running())
            self.assertEqual(widget.busy_operation(), "PowerPoint report export")
            self.assertFalse(widget.controls_content.isEnabled())
            release.set()
            deadline = time.monotonic() + 4.0
            while widget.job_is_running() and time.monotonic() < deadline:
                QCoreApplication.processEvents()
                time.sleep(0.005)
            QCoreApplication.processEvents()
            self.assertFalse(widget.job_is_running())
            self.assertIsNone(widget.busy_operation())
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], frozen_plan)
            self.assertEqual(calls[0][1], output)
            self.assertEqual(calls[0][2], str(template))
            self.assertEqual(
                calls[0][3],
                {
                    "azimuth_3x2": "Team Master :: Six Up",
                    "frequency_single": "Team Master :: One Up",
                },
            )
            self.assertEqual(exported, [output])
            self.assertTrue(Path(output).is_file())

    def test_existing_output_requires_explicit_replace_confirmation(self):
        exporter = mock.Mock()
        widget = self.workspace(exporter=exporter)
        widget.set_dataset_catalog(self.entries())
        with mock.patch.object(widget.preview_canvas, "render_slide"):
            self.assertTrue(widget.build_preview())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.pptx"
            output.write_bytes(b"keep me")
            widget.output_edit.setText(str(output))
            with mock.patch(
                "ppt_workspace.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question:
                self.assertFalse(widget.export_report())
            question.assert_called_once()
            self.assertEqual(output.read_bytes(), b"keep me")
            exporter.assert_not_called()
            self.assertIn("existing file was kept", widget.status_label.text())

    def test_empty_catalog_has_actionable_preview_error_and_no_export(self):
        widget = self.workspace()
        self.assertFalse(widget.build_preview())
        self.assertIsNone(widget.preview_plan)
        self.assertFalse(widget.export_button.isEnabled())
        self.assertIn("Select at least one loaded dataset", widget.last_error)


if __name__ == "__main__":
    unittest.main()
