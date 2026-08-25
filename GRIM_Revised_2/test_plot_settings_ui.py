"""Focused GUI regressions for the independent Plotting/ISAR settings popups."""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QScrollArea, QWidget

import grim_cut_gui
from plot_models import PlotContext


class _SettingsBuilderWindow(QMainWindow):
    """Construct only the QWidget base needed by the settings/context builder."""

    _build_plot_left_context = grim_cut_gui.GrimCutWindow._build_plot_left_context
    _activate_plot_tab = grim_cut_gui.GrimCutWindow._activate_plot_tab

    def __init__(self) -> None:
        super().__init__()
        self._plot_controls_by_tab: dict[str, dict] = {}

    def _schedule_hover(self, *_args) -> None:
        pass

    def _reset_hover_readout(self, *_args) -> None:
        pass

    def _move_shared_right_panel(self, _tab_key: str) -> None:
        pass


class PlotSettingsUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = _SettingsBuilderWindow()
        self.plot_panel = QWidget(self.window)
        self.isar_panel = QWidget(self.window)
        self.plot_context = self.window._build_plot_left_context(
            self.plot_panel, "plotting"
        )
        self.isar_context = self.window._build_plot_left_context(
            self.isar_panel, "isar"
        )

    def tearDown(self) -> None:
        self.plot_context.settings_frame.close()
        self.isar_context.settings_frame.close()
        self.window.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _label(popup, text: str) -> QLabel:
        return next(
            label
            for label in popup.content_widget.findChildren(QLabel)
            if label.text() == text
        )

    def test_popups_have_independent_titles_scroll_views_and_isar_content(self) -> None:
        plot_popup = self.plot_context.settings_frame
        isar_popup = self.isar_context.settings_frame

        self.assertEqual(plot_popup.windowTitle(), "Plot Settings")
        self.assertEqual(isar_popup.windowTitle(), "ISAR Settings")
        self.assertEqual(self.plot_context.btn_settings.text(), "Plot Settings")
        self.assertEqual(self.isar_context.btn_settings.text(), "ISAR Settings")
        self.assertIsNot(plot_popup.filter_edit, isar_popup.filter_edit)

        for popup in (plot_popup, isar_popup):
            self.assertIsInstance(popup.scroll_area, QScrollArea)
            self.assertIs(popup.scroll_area.widget(), popup.content_widget)
            self.assertTrue(popup.scroll_area.widgetResizable())
            self.assertEqual(
                popup.scroll_area.verticalScrollBarPolicy(), Qt.ScrollBarAsNeeded
            )
            self.assertGreaterEqual(popup.minimumWidth(), 420)
            self.assertGreaterEqual(popup.minimumHeight(), 280)

        plot_section = plot_popup.content_widget.findChild(
            QWidget, "isarSettingsSection"
        )
        isar_section = isar_popup.content_widget.findChild(
            QWidget, "isarSettingsSection"
        )
        self.assertIsNotNone(plot_section)
        self.assertIsNotNone(isar_section)
        self.assertTrue(plot_section.isHidden())
        self.assertFalse(isar_section.isHidden())

        plot_popup.show()
        isar_popup.resize(480, 300)
        isar_popup.show()
        self.app.processEvents()
        self.assertFalse(plot_section.isVisible())
        self.assertTrue(isar_section.isVisible())
        self.assertTrue(self.isar_context.combo_isar_recon.isVisible())
        self.assertTrue(self._label(isar_popup, "Image Scale").isVisible())
        self.assertFalse(self._label(isar_popup, "Plot X Min").isVisible())
        self.assertGreater(isar_popup.scroll_area.verticalScrollBar().maximum(), 0)

    def test_filter_is_row_based_case_insensitive_and_preserves_values(self) -> None:
        popup = self.isar_context.settings_frame
        popup.show()
        self.app.processEvents()
        original_width = self.isar_context.spin_isar_ap_width.value()
        changed_values: list[float] = []
        changed_scales: list[int] = []
        self.isar_context.spin_isar_ap_width.valueChanged.connect(
            changed_values.append
        )
        self.isar_context.combo_plot_scale.currentIndexChanged.connect(
            changed_scales.append
        )

        popup.filter_edit.setText("ApErTuRe center")
        self.app.processEvents()
        self.assertTrue(self.isar_context.chk_isar_aperture.isVisible())
        self.assertTrue(self.isar_context.spin_isar_ap_center.isVisible())
        self.assertFalse(self.isar_context.combo_isar_recon.isVisible())
        self.assertFalse(self._label(popup, "Plot X Min").isVisible())
        self.assertFalse(popup.no_matches_label.isVisible())
        self.assertEqual(self.isar_context.spin_isar_ap_width.value(), original_width)
        self.assertEqual(changed_values, [])
        self.assertEqual(changed_scales, [])

        popup.filter_edit.clear()
        self.app.processEvents()
        self.assertTrue(self.isar_context.combo_isar_recon.isVisible())
        self.assertFalse(self._label(popup, "Plot X Min").isVisible())
        self.assertEqual(self.isar_context.spin_isar_ap_width.value(), original_width)
        self.assertEqual(changed_values, [])
        self.assertEqual(changed_scales, [])

        # Combo-box choices participate in search without changing selection.
        popup.filter_edit.setText("image linear")
        self.app.processEvents()
        self.assertTrue(self.isar_context.combo_plot_scale.isVisible())
        self.assertFalse(self.isar_context.combo_isar_recon.isVisible())
        self.assertEqual(changed_scales, [])

        popup.filter_edit.setText("setting-that-does-not-exist")
        self.app.processEvents()
        self.assertTrue(popup.no_matches_label.isVisible())
        self.assertFalse(
            popup.content_widget.findChild(QWidget, "isarSettingsSection").isVisible()
        )

    def test_plotting_search_never_reveals_isar_and_tab_state_stays_independent(self) -> None:
        plot_popup = self.plot_context.settings_frame
        isar_popup = self.isar_context.settings_frame
        plot_section = plot_popup.content_widget.findChild(
            QWidget, "isarSettingsSection"
        )

        plot_popup.show()
        plot_popup.filter_edit.setText("reconstruction")
        self.app.processEvents()
        self.assertTrue(plot_popup.no_matches_label.isVisible())
        self.assertFalse(plot_section.isVisible())
        plot_popup.filter_edit.clear()
        self.assertTrue(plot_section.isHidden())

        self.plot_context.spin_plot_xmin.setValue(-12.0)
        self.isar_context.spin_plot_xmin.setValue(-34.0)
        isar_popup.filter_edit.setText("aperture")
        self.plot_context.btn_settings.setChecked(True)

        self.window._plot_contexts = {
            "plotting": self.plot_context,
            "isar": self.isar_context,
        }
        self.window._active_plot_tab = "plotting"
        self.window._dataset_ops_panel = QWidget(self.window)
        self.window._dataset_ops_visible = False
        for field in PlotContext.__dataclass_fields__:
            setattr(self.window, field, getattr(self.plot_context, field))

        with mock.patch.object(self.window, "_move_shared_right_panel"):
            self.window._activate_plot_tab("isar")

        self.app.processEvents()
        self.assertFalse(plot_popup.isVisible())
        self.assertFalse(self.plot_context.btn_settings.isChecked())
        self.assertIs(self.window.combo_isar_recon, self.isar_context.combo_isar_recon)
        self.assertEqual(self.plot_context.spin_plot_xmin.value(), -12.0)
        self.assertEqual(self.window.spin_plot_xmin.value(), -34.0)
        self.assertEqual(plot_popup.filter_edit.text(), "")
        self.assertEqual(isar_popup.filter_edit.text(), "aperture")


if __name__ == "__main__":
    unittest.main()
