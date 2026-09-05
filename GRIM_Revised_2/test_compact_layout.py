"""Real embedded workspaces must fit without clipping sidebar actions."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect, QSettings, QSize
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from grim_cut_gui import GrimCutWindow
from dataset_sidebar import DatasetSidebar
from grim_widgets import initial_window_size


class CompactLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # GHOST's legacy flat imports alter process state. Restore that boundary
        # after exercising the actual workspace so diagnostics can test other
        # installations independently of this suite's execution order.
        backend = Path(__file__).resolve().parents[1] / "tools" / "GHOST" / "Backend"
        cls.original_backend_modules = {
            path.stem: sys.modules.get(path.stem) for path in backend.glob("*.py")
            if path.stem != "__init__"
        }
        cls.original_sys_path = sys.path[:]
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
        sys.path[:] = cls.original_sys_path
        for name, original in cls.original_backend_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        settings = QSettings(str(Path(self.temp.name) / "window.ini"), QSettings.IniFormat)
        self.window = GrimCutWindow(settings=settings)

    def tearDown(self):
        self.window.hide()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def test_embedded_tabs_fit_compact_windows(self):
        for size in (QSize(1200, 680), QSize(1280, 720), QSize(1366, 768)):
            for index in range(self.window.main_tabs.count()):
                with self.subTest(size=size, tab=index):
                    self.window.main_tabs.setCurrentIndex(index)
                    self.window.resize(size)
                    self.window.show()
                    self.app.processEvents()
                    self.assertEqual(self.window.size(), size)

    def test_all_sidebar_controls_fit_even_with_cancel_visible(self):
        self.window.resize(1280, 720)
        self.window.show()
        self.window.btn_dataset_cancel.show()
        for tab in (0, 1):
            self.window.main_tabs.setCurrentIndex(tab)
            self.app.processEvents()
            sidebar = self.window.dataset_sidebar
            viewport = sidebar.viewport()
            self.assertLessEqual(sidebar.widget().width(), viewport.width())
            for name in (
                "btn_dataset_load", "btn_dataset_save", "btn_dataset_save_all",
                "btn_dataset_export", "btn_dataset_delete", "btn_dataset_undo_delete",
                "btn_dataset_cancel", "list_pol", "list_freq", "list_elev", "list_az",
            ):
                control = getattr(self.window, name)
                bounds = QRect(control.mapTo(viewport, QPoint()), control.size())
                with self.subTest(tab=tab, control=name):
                    self.assertTrue(viewport.rect().contains(bounds), (bounds, viewport.rect()))

    def test_solver_actions_stay_visible_with_advanced_form_scrolled(self):
        self.window.resize(1280, 680)
        self.window.main_tabs.setCurrentWidget(self.window.ghost_integration)
        ghost = self.window.ghost_integration.workspace
        solver = ghost.solver_tab
        ghost.setCurrentWidget(solver)
        solver.btn_advanced_settings.setChecked(True)
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.size(), QSize(1280, 680))
        bar = solver.controls_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        for value in (bar.minimum(), bar.maximum()):
            bar.setValue(value)
            self.app.processEvents()
            for control in (solver.btn_run, solver.btn_cancel, solver.progress):
                self.assertFalse(solver.controls_scroll.isAncestorOf(control))
                bounds = QRect(control.mapTo(solver, QPoint()), control.size())
                self.assertTrue(solver.rect().contains(bounds))

    def test_sidebar_exports_emit_intentions_without_owning_file_operations(self):
        sidebar = DatasetSidebar(self.window)
        for action, signal in zip(sidebar.btn_dataset_export.menu().actions(), (
            sidebar.export_pio_requested, sidebar.export_ptm_requested,
            sidebar.export_csv_requested,
        )):
            with self.subTest(action=action.text()):
                spy = QSignalSpy(signal)
                action.trigger()
                self.assertEqual(spy.count(), 1)

    def test_initial_size_accounts_for_available_desktop_area(self):
        size = initial_window_size(QSize(1366, 728))
        self.assertLess(size.width(), 1366)
        self.assertLess(size.height(), 728)
        self.window.resize(size)
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.size(), size)

    def test_neutral_palette_is_persisted_and_reaches_embedded_plots(self):
        self.window._application_palette_actions["Neutral Dark"].trigger()
        self.assertEqual(self.window._settings.value("appearance/application_palette"), "Neutral Dark")
        self.assertIn("#1f2937", self.window.styleSheet())
        freddy_colors = self.window.freddy_integration.workspace._theme_colors()
        self.assertEqual(freddy_colors["window_bg"], "#111827")
        self.assertEqual(freddy_colors["plot_bg"], "#1f2937")
        from matplotlib.colors import to_hex
        ghost_canvas = self.window.ghost_integration.workspace.geometry_tab.canvas
        self.assertEqual(to_hex(ghost_canvas.figure.get_facecolor()), "#1f2937")


if __name__ == "__main__":
    unittest.main()
