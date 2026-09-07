"""Progress delivery across the real worker/Qt boundary, including failures."""
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
gui = None
if importlib.util.find_spec('PySide6') or importlib.util.find_spec('PySide2'):
    from cem_tools import gui

from cem_tools.operations import BatchResult
from cem_tools.registry import ToolRegistry, ToolSpec


@unittest.skipIf(gui is None, 'Qt is optional for headless CEM Tools')
class GuiProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = gui.QApplication.instance() or gui.QApplication([])

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.002)
        self.assertTrue(predicate(), 'Qt worker did not reach the expected state')

    def test_live_progress_failure_and_subsequent_success(self):
        release = threading.Event()
        fail = [True]
        def operation(*, progress):
            progress(1, 2, 'Saved first.grim')
            if not release.wait(5):
                raise RuntimeError('Test worker timed out')
            if fail[0]:
                raise OSError('second.grim could not be written')
            progress(2, 2, 'Saved second.grim')
            return BatchResult((Path('first.grim'), Path('second.grim')))
        registry = ToolRegistry()
        registry.register(ToolSpec('probe', 'Probe', 'Progress test', operation, ()))
        with tempfile.TemporaryDirectory() as directory:
            settings = gui.QSettings(str(Path(directory)/'settings.ini'), gui.QSettings.IniFormat)
            with mock.patch.object(gui, 'QSettings', return_value=settings):
                window = gui.MainWindow(registry)
            try:
                with mock.patch.object(gui.QMessageBox, 'critical') as error:
                    window._run()
                    self.wait_until(lambda: window.progress_bar.value() == 500)
                    self.assertIn('1 / 2 completed', window.progress_label.text())
                    self.assertFalse(window.run_button.isEnabled())
                    self.assertFalse(window.tool_list.isEnabled())
                    self.assertFalse(window.form_host.isEnabled())
                    release.set()
                    self.wait_until(lambda: not window._running)
                    self.assertEqual(window.progress_bar.value(), 500)
                    self.assertIn('Failed after 1 / 2', window.progress_label.text())
                    self.assertTrue(window.run_button.isEnabled())
                    self.assertTrue(window.tool_list.isEnabled())
                    self.assertTrue(window.form_host.isEnabled())
                    error.assert_called_once()
                    fail[0] = False
                    window._run()
                    self.wait_until(lambda: not window._running)
                    self.assertEqual(window.progress_bar.value(), 1000)
                    self.assertIn('Complete - Wrote 2', window.progress_label.text())
            finally:
                release.set()
                window.pool.waitForDone(5000)
                self.app.processEvents()
                window.close()

    def test_fast_batch_bounds_gui_events_and_preserves_final_count(self):
        def operation(*, progress):
            for completed in range(10001):
                progress(completed, 10000, 'Copying small files')
            return BatchResult(())
        worker = gui.ToolWorker(ToolSpec('probe', 'Probe', '', operation, ()), {})
        events = []
        worker.signals.progress.connect(lambda *event: events.append(event))
        with mock.patch.object(gui.time, 'monotonic', return_value=1.):
            worker.run()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1][:2], (10000, 10000))

    def test_fast_failure_flushes_the_latest_completed_count(self):
        def operation(*, progress):
            progress(0, 2, 'Writing first.grim')
            progress(1, 2, 'Saved first.grim')
            raise OSError('second.grim failed')
        worker = gui.ToolWorker(ToolSpec('probe', 'Probe', '', operation, ()), {})
        events, errors = [], []
        worker.signals.progress.connect(lambda *event: events.append(event))
        worker.signals.failed.connect(errors.append)
        with mock.patch.object(gui.time, 'monotonic', return_value=1.):
            worker.run()
        self.assertEqual(events[-1][:2], (1, 2))
        self.assertEqual(len(errors), 1)


if __name__ == '__main__':
    unittest.main()
