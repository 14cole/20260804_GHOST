"""Preview state must remain tied to the validated Assembly and radar sample."""
import os
from types import SimpleNamespace
from unittest import mock
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from assembly_interference import InterferenceInspector
from ghost_integration import load_ghost_module


class InterferencePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widget = InterferenceInspector()
        self.plan = SimpleNamespace(prepared_plan_sha256="validated-plan")
        self.widget.plan_provider = lambda: self.plan
        self.result = dict(key=("validated-plan", 10., 30., 5.), labels=["Line seam"],
                           body=np.ones(3, complex), fields=np.full((1, 3), -.5+0j))

    def tearDown(self):
        self.widget.hide()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_change_during_evaluation_or_preview_clears_the_old_result(self):
        self.widget._show(self.result)
        self.assertIn("10 GHz, az 30, el 5", self.widget.total.text())
        self.plan = None
        self.widget._check_current_plan()
        self.assertIsNone(self.widget._result)
        self.assertEqual(self.widget.table.rowCount(), 0)
        self.assertEqual(self.widget.total.text(), "")
        self.widget._show(self.result)
        self.assertIsNone(self.widget._result)
        self.assertIn("Assembly changed", self.widget.status.text())

    def test_toggle_reuses_fields_and_removes_the_interfering_contribution(self):
        self.widget._show(self.result)
        self.assertLess(float(self.widget.table.item(0, 7).text()), 0.)
        with mock.patch.object(load_ghost_module("assembly_inspector").ContributionInspector,
                               "evaluate", side_effect=AssertionError("No new solve during preview")):
            self.widget.table.item(0, 0).setCheckState(Qt.Unchecked)
        self.assertEqual(float(self.widget.table.item(0, 7).text()), 0.)
        self.assertEqual(self.widget.table.item(0, 8).text(), "undefined")
        self.assertIn("12.5664 m2", self.widget.total.text())


if __name__ == "__main__":
    unittest.main()
