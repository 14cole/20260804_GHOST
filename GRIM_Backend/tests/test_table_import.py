"""Numerical, dialog, and actual background import-queue coverage."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialog

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.execution.dataset_jobs import _DatasetLoadWorker, _load_dataset_path_task
from GRIM_Backend.io.loaders import load_dataset, UnrecognizedTableError
from GRIM_Backend.io.mapped_table import grid_from_mapped_rows
from GRIM_Backend.ui.dataset_actions import DATASET_DIRTY_ROLE, DATASET_PATH_ROLE, DatasetOpsMixin
from GRIM_Backend.ui.table_import import create_table_import_dialog


def records(magnitude=2.0, phase=90.0):
    row = dict(frequency=1.0, azimuth=0.0, elevation=0.0, polarization="VV", magnitude=magnitude)
    if phase is not None:
        row["phase"] = phase
    return [(2, row)]


class MappedGridTests(unittest.TestCase):
    def test_amplitude_power_dbsm_and_ratios(self):
        for representation, magnitude, power, quantity in (
            ("RCS amplitude (sqrt(m²))", 2.0, 4.0, "sigma_3d"),
            ("RCS power (m²)", 2.0, 2.0, "sigma_3d"),
            ("RCS (dBsm)", 10.0, 10.0, "sigma_3d"),
            ("RCS (dBsm)", -math.inf, 0.0, "sigma_3d"),
            ("Scattering width (m)", 2.0, 2.0, "sigma_2d"),
            ("Scattering width (dBke)", 0.0, 299792458 / (2 * math.pi * 1e9), "sigma_2d"),
            ("Amplitude ratio", 2.0, 4.0, "power_ratio"),
            ("Power ratio (dB)", 10.0, 10.0, "power_ratio"),
        ):
            with self.subTest(representation=representation):
                grid = grid_from_mapped_rows(records(magnitude), "input.dat", representation)
                self.assertAlmostEqual(grid.rcs_power.item(), power)
                if power:
                    self.assertAlmostEqual(grid.rcs_phase.item(), math.pi / 2)
                self.assertEqual(grid.units["rcs_linear_quantity"], quantity)
                self.assertEqual(grid.units["frequency"], "GHz")

    def test_missing_phase_stays_unknown_and_mapping_survives_save(self):
        grid = grid_from_mapped_rows(records(phase=None), "original.asc", "RCS power (m²)",
                                     mapping={"elevation": {"constant": 0}})
        self.assertTrue(np.isnan(grid.rcs_phase.item()))
        self.assertNotIn("grim-column-import", grid.history)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mapped.grim"
            grid.save(str(target))
            loaded = RcsGrid.load(str(target))
            self.assertEqual(json.loads(str(loaded.extra["column_mapping_json"].item()))["elevation"]["constant"], 0)
            self.assertTrue(np.isnan(loaded.rcs_phase.item()))

    def test_sparse_samples_and_conflicting_duplicates(self):
        rows = records() + [(3, dict(records()[0][1], frequency=2.0, azimuth=10.0))]
        grid = grid_from_mapped_rows(rows, "input.txt", "RCS power (m²)")
        self.assertEqual(grid.rcs_power.shape, (2, 1, 2, 1))
        self.assertEqual(np.count_nonzero(np.isnan(grid.rcs_power)), 2)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            grid_from_mapped_rows(records(2) + records(3), "input.txt", "RCS power (m²)")

    def test_invalid_data_and_ambiguous_magnitude(self):
        for magnitude in (-2, math.nan, math.inf):
            with self.assertRaisesRegex(ValueError, "Line 2"):
                grid_from_mapped_rows(records(magnitude), "input.txt", "RCS amplitude (sqrt(m²))")
        with self.assertRaisesRegex(ValueError, "Choose"):
            grid_from_mapped_rows(records(), "input.txt", "magnitude")
        with self.assertRaisesRegex(ValueError, "positive"):
            grid_from_mapped_rows([(2, dict(records()[0][1], frequency=0))], "input.txt", "Power ratio")

    def test_worker_separates_unknown_text_from_corrupt_native_and_standard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / "custom.dat"
            unknown.write_text("1e9 0 90 2\n")
            corrupt = root / "broken.grim"
            corrupt.write_text("not an archive")
            known = root / "standard.grim"
            grid_from_mapped_rows(records(), "source", "RCS power (m²)").save(str(known))
            self.assertEqual(_load_dataset_path_task((0, str(unknown)))["status"], "mapping_required")
            self.assertEqual(_load_dataset_path_task((0, str(corrupt)))["status"], "error")
            with self.assertRaises(UnrecognizedTableError):
                load_dataset(str(unknown))
            worker = _DatasetLoadWorker([(i, str(p)) for i, p in enumerate((unknown, known, corrupt))])
            summaries = []
            worker.finished.connect(summaries.append)
            worker.run()
            self.assertEqual(len(summaries), 1)
            self.assertEqual([len(summaries[0][k]) for k in ("loaded", "mapping_required", "failed")], [1, 1, 1])


class TableImportDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_named_ascii_with_hz_and_radians_imports_using_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "custom.asc"
            source.write_text("frequency_Hz azimuth_rad phase_rad magnitude\n1D9 0 1.5707963267948966 2\n2D9 0 0 3\n")
            original = source.read_bytes()
            dialog = create_table_import_dialog(None, source)
            try:
                self.assertEqual(dialog.mapping.rowCount(), 6)
                self.assertEqual(dialog.mapping.cellWidget(2, 2).text(), "0")
                dialog.magnitude_format.setCurrentText("RCS amplitude (sqrt(m²))")
                dialog.submit()
                deadline = time.monotonic() + 10
                while dialog.job is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.005)
                self.assertIsNotNone(dialog.dataset, dialog.status.text())
                np.testing.assert_allclose(dialog.dataset.frequencies, [1, 2])
                np.testing.assert_allclose(dialog.dataset.rcs_power.reshape(-1), [4, 9])
                np.testing.assert_allclose(dialog.dataset.rcs_phase.reshape(-1), [math.pi / 2, 0])
                self.assertEqual(source.read_bytes(), original)
            finally:
                dialog.deleteLater()
                self.app.processEvents()

    def test_headerless_csv_requires_frequency_units_and_magnitude_meaning(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "custom.csv"
            source.write_text("1e9,0,90,2\n")
            dialog = create_table_import_dialog(None, source)
            try:
                for row, column in ((0, 0), (1, 1), (4, 3), (5, 2)):
                    box = dialog.mapping.cellWidget(row, 1)
                    box.setCurrentIndex(box.findData(column))
                dialog.mapping.item(5, 0).setCheckState(Qt.Checked)
                dialog.submit()
                self.assertIsNone(dialog.job)
                self.assertIn("units", dialog.status.text())
                dialog.mapping.cellWidget(0, 3).setCurrentText("Hz")
                dialog.submit()
                self.assertIsNone(dialog.job)
                self.assertIn("magnitude", dialog.status.text())
                self.assertIsNone(dialog.dataset)
            finally:
                dialog.deleteLater()
                self.app.processEvents()


class ImportQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import test_gui_shell as shell
        shell.UnifiedGuiShellTest.setUp(self)

    def tearDown(self):
        import test_gui_shell as shell
        shell.UnifiedGuiShellTest.tearDown(self)

    def test_mixed_batch_cancel_and_queued_import_resume(self):
        import GRIM_Backend.ui.table_import as importer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / "custom.dat"
            unknown.write_text("frequency_Hz azimuth phase magnitude\n1e9 0 90 2\n")
            cancelled = root / "cancelled.csv"
            cancelled.write_text("f,mag\n1e9,2\n")
            standard, queued = root / "standard.grim", root / "queued.grim"
            grid = grid_from_mapped_rows(records(), "source", "RCS power (m²)")
            grid.save(str(standard))
            grid.save(str(queued))
            seen = []
            def factory(parent, path):
                self.assertIsNone(parent._background_worker_thread)
                self.assertTrue(parent._background_job_active())
                dialog = create_table_import_dialog(parent, path)
                seen.append(Path(path).name)
                # Queue a second batch while the actual modal mapping dialog is active.
                if Path(path).name == "custom.dat":
                    DatasetOpsMixin._handle_files_dropped(parent, [str(queued)])
                    dialog.magnitude_format.setCurrentText("RCS amplitude (sqrt(m²))")
                    QTimer.singleShot(0, dialog.submit)
                else:
                    QTimer.singleShot(0, dialog.reject)
                # Bound the test even if the form stops accepting the supplied mapping.
                QTimer.singleShot(10000, dialog, dialog.reject)
                return dialog
            with mock.patch.object(importer, "create_table_import_dialog", side_effect=factory):
                DatasetOpsMixin._handle_files_dropped(self.window, [str(unknown), str(standard), str(cancelled)])
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    if self.window._background_worker_thread is None and not self.window._pending_import_batches:
                        break
                    time.sleep(.005)
            self.assertEqual(seen, ["custom.dat", "cancelled.csv"])
            self.assertEqual(self.window.table.rowCount(), 3, self.window.status.currentMessage())
            mapped_row = next(r for r in range(3) if self.window.table.item(r, 0).text() == "custom")
            self.assertTrue(self.window.table.item(mapped_row, 0).data(DATASET_DIRTY_ROLE))
            self.assertEqual(self.window.table.item(mapped_row, 1).data(DATASET_PATH_ROLE), "")
            mapped = self.window.table.item(mapped_row, 0).data(Qt.UserRole)
            self.assertAlmostEqual(mapped.rcs_power.item(), 4.0)
            self.assertFalse(self.window._table_mapping_active)
            self.assertIn("cancelled", self.window._last_import_summary)
