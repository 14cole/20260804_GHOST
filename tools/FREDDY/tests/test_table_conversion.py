from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ibc.table_conversion import (
    ColumnMapping as Column, TableOptions, converted_rows, export_table,
    output_header, preview_table, suggest_column,
)


class TableConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.txt"
        self.target = self.root / "output.csv"

    def test_hz_ghz_roundtrip_and_constant_columns(self):
        self.source.write_text("frequency_Hz phase_rad\n1D9 3.141592653589793\n2e9 0\n")
        columns = [Column("frequency", 0, input_unit="Hz", output_unit="GHz"),
                   Column("phase", 1, input_unit="rad", output_unit="deg"),
                   Column("elevation", constant="0", input_unit="deg", output_unit="deg")]
        original = self.source.read_bytes()
        self.assertEqual(export_table(self.source, self.target, TableOptions(), columns), 2)
        with self.target.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0], {"frequency_ghz": "1", "phase_deg": "180", "elevation_deg": "0"})
        roundtrip = list(converted_rows(self.target, TableOptions(),
            [Column("frequency", 0, input_unit="GHz", output_unit="Hz")]))
        self.assertEqual([r[1]["frequency"] for r in roundtrip], [1e9, 2e9])
        self.assertEqual(self.source.read_bytes(), original)

    def test_material_conversion_is_readable_by_freddy(self):
        from ibc.io import read_material_table
        self.source.write_text("frequency_ghz,eps_real,eps_imag,mu_real,mu_imag\n1,2,-0.1,1,0\n2,3,-0.2,1,0\n")
        columns = [Column("frequency", 0, input_unit="GHz", output_unit="Hz")]
        columns += [Column(name, i) for i, name in enumerate(
            ("frequency", "eps_real", "eps_imag", "mu_real", "mu_imag")) if i]
        export_table(self.source, self.target, TableOptions(), columns)
        self.assertIsNotNone(read_material_table(self.target))

    def test_explicit_headers_only_and_supported_delimiters(self):
        for text, options, expected in (
            ("frequency,azimuth,phase,magnitude\n1e9,0,90,2\n", TableOptions(), "frequency"),
            ("1e9   0\t90  2\n", TableOptions(), "Column 1"),
            ("notes\n# comment\n% comment\n! comment\nf;a;p;m\n1D9;0;90;2\n", TableOptions(skip_rows=1), "f"),
            ('"freq","channel"\n1e9,"V,V"\n', TableOptions(), "freq"),
            ("ignored\n1e9\t0\n", TableOptions(delimiter="Tab", skip_rows=1, header="No"), "Column 1"),
        ):
            with self.subTest(text=text):
                self.source.write_text(text, encoding="utf-8-sig")
                names, rows = preview_table(self.source, options)
                self.assertEqual(names[0], expected)
                self.assertEqual(len(rows), 1)
        self.assertEqual(suggest_column("frequency (GHz)"), ("frequency", "GHz"))
        self.assertEqual(suggest_column("frequency"), ("frequency", "As written"))
        self.assertEqual(output_header(Column("frequency_hz", output_unit="GHz")), "frequency_ghz")

    def test_preview_is_bounded_but_export_validates_later_rows(self):
        self.source.write_text("f\n" + "1000000000\n" * 22 + "invalid\n")
        self.assertEqual(len(preview_table(self.source)[1]), 20)
        self.target.write_text("existing output")
        with self.assertRaisesRegex(ValueError, "Line 24.*invalid"):
            export_table(self.source, self.target, TableOptions(),
                [Column("frequency", 0, input_unit="Hz", output_unit="GHz")])
        self.assertEqual(self.target.read_text(), "existing output")
        self.assertEqual(list(self.root.glob(".converted-*")), [])

    def test_invalid_units_values_and_source_replacement_are_rejected(self):
        self.source.write_text("f,x\n1e9,2\n")
        for columns, message in (
            ([Column("f", 0, input_unit="Hz", output_unit="deg")], "cannot convert"),
            ([Column("f", 0, input_unit="Hz")], "both input"),
            ([Column("f", 9)], "source column"),
            ([Column("f", constant="")], "constant"),
            ([Column("f", 0), Column("F", 1)], "unique"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                list(converted_rows(self.source, TableOptions(), columns))
        with self.assertRaisesRegex(ValueError, "different output"):
            export_table(self.source, self.source, TableOptions(), [Column("f", 0)])
        for bad in ("nan", "inf", "1e999"):
            self.source.write_text("f\n" + bad + "\n")
            with self.assertRaisesRegex(ValueError, "finite"):
                list(converted_rows(self.source, TableOptions(), [Column("f", 0, input_unit="Hz", output_unit="GHz")]))

    def test_ragged_binary_and_empty_files(self):
        for text, error in (("f,a\n1,2,3\n", "Line 2"), ("\x00junk", "binary"),
                            ("# none\n", "no table"), ("f,a\n", "no data")):
            self.source.write_text(text)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, error):
                preview_table(self.source)

    def test_failed_publish_preserves_output_and_removes_staging(self):
        self.source.write_text("f\n1e9\n")
        self.target.write_text("original")
        with mock.patch("ibc.table_conversion.os.replace", side_effect=OSError("denied")):
            with self.assertRaises(OSError):
                export_table(self.source, self.target, TableOptions(), [Column("f", 0)])
        self.assertEqual(self.target.read_text(), "original")
        self.assertEqual(list(self.root.glob(".converted-*")), [])


class ConverterUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_edit_units_preview_and_background_export(self):
        import time
        from ibc.converter_dialog import FileConverterDialog
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frequency.csv"
            output = Path(directory) / "converted.csv"
            source.write_text("frequency_hz,value\n1000000000,2\n")
            dialog = FileConverterDialog(path=source)
            try:
                self.assertEqual(dialog.mapping.rowCount(), 2)
                dialog.mapping.cellWidget(0, 4).setCurrentText("GHz")
                dialog.preview_conversion()
                self.assertEqual(dialog.preview.item(0, 0).text(), "1")
                with mock.patch("ibc.converter_dialog.QFileDialog.getSaveFileName", return_value=(str(output), "")):
                    dialog.submit()
                self.assertIsNotNone(dialog.job)
                deadline = time.monotonic() + 10
                while dialog.job is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.005)
                self.assertIsNone(dialog.job)
                self.assertIn("Saved 1 rows", dialog.status.text())
                self.assertIn("frequency_ghz", output.read_text())
                dialog.header.setCurrentText("No")
                with self.assertRaisesRegex(ValueError, "Refresh"):
                    dialog.snapshot()
            finally:
                dialog.deleteLater()
                self.app.processEvents()
