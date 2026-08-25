"""Focused tests for the typed FREDDY-to-GHOST material handoff."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from geometry_tab import GeometryTab  # noqa: E402

try:  # noqa: E402
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:  # noqa: E402
    from PySide2.QtWidgets import QApplication, QMessageBox  # type: ignore


IBC_TEXT = (
    "frequency_hz,resistance_ohm,reactance_ohm\n"
    "1000000000,50,0\n"
    "2000000000,55,2\n"
)
MATERIAL_TEXT = (
    "frequency_hz,eps_real,eps_imag,mu_real,mu_imag\n"
    "1000000000,2.5,-0.1,1,0\n"
    "2000000000,2.4,-0.1,1,0\n"
)


class FreddyMaterialHandoffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.tab = GeometryTab()

    def tearDown(self) -> None:
        self.tab.close()
        self.tab.deleteLater()
        self.app.processEvents()

    def _saved_geometry(self, root: Path) -> Path:
        geometry = root / "body.geo"
        geometry.write_text("title: body\n", encoding="utf-8")
        self.tab.loaded_path = str(geometry)
        return geometry

    @mock.patch("geometry_tab.QMessageBox.information")
    def test_nominal_ibc_is_copied_and_added_to_ibc_table(
        self, _information: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry_dir = root / "geometry"
            export_dir = root / "exports"
            geometry_dir.mkdir()
            export_dir.mkdir()
            self._saved_geometry(geometry_dir)
            source = export_dir / "coating_ibc.csv"
            source.write_text(IBC_TEXT, encoding="utf-8")

            self.assertTrue(
                self.tab.attach_material_artifact("ibc", str(source))
            )

            self.assertEqual(
                (geometry_dir / source.name).read_text(encoding="utf-8"),
                IBC_TEXT,
            )
            self.assertEqual(
                self.tab._read_small_table(self.tab.table_ibc),
                [["1", source.name]],
            )
            self.assertEqual(
                self.tab._read_small_table(self.tab.table_diel), []
            )

    @mock.patch("geometry_tab.QMessageBox.information")
    def test_nominal_material_is_added_to_dielectric_table(
        self, _information: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry_dir = root / "geometry"
            export_dir = root / "exports"
            geometry_dir.mkdir()
            export_dir.mkdir()
            self._saved_geometry(geometry_dir)
            source = export_dir / "blend.csv"
            source.write_text(MATERIAL_TEXT, encoding="utf-8")

            self.assertTrue(
                self.tab.attach_material_artifact("material", str(source))
            )

            self.assertEqual(
                self.tab._read_small_table(self.tab.table_diel),
                [["1", source.name]],
            )
            self.assertEqual(
                self.tab._read_small_table(self.tab.table_ibc), []
            )

    @mock.patch("geometry_tab.QMessageBox.warning")
    def test_active_saved_geo_is_required(self, warning: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "nominal.csv"
            source.write_text(IBC_TEXT, encoding="utf-8")

            self.assertFalse(
                self.tab.attach_material_artifact("ibc", str(source))
            )
            self.assertIn("No Active Saved Geometry", warning.call_args.args)

    @mock.patch("geometry_tab.QMessageBox.critical")
    def test_analysis_csv_is_rejected_by_production_schema(
        self, critical: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry_dir = root / "geometry"
            export_dir = root / "exports"
            geometry_dir.mkdir()
            export_dir.mkdir()
            self._saved_geometry(geometry_dir)
            source = export_dir / "renamed_analysis.csv"
            source.write_text(
                "frequency_hz,zr_nom,zi_nom,zr_min,zr_max,zi_min,zi_max\n"
                "1000000000,50,0,40,60,-1,1\n",
                encoding="utf-8",
            )

            self.assertFalse(
                self.tab.attach_material_artifact("ibc", str(source))
            )
            self.assertFalse((geometry_dir / source.name).exists())
            self.assertEqual(self.tab.table_ibc.rowCount(), 0)
            self.assertIn("Invalid FREDDY Artifact", critical.call_args.args)

    @mock.patch("geometry_tab.QMessageBox.information")
    @mock.patch("geometry_tab.QMessageBox.question")
    def test_existing_sidecar_requires_explicit_replace_confirmation(
        self, question: mock.Mock, _information: mock.Mock
    ) -> None:
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry_dir = root / "geometry"
            export_dir = root / "exports"
            geometry_dir.mkdir()
            export_dir.mkdir()
            self._saved_geometry(geometry_dir)
            source = export_dir / "coating.csv"
            source.write_text(IBC_TEXT, encoding="utf-8")
            destination = geometry_dir / source.name
            destination.write_text("existing content\n", encoding="utf-8")

            question.return_value = buttons.No
            self.assertFalse(
                self.tab.attach_material_artifact("ibc", str(source))
            )
            self.assertEqual(
                destination.read_text(encoding="utf-8"), "existing content\n"
            )
            self.assertEqual(self.tab.table_ibc.rowCount(), 0)

            question.return_value = buttons.Yes
            self.assertTrue(
                self.tab.attach_material_artifact("ibc", str(source))
            )
            self.assertEqual(
                destination.read_text(encoding="utf-8"), IBC_TEXT
            )
            self.assertEqual(self.tab.table_ibc.rowCount(), 1)

    @mock.patch("geometry_tab.QMessageBox.information")
    @mock.patch("geometry_tab.QMessageBox.question")
    def test_casefold_match_updates_row_to_exact_attached_filename(
        self, question: mock.Mock, _information: mock.Mock
    ) -> None:
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        question.return_value = buttons.Yes
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry_dir = root / "geometry"
            export_dir = root / "exports"
            geometry_dir.mkdir()
            export_dir.mkdir()
            self._saved_geometry(geometry_dir)
            source = export_dir / "foo.csv"
            source.write_text(IBC_TEXT, encoding="utf-8")
            (geometry_dir / "Foo.csv").write_text(IBC_TEXT, encoding="utf-8")
            existing = [["7", "Foo.csv"]]
            self.tab.ibcs_entries = existing
            self.tab._populate_small_table(
                self.tab.table_ibc,
                existing,
                label=self.tab.lbl_ibc,
                title_prefix="IBCS/Resistances",
            )

            self.assertTrue(self.tab.attach_material_artifact("ibc", str(source)))
            self.assertEqual(
                self.tab._read_small_table(self.tab.table_ibc),
                [["7", "foo.csv"]],
            )

    @mock.patch("geometry_tab.QMessageBox.critical")
    @mock.patch("geometry_tab.QFileDialog.getSaveFileName")
    def test_failed_atomic_geometry_save_preserves_existing_file(
        self, save_dialog: mock.Mock, critical: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "body.geo"
            target.write_text("previous geometry\n", encoding="utf-8")
            save_dialog.return_value = (str(target), "Geometry Files (*.geo)")
            with (
                mock.patch(
                    "geometry_tab.build_geometry_text",
                    return_value="replacement geometry\n",
                ),
                mock.patch(
                    "geometry_tab.os.replace",
                    side_effect=OSError("simulated publish failure"),
                ),
            ):
                self.tab.save_geo()

            self.assertEqual(
                target.read_text(encoding="utf-8"), "previous geometry\n"
            )
            self.assertEqual(set(Path(folder).iterdir()), {target})
            critical.assert_called_once()

    @mock.patch("geometry_tab.QMessageBox.warning")
    def test_same_filename_cannot_be_reused_for_opposite_schema(
        self, warning: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry_dir = root / "geometry"
            export_dir = root / "exports"
            geometry_dir.mkdir()
            export_dir.mkdir()
            self._saved_geometry(geometry_dir)
            source = export_dir / "shared.csv"
            source.write_text(IBC_TEXT, encoding="utf-8")
            (geometry_dir / source.name).write_text(
                MATERIAL_TEXT, encoding="utf-8"
            )
            existing = [["2", source.name]]
            self.tab.dielectric_entries = existing
            self.tab._populate_small_table(
                self.tab.table_diel,
                existing,
                label=self.tab.lbl_diel,
                title_prefix="Dielectrics",
            )

            self.assertFalse(
                self.tab.attach_material_artifact("ibc", str(source))
            )
            self.assertEqual(
                (geometry_dir / source.name).read_text(encoding="utf-8"),
                MATERIAL_TEXT,
            )
            self.assertIn(
                "Geometry Sidecar Type Conflict", warning.call_args.args
            )


if __name__ == "__main__":
    unittest.main()
