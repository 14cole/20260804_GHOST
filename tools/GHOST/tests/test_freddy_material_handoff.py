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

from geometry_io import parse_geometry  # noqa: E402

try:  # noqa: E402
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        from PySide2.QtWidgets import (  # type: ignore
            QApplication,
            QMessageBox,
        )
    import matplotlib  # noqa: F401
    GUI_DEPENDENCIES_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency gate exercised by CI images
    QApplication = None  # type: ignore[assignment]
    QMessageBox = None  # type: ignore[assignment]
    GUI_DEPENDENCIES_AVAILABLE = False

if GUI_DEPENDENCIES_AVAILABLE:  # noqa: E402
    from geometry_tab import GeometryTab
else:
    GeometryTab = None  # type: ignore[assignment]


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


@unittest.skipUnless(
    GUI_DEPENDENCIES_AVAILABLE, "GUI dependencies are unavailable"
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
    def test_actual_headerless_freddy_exports_attach_to_geometry(self, _information):
        freddy = Path(__file__).resolve().parents[2]/"FREDDY"
        if str(freddy) not in sys.path:
            sys.path.insert(0, str(freddy))
        from ibc.io import write_output, write_material_table
        from ibc.compute import MaterialTable
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._saved_geometry(root)
            ibc, medium = root/"coating.csv", root/"dielectric.csv"
            write_output(ibc, [(1., 20., -3.), (2., 30., 4.)], False)
            write_material_table(medium, MaterialTable([1., 2.], [2-.1j, 3-.2j], [1+0j, 1+0j]), False)
            self.assertTrue(self.tab.attach_material_artifact("ibc", str(ibc)))
            self.assertTrue(self.tab.attach_material_artifact("material", str(medium)))

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

    @mock.patch("geometry_tab.QMessageBox.critical")
    def test_whitespace_filename_is_rejected_before_copy(
        self, critical: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry_dir = root / "geometry"
            export_dir = root / "exports"
            geometry_dir.mkdir()
            export_dir.mkdir()
            self._saved_geometry(geometry_dir)
            source = export_dir / "My IBC.csv"
            source.write_text(IBC_TEXT, encoding="utf-8")

            self.assertFalse(
                self.tab.attach_material_artifact("ibc", str(source))
            )

            self.assertFalse((geometry_dir / source.name).exists())
            self.assertEqual(self.tab.table_ibc.rowCount(), 0)
            self.assertIn(
                "Invalid FREDDY Artifact Filename", critical.call_args.args
            )

    @mock.patch("geometry_tab.QMessageBox.warning")
    @mock.patch("geometry_tab.QFileDialog.getOpenFileName")
    def test_manual_csv_picker_rejects_whitespace_filename(
        self, open_dialog: mock.Mock, warning: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._saved_geometry(root)
            source = root / "My Material.csv"
            source.write_text(MATERIAL_TEXT, encoding="utf-8")
            open_dialog.return_value = (str(source), "CSV Files (*.csv)")

            self.assertEqual(self.tab._choose_material_csv("Choose CSV"), "")
            self.assertIn(
                "Unsupported Material Filename", warning.call_args.args
            )

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
    @mock.patch("geometry_tab.QMessageBox.question")
    def test_attachment_ui_failure_restores_file_and_table(
        self, question: mock.Mock, critical: mock.Mock
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
            source = export_dir / "coating.csv"
            source.write_text(IBC_TEXT, encoding="utf-8")
            destination = geometry_dir / source.name
            destination.write_text("previous sidecar\n", encoding="utf-8")

            with mock.patch.object(
                self.tab,
                "_refresh_segment_dropdowns",
                side_effect=RuntimeError("simulated UI failure"),
            ):
                self.assertFalse(
                    self.tab.attach_material_artifact("ibc", str(source))
                )

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "previous sidecar\n",
            )
            self.assertEqual(self.tab._read_small_table(self.tab.table_ibc), [])
            self.assertEqual(self.tab.ibcs_entries, [])
            self.assertIn(
                "FREDDY Artifact Attachment Failed", critical.call_args.args
            )

    @mock.patch("geometry_tab.QMessageBox.information")
    @mock.patch("geometry_tab.QFileDialog.getSaveFileName")
    def test_save_as_copies_referenced_sidecars_to_new_directory(
        self, save_dialog: mock.Mock, _information: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_dir = root / "old"
            new_dir = root / "new"
            old_dir.mkdir()
            new_dir.mkdir()
            self._saved_geometry(old_dir)
            sidecar = old_dir / "coating.csv"
            sidecar.write_text(IBC_TEXT, encoding="utf-8")
            rows = [["3", sidecar.name]]
            self.tab.ibcs_entries = rows
            self.tab._populate_small_table(
                self.tab.table_ibc,
                rows,
                label=self.tab.lbl_ibc,
                title_prefix="IBCS/Resistances",
            )
            target = new_dir / "body.geo"
            save_dialog.return_value = (str(target), "Geometry Files (*.geo)")

            self.tab.save_geo()

            self.assertEqual(
                (new_dir / sidecar.name).read_text(encoding="utf-8"),
                IBC_TEXT,
            )
            _title, _segments, ibcs, _dielectrics = parse_geometry(
                target.read_text(encoding="utf-8")
            )
            self.assertEqual(ibcs, rows)
            self.assertEqual(self.tab.loaded_path, str(target.resolve()))

    @mock.patch("geometry_tab.QMessageBox.critical")
    @mock.patch("geometry_tab.QFileDialog.getSaveFileName")
    def test_save_as_missing_sidecar_is_blocked(
        self, save_dialog: mock.Mock, critical: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_dir = root / "old"
            new_dir = root / "new"
            old_dir.mkdir()
            new_dir.mkdir()
            self._saved_geometry(old_dir)
            rows = [["3", "missing.csv"]]
            self.tab.ibcs_entries = rows
            self.tab._populate_small_table(
                self.tab.table_ibc,
                rows,
                label=self.tab.lbl_ibc,
                title_prefix="IBCS/Resistances",
            )
            target = new_dir / "body.geo"
            save_dialog.return_value = (str(target), "Geometry Files (*.geo)")

            self.tab.save_geo()

            self.assertFalse(target.exists())
            self.assertEqual(self.tab.loaded_path, str((old_dir / "body.geo").resolve()))
            self.assertIn("Geometry Save Blocked", critical.call_args.args)

    @mock.patch("geometry_tab.QMessageBox.information")
    @mock.patch("geometry_tab.QMessageBox.critical")
    @mock.patch("geometry_tab.QMessageBox.question")
    @mock.patch("geometry_tab.QFileDialog.getSaveFileName")
    def test_save_as_publish_failure_restores_geometry_and_sidecar(
        self,
        save_dialog: mock.Mock,
        question: mock.Mock,
        critical: mock.Mock,
        _information: mock.Mock,
    ) -> None:
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        question.return_value = buttons.Yes
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_dir = root / "old"
            new_dir = root / "new"
            old_dir.mkdir()
            new_dir.mkdir()
            geometry = self._saved_geometry(old_dir)
            (old_dir / "coating.csv").write_text(IBC_TEXT, encoding="utf-8")
            rows = [["3", "coating.csv"]]
            self.tab.ibcs_entries = rows
            self.tab._populate_small_table(
                self.tab.table_ibc,
                rows,
                label=self.tab.lbl_ibc,
                title_prefix="IBCS/Resistances",
            )

            target = new_dir / "body.geo"
            target.write_text("previous target geometry\n", encoding="utf-8")
            target_sidecar = new_dir / "coating.csv"
            target_sidecar.write_text(
                "previous target sidecar\n", encoding="utf-8"
            )
            save_dialog.return_value = (str(target), "Geometry Files (*.geo)")

            real_replace = os.replace
            call_count = 0

            def fail_geometry_publish(source: object, destination: object) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("simulated geometry publication failure")
                real_replace(source, destination)

            with mock.patch(
                "geometry_tab.os.replace", side_effect=fail_geometry_publish
            ):
                self.tab.save_geo()

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "previous target geometry\n",
            )
            self.assertEqual(
                target_sidecar.read_text(encoding="utf-8"),
                "previous target sidecar\n",
            )
            self.assertEqual(self.tab.loaded_path, str(geometry.resolve()))
            critical.assert_called_once()

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
