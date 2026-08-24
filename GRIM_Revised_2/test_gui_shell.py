"""Focused shell regressions for the unified GRIM application."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QMimeData, QUrl, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import grim_cut_gui
import grim_cut_dataset_mixin
import freddy_integration
import ghost_integration
from grim_cut_dataset_mixin import ConicGCDialog, RangeCalibrationDialog
from assembly_tree import (
    AssemblyTreePanel,
    _TYPE_BRANCH,
    _TYPE_ROOT,
    _attach,
)
from grim_dataset import GRIM_GC_CONVENTION, RcsGrid


class _FakeGhostIntegration(QWidget):
    files_exported = Signal(list, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.backend_path = None
        self.running = False
        self.focus_called = False
        self.setLayout(QVBoxLayout())

    def solve_is_running(self) -> bool:
        return self.running

    def focus_solver(self) -> None:
        self.focus_called = True


class _FakeFreddyIntegration(QWidget):
    # Deliberately expose a GHOST-shaped signal: the shell must not connect
    # FREDDY material/IBC CSV exports to GRIM's RCS dataset loader.
    files_exported = Signal(list, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.running = False
        self.focus_called = False
        self.setLayout(QVBoxLayout())

    def job_is_running(self) -> bool:
        return self.running

    def focus_workspace(self) -> None:
        self.focus_called = True


class _RecordingWindow(grim_cut_gui.GrimCutWindow):
    def __init__(self) -> None:
        self.loaded_path_batches: list[list[str]] = []
        super().__init__()

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
        self.ghost_patch.start()
        self.feature_patch.start()
        self.freddy_patch.start()
        self.window = _RecordingWindow()

    def tearDown(self) -> None:
        self.window.ghost_integration.running = False
        self.window.freddy_integration.running = False
        self.window.deleteLater()
        self.app.processEvents()
        self.freddy_patch.stop()
        self.feature_patch.stop()
        self.ghost_patch.stop()

    def test_tabs_have_one_canonical_assembly_workspace(self) -> None:
        labels = [
            self.window.main_tabs.tabText(index)
            for index in range(self.window.main_tabs.count())
        ]
        self.assertEqual(
            labels, ["Plotting", "ISAR", "Assembly", "GHOST", "FREDDY"]
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.assembly_workspace), 2
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.ghost_integration), 3
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.freddy_integration), 4
        )

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
        self.assertFalse(ok_button.isEnabled())
        dialog.combo_measured.setCurrentIndex(1)
        dialog.combo_exact.setCurrentIndex(2)
        dialog.spin_offset_m.setValue(0.125)
        dialog.chk_broadcast.setChecked(True)
        dialog.chk_attest.setChecked(True)
        self.assertTrue(ok_button.isEnabled())
        params = dialog.get_params()
        self.assertEqual(params["measured"][0], "Measured cylinder")
        self.assertEqual(params["exact"][0], "Exact")
        self.assertAlmostEqual(params["range_offset_m"], 0.125)
        self.assertTrue(params["allow_singleton_angular_broadcast"])
        self.assertTrue(params["convention_attested"])
        dialog.deleteLater()

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

    def test_conic_gc_dialog_is_symmetric_and_legacy_attestation_is_explicit(self):
        conic_dialog = ConicGCDialog(source_coordinate_system="conic")
        self.assertEqual(
            conic_dialog.get_params()["direction"], "conic_to_gc"
        )
        self.assertFalse(conic_dialog._radio_regrid.isEnabled())
        self.assertFalse(conic_dialog._chk_attest_legacy.isEnabled())
        conic_dialog.deleteLater()

        legacy_dialog = ConicGCDialog(
            source_coordinate_system="great_circle",
            source_gc_convention="legacy_ptm_unspecified",
        )
        self.assertEqual(
            legacy_dialog.get_params()["direction"], "gc_to_conic"
        )
        self.assertTrue(legacy_dialog._chk_attest_legacy.isEnabled())
        legacy_dialog._chk_attest_legacy.setChecked(True)
        self.assertTrue(
            legacy_dialog.get_params()["attest_legacy_ptm_convention"]
        )
        legacy_dialog.deleteLater()

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

    def test_running_feature_job_blocks_close_on_assembly_tab(self) -> None:
        event = QCloseEvent()
        with (
            mock.patch.object(
                self.window.feature_assembly_panel,
                "job_is_running",
                return_value=True,
            ),
            mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
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


if __name__ == "__main__":
    unittest.main()
