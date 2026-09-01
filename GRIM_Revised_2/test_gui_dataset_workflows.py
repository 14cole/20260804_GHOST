"""GUI regressions for parameter edits and multi-dataset overlap workflows."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QDialog, QMessageBox

import grim_cut_gui
import grim_cut_dataset_mixin
from grim_cut_dataset_mixin import (
    CropDialog,
    DATASET_DIRTY_ROLE,
    DATASET_ID_ROLE,
    DATASET_PATH_ROLE,
    DatasetAuditDialog,
    RegridDialog,
    StitchDialog,
    WrapDialog,
)
from grim_dataset import RcsGrid
from grim_python import DatasetReference, PythonScriptRecorder
from test_gui_shell import (
    _FakeFeatureWorkflow,
    _FakeFreddyIntegration,
    _FakeGhostIntegration,
    _FakeRunsWorkspace,
    _RecordingWindow,
)


def _axis_grid(azimuths: list[float], value: float) -> RcsGrid:
    shape = (len(azimuths), 1, 1, 1)
    return RcsGrid(
        azimuths,
        [0.0],
        [10.0],
        ["VV"],
        rcs=np.full(shape, complex(value), dtype=np.complex128),
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        },
    )


def _mixed_unit_grid(*, radians: bool, frequency_hz: bool) -> RcsGrid:
    azimuth_deg = np.asarray([0.0, 90.0, 180.0])
    elevation_deg = np.asarray([-10.0, 0.0, 10.0])
    frequency_ghz = np.asarray([8.0, 9.0, 10.0])
    azimuths = np.deg2rad(azimuth_deg) if radians else azimuth_deg
    elevations = np.deg2rad(elevation_deg) if radians else elevation_deg
    frequencies = frequency_ghz * 1.0e9 if frequency_hz else frequency_ghz
    shape = (3, 3, 3, 1)
    amplitude = (
        np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1.0
    ).astype(np.complex128)
    return RcsGrid(
        azimuths,
        elevations,
        frequencies,
        ["VV"],
        rcs=amplitude,
        units={
            "azimuth": "rad" if radians else "deg",
            "elevation": "rad" if radians else "deg",
            "frequency": "Hz" if frequency_hz else "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
            "angular_coordinate_system": "conic",
        },
    )


class AxisEditTransactionTest(unittest.TestCase):
    def test_long_polarization_edit_is_trimmed_without_dtype_truncation(self) -> None:
        power = np.asarray([10.0, 20.0, 30.0]).reshape(1, 1, 1, 3)
        source = RcsGrid(
            [0.0],
            [0.0],
            [10.0],
            ["HH", "VV", "CUSTOM"],
            rcs_power=power,
            rcs_phase=np.zeros_like(power),
            history="Loaded source",
            units={"frequency": "GHz"},
            extra={
                "polarization_alias_primary": "VV",
                "polarization_aliases_json": '["VV", "HH"]',
            },
        )

        edited = source.edit_axis_value(
            "polarization", 1, "  VERY_LONG_POLARIZATION  "
        )

        self.assertIsNot(edited, source)
        np.testing.assert_array_equal(
            source.polarizations, ["HH", "VV", "CUSTOM"]
        )
        np.testing.assert_array_equal(
            edited.polarizations,
            ["HH", "VERY_LONG_POLARIZATION", "CUSTOM"],
        )
        np.testing.assert_array_equal(edited.rcs_power, power)
        self.assertNotIn("polarization_alias_primary", edited.extra)
        self.assertNotIn("polarization_aliases_json", edited.extra)
        self.assertIn(
            "Edit polarization axis[1]: 'VV' -> 'VERY_LONG_POLARIZATION'",
            edited.history,
        )

    def test_nonreordering_axis_edit_owns_independent_sample_arrays(self) -> None:
        source = _axis_grid([0.0, 1.0, 2.0], 1.0)

        numeric_edit = source.edit_axis_value("azimuth", 1, 1.5)
        polarization_edit = source.edit_axis_value("polarization", 0, "CUSTOM")

        for edited in (numeric_edit, polarization_edit):
            with self.subTest(axis=edited.history.splitlines()[-1]):
                self.assertFalse(
                    np.shares_memory(source.rcs_power, edited.rcs_power)
                )
                self.assertFalse(
                    np.shares_memory(source.rcs_phase, edited.rcs_phase)
                )
                edited.rcs_power.flat[0] = 1234.0
                edited.rcs_phase.flat[0] = 2.5
                self.assertNotEqual(float(source.rcs_power.flat[0]), 1234.0)
                self.assertNotEqual(float(source.rcs_phase.flat[0]), 2.5)

    def test_polarization_edit_rejects_blank_and_trimmed_casefold_duplicate(self) -> None:
        source = RcsGrid(
            [0.0],
            [0.0],
            [10.0],
            ["HH", "VV"],
            rcs=np.ones((1, 1, 1, 2), dtype=np.complex128),
            units={"frequency": "GHz"},
        )
        original_labels = source.polarizations.copy()

        for value, message in (
            ("   ", "must not be blank"),
            ("  vv  ", "duplicates another channel"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    source.edit_axis_value("polarization", 0, value)
                np.testing.assert_array_equal(
                    source.polarizations, original_labels
                )

    def test_numeric_edit_stable_sorts_axis_samples_and_aligned_extra_arrays(self) -> None:
        power = np.arange(1.0, 7.0).reshape(3, 2, 1, 1)
        phase = power / 10.0
        aligned_extra = np.stack((power, power + 100.0), axis=-1)
        source = RcsGrid(
            [0.0, 1.0, 2.0],
            [-1.0, 1.0],
            [10.0],
            ["VV"],
            rcs_power=power,
            rcs_phase=phase,
            history="Loaded source",
            units={"frequency": "GHz"},
            extra={
                "aligned": aligned_extra,
                "solver_metadata_json": "stale",
            },
        )

        edited = source.edit_axis_value("azimuth", 0, 3.0)
        expected_order = [1, 2, 0]

        np.testing.assert_array_equal(source.azimuths, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(edited.azimuths, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            edited.rcs_power, np.take(power, expected_order, axis=0)
        )
        np.testing.assert_array_equal(
            edited.rcs_phase, np.take(phase, expected_order, axis=0)
        )
        np.testing.assert_array_equal(
            edited.extra["aligned"],
            np.take(aligned_extra, expected_order, axis=0),
        )
        self.assertNotIn("solver_metadata_json", edited.extra)
        self.assertIn("stable-sorted axis and sample arrays", edited.history)

    def test_numeric_edit_rejects_nonfinite_duplicate_and_invalid_index_atomically(self) -> None:
        source = _axis_grid([0.0, 1.0, 2.0], 1.0)
        original_axis = source.azimuths.copy()
        original_power = source.rcs_power.copy()

        for value, message in (
            (float("nan"), "must be finite"),
            (float("inf"), "must be finite"),
            (1.0, "duplicates another coordinate"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, message):
                    source.edit_axis_value("azimuth", 0, value)
                np.testing.assert_array_equal(source.azimuths, original_axis)
                np.testing.assert_array_equal(source.rcs_power, original_power)

        with self.assertRaisesRegex(IndexError, "outside"):
            source.edit_axis_value("azimuth", 99, 4.0)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            source.edit_axis_value("azimuth", True, 4.0)

        for value in (0.0, -1.0):
            with self.subTest(frequency=value):
                with self.assertRaisesRegex(ValueError, "greater than zero"):
                    source.edit_axis_value("frequency", 0, value)
                np.testing.assert_array_equal(source.frequencies, [10.0])

    def test_recorder_chains_in_place_axis_edits_by_stable_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = str(Path(temp_dir) / "source.grim")
            reference = DatasetReference("stable-id", "Measured", source_path)
            recorder = PythonScriptRecorder()
            recorder.record_method(
                reference,
                reference,
                "edit_axis_value",
                args=("polarization", 0, "LONG_LABEL"),
                comment="First edit",
            )
            recorder.record_method(
                reference,
                reference,
                "edit_axis_value",
                args=("frequency", 0, 12.0),
                comment="Second edit",
            )

        self.assertIn(
            "dataset_2 = dataset_1.edit_axis_value(", recorder.script
        )
        self.assertIn(
            "dataset_3 = dataset_2.edit_axis_value(", recorder.script
        )
        self.assertLess(
            recorder.script.index("# First edit"),
            recorder.script.index("# Second edit"),
        )


class DatasetOperationDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_audit_statuses_use_one_pass_warn_fail_vocabulary(self) -> None:
        rendered = DatasetAuditDialog._format_reports(
            [
                ("healthy", {"status": "ok"}),
                ("review", {"status": "warning"}),
                ("invalid", {"status": "error"}),
                ("legacy", {"status": "fail"}),
                ("unknown", {"status": "unexpected"}),
            ]
        )
        self.assertIn("healthy\nStatus: PASS", rendered)
        self.assertIn("review\nStatus: WARN", rendered)
        self.assertIn("invalid\nStatus: FAIL", rendered)
        self.assertIn("legacy\nStatus: FAIL", rendered)
        self.assertIn("unknown\nStatus: WARN", rendered)

    def test_dialogs_expose_physical_units_frames_and_half_open_wraps(self) -> None:
        reference = _mixed_unit_grid(radians=True, frequency_hz=False)
        crop = CropDialog(reference, has_selected_values=True)
        regrid = RegridDialog(reference)
        stitch = StitchDialog(["first", "second"])
        wrap = WrapDialog()
        self.addCleanup(crop.deleteLater)
        self.addCleanup(regrid.deleteLater)
        self.addCleanup(stitch.deleteLater)
        self.addCleanup(wrap.deleteLater)

        self.assertEqual(crop._range_controls["azimuth"][0].text(), "Azimuth (deg)")
        self.assertEqual(crop._range_controls["frequency"][0].text(), "Frequency (GHz)")
        self.assertAlmostEqual(crop._range_controls["azimuth"][1].value(), 0.0)
        self.assertAlmostEqual(crop._range_controls["azimuth"][2].value(), 180.0)
        self.assertEqual(regrid._axis.itemText(0), "Azimuth")
        self.assertEqual(regrid.get_params()["unit"], "deg")
        self.assertIn("overlap phase removed", stitch._policy.itemText(2))
        self.assertIn("declared native axis units", stitch._tolerance_help.text())
        self.assertIn("same units", stitch._tolerance_help.text())
        self.assertEqual(wrap._rb_0_360.text(), "[0°, 360°)")
        self.assertEqual(wrap._rb_pm180.text(), "[-180°, 180°)")

        great_circle_units = dict(reference.units)
        great_circle_units.update(
            angular_coordinate_system="great_circle",
            great_circle_coordinate_convention="GRIM_GC_V1",
        )
        great_circle = RcsGrid(
            reference.azimuths,
            reference.elevations,
            reference.frequencies,
            reference.polarizations,
            rcs_power=reference.rcs_power,
            rcs_phase=reference.rcs_phase,
            units=great_circle_units,
        )
        gc_crop = CropDialog(great_circle, has_selected_values=True)
        gc_regrid = RegridDialog(great_circle)
        self.addCleanup(gc_crop.deleteLater)
        self.addCleanup(gc_regrid.deleteLater)
        self.assertEqual(gc_crop._range_controls["azimuth"][0].text(), "Aspect (deg)")
        self.assertEqual(gc_crop._range_controls["elevation"][0].text(), "Pitch (deg)")
        self.assertEqual(gc_regrid._axis.itemText(0), "Aspect")
        self.assertEqual(gc_regrid._axis.itemText(1), "Pitch")


class GuiDatasetWorkflowTest(unittest.TestCase):
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
        self.runs_patch = mock.patch.object(
            grim_cut_gui, "RunsWorkspace", _FakeRunsWorkspace
        )
        self.ghost_patch.start()
        self.feature_patch.start()
        self.freddy_patch.start()
        self.runs_patch.start()
        self.window = _RecordingWindow()

    def tearDown(self) -> None:
        self.window.ghost_integration.running = False
        self.window.freddy_integration.running = False
        self.window.runs_workspace.running = False
        self.window.deleteLater()
        self.app.processEvents()
        self.runs_patch.stop()
        self.freddy_patch.stop()
        self.feature_patch.stop()
        self.ghost_patch.stop()

    def _select_rows_in_order(self, *rows: int) -> None:
        table = self.window.table
        table.clearSelection()
        flags = (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        )
        for row in rows:
            table.setCurrentCell(
                row,
                0,
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
            table.selectionModel().select(table.model().index(row, 0), flags)
            self.app.processEvents()
        self.assertEqual(self.window._dataset_selection_order, list(rows))

    def _wait_for_background(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self.window._background_job_active() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertFalse(self.window._background_job_active())

    @staticmethod
    def _select_parameter_row(widget, row: int) -> None:
        widget.clearSelection()
        item = widget.item(int(row))
        if item is None:
            raise AssertionError(f"parameter row {row} does not exist")
        item.setSelected(True)
        widget.setCurrentItem(item)

    def test_join_and_merge_actions_explain_their_overlap_workflows(self) -> None:
        self.assertEqual(self.window.btn_join.text(), "Join")
        self.assertEqual(self.window.btn_stitch.text(), "Merge Overlaps...")
        self.assertIn("without interpolation", self.window.btn_join.toolTip())
        self.assertIn("conflicting finite overlap", self.window.btn_join.toolTip())
        self.assertIn("without interpolation", self.window.btn_stitch.toolTip())
        self.assertIn("resolve conflicting overlaps", self.window.btn_stitch.toolTip())

    def test_open_and_overlap_have_distinct_discoverable_shortcuts(self) -> None:
        shortcuts = {
            shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
            for shortcut in self.window.findChildren(QShortcut)
        }
        self.assertIn("Ctrl+O", shortcuts)
        self.assertIn("Ctrl+Shift+O", shortcuts)
        self.assertIn("Ctrl+Shift+O", self.window.btn_overlap.toolTip())

    def test_audit_publishes_core_statuses_and_matching_summary_counts(self) -> None:
        healthy = _axis_grid([0.0], 1.0)
        healthy.units.update(
            phase_reference="origin",
            time_convention="exp(-jwt)",
            polarization_basis="V/H",
        )
        warning = _axis_grid([0.0], 2.0)
        warning.rcs_phase.flat[0] = np.nan
        invalid = _axis_grid([0.0], 3.0)
        invalid.rcs_power.flat[0] = -1.0
        for name, dataset in (
            ("Healthy", healthy),
            ("Review", warning),
            ("Invalid", invalid),
        ):
            self.window._add_dataset_row(dataset, name, "Loaded", "")
        self._select_rows_in_order(0, 1, 2)

        with mock.patch(
            "grim_cut_dataset_mixin.DatasetAuditDialog"
        ) as dialog_type:
            self.window._audit_selected_datasets()
            self._wait_for_background()

        reports = dialog_type.call_args.args[0]
        self.assertEqual(
            [report["status"] for _name, report in reports],
            ["ok", "warning", "error"],
        )
        self.assertEqual(
            self.window.status.currentMessage(),
            "Dataset audit complete: 1 pass, 1 warning, 1 fail.",
        )

    def test_stitch_auto_adds_and_status_uses_exact_core_report_counts(self) -> None:
        self.window._add_dataset_row(
            _axis_grid([0.0, 1.0], 1.0), "First", "Loaded", ""
        )
        self.window._add_dataset_row(
            _axis_grid([1.0, 2.0], 2.0), "Second", "Loaded", ""
        )
        self._select_rows_in_order(0, 1)

        with mock.patch("grim_cut_dataset_mixin.StitchDialog") as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = {
                "policy": "priority-first",
                "tol": 1.0e-6,
            }
            self.window._stitch_selected_datasets()
            self._wait_for_background()

        status = self.window.status.currentMessage()
        for expected in (
            "1 overlap cell(s)",
            "1 conflict(s) resolved by priority-first",
            "0 missing output cell(s)",
            "maximum 2 contributor(s) per cell",
        ):
            self.assertIn(expected, status)
        self.assertEqual(self.window.table.rowCount(), 3)
        stitched = self.window.table.item(2, 0).data(Qt.UserRole)
        np.testing.assert_array_equal(stitched.azimuths, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(stitched.rcs_power.ravel(), [1.0, 1.0, 4.0])

    def test_crop_selected_reference_values_convert_for_each_native_unit(self) -> None:
        self.window._add_dataset_row(
            _mixed_unit_grid(radians=False, frequency_hz=True),
            "Degrees Hz",
            "Loaded",
            "",
        )
        self.window._add_dataset_row(
            _mixed_unit_grid(radians=True, frequency_hz=False),
            "Radians GHz reference",
            "Loaded",
            "",
        )
        self._select_rows_in_order(0, 1)
        self._select_parameter_row(self.window.list_az, 1)
        self._select_parameter_row(self.window.list_elev, 1)
        self._select_parameter_row(self.window.list_freq, 1)

        with mock.patch(
            "grim_cut_dataset_mixin.CropDialog"
        ) as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = {
                "mode": "selected",
                "ranges": {
                    "azimuth": None,
                    "elevation": None,
                    "frequency": None,
                },
                "strides": {
                    "azimuth": 1,
                    "elevation": 1,
                    "frequency": 1,
                },
                "selected_polarizations": False,
            }
            self.window._slice_selected()
            self._wait_for_background()

        self.assertEqual(self.window.table.rowCount(), 4)
        degrees_hz = self.window.table.item(2, 0).data(Qt.UserRole)
        radians_ghz = self.window.table.item(3, 0).data(Qt.UserRole)
        np.testing.assert_allclose(degrees_hz.azimuths, [90.0])
        np.testing.assert_allclose(degrees_hz.elevations, [0.0])
        np.testing.assert_allclose(degrees_hz.frequencies, [9.0e9])
        np.testing.assert_allclose(radians_ghz.azimuths, [np.pi / 2.0])
        np.testing.assert_allclose(radians_ghz.elevations, [0.0])
        np.testing.assert_allclose(radians_ghz.frequencies, [9.0])

    def test_crop_rejects_cross_frame_reference_value_transfer_atomically(self) -> None:
        conic = _mixed_unit_grid(radians=False, frequency_hz=False)
        great_circle = _mixed_unit_grid(radians=False, frequency_hz=False)
        great_circle.units.update(
            angular_coordinate_system="great_circle",
            great_circle_coordinate_convention="GRIM_GC_V1",
        )
        self.window._add_dataset_row(conic, "Conic", "Loaded", "")
        self.window._add_dataset_row(great_circle, "GC reference", "Loaded", "")
        self._select_rows_in_order(0, 1)
        self._select_parameter_row(self.window.list_az, 1)

        with mock.patch(
            "grim_cut_dataset_mixin.CropDialog"
        ) as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = {
                "mode": "selected",
                "ranges": {
                    "azimuth": None,
                    "elevation": None,
                    "frequency": None,
                },
                "strides": {
                    "azimuth": 1,
                    "elevation": 1,
                    "frequency": 1,
                },
                "selected_polarizations": False,
            }
            self.window._slice_selected()

        self.assertEqual(self.window.table.rowCount(), 2)
        self.assertIn(
            "angular coordinate system differs",
            self.window.status.currentMessage(),
        )

    def test_regrid_all_axes_converts_active_reference_to_native_units(self) -> None:
        self.window._add_dataset_row(
            _mixed_unit_grid(radians=False, frequency_hz=True),
            "Degrees Hz",
            "Loaded",
            "",
        )
        self.window._add_dataset_row(
            _mixed_unit_grid(radians=True, frequency_hz=False),
            "Radians GHz reference",
            "Loaded",
            "",
        )
        cases = (
            (
                "azimuth",
                0.0,
                180.0,
                45.0,
                "deg",
                np.asarray([0.0, 45.0, 90.0, 135.0, 180.0]),
                np.deg2rad([0.0, 45.0, 90.0, 135.0, 180.0]),
            ),
            (
                "elevation",
                -10.0,
                10.0,
                5.0,
                "deg",
                np.asarray([-10.0, -5.0, 0.0, 5.0, 10.0]),
                np.deg2rad([-10.0, -5.0, 0.0, 5.0, 10.0]),
            ),
            (
                "frequency",
                8.0,
                10.0,
                0.5,
                "GHz",
                np.asarray([8.0, 8.5, 9.0, 9.5, 10.0]) * 1.0e9,
                np.asarray([8.0, 8.5, 9.0, 9.5, 10.0]),
            ),
        )
        for axis, start, stop, step, unit, expected_first, expected_second in cases:
            with self.subTest(axis=axis):
                self._select_rows_in_order(0, 1)
                row_before = self.window.table.rowCount()
                with mock.patch(
                    "grim_cut_dataset_mixin.RegridDialog"
                ) as dialog_type:
                    dialog = dialog_type.return_value
                    dialog.exec.return_value = QDialog.Accepted
                    dialog.get_params.return_value = {
                        "axis": axis,
                        "start": start,
                        "stop": stop,
                        "step": step,
                        "unit": unit,
                    }
                    self.window._interpolate_selected()
                    self._wait_for_background()

                first = self.window.table.item(row_before, 0).data(Qt.UserRole)
                second = self.window.table.item(row_before + 1, 0).data(Qt.UserRole)
                np.testing.assert_allclose(first.get_axis(axis), expected_first)
                np.testing.assert_allclose(second.get_axis(axis), expected_second)

    def test_regrid_recorder_uses_compact_grid_expression(self) -> None:
        source_path = os.path.abspath("regrid_source.grim")
        self.window._add_dataset_row(
            _mixed_unit_grid(radians=False, frequency_hz=False),
            "Regrid source",
            "Loaded",
            source_path,
        )
        self._select_rows_in_order(0)
        with mock.patch(
            "grim_cut_dataset_mixin.RegridDialog"
        ) as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = {
                "axis": "azimuth",
                "start": 0.0,
                "stop": 180.0,
                "step": 0.09,
                "unit": "deg",
            }
            self.window._interpolate_selected()
            self._wait_for_background()

        script = self.window.python_recorder.script
        self.assertIn("np.arange(2001, dtype=float)", script)
        self.assertIn("regrid_axis(", script)
        self.assertLess(len(script), 10_000)
        result = self.window.table.item(1, 0).data(Qt.UserRole)
        self.assertEqual(len(result.azimuths), 2001)

    def test_wrap_phase_only_preserves_missing_phase_and_declares_interval(self) -> None:
        grid = RcsGrid(
            [0.0],
            [0.0],
            [10.0],
            ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            rcs_phase=np.full((1, 1, 1, 1), np.nan),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
        )
        self.window._add_dataset_row(grid, "Magnitude only", "Loaded", "")
        self._select_rows_in_order(0)
        with mock.patch("grim_cut_dataset_mixin.WrapDialog") as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = {
                "azimuth": False,
                "phase": True,
                "mode": "0_360",
            }
            self.window._wrap_selected()
            self._wait_for_background()

        wrapped = self.window.table.item(1, 0).data(Qt.UserRole)
        np.testing.assert_array_equal(wrapped.rcs_power, grid.rcs_power)
        self.assertTrue(np.isnan(wrapped.rcs_phase).all())
        self.assertEqual(wrapped.units["phase_wrap"], "0_360")

    def test_wrap_azimuth_only_honors_radian_axis_without_touching_phase(self) -> None:
        field = np.asarray([1.0 + 1.0j, 2.0 + 2.0j]).reshape(2, 1, 1, 1)
        grid = RcsGrid(
            [-np.pi, 0.0],
            [0.0],
            [10.0],
            ["VV"],
            rcs=field,
            units={
                "azimuth": "rad",
                "elevation": "rad",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
        )
        self.window._add_dataset_row(grid, "Radians", "Loaded", "")
        self._select_rows_in_order(0)
        with mock.patch("grim_cut_dataset_mixin.WrapDialog") as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = {
                "azimuth": True,
                "phase": False,
                "mode": "0_360",
            }
            self.window._wrap_selected()
            self._wait_for_background()

        wrapped = self.window.table.item(1, 0).data(Qt.UserRole)
        np.testing.assert_allclose(wrapped.azimuths, [0.0, np.pi])
        np.testing.assert_allclose(wrapped.rcs_power.ravel(), [8.0, 2.0])
        np.testing.assert_allclose(wrapped.rcs_phase.ravel(), [np.pi / 4.0] * 2)
        self.assertNotIn("phase_wrap", wrapped.units)

    def test_delta_db_requires_exactly_two_operands(self) -> None:
        for index, amplitude in enumerate((1.0, 2.0, 3.0), start=1):
            self.window._add_dataset_row(
                _axis_grid([0.0], amplitude), f"Dataset {index}", "Loaded"
            )
        self._select_rows_in_order(0, 1, 2)
        row_count = self.window.table.rowCount()

        self.window._dbdiff_selected()

        self.assertEqual(self.window.table.rowCount(), row_count)
        self.assertIn("select exactly 2", self.window.status.currentMessage())

    def test_async_statistics_keeps_launch_time_recorder_reference(self) -> None:
        dataset = _axis_grid([0.0, 1.0], 2.0)
        source_path = os.path.abspath("statistics_source.grim")
        dataset_id = self.window._add_dataset_row(
            dataset, "Statistics source", "Loaded", source_path
        )
        self.window.python_recorder.bind_loaded(
            DatasetReference(dataset_id, "Statistics source", source_path)
        )
        self._select_rows_in_order(0)
        started = threading.Event()
        release = threading.Event()
        real_statistics = RcsGrid.statistics_dataset

        def delayed_statistics(source, *args, **kwargs):
            started.set()
            if not release.wait(5.0):
                raise TimeoutError("test did not release statistics worker")
            return real_statistics(source, *args, **kwargs)

        with (
            mock.patch.object(
                RcsGrid,
                "statistics_dataset",
                autospec=True,
                side_effect=delayed_statistics,
            ),
            mock.patch(
                "grim_cut_dataset_mixin.StatisticsDialog"
            ) as dialog_type,
        ):
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = (
                "mean",
                50.0,
                ["azimuth"],
                False,
            )
            self.window._statistics_selected()
            self.assertTrue(started.wait(2.0))
            self.window.table.removeRow(0)
            release.set()
            self._wait_for_background()

        self.assertEqual(self.window.table.rowCount(), 1)
        self.assertIn("statistics_dataset", self.window.python_recorder.script)
        self.assertIn("statistics_source.grim", self.window.python_recorder.script)

    def test_statistics_broadcast_is_memory_guarded_before_worker_start(self) -> None:
        self.window._add_dataset_row(
            _axis_grid([0.0, 1.0, 2.0], 1.0), "Large broadcast", "Loaded"
        )
        self._select_rows_in_order(0)
        with (
            mock.patch(
                "grim_cut_dataset_mixin.StatisticsDialog"
            ) as dialog_type,
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_derived_grid_memory_limit",
                return_value=1,
            ),
        ):
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_params.return_value = (
                "percentile",
                90.0,
                ["azimuth"],
                True,
            )
            self.window._statistics_selected()

        self.assertFalse(self.window._background_job_active())
        self.assertEqual(self.window.table.rowCount(), 1)
        self.assertIn(
            "Statistics blocked before allocation",
            self.window.status.currentMessage(),
        )

    def test_percentile_button_reduces_azimuth_with_default_90(self) -> None:
        azimuths = np.asarray([-30.0, -10.0, 10.0, 30.0])
        base_power = np.asarray(
            [
                [[[1.0]], [[10.0]]],
                [[[2.0]], [[20.0]]],
                [[[3.0]], [[30.0]]],
                [[[100.0]], [[40.0]]],
            ]
        )
        for name, scale in (("First", 1.0), ("Second", 2.0)):
            power = scale * base_power
            dataset = RcsGrid(
                azimuths,
                [-5.0, 5.0],
                [10.0],
                ["VV"],
                rcs_power=power,
                rcs_phase=np.zeros_like(power),
                units={
                    "azimuth": "deg",
                    "elevation": "deg",
                    "frequency": "GHz",
                    "rcs_log_unit": "dBsm",
                    "rcs_linear_quantity": "sigma_3d",
                },
            )
            self.window._add_dataset_row(dataset, name, "Loaded", "")
        self._select_rows_in_order(0, 1)

        with mock.patch(
            "grim_cut_dataset_mixin.QInputDialog.getDouble",
            return_value=(90.0, True),
        ) as prompt:
            self.window._percentile_selected()
            self._wait_for_background()

        self.assertEqual(prompt.call_args.args[3], 90.0)
        self.assertEqual(self.window.table.rowCount(), 4)
        for row, scale in ((2, 1.0), (3, 2.0)):
            result = self.window.table.item(row, 0).data(Qt.UserRole)
            self.assertEqual(result.rcs_power.shape, (4, 2, 1, 1))
            np.testing.assert_allclose(result.azimuths, azimuths)
            expected = np.nanpercentile(
                scale * base_power[:, :, 0, 0],
                90.0,
                axis=0,
                keepdims=True,
            )
            np.testing.assert_allclose(
                result.rcs_power[:, :, 0, 0],
                np.broadcast_to(expected, (azimuths.size, expected.shape[1])),
            )
            self.assertTrue(np.all(np.isnan(result.rcs_phase)))
            self.assertIn("[p90 az]", self.window.table.item(row, 0).text())
        self.assertIn(
            "Percentile created 2 dataset",
            self.window.status.currentMessage(),
        )
        self.assertIn(
            "p90 statistics on linear power",
            self.window.python_recorder.script,
        )

    def test_dataset_add_runs_off_the_gui_thread(self) -> None:
        left = _axis_grid([0.0, 1.0], 1.0)
        right = _axis_grid([0.0, 1.0], 2.0)
        self.window._add_dataset_row(left, "Left", "Loaded")
        self.window._add_dataset_row(right, "Right", "Loaded")
        self._select_rows_in_order(0, 1)
        started = threading.Event()
        release = threading.Event()
        worker_threads: list[int] = []
        gui_thread = threading.get_ident()
        real_add = RcsGrid.incoherent_add

        def delayed_add(source, *args, **kwargs):
            worker_threads.append(threading.get_ident())
            started.set()
            if not release.wait(5.0):
                raise TimeoutError("test did not release dataset-add worker")
            return real_add(source, *args, **kwargs)

        with mock.patch.object(
            RcsGrid,
            "incoherent_add",
            autospec=True,
            side_effect=delayed_add,
        ):
            self.window._incoherent_add_selected()
            self.assertTrue(started.wait(2.0))
            self.assertTrue(self.window._background_job_active())
            self.assertFalse(self.window.dataset_job_progress.isHidden())
            self.assertEqual(self.window.dataset_job_progress.minimum(), 0)
            self.assertEqual(self.window.dataset_job_progress.maximum(), 0)
            self.assertEqual(self.window.table.rowCount(), 2)
            self.assertNotEqual(worker_threads, [gui_thread])
            release.set()
            self._wait_for_background()

        self.assertTrue(self.window.dataset_job_progress.isHidden())
        self.assertEqual(self.window.table.rowCount(), 3)
        result = self.window.table.item(2, 0).data(Qt.UserRole)
        np.testing.assert_allclose(result.rcs_power, 5.0)

    def test_align_runs_off_gui_thread_and_reports_per_dataset_progress(self) -> None:
        reference = _axis_grid([0.0, 1.0], 1.0)
        source = _axis_grid([0.0, 1.0], 2.0)
        self.window._add_dataset_row(reference, "Reference", "Loaded")
        self.window._add_dataset_row(source, "Source", "Loaded")
        self._select_rows_in_order(0, 1)
        started = threading.Event()
        release = threading.Event()
        worker_threads: list[int] = []
        gui_thread = threading.get_ident()
        real_align = RcsGrid.align_to

        def delayed_align(dataset, *args, **kwargs):
            worker_threads.append(threading.get_ident())
            started.set()
            if not release.wait(5.0):
                raise TimeoutError("test did not release alignment worker")
            return real_align(dataset, *args, **kwargs)

        with (
            mock.patch.object(
                RcsGrid, "align_to", autospec=True, side_effect=delayed_align
            ),
            mock.patch("grim_cut_dataset_mixin.AlignDialog") as dialog_type,
        ):
            dialog = dialog_type.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.get_mode.return_value = "exact"
            self.window._align_selected()
            self.assertTrue(started.wait(2.0))
            self.assertTrue(self.window._background_job_active())
            self.assertFalse(self.window.dataset_job_progress.isHidden())
            self.assertEqual(self.window.table.rowCount(), 2)
            self.assertNotEqual(worker_threads, [gui_thread])
            release.set()
            self._wait_for_background()

        self.assertEqual(self.window.table.rowCount(), 3)
        self.assertEqual(self.window.table.item(2, 0).text(), "Source [Aligned]")
        self.assertTrue(self.window.dataset_job_progress.isHidden())

    def test_single_ptm_export_runs_off_gui_thread(self) -> None:
        dataset = _axis_grid([0.0, 1.0], 1.0)
        self.window._add_dataset_row(dataset, "PTM source", "Loaded")
        self._select_rows_in_order(0)
        started = threading.Event()
        release = threading.Event()
        worker_threads: list[int] = []
        gui_thread = threading.get_ident()
        output_path = os.path.abspath("async-export.ptm")

        def delayed_save(_dataset, path, **_kwargs):
            worker_threads.append(threading.get_ident())
            started.set()
            if not release.wait(5.0):
                raise TimeoutError("test did not release PTM writer")
            return path

        with (
            mock.patch.object(
                RcsGrid, "save_ptm", autospec=True, side_effect=delayed_save
            ),
            mock.patch(
                "grim_cut_dataset_mixin.QFileDialog.getSaveFileName",
                return_value=(output_path, "PTM Files (*.ptm)"),
            ),
        ):
            self.window._export_ptm_selected()
            self.assertTrue(started.wait(2.0))
            self.assertTrue(self.window._background_job_active())
            self.assertNotEqual(worker_threads, [gui_thread])
            release.set()
            self._wait_for_background()

        self.assertTrue(self.window.dataset_job_progress.isHidden())
        self.assertIn("Exported async-export.ptm", self.window.status.currentMessage())

    def test_polarization_edit_uses_stored_index_and_updates_workflow_state(self) -> None:
        power = np.asarray([10.0, 20.0, 30.0]).reshape(1, 1, 1, 3)
        source = RcsGrid(
            [0.0],
            [0.0],
            [10.0],
            ["HH", "VV", "CUSTOM"],
            rcs_power=power,
            rcs_phase=np.zeros_like(power),
            units={"frequency": "GHz"},
        )
        source_path = os.path.abspath("source.grim")
        self.window._add_dataset_row(
            source, "Measured", "Loaded source", source_path
        )
        name_item = self.window.table.item(0, 0)
        source_item = self.window.table.item(0, 1)
        dataset_id = str(name_item.data(DATASET_ID_ROLE))
        self.window.table.selectRow(0)
        self.app.processEvents()

        # Polarizations are display-sorted (VV before HH), while UserRole+1
        # deliberately retains VV's underlying source index of 1.
        self.assertEqual(
            [
                self.window.list_pol.item(row).text()
                for row in range(self.window.list_pol.count())
            ],
            ["VV", "HH", "CUSTOM"],
        )
        vv_item = self.window.list_pol.item(0)
        self.assertEqual(vv_item.data(Qt.UserRole + 1), 1)

        with (
            mock.patch.object(
                self.window.python_recorder, "record_method"
            ) as record,
            mock.patch.object(self.window, "_clear_plot") as clear_plot,
        ):
            vv_item.setText("  VERY_LONG_POLARIZATION  ")
            self.app.processEvents()

        edited = name_item.data(Qt.UserRole)
        self.assertIs(self.window.active_dataset, edited)
        np.testing.assert_array_equal(
            edited.polarizations,
            ["HH", "VERY_LONG_POLARIZATION", "CUSTOM"],
        )
        np.testing.assert_array_equal(edited.rcs_power, power)
        self.assertEqual(str(name_item.data(DATASET_ID_ROLE)), dataset_id)
        self.assertTrue(name_item.data(DATASET_DIRTY_ROLE))
        self.assertTrue(name_item.font().bold())
        self.assertEqual(source_item.text(), "Unsaved")
        self.assertEqual(source_item.data(DATASET_PATH_ROLE), source_path)
        self.assertEqual(self.window.table.item(0, 2).text(), edited.history)
        self.assertIn(
            "Edit polarization axis[1]: 'VV' -> 'VERY_LONG_POLARIZATION'",
            edited.history,
        )
        catalog = self.window._dataset_catalog()
        self.assertEqual(len(catalog), 1)
        self.assertIs(catalog[0].grid, edited)
        self.assertEqual(catalog[0].source, "")
        clear_plot.assert_called_once_with()

        record.assert_called_once()
        output_ref, input_ref, method = record.call_args.args
        self.assertEqual(output_ref.dataset_id, dataset_id)
        self.assertEqual(input_ref.dataset_id, dataset_id)
        self.assertEqual(input_ref.path, source_path)
        self.assertEqual(method, "edit_axis_value")
        self.assertEqual(
            record.call_args.kwargs,
            {
                "args": (
                    "polarization",
                    1,
                    "VERY_LONG_POLARIZATION",
                ),
                "comment": "Edit polarization parameter for Measured",
            },
        )
        self.assertEqual(
            self.window.status.currentMessage(),
            "Edited polarization parameter for Measured; dataset is unsaved.",
        )

    def test_invalid_and_noop_polarization_edits_have_no_workflow_side_effects(self) -> None:
        power = np.asarray([10.0, 20.0]).reshape(1, 1, 1, 2)
        source = RcsGrid(
            [0.0],
            [0.0],
            [10.0],
            ["HH", "VV"],
            rcs_power=power,
            rcs_phase=np.zeros_like(power),
            units={"frequency": "GHz"},
        )
        source_path = os.path.abspath("source.grim")
        self.window._add_dataset_row(
            source, "Measured", "Loaded source", source_path
        )
        self.window.table.selectRow(0)
        self.app.processEvents()
        row_dataset = self.window.table.item(0, 0).data(Qt.UserRole)
        original_history = self.window.table.item(0, 2).text()
        vv_item = self.window.list_pol.item(0)
        self.assertEqual(vv_item.text(), "VV")

        with (
            mock.patch.object(
                self.window.python_recorder, "record_method"
            ) as record,
            mock.patch.object(self.window, "_clear_plot") as clear_plot,
        ):
            for entered, status_text in (
                ("  hh  ", "duplicates another channel"),
                ("   ", "must not be blank"),
                ("  VV  ", "value is unchanged"),
            ):
                with self.subTest(entered=entered):
                    vv_item.setText(entered)
                    self.app.processEvents()
                    self.assertEqual(vv_item.text(), "VV")
                    self.assertEqual(vv_item.data(Qt.UserRole), "VV")
                    self.assertIn(status_text, self.window.status.currentMessage())

        name_item = self.window.table.item(0, 0)
        self.assertIs(name_item.data(Qt.UserRole), row_dataset)
        self.assertFalse(name_item.data(DATASET_DIRTY_ROLE))
        self.assertFalse(name_item.font().bold())
        self.assertEqual(self.window.table.item(0, 1).text(), "source.grim")
        self.assertEqual(self.window.table.item(0, 2).text(), original_history)
        record.assert_not_called()
        clear_plot.assert_not_called()

    def test_numeric_gui_edit_reorders_samples_but_rejections_are_atomic(self) -> None:
        power = np.asarray([10.0, 20.0, 30.0]).reshape(3, 1, 1, 1)
        source = RcsGrid(
            [0.0, 1.0, 2.0],
            [0.0],
            [10.0],
            ["VV"],
            rcs_power=power,
            rcs_phase=power / 10.0,
            units={"frequency": "GHz"},
        )
        source_path = os.path.abspath("numeric.grim")
        self.window._add_dataset_row(
            source, "Numeric", "Loaded source", source_path
        )
        self.window.table.selectRow(0)
        self.app.processEvents()
        original = self.window.table.item(0, 0).data(Qt.UserRole)
        azimuth_item = self.window.list_az.item(0)
        self.assertEqual(azimuth_item.data(Qt.UserRole + 1), 0)

        with (
            mock.patch.object(
                self.window.python_recorder, "record_method"
            ) as record,
            mock.patch.object(self.window, "_clear_plot") as clear_plot,
        ):
            for entered, message in (
                ("nan", "must be finite"),
                ("inf", "must be finite"),
                ("1", "duplicates another coordinate"),
            ):
                with self.subTest(rejected=entered):
                    azimuth_item.setText(entered)
                    self.app.processEvents()
                    self.assertEqual(azimuth_item.text(), "0.0")
                    self.assertIs(
                        self.window.table.item(0, 0).data(Qt.UserRole),
                        original,
                    )
                    self.assertIn(message, self.window.status.currentMessage())
            record.assert_not_called()
            clear_plot.assert_not_called()

            azimuth_item.setText("3")
            self.app.processEvents()

        edited = self.window.table.item(0, 0).data(Qt.UserRole)
        np.testing.assert_array_equal(edited.azimuths, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(edited.rcs_power.ravel(), [20.0, 30.0, 10.0])
        np.testing.assert_array_equal(edited.rcs_phase.ravel(), [2.0, 3.0, 1.0])
        self.assertTrue(self.window.table.item(0, 0).data(DATASET_DIRTY_ROLE))
        self.assertIn("stable-sorted axis", edited.history)
        record.assert_called_once()
        self.assertEqual(
            record.call_args.kwargs["args"], ("azimuth", 0, 3.0)
        )
        clear_plot.assert_called_once_with()

    def test_overlap_uses_every_selected_dataset_and_preserves_selection_order(self) -> None:
        datasets = (
            ("A", _axis_grid([0.0, 1.0, 2.0, 3.0], 1.0)),
            ("B", _axis_grid([1.0, 2.0, 3.0], 2.0)),
            ("C", _axis_grid([2.0, 3.0, 4.0], 3.0)),
        )
        for name, dataset in datasets:
            self.window._add_dataset_row(
                dataset,
                name,
                f"Loaded {name}",
                f"{name.lower()}.grim",
            )

        self._select_rows_in_order(2, 0, 1)
        with mock.patch.object(
            self.window.python_recorder, "record_multi_function"
        ) as record:
            self.window._overlap_selected_datasets()
            self._wait_for_background()

        self.assertEqual(self.window.table.rowCount(), 6)
        self.assertEqual(
            [self.window.table.item(row, 0).text() for row in range(3, 6)],
            ["C [Overlap]", "A [Overlap]", "B [Overlap]"],
        )
        for row, source_name in zip(range(3, 6), ("C", "A", "B")):
            result = self.window.table.item(row, 0).data(Qt.UserRole)
            np.testing.assert_array_equal(result.azimuths, [2.0, 3.0])
            self.assertIn(
                f"Overlap with [C, A, B]: {source_name}",
                self.window.table.item(row, 2).text(),
            )

        record.assert_called_once()
        output_refs, function, input_refs = record.call_args.args
        self.assertEqual(function, "RcsGrid.overlap_many")
        self.assertEqual(
            [reference.name for reference in input_refs], ["C", "A", "B"]
        )
        self.assertEqual(
            [reference.name for reference in output_refs],
            ["C [Overlap]", "A [Overlap]", "B [Overlap]"],
        )
        self.assertEqual(record.call_args.kwargs, {
            "kwargs": {"tol": 1.0e-6},
            "comment": "Crop datasets to their common finite overlap",
        })
        self.assertEqual(
            self.window.status.currentMessage(), "Overlap created 3 dataset(s)."
        )

    def test_empty_all_selected_intersection_is_atomic_and_not_recorded(self) -> None:
        # A intersects B at 1 and A intersects C at 0, but there is no point
        # common to A, B, and C. This catches pairwise-to-the-first behavior.
        for name, dataset in (
            ("A", _axis_grid([0.0, 1.0], 1.0)),
            ("B", _axis_grid([1.0, 2.0], 2.0)),
            ("C", _axis_grid([0.0], 3.0)),
        ):
            self.window._add_dataset_row(dataset, name, f"Loaded {name}", "")

        self._select_rows_in_order(0, 1, 2)
        with mock.patch.object(
            self.window.python_recorder, "record_multi_function"
        ) as record:
            self.window._overlap_selected_datasets()
            self._wait_for_background()

        self.assertEqual(self.window.table.rowCount(), 3)
        record.assert_not_called()
        self.assertEqual(
            self.window.status.currentMessage(),
            "Dataset overlap failed: no overlap across one or more axes",
        )


if __name__ == "__main__":
    unittest.main()
