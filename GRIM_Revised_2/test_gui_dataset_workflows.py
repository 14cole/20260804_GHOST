"""GUI regressions for parameter edits and multi-dataset overlap workflows."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication

import grim_cut_gui
from grim_cut_dataset_mixin import (
    DATASET_DIRTY_ROLE,
    DATASET_ID_ROLE,
    DATASET_PATH_ROLE,
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

        self.assertEqual(self.window.table.rowCount(), 3)
        record.assert_not_called()
        self.assertEqual(
            self.window.status.currentMessage(),
            "no overlap across one or more axes",
        )


if __name__ == "__main__":
    unittest.main()
