"""Focused shell regressions for the unified GRIM application."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QMimeData, QUrl, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
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
from grim_cut_dataset_mixin import (
    DATASET_DIRTY_ROLE,
    DATASET_ID_ROLE,
    ConicGCDialog,
    RangeCalibrationDialog,
)
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
        self.attached_artifacts: list[tuple[str, str]] = []
        self.setLayout(QVBoxLayout())

    def solve_is_running(self) -> bool:
        return self.running

    def focus_solver(self) -> None:
        self.focus_called = True

    def attach_material_artifact(self, kind: str, path: str) -> None:
        self.attached_artifacts.append((kind, path))


class _FakeFreddyIntegration(QWidget):
    # Deliberately expose a GHOST-shaped signal: the shell must not connect
    # FREDDY material/IBC CSV exports to GRIM's RCS dataset loader.
    files_exported = Signal(list, str)
    attach_to_ghost_requested = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.running = False
        self.focus_called = False
        self.setLayout(QVBoxLayout())

    def job_is_running(self) -> bool:
        return self.running

    def focus_workspace(self) -> None:
        self.focus_called = True


class _FakeRunsWorkspace(QWidget):
    status_changed = Signal(str)
    results_downloaded = Signal(str)

    def __init__(self, parent=None, **_kwargs) -> None:
        super().__init__(parent)
        self.running = False
        self.focus_called = False
        self.save_count = 0
        self.setLayout(QVBoxLayout())

    def job_is_running(self) -> bool:
        return self.running

    def busy_operation(self) -> str | None:
        return "HPC upload and submission" if self.running else None

    def focus_workspace(self) -> None:
        self.focus_called = True

    def save_settings(self) -> None:
        self.save_count += 1


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

    def test_tabs_have_one_canonical_assembly_workspace(self) -> None:
        labels = [
            self.window.main_tabs.tabText(index)
            for index in range(self.window.main_tabs.count())
        ]
        self.assertEqual(
            labels,
            [
                "Plotting",
                "ISAR",
                "PPT",
                "Assembly",
                "GHOST",
                "FREDDY",
                "Runs",
                "Python",
            ],
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.ppt_workspace), 2
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.assembly_workspace), 3
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.ghost_integration), 4
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.freddy_integration), 5
        )
        self.assertEqual(
            self.window.main_tabs.indexOf(self.window.runs_workspace), 6
        )
        self.assertNotIn("ppt", self.window._plot_contexts)

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
        left_tabs = self.window.assembly_workspace.left_tabs
        self.assertEqual(
            [left_tabs.tabText(index) for index in range(left_tabs.count())],
            ["Place Features", "Datasets + Preview Layers"],
        )
        self.assertIs(
            left_tabs.currentWidget(),
            self.window.assembly_workspace.place_features_tab,
        )

    def test_dark_theme_explicitly_styles_logs_and_workspace_scroll_surfaces(self) -> None:
        qss = self.window.styleSheet().lower()
        self.assertIn("qplaintextedit", qss)
        self.assertIn("background: #0b1222", qss)
        self.assertIn("color: #dbeafe", qss)
        self.assertIn("qscrollarea#runscontrolsscroll", qss)
        self.assertIn("qscrollarea#pptcontrolsscroll", qss)
        self.assertEqual(
            self.window.ppt_workspace.controls_content.objectName(),
            "pptControlsContent",
        )

    def test_python_clear_confirms_and_resets_dirty_state(self) -> None:
        self.window.python_recorder._lines.extend(["answer = 42", ""])
        self.window.python_recorder._notify()
        script = self.window.python_recorder.script
        self.assertTrue(self.window._python_script_is_dirty())
        self.assertEqual(
            self.window.main_tabs.tabText(
                self.window.main_tabs.indexOf(self.window.tab_python)
            ),
            "Python*",
        )
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        with mock.patch.object(
            grim_cut_gui.QMessageBox, "question", return_value=buttons.No
        ):
            self.window._clear_python_script()
        self.assertEqual(self.window.python_recorder.script, script)

        with mock.patch.object(
            grim_cut_gui.QMessageBox, "question", return_value=buttons.Yes
        ):
            self.window._clear_python_script()
        self.assertEqual(
            self.window.python_recorder.script, self.window._python_empty_script
        )
        self.assertFalse(self.window._python_script_is_dirty())
        self.assertEqual(
            self.window.main_tabs.tabText(
                self.window.main_tabs.indexOf(self.window.tab_python)
            ),
            "Python",
        )

    def test_python_save_failure_preserves_existing_script(self) -> None:
        self.window.python_recorder._lines.extend(["answer = 42", ""])
        self.window.python_recorder._notify()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "workflow.py"
            target.write_text("original\n", encoding="utf-8")
            with (
                mock.patch.object(
                    grim_cut_gui.QFileDialog,
                    "getSaveFileName",
                    return_value=(str(target), "Python Files (*.py)"),
                ),
                mock.patch.object(
                    grim_cut_gui.os,
                    "replace",
                    side_effect=OSError("publication failed"),
                ),
                mock.patch.object(grim_cut_gui.QMessageBox, "critical"),
            ):
                self.assertFalse(self.window._save_python_script())

            self.assertEqual(target.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(list(Path(temp_dir).glob(".workflow.py.*.tmp")), [])
            self.assertTrue(self.window._python_script_is_dirty())

    def test_ppt_catalog_tracks_stable_dataset_ids_and_main_selection(self) -> None:
        first = _grid(1.0)
        second = _grid(2.0)
        self.window._add_dataset_row(first, "First", "", "first.grim")
        self.window._add_dataset_row(second, "Second", "", "second.grim")
        first_item = self.window.table.item(0, 0)
        second_item = self.window.table.item(1, 0)
        first_id = str(first_item.data(DATASET_ID_ROLE))
        second_id = str(second_item.data(DATASET_ID_ROLE))
        self.assertTrue(first_id)
        self.assertTrue(second_id)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(
            self.window.ppt_workspace.dataset_ids_in_order(),
            (first_id, second_id),
        )

        self.window.ppt_workspace.select_dataset_ids((second_id,))
        first_item.setText("Renamed first")
        self.app.processEvents()
        self.assertEqual(
            self.window.ppt_workspace.selected_dataset_ids(), (second_id,)
        )
        self.assertEqual(
            self.window.ppt_workspace._catalog[first_id].name, "Renamed first"
        )

        self.window.table.clearSelection()
        self.window.table.selectRow(0)
        self.window.ppt_workspace.use_selected_button.click()
        self.assertEqual(
            self.window.ppt_workspace.selected_dataset_ids(), (first_id,)
        )

        self.window.table.clearSelection()
        self.window.table.selectRow(1)
        self.window._delete_selected_datasets()
        self.assertEqual(
            self.window.ppt_workspace.dataset_ids_in_order(), (first_id,)
        )

    def test_feature_input_preview_becomes_visibly_stale_after_edit(self) -> None:
        plan = SimpleNamespace(
            preview_stage="input",
            surface_triangles_cad_m=np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
            ),
            body_profile_rho_z_m=None,
            point_locations_cad_m={"fastener": np.asarray([[0.2, 0.2, 0.0]])},
            line_paths_cad_m={},
        )

        self.window._on_feature_preview_ready(plan)
        self.assertEqual(
            self.window.assembly_workspace.scene_canvas.preview_stage, "input"
        )

        self.window.feature_assembly_panel.preview_stale.emit("CSV changed.")

        self.assertEqual(
            self.window.assembly_workspace.scene_canvas.preview_stage, "stale"
        )
        self.assertIn("CSV changed", self.window.assembly_workspace.lbl_status.text())

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

    def test_freddy_material_handoff_is_typed_directly_to_ghost(self) -> None:
        self.window.freddy_integration.attach_to_ghost_requested.emit(
            "ibc", "nominal_ibc.csv"
        )
        self.assertEqual(
            self.window.ghost_integration.attached_artifacts,
            [("ibc", "nominal_ibc.csv")],
        )
        self.assertEqual(self.window.loaded_path_batches, [])

    def test_busy_imports_queue_fifo_and_deduplicate_casefolded_paths(self) -> None:
        # _RecordingWindow overrides the public hook so ordinary shell-routing
        # tests stay synchronous. Exercise the mixin implementation directly.
        with mock.patch.object(
            self.window, "_background_job_active", return_value=True
        ):
            grim_cut_dataset_mixin.DatasetOpsMixin._handle_files_dropped(
                self.window, ["first.grim", "FIRST.GRIM", "second.grim"]
            )
            grim_cut_dataset_mixin.DatasetOpsMixin._handle_files_dropped(
                self.window, ["SECOND.grim", "third.grim"]
            )

        self.assertEqual(
            [batch[0] for batch in self.window._pending_import_batches],
            [("first.grim", "second.grim"), ("third.grim",)],
        )
        with mock.patch.object(
            self.window, "_start_dataset_import_batch", return_value=True
        ) as start:
            self.window._on_background_thread_finished()
            self.window._on_background_thread_finished()
        self.assertEqual(
            [call.args[0] for call in start.call_args_list],
            [["first.grim", "second.grim"], ["third.grim"]],
        )

    def test_loaded_catalog_prefers_container_over_solver_source_path(self) -> None:
        dataset = _grid()
        dataset.source_path = os.path.abspath("source_geometry.geo")
        container = os.path.abspath("solver_output.grim")
        self.window._ensure_background_worker_state()
        self.window._on_load_worker_finished(
            {
                "loaded": [
                    {
                        "index": 0,
                        "path": container,
                        "file_name": "solver_output.grim",
                        "name": "solver_output",
                        "history": "GHOST solve",
                        "dataset": dataset,
                    }
                ],
                "failed": [],
                "ignored": 0,
                "total_supported": 1,
            }
        )

        self.assertEqual(self.window._dataset_catalog()[0].source, container)
        self.assertEqual(
            self.window.table.item(0, 1).toolTip(), container
        )

    def test_parameter_headers_follow_units_and_axes_are_editable(self) -> None:
        dataset = _grid()
        dataset.units.update(
            {
                "frequency": "Hz",
                "azimuth": "rad",
                "elevation": "rad",
                "angular_coordinate_system": "great_circle",
            }
        )
        self.window._add_dataset_row(dataset, "GC", "Loaded", "gc.grim")
        self.window.table.selectRow(0)
        self.app.processEvents()

        self.assertEqual(self.window.lbl_freq.text(), "Frequency (Hz)")
        self.assertEqual(self.window.lbl_elev.text(), "Pitch (rad)")
        self.assertEqual(self.window.lbl_az.text(), "Aspect (rad)")
        for widget in (
            self.window.list_pol,
            self.window.list_freq,
            self.window.list_elev,
            self.window.list_az,
        ):
            self.assertTrue(
                widget.editTriggers() & QAbstractItemView.DoubleClicked
            )
            self.assertTrue(widget.item(0).flags() & Qt.ItemIsEditable)

        self.window.list_pol.item(0).setText("TOTAL")
        self.app.processEvents()
        edited_item = self.window.table.item(0, 0)
        edited = edited_item.data(Qt.UserRole)
        self.assertEqual(edited.polarizations.tolist(), ["TOTAL"])
        self.assertIs(self.window.active_dataset, edited)
        self.assertTrue(edited_item.data(DATASET_DIRTY_ROLE))
        self.assertEqual(self.window.table.item(0, 1).text(), "Unsaved")
        self.assertIn("Edit polarization axis[0]", edited.history)

    def test_batch_save_preserves_per_row_provenance_for_shared_grid(self) -> None:
        shared = _grid()
        shared.source_path = os.path.abspath("original_source.grim")
        self.window._add_dataset_row(shared, "First branch", "First operation")
        self.window._add_dataset_row(shared, "Second branch", "Second operation")
        first_history = self.window.table.item(0, 2).text()
        second_history = self.window.table.item(1, 2).text()
        self.assertEqual(first_history, "First operation")
        self.assertEqual(second_history, "Second operation")
        self.assertIsNot(
            self.window.table.item(0, 0).data(Qt.UserRole),
            self.window.table.item(1, 0).data(Qt.UserRole),
        )
        self.assertEqual(
            [entry.source for entry in self.window._dataset_catalog()], ["", ""]
        )

        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(
                self.window._save_rows_to_directory(
                    [0, 1], tmp, dialog_title="Test Save"
                )
            )
            first_saved = RcsGrid.load(os.path.join(tmp, "First branch.grim"))
            second_saved = RcsGrid.load(os.path.join(tmp, "Second branch.grim"))
            self.assertEqual(
                [entry.source for entry in self.window._dataset_catalog()],
                [
                    os.path.join(tmp, "First branch.grim"),
                    os.path.join(tmp, "Second branch.grim"),
                ],
            )

        self.assertEqual(first_saved.history, first_history)
        self.assertEqual(second_saved.history, second_history)
        self.assertEqual(self.window.table.item(0, 2).text(), first_history)
        self.assertEqual(self.window.table.item(1, 2).text(), second_history)
        self.assertFalse(self.window._dataset_row_is_dirty(0))
        self.assertFalse(self.window._dataset_row_is_dirty(1))

    def test_dataset_loader_always_finishes_after_pool_setup_failure(self) -> None:
        worker = grim_cut_dataset_mixin._DatasetLoadWorker(
            [(0, "first.grim"), (1, "second.grim")]
        )
        payloads = []
        worker.finished.connect(payloads.append)
        with mock.patch.object(
            grim_cut_dataset_mixin,
            "_recommended_loader_workers",
            side_effect=RuntimeError("pool unavailable"),
        ):
            worker.run()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["loaded"], [])
        self.assertIn("pool unavailable", payloads[0]["failed"][0])

    def test_grim_archive_expansion_caps_parallel_loader_workers(self) -> None:
        tasks = [(0, "first.grim"), (1, "second.grim")]
        expanded = 80 * 1024**2
        with (
            mock.patch.object(os.path, "getsize", return_value=1024**2),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_grim_archive_uncompressed_bytes",
                return_value=expanded,
            ),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_available_memory_bytes",
                return_value=1024**3,
            ),
            mock.patch.object(grim_cut_dataset_mixin.os, "cpu_count", return_value=8),
        ):
            workers = grim_cut_dataset_mixin._recommended_loader_workers(tasks)

        self.assertEqual(workers, 1)

    def test_grim_memory_estimate_reads_uncompressed_archive_metadata(self) -> None:
        payload_size = 2 * 1024**2
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "compressed.grim")
            with zipfile.ZipFile(
                path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("rcs_power.npy", b"\0" * payload_size)

            retained, peak = (
                grim_cut_dataset_mixin._dataset_load_memory_estimate(path)
            )

        self.assertEqual(retained, payload_size)
        self.assertGreater(peak, retained)

    def test_unknown_memory_rejects_unsafe_compressed_grim_batch(self) -> None:
        tasks = [(0, "first.grim"), (1, "second.grim")]
        with (
            mock.patch.object(os.path, "getsize", return_value=1024**2),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_grim_archive_uncompressed_bytes",
                return_value=300 * 1024**2,
            ),
            mock.patch.object(
                grim_cut_dataset_mixin,
                "_available_memory_bytes",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(MemoryError, "Load fewer .grim files"):
                grim_cut_dataset_mixin._recommended_loader_workers(tasks)

    def test_batch_save_rejects_sanitized_casefold_collision_before_write(self) -> None:
        self.window._add_dataset_row(_grid(), "Body/A", "first")
        self.window._add_dataset_row(_grid(), "body:a", "second")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                grim_cut_dataset_mixin.QMessageBox, "critical"
            ) as critical:
                result = self.window._save_rows_to_directory(
                    [0, 1], tmp, dialog_title="Test Save"
                )
            self.assertFalse(result)
            self.assertEqual(os.listdir(tmp), [])
        critical.assert_called_once()

    def test_delete_requires_confirmation_for_unsaved_derived_rows(self) -> None:
        start_count = self.window.table.rowCount()
        self.window._add_dataset_row(_grid(), "Unsaved branch", "Derived")
        row = self.window.table.rowCount() - 1
        self.window.table.clearSelection()
        self.window.table.selectRow(row)
        buttons = getattr(
            grim_cut_dataset_mixin.QMessageBox,
            "StandardButton",
            grim_cut_dataset_mixin.QMessageBox,
        )
        with mock.patch.object(
            grim_cut_dataset_mixin.QMessageBox,
            "question",
            return_value=buttons.No,
        ) as question:
            self.window._delete_selected_datasets()
        self.assertEqual(self.window.table.rowCount(), start_count + 1)
        self.assertIn("Unsaved branch", question.call_args.args[2])

        with mock.patch.object(
            grim_cut_dataset_mixin.QMessageBox,
            "question",
            return_value=buttons.Yes,
        ):
            self.window._delete_selected_datasets()
        self.assertEqual(self.window.table.rowCount(), start_count)

    def test_batch_save_confirms_all_existing_replacements_once(self) -> None:
        self.window._add_dataset_row(_grid(1.0), "First", "first")
        self.window._add_dataset_row(_grid(2.0), "Second", "second")
        buttons = getattr(
            grim_cut_dataset_mixin.QMessageBox,
            "StandardButton",
            grim_cut_dataset_mixin.QMessageBox,
        )
        with tempfile.TemporaryDirectory() as tmp:
            for filename in ("First.grim", "Second.grim"):
                with open(os.path.join(tmp, filename), "wb") as stream:
                    stream.write(b"old")
            with mock.patch.object(
                grim_cut_dataset_mixin.QMessageBox,
                "question",
                return_value=buttons.Yes,
            ) as question:
                result = self.window._save_rows_to_directory(
                    [0, 1], tmp, dialog_title="Test Save"
                )

            self.assertTrue(result)
            question.assert_called_once()
            self.assertEqual(float(RcsGrid.load(os.path.join(tmp, "First.grim")).rcs_power.item()), 1.0)
            self.assertEqual(float(RcsGrid.load(os.path.join(tmp, "Second.grim")).rcs_power.item()), 4.0)

    def test_batch_save_retains_prior_backup_when_restore_itself_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "First.grim"
            second = Path(tmp) / "Second.grim"
            first.write_bytes(b"old first")
            second.write_bytes(b"old second")
            original_replace = grim_cut_dataset_mixin.os.replace

            def fail_publish_then_restore(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    destination_path == second
                    and source_path.name.endswith(".staging.grim")
                ):
                    raise OSError("second publication failed")
                if (
                    destination_path == first
                    and source_path.name.endswith(".backup")
                ):
                    raise OSError("first restoration failed")
                return original_replace(source, destination)

            with mock.patch.object(
                grim_cut_dataset_mixin.os,
                "replace",
                side_effect=fail_publish_then_restore,
            ):
                with self.assertRaisesRegex(
                    grim_cut_dataset_mixin._GrimBatchRollbackError,
                    "Retained .grim-backup file",
                ):
                    grim_cut_dataset_mixin._stage_and_publish_grim_batch(
                        [
                            (_grid(1.0), str(first), "first"),
                            (_grid(2.0), str(second), "second"),
                        ]
                    )

            retained = list(Path(tmp).glob(".grim-backup-*.backup"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"old first")
            self.assertEqual(second.read_bytes(), b"old second")

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

    def test_native_workspace_cancel_blocks_unified_close(self) -> None:
        request_close = mock.Mock(return_value=False)
        self.window.ghost_integration.workspace = SimpleNamespace(
            request_close=request_close
        )
        event = QCloseEvent()
        self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        request_close.assert_called_once_with(self.window)
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.ghost_integration
        )

    def test_unsaved_python_script_can_cancel_unified_close(self) -> None:
        self.window.python_recorder._lines.extend(["value = 1", ""])
        self.window.python_recorder._notify()
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        event = QCloseEvent()
        with mock.patch.object(
            grim_cut_gui.QMessageBox, "warning", return_value=buttons.Cancel
        ) as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertIn("Python recorder", warning.call_args.args[2])
        self.assertIs(self.window.main_tabs.currentWidget(), self.window.tab_python)

    def test_accepted_unified_close_disposes_ppt_preview_directory(self) -> None:
        preview_directory = Path(self.window.ppt_workspace._preview_temp.name)
        (preview_directory / "marker.png").write_bytes(b"preview")
        event = QCloseEvent()
        self.window.closeEvent(event)

        self.assertTrue(event.isAccepted())
        self.assertFalse(preview_directory.exists())

    def test_running_ppt_export_blocks_close_and_focuses_workspace(self) -> None:
        event = QCloseEvent()
        with (
            mock.patch.object(
                self.window.ppt_workspace, "job_is_running", return_value=True
            ),
            mock.patch.object(
                self.window.ppt_workspace,
                "busy_operation",
                return_value="PowerPoint report export",
            ),
            mock.patch.object(
                self.window.ppt_workspace, "focus_workspace"
            ) as focus_workspace,
            mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning,
        ):
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        focus_workspace.assert_called_once()
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.ppt_workspace
        )

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

    def test_isar_worker_is_cancelled_and_blocks_close_until_done(self) -> None:
        current_cancel = threading.Event()
        pending_cancel = threading.Event()
        self.window._isar_busy = True
        self.window._isar_cancel_event = current_cancel
        self.window._isar_pending = {"_cancel_event": pending_cancel}
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertTrue(current_cancel.is_set())
        self.assertTrue(pending_cancel.is_set())
        self.assertIsNone(self.window._isar_pending)
        self.assertIs(self.window.main_tabs.currentWidget(), self.window.tab_isar)
        warning.assert_called_once()

    def test_unsaved_derived_dataset_blocks_close_when_cancelled(self) -> None:
        self.window._add_dataset_row(_grid(), "Unsaved result", "Derived")
        buttons = getattr(
            grim_cut_gui.QMessageBox,
            "StandardButton",
            grim_cut_gui.QMessageBox,
        )
        event = QCloseEvent()
        with mock.patch.object(
            grim_cut_gui.QMessageBox, "warning", return_value=buttons.Cancel
        ) as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertIn("Unsaved result", warning.call_args.args[2])

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

    def test_running_hpc_transfer_blocks_close_but_remote_jobs_do_not(self) -> None:
        self.window.runs_workspace.running = True
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        warning.assert_called_once()
        self.assertIn("HPC upload", warning.call_args.args[2])
        self.assertIs(
            self.window.main_tabs.currentWidget(), self.window.runs_workspace
        )
        self.assertTrue(self.window.runs_workspace.focus_called)

        # A tracked SLURM job is owned by the remote scheduler. Once the
        # foreground SSH process is done, it must not trap GRIM open.
        self.window.runs_workspace.running = False
        event = QCloseEvent()
        with mock.patch.object(grim_cut_gui.QMessageBox, "warning") as warning:
            self.window.closeEvent(event)
        self.assertTrue(event.isAccepted())
        warning.assert_not_called()
        self.assertGreater(self.window.runs_workspace.save_count, 0)

    def test_downloaded_hpc_result_tree_reuses_dataset_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "results" / "FRD" / "first.grim"
            second = root / "results" / "OPN" / "second.ptm"
            ignored = root / "results" / "solver.log"
            for path in (first, second, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            self.window.runs_workspace.results_downloaded.emit(str(root))

        self.assertEqual(
            self.window.loaded_path_batches,
            [[str(first), str(second)]],
        )

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
