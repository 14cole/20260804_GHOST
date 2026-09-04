import os
import json
import threading
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools/GHOST/Backend"))
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread
from assembly_placement_editor import PlacementEditor, point_array, point_path, project_to_surface
from assembly_response_comparison import read_response_cut, ResponseComparison
from feature_assembly_panel import POINT_PLACEMENT_COLUMNS
import feature_workflow as fw

COMPARISON_UNITS = json.dumps({"azimuth": "deg", "elevation": "deg", "frequency": "GHz", "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm"})


class AuthoringUpdatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_row_and_circle_have_unique_ids_and_respect_local_frame(self):
        seed = ["p", "family", "1", "2", "3", "0", "0", "1", "1", "0", "0"]
        row = point_array(seed, {"p", "p_2"}, count=3, step=[0, .25, 0])
        self.assertEqual([p[0] for p in row], ["p", "p_3", "p_4"])
        np.testing.assert_allclose(np.asarray([p[2:5] for p in row], float), [[1, 2, 3], [1, 2.25, 3], [1, 2.5, 3]])
        ring = point_array(seed, {"p"}, count=4, radius=2.)
        np.testing.assert_allclose(np.asarray([p[2:5] for p in ring], float), [[3, 2, 3], [1, 4, 3], [-1, 2, 3], [1, 0, 3]], atol=1e-14)

    def wait_for(self, condition):
        deadline = time.monotonic()+10
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.002)
        self.assertTrue(condition(), "Background operation did not complete")

    def test_surface_helper_runs_off_ui_and_preserves_positions_for_normals_only(self):
        calls = []
        def load_surface():
            calls.append(QThread.currentThread() != self.app.thread())
            return None, np.asarray([[0, 1], [1, 1], [1, -1], [0, -1]], float), 1.
        editor = PlacementEditor("point", columns=POINT_PLACEMENT_COLUMNS, units="meters", surface_loader=load_surface)
        editor.change([["p", "f", "0", ".25", "1.1", "0", "1", "0", "1", "0", "0"]])
        editor.table.selectRow(0)
        editor.project(False)
        self.wait_for(lambda: editor._thread is None)
        self.assertEqual(calls, [True])
        np.testing.assert_allclose(np.asarray(editor.rows()[0][2:8], float), [0, .25, 1.1, 0, 0, 1])
        editor.undo()
        np.testing.assert_allclose(np.asarray(editor.rows()[0][5:8], float), [0, 1, 0])
        editor._saved_rows = editor.rows()
        editor.reject()

    def test_tree_worker_uses_snapshot_preserves_version_and_honors_cancel(self):
        import assembly_tree as tree_module
        from grim_dataset import RcsGrid
        tree = tree_module.AssemblyTree()
        root = tree._make_node("Parts", tree_module._TYPE_ROOT, edit=False)
        for index in range(2):
            grid = RcsGrid([0.], [0.], [1.], ["VV"], rcs=np.full((1, 1, 1, 1), index+1, complex), units={"rcs_linear_quantity": "sigma_2d", "rcs_log_unit": "dBke"}, extra={"amplitude_version": "2", "combine_role": "coherent"})
            tree_module._attach(tree, tree._make_leaf(str(index), grid), root)
        snapshot = tree_module._BuildNode(root)
        tree.clear()  # all original Qt items are gone before worker execution
        original, calls, results = tree_module.build_assembly_grid, [], []
        def checked(*args, **kwargs):
            calls.append(QThread.currentThread() != self.app.thread())
            return original(*args, **kwargs)
        thread, cancel = QThread(), threading.Event()
        worker = tree_module._AssemblyBuildWorker(snapshot, "strict", cancel)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(results.append)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        try:
            with patch.object(tree_module, "build_assembly_grid", side_effect=checked):
                thread.start()
                self.wait_for(lambda: bool(results) and not thread.isRunning())
            self.assertTrue(all(calls))
            self.assertNotIn("error", results[0])
            np.testing.assert_allclose(results[0]["grid"].rcs, 3.)
            self.assertEqual(str(results[0]["grid"].extra["amplitude_version"]), "2")
            with self.assertRaises(InterruptedError):
                original(snapshot, axis_mode="strict", cancel_check=lambda: True)
        finally:
            thread.quit()
            thread.wait(10000)

    def test_comparison_rejects_2d_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"line.grim"
            units = json.loads(COMPARISON_UNITS)
            units.update(rcs_linear_quantity="sigma_2d", rcs_log_unit="dBke")
            with path.open("wb") as stream:
                np.savez(stream, units=json.dumps(units))
            with self.assertRaisesRegex(ValueError, "sigma_3d"):
                read_response_cut(path, frequency=1., elevation=0., polarization="VV")

    def test_db_difference_uses_only_shared_stored_looks(self):
        comparison = ResponseComparison()
        comparison.show_difference.setChecked(True)
        comparison._show({"curves": [("Body", np.array([0., 30., 60.]), np.array([-10., -20., -30.])), ("Coherent total", np.array([0., 60.]), np.array([-12., -27.]))], "errors": []})
        curve = comparison._difference_axes.lines[0]
        np.testing.assert_array_equal(curve.get_xdata(), [0., 60.])
        np.testing.assert_array_equal(curve.get_ydata(), [-2., 3.])
        comparison.close()

    def test_profile_projection_uses_cad_forward_axis(self):
        profile = np.asarray([[0, 1], [1, 1], [1, -1], [0, -1]], float)
        nearest, normals = project_to_surface([[0, .25, 1.1], [1.1, .5, 0]], profile=profile)
        np.testing.assert_allclose(nearest, [[0, .25, 1], [1, .5, 0]])
        np.testing.assert_allclose(normals, [[0, 0, 1], [1, 0, 0]])

    def test_closed_point_path_uses_uniform_arclength_without_duplicate_end(self):
        seed = ["p", "f", "0", "0", "0", "0", "0", "1", "1", "0", "0"]
        rows, length, spacing = point_path(seed, {"p"}, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], 8)
        self.assertEqual(length, 4.)
        self.assertEqual(spacing, .5)
        positions = np.asarray([row[2:5] for row in rows], float)
        self.assertEqual(len(np.unique(positions, axis=0)), 8)
        np.testing.assert_allclose(positions[[0, 3, 7]], [[0, 0, 0], [1, .5, 0], [0, .5, 0]])

    def test_editor_undo_redo_and_invalid_save_leave_source_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"points.csv"
            initial = ",".join(POINT_PLACEMENT_COLUMNS)+"\np,f,0,0,0,0,0,1,1,0,0\n"
            path.write_text(initial)
            editor = PlacementEditor("point", columns=POINT_PLACEMENT_COLUMNS, units="meters", path=path, validator=fw.read_point_placement_csv)
            editor.table.selectRow(0)
            editor.duplicate()
            self.assertEqual(len(editor.rows()), 2)
            editor.undo()
            self.assertEqual(len(editor.rows()), 1)
            editor.redo()
            self.assertEqual(editor.rows()[1][0], "p_2")
            editor.table.item(1, 0).setText("p")
            with patch("assembly_placement_editor.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                editor.save()
            self.assertEqual(path.read_text(), initial)
            self.assertIn("duplicate", editor.status.text().lower())
            editor._saved_rows = editor.rows()
            editor.reject()

    def test_cut_reader_handles_c_and_fortran_storage_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.arange(1., 73.).reshape(3, 3, 2, 4)
            for storage in (values, np.asfortranarray(values)):
                path = Path(directory)/"response.grim"
                with path.open("wb") as stream:
                    np.savez_compressed(stream, azimuths=[0, 10, 20], elevations=[-10, 0, 10], frequencies=[1., 2.], polarizations=["VV", "HH", "VH", "HV"], rcs_power=storage, units=COMPARISON_UNITS)
                az, db = read_response_cut(path, frequency=2., elevation=0., polarization="VH")
                np.testing.assert_array_equal(az, [0, 10, 20])
                np.testing.assert_allclose(db, 10*np.log10(values[:, 1, 1, 2]))
                with self.assertRaises(InterruptedError):
                    read_response_cut(path, frequency=2., elevation=0., polarization="VH", cancel_check=lambda: True)


if __name__ == "__main__":
    unittest.main()
