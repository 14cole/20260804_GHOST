"""Cooperative progress/cancellation and atomic-output guarantees."""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import feature_sum  # noqa: E402
import feature_workflow  # noqa: E402
import grim_io  # noqa: E402
from line_expand import SeamCoefficients, expand_perimeter  # noqa: E402


GRID = {
    "frequencies_ghz": [1.0],
    "azimuths_deg": [0.0, 45.0],
    "elevations_deg": [0.0],
    "axis_az_deg": 0.0,
    "axis_el_deg": 0.0,
    "roll_deg": 0.0,
}


def _write_empty_base(path: Path) -> None:
    feature_sum.export_radar_grim(
        str(path), bor_result=None, placements=[], **GRID
    )


class FeatureJobControlTests(unittest.TestCase):
    def test_cancelled_atomic_build_keeps_existing_output_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            original = b"existing verified output"
            output.write_bytes(original)
            calls = 0

            def cancel_after_setup():
                nonlocal calls
                calls += 1
                return calls >= 4

            with self.assertRaisesRegex(InterruptedError, "cancelled"):
                feature_sum.add_features_to_monostatic_grim(
                    str(base),
                    str(output),
                    radar_grid=GRID,
                    declared_coherent_base=True,
                    cancel_check=cancel_after_setup,
                )

            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
            self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_final_cancel_callback_cannot_replace_unreviewed_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            calls = 0
            final_stage_written = False
            mutated = False
            intruder = b"created by side-effecting callback"
            real_save = grim_io._save_grim_npz

            def track_final_stage(payload, path):
                nonlocal final_stage_written
                saved = real_save(payload, path)
                if ".tmp." in Path(path).name:
                    final_stage_written = True
                return saved

            def mutate_on_final_check():
                nonlocal calls, mutated
                calls += 1
                if final_stage_written and not mutated:
                    output.write_bytes(intruder)
                    mutated = True
                return False

            with mock.patch.object(
                grim_io, "_save_grim_npz", side_effect=track_final_stage
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "output was created by another process"
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                        cancel_check=mutate_on_final_check,
                    )
            self.assertTrue(mutated)
            self.assertGreater(calls, 0)
            self.assertEqual(output.read_bytes(), intruder)
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
            self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_successful_build_reports_monotone_progress_through_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            updates = []

            saved = feature_sum.add_features_to_monostatic_grim(
                str(base),
                str(output),
                radar_grid=GRID,
                declared_coherent_base=True,
                progress_callback=lambda done, total, message: updates.append(
                    (int(done), int(total), str(message))
                ),
            )

            self.assertEqual(Path(saved), output.resolve())
            percentages = [round(100.0 * done / total) for done, total, _ in updates]
            self.assertEqual(percentages[0], 0)
            self.assertEqual(percentages[-1], 100)
            self.assertEqual(percentages, sorted(percentages))
            self.assertTrue(any("GHz" in message for _, _, message in updates))
            self.assertIn("published", updates[-1][2])

    def test_line_direction_loop_observes_cancellation(self):
        coefficient = SeamCoefficients(
            1.0,
            np.asarray([0.0, 90.0, 180.0]),
            np.ones(3, dtype=np.complex128),
            np.ones(3, dtype=np.complex128),
        )
        segment = np.asarray([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
        normals = np.asarray([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            expand_perimeter(
                segment,
                coefficient,
                None,
                np.asarray([[0.0, 0.0, 1.0]]),
                segment_normals=normals,
                cancel_check=lambda: True,
            )

    def test_sidecar_created_during_build_keeps_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            sidecar = root / "feature.grim.feature.json"
            _write_empty_base(base)
            original = b"existing verified output"
            output.write_bytes(original)
            original_save = grim_io._save_grim_npz

            def save_then_create_sidecar(payload, path):
                saved = original_save(payload, path)
                sidecar.write_text("{}", encoding="utf-8")
                return saved

            with mock.patch.object(
                grim_io, "_save_grim_npz", side_effect=save_then_create_sidecar
            ):
                with self.assertRaisesRegex(RuntimeError, "sidecar state changed"):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                        expected_absent_paths=[str(sidecar)],
                    )

            self.assertEqual(output.read_bytes(), original)

    def test_final_progress_callback_failure_does_not_mask_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)

            def callback(done, total, _message):
                if int(done) == int(total):
                    raise RuntimeError("display disconnected")

            saved = feature_sum.add_features_to_monostatic_grim(
                str(base),
                str(output),
                radar_grid=GRID,
                declared_coherent_base=True,
                progress_callback=callback,
            )
            self.assertEqual(Path(saved), output.resolve())
            self.assertTrue(output.is_file())

    def test_second_temp_creation_failure_cleans_first_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            real_mkstemp = tempfile.mkstemp
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated second staging failure")
                return real_mkstemp(*args, **kwargs)

            with mock.patch.object(
                feature_sum.tempfile, "mkstemp", side_effect=fail_second
            ):
                with self.assertRaisesRegex(OSError, "second staging failure"):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            self.assertEqual(list(root.glob(".*.features.*.grim")), [])
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])

    def test_run_progress_is_monotone_across_prepare_and_execute(self):
        updates = []
        request = feature_workflow.FeatureAssemblyRequest(
            base_grim="clean.grim", output_grim="assembled.grim"
        )

        def fake_prepare(_request, *, cancel_check, progress_callback):
            self.assertIsNone(cancel_check)
            progress_callback(0, 100, "prepare start")
            progress_callback(100, 100, "prepare done")
            return object()

        def fake_execute(_plan, *, cancel_check, progress_callback):
            self.assertIsNone(cancel_check)
            progress_callback(0, 100, "execute start")
            progress_callback(100, 100, "execute done")
            return "assembled.grim"

        with (
            mock.patch.object(
                feature_workflow,
                "prepare_feature_assembly",
                side_effect=fake_prepare,
            ),
            mock.patch.object(
                feature_workflow,
                "execute_feature_assembly",
                side_effect=fake_execute,
            ),
        ):
            saved = feature_workflow.run_feature_assembly(
                request,
                progress_callback=lambda done, total, message: updates.append(
                    (int(done), int(total), str(message))
                ),
            )

        self.assertEqual(saved, "assembled.grim")
        self.assertEqual([done for done, _total, _message in updates], [0, 25, 25, 100])

    def test_concurrent_writers_cannot_silently_replace_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            first_entered = threading.Event()
            release_first = threading.Event()
            real_save = grim_io._save_grim_npz
            results = []
            errors = []

            def gated_save(payload, path):
                if threading.current_thread().name == "first-writer":
                    first_entered.set()
                    self.assertTrue(release_first.wait(timeout=10.0))
                return real_save(payload, path)

            def build(history):
                try:
                    results.append(feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                        history=history,
                    ))
                except Exception as exc:  # captured for cross-thread assertion
                    errors.append(exc)

            with mock.patch.object(
                grim_io, "_save_grim_npz", side_effect=gated_save
            ):
                first = threading.Thread(
                    target=build, args=("first",), name="first-writer"
                )
                first.start()
                self.assertTrue(first_entered.wait(timeout=10.0))
                second = threading.Thread(
                    target=build, args=("second",), name="second-writer"
                )
                second.start()
                second.join(timeout=10.0)
                self.assertFalse(second.is_alive())
                release_first.set()
                first.join(timeout=10.0)
                self.assertFalse(first.is_alive())

            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 1)
            self.assertRegex(str(errors[0]), "already publishing")
            with np.load(output, allow_pickle=False) as stored:
                self.assertIn("first", str(stored["history"]))

    def test_cleanup_failure_after_publish_does_not_report_false_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            real_unlink = feature_sum.os.unlink
            failed_once = False

            def fail_component_cleanup(path):
                nonlocal failed_once
                if not failed_once and ".features." in str(path):
                    failed_once = True
                    raise PermissionError("simulated antivirus staging lock")
                return real_unlink(path)

            with mock.patch.object(
                feature_sum.os, "unlink", side_effect=fail_component_cleanup
            ):
                saved = feature_sum.add_features_to_monostatic_grim(
                    str(base),
                    str(output),
                    radar_grid=GRID,
                    declared_coherent_base=True,
                )
            self.assertEqual(Path(saved), output.resolve())
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
