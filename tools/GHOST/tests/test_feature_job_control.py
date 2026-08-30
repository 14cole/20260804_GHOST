"""Cooperative progress/cancellation and atomic-output guarantees."""

from __future__ import annotations

import os
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
    @staticmethod
    def _scalar(payload, key):
        value = np.asarray(payload[key]).reshape(()).item()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def test_feature_progress_weights_scale_with_exact_shadow_and_field_work(self):
        point_shadow, line_shadow, per_frequency = (
            feature_sum._feature_export_progress_weights(
                look_count=100,
                frequency_count=3,
                point_count=2,
                line_count=1,
                line_shadow_piece_count=50,
                has_point_shadow=True,
                has_line_shadow=True,
            )
        )

        self.assertEqual(point_shadow, 200)
        self.assertEqual(line_shadow, 5_000)
        self.assertEqual(per_frequency, 5_200)

        without_shadow = feature_sum._feature_export_progress_weights(
            look_count=100,
            frequency_count=3,
            point_count=2,
            line_count=1,
            line_shadow_piece_count=0,
            has_point_shadow=False,
            has_line_shadow=False,
        )
        self.assertEqual(without_shadow, (0, 0, 300))

    def test_success_publishes_total_and_reusable_feature_delta_with_shared_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            _write_empty_base(base)

            saved = feature_sum.add_features_to_monostatic_grim(
                str(base),
                str(output),
                radar_grid=GRID,
                declared_coherent_base=True,
                feature_provenance={"case": "publication-test"},
            )

            self.assertEqual(Path(saved), output.resolve())
            self.assertEqual(
                Path(feature_sum.feature_only_output_path(str(output))),
                features.resolve(),
            )
            self.assertTrue(features.is_file())
            with (
                np.load(base, allow_pickle=False) as clean,
                np.load(output, allow_pickle=False) as total,
                np.load(features, allow_pickle=False) as delta,
            ):
                self.assertEqual(
                    self._scalar(total, "assembly_response_role"),
                    "body_plus_features",
                )
                self.assertEqual(
                    self._scalar(delta, "assembly_response_role"),
                    "features_only_delta",
                )
                self.assertEqual(
                    self._scalar(total, "assembly_base_sha256"),
                    self._scalar(delta, "assembly_base_sha256"),
                )
                self.assertEqual(
                    self._scalar(total, "assembly_base_response_sha256"),
                    self._scalar(delta, "assembly_base_response_sha256"),
                )
                self.assertEqual(
                    self._scalar(total, "assembly_base_response_sha256"),
                    feature_sum.assembly_response_physics_sha256(
                        feature_sum._load_grim(str(base)), base.name
                    ),
                )
                self.assertEqual(
                    self._scalar(total, "feature_provenance_json"),
                    self._scalar(delta, "feature_provenance_json"),
                )
                delta_records = feature_sum.json.loads(
                    self._scalar(delta, "feature_provenance_json")
                )
                self.assertEqual(len(delta_records), 1)
                self.assertEqual(
                    delta_records[0]["details"]["case"],
                    "publication-test",
                )
                clean_amp = clean["rcs_amp_real"] + 1j * clean["rcs_amp_imag"]
                total_amp = total["rcs_amp_real"] + 1j * total["rcs_amp_imag"]
                delta_amp = delta["rcs_amp_real"] + 1j * delta["rcs_amp_imag"]
                np.testing.assert_allclose(total_amp - clean_amp, delta_amp)
            self.assertEqual(list(root.glob(".*.backup")), [])
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
            self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_unreviewed_feature_only_sibling_blocks_both_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            _write_empty_base(base)
            intruder = b"unreviewed feature-only output"
            features.write_bytes(intruder)

            with self.assertRaisesRegex(RuntimeError, "feature-only output was created"):
                feature_sum.add_features_to_monostatic_grim(
                    str(base),
                    str(output),
                    radar_grid=GRID,
                    declared_coherent_base=True,
                    expect_output_absent=True,
                    expect_features_only_output_absent=True,
                )

            self.assertFalse(output.exists())
            self.assertEqual(features.read_bytes(), intruder)

    def test_feature_bearing_total_cannot_be_used_as_another_build_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.grim"
            first = root / "first.grim"
            second = root / "second.grim"
            _write_empty_base(clean)
            feature_sum.add_features_to_monostatic_grim(
                str(clean),
                str(first),
                radar_grid=GRID,
                declared_coherent_base=True,
            )

            with self.assertRaisesRegex(
                ValueError, "feature-bearing base"
            ):
                feature_sum.add_features_to_monostatic_grim(
                    str(first),
                    str(second),
                    radar_grid=GRID,
                    declared_coherent_base=True,
                )

            self.assertFalse(second.exists())
            self.assertFalse(
                Path(feature_sum.feature_only_output_path(str(second))).exists()
            )

    def test_pair_publication_failure_rolls_back_both_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            _write_empty_base(base)
            old_total = b"previous total"
            old_features = b"previous feature delta"
            output.write_bytes(old_total)
            features.write_bytes(old_features)
            real_link = feature_sum.os.link

            def fail_total_publication(source, destination):
                if (
                    Path(destination) == output
                    and ".tmp." in Path(source).name
                    and Path(source).suffix == ".grim"
                ):
                    raise OSError("simulated total publication failure")
                return real_link(source, destination)

            with mock.patch.object(
                feature_sum.os, "link", side_effect=fail_total_publication
            ):
                with self.assertRaisesRegex(
                    OSError, "total publication failure"
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            self.assertEqual(output.read_bytes(), old_total)
            self.assertEqual(features.read_bytes(), old_features)
            self.assertEqual(list(root.glob(".*.backup")), [])
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
            self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_failed_replace_does_not_delete_external_file_at_absent_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            _write_empty_base(base)
            intruder = b"external writer won the race"
            real_link = feature_sum.os.link

            def race_before_total_link(source, destination):
                if Path(destination) == output:
                    output.write_bytes(intruder)
                return real_link(source, destination)

            with mock.patch.object(
                feature_sum.os, "link", side_effect=race_before_total_link
            ):
                with self.assertRaises(OSError):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            self.assertEqual(output.read_bytes(), intruder)
            self.assertFalse(features.exists())
            self.assertEqual(list(root.glob(".*.backup")), [])

    def test_external_replacement_between_link_and_verification_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            intruder_source = root / "independent-feature-output"
            _write_empty_base(base)
            old_total = b"reviewed previous total"
            old_features = b"reviewed previous feature delta"
            intruder = b"independent writer replaced the new link"
            output.write_bytes(old_total)
            features.write_bytes(old_features)
            intruder_source.write_bytes(intruder)
            real_link = feature_sum.os.link

            def replace_immediately_after_feature_link(source, destination):
                result = real_link(source, destination)
                if (
                    Path(destination) == features
                    and ".features." in Path(source).name
                ):
                    os.replace(intruder_source, features)
                return result

            with mock.patch.object(
                feature_sum.os,
                "link",
                side_effect=replace_immediately_after_feature_link,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "without overwriting an independent file"
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            self.assertEqual(features.read_bytes(), intruder)
            self.assertEqual(output.read_bytes(), old_total)
            backups = list(root.glob(".*.backup"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), old_features)
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
            self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_external_replacement_before_pair_commit_is_detected_and_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            intruder_source = root / "independent-feature-output"
            _write_empty_base(base)
            old_total = b"reviewed previous total"
            old_features = b"reviewed previous feature delta"
            intruder = b"independent writer won before paired commit"
            output.write_bytes(old_total)
            features.write_bytes(old_features)
            intruder_source.write_bytes(intruder)
            real_link = feature_sum.os.link

            def replace_feature_while_total_is_published(source, destination):
                result = real_link(source, destination)
                if (
                    Path(destination) == output
                    and ".tmp." in Path(source).name
                ):
                    os.replace(intruder_source, features)
                return result

            with mock.patch.object(
                feature_sum.os,
                "link",
                side_effect=replace_feature_while_total_is_published,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "without overwriting an independent file"
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            self.assertEqual(features.read_bytes(), intruder)
            self.assertEqual(output.read_bytes(), old_total)
            backups = list(root.glob(".*.backup"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), old_features)
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
            self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_in_place_mutation_before_pair_commit_restores_reviewed_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            _write_empty_base(base)
            old_total = b"reviewed previous total"
            old_features = b"reviewed previous feature delta"
            output.write_bytes(old_total)
            features.write_bytes(old_features)
            real_link = feature_sum.os.link

            def mutate_feature_while_total_is_published(source, destination):
                result = real_link(source, destination)
                if (
                    Path(destination) == output
                    and ".tmp." in Path(source).name
                ):
                    features.write_bytes(b"mutated through the public hard link")
                return result

            with mock.patch.object(
                feature_sum.os,
                "link",
                side_effect=mutate_feature_while_total_is_published,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "staged Assembly output changed"
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            self.assertEqual(features.read_bytes(), old_features)
            self.assertEqual(output.read_bytes(), old_total)
            self.assertEqual(list(root.glob(".*.backup")), [])
            self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
            self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_existing_target_recreation_race_keeps_intruder_for_both_outputs(self):
        for raced_name in ("assembled_features_only.grim", "assembled.grim"):
            with self.subTest(raced_name=raced_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = root / "clean.grim"
                output = root / "assembled.grim"
                features = root / "assembled_features_only.grim"
                raced = root / raced_name
                _write_empty_base(base)
                old_total = b"reviewed previous total"
                old_features = b"reviewed previous delta"
                intruder = b"independent writer recreated this target"
                output.write_bytes(old_total)
                features.write_bytes(old_features)
                real_link = feature_sum.os.link

                def recreate_before_exclusive_publish(source, destination):
                    if Path(destination) == raced and (
                        ".features." in Path(source).name
                        or ".tmp." in Path(source).name
                    ):
                        raced.write_bytes(intruder)
                    return real_link(source, destination)

                with mock.patch.object(
                    feature_sum.os,
                    "link",
                    side_effect=recreate_before_exclusive_publish,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "without overwriting an independent file"
                    ):
                        feature_sum.add_features_to_monostatic_grim(
                            str(base),
                            str(output),
                            radar_grid=GRID,
                            declared_coherent_base=True,
                        )

                self.assertEqual(raced.read_bytes(), intruder)
                other = output if raced == features else features
                expected_other = old_total if other == output else old_features
                self.assertEqual(other.read_bytes(), expected_other)
                backups = list(root.glob(".*.backup"))
                self.assertEqual(len(backups), 1)
                expected_backup = old_features if raced == features else old_total
                self.assertEqual(backups[0].read_bytes(), expected_backup)
                self.assertEqual(list(root.glob(".*.tmp.*.grim")), [])
                self.assertEqual(list(root.glob(".*.features.*.grim")), [])

    def test_backup_digest_recheck_detects_existing_target_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            features = root / "assembled_features_only.grim"
            _write_empty_base(base)
            old_total = b"reviewed total"
            old_features = b"reviewed delta"
            intruder = b"changed after final digest check"
            output.write_bytes(old_total)
            features.write_bytes(old_features)
            real_replace = feature_sum.os.replace
            swapped = False

            def swap_before_backup(source, destination):
                nonlocal swapped
                if (
                    not swapped
                    and Path(source) == output
                    and str(destination).endswith(".backup")
                ):
                    output.write_bytes(intruder)
                    swapped = True
                return real_replace(source, destination)

            with mock.patch.object(
                feature_sum.os, "replace", side_effect=swap_before_backup
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "changed after its final reviewed digest check"
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            self.assertTrue(swapped)
            self.assertEqual(output.read_bytes(), intruder)
            self.assertEqual(features.read_bytes(), old_features)
            self.assertEqual(list(root.glob(".*.backup")), [])

    def test_unsupported_hard_links_fail_before_feature_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "clean.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)

            with (
                mock.patch.object(
                    feature_sum.os,
                    "link",
                    side_effect=OSError("hard links unsupported"),
                ),
                mock.patch.object(feature_sum, "export_radar_grim") as export,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "filesystem does not support atomic, no-overwrite hard links",
                ):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base),
                        str(output),
                        radar_grid=GRID,
                        declared_coherent_base=True,
                    )

            export.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(
                Path(feature_sum.feature_only_output_path(str(output))).exists()
            )
            self.assertEqual(
                list(root.glob(".ghost-assembly-link-probe.*")), []
            )

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
