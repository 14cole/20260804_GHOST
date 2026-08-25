from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ibc.compute import MaterialTable
from ibc.io import (
    load_project_file,
    save_project_file,
    write_impedance_bundle,
    write_material_table,
    write_output,
)


def _project_state(root: Path) -> dict[str, object]:
    materials = root / "materials"
    outputs = root / "outputs"
    return {
        "layers": [
            {
                "thickness_in": 0.125,
                "anisotropic": True,
                "file_0deg": str(materials / "x.csv"),
                "file_90deg": str(materials / "y.csv"),
                "polarization_deg": 0.0,
            }
        ],
        "controls": {
            "output": str(outputs / "nominal.csv"),
            "ibc_batch_output_dir": str(outputs / "ibc_batch"),
            "angle_output": str(outputs / "angles.csv"),
            "thk_output": str(outputs / "thickness.csv"),
            "mix_prop_file": str(materials / "target.csv"),
        },
        "mixes": {
            "components": [
                {"file": str(materials / "host.csv"), "parts": 1.0}
            ]
        },
    }


class ProjectPathPortabilityTests(unittest.TestCase):
    def test_legacy_schema_one_relative_paths_keep_working_directory_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "legacy.json"
            payload = {
                "schema_version": 1,
                "state": {
                    "layers": [
                        {
                            "file_0deg": "materials/legacy.csv",
                            "file_90deg": "",
                        }
                    ],
                    "controls": {"output": "outputs/legacy.csv"},
                },
            }
            project.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertWarnsRegex(
                RuntimeWarning, "historical working-directory meaning"
            ):
                loaded = load_project_file(project)

            self.assertEqual(
                loaded["layers"][0]["file_0deg"],
                "materials/legacy.csv",
            )
            self.assertEqual(
                loaded["controls"]["output"],
                "outputs/legacy.csv",
            )

    def test_relative_live_path_is_resolved_from_runtime_not_project_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            project_directory = root / "project"
            runtime.mkdir()
            project_directory.mkdir()
            (runtime / "material.csv").write_text("runtime material", encoding="utf-8")
            (project_directory / "material.csv").write_text(
                "different project material", encoding="utf-8"
            )
            state = {
                "layers": [{"file_0deg": "material.csv", "file_90deg": ""}],
                "controls": {},
            }
            project = project_directory / "design.json"

            with mock.patch("ibc.io.Path.cwd", return_value=runtime):
                save_project_file(project, state)

            payload = json.loads(project.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["state"]["layers"][0]["file_0deg"],
                "../runtime/material.csv",
            )
            loaded = load_project_file(project)
            selected = Path(loaded["layers"][0]["file_0deg"])
            self.assertEqual(selected, (runtime / "material.csv").resolve())
            self.assertEqual(selected.read_text(encoding="utf-8"), "runtime material")

    def test_relative_paths_follow_a_moved_project_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temporary_root = Path(tmp)
            original = temporary_root / "original"
            original.mkdir()
            state = _project_state(original)
            untouched = copy.deepcopy(state)
            project = original / "design.json"

            save_project_file(project, state)

            self.assertEqual(state, untouched, "saving must not mutate live UI state")
            payload = json.loads(project.read_text(encoding="utf-8"))
            stored = payload["state"]
            self.assertEqual(stored["layers"][0]["file_0deg"], "materials/x.csv")
            self.assertEqual(stored["controls"]["output"], "outputs/nominal.csv")
            self.assertEqual(
                stored["controls"]["ibc_batch_output_dir"],
                "outputs/ibc_batch",
            )
            self.assertEqual(
                stored["mixes"]["components"][0]["file"],
                "materials/host.csv",
            )
            self.assertEqual(
                payload["path_portability"]["external_absolute_paths"], []
            )

            moved = temporary_root / "moved"
            shutil.copytree(original, moved)
            loaded = load_project_file(moved / "design.json")
            self.assertEqual(
                Path(loaded["layers"][0]["file_90deg"]),
                (moved / "materials" / "y.csv").resolve(),
            )
            self.assertEqual(
                Path(loaded["controls"]["thk_output"]),
                (moved / "outputs" / "thickness.csv").resolve(),
            )
            self.assertEqual(
                Path(loaded["controls"]["ibc_batch_output_dir"]),
                (moved / "outputs" / "ibc_batch").resolve(),
            )
            self.assertEqual(
                Path(loaded["mixes"]["components"][0]["file"]),
                (moved / "materials" / "host.csv").resolve(),
            )

    def test_neighboring_material_directory_uses_one_parent_hop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_directory = root / "projects"
            project_directory.mkdir()
            state = _project_state(root)
            project = project_directory / "design.json"

            save_project_file(project, state)

            payload = json.loads(project.read_text(encoding="utf-8"))
            stored_path = payload["state"]["layers"][0]["file_0deg"]
            self.assertEqual(stored_path, "../materials/x.csv")
            loaded = load_project_file(project)
            self.assertEqual(
                Path(loaded["layers"][0]["file_0deg"]),
                (root / "materials" / "x.csv").resolve(),
            )

    def test_distant_absolute_path_is_preserved_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_directory = root / "bundle" / "projects"
            project_directory.mkdir(parents=True)
            external = (root / "external" / "measured.csv").resolve()
            state = _project_state(project_directory)
            state["layers"][0]["file_0deg"] = str(external)
            project = project_directory / "design.json"

            save_project_file(project, state)

            payload = json.loads(project.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["state"]["layers"][0]["file_0deg"], str(external)
            )
            metadata = payload["path_portability"]["external_absolute_paths"]
            self.assertIn(
                {"field": "layers[0].file_0deg", "path": str(external)},
                metadata,
            )
            with self.assertWarnsRegex(RuntimeWarning, "external absolute path"):
                loaded = load_project_file(project)
            self.assertEqual(loaded["layers"][0]["file_0deg"], str(external))


class AtomicFreddyWriterTests(unittest.TestCase):
    def _assert_no_temporary_files(self, path: Path) -> None:
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_failed_impedance_publish_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nominal.csv"
            path.write_text("previous\n", encoding="utf-8")
            with mock.patch("ibc.io.os.replace", side_effect=OSError("blocked")):
                with self.assertRaisesRegex(OSError, "blocked"):
                    write_output(path, [(1.0, 120.0, 15.0)], True)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous\n")
            self._assert_no_temporary_files(path)

    def test_failed_material_publish_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "material.csv"
            path.write_text("previous\n", encoding="utf-8")
            table = MaterialTable([1.0], [2.5 - 0.1j], [1.0 + 0.0j])
            with mock.patch("ibc.io.os.replace", side_effect=OSError("blocked")):
                with self.assertRaisesRegex(OSError, "blocked"):
                    write_material_table(path, table)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous\n")
            self._assert_no_temporary_files(path)

    def test_failed_project_serialization_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "design.json"
            path.write_text("previous\n", encoding="utf-8")
            with mock.patch(
                "ibc.io.json.dump", side_effect=RuntimeError("serialization failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                    save_project_file(path, {"layers": [], "controls": {}})
            self.assertEqual(path.read_text(encoding="utf-8"), "previous\n")
            self._assert_no_temporary_files(path)

    def test_failed_second_impedance_stage_preserves_nominal_and_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nominal = Path(tmp) / "nominal.csv"
            uncertainty = Path(tmp) / "nominal_uncertainty.csv"
            nominal.write_text("old nominal\n", encoding="utf-8")
            uncertainty.write_text("old uncertainty\n", encoding="utf-8")
            nominal_rows = [(1.0, 120.0, 15.0)]
            uncertainty_rows = [
                (1.0, 120.0, 15.0, 110.0, 130.0, 10.0, 20.0)
            ]
            from ibc import io as freddy_io

            original_stage = freddy_io._stage_text_file
            calls = 0

            def fail_second(path, writer):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("second stage failed")
                return original_stage(path, writer)

            with mock.patch.object(
                freddy_io, "_stage_text_file", side_effect=fail_second
            ):
                with self.assertRaisesRegex(OSError, "second stage failed"):
                    write_impedance_bundle(
                        nominal,
                        nominal_rows,
                        True,
                        uncertainty,
                        uncertainty_rows,
                    )

            self.assertEqual(nominal.read_text(encoding="utf-8"), "old nominal\n")
            self.assertEqual(
                uncertainty.read_text(encoding="utf-8"), "old uncertainty\n"
            )
            self.assertEqual(set(Path(tmp).iterdir()), {nominal, uncertainty})


if __name__ == "__main__":
    unittest.main()
