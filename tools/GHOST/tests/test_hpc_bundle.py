#!/usr/bin/env python3
"""Contract tests for portable Windows-to-Linux HPC request bundles."""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


BACKEND = Path(__file__).resolve().parent.parent / "Backend"
sys.path.insert(0, str(BACKEND))

import hpc_bundle  # noqa: E402


def _write_geometry(root: Path, name: str = "body.geo", *, sidecar: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    material_rows = "IBCS_Resistances:\n7 coating.csv\n" if sidecar else "IBCS_Resistances:\n"
    path = root / name
    path.write_text(
        "Title: portable bundle test\n"
        "Segment: body 2\n"
        "properties: 2 7 0 0 0\n"
        "-0.02 -0.02 -0.02 0.02\n"
        "-0.02 0.02 0.02 0.02\n"
        "0.02 0.02 0.02 -0.02\n"
        "0.02 -0.02 -0.02 -0.02\n"
        + material_rows
        + "Dielectrics:\n",
        encoding="utf-8",
    )
    if sidecar:
        (root / "coating.csv").write_text(
            "frequency_GHz,resistance_ohm,reactance_ohm\n1,50,2\n",
            encoding="utf-8",
        )
    return path


class PortableBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="ghost-bundle-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary, ignore_errors=True)

    def _create_2d(self, *, settings=None) -> tuple[Path, dict]:
        source = _write_geometry(self.temporary / "source")
        target = self.temporary / "portable_request"
        request = hpc_bundle.create_portable_bundle(
            target,
            solver="2d",
            geometries=[{"role": "FRD", "path": str(source)}],
            settings=settings
            or {
                "FREQUENCIES_GHZ": [2.0, 4.0],
                "AZIMUTHS_DEG": [0.0, 90.0],
                "GEOMETRY_UNITS": "meters",
                "N_NODES": 2,
                "SLURM_PARTITION": "compute",
            },
        )
        return target, request

    def test_create_is_relative_complete_and_self_documenting(self) -> None:
        target, request = self._create_2d()
        raw = (target / hpc_bundle.REQUEST_NAME).read_text(encoding="utf-8")
        self.assertEqual(request["schema"], hpc_bundle.REQUEST_SCHEMA)
        self.assertNotIn(str(self.temporary), raw)
        self.assertNotIn("C:\\", raw)
        self.assertEqual(request["geometries"][0]["role"], "FRD")
        self.assertEqual(len(request["geometries"][0]["sidecars"]), 1)
        readme = (target / hpc_bundle.README_NAME).read_text(encoding="utf-8")
        self.assertIn("hpc_bundle.py stage", readme)
        self.assertIn("--run-driver --submit", readme)
        self.assertIn("matching GHOST checkout", readme)
        self.assertEqual(hpc_bundle.verify_portable_bundle(target), request)

    def test_each_geometry_keeps_its_own_same_named_material(self) -> None:
        one = _write_geometry(self.temporary / "one", "first.geo")
        two = _write_geometry(self.temporary / "two", "second.geo")
        (two.parent / "coating.csv").write_text(
            "frequency_GHz,resistance_ohm,reactance_ohm\n1,99,4\n",
            encoding="utf-8",
        )
        target = self.temporary / "two_geometries"
        request = hpc_bundle.create_portable_bundle(
            target,
            solver="2d",
            geometries=[
                {"role": "FRD", "path": str(one)},
                {"role": "OPN", "path": str(two)},
            ],
            settings={},
        )
        sidecars = [record["sidecars"][0] for record in request["geometries"]]
        self.assertEqual(len(set(sidecars)), 2)
        hashes = {
            record["sha256"]
            for record in request["files"]
            if record["kind"] == "material"
        }
        self.assertEqual(len(hashes), 2)

    def test_tampering_and_unexpected_files_fail_closed(self) -> None:
        target, request = self._create_2d()
        geometry = target.joinpath(*request["geometries"][0]["path"].split("/"))
        geometry.write_text(geometry.read_text(encoding="utf-8") + "# changed\n")
        with self.assertRaisesRegex(hpc_bundle.BundleError, "differs from its inventory"):
            hpc_bundle.verify_portable_bundle(target)

        target, _request = self._create_2d_at(self.temporary / "second_bundle")
        (target / "surprise.sh").write_text("echo no\n", encoding="utf-8")
        with self.assertRaisesRegex(hpc_bundle.BundleError, "exact inventory"):
            hpc_bundle.verify_portable_bundle(target)

    def _create_2d_at(self, target: Path) -> tuple[Path, dict]:
        source_root = self.temporary / f"source_{target.name}"
        source = _write_geometry(source_root)
        request = hpc_bundle.create_portable_bundle(
            target,
            solver="2d",
            geometries=[{"role": "FRD", "path": str(source)}],
            settings={},
        )
        return target, request

    def test_roles_stems_and_settings_are_strict(self) -> None:
        source = _write_geometry(self.temporary / "strict")
        with self.assertRaisesRegex(hpc_bundle.BundleError, "role must be"):
            hpc_bundle.create_portable_bundle(
                self.temporary / "bad_role",
                solver="bor",
                geometries=[{"role": "FRD", "path": str(source)}],
                settings={},
            )
        with self.assertRaisesRegex(hpc_bundle.BundleError, "execution-owned"):
            hpc_bundle.create_portable_bundle(
                self.temporary / "shell_injection",
                solver="2d",
                geometries=[{"role": "FRD", "path": str(source)}],
                settings={"JOB_PROLOGUE": ["curl bad | sh"]},
            )
        with self.assertRaisesRegex(hpc_bundle.BundleError, "conservative SLURM token"):
            hpc_bundle.create_portable_bundle(
                self.temporary / "newline_injection",
                solver="2d",
                geometries=[{"role": "FRD", "path": str(source)}],
                settings={"SLURM_PARTITION": "compute\n/bin/false"},
            )
        with self.assertRaisesRegex(hpc_bundle.BundleError, "duplicate stem"):
            hpc_bundle.create_portable_bundle(
                self.temporary / "duplicate_stem",
                solver="2d",
                geometries=[
                    {"role": "FRD", "path": str(source)},
                    {
                        "role": "OPN",
                        "path": str(_write_geometry(self.temporary / "other")),
                    },
                ],
                settings={},
            )

    def test_every_allowlisted_setting_exists_in_its_canonical_driver(self) -> None:
        for solver, driver in (
            ("2d", hpc_bundle.TWOD_DRIVER),
            ("bor", hpc_bundle.BOR_DRIVER),
        ):
            assigned = set(
                re.findall(
                    r"^([A-Z_][A-Z0-9_]*)\s*=",
                    Path(driver).read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                )
            )
            self.assertEqual(
                hpc_bundle._SETTINGS_BY_SOLVER[solver] - assigned,
                set(),
                f"{solver} allowlist drifted from {Path(driver).name}",
            )

    def test_request_path_traversal_is_rejected_before_read(self) -> None:
        target, request = self._create_2d()
        request["files"][0]["path"] = "../outside.geo"
        (target / hpc_bundle.REQUEST_NAME).write_text(
            json.dumps(request), encoding="utf-8"
        )
        with self.assertRaisesRegex(hpc_bundle.BundleError, "dot path component"):
            hpc_bundle.verify_portable_bundle(target)

    def test_stage_derives_linux_owned_execution_settings(self) -> None:
        target, request = self._create_2d()
        workspace = self.temporary / "workspace"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            result = hpc_bundle.stage_portable_bundle(target, workspace)
        self.assertTrue(result["ok"])
        self.assertFalse(result["driver_ran"])
        self.assertIsNone(result["run_dir"])
        driver = Path(result["driver_path"]).read_text(encoding="utf-8")
        self.assertIn(f"OUTPUT_DIR = {str(Path(result['output_root']))!r}", driver)
        self.assertIn(f"PYTHON_EXE = {sys.executable!r}", driver)
        self.assertIn("SUBMIT = False", driver)
        self.assertNotIn(str(self.temporary / "source"), driver)
        stage_dir = Path(result["stage_dir"])
        staged_geometry = stage_dir.joinpath(
            *request["geometries"][0]["path"].split("/")
        )
        self.assertTrue(staged_geometry.is_file())

    def test_bor_stage_derives_only_bor_geometry_root(self) -> None:
        source = _write_geometry(self.temporary / "bor_source", "axisymmetric.geo")
        target = self.temporary / "bor_request"
        request = hpc_bundle.create_portable_bundle(
            target,
            solver="bor",
            geometries=[{"role": "BOR", "path": str(source)}],
            settings={
                "FREQUENCIES_GHZ": [1.0, 2.0],
                "AZIMUTHS_DEG": [0.0, 180.0],
                "ELEVATIONS_DEG": [-30.0, 30.0],
                "BODY_AXIS_AZ_DEG": 0.0,
                "BODY_AXIS_EL_DEG": 0.0,
                "BODY_ROLL_DEG": 0.0,
            },
        )
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            result = hpc_bundle.stage_portable_bundle(
                target, self.temporary / "bor_workspace"
            )
        driver = Path(result["driver_path"]).read_text(encoding="utf-8")
        bor_root = Path(result["stage_dir"]) / "payload" / "BOR"
        self.assertIn(f"GEOMETRY_DIRS = {[str(bor_root)]!r}", driver)
        self.assertNotIn("FRD_DIR =", driver)
        self.assertEqual(request["geometries"][0]["role"], "BOR")

    def test_driver_result_is_machine_readable_and_idempotent(self) -> None:
        target, request = self._create_2d()
        workspace = self.temporary / "workspace"
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            run_root = workspace / f"grim_{request['bundle_id']}" / "runs" / "run_123"
            run_root.mkdir(parents=True)
            return SimpleNamespace(
                returncode=0,
                stdout="planning complete\nSubmitted batch job 81234\n",
            )

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle.subprocess, "run", side_effect=fake_run),
        ):
            first = hpc_bundle.stage_portable_bundle(
                target, workspace, run_driver=True, submit=True
            )
            second = hpc_bundle.stage_portable_bundle(
                target, workspace, run_driver=True, submit=True
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(second, first)
        self.assertTrue(first["submitted"])
        self.assertEqual(first["job_ids"], ["81234"])
        self.assertTrue(first["run_dir"].endswith("run_123"))
        self.assertEqual(first["run_id"], "run_123")
        self.assertEqual(first["schema"], hpc_bundle.STAGE_RESULT_SCHEMA)
        self.assertEqual(
            json.loads(Path(first["stage_dir"], "stage_result.json").read_text()),
            first,
        )

    def test_staged_payload_rejects_extra_files_before_driver_execution(self) -> None:
        target, _request = self._create_2d()
        workspace = self.temporary / "workspace_exact"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        extra = Path(staged["stage_dir"]) / "payload" / "FRD" / "surprise.geo"
        extra.write_text("# not declared\n", encoding="utf-8")

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle.subprocess, "run") as runner,
        ):
            with self.assertRaisesRegex(hpc_bundle.BundleError, "exact request inventory"):
                hpc_bundle.stage_portable_bundle(
                    target, workspace, run_driver=True, submit=True
                )
        runner.assert_not_called()

    def test_driver_result_recovers_each_persisted_partial_job_id(self) -> None:
        target, request = self._create_2d()
        workspace = self.temporary / "workspace_partial"

        def fake_run(command, **kwargs):
            run_root = workspace / f"grim_{request['bundle_id']}" / "runs" / "run_partial"
            run_root.mkdir(parents=True)
            (run_root / "submitted_jobs.json").write_text(
                json.dumps(
                    {
                        "schema": "ghost.hpc.submitted-jobs.v1",
                        "job_ids": ["81234", "81235"],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(
                returncode=2,
                stdout="Submitted batch job 81234\nsecond sbatch failed\n",
            )

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle.subprocess, "run", side_effect=fake_run),
        ):
            result = hpc_bundle.stage_portable_bundle(
                target, workspace, run_driver=True, submit=True
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["job_ids"], ["81234", "81235"])
        self.assertTrue(result["submitted"])

    def test_recover_reconstructs_interrupted_stage_without_resubmitting(self) -> None:
        target, request = self._create_2d()
        workspace = self.temporary / "workspace_recover"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        stage_dir = Path(staged["stage_dir"])
        (stage_dir / "stage_result.json").unlink()
        (stage_dir / "stage_state.json").write_text(
            json.dumps(
                {
                    "schema": "ghost.hpc.stage-state.v1",
                    "status": "running",
                    "submission_requested": True,
                }
            ),
            encoding="utf-8",
        )
        run_dir = stage_dir / "runs" / "run_interrupted"
        run_dir.mkdir()
        (run_dir / "submitted_jobs.json").write_text(
            json.dumps(
                {
                    "schema": "ghost.hpc.submitted-jobs.v1",
                    "job_ids": ["81234"],
                }
            ),
            encoding="utf-8",
        )

        recovered = hpc_bundle.recover_staged_bundle(stage_dir)

        self.assertFalse(recovered["ok"])
        self.assertTrue(recovered["recovered"])
        self.assertEqual(recovered["bundle_id"], request["bundle_id"])
        self.assertEqual(recovered["run_id"], "run_interrupted")
        self.assertEqual(recovered["job_ids"], ["81234"])
        self.assertEqual(recovered["stage_state"], "running")

    def test_cli_emits_one_json_object(self) -> None:
        target, request = self._create_2d()
        output = io.StringIO()
        with redirect_stdout(output):
            returncode = hpc_bundle.main(["verify", str(target)])
        self.assertEqual(returncode, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["bundle_id"], request["bundle_id"])
        self.assertEqual(output.getvalue().count("\n"), 1)

    def test_windows_cannot_create_final_stage_provenance(self) -> None:
        target, _request = self._create_2d()
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=False):
            with self.assertRaisesRegex(hpc_bundle.BundleError, "Linux login node"):
                hpc_bundle.stage_portable_bundle(target, self.temporary / "workspace")


if __name__ == "__main__":
    unittest.main(verbosity=2)
