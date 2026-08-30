#!/usr/bin/env python3
"""Contract tests for portable Windows-to-Linux HPC request bundles."""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


BACKEND = Path(__file__).resolve().parent.parent / "Backend"
sys.path.insert(0, str(BACKEND))

import hpc_bundle  # noqa: E402
import hpc_common  # noqa: E402
import feature_sum  # noqa: E402
import workflow_provenance  # noqa: E402


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

    def _minimal_run(self, *, solver: str = "2d") -> tuple[Path, dict]:
        run_dir = self.temporary / f"run_status_{solver}"
        run_dir.mkdir()
        common = {
            "run_id": run_dir.name,
            "solver_source_sha256": "a" * 64,
            "runtime_environment_sha256": "b" * 64,
            "solver_config": {"geometry_units": "meters"},
        }
        if solver == "2d":
            units = [{
                "geometry": "/tmp/body.geo",
                "geometry_stem": "body",
                "geometry_input_sha256": "c" * 64,
                "frequency_ghz": 1.0,
                "polarizations": ["VV", "HH"],
                "role": "FRD",
            }]
            manifest = {
                **common,
                "schema": "ghost.hpc.2d-run.v2",
                "unit_output_dir": "results",
                "azimuths_deg": [0.0],
                "units": units,
                "n_units": len(units),
            }
        else:
            units = [
                {
                    "geometry": "/tmp/body.geo",
                    "geometry_stem": "body",
                    "geometry_input_sha256": "c" * 64,
                    "frequency_ghz": 1.0,
                    "polarization": polarization,
                }
                for polarization in ("VV", "HH")
            ]
            manifest = {
                **common,
                "schema": "ghost.hpc.bor-run.v1",
                "unit_output_dir": "results/by_frequency",
                "aspects_deg": [0.0],
                "units": units,
                "n_units": len(units),
            }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return run_dir, manifest

    def test_run_status_requires_exact_attested_unit_inventory(self) -> None:
        run_dir, _manifest = self._minimal_run(solver="2d")
        expected = run_dir / "results" / "FRD" / "1.000GHz_body.grim"
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"unit")

        with mock.patch.object(
            hpc_common, "require_hpc_output_attestations", return_value=None
        ):
            complete = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["attestation_verified"])
        self.assertEqual(complete["n_done"], 1)
        self.assertEqual(complete["missing"], [])

        unexpected = run_dir / "results" / "unexpected.grim"
        unexpected.write_bytes(b"extra")
        with mock.patch.object(
            hpc_common, "require_hpc_output_attestations", return_value=None
        ):
            extra = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertFalse(extra["complete"])
        self.assertEqual(extra["unexpected"], ["results/unexpected.grim"])
        unexpected.unlink()

        with mock.patch.object(
            hpc_common,
            "require_hpc_output_attestations",
            side_effect=ValueError("wrong embedded run attestation"),
        ):
            untrusted = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertFalse(untrusted["complete"])
        self.assertFalse(untrusted["attestation_verified"])
        self.assertIn("wrong embedded run attestation", untrusted["attestation_error"])

        with mock.patch.object(
            hpc_common, "require_hpc_output_attestations", return_value=None
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                returncode = hpc_bundle.main(["run-status", str(run_dir)])
        self.assertEqual(returncode, 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["schema"],
            "ghost.hpc.run-status.v1",
        )

    def test_bor_run_status_requires_verified_publication(self) -> None:
        run_dir, manifest = self._minimal_run(solver="bor")
        unit_root = run_dir / "results" / "by_frequency"
        unit_root.mkdir(parents=True)
        for polarization in ("VV", "HH"):
            (unit_root / f"{polarization}_1.000GHz_body.grim").write_bytes(b"unit")

        with mock.patch.object(
            hpc_common, "require_hpc_output_attestations", return_value=None
        ):
            missing = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertTrue(missing["unit_complete"])
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["derived_missing"], ["results/body.grim"])

        publication = run_dir / "results" / "body.grim"
        publication.write_bytes(b"published")
        run_spec = "d" * 64
        payload = {
            "geometry_input_sha256": manifest["units"][0][
                "geometry_input_sha256"
            ],
            "solver_source_sha256": manifest["solver_source_sha256"],
            "runtime_environment_sha256": manifest[
                "runtime_environment_sha256"
            ],
            "run_solve_spec_sha256": run_spec,
        }
        with (
            mock.patch.object(
                hpc_common, "require_hpc_output_attestations", return_value=None
            ),
            mock.patch.object(feature_sum, "_load_grim", return_value=payload),
            mock.patch.object(feature_sum, "load_body_grim", return_value={}),
            mock.patch.object(
                workflow_provenance,
                "manifest_solve_spec_fingerprint",
                return_value=run_spec,
            ),
        ):
            complete = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["publication_verified"])
        self.assertEqual(complete["n_derived_done"], 1)

        wrong_payload = dict(payload, run_solve_spec_sha256="wrong-run")
        with (
            mock.patch.object(
                hpc_common, "require_hpc_output_attestations", return_value=None
            ),
            mock.patch.object(
                feature_sum, "_load_grim", return_value=wrong_payload
            ),
            mock.patch.object(feature_sum, "load_body_grim", return_value={}),
            mock.patch.object(
                workflow_provenance,
                "manifest_solve_spec_fingerprint",
                return_value=run_spec,
            ),
        ):
            wrong_run = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertFalse(wrong_run["complete"])
        self.assertIn("does not match this run", wrong_run["publication_error"])

    def test_bor_run_status_reads_real_run_bound_body_grim(self) -> None:
        run_dir, manifest = self._minimal_run(solver="bor")
        unit_root = run_dir / "results" / "by_frequency"
        unit_root.mkdir(parents=True)
        for polarization in ("VV", "HH"):
            (unit_root / f"{polarization}_1.000GHz_body.grim").write_bytes(b"unit")

        amplitude = 1.0 / math.sqrt(4.0 * math.pi)
        bodies = {1.0: {
            "theta_deg": np.asarray([0.0, 90.0, 180.0]),
            "amp_vv": np.asarray([amplitude] * 3, dtype=complex),
            "amp_hh": np.asarray([amplitude] * 3, dtype=complex),
        }}
        profile = np.asarray([[0.0, 1.0], [0.2, 0.0], [0.0, -1.0]])
        run_spec = workflow_provenance.manifest_solve_spec_fingerprint(manifest)
        metadata = {
            "geometry_input_sha256": manifest["units"][0][
                "geometry_input_sha256"
            ],
            "solver_source_sha256": manifest["solver_source_sha256"],
            "runtime_environment_sha256": manifest[
                "runtime_environment_sha256"
            ],
            "run_solve_spec_sha256": run_spec,
        }
        publication = run_dir / "results" / "body.grim"
        feature_sum.save_monostatic_grim(
            bodies,
            profile,
            str(publication),
            azimuths_deg=[0.0, 90.0],
            elevations_deg=[0.0],
            artifact_metadata=metadata,
        )
        with mock.patch.object(
            hpc_common, "require_hpc_output_attestations", return_value=None
        ):
            complete = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["publication_verified"])

        feature_sum.save_monostatic_grim(
            bodies,
            profile,
            str(publication),
            azimuths_deg=[0.0, 90.0],
            elevations_deg=[0.0],
            artifact_metadata={**metadata, "run_solve_spec_sha256": "wrong-run"},
        )
        with mock.patch.object(
            hpc_common, "require_hpc_output_attestations", return_value=None
        ):
            rejected = hpc_bundle.inspect_hpc_run(run_dir)
        self.assertFalse(rejected["complete"])
        self.assertIn("does not match this run", rejected["publication_error"])

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

    def test_bor_bundle_rejects_pure_efie_cfie_endpoint(self) -> None:
        source = _write_geometry(self.temporary / "cfie_endpoint")
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    hpc_bundle.BundleError, "strictly between 0 and 1"
                ):
                    hpc_bundle.create_portable_bundle(
                        self.temporary / f"bad_cfie_{alpha:g}",
                        solver="bor",
                        geometries=[{"role": "BODY", "path": str(source)}],
                        settings={"CFIE_ALPHA": alpha},
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

    def test_interrupted_initial_copy_never_publishes_partial_stage(self) -> None:
        target, request = self._create_2d()
        workspace = self.temporary / "workspace_atomic_stage"

        def interrupted_copy(_bundle_root, private_stage, _bundle_request):
            (private_stage / "partial-copy-marker").write_text(
                "interrupted", encoding="utf-8"
            )
            raise OSError("simulated copy interruption")

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle, "_copy_verified_files", side_effect=interrupted_copy),
        ):
            with self.assertRaisesRegex(OSError, "simulated copy interruption"):
                hpc_bundle.stage_portable_bundle(target, workspace)

        final_stage = workspace / f"grim_{request['bundle_id']}"
        self.assertFalse(final_stage.exists())
        self.assertEqual(
            list(workspace.glob(f".{final_stage.name}.*.stage-tmp")), []
        )

        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        self.assertEqual(Path(staged["stage_dir"]), final_stage)
        self.assertTrue((final_stage / "stage_metadata.json").is_file())

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

    def test_windows_stage_guard_initialization_locks_before_seeding(self) -> None:
        class OrderedWindowsLock:
            LK_NBLCK = 1
            LK_UNLCK = 2

            def __init__(self, events, *, unavailable=False):
                self.events = events
                self.unavailable = unavailable

            def locking(self, fd, mode, count):
                self.events.append("unlock" if mode == self.LK_UNLCK else "lock")
                self.assertion = (os.fstat(fd).st_size, count)
                if self.unavailable and mode == self.LK_NBLCK:
                    raise OSError(hpc_bundle.errno.EACCES, "lock held")

        guard_path = self.temporary / "windows-first-use.guard"
        events = []
        fake_lock = OrderedWindowsLock(events)
        real_write = os.write

        def record_write(fd, payload):
            events.append("write")
            return real_write(fd, payload)

        with (
            mock.patch.object(hpc_bundle, "fcntl", None),
            mock.patch.object(hpc_bundle, "msvcrt", fake_lock),
            mock.patch.object(hpc_bundle.os, "write", side_effect=record_write),
        ):
            fd = hpc_bundle._StageLease._try_lock_file(guard_path)
            self.assertIsNotNone(fd)
            self.assertEqual(events[:2], ["lock", "write"])
            self.assertEqual(fake_lock.assertion, (0, 1))
            hpc_bundle._StageLease._unlock_file(fd)
        self.assertEqual(guard_path.read_bytes(), b"\0")
        self.assertEqual(events[-1], "unlock")

        # A losing first-use contender must neither seed the file nor leak the
        # descriptor it opened; it reports ordinary nonblocking unavailability.
        busy_path = self.temporary / "windows-busy-first-use.guard"
        captured_fds = []
        busy_events = []
        busy_lock = OrderedWindowsLock(busy_events, unavailable=True)
        real_open = os.open

        def record_open(*args, **kwargs):
            opened_fd = real_open(*args, **kwargs)
            captured_fds.append(opened_fd)
            return opened_fd

        with (
            mock.patch.object(hpc_bundle, "fcntl", None),
            mock.patch.object(hpc_bundle, "msvcrt", busy_lock),
            mock.patch.object(hpc_bundle.os, "open", side_effect=record_open),
            mock.patch.object(hpc_bundle.os, "write", wraps=real_write) as write,
        ):
            result = hpc_bundle._StageLease._try_lock_file(busy_path)
        self.assertIsNone(result)
        self.assertEqual(busy_events, ["lock"])
        write.assert_not_called()
        with self.assertRaises(OSError):
            os.fstat(captured_fds[-1])

    def test_concurrent_stage_submit_has_one_driver_invocation(self) -> None:
        target, _request = self._create_2d()
        workspace = self.temporary / "workspace_concurrent"
        started = threading.Event()
        finish = threading.Event()
        calls = []
        first_result = []
        first_error = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            started.set()
            if not finish.wait(5.0):
                raise RuntimeError("test did not release the staged driver")
            return SimpleNamespace(returncode=0, stdout="planning complete\n")

        def first_stage():
            try:
                first_result.append(
                    hpc_bundle.stage_portable_bundle(
                        target, workspace, run_driver=True, submit=True
                    )
                )
            except BaseException as exc:  # surfaced in the test thread below
                first_error.append(exc)

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle.subprocess, "run", side_effect=fake_run),
        ):
            thread = threading.Thread(target=first_stage)
            thread.start()
            self.assertTrue(started.wait(5.0))
            try:
                with self.assertRaisesRegex(hpc_bundle.BundleError, "active stage/submit lease"):
                    hpc_bundle.stage_portable_bundle(
                        target, workspace, run_driver=True, submit=True
                    )
            finally:
                finish.set()
                thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(first_error, [])
        self.assertEqual(len(first_result), 1)
        self.assertEqual(len(calls), 1)

    def test_stage_lease_stale_takeover_advances_generation(self) -> None:
        lease_path = self.temporary / "workspace_lease" / ".stage-lease.json"
        lease_path.parent.mkdir()
        lease_path.write_text(
            json.dumps(
                {
                    "schema": "ghost.hpc.stage-lease.v1",
                    "owner_token": "former-owner",
                    "generation": 7,
                    "host": "dead-login",
                    "pid": 123,
                }
            ),
            encoding="utf-8",
        )
        old = time.time() - 120.0
        os.utime(lease_path, (old, old))

        with hpc_bundle._StageLease(
            lease_path, stale_seconds=1.0, heartbeat_seconds=0.05
        ) as lease:
            current = json.loads(lease_path.read_text(encoding="utf-8"))
            self.assertTrue(lease.recovered_stale)
            self.assertEqual(lease.generation, 8)
            self.assertEqual(current["owner_token"], lease.owner_token)
            self.assertEqual(current["generation"], 8)
            hpc_bundle._StageLease._release_owned_file(
                lease_path, "former-owner", 7
            )
            self.assertTrue(lease_path.is_file())
        self.assertFalse(lease_path.exists())

    def test_stage_lease_heartbeat_prevents_false_stale_takeover(self) -> None:
        lease_path = self.temporary / "workspace_heartbeat" / ".stage-lease.json"
        with hpc_bundle._StageLease(
            lease_path, stale_seconds=0.08, heartbeat_seconds=0.01
        ):
            time.sleep(0.16)
            with self.assertRaisesRegex(
                hpc_bundle.BundleError,
                "(?:active stage/submit lease|acquiring this bundle's stage lease)",
            ):
                with hpc_bundle._StageLease(
                    lease_path, stale_seconds=0.08, heartbeat_seconds=0.01
                ):
                    self.fail("a live heartbeat lease must not be stolen")

    def test_old_looking_live_stage_guard_cannot_be_stolen(self) -> None:
        lease_path = self.temporary / "workspace_guard" / ".stage-lease.json"
        lease_path.parent.mkdir()
        holder = hpc_bundle._StageLease(lease_path)
        contender = hpc_bundle._StageLease(lease_path)
        guard = holder._acquire_guard()
        try:
            old = time.time() - 3600.0
            os.utime(holder.guard_path, (old, old))
            with self.assertRaisesRegex(hpc_bundle.BundleError, "Another process is acquiring"):
                contender._acquire_guard()
        finally:
            holder._release_guard(guard)

    def test_stage_heartbeat_retries_transient_filesystem_error(self) -> None:
        lease_path = self.temporary / "workspace_heartbeat_io" / ".stage-lease.json"
        lease = hpc_bundle._StageLease(
            lease_path, stale_seconds=1.0, heartbeat_seconds=0.01
        )
        lease.__enter__()
        original_acquire = lease._acquire_guard
        recovered = threading.Event()
        calls = [0]

        def flaky_acquire():
            calls[0] += 1
            if calls[0] == 1:
                raise OSError("transient shared-filesystem error")
            guard = original_acquire()
            recovered.set()
            return guard

        try:
            with mock.patch.object(lease, "_acquire_guard", side_effect=flaky_acquire):
                self.assertTrue(recovered.wait(1.0))
                self.assertIsNotNone(lease._thread)
                self.assertTrue(lease._thread.is_alive())
        finally:
            lease.__exit__(None, None, None)

    def test_displaced_stage_owner_cannot_remove_successor_lease(self) -> None:
        lease_path = self.temporary / "workspace_displaced" / ".stage-lease.json"
        former = hpc_bundle._StageLease(
            lease_path, stale_seconds=1.0, heartbeat_seconds=60.0
        )
        successor = hpc_bundle._StageLease(
            lease_path, stale_seconds=1.0, heartbeat_seconds=60.0
        )
        former.__enter__()
        old = time.time() - 120.0
        os.utime(lease_path, (old, old))
        successor.__enter__()
        try:
            with self.assertRaisesRegex(hpc_bundle.BundleError, "no longer owns"):
                former.require_current()
            former.__exit__(None, None, None)
            current = json.loads(lease_path.read_text(encoding="utf-8"))
            self.assertEqual(current["owner_token"], successor.owner_token)
            self.assertEqual(current["generation"], 2)
        finally:
            successor.__exit__(None, None, None)

    def test_interrupted_submit_without_job_id_fails_closed(self) -> None:
        target, _request = self._create_2d()
        workspace = self.temporary / "workspace_stale_state"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        stage_dir = Path(staged["stage_dir"])
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

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle.subprocess, "run") as runner,
        ):
            with self.assertRaisesRegex(
                hpc_bundle.BundleError, "may have accepted.*automatic resubmission"
            ):
                hpc_bundle.stage_portable_bundle(
                    target, workspace, run_driver=True, submit=True
                )

        runner.assert_not_called()
        state = json.loads((stage_dir / "stage_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "running")

    def test_terminal_state_without_result_is_recovered_not_resubmitted(self) -> None:
        target, _request = self._create_2d()
        workspace = self.temporary / "workspace_terminal_recovery"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        stage_dir = Path(staged["stage_dir"])
        (stage_dir / "stage_result.json").unlink()
        (stage_dir / "stage_state.json").write_text(
            json.dumps(
                {
                    "schema": "ghost.hpc.stage-state.v1",
                    "status": "complete",
                    "submission_requested": True,
                    "returncode": 0,
                    "job_ids": ["92345"],
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle.subprocess, "run") as runner,
        ):
            recovered = hpc_bundle.stage_portable_bundle(
                target, workspace, run_driver=True, submit=True
            )

        runner.assert_not_called()
        self.assertTrue(recovered["ok"])
        self.assertTrue(recovered["recovered"])
        self.assertEqual(recovered["job_ids"], ["92345"])

    def test_interrupted_non_submitting_driver_can_retry(self) -> None:
        target, _request = self._create_2d()
        workspace = self.temporary / "workspace_stale_planning"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        stage_dir = Path(staged["stage_dir"])
        (stage_dir / "stage_state.json").write_text(
            json.dumps(
                {
                    "schema": "ghost.hpc.stage-state.v1",
                    "status": "running",
                    "submission_requested": False,
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(
                hpc_bundle.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="planning complete\n"),
            ) as runner,
        ):
            result = hpc_bundle.stage_portable_bundle(
                target, workspace, run_driver=True, submit=False
            )

        self.assertTrue(result["ok"])
        runner.assert_called_once()
        state = json.loads((stage_dir / "stage_state.json").read_text(encoding="utf-8"))
        self.assertTrue(state["recovered_prior_running_state"])
        self.assertEqual(state["status"], "complete")

    def test_stale_running_state_with_job_journal_refuses_resubmit(self) -> None:
        target, _request = self._create_2d()
        workspace = self.temporary / "workspace_stale_submitted"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        stage_dir = Path(staged["stage_dir"])
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

        with (
            mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True),
            mock.patch.object(hpc_bundle.subprocess, "run") as runner,
        ):
            with self.assertRaisesRegex(hpc_bundle.BundleError, "81234"):
                hpc_bundle.stage_portable_bundle(
                    target, workspace, run_driver=True, submit=True
                )
        runner.assert_not_called()

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

    def test_recovery_running_state_and_journal_override_stale_result(self) -> None:
        target, request = self._create_2d()
        workspace = self.temporary / "workspace_stale_result"
        with mock.patch.object(hpc_bundle, "_linux_staging_available", return_value=True):
            staged = hpc_bundle.stage_portable_bundle(target, workspace)
        stage_dir = Path(staged["stage_dir"])
        (stage_dir / "stage_state.json").write_text(
            json.dumps(
                {
                    "schema": "ghost.hpc.stage-state.v1",
                    "status": "running",
                    "submission_requested": True,
                    "lease_owner_token": "newer-attempt",
                    "lease_generation": 2,
                }
            ),
            encoding="utf-8",
        )
        run_dir = stage_dir / "runs" / "run_newer"
        run_dir.mkdir()
        (run_dir / "submitted_jobs.json").write_text(
            json.dumps(
                {
                    "schema": "ghost.hpc.submitted-jobs.v1",
                    "job_ids": ["91234"],
                }
            ),
            encoding="utf-8",
        )

        recovered = hpc_bundle.recover_staged_bundle(stage_dir)

        self.assertFalse(recovered["ok"])
        self.assertTrue(recovered["recovered"])
        self.assertEqual(recovered["bundle_id"], request["bundle_id"])
        self.assertEqual(recovered["stage_state"], "running")
        self.assertEqual(recovered["run_id"], "run_newer")
        self.assertEqual(recovered["job_ids"], ["91234"])
        self.assertTrue(recovered["submission_requested"])

    def test_stage_json_is_synced_before_atomic_replace(self) -> None:
        destination = self.temporary / "durable" / "state.json"
        events = []
        real_fsync = hpc_bundle.os.fsync
        real_replace = hpc_bundle.os.replace

        def recording_fsync(fd):
            events.append("fsync")
            return real_fsync(fd)

        def recording_replace(source, target):
            events.append("replace")
            return real_replace(source, target)

        with (
            mock.patch.object(hpc_bundle.os, "fsync", side_effect=recording_fsync),
            mock.patch.object(hpc_bundle.os, "replace", side_effect=recording_replace),
        ):
            hpc_bundle._write_json_atomic(destination, {"ok": True})

        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"ok": True})
        self.assertIn("replace", events)
        self.assertIn("fsync", events)
        self.assertLess(events.index("fsync"), events.index("replace"))

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
