"""Focused off-screen regressions for the GRIM HPC Runs workspace."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from runs_workspace import (
    ConnectionValues,
    RunsWorkspace,
    TrackedRun,
    _parse_number_list,
)


class _MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.sync_count = 0

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:  # noqa: N802 - Qt-compatible fake
        self.values[key] = value

    def sync(self) -> None:
        self.sync_count += 1


class _FakeBundleService:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.verified: list[str] = []

    def create_portable_bundle(
        self,
        bundle_dir,
        *,
        solver,
        geometries,
        settings,
        bundle_id=None,
    ):
        path = Path(bundle_dir)
        if path.exists():
            raise ValueError("destination already exists")
        path.mkdir(parents=True)
        (path / "request.json").write_text("{}", encoding="utf-8")
        call = {
            "path": str(path),
            "solver": solver,
            "geometries": [dict(value) for value in geometries],
            "settings": dict(settings),
            "bundle_id": bundle_id,
        }
        self.created.append(call)
        actual_bundle_id = bundle_id or "0123456789abcdef0123456789abcdef"
        return {
            "schema": "ghost.hpc.portable-request.v1",
            "bundle_id": actual_bundle_id,
            "solver": solver,
        }

    def verify_portable_bundle(self, bundle_dir):
        path = Path(bundle_dir)
        if not (path / "request.json").is_file():
            raise ValueError("not a portable bundle")
        self.verified.append(str(path))
        solver = self.created[-1]["solver"] if self.created else "2d"
        return {
            "schema": "ghost.hpc.portable-request.v1",
            "bundle_id": (
                str(self.created[-1]["bundle_id"])
                if self.created and self.created[-1]["bundle_id"]
                else "0123456789abcdef0123456789abcdef"
            ),
            "solver": solver,
        }


class _FakeRemoteClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.upload_local_existed = False
        self.upload_local_path = ""
        self.statuses = (
            SimpleNamespace(job_id="321", state="RUNNING"),
            SimpleNamespace(job_id="654", state="PENDING"),
        )
        self.stage_exception: Exception | None = None
        self.stage_result = None

    def test_connection(self):
        self.calls.append(("test",))
        command = SimpleNamespace(stdout="GRIM_HPC_OK\nlogin01\nanalyst\n")
        return SimpleNamespace(hostname="login01", username="analyst", result=command)

    def upload_bundle(self, local_path, remote_directory):
        local = Path(local_path)
        self.calls.append(("upload", str(local), remote_directory))
        self.upload_local_existed = local.is_dir()
        self.upload_local_path = str(local)
        return SimpleNamespace(
            remote_path=f"{remote_directory}/{local.name}",
            local_path=local,
        )

    def stage_hpc_bundle(
        self,
        remote_cli,
        remote_bundle,
        remote_workspace,
        *,
        run_driver,
        submit,
        python_executable="python3",
    ):
        bundle_id = Path(remote_bundle).name.removeprefix("bundle_")
        self.calls.append(
            (
                "stage",
                remote_cli,
                remote_bundle,
                remote_workspace,
                run_driver,
                submit,
                python_executable,
            )
        )
        if self.stage_exception is not None:
            raise self.stage_exception
        self.stage_result = SimpleNamespace(
            payload={
                "schema": "ghost.hpc.stage-result.v1",
                "ok": True,
                "run_id": "backend-run-uuid",
                "bundle_id": bundle_id,
                "solver": "2d",
                "stage_dir": f"/cluster/grim/workspaces/grim_{bundle_id}",
                "run_dir": "/cluster/grim/workspaces/rcs_runs/run_20260101_010101",
                "output_root": "/cluster/grim/workspaces/rcs_runs",
                "log_path": "/cluster/grim/workspaces/stage-abcd/submit.log",
                "job_ids": ["321", "654"],
            },
            run_id="backend-run-uuid",
            job_ids=("321", "654"),
        )
        return self.stage_result

    def read_stage_result(self, remote_stage_result):
        self.calls.append(("read-stage-result", remote_stage_result))
        if self.stage_result is None:
            raise RuntimeError("stage_result.json is not ready")
        return self.stage_result

    def recover_hpc_bundle(
        self, remote_cli, remote_stage_directory, *, python_executable="python3"
    ):
        self.calls.append(
            (
                "recover-stage",
                remote_cli,
                remote_stage_directory,
                python_executable,
            )
        )
        if self.stage_result is None:
            raise RuntimeError("no recoverable stage state")
        return self.stage_result

    def query_jobs(self, job_ids):
        self.calls.append(("query", tuple(job_ids)))
        return self.statuses

    def tail_log(self, remote_path, *, lines):
        self.calls.append(("tail", remote_path, lines))
        return "solver progress: 8 / 10"

    def tail_run_logs(self, remote_path, *, lines, files):
        self.calls.append(("tail-tasks", remote_path, lines, files))
        return "task 0: frequency complete"

    def cancel(self, job_ids):
        self.calls.append(("cancel", tuple(job_ids)))
        return "cancelled"

    def download_results(self, remote_path, local_directory):
        self.calls.append(("download", remote_path, local_directory))
        return SimpleNamespace(
            remote_path=remote_path,
            local_path=Path(local_directory) / Path(remote_path).name,
        )


def _wait_for_idle(app: QApplication, workspace: RunsWorkspace) -> None:
    deadline = time.monotonic() + 5.0
    while workspace._thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    if workspace._thread is not None:
        raise AssertionError(f"workspace did not become idle: {workspace.busy_operation()}")


class NumberListParserTests(unittest.TestCase):
    def test_numbers_and_inclusive_ranges(self) -> None:
        self.assertEqual(
            _parse_number_list("1, 2.5, 3:5:1", label="Values"),
            [1.0, 2.5, 3.0, 4.0, 5.0],
        )
        self.assertEqual(
            _parse_number_list("2:-2:-2", label="Values"),
            [2.0, 0.0, -2.0],
        )

    def test_invalid_ranges_are_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "pointing away"):
            _parse_number_list("0:10:-1", label="Azimuths")
        with self.assertRaisesRegex(ValueError, "start:stop:step"):
            _parse_number_list("0:10", label="Azimuths")


class RunsWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frd = self.root / "target_FRD.geo"
        self.opn = self.root / "target_OPN.geo"
        self.bor = self.root / "body.geo"
        for path in (self.frd, self.opn, self.bor):
            path.write_text("# geometry\n", encoding="utf-8")
        self.settings = _MemorySettings()
        self.bundle_service = _FakeBundleService()
        self.remote_client = _FakeRemoteClient()
        self.factory_configs: list[ConnectionValues] = []

        def remote_factory(config):
            self.factory_configs.append(config)
            return self.remote_client

        self.workspace = RunsWorkspace(
            bundle_service=self.bundle_service,
            remote_client_factory=remote_factory,
            connection_config_factory=lambda values: values,
            settings=self.settings,
        )

    def tearDown(self) -> None:
        _wait_for_idle(self.app, self.workspace)
        self.workspace.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def _configure_connection(self) -> None:
        self.workspace.host_edit.setText("login.cluster.example")
        self.workspace.username_edit.setText("analyst")
        self.workspace.remote_root_edit.setText("/cluster/grim")
        self.workspace.remote_cli_edit.setText(
            "/cluster/GHOST/Backend/hpc_bundle.py"
        )

    def _configure_2d_request(self) -> None:
        self.workspace.add_geometry("FRD", self.frd)
        self.workspace.add_geometry("OPN", self.opn)
        self.workspace.frequency_edit.setText("2:6:2")
        self.workspace.azimuth_edit.setText("0, 45, 90")
        self.workspace.nodes_spin.setValue(4)
        self.workspace.jobs_spin.setValue(2)
        self.workspace.partition_edit.setText("compute")
        self.workspace.account_edit.setText("project_a")
        self.workspace.walltime_edit.setText("12:30:00")

    def test_construction_never_creates_a_bundle_or_remote_client(self) -> None:
        self.assertEqual(self.bundle_service.created, [])
        self.assertEqual(self.factory_configs, [])
        self.assertFalse(self.workspace.job_is_running())
        self.assertEqual(
            [self.workspace.transport_combo.itemData(index) for index in range(2)],
            ["openssh", "putty"],
        )

    def test_2d_and_bor_forms_emit_the_authoritative_settings(self) -> None:
        self._configure_2d_request()
        bundle = self.root / "manual_bundle"
        self.workspace.bundle_path_edit.setText(str(bundle))
        request = self.workspace._request_snapshot()
        self.assertEqual(request["solver"], "2d")
        self.assertEqual(
            request["settings"]["FREQUENCIES_GHZ"], [2.0, 4.0, 6.0]
        )
        self.assertEqual(request["settings"]["AZIMUTHS_DEG"], [0.0, 45.0, 90.0])
        self.assertEqual(request["settings"]["N_NODES"], 4)
        self.assertEqual(request["settings"]["SLURM_ACCOUNT"], "project_a")
        self.assertNotIn("ELEVATIONS_DEG", request["settings"])
        self.assertIsNone(request["bundle_id"])

        self.workspace.geometry_list.clear()
        self.workspace.solver_combo.setCurrentIndex(
            self.workspace.solver_combo.findData("bor")
        )
        self.workspace.add_geometry("BOR", self.bor)
        self.workspace.elevation_edit.setText("-10:10:10")
        self.workspace.body_axis_az_spin.setValue(15.0)
        self.workspace.body_axis_el_spin.setValue(-4.0)
        self.workspace.body_roll_spin.setValue(22.5)
        request = self.workspace._request_snapshot()
        self.assertEqual(request["solver"], "bor")
        self.assertEqual(request["settings"]["ELEVATIONS_DEG"], [-10.0, 0.0, 10.0])
        self.assertEqual(request["settings"]["BODY_AXIS_AZ_DEG"], 15.0)
        self.assertEqual(request["settings"]["BODY_AXIS_EL_DEG"], -4.0)
        self.assertEqual(request["settings"]["BODY_ROLL_DEG"], 22.5)

    def test_export_bundle_uses_current_form_and_persistent_target(self) -> None:
        self._configure_2d_request()
        target = self.root / "exported_bundle"
        self.workspace.bundle_path_edit.setText(str(target))

        self.assertTrue(self.workspace.export_bundle())
        _wait_for_idle(self.app, self.workspace)

        self.assertTrue((target / "request.json").is_file())
        self.assertEqual(len(self.bundle_service.created), 1)
        call = self.bundle_service.created[0]
        self.assertEqual(call["solver"], "2d")
        self.assertEqual(call["bundle_id"], None)
        self.assertEqual(
            [entry["role"] for entry in call["geometries"]], ["FRD", "OPN"]
        )
        self.assertIn("Portable HPC bundle ready", self.workspace.status_label.text())

    def test_current_form_is_accepted_by_real_hpc_bundle_contract(self) -> None:
        from ghost_integration import discover_ghost_backend, load_ghost_module

        self._configure_2d_request()
        target = self.root / "real_contract_bundle"
        self.workspace.bundle_path_edit.setText(str(target))
        request = self.workspace._request_snapshot()
        backend = discover_ghost_backend()
        flat_names = {path.stem for path in backend.glob("*.py")}
        missing = object()
        modules_before = {
            name: sys.modules.get(name, missing) for name in flat_names
        }
        path_before = list(sys.path)
        try:
            service = load_ghost_module("hpc_bundle")
            created = service.create_portable_bundle(
                request["bundle_path"],
                solver=request["solver"],
                geometries=request["geometries"],
                settings=request["settings"],
                bundle_id=request["bundle_id"],
            )
        finally:
            sys.path[:] = path_before
            for name, previous in modules_before.items():
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.assertEqual(created["schema"], "ghost.hpc.portable-request.v1")
        self.assertEqual(created["settings"]["FREQUENCIES_GHZ"], [2.0, 4.0, 6.0])
        self.assertEqual(
            [entry["role"] for entry in created["geometries"]], ["FRD", "OPN"]
        )

    def test_upload_submit_builds_fresh_temporary_bundle_from_same_form(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.bundle_path_edit.clear()  # Direct mode needs no prior export.
        self.workspace.run_name_edit.setText("survey_01")

        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)

        self.assertEqual(len(self.bundle_service.created), 1)
        bundle_id = str(self.bundle_service.created[0]["bundle_id"])
        self.assertRegex(bundle_id, r"^[0-9a-f]{32}$")
        self.assertTrue(self.remote_client.upload_local_existed)
        self.assertFalse(Path(self.remote_client.upload_local_path).exists())
        stage_call = next(call for call in self.remote_client.calls if call[0] == "stage")
        upload_call = next(call for call in self.remote_client.calls if call[0] == "upload")
        self.assertEqual(stage_call[2], f"{upload_call[2]}/{Path(upload_call[1]).name}")
        self.assertEqual(stage_call[4:], (True, True, "python3"))

        run = self.workspace.tracked_runs[0]
        self.assertEqual(run.run_id, "survey_01")
        self.assertEqual(run.job_ids, ("321", "654"))
        self.assertEqual(run.state, "SUBMITTED")
        self.assertEqual(run.bundle_id, bundle_id)
        self.assertEqual(
            run.remote_stage_result,
            f"/cluster/grim/workspaces/grim_{bundle_id}/stage_result.json",
        )
        self.assertEqual(
            run.remote_results,
            "/cluster/grim/workspaces/rcs_runs/run_20260101_010101/results",
        )
        self.assertNotEqual(
            run.remote_results,
            "/cluster/grim/workspaces/rcs_runs",
        )
        self.assertFalse(self.workspace.job_is_running())

    def test_duplicate_run_name_is_rejected_without_a_second_submission(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.run_name_edit.setText("survey_unique")
        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)

        self.assertFalse(self.workspace.upload_and_submit())
        self.assertIn("already tracked", self.workspace.last_error)
        self.assertEqual(len(self.bundle_service.created), 1)

    def test_remote_python_is_passed_to_staging_and_persisted(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.remote_python_edit.setText("/cluster/venv/bin/python")
        self.workspace.run_name_edit.setText("survey_venv")

        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)

        stage_call = next(
            call for call in self.remote_client.calls if call[0] == "stage"
        )
        self.assertEqual(stage_call[-1], "/cluster/venv/bin/python")
        self.assertEqual(
            self.workspace.tracked_runs[0].connection["python_executable"],
            "/cluster/venv/bin/python",
        )

    def test_interrupted_stage_is_retained_and_refresh_recovers_job_ids(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.run_name_edit.setText("survey_recover")
        self.remote_client.stage_exception = ConnectionError("SSH connection reset")

        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)
        run = self.workspace.tracked_runs[0]
        self.assertEqual(run.state, "SUBMISSION UNKNOWN")
        self.assertFalse(run.job_ids)

        bundle_id = run.bundle_id
        self.remote_client.stage_exception = None
        self.remote_client.stage_result = SimpleNamespace(
            payload={
                "schema": "ghost.hpc.stage-result.v1",
                "ok": True,
                "bundle_id": bundle_id,
                "solver": "2d",
                "stage_dir": f"/cluster/grim/workspaces/grim_{bundle_id}",
                "run_dir": "/cluster/grim/workspaces/runs/run_recovered",
                "log_path": f"/cluster/grim/workspaces/grim_{bundle_id}/driver_submit.log",
                "job_ids": ["321", "654"],
            },
            job_ids=("321", "654"),
        )
        self.workspace.jobs_table.selectRow(0)
        self.assertTrue(self.workspace.refresh_selected_run())
        _wait_for_idle(self.app, self.workspace)

        recovered = self.workspace.tracked_runs[0]
        self.assertEqual(recovered.job_ids, ("321", "654"))
        self.assertEqual(recovered.state, "RUNNING")
        self.assertTrue(
            any(call[0] == "recover-stage" for call in self.remote_client.calls)
        )

    def test_partial_driver_failure_preserves_already_submitted_jobs(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.run_name_edit.setText("survey_partial")

        class PartialFailure(RuntimeError):
            pass

        failure = PartialFailure("second sbatch failed")
        failure.bundle_result = SimpleNamespace(
            payload={
                "schema": "ghost.hpc.stage-result.v1",
                "ok": False,
                "bundle_id": "0123456789abcdef0123456789abcdef",
                "run_dir": "/cluster/grim/workspaces/runs/run_partial",
                "log_path": "/cluster/grim/workspaces/driver_submit.log",
                "job_ids": ["321"],
            },
            job_ids=("321",),
        )
        self.remote_client.stage_exception = failure

        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)

        run = self.workspace.tracked_runs[0]
        self.assertEqual(run.state, "PARTIAL SUBMISSION")
        self.assertEqual(run.job_ids, ("321",))
        self.assertEqual(
            run.remote_results,
            "/cluster/grim/workspaces/runs/run_partial/results",
        )

    def test_refresh_updates_scheduler_state_and_log_without_marking_remote_job_busy(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.run_name_edit.setText("survey_02")
        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)
        self.workspace.jobs_table.selectRow(0)
        self.remote_client.statuses = (
            SimpleNamespace(job_id="321", state="COMPLETED"),
            SimpleNamespace(job_id="654", state="COMPLETED"),
        )

        self.assertTrue(self.workspace.refresh_selected_run())
        _wait_for_idle(self.app, self.workspace)

        run = self.workspace.tracked_runs[0]
        self.assertEqual(run.state, "COMPLETED")
        self.assertIn("solver progress", self.workspace.log_view.toPlainText())
        self.assertIn("frequency complete", self.workspace.log_view.toPlainText())
        self.assertFalse(self.workspace.job_is_running())
        self.assertIsNone(self.workspace.busy_operation())

    def test_cancel_uses_only_recorded_job_ids(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.run_name_edit.setText("survey_cancel")
        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)
        self.workspace.jobs_table.selectRow(0)

        with mock.patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.assertTrue(self.workspace.cancel_selected_run())
        _wait_for_idle(self.app, self.workspace)

        self.assertIn(("cancel", ("321", "654")), self.remote_client.calls)
        self.assertEqual(self.workspace.tracked_runs[0].state, "CANCEL REQUESTED")

    def test_download_uses_recorded_run_results_not_stage_root(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.run_name_edit.setText("survey_download")
        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)
        self.workspace.tracked_runs[0].state = "COMPLETED"
        self.workspace._render_runs(select_run_id="survey_download")
        self.workspace.jobs_table.selectRow(0)
        destination = self.root / "downloaded"
        destination.mkdir()

        with mock.patch.object(
            QFileDialog,
            "getExistingDirectory",
            return_value=str(destination),
        ):
            self.assertTrue(self.workspace.download_selected_results())
        _wait_for_idle(self.app, self.workspace)

        download = next(
            call for call in self.remote_client.calls if call[0] == "download"
        )
        self.assertEqual(
            download[1],
            "/cluster/grim/workspaces/rcs_runs/run_20260101_010101/results",
        )
        self.assertEqual(download[2], str(destination))
        self.assertIn(
            str(destination / "results"), self.workspace.status_label.text()
        )

    def test_download_waits_for_terminal_state_and_refuses_local_merge(self) -> None:
        self._configure_2d_request()
        self._configure_connection()
        self.workspace.run_name_edit.setText("survey_safe_download")
        self.assertTrue(self.workspace.upload_and_submit())
        _wait_for_idle(self.app, self.workspace)
        self.workspace.jobs_table.selectRow(0)

        with mock.patch.object(QFileDialog, "getExistingDirectory") as chooser:
            self.assertFalse(self.workspace.download_selected_results())
        chooser.assert_not_called()
        self.assertIn("terminal state", self.workspace.last_error)

        destination = self.root / "collision"
        (destination / "results").mkdir(parents=True)
        self.workspace.tracked_runs[0].state = "COMPLETED"
        self.workspace._render_runs(select_run_id="survey_safe_download")
        with mock.patch.object(
            QFileDialog, "getExistingDirectory", return_value=str(destination)
        ):
            self.assertFalse(self.workspace.download_selected_results())
        self.assertIn("already exists", self.workspace.last_error)
        self.assertFalse(
            any(call[0] == "download" for call in self.remote_client.calls)
        )

    def test_bundle_chooser_distinguishes_existing_bundle_from_parent(self) -> None:
        parent = self.root / "exports"
        parent.mkdir()
        self.workspace.run_name_edit.setText("survey_choice")
        with mock.patch.object(
            QFileDialog, "getExistingDirectory", return_value=str(parent)
        ):
            self.workspace._choose_bundle()
        self.assertEqual(
            Path(self.workspace.bundle_path_edit.text()),
            parent / "survey_choice_bundle",
        )

        existing = self.root / "existing_bundle"
        existing.mkdir()
        (existing / "request.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(
            QFileDialog, "getExistingDirectory", return_value=str(existing)
        ):
            self.workspace._choose_bundle()
        self.assertEqual(Path(self.workspace.bundle_path_edit.text()), existing)

    def test_corrupt_registry_entry_does_not_discard_valid_runs(self) -> None:
        valid = TrackedRun(
            run_id="kept_run",
            solver="2d",
            job_ids=("42",),
            state="RUNNING",
        ).to_json_value()
        loaded_settings = _MemorySettings()
        loaded_settings.values["runs/tracked_runs"] = json.dumps(
            {
                "schema": "grim.runs.registry.v1",
                "runs": [valid, {"run_id": "not/a/valid/run"}],
            }
        )
        loaded = RunsWorkspace(
            bundle_service=self.bundle_service,
            remote_client_factory=lambda config: self.remote_client,
            connection_config_factory=lambda values: values,
            settings=loaded_settings,
        )
        try:
            self.assertEqual([run.run_id for run in loaded.tracked_runs], ["kept_run"])
            self.assertIn("skipped 1 corrupt", loaded.log_view.toPlainText())
        finally:
            loaded.deleteLater()
            self.app.processEvents()

    def test_settings_persist_only_non_secret_connection_metadata_and_runs(self) -> None:
        self._configure_connection()
        self.workspace.identity_edit.setText("")
        self.workspace.save_settings()

        self.assertEqual(
            self.settings.values["runs/host"], "login.cluster.example"
        )
        self.assertEqual(self.settings.values["runs/username"], "analyst")
        self.assertIn("runs/tracked_runs", self.settings.values)
        lowered_keys = " ".join(self.settings.values).casefold()
        self.assertNotIn("password", lowered_keys)
        self.assertNotIn("passphrase", lowered_keys)
        self.assertGreater(self.settings.sync_count, 0)

    def test_putty_mode_requires_only_a_saved_session(self) -> None:
        index = self.workspace.transport_combo.findData("putty")
        self.workspace.transport_combo.setCurrentIndex(index)
        self.workspace.profile_edit.setText("HPC Production")
        self.workspace.putty_host_key_edit.setText("ssh-ed25519 255 SHA256:trusted")

        self.assertTrue(self.workspace.test_connection())
        _wait_for_idle(self.app, self.workspace)

        config = self.factory_configs[-1]
        self.assertEqual(config.transport, "putty")
        self.assertEqual(config.profile, "HPC Production")
        self.assertIn("Connection successful", self.workspace.status_label.text())

    def test_default_config_adapter_matches_hpc_remote_factories(self) -> None:
        from hpc_remote import Transport

        self.workspace._connection_config_factory_value = None
        direct = self.workspace._connection_config(
            ConnectionValues(
                transport="openssh",
                host="login.cluster.example",
                username="analyst",
                port=2222,
            )
        )
        self.assertEqual(direct.transport, Transport.OPENSSH)
        self.assertEqual(direct.host, "login.cluster.example")
        self.assertEqual(direct.port, 2222)

        alias = self.workspace._connection_config(
            ConnectionValues(transport="openssh", profile="approved-hpc")
        )
        self.assertEqual(alias.ssh_config_alias, "approved-hpc")

        putty = self.workspace._connection_config(
            ConnectionValues(
                transport="putty",
                profile="HPC Production",
                putty_host_key="ssh-ed25519 255 SHA256:trusted",
            )
        )
        self.assertEqual(putty.transport, Transport.PUTTY)
        self.assertEqual(putty.putty_session, "HPC Production")


if __name__ == "__main__":
    unittest.main()
