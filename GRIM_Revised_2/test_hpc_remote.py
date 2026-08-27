from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest import mock

import hpc_remote
from hpc_remote import (
    CommandFailedError,
    CommandTimeoutError,
    ConfigurationError,
    ConnectionConfig,
    HpcBundleFailedError,
    HpcRemoteClient,
    ProtocolError,
    Transport,
)


class FakeRunner:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(
        self, argv: tuple[str, ...], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((tuple(argv), timeout))
        if not self.responses:
            raise AssertionError("unexpected process invocation")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            returncode, stdout, stderr = response
            return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
        assert isinstance(response, subprocess.CompletedProcess)
        return response


class MaterializingDownloadRunner(FakeRunner):
    """Fake SCP runner that creates the downloaded basename in its target."""

    def __init__(self, name: str, *responses: object) -> None:
        super().__init__(*responses)
        self.name = name

    def __call__(
        self, argv: tuple[str, ...], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        result = super().__call__(argv, timeout=timeout)
        destination = Path(argv[-1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / self.name).mkdir()
        return result


def completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> tuple[int, str, str]:
    return returncode, stdout, stderr


class ConnectionConfigTests(unittest.TestCase):
    def test_direct_openssh_configuration_is_validated(self) -> None:
        config = ConnectionConfig.openssh_host(
            "login.cluster.example",
            username="analyst",
            port=2222,
            identity_file=Path("C:/keys/grim_key"),
            known_hosts_file=Path("C:/keys/known_hosts"),
        )

        self.assertIs(config.transport, Transport.OPENSSH)
        self.assertEqual(config.port, 2222)
        self.assertNotIn("password", {field.name for field in fields(config)})

    def test_alias_and_direct_endpoint_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "exactly one"):
            ConnectionConfig()
        with self.assertRaisesRegex(ConfigurationError, "exactly one"):
            ConnectionConfig(host="cluster", ssh_config_alias="saved")
        with self.assertRaisesRegex(ConfigurationError, "defined by the SSH config"):
            ConnectionConfig(
                ssh_config_alias="saved", username="somebody", port=None
            )

    def test_host_port_and_timeouts_reject_option_or_unbounded_values(self) -> None:
        with self.assertRaises(ConfigurationError):
            ConnectionConfig.openssh_host("-oProxyCommand=bad")
        with self.assertRaises(ConfigurationError):
            ConnectionConfig.openssh_host("host", port=70000)
        with self.assertRaises(ConfigurationError):
            ConnectionConfig.openssh_host("host", command_timeout=float("inf"))
        with self.assertRaises(ConfigurationError):
            ConnectionConfig.openssh_host("host", transfer_timeout=86401)

    def test_putty_requires_a_safe_saved_session(self) -> None:
        config = ConnectionConfig.putty_saved_session(
            "Lab Cluster", host_key="ssh-ed25519 255 SHA256:example"
        )
        self.assertIs(config.transport, Transport.PUTTY)
        self.assertEqual(config.putty_session, "Lab Cluster")
        with self.assertRaises(ConfigurationError):
            ConnectionConfig.putty_saved_session("-ssh evil")
        with self.assertRaises(ConfigurationError):
            ConnectionConfig(
                transport="putty", putty_session="saved", host="other-host"
            )


class InvocationTests(unittest.TestCase):
    def test_openssh_connection_probe_uses_strict_noninteractive_argv(self) -> None:
        runner = FakeRunner(completed("GRIM_HPC_OK\nlogin01\nanalyst\n"))
        client = HpcRemoteClient(
            ConnectionConfig.openssh_host(
                "login.cluster.example",
                username="analyst",
                port=2207,
                identity_file="C:/keys/grim key",
            ),
            runner,
        )

        info = client.test_connection(timeout=9)

        self.assertEqual((info.hostname, info.username), ("login01", "analyst"))
        argv, timeout = runner.calls[0]
        self.assertEqual(argv[0:2], ("ssh", "-T"))
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertIn("ForwardAgent=no", argv)
        self.assertIn("ForwardX11=no", argv)
        self.assertIn("ClearAllForwardings=yes", argv)
        self.assertIn("PermitLocalCommand=no", argv)
        self.assertIn("RemoteCommand=none", argv)
        self.assertIn("ConnectTimeout=15", argv)
        self.assertIn(str(Path("C:/keys/grim key")), argv)
        self.assertIn("-p", argv)
        self.assertIn("2207", argv)
        self.assertEqual(argv[-2], "analyst@login.cluster.example")
        self.assertEqual(shlex.split(argv[-1])[0:2], ["sh", "-c"])
        self.assertEqual(timeout, 9)

    def test_openssh_alias_leaves_user_port_and_key_to_ssh_config(self) -> None:
        runner = FakeRunner(completed("ok"))
        client = HpcRemoteClient(
            ConnectionConfig.openssh_alias(
                "lab-hpc", ssh_config_file="C:/ssh/grim_config"
            ),
            runner,
        )

        client.run_remote(("true",))

        argv, _ = runner.calls[0]
        self.assertEqual(argv[-2], "lab-hpc")
        self.assertIn("-F", argv)
        self.assertNotIn("-p", argv)
        self.assertNotIn("-i", argv)

    def test_putty_uses_batch_saved_session_and_optional_host_pin(self) -> None:
        runner = FakeRunner(completed("ok"))
        client = HpcRemoteClient(
            ConnectionConfig.putty_saved_session(
                "Lab Cluster", host_key="ssh-ed25519 255 SHA256:abc"
            ),
            runner,
        )

        client.run_remote(("printf", "%s", "safe"))

        argv, _ = runner.calls[0]
        self.assertEqual(
            argv[:5], ("plink", "-batch", "-T", "-load", "Lab Cluster")
        )
        self.assertIn("-hostkey", argv)
        self.assertNotIn("-pw", argv)

    def test_remote_tokens_are_posix_quoted_not_locally_shell_executed(self) -> None:
        runner = FakeRunner(completed())
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)
        dangerous_literal = "value; touch /tmp/not-created"

        client.run_remote(("printf", "%s", dangerous_literal))

        remote_command = runner.calls[0][0][-1]
        self.assertEqual(
            shlex.split(remote_command), ["printf", "%s", dangerous_literal]
        )
        with self.assertRaises(ConfigurationError):
            client.run_remote(("printf", "line1\nline2"))

    def test_default_runner_explicitly_disables_shell_and_stdin(self) -> None:
        fake_completed = subprocess.CompletedProcess(["ssh"], 0, "", "")
        with mock.patch.object(
            hpc_remote.subprocess, "run", return_value=fake_completed
        ) as run:
            result = hpc_remote._default_process_runner(["ssh", "host"], timeout=3)

        self.assertEqual(result.returncode, 0)
        kwargs = run.call_args.kwargs
        self.assertIs(kwargs["shell"], False)
        self.assertIs(kwargs["check"], False)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["timeout"], 3)

    def test_timeout_and_nonzero_exit_have_typed_errors(self) -> None:
        timeout_runner = FakeRunner(subprocess.TimeoutExpired(("ssh",), 4))
        timeout_client = HpcRemoteClient(
            ConnectionConfig.openssh_alias("cluster"), timeout_runner
        )
        with self.assertRaises(CommandTimeoutError) as caught:
            timeout_client.run_remote(("true",), timeout=4)
        self.assertEqual(caught.exception.timeout, 4)

        failure_runner = FakeRunner(completed("", "permission denied", 255))
        failure_client = HpcRemoteClient(
            ConnectionConfig.openssh_alias("cluster"), failure_runner
        )
        with self.assertRaisesRegex(CommandFailedError, "permission denied"):
            failure_client.run_remote(("true",))

    def test_putty_batch_prompt_failure_keeps_error_and_adds_guidance(self) -> None:
        raw_error = "FATAL ERROR: Cannot Answer Interactive Prompts In Batch Mode"
        runner = FakeRunner(completed(stderr=raw_error, returncode=1))
        client = HpcRemoteClient(
            ConnectionConfig.putty_saved_session("HPC Production"), runner
        )

        with self.assertRaises(CommandFailedError) as caught:
            client.run_remote(("true",))

        message = str(caught.exception)
        self.assertIn(raw_error, message)
        self.assertIn("saved PuTTY session interactively", message)
        self.assertIn("auto-login username", message)
        self.assertIn("verified host key", message)
        self.assertIn("Pageant", message)
        self.assertIn("connection sharing", message)
        self.assertIn("Export Bundle", message)
        self.assertIn("-batch", runner.calls[0][0])
        self.assertNotIn("-pw", runner.calls[0][0])

    def test_missing_client_executable_is_a_clear_configuration_error(self) -> None:
        runner = FakeRunner(FileNotFoundError("missing"))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)
        with self.assertRaisesRegex(ConfigurationError, "'ssh' was not found"):
            client.run_remote(("true",))


class TransferTests(unittest.TestCase):
    def test_upload_creates_remote_directory_then_copies_actual_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "portable-bundle.zip"
            local.write_bytes(b"bundle")
            runner = FakeRunner(completed(), completed())
            client = HpcRemoteClient(
                ConnectionConfig.openssh_host(
                    "login", username="me", port=2022, transfer_timeout=500
                ),
                runner,
            )

            transfer = client.upload_bundle(local, "/scratch/me/grim uploads")

        self.assertEqual(
            transfer.remote_path, "/scratch/me/grim uploads/portable-bundle.zip"
        )
        self.assertEqual(len(runner.calls), 2)
        mkdir_command = shlex.split(runner.calls[0][0][-1])
        self.assertEqual(
            mkdir_command, ["mkdir", "-p", "--", "/scratch/me/grim uploads"]
        )
        copy_argv, copy_timeout = runner.calls[1]
        self.assertEqual(copy_argv[0], "scp")
        self.assertIn("-B", copy_argv)
        self.assertIn("StrictHostKeyChecking=yes", copy_argv)
        self.assertIn("-P", copy_argv)
        self.assertEqual(copy_argv[-1], "me@login:/scratch/me/grim uploads")
        self.assertEqual(copy_timeout, 500)

    def test_direct_ipv6_copy_target_is_bracketed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "bundle"
            local.mkdir()
            runner = FakeRunner(completed(), completed())
            client = HpcRemoteClient(
                ConnectionConfig.openssh_host("2001:db8::7", username="me"),
                runner,
            )
            client.upload_bundle(local, "/scratch/grim")

        self.assertEqual(
            runner.calls[1][0][-1], "me@[2001:db8::7]:/scratch/grim"
        )

    def test_directory_upload_is_recursive_and_unsafe_remote_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "bundle"
            local.mkdir()
            runner = FakeRunner(completed(), completed())
            client = HpcRemoteClient(
                ConnectionConfig.openssh_alias("cluster"), runner
            )
            client.upload_bundle(local, "/scratch/jobs")
            self.assertIn("-r", runner.calls[1][0])

            with self.assertRaises(ConfigurationError):
                client.upload_bundle(local, "/scratch/jobs; shutdown")
            with self.assertRaises(ConfigurationError):
                client.upload_bundle(local, "/scratch/../etc")

    def test_putty_download_uses_pscp_saved_session_without_password(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = MaterializingDownloadRunner("results", completed())
            client = HpcRemoteClient(
                ConnectionConfig.putty_saved_session(
                    "HPC Profile", host_key="ssh-rsa 2048 SHA256:pinned"
                ),
                runner,
            )

            transfer = client.download_results(
                "/scratch/me/run 17/results", directory
            )

            self.assertTrue(transfer.local_path.is_dir())
            self.assertFalse(
                any(Path(directory).glob(".results.grim-download-*"))
            )

        argv, timeout = runner.calls[0]
        self.assertEqual(argv[:4], ("pscp", "-batch", "-load", "HPC Profile"))
        self.assertIn("-hostkey", argv)
        self.assertIn("-r", argv)
        self.assertNotIn("-pw", argv)
        self.assertEqual(argv[-2], "HPC Profile:/scratch/me/run 17/results")
        self.assertEqual(transfer.local_path.name, "results")
        self.assertEqual(timeout, 1800)

    def test_pscp_batch_prompt_failure_adds_same_guidance(self) -> None:
        raw_error = "cannot answer interactive prompts in batch mode"
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(
                completed(), completed(stderr=raw_error, returncode=1)
            )
            client = HpcRemoteClient(
                ConnectionConfig.putty_saved_session("HPC Profile"), runner
            )
            bundle = Path(directory) / "bundle.zip"
            bundle.write_bytes(b"bundle")

            with self.assertRaises(CommandFailedError) as caught:
                client.upload_bundle(bundle, "/scratch/grim")

        message = str(caught.exception)
        self.assertIn(raw_error, message)
        self.assertIn("saved PuTTY session interactively", message)
        self.assertEqual(runner.calls[-1][0][0:2], ("pscp", "-batch"))
        self.assertNotIn("-pw", runner.calls[-1][0])

    def test_download_refuses_to_merge_with_an_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "results").mkdir()
            runner = FakeRunner(completed())
            client = HpcRemoteClient(
                ConnectionConfig.openssh_alias("cluster"), runner
            )
            with self.assertRaisesRegex(ConfigurationError, "already exists"):
                client.download_results("/scratch/run/results", directory)
        self.assertEqual(runner.calls, [])

    def test_failed_download_removes_private_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FakeRunner(completed(stderr="transfer failed", returncode=1))
            client = HpcRemoteClient(
                ConnectionConfig.openssh_alias("cluster"), runner
            )
            with self.assertRaises(CommandFailedError):
                client.download_results("/scratch/run/results", directory)
            self.assertFalse((Path(directory) / "results").exists())
            self.assertFalse(
                any(Path(directory).glob(".results.grim-download-*"))
            )


class HpcBundleTests(unittest.TestCase):
    def _stage_payload(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "ghost.hpc.stage-result.v1",
            "ok": True,
            "bundle_id": "bundle-a1",
            "run_id": "run-42",
            "solver": "GHOST",
            "bundle_sha256": "abc",
            "stage_dir": "/scratch/grim/staged/bundle-a1",
            "driver_path": "/scratch/grim/staged/bundle-a1/run.py",
            "output_root": "/scratch/grim/results",
            "driver_ran": True,
            "submission_requested": True,
            "submitted": True,
            "run_dir": "/scratch/grim/results/run-42",
            "job_ids": [8123, "8124_7"],
            "log_path": "/scratch/grim/results/run-42/driver.log",
            "returncode": 0,
        }
        payload.update(overrides)
        return payload

    def test_stage_invokes_finalized_cli_contract_and_extracts_ids(self) -> None:
        payload = self._stage_payload()
        runner = FakeRunner(completed(json.dumps(payload, separators=(",", ":"))))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        result = client.stage_hpc_bundle(
            "/opt/grim/Backend/hpc_bundle.py",
            "/scratch/uploads/portable-bundle",
            "/scratch/grim workspace",
            run_driver=True,
            submit=True,
            timeout=75,
        )

        self.assertEqual(result.schema, "ghost.hpc.stage-result.v1")
        self.assertTrue(result.ok)
        self.assertEqual(result.bundle_id, "bundle-a1")
        self.assertEqual(result.run_id, "run-42")
        self.assertEqual(result.job_ids, ("8123", "8124_7"))
        argv, timeout = runner.calls[0]
        self.assertEqual(
            shlex.split(argv[-1]),
            [
                "python3",
                "/opt/grim/Backend/hpc_bundle.py",
                "stage",
                "/scratch/uploads/portable-bundle",
                "--workspace-root",
                "/scratch/grim workspace",
                "--run-driver",
                "--submit",
            ],
        )
        self.assertEqual(timeout, 75)

    def test_bundle_id_is_a_fallback_run_identifier(self) -> None:
        payload = self._stage_payload()
        payload.pop("run_id")
        runner = FakeRunner(completed(json.dumps(payload)))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        result = client.invoke_hpc_bundle(
            "/opt/hpc_bundle.py", ("stage", "/scratch/bundle")
        )

        self.assertEqual(result.run_id, "bundle-a1")

    def test_stage_defaults_to_long_transfer_timeout_for_driver_planning(self) -> None:
        payload = self._stage_payload()
        runner = FakeRunner(completed(json.dumps(payload)))
        client = HpcRemoteClient(
            ConnectionConfig.openssh_alias(
                "cluster", command_timeout=12, transfer_timeout=987
            ),
            runner,
        )

        client.stage_hpc_bundle(
            "/opt/hpc_bundle.py",
            "/scratch/bundle",
            "/scratch/workspace",
            run_driver=True,
        )

        self.assertEqual(runner.calls[0][1], 987)

    def test_nonzero_bundle_response_is_parsed_into_typed_error(self) -> None:
        payload = self._stage_payload(
            ok=False,
            job_ids=[],
            returncode=2,
            error={"code": "invalid_bundle", "message": "manifest mismatch"},
        )
        runner = FakeRunner(completed(json.dumps(payload), returncode=2))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        with self.assertRaisesRegex(
            HpcBundleFailedError, "manifest mismatch"
        ) as caught:
            client.stage_hpc_bundle(
                "/opt/hpc_bundle.py", "/scratch/bundle", "/scratch/workspace"
            )

        self.assertFalse(caught.exception.bundle_result.ok)
        self.assertEqual(caught.exception.result.returncode, 2)

    def test_read_stage_result_recovers_a_previous_submission(self) -> None:
        payload = self._stage_payload()
        runner = FakeRunner(completed(json.dumps(payload)))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        result = client.read_stage_result(
            "/scratch/grim/staged/bundle-a1/stage_result.json"
        )

        self.assertEqual(result.job_ids, ("8123", "8124_7"))
        self.assertEqual(result.run_id, "run-42")
        self.assertEqual(
            shlex.split(runner.calls[0][0][-1]),
            [
                "cat",
                "--",
                "/scratch/grim/staged/bundle-a1/stage_result.json",
            ],
        )

    def test_recover_command_returns_partial_job_ids_despite_nonzero_status(self) -> None:
        payload = self._stage_payload(
            ok=False,
            recovered=True,
            stage_state="running",
            job_ids=["8123"],
            error="stage wrapper was interrupted",
        )
        runner = FakeRunner(completed(json.dumps(payload), returncode=1))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        result = client.recover_hpc_bundle(
            "/opt/grim/Backend/hpc_bundle.py",
            "/scratch/grim/grim_0123456789abcdef0123456789abcdef",
            python_executable="/opt/grim/venv/bin/python",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.job_ids, ("8123",))
        self.assertEqual(
            shlex.split(runner.calls[0][0][-1]),
            [
                "/opt/grim/venv/bin/python",
                "/opt/grim/Backend/hpc_bundle.py",
                "recover",
                "/scratch/grim/grim_0123456789abcdef0123456789abcdef",
            ],
        )

    def test_malformed_json_schema_and_job_id_are_rejected(self) -> None:
        client = HpcRemoteClient(
            ConnectionConfig.openssh_alias("cluster"), FakeRunner(completed("not json"))
        )
        with self.assertRaises(ProtocolError):
            client.invoke_hpc_bundle("/opt/hpc_bundle.py")

        wrong_schema = self._stage_payload(schema="unexpected.v1")
        client = HpcRemoteClient(
            ConnectionConfig.openssh_alias("cluster"),
            FakeRunner(completed(json.dumps(wrong_schema))),
        )
        with self.assertRaisesRegex(ProtocolError, "unsupported schema"):
            client.stage_hpc_bundle(
                "/opt/hpc_bundle.py", "/scratch/bundle", "/scratch/workspace"
            )

        bad_job = self._stage_payload(job_ids=["123; scancel 1"])
        client = HpcRemoteClient(
            ConnectionConfig.openssh_alias("cluster"),
            FakeRunner(completed(json.dumps(bad_job))),
        )
        with self.assertRaisesRegex(ProtocolError, "invalid job ID"):
            client.invoke_hpc_bundle("/opt/hpc_bundle.py")


class SchedulerTests(unittest.TestCase):
    def test_squeue_and_sacct_rows_are_normalized(self) -> None:
        runner = FakeRunner(
            completed("8123|RUNNING|grim-a|00:03|node01\n"),
            completed("8124|CANCELLED+|grim-b|01:02:03|0:0\n"),
        )
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        active = client.query_squeue([8123])
        history = client.query_sacct(["8124"])

        self.assertEqual(active[0].state, "RUNNING")
        self.assertEqual(active[0].source, "squeue")
        self.assertEqual(active[0].detail, "node01")
        self.assertEqual(history[0].state, "CANCELLED")
        self.assertEqual(history[0].source, "sacct")
        self.assertEqual(history[0].detail, "0:0")
        self.assertEqual(
            shlex.split(runner.calls[0][0][-1]),
            [
                "squeue",
                "--noheader",
                "--jobs",
                "8123",
                "--format=%i|%T|%j|%M|%R",
            ],
        )
        self.assertIn(
            "--format=JobID,State%32,JobName,Elapsed,ExitCode",
            shlex.split(runner.calls[1][0][-1]),
        )
        self.assertIn("--array", shlex.split(runner.calls[1][0][-1]))

    def test_query_jobs_uses_sacct_only_for_jobs_missing_from_queue(self) -> None:
        runner = FakeRunner(
            completed("10|RUNNING|first|00:01|node1\n"),
            completed("11|COMPLETED|second|00:02|0:0\n"),
        )
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        statuses = client.query_jobs(["10", "11"])

        self.assertEqual([row.job_id for row in statuses], ["10", "11"])
        self.assertEqual([row.source for row in statuses], ["squeue", "sacct"])
        self.assertIn("11", shlex.split(runner.calls[1][0][-1]))
        self.assertNotIn("10", shlex.split(runner.calls[1][0][-1]))

    def test_query_jobs_preserves_unknown_rows_for_every_requested_id(self) -> None:
        runner = FakeRunner(
            completed("10|COMPLETED|first|00:01|0:0\n"),
            completed(""),
        )
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        statuses = client.query_jobs(["10", "11"])

        self.assertEqual([row.job_id for row in statuses], ["10", "11"])
        self.assertEqual([row.state for row in statuses], ["COMPLETED", "UNKNOWN"])
        self.assertEqual(statuses[1].source, "unavailable")

    def test_query_jobs_aggregates_slurm_array_tasks_under_parent_id(self) -> None:
        runner = FakeRunner(
            completed(
                "10_0|COMPLETED|grim|00:02|node1\n"
                "10_1|RUNNING|grim|00:01|node2\n"
                "11_[0-3]|PENDING|grim|00:00|(Resources)\n"
            )
        )
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        statuses = client.query_jobs(["10", "11"])

        parents = {row.job_id: row for row in statuses if row.job_id in {"10", "11"}}
        self.assertEqual(parents["10"].state, "RUNNING")
        self.assertEqual(parents["11"].state, "PENDING")
        self.assertIn("1 COMPLETED", parents["10"].detail)
        self.assertIn("1 RUNNING", parents["10"].detail)
        self.assertEqual(len(runner.calls), 1, "active array parents must not query sacct")

    def test_query_jobs_aggregates_completed_array_history_and_failure(self) -> None:
        runner = FakeRunner(
            completed(""),
            completed(
                "12_0|COMPLETED|grim|00:02|0:0\n"
                "12_1|FAILED|grim|00:01|1:0\n"
            ),
        )
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        statuses = client.query_jobs(["12"])

        self.assertEqual(statuses[0].job_id, "12")
        self.assertEqual(statuses[0].state, "FAILED")

    def test_query_jobs_falls_back_to_sacct_after_squeue_purges_job(self) -> None:
        runner = FakeRunner(
            completed(
                stderr="slurm_load_jobs error: Invalid job id specified\n",
                returncode=1,
            ),
            completed(
                "12_0|COMPLETED|grim|00:02|0:0\n"
                "12_1|FAILED|grim|00:01|1:0\n"
            ),
        )
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        statuses = client.query_jobs(["12"])

        self.assertEqual(statuses[0].job_id, "12")
        self.assertEqual(statuses[0].state, "FAILED")
        self.assertEqual(statuses[0].source, "sacct")

    def test_squeue_does_not_hide_transport_or_authentication_failures(self) -> None:
        runner = FakeRunner(
            completed(stderr="Permission denied (publickey).\n", returncode=255)
        )
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        with self.assertRaisesRegex(CommandFailedError, "Permission denied"):
            client.query_squeue(["12"])

    def test_cancel_validates_ids_and_passes_each_as_a_quoted_token(self) -> None:
        runner = FakeRunner(completed())
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        client.cancel(["123", "124_7"])

        self.assertEqual(
            shlex.split(runner.calls[0][0][-1]),
            ["scancel", "--", "123", "124_7"],
        )
        with self.assertRaises(ConfigurationError):
            client.cancel(["123; touch /tmp/pwned"])
        self.assertEqual(len(runner.calls), 1)

    def test_tail_log_quotes_spaces_and_rejects_metacharacters(self) -> None:
        runner = FakeRunner(completed("last line\n"))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        output = client.tail_log("/scratch/run 7/slurm.log", lines=25)

        self.assertEqual(output, "last line\n")
        self.assertEqual(
            shlex.split(runner.calls[0][0][-1]),
            ["tail", "--lines", "25", "--", "/scratch/run 7/slurm.log"],
        )
        with self.assertRaises(ConfigurationError):
            client.tail_log("/scratch/log; cat /etc/passwd")
        with self.assertRaises(ConfigurationError):
            client.tail_log("/scratch/log", lines=0)

    def test_tail_run_logs_passes_path_and_limits_as_data_tokens(self) -> None:
        runner = FakeRunner(completed("===== task.out =====\ndone\n"))
        client = HpcRemoteClient(ConnectionConfig.openssh_alias("cluster"), runner)

        output = client.tail_run_logs("/scratch/run 7", lines=40, files=3)

        self.assertIn("done", output)
        tokens = shlex.split(runner.calls[0][0][-1])
        self.assertEqual(tokens[0:2], ["sh", "-c"])
        self.assertEqual(tokens[-4:], ["grim-tail-logs", "/scratch/run 7", "40", "3"])
        with self.assertRaises(ConfigurationError):
            client.tail_run_logs("/scratch/run; rm -rf /", lines=40)

    def test_malformed_scheduler_output_is_not_silently_accepted(self) -> None:
        client = HpcRemoteClient(
            ConnectionConfig.openssh_alias("cluster"),
            FakeRunner(completed("123|RUNNING|missing-fields\n")),
        )
        with self.assertRaisesRegex(ProtocolError, "malformed row"):
            client.query_squeue(["123"])


if __name__ == "__main__":
    unittest.main()
