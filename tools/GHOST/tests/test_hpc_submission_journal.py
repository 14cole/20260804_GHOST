#!/usr/bin/env python3
"""Focused durability checks for both SLURM submission journals."""

import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

# The journal helper itself has no numerical dependency.  Keep this focused
# test runnable on a headless submission host that has not loaded NumPy yet;
# solver_quality only needs the module to exist while the driver is imported.
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")

import run_hpc_bor_monostatic  # noqa: E402
import run_hpc_monostatic  # noqa: E402


DRIVERS = (run_hpc_monostatic, run_hpc_bor_monostatic)


class SubmissionJournalDurabilityTests(unittest.TestCase):
    def test_file_fsync_precedes_atomic_replace(self) -> None:
        for driver in DRIVERS:
            with self.subTest(driver=driver.__name__), tempfile.TemporaryDirectory() as raw:
                target = Path(raw) / "submitted_jobs.json"
                document = {
                    "schema": "ghost.hpc.submitted-jobs.v1",
                    "job_ids": ["12345"],
                    "updated_utc": "2026-08-25T00:00:00Z",
                }
                events = []
                real_fsync = os.fsync
                real_replace = os.replace

                def recording_fsync(fd):
                    kind = "directory-fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-fsync"
                    events.append(kind)
                    return real_fsync(fd)

                def recording_replace(source, destination):
                    events.append("replace")
                    return real_replace(source, destination)

                with mock.patch.object(driver.os, "fsync", side_effect=recording_fsync), \
                        mock.patch.object(driver.os, "replace", side_effect=recording_replace):
                    driver._durably_publish_submitted_jobs(target, document)

                self.assertEqual(events[:2], ["file-fsync", "replace"])
                if os.name == "posix":
                    self.assertEqual(events[2:], ["directory-fsync"])
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")), document)
                self.assertFalse(target.with_suffix(".json.tmp").exists())

    def test_file_fsync_failure_prevents_publication(self) -> None:
        for driver in DRIVERS:
            with self.subTest(driver=driver.__name__), tempfile.TemporaryDirectory() as raw:
                target = Path(raw) / "submitted_jobs.json"
                with mock.patch.object(driver.os, "fsync", side_effect=OSError("disk error")), \
                        mock.patch.object(driver.os, "replace") as replace:
                    with self.assertRaisesRegex(OSError, "disk error"):
                        driver._durably_publish_submitted_jobs(target, {"job_ids": ["12"]})
                replace.assert_not_called()
                self.assertFalse(target.exists())

    @unittest.skipUnless(os.name == "posix", "directory fsync is POSIX-only")
    def test_directory_fsync_failure_is_best_effort(self) -> None:
        for driver in DRIVERS:
            with self.subTest(driver=driver.__name__), tempfile.TemporaryDirectory() as raw:
                target = Path(raw) / "submitted_jobs.json"
                real_fsync = os.fsync

                def fail_only_for_directory(fd):
                    if stat.S_ISDIR(os.fstat(fd).st_mode):
                        raise OSError("directory fsync unsupported")
                    return real_fsync(fd)

                with mock.patch.object(driver.os, "fsync", side_effect=fail_only_for_directory):
                    driver._durably_publish_submitted_jobs(target, {"job_ids": ["67890"]})

                self.assertEqual(
                    json.loads(target.read_text(encoding="utf-8")),
                    {"job_ids": ["67890"]},
                )


if __name__ == "__main__":
    unittest.main()
