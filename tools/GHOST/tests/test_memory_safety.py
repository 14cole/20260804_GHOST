#!/usr/bin/env python3
"""Desktop and scheduler memory-gate regressions for GHOST."""

from pathlib import Path
from types import SimpleNamespace
import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "Backend"
sys.path.insert(0, str(BACKEND))

import bor_solver  # noqa: E402
import rcs_solver as rcs  # noqa: E402


GIB = 1024 ** 3


class AvailableMemoryDetectionTests(unittest.TestCase):
    def test_tightest_host_scheduler_and_cgroup_headroom_wins(self):
        with (
            mock.patch.object(rcs, "_psutil_available_bytes", return_value=12 * GIB),
            mock.patch.object(rcs, "_slurm_available_bytes", return_value=8 * GIB),
            mock.patch.object(rcs, "_cgroup_available_bytes", return_value=6 * GIB),
        ):
            self.assertEqual(rcs._detect_available_gb(), 6.0)

    def test_native_fallback_is_used_when_psutil_is_unavailable(self):
        with (
            mock.patch.object(rcs, "_psutil_available_bytes", return_value=None),
            mock.patch.object(rcs, "_windows_available_bytes", return_value=7 * GIB),
            mock.patch.object(rcs, "_macos_available_bytes") as macos,
            mock.patch.object(rcs, "_posix_available_bytes") as posix,
            mock.patch.object(rcs, "_slurm_available_bytes", return_value=None),
            mock.patch.object(rcs, "_cgroup_available_bytes", return_value=None),
        ):
            self.assertEqual(rcs._detect_available_gb(), 7.0)
        macos.assert_not_called()
        posix.assert_not_called()

    def test_windows_global_memory_status_fallback_reports_available_physical(self):
        class _Kernel32:
            @staticmethod
            def GlobalMemoryStatusEx(pointer):
                pointer._obj.ullAvailPhys = 5 * GIB
                return 1

        fake_windll = SimpleNamespace(kernel32=_Kernel32())
        with (
            mock.patch.object(rcs.os, "name", "nt"),
            mock.patch.object(rcs.ctypes, "windll", fake_windll, create=True),
        ):
            self.assertEqual(rcs._windows_available_bytes(), 5 * GIB)

    def test_macos_vm_stat_fallback_is_conservative(self):
        output = (
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
            "Pages free:                              1000.\n"
            "Pages inactive:                          2000.\n"
            "Pages speculative:                       9000.\n"
        )
        completed = SimpleNamespace(stdout=output)
        with (
            mock.patch.object(rcs.sys, "platform", "darwin"),
            mock.patch.object(rcs.subprocess, "run", return_value=completed),
        ):
            self.assertEqual(rcs._macos_available_bytes(), 3000 * 4096)

    def test_cgroup_limit_without_readable_usage_fails_closed(self):
        with mock.patch.object(
            rcs, "_read_cgroup_int", side_effect=[4 * GIB, None]
        ):
            self.assertEqual(rcs._cgroup_available_bytes(), 0)

    def test_slurm_mem_per_cpu_defaults_to_one_cpu_per_task(self):
        scheduler = {
            key: value
            for key, value in os.environ.items()
            if key not in {
                "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU",
                "SLURM_CPUS_PER_TASK",
            }
        }
        scheduler["SLURM_MEM_PER_CPU"] = "2048"
        with (
            mock.patch.dict(os.environ, scheduler, clear=True),
            mock.patch.object(rcs, "_process_rss_bytes", return_value=0),
        ):
            self.assertEqual(rcs._slurm_available_bytes(), 2 * GIB)

    def test_unknown_memory_has_no_historical_32_gb_floor(self):
        with (
            mock.patch.dict(os.environ, {"GHOST_MAX_SOLVE_GB": ""}, clear=False),
            mock.patch.object(rcs, "_detect_available_gb", return_value=0.0),
        ):
            self.assertEqual(rcs._solve_memory_limit_gb(), 0.0)

    def test_explicit_limit_remains_available_for_confirmed_allocations(self):
        with mock.patch.dict(
            os.environ, {"GHOST_MAX_SOLVE_GB": "48"}, clear=False
        ):
            self.assertEqual(rcs._solve_memory_limit_gb(), 48.0)

    def test_nonfinite_explicit_limit_cannot_disable_safety_gate(self):
        with (
            mock.patch.dict(
                os.environ, {"GHOST_MAX_SOLVE_GB": "inf"}, clear=False
            ),
            mock.patch.object(rcs, "_detect_available_gb", return_value=10.0),
        ):
            self.assertEqual(rcs._solve_memory_limit_gb(), 9.0)

    def test_error_states_required_available_and_safe_limit(self):
        with mock.patch.object(rcs, "_detect_available_gb", return_value=3.25):
            message = rcs._memory_gate_message(
                7.5,
                2.925,
                "The test solve",
                "Planned system: 1000 DOFs.",
            )
        self.assertIn("requires an estimated 7.50 GB", message)
        self.assertIn("3.25 GB is currently available", message)
        self.assertIn("safe allocation limit is 2.92 GB", message)
        self.assertIn("GHOST_MAX_SOLVE_GB", message)


class BorMemoryGateTests(unittest.TestCase):
    def test_dense_peak_estimate_multiplies_actual_worker_concurrency(self):
        serial = bor_solver.estimate_bor_dense_peak_gb(
            1200, 18, workers=1, mode_tasks=8
        )
        parallel = bor_solver.estimate_bor_dense_peak_gb(
            1200, 18, workers=3, mode_tasks=8
        )
        capped = bor_solver.estimate_bor_dense_peak_gb(
            1200, 18, workers=20, mode_tasks=2
        )

        self.assertAlmostEqual(parallel, 3.0 * serial)
        self.assertAlmostEqual(capped, 2.0 * serial)

    def test_bor_uses_process_limit_instead_of_fixed_32_gb_gate(self):
        points = np.asarray([[0.0, 1.0], [0.5, 0.0], [0.0, -1.0]])
        with (
            mock.patch.object(bor_solver, "estimate_bor_table_gb", return_value=2.0),
            mock.patch.object(bor_solver, "_solve_memory_limit_gb", return_value=1.0),
            mock.patch.object(rcs, "_detect_available_gb", return_value=1.2),
            mock.patch.object(
                bor_solver.BorPecSolver,
                "prepare_operators",
                side_effect=AssertionError("far assembly started"),
            ),
        ):
            with self.assertRaisesRegex(
                MemoryError, "requires an estimated 2.00 GB"
            ):
                bor_solver.solve_bor(
                    points,
                    1.0e9,
                    [0.0],
                    formulation="efie",
                    n_modes=1,
                    assembly="tables",
                    table_precision="double",
                )

    def test_direct_bor_combines_resident_tables_with_dense_solve_peak(self):
        points = np.asarray([[0.0, 1.0], [0.5, 0.0], [0.0, -1.0]])
        with (
            mock.patch.object(bor_solver, "estimate_bor_table_gb", return_value=2.0),
            mock.patch.object(
                bor_solver, "estimate_bor_dense_peak_gb", return_value=0.25
            ),
            mock.patch.object(bor_solver, "_solve_memory_limit_gb", return_value=2.1),
            mock.patch.object(rcs, "_detect_available_gb", return_value=2.5),
            mock.patch.object(
                bor_solver.BorPecSolver,
                "prepare_operators",
                side_effect=AssertionError("far assembly started"),
            ),
        ):
            with self.assertRaisesRegex(
                MemoryError, "requires an estimated 2.25 GB"
            ):
                bor_solver.solve_bor(
                    points,
                    1.0e9,
                    [0.0],
                    formulation="efie",
                    n_modes=1,
                    assembly="tables",
                    table_precision="double",
                )

    def test_dielectric_dense_gate_runs_before_operator_preparation(self):
        points = np.asarray([[0.0, 1.0], [0.5, 0.0], [0.0, -1.0]])
        with (
            mock.patch.object(
                bor_solver, "estimate_bor_dense_peak_gb", return_value=1.5
            ),
            mock.patch.object(bor_solver, "_solve_memory_limit_gb", return_value=1.0),
            mock.patch.object(rcs, "_detect_available_gb", return_value=1.2),
            mock.patch.object(
                bor_solver.BorPecSolver,
                "prepare_operators",
                side_effect=AssertionError("non-PEC operator preparation started"),
            ) as prepare,
        ):
            with self.assertRaisesRegex(
                MemoryError, "dielectric BoR solve.*dense peak 1.50 GB"
            ):
                bor_solver.solve_bor_dielectric(
                    points,
                    1.0e9,
                    [0.0],
                    eps_r=2.5 - 0.05j,
                    n_modes=1,
                    workers=2,
                )
        prepare.assert_not_called()

    def test_dielectric_combines_operator_tables_with_dense_peak(self):
        points = np.asarray([[0.0, 1.0], [0.5, 0.0], [0.0, -1.0]])
        with (
            mock.patch.object(
                bor_solver,
                "estimate_bor_operator_storage_gb",
                return_value=2.0,
            ) as operator_estimate,
            mock.patch.object(
                bor_solver, "estimate_bor_dense_peak_gb", return_value=0.25
            ),
            mock.patch.object(bor_solver, "_solve_memory_limit_gb", return_value=2.1),
            mock.patch.object(rcs, "_detect_available_gb", return_value=2.5),
            mock.patch.object(
                bor_solver.BorPecSolver,
                "prepare_operators",
                side_effect=AssertionError("non-PEC operator preparation started"),
            ) as prepare,
        ):
            with self.assertRaisesRegex(
                MemoryError, "requires an estimated 2.25 GB"
            ):
                bor_solver.solve_bor_dielectric(
                    points,
                    1.0e9,
                    [0.0],
                    eps_r=2.5 - 0.05j,
                    n_modes=1,
                    workers=2,
                )
        operator_estimate.assert_called_once()
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
