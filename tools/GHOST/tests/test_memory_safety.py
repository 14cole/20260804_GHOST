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
import bor_streaming  # noqa: E402
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
            mock.patch.object(rcs, "_posix_available_bytes") as posix,
            mock.patch.object(rcs, "_slurm_available_bytes", return_value=None),
            mock.patch.object(rcs, "_cgroup_available_bytes", return_value=None),
        ):
            self.assertEqual(rcs._detect_available_gb(), 7.0)
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
    def test_streaming_tile_shape_honors_conservative_live_set_budget(self):
        budget_gb = 0.25
        workers = 3
        n_xi = 8192
        gauss_order = 4
        tile_elements, source_columns = bor_streaming._streaming_tile_shape(
            n_elements=600,
            gauss_order=gauss_order,
            point_count=2400,
            n_xi=n_xi,
            tile_budget_gb=budget_gb,
            workers=workers,
        )
        accounted_bytes = (
            tile_elements
            * gauss_order
            * source_columns
            * n_xi
            * bor_streaming.BOR_STREAM_TILE_BYTES_PER_SAMPLE
            * workers
        )
        self.assertLessEqual(accounted_bytes, budget_gb * 1.0e9)
        self.assertEqual(
            bor_streaming.BOR_STREAM_TILE_BYTES_PER_SAMPLE,
            256.0,
            "the fallback kernel's 200+ byte live set needs FFT/allocator slack",
        )

    def test_streaming_tile_shape_caps_rows_to_real_element_count(self):
        tile_elements, source_columns = bor_streaming._streaming_tile_shape(
            n_elements=7,
            gauss_order=4,
            point_count=28,
            n_xi=256,
            tile_budget_gb=1.0,
            workers=1,
        )
        self.assertEqual(tile_elements, 7)
        self.assertEqual(source_columns, 28)

    def test_streaming_sampling_workers_respect_one_column_floor(self):
        budget_gb = 1.0
        workers = bor_streaming._streaming_worker_count(
            gauss_order=4,
            n_xi=8192,
            tile_budget_gb=budget_gb,
            workers=256,
        )
        minimum_live_bytes = (
            workers
            * 4
            * 8192
            * bor_streaming.BOR_STREAM_TILE_BYTES_PER_SAMPLE
        )
        self.assertLess(workers, 256)
        self.assertLessEqual(minimum_live_bytes, budget_gb * 1.0e9)

    def test_streaming_rejects_budget_below_one_tile_floor(self):
        minimum_gb = (
            4
            * 8192
            * bor_streaming.BOR_STREAM_TILE_BYTES_PER_SAMPLE
            / 1.0e9
        )
        with self.assertRaisesRegex(ValueError, "one-column minimum"):
            bor_streaming._streaming_worker_count(
                gauss_order=4,
                n_xi=8192,
                tile_budget_gb=0.5 * minimum_gb,
                workers=1,
            )

    def test_streaming_mode_block_plan_matches_runtime_alignment_and_peak(self):
        mode_block, retained_gb, effective_workers = (
            bor_streaming.plan_streaming_mode_block(
                n_elems=1000,
                m_max=100,
                formulation="cfie",
                has_ibc=False,
                single_blocks=False,
                stream_budget_gb=8.0,
                workers=64,
            )
        )
        runtime_block = bor_streaming._aligned_stream_mode_block(
            100, mode_block, effective_workers
        )
        self.assertEqual(mode_block, runtime_block)
        self.assertEqual(mode_block, 28)
        self.assertEqual(effective_workers, 28)
        self.assertAlmostEqual(
            retained_gb,
            bor_streaming.estimate_streaming_block_gb(
                1000, 100, mode_block, "cfie", False, False
            ),
        )
        self.assertLessEqual(retained_gb, 8.0)

    def test_streaming_mode_block_rejects_impossible_retained_budget(self):
        minimum = bor_streaming.estimate_streaming_block_gb(
            1000, 100, 1, "cfie", False, False
        )
        with self.assertRaisesRegex(ValueError, "one-mode retained minimum"):
            bor_streaming.plan_streaming_mode_block(
                n_elems=1000,
                m_max=100,
                formulation="cfie",
                has_ibc=False,
                single_blocks=False,
                stream_budget_gb=0.5 * minimum,
                workers=64,
            )

    def test_streaming_runtime_uses_budget_capped_outer_workers(self):
        points = np.asarray([[0.0, 1.0], [0.5, 0.0], [0.0, -1.0]])
        retained_two = bor_streaming.estimate_streaming_block_gb(
            2, 4, 2, "cfie", False, False
        )
        retained_three = bor_streaming.estimate_streaming_block_gb(
            2, 4, 3, "cfie", False, False
        )
        budget = 0.5 * (retained_two + retained_three)
        captured = {}

        def fake_mode_sweep(*args, **kwargs):
            captured["workers"] = kwargs["workers"]
            captured["assembly_peak_gb"] = kwargs["assembly_peak_gb"]
            kwargs["prepare"](4)
            return (
                np.zeros((2, 1), dtype=np.complex128),
                0,
                {"mode_converged": True},
            )

        with (
            mock.patch.object(bor_solver, "_mode_sweep", side_effect=fake_mode_sweep),
            mock.patch.object(
                bor_solver.BorPecSolver, "enable_streaming"
            ) as enable_streaming,
            mock.patch.object(bor_solver.BorPecSolver, "prepare_operators"),
        ):
            bor_solver.solve_bor(
                points,
                1.0e9,
                [90.0],
                formulation="cfie",
                n_modes=4,
                workers=4,
                assembly="streaming",
                table_precision="double",
                stream_budget_gb=budget,
            )

        self.assertEqual(captured["workers"], 2)
        self.assertLessEqual(
            captured["assembly_peak_gb"]
            - bor_streaming.BOR_STREAM_TILE_BUDGET_GB,
            budget,
        )
        self.assertEqual(enable_streaming.call_args.kwargs["workers"], 2)
        self.assertEqual(enable_streaming.call_args.kwargs["mode_block"], 2)

    def test_streaming_estimator_counts_all_nine_efie_primitives(self):
        n_elems = 24
        m_max = 12
        nodes = float(n_elems + 1)
        expected_efie = 9.0 * nodes * nodes * (m_max + 2) * 16.0 / 1.0e9
        self.assertAlmostEqual(
            bor_streaming.estimate_streaming_gb(
                n_elems, m_max, formulation="efie"
            ),
            expected_efie,
        )

    def test_total_peak_adds_the_same_scheduler_margin_once(self):
        self.assertAlmostEqual(
            bor_solver.estimate_bor_total_peak_gb(7.0, 0.25),
            9.2,
        )

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
                MemoryError, "requires an estimated 8.90 GB"
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
                MemoryError, "requires an estimated 9.20 GB"
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
                MemoryError, "requires an estimated 3.20 GB"
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
