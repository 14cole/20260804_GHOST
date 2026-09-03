"""Retired methods must not fall through to an unrequested dense allocation."""

from pathlib import Path
import sys
import unittest
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1] / "Backend"
sys.path.insert(0, str(BACKEND))

import rcs_solver as rcs
from test_2d_capability_acceptance import _circle


class DirectSolverMethodTests(unittest.TestCase):
    def test_all_supported_methods_enforce_dense_memory_gate_before_assembly(self):
        snapshot = {
            "segments": [_circle("pec", 0.08, 16, 2)],
            "ibcs": [], "dielectrics": [],
        }
        for method in ("auto", "direct"):
            with self.subTest(method=method):
                with (
                    mock.patch.object(rcs, "_solve_memory_limit_gb", return_value=1.0e-12),
                    mock.patch.object(rcs, "_assemble_linear_operator_matrices") as assemble,
                ):
                    with self.assertRaises(MemoryError):
                        rcs.solve_monostatic_rcs_2d_single_polarization(
                            snapshot, [0.1], [0.0], "TE",
                            geometry_units="meters", solver_method=method,
                        )
                assemble.assert_not_called()

    def test_removed_method_is_rejected_before_any_geometry_work(self):
        for method in ("fmm", " FMM ", "gmres", "unknown"):
            for solver, angles in (
                (rcs.solve_monostatic_rcs_2d_single_polarization, {"elevations_deg": [0.0]}),
                (rcs.solve_bistatic_rcs_2d_single_polarization,
                 {"incidence_angles_deg": [0.0], "observation_angles_deg": [90.0]}),
                (rcs.solve_monostatic_rcs_2d_certified_single_polarization,
                 {"elevations_deg": [0.0]}),
                (rcs.solve_bistatic_rcs_2d_certified_single_polarization,
                 {"incidence_angles_deg": [0.0], "observation_angles_deg": [90.0]}),
            ):
                with self.subTest(method=method, solver=solver.__name__):
                    with mock.patch.object(rcs, "_build_panels") as build:
                        with self.assertRaisesRegex(ValueError, "auto.*direct"):
                            solver(
                                geometry_snapshot={}, frequencies_ghz=[1.0],
                                polarization="TE", solver_method=method, **angles,
                            )
                    build.assert_not_called()

    def test_private_formulations_also_reject_retired_method(self):
        for solver in (rcs._solve_te_robin_mfie, rcs._solve_multi_region_indirect):
            with self.subTest(solver=solver.__name__):
                with self.assertRaisesRegex(ValueError, "FMM solver has been removed"):
                    solver(None, [], "TE", 1.0, [0.0], solver_method="fmm")

    def test_auto_and_direct_reach_geometry_preflight(self):
        for method in ("auto", "direct"):
            with self.subTest(method=method):
                with mock.patch.object(
                    rcs, "_material_base_dir_for_snapshot",
                    side_effect=RuntimeError("geometry preflight reached"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "geometry preflight reached"):
                        rcs.solve_monostatic_rcs_2d_single_polarization(
                            {}, [1.0], [0.0], "TE", solver_method=method,
                        )


if __name__ == "__main__":
    unittest.main()
