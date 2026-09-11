"""Release-safety regressions promoted from the external audit."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


GHOST = Path(__file__).resolve().parents[1]
ROOT = GHOST.parents[2]
BACKEND = GHOST
GRIM = ROOT / "GRIM_Backend"
for entry in (str(BACKEND), str(GRIM)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import ghost_backend.bor.dispatch as bor_dispatch
import ghost_backend.bor.solver as bor_solver
import ghost_backend.io.viewer_bridge as grim_compat
import ghost_backend.hpc.scheduler as hpc_scheduler
from ghost_backend.geometry.io import Segment, build_geometry_text
from GRIM_Backend.datasets.api import RcsGrid  # noqa: E402


class AuditReleaseSafetyTests(unittest.TestCase):
    def test_staging_rejects_values_rejected_by_drivers(self):
        import importlib
        import ghost_backend.runs.config as driver_config
        import ghost_backend.hpc.common as hpc_common
        for name, settings in (
            ('run_local_bor', {'CFIE_ALPHA': 0}),
            ('run_local_bor', {'CFIE_ALPHA': 1}),
            ('run_local_monostatic', {'MEMORY_SAFETY': .5}),
            ('run_hpc_monostatic', {'CLAIM_STALE_SECONDS': 1}),
            ('run_hpc_monostatic', {'FREQUENCIES_GHZ': [1.0001, 1.0002]}),
        ):
            module = importlib.import_module(name)
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / 'driver.py'
                with self.assertRaises(ValueError):
                    hpc_common.configure_driver(Path(module.__file__), destination, settings)
                self.assertFalse(destination.exists())
                with self.assertRaises(ValueError):
                    driver_config.validate_settings(settings, module._CONFIG_KEYS)

    def test_bor_streaming_annotations_resolve(self):
        from typing import get_type_hints
        self.assertTrue(get_type_hints(bor_solver._plan_multisurface_assembly))
        self.assertTrue(get_type_hints(bor_solver._MultiRegionBor.enable_streaming))

    def test_bor_rejects_invalid_cfie_alpha_before_geometry_work(self) -> None:
        for value in (0.0, -0.1, 1.0, 1.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "0 < alpha < 1"):
                    bor_dispatch.solve_monostatic_rcs_bor(
                        {}, [1.0], [0.0], cfie_alpha=value
                    )

    def test_direct_bor_cfie_rejects_pure_efie_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 < alpha < 1"):
            bor_solver.solve_bor(
                {}, 1.0e9, [0.0], formulation="cfie", cfie_alpha=1.0
            )

    def test_geometry_writer_rejects_unroundtrippable_segment_names(self) -> None:
        for name in ("", "two words", "line\nbreak", "bad:name"):
            with self.subTest(name=name):
                segment = Segment(name, "2", ["2", "1", "0", "0", "0"], [0, 1], [0, 0])
                with self.assertRaisesRegex(ValueError, "Segment name"):
                    build_geometry_text("body", [segment], [], [])

    def test_sbatch_log_paths_are_quoted_for_space_containing_workspace(self) -> None:
        script = hpc_scheduler.build_sbatch_script(
            job_name="safe",
            run_dir="/cluster/My Workspace/run",
            script_path="/cluster/My Workspace/run/submit/job.sh",
            array_size=1,
            array_throttle=None,
            partition="compute",
            cpus_per_node=1,
            mem_per_node=None,
            walltime=None,
            account=None,
            qos=None,
            mail_type=None,
            mail_user=None,
            extra_sbatch=(),
            prologue=(),
            python_exe="python3",
            worker_args="worker.py",
            submission_index=2,
            blas_threads=1,
        )
        self.assertIn(
            '#SBATCH --output="/cluster/My Workspace/run/logs/sub2_%A_%a.out"',
            script,
        )
        self.assertIn(
            '#SBATCH --error="/cluster/My Workspace/run/logs/sub2_%A_%a.err"',
            script,
        )

    def test_declared_non_sigma3d_pattern_never_falls_back_to_sigma3d(self) -> None:
        grid = RcsGrid(
            [0.0], [0.0], [10.0], ["VV"],
            rcs=np.ones((1, 1, 1, 1), dtype=np.complex128),
            units={
                "frequency": "GHz",
                "rcs_log_unit": "dB",
                "rcs_linear_quantity": "power_ratio",
            },
        )
        with tempfile.TemporaryDirectory() as folder:
            path = grid.save(str(Path(folder) / "ratio.grim"))
            with self.assertRaisesRegex(ValueError, "refusing to guess sigma_3d"):
                grim_compat.load_pattern_any(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
