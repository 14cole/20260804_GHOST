"""Focused contracts for collision-safe GHOST desktop output planning."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    import solver_tab
except (ImportError, RuntimeError) as exc:  # GUI dependency is optional in lean CI.
    solver_tab = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(solver_tab is None, f"GHOST GUI dependencies unavailable: {_IMPORT_ERROR}")
class SolverOutputSafetyTests(unittest.TestCase):
    def test_density_solver_honors_pre_set_abort_before_geometry_work(self) -> None:
        abort = threading.Event()
        abort.set()
        with self.assertRaisesRegex(InterruptedError, "canceled by user"):
            solver_tab.compute_boundary_densities(
                {},
                frequency_ghz=1.0,
                elevation_deg=0.0,
                polarization="TE",
                abort_event=abort,
            )

    def test_solve_worker_routes_interrupted_error_to_canceled_signal(self) -> None:
        worker = solver_tab._SolveWorker(
            snapshot={},
            source_path="",
            base_dir="",
            frequencies=[1.0],
            elevations=[0.0],
            units="meters",
            quality_thresholds={},
            abort_event=threading.Event(),
        )
        canceled = []
        errors = []
        finished = []
        worker.canceled.connect(canceled.append)
        worker.error.connect(errors.append)
        worker.finished.connect(lambda *args: finished.append(args))
        with mock.patch.object(
            worker, "_run_2d", side_effect=InterruptedError("operator canceled")
        ):
            worker.run()

        self.assertEqual(canceled, ["operator canceled"])
        self.assertEqual(errors, [])
        self.assertEqual(finished, [])

    def test_solve_worker_honors_cancel_requested_during_final_solver_step(self) -> None:
        abort = threading.Event()
        worker = solver_tab._SolveWorker(
            snapshot={},
            source_path="",
            base_dir="",
            frequencies=[1.0],
            elevations=[0.0],
            units="meters",
            quality_thresholds={},
            abort_event=abort,
        )
        canceled = []
        finished = []
        worker.canceled.connect(canceled.append)
        worker.finished.connect(lambda *args: finished.append(args))

        def finish_after_cancel(*_args):
            abort.set()
            return {"samples": [], "metadata": {}}

        with mock.patch.object(worker, "_run_2d", side_effect=finish_after_cancel):
            worker.run()

        self.assertEqual(canceled, ["Solve canceled by user."])
        self.assertEqual(finished, [])

    def test_boundary_worker_returns_both_channels_without_writing_files(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            abort = threading.Event()
            worker = solver_tab._BoundaryDensityWorker(
                run_id=7, snapshot={"segment_count": 1}, source_path="body.geo",
                base_dir=folder, frequency_ghz=3.0, elevation_deg=12.0, units="meters",
                snapshot_sha256="snapshot-id", input_sha256={}, abort_event=abort,
            )
            finished, errors = [], []
            worker.finished.connect(lambda *args: finished.append(args))
            worker.error.connect(lambda *args: errors.append(args))
            def fake_density(**kwargs):
                return {"element_count": 2, "formulation": kwargs["polarization"]}
            with mock.patch.object(solver_tab, "compute_boundary_densities", side_effect=fake_density) as compute:
                worker.run()
            self.assertEqual(errors, [])
            self.assertEqual(compute.call_count, 2)
            self.assertEqual([call.kwargs["polarization"] for call in compute.call_args_list], ["TE", "TM"])
            self.assertEqual(len(finished), 1)
            run_id, payload = finished[0]
            self.assertEqual(run_id, 7)
            self.assertEqual(payload["polarizations"], ["VV", "HH"])
            self.assertEqual(payload["geometry_snapshot_sha256"], "snapshot-id")
            self.assertEqual(payload["channels"]["VV"]["formulation"], "TE")
            self.assertEqual(payload["channels"]["HH"]["formulation"], "TM")
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_boundary_worker_cancel_between_channels_leaves_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "densities.json"
            abort = threading.Event()
            worker = solver_tab._BoundaryDensityWorker(
                run_id=4,
                snapshot={"segment_count": 1},
                source_path="",
                base_dir=folder,
                frequency_ghz=1.0,
                elevation_deg=0.0,
                units="meters",
                snapshot_sha256="snapshot-id",
                input_sha256={},
                abort_event=abort,
            )
            canceled = []
            worker.canceled.connect(lambda *args: canceled.append(args))

            def cancel_after_first(**_kwargs):
                abort.set()
                return {"element_count": 1, "formulation": "test"}

            with mock.patch.object(
                solver_tab,
                "compute_boundary_densities",
                side_effect=cancel_after_first,
            ) as compute:
                worker.run()

            self.assertEqual(compute.call_count, 1)
            self.assertEqual(canceled[0][0], 4)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(folder).glob(".*.tmp")), [])


    def test_density_input_verification_rejects_changed_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "body.geo"
            source.write_text("original", encoding="utf-8")
            identities = {
                str(source.resolve()): solver_tab._stable_sha256(str(source))
            }
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Input changed"):
                solver_tab._verify_input_sha256(identities)

    def test_monostatic_and_bistatic_paths_are_known_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mono = solver_tab._planned_export_paths(
                {"scattering_mode": "monostatic"},
                str(root / "body"),
            )
            self.assertEqual(mono, [str((root / "body.grim").resolve())])

            bistatic = solver_tab._planned_export_paths(
                {
                    "scattering_mode": "bistatic",
                    "samples": [
                        {"theta_inc_deg": 10.0},
                        {"theta_inc_deg": -2.5},
                        {"theta_inc_deg": 10.0},
                    ],
                },
                str(root / "field.grim"),
            )
            self.assertEqual(
                bistatic,
                [
                    str((root / "field_inc_m2p5.grim").resolve()),
                    str((root / "field_inc_10.grim").resolve()),
                ],
            )

    def test_relative_explicit_output_is_resolved_beside_geometry(self) -> None:
        class Resolver:
            _documents_output_dir = staticmethod(lambda: Path("unused"))

        with tempfile.TemporaryDirectory() as folder:
            geometry = Path(folder) / "geometry" / "body.geo"
            geometry.parent.mkdir()
            geometry.write_text("title: body\n", encoding="utf-8")
            resolved = solver_tab.SolverTab._resolve_output_path(
                Resolver(),
                "results/body_response",
                str(geometry),
            )
            self.assertEqual(
                Path(resolved),
                geometry.parent / "results" / "body_response.grim",
            )


if __name__ == "__main__":
    unittest.main()
