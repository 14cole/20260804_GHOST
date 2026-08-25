from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from ibc.batch import (
    MAX_IBC_BATCH_TOTAL_POINTS,
    export_pec_ibc_thickness_batch,
    ibc_batch_frequency_count,
    plan_ibc_thickness_batch,
    validate_ibc_batch_workload,
)
from ibc.compute import (
    INCH_TO_M,
    LoadedLayer,
    MaterialTable,
    compute_stack_impedance_many,
)
from ibc.io import IMPEDANCE_HEADER, HZ_PER_GHZ, write_impedance_batch


class IbcBatchPlanningTests(unittest.TestCase):
    def test_workload_is_bounded_without_allocating_frequency_rows(self) -> None:
        count = ibc_batch_frequency_count(1.0, 18.0, 0.1)
        self.assertEqual(count, 171)
        self.assertEqual(validate_ibc_batch_workload(4, count), 684)
        with self.assertRaisesRegex(ValueError, "safe desktop limit"):
            validate_ibc_batch_workload(
                1000,
                MAX_IBC_BATCH_TOTAL_POINTS // 1000 + 1,
            )

    def test_mil_sweep_has_exact_values_and_canonical_names(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_ibc_thickness_batch(
                folder, "coating", "15", "30", "5", "mil"
            )

        self.assertEqual(
            [item.path.name for item in plan],
            [
                "coating_15mil.csv",
                "coating_20mil.csv",
                "coating_25mil.csv",
                "coating_30mil.csv",
            ],
        )
        self.assertEqual(
            [item.thickness_in for item in plan],
            [0.015, 0.020, 0.025, 0.030],
        )

    def test_decimal_names_are_stable_and_stop_is_not_overshot(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_ibc_thickness_batch(
                folder, "skin", "15.5", "21", "2.5", "mil"
            )
        self.assertEqual(
            [item.path.name for item in plan],
            ["skin_15p5mil.csv", "skin_18mil.csv", "skin_20p5mil.csv"],
        )

    def test_inch_and_millimeter_units_have_exact_names_and_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            inch_plan = plan_ibc_thickness_batch(
                folder, "inch", "0.015", "0.025", "0.01", "in"
            )
            mm_plan = plan_ibc_thickness_batch(
                folder, "metric", "0.5", "1.0", "0.5", "mm"
            )

        self.assertEqual(
            [item.path.name for item in inch_plan],
            ["inch_0p015in.csv", "inch_0p025in.csv"],
        )
        self.assertEqual(
            [item.thickness_in for item in inch_plan],
            [0.015, 0.025],
        )
        self.assertEqual(
            [item.path.name for item in mm_plan],
            ["metric_0p5mm.csv", "metric_1mm.csv"],
        )
        self.assertAlmostEqual(mm_plan[0].thickness_in, 0.5 / 25.4)
        self.assertAlmostEqual(mm_plan[1].thickness_in, 1.0 / 25.4)

    def test_invalid_plan_is_rejected_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "prefix"):
                plan_ibc_thickness_batch(
                    folder, "bad/name.csv", "15", "30", "1", "mil"
                )
            with self.assertRaisesRegex(ValueError, "limit is 1000"):
                plan_ibc_thickness_batch(
                    folder, "ibc", "1", "2000", "1", "mil"
                )
            with self.assertRaisesRegex(ValueError, "stop must be >= start"):
                plan_ibc_thickness_batch(
                    folder, "ibc", "30", "15", "1", "mil"
                )


class IbcBatchExportTests(unittest.TestCase):
    @staticmethod
    def _material(value: complex) -> MaterialTable:
        return MaterialTable(
            freq_ghz=[1.0, 2.0, 3.0],
            eps_r=[value, value, value],
            mu_r=[1.0 + 0.0j] * 3,
        )

    def test_batch_matches_authoritative_stack_compute_and_preserves_stack(
        self,
    ) -> None:
        top = LoadedLayer(
            thickness_m=0.050 * INCH_TO_M,
            anisotropic=False,
            polarization_deg=0.0,
            table_0deg=self._material(2.5 - 0.1j),
            table_90deg=None,
        )
        selected = LoadedLayer(
            thickness_m=0.100 * INCH_TO_M,
            anisotropic=False,
            polarization_deg=0.0,
            table_0deg=self._material(4.0 - 0.2j),
            table_90deg=None,
        )
        layers = [top, selected]
        original = copy.deepcopy(layers)
        frequencies = [1.0, 2.0, 3.0]

        with tempfile.TemporaryDirectory() as folder:
            plan = plan_ibc_thickness_batch(
                folder, "ibc", "15", "30", "15", "mil"
            )
            count = export_pec_ibc_thickness_batch(
                plan, layers, 1, frequencies
            )
            self.assertEqual(count, 2)

            for item in plan:
                expected_stack = list(layers)
                expected_stack[1] = replace(
                    selected, thickness_m=item.thickness_in * INCH_TO_M
                )
                expected = compute_stack_impedance_many(
                    frequencies, expected_stack, "pec"
                )
                lines = item.path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines[0], IMPEDANCE_HEADER)
                rows = [
                    [float(value) for value in line.split(",")]
                    for line in lines[1:]
                ]
                self.assertEqual(
                    [row[0] for row in rows],
                    [f * HZ_PER_GHZ for f in frequencies],
                )
                for row, impedance in zip(rows, expected):
                    self.assertAlmostEqual(
                        row[1],
                        impedance.real,
                        delta=max(1e-9, abs(impedance.real) * 1e-11),
                    )
                    self.assertAlmostEqual(
                        row[2],
                        impedance.imag,
                        delta=max(1e-9, abs(impedance.imag) * 1e-11),
                    )

        self.assertEqual(layers, original)

    def test_compute_failure_publishes_no_partial_batch(self) -> None:
        layer = LoadedLayer(
            thickness_m=0.050 * INCH_TO_M,
            anisotropic=False,
            polarization_deg=0.0,
            table_0deg=self._material(2.5 - 0.1j),
            table_90deg=None,
        )
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_ibc_thickness_batch(
                folder, "ibc", "15", "20", "5", "mil"
            )
            with mock.patch(
                "ibc.batch.compute_stack_impedance_many",
                side_effect=[[100.0 + 10.0j], RuntimeError("second compute failed")],
            ):
                with self.assertRaisesRegex(RuntimeError, "second compute failed"):
                    export_pec_ibc_thickness_batch(plan, [layer], 0, [1.0])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_work_cap_fails_before_compute_or_output(self) -> None:
        layer = LoadedLayer(
            thickness_m=0.050 * INCH_TO_M,
            anisotropic=False,
            polarization_deg=0.0,
            table_0deg=self._material(2.5 - 0.1j),
            table_90deg=None,
        )
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_ibc_thickness_batch(
                folder, "ibc", "15", "20", "5", "mil"
            )
            with mock.patch(
                "ibc.batch.compute_stack_impedance_many"
            ) as compute:
                with self.assertRaisesRegex(ValueError, "4 total frequency rows"):
                    export_pec_ibc_thickness_batch(
                        plan,
                        [layer],
                        0,
                        [1.0, 2.0],
                        max_total_points=3,
                    )
            compute.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_each_result_is_staged_before_the_next_is_computed(self) -> None:
        layer = LoadedLayer(
            thickness_m=0.050 * INCH_TO_M,
            anisotropic=False,
            polarization_deg=0.0,
            table_0deg=self._material(2.5 - 0.1j),
            table_90deg=None,
        )
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_ibc_thickness_batch(
                folder, "ibc", "15", "25", "5", "mil"
            )
            from ibc import io as freddy_io

            events: list[str] = []
            original_stage = freddy_io._stage_text_file

            def compute(_frequencies, _stack, _backing):
                events.append("compute")
                return [100.0 + 10.0j]

            def stage(path, writer):
                events.append("stage")
                return original_stage(path, writer)

            with mock.patch(
                "ibc.batch.compute_stack_impedance_many", side_effect=compute
            ), mock.patch.object(
                freddy_io, "_stage_text_file", side_effect=stage
            ):
                export_pec_ibc_thickness_batch(plan, [layer], 0, [1.0])

            self.assertEqual(
                events,
                ["compute", "stage", "compute", "stage", "compute", "stage"],
            )

    def test_stage_failure_preserves_every_existing_batch_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "ibc_15mil.csv"
            second = Path(folder) / "ibc_20mil.csv"
            first.write_text("old first\n", encoding="utf-8")
            second.write_text("old second\n", encoding="utf-8")
            outputs = [
                (first, [(1.0, 100.0, 10.0)]),
                (second, [(1.0, 110.0, 11.0)]),
            ]
            from ibc import io as freddy_io

            original_stage = freddy_io._stage_text_file
            calls = 0

            def fail_second(path, writer):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("second stage failed")
                return original_stage(path, writer)

            with mock.patch.object(
                freddy_io, "_stage_text_file", side_effect=fail_second
            ):
                with self.assertRaisesRegex(OSError, "second stage failed"):
                    write_impedance_batch(outputs)

            self.assertEqual(first.read_text(encoding="utf-8"), "old first\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "old second\n")
            self.assertEqual(set(Path(folder).iterdir()), {first, second})

    def test_partial_publication_failure_rolls_back_complete_batch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "ibc_15mil.csv"
            second = Path(folder) / "ibc_20mil.csv"
            first.write_text("old first\n", encoding="utf-8")
            second.write_text("old second\n", encoding="utf-8")
            outputs = [
                (first, [(1.0, 100.0, 10.0)]),
                (second, [(1.0, 110.0, 11.0)]),
            ]
            from ibc import io as freddy_io

            original_replace = freddy_io.os.replace
            calls = 0

            def fail_second_publication(source, destination):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise OSError("second publication failed")
                return original_replace(source, destination)

            with mock.patch.object(
                freddy_io.os, "replace", side_effect=fail_second_publication
            ):
                with self.assertRaisesRegex(OSError, "second publication failed"):
                    write_impedance_batch(outputs)

            self.assertEqual(first.read_text(encoding="utf-8"), "old first\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "old second\n")
            self.assertEqual(set(Path(folder).iterdir()), {first, second})


if __name__ == "__main__":
    unittest.main()
