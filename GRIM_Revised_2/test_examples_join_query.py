"""Focused tests for the runnable join and point-query examples."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples import join_folder as join_example
from examples import query_dataset as query_example
from examples.join_folder import discover_dataset_files, join_folder
from examples.query_dataset import query_sample
from grim_dataset import RcsGrid


class JoinFolderExampleTests(unittest.TestCase):
    @staticmethod
    def _cut(azimuth: float, power: float) -> RcsGrid:
        return RcsGrid(
            [azimuth],
            [0.0],
            [10.0],
            ["VV"],
            rcs_power=np.asarray([[[[power]]]], dtype=np.float64),
            rcs_phase=np.zeros((1, 1, 1, 1), dtype=np.float64),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
        )

    def test_join_folder_is_sorted_and_excludes_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self._cut(1.0, 2.0).save(folder / "b.grim")
            self._cut(0.0, 1.0).save(folder / "a.grim")
            output = folder / "joined.grim"

            joined, written, inputs = join_folder(folder, output)

            self.assertEqual([path.name for path in inputs], ["a.grim", "b.grim"])
            self.assertEqual(written, output.resolve())
            np.testing.assert_allclose(joined.azimuths, [0.0, 1.0])
            np.testing.assert_allclose(joined.rcs_power[:, 0, 0, 0], [1.0, 2.0])

            # A repeat run must not consume its own prior result.
            repeated, _, repeated_inputs = join_folder(
                folder, output, overwrite=True
            )
            self.assertEqual(
                [path.name for path in repeated_inputs], ["a.grim", "b.grim"]
            )
            np.testing.assert_allclose(repeated.rcs_power, joined.rcs_power)

    def test_strict_join_rejects_conflicting_finite_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self._cut(0.0, 1.0).save(folder / "a.grim")
            self._cut(0.0, 2.0).save(folder / "b.grim")
            with self.assertRaisesRegex(ValueError, "conflicting finite samples"):
                join_folder(folder, folder / "joined.grim")

    def test_discovery_accepts_supported_extensions_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self._cut(0.0, 1.0).save(folder / "cut.grim")
            (folder / "notes.md").write_text("not a dataset", encoding="utf-8")
            self.assertEqual(
                [path.name for path in discover_dataset_files(folder)], ["cut.grim"]
            )

    def test_main_uses_editable_configuration_constants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self._cut(0.0, 1.0).save(folder / "a.grim")
            self._cut(1.0, 2.0).save(folder / "b.grim")
            output = folder / "configured.grim"

            with patch.multiple(
                join_example,
                INPUT_FOLDER=folder,
                OUTPUT_FILE=output,
                FILE_PATTERN="*.grim",
                SEARCH_SUBFOLDERS=False,
                PARALLEL_LOADERS=1,
                OVERLAP_POLICY="error",
                COORDINATE_TOLERANCE=1.0e-6,
                MAX_OUTPUT_GIB=None,
                OVERWRITE_OUTPUT=False,
            ):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    status = join_example.main()

            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            self.assertIn("Loaded 2 dataset(s)", stdout.getvalue())
            np.testing.assert_allclose(
                RcsGrid.load(output).rcs_power[:, 0, 0, 0], [1.0, 2.0]
            )


class QueryDatasetExampleTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> RcsGrid:
        shape = (2, 1, 2, 2)
        power = np.ones(shape, dtype=np.float64)
        phase = np.zeros(shape, dtype=np.float64)
        power[1, 0, 1, 0] = 4.0
        phase[1, 0, 1, 0] = math.radians(-10.0)
        return RcsGrid(
            [0.0, math.pi / 2.0],
            [0.0],
            [1.0e9, 2.0e9],
            ["VV", "HH"],
            rcs_power=power,
            rcs_phase=phase,
            source_path="unit-aware.grim",
            units={
                "azimuth": "rad",
                "elevation": "rad",
                "frequency": "Hz",
                "phase_wrap": "0_360",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
        )

    def test_query_converts_units_returns_indices_and_all_sample_forms(self) -> None:
        result = query_sample(
            self._dataset(),
            azimuth=90.0,
            elevation=0.0,
            frequency=2000.0,
            polarization="vv",
            angle_unit="deg",
            frequency_unit="MHz",
        )

        self.assertEqual(
            result["indices"],
            {"azimuth": 1, "elevation": 0, "frequency": 1, "polarization": 0},
        )
        self.assertEqual(result["matches"]["azimuth"]["native_unit"], "rad")
        self.assertEqual(result["matches"]["frequency"]["native_unit"], "Hz")
        sample = result["sample"]
        self.assertAlmostEqual(sample["linear_power"], 4.0)
        self.assertAlmostEqual(sample["field_magnitude"], 2.0)
        self.assertAlmostEqual(sample["display_magnitude_db"], 10.0 * math.log10(4.0))
        self.assertEqual(sample["display_magnitude_unit"], "dBsm")
        self.assertAlmostEqual(sample["phase_degrees"], 350.0, places=5)
        self.assertAlmostEqual(sample["complex"]["real"], 2.0 * math.cos(math.radians(350.0)))
        self.assertAlmostEqual(sample["complex"]["imag"], 2.0 * math.sin(math.radians(350.0)))

    def test_strict_query_explains_miss_and_nearest_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "nearest=True"):
            query_sample(
                self._dataset(),
                azimuth=80.0,
                elevation=0.0,
                frequency=2000.0,
                polarization="VV",
                angle_unit="deg",
                frequency_unit="MHz",
            )
        nearest = query_sample(
            self._dataset(),
            azimuth=80.0,
            elevation=0.0,
            frequency=2000.0,
            polarization="VV",
            angle_unit="deg",
            frequency_unit="MHz",
            nearest=True,
        )
        self.assertEqual(nearest["indices"]["azimuth"], 1)
        self.assertAlmostEqual(
            nearest["matches"]["azimuth"]["difference_in_requested_unit"], 10.0
        )

    def test_main_uses_configuration_and_emits_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "query.grim"
            self._dataset().save(path)
            stdout = StringIO()
            with patch.multiple(
                query_example,
                DATASET_PATH=path,
                QUERY_AZIMUTH=90.0,
                QUERY_ELEVATION=0.0,
                QUERY_FREQUENCY=2.0,
                QUERY_POLARIZATION="VV",
                QUERY_ANGLE_UNIT="deg",
                QUERY_FREQUENCY_UNIT="GHz",
                NEAREST_MATCH=False,
                ANGLE_TOLERANCE=None,
                FREQUENCY_TOLERANCE=None,
                JSON_OUTPUT_PATH=None,
                OVERWRITE_JSON=False,
            ):
                with redirect_stdout(stdout):
                    status = query_example.main()
            self.assertEqual(status, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["indices"]["frequency"], 1)
            self.assertNotIn("NaN", stdout.getvalue())

    def test_main_preserves_existing_json_without_configured_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dataset_path = folder / "query.grim"
            output_path = folder / "sample.json"
            self._dataset().save(dataset_path)
            output_path.write_text("keep me\n", encoding="utf-8")
            configured = {
                "DATASET_PATH": dataset_path,
                "QUERY_AZIMUTH": 90.0,
                "QUERY_ELEVATION": 0.0,
                "QUERY_FREQUENCY": 2.0,
                "QUERY_POLARIZATION": "VV",
                "QUERY_ANGLE_UNIT": "deg",
                "QUERY_FREQUENCY_UNIT": "GHz",
                "NEAREST_MATCH": False,
                "ANGLE_TOLERANCE": None,
                "FREQUENCY_TOLERANCE": None,
                "JSON_OUTPUT_PATH": output_path,
                "OVERWRITE_JSON": False,
            }

            with patch.multiple(query_example, **configured):
                with self.assertRaisesRegex(SystemExit, "OVERWRITE_JSON = True"):
                    query_example.main()
                self.assertEqual(
                    output_path.read_text(encoding="utf-8"), "keep me\n"
                )

            configured["OVERWRITE_JSON"] = True
            with patch.multiple(query_example, **configured):
                with redirect_stdout(StringIO()):
                    self.assertEqual(query_example.main(), 0)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["indices"][
                    "frequency"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
