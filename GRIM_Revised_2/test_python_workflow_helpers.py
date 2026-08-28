from __future__ import annotations

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import grim_headless
import grim_python
from grim_csv_schema import write_flat_csv
from grim_dataset import RcsGrid
from grim_headless import audit_dataset, combine_datasets
from grim_python import (
    PythonScriptRecorder,
    crop_dataset,
    regrid_axis,
    stitch_datasets,
    wrap_phase,
)


def _grid() -> RcsGrid:
    shape = (4, 3, 4, 2)
    power = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    phase = np.linspace(-0.5, 0.5, np.prod(shape), dtype=float).reshape(shape)
    return RcsGrid(
        np.asarray([0.0, 10.0, 20.0, 30.0]),
        np.asarray([-5.0, 0.0, 5.0]),
        np.asarray([8.0, 9.0, 10.0, 11.0]),
        np.asarray(["HH", "VV"]),
        rcs_power=power,
        rcs_phase=phase,
        source_path="source.grim",
        history="loaded source",
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_linear_quantity": "sigma_3d",
            "rcs_log_unit": "dBsm",
        },
        extra={
            "phase_reference": "vehicle origin",
            "time_convention": "exp(+j*omega*t)",
            "polarization_basis": "H/V",
        },
    )


class PythonWorkflowHelperTests(unittest.TestCase):
    def test_crop_applies_inclusive_native_ranges_before_strides(self):
        source = _grid()

        result = crop_dataset(
            source,
            azimuth_range=(5.0, 30.0),
            elevation_range=(-5.0, 5.0),
            frequency_range=(8.5, 11.0),
            azimuth_stride=2,
            elevation_stride=2,
            frequency_stride=2,
            polarizations=["HH"],
        )

        np.testing.assert_array_equal(result.azimuths, [10.0, 30.0])
        np.testing.assert_array_equal(result.elevations, [-5.0, 5.0])
        np.testing.assert_array_equal(result.frequencies, [9.0, 11.0])
        np.testing.assert_array_equal(result.polarizations, ["HH"])
        self.assertEqual(result.rcs_power.shape, (2, 2, 2, 1))
        self.assertEqual(result.source_path, source.source_path)
        self.assertEqual(result.history, source.history)
        self.assertEqual(result.units, source.units)
        self.assertEqual(result.extra["phase_reference"], "vehicle origin")

    def test_crop_rejects_nonintegral_or_nonpositive_strides(self):
        source = _grid()
        for value in (True, 1.5, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "positive integer"):
                    crop_dataset(source, azimuth_stride=value)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            crop_dataset(source, frequency_stride=0)

    def test_regrid_axis_accepts_resolved_values_or_start_stop_step(self):
        source = _grid()

        explicit = regrid_axis(source, "az", values=[0.0, 5.0, 10.0])
        resolved = regrid_axis(
            source, "frequency", start=8.0, stop=10.0, step=0.5
        )

        np.testing.assert_array_equal(explicit.azimuths, [0.0, 5.0, 10.0])
        np.testing.assert_array_equal(
            resolved.frequencies, [8.0, 8.5, 9.0, 9.5, 10.0]
        )
        for result in (explicit, resolved):
            self.assertEqual(result.source_path, source.source_path)
            self.assertEqual(result.history, source.history)
            self.assertEqual(result.extra["phase_reference"], "vehicle origin")

    def test_regrid_axis_rejects_ambiguous_or_unsafe_grid_specs(self):
        source = _grid()
        with self.assertRaisesRegex(ValueError, "either values"):
            regrid_axis(
                source,
                "azimuth",
                values=[0.0, 1.0],
                start=0.0,
                stop=1.0,
                step=1.0,
            )
        with self.assertRaisesRegex(ValueError, "required"):
            regrid_axis(source, "azimuth", start=0.0, stop=1.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            regrid_axis(source, "azimuth", values=[0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "safety limit"):
            regrid_axis(
                source,
                "azimuth",
                start=0.0,
                stop=2.0,
                step=1.0e-7,
            )

    def test_phase_wrap_delegates_user_modes_and_preserves_complex_field(self):
        phase_deg = np.asarray([-190.0, 180.0, 370.0])
        power = np.asarray([1.0, 2.0, 3.0]).reshape(3, 1, 1, 1)
        source = RcsGrid(
            [0.0, 1.0, 2.0],
            [0.0],
            [10.0],
            ["HH"],
            rcs_power=power,
            rcs_phase=np.deg2rad(phase_deg).reshape(3, 1, 1, 1),
            source_path="phase.grim",
            history="loaded phase",
            extra={"phase_reference": "origin"},
        )

        positive = wrap_phase(source, mode="0_360")
        signed = wrap_phase(source)

        np.testing.assert_allclose(
            np.rad2deg(positive.rcs_phase[:, 0, 0, 0]), [170.0, 180.0, 10.0]
        )
        np.testing.assert_allclose(
            np.rad2deg(signed.rcs_phase[:, 0, 0, 0]), [170.0, -180.0, 10.0]
        )
        np.testing.assert_allclose(np.abs(positive.rcs), np.abs(source.rcs))
        np.testing.assert_allclose(positive.rcs_power, source.rcs_power)
        self.assertEqual(positive.source_path, source.source_path)
        self.assertIn("loaded phase", positive.history)
        self.assertIn("Wrap phase", positive.history)
        with self.assertRaisesRegex(ValueError, "mode must"):
            wrap_phase(source, mode="principal")

    def test_versioned_flat_csv_preserves_phase_wrap_representation(self):
        wrapped = wrap_phase(_grid(), mode="0_360")
        with tempfile.TemporaryDirectory() as root_text:
            path = Path(root_text) / "wrapped.csv"
            write_flat_csv(wrapped, path, scale="linear", include_phase=True)

            with path.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertEqual({row["phase_wrap"] for row in rows}, {"0_360"})
            phase_degrees = np.asarray(
                [float(row["phase_deg"]) for row in rows if row["phase_deg"]]
            )
            self.assertTrue(np.all((phase_degrees >= 0.0) & (phase_degrees < 360.0)))

            restored = grim_headless.load_flat_csv(path)

        self.assertEqual(restored.units["phase_wrap"], "0_360")
        np.testing.assert_allclose(restored.rcs_power, wrapped.rcs_power)
        np.testing.assert_allclose(
            restored.rcs,
            wrapped.rcs,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_stitch_helpers_delegate_every_policy_and_provenance_option(self):
        first = _grid()
        second = _grid()
        expected = object()
        with mock.patch.object(
            RcsGrid, "stitch_many", return_value=expected, create=True
        ) as stitch_many:
            result = stitch_datasets(
                first,
                second,
                policy="priority-first",
                tol=2.5e-5,
                metadata_attested=True,
                max_output_bytes=1234,
                return_report=True,
            )

        self.assertIs(result, expected)
        stitch_many.assert_called_once_with(
            first,
            second,
            policy="priority-first",
            tol=2.5e-5,
            metadata_attested=True,
            max_output_bytes=1234,
            return_report=True,
        )

        with mock.patch.object(
            RcsGrid, "stitch_many", return_value=expected, create=True
        ) as stitch_many:
            result = combine_datasets(
                (first, second),
                "stitch",
                stitch_policy="priority-first",
                tol=4.0e-6,
                coherent_metadata_attested=True,
                max_output_bytes=5678,
            )
        self.assertIs(result, expected)
        stitch_many.assert_called_once_with(
            first,
            second,
            policy="priority-first",
            tol=4.0e-6,
            metadata_attested=True,
            max_output_bytes=5678,
            return_report=False,
        )

    def test_headless_join_forwards_the_requested_coordinate_tolerance(self):
        first = _grid()
        second = _grid()
        expected = object()
        with mock.patch.object(
            RcsGrid, "join_many", return_value=expected
        ) as join_many:
            result = combine_datasets(
                (first, second),
                "join",
                overlap="error",
                tol=2.5e-5,
                max_output_bytes=4321,
            )

        self.assertIs(result, expected)
        join_many.assert_called_once_with(
            first,
            second,
            tol=2.5e-5,
            overlap="error",
            max_output_bytes=4321,
        )

    def test_stitch_helper_returns_core_report_and_durable_provenance(self):
        source = _grid()
        first = source.axis_crop(azimuth_range=(0.0, 20.0))
        second = source.axis_crop(azimuth_range=(20.0, 30.0))

        stitched, report = stitch_datasets(
            first,
            second,
            policy="priority-first",
            return_report=True,
        )

        np.testing.assert_array_equal(stitched.azimuths, source.azimuths)
        np.testing.assert_allclose(stitched.rcs_power, source.rcs_power)
        self.assertEqual(stitched.source_path, source.source_path)
        self.assertIn("loaded source", stitched.history)
        self.assertIn("Stitch (priority-first", stitched.history)
        self.assertEqual(report["policy"], "priority-first")
        provenance = json.loads(stitched.extra["stitch_provenance_json"])
        self.assertEqual(provenance["schema"], "grim.stitch-provenance.v1")
        self.assertEqual(provenance["input_sources"], ["source.grim", "source.grim"])

    def test_headless_cli_stitch_preserves_existing_invocation_shape(self):
        source = _grid()
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            first_path = root / "first.grim"
            second_path = root / "second.grim"
            output_path = root / "stitched.grim"
            source.axis_crop(azimuth_range=(0.0, 20.0)).save(first_path)
            source.axis_crop(azimuth_range=(20.0, 30.0)).save(second_path)

            with contextlib.redirect_stdout(io.StringIO()):
                return_code = grim_headless.main(
                    [
                        str(first_path),
                        str(second_path),
                        "--operation",
                        "stitch",
                        "--stitch-policy",
                        "priority-first",
                        "--tol",
                        "1e-6",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(return_code, 0)
            stitched = RcsGrid.load(output_path)
            np.testing.assert_array_equal(stitched.azimuths, source.azimuths)
            np.testing.assert_allclose(stitched.rcs_power, source.rcs_power)
            self.assertIn("Stitch (priority-first", stitched.history)

    def test_audit_helper_and_cli_are_read_only_and_json_serializable(self):
        source = _grid()
        real_report = audit_dataset(source)
        self.assertIn("status", real_report)
        self.assertIn("metrics", real_report)
        json.dumps(real_report, allow_nan=False)

        source.audit = mock.Mock(
            return_value={"finite_count": np.int64(96), "shape": np.asarray([4, 3, 4, 2])}
        )
        self.assertEqual(audit_dataset(source)["finite_count"], 96)
        source.audit.assert_called_once_with()

        report = {
            "finite_count": np.int64(96),
            "shape": np.asarray([4, 3, 4, 2]),
            "flags": {"complete", "phase-ready"},
        }
        stream = io.StringIO()
        with mock.patch.object(grim_headless, "load_dataset", return_value=source), mock.patch.object(
            grim_headless, "audit_dataset", return_value=report
        ), mock.patch.object(source, "save") as save, contextlib.redirect_stdout(stream):
            return_code = grim_headless.main(["input.grim", "--audit"])

        self.assertEqual(return_code, 0)
        save.assert_not_called()
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["operation"], "audit")
        self.assertEqual(payload["datasets"][0]["input"], os.path.abspath("input.grim"))
        self.assertEqual(payload["datasets"][0]["report"]["finite_count"], 96)
        self.assertEqual(payload["datasets"][0]["report"]["shape"], [4, 3, 4, 2])

        with mock.patch.object(
            grim_headless, "load_dataset", return_value=source
        ), mock.patch.object(grim_headless, "audit_dataset", return_value=report):
            with self.assertRaisesRegex(SystemExit, "must not overwrite"):
                grim_headless.main(
                    ["input.grim", "--audit", "--output", "input.grim"]
                )

        with tempfile.TemporaryDirectory() as root_text:
            report_path = Path(root_text) / "audit.json"
            with mock.patch.object(
                grim_headless, "load_dataset", return_value=source
            ), mock.patch.object(
                grim_headless, "audit_dataset", return_value=report
            ), contextlib.redirect_stdout(io.StringIO()):
                return_code = grim_headless.main(
                    [
                        "input.grim",
                        "--audit",
                        "--output",
                        str(report_path),
                    ]
                )
            self.assertEqual(return_code, 0)
            saved_payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["operation"], "audit")

    def test_recorder_header_and_public_exports_include_workflow_helpers(self):
        names = {
            "crop_dataset",
            "regrid_axis",
            "stitch_datasets",
            "wrap_phase",
        }
        script = PythonScriptRecorder().script
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, grim_python.__all__)
                self.assertIn(f"        {name},", script)

    def test_headless_help_names_audit_and_every_stitch_policy(self):
        help_text = grim_headless._parser().format_help()
        self.assertIn("--audit", help_text)
        for policy in (
            "priority-first",
            "priority-last",
            "power-mean",
            "coherent-mean",
        ):
            with self.subTest(policy=policy):
                self.assertIn(policy, help_text)


if __name__ == "__main__":
    unittest.main()
