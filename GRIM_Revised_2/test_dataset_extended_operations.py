"""Focused regressions for safe subtraction, wrapping, audit, and stitching."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import grim_dataset
from grim_csv_schema import write_flat_csv
from grim_dataset import RcsGrid
from grim_headless import load_flat_csv
from grim_python import _display_values, crop_dataset, regrid_axis


_UNITS = {
    "azimuth": "deg",
    "elevation": "deg",
    "frequency": "GHz",
    "rcs_log_unit": "dBsm",
    "rcs_linear_quantity": "sigma_3d",
}
_COHERENT_EXTRA = {
    "phase_reference": "vehicle origin",
    "time_convention": "exp(-jwt)",
    "polarization_basis": "linear V/H",
}


def _line_grid(
    azimuths,
    power,
    phase=None,
    *,
    units=None,
    extra=None,
    history="source history",
):
    azimuths = np.asarray(azimuths, dtype=float)
    power = np.asarray(power)
    shape = (len(azimuths), 1, 1, 1)
    if phase is None:
        phase = np.full(power.shape, np.nan, dtype=float)
    return RcsGrid(
        azimuths,
        [0.0],
        [10.0],
        ["HH"],
        rcs_power=np.asarray(power).reshape(shape),
        rcs_phase=np.asarray(phase).reshape(shape),
        units=dict(_UNITS if units is None else units),
        extra=dict(_COHERENT_EXTRA if extra is None else extra),
        history=history,
    )


class IncoherentSubtractSafetyTests(unittest.TestCase):
    def test_material_negative_power_is_rejected_with_count_and_minimum(self):
        left = _line_grid([0.0, 1.0, 2.0], [1.0, 1.0, 3.0])
        right = _line_grid(
            [0.0, 1.0, 2.0],
            [2.0, np.nextafter(1.0, 2.0), 5.0],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"materially negative linear power in 2 cell\(s\).*minimum difference is -2",
        ):
            left.incoherent_subtract(right)

    def test_only_roundoff_negative_is_clamped_to_exact_zero(self):
        left = _line_grid([0.0, 1.0], [1.0, 4.0])
        right = _line_grid(
            [0.0, 1.0], [np.nextafter(1.0, 2.0), 3.0]
        )

        result = left.incoherent_subtract(right)

        np.testing.assert_array_equal(result.rcs_power.ravel(), [0.0, 1.0])
        self.assertTrue(np.isnan(result.rcs_phase).all())


class PhaseWrapTests(unittest.TestCase):
    def setUp(self):
        self.grid = _line_grid(
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 4.0, 9.0, 16.0],
            [-0.5 * np.pi, 0.0, 3.0 * np.pi, np.nan],
        )

    def test_wrap_phase_zero_to_360_preserves_power_missingness_and_field(self):
        original_field = self.grid.rcs.copy()

        wrapped = self.grid.wrap_phase("0_360")

        np.testing.assert_array_equal(wrapped.rcs_power, self.grid.rcs_power)
        np.testing.assert_allclose(
            wrapped.rcs_phase.ravel()[:3],
            [1.5 * np.pi, 0.0, np.pi],
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertTrue(np.isnan(wrapped.rcs_phase.ravel()[3]))
        self.assertEqual(wrapped.units["phase_wrap"], "0_360")
        self.assertEqual(wrapped.audit()["metrics"]["phase"]["outside_declared_wrap_count"], 0)
        np.testing.assert_allclose(
            wrapped.rcs.ravel()[:3],
            original_field.ravel()[:3],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        self.assertEqual(wrapped.extra["phase_reference"], "vehicle origin")
        self.assertIn("Wrap phase to [0, 360) deg", wrapped.history)
        np.testing.assert_array_equal(self.grid.rcs_phase.ravel()[:3], [-0.5 * np.pi, 0.0, 3.0 * np.pi])

    def test_wrap_phase_minus_180_to_180_and_invalid_mode(self):
        wrapped = self.grid.wrap_phase("-180_180")
        np.testing.assert_allclose(
            wrapped.rcs_phase.ravel()[:3],
            [-0.5 * np.pi, 0.0, -np.pi],
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertEqual(wrapped.units["phase_wrap"], "-180_180")
        with self.assertRaisesRegex(ValueError, "phase wrap mode"):
            self.grid.wrap_phase("degrees")


class PhaseWrapPersistenceAndDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _path(self, name):
        return os.path.join(self.temp.name, name)

    def test_versioned_flat_csv_round_trip_preserves_zero_360(self):
        units = dict(_UNITS)
        units["phase_wrap"] = "0_360"
        source = _line_grid(
            [0.0],
            [4.0],
            [np.deg2rad(350.0)],
            units=units,
        )
        path = self._path("zero_360.csv")

        write_flat_csv(source, path, scale="both", include_phase=True)

        with open(path, "r", newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["grim_csv_schema"], "grim.flat-rcs.v1")
        self.assertEqual(row["phase_wrap"], "0_360")
        self.assertAlmostEqual(float(row["phase_deg"]), 350.0)

        restored = load_flat_csv(path)
        self.assertEqual(restored.units["phase_wrap"], "0_360")
        self.assertAlmostEqual(
            float(np.rad2deg(restored.rcs_phase.item())), 350.0
        )
        np.testing.assert_allclose(
            restored.rcs,
            source.rcs,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        self.assertEqual(
            restored.audit()["metrics"]["phase"]["outside_declared_wrap_count"],
            0,
        )

    def test_native_grim_round_trip_preserves_zero_360(self):
        units = dict(_UNITS)
        units["phase_wrap"] = "0_360"
        source = _line_grid(
            [0.0], [4.0], [np.deg2rad(350.0)], units=units
        )
        path = self._path("zero_360.grim")

        source.save(path)
        restored = RcsGrid.load(path)

        self.assertEqual(restored.units["phase_wrap"], "0_360")
        self.assertAlmostEqual(
            float(np.rad2deg(restored.rcs_phase.item())), 350.0
        )
        np.testing.assert_allclose(restored.rcs, source.rcs, atol=1.0e-13)

    def test_versioned_flat_csv_without_phase_wrap_defaults_to_legacy_range(self):
        source = _line_grid(
            [0.0], [4.0], [np.deg2rad(-10.0)]
        )
        complete_path = self._path("complete.csv")
        legacy_path = self._path("without_phase_wrap.csv")
        write_flat_csv(source, complete_path, include_phase=True)
        with open(complete_path, "r", newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        phase_wrap_index = rows[0].index("phase_wrap")
        for row in rows:
            del row[phase_wrap_index]
        with open(legacy_path, "w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

        restored = load_flat_csv(legacy_path)

        self.assertEqual(restored.units["phase_wrap"], "-180_180")
        self.assertAlmostEqual(
            float(np.rad2deg(restored.rcs_phase.item())), -10.0
        )

    def test_headless_plot_display_honors_zero_360_marker(self):
        units = dict(_UNITS)
        units["phase_wrap"] = "0_360"
        dataset = _line_grid(
            [0.0], [1.0], [np.deg2rad(350.0)], units=units
        )

        displayed = _display_values(
            dataset,
            dataset.rcs.ravel(),
            frequency=10.0,
            phase=True,
            scale="dbsm",
        )

        np.testing.assert_allclose(displayed, [350.0], atol=1.0e-12)

    def test_regrid_normalizes_signed_interpolation_to_declared_zero_360(self):
        units = dict(_UNITS)
        units["phase_wrap"] = "0_360"
        source = _line_grid(
            [0.0, 2.0],
            [1.0, 1.0],
            np.deg2rad([300.0, 320.0]),
            units=units,
        )

        regridded = regrid_axis(
            source, "azimuth", values=[0.0, 1.0, 2.0]
        )

        self.assertEqual(regridded.units["phase_wrap"], "0_360")
        np.testing.assert_allclose(
            np.rad2deg(regridded.rcs_phase.ravel()),
            [300.0, 310.0, 320.0],
            atol=1.0e-10,
        )
        audit = regridded.audit()
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(
            audit["metrics"]["phase"]["outside_declared_wrap_count"], 0
        )

    def test_constructor_rejects_unsupported_declared_phase_wrap(self):
        units = dict(_UNITS)
        units["phase_wrap"] = "positive"
        with self.assertRaisesRegex(ValueError, "phase_wrap must be"):
            _line_grid([0.0], [1.0], [0.0], units=units)


class CropDatasetReplayTests(unittest.TestCase):
    def test_explicit_selected_axis_values_preserve_exact_sample_mapping(self):
        azimuths = np.asarray([0.0, 10.0, 20.0])
        elevations = np.asarray([-5.0, 5.0])
        frequencies = np.asarray([8.0, 9.0, 10.0])
        polarizations = np.asarray(["VV", "HH"])
        shape = (3, 2, 3, 2)
        power = np.arange(1.0, np.prod(shape) + 1.0).reshape(shape)
        phase = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100.0
        units = dict(_UNITS)
        units["phase_wrap"] = "0_360"
        source = RcsGrid(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            source_path="measured.grim",
            history="loaded",
            units=units,
            extra=dict(_COHERENT_EXTRA),
        )

        cropped = crop_dataset(
            source,
            azimuths=[0.0, 20.0],
            elevations=[5.0],
            frequencies=[8.0, 10.0],
            polarizations=["HH"],
        )

        np.testing.assert_array_equal(cropped.azimuths, [0.0, 20.0])
        np.testing.assert_array_equal(cropped.elevations, [5.0])
        np.testing.assert_array_equal(cropped.frequencies, [8.0, 10.0])
        np.testing.assert_array_equal(cropped.polarizations, ["HH"])
        selection = np.ix_([0, 2], [1], [0, 2], [1])
        np.testing.assert_array_equal(cropped.rcs_power, power[selection])
        np.testing.assert_array_equal(cropped.rcs_phase, phase[selection])
        self.assertEqual(cropped.units["phase_wrap"], "0_360")
        self.assertEqual(cropped.extra["phase_reference"], "vehicle origin")
        self.assertEqual(cropped.source_path, "measured.grim")
        np.testing.assert_array_equal(source.rcs_power, power)


class DatasetAuditTests(unittest.TestCase):
    def test_healthy_audit_is_strict_json_and_non_mutating(self):
        shape = (2, 1, 3, 1)
        grid = RcsGrid(
            [0.0, 1.0],
            [0.0],
            [8.0, 9.0, 10.0],
            ["HH"],
            rcs_power=np.arange(1.0, 7.0).reshape(shape),
            rcs_phase=np.zeros(shape),
            units=dict(_UNITS),
            extra=dict(_COHERENT_EXTRA),
        )
        snapshots = tuple(
            np.array(value, copy=True)
            for value in (
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                grid.polarizations,
                grid.rcs_power,
                grid.rcs_phase,
            )
        )

        report = grid.audit()

        json.dumps(report, allow_nan=False)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["warnings"], [])
        metrics = report["metrics"]
        self.assertEqual(metrics["grid"]["cell_count"], 6)
        self.assertEqual(metrics["grid"]["finite_power_count"], 6)
        self.assertTrue(metrics["frequency_uniformity"]["uniform"])
        self.assertTrue(metrics["readiness"]["coherent_arithmetic"])
        self.assertTrue(metrics["readiness"]["frequency_transform"])
        for actual, expected in zip(
            (
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                grid.polarizations,
                grid.rcs_power,
                grid.rcs_phase,
            ),
            snapshots,
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_audit_reports_public_mutation_seam_and_readiness_failures(self):
        grid = RcsGrid(
            [0.0, 360.0],
            [0.0],
            [1.0, 2.0, 4.0],
            ["HH"],
            rcs_power=np.ones((2, 1, 3, 1)),
            rcs_phase=np.zeros((2, 1, 3, 1)),
            units=dict(_UNITS),
            extra={},
        )
        grid.rcs_power[0, 0, 0, 0] = -0.25
        grid.rcs_power[-1, 0, 0, 0] = 2.0
        grid.rcs_phase[1, 0, 1, 0] = np.nan

        report = grid.audit()

        json.dumps(report, allow_nan=False)
        self.assertEqual(report["status"], "error")
        error_codes = {item["code"] for item in report["errors"]}
        warning_codes = {item["code"] for item in report["warnings"]}
        self.assertIn("negative_power", error_codes)
        self.assertIn("missing_coherent_phase", warning_codes)
        self.assertIn("nonuniform_frequency", warning_codes)
        self.assertIn("conflicting_azimuth_seam", warning_codes)
        self.assertIn("missing_phase_reference", warning_codes)
        self.assertEqual(report["metrics"]["seam"]["conflict_cell_count"], 1)
        self.assertFalse(report["metrics"]["readiness"]["incoherent_arithmetic"])
        self.assertFalse(report["metrics"]["readiness"]["coherent_arithmetic"])

    def test_audit_survives_shape_corruption(self):
        grid = _line_grid([0.0, 1.0], [1.0, 2.0], [0.0, 0.0])
        grid.rcs_phase = np.zeros((2,), dtype=float)

        report = grid.audit()

        self.assertEqual(report["status"], "error")
        self.assertIn(
            "phase_shape_mismatch", {item["code"] for item in report["errors"]}
        )
        json.dumps(report, allow_nan=False)


class StitchManyTests(unittest.TestCase):
    def setUp(self):
        self.first = _line_grid(
            [0.0, 1.0], [1.0, 4.0], [0.0, 0.0], history="first"
        )
        self.last = _line_grid(
            [1.0, 2.0], [9.0, 16.0], [0.5 * np.pi, np.pi], history="last"
        )

    def test_all_stitch_policies_resolve_overlap_and_report_cells(self):
        with mock.patch.object(grim_dataset, "_JOIN_MERGE_BLOCK_CELLS", 1):
            priority_first, first_report = RcsGrid.stitch_many(
                self.first,
                self.last,
                policy="priority-first",
                return_report=True,
            )
            priority_last, last_report = RcsGrid.stitch_many(
                self.first,
                self.last,
                policy="priority-last",
                return_report=True,
            )
            power_mean, power_report = RcsGrid.stitch_many(
                self.first,
                self.last,
                policy="power-mean",
                return_report=True,
            )
            coherent_mean, coherent_report = RcsGrid.stitch_many(
                self.first,
                self.last,
                policy="coherent-mean",
                return_report=True,
            )

        np.testing.assert_array_equal(priority_first.azimuths, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(priority_first.rcs_power.ravel(), [1.0, 4.0, 16.0])
        np.testing.assert_allclose(priority_first.rcs_phase.ravel(), [0.0, 0.0, -np.pi])
        np.testing.assert_allclose(priority_last.rcs_power.ravel(), [1.0, 9.0, 16.0])
        np.testing.assert_allclose(priority_last.rcs_phase.ravel(), [0.0, 0.5 * np.pi, -np.pi])
        np.testing.assert_allclose(power_mean.rcs_power.ravel(), [1.0, 6.5, 16.0])
        self.assertTrue(np.isnan(power_mean.rcs_phase.ravel()[1]))

        expected_field = (2.0 + 3.0j) / 2.0
        self.assertAlmostEqual(
            coherent_mean.rcs_power.ravel()[1], abs(expected_field) ** 2
        )
        self.assertAlmostEqual(
            coherent_mean.rcs_phase.ravel()[1], np.angle(expected_field)
        )
        for report, policy in (
            (first_report, "priority-first"),
            (last_report, "priority-last"),
            (power_report, "power-mean"),
            (coherent_report, "coherent-mean"),
        ):
            json.dumps(report, allow_nan=False)
            self.assertEqual(report["policy"], policy)
            self.assertEqual(report["overlap_count"], 1)
            self.assertEqual(report["conflict_count"], 1)
            self.assertEqual(report["equal_count"], 0)
            self.assertEqual(report["contributing_count"], 4)
            self.assertEqual(report["output_finite_count"], 3)

        provenance = json.loads(
            coherent_mean.extra["stitch_provenance_json"]
        )
        self.assertEqual(provenance["policy"], "coherent-mean")
        self.assertEqual(provenance["conflict_count"], 1)
        self.assertIn("Stitch (coherent-mean", coherent_mean.history)

    def test_equal_overlap_is_reported_and_strict_join_remains_strict(self):
        equal_last = _line_grid([1.0, 2.0], [4.0, 16.0], [0.0, np.pi])

        stitched, report = RcsGrid.stitch_many(
            self.first, equal_last, return_report=True
        )

        self.assertIsInstance(stitched, RcsGrid)
        self.assertEqual(report["overlap_count"], 1)
        self.assertEqual(report["equal_count"], 1)
        self.assertEqual(report["conflict_count"], 0)
        with self.assertRaisesRegex(ValueError, "conflicting finite samples"):
            RcsGrid.join_many(self.first, self.last)

    def test_coherent_mean_requires_phase_and_metadata_or_attestation(self):
        missing_phase = _line_grid(
            [0.0, 1.0], [1.0, 4.0], [0.0, np.nan]
        )
        with self.assertRaisesRegex(ValueError, "requires finite phase"):
            RcsGrid.stitch_many(
                missing_phase, self.first, policy="coherent-mean"
            )

        unknown_a = _line_grid([0.0], [1.0], [0.0], extra={})
        unknown_b = _line_grid([0.0], [1.0], [0.0], extra={})
        with self.assertRaisesRegex(ValueError, "requires declared coherent metadata"):
            RcsGrid.stitch_many(
                unknown_a, unknown_b, policy="coherent-mean"
            )
        attested = RcsGrid.stitch_many(
            unknown_a,
            unknown_b,
            policy="coherent-mean",
            metadata_attested=True,
        )
        attestation = json.loads(
            attested.extra["coherent_metadata_attestation_json"]
        )
        self.assertTrue(attestation["user_attested"])
        self.assertEqual(attestation["operation"], "coherent-mean-stitch")

    def test_attestation_never_overrides_explicit_metadata_conflict(self):
        opposite_extra = dict(_COHERENT_EXTRA)
        opposite_extra["time_convention"] = "exp(+jwt)"
        opposite = _line_grid([1.0, 2.0], [9.0, 16.0], [0.0, 0.0], extra=opposite_extra)

        with self.assertRaisesRegex(ValueError, "matching time conventions"):
            RcsGrid.stitch_many(
                self.first,
                opposite,
                policy="coherent-mean",
                metadata_attested=True,
            )

    def test_memory_cap_policy_validation_and_physical_units(self):
        with self.assertRaises(MemoryError):
            RcsGrid.stitch_many(
                self.first, self.last, max_output_bytes=1
            )
        with self.assertRaisesRegex(ValueError, "policy must be"):
            RcsGrid.stitch_many(self.first, self.last, policy="overwrite")
        with self.assertRaisesRegex(TypeError, "return_report"):
            RcsGrid.stitch_many(
                self.first, self.last, return_report="yes"
            )

        other_units = dict(_UNITS)
        other_units["frequency"] = "MHz"
        other = _line_grid(
            [1.0, 2.0],
            [9.0, 16.0],
            [0.0, 0.0],
            units=other_units,
        )
        with self.assertRaisesRegex(ValueError, "frequency unit mismatch"):
            RcsGrid.stitch_many(self.first, other)


if __name__ == "__main__":
    unittest.main()
