import json
import unittest

import numpy as np

from grim_dataset import RcsGrid
from grim_headless import combine_datasets
from grim_python import (
    coherent_divide,
    convert_extrusion,
    medianize_azimuth,
    offset_db,
    shift_dataset,
)


def _grid(
    azimuths=(0.0, 90.0, 180.0, 270.0),
    *,
    angle_unit="deg",
    quantity="sigma_3d",
    log_unit="dBsm",
    values=(1.0, 2.0, 3.0, 4.0),
    extra=None,
):
    power = np.asarray(values, dtype=float).reshape(-1, 1, 1, 1)
    return RcsGrid(
        np.asarray(azimuths, dtype=float),
        np.asarray([0.0]),
        np.asarray([10.0]),
        np.asarray(["HH"]),
        rcs_power=power,
        rcs_phase=np.zeros_like(power),
        units={
            "azimuth": angle_unit,
            "elevation": angle_unit,
            "frequency": "GHz",
            "rcs_linear_quantity": quantity,
            "rcs_log_unit": log_unit,
        },
        extra=dict(extra or {}),
    )


class PythonDatasetHelperTest(unittest.TestCase):
    def test_medianize_is_degree_radian_equivalent_and_retains_native_unit(self):
        degree = _grid()
        radian = _grid(
            np.deg2rad(degree.azimuths),
            angle_unit="rad",
        )

        degree_result = medianize_azimuth(
            degree, window_degrees=200.0, slide_degrees=90.0
        )
        radian_result = medianize_azimuth(
            radian, window_degrees=200.0, slide_degrees=90.0
        )

        np.testing.assert_allclose(
            degree_result.azimuths, np.rad2deg(radian_result.azimuths)
        )
        np.testing.assert_allclose(
            degree_result.rcs_power, radian_result.rcs_power
        )
        self.assertEqual(radian_result.units["azimuth"], "rad")
        # The first periodic window crosses the 0/360 seam and includes the
        # 270-degree sample exactly once: median([1, 2, 4]) == 2.
        self.assertEqual(float(degree_result.rcs_power[0, 0, 0, 0]), 2.0)
        self.assertTrue(np.isnan(degree_result.rcs_phase).all())
        self.assertNotIn("phase_reference", degree_result.extra)

    def test_periodic_median_counts_closed_sweep_seam_once(self):
        closed = _grid(
            (0.0, 90.0, 180.0, 270.0, 360.0),
            values=(1.0, 2.0, 100.0, 100.0, 1.0),
        )
        closed.source_path = "closed.grim"
        closed.history = "loaded closed sweep"

        result = medianize_azimuth(
            closed,
            window_degrees=360.0,
            slide_degrees=360.0,
        )

        # Four physical directions remain: median([1, 2, 100, 100]) = 51.
        self.assertEqual(float(result.rcs_power.item()), 51.0)
        self.assertEqual(result.source_path, "closed.grim")
        self.assertEqual(result.history, "loaded closed sweep")

    def test_periodic_median_rejects_conflicting_closed_sweep_seam(self):
        conflict = _grid(
            (-180.0, -90.0, 0.0, 90.0, 180.0),
            values=(1.0, 2.0, 3.0, 4.0, 9.0),
        )

        with self.assertRaisesRegex(ValueError, "conflicting finite seam"):
            medianize_azimuth(
                conflict,
                window_degrees=180.0,
                slide_degrees=90.0,
            )

    def test_extrusion_conversion_round_trips_and_rejects_ratios(self):
        body_profile = np.asarray([1.0, 2.0])
        source = _grid(extra={
            "phase_reference": "vehicle origin",
            "rcs_amp_real": np.ones((4, 1, 1, 1)),
            "solver_certification": "source-only",
            "amplitude_convention": "producer-specific normalization",
            "rcs_domain": "producer-specific-domain",
            "power_domain": "producer-specific-power-domain",
            "body_profile_radius_m": body_profile,
        })
        source.source_path = "vehicle.grim"
        source.history = "loaded vehicle"
        width = convert_extrusion(source, to="dbke", length_m=2.0)
        restored = convert_extrusion(width, to="dbsm", length_m=2.0)

        np.testing.assert_allclose(restored.rcs_power, source.rcs_power)
        np.testing.assert_allclose(restored.rcs_phase, source.rcs_phase)
        self.assertEqual(width.linear_quantity(), "sigma_2d")
        self.assertEqual(width.extra["phase_reference"], "vehicle origin")
        self.assertNotIn("rcs_amp_real", width.extra)
        self.assertNotIn("solver_certification", width.extra)
        self.assertNotIn("amplitude_convention", width.extra)
        self.assertNotIn("rcs_domain", width.extra)
        self.assertNotIn("power_domain", width.extra)
        self.assertEqual(width.source_path, "vehicle.grim")
        self.assertEqual(width.history, "loaded vehicle")
        np.testing.assert_array_equal(
            width.extra["body_profile_radius_m"], [1.0, 2.0]
        )
        self.assertTrue(
            np.shares_memory(width.extra["body_profile_radius_m"], body_profile)
        )
        self.assertFalse(width.extra["body_profile_radius_m"].flags.writeable)

        ratio = _grid(quantity="power_ratio", log_unit="dB")
        with self.assertRaisesRegex(ValueError, "requires a sigma_3d/dBsm"):
            convert_extrusion(ratio, to="dbke", length_m=2.0)

    def test_field_edits_drop_stale_solver_payload_but_keep_phase_reference(self):
        body_profile = np.asarray([1.0, 2.0])
        source = _grid(extra={
            "phase_reference": "vehicle origin",
            "rcs_amp_real": np.ones((4, 1, 1, 1)),
            "solver_certification": "source-only",
            "amplitude_convention": "producer-specific normalization",
            "rcs_domain": "producer-specific-domain",
            "power_domain": "producer-specific-power-domain",
            "body_profile_radius_m": body_profile,
        })
        source.source_path = "vehicle.grim"
        source.history = "loaded vehicle"
        shifted = shift_dataset(source, phase_degrees=30.0)
        offset = offset_db(source, 3.0)
        for result in (shifted, offset):
            self.assertEqual(result.extra["phase_reference"], "vehicle origin")
            self.assertNotIn("rcs_amp_real", result.extra)
            self.assertNotIn("solver_certification", result.extra)
            self.assertNotIn("amplitude_convention", result.extra)
            self.assertNotIn("rcs_domain", result.extra)
            self.assertNotIn("power_domain", result.extra)
            self.assertEqual(result.source_path, "vehicle.grim")
            self.assertEqual(result.history, "loaded vehicle")
            np.testing.assert_array_equal(
                result.extra["body_profile_radius_m"], [1.0, 2.0]
            )
            self.assertTrue(
                np.shares_memory(result.extra["body_profile_radius_m"], body_profile)
            )
            self.assertFalse(result.extra["body_profile_radius_m"].flags.writeable)

    def test_coherent_divide_requires_explicit_unknown_metadata_attestation(self):
        numerator = _grid(values=(4.0, 4.0, 4.0, 4.0))
        denominator = _grid(values=(1.0, 1.0, 1.0, 1.0))
        numerator.source_path = "numerator.grim"
        numerator.history = "loaded numerator"

        with self.assertRaisesRegex(ValueError, "phase reference"):
            coherent_divide(numerator, denominator)
        with self.assertRaisesRegex(TypeError, "must be True or False"):
            coherent_divide(numerator, denominator, metadata_attested="false")
        result = coherent_divide(
            numerator, denominator, metadata_attested=True
        )

        np.testing.assert_allclose(result.rcs_power, 4.0)
        self.assertEqual(result.linear_quantity(), "power_ratio")
        self.assertNotIn("phase_reference", result.extra)
        self.assertEqual(result.source_path, "numerator.grim")
        self.assertIn("loaded numerator", result.history)
        self.assertIn("User-attested coherent metadata", result.history)
        attestation = json.loads(
            result.extra["coherent_metadata_attestation_json"]
        )
        self.assertEqual(attestation["operation"], "coherent-divide")
        self.assertFalse(attestation["declarations_inferred"])
        self.assertEqual(
            attestation["missing_declarations_by_input"]["phase_reference"],
            [1, 2],
        )

    def test_headless_coherent_add_retains_attestation_provenance(self):
        left = _grid()
        right = _grid()
        left.source_path = "left.grim"
        left.history = "loaded left"

        result = combine_datasets(
            (left, right),
            "coherent-add",
            coherent_metadata_attested=True,
        )

        self.assertEqual(result.source_path, "left.grim")
        self.assertNotIn("phase_reference", result.extra)
        self.assertIn("User-attested coherent metadata", result.history)
        record = json.loads(result.extra["coherent_metadata_attestation_json"])
        self.assertEqual(record["operation"], "coherent-add")
        self.assertTrue(record["user_attested"])

        declared_a = _grid(extra={"phase_reference": "origin A"})
        declared_b = _grid(extra={"phase_reference": "origin B"})
        with self.assertRaisesRegex(ValueError, "matching phase references"):
            combine_datasets(
                (left, declared_a, declared_b),
                "coherent-add",
                coherent_metadata_attested=True,
            )
        with self.assertRaisesRegex(TypeError, "must be True or False"):
            combine_datasets(
                (left, right),
                "coherent-add",
                coherent_metadata_attested="false",
            )


if __name__ == "__main__":
    unittest.main()
