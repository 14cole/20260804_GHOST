"""Safety regressions for ISAR sampling and physical metadata preflight."""

from __future__ import annotations

import unittest

import numpy as np

from grim_dataset import RcsGrid
from plot_modes import isar_mode


class TestIsarCoreSafety(unittest.TestCase):
    def setUp(self):
        isar_mode._PREPROCESS_CACHE.clear()
        isar_mode._PREPROCESS_CACHE_BYTES = 0

    @staticmethod
    def _grid(
        *,
        azimuths=(-3.0, -1.0, 1.0, 3.0),
        frequencies=(9.0, 9.1, 9.2, 9.3),
        units=None,
        extra=None,
    ):
        azimuths = np.asarray(azimuths, dtype=float)
        frequencies = np.asarray(frequencies, dtype=float)
        shape = (azimuths.size, 1, frequencies.size, 1)
        base_units = {
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "angular_coordinate_system": "conic",
            "time_convention": "exp(+jwt)",
        }
        if units is not None:
            base_units = dict(units)
        return RcsGrid(
            azimuths,
            [0.0],
            frequencies,
            ["HH"],
            rcs=np.ones(shape, dtype=np.complex64),
            units=base_units,
            extra={
                "phase_reference": "fixed origin",
                "measurement_geometry": "far-field monostatic",
                "motion_compensation": "stable",
                "range_phase_convention": "S~exp(-j*2*k*R)",
            } if extra is None else extra,
        )

    def test_disjoint_frequency_band_is_zero_weighted_not_fabricated(self):
        grid = self._grid(frequencies=(8.0, 8.1, 10.0, 10.1))
        bands, _elapsed = isar_mode.form_isar(grid, window="Rectangular")
        result = bands[0]
        self.assertEqual(result["freq_gap_count"], 1)
        self.assertGreater(result["freq_gap_fraction"], 0.75)
        self.assertLess(result["phase_coverage"], 0.30)
        self.assertAlmostEqual(result["freq_largest_gap"] / 1.0e9, 1.9, places=6)

    def test_two_omitted_nominal_samples_are_already_a_gap(self):
        plan = isar_mode._uniform_resample_plan(
            np.asarray([8.0, 8.1, 8.4, 8.5])
        )
        self.assertEqual(plan["info"]["gap_count"], 1)
        target = plan["target"]
        unsupported = ~plan["support"]
        self.assertTrue(np.all(unsupported[(target > 8.1) & (target < 8.4)]))

    def test_disjoint_azimuth_target_has_no_interpolated_bridge(self):
        source = np.asarray([-10.0, -9.0, 9.0, 10.0])
        target = np.arange(-10.0, 11.0)
        samples = np.ones((source.size, 2), dtype=np.complex64)
        returned_target, resampled, info = isar_mode._resample_azimuth_to_target(
            source, samples, target, axis=0
        )
        np.testing.assert_array_equal(returned_target, target)
        self.assertEqual(info["gap_count"], 1)
        self.assertTrue(np.all(resampled[(target > -9.0) & (target < 9.0)] == 0.0))
        self.assertTrue(np.all(resampled[np.isin(target, source)] == 1.0))

    def test_pathological_gap_expansion_is_rejected_with_action(self):
        with self.assertRaisesRegex(ValueError, "Select one contiguous band/aperture"):
            isar_mode._uniform_resample_plan(
                np.asarray([0.0, 1.0, 1.0e9]), max_output_samples=100
            )

    def test_explicit_azimuth_target_cannot_create_unmeasured_aperture(self):
        grid = self._grid(azimuths=(-5.0, -2.5, 0.0, 2.5, 5.0))
        with self.assertRaisesRegex(ValueError, "unmeasured angles|full revolution"):
            isar_mode.form_isar(
                grid,
                azimuth_target_degrees=np.linspace(0.0, 360.0, 361),
            )
        with self.assertRaisesRegex(ValueError, "outside the measured aperture"):
            isar_mode.form_isar(
                grid,
                azimuth_target_degrees=np.linspace(-6.0, 5.0, 23),
            )

    def test_explicit_azimuth_target_must_be_finite_uniform_and_increasing(self):
        grid = self._grid()
        for target, message in (
            (np.asarray([-3.0, np.nan, 3.0]), "finite and strictly increasing"),
            (np.asarray([-3.0, 1.0, 0.0]), "finite and strictly increasing"),
            (np.asarray([-3.0, -1.0, 2.0, 3.0]), "uniformly spaced"),
        ):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, message):
                    isar_mode.form_isar(
                        grid, azimuth_target_degrees=target
                    )

    def test_selected_source_digest_binds_physical_axes_and_slice_identity(self):
        first = self._grid()
        second = self._grid(frequencies=(90.0, 90.1, 90.2, 90.3))
        selection = ([0, 1, 2, 3], 0, [0, 1, 2, 3], 0)
        token_first = isar_mode._selected_data_token(first, *selection)
        token_second = isar_mode._selected_data_token(second, *selection)
        self.assertNotEqual(token_first, token_second)

        second.frequencies[:] = first.frequencies
        second.elevations[0] = 1.0
        token_elevation = isar_mode._selected_data_token(second, *selection)
        self.assertNotEqual(token_first, token_elevation)

    def test_missing_frequency_unit_is_never_guessed(self):
        grid = self._grid(
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "time_convention": "exp(+jwt)",
            }
        )
        with self.assertRaisesRegex(ValueError, "frequency units are missing"):
            isar_mode.form_isar(grid, legacy_metadata_attested=True)

    def test_unknown_frequency_unit_is_blocked_by_isar_preflight(self):
        grid = self._grid(
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "cycles",
                "time_convention": "exp(+jwt)",
            }
        )
        with self.assertRaisesRegex(ValueError, "unsupported frequency unit"):
            isar_mode.form_isar(grid, legacy_metadata_attested=True)

    def test_semantically_unsafe_phase_reference_cannot_be_attested_away(self):
        for phase_reference in (
            "drifting phase center",
            "uncompensated range reference",
        ):
            with self.subTest(phase_reference=phase_reference):
                grid = self._grid(extra={"phase_reference": phase_reference})
                with self.assertRaisesRegex(ValueError, "not a verified fixed phase center"):
                    isar_mode.form_isar(grid, legacy_metadata_attested=True)

    def test_unrecognized_phase_reference_is_recorded_as_user_assumed(self):
        for phase_reference in ("unknown", "banana reference", "N/A"):
            with self.subTest(phase_reference=phase_reference):
                grid = self._grid(extra={"phase_reference": phase_reference})
                bands, _elapsed = isar_mode.form_isar(grid)
                self.assertIn(
                    "a fixed phase reference/center",
                    bands[0]["isar_contract_undeclared_fields"],
                )

    def test_explicit_near_field_and_bistatic_geometry_are_rejected(self):
        cases = (
            ("complex_field_domain", "near_field_scattering_amplitude", "far-field"),
            ("measurement_geometry", "bistatic", "monostatic"),
            (
                "measurement_geometry",
                "far-field quasi-monostatic",
                "verify and declare",
            ),
            (
                "measurement_geometry",
                "far-field pseudomonostatic",
                "verify and declare",
            ),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                grid = self._grid(
                    extra={"phase_reference": "fixed origin", key: value}
                )
                with self.assertRaisesRegex(ValueError, message):
                    isar_mode.form_isar(grid, legacy_metadata_attested=True)

    def test_unrecognized_geometry_is_recorded_but_known_incompatible_is_rejected(self):
        grid = self._grid(
            extra={
                "phase_reference": "fixed origin",
                "measurement_geometry": "banana geometry",
            }
        )
        bands, _elapsed = isar_mode.form_isar(grid)
        self.assertIn(
            "far-field monostatic acquisition geometry",
            bands[0]["isar_contract_undeclared_fields"],
        )

        incompatible = self._grid(
            extra={
                "phase_reference": "fixed origin",
                "measurement_geometry": "not monostatic; Fresnel zone",
            }
        )
        with self.assertRaisesRegex(ValueError, "geometry|far-field"):
            isar_mode.form_isar(incompatible)

    def test_explicit_far_field_monostatic_geometry_is_accepted(self):
        grid = self._grid(
            extra={
                "phase_reference": "fixed origin",
                "measurement_geometry": "far-field monostatic",
            }
        )
        bands, _elapsed = isar_mode.form_isar(grid)
        self.assertEqual(len(bands), 1)
        self.assertTrue(bands[0]["isar_contract_user_assumed"])

    def test_frequency_domain_metadata_is_not_mistaken_for_propagation_zone(self):
        grid = self._grid()
        grid.extra["measurement_domain"] = "frequency-domain"
        bands, _elapsed = isar_mode.form_isar(grid)
        self.assertEqual(len(bands), 1)

        grid.extra["measurement_domain"] = "near-field Fresnel"
        with self.assertRaisesRegex(ValueError, "far-field phase history"):
            isar_mode.form_isar(grid, legacy_metadata_attested=True)

    def test_physically_derived_far_field_domain_remains_imageable(self):
        grid = self._grid(
            extra={
                "phase_reference": "fixed origin",
                "measurement_geometry": "far-field monostatic",
                "complex_field_domain": "support_referenced_complex_difference",
            }
        )
        bands, _elapsed = isar_mode.form_isar(
            grid, legacy_metadata_attested=True
        )
        self.assertEqual(len(bands), 1)

    def test_bistatic_ss_coordinate_marker_is_rejected(self):
        grid = self._grid(
            extra={
                "phase_reference": "fixed origin",
                "fixed_incident_azimuth_deg": 0.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "bistatic acquisition"):
            isar_mode.form_isar(grid, legacy_metadata_attested=True)

    def test_uncompensated_motion_metadata_is_rejected(self):
        for declared in ("false", False, 0, "phase center drifting"):
            with self.subTest(declared=declared):
                grid = self._grid(
                    extra={
                        "phase_reference": "fixed origin",
                        "motion_compensation": declared,
                    }
                )
                with self.assertRaisesRegex(ValueError, "motion-compensated phase center"):
                    isar_mode.form_isar(grid, legacy_metadata_attested=True)

    def test_phase_center_motion_uses_inverse_boolean_polarity(self):
        for declared in (False, 0, "no motion", "no drift"):
            with self.subTest(stable=declared):
                grid = self._grid()
                grid.extra.pop("motion_compensation")
                grid.extra["phase_center_motion"] = declared
                bands, _elapsed = isar_mode.form_isar(grid)
                self.assertEqual(len(bands), 1)

        for declared in (True, 1, "moving"):
            with self.subTest(unsafe=declared):
                grid = self._grid()
                grid.extra.pop("motion_compensation")
                grid.extra["phase_center_motion"] = declared
                with self.assertRaisesRegex(
                    ValueError, "motion-compensated phase center"
                ):
                    isar_mode.form_isar(
                        grid, legacy_metadata_attested=True
                    )

    def test_unrecognized_motion_metadata_is_recorded_as_user_assumed(self):
        for declaration in ("banana state", "N/A"):
            with self.subTest(declaration=declaration):
                grid = self._grid(
                    extra={
                        "phase_reference": "fixed origin",
                        "motion_compensation": declaration,
                    }
                )
                bands, _elapsed = isar_mode.form_isar(grid)
                self.assertIn(
                    "a stable or motion-compensated phase center",
                    bands[0]["isar_contract_undeclared_fields"],
                )

    def test_missing_geometry_metadata_remains_legacy_compatible(self):
        grid = self._grid(
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
            },
            extra={},
        )
        bands, _elapsed = isar_mode.form_isar(grid)
        self.assertEqual(len(bands), 1)
        self.assertTrue(bands[0]["isar_contract_user_assumed"])

    def test_phase_and_time_alone_record_geometry_and_motion_assumptions(self):
        grid = self._grid(
            extra={
                "phase_reference": "fixed origin",
                "range_phase_convention": "S~exp(-j*2*k*R)",
            }
        )
        bands, _elapsed = isar_mode.form_isar(grid)
        self.assertEqual(len(bands), 1)
        undeclared = bands[0]["isar_contract_undeclared_fields"]
        self.assertIn("far-field monostatic acquisition geometry", undeclared)
        self.assertIn("a stable or motion-compensated phase center", undeclared)

    def test_explicit_opposite_range_phase_requires_both_axis_flips(self):
        grid = self._grid()
        grid.extra["range_phase_convention"] = "S proportional to exp(+j*2*k*R)"
        with self.assertRaisesRegex(ValueError, "enable both Flip X and Flip Y"):
            isar_mode.form_isar(grid)
        with self.assertRaisesRegex(ValueError, "enable both Flip X and Flip Y"):
            isar_mode.form_isar(grid, flip_y=True)
        bands, _elapsed = isar_mode.form_isar(
            grid, flip_x=True, flip_y=True
        )
        self.assertEqual(len(bands), 1)

    def test_contradictory_range_phase_declarations_are_rejected(self):
        grid = self._grid()
        grid.extra["phase_law"] = "S~exp(+j*2*k*R)"
        with self.assertRaisesRegex(ValueError, "contradictory two-way range-phase"):
            isar_mode.form_isar(grid, flip_x=True, flip_y=True)

    def test_two_sample_hann_aperture_has_nonzero_coherent_gain(self):
        theta = np.deg2rad(np.asarray([-1.0, 1.0]))
        frequency = np.asarray([9.0e9, 9.1e9])
        formed = isar_mode._compute_band_polar_format(
            "Hanning",
            np.ones((2, 2), dtype=np.complex64),
            theta,
            frequency,
            1.0e8,
            1.0,
        )
        self.assertNotIsInstance(formed, str)
        self.assertAlmostEqual(float(np.abs(formed[0]).max()), 1.0, places=6)

    def test_scene_extent_guard_is_independent_of_display_length_unit(self):
        grid = self._grid(
            azimuths=(-0.02, -0.01, 0.0, 0.01),
            frequencies=(9.0, 9.0001, 9.0002, 9.0003),
        )
        metric, _ = isar_mode.form_isar(grid, length_unit="m")
        inches, _ = isar_mode.form_isar(grid, length_unit="in")
        self.assertEqual(len(metric), 1)
        self.assertEqual(len(inches), 1)
        self.assertAlmostEqual(
            float(np.ptp(inches[0]["y_range"]) / np.ptp(metric[0]["y_range"])),
            1.0 / 0.0254,
            places=5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
