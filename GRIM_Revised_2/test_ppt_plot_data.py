"""Focused physics/unit regressions for PowerPoint plot-data extraction."""

from __future__ import annotations

import unittest

import numpy as np

from grim_dataset import C0, GRIM_GC_CONVENTION, RcsGrid
from ppt_plot_data import (
    DUAL_COPOLARIZATION,
    NamedGrid,
    build_azimuth_specs,
    build_elevation_specs,
    build_frequency_spec,
    build_frequency_specs,
    get_plot_availability,
)


def _grid(
    *,
    azimuths=(-90.0, 0.0, 90.0),
    elevations=(0.0,),
    frequencies=(1.0, 2.0),
    polarizations=("VV",),
    power=None,
    phase=None,
    angle_unit="deg",
    frequency_unit="GHz",
    log_unit="dBsm",
    quantity="sigma_3d",
    coordinate_system="conic",
    gc_convention=None,
    phase_reference="origin=(0,0,0); exp(-jkr)",
) -> RcsGrid:
    shape = (
        len(azimuths),
        len(elevations),
        len(frequencies),
        len(polarizations),
    )
    power_values = np.ones(shape, dtype=float)
    if power is not None:
        power_values = np.broadcast_to(np.asarray(power, dtype=float), shape).copy()
    phase_values = np.zeros(shape, dtype=float)
    if phase is not None:
        phase_values = np.broadcast_to(np.asarray(phase, dtype=float), shape).copy()
    units = {
        "azimuth": angle_unit,
        "elevation": angle_unit,
        "frequency": frequency_unit,
        "rcs_log_unit": log_unit,
        "rcs_linear_quantity": quantity,
        "angular_coordinate_system": coordinate_system,
    }
    if gc_convention is not None:
        units["great_circle_coordinate_convention"] = gc_convention
    extra = {"phase_reference": phase_reference} if phase_reference else {}
    return RcsGrid(
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs_power=power_values,
        rcs_phase=phase_values,
        units=units,
        extra=extra,
    )


class PptPlotDataTests(unittest.TestCase):
    def test_dense_tolerant_axis_intersection_preserves_unique_sorted_values(self):
        axis = np.arange(36_000, dtype=float) * 0.01
        first = _grid(
            azimuths=axis,
            frequencies=(1.0,),
        )
        second = _grid(
            azimuths=axis + 5.0e-7,
            frequencies=(1.0,),
        )
        availability = get_plot_availability(
            [("First", first), ("Second", second)],
            evaluate_phase=False,
        )
        self.assertEqual(len(availability.azimuths), axis.size)
        self.assertEqual(availability.azimuths[0], 0.0)
        self.assertAlmostEqual(availability.azimuths[-1], float(axis[-1]))

    def test_azimuth_dbsm_overlays_sort_each_swept_axis_and_match_fixed_axes(self):
        first = _grid(
            azimuths=(90.0, -90.0, 0.0),
            frequencies=(2.0, 1.0),
            power=np.asarray([100.0, 1.0, 10.0]).reshape(3, 1, 1, 1),
        )
        second = _grid(
            azimuths=(0.0, 90.0, -90.0),
            frequencies=(1.0 + 5.0e-7, 2.0 + 5.0e-7),
            power=np.asarray([4.0, 16.0, 1.0]).reshape(3, 1, 1, 1),
        )
        availability = get_plot_availability(
            [NamedGrid("Reference", first), NamedGrid("Comparison", second)]
        )
        self.assertEqual(availability.frequencies, (1.0, 2.0))

        spec = build_azimuth_specs(
            [("Reference", first), ("Comparison", second)],
            frequencies=[1.0],
            elevation=0.0,
            polarization="VV",
        )[0]
        self.assertEqual(spec.kind, "azimuth_rect")
        self.assertEqual(spec.y_label, "RCS (dBsm)")
        self.assertEqual(spec.series[0].x, (-90.0, 0.0, 90.0))
        np.testing.assert_allclose(spec.series[0].y, [0.0, 10.0, 20.0])
        self.assertEqual(spec.series[1].x, (-90.0, 0.0, 90.0))
        np.testing.assert_allclose(
            spec.series[1].y,
            [0.0, 10.0 * np.log10(4.0), 10.0 * np.log10(16.0)],
        )

    def test_elevation_sweep_sorts_native_axis_without_interpolation(self):
        grid = _grid(
            azimuths=(15.0,),
            elevations=(30.0, -30.0, 0.0),
            frequencies=(2.0,),
            power=np.asarray([100.0, 1.0, 10.0]).reshape(1, 3, 1, 1),
        )
        spec = build_elevation_specs(
            [("Elevation", grid)],
            frequencies=(2.0,),
            azimuth=15.0,
            polarization="VV",
        )[0]
        self.assertEqual(spec.kind, "elevation")
        self.assertEqual(spec.x_label, "Elevation (deg)")
        self.assertIn("Azimuth 15 deg", spec.title)
        self.assertEqual(spec.series[0].x, (-30.0, 0.0, 30.0))
        np.testing.assert_allclose(spec.series[0].y, (0.0, 10.0, 20.0))

    def test_elevation_sweep_requires_a_real_swept_axis(self):
        with self.assertRaisesRegex(ValueError, "at least two stored elevation"):
            build_elevation_specs(
                [("Single cut", _grid(elevations=(0.0,)))],
                frequencies=(1.0,),
                azimuth=0.0,
                polarization="VV",
            )

    def test_frequency_dbke_passes_sorted_frequency_vector_to_conversion(self):
        frequencies = np.asarray([2.0, 1.0])
        frequency_hz = frequencies * 1.0e9
        # sigma_2d chosen so 10log10(k*sigma) is exactly 0 dBke.
        power = (C0 / (2.0 * np.pi * frequency_hz)).reshape(1, 1, 2, 1)
        grid = _grid(
            azimuths=(0.0,),
            frequencies=frequencies,
            power=power,
            log_unit="dBke",
            quantity="sigma_2d",
        )
        spec = build_frequency_spec(
            [("2-D", grid)],
            azimuth=0.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertEqual(spec.series[0].x, (1.0, 2.0))
        np.testing.assert_allclose(spec.series[0].y, [0.0, 0.0], atol=1.0e-12)
        self.assertEqual(spec.y_label, "RCS (dBke)")

    def test_native_rad_and_hz_axes_convert_only_for_display(self):
        grid = _grid(
            azimuths=(-np.pi / 2.0, 0.0),
            elevations=(np.pi / 6.0,),
            frequencies=(2.0e9, 1.0e9),
            angle_unit="rad",
            frequency_unit="Hz",
        )
        azimuth_spec = build_azimuth_specs(
            [("Radians", grid)],
            frequencies=[1.0e9],
            elevation=np.pi / 6.0,
            polarization="VV",
            angle_display_unit="deg",
            frequency_display_unit="GHz",
        )[0]
        np.testing.assert_allclose(azimuth_spec.series[0].x, [-90.0, 0.0])
        self.assertIn("1 GHz", azimuth_spec.title)
        self.assertIn("Elevation 30 deg", azimuth_spec.title)

        frequency_spec = build_frequency_spec(
            [("Radians", grid)],
            azimuth=-np.pi / 2.0,
            elevation=np.pi / 6.0,
            polarization="VV",
            angle_display_unit="deg",
            frequency_display_unit="GHz",
        )
        self.assertEqual(frequency_spec.series[0].x, (1.0, 2.0))
        self.assertIn("Azimuth -90 deg", frequency_spec.title)

    def test_multi_frequency_specs_share_prepared_azimuth_coordinates(self):
        grid = _grid(frequencies=(1.0, 2.0))
        specs = build_azimuth_specs(
            (NamedGrid("shared", grid),),
            frequencies=(1.0, 2.0),
            elevation=0.0,
            polarization="VV",
        )
        self.assertIs(specs[0].series[0].x, specs[1].series[0].x)

    def test_dual_copolarization_emits_separate_unique_plots_in_one_request(self):
        power = np.ones((3, 1, 2, 2), dtype=float)
        power[:, :, :, 1] = 100.0
        grid = _grid(
            frequencies=(1.0, 2.0),
            polarizations=("VV", "HH"),
            power=power,
        )

        azimuth_specs = build_azimuth_specs(
            [("Dual", grid)],
            frequencies=(1.0, 2.0),
            elevation=0.0,
            polarization=DUAL_COPOLARIZATION,
        )
        self.assertEqual(len(azimuth_specs), 4)
        self.assertTrue(all("| VV" in spec.title for spec in azimuth_specs[:2]))
        self.assertTrue(all("| HH" in spec.title for spec in azimuth_specs[2:]))
        self.assertEqual(len({spec.plot_id for spec in azimuth_specs}), 4)
        np.testing.assert_allclose(azimuth_specs[0].series[0].y, 0.0)
        np.testing.assert_allclose(azimuth_specs[2].series[0].y, 20.0)

        frequency_specs = build_frequency_specs(
            [("Dual", grid)],
            azimuth=0.0,
            elevation=0.0,
            polarization=("VV", "HH"),
        )
        self.assertEqual(len(frequency_specs), 2)
        self.assertIn("| VV", frequency_specs[0].title)
        self.assertIn("| HH", frequency_specs[1].title)
        np.testing.assert_allclose(frequency_specs[0].series[0].y, 0.0)
        np.testing.assert_allclose(frequency_specs[1].series[0].y, 20.0)

    def test_dual_copolarization_reports_missing_common_component(self):
        with self.assertRaisesRegex(ValueError, "HH.*not common"):
            build_frequency_specs(
                [("VV only", _grid())],
                azimuth=0.0,
                elevation=0.0,
                polarization=DUAL_COPOLARIZATION,
            )

    def test_frequency_azimuth_band_is_display_domain_percentile_per_frequency(self):
        # Columns are stored in native frequency order (2, 1 GHz).  The first
        # overlay has azimuth distributions [0, 10, 20] and [10, 20, 30] dB;
        # the second is 10 dB higher at every sample.
        db_values = np.asarray(
            [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]],
            dtype=float,
        )
        first = _grid(
            azimuths=(-10.0, 0.0, 10.0),
            frequencies=(2.0, 1.0),
            power=np.power(10.0, db_values / 10.0).reshape(3, 1, 2, 1),
        )
        second = _grid(
            azimuths=(-10.0, 0.0, 10.0),
            frequencies=(2.0, 1.0),
            power=np.power(10.0, (db_values + 10.0) / 10.0).reshape(3, 1, 2, 1),
        )
        spec = build_frequency_spec(
            [("First", first), ("Second", second)],
            azimuth=None,
            azimuth_band=(-10.0, 10.0),
            azimuth_percentile=90.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertEqual(spec.series[0].x, (1.0, 2.0))
        np.testing.assert_allclose(spec.series[0].y, (28.0, 18.0))
        np.testing.assert_allclose(spec.series[1].y, (38.0, 28.0))
        self.assertIn("P90 across Azimuth [-10, 10] deg", spec.title)

    def test_frequency_band_retains_frequency_nan_gaps_and_rejects_phase(self):
        power = np.asarray(
            [[np.nan, 1.0], [np.nan, 100.0], [np.nan, 10000.0]],
            dtype=float,
        ).reshape(3, 1, 2, 1)
        grid = _grid(frequencies=(1.0, 2.0), power=power)
        spec = build_frequency_spec(
            [("Gaps", grid)],
            azimuth=None,
            azimuth_band=(-90.0, 90.0),
            azimuth_percentile=50.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertTrue(np.isnan(spec.series[0].y[0]))
        self.assertAlmostEqual(spec.series[0].y[1], 20.0)

        with self.assertRaisesRegex(ValueError, "unavailable for wrapped phase"):
            build_frequency_spec(
                [("Phase", grid)],
                azimuth=None,
                azimuth_band=(-90.0, 90.0),
                azimuth_percentile=50.0,
                elevation=0.0,
                polarization="VV",
                quantity="phase",
            )

    def test_frequency_band_supports_seam_wrap_and_requires_common_samples(self):
        grid = _grid(
            azimuths=(-180.0, -170.0, 0.0, 170.0),
            frequencies=(1.0,),
            power=np.asarray([1.0, 10.0, 1.0e6, 1000.0]).reshape(4, 1, 1, 1),
        )
        spec = build_frequency_spec(
            [("Wrapped", grid)],
            azimuth=None,
            azimuth_band=(170.0, -170.0),
            azimuth_percentile=50.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertAlmostEqual(spec.series[0].y[0], 10.0)
        self.assertIn("(wrap)", spec.title)

        with self.assertRaisesRegex(ValueError, "contains no samples common"):
            build_frequency_spec(
                [("Wrapped", grid)],
                azimuth=None,
                azimuth_band=(20.0, 30.0),
                azimuth_percentile=90.0,
                elevation=0.0,
                polarization="VV",
            )

    def test_frequency_band_counts_periodic_seam_once_and_prefers_upper_alias(self):
        # 0/360 and -pi/+pi are the same physical direction.  The upper
        # stored representation wins, matching the SENTRi import policy, and
        # the direction contributes only once to the percentile.
        degree_grid = _grid(
            azimuths=(0.0, 90.0, 180.0, 270.0, 360.0),
            frequencies=(1.0,),
            power=np.power(10.0, np.asarray([0, 10, 20, 30, 40]) / 10.0).reshape(
                5, 1, 1, 1
            ),
        )
        degree_spec = build_frequency_spec(
            [("Degrees", degree_grid)],
            azimuth=None,
            azimuth_band=(0.0, 360.0),
            azimuth_percentile=50.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertAlmostEqual(degree_spec.series[0].y[0], 25.0)
        self.assertIn("(4 common samples)", degree_spec.title)

        radian_grid = _grid(
            azimuths=(-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi),
            frequencies=(1.0,),
            power=np.power(10.0, np.asarray([0, 10, 20, 30, 40]) / 10.0).reshape(
                5, 1, 1, 1
            ),
            angle_unit="rad",
        )
        radian_spec = build_frequency_spec(
            [("Radians", radian_grid)],
            azimuth=None,
            azimuth_band=(-np.pi, np.pi),
            azimuth_percentile=50.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertAlmostEqual(radian_spec.series[0].y[0], 25.0)
        self.assertIn("[-180, 180] deg", radian_spec.title)

    def test_frequency_band_rejects_limits_from_a_different_azimuth_convention(self):
        grid = _grid(
            azimuths=(-180.0, -170.0, 0.0, 170.0, 180.0),
            frequencies=(1.0,),
        )
        with self.assertRaisesRegex(ValueError, "outside the stored azimuth range"):
            build_frequency_spec(
                [("Signed convention", grid)],
                azimuth=None,
                azimuth_band=(350.0, 10.0),
                azimuth_percentile=90.0,
                elevation=0.0,
                polarization="VV",
            )

    def test_phase_uses_stored_values_and_reference_differences_are_notes(self):
        phase = np.asarray([0.0, np.nan, np.pi]).reshape(3, 1, 1, 1)
        grid = _grid(frequencies=(1.0,), phase=phase)
        spec = build_azimuth_specs(
            [("Phase", grid)],
            frequencies=[1.0],
            elevation=0.0,
            polarization="VV",
            quantity="phase",
        )[0]
        self.assertEqual(spec.y_label, "Phase (deg)")
        y = np.asarray(spec.series[0].y)
        np.testing.assert_allclose(y[[0, 2]], [0.0, 180.0])
        self.assertTrue(np.isnan(spec.series[0].y[1]))

        other_reference = _grid(
            frequencies=(1.0,), phase=phase, phase_reference="different origin"
        )
        availability = get_plot_availability(
            [("Phase", grid), ("Other", other_reference)]
        )
        self.assertTrue(availability.phase_available)
        self.assertIn("different phase references", availability.phase_reason)
        specs = build_azimuth_specs(
            [("Phase", grid), ("Other", other_reference)],
            frequencies=[1.0],
            elevation=0.0,
            polarization="VV",
            quantity="phase",
        )
        self.assertEqual(len(specs[0].series), 2)

    def test_blank_phase_reference_is_a_nonblocking_availability_note(self):
        grid = _grid(frequencies=(1.0,), phase_reference="")
        availability = get_plot_availability([("Unknown reference", grid)])
        self.assertTrue(availability.phase_available)
        self.assertIn("unspecified", availability.phase_reason)
        spec = build_frequency_spec(
            [("Unknown reference", grid)],
            azimuth=0.0,
            elevation=0.0,
            polarization="VV",
            quantity="phase",
        )
        self.assertEqual(spec.y_label, "Phase (deg)")

    def test_phase_export_still_requires_at_least_one_finite_phase_sample(self):
        grid = _grid(frequencies=(1.0,), phase=np.nan, phase_reference="")
        availability = get_plot_availability([("No phase", grid)])
        self.assertFalse(availability.phase_available)
        with self.assertRaisesRegex(ValueError, "finite stored phase"):
            build_frequency_spec(
                [("No phase", grid)],
                azimuth=0.0,
                elevation=0.0,
                polarization="VV",
                quantity="phase",
            )

    def test_all_nan_cut_fails_but_partial_nan_magnitude_gap_survives(self):
        partial = _grid(
            frequencies=(1.0,),
            power=np.asarray([1.0, np.nan, 100.0]).reshape(3, 1, 1, 1),
        )
        spec = build_azimuth_specs(
            [("Partial", partial)],
            frequencies=[1.0],
            elevation=0.0,
            polarization="VV",
        )[0]
        self.assertTrue(np.isnan(spec.series[0].y[1]))
        y = np.asarray(spec.series[0].y)
        np.testing.assert_allclose(y[[0, 2]], [0.0, 20.0])

        empty = _grid(frequencies=(1.0,), power=np.nan)
        with self.assertRaisesRegex(ValueError, "no finite magnitude samples"):
            build_azimuth_specs(
                [("Empty", empty)],
                frequencies=[1.0],
                elevation=0.0,
                polarization="VV",
            )

    def test_frequency_cut_is_direct_exact_and_sorted_without_interpolation(self):
        grid = _grid(
            azimuths=(20.0, 10.0),
            frequencies=(3.0, 1.0, 2.0),
            power=np.asarray(
                [1000.0, 10.0, 100.0, 1.0, 1.0, 1.0]
            ).reshape(2, 1, 3, 1),
        )
        spec = build_frequency_spec(
            [("Exact", grid)],
            azimuth=20.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertEqual(spec.series[0].x, (1.0, 2.0, 3.0))
        np.testing.assert_allclose(spec.series[0].y, [10.0, 20.0, 30.0])
        with self.assertRaisesRegex(ValueError, "no interpolation is performed"):
            build_frequency_spec(
                [("Exact", grid)],
                azimuth=15.0,
                elevation=0.0,
                polarization="VV",
            )

    def test_physical_metadata_and_common_axis_mismatches_are_actionable(self):
        three_d = _grid(frequencies=(1.0,), log_unit="dBsm", quantity="sigma_3d")
        two_d = _grid(frequencies=(1.0,), log_unit="dBke", quantity="sigma_2d")
        with self.assertRaisesRegex(ValueError, "physically compatible.*linear quantity mismatch"):
            get_plot_availability([("3-D", three_d), ("2-D", two_d)])

        radians = _grid(
            azimuths=(-np.pi / 2.0, 0.0, np.pi / 2.0),
            frequencies=(1.0,),
            angle_unit="rad",
        )
        with self.assertRaisesRegex(ValueError, "azimuth unit mismatch"):
            get_plot_availability([("Degrees", three_d), ("Radians", radians)])

        offset = _grid(azimuths=(-80.0, 10.0, 100.0), frequencies=(1.0,))
        availability = get_plot_availability([("A", three_d), ("B", offset)])
        self.assertEqual(availability.azimuths, ())
        with self.assertRaisesRegex(ValueError, "common azimuth"):
            build_frequency_spec(
                [("A", three_d), ("B", offset)],
                azimuth=0.0,
                elevation=0.0,
                polarization="VV",
            )

    def test_unmarked_legacy_great_circle_polar_uses_stored_angles_with_note(self):
        legacy = _grid(
            frequencies=(1.0,),
            coordinate_system="great_circle",
            gc_convention=None,
        )
        availability = get_plot_availability([("Legacy PTM", legacy)])
        self.assertTrue(availability.polar_available)
        self.assertIn("stored aspect angles", availability.polar_reason)
        legacy_polar = build_azimuth_specs(
            [("Legacy PTM", legacy)],
            frequencies=[1.0],
            elevation=0.0,
            polarization="VV",
            kind="azimuth_polar",
        )[0]
        self.assertEqual(legacy_polar.kind, "azimuth_polar")
        # Rectangular display does not assert an unknown compass orientation.
        self.assertEqual(
            build_azimuth_specs(
                [("Legacy PTM", legacy)],
                frequencies=[1.0],
                elevation=0.0,
                polarization="VV",
                kind="azimuth_rect",
            )[0].kind,
            "azimuth_rect",
        )

        marked = _grid(
            frequencies=(1.0,),
            coordinate_system="great_circle",
            gc_convention=GRIM_GC_CONVENTION,
        )
        polar = build_azimuth_specs(
            [("GRIM GC", marked)],
            frequencies=[1.0],
            elevation=0.0,
            polarization="VV",
            kind="azimuth_polar",
        )[0]
        self.assertEqual(polar.kind, "azimuth_polar")
        self.assertEqual(polar.x_label, "Aspect (deg)")
        self.assertIn("Pitch 0 deg", polar.title)

        frequency = build_frequency_spec(
            [("GRIM GC", marked)],
            azimuth=0.0,
            elevation=0.0,
            polarization="VV",
        )
        self.assertIn("Aspect 0 deg", frequency.title)
        self.assertIn("Pitch 0 deg", frequency.title)


if __name__ == "__main__":
    unittest.main()
