import math
import unittest

from ibc.inverse_performance import band_performance, response_curve, search_progress


class InversePerformanceTests(unittest.TestCase):
    def test_broad_shallower_null_has_more_useful_band_than_deep_spike(self):
        frequencies = [1, 2, 3, 4, 5, 6, 7]
        narrow = band_performance(frequencies, [-2, -2, -2, -50, -2, -2, -2], -10, continuous=True)
        broad = band_performance(frequencies, [-2, -15, -15, -15, -15, -15, -2], -10, continuous=True)
        self.assertLess(narrow.deepest_db, broad.deepest_db)
        self.assertGreater(broad.widest_ghz, narrow.widest_ghz)
        self.assertGreater(broad.coverage_pct, narrow.coverage_pct)

    def test_nonuniform_grid_uses_interpolated_width_not_point_count(self):
        metric = band_performance([1, 2, 10], [-20, -20, 0], -10, continuous=True)
        self.assertEqual(metric.passing_bands, ((1, 6),))
        self.assertAlmostEqual(metric.coverage_pct, 500 / 9)
        self.assertEqual(metric.widest_ghz, 5)

    def test_disjoint_bands_are_not_added_as_one_wide_band(self):
        metric = band_performance([1, 2, 3, 4, 5], [-20, -20, 0, -20, -20], -10, continuous=True)
        self.assertEqual(metric.passing_bands, ((1, 2.5), (3.5, 5)))
        self.assertEqual(metric.widest_ghz, 1.5)
        self.assertEqual(metric.coverage_pct, 75)

    def test_discrete_and_single_frequency_have_no_inferred_bandwidth(self):
        metric = band_performance([1, 2, 10], [-20, -20, 0], -10, continuous=False)
        self.assertAlmostEqual(metric.coverage_pct, 200 / 3)
        self.assertIsNone(metric.widest_ghz)
        self.assertFalse(metric.passing_bands)
        single = band_performance([1], [-10], -10, continuous=True)
        self.assertEqual(single.coverage_pct, 100)
        self.assertIsNone(single.widest_ghz)

    def test_all_pass_all_fail_and_touching_threshold(self):
        for values, width, pct in [([-20, -10], 2, 100), ([0, 0], 0, 0), ([-10, 0], 0, 0)]:
            metric = band_performance([1, 3], values, -10, continuous=True)
            self.assertEqual(metric.widest_ghz, width)
            self.assertEqual(metric.coverage_pct, pct)
        metric = band_performance([1, 2, 3], [-20, -10, -20], -10, continuous=True)
        self.assertEqual(metric.passing_bands, ((1, 3),))

    def test_worst_case_envelope_is_conservative_at_each_frequency(self):
        samples = [[-30, -10, -20], [-8, -30, -15]]
        self.assertEqual(response_curve(samples), [-10, -8])
        self.assertEqual(response_curve(samples, 50), [-20, -15])
        self.assertEqual(response_curve(samples, 10), [-28, -27])

    def test_rejects_incomplete_nonfinite_and_duplicate_data(self):
        for frequencies, values in [([], []), ([1, 1], [-20, -10]), ([1], []), ([1], [math.nan]), ([-1], [-20])]:
            with self.assertRaises(ValueError):
                band_performance(frequencies, values, -10, continuous=True)
        for samples in [[[]], [[-20, math.inf]]]:
            with self.assertRaises(ValueError):
                response_curve(samples)
        with self.assertRaises(ValueError):
            response_curve([[-20]], 101)

    def test_best_so_far_history_includes_every_completed_score(self):
        scores, best = search_progress([-5, -9, -6, -10, -10])
        self.assertEqual(scores, [-5, -9, -6, -10, -10])
        self.assertEqual(best, [-5, -9, -9, -10, -10])
        self.assertEqual(search_progress([]), ([], []))
