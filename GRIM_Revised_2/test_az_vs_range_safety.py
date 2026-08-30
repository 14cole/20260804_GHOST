"""Sampling-safety regressions for azimuth versus down-range processing."""

from __future__ import annotations

import unittest

import numpy as np

from plot_modes.az_vs_range_mode import _prepare_uniform_frequency_history


class TestAzimuthRangeSamplingSafety(unittest.TestCase):
    def test_disjoint_frequency_band_stays_zero_weighted(self):
        frequency = np.asarray([8.0, 8.1, 10.0, 10.1]) * 1.0e9
        history = np.ones((2, 4), dtype=np.complex64)
        target, resampled, weights, info = _prepare_uniform_frequency_history(
            frequency, history
        )
        gap = (target > 8.1e9) & (target < 10.0e9)
        self.assertEqual(info["gap_count"], 1)
        self.assertTrue(np.any(gap))
        self.assertTrue(np.all(weights[:, gap] == 0.0))
        self.assertTrue(np.all(resampled[:, gap] == 0.0))

    def test_interpolation_touching_unknown_phase_is_not_observed(self):
        frequency = np.asarray([9.0, 9.1, 9.2, 9.3]) * 1.0e9
        history = np.ones((2, 4), dtype=np.complex64)
        history[0, 1] = np.nan + 1j * np.nan
        target, resampled, weights, _info = _prepare_uniform_frequency_history(
            frequency, history
        )
        missing = int(np.flatnonzero(np.isclose(target, 9.1e9))[0])
        self.assertEqual(float(weights[0, missing]), 0.0)
        self.assertEqual(complex(resampled[0, missing]), 0.0j)
        self.assertTrue(np.all(weights[1] == 1.0))

    def test_pathological_gap_expansion_fails_before_allocation(self):
        frequency = np.asarray([1.0, 2.0, 1.0e12])
        history = np.ones((1000, 3), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "Select one contiguous band/aperture"):
            _prepare_uniform_frequency_history(frequency, history)


if __name__ == "__main__":
    unittest.main()
