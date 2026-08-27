"""Synthetic physics regressions for ISAR image formation."""

import unittest

import numpy as np

from grim_dataset import RcsGrid
from grim_headless import form_isar
from plot_modes import isar_mode


C0 = 299_792_458.0


class TestIsarPhysics(unittest.TestCase):
    def setUp(self):
        isar_mode._PREPROCESS_CACHE.clear()
        isar_mode._PREPROCESS_CACHE_BYTES = 0
        self.theta = np.deg2rad(np.linspace(-10.0, 10.0, 401))
        self.frequency = np.linspace(8.0e9, 12.0e9, 401)

    def _image(self, samples, theta=None, frequency=None, window="Rectangular", weights=None,
               elevation=0.0):
        theta = self.theta if theta is None else theta
        frequency = self.frequency if frequency is None else frequency
        result = isar_mode._compute_band_polar_format(
            window, samples, theta, frequency, float(np.mean(np.diff(frequency))), 1.0,
            sample_weights=weights, elevation_deg=elevation,
        )
        self.assertNotIsInstance(result, str)
        return result

    def test_all_windows_preserve_unit_point_peak(self):
        samples = np.ones((self.theta.size, self.frequency.size), dtype=np.complex64)
        for window in (
            "Rectangular", "Hanning", "Hamming", "Blackman",
            "Blackman-Harris", "Kaiser β=15",
        ):
            with self.subTest(window=window):
                image, _, _ = self._image(samples, window=window)
                self.assertAlmostEqual(float(np.abs(image).max()), 1.0, places=5)

    def test_missing_sample_weights_preserve_coherent_gain(self):
        weights = np.ones((self.theta.size, self.frequency.size), dtype=np.float32)
        weights[::5, ::3] = 0.0
        samples = weights.astype(np.complex64)
        image, _, _ = self._image(samples, window="Hanning", weights=weights)
        self.assertAlmostEqual(float(np.abs(image).max()), 1.0, places=5)

    def test_cartesian_pfa_focuses_off_center_point(self):
        ones = np.ones((self.theta.size, self.frequency.size), dtype=np.complex64)
        _, q, v = isar_mode._pfa_regrid_cartesian(ones, self.theta, self.frequency)
        _, x_axis, y_axis = self._image(ones, theta=q, frequency=v)
        x0 = float(x_axis[np.abs(x_axis - 2.13).argmin()])
        y0 = float(y_axis[np.abs(y_axis - 3.07).argmin()])
        phase_history = np.exp(
            -1j * 4.0 * np.pi / C0 * self.frequency[None, :]
            * (x0 * np.sin(self.theta[:, None]) + y0 * np.cos(self.theta[:, None]))
        ).astype(np.complex64)

        accurate, q, v = isar_mode._pfa_regrid_cartesian(
            phase_history, self.theta, self.frequency
        )
        accurate_image, xa, ya = self._image(accurate, theta=q, frequency=v)
        fast = isar_mode._pfa_regrid_azimuth(
            phase_history, self.theta, self.frequency
        )
        fast_image, _, _ = self._image(fast)

        peak = np.unravel_index(np.abs(accurate_image).argmax(), accurate_image.shape)
        self.assertAlmostEqual(float(xa[peak[0]]), x0, places=6)
        self.assertAlmostEqual(float(ya[peak[1]]), y0, places=6)
        accurate_concentration = float(np.abs(accurate_image[peak]) ** 2 / np.sum(np.abs(accurate_image) ** 2))
        fast_concentration = float(np.max(np.abs(fast_image)) ** 2 / np.sum(np.abs(fast_image) ** 2))
        self.assertGreater(accurate_concentration, 0.85)
        self.assertGreater(accurate_concentration, 10.0 * fast_concentration)

    def test_cartesian_pfa_axis_preserves_true_u_spacing(self):
        ones = np.ones((self.theta.size, self.frequency.size), dtype=np.complex64)
        _, axis_q, v = isar_mode._pfa_regrid_cartesian(
            ones, self.theta, self.frequency
        )
        fc = float(np.mean(self.frequency))
        psi = self.theta - float(np.mean(self.theta))
        ratio_min = float(self.frequency[0] / fc)
        raw_q = np.linspace(
            ratio_min * np.sin(psi[0]),
            ratio_min * np.sin(psi[-1]),
            self.theta.size,
        )
        np.testing.assert_allclose(
            float(np.mean(v) * np.mean(np.diff(axis_q))),
            float(fc * np.mean(np.diff(raw_q))),
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_elevation_projects_horizontal_image_axes(self):
        samples = np.ones((self.theta.size, self.frequency.size), dtype=np.complex64)
        _, x0, y0 = self._image(samples, elevation=0.0)
        _, x60, y60 = self._image(samples, elevation=60.0)
        self.assertAlmostEqual(float(np.ptp(x60) / np.ptp(x0)), 2.0, places=6)
        self.assertAlmostEqual(float(np.ptp(y60) / np.ptp(y0)), 2.0, places=6)

    def test_circular_unwrap_keeps_zero_degree_aperture_contiguous(self):
        values = np.asarray([355.0, 358.0, 0.0, 2.0, 5.0])
        unwrapped = np.sort(isar_mode._unwrap_degrees(values, 0.0))
        np.testing.assert_allclose(unwrapped, [-5.0, -2.0, 0.0, 2.0, 5.0])
        self.assertEqual(float(np.ptp(unwrapped)), 10.0)

    def test_headless_formation_uses_same_full_resolution_path(self):
        azimuth = np.linspace(-2.0, 2.0, 17)
        frequency = np.linspace(8.0, 9.0, 19)
        samples = np.ones((17, 1, 19, 1), dtype=np.complex64)
        grid = RcsGrid(
            azimuth, [0.0], frequency, ["VV"], rcs=samples,
            units={"frequency": "GHz"},
        )
        bands, elapsed = form_isar(grid, reconstruction="accurate", window="Hamming")
        self.assertEqual(len(bands), 1)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(bands[0]["magnitude"].shape, (256, 256))
        self.assertAlmostEqual(float(bands[0]["magnitude"].max()), 1.0, places=5)

    def test_preprocess_cache_invalidates_after_in_place_data_change(self):
        azimuth = np.asarray([-15.0, -5.0, 5.0, 15.0])
        frequency_ghz = np.asarray([9.0, 9.1, 9.2, 9.3])
        power = np.ones((4, 1, 4, 1), dtype=np.float32)
        grid = RcsGrid(
            azimuth,
            [0.0],
            frequency_ghz,
            ["VV"],
            rcs_power=power,
            rcs_phase=np.zeros_like(power),
            units={"frequency": "GHz"},
        )
        args = (
            grid,
            "Hanning",
            list(range(4)),
            list(range(4)),
            0,
            0,
            frequency_ghz * 1.0e9,
            1.0e8,
            1.0,
        )
        first = isar_mode._compute_band(*args)
        cached = isar_mode._compute_band(*args)
        self.assertFalse(first["preprocess_cache_hit"])
        self.assertTrue(cached["preprocess_cache_hit"])

        grid.rcs_phase[:] = np.asarray([0.0, 0.3, 1.1, 2.7]).reshape(1, 1, 4, 1)
        changed = isar_mode._compute_band(*args)
        self.assertFalse(changed["preprocess_cache_hit"])
        self.assertGreater(
            float(np.max(np.abs(first["magnitude"] - changed["magnitude"]))),
            0.1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
