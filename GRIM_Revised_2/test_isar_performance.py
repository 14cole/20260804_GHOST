"""Memory, cancellation, and streaming regressions for ISAR formation."""

import unittest
from unittest import mock
import weakref

import numpy as np

from grim_dataset import RcsGrid
from plot_modes import isar_mode


def _grid(azimuth, frequency):
    azimuth = np.asarray(azimuth, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    field = np.ones(
        (azimuth.size, 1, frequency.size, 1), dtype=np.complex64
    )
    return RcsGrid(
        azimuth,
        [0.0],
        frequency,
        ["VV"],
        rcs=field,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "angular_coordinate_system": "conic",
            "time_convention": "exp(+jwt)",
        },
        extra={
            "phase_reference": "fixed origin",
            "measurement_geometry": "far-field monostatic",
            "motion_compensation": "stable",
            "range_phase_convention": "S~exp(-j*2*k*R)",
        },
    )


class TestIsarWorkingSet(unittest.TestCase):
    def test_estimator_accounts_for_algorithm_and_retained_complex_output(self):
        fast = isar_mode._estimate_band_working_set_bytes(
            400, 300, reconstruction="fft", retain_complex=False
        )
        retained = isar_mode._estimate_band_working_set_bytes(
            400, 300, reconstruction="fft", retain_complex=True
        )
        sparse = isar_mode._estimate_band_working_set_bytes(
            400, 300, reconstruction="sparse", retain_complex=False
        )
        self.assertGreater(retained, fast)
        self.assertGreater(sparse, retained)
        self.assertGreater(fast, 400 * 300 * np.dtype(np.complex64).itemsize)

    def test_byte_preflight_reports_resident_results_and_action(self):
        with self.assertRaisesRegex(ValueError, "already retained") as caught:
            isar_mode._validate_isar_working_set(
                900,
                resident_bytes=200,
                limit_bytes=1000,
                operation="test ISAR",
            )
        self.assertIn("GRIM_ISAR_WORKING_SET_MB", str(caught.exception))

    def test_band_preflight_runs_before_complex_slice_allocation(self):
        grid = _grid(np.linspace(-2.0, 2.0, 9), np.linspace(8.0, 9.0, 9))
        with mock.patch.object(
            grid, "rcs_slice", wraps=grid.rcs_slice
        ) as slice_method, mock.patch.object(
            isar_mode, "_ISAR_WORKING_SET_LIMIT", 1024
        ):
            result = isar_mode._compute_band(
                grid,
                "Rectangular",
                list(range(9)),
                list(range(9)),
                0,
                0,
                np.linspace(8.0e9, 9.0e9, 9),
                1.25e8,
                1.0,
            )
        self.assertIsInstance(result, str)
        self.assertIn("working-set preflight blocked", result)
        slice_method.assert_not_called()

    def test_retained_array_counter_does_not_double_count_views(self):
        base = np.zeros((8, 8), dtype=np.float32)
        result = {"magnitude": base, "view": base[::-1]}
        self.assertEqual(isar_mode._result_array_bytes([result]), base.nbytes)


class TestIsarCancellationAndStreaming(unittest.TestCase):
    def test_source_hash_can_cancel_between_rows(self):
        grid = _grid(np.linspace(-10.0, 10.0, 65), np.linspace(8.0, 9.0, 17))
        checks = 0

        def cancel_soon():
            nonlocal checks
            checks += 1
            # Initial/metadata/source checks consume the first three calls;
            # call five is the checkpoint before hashing row 16.
            return checks >= 5

        token = isar_mode._selected_data_token(
            grid,
            list(range(65)),
            0,
            list(range(17)),
            0,
            cancel_check=cancel_soon,
        )
        self.assertIsNone(token)
        self.assertEqual(checks, 5)

    def test_fft_path_checks_cancellation_before_work(self):
        samples = np.ones((8, 8), dtype=np.complex64)
        result = isar_mode._compute_band_polar_format(
            "Rectangular",
            samples,
            np.deg2rad(np.linspace(-1.0, 1.0, 8)),
            np.linspace(8.0e9, 9.0e9, 8),
            1.0e8,
            1.0,
            cancel_check=lambda: True,
        )
        self.assertEqual(result, "ISAR computation superseded.")

    def test_wide_composite_releases_each_sublook_before_forming_next(self):
        grid = _grid(np.linspace(-20.0, 20.0, 41), np.linspace(8.0, 9.0, 9))
        magnitude_refs = []
        alive_at_entry = []

        def fake_band(*_args, **_kwargs):
            alive_at_entry.append(
                sum(reference() is not None for reference in magnitude_refs)
            )
            magnitude = np.ones((32, 32), dtype=np.float32)
            magnitude_refs.append(weakref.ref(magnitude))
            return {
                "magnitude": magnitude,
                "x_range": np.linspace(-10.0, 10.0, 32),
                "y_range": np.linspace(-10.0, 10.0, 32),
                "phase_coverage": 1.0,
            }

        with mock.patch.object(
            isar_mode, "_COMPOSITE_GRID_SIDE", 32
        ), mock.patch.object(isar_mode, "_compute_band", side_effect=fake_band):
            result = isar_mode._compute_band_composite(
                grid,
                "Rectangular",
                list(range(41)),
                list(range(9)),
                0,
                0,
                np.linspace(8.0e9, 9.0e9, 9),
                1.25e8,
                1.0,
                recon="fft",
                l1_strength=0.05,
                l1_iters=10,
                elevation_deg=0.0,
            )

        self.assertNotIsInstance(result, str)
        self.assertTrue(result["composite_streamed"])
        self.assertEqual(result["composite"], 4)
        self.assertEqual(result["magnitude"].dtype, np.float32)
        self.assertEqual(alive_at_entry, [0, 0, 0, 0])

    def test_streaming_axis_preflight_matches_formed_sublook(self):
        grid = _grid(np.linspace(-5.0, 5.0, 21), np.linspace(8.0, 9.0, 17))
        frequency_hz = np.linspace(8.0e9, 9.0e9, 17)
        for reconstruction in ("fft", "accurate"):
            with self.subTest(reconstruction=reconstruction):
                predicted_x, predicted_y = isar_mode._predict_sublook_scene_axes(
                    grid,
                    list(range(21)),
                    frequency_hz,
                    1.0,
                    reconstruction=reconstruction,
                    elevation_deg=0.0,
                    az_center_deg=None,
                )
                formed = isar_mode._compute_band(
                    grid,
                    "Rectangular",
                    list(range(21)),
                    list(range(17)),
                    0,
                    0,
                    frequency_hz,
                    6.25e7,
                    1.0,
                    recon=reconstruction,
                )
                self.assertNotIsInstance(formed, str)
                np.testing.assert_array_equal(predicted_x, formed["x_range"])
                np.testing.assert_array_equal(predicted_y, formed["y_range"])

    def test_display_magnitude_is_float32_while_complex_result_is_preserved(self):
        grid = _grid(np.linspace(-2.0, 2.0, 9), np.linspace(8.0, 9.0, 9))
        bands, _elapsed = isar_mode.form_isar(
            grid,
            window="Rectangular",
            retain_complex=True,
        )
        self.assertEqual(bands[0]["magnitude"].dtype, np.float32)
        self.assertTrue(np.iscomplexobj(bands[0]["complex_image"]))
        self.assertEqual(bands[0]["complex_image"].dtype, np.complex64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
