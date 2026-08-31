"""Physics and provenance regressions for complex Range Cal."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from grim_dataset import C0, RcsGrid


def _grid(
    amplitude,
    *,
    azimuths=(0.0,),
    elevations=(0.0,),
    frequencies=(9.0, 10.0),
    polarizations=("VV", "HH"),
    quantity="sigma_3d",
    phase_reference="",
    extra=None,
) -> RcsGrid:
    values = np.asarray(amplitude, dtype=np.complex128)
    shape = (
        len(azimuths), len(elevations), len(frequencies), len(polarizations)
    )
    values = np.broadcast_to(values, shape).copy()
    metadata = dict(extra or {})
    if phase_reference:
        metadata["phase_reference"] = phase_reference
    return RcsGrid(
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=values,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm" if quantity == "sigma_3d" else "dBke",
            "rcs_linear_quantity": quantity,
            "angular_coordinate_system": "conic",
        },
        extra=metadata,
    )


class RangeCalibrationTest(unittest.TestCase):
    @staticmethod
    def _grid_with_authoritative_raw(stored_power, raw_field):
        shape = (1, 1, 2, 2)
        raw_field = np.broadcast_to(
            np.asarray(raw_field, dtype=np.complex128), shape
        ).copy()
        # sigma_3d raw GHOST amplitudes are converted by sqrt(4*pi) in
        # RcsGrid.rcs. Store the inverse-scaled values so raw_field is the
        # exact authoritative physical amplitude used by Range Cal.
        ghost_raw = raw_field / np.sqrt(4.0 * np.pi)
        return RcsGrid(
            [0.0],
            [0.0],
            [9.0, 10.0],
            ["VV", "HH"],
            rcs_power=np.broadcast_to(
                np.asarray(stored_power, dtype=np.float32), shape
            ).copy(),
            rcs_phase=np.zeros(shape, dtype=np.float32),
            units=dict(_grid(1.0 + 0.0j).units),
            extra={
                "rcs_amp_real": ghost_raw.real,
                "rcs_amp_imag": ghost_raw.imag,
            },
        )

    def test_recovers_known_complex_dut_with_signed_range_offset(self) -> None:
        frequencies = np.asarray([9.0, 10.0])
        frequency_hz = frequencies * 1.0e9
        offset_m = 0.125
        range_phase = np.exp(-1j * 4.0 * np.pi * frequency_hz * offset_m / C0)

        exact_amp = np.asarray([[[[2.0 + 0.5j, 1.0 - 0.25j],
                                  [1.5 - 0.2j, 0.8 + 0.4j]]]])
        system_gain = np.asarray([[[[0.5 + 0.3j, 1.2 - 0.1j],
                                    [0.8 - 0.4j, 0.7 + 0.2j]]]])
        measured_amp = exact_amp * system_gain * range_phase.reshape(1, 1, -1, 1)
        truth = np.asarray(
            [
                [[[1.0 + 2.0j, 0.5 - 0.4j], [2.0 - 0.2j, 1.0 + 0.1j]]],
                [[[0.8 - 0.6j, 1.3 + 0.7j], [0.4 + 0.9j, 2.0 - 1.0j]]],
            ]
        )
        dut_measured = truth * system_gain

        dut = _grid(dut_measured, azimuths=(-10.0, 10.0), frequencies=frequencies)
        exact = _grid(exact_amp, frequencies=frequencies, phase_reference="exact origin")
        measured = _grid(measured_amp, frequencies=frequencies)

        result = dut.range_calibrate(
            measured,
            exact,
            offset_m,
            allow_singleton_angular_broadcast=True,
            convention_attested=True,
            measured_label="measured-cylinder",
            exact_label="exact-cylinder",
        )
        np.testing.assert_allclose(result.rcs, truth, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(result.rcs_power, np.abs(truth) ** 2)
        self.assertEqual(result.rcs_domain, "complex_amplitude")
        self.assertIsNone(result.source_path)

        metadata = json.loads(result.extra["range_calibration_json"])
        self.assertEqual(metadata["schema"], "grim.range-calibration.v1")
        self.assertAlmostEqual(metadata["range_offset_m"], offset_m)
        self.assertEqual(
            metadata["range_offset_positive_direction"], "away_from_radar"
        )
        self.assertIn("exact-cylinder", result.history)

    def test_polarization_reorder_and_field_amplitude_ratio(self) -> None:
        dut = _grid([[[[4.0 + 0.0j, 8.0 + 0.0j]]]], frequencies=(10.0,))
        exact = _grid(
            [[[[3.0 + 0.0j, 6.0 + 0.0j]]]], frequencies=(10.0,)
        )
        measured = _grid(
            [[[[4.0 + 0.0j, 2.0 + 0.0j]]]],
            frequencies=(10.0,),
            polarizations=("HH", "VV"),
        )
        result = dut.range_calibrate(
            measured, exact, 0.0, convention_attested=True
        )
        # VV: exact 3 / measured VV 2. HH: exact 6 / measured HH 4.
        np.testing.assert_allclose(result.rcs[0, 0, 0, :], [6.0, 12.0])
        np.testing.assert_allclose(result.rcs_power[0, 0, 0, :], [36.0, 144.0])

    def test_ignores_extra_measured_crosspol_for_copol_exact_standard(self) -> None:
        dut = _grid(
            [[[[2.0 + 0.0j, 4.0 + 0.0j]]]],
            frequencies=(10.0,),
            polarizations=("VV", "HH"),
        )
        measured = _grid(
            [[[[1.0 + 0.0j, 2.0 + 0.0j, 0.1 + 0.0j, 0.2 + 0.0j]]]],
            frequencies=(10.0,),
            polarizations=("VV", "HH", "VH", "HV"),
        )
        exact = _grid(
            [[[[1.0 + 0.0j, 2.0 + 0.0j]]]],
            frequencies=(10.0,),
            polarizations=("VV", "HH"),
        )
        result = dut.range_calibrate(
            measured, exact, 0.0, convention_attested=True
        )
        np.testing.assert_allclose(result.rcs, dut.rcs)

    def test_requires_explicit_broadcast_but_not_metadata_attestation(self) -> None:
        dut = _grid(1.0 + 0.0j, azimuths=(-10.0, 10.0))
        measured = _grid(1.0 + 0.0j)
        exact = _grid(1.0 + 0.0j)
        with self.assertRaisesRegex(ValueError, "broadcast confirmation"):
            dut.range_calibrate(measured, exact, 0.0)
        result = dut.range_calibrate(
            measured,
            exact,
            0.0,
            allow_singleton_angular_broadcast=True,
        )
        provenance = json.loads(result.extra["range_calibration_json"])
        self.assertFalse(provenance["user_convention_attested"])

        for option_name, options in (
            ("convention_attested", {"convention_attested": "false"}),
            (
                "allow_singleton_angular_broadcast",
                {
                    "convention_attested": True,
                    "allow_singleton_angular_broadcast": "false",
                },
            ),
        ):
            with self.subTest(option_name=option_name):
                with self.assertRaisesRegex(TypeError, option_name):
                    dut.range_calibrate(measured, exact, 0.0, **options)

    def test_rejects_wrong_quantity_measured_null_and_axis_mismatch(self) -> None:
        dut = _grid(1.0 + 0.0j)
        measured = _grid(1.0 + 0.0j)
        exact = _grid(1.0 + 0.0j)

        wrong_quantity = _grid(1.0 + 0.0j, quantity="sigma_2d")
        with self.assertRaisesRegex(ValueError, "sigma_3d"):
            dut.range_calibrate(
                measured, wrong_quantity, 0.0, convention_attested=True
            )

        missing_phase = RcsGrid(
            [0.0], [0.0], [9.0, 10.0], ["VV", "HH"],
            rcs_power=np.ones((1, 1, 2, 2)),
            rcs_phase=np.zeros((1, 1, 2, 2)),
            units=dict(measured.units),
        )
        missing_phase.rcs_phase[0, 0, 0, 0] = np.nan
        masked = dut.range_calibrate(
            missing_phase, exact, 0.0, convention_attested=True
        )
        self.assertTrue(np.isnan(masked.rcs_power[0, 0, 0, 0]))
        self.assertEqual(int(np.isfinite(masked.rcs_power).sum()), 3)
        self.assertEqual(
            json.loads(masked.extra["range_calibration_json"])[
                "correction_gain_db"
            ]["masked_output_bin_count"],
            1,
        )

        all_missing_phase = RcsGrid(
            [0.0], [0.0], [9.0, 10.0], ["VV", "HH"],
            rcs_power=np.ones((1, 1, 2, 2)),
            units=dict(measured.units),
        )
        with self.assertRaisesRegex(ValueError, "no calibratable bins"):
            dut.range_calibrate(
                all_missing_phase, exact, 0.0, convention_attested=True
            )

        # A magnitude-only exact null has no meaningful phase, but the zero
        # field is still an exact and valid calibration response.
        null_reference = RcsGrid(
            [0.0],
            [0.0],
            [9.0, 10.0],
            ["VV", "HH"],
            rcs_power=np.zeros((1, 1, 2, 2)),
            units=dict(exact.units),
        )
        zeroed = dut.range_calibrate(
            measured, null_reference, 0.0, convention_attested=True
        )
        np.testing.assert_array_equal(zeroed.rcs_power, 0.0)

        null_measured = _grid(0.0 + 0.0j)
        with self.assertRaisesRegex(ValueError, "no calibratable bins"):
            dut.range_calibrate(
                null_measured, exact, 0.0, convention_attested=True
            )

        wrong_frequency = _grid(1.0 + 0.0j, frequencies=(8.0, 10.0))
        with self.assertRaisesRegex(ValueError, "frequency axes differ"):
            dut.range_calibrate(
                measured, wrong_frequency, 0.0, convention_attested=True
            )

    def test_authoritative_raw_field_controls_zero_semantics(self) -> None:
        # The separately stored float32 power can underflow or disagree with a
        # finite GHOST raw field. Do not replace a finite authoritative DUT or
        # exact amplitude merely because stored power is zero.
        dut = self._grid_with_authoritative_raw(0.0, 3.0 + 0.0j)
        exact = self._grid_with_authoritative_raw(0.0, 2.0 + 0.0j)
        measured = _grid(1.0 + 0.0j)
        result = dut.range_calibrate(
            measured, exact, 0.0, convention_attested=True
        )
        np.testing.assert_allclose(result.rcs, 6.0 + 0.0j)

        # Conversely, an actual raw denominator zero remains invalid even if
        # the stored power array incorrectly says it is positive.
        measured_raw_zero = self._grid_with_authoritative_raw(1.0, 0.0 + 0.0j)
        with self.assertRaisesRegex(ValueError, "no calibratable bins"):
            dut.range_calibrate(
                measured_raw_zero, exact, 0.0, convention_attested=True
            )

    def test_rejects_declared_opposite_time_sign_and_excessive_gain(self) -> None:
        dut = _grid(
            1.0 + 0.0j,
            extra={"time_convention": "exp(+j*omega*t)"},
        )
        measured_wrong_sign = _grid(
            1.0 + 0.0j,
            extra={"time_convention": "exp(-j*omega*t)"},
        )
        exact = _grid(1.0 + 0.0j)
        with self.assertRaisesRegex(ValueError, "will not override"):
            dut.range_calibrate(
                measured_wrong_sign, exact, 0.0, convention_attested=True
            )

        weak_measured = _grid(1.0 + 0.0j)
        weak_measured.rcs_power.flat[0] = 1.0e-8
        weak_measured.rcs_phase.flat[0] = 0.0
        masked = dut.range_calibrate(
            weak_measured,
            exact,
            0.0,
            maximum_correction_gain_db=60.0,
        )
        self.assertTrue(np.isnan(masked.rcs_power.flat[0]))
        self.assertEqual(int(np.isfinite(masked.rcs_power).sum()), 3)
        masked_provenance = json.loads(masked.extra["range_calibration_json"])
        self.assertEqual(
            masked_provenance["correction_gain_db"][
                "over_limit_correction_bin_count"
            ],
            1,
        )
        allowed = dut.range_calibrate(
            weak_measured,
            exact,
            0.0,
            convention_attested=True,
            maximum_correction_gain_db=None,
        )
        self.assertAlmostEqual(float(np.abs(allowed.rcs.flat[0])), 1.0e4)

        unsupported_unit = _grid(1.0 + 0.0j)
        unsupported_unit.units["frequency"] = "THz"
        with self.assertRaisesRegex(ValueError, "unsupported frequency unit"):
            unsupported_unit.range_calibrate(
                unsupported_unit,
                unsupported_unit,
                0.0,
                convention_attested=True,
            )

    def test_strips_stale_solver_metadata_and_roundtrips_provenance(self) -> None:
        stale = {
            "rcs_amp_real": np.ones((1, 1, 2, 2)),
            "rcs_amp_imag": np.zeros((1, 1, 2, 2)),
            "solver_metadata_json": "stale",
            "production_mesh_certification_json": "stale",
            "source_body_mesh_certification_json": "stale",
            "rcs_domain": "delta",
            "power_domain": "delta_amp_sq",
            "raw_complex_amplitude_preserved": True,
            "complex_field_domain": "featured_minus_clean",
            "combination_estimate_power": 99.0,
        }
        dut = _grid(2.0 + 1.0j, extra=stale)
        measured = _grid(1.0 + 0.0j)
        exact = _grid(1.0 + 0.0j)
        result = dut.range_calibrate(
            measured, exact, 0.0, convention_attested=True
        )
        for key in stale:
            self.assertNotIn(key, result.extra)
        self.assertIn("range_calibration_json", result.extra)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calibrated.grim")
            result.save(path)
            loaded = RcsGrid.load(path)
        np.testing.assert_allclose(loaded.rcs, result.rcs)
        loaded_metadata = str(
            np.asarray(loaded.extra["range_calibration_json"]).reshape(()).item()
        )
        self.assertEqual(
            json.loads(loaded_metadata)["schema"],
            "grim.range-calibration.v1",
        )
        recalibrated = loaded.range_calibrate(measured, exact, 0.0)
        recalibration = json.loads(recalibrated.extra["range_calibration_json"])
        self.assertTrue(recalibration["input_was_previously_range_calibrated"])
        self.assertIsInstance(recalibration["prior_range_calibration"], dict)

    def test_blank_exact_phase_centers_get_content_bound_references(self) -> None:
        dut = _grid(1.0 + 0.0j)
        measured = _grid(1.0 + 0.0j)
        exact_a = _grid(1.0 + 0.0j)
        exact_b = _grid(2.0 + 0.0j)
        result_a = dut.range_calibrate(
            measured, exact_a, 0.0, convention_attested=True
        )
        result_b = dut.range_calibrate(
            measured, exact_b, 0.0, convention_attested=True
        )
        self.assertNotEqual(result_a._phase_reference(), result_b._phase_reference())
        metadata_a = json.loads(result_a.extra["range_calibration_json"])
        self.assertIn("exact_reference_content_sha256", metadata_a)

    def test_content_hash_binds_authoritative_raw_amplitude(self) -> None:
        dut = _grid(1.0 + 0.0j)
        measured = _grid(1.0 + 0.0j)
        exact_a = self._grid_with_authoritative_raw(1.0, 1.0 + 0.0j)
        exact_b = self._grid_with_authoritative_raw(1.0, 2.0 + 0.0j)
        result_a = dut.range_calibrate(
            measured, exact_a, 0.0, convention_attested=True
        )
        result_b = dut.range_calibrate(
            measured, exact_b, 0.0, convention_attested=True
        )
        hash_a = json.loads(result_a.extra["range_calibration_json"])[
            "exact_reference_content_sha256"
        ]
        hash_b = json.loads(result_b.extra["range_calibration_json"])[
            "exact_reference_content_sha256"
        ]
        self.assertNotEqual(hash_a, hash_b)


if __name__ == "__main__":
    unittest.main()
