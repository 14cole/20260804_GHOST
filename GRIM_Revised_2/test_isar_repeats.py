from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest import mock

import numpy as np

from grim_dataset import RcsGrid
from isar_repeats import IsarSweep, RepeatAcquisitionStack


class TestIsarRepeatContract(unittest.TestCase):
    def _grid(self, field, *, phase_reference="fixed origin"):
        values = np.asarray(field, dtype=np.complex128)
        return RcsGrid(
            np.arange(values.shape[0], dtype=float),
            [0.0],
            np.linspace(9.0, 10.0, values.shape[1]),
            ["VV"],
            rcs=values[:, None, :, None],
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_linear_quantity": "sigma_3d",
                "time_convention": "exp(+jwt)",
                "angular_coordinate_system": "conic",
            },
            extra={
                "phase_reference": phase_reference,
                "polarization_basis": "GRIM conic V/H",
                "measurement_geometry": "far-field monostatic",
                "motion_compensation": "not required; static turntable",
                "range_phase_convention": "S~exp(-j*2*k*R)",
            },
        )

    def _sweeps(self, arrays):
        start = datetime(2026, 8, 30, tzinfo=timezone.utc)
        return [
            IsarSweep(f"run-{index}", start + timedelta(minutes=index), self._grid(array))
            for index, array in enumerate(arrays)
        ]

    def test_contract_rejects_axis_and_convention_mismatch(self):
        field = np.ones((3, 4), dtype=np.complex128)
        first, second = self._sweeps([field, field])
        second.grid.frequencies[1] += 0.01
        with self.assertRaisesRegex(ValueError, "frequencies axis"):
            RepeatAcquisitionStack([first, second])

        first, second = self._sweeps([field, field])
        second.grid.extra["phase_reference"] = "another origin"
        with self.assertRaisesRegex(ValueError, "phase_reference"):
            RepeatAcquisitionStack([first, second])

        first, second = self._sweeps([field, field])
        first.grid.units["azimuth"] = "deg"
        second.grid.units["azimuth"] = "rad"
        with self.assertRaisesRegex(ValueError, "azimuth metadata"):
            RepeatAcquisitionStack([first, second])

        first, second = self._sweeps([field, field])
        first.grid.units["rcs_linear_quantity"] = "sigma_3d"
        second.grid.units["rcs_linear_quantity"] = "power_ratio"
        with self.assertRaisesRegex(ValueError, "response quantity"):
            RepeatAcquisitionStack([first, second])

    def test_range_phase_aliases_must_be_explicitly_compatible(self):
        field = np.ones((3, 4), dtype=np.complex128)
        first, second = self._sweeps([field, field])
        second.grid.extra["range_phase_convention"] = "S~exp(+j*2*k*R)"
        with self.assertRaisesRegex(ValueError, "two-way range-phase convention"):
            RepeatAcquisitionStack(
                [first, second], legacy_metadata_attested=True
            )

        first, second = self._sweeps([field, field])
        second.grid.extra.pop("range_phase_convention")
        second.grid.extra["phase_law"] = (
            "exp(+j*omega*t); S(range) proportional to exp(-j*2*k*R)"
        )
        stack = RepeatAcquisitionStack([first, second])
        self.assertEqual(
            stack.metadata_contract["metadata_profiles"]["run-1"][
                "range_phase"
            ]["sign"],
            -1,
        )

        first, second = self._sweeps([field, field])
        second.grid.extra.pop("range_phase_convention")
        with self.assertRaisesRegex(ValueError, "range_phase_convention"):
            RepeatAcquisitionStack([first, second])
        attested = RepeatAcquisitionStack(
            [first, second], legacy_metadata_attested=True
        )
        self.assertIn(
            "range_phase_convention",
            attested.metadata_contract["missing_declarations_by_acquisition"][
                "run-1"
            ],
        )

    def test_semantic_alias_families_accept_equivalent_crossed_keys(self):
        field = np.ones((3, 4), dtype=np.complex128)
        first, second = self._sweeps([field, field])
        first.grid.extra.update(
            {
                "calibration_id": "CAL-42",
                "measurement_setup_id": "RANGE-A",
            }
        )
        second.grid.extra.pop("measurement_geometry")
        second.grid.extra.pop("motion_compensation")
        second.grid.extra.update(
            {
                "acquisition_geometry": "monostatic Fraunhofer",
                "phase_center_stability": "stable",
                "calibration_identifier": "cal-42",
                "radar_setup_id": "range-a",
            }
        )

        stack = RepeatAcquisitionStack([first, second])

        self.assertFalse(stack.metadata_contract["legacy_metadata_attested"])
        profiles = stack.metadata_contract["metadata_profiles"]
        self.assertEqual(
            profiles["run-1"]["semantic_families"]["acquisition_geometry"][
                "canonical_dimensions"
            ],
            {
                "propagation_regime": "far_field",
                "scattering_configuration": "monostatic",
            },
        )

    def test_crossed_and_intra_sweep_alias_contradictions_are_rejected(self):
        field = np.ones((3, 4), dtype=np.complex128)
        first, second = self._sweeps([field, field])
        second.grid.extra.pop("measurement_geometry")
        second.grid.extra["scattering_geometry"] = "far-field bistatic"
        with self.assertRaisesRegex(ValueError, "measurement geometry"):
            RepeatAcquisitionStack(
                [first, second], legacy_metadata_attested=True
            )

        first, second = self._sweeps([field, field])
        first.grid.extra["acquisition_geometry"] = "far-field bistatic"
        with self.assertRaisesRegex(
            ValueError, "contradictory measurement geometry"
        ):
            RepeatAcquisitionStack(
                [first, second], legacy_metadata_attested=True
            )

        first, second = self._sweeps([field, field])
        first.grid.extra["calibration_id"] = "CAL-A"
        second.grid.extra["calibration_identifier"] = "CAL-B"
        with self.assertRaisesRegex(ValueError, "calibration ID"):
            RepeatAcquisitionStack(
                [first, second], legacy_metadata_attested=True
            )

    def test_all_repeat_pairs_are_checked_when_reference_is_blank(self):
        field = np.ones((3, 4), dtype=np.complex128)
        first, second, third = self._sweeps([field, field, field])
        first.grid.extra.pop("measurement_geometry")
        second.grid.extra["measurement_geometry"] = "far-field monostatic"
        third.grid.extra["measurement_geometry"] = "far-field bistatic"
        with self.assertRaisesRegex(ValueError, "measurement geometry"):
            RepeatAcquisitionStack(
                [first, second, third], legacy_metadata_attested=True
            )

    def test_legacy_attestation_covers_only_missing_metadata(self):
        field = np.ones((3, 4), dtype=np.complex128)
        first, second = self._sweeps([field, field])
        for sweep in (first, second):
            sweep.grid.extra.pop("measurement_geometry")
            sweep.grid.extra.pop("motion_compensation")
            sweep.grid.extra.pop("phase_reference")
            sweep.grid.extra.pop("polarization_basis")
            sweep.grid.units.pop("time_convention")
        with self.assertRaisesRegex(ValueError, "metadata is incomplete"):
            RepeatAcquisitionStack([first, second])

        stack = RepeatAcquisitionStack(
            [first, second], legacy_metadata_attested=True
        )
        contract = stack.metadata_contract
        self.assertTrue(contract["legacy_metadata_attested"])
        self.assertFalse(contract["declarations_inferred"])
        self.assertIn(
            "phase_reference",
            contract["missing_declarations_by_acquisition"]["run-0"],
        )

        first, second = self._sweeps([field, field])
        second.grid.units["time_convention"] = "exp(-jwt)"
        with self.assertRaisesRegex(ValueError, "time conventions"):
            RepeatAcquisitionStack(
                [first, second], legacy_metadata_attested=True
            )

        first, second = self._sweeps([field, field])
        second.grid.extra["motion_compensation"] = "uncompensated"
        with self.assertRaisesRegex(ValueError, "stable/aligned acquisitions"):
            RepeatAcquisitionStack(
                [first, second], legacy_metadata_attested=True
            )

    def test_motion_booleans_and_per_acquisition_calibration_runs(self):
        field = np.ones((3, 4), dtype=np.complex128)
        first, second = self._sweeps([field, field])
        first.grid.extra.pop("motion_compensation")
        second.grid.extra.pop("motion_compensation")
        first.grid.extra["motion_compensated"] = True
        second.grid.extra["phase_center_motion"] = False
        first.grid.extra["calibration_run_id"] = "RUN-A"
        second.grid.extra["calibration_run_id"] = "RUN-B"
        RepeatAcquisitionStack([first, second])

        second.grid.extra["phase_center_motion"] = True
        with self.assertRaisesRegex(ValueError, "stable/aligned acquisitions"):
            RepeatAcquisitionStack(
                [first, second], legacy_metadata_attested=True
            )

    def test_legacy_attestation_must_be_boolean(self):
        field = np.ones((3, 4), dtype=np.complex128)
        with self.assertRaisesRegex(TypeError, "must be True or False"):
            RepeatAcquisitionStack(
                self._sweeps([field, field]),
                legacy_metadata_attested="yes",
            )

    def test_screen_registers_global_drift_and_flags_transient_without_editing(self):
        base = np.exp(1j * np.linspace(0.1, 1.2, 20)).reshape(4, 5)
        arrays = [base.copy() for _ in range(5)]
        drift = 1.3 * np.exp(1j * 0.4)
        arrays[1] = arrays[1] * drift
        arrays[3][2, 4] += 20.0 - 5.0j
        originals = [array.copy() for array in arrays]
        stack = RepeatAcquisitionStack(self._sweeps(arrays))
        guard = np.ones((4, 5), dtype=bool)
        guard[2, 4] = False
        result = stack.screen_transients(
            elevation_index=0,
            polarization_index=0,
            registration_guard=guard,
            threshold=6.0,
        )
        np.testing.assert_allclose(result.registered_stack[1], base, atol=1.0e-12)
        self.assertTrue(result.candidate_outlier_mask[3, 2, 4])
        self.assertEqual(int(result.candidate_outlier_mask.sum()), 1)
        for before, sweep in zip(originals, stack.sweeps):
            np.testing.assert_allclose(
                sweep.grid.rcs_slice(
                    np.ix_(np.arange(4), [0], np.arange(5), [0])
                )[:, 0, :, 0],
                before,
            )

    def test_screen_requires_three_repeats(self):
        field = np.ones((3, 4), dtype=np.complex128)
        stack = RepeatAcquisitionStack(self._sweeps([field, field]))
        with self.assertRaisesRegex(ValueError, "at least three"):
            stack.screen_transients(elevation_index=0, polarization_index=0)

    def test_repeat_allocations_fail_before_reading_when_byte_cap_is_too_small(self):
        field = np.ones((3, 4), dtype=np.complex128)
        two = RepeatAcquisitionStack(self._sweeps([field, field]))
        with mock.patch.object(
            RcsGrid,
            "rcs_slice",
            side_effect=AssertionError("data read happened before preflight"),
        ):
            with self.assertRaisesRegex(ValueError, "estimated.*working set"):
                two.complex_stack(
                    elevation_index=0,
                    polarization_index=0,
                    maximum_working_bytes=1,
                )

        three = RepeatAcquisitionStack(self._sweeps([field, field, field]))
        with mock.patch.object(
            RcsGrid,
            "rcs_slice",
            side_effect=AssertionError("data read happened before preflight"),
        ):
            with self.assertRaisesRegex(ValueError, "estimated.*working set"):
                three.screen_transients(
                    elevation_index=0,
                    polarization_index=0,
                    maximum_working_bytes=1,
                )

    def test_repeat_working_set_limit_rejects_boolean(self):
        field = np.ones((3, 4), dtype=np.complex128)
        stack = RepeatAcquisitionStack(self._sweeps([field, field]))
        with self.assertRaisesRegex(TypeError, "positive integer"):
            stack.complex_stack(
                elevation_index=0,
                polarization_index=0,
                maximum_working_bytes=True,
            )

    def test_equal_nonzero_repeat_residuals_are_not_all_outliers(self):
        base = np.ones((3, 4), dtype=np.complex128)
        arrays = [base + offset for offset in (-0.1, 0.0, 0.1)]
        stack = RepeatAcquisitionStack(self._sweeps(arrays))
        result = stack.screen_transients(
            elevation_index=0,
            polarization_index=0,
            minimum_coherence=0.0,
        )
        self.assertFalse(np.any(result.candidate_outlier_mask))


if __name__ == "__main__":
    unittest.main()
