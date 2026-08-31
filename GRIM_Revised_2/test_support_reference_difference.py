"""Physics, QA, and provenance tests for guided support subtraction."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from grim_dataset import RcsGrid


def _grid(field, *, azimuths=(0.0, 10.0), extra=None) -> RcsGrid:
    values = np.asarray(field, dtype=np.complex128)
    shape = (len(azimuths), 1, 1, 1)
    values = np.broadcast_to(values.reshape(shape), shape).copy()
    metadata = {
        "phase_reference": "turntable origin",
        "time_convention": "exp(+j*omega*t)",
        "polarization_basis": "GRIM conic V/H",
    }
    metadata.update(extra or {})
    return RcsGrid(
        azimuths,
        [0.0],
        [10.0],
        ["VV"],
        rcs=values,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
            "angular_coordinate_system": "conic",
        },
        extra=metadata,
    )


class SupportReferenceDifferenceTest(unittest.TestCase):
    def test_exact_complex_difference_records_bounded_qa_and_content(self) -> None:
        combined = _grid([3.0 + 0.0j, 1.0 + 1.0j])
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j])

        result = combined.support_referenced_difference(
            support,
            assumptions_attested=True,
            target_label="vehicle on pylon",
            support_label="pylon only",
        )

        np.testing.assert_allclose(result.rcs.reshape(-1), [2.0 + 0.0j, 0.0 + 1.0j])
        self.assertIsNone(result.source_path)
        self.assertIn("not a reconstructed free-space", result.history)
        self.assertEqual(
            result.extra["complex_field_domain"],
            "support_referenced_complex_difference",
        )

        provenance = json.loads(result.extra["support_reference_difference_json"])
        self.assertEqual(
            provenance["schema"], "grim.support-reference-difference.v1"
        )
        self.assertTrue(provenance["not_free_space_target"])
        self.assertEqual(provenance["target_plus_support"], "vehicle on pylon")
        self.assertEqual(provenance["support_only_reference"], "pylon only")
        for key in (
            "target_plus_support_content_sha256",
            "support_only_reference_content_sha256",
            "result_content_sha256",
        ):
            self.assertRegex(provenance[key], r"^[0-9a-f]{64}$")

        qa = provenance["qa"]
        self.assertEqual(qa["common_finite_sample_count"], 2)
        self.assertEqual(qa["excluded_sample_count"], 0)
        energies = qa["energy_sum_linear"]
        self.assertAlmostEqual(energies["pre_target_plus_support"], 11.0)
        self.assertAlmostEqual(energies["subtracted_support_reference"], 2.0)
        self.assertAlmostEqual(
            energies["post_support_referenced_difference"], 5.0
        )
        self.assertLess(energies["algebraic_closure_residual"], 1.0e-28)
        self.assertAlmostEqual(
            qa["complex_coherence"], abs(4.0 + 1.0j) / np.sqrt(22.0)
        )
        self.assertTrue(qa["complex_coherence_meaningful"])

    def test_working_set_gate_fires_before_any_complex_tile_is_read(self) -> None:
        combined = _grid([3.0 + 0.0j, 1.0 + 1.0j])
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j])
        left_read = mock.Mock(
            side_effect=AssertionError("left numerical tile read before preflight")
        )
        right_read = mock.Mock(
            side_effect=AssertionError("right numerical tile read before preflight")
        )
        with (
            mock.patch.object(
                combined,
                "_bounded_complex_slice_reader",
                return_value=(left_read, np.dtype(np.float64)),
            ) as left_reader_factory,
            mock.patch.object(
                support,
                "_bounded_complex_slice_reader",
                return_value=(right_read, np.dtype(np.float64)),
            ) as right_reader_factory,
            self.assertRaisesRegex(MemoryError, "working set"),
        ):
            combined.support_referenced_difference(
                support,
                assumptions_attested=True,
                maximum_working_bytes=1,
            )
        left_reader_factory.assert_not_called()
        right_reader_factory.assert_not_called()
        left_read.assert_not_called()
        right_read.assert_not_called()

    def test_tiled_difference_transfers_fresh_arrays_without_constructor_copy(self) -> None:
        combined = _grid([3.0 + 0.0j, 1.0 + 1.0j])
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j])
        with mock.patch.object(
            RcsGrid,
            "_clean_power",
            side_effect=AssertionError("fresh difference result was copied"),
        ):
            result = combined.support_referenced_difference(
                support,
                assumptions_attested=True,
                maximum_working_bytes=1024**2,
            )
        np.testing.assert_allclose(
            result.rcs.reshape(-1), [2.0 + 0.0j, 0.0 + 1.0j]
        )

    def test_requires_distinct_roles_and_exact_compatible_axes(self) -> None:
        combined = _grid([2.0 + 0.0j, 2.0 + 0.0j])
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j])
        result = combined.support_referenced_difference(support)
        provenance = json.loads(result.extra["support_reference_difference_json"])
        self.assertTrue(
            provenance["operation_selected_as_assumption_of_compatible_acquisition"]
        )
        self.assertFalse(provenance["user_assumptions_attested"])
        with self.assertRaisesRegex(ValueError, "different datasets"):
            combined.support_referenced_difference(
                combined, assumptions_attested=True
            )
        mismatched = _grid([1.0 + 0.0j, 1.0 + 0.0j], azimuths=(0.0, 11.0))
        with self.assertRaisesRegex(ValueError, "azimuth axis mismatch"):
            combined.support_referenced_difference(
                mismatched, assumptions_attested=True
            )

        incompatible = _grid(
            [1.0 + 0.0j, 1.0 + 0.0j],
            extra={"time_convention": "exp(-j*omega*t)"},
        )
        with self.assertRaisesRegex(ValueError, "time conventions"):
            combined.support_referenced_difference(
                incompatible,
                assumptions_attested=True,
                metadata_attested=True,
            )

    def test_explicit_acquisition_or_calibration_contradictions_cannot_be_attested(self) -> None:
        cases = (
            (
                {"measurement_geometry": "far-field monostatic"},
                {"measurement_geometry": "bistatic"},
                "measurement geometry",
            ),
            (
                {"motion_compensation": "stable"},
                {"motion_compensation": "uncompensated"},
                "stable/aligned acquisitions",
            ),
            (
                {"calibration_id": "CAL-A"},
                {"calibration_id": "CAL-B"},
                "calibration ID",
            ),
        )
        for left_extra, right_extra, message in cases:
            with self.subTest(message=message):
                combined = _grid([2.0 + 0.0j, 2.0 + 0.0j], extra=left_extra)
                support = _grid([1.0 + 0.0j, 1.0 + 0.0j], extra=right_extra)
                with self.assertRaisesRegex(ValueError, message):
                    combined.support_referenced_difference(
                        support,
                        assumptions_attested=True,
                        metadata_attested=True,
                    )

    def test_matching_explicit_acquisition_metadata_is_recorded_and_propagated(self) -> None:
        declared = {
            "measurement_geometry": "far-field monostatic",
            "motion_compensation": "stable",
            "calibration_id": "CAL-42",
            "measurement_setup_id": "RANGE-A",
        }
        combined = _grid([2.0 + 0.0j, 2.0 + 0.0j], extra=declared)
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j], extra=declared)
        result = combined.support_referenced_difference(
            support, assumptions_attested=True
        )
        provenance = json.loads(result.extra["support_reference_difference_json"])
        contract = provenance["support_metadata_contract"]
        self.assertFalse(contract["explicit_contradictions_allowed"])
        self.assertEqual(
            contract["matching_explicit_declarations"]["calibration_id"],
            "CAL-42",
        )
        for key, value in declared.items():
            self.assertEqual(result.extra[key], value)

    def test_crossed_semantic_aliases_are_compared_and_recorded(self) -> None:
        combined = _grid(
            [2.0 + 0.0j, 2.0 + 0.0j],
            extra={
                "measurement_geometry": "far-field monostatic",
                "calibration_id": "CAL-42",
                "measurement_setup_id": "RANGE-A",
            },
        )
        support = _grid(
            [1.0 + 0.0j, 1.0 + 0.0j],
            extra={
                "acquisition_geometry": "monostatic Fraunhofer",
                "calibration_identifier": "cal-42",
                "radar_setup_id": "range-a",
            },
        )
        result = combined.support_referenced_difference(
            support, assumptions_attested=True
        )
        provenance = json.loads(result.extra["support_reference_difference_json"])
        contract = provenance["support_metadata_contract"]
        self.assertEqual(
            contract["schema"], "grim.support-reference-metadata-contract.v3"
        )
        geometry = contract["semantic_families"]["acquisition_geometry"]
        self.assertEqual(
            geometry["canonical_dimensions_by_role"]["target_plus_support"],
            {
                "propagation_regime": "far_field",
                "scattering_configuration": "monostatic",
            },
        )
        self.assertEqual(result.extra["calibration_id"], "CAL-42")
        self.assertEqual(result.extra["measurement_setup_id"], "RANGE-A")

    def test_two_way_range_phase_aliases_cannot_hide_an_opposite_sign(self) -> None:
        combined = _grid(
            [2.0 + 0.0j, 2.0 + 0.0j],
            extra={"range_phase_convention": "S~exp(-j*2*k*R)"},
        )
        support = _grid(
            [1.0 + 0.0j, 1.0 + 0.0j],
            extra={
                "phase_law": (
                    "exp(+j*omega*t); S(range) proportional to exp(-j*2*k*R)"
                )
            },
        )
        result = combined.support_referenced_difference(
            support, assumptions_attested=True
        )
        contract = json.loads(result.extra["support_reference_difference_json"])[
            "support_metadata_contract"
        ]
        range_contract = contract["semantic_families"]["range_phase_law"]
        self.assertEqual(
            range_contract["canonical_dimensions_by_role"][
                "target_plus_support"
            ]["two_way_sign"],
            "negative",
        )

        support.extra["phase_law"] = "S~exp(+j*2*k*R)"
        with self.assertRaisesRegex(ValueError, "two-way range-phase convention"):
            combined.support_referenced_difference(
                support,
                assumptions_attested=True,
                metadata_attested=True,
            )

        support.extra["phase_law"] = "S~exp(-j*2*k*R)"
        combined.extra["phase_law"] = "S~exp(+j*2*k*R)"
        with self.assertRaisesRegex(ValueError, "contradictory.*range-phase"):
            combined.support_referenced_difference(
                support,
                assumptions_attested=True,
                metadata_attested=True,
            )

    def test_crossed_alias_and_intra_dataset_contradictions_are_rejected(self) -> None:
        cases = (
            (
                {"measurement_geometry": "far-field monostatic"},
                {"acquisition_geometry": "far-field bistatic"},
                "measurement geometry",
            ),
            (
                {"calibration_id": "CAL-A"},
                {"calibration_identifier": "CAL-B"},
                "calibration ID",
            ),
            (
                {"measurement_setup_id": "RANGE-A"},
                {"radar_setup_id": "RANGE-B"},
                "measurement-setup ID",
            ),
        )
        for left_extra, right_extra, message in cases:
            with self.subTest(message=message):
                combined = _grid(
                    [2.0 + 0.0j, 2.0 + 0.0j], extra=left_extra
                )
                support = _grid(
                    [1.0 + 0.0j, 1.0 + 0.0j], extra=right_extra
                )
                with self.assertRaisesRegex(ValueError, message):
                    combined.support_referenced_difference(
                        support,
                        assumptions_attested=True,
                        metadata_attested=True,
                    )

        internally_conflicted = _grid(
            [2.0 + 0.0j, 2.0 + 0.0j],
            extra={
                "measurement_geometry": "far-field monostatic",
                "acquisition_geometry": "far-field bistatic",
            },
        )
        support = _grid(
            [1.0 + 0.0j, 1.0 + 0.0j],
            extra={"measurement_geometry": "far-field monostatic"},
        )
        with self.assertRaisesRegex(ValueError, "contradictory measurement geometry"):
            internally_conflicted.support_referenced_difference(
                support,
                assumptions_attested=True,
                metadata_attested=True,
            )

    def test_motion_alias_boolean_semantics_do_not_erase_false(self) -> None:
        combined = _grid(
            [2.0 + 0.0j, 2.0 + 0.0j],
            extra={"motion_compensated": True},
        )
        support = _grid(
            [1.0 + 0.0j, 1.0 + 0.0j],
            extra={"phase_center_motion": False},
        )
        combined.support_referenced_difference(
            support, assumptions_attested=True
        )

        support.extra["phase_center_motion"] = True
        with self.assertRaisesRegex(ValueError, "stable/aligned acquisitions"):
            combined.support_referenced_difference(
                support,
                assumptions_attested=True,
                metadata_attested=True,
            )

    def test_missing_declarations_are_recorded_as_assumptions(self) -> None:
        combined = _grid([2.0 + 0.0j, 2.0 + 0.0j])
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j])
        support.extra.pop("phase_reference")
        result = combined.support_referenced_difference(support)
        provenance = json.loads(result.extra["support_reference_difference_json"])
        self.assertFalse(provenance["metadata_attestation_used"])
        assumption = json.loads(result.extra["coherent_metadata_assumption_json"])
        self.assertEqual(assumption["operation"], "coherent-subtract")

    def test_missing_samples_are_reported_and_provenance_roundtrips(self) -> None:
        combined = _grid([2.0 + 0.0j, np.nan + 1j * np.nan])
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j])
        result = combined.support_referenced_difference(
            support, assumptions_attested=True
        )
        provenance = json.loads(result.extra["support_reference_difference_json"])
        self.assertEqual(provenance["qa"]["common_finite_sample_count"], 1)
        self.assertEqual(provenance["qa"]["excluded_sample_count"], 1)
        self.assertFalse(provenance["qa"]["complex_coherence_meaningful"])
        self.assertIsNone(provenance["qa"]["complex_coherence"])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "support-reference.grim")
            result.save(path)
            loaded = RcsGrid.load(path)
        loaded_raw = np.asarray(
            loaded.extra["support_reference_difference_json"]
        ).reshape(()).item()
        self.assertEqual(json.loads(str(loaded_raw)), provenance)
        chained = loaded.support_referenced_difference(support)
        chained_provenance = json.loads(
            chained.extra["support_reference_difference_json"]
        )
        self.assertEqual(
            chained_provenance["chained_support_difference_input_roles"],
            ["target_plus_support"],
        )

        all_missing = _grid(
            [np.nan + 1j * np.nan, np.nan + 1j * np.nan]
        )
        with self.assertRaisesRegex(ValueError, "no common usable complex samples"):
            all_missing.support_referenced_difference(
                support, assumptions_attested=True
            )

    def test_lightweight_dialog_precheck_defers_full_phase_scan_to_worker(self) -> None:
        combined = _grid([2.0 + 0.0j, 2.0 + 0.0j])
        support = _grid([1.0 + 0.0j, 1.0 + 0.0j])
        support.rcs_phase[0, 0, 0, 0] = np.nan

        combined._assert_compatible(
            support,
            coherent=True,
            _scan_phase_samples=False,
        )
        with self.assertRaisesRegex(ValueError, "requires phase"):
            combined._assert_compatible(support, coherent=True)
        result = combined.support_referenced_difference(support)
        provenance = json.loads(result.extra["support_reference_difference_json"])
        self.assertEqual(provenance["qa"]["common_finite_sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
