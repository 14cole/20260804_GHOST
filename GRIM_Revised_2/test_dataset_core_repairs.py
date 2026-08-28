"""Regressions for native fidelity and physically safe dataset operations."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

from grim_dataset import RcsGrid


def _grid(
    power=1.0,
    *,
    phase=0.0,
    azimuths=(0.0,),
    elevations=(0.0,),
    frequencies=(1.0,),
    polarizations=("VV",),
    units=None,
    extra=None,
):
    shape = (
        len(azimuths),
        len(elevations),
        len(frequencies),
        len(polarizations),
    )
    raw_power = np.asarray(power, dtype=float)
    raw_phase = np.asarray(phase, dtype=float)
    power_values = (
        raw_power.reshape(shape).copy()
        if raw_power.size == int(np.prod(shape))
        else np.broadcast_to(raw_power, shape).copy()
    )
    phase_values = (
        raw_phase.reshape(shape).copy()
        if raw_phase.size == int(np.prod(shape))
        else np.broadcast_to(raw_phase, shape).copy()
    )
    return RcsGrid(
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs_power=power_values,
        rcs_phase=phase_values,
        units=units
        or {
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        },
        extra=extra,
    )


class NativeFidelityTests(unittest.TestCase):
    def test_roundtrip_preserves_non_grid_ghost_ancillary_arrays(self):
        grid = _grid()
        grid.extra.update(
            {
                "body_profile_rho_m": np.asarray([0.0, 0.2, 0.0]),
                "body_profile_z_m": np.asarray([-1.0, 0.0, 1.0]),
                "body_model_aspects_deg": np.asarray([-90.0, 0.0, 90.0]),
                "body_model_amp_vv_real": np.arange(6.0).reshape(3, 2),
                "body_model_amp_vv_imag": np.arange(6.0, 12.0).reshape(3, 2),
                "surface_triangles_cad_m": np.arange(18.0).reshape(2, 3, 3),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            restored = RcsGrid.load(grid.save(os.path.join(directory, "body")))
        for key, expected in grid.extra.items():
            np.testing.assert_array_equal(restored.extra[key], expected)

    def test_compact_default_and_fast_native_modes_roundtrip(self):
        grid = _grid(
            np.ones((20, 1, 20, 1)),
            phase=np.zeros((20, 1, 20, 1)),
            azimuths=tuple(range(20)),
            frequencies=tuple(range(1, 21)),
        )
        with tempfile.TemporaryDirectory() as directory:
            compact = grid.save(os.path.join(directory, "compact"))
            fast = grid.save(os.path.join(directory, "fast"), compressed=False)
            with zipfile.ZipFile(compact) as archive:
                self.assertTrue(
                    all(
                        member.compress_type == zipfile.ZIP_DEFLATED
                        for member in archive.infolist()
                    )
                )
            with zipfile.ZipFile(fast) as archive:
                self.assertTrue(
                    all(
                        member.compress_type == zipfile.ZIP_STORED
                        for member in archive.infolist()
                    )
                )
            self.assertLess(os.path.getsize(compact), os.path.getsize(fast))
            np.testing.assert_array_equal(
                RcsGrid.load(compact).rcs_power, RcsGrid.load(fast).rcs_power
            )

    def _write_native(self, path, **updates):
        payload = {
            "azimuths": np.asarray([0.0]),
            "elevations": np.asarray([0.0]),
            "frequencies": np.asarray([1.0]),
            "polarizations": np.asarray(["VV"]),
            "rcs_power": np.asarray([[[[1.0]]]]),
            "rcs_phase": np.asarray([[[[0.0]]]]),
            "units": json.dumps(
                {"azimuth": "deg", "elevation": "deg", "frequency": "GHz"}
            ),
        }
        payload.update(updates)
        with open(path, "wb") as stream:
            np.savez(stream, **payload)

    def test_native_validation_allows_nan_sparsity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sparse.grim")
            self._write_native(
                path,
                azimuths=np.asarray([0.0, 1.0]),
                rcs_power=np.asarray([1.0, np.nan]).reshape(2, 1, 1, 1),
                rcs_phase=np.asarray([0.0, np.nan]).reshape(2, 1, 1, 1),
            )
            loaded = RcsGrid.load(path)
        self.assertTrue(np.isnan(loaded.rcs_power[1]).all())

    def test_native_float32_axes_keep_decimal_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "float32-axis.grim")
            self._write_native(path, azimuths=np.asarray([0.1], dtype=np.float32))
            loaded = RcsGrid.load(path)
        self.assertEqual(float(loaded.azimuths[0]), 0.1)

    def test_native_validation_rejects_malformed_physical_payloads(self):
        cases = {
            "duplicate azimuth": {
                "azimuths": np.asarray([0.0, 0.0]),
                "rcs_power": np.ones((2, 1, 1, 1)),
                "rcs_phase": np.zeros((2, 1, 1, 1)),
            },
            "nonpositive frequency": {"frequencies": np.asarray([0.0])},
            "duplicate polarization": {
                "polarizations": np.asarray(["VV", "vv"]),
                "rcs_power": np.ones((1, 1, 1, 2)),
                "rcs_phase": np.zeros((1, 1, 1, 2)),
            },
            "negative finite rcs_power": {
                "rcs_power": np.asarray([[[[-1.0]]]])
            },
            "infinite rcs_phase": {"rcs_phase": np.asarray([[[[np.inf]]]])},
            "unsupported frequency unit": {
                "units": json.dumps({"frequency": "THz"})
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            for expected, updates in cases.items():
                with self.subTest(expected=expected):
                    path = os.path.join(
                        directory, expected.replace(" ", "_") + ".grim"
                    )
                    self._write_native(path, **updates)
                    with self.assertRaisesRegex(ValueError, expected):
                        RcsGrid.load(path)

    def test_native_save_revalidates_publicly_mutated_arrays(self):
        grid = _grid()
        grid.rcs_power[0, 0, 0, 0] = -1.0
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invalid.grim")
            with self.assertRaisesRegex(ValueError, "negative finite rcs_power"):
                grid.save(path)
            self.assertFalse(os.path.exists(path))


class CoreOperationTests(unittest.TestCase):
    def test_value_lookup_applies_tolerance_only_to_numeric_axes(self):
        grid = _grid(
            power=[4.0],
            phase=[0.25],
            azimuths=(10.0,),
            elevations=(20.0,),
            frequencies=(3.0,),
            polarizations=("VV",),
        )

        sample = grid.get_by_value(10.0001, 19.9999, 3.0001, "VV", tol=1.0e-3)

        self.assertAlmostEqual(float(np.abs(sample)), 2.0)
        self.assertAlmostEqual(float(np.angle(sample)), 0.25)
        with self.assertRaisesRegex(ValueError, "value vv not found"):
            grid.get_by_value(10.0, 20.0, 3.0, "vv", tol=1.0e-3)

    def test_degree_valued_transforms_convert_for_radian_axes(self):
        units = {
            "azimuth": "rad",
            "elevation": "rad",
            "frequency": "GHz",
        }
        grid = _grid(
            [1.0, 2.0],
            phase=[0.0, 0.1],
            azimuths=(0.0, np.pi / 2.0),
            units=units,
        )
        np.testing.assert_allclose(
            grid.shift_azimuth(180.0).azimuths,
            [np.pi, 1.5 * np.pi],
        )
        np.testing.assert_allclose(
            grid.mirror_about_azimuth(90.0).azimuths,
            [np.pi / 2.0, np.pi],
        )
        np.testing.assert_allclose(
            grid.shift_elevation(90.0).elevations,
            [np.pi / 2.0],
        )

    def test_swap_exchanges_angular_unit_metadata(self):
        grid = _grid(
            np.ones((2, 3, 1, 1)),
            phase=np.zeros((2, 3, 1, 1)),
            azimuths=(0.0, 1.0),
            elevations=(10.0, 20.0, 30.0),
            units={"azimuth": "rad", "elevation": "deg", "frequency": "GHz"},
        )
        swapped = grid.swap_elevation_azimuth()
        self.assertEqual(swapped.units["azimuth"], "deg")
        self.assertEqual(swapped.units["elevation"], "rad")
        self.assertEqual(swapped.rcs_power.shape, (3, 2, 1, 1))

    def test_wrap_merges_equivalent_rad_seam_and_rejects_conflict(self):
        units = {"azimuth": "rad", "elevation": "deg", "frequency": "GHz"}
        equivalent = _grid(
            [1.0, 2.0, 1.0],
            phase=[0.25, 0.5, 0.25],
            azimuths=(-np.pi, 0.0, np.pi),
            units=units,
        )
        wrapped = equivalent.wrap_azimuth("-180_180")
        np.testing.assert_allclose(wrapped.azimuths, [-np.pi, 0.0])
        np.testing.assert_allclose(wrapped.rcs_power.ravel(), [1.0, 2.0])

        conflicting = _grid(
            [1.0, 2.0],
            phase=[0.0, 0.0],
            azimuths=(0.0, 2.0 * np.pi),
            units=units,
        )
        with self.assertRaisesRegex(ValueError, "conflicting finite seam"):
            conflicting.wrap_azimuth("0_360")

    def test_el_to_az360_rejects_conflicting_overlap(self):
        power = np.asarray(
            [
                [[[1.0]], [[2.0]]],
                [[[3.0]], [[4.0]]],
            ]
        )
        grid = _grid(
            power,
            phase=np.zeros_like(power),
            azimuths=(0.0, 180.0),
            elevations=(-1.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "conflicting finite seam"):
            grid.combine_elevation_pair_to_azimuth_360()

    def test_el_to_az360_merges_equivalent_overlap(self):
        power = np.asarray(
            [
                [[[1.0]], [[2.0]]],
                [[[2.0]], [[4.0]]],
            ]
        )
        grid = _grid(
            power,
            phase=np.zeros_like(power),
            azimuths=(0.0, 180.0),
            elevations=(-1.0, 1.0),
        )
        combined = grid.combine_elevation_pair_to_azimuth_360()
        np.testing.assert_array_equal(combined.azimuths, [0.0, 180.0, 360.0])
        np.testing.assert_array_equal(combined.rcs_power.ravel(), [1.0, 2.0, 4.0])

        radian_grid = _grid(
            power,
            phase=np.zeros_like(power),
            azimuths=(0.0, np.pi),
            elevations=(-0.1, 0.1),
            units={"azimuth": "rad", "elevation": "rad", "frequency": "GHz"},
        )
        np.testing.assert_allclose(
            radian_grid.combine_elevation_pair_to_azimuth_360().azimuths,
            [0.0, np.pi, 2.0 * np.pi],
        )

    def test_db_difference_preserves_undefined_zero_ratios(self):
        left = _grid(
            np.asarray([0.0, 1.0, 0.0]).reshape(3, 1, 1, 1),
            phase=np.zeros((3, 1, 1, 1)),
            azimuths=(0.0, 1.0, 2.0),
        )
        right = _grid(
            np.asarray([0.0, 0.0, 2.0]).reshape(3, 1, 1, 1),
            phase=np.zeros((3, 1, 1, 1)),
            azimuths=(0.0, 1.0, 2.0),
        )
        result = left.arithmetic_db_subtract(right)
        self.assertTrue(np.isnan(result.rcs_power[0, 0, 0, 0]))
        self.assertTrue(np.isnan(result.rcs_power[1, 0, 0, 0]))
        self.assertEqual(float(result.rcs_power[2, 0, 0, 0]), 0.0)

    def test_db_difference_computes_float32_extreme_ratios_in_float64(self):
        shape = (2, 1, 1, 1)
        left = RcsGrid(
            [0.0, 1.0],
            [0.0],
            [1.0],
            ["VV"],
            rcs_power=np.asarray([1.0e30, 1.0e-30], dtype=np.float32).reshape(shape),
            rcs_phase=np.zeros(shape, dtype=np.float32),
        )
        right = RcsGrid(
            [0.0, 1.0],
            [0.0],
            [1.0],
            ["VV"],
            rcs_power=np.asarray([1.0e-30, 1.0e30], dtype=np.float32).reshape(shape),
            rcs_phase=np.zeros(shape, dtype=np.float32),
        )
        result = left.arithmetic_db_subtract(right)
        np.testing.assert_allclose(result.rcs_power.ravel(), [1.0e60, 1.0e-60])
        np.testing.assert_allclose(
            10.0 * np.log10(result.rcs_power.ravel()), [600.0, -600.0]
        )

    def test_align_intersect_never_reuses_one_source_sample(self):
        source = _grid(
            [10.0, 20.0],
            phase=[0.0, 0.0],
            azimuths=(0.0, 1.0e-6),
        )
        target = _grid(
            [1.0, 1.0],
            phase=[0.0, 0.0],
            azimuths=(0.6e-6, 1.4e-6),
        )
        aligned = source.align_to(target, mode="intersect")
        np.testing.assert_array_equal(aligned.rcs_power.ravel(), [10.0, 20.0])
        np.testing.assert_array_equal(aligned.azimuths, target.azimuths)

    def test_align_intersect_preserves_unsorted_target_order(self):
        source = _grid(
            [10.0, 20.0],
            phase=[0.0, 0.0],
            azimuths=(0.0, 1.0e-6),
        )
        target = _grid(
            [1.0, 1.0],
            phase=[0.0, 0.0],
            azimuths=(1.0e-6, 0.0),
        )
        aligned = source.align_to(target, mode="intersect")
        np.testing.assert_array_equal(aligned.azimuths, target.azimuths)
        np.testing.assert_array_equal(aligned.rcs_power.ravel(), [20.0, 10.0])

    def test_coherent_metadata_requires_reference_or_attestation(self):
        left = _grid()
        right = _grid()
        left.source_path = "left.grim"
        left.history = "loaded left"
        with self.assertRaisesRegex(ValueError, "nonblank phase reference"):
            left.coherent_add(right)
        added = left.coherent_add(right, metadata_attested=True)
        np.testing.assert_allclose(added.rcs_power, 4.0)
        self.assertEqual(added.source_path, "left.grim")
        self.assertNotIn("phase_reference", added.extra)
        self.assertIn("loaded left", added.history)
        self.assertIn("User-attested coherent metadata", added.history)
        added_record = json.loads(
            added.extra["coherent_metadata_attestation_json"]
        )
        self.assertEqual(added_record["operation"], "coherent-add")
        self.assertFalse(added_record["declarations_inferred"])

        subtracted = left.coherent_subtract(right, metadata_attested=True)
        subtract_record = json.loads(
            subtracted.extra["coherent_metadata_attestation_json"]
        )
        self.assertEqual(subtract_record["operation"], "coherent-subtract")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "attested.grim")
            added.save(path)
            restored = RcsGrid.load(path)
        self.assertIn("User-attested coherent metadata", restored.history)
        restored_attestation = np.asarray(
            restored.extra["coherent_metadata_attestation_json"]
        ).reshape(-1)[0]
        self.assertEqual(
            json.loads(str(restored_attestation)),
            added_record,
        )

        known = _grid(extra={"phase_reference": "origin A"})
        with self.assertRaisesRegex(ValueError, "phase references"):
            known.coherent_add(right)
        known.coherent_add(right, metadata_attested=True)
        adopted = right.coherent_add(known, metadata_attested=True)
        self.assertEqual(adopted._phase_reference(), "origin A")

        conflicting_known = _grid(extra={"phase_reference": "origin B"})
        with self.assertRaisesRegex(ValueError, "matching phase reference"):
            right.coherent_add_many(
                known,
                conflicting_known,
                metadata_attested=True,
            )

    def test_explicit_coherent_metadata_mismatch_is_not_overridable(self):
        base_units = {
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "polarization_basis": "basis-a",
        }
        left = _grid(
            units=base_units,
            extra={
                "phase_reference": "origin A",
                "time_convention": "exp(+j*omega*t)",
            },
        )
        mismatch_cases = (
            _grid(
                units=base_units,
                extra={
                    "phase_reference": "origin B",
                    "time_convention": "exp(+j*omega*t)",
                },
            ),
            _grid(
                units=base_units,
                extra={
                    "phase_reference": "origin A",
                    "time_convention": "exp(-j*omega*t)",
                },
            ),
            _grid(
                units={**base_units, "polarization_basis": "basis-b"},
                extra={
                    "phase_reference": "origin A",
                    "time_convention": "exp(+j*omega*t)",
                },
            ),
        )
        for other in mismatch_cases:
            with self.subTest(other=other.units.get("polarization_basis")):
                with self.assertRaisesRegex(ValueError, "requires matching"):
                    left.coherent_add(other, metadata_attested=True)

    def test_join_first_does_not_pair_kept_power_with_conflicting_phase(self):
        first = _grid(1.0, phase=np.nan)
        second = _grid(2.0, phase=0.75)
        joined = RcsGrid.join_many(first, second, overlap="first")
        self.assertEqual(float(joined.rcs_power.item()), 1.0)
        self.assertTrue(np.isnan(joined.rcs_phase.item()))

    def test_join_allows_irrelevant_phase_difference_at_exact_zero(self):
        first = _grid(0.0, phase=0.0)
        second = _grid(0.0, phase=1.5)
        joined = RcsGrid.join_many(first, second, overlap="error")
        self.assertEqual(float(joined.rcs_power.item()), 0.0)

    def test_unknown_frequency_unit_never_falls_back_to_ghz(self):
        grid = _grid(units={"frequency": "THz"})
        with self.assertRaisesRegex(ValueError, "unsupported frequency unit"):
            grid.linear_to_dbke(grid.rcs_power, grid.frequencies)


class OutParserTests(unittest.TestCase):
    def test_out_rejects_conflicting_duplicates_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = os.path.join(directory, "sample_HH.out")
            with open(duplicate, "w", encoding="utf-8") as stream:
                stream.write("1 0 10 0\n1 0 20 0\n")
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                RcsGrid.load_out(duplicate)

            extra = os.path.join(directory, "sample_VV.out")
            with open(extra, "w", encoding="utf-8") as stream:
                stream.write("1 0 10 0 unexpected\n")
            with self.assertRaisesRegex(ValueError, "exactly 4 columns"):
                RcsGrid.load_out(extra)


class PioStreamingTests(unittest.TestCase):
    def test_multiblock_double_precision_roundtrip_preserves_fortran_order(self):
        values = (
            np.arange(12, dtype=np.float64).reshape(3, 1, 4, 1)
            + 1j
            * np.arange(100, 112, dtype=np.float64).reshape(3, 1, 4, 1)
        )
        grid = RcsGrid(
            [-10.0, 0.0, 10.0],
            [5.0],
            [8.0, 9.0, 10.0, 11.0],
            ["VV"],
            rcs=values,
        )
        with tempfile.TemporaryDirectory() as directory:
            # Three azimuths and a four-cell scratch target force one
            # frequency per tile, exercising every block boundary.
            with mock.patch("grim_dataset._PIO_WRITE_BLOCK_CELLS", 4):
                path = grid.save_pio(
                    os.path.join(directory, "streamed"), precision="double"
                )
            restored = RcsGrid.load_pio(path)
        np.testing.assert_array_equal(restored.azimuths, grid.azimuths)
        np.testing.assert_array_equal(restored.frequencies, grid.frequencies)
        # load_pio stores power/phase and reconstructs the field, so permit
        # only roundoff from the polar/cartesian conversion (not reordering).
        np.testing.assert_allclose(restored.rcs, values, rtol=1.0e-14, atol=1.0e-13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
