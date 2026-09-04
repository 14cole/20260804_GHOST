"""Regressions for native fidelity and physically safe dataset operations."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

import grim_dataset
from grim_dataset import (
    C0,
    RcsGrid,
    _checked_dense_import_allocation,
    _preflight_native_archive_allocation,
)


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
    def test_native_load_closes_numpy_archive_before_return(self):
        grid = _grid()
        with tempfile.TemporaryDirectory() as directory:
            path = grid.save(os.path.join(directory, "close-check"))
            real_load = np.load
            state = {"exited": False}

            class TrackingArchive:
                def __init__(self, archive):
                    self.archive = archive

                def __enter__(self):
                    return self.archive

                def __exit__(self, exc_type, exc, traceback):
                    state["exited"] = True
                    self.archive.close()

            def tracked_load(*args, **kwargs):
                return TrackingArchive(real_load(*args, **kwargs))

            with mock.patch.object(grim_dataset.np, "load", side_effect=tracked_load):
                restored = RcsGrid.load(path)

        self.assertTrue(state["exited"])
        np.testing.assert_array_equal(restored.rcs_power, grid.rcs_power)

    def test_native_load_preflights_peak_bytes_before_numpy_extraction(self):
        grid = _grid()
        with tempfile.TemporaryDirectory() as directory:
            path = grid.save(os.path.join(directory, "bounded"), compressed=False)
            report = _preflight_native_archive_allocation(
                path, max_output_bytes=1024**2
            )
            required = int(report["estimated_peak_bytes"])
            with (
                mock.patch.object(grim_dataset.np, "load") as numpy_load,
                self.assertRaisesRegex(
                    MemoryError, "peak memory|uncompressed native members"
                ),
            ):
                RcsGrid.load(path, max_output_bytes=required - 1)
            numpy_load.assert_not_called()
            restored = RcsGrid.load(path, max_output_bytes=required)

        np.testing.assert_array_equal(restored.rcs_phase, grid.rcs_phase)

    def test_native_load_rejects_huge_declared_npy_shape_before_allocation(self):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": np.lib.format.dtype_to_descr(np.dtype("<f8")),
                "fortran_order": False,
                "shape": (2**31,),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "shape-bomb.grim")
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("rcs_power.npy", header.getvalue())
            with (
                mock.patch.object(grim_dataset.np, "load") as numpy_load,
                self.assertRaisesRegex(ValueError, "shape/data framing"),
            ):
                RcsGrid.load(path, max_output_bytes=32 * 1024**3)
            numpy_load.assert_not_called()

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

            contradictory = _grid(
                extra={
                    "rcs_amp_real": np.asarray([[[[100.0]]]], dtype=np.float64),
                    "rcs_amp_imag": np.asarray([[[[0.0]]]], dtype=np.float64),
                }
            )
            with self.assertRaisesRegex(ValueError, "rcs_power is inconsistent"):
                contradictory.save(path)
            self.assertFalse(os.path.exists(path))

    def test_native_rejects_raw_complex_field_that_disagrees_with_display(self):
        """Plots and coherent exports must never see two different answers."""

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "contradictory.grim")
            self._write_native(
                path,
                rcs_amp_real=np.asarray([[[[100.0]]]], dtype=np.float64),
                rcs_amp_imag=np.asarray([[[[0.0]]]], dtype=np.float64),
                raw_complex_amplitude_preserved=np.asarray(True),
                units=json.dumps(
                    {
                        "frequency": "GHz",
                        "rcs_log_unit": "dBsm",
                        "rcs_linear_quantity": "sigma_3d",
                    }
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "rcs_power is inconsistent.*sigma_3d"
            ):
                RcsGrid.load(path)

    def test_native_audit_reports_raw_power_and_phase_disagreement(self):
        grid = _grid(
            extra={
                "rcs_amp_real": np.asarray([[[[100.0]]]], dtype=np.float64),
                "rcs_amp_imag": np.asarray([[[[0.0]]]], dtype=np.float64),
                "raw_complex_amplitude_preserved": True,
            }
        )
        grid.rcs_phase[...] = 0.5

        report = grid.audit()

        self.assertEqual(report["status"], "error")
        codes = {entry["code"] for entry in report["errors"]}
        self.assertIn("raw_complex_power_mismatch", codes)
        self.assertIn("raw_complex_phase_mismatch", codes)
        self.assertGreater(
            report["metrics"]["raw_complex"]["maximum_power_absolute_error"],
            1.0,
        )
        self.assertFalse(report["metrics"]["readiness"]["coherent_arithmetic"])

    def test_native_accepts_normalized_sparse_raw_pair_with_float32_roundoff(self):
        raw_real = np.asarray(
            [1.0 / np.sqrt(4.0 * np.pi), np.nan], dtype=np.float64
        ).reshape(2, 1, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "valid-sparse.grim")
            self._write_native(
                path,
                azimuths=np.asarray([0.0, 1.0]),
                rcs_power=np.asarray([1.0, np.nan], dtype=np.float32).reshape(
                    2, 1, 1, 1
                ),
                rcs_phase=np.asarray([0.0, np.nan], dtype=np.float32).reshape(
                    2, 1, 1, 1
                ),
                rcs_amp_real=raw_real,
                rcs_amp_imag=np.asarray([0.0, np.nan], dtype=np.float64).reshape(
                    2, 1, 1, 1
                ),
                raw_complex_amplitude_preserved=np.asarray(True),
                units=json.dumps(
                    {
                        "frequency": "GHz",
                        "rcs_log_unit": "dBsm",
                        "rcs_linear_quantity": "sigma_3d",
                    }
                ),
            )
            loaded = RcsGrid.load(path)

        report = loaded.audit()
        self.assertNotEqual(report["status"], "error")
        self.assertEqual(
            report["metrics"]["raw_complex"]["finite_pair_count"], 1
        )
        self.assertEqual(
            report["metrics"]["raw_complex"]["missing_pair_count"], 1
        )

    def test_native_sigma2d_raw_validation_honors_declared_frequency_unit(self):
        raw = 1.0 + 2.0j
        k0 = 2.0 * np.pi * 1.0e9 / C0
        power = abs(raw) ** 2 / (4.0 * k0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sigma2d-mhz.grim")
            self._write_native(
                path,
                frequencies=np.asarray([1000.0]),
                rcs_power=np.asarray([[[[power]]]], dtype=np.float32),
                rcs_phase=np.asarray([[[[np.angle(raw)]]]], dtype=np.float32),
                rcs_amp_real=np.asarray([[[[raw.real]]]], dtype=np.float64),
                rcs_amp_imag=np.asarray([[[[raw.imag]]]], dtype=np.float64),
                raw_complex_amplitude_preserved=np.asarray(True),
                units=json.dumps(
                    {
                        "frequency": "MHz",
                        "rcs_log_unit": "dBke",
                        "rcs_linear_quantity": "sigma_2d",
                    }
                ),
            )
            loaded = RcsGrid.load(path)

        np.testing.assert_allclose(
            loaded.rcs.item(), raw / (2.0 * np.sqrt(k0)), rtol=2.0e-15
        )
        self.assertEqual(
            loaded.audit()["metrics"]["raw_complex"]["normalization"],
            "sigma_2d",
        )

    def test_native_rejects_partial_raw_complex_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "partial-raw.grim")
            self._write_native(
                path,
                rcs_amp_real=np.asarray([[[[1.0]]]], dtype=np.float64),
            )
            with self.assertRaisesRegex(
                ValueError, "must provide both rcs_amp_real and rcs_amp_imag"
            ):
                RcsGrid.load(path)


class CoreOperationTests(unittest.TestCase):
    def test_dense_import_guard_uses_exact_checked_byte_arithmetic(self):
        report = _checked_dense_import_allocation(
            (2, 3, 4),
            (np.float32, np.float64),
            source="unit test",
            max_output_bytes=388,
            resident_bytes=100,
        )
        self.assertEqual(report["cell_count"], 24)
        self.assertEqual(report["dense_bytes"], 288)
        self.assertEqual(report["resident_bytes"], 100)
        self.assertEqual(report["peak_bytes"], 388)
        with self.assertRaisesRegex(MemoryError, "exceeding"):
            _checked_dense_import_allocation(
                (2, 3, 4),
                (np.float32, np.float64),
                source="unit test",
                max_output_bytes=387,
                resident_bytes=100,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _checked_dense_import_allocation(
                (1,), (np.float32,), source="unit test", max_output_bytes=4.5
            )
        with self.assertRaisesRegex(MemoryError, "addressable allocation size"):
            _checked_dense_import_allocation(
                (2**62, 2**62),
                (np.float64,),
                source="unit test",
                max_output_bytes=2**127,
            )
        with mock.patch(
            "grim_dataset._available_import_memory_bytes", return_value=1000
        ):
            dynamic = _checked_dense_import_allocation(
                (100,), (np.float32,), source="unit test"
            )
            self.assertEqual(dynamic["limit_bytes"], 500)
            with self.assertRaisesRegex(MemoryError, "exceeding"):
                _checked_dense_import_allocation(
                    (126,), (np.float32,), source="unit test"
                )

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
        lower_case_sample = grid.get_by_value(
            10.0, 20.0, 3.0, "vv", tol=1.0e-3
        )
        self.assertEqual(lower_case_sample, sample)

    def test_polarization_identity_is_case_canonical_across_construction_and_join(self):
        lower = _grid(
            power=[1.0],
            frequencies=(1.0,),
            polarizations=(" vv ",),
        )
        upper = _grid(
            power=[2.0],
            frequencies=(2.0,),
            polarizations=("VV",),
        )
        joined = RcsGrid.join_many(lower, upper)

        np.testing.assert_array_equal(lower.polarizations, ["VV"])
        np.testing.assert_array_equal(joined.polarizations, ["VV"])
        np.testing.assert_array_equal(joined.rcs_power.ravel(), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "unique after case normalization"):
            RcsGrid(
                [0.0],
                [0.0],
                [1.0],
                ["HH", "hh"],
                rcs_power=np.ones((1, 1, 1, 2)),
                rcs_phase=np.zeros((1, 1, 1, 2)),
            )

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
            grid.combine_elevation_pair_to_azimuth_360(
                assumptions_attested=True
            )

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
        combined = grid.combine_elevation_pair_to_azimuth_360(
            assumptions_attested=True
        )
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
            radian_grid.combine_elevation_pair_to_azimuth_360(
                assumptions_attested=True
            ).azimuths,
            [0.0, np.pi, 2.0 * np.pi],
        )

    def test_el_to_az360_requires_attestation_and_opposite_elevations(self):
        power = np.arange(1.0, 5.0).reshape(2, 2, 1, 1)
        symmetric = _grid(
            power,
            phase=np.zeros_like(power),
            azimuths=(0.0, 90.0),
            elevations=(-10.0, 10.0),
        )
        with self.assertRaisesRegex(ValueError, "acquisition-specific relabel"):
            symmetric.combine_elevation_pair_to_azimuth_360()

        asymmetric = _grid(
            power,
            phase=np.zeros_like(power),
            azimuths=(0.0, 90.0),
            elevations=(0.0, 10.0),
        )
        with self.assertRaisesRegex(ValueError, "equal-and-opposite"):
            asymmetric.combine_elevation_pair_to_azimuth_360(
                assumptions_attested=True
            )

        result = symmetric.combine_elevation_pair_to_azimuth_360(
            assumptions_attested=True
        )
        provenance = json.loads(result.extra["elevation_pair_to_azimuth_json"])
        self.assertTrue(provenance["user_assumptions_attested"])
        self.assertEqual(provenance["elevation_pair_deg"], [-10.0, 10.0])

    def test_axis_unit_conversion_preserves_physical_grid_and_samples(self):
        power = np.arange(1.0, 5.0).reshape(2, 1, 2, 1)
        phase = np.linspace(0.1, 0.4, 4).reshape(power.shape)
        grid = _grid(
            power,
            phase=phase,
            azimuths=(0.0, 180.0),
            elevations=(30.0,),
            frequencies=(1.0, 2.0),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
            },
        )
        converted = grid.convert_axis_units(
            azimuth="rad", elevation="rad", frequency="MHz"
        )
        np.testing.assert_allclose(converted.azimuths, [0.0, np.pi])
        np.testing.assert_allclose(converted.elevations, [np.pi / 6.0])
        np.testing.assert_allclose(converted.frequencies, [1000.0, 2000.0])
        np.testing.assert_array_equal(converted.rcs_power, power)
        np.testing.assert_array_equal(converted.rcs_phase, phase)
        self.assertEqual(converted.units["frequency"], "MHz")

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

    def test_align_only_requires_coordinate_metadata_compatibility(self):
        source = _grid(
            power=[1.0, 2.0],
            azimuths=(0.0, 1.0),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
        )
        target = _grid(
            power=[3.0, 4.0],
            azimuths=(0.0, 1.0),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBke",
                "rcs_linear_quantity": "sigma_2d",
            },
        )

        self.assertIs(source.align_to(target, mode="exact"), source)
        intersected = source.align_to(target, mode="intersect")
        self.assertEqual(intersected.linear_quantity(), "sigma_3d")
        np.testing.assert_array_equal(intersected.rcs_power, source.rcs_power)

    def test_coherent_metadata_missing_is_recorded_without_blocking(self):
        left = _grid()
        right = _grid()
        left.source_path = "left.grim"
        left.history = "loaded left"
        assumed = left.coherent_add(right)
        np.testing.assert_allclose(assumed.rcs_power, 4.0)
        assumption = json.loads(
            assumed.extra["coherent_metadata_assumption_json"]
        )
        self.assertFalse(assumption["user_attested"])
        self.assertIn("phase_reference", assumption["missing_declarations_by_input"])
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
        one_sided = known.coherent_add(right)
        self.assertNotIn("phase_reference", one_sided.extra)
        self.assertIn("coherent_source_conventions_json", one_sided.extra)
        known.coherent_add(right, metadata_attested=True)
        adopted = right.coherent_add(known, metadata_attested=True)
        self.assertEqual(adopted._phase_reference(), "")

        conflicting_known = _grid(extra={"phase_reference": "origin B"})
        combined = right.coherent_add_many(known, conflicting_known)
        np.testing.assert_allclose(combined.rcs_power, 9.0)
        self.assertNotIn("phase_reference", combined.extra)
        self.assertIn("origin B", combined.extra["coherent_source_conventions_json"])

    def test_explicit_coherent_metadata_mismatch_is_advisory(self):
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
                result = left.coherent_subtract(other)
                np.testing.assert_array_equal(result.rcs_power, 0.0)
                record = json.loads(result.extra["coherent_metadata_assumption_json"])
                self.assertTrue(record["advisories"])
                self.assertIn("no phase or amplitude conversion", record["metadata_policy"])

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
    @staticmethod
    def _write_pio_fixture(
        path,
        *,
        xsize="1",
        ysize="1",
        xvals="0",
        yvals="1",
        data_type="Complex",
        precision="Single",
        data_format="Binary",
        offset_value=None,
        payload=None,
        xstart=None,
        xstop=None,
        xstep=None,
        ystart=None,
        ystop=None,
        ystep=None,
        xunits="deg",
        yunits="GHz",
        elevation=None,
        elevationunits=None,
    ):
        def _summary(raw_values, raw_size, fallback):
            try:
                size = int(raw_size)
                values = [float(value) for value in str(raw_values).split(":")]
                if len(values) != size or not np.all(np.isfinite(values)):
                    raise ValueError
            except (TypeError, ValueError):
                return fallback, fallback, "0"
            step = 0.0 if size == 1 else (values[-1] - values[0]) / (size - 1)
            return f"{values[0]:.17g}", f"{values[-1]:.17g}", f"{step:.17g}"

        default_xstart, default_xstop, default_xstep = _summary(
            xvals, xsize, "0"
        )
        default_ystart, default_ystop, default_ystep = _summary(
            yvals, ysize, "1"
        )
        lines = [
            "Name=fixture",
            f"XStart={default_xstart if xstart is None else xstart}",
            f"XStop={default_xstop if xstop is None else xstop}",
            f"XStep={default_xstep if xstep is None else xstep}",
            f"XSize={xsize}",
            "XName=azimuth",
            *([] if xunits is None else [f"XUnits={xunits}"]),
            f"XVals={xvals}",
            f"YStart={default_ystart if ystart is None else ystart}",
            f"YStop={default_ystop if ystop is None else ystop}",
            f"YStep={default_ystep if ystep is None else ystep}",
            f"YSize={ysize}",
            "YName=frequency",
            *([] if yunits is None else [f"YUnits={yunits}"]),
            f"YVals={yvals}",
            *([] if elevation is None else [f"Elevation={elevation}"]),
            *(
                []
                if elevationunits is None
                else [f"ElevationUnits={elevationunits}"]
            ),
            f"Type={data_type}",
            f"Precision={precision}",
            "Order=Little Endian",
            f"DataFormat={data_format}",
        ]
        header = ("\n".join(lines) + "\n").encode("ascii")
        if offset_value is None:
            offset = len(header) + 18
            offset_line = f"Offset={offset:010d}\n".encode("ascii")
        else:
            offset_line = f"Offset={offset_value}\n".encode("ascii")
            offset = None
        if payload is None:
            payload = np.asarray([1.0, 2.0], dtype="<f4").tobytes()
        with open(path, "wb") as stream:
            stream.write(header)
            stream.write(offset_line)
            if offset is not None:
                self_position = stream.tell()
                if offset > self_position:
                    stream.write(b"\0" * (offset - self_position))
            stream.write(payload)

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

    def test_pio_removes_arbitrary_angle_closed_sweep_endpoint(self):
        samples = np.asarray(
            [10.0 + 1.0j, 20.0 + 2.0j, 30.0 + 3.0j, 40.0 + 4.0j,
             99.0 + 9.0j],
            dtype=np.complex64,
        )
        payload = np.empty(samples.size * 2, dtype="<f4")
        payload[0::2] = samples.real
        payload[1::2] = samples.imag
        with tempfile.TemporaryDirectory() as directory:
            for label, closing in (
                ("unwrapped", "181.16"),
                ("wrapped", "-178.84"),
                ("wrapped-rounded", "-178.8399"),
            ):
                with self.subTest(closing=closing):
                    path = os.path.join(directory, f"arbitrary-{label}.pio")
                    self._write_pio_fixture(
                        path,
                        xsize="5",
                        xvals=f"-178.84:-90:0:90:{closing}",
                        payload=payload.tobytes(),
                    )
                    loaded = RcsGrid.load_pio(path)
                    np.testing.assert_allclose(
                        loaded.azimuths, [-178.84, -90.0, 0.0, 90.0]
                    )
                    np.testing.assert_allclose(
                        loaded.rcs[:, 0, 0, 0], samples[:-1],
                        rtol=1.0e-6, atol=1.0e-6,
                    )
                    self.assertIn("removed repeated closing azimuth", loaded.history)

    def test_pio_writer_omits_arbitrary_angle_closed_sweep_endpoint(self):
        samples = np.asarray(
            [10.0 + 1.0j, 20.0 + 2.0j, 30.0 + 3.0j, 40.0 + 4.0j,
             99.0 + 9.0j],
            dtype=np.complex64,
        )
        with tempfile.TemporaryDirectory() as directory:
            for label, closing in (
                ("unwrapped", 181.16),
                ("wrapped", -178.84),
            ):
                with self.subTest(closing=closing):
                    grid = RcsGrid(
                        [-178.84, -90.0, 0.0, 90.0, closing],
                        [0.0],
                        [10.0],
                        ["VV"],
                        rcs=samples[:, None, None, None],
                    )
                    path = grid.save_pio(
                        os.path.join(directory, f"arbitrary-{label}.pio")
                    )
                    with open(path, "rb") as stream:
                        header = stream.read().split(b"Offset=", 1)[0]
                    loaded = RcsGrid.load_pio(path)

                    self.assertIn(b"XSize=4\n", header)
                    np.testing.assert_allclose(
                        loaded.azimuths, [-178.84, -90.0, 0.0, 90.0]
                    )
                    np.testing.assert_allclose(
                        loaded.rcs[:, 0, 0, 0], samples[:-1],
                        rtol=1.0e-6, atol=1.0e-6,
                    )

    def test_pio_rejects_offset_into_header_before_decoding_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "header-as-data.pio")
            self._write_pio_fixture(path, offset_value="0")
            with self.assertRaisesRegex(ValueError, "precedes the end of the header"):
                RcsGrid.load_pio(path)

    def test_pio_requires_exact_positive_dimensions_and_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = (
                ({"xsize": "1.5"}, "xsize must be an exact integer"),
                ({"ysize": "0"}, "ysize must be greater than zero"),
                ({"offset_value": "12.5"}, "offset must be an exact integer"),
                ({"offset_value": "999999"}, "lies beyond"),
            )
            for index, (updates, message) in enumerate(cases):
                with self.subTest(updates=updates):
                    path = os.path.join(directory, f"invalid-{index}.pio")
                    self._write_pio_fixture(path, **updates)
                    with self.assertRaisesRegex(ValueError, message):
                        RcsGrid.load_pio(path)

    def test_pio_rejects_unknown_type_format_and_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = (
                ({"data_type": "Magnitude"}, "Unsupported PIO Type"),
                ({"data_format": "ASCII"}, "Unsupported PIO DataFormat"),
                (
                    {"xsize": "2", "xvals": "0:1"},
                    "PIO data block truncated",
                ),
            )
            for index, (updates, message) in enumerate(cases):
                with self.subTest(updates=updates):
                    path = os.path.join(directory, f"framing-{index}.pio")
                    self._write_pio_fixture(path, **updates)
                    with self.assertRaisesRegex(ValueError, message):
                        RcsGrid.load_pio(path)

    def test_pio_requires_explicit_axis_units(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, (updates, message) in enumerate((
                ({"xunits": None}, "missing XUnits"),
                ({"yunits": None}, "missing YUnits"),
                (
                    {"elevation": "10", "elevationunits": None},
                    "Elevation but no ElevationUnits",
                ),
            )):
                with self.subTest(updates=updates):
                    path = os.path.join(directory, f"missing-units-{index}.pio")
                    self._write_pio_fixture(path, **updates)
                    with self.assertRaisesRegex(ValueError, message):
                        RcsGrid.load_pio(path)

    def test_pio_rejects_nonfinite_duplicate_and_nonmonotonic_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = np.asarray([1.0, 0.0, 2.0, 0.0], dtype="<f4").tobytes()
            for index, (xvals, message) in enumerate(
                (
                    ("0:nan", "nonfinite axis value"),
                    ("0:0", "strictly monotonic"),
                )
            ):
                with self.subTest(xvals=xvals):
                    path = os.path.join(directory, f"axis-{index}.pio")
                    self._write_pio_fixture(
                        path,
                        xsize="2",
                        xvals=xvals,
                        payload=payload,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        RcsGrid.load_pio(path)

            path = os.path.join(directory, "axis-nonmonotonic.pio")
            self._write_pio_fixture(
                path,
                xsize="3",
                xvals="0:2:1",
                payload=np.asarray(
                    [1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype="<f4"
                ).tobytes(),
            )
            with self.assertRaisesRegex(ValueError, "strictly monotonic"):
                RcsGrid.load_pio(path)

    def test_pio_rejects_conflicting_explicit_and_summary_axis_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "conflicting-axis.pio")
            self._write_pio_fixture(
                path,
                xsize="2",
                xvals="0:1",
                xstop="2",
                payload=np.asarray(
                    [1.0, 0.0, 2.0, 0.0], dtype="<f4"
                ).tobytes(),
            )
            with self.assertRaisesRegex(
                ValueError, "XStop=2 conflicts with explicit XVals endpoint 1"
            ):
                RcsGrid.load_pio(path)

    def test_pio_accepts_rounded_axis_summaries_when_xvals_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rounded-axis-summary.pio")
            self._write_pio_fixture(
                path,
                xsize="2",
                xvals="-18.2184:-17.2184",
                xstart="-18.218",
                xstop="-17.218",
                xstep="1",
                payload=np.asarray(
                    [1.0, 0.0, 2.0, 0.0], dtype="<f4"
                ).tobytes(),
            )
            loaded = RcsGrid.load_pio(path)

        np.testing.assert_allclose(loaded.azimuths, [-18.2184, -17.2184])
        np.testing.assert_allclose(loaded.rcs.real.ravel(), [1.0, 2.0])

    def test_pio_descending_axis_is_canonicalized_with_its_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "descending.pio")
            self._write_pio_fixture(
                path,
                xsize="2",
                xvals="1:0",
                payload=np.asarray(
                    [10.0, 0.0, 20.0, 0.0], dtype="<f4"
                ).tobytes(),
            )
            loaded = RcsGrid.load_pio(path)
        np.testing.assert_array_equal(loaded.azimuths, [0.0, 1.0])
        np.testing.assert_array_equal(loaded.rcs.real.ravel(), [20.0, 10.0])

    def test_pio_real_payload_remains_a_supported_legacy_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "real.pio")
            self._write_pio_fixture(
                path,
                data_type="Real",
                payload=(
                    np.asarray([2.5], dtype="<f4").tobytes()
                    + b"\nPolarity=VV\n"
                ),
            )
            loaded = RcsGrid.load_pio(path)
        self.assertAlmostEqual(complex(loaded.rcs.item()).real, 2.5)
        self.assertAlmostEqual(complex(loaded.rcs.item()).imag, 0.0)
        np.testing.assert_array_equal(loaded.polarizations, ["VV"])

    def test_pio_rejects_binary_tail_that_contradicts_declared_real_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "false-real.pio")
            self._write_pio_fixture(
                path,
                data_type="Real",
                payload=np.asarray([2.5, 7.25], dtype="<f4").tobytes(),
            )
            with self.assertRaisesRegex(
                ValueError,
                "after the declared data block|binary control bytes|"
                "footer must contain ASCII",
            ):
                RcsGrid.load_pio(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
