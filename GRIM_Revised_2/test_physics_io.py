"""Physical-semantics and solver-interchange regressions for GRIM."""

import json
import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from assembly_tree import _b64_to_grid, _combine_children, _grid_to_b64
import grim_dataset
from grim_dataset import C0, RcsGrid
from grim_headless import load_dataset, load_folder


class TestPhysicsAndIo(unittest.TestCase):
    def _grid(self, value=1.0 + 0.0j, *, units=None, dtype=np.complex128):
        return RcsGrid(
            [0.0], [0.0], [3.0], ["VV"],
            rcs=np.asarray([value], dtype=dtype).reshape(1, 1, 1, 1),
            units=units or {
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
        )

    def test_units_and_unknown_phase_block_coherent_arithmetic(self):
        ghz = self._grid(units={"frequency": "GHz"})
        hz = self._grid(units={"frequency": "Hz"})
        with self.assertRaisesRegex(ValueError, "frequency unit mismatch"):
            ghz.coherent_add(hz)

        power_only = RcsGrid(
            [0.0], [0.0], [3.0], ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            units=ghz.units,
        )
        with self.assertRaisesRegex(ValueError, "requires phase"):
            ghz.coherent_add(power_only)

    def test_mixed_assembly_has_no_invented_phase(self):
        coherent = self._grid(1.0 + 0.0j)
        incoherent = RcsGrid(
            [0.0], [0.0], [3.0], ["VV"],
            rcs_power=np.ones((1, 1, 1, 1)),
            units=coherent.units,
        )
        mixed = _combine_children([coherent], [incoherent], coherent)
        self.assertEqual(float(mixed.rcs_power.item()), 2.0)
        self.assertTrue(np.isnan(mixed.rcs_phase.item()))
        with self.assertRaisesRegex(ValueError, "cannot be used as a coherent"):
            _combine_children([mixed, coherent], [], coherent)

    def test_float64_coherent_cancellation_is_preserved(self):
        left = self._grid(1.0 + 0.0j)
        right = self._grid(-1.0 + 1.0e-9j)
        result = left.coherent_add(right)
        self.assertEqual(result.rcs_power.dtype, np.float64)
        self.assertAlmostEqual(float(result.rcs_power.item()), 1.0e-18, delta=1.0e-24)

    def test_assembly_serialization_preserves_precision_and_solver_metadata(self):
        grid = self._grid(1.0 + 1.0e-12j)
        grid.extra["phase_reference"] = "origin=(0,0), convention=exp(+jwt)"
        scale = np.sqrt(4.0 * np.pi)
        grid.extra["rcs_amp_real"] = np.full((1, 1, 1, 1), 1.0 / scale, dtype=np.float64)
        grid.extra["rcs_amp_imag"] = np.full((1, 1, 1, 1), 1.0e-12 / scale, dtype=np.float64)
        restored = _b64_to_grid(_grid_to_b64(grid))
        self.assertEqual(restored.rcs_power.dtype, np.float64)
        self.assertEqual(restored.rcs_phase.dtype, np.float64)
        np.testing.assert_array_equal(restored.extra["rcs_amp_imag"], grid.extra["rcs_amp_imag"])
        self.assertEqual(restored._phase_reference(), grid._phase_reference())

    def test_solver_raw_amplitude_drives_coherent_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "solver.grim")
            frequency = 3.0
            k0 = 2.0 * np.pi * frequency * 1.0e9 / C0
            raw = np.asarray([1.0 + 2.0j], dtype=np.complex128).reshape(1, 1, 1, 1)
            power = np.abs(raw) ** 2 / (4.0 * k0)
            with open(path, "wb") as stream:
                np.savez_compressed(
                    stream,
                    azimuths=np.asarray([0.0]), elevations=np.asarray([0.0]),
                    frequencies=np.asarray([frequency]), polarizations=np.asarray(["HH"]),
                    rcs_power=power.astype(np.float32),
                    rcs_phase=np.angle(raw).astype(np.float32),
                    rcs_amp_real=raw.real, rcs_amp_imag=raw.imag,
                    rcs_domain="power_phase", power_domain="linear_rcs",
                    source_path="", history="solver fixture",
                    units=json.dumps({
                        "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                        "rcs_log_unit": "dBke", "rcs_linear_quantity": "sigma_2d",
                    }),
                    phase_reference="origin=(0,0), convention=exp(+jwt)",
                    amplitude_convention="solver B field",
                    raw_complex_amplitude_preserved=True,
                )
            grid = RcsGrid.load(path)
            expected = raw / (2.0 * np.sqrt(k0))
            self.assertEqual(grid.rcs.dtype, np.complex128)
            np.testing.assert_allclose(grid.rcs, expected, rtol=0.0, atol=1.0e-15)
            roundtrip = grid.save(os.path.join(tmp, "roundtrip.grim"))
            reloaded = RcsGrid.load(roundtrip)
            np.testing.assert_array_equal(reloaded.extra["rcs_amp_real"], raw.real)
            np.testing.assert_array_equal(reloaded.extra["rcs_amp_imag"], raw.imag)
            self.assertEqual(reloaded.linear_quantity(), "sigma_2d")
            self.assertEqual(reloaded._phase_reference(), "origin=(0,0), convention=exp(+jwt)")

    def test_grim_save_failure_preserves_existing_artifact(self):
        grid = self._grid()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "protected.grim")
            with open(path, "wb") as stream:
                stream.write(b"existing artifact")

            with mock.patch.object(
                grim_dataset.np,
                "savez",
                side_effect=RuntimeError("simulated write failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                    grid.save(path)

            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), b"existing artifact")
            self.assertEqual(os.listdir(tmp), ["protected.grim"])

    def test_db_difference_is_a_dimensionless_ratio(self):
        a = self._grid(np.sqrt(1000.0) + 0.0j)
        b = self._grid(np.sqrt(10.0 ** 2.5) + 0.0j)
        result = a.arithmetic_db_subtract(b)
        self.assertEqual(result.default_log_unit(), "dB")
        self.assertEqual(result.linear_quantity(), "power_ratio")
        self.assertAlmostEqual(float(result.linear_to_default_db(result.rcs_power).item()), 5.0)

    def test_interpolation_keeps_phase_local(self):
        power = np.ones((2, 2, 1, 1), dtype=np.float64)
        phase = np.zeros_like(power)
        phase[:, 1, :, :] = np.nan
        grid = RcsGrid([0.0, 1.0], [0.0, 1.0], [3.0], ["VV"],
                       rcs_power=power, rcs_phase=phase)
        result = grid.interpolate_axis("azimuth", [0.0, 0.5, 1.0])
        self.assertTrue(np.isfinite(result.rcs_phase[:, 0, :, :]).all())
        self.assertTrue(np.isnan(result.rcs_phase[:, 1, :, :]).all())

    def test_join_rejects_conflicting_overlap(self):
        a = self._grid(1.0 + 0.0j)
        b = self._grid(2.0 + 0.0j)
        with self.assertRaisesRegex(ValueError, "conflicting finite samples"):
            RcsGrid.join_many(a, b)

    def test_pio_rejects_power_only_and_preserves_double(self):
        with tempfile.TemporaryDirectory() as tmp:
            power_only = RcsGrid(
                [0.0], [0.0], [3.0], ["VV"],
                rcs_power=np.full((1, 1, 1, 1), 4.0),
            )
            with self.assertRaisesRegex(ValueError, "requires phase"):
                power_only.save_pio(os.path.join(tmp, "power.pio"))

            source = self._grid(1.0 + 1.0e-12j)
            path = source.save_pio(os.path.join(tmp, "double.pio"), precision="double")
            loaded = RcsGrid.load_pio(path)
            self.assertEqual(loaded.rcs_power.dtype, np.float64)
            np.testing.assert_allclose(loaded.rcs, source.rcs, rtol=0.0, atol=1.0e-15)

            with open(path, "rb") as stream:
                contents = stream.read()
            contents = contents.replace(b"XName=azimuth", b"XName=invalid")
            bad_path = os.path.join(tmp, "bad_axis.pio")
            with open(bad_path, "wb") as stream:
                stream.write(contents)
            with self.assertRaisesRegex(ValueError, "Unsupported PIO axes"):
                RcsGrid.load_pio(bad_path)

    def test_headless_flat_csv_and_folder_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "flat.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "azimuth", "elevation", "frequency", "frequency_unit",
                    "polarization", "rcs_log_unit", "magnitude_linear", "phase_deg",
                ])
                writer.writerow([0.0, 0.0, 3.0, "GHz", "VV", "dBsm", 1.0, 0.0])
            loaded = load_dataset(csv_path)
            self.assertEqual(float(loaded.rcs_power.item()), 1.0)

            units = {
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            }
            for frequency in (3.0, 4.0):
                grid = RcsGrid(
                    [0.0], [0.0], [frequency], ["VV"],
                    rcs=np.ones((1, 1, 1, 1), dtype=np.complex128), units=units,
                )
                grid.save(os.path.join(tmp, f"part_{frequency:g}.grim"))
            joined = load_folder(tmp, pattern="part_*.grim", operation="join")
            np.testing.assert_allclose(joined.frequencies, [3.0, 4.0])
            np.testing.assert_allclose(joined.rcs_power.ravel(), [1.0, 1.0])

    def test_pickle_is_opt_in_for_legacy_grim_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.grim")
            shape = (1, 1, 1, 1)
            with open(path, "wb") as stream:
                np.savez(
                    stream,
                    azimuths=[0.0], elevations=[0.0], frequencies=[3.0],
                    polarizations=np.asarray(["VV"], dtype=object),
                    rcs_power=np.ones(shape), rcs_phase=np.zeros(shape),
                    units="{}", source_path="", history="",
                )
            with self.assertRaises(ValueError):
                RcsGrid.load(path)
            legacy = RcsGrid.load(path, allow_legacy_pickle=True)
            self.assertEqual(str(legacy.polarizations[0]), "VV")


if __name__ == "__main__":
    unittest.main(verbosity=2)
