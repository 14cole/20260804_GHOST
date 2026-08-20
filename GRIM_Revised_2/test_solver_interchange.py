#!/usr/bin/env python3
"""Consumer-side acceptance tests for current GHOST solver `.grim` output."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from grim_dataset import C0, RcsGrid  # noqa: E402


class SolverInterchangeAcceptanceTests(unittest.TestCase):
    def _write_solver_style_file(self, path):
        azimuths = np.asarray([-20.0, 30.0], dtype=float)
        elevations = np.asarray([0.0], dtype=float)
        frequencies = np.asarray([1.0, 1.5], dtype=float)
        polarizations = np.asarray(["HH"])
        shape = (2, 1, 2, 1)
        raw = np.empty(shape, dtype=np.complex128)
        for ai, angle in enumerate(azimuths):
            for fi, frequency_ghz in enumerate(frequencies):
                raw[ai, 0, fi, 0] = complex(
                    1.0 + angle / 100.0, -0.2 * frequency_ghz
                )
        k0 = 2.0 * math.pi * frequencies * 1.0e9 / C0
        power = np.abs(raw) ** 2 / (4.0 * k0[None, None, :, None])
        units = {
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBke",
            "rcs_linear_quantity": "sigma_2d",
        }
        with open(path, "wb") as stream:
            np.savez_compressed(
                stream,
                azimuths=azimuths,
                elevations=elevations,
                frequencies=frequencies,
                polarizations=polarizations,
                polarization_alias_primary="TM",
                polarization_aliases_json=json.dumps(
                    ["TM", "HH", "H", "HORIZONTAL"]
                ),
                rcs_power=power.astype(np.float32),
                rcs_phase=np.angle(raw).astype(np.float32),
                rcs_amp_real=raw.real.astype(np.float64),
                rcs_amp_imag=raw.imag.astype(np.float64),
                rcs_domain="power_phase",
                power_domain="linear_rcs",
                source_path="acceptance.geo",
                history="solver interchange acceptance",
                units=json.dumps(units),
                phase_reference=(
                    "origin=(0,0), convention=exp(+jwt); stored complex field "
                    "is the 2D layer-potential bare-integral amplitude B"
                ),
                amplitude_convention="A_physical_asymptotic = +j * B_stored",
                raw_complex_amplitude_preserved=True,
                complex_field_domain=(
                    "2d_layer_potential_bare_integral_amplitude_B"
                ),
            )
        return raw, power

    def test_loads_solver_2d_field_with_dBke_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solver.grim"
            raw, power = self._write_solver_style_file(path)
            grid = RcsGrid.load(str(path))

            self.assertEqual(grid.default_log_unit(), "dBke")
            self.assertEqual(grid.linear_quantity(), "sigma_2d")
            self.assertEqual(grid.rcs_power.shape, (2, 1, 2, 1))
            np.testing.assert_allclose(
                grid.rcs_power, power, rtol=2.0e-7, atol=0.0
            )
            self.assertIn("rcs_amp_real", grid.extra)
            self.assertIn("rcs_amp_imag", grid.extra)

            k0 = 2.0 * math.pi * grid.frequencies * 1.0e9 / C0
            expected_coherent = raw / (2.0 * np.sqrt(k0)[None, None, :, None])
            np.testing.assert_allclose(
                grid.rcs, expected_coherent, rtol=2.0e-15, atol=2.0e-15
            )
            np.testing.assert_allclose(
                np.abs(grid.rcs) ** 2,
                grid.rcs_power,
                rtol=2.0e-7,
                atol=0.0,
            )

    def test_load_save_roundtrip_preserves_solver_complex_amplitude(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.grim"
            self._write_solver_style_file(source)
            original = RcsGrid.load(str(source))
            saved = original.save(str(Path(tmp) / "roundtrip"))
            reloaded = RcsGrid.load(saved)

            self.assertEqual(reloaded.default_log_unit(), "dBke")
            self.assertEqual(reloaded.linear_quantity(), "sigma_2d")
            np.testing.assert_array_equal(
                reloaded.extra["rcs_amp_real"],
                original.extra["rcs_amp_real"],
            )
            np.testing.assert_array_equal(
                reloaded.extra["rcs_amp_imag"],
                original.extra["rcs_amp_imag"],
            )
            np.testing.assert_allclose(
                reloaded.rcs, original.rcs, rtol=0.0, atol=0.0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
